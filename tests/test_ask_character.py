"""
test_ask_character.py — ask_character MCP サーバーのユニットテスト

Phase 3: AITuber 掛け合い
"""

from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.mcp_servers.ask_character import (
    _ask_character_impl,
    reset_ask_character_context,
    set_ask_character_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """各テスト後に contextvars をリセット。"""
    yield
    reset_ask_character_context()


class TestSetAskCharacterContext:
    """contextvars のセット/リセットを検証する。"""

    def test_set_and_get(self):
        """セットした値が取得できること。"""
        from lab_lounge.mcp_servers.ask_character import (
            _on_tts_chunk_var, _tts_output_dir_var, _common_var, _caller_slug_var,
        )

        mock_callback = MagicMock()
        set_ask_character_context(
            on_tts_chunk=mock_callback,
            tts_output_dir="/tmp/audio",
            common={"stream_id": "s1"},
            caller_slug="mimi",
        )

        assert _on_tts_chunk_var.get() is mock_callback
        assert _tts_output_dir_var.get() == "/tmp/audio"
        assert _common_var.get() == {"stream_id": "s1"}
        assert _caller_slug_var.get() == "mimi"

    def test_reset(self):
        """リセット後にデフォルト値に戻ること。"""
        from lab_lounge.mcp_servers.ask_character import _caller_slug_var

        set_ask_character_context(caller_slug="mimi")
        reset_ask_character_context()
        assert _caller_slug_var.get() == ""


class TestAskCharacterImpl:
    """_ask_character_impl の動作を検証する。"""

    def test_self_call_returns_error(self):
        """自分自身への呼び出しはエラーメッセージを返す。"""
        set_ask_character_context(caller_slug="mimi")
        result = _ask_character_impl("mimi", "テスト質問")
        assert "エラー" in result
        assert "自分自身" in result

    def test_unknown_character_returns_error(self):
        """未知のキャラクター slug はエラーメッセージを返す。"""
        set_ask_character_context(caller_slug="mimi")
        result = _ask_character_impl("nonexistent_char", "テスト質問")
        assert "エラー" in result
        assert "存在しません" in result

    def test_returns_response_text(self):
        """正常系: 協働先の応答テキストが返ること。"""
        set_ask_character_context(
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        mock_result = MagicMock()
        mock_result.text = "データによると問題ありません"

        with patch("lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
                    return_value="データによると問題ありません"):
            result = _ask_character_impl("chisame", "データは大丈夫？")

        assert "データによると問題ありません" in result
        assert "ちさめ" in result or "chisame" in result.lower()

    def test_tts_called_when_enabled(self, monkeypatch):
        """L2_USE_REAL_TTS=true のとき TTS 合成が呼ばれること（導入 + 本応答の 2 回）。"""
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_tts_chunk = MagicMock()
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch("lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
                    return_value="テスト応答"), \
             patch("lab_lounge.tts.synthesize") as mock_synth:
            _ask_character_impl("chisame", "質問")

        # 導入セリフ (caller=mimi) + 本応答 (target=chisame) の 2 回
        assert mock_synth.call_count >= 2
        # 最後の呼び出しは chisame の応答
        last_call = mock_synth.call_args_list[-1]
        assert last_call.kwargs.get("speaker") == "chisame" or \
               last_call[1].get("speaker") == "chisame"

    def test_tts_not_called_when_disabled(self, monkeypatch):
        """L2_USE_REAL_TTS=false のとき TTS 合成がスキップされること。"""
        monkeypatch.setenv("L2_USE_REAL_TTS", "false")

        set_ask_character_context(
            on_tts_chunk=MagicMock(),
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch("lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
                    return_value="テスト応答"), \
             patch("lab_lounge.tts.synthesize") as mock_synth:
            _ask_character_impl("chisame", "質問")

        mock_synth.assert_not_called()

    def test_model_override(self, monkeypatch):
        """L2_ASK_CHARACTER_MODEL_OVERRIDE が設定されている場合にそのモデルが使われること。"""
        monkeypatch.setenv("L2_ASK_CHARACTER_MODEL_OVERRIDE", "gpt-5.4-nano")

        set_ask_character_context(
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch("lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
                    return_value="テスト") as mock_collab:
            _ask_character_impl("chisame", "質問")

        call_kwargs = mock_collab.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4-nano"


class TestMCPServer:
    """MCP サーバーインスタンスの生成を検証する。"""

    def test_get_server_returns_fastmcp(self):
        """get_server() が FastMCP インスタンスを返すこと。"""
        from lab_lounge.mcp_servers.ask_character import get_server
        server = get_server()
        assert server is not None
