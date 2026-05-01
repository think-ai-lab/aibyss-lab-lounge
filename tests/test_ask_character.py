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

    def test_set_resets_ask_count_and_previous_target(self):
        """set_ask_character_context は session_id ごとの ask_count と previous_target をリセットする。

        ターン開始時に呼ばれる set_ask_character_context が、前ターンの
        session 状態 (count, previous_target) を 0 / "" に戻すことを保証。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _ask_counts, _previous_targets, _ask_state_lock,
        )

        # 前ターン状態を残しておく
        with _ask_state_lock:
            _ask_counts["sess-x"] = 3
            _previous_targets["sess-x"] = "波心ちさめ"

        # 同 session_id で新ターン開始
        set_ask_character_context(
            caller_slug="mimi",
            common={"session_id": "sess-x", "stream_id": "s1", "trace_id": "t1"},
        )

        with _ask_state_lock:
            assert _ask_counts["sess-x"] == 0
            assert _previous_targets["sess-x"] == ""


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
        """L2_USE_REAL_TTS=true のとき TTS 合成が呼ばれること（導入 + 本応答の 2 回）。

        本応答 TTS はバックグラウンドスレッドで実行されるため、テストでは短い
        sleep でスレッド完了を待つ。mock の synthesize は処理時間 0 のため
        100ms で十分。
        """
        import time
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
            # バックグラウンド TTS スレッドが synthesize を呼ぶまで少し待つ
            for _ in range(20):
                if mock_synth.call_count >= 2:
                    break
                time.sleep(0.05)

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


class TestAskCharacterImplCountAndPrevious:
    """同一ターン内の連続 ask_character 呼出しで count/previous が更新されることを検証。"""

    def test_call_count_increments_across_calls(self):
        """同一 session_id 内で 2 回 _ask_character_impl を呼ぶと count が 1→2 と進む。"""
        from lab_lounge.mcp_servers.ask_character import _ask_counts, _ask_state_lock

        set_ask_character_context(
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value="テスト応答",
        ):
            _ask_character_impl("chisame", "質問1")
            with _ask_state_lock:
                assert _ask_counts["ss1"] == 1

            _ask_character_impl("sakura", "質問2")
            with _ask_state_lock:
                assert _ask_counts["ss1"] == 2

    def test_previous_target_updated_after_call(self):
        """1 回目呼出し後、_previous_targets[session_id] が target の display_name で更新される。"""
        from lab_lounge.mcp_servers.ask_character import _previous_targets, _ask_state_lock

        set_ask_character_context(
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value="テスト応答",
        ):
            _ask_character_impl("chisame", "質問1")

        # ちさめの display_name が previous として保持される
        with _ask_state_lock:
            assert _previous_targets["ss1"] == "波心ちさめ"

    def test_generate_intro_called_with_ask_index(self):
        """_generate_intro に ask_index=1 / 2 が伝わる (連続呼出しで)。"""
        set_ask_character_context(
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value="テスト応答",
        ), patch(
            "lab_lounge.mcp_servers.ask_character._generate_intro",
            return_value="導入セリフ",
        ) as mock_intro:
            _ask_character_impl("chisame", "質問1")
            _ask_character_impl("sakura", "質問2")

        assert mock_intro.call_count == 2
        first_kwargs = mock_intro.call_args_list[0].kwargs
        second_kwargs = mock_intro.call_args_list[1].kwargs
        assert first_kwargs["ask_index"] == 1
        assert first_kwargs["previous_target_display"] == ""
        assert second_kwargs["ask_index"] == 2
        # 2 回目では 1 回目の target display_name (波心ちさめ) が previous として伝わる
        assert second_kwargs["previous_target_display"] == "波心ちさめ"


class TestGenerateIntroPromptBranch:
    """_generate_intro が ask_index でプロンプト分岐することを検証。"""

    def test_first_call_prompt_includes_luca_acknowledgement(self):
        """ask_index=1 のプロンプトはルカへの受け止めコメントを要求する。"""
        from lab_lounge.mcp_servers.ask_character import _generate_intro
        from lab_lounge.characters import get_character

        captured_prompts: list[str] = []

        def _capture_call_llm(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            result = MagicMock()
            result.text = "テスト導入"
            return result

        caller = get_character("mimi")
        target = get_character("chisame")

        with patch(
            "lab_lounge.llm.call_llm",
            side_effect=_capture_call_llm,
        ):
            _generate_intro(
                caller_char=caller,
                target_char=target,
                user_question="あなたたちはAIですか?",
                ask_index=1,
                previous_target_display="",
            )

        assert len(captured_prompts) == 1
        first_prompt = captured_prompts[0]
        assert "ルカの質問を受け止めるコメント" in first_prompt
        assert target.display_name in first_prompt

    def test_second_call_prompt_omits_luca_greeting(self):
        """ask_index>=2 のプロンプトはルカへの受け止めを禁止し、繋ぎ語からの開始を指示する。"""
        from lab_lounge.mcp_servers.ask_character import _generate_intro
        from lab_lounge.characters import get_character

        captured_prompts: list[str] = []

        def _capture_call_llm(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            result = MagicMock()
            result.text = "テスト導入"
            return result

        caller = get_character("mimi")
        target = get_character("sakura")

        with patch(
            "lab_lounge.llm.call_llm",
            side_effect=_capture_call_llm,
        ):
            _generate_intro(
                caller_char=caller,
                target_char=target,
                user_question="あなたたちはAIですか?",
                ask_index=2,
                previous_target_display="波心ちさめ",
            )

        assert len(captured_prompts) == 1
        second_prompt = captured_prompts[0]
        # ルカへの受け止めコメント要求が消えている
        assert "ルカの質問を受け止めるコメント" not in second_prompt
        # ルカ呼びかけ禁止が明示されている
        assert "ルカ呼びかけ" in second_prompt or "ルカ" in second_prompt
        # 直前 target が文中に含まれる
        assert "波心ちさめ" in second_prompt
        # 現 target が文中に含まれる
        assert target.display_name in second_prompt


class TestMCPServer:
    """MCP サーバーインスタンスの生成を検証する。"""

    def test_get_server_returns_fastmcp(self):
        """get_server() が FastMCP インスタンスを返すこと。"""
        from lab_lounge.mcp_servers.ask_character import get_server
        server = get_server()
        assert server is not None
