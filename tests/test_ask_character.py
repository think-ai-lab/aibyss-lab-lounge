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


# ─── Phase 0.5-B-β-1 commit 4: target キャラのステータス反映 ────────────
# WHY: Phase 0.5-B-α では caller のステータス反映 (= mimi: thinking → tool_calling
# → talking → ready) のみ実装され、ask_character 経由で話す target (= chisame の
# 音声が流れている数十秒) のステータスが ready のまま (= ask_character.py に
# set_status 呼出が 0 件) で HUD カードが更新されない穴があった。本 commit で
# target の THINKING (bridge filler 時) / TALKING (本応答 chunk 1 時、metadata: pose
# + full response_text) / READY (合成完了時) を反映し、HUD 網羅性を向上する。


class TestStatusReflection:
    """target キャラの HUD ステータス反映 (Phase 0.5-B-β-1 commit 4)。"""

    def test_set_ask_character_context_accepts_status_manager(self):
        """set_ask_character_context に status_manager 引数を渡せ、contextvar に格納される。

        WHY: graph._generation_node が _gen_status_manager を ask_character へ
        注入する経路。contextvar 経由で _ask_character_impl 内から取得できる
        ことを保証する (= 後段の THINKING/TALKING/READY 反映の前提)。
        """
        from lab_lounge.mcp_servers.ask_character import _status_manager_var

        mock_mgr = MagicMock()
        set_ask_character_context(
            caller_slug="mimi",
            status_manager=mock_mgr,
        )
        assert _status_manager_var.get() is mock_mgr

        # reset で None に戻ること (= 他テストへの漏れ防止、autouse fixture 連動)
        reset_ask_character_context()
        assert _status_manager_var.get() is None

    def test_target_thinking_reflected_at_bridge_filler(self, monkeypatch):
        """bridge filler 投入直前に target が THINKING で反映される。

        WHY: bridge filler (= 例: ちさめ「ええと…」) が再生される間、HUD カード
        も同期的に thinking (黄色) を表示する。bubble.update("thinking") と対を
        なす SSE 経路で、視聴者には「target が考え中」と分かる。
        """
        from lab_lounge.character_status import CharacterStatus

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_status = MagicMock()
        mock_tts_chunk = MagicMock()
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
            status_manager=mock_status,
        )

        # tts.synthesize は呼ばれた瞬間に return (= chunks 投入なし、bg_tts 完了)
        # → bridge filler 投入直前の THINKING 反映だけ走る (= 同期パスでテスト容易)
        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "テスト", "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize"):
            _ask_character_impl("chisame", "質問")

        # 期待: target=chisame で THINKING 反映が 1 回以上発生
        thinking_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.THINKING)
        ]
        assert len(thinking_calls) >= 1, (
            "bridge filler 投入時に target が THINKING で set_status されること"
        )

    def test_target_talking_reflected_at_first_chunk(self, monkeypatch):
        """本応答 chunk 1 投入時に target が TALKING + metadata で反映される。

        WHY: HUD カードを talking (緑) に切替、metadata (= pose + full
        response_text) で発話全文と立ち絵を視認可能にする。graph._tts_node が
        通常応答経路で渡す metadata 形と統一 (= V2 SSE 受信側で同じ shape)。
        """
        from lab_lounge.character_status import CharacterStatus
        from lab_lounge.mcp_servers.ask_character import wait_bg_tts_complete

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_status = MagicMock()
        mock_tts_chunk = MagicMock()
        session_id = "ss1"
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            status_manager=mock_status,
        )

        response_text = (
            '{"response": "データを分析しました", '
            '"emotion": {"happy": 30}, "speed": 100, "pose": "special_doya"}'
        )

        def fake_synth(*args, **kwargs):
            """target chunk として on_chunk_ready callback を 1 回呼ぶ。

            導入セリフ呼出 (= speaker=mimi) は callback 不要、本応答呼出
            (= speaker=chisame) で _wrapped_on_chunk_ready を 1 回起動する。
            """
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk and speaker == "chisame":
                on_chunk("file://chunk1.wav", "データを分析しました", True, "chisame")

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=response_text,
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            # bg_tts daemon thread の完了を確実に待つ (= wait_bg_tts_complete は
            # session_id に紐付いた completion event を join する設計)
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 期待: target=chisame で TALKING 反映 (metadata 付き) が 1 回
        talking_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.TALKING)
        ]
        assert len(talking_calls) == 1, (
            "本応答 chunk 1 時に target が TALKING で set_status されること"
        )
        # metadata が pose + text を含むこと
        metadata = talking_calls[0].kwargs.get("metadata")
        assert metadata is not None
        assert metadata.get("pose") == "special_doya"
        # Phase 0.5-B-β-3 commit 1: text は say_text (= JSON parse 後の response
        # 部分のみ) を期待する。JSON 全文ではない (= シナリオ 2 で観察した
        # 「HUD に JSON が表示される」不具合の修正)。
        assert metadata.get("text") == "データを分析しました"

    def test_target_talking_metadata_text_falls_back_to_raw_on_parse_failure(
        self, monkeypatch,
    ):
        """response_text が JSON でない場合、metadata.text は raw 文字列にフォールバック。

        WHY: _parse_voicepeak_json が失敗したケース (= 協働先 LLM が JSON 形式を
        返さなかった、あるいは structured_output 未指定で plain text 返却)。
        metadata.text を None にしてしまうと HUD に何も表示されないため、raw を
        fallback として使う (= 視認可能性を最優先、UI stuck 防止)。
        """
        import time

        from lab_lounge.character_status import CharacterStatus
        from lab_lounge.mcp_servers.ask_character import wait_bg_tts_complete

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_status = MagicMock()
        mock_tts_chunk = MagicMock()
        session_id = "ss-fallback"
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            status_manager=mock_status,
        )

        # JSON parse 不可な raw 文字列
        raw_response = "ただのテキスト応答 (JSON ではない)"

        def fake_synth(*args, **kwargs):
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk and speaker == "chisame":
                on_chunk("file://chunk1.wav", "ただの…", True, "chisame")

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=raw_response,
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        talking_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.TALKING)
        ]
        assert len(talking_calls) == 1
        metadata = talking_calls[0].kwargs.get("metadata")
        assert metadata is not None
        # JSON parse 失敗時は raw response_text にフォールバック
        assert metadata.get("text") == raw_response

    def test_target_ready_NOT_reflected_when_synth_normal(self, monkeypatch):
        """正常合成時は bg_tts_synthesize finally で READY 反映しない (Phase 0.5-B-β-3 commit 2)。

        WHY: bg_tts 合成完了 != 物理再生完了。シナリオ 2 で観察した「HUD で発話
        途中に灰色化する」不具合の修正。正常系の READY 反映は playback worker
        (= is_last chunk 物理再生完了時) に移動。bg_tts_synthesize の finally では
        READY 反映しない (= playback worker 経由で適切なタイミングに反映される)。
        """
        from lab_lounge.character_status import CharacterStatus
        from lab_lounge.mcp_servers.ask_character import wait_bg_tts_complete

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_status = MagicMock()
        mock_tts_chunk = MagicMock()
        session_id = "ss1-normal"
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            status_manager=mock_status,
        )

        # tts.synthesize は単純 return (= 合成成功シナリオ、例外なし)
        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "テスト応答", "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize"):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 期待: 正常系では target=chisame で READY 反映が呼ばれない (= 0 回)。
        # playback worker 経由で is_last 再生完了時に反映される設計のため。
        ready_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.READY)
        ]
        assert len(ready_calls) == 0, (
            f"正常系では bg_tts_synthesize finally で READY 反映されないこと "
            f"(playback worker 経由に移動): ready_calls={ready_calls}"
        )

    def test_target_ready_fallback_when_synth_exception(self, monkeypatch):
        """合成例外時は bg_tts_synthesize finally で fallback READY 反映 (Phase 0.5-B-β-3 commit 2)。

        WHY: tts.synthesize が例外を投げた場合、chunks が playback queue に入らない
        ため、playback worker 経由の READY 反映が走らない。HUD カードが talking
        のまま stuck するのを防ぐため、本 finally で fallback として READY 反映する。
        """
        from lab_lounge.character_status import CharacterStatus
        from lab_lounge.mcp_servers.ask_character import wait_bg_tts_complete

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_status = MagicMock()
        mock_tts_chunk = MagicMock()
        session_id = "ss1-exc"
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            status_manager=mock_status,
        )

        # tts.synthesize が例外を投げる (= 合成失敗、VOICEPEAK クラッシュ等)
        def fake_synth_raises(*args, **kwargs):
            raise RuntimeError("VOICEPEAK 合成失敗")

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "テスト応答", "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth_raises):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 期待: 例外系では fallback で target=chisame の READY 反映が 1 回呼ばれる
        ready_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.READY)
        ]
        assert len(ready_calls) >= 1, (
            "tts.synthesize 例外時は finally で fallback として target が "
            "READY で set_status されること (UI stuck 防止)"
        )

    def test_status_manager_none_skips_reflection(self, monkeypatch):
        """status_manager=None なら set_status は一切呼ばれない (後方互換)。

        WHY: Phase 0.5-B-α 以前の呼出元 (= status_manager 引数を渡さない) で
        ask_character が動作することを保証する。後方互換性 + 「注入忘れ」の
        フォールバック挙動 (= ステータス反映 no-op、bug にならない)。
        """
        import time

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        mock_tts_chunk = MagicMock()
        # status_manager 渡さず (= 旧呼出パターン)
        set_ask_character_context(
            on_tts_chunk=mock_tts_chunk,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        # set_status を spy するため CharacterStatusManager 全体を MagicMock 化して
        # `_status_manager_var.get()` が None を返す状態を作る (= 上の
        # set_ask_character_context で status_manager 未指定)
        # → 内部の `if _status_manager_for_target is not None:` ガードで
        #    set_status が呼ばれないことを確認する
        from lab_lounge.mcp_servers.ask_character import _status_manager_var

        assert _status_manager_var.get() is None  # 前提確認

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "x", "emotion": {"happy": 0}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize"):
            # 例外無く完了すれば OK (= None ガードで no-op 経路を通過)
            result = _ask_character_impl("chisame", "質問")
            time.sleep(0.2)

        # 例外なく応答テキストが返る
        assert result is not None
        assert "テキスト" in result or "x" in result or "chisame" in result.lower() or "ちさめ" in result


# ─── Phase 0.5-B-β-1 commit 5: BG LLM → ask_character → playback queue 貫通 ──
# end-to-end wiring を統合的に保証する。pipeline.py / run_loop.py / graph.py /
# ask_character.py の修正が全て繋がっていれば、run_pipeline_llm_only 経由で渡された
# callback が ask_character ツール起動時に呼ばれる (= A1 主機能修正)。callback=None
# なら TTS スキップ (= バグ前の状態を再現する後方互換テスト)。
#
# 注意: dummy mode (= L2_USE_REAL_LLM 未設定) では graph._generation_node の
# Agent 実行 (= ask_character ツール呼出) に入らないため、テストでは
# set_ask_character_context で contextvars を直接セット → _ask_character_impl を
# 直接呼ぶ形で wiring の最終セグメントを確認する。pipeline.py の引数 →
# initial_state → set_ask_character_context までの上流 wiring は β-1-1 / β-1-2 の
# tests/test_pipeline.py で検証済み。run_pipeline_llm_only 自体の events 不変性
# (= tts.done が含まれないこと) は TestRunPipelineLlmOnly に追加した
# test_no_tts_done_event_with_callback_set で別途検証。


class TestLlmOnlyTtsPenetration:
    """end-to-end wiring 統合テスト: BG LLM 経路の対話 TTS callback 起動 (Phase 0.5-B-β-1 commit 5)。"""

    def test_callback_invoked_when_set(self, monkeypatch):
        """contextvars 経由の callback が ask_character の対話 TTS で起動される。

        WHY: pipeline.py / run_loop.py / graph.py / ask_character.py の wiring が
        全て繋がっていれば、ask_character ツール起動時に渡された callback が
        呼ばれる。複数回 (= bridge filler + 本応答 chunks) で呼ばれることで、
        バグ修正前 (= 0 回) と区別。
        """
        from lab_lounge.mcp_servers.ask_character import wait_bg_tts_complete

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        callback_invocations: list[tuple] = []

        def mock_cb(url, chunk_text, is_last, character):
            callback_invocations.append((url, chunk_text, is_last, character))

        session_id = "ss1"
        set_ask_character_context(
            on_tts_chunk=mock_cb,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
        )

        response_text = (
            '{"response": "テスト応答", "emotion": {"happy": 50}, '
            '"speed": 100, "pose": "neutral"}'
        )

        def fake_synth(*args, **kwargs):
            """on_chunk_ready が渡された TTS 呼出 (= 本応答) で chunk 1 投入。"""
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk:
                on_chunk(f"file://{speaker}.wav", "テスト", True, speaker)

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=response_text,
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # ask_character.py:441 の bridge filler 投入 (= 1 回、空 text で chunk 投入)
        # + 本応答 chunk 1 投入 (= fake_synth → _wrapped_on_chunk_ready → on_tts_chunk)
        # の最低 2 回 (= 修正前は 0 回) 呼ばれる
        assert len(callback_invocations) >= 2, (
            f"callback が 2 回以上呼ばれること (実際: {len(callback_invocations)} 回)"
        )

    def test_no_callback_skips_tts(self, monkeypatch):
        """callback=None (= Phase 0.5-A 以前のバグ状態) で tts.synthesize 呼ばれない。

        WHY: A1 バグの本体 (= 「導入セリフ + 協働応答 TTS が完全スキップ」) を
        再現する後方互換テスト。本 commit 群のロールバック (= 部分 revert) 後の
        挙動を保証することで、partial revert デバッグパターン (= memory
        feedback_partial_revert_debug_pattern.md) の整合性を確保する。
        ask_character.py:376 / :408 の gating `if on_tts_chunk and use_real_tts:`
        で False になり、tts.synthesize は一切呼ばれない。
        """
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        # callback=None で contextvars セット (= バグ再現状態)
        set_ask_character_context(
            on_tts_chunk=None,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "x", "emotion": {"happy": 0}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize") as mock_synth:
            _ask_character_impl("chisame", "質問")

        # gating False → tts.synthesize 一切呼ばれない (= バグ前の状態)
        mock_synth.assert_not_called()


# ─── Phase 0.5-B-β-2 commit 2: cancel_bg_tts API + bg_tts キャンセルガード ────
# 却下/lapse 時に ask_character の bg_tts daemon thread を阻止する経路。run_loop
# の on_handraise_close callback (β-2-3 で実装) が cancel_bg_tts(session_id) を
# 呼ぶと、該当 session の Event が set される。導入セリフ TTS / _wrapped_on_chunk_ready
# / _bg_tts_synthesize の 3 箇所で is_set() チェックして以降の処理を skip する。
# 既に subprocess 中の VOICEPEAK 合成は止められないが、未起動 thread / 未投入
# chunk / 導入セリフ起動を阻止することで、案 A の音声漏れを最小化する。


class TestCancelBgTts:
    """cancel_bg_tts API + bg_tts キャンセルガード (Phase 0.5-B-β-2 commit 2)。"""

    def test_cancel_bg_tts_returns_zero_for_unknown_session(self):
        """未登録 session_id で 0 を返す (= flag が無いので set もしない)。

        WHY: dispatcher の on_handraise_close から呼ばれる際、稀に session_id が
        既にクリーンアップ済 (= 別ターン開始等) のケースで安全に no-op で帰る。
        """
        from lab_lounge.mcp_servers.ask_character import cancel_bg_tts

        result = cancel_bg_tts("unknown_session_xyz")
        assert result == 0

    def test_cancel_bg_tts_empty_session_no_op(self):
        """空文字 session_id で no-op で 0 を返す。

        WHY: テスト等で session_id 未指定 (= 空文字) で呼ばれるパターンに対応。
        """
        from lab_lounge.mcp_servers.ask_character import cancel_bg_tts

        result = cancel_bg_tts("")
        assert result == 0

    def test_cancel_bg_tts_sets_flag_and_returns_count(self):
        """set_ask_character_context 後、cancel_bg_tts で flag set + count 返却。

        WHY: 登録済 session に対する cancel の本来の動作。flag set されると、
        以降の bg_tts ガード (= ask_character.py の 3 箇所) が False → return で
        skip 動作する。count は影響範囲を示す診断値 (= 登録済 bg_tts events 数)。
        """
        import threading

        from lab_lounge.mcp_servers.ask_character import (
            _ask_state_lock,
            _bg_cancel_flags,
            _register_bg_tts_event,
            cancel_bg_tts,
        )

        session_id = "ss-cancel-test"
        set_ask_character_context(
            common={"session_id": session_id, "stream_id": "s1", "trace_id": "t1"},
        )

        # bg_tts events を 2 つ登録 (= bg_tts thread 2 つ起動済の状態を模擬)
        ev1 = threading.Event()
        ev2 = threading.Event()
        _register_bg_tts_event(session_id, ev1)
        _register_bg_tts_event(session_id, ev2)

        # cancel 実行
        result = cancel_bg_tts(session_id)

        # flag が set されている
        with _ask_state_lock:
            assert _bg_cancel_flags[session_id].is_set()
        # 戻り値 = 登録済 bg_tts events 数 (= 影響範囲指標)
        assert result == 2

    def test_cancel_blocks_subsequent_bg_tts_synthesize(self, monkeypatch):
        """cancel_bg_tts 後の _ask_character_impl で tts.synthesize 起動が skip される。

        WHY: end-to-end の整合性確認。flag set 状態で _ask_character_impl を呼ぶと、
        ask_character.py:376 (導入セリフ) と _bg_tts_synthesize の冒頭ガードで
        tts.synthesize が一切呼ばれない (= VOICEPEAK 合成も起動しない、CPU/GPU
        浪費なし)。実走では「却下後にミミ様の問いかけが流れない」「ちさめの応答も
        流れない」状態になる (= 案 A の音声漏れの最小化)。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _ask_character_impl,
            cancel_bg_tts,
            wait_bg_tts_complete,
        )

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        session_id = "ss-cancel-integ"
        set_ask_character_context(
            on_tts_chunk=MagicMock(),
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
        )
        # 予め cancel (= 「ask_character 起動時には既に却下されている」シナリオ)
        cancel_bg_tts(session_id)

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "x", "emotion": {"happy": 0}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize") as mock_synth:
            _ask_character_impl("chisame", "質問")
            # bg_tts thread の完了 (= cancel ガードで早期 return) を待つ
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 導入セリフ (caller=mimi) も本応答 (target=chisame) も両方 skip される
        mock_synth.assert_not_called()


class TestBgChunkBuffers:
    """Phase 0.5-D-1a: BG LLM 経路用の chunk buffer 機構の単体テスト。

    本 phase (D-1a) では _bg_chunk_buffers dict と操作 API (_append/_drain/_peek)
    を追加するのみ (= 未配線、_wrapped_on_chunk_ready からは呼ばれない)。
    D-1b で `set_ask_character_context(defer_chunks=True)` 時の defer モード分岐
    から本 API を呼ぶよう配線する。

    本クラスでは API 単体の挙動 (= 蓄積/atomic 取得 + クリア/カウント/初期化) を
    検証する。
    """

    def test_append_bg_chunk_stores_in_session_dict(self):
        """_append_bg_chunk で session_id ごとに chunks が累積する。"""
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _peek_bg_chunks_count,
        )
        chunk = {"url": "file:///a.wav", "text": "hello",
                 "is_last": False, "character": "mimi", "pose": None}
        _append_bg_chunk("session-A", chunk)
        assert _peek_bg_chunks_count("session-A") == 1
        _append_bg_chunk("session-A", chunk)
        assert _peek_bg_chunks_count("session-A") == 2

    def test_drain_bg_chunks_returns_and_clears(self):
        """_drain_bg_chunks は chunks list を順序保証で返し、同時に dict から削除する。

        WHY atomic: 並行 daemon thread からの append と承認 callback からの drain
        の race を防ぐため、_ask_state_lock 配下で取得 + クリアを 1 操作にする
        (= 取得後の追加チャンクは新 buffer に蓄積される、見逃しなし)。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _drain_bg_chunks, _peek_bg_chunks_count,
        )
        c1 = {"url": "1", "text": "a"}
        c2 = {"url": "2", "text": "b"}
        _append_bg_chunk("session-B", c1)
        _append_bg_chunk("session-B", c2)
        result = _drain_bg_chunks("session-B")
        # 順序保証 (= append 順)
        assert result == [c1, c2]
        # drain 後は完全クリア (= 2 度目 drain は空)
        assert _peek_bg_chunks_count("session-B") == 0
        assert _drain_bg_chunks("session-B") == []

    def test_drain_bg_chunks_unknown_session_returns_empty_list(self):
        """未知 session_id の drain は [] を返す (= 未蓄積/未配線時の後方互換)。"""
        from lab_lounge.mcp_servers.ask_character import _drain_bg_chunks
        assert _drain_bg_chunks("session-unknown") == []

    def test_empty_session_id_no_op(self):
        """空文字 session_id では append/drain/peek 全部 no-op (= 既存 dict 群と同じパターン)。

        WHY: テストやエッジケースで session_id 未指定で呼ばれることがあるため、
        既存の _ask_counts/_previous_targets 等と同じく空文字は副作用なし扱い。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _bg_chunk_buffers, _drain_bg_chunks, _peek_bg_chunks_count,
        )
        _append_bg_chunk("", {"url": "x"})
        # 空文字 session_id では dict 自体に何も登録されない
        assert "" not in _bg_chunk_buffers
        assert _peek_bg_chunks_count("") == 0
        assert _drain_bg_chunks("") == []

    def test_buffer_resets_on_set_ask_character_context(self):
        """set_ask_character_context (= ターン開始時) で前ターンの buffer がリセットされる。

        WHY: 前ターン残留 buffer が次ターンに混入することを構造的に防ぐ
        (= _reset_session_state 内の _bg_chunk_buffers.pop で担保)。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _peek_bg_chunks_count,
        )
        _append_bg_chunk("sess-C", {"url": "leaked"})
        assert _peek_bg_chunks_count("sess-C") == 1
        # set_ask_character_context で同一 session_id 指定 → _reset_session_state 経由でクリア
        set_ask_character_context(common={"session_id": "sess-C"})
        assert _peek_bg_chunks_count("sess-C") == 0

    def test_buffer_isolated_per_session_id(self):
        """異なる session_id 間で buffer が完全分離される (= dict ベース、明示テスト)。"""
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _drain_bg_chunks, _peek_bg_chunks_count,
        )
        _append_bg_chunk("sess-X", {"url": "x1"})
        _append_bg_chunk("sess-Y", {"url": "y1"})
        _append_bg_chunk("sess-Y", {"url": "y2"})
        assert _peek_bg_chunks_count("sess-X") == 1
        assert _peek_bg_chunks_count("sess-Y") == 2
        # X drain しても Y には影響なし
        assert _drain_bg_chunks("sess-X") == [{"url": "x1"}]
        assert _peek_bg_chunks_count("sess-Y") == 2

    def test_reset_ask_character_context_clears_all_session_buffers(self):
        """reset_ask_character_context は全 session の buffer を一括クリーンする。

        WHY: テスト間の漏れ防止 (= autouse fixture の _reset_context で全 session
        buffer が空になっていることを担保、case 同士の干渉を防ぐ)。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _append_bg_chunk, _bg_chunk_buffers, reset_ask_character_context,
        )
        _append_bg_chunk("sess-1", {"url": "1"})
        _append_bg_chunk("sess-2", {"url": "2"})
        assert len(_bg_chunk_buffers) == 2
        reset_ask_character_context()
        assert len(_bg_chunk_buffers) == 0


class TestDeferChunksMode:
    """Phase 0.5-D-1b: defer_chunks フラグによる _wrapped_on_chunk_ready 分岐テスト。

    set_ask_character_context(defer_chunks=True) で起動された ask_character の本応答
    chunks (= _wrapped_on_chunk_ready 経由) は _bg_chunk_buffers に蓄積され、
    on_tts_chunk callback には流れない。通常応答経路 (defer_chunks=False、default)
    では既存挙動 (= 即時 callback) を完全維持する。

    【BG LLM 経路と通常応答経路の分岐根拠】
    通常応答経路では ask_character の戻り値文字列を caller LLM が読んで「target が
    既に話した前提でリアクション」を組み立てる。即時再生でないと caller のリアクション
    が target の発話前に流れる逆順バグになる。BG LLM 経路では承認時まで再生を遅らせて
    いいため、ターン跨ぎ問題回避のため buffer 経路にする。
    """

    def test_set_ask_character_context_default_is_false(self):
        """set_ask_character_context の defer_chunks 引数の default は False (= 後方互換)。"""
        from lab_lounge.mcp_servers.ask_character import _defer_chunks_var
        set_ask_character_context()
        assert _defer_chunks_var.get() is False

    def test_set_ask_character_context_accepts_defer_chunks_true(self):
        """set_ask_character_context(defer_chunks=True) で contextvars に True がセットされる。"""
        from lab_lounge.mcp_servers.ask_character import _defer_chunks_var
        set_ask_character_context(defer_chunks=True)
        assert _defer_chunks_var.get() is True

    def test_reset_ask_character_context_resets_defer_flag(self):
        """reset_ask_character_context で defer_chunks も False にリセットされる。"""
        from lab_lounge.mcp_servers.ask_character import (
            _defer_chunks_var, reset_ask_character_context,
        )
        set_ask_character_context(defer_chunks=True)
        assert _defer_chunks_var.get() is True
        reset_ask_character_context()
        assert _defer_chunks_var.get() is False

    def test_defer_true_appends_response_chunks_to_buffer(self, monkeypatch):
        """defer_chunks=True で target 本応答 chunks は buffer に蓄積、callback には流れない。

        WHY: BG LLM 経路では target (= 質問先キャラ) の応答 TTS を承認時まで再生
        遅延させる。bridge filler や caller (= 質問する側) の導入セリフ TTS は
        on_tts_chunk 直接呼出 (= _wrapped_on_chunk_ready 経由でない、ask_character.py
        内の独立経路) なので defer モードでも生 callback に流れる (= 設計通り、UX
        として「考え中の繋ぎ」を即時再生する)。本テストでは「target 応答のみが
        buffer に行く」ことを character フィルタで区別検証する。

        【テスト隔離】
        _generate_intro は内部で call_llm (= API key 必要) を呼ぶため、本テストの
        範囲では mock して空文字を返させる (= 導入セリフ TTS skip パス、
        ask_character.py:591 の `if on_tts_chunk and use_real_tts and caller_char
        and intro_response_text:` が False に)。これにより fake_synth は target
        本応答の 1 回だけ呼ばれ、テスト挙動が決定的になる。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _drain_bg_chunks, wait_bg_tts_complete,
        )
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        callback_invocations: list[tuple] = []

        def mock_cb(url, chunk_text, is_last, character):
            callback_invocations.append((url, chunk_text, is_last, character))

        session_id = "defer_test_buffer"
        set_ask_character_context(
            on_tts_chunk=mock_cb,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            defer_chunks=True,
        )

        response_text = (
            '{"response": "テスト応答", "emotion": {"happy": 50}, '
            '"speed": 100, "pose": "doya"}'
        )

        def fake_synth(*args, **kwargs):
            """on_chunk_ready 経由で 2 件 chunk 投入 (= 本応答 = target=chisame)。"""
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk:
                on_chunk(f"file://{speaker}_c1.wav", "テスト", False, speaker)
                on_chunk(f"file://{speaker}_c2.wav", "応答", True, speaker)

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=response_text,
        ), patch(
            "lab_lounge.mcp_servers.ask_character._generate_intro",
            return_value="",  # 導入セリフ skip → 本応答 TTS のみ実行
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # target = chisame の chunks (= 本応答) は callback に流れない (= defer モード)
        target_callbacks = [c for c in callback_invocations if c[3] == "chisame"
                            and c[1] != ""]  # text 非空 = bridge filler を除外
        assert len(target_callbacks) == 0, (
            f"defer=True で target 本応答 chunks は callback に流れない "
            f"(実際: {len(target_callbacks)} 件、{target_callbacks})"
        )
        # buffer に target 本応答 2 件蓄積される
        chunks = _drain_bg_chunks(session_id)
        assert len(chunks) == 2, (
            f"buffer に本応答 chunks 2 件蓄積される (実際: {len(chunks)} 件)"
        )
        for c in chunks:
            assert c["character"] == "chisame", (
                f"buffer chunks の character は target=chisame "
                f"(実際: {c['character']})"
            )

    def test_defer_false_preserves_existing_callback_behavior(self, monkeypatch):
        """defer_chunks=False (= 通常応答経路) で chunks は既存通り callback 経由で流れる。

        WHY: 通常応答経路の不変保証。Phase 0.5-D-1b で BG LLM 経路だけ分岐させ、
        通常応答経路は完全に既存挙動を維持していることを担保する (= TestStatusReflection
        等 36 件への影響なし)。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _peek_bg_chunks_count, wait_bg_tts_complete,
        )
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        callback_invocations: list[tuple] = []

        def mock_cb(url, chunk_text, is_last, character):
            callback_invocations.append((url, chunk_text, is_last, character))

        session_id = "non_defer_test"
        set_ask_character_context(
            on_tts_chunk=mock_cb,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            defer_chunks=False,  # 既存挙動 (= 通常応答経路)
        )

        response_text = (
            '{"response": "テスト", "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}'
        )

        def fake_synth(*args, **kwargs):
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk:
                on_chunk(f"file://{speaker}_c1.wav", "テスト応答", True, speaker)

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=response_text,
        ), patch(
            "lab_lounge.mcp_servers.ask_character._generate_intro",
            return_value="",  # 導入セリフ skip でテスト決定性確保
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 本応答 chunks (= target=chisame, text 非空) が callback で受け取られる (= 既存挙動)
        response_callbacks = [c for c in callback_invocations if c[3] == "chisame"
                              and c[1] != ""]
        assert len(response_callbacks) >= 1, (
            f"defer=False で本応答 chunks は callback 経由 "
            f"(実際: {len(response_callbacks)} 件)"
        )
        # buffer は空 (= 通常応答経路では使われない)
        assert _peek_bg_chunks_count(session_id) == 0, (
            f"defer=False で buffer は使われない "
            f"(実際: {_peek_bg_chunks_count(session_id)} 件)"
        )

    def test_defer_true_chunk_dict_carries_pose_directly(self, monkeypatch):
        """defer モードで pose は chunk dict に直接埋め込まれる (= on_pose_ready callback でなく)。

        WHY: 通常応答経路は on_pose_ready callback で run_loop の _pending_poses に
        セット → 直後の _on_tts_chunk で pop → chunk dict にセット、という流れだが、
        defer 経路では run_loop closure に依存しない (= ターン跨ぎ独立性のため)。
        chunk dict の "pose" に直接埋め込み、専用 mini worker (= D-2 で配線) が
        OBS 立ち絵切替する経路を担保する。
        """
        from lab_lounge.mcp_servers.ask_character import (
            _drain_bg_chunks, wait_bg_tts_complete,
        )
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        pose_callback_invocations: list[tuple] = []

        def mock_pose_cb(slug, pose):
            pose_callback_invocations.append((slug, pose))

        session_id = "defer_test_pose"
        set_ask_character_context(
            on_tts_chunk=lambda *a: None,  # bridge filler 用、no-op
            on_pose_ready=mock_pose_cb,
            tts_output_dir="/tmp/audio",
            caller_slug="mimi",
            common={"stream_id": "s1", "session_id": session_id, "trace_id": "t1"},
            defer_chunks=True,
        )

        # 本応答に pose=special_doya を含める
        response_text = (
            '{"response": "ふふ、その通りですわ", "emotion": {"happy": 80}, '
            '"speed": 100, "pose": "special_doya"}'
        )

        def fake_synth(*args, **kwargs):
            on_chunk = kwargs.get("on_chunk_ready")
            speaker = kwargs.get("speaker")
            if on_chunk:
                on_chunk(f"file://{speaker}_c1.wav", "ふふ", False, speaker)
                on_chunk(f"file://{speaker}_c2.wav", "その通り", True, speaker)

        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value=response_text,
        ), patch(
            "lab_lounge.mcp_servers.ask_character._generate_intro",
            return_value="",  # 導入セリフ skip でテスト決定性確保
        ), patch("lab_lounge.tts.synthesize", side_effect=fake_synth):
            _ask_character_impl("chisame", "質問")
            wait_bg_tts_complete(session_id, timeout=5.0)

        # defer モードでは on_pose_ready callback は呼ばれない
        # (= chunk dict に直接埋込が代替経路)
        assert len(pose_callback_invocations) == 0, (
            f"defer=True で on_pose_ready callback は呼ばれない "
            f"(実際: {pose_callback_invocations})"
        )
        # buffer の first_chunk に pose=special_doya が埋め込まれている
        chunks = _drain_bg_chunks(session_id)
        assert len(chunks) == 2
        assert chunks[0].get("pose") == "special_doya", (
            f"first_chunk の pose は special_doya (実際: {chunks[0].get('pose')})"
        )
        # 後続 chunk は pose 未指定 (= 維持の意味、実装で None or 欠如)
        assert chunks[1].get("pose") in (None, ""), (
            f"後続 chunk の pose は未指定 (実際: {chunks[1].get('pose')})"
        )


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
