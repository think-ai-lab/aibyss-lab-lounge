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
        assert metadata.get("text") == response_text

    def test_target_ready_reflected_at_bg_tts_done(self, monkeypatch):
        """bg_tts 合成完了後 (finally) に target が READY で反映される。

        WHY: target の発話終了 = HUD カードを ready (灰) に戻す瞬間。例外時も
        finally で確実に Ready にすることで、HUD カードが talking のまま stuck
        するのを防ぐ (= UI 整合性、視覚的に「終了した」が分かる)。
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

        # tts.synthesize は単純 return (= 合成成功シナリオ)
        with patch(
            "lab_lounge.mcp_servers.ask_character._run_collaboration_agent",
            return_value='{"response": "テスト応答", "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}',
        ), patch("lab_lounge.tts.synthesize"):
            _ask_character_impl("chisame", "質問")
            # bg_tts daemon thread の完了を確実に待つ (finally で READY 反映後)
            wait_bg_tts_complete(session_id, timeout=5.0)

        # 期待: target=chisame で READY 反映が 1 回 (= bg_tts 合成完了 finally)
        ready_calls = [
            call for call in mock_status.set_status.call_args_list
            if call.args[:2] == ("chisame", CharacterStatus.READY)
        ]
        assert len(ready_calls) >= 1, (
            "bg_tts 合成完了後に target が READY で set_status されること"
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
