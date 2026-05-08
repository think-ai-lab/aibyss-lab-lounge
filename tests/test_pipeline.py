"""
test_pipeline.py — pipeline.run_pipeline テスト

publisher をモックして Redis 接続なしで検証する。
"""

from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.pipeline as pipeline_mod
from lab_lounge.pipeline import run_pipeline


COMMON = dict(
    stream_id="stream-pipe-001",
    session_id="sess-pipe-001",
    trace_id="trace-pipe-001",
)


@pytest.fixture()
def mock_publish():
    """bus.publish をモックして XADD を発行しない。"""
    with patch.object(pipeline_mod, "publish", return_value="1-0") as m:
        yield m


class TestRunPipeline:
    def test_returns_three_events(self, mock_publish):
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert len(result.events) == 3

    def test_event_types_in_order(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final", "tts.done"]

    def test_publish_called_five_times(self, mock_publish):
        """3 メインイベント + 2 bubble.update (thinking/answering) = 5 回。

        Sprint Axis D Block 1: done は run_loop の再生 worker が発行。
        Sprint Axis D Block 3: retrieval ノード削除で bubble("thinking") が 1 つ減少。
        routing が "thinking"、generation が "answering" の 2 回。
        """
        run_pipeline("hello", **COMMON)
        assert mock_publish.call_count == 5

    def test_llm_links_utterance(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        utt_id = result.events[0]["event_id"]
        llm_links = result.events[1]["links"]
        assert utt_id in llm_links

    def test_tts_links_llm(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        llm_id = result.events[1]["event_id"]
        tts_links = result.events[2]["links"]
        assert llm_id in tts_links

    def test_llm_text_contains_input(self, mock_publish):
        result = run_pipeline("天気の話", **COMMON)
        assert "天気の話" in result.events[1]["payload"]["text"]

    def test_tts_text_matches_llm_text(self, mock_publish):
        result = run_pipeline("天気の話", **COMMON)
        assert result.events[2]["payload"]["text"] == result.events[1]["payload"]["text"]

    def test_result_ids_match_common(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        assert result.stream_id == COMMON["stream_id"]
        assert result.session_id == COMMON["session_id"]
        assert result.trace_id == COMMON["trace_id"]

    def test_seq_increments(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        seqs = [ev["seq"] for ev in result.events]
        assert seqs == [0, 1, 2]

    def test_source_all_lab_lounge(self, mock_publish):
        result = run_pipeline("hello", **COMMON)
        for ev in result.events:
            assert ev["source"] == "lab-lounge"

    def test_no_stream_idx_in_any_event(self, mock_publish):
        """Guardrail G-1: どのイベントにも stream_idx を含めない。"""
        result = run_pipeline("hello", **COMMON)
        for ev in result.events:
            assert "stream_idx" not in ev

    def test_utterance_text_exact_japanese(self, mock_publish):
        """utterance.final の payload.text は入力テキストと完全一致する"""
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[0]["payload"]["text"] == "今日の天気を教えて"

    def test_llm_text_exact_japanese(self, mock_publish):
        """llm.final の payload.text はダミープレフィックス + 入力テキストと完全一致する"""
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[1]["payload"]["text"] == "ダミー応答: 今日の天気を教えて"

    def test_tts_text_exact_japanese(self, mock_publish):
        """ダミーモード: tts.done の payload.text は llm.final と同一テキストになる"""
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[2]["payload"]["text"] == "ダミー応答: 今日の天気を教えて"


class TestRunPipelineRealMode:
    """L2_USE_REAL_LLM=true のとき graph.py 経由で real LLM を呼ぶテスト"""

    @pytest.fixture()
    def mock_real_mode(self, monkeypatch):
        """L2_USE_REAL_LLM=true を定義する"""
        monkeypatch.setenv("L2_USE_REAL_LLM", "true")
        monkeypatch.setenv("L2_LLM_PROVIDER", "openai")
        monkeypatch.setenv("L2_LLM_MODEL", "gpt-4o-mini")

    @pytest.fixture()
    def fake_llm_result(self):
        from lab_lounge.llm import LLMResult
        return LLMResult(
            text="リアル LLM 応答テスト",
            model="gpt-4o-mini",
            input_tokens=15,
            output_tokens=8,
            latency_ms=250,
            finish_reason="stop",
        )

    def test_real_mode_llm_text_from_graph(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[1]["payload"]["text"] == "リアル LLM 応答テスト"

    def test_real_mode_model_in_payload(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[1]["payload"]["model"] == "gpt-4o-mini"

    def test_real_mode_tokens_in_payload(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        payload = result.events[1]["payload"]
        assert payload["input_tokens"] == 15
        assert payload["output_tokens"] == 8

    def test_real_mode_latency_ms_in_payload(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[1]["payload"]["latency_ms"] == 250

    def test_real_mode_finish_reason_in_payload(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[1]["payload"]["finish_reason"] == "stop"

    def test_real_mode_rag_used_false(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[1]["payload"]["rag_used"] is False

    def test_real_mode_still_three_events(self, mock_real_mode, mock_publish, fake_llm_result):
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert len(result.events) == 3

    def test_real_mode_no_stream_idx(self, mock_real_mode, mock_publish, fake_llm_result):
        """Guardrail G-1: real mode でも stream_idx を含めない"""
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        for ev in result.events:
            assert "stream_idx" not in ev

    def test_real_mode_run_graph_called_with_env_args(self, mock_real_mode, mock_publish, fake_llm_result):
        """run_graph に ENV から読んだ model/provider が渡る"""
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result) as mock_graph:
            run_pipeline("テスト質問", **COMMON)
        call_kw = mock_graph.call_args.kwargs
        assert call_kw["model"] == "gpt-4o-mini"
        assert call_kw["provider"] == "openai"
        assert "run_metadata" in call_kw

    def test_real_mode_tts_text_matches_llm_text(self, mock_real_mode, mock_publish, fake_llm_result):
        """tts.done の text は real LLM 出力と一致する"""
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト質問", **COMMON)
        assert result.events[2]["payload"]["text"] == "リアル LLM 応答テスト"

    def test_dummy_mode_preserved(self, mock_publish, monkeypatch):
        """L2_USE_REAL_LLM が未設定のとき既存のダミー動作を維持"""
        monkeypatch.delenv("L2_USE_REAL_LLM", raising=False)
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[1]["payload"]["text"] == "ダミー応答: 今日の天気を教えて"
        assert result.events[1]["payload"]["model"] == "dummy-1.0"


class TestRunPipelineRealTTSMode:
    """L2_USE_REAL_TTS=true のとき tts.py 経由で real TTS を呼ぶテスト"""

    @pytest.fixture()
    def mock_real_tts_mode(self, monkeypatch):
        """L2_USE_REAL_TTS=true を設定する"""
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")
        monkeypatch.setenv("L2_TTS_PROVIDER", "edge_tts")
        monkeypatch.setenv("L2_TTS_VOICE", "ja-JP-NanamiNeural")
        monkeypatch.setenv("L2_TTS_SPEAKER", "Nanami")
        monkeypatch.setenv("L2_TTS_OUTPUT_DIR", "./data/audio")

    @pytest.fixture()
    def fake_tts_result(self):
        from lab_lounge.tts import TTSResult
        return TTSResult(
            audio_url="file:///tmp/tts-test.mp3",
            duration_ms=2500,
            voice="ja-JP-NanamiNeural",
            format="mp3",
            sample_rate=24000,
            speaker="Nanami",
        )

    def test_real_tts_audio_url_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["audio_url"] == "file:///tmp/tts-test.mp3"

    def test_real_tts_duration_ms_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["duration_ms"] == 2500

    def test_real_tts_voice_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["voice"] == "ja-JP-NanamiNeural"

    def test_real_tts_format_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["format"] == "mp3"

    def test_real_tts_sample_rate_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["sample_rate"] == 24000

    def test_real_tts_speaker_in_payload(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["speaker"] == "Nanami"

    def test_real_tts_still_three_events(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_real_tts_no_stream_idx(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        """Guardrail G-1: real TTS mode でも stream_idx を含めない"""
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        for ev in result.events:
            assert "stream_idx" not in ev

    def test_real_tts_synthesize_called_with_character_args(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        """synthesize にデフォルトキャラクターの TTS 設定が渡る"""
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result) as mock_synth:
            run_pipeline("テスト", **COMMON)
        # デフォルトキャラクター (octamaid) の設定が使われる
        mock_synth.assert_called_once_with(
            "ダミー応答: テスト",
            provider="voicevox",
            voice="89",
            speaker="octamaid",
            output_dir="./data/audio",
            on_chunk_ready=None,
        )

    def test_real_tts_text_matches_llm_text(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        """tts.done の text は LLM 出力テキストと一致する"""
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[2]["payload"]["text"] == result.events[1]["payload"]["text"]

    def test_dummy_tts_mode_preserved(self, mock_publish, monkeypatch):
        """L2_USE_REAL_TTS が未設定のとき dummy audio_url が使われる"""
        monkeypatch.delenv("L2_USE_REAL_TTS", raising=False)
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[2]["payload"]["audio_url"] == "file://dummy/audio.opus"
        assert result.events[2]["payload"]["voice"] == "dummy-voice"


class TestRunPipelineWithUtteranceMeta:
    """utterance_meta を渡したとき utterance.final payload に反映されるテスト"""

    def test_utterance_meta_confidence_reflected(self, mock_publish):
        """utterance_meta.confidence が payload に反映される"""
        result = run_pipeline(
            "テスト", utterance_meta={"confidence": 0.87}, **COMMON
        )
        assert result.events[0]["payload"]["confidence"] == 0.87

    def test_utterance_meta_lang_reflected(self, mock_publish):
        """utterance_meta.lang が payload に反映される"""
        result = run_pipeline(
            "テスト", utterance_meta={"lang": "ja"}, **COMMON
        )
        assert result.events[0]["payload"]["lang"] == "ja"

    def test_utterance_meta_duration_ms_reflected(self, mock_publish):
        """utterance_meta.duration_ms が payload に反映される"""
        result = run_pipeline(
            "テスト", utterance_meta={"duration_ms": 3500}, **COMMON
        )
        assert result.events[0]["payload"]["duration_ms"] == 3500

    def test_utterance_meta_words_reflected(self, mock_publish):
        """utterance_meta.words が payload に反映される"""
        words = [{"word": "テスト", "start": 0.0, "end": 0.5}]
        result = run_pipeline(
            "テスト", utterance_meta={"words": words}, **COMMON
        )
        assert result.events[0]["payload"]["words"] == words

    def test_utterance_meta_none_uses_defaults(self, mock_publish):
        """utterance_meta=None（省略時）はデフォルト値が使われる"""
        result = run_pipeline("テスト", **COMMON)
        payload = result.events[0]["payload"]
        assert payload["lang"] == "ja-JP"
        assert payload["confidence"] == 0.95
        assert payload["duration_ms"] == 0
        assert "words" not in payload

    def test_utterance_meta_full_stt_fields(self, mock_publish):
        """real STT 相当の全フィールドを渡してもスキーマ検証が通る"""
        from lab_lounge.events import validate_event
        words = [{"word": "こんにちは", "start": 0.0, "end": 0.6}]
        meta = {
            "lang": "ja",
            "confidence": 0.0,
            "duration_ms": 2800,
            "words": words,
        }
        result = run_pipeline("こんにちは", utterance_meta=meta, **COMMON)
        ev = result.events[0]
        validate_event(ev)
        assert ev["payload"]["duration_ms"] == 2800
        assert ev["payload"]["words"] == words

    def test_utterance_meta_does_not_affect_llm_tts(self, mock_publish):
        """utterance_meta の STT 固有フィールドは llm.final / tts.done に漏れない"""
        meta = {"confidence": 0.75, "duration_ms": 1000}
        result = run_pipeline("テスト", utterance_meta=meta, **COMMON)
        # confidence は STT 固有フィールド — llm.final にも tts.done にも存在しない
        assert "confidence" not in result.events[1]["payload"]
        assert "confidence" not in result.events[2]["payload"]
        # tts.done は TTS 音声長としての duration_ms を持つが、
        # STT の duration_ms (1000) とは別物。TTS のデフォルト値 (3000) のまま。
        assert result.events[2]["payload"]["duration_ms"] == 3000

    def test_still_three_events_with_meta(self, mock_publish):
        """utterance_meta があっても 3 イベント publish される"""
        result = run_pipeline(
            "テスト", utterance_meta={"duration_ms": 1000}, **COMMON
        )
        assert len(result.events) == 3

    def test_no_stream_idx_with_meta(self, mock_publish):
        """Guardrail G-1: utterance_meta があっても stream_idx を含めない"""
        result = run_pipeline(
            "テスト", utterance_meta={"lang": "ja"}, **COMMON
        )
        for ev in result.events:
            assert "stream_idx" not in ev


class TestRunPipelineWithLangSmith:
    """
    LangSmith 観測関連のテスト。
    実際の LangSmith API は呼ばない。モックで検証する。
    """

    @pytest.fixture()
    def mock_real_llm(self, monkeypatch):
        monkeypatch.setenv("L2_USE_REAL_LLM", "true")
        monkeypatch.setenv("L2_LLM_PROVIDER", "openai")
        monkeypatch.setenv("L2_LLM_MODEL", "gpt-4o-mini")

    @pytest.fixture()
    def fake_llm_result(self):
        from lab_lounge.llm import LLMResult
        return LLMResult(
            text="LangSmith テスト応答",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=5,
            latency_ms=200,
            finish_reason="stop",
        )

    def test_run_graph_called_with_run_metadata_in_real_mode(
        self, mock_real_llm, mock_publish, fake_llm_result
    ):
        """real LLM 時に run_graph に run_metadata が渡ること"""
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result) as mock_graph:
            run_pipeline("テスト", **COMMON)
        call_kw = mock_graph.call_args.kwargs
        assert "run_metadata" in call_kw
        assert call_kw["run_metadata"] is not None

    def test_run_metadata_contains_correct_ids(
        self, mock_real_llm, mock_publish, fake_llm_result
    ):
        """run_metadata に stream_id / session_id / trace_id が含まれること"""
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result) as mock_graph:
            run_pipeline("テスト", **COMMON)
        meta = mock_graph.call_args.kwargs["run_metadata"]
        assert meta["aibyss.stream_id"] == COMMON["stream_id"]
        assert meta["aibyss.session_id"] == COMMON["session_id"]
        assert meta["aibyss.trace_id"] == COMMON["trace_id"]
        assert meta["aibyss.source"] == "aibyss-lab-lounge"

    def test_langsmith_tracing_on_does_not_change_events(
        self, mock_real_llm, mock_publish, fake_llm_result, monkeypatch
    ):
        """LANGSMITH_TRACING=true でも発行されるイベントは変わらない"""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result):
            result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final", "tts.done"]

    def test_langsmith_tracing_off_still_passes_run_metadata(
        self, mock_real_llm, mock_publish, fake_llm_result, monkeypatch
    ):
        """LANGSMITH_TRACING=false のときも run_metadata は渡される（副作用なし）"""
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result) as mock_graph:
            run_pipeline("テスト", **COMMON)
        assert "run_metadata" in mock_graph.call_args.kwargs

    def test_audio_file_path_with_tracing_on(
        self, mock_real_llm, mock_publish, fake_llm_result, monkeypatch
    ):
        """utterance_meta (audio-file 想定) + LANGSMITH_TRACING=true の組み合わせ"""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        utterance_meta = {"lang": "ja", "confidence": 0.9, "duration_ms": 3049}
        with patch("lab_lounge.graph.run_graph", return_value=fake_llm_result) as mock_graph:
            result = run_pipeline(
                "音声テスト", utterance_meta=utterance_meta, **COMMON
            )
        # イベントが正しく生成される
        assert len(result.events) == 3
        assert result.events[0]["payload"]["confidence"] == 0.9
        # run_metadata に trace_id が入る
        meta = mock_graph.call_args.kwargs["run_metadata"]
        assert meta["aibyss.trace_id"] == COMMON["trace_id"]

    def test_dummy_mode_does_not_call_run_graph(self, mock_publish, monkeypatch):
        """dummy LLM モード局 run_graph が呼ばれないこと"""
        monkeypatch.delenv("L2_USE_REAL_LLM", raising=False)
        with patch("lab_lounge.graph.run_graph") as mock_graph:
            run_pipeline("テスト", **COMMON)
        mock_graph.assert_not_called()


# ─── Phase 0.5-A 案 W'-1: LLM-only / TTS-only パイプライン ───────


class TestRunPipelineLlmOnly:
    """run_pipeline_llm_only: LLM のみ先行実行 (Phase 0.5-A 案 W'-1)。

    既存 run_pipeline と異なり TTS ノードを実行しない。挙手 BG 先行生成で
    使われ、承認時に run_pipeline_tts_only(llm_result) で TTS を再開する。
    """

    def test_returns_pipeline_result(self, mock_publish):
        from lab_lounge.pipeline import run_pipeline_llm_only, PipelineResult
        result = run_pipeline_llm_only("hello", **COMMON)
        assert isinstance(result, PipelineResult)

    def test_returns_two_events(self, mock_publish):
        """events に utterance.final + llm.final の 2 件のみ含まれる。"""
        from lab_lounge.pipeline import run_pipeline_llm_only
        result = run_pipeline_llm_only("hello", **COMMON)
        assert len(result.events) == 2

    def test_event_types_in_order(self, mock_publish):
        """types == [utterance.final, llm.final] (tts.done なし)。"""
        from lab_lounge.pipeline import run_pipeline_llm_only
        result = run_pipeline_llm_only("hello", **COMMON)
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final"]

    def test_no_tts_done_event(self, mock_publish):
        """events に tts.done が含まれない (= TTS ノード非実行の証)。

        WHY: 案 W'-1 の核心。BG LLM が TTS まで走らないので VOICEPEAK FIFO に
        投入されない (= 却下/lapse 時のリソース無駄を防ぐ)。
        """
        from lab_lounge.pipeline import run_pipeline_llm_only
        result = run_pipeline_llm_only("hello", **COMMON)
        types = [ev["type"] for ev in result.events]
        assert "tts.done" not in types

    def test_publish_called_three_times(self, mock_publish):
        """3 publish: utterance.final + bubble.update("thinking") + llm.final。

        WHY: run_pipeline は 5 publish (= utterance + thinking + answering + llm + tts)
        だが、run_pipeline_llm_only は TTS / answering を含まない (suppress=True)
        ので 3 件まで。
        """
        from lab_lounge.pipeline import run_pipeline_llm_only
        run_pipeline_llm_only("hello", **COMMON)
        assert mock_publish.call_count == 3

    def test_llm_links_utterance(self, mock_publish):
        """llm.final.links に utterance.final.event_id が含まれる (因果関係保持)。"""
        from lab_lounge.pipeline import run_pipeline_llm_only
        result = run_pipeline_llm_only("hello", **COMMON)
        utt_id = result.events[0]["event_id"]
        llm_links = result.events[1]["links"]
        assert utt_id in llm_links

    def test_dummy_llm_text_format(self, mock_publish, monkeypatch):
        """ダミーモード (L2_USE_REAL_LLM 未設定) で「ダミー応答: <input>」を返す。"""
        from lab_lounge.pipeline import run_pipeline_llm_only
        monkeypatch.delenv("L2_USE_REAL_LLM", raising=False)
        result = run_pipeline_llm_only("天気の話", **COMMON)
        assert result.events[1]["payload"]["text"] == "ダミー応答: 天気の話"

    def test_speaker_hint_routes_correctly(self, mock_publish):
        """speaker_hint='mimi' で result.speaker == 'mimi'。"""
        from lab_lounge.pipeline import run_pipeline_llm_only
        result = run_pipeline_llm_only("hello", speaker_hint="mimi", **COMMON)
        assert result.speaker == "mimi"


class TestRunPipelineTtsOnly:
    """run_pipeline_tts_only: LLM 結果を再利用して TTS のみ実行 (Phase 0.5-A 案 W'-1)。

    挙手承認時に run_loop が呼ぶ。TTS-only graph (= tts → END) を invoke して
    _tts_node 内の全ロジック (pose 切替 / wait_bg / build_tts_done) を再利用する。
    """

    def test_returns_pipeline_result(self, mock_publish):
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only, PipelineResult,
        )
        llm_result = run_pipeline_llm_only("hello", **COMMON)
        result = run_pipeline_tts_only(llm_result)
        assert isinstance(result, PipelineResult)

    def test_appends_tts_done_to_initial_events(self, mock_publish):
        """initial events (2 件) + tts.done で計 3 件になる。"""
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only,
        )
        llm_result = run_pipeline_llm_only("hello", **COMMON)
        assert len(llm_result.events) == 2  # 前提確認

        result = run_pipeline_tts_only(llm_result)
        assert len(result.events) == 3
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final", "tts.done"]

    def test_speaker_preserved_from_llm_result(self, mock_publish):
        """result.speaker は llm_result.speaker を引き継ぐ。"""
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only,
        )
        llm_result = run_pipeline_llm_only("hello", speaker_hint="mimi", **COMMON)
        result = run_pipeline_tts_only(llm_result)
        assert result.speaker == "mimi"

    def test_tts_done_text_matches_llm_final_text(self, mock_publish):
        """tts.done.payload.text は llm.final.text と一致する (= TTS 入力の再利用)。

        WHY: _tts_node は state["llm_text"] を TTS 入力として使い、build_tts_done
        の text にも入れる。run_pipeline_tts_only は llm_result.events から llm.final
        を抽出して state["llm_text"] に設定する。
        """
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only,
        )
        llm_result = run_pipeline_llm_only("test 入力", **COMMON)
        llm_text = llm_result.events[1]["payload"]["text"]

        result = run_pipeline_tts_only(llm_result)
        tts_text = result.events[2]["payload"]["text"]
        assert tts_text == llm_text

    def test_common_ids_preserved(self, mock_publish):
        """stream_id / session_id / trace_id は llm_result から引き継がれる。"""
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only,
        )
        llm_result = run_pipeline_llm_only("hello", **COMMON)
        result = run_pipeline_tts_only(llm_result)
        assert result.stream_id == COMMON["stream_id"]
        assert result.session_id == COMMON["session_id"]
        assert result.trace_id == COMMON["trace_id"]

    def test_on_tts_chunk_ready_callback_passed_through(self, mock_publish, monkeypatch):
        """on_tts_chunk_ready callback が graph state に伝播される。

        ダミー TTS モード (L2_USE_REAL_TTS 未設定) では callback は呼ばれないので、
        ここでは「state の中で参照できる」ことを verify する。
        """
        from lab_lounge.pipeline import (
            run_pipeline_llm_only, run_pipeline_tts_only,
        )
        chunks_received: list = []
        def on_chunk(url, text, is_last, character, pose=None):
            chunks_received.append((url, text, is_last, character))

        llm_result = run_pipeline_llm_only("hello", **COMMON)
        # ダミー TTS モードでは callback は呼ばれない (= chunks_received は空)。
        # 重要なのは「呼出が例外無く完了する」こと (= callback が graph state に
        # 渡される経路の sanity check)。
        monkeypatch.delenv("L2_USE_REAL_TTS", raising=False)
        result = run_pipeline_tts_only(llm_result, on_tts_chunk_ready=on_chunk)
        # ダミーモードでも tts.done event は発行される (= dummy meta only)
        assert any(ev["type"] == "tts.done" for ev in result.events)
