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

    def test_publish_called_three_times(self, mock_publish):
        run_pipeline("hello", **COMMON)
        assert mock_publish.call_count == 3

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
        mock_graph.assert_called_once_with("テスト質問", model="gpt-4o-mini", provider="openai")

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

    def test_real_tts_synthesize_called_with_env_args(
        self, mock_real_tts_mode, mock_publish, fake_tts_result
    ):
        """synthesize に ENV から読んだ provider/voice/speaker/output_dir が渡る"""
        with patch("lab_lounge.tts.synthesize", return_value=fake_tts_result) as mock_synth:
            run_pipeline("テスト", **COMMON)
        mock_synth.assert_called_once_with(
            "ダミー応答: テスト",
            provider="edge_tts",
            voice="ja-JP-NanamiNeural",
            speaker="Nanami",
            output_dir="./data/audio",
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
