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
        """tts.done の payload.text は llm.final と同一テキストになる"""
        result = run_pipeline("今日の天気を教えて", **COMMON)
        assert result.events[2]["payload"]["text"] == "ダミー応答: 今日の天気を教えて"
