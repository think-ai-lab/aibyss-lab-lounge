"""
test_events.py — Event Envelope ビルダー + スキーマ検証テスト
"""

import re

import jsonschema
import pytest

from lab_lounge.events import (
    build_llm_final,
    build_tts_done,
    build_utterance_final,
    validate_event,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

COMMON = dict(
    stream_id="stream-test-001",
    session_id="sess-test-001",
    trace_id="trace-test-001",
)


class TestUtteranceFinal:
    def test_schema_valid(self):
        ev = build_utterance_final(text="今日の天気を教えて", **COMMON)
        # 検証が通った = 例外が出ない
        validate_event(ev)

    def test_required_fields(self):
        ev = build_utterance_final(text="hello", **COMMON)
        for f in ("ver", "event_id", "ts", "stream_id", "session_id", "trace_id", "type", "source", "payload"):
            assert f in ev

    def test_event_id_is_uuid(self):
        ev = build_utterance_final(text="hello", **COMMON)
        assert UUID_PATTERN.match(ev["event_id"]), f"UUID 形式でない: {ev['event_id']}"

    def test_ver_and_type(self):
        ev = build_utterance_final(text="hello", **COMMON)
        assert ev["ver"] == "0.1"
        assert ev["type"] == "utterance.final"
        assert ev["source"] == "lab-lounge"

    def test_payload_contents(self):
        ev = build_utterance_final(text="テスト", **COMMON)
        assert ev["payload"]["text"] == "テスト"
        assert ev["payload"]["lang"] == "ja-JP"
        assert ev["payload"]["confidence"] == 0.95

    def test_no_stream_idx(self):
        """Guardrail G-1: stream_idx を含めない"""
        ev = build_utterance_final(text="hello", **COMMON)
        assert "stream_idx" not in ev

    def test_schema_rejects_extra_field(self):
        ev = build_utterance_final(text="hello", **COMMON)
        ev["stream_idx"] = 0  # C2 専権フィールドを手動追加
        with pytest.raises(jsonschema.ValidationError):
            validate_event(ev)


class TestLlmFinal:
    def test_schema_valid(self):
        ev = build_llm_final(
            text="ダミー応答: 今日の天気を教えて",
            links=["11111111-2222-4333-8444-555555555555"],
            **COMMON,
        )
        validate_event(ev)

    def test_links_contains_parent_event_id(self):
        parent_id = "11111111-2222-4333-8444-555555555555"
        ev = build_llm_final(text="dummy", links=[parent_id], **COMMON)
        assert ev["links"] == [parent_id]

    def test_payload_model(self):
        ev = build_llm_final(text="dummy", links=[], **COMMON)
        assert ev["payload"]["model"] == "dummy-1.0"


class TestTtsDone:
    def test_schema_valid(self):
        ev = build_tts_done(
            text="ダミー応答",
            links=["11111111-2222-4333-8444-555555555555"],
            **COMMON,
        )
        validate_event(ev)

    def test_links_contains_parent_event_id(self):
        parent_id = "11111111-2222-4333-8444-555555555555"
        ev = build_tts_done(text="dummy", links=[parent_id], **COMMON)
        assert ev["links"] == [parent_id]

    def test_payload_audio_url(self):
        ev = build_tts_done(text="dummy", links=[], **COMMON)
        assert ev["payload"]["audio_url"] == "file://dummy/audio.opus"
        assert ev["payload"]["duration_ms"] == 3000
