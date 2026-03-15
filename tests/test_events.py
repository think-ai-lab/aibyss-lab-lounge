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

    def test_payload_duration_ms_default(self):
        """duration_ms のデフォルト値は 0"""
        ev = build_utterance_final(text="hello", **COMMON)
        assert ev["payload"]["duration_ms"] == 0

    def test_payload_lang_custom(self):
        """lang 引数を渡すと payload に反映される"""
        ev = build_utterance_final(text="hello", lang="en-US", **COMMON)
        assert ev["payload"]["lang"] == "en-US"

    def test_payload_confidence_custom(self):
        """confidence 引数を渡すと payload に反映される"""
        ev = build_utterance_final(text="hello", confidence=0.0, **COMMON)
        assert ev["payload"]["confidence"] == 0.0

    def test_payload_stt_fields_full(self):
        """real STT レスポンスの全フィールドを渡すと payload に反映されスキーマ検証も通る"""
        words = [{"word": "こんにちは", "start": 0.0, "end": 0.5}]
        ev = build_utterance_final(
            text="こんにちは",
            lang="ja",
            confidence=0.0,
            duration_ms=3500,
            words=words,
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["lang"] == "ja"
        assert payload["confidence"] == 0.0
        assert payload["duration_ms"] == 3500
        assert payload["words"] == words
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

    def test_payload_extended_fields_defaults(self):
        """model/トークン・レイテンシのデフォルト値が正しい"""
        ev = build_llm_final(text="dummy", links=[], **COMMON)
        assert ev["payload"]["input_tokens"] == 0
        assert ev["payload"]["output_tokens"] == 0
        assert ev["payload"]["latency_ms"] == 0
        assert ev["payload"]["finish_reason"] == "stop"
        assert ev["payload"]["rag_used"] is False

    def test_payload_model_custom(self):
        """model 引数を渡すと payload に反映される"""
        ev = build_llm_final(text="dummy", links=[], model="gpt-4o", **COMMON)
        assert ev["payload"]["model"] == "gpt-4o"

    def test_payload_real_llm_fields(self):
        """real LLM のフィールドを渡すと payload に反映されスキーマ検証も通る"""
        ev = build_llm_final(
            text="response",
            links=["11111111-2222-4333-8444-555555555555"],
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            finish_reason="stop",
            rag_used=False,
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["model"] == "gpt-4o-mini"
        assert payload["input_tokens"] == 100
        assert payload["output_tokens"] == 50
        assert payload["latency_ms"] == 500
        assert payload["finish_reason"] == "stop"
        assert payload["rag_used"] is False
        # スキーマ検証も通る
        validate_event(ev)


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

    def test_payload_extended_fields_defaults(self):
        """デフォルト値で新フィールド voice/format/sample_rate/speaker が設定される"""
        ev = build_tts_done(text="dummy", links=[], **COMMON)
        assert ev["payload"]["voice"] == "dummy-voice"
        assert ev["payload"]["format"] == "opus"
        assert ev["payload"]["sample_rate"] == 24000
        assert ev["payload"]["speaker"] == "dummy"

    def test_payload_custom_audio_url(self):
        """audio_url 引数を渡すと payload に反映される"""
        ev = build_tts_done(
            text="hello", links=[], audio_url="file:///audio/test.mp3", **COMMON
        )
        assert ev["payload"]["audio_url"] == "file:///audio/test.mp3"

    def test_payload_real_tts_fields(self):
        """real TTS の全フィールドを渡すと payload に反映されスキーマ検証も通る"""
        ev = build_tts_done(
            text="こんにちは",
            links=["11111111-2222-4333-8444-555555555555"],
            audio_url="file:///tmp/tts-abc.mp3",
            duration_ms=2500,
            voice="ja-JP-NanamiNeural",
            format="mp3",
            sample_rate=24000,
            speaker="Nanami",
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["audio_url"] == "file:///tmp/tts-abc.mp3"
        assert payload["duration_ms"] == 2500
        assert payload["voice"] == "ja-JP-NanamiNeural"
        assert payload["format"] == "mp3"
        assert payload["sample_rate"] == 24000
        assert payload["speaker"] == "Nanami"
        # スキーマ検証も通る
        validate_event(ev)
