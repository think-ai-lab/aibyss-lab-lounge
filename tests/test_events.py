"""
test_events.py — Event Envelope ビルダー + スキーマ検証テスト
"""

import re

import jsonschema
import pytest

from lab_lounge.events import (
    build_bubble_update,
    build_dispatcher_handraise_update,
    build_dispatcher_queue_update,
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

    def test_character_in_payload_when_set(self):
        """ログ強化 L-2: character を渡すと payload.character に格納される。"""
        ev = build_llm_final(
            text="response", links=[], character="mimi", **COMMON,
        )
        assert ev["payload"]["character"] == "mimi"
        validate_event(ev)

    def test_character_omitted_when_none(self):
        """character=None (default) なら payload に含まれない (後方互換)。"""
        ev = build_llm_final(text="dummy", links=[], **COMMON)
        assert "character" not in ev["payload"]


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

    def test_character_in_payload_when_set(self):
        """ログ強化 L-2: character を渡すと payload.character に格納される。

        speaker (= voicepeak narrator) と独立した aibyss キャラ slug 用フィールド。
        """
        ev = build_tts_done(
            text="dummy", links=[],
            speaker="Haruno Sora",
            character="sakura",
            **COMMON,
        )
        assert ev["payload"]["speaker"] == "Haruno Sora"
        assert ev["payload"]["character"] == "sakura"
        validate_event(ev)

    def test_character_omitted_when_none(self):
        """character=None (default) なら payload に含まれない (後方互換)。"""
        ev = build_tts_done(text="dummy", links=[], **COMMON)
        assert "character" not in ev["payload"]


class TestDispatcherQueueUpdate:
    """build_dispatcher_queue_update — Block 0 (録音常時化) で導入。
    HUD のデバッグ dashboard 用。配信画面非表示。"""

    def test_schema_valid_empty_queue(self):
        """queue 空でもスキーマ検証を通る。"""
        ev = build_dispatcher_queue_update(
            queue=[],
            max_size=3,
            ttl_sec=60.0,
            **COMMON,
        )
        validate_event(ev)

    def test_schema_valid_with_entries(self):
        """queue にエントリがあってもスキーマ検証を通る。"""
        queue = [
            {
                "character_slug": "mimi",
                "keyword": "ミミ様",
                "transcript": "深海って怖い場所?",
                "age_sec": 5.2,
            },
            {
                "character_slug": "chisame",
                "keyword": "ちさめさん",
                "transcript": "データ的には?",
                "age_sec": 1.0,
            },
        ]
        ev = build_dispatcher_queue_update(
            queue=queue,
            max_size=3,
            ttl_sec=60.0,
            **COMMON,
        )
        validate_event(ev)

    def test_required_fields(self):
        ev = build_dispatcher_queue_update(
            queue=[],
            max_size=3,
            ttl_sec=60.0,
            **COMMON,
        )
        for f in ("ver", "event_id", "ts", "stream_id", "session_id", "trace_id", "type", "source", "payload"):
            assert f in ev

    def test_event_id_is_uuid(self):
        ev = build_dispatcher_queue_update(
            queue=[], max_size=3, ttl_sec=60.0, **COMMON,
        )
        assert UUID_PATTERN.match(ev["event_id"]), f"UUID 形式でない: {ev['event_id']}"

    def test_ver_type_source(self):
        ev = build_dispatcher_queue_update(
            queue=[], max_size=3, ttl_sec=60.0, **COMMON,
        )
        assert ev["ver"] == "0.1"
        assert ev["type"] == "dispatcher.queue.update"
        assert ev["source"] == "lab-lounge"

    def test_payload_contents(self):
        queue = [
            {
                "character_slug": "sakura",
                "keyword": "さくらさん",
                "transcript": "気持ちは?",
                "age_sec": 2.5,
            },
        ]
        ev = build_dispatcher_queue_update(
            queue=queue,
            max_size=3,
            ttl_sec=60.0,
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["queue"] == queue
        assert payload["max_size"] == 3
        assert payload["ttl_sec"] == 60.0
        assert payload["state"] == "idle"  # default

    def test_state_field(self):
        ev = build_dispatcher_queue_update(
            queue=[], max_size=3, ttl_sec=60.0, state="responding", **COMMON,
        )
        assert ev["payload"]["state"] == "responding"

    def test_no_stream_idx(self):
        """Guardrail G-1: stream_idx を含めない"""
        ev = build_dispatcher_queue_update(
            queue=[], max_size=3, ttl_sec=60.0, **COMMON,
        )
        assert "stream_idx" not in ev


class TestBubbleUpdate:
    """build_bubble_update — Axis B (意図ゲート) 以降で導入。
    パイプライン進捗を OBS 吹き出しに表示するイベント。
    Phase 0.5 で挙手ステップ (handraise/denied/lapsed/cancelled) と ttl_ms を追加。
    """

    def test_schema_valid_basic(self):
        """基本フィールドだけでスキーマ検証を通る。"""
        ev = build_bubble_update(
            character="mimi",
            step="thinking",
            text="考え中…",
            **COMMON,
        )
        validate_event(ev)

    def test_required_fields(self):
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        for f in ("ver", "event_id", "ts", "stream_id", "session_id", "trace_id", "type", "source", "payload"):
            assert f in ev

    def test_event_id_is_uuid(self):
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert UUID_PATTERN.match(ev["event_id"]), f"UUID 形式でない: {ev['event_id']}"

    def test_ver_type_source(self):
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert ev["ver"] == "0.1"
        assert ev["type"] == "bubble.update"
        assert ev["source"] == "lab-lounge"

    def test_payload_contents(self):
        ev = build_bubble_update(
            character="chisame",
            step="answering",
            text="それは興味深いですね",
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["character"] == "chisame"
        assert payload["step"] == "answering"
        assert payload["text"] == "それは興味深いですね"

    def test_no_stream_idx(self):
        """Guardrail G-1: stream_idx を含めない。"""
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert "stream_idx" not in ev

    def test_links_optional(self):
        """links を渡すと top-level に追加される。"""
        parent = "11111111-2222-4333-8444-555555555555"
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…",
            links=[parent], **COMMON,
        )
        assert ev["links"] == [parent]

    def test_links_omitted_when_none(self):
        """links を渡さないと event に含まれない。"""
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert "links" not in ev


class TestBubbleUpdateTtlMs:
    """Phase 0.5-A で追加された ttl_ms 引数 + 新 step 値の挙動を確認する。"""

    def test_ttl_ms_in_payload_when_set(self):
        """ttl_ms を渡すと payload.ttl_ms に格納される。"""
        ev = build_bubble_update(
            character="mimi", step="denied", text="また今度",
            ttl_ms=2000, **COMMON,
        )
        assert ev["payload"]["ttl_ms"] == 2000

    def test_ttl_ms_omitted_when_none(self):
        """ttl_ms=None (default) なら payload に ttl_ms キーが含まれない。"""
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert "ttl_ms" not in ev["payload"]

    def test_ttl_ms_explicit_none(self):
        """ttl_ms=None を明示しても payload に含まれない (handraise 中の永続表示)。"""
        ev = build_bubble_update(
            character="mimi", step="handraise", text="ちょっと、いい?",
            ttl_ms=None, **COMMON,
        )
        assert "ttl_ms" not in ev["payload"]

    def test_step_handraise_validates(self):
        """新 step 値 "handraise" でスキーマ検証を通る。"""
        ev = build_bubble_update(
            character="mimi", step="handraise", text="ちょっと、いい?",
            **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["step"] == "handraise"

    def test_step_denied_with_ttl(self):
        """新 step 値 "denied" + ttl_ms=2000 でスキーマ検証を通る。"""
        ev = build_bubble_update(
            character="mimi", step="denied", text="また今度",
            ttl_ms=2000, **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["step"] == "denied"
        assert ev["payload"]["ttl_ms"] == 2000

    def test_step_lapsed_with_ttl(self):
        """新 step 値 "lapsed" + ttl_ms=2000 でスキーマ検証を通る。"""
        ev = build_bubble_update(
            character="sakura", step="lapsed", text="…静まりました",
            ttl_ms=2000, **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["step"] == "lapsed"

    def test_step_cancelled_with_ttl(self):
        """新 step 値 "cancelled" + ttl_ms でスキーマ検証を通る。"""
        ev = build_bubble_update(
            character="chisame", step="cancelled", text="撤回します",
            ttl_ms=2000, **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["step"] == "cancelled"

    def test_ttl_ms_with_links(self):
        """ttl_ms と links を同時に渡せる。"""
        parent = "11111111-2222-4333-8444-555555555555"
        ev = build_bubble_update(
            character="mimi", step="denied", text="また今度",
            links=[parent], ttl_ms=2000, **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["ttl_ms"] == 2000
        assert ev["links"] == [parent]


class TestBubbleUpdateCategory:
    """Phase 0.5-A 8-10 で追加された category 引数の挙動を確認する。

    category は V2 HUD 側で表示エリアを分岐させるためのフィールド:
      - "speech":    通常応答 + 承認後応答
      - "handraise": 挙手系 (handraise/denied/lapsed/cancelled)
    None なら payload に含めない (受信側 default = "speech" 解釈、後方互換)。
    """

    def test_category_in_payload_when_set(self):
        """category を渡すと payload.category に格納される。"""
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…",
            category="speech", **COMMON,
        )
        assert ev["payload"]["category"] == "speech"

    def test_category_omitted_when_none(self):
        """category=None (default) なら payload に category キーが含まれない (後方互換)。"""
        ev = build_bubble_update(
            character="mimi", step="thinking", text="…", **COMMON,
        )
        assert "category" not in ev["payload"]

    def test_category_speech_validates(self):
        """category="speech" でスキーマ検証を通る (通常応答パス)。"""
        ev = build_bubble_update(
            character="mimi", step="answering", text="わたくしの見解は…",
            category="speech", **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["category"] == "speech"

    def test_category_handraise_validates(self):
        """category="handraise" + step="handraise" でスキーマ検証を通る (挙手系)。"""
        ev = build_bubble_update(
            character="sakura", step="handraise", text="あ、わたくし……",
            category="handraise", **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["category"] == "handraise"
        assert ev["payload"]["step"] == "handraise"

    def test_category_with_ttl_ms(self):
        """category と ttl_ms を同時に渡せる (denied/lapsed パス)。"""
        ev = build_bubble_update(
            character="mimi", step="denied", text="また今度",
            ttl_ms=2000, category="handraise", **COMMON,
        )
        validate_event(ev)
        assert ev["payload"]["category"] == "handraise"
        assert ev["payload"]["ttl_ms"] == 2000


class TestDispatcherHandraiseUpdate:
    """build_dispatcher_handraise_update — Phase 0.5-A で導入。
    HUD のデバッグ dashboard 用 (配信画面非表示)。dispatcher.queue.update の兄弟。
    """

    def test_schema_valid_empty(self):
        """挙手中なし + cooldown なしでもスキーマ検証を通る (リセット時に発行)。"""
        ev = build_dispatcher_handraise_update(
            handraise_states=[],
            cooldowns={},
            **COMMON,
        )
        validate_event(ev)

    def test_schema_valid_with_entries(self):
        """挙手中キャラと cooldown 状態を持っていてもスキーマ検証を通る。"""
        states = [
            {
                "target_slug": "mimi",
                "started_at_age_sec": 12.5,
                "phrase": "ちょっと、いい?",
                "bg_completed": False,
                "trace_id": "11111111-2222-4333-8444-555555555555",
                "utterance_count_since": 2,
            },
        ]
        cooldowns = {
            "chisame": {
                "consecutive_denials": 1,
                "cooldown_until_sec_remaining": 0.0,
                "threshold_multiplier": 1.0,
            },
        }
        ev = build_dispatcher_handraise_update(
            handraise_states=states,
            cooldowns=cooldowns,
            **COMMON,
        )
        validate_event(ev)

    def test_required_fields(self):
        ev = build_dispatcher_handraise_update(
            handraise_states=[], cooldowns={}, **COMMON,
        )
        for f in ("ver", "event_id", "ts", "stream_id", "session_id", "trace_id", "type", "source", "payload"):
            assert f in ev

    def test_event_id_is_uuid(self):
        ev = build_dispatcher_handraise_update(
            handraise_states=[], cooldowns={}, **COMMON,
        )
        assert UUID_PATTERN.match(ev["event_id"]), f"UUID 形式でない: {ev['event_id']}"

    def test_ver_type_source(self):
        ev = build_dispatcher_handraise_update(
            handraise_states=[], cooldowns={}, **COMMON,
        )
        assert ev["ver"] == "0.1"
        assert ev["type"] == "dispatcher.handraise.update"
        assert ev["source"] == "lab-lounge"

    def test_payload_contents(self):
        states = [{"target_slug": "mimi", "started_at_age_sec": 1.0,
                   "phrase": "ねぇ", "bg_completed": False,
                   "trace_id": "t", "utterance_count_since": 0}]
        cooldowns = {"chisame": {"consecutive_denials": 2,
                                  "cooldown_until_sec_remaining": 0.0,
                                  "threshold_multiplier": 1.0}}
        ev = build_dispatcher_handraise_update(
            handraise_states=states,
            cooldowns=cooldowns,
            **COMMON,
        )
        payload = ev["payload"]
        assert payload["handraise_states"] == states
        assert payload["cooldowns"] == cooldowns

    def test_no_stream_idx(self):
        """Guardrail G-1: stream_idx を含めない。"""
        ev = build_dispatcher_handraise_update(
            handraise_states=[], cooldowns={}, **COMMON,
        )
        assert "stream_idx" not in ev

    def test_payload_passes_through_unchanged(self):
        """events.py は handraise_states / cooldowns の中身に介入せず、そのまま payload に載せる。"""
        # events.py が想定外のキーを足したり消したりしないことを確認
        states = [{"target_slug": "mimi", "extra_field": "xxx",
                   "started_at_age_sec": 0.0, "phrase": "", "bg_completed": True,
                   "trace_id": "", "utterance_count_since": 0}]
        ev = build_dispatcher_handraise_update(
            handraise_states=states, cooldowns={}, **COMMON,
        )
        # extra_field がそのまま残る (= events.py が dict の中身を変えない)
        assert ev["payload"]["handraise_states"][0]["extra_field"] == "xxx"


