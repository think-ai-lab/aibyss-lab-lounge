"""
test_bus.py — bus.publish テスト

redis.from_url をモックして実際の Redis 接続なしで検証する。
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.bus as bus_mod
from lab_lounge.bus import _summarize_event, publish

SAMPLE_EVENT = {
    "ver": "0.1",
    "event_id": "11111111-2222-4333-8444-555555555555",
    "ts": "2026-03-08T10:00:00.000Z",
    "stream_id": "stream-bus-001",
    "session_id": "sess-bus-001",
    "trace_id": "trace-bus-001",
    "type": "utterance.final",
    "source": "lab-lounge",
    "payload": {"text": "テスト"},
}


@pytest.fixture()
def mock_redis_client():
    client = MagicMock()
    # decode_responses=False のとき xadd は bytes を返す
    client.xadd.return_value = b"1741000000000-0"
    with patch.object(bus_mod.redis, "from_url", return_value=client) as _:
        yield client


class TestPublish:
    def test_xadd_called_once(self, mock_redis_client):
        publish(SAMPLE_EVENT)
        mock_redis_client.xadd.assert_called_once()

    def test_xadd_field_name_is_event(self, mock_redis_client):
        """C2 との規約: フィールド名は "event" 固定"""
        publish(SAMPLE_EVENT)
        _, kwargs = mock_redis_client.xadd.call_args
        # xadd(stream, fields) の positional arg として渡される場合も考慮
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        assert "event" in fields

    def test_xadd_value_is_valid_json(self, mock_redis_client):
        """フィールド値が Event Envelope JSON 文字列になっている"""
        publish(SAMPLE_EVENT)
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        parsed = json.loads(fields["event"])
        assert parsed["event_id"] == SAMPLE_EVENT["event_id"]

    def test_xadd_stream_key_default(self, mock_redis_client, monkeypatch):
        monkeypatch.delenv("REDIS_STREAM_KEY", raising=False)
        publish(SAMPLE_EVENT)
        call_args = mock_redis_client.xadd.call_args
        stream = call_args.args[0] if call_args.args else call_args.kwargs.get("name")
        assert stream == "aibyss:events"

    def test_returns_msg_id(self, mock_redis_client):
        """bytes の msg_id が str として返る"""
        msg_id = publish(SAMPLE_EVENT)
        assert msg_id == "1741000000000-0"

    def test_client_closed_after_publish(self, mock_redis_client):
        publish(SAMPLE_EVENT)
        mock_redis_client.close.assert_called_once()

    def test_japanese_text_preserved(self, mock_redis_client):
        """日本語テキストが UTF-8 JSON バイト列として送られ、decode しても元の文字列に戻る"""
        japanese_event = dict(SAMPLE_EVENT)
        japanese_event["payload"] = {"text": "今日の天気を教えて"}
        publish(japanese_event)
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        raw = fields["event"]
        # bytes として受け取り UTF-8 decode → JSON parse しても日本語が壊れない
        assert isinstance(raw, bytes), "publish は UTF-8 bytes を送るべき"
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["payload"]["text"] == "今日の天気を教えて"

    def test_log_includes_summary(self, mock_redis_client, caplog):
        """ログ強化 L-1: publish 時のログに type 別 summary が含まれる。

        bubble.update なら character / step / category が、調査時の手掛かりとして
        パッと分かる形で出ること。
        """
        bubble_event = {
            **SAMPLE_EVENT,
            "type": "bubble.update",
            "payload": {
                "character": "mimi",
                "step": "thinking",
                "text": "考えていますわ",
                "category": "speech",
            },
        }
        with caplog.at_level(logging.INFO, logger="lab_lounge.bus"):
            publish(bubble_event)
        # 「published type=bubble.update [character=mimi step=thinking category=speech]」のような形
        msg = "\n".join(rec.message for rec in caplog.records)
        assert "type=bubble.update" in msg
        assert "character=mimi" in msg
        assert "step=thinking" in msg
        assert "category=speech" in msg


# ─── ログ強化 L-1: _summarize_event の type 別挙動 ─────────────────


class TestSummarizeEvent:
    """_summarize_event が type 別に payload から手掛かりを抽出することを検証する。

    Phase 0.5-A 後のログ強化 L-1: 実走ログで「どのキャラの動作か」を即座に
    把握できるようにする目的。
    """

    def test_utterance_final_includes_text_preview(self):
        """utterance.final はルカ発話 → text 先頭でターン識別。"""
        ev = {
            "type": "utterance.final",
            "payload": {"text": "最近のAI倫理について気になっています"},
        }
        s = _summarize_event(ev)
        assert "text=" in s
        assert "最近のAI倫理について" in s  # 先頭部分
        assert "chars=" in s

    def test_utterance_final_long_text_truncated(self):
        """30 文字超の text は ... で切り詰める (ログ可読性)。"""
        long_text = "あ" * 100
        ev = {"type": "utterance.final", "payload": {"text": long_text}}
        s = _summarize_event(ev)
        assert "..." in s
        assert "chars=100" in s

    def test_llm_final_includes_character_when_present(self):
        """llm.final に character フィールドがあれば summary に含まれる (L-2 後対応)。"""
        ev = {
            "type": "llm.final",
            "payload": {"character": "mimi", "model": "gpt-5.5", "text": "あら、ルカ"},
        }
        s = _summarize_event(ev)
        assert "character=mimi" in s
        assert "model=gpt-5.5" in s
        assert "text_len=" in s

    def test_llm_final_character_unknown_when_missing(self):
        """character フィールド無しは ? 表示 (= 「明示的に欠けている」と分かる)。"""
        ev = {"type": "llm.final", "payload": {"model": "x", "text": ""}}
        s = _summarize_event(ev)
        assert "character=?" in s

    def test_tts_done_uses_character_or_speaker(self):
        """tts.done は character > speaker の優先順位で character を表示。"""
        ev1 = {
            "type": "tts.done",
            "payload": {"character": "sakura", "speaker": "sakura", "duration_ms": 1234},
        }
        s1 = _summarize_event(ev1)
        assert "character=sakura" in s1
        assert "duration_ms=1234" in s1

        # character 無しなら speaker にフォールバック
        ev2 = {"type": "tts.done", "payload": {"speaker": "chisame", "duration_ms": 999}}
        s2 = _summarize_event(ev2)
        assert "character=chisame" in s2

    def test_bubble_update_includes_character_step_category(self):
        """bubble.update は character / step / category を全部表示。"""
        ev = {
            "type": "bubble.update",
            "payload": {
                "character": "sakura",
                "step": "handraise",
                "text": "あのぉ、ちょっといいですかぁ",
                "category": "handraise",
            },
        }
        s = _summarize_event(ev)
        assert "character=sakura" in s
        assert "step=handraise" in s
        assert "category=handraise" in s

    def test_bubble_update_default_category_speech(self):
        """category 省略時は default の "speech" 表示 (受信側 default 解釈と一致)。"""
        ev = {
            "type": "bubble.update",
            "payload": {"character": "mimi", "step": "thinking", "text": "..."},
        }
        s = _summarize_event(ev)
        assert "category=speech" in s

    def test_dispatcher_queue_update_lists_slugs(self):
        """dispatcher.queue.update は queue 内の character_slug 一覧 + state を表示。"""
        ev = {
            "type": "dispatcher.queue.update",
            "payload": {
                "queue": [
                    {"character_slug": "mimi", "transcript": "..."},
                    {"character_slug": "chisame", "transcript": "..."},
                ],
                "state": "responding",
            },
        }
        s = _summarize_event(ev)
        assert "state=responding" in s
        assert "queue_size=2" in s
        assert "mimi" in s
        assert "chisame" in s

    def test_dispatcher_handraise_update_lists_slugs(self):
        """dispatcher.handraise.update は挙手中キャラ + cooldown キーを表示。"""
        ev = {
            "type": "dispatcher.handraise.update",
            "payload": {
                "handraise_states": [
                    {"target_slug": "sakura", "phrase": "..."},
                ],
                "cooldowns": {"mimi": {"consecutive_denials": 1}},
            },
        }
        s = _summarize_event(ev)
        assert "handraise_count=1" in s
        assert "sakura" in s
        assert "mimi" in s

    def test_unknown_type_returns_no_summary(self):
        """未知の type は (no summary) を返して後方互換維持。"""
        ev = {"type": "unknown.weird", "payload": {"some": "data"}}
        s = _summarize_event(ev)
        assert s == "(no summary)"

    def test_missing_payload_handled(self):
        """payload が無い event でも例外で死なない。"""
        ev = {"type": "bubble.update"}
        s = _summarize_event(ev)
        # payload 無し → デフォルト値 (?) でフォールバック
        assert "character=?" in s
