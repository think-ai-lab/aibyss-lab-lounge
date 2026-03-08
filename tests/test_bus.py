"""
test_bus.py — bus.publish テスト

redis.from_url をモックして実際の Redis 接続なしで検証する。
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.bus as bus_mod
from lab_lounge.bus import publish

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
