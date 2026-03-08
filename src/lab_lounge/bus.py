"""
bus.py — Redis Streams Publisher

責務:
  - Event Envelope を JSON 文字列化して Redis Stream に XADD する
  - フィールド名は "event" 固定 (C2 Consumer との規約)
  - C2 への HTTP POST は絶対に行わない (Guardrail G-2)

【Redis Streams メッセージ規約 (v0.1)】
  XADD <stream_key> * event '<Event Envelope JSON 文字列>'
"""

import json
import logging
import os

import redis

logger = logging.getLogger(__name__)


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def _stream_key() -> str:
    return os.environ.get("REDIS_STREAM_KEY", "aibyss:events")


def publish(event: dict) -> str:
    """
    Event Envelope を Redis Stream に publish する。

    エンコーディング方針:
      - json.dumps(..., ensure_ascii=False) で Unicode 文字をそのまま JSON 化する
      - .encode("utf-8") で明示的に UTF-8 バイト列に変換してから Redis に送る
      - decode_responses=False で redis-py の内部エンコーディング推測に依存しない
      - 受け取った msg_id (bytes) は UTF-8 で decode して str として返す

    Returns:
        XADD で返された Redis メッセージ ID 文字列。
    """
    url = _redis_url()
    stream = _stream_key()
    # decode_responses=False: 送受信ともにバイト列を明示的に扱う
    client = redis.from_url(url, decode_responses=False)
    try:
        # locale 依存の encode を避け、UTF-8 バイト列として明示的に組み立てる
        payload_bytes = json.dumps(event, ensure_ascii=False).encode("utf-8")
        msg_id: bytes = client.xadd(stream, {"event": payload_bytes})
        msg_id_str = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
        logger.info("published type=%s event_id=%s msg_id=%s", event.get("type"), event.get("event_id"), msg_id_str)
        return msg_id_str
    finally:
        client.close()
