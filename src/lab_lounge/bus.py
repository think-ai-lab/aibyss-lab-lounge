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
from typing import Any

import redis

logger = logging.getLogger(__name__)

# 日本語テキストをログに含めるとき、長すぎると 1 行が見にくいので先頭のみ。
# トラブル調査では「どの発話で何が起きたか」が分かれば十分なので 30 文字で打ち切る。
_LOG_TEXT_PREVIEW_CHARS = 30


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def _stream_key() -> str:
    return os.environ.get("REDIS_STREAM_KEY", "aibyss:events")


def _summarize_event(event: dict[str, Any]) -> str:
    """ログ用に payload から「どのキャラの何の動作か」が分かる手掛かりを抽出する。

    Phase 0.5-A 後 ログ強化 L-1 で追加。実走ログで「published type=xxx event_id=yyy」
    だけだと「どのキャラの応答か」「何が再生されたか」がパッと分からず、調査効率が
    悪かった。各 event type ごとに調査時に欲しい識別子・コンテキストを整形する。

    payload に欲しい情報が無い type は ``(no summary)`` を返す (= 未知 type の
    後方互換、ログは type / event_id / msg_id のみで継続)。

    Args:
        event: Event Envelope dict (build_* 関数の戻り値)

    Returns:
        ログに含める短い summary 文字列 (例: ``character=mimi step=thinking category=speech``)。
        フィールド未定義時は ``?`` を入れる (= 「明示的に欠けている」と分かるよう)。
    """
    type_ = event.get("type", "")
    payload = event.get("payload", {}) or {}

    if type_ == "utterance.final":
        # ルカ発話なので character なし。発話テキスト先頭でターン識別。
        text = payload.get("text", "")
        preview = text[:_LOG_TEXT_PREVIEW_CHARS]
        suffix = "..." if len(text) > _LOG_TEXT_PREVIEW_CHARS else ""
        return f"text={preview!r}{suffix} chars={len(text)}"

    if type_ == "llm.final":
        # character は L-2 で payload に追加。それまでは ? の場合あり。
        char = payload.get("character", "?")
        model = payload.get("model", "?")
        text_len = len(payload.get("text", ""))
        return f"character={char} model={model} text_len={text_len}"

    if type_ == "tts.done":
        # speaker は voicepeak narrator slug 兼キャラ slug。L-2 で character 統一予定。
        char = payload.get("character") or payload.get("speaker", "?")
        duration = payload.get("duration_ms", "?")
        return f"character={char} duration_ms={duration}"

    if type_ == "bubble.update":
        # Phase 0.5-A 8-10 で category 追加済。speech / handraise の分岐軸。
        char = payload.get("character", "?")
        step = payload.get("step", "?")
        cat = payload.get("category", "speech_status")  # default 解釈と合わせる (Phase 0.5-E: bubble 3 系統分離)
        return f"character={char} step={step} category={cat}"

    if type_ == "dispatcher.queue.update":
        q = payload.get("queue", [])
        slugs = [item.get("character_slug") for item in q]
        state = payload.get("state", "?")
        return f"state={state} queue_size={len(q)} slugs={slugs}"

    if type_ == "dispatcher.handraise.update":
        states = payload.get("handraise_states", [])
        slugs = [s.get("target_slug") for s in states]
        cooldowns = payload.get("cooldowns", {})
        return (
            f"handraise_count={len(states)} slugs={slugs} "
            f"cooldown_slugs={list(cooldowns.keys())}"
        )

    if type_ == "character.status.update":
        # Phase 0.5-B-α 導入。Ready/Thinking/ToolCalling/Raisehand/Talking の遷移を
        # ログで観察可能にする。Talking 時は metadata の pose / text を簡略表示
        # (text は長文になりうるので _LOG_TEXT_PREVIEW_CHARS で truncate)。
        char = payload.get("character", "?")
        status = payload.get("status", "?")
        prev = payload.get("previous_status", "?")
        metadata = payload.get("metadata") or {}
        meta_parts = []
        if "pose" in metadata:
            meta_parts.append(f"pose={metadata['pose']}")
        if "text" in metadata:
            text = metadata["text"]
            preview = text[:_LOG_TEXT_PREVIEW_CHARS]
            suffix = "..." if len(text) > _LOG_TEXT_PREVIEW_CHARS else ""
            meta_parts.append(f"text={preview!r}{suffix}")
        meta_summary = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        return f"character={char} status={prev}->{status}{meta_summary}"

    return "(no summary)"


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
        # ログ強化 L-1: type 別 summary でキャラ識別 + 主要 context をログに残す
        summary = _summarize_event(event)
        logger.info(
            "published type=%s [%s] event_id=%s msg_id=%s",
            event.get("type"), summary, event.get("event_id"), msg_id_str,
        )
        return msg_id_str
    finally:
        client.close()
