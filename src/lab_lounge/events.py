"""
events.py — Event Envelope ビルダー + スキーマ検証

責務:
  - Event Envelope (v0.1) を組み立てる
  - publish 前に event-envelope-0.1.schema.json で検証する
  - stream_idx は C2 の専権フィールドのため、絶対に含めない (Guardrail G-1)

【スキーマ解決順序】
  1. 環境変数 AIBYSS_SCHEMA_PATH が指定されていればそれを使う
  2. 兄弟ディレクトリ aibyss-workspace/specs/ を自動探索する
  どちらも見つからなければ RuntimeError を送出する

【trace_id 注意】
  v0.1 では UUIDv4 を仮使用。将来 W3C Trace Context に揃える予定 (Guardrail G-7)。
"""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonschema
import jsonschema.protocols


# ---------------------------------------------------------------------------
# スキーマ管理
# ---------------------------------------------------------------------------

def _load_schema() -> dict:
    """スキーマ JSON を読み込んで返す。"""
    env_path = os.environ.get("AIBYSS_SCHEMA_PATH", "")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        raise RuntimeError(f"AIBYSS_SCHEMA_PATH が指定されていますが見つかりません: {env_path}")

    # 兄弟ディレクトリ自動探索 (src/lab_lounge/ から 5 階層上まで遡る)
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents[:5]]:
        candidate = ancestor.parent / "aibyss-workspace" / "specs" / "event-envelope-0.1.schema.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))

    raise RuntimeError(
        "event-envelope-0.1.schema.json が見つかりません。"
        "AIBYSS_SCHEMA_PATH を設定するか、aibyss-workspace を兄弟ディレクトリに配置してください。"
    )


@lru_cache(maxsize=1)
def _get_validator() -> jsonschema.protocols.Validator:
    schema = _load_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def _reset_validator_cache() -> None:
    """テスト用: キャッシュをクリアする。"""
    _get_validator.cache_clear()


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _new_uuid() -> str:
    return str(uuid4())


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------

def validate_event(event: dict) -> None:
    """スキーマ検証。違反があれば jsonschema.ValidationError を送出。"""
    _get_validator().validate(event)


# ---------------------------------------------------------------------------
# イベントビルダー
# ---------------------------------------------------------------------------

def build_utterance_final(
    *,
    text: str,
    stream_id: str,
    session_id: str,
    trace_id: str,
    seq: int = 0,
    lang: str = "ja-JP",
    confidence: float = 0.95,
    duration_ms: int = 0,
    words: list | None = None,
) -> dict[str, Any]:
    """utterance.final イベントを組み立てて検証する。

    Args:
        lang:        音声認識言語コード (デフォルト: "ja-JP")
        confidence:  書き起こし信頼度 (デフォルト: 0.95)
        duration_ms: 音声ファイルの時間長 [ms] (デフォルト: 0)
        words:       単語タイムスタンプ (optional)
                     [{"word": str, "start": float, "end": float}, ...]
    """
    payload: dict[str, Any] = {
        "text": text,
        "lang": lang,
        "confidence": confidence,
        "duration_ms": duration_ms,
    }
    if words is not None:
        payload["words"] = words

    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "utterance.final",
        "source": "lab-lounge",
        "seq": seq,
        "payload": payload,
    }
    validate_event(event)
    return event


def build_llm_final(
    *,
    text: str,
    stream_id: str,
    session_id: str,
    trace_id: str,
    links: list[str],
    seq: int = 1,
    model: str = "dummy-1.0",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    finish_reason: str = "stop",
    rag_used: bool = False,
    answer_mode: str = "fallback",
    retrieval_latency_ms: int = 0,
    retrieved_doc_count: int = 0,
    retrieved_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    """llm.final イベントを組み立てて検証する。"""
    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "llm.final",
        "source": "lab-lounge",
        "seq": seq,
        "links": links,
        "payload": {
            "text": text,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "rag_used": rag_used,
            "answer_mode": answer_mode,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieved_doc_count": retrieved_doc_count,
            "retrieved_doc_ids": retrieved_doc_ids if retrieved_doc_ids is not None else [],
        },
    }
    validate_event(event)
    return event


def build_tts_done(
    *,
    text: str,
    stream_id: str,
    session_id: str,
    trace_id: str,
    links: list[str],
    seq: int = 2,
    audio_url: str = "file://dummy/audio.opus",
    chunk_audio_urls: list[str] | None = None,
    duration_ms: int = 3000,
    voice: str = "dummy-voice",
    format: str = "opus",
    sample_rate: int = 24000,
    speaker: str = "dummy",
) -> dict[str, Any]:
    """tts.done イベントを組み立てて検証する。"""
    payload: dict[str, Any] = {
        "text": text,
        "audio_url": audio_url,
        "duration_ms": duration_ms,
        "voice": voice,
        "format": format,
        "sample_rate": sample_rate,
        "speaker": speaker,
    }
    if chunk_audio_urls:
        payload["chunk_audio_urls"] = chunk_audio_urls
    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "tts.done",
        "source": "lab-lounge",
        "seq": seq,
        "links": links,
        "payload": payload,
    }
    validate_event(event)
    return event


def build_bubble_update(
    *,
    character: str,
    step: str,
    text: str,
    stream_id: str,
    session_id: str,
    trace_id: str,
    links: list[str] | None = None,
    ttl_ms: int | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    bubble.update イベントを組み立てて検証する。

    パイプライン実行中の進捗を視聴者に伝えるイベント。
    キャラ口調の短い一言を OBS 吹き出しに表示する。

    Args:
        character: キャラクター slug (e.g., "mimi")
        step:      進捗ステップ。次のいずれかを想定:
                     - 通常応答: "searching" / "thinking" / "answering" / "done"
                     - Phase 0.5 (挙手): "handraise" / "denied" / "lapsed" / "cancelled"
                   (events.py は文字列の中身に介入しない。受信側 V2 が解釈する)
        text:      表示テキスト（キャラクター口調の固定文字列）
        ttl_ms:    表示後の自動消去ミリ秒数。None なら受信側で next step まで保持。
                   Phase 0.5 では denied/lapsed=2000ms を想定し、handraise 自体は
                   None (承認/却下/lapse まで保持) で発行する。
                   payload 内に追加するため event-envelope-0.1 の schema 変更は不要。
        category:  V2 HUD 側で表示エリアを分岐させるための種別フィールド (Phase 0.5-A 8-10)。
                     - "speech":    通常応答 + 承認後応答 (thinking/searching/answering/
                                    speaking/done/pose_change)
                     - "handraise": 挙手系 (handraise/denied/lapsed/cancelled)
                   None 時は payload に含めない (受信側 default = "speech" 解釈、後方互換)。
                   step は「進捗状態」、category は「分岐軸」として独立した概念。step 拡張で
                   将来の bubble 種別が増えても category 固定で受信側ロジックを単純に保てる。
                   payload 内に追加するため event-envelope-0.1 の schema 変更は不要。
    """
    payload: dict[str, Any] = {
        "character": character,
        "step": step,
        "text": text,
    }
    # ttl_ms / category はオプション。None 時は payload に含めない (受信側 default 挙動を維持)。
    if ttl_ms is not None:
        payload["ttl_ms"] = ttl_ms
    if category is not None:
        payload["category"] = category
    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "bubble.update",
        "source": "lab-lounge",
        "payload": payload,
    }
    if links:
        event["links"] = links
    validate_event(event)
    return event


def build_dispatcher_queue_update(
    *,
    queue: list[dict[str, Any]],
    max_size: int,
    ttl_sec: float,
    stream_id: str,
    session_id: str,
    trace_id: str,
    state: str = "idle",
) -> dict[str, Any]:
    """
    dispatcher.queue.update イベントを組み立てて検証する。

    Block 0 (録音常時化) で導入。Dispatcher の wake_event_queue が変化したとき
    (add / dequeue / evict) に発行され、HUD のデバッグ dashboard で「現在
    スタックしている応答」を可視化するためのイベント。

    配信画面には出さない想定（運用デバッグ用途）。Phase 0.5 では同じパターンで
    ``dispatcher.handraise.update`` を兄弟イベントとして追加できる。

    Args:
        queue:    queue 内の各 event を表す dict のリスト。各要素は
                  ``{"character_slug": str, "keyword": str | None,
                     "transcript": str | None, "age_sec": float}`` を含む想定
                  （events.py は中身に介入せず、payload にそのまま載せる）
        max_size: queue の最大保持件数 (Dispatcher.DRAIN_MAX_EVENTS と一致)
        ttl_sec:  期限切れ閾値秒数 (Dispatcher.DRAIN_MAX_AGE_SEC と一致)
        state:    Dispatcher の現在状態 ("idle" / "responding" / "handraising")
    """
    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "dispatcher.queue.update",
        "source": "lab-lounge",
        "payload": {
            "queue": queue,
            "max_size": max_size,
            "ttl_sec": ttl_sec,
            "state": state,
        },
    }
    validate_event(event)
    return event


def build_dispatcher_handraise_update(
    *,
    handraise_states: list[dict[str, Any]],
    cooldowns: dict[str, dict[str, Any]],
    stream_id: str,
    session_id: str,
    trace_id: str,
) -> dict[str, Any]:
    """
    dispatcher.handraise.update イベントを組み立てて検証する。

    Phase 0.5 (挙手システム) で導入。Dispatcher の挙手中キャラ状態 + 連続却下
    cooldown 状態が変化したとき (start / approval / denial / lapse) に発行され、
    HUD のデバッグ dashboard で「現在挙手中のキャラ」「連続却下回数」を可視化する。
    配信画面には出さない想定 (運用デバッグ用途)。Phase 0.5-B で V2 側 UI が
    本イベントを購読して可視化する予定。

    `dispatcher.queue.update` (Block 0 で導入) の兄弟イベント。両者は別々の payload
    を持つが、HUD 側は同じ「dispatcher の内部状態スナップショット」として扱う。

    Args:
        handraise_states: 挙手中キャラ各々の dict のリスト。各要素は次を含む想定:
                            {"target_slug": str,
                             "started_at_age_sec": float,
                             "phrase": str,
                             "bg_completed": bool,
                             "trace_id": str,
                             "utterance_count_since": int}
                          (events.py は中身に介入せず、payload にそのまま載せる)
        cooldowns:        slug → cooldown 状態の dict。各値は次を含む想定:
                            {"consecutive_denials": int,
                             "cooldown_until_sec_remaining": float,
                             "threshold_multiplier": float}
                          Phase 0.5-A は threshold_multiplier=1.0 固定で発行する。
    """
    event: dict[str, Any] = {
        "ver": "0.1",
        "event_id": _new_uuid(),
        "ts": _now_iso(),
        "stream_id": stream_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "type": "dispatcher.handraise.update",
        "source": "lab-lounge",
        "payload": {
            "handraise_states": handraise_states,
            "cooldowns": cooldowns,
        },
    }
    validate_event(event)
    return event


