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
) -> dict[str, Any]:
    """utterance.final イベントを組み立てて検証する。"""
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
        "payload": {
            "text": text,
            "lang": "ja-JP",
            "confidence": 0.95,
        },
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
            "model": "dummy-1.0",
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
) -> dict[str, Any]:
    """tts.done イベントを組み立てて検証する。"""
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
        "payload": {
            "text": text,
            "audio_url": "file://dummy/audio.opus",
            "duration_ms": 3000,
        },
    }
    validate_event(event)
    return event
