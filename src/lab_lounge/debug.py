"""
debug.py — ローカル debug artifacts 書き出し

責務:
  - pipeline.py の各ステップ結果をローカルファイルに書き出す
  - L2_DEBUG_ARTIFACTS=true のときのみ書き出す（デフォルト: false）
  - 書き出し先は L2_DEBUG_LOG_DIR（デフォルト: ./logs）

【出力ファイル】
  logs/stt_output.json    ← STT 結果（テキスト・信頼度・duration 等）
  logs/retrieval.json     ← 検索結果（doc_ids, scores, latency）
  logs/llm_prompt.txt     ← LLM に渡したプロンプト全文
  logs/llm_response.txt   ← LLM の応答全文

【有効化】
  .env に追加:
    L2_DEBUG_ARTIFACTS=true
    L2_DEBUG_LOG_DIR=./logs   # 省略可（デフォルト: ./logs）
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_debug_enabled() -> bool:
    """L2_DEBUG_ARTIFACTS=true のとき True を返す。"""
    return os.environ.get("L2_DEBUG_ARTIFACTS", "false").lower() in ("true", "1", "yes")


def _log_dir() -> str:
    return os.environ.get("L2_DEBUG_LOG_DIR", "./logs")


def _write(filename: str, content: str) -> None:
    """logs/ 以下にファイルを書き出す。"""
    log_dir = _log_dir()
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("debug artifact: %s", os.path.basename(path))


def write_stt_output(text: str, utterance_meta: dict[str, Any] | None) -> None:
    """
    STT 結果を logs/stt_output.json に書き出す。

    Args:
        text:          STT で得たテキスト
        utterance_meta: STT メタデータ（confidence / lang / duration_ms 等）
    """
    if not is_debug_enabled():
        return
    data = {"text": text, **(utterance_meta or {})}
    _write("stt_output.json", json.dumps(data, ensure_ascii=False, indent=2))


def write_retrieval(
    doc_ids: list[str],
    scores: list[float],
    latency_ms: int,
    rag_enabled: bool,
) -> None:
    """
    検索結果を logs/retrieval.json に書き出す。

    Args:
        doc_ids:    取得した文書の ID リスト
        scores:     各文書のコサイン類似度スコア
        latency_ms: 検索にかかった時間 [ms]
        rag_enabled: RAG が有効だったかどうか
    """
    if not is_debug_enabled():
        return
    data = {
        "rag_enabled": rag_enabled,
        "latency_ms": latency_ms,
        "results": [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in zip(doc_ids, scores)
        ],
    }
    _write("retrieval.json", json.dumps(data, ensure_ascii=False, indent=2))


def write_llm_prompt(text: str, context: str | None) -> None:
    """
    LLM に渡したプロンプト全文を logs/llm_prompt.txt に書き出す。

    Args:
        text:    ユーザー発話テキスト
        context: RAG で取得した参照テキスト（None = non-RAG）
    """
    if not is_debug_enabled():
        return
    parts: list[str] = []
    if context:
        parts.append("[SYSTEM]")
        parts.append("以下の参照情報をもとに回答してください。")
        parts.append("")
        parts.append(context)
        parts.append("")
    parts.append("[USER]")
    parts.append(text)
    _write("llm_prompt.txt", "\n".join(parts))


def write_llm_response(text: str) -> None:
    """
    LLM の応答全文を logs/llm_response.txt に書き出す。

    Args:
        text: LLM の応答テキスト
    """
    if not is_debug_enabled():
        return
    _write("llm_response.txt", text)
