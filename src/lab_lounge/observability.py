"""
observability.py — LangSmith 観測ヘルパー

責務:
  - LangSmith 有効/無効の判定を集約する (LANGSMITH_TRACING 環境変数)
  - LangGraph run に渡す run_metadata dict を組み立てる
  - LangSmith 依存は optional — 無効時に langsmith パッケージは不要

設計:
  - LangSmith tracing は LangChain / LangGraph のコールバックシステムを通じて
    環境変数だけで自動 on/off できる。個別 API 呼び出しは不要。
    LANGSMITH_TRACING=true + LANGSMITH_API_KEY を設定するだけで
    LangChain / LangGraph がトレースを自動送信する。
  - run_metadata を graph.invoke() の config["metadata"] に渡すことで
    stream_id / session_id / trace_id が LangSmith のトレース画面に表示される。
  - tracing off 時: config["metadata"] は組み立てて渡すが LangSmith サーバには
    送信されない。アプリケーション動作への影響はゼロ。

【有効化手順】
  uv sync --extra obs
  .env に追加:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_pt_...
    LANGSMITH_PROJECT=aibyss-lab-lounge
"""

import logging
import os

logger = logging.getLogger(__name__)


def is_langsmith_enabled() -> bool:
    """
    LangSmith tracing が有効かどうかを返す。

    LANGSMITH_TRACING=true のとき True を返す。
    LANGSMITH_API_KEY と langsmith パッケージが別途必要。
    """
    return os.environ.get("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")


def build_run_metadata(
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    rag_used: bool = False,
    answer_mode: str = "fallback",
    retrieval_latency_ms: int = 0,
    retrieved_doc_count: int = 0,
    retrieved_doc_ids: list[str] | None = None,
) -> dict:
    """
    LangGraph run に渡す metadata dict を組み立てる。

    LangSmith のトレース画面で stream_id / session_id / trace_id および
    RAG 関連フィールドをフィルタ/検索キーとして使えるようになる。

    LangSmith が無効のときも呼んでよい。config["metadata"] に渡されるが、
    送信先がないため何の副作用も持たない。

    Args:
        stream_id:             A.I.byss ストリーム識別子
        session_id:            A.I.byss セッション識別子
        trace_id:              A.I.byss トレース識別子
        rag_used:              RAG を使ったかどうか
        answer_mode:           "grounded" または "fallback"
        retrieval_latency_ms:  検索にかかった時間 [ms]
        retrieved_doc_count:   取得した文書数
        retrieved_doc_ids:     取得した文書の ID リスト

    Returns:
        LangGraph config["metadata"] として渡せる dict
    """
    return {
        "aibyss.stream_id": stream_id,
        "aibyss.session_id": session_id,
        "aibyss.trace_id": trace_id,
        "aibyss.source": "aibyss-lab-lounge",
        "aibyss.rag_used": rag_used,
        "aibyss.answer_mode": answer_mode,
        "aibyss.retrieval_latency_ms": retrieval_latency_ms,
        "aibyss.retrieved_doc_count": retrieved_doc_count,
        "aibyss.retrieved_doc_ids": retrieved_doc_ids or [],
    }
