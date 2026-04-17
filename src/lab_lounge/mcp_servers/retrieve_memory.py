"""
retrieve_memory.py — MCP 記憶検索サーバー

CompositeRetriever (Local kb + C2 semantic + C2 recent) を使った
記憶・知識ベース検索ツールを MCP サーバーとして提供する。
全 AITuber キャラクター（ミミ/ちさめ/さくら/オクタメイド）が共有して使用する。

Sprint Axis D Block 3: RAG を Agent ツールに統合

【セッションコンテキスト】
  retrieve_memory は stream_id / exclude_event_ids を必要とするが、
  ツールのシグネチャは (query, top_k) のみにしたいため、
  contextvars でセッション情報を注入する。
  呼び出し側 (graph.py _generation_node) が set_retrieval_context() を
  Agent 実行前に呼ぶ。

【使い方】
  # LangGraph Agent から使用（通常はこちら）
  graph.py の _load_mcp_tools() が自動的にツール登録する。

  # スタンドアロン起動（デバッグ用）
  uv run python -m lab_lounge.mcp_servers.retrieve_memory
"""

import contextvars
import logging
import os

logger = logging.getLogger(__name__)

# ─── セッションコンテキスト (contextvars) ────────────────────────────
# _generation_node がターンごとにセットし、ツール関数が読む。
# create_react_agent は同一スレッド同期実行のため、thread safety は問題なし。

_stream_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieve_memory_stream_id", default=None,
)
_exclude_event_ids_var: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "retrieve_memory_exclude_event_ids", default=[],
)


def set_retrieval_context(
    *,
    stream_id: str | None = None,
    exclude_event_ids: list[str] | None = None,
) -> None:
    """Agent 実行前にセッションコンテキストをセットする。

    graph.py の _generation_node から呼ばれる。
    """
    _stream_id_var.set(stream_id)
    _exclude_event_ids_var.set(exclude_event_ids or [])


def reset_retrieval_context() -> None:
    """テスト用: contextvars をデフォルトにリセットする。"""
    _stream_id_var.set(None)
    _exclude_event_ids_var.set([])


# ─── MCP サーバー ──────────────────────────────────────────────────

_mcp = None


def _get_mcp():
    """FastMCP インスタンスを遅延初期化する。"""
    global _mcp
    if _mcp is None:
        from fastmcp import FastMCP
        _mcp = FastMCP(
            "aibyss-retrieve-memory",
            instructions=(
                "記憶・知識ベース検索ツール。"
                "過去の会話や知識ベースから関連情報を検索します。"
                "「前に話した〜」「さっきの〜」など過去への言及がある場合に使用してください。"
            ),
        )
        _register_tools(_mcp)
    return _mcp


def _register_tools(mcp):
    """ツールを MCP サーバーに登録する。"""

    @mcp.tool()
    def retrieve_memory(query: str, top_k: int = 5) -> str:
        """
        過去の会話や知識ベースから関連情報を検索する。

        以下の場合に使用する:
        - 「前に話した〜」「さっきの〜」「以前〜」など過去への言及
        - キャラクターや設定についての具体的な質問
        - 迷った場合はこちらを選ぶ（呼ばない判断ミスの方がコストが高い）
        """
        return _retrieve_memory_impl(query, top_k)


def _retrieve_memory_impl(query: str, top_k: int = 5) -> str:
    """
    CompositeRetriever で記憶・知識ベース検索を実行する。

    MCP ツールおよび直接呼び出しの両方から使用可能。
    """
    # 遅延 import で循環参照を回避 (retrieve_memory.py → graph.py → mcp_servers)
    from ..graph import _build_retriever_from_env

    stream_id = _stream_id_var.get()
    exclude_event_ids = _exclude_event_ids_var.get()

    # 環境変数から RAG 設定を取得
    kb_path = os.environ.get("L2_KB_PATH", "./data/index")
    rag_top_k = int(os.environ.get("L2_RAG_TOP_K", str(top_k)))

    logger.info(
        "記憶検索実行: %r (top_k=%d, stream_id=%s)",
        query, rag_top_k, stream_id or "None",
    )

    try:
        retriever = _build_retriever_from_env(
            kb_path=kb_path,
            stream_id=stream_id,
            exclude_event_ids=exclude_event_ids,
        )
        docs = retriever.retrieve(query, top_k=rag_top_k)
    except Exception as exc:
        logger.warning("記憶検索失敗: %s", exc)
        return f"記憶検索でエラーが発生しました: {exc}"

    if not docs:
        logger.info("記憶検索完了: %r → 結果なし", query)
        return "関連する記憶は見つかりませんでした。"

    # RetrievedDoc リストを Agent が読める文字列に変換
    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        lines.append(
            f"[{i}] (source: {doc.source}, score: {doc.score:.2f})\n{doc.text}"
        )
    result = "\n\n---\n\n".join(lines)
    logger.info("記憶検索完了: %r → %d 件 (%d chars)", query, len(docs), len(result))
    return result


# 直接呼び出し用のエイリアス
retrieve_memory = _retrieve_memory_impl


def get_server():
    """MCP サーバーインスタンスを返す（遅延初期化）。"""
    return _get_mcp()


if __name__ == "__main__":
    server = get_server()
    server.run()
