"""
web_search.py — MCP Web Search サーバー

Tavily Search API を使ったインターネット検索ツールを MCP サーバーとして提供する。
全 AITuber キャラクター（ミミ/ちさめ/さくら）が共有して使用する。

【前提】
  TAVILY_API_KEY 環境変数が必要。
  uv sync --extra tools でインストール。

【使い方】
  # スタンドアロン起動（デバッグ用）
  uv run python -m lab_lounge.mcp_servers.web_search

  # LangGraph Agent から使用（通常はこちら）
  graph.py の _build_agent_graph() が自動的に接続する。
"""

import logging
import os

logger = logging.getLogger(__name__)

_mcp = None


def _get_mcp():
    """FastMCP インスタンスを遅延初期化する。"""
    global _mcp
    if _mcp is None:
        from fastmcp import FastMCP
        _mcp = FastMCP(
            "aibyss-web-search",
            instructions=(
                "インターネット検索ツール。"
                "最新情報や事実確認が必要な質問に対して使用してください。"
            ),
        )
        _register_tools(_mcp)
    return _mcp


def _register_tools(mcp):
    """ツールを MCP サーバーに登録する。"""

    @mcp.tool()
    def web_search(query: str, max_results: int = 5, time_range: str | None = None) -> str:
        """
        インターネットで情報を検索する。

        最新のニュース、天気、事実確認など、リアルタイムの情報が必要な場合に使用する。
        検索クエリは日本語または英語で指定可能。

        Args:
            query:       検索クエリ。
            max_results: 取得件数 (既定 5)。
            time_range:  結果を直近に絞る期間。"day" / "week" / "month" / "year" のいずれか、
                         または None (期間指定なし)。モデル・製品・バージョン・ニュースなど
                         **時事性が高く鮮度が重要な検索では設定を推奨**。不正値は無視される。
        """
        return _web_search_impl(query, max_results, time_range)


# Tavily が受け付ける time_range の許容値 (これ以外は無視してクラッシュを防ぐ)。
_VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})


def _web_search_impl(query: str, max_results: int = 5, time_range: str | None = None) -> str:
    """
    Tavily API でインターネット検索を実行する。

    MCP ツールおよび直接呼び出しの両方から使用可能。time_range を渡すと結果を直近に
    絞る (鮮度重視の検索用)。許容値以外は無視する。
    """
    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise ImportError(
            "langchain-tavily が必要です。"
            " uv sync --extra tools でインストールしてください。"
        ) from exc

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY が設定されていません。"
            " https://tavily.com/ で API キーを取得し、.env に追加してください。"
        )

    # time_range は許容値のみ採用 (不正値は無視)。kwargs を条件付きで組むことで、
    # 未指定時は従来どおり TavilySearch(max_results, topic, search_depth) のまま保つ。
    tavily_kwargs: dict = {
        "max_results": max_results,
        "topic": "general",
        "search_depth": "basic",
    }
    effective_range = time_range if time_range in _VALID_TIME_RANGES else None
    if effective_range:
        tavily_kwargs["time_range"] = effective_range

    logger.info(
        "Web 検索実行: %r (max_results=%d, time_range=%s)",
        query, max_results, effective_range or "なし",
    )

    tool = TavilySearch(**tavily_kwargs)
    results = tool.invoke(query)

    logger.info("Web 検索完了: %r → %d chars", query, len(str(results)))
    return str(results)


# 直接呼び出し用のエイリアス
web_search = _web_search_impl


def get_server():
    """MCP サーバーインスタンスを返す（遅延初期化）。"""
    return _get_mcp()


if __name__ == "__main__":
    server = get_server()
    server.run()
