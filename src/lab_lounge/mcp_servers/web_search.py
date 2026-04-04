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
    def web_search(query: str, max_results: int = 5) -> str:
        """
        インターネットで情報を検索する。

        最新のニュース、天気、事実確認など、リアルタイムの情報が必要な場合に使用する。
        検索クエリは日本語または英語で指定可能。
        """
        return _web_search_impl(query, max_results)


def _web_search_impl(query: str, max_results: int = 5) -> str:
    """
    Tavily API でインターネット検索を実行する。

    MCP ツールおよび直接呼び出しの両方から使用可能。
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

    logger.info("Web 検索実行: %r (max_results=%d)", query, max_results)

    tool = TavilySearch(
        max_results=max_results,
        topic="general",
        search_depth="basic",
    )
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
