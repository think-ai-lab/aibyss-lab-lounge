"""
test_web_search.py — MCP Web Search サーバーのテスト
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestWebSearchTool:
    """web_search ツールの単体テスト。"""

    def test_search_returns_results(self):
        mock_tavily = MagicMock()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "検索結果: テスト"
        mock_tavily.TavilySearch.return_value = mock_tool

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            from lab_lounge.mcp_servers.web_search import web_search
            result = web_search("テスト検索")

        assert "テスト" in result
        mock_tool.invoke.assert_called_once_with("テスト検索")

    def test_search_respects_max_results(self):
        mock_tavily = MagicMock()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "results"
        mock_tavily.TavilySearch.return_value = mock_tool

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            from lab_lounge.mcp_servers.web_search import web_search
            web_search("query", max_results=3)

        mock_tavily.TavilySearch.assert_called_once_with(
            max_results=3,
            topic="general",
            search_depth="basic",
        )

    def test_missing_api_key_raises(self):
        mock_tavily = MagicMock()

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {}, clear=False):
            # TAVILY_API_KEY を確実に削除
            os.environ.pop("TAVILY_API_KEY", None)
            from lab_lounge.mcp_servers.web_search import web_search
            with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                web_search("test")


class TestMCPServer:
    """MCP サーバーインスタンスのテスト。"""

    def test_get_server_returns_fastmcp(self):
        from lab_lounge.mcp_servers.web_search import get_server
        server = get_server()
        assert server is not None
        assert server.name == "aibyss-web-search"
