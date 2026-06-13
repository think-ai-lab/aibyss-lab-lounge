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

    def test_time_range_passed_when_valid(self):
        """time_range が許容値なら TavilySearch に渡る (鮮度重視の検索)。"""
        mock_tavily = MagicMock()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "results"
        mock_tavily.TavilySearch.return_value = mock_tool

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            from lab_lounge.mcp_servers.web_search import web_search
            web_search("最新モデル", time_range="week")

        mock_tavily.TavilySearch.assert_called_once_with(
            max_results=5,
            topic="general",
            search_depth="basic",
            time_range="week",
        )

    def test_invalid_time_range_ignored(self):
        """不正な time_range は無視され、TavilySearch には渡さない (クラッシュ防止)。"""
        mock_tavily = MagicMock()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "results"
        mock_tavily.TavilySearch.return_value = mock_tool

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            from lab_lounge.mcp_servers.web_search import web_search
            web_search("query", time_range="yesterday")  # 許容外

        mock_tavily.TavilySearch.assert_called_once_with(
            max_results=5,
            topic="general",
            search_depth="basic",
        )

    def test_no_time_range_keeps_legacy_call(self):
        """time_range 未指定なら従来どおりの呼び出し (後方互換)。"""
        mock_tavily = MagicMock()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "results"
        mock_tavily.TavilySearch.return_value = mock_tool

        with patch.dict("sys.modules", {"langchain_tavily": mock_tavily}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            from lab_lounge.mcp_servers.web_search import web_search
            web_search("query")

        mock_tavily.TavilySearch.assert_called_once_with(
            max_results=5,
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
