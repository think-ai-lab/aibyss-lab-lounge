"""
test_retrieve_memory.py — retrieve_memory MCP サーバーのユニットテスト

Sprint Axis D Block 3: RAG を Agent ツールに統合
"""

from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.mcp_servers.retrieve_memory import (
    _retrieve_memory_impl,
    reset_retrieval_context,
    set_retrieval_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """各テスト後に contextvars をリセット。"""
    yield
    reset_retrieval_context()


class TestRetrieveMemoryImpl:
    """_retrieve_memory_impl の動作を検証する。"""

    def test_returns_formatted_results(self):
        """RetrievedDoc がテキスト形式に変換されること。"""
        from lab_lounge.retriever import RetrievedDoc

        fake_docs = [
            RetrievedDoc(doc_id="d1", text="過去の会話1", score=0.85, source="c2:utterance.final"),
            RetrievedDoc(doc_id="d2", text="知識ベース", score=0.70, source="ai_agents.md"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = fake_docs

        with patch("lab_lounge.graph._build_retriever_from_env",
                    return_value=mock_retriever):
            result = _retrieve_memory_impl("テストクエリ", top_k=5)

        assert "[1]" in result
        assert "過去の会話1" in result
        assert "[2]" in result
        assert "知識ベース" in result
        assert "score: 0.85" in result

    def test_returns_no_results_message(self):
        """検索結果 0 件のとき適切なメッセージを返すこと。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch("lab_lounge.graph._build_retriever_from_env",
                    return_value=mock_retriever):
            result = _retrieve_memory_impl("見つからないクエリ")

        assert "見つかりませんでした" in result

    def test_returns_error_string_on_retriever_failure(self):
        """Retriever 例外時にエラー文字列を返すこと (Agent が理解できる形式)。"""
        with patch("lab_lounge.graph._build_retriever_from_env",
                    side_effect=RuntimeError("C2 接続失敗")):
            result = _retrieve_memory_impl("テスト")

        assert "エラー" in result
        assert "C2 接続失敗" in result

    def test_uses_contextvars_for_stream_id(self):
        """set_retrieval_context で設定した stream_id が retriever に渡ること。"""
        set_retrieval_context(stream_id="test-stream-123", exclude_event_ids=["ev-1"])

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch("lab_lounge.graph._build_retriever_from_env",
                    return_value=mock_retriever) as mock_build:
            _retrieve_memory_impl("テスト")

        _, kwargs = mock_build.call_args
        assert kwargs["stream_id"] == "test-stream-123"
        assert kwargs["exclude_event_ids"] == ["ev-1"]

    def test_default_contextvars_are_none(self):
        """contextvars 未設定時はデフォルト値 (None / []) が渡ること。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch("lab_lounge.graph._build_retriever_from_env",
                    return_value=mock_retriever) as mock_build:
            _retrieve_memory_impl("テスト")

        _, kwargs = mock_build.call_args
        assert kwargs["stream_id"] is None
        assert kwargs["exclude_event_ids"] == []


class TestMCPServer:
    """MCP サーバーインスタンスの生成を検証する。"""

    def test_get_server_returns_fastmcp(self):
        """get_server() が FastMCP インスタンスを返すこと。"""
        from lab_lounge.mcp_servers.retrieve_memory import get_server
        server = get_server()
        assert server is not None
