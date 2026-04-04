"""
test_graph.py — graph.run_graph テスト

langgraph / LLM 呼び出しをモックして実 API なしで検証する。
langgraph がインストールされていなくても動作する。
"""

import builtins
import pytest
from unittest.mock import MagicMock, patch

from lab_lounge.llm import LLMResult


FAKE_RESULT = LLMResult(
    text="グラフ応答テスト",
    model="gpt-4o-mini",
    input_tokens=20,
    output_tokens=10,
    latency_ms=300,
    finish_reason="stop",
)


class TestRunGraph:
    def test_run_graph_returns_llm_result(self):
        """_build_simple_graph をモックして run_graph が LLMResult を返すことを確認する"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("テスト", model="gpt-4o-mini", provider="openai")

        assert isinstance(result, LLMResult)
        assert result.text == "グラフ応答テスト"

    def test_run_graph_invoke_called_with_correct_state(self):
        """graph.invoke に正しい初期 state が渡る"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト入力",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("テスト入力", model="gpt-4o-mini", provider="openai")

        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["text"] == "テスト入力"
        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["provider"] == "openai"
        assert call_args["result"] is None

    def test_run_graph_returns_correct_text(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("x", model="m")

        assert result.text == "グラフ応答テスト"
        assert result.model == "gpt-4o-mini"

    def test_run_graph_raises_if_result_is_none(self):
        """Graph が result=None を返した場合は RuntimeError"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": None,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            with pytest.raises(RuntimeError):
                run_graph("テスト", model="gpt-4o-mini")

    def test_build_simple_graph_raises_import_error_without_langgraph(self):
        """langgraph が未インストールの場合、ImportError を送出する"""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langgraph" in name:
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        from lab_lounge.graph import _build_simple_graph
        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="langgraph"):
                _build_simple_graph()


class TestRunGraphWithMetadata:
    """run_metadata を渡したとき config["metadata"] に反映されるテスト"""

    def test_run_metadata_passed_to_invoke_config(self):
        """run_metadata が graph.invoke の config["metadata"] に渡ること"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }
        meta = {"aibyss.trace_id": "t1", "aibyss.stream_id": "s1"}

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("x", model="m", run_metadata=meta)

        positional_args = mock_graph.invoke.call_args[0]
        config_arg = positional_args[1]
        assert config_arg == {"metadata": meta}

    def test_no_run_metadata_passes_none_config(self):
        """run_metadata=None のとき config は None で invoke されること"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("x", model="m")

        positional_args = mock_graph.invoke.call_args[0]
        config_arg = positional_args[1]
        assert config_arg is None

    def test_run_metadata_does_not_affect_result(self):
        """run_metadata を渡しても LLMResult の内容は変わらないこと"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }
        meta = {"aibyss.trace_id": "t1"}

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("x", model="m", run_metadata=meta)

        assert result.text == FAKE_RESULT.text
        assert result.model == FAKE_RESULT.model
