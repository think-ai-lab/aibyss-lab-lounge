"""
test_graph_agent.py — Agent 化した graph のテスト
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.graph import (
    _is_tools_enabled,
    run_graph,
)
from lab_lounge.llm import LLMResult


class TestToolsEnabled:
    """ツールモードの有効/無効テスト。"""

    def test_default_disabled(self):
        assert _is_tools_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_ENABLE_TOOLS", "true")
        assert _is_tools_enabled() is True

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_ENABLE_TOOLS", "false")
        assert _is_tools_enabled() is False


class TestRunGraphSimpleMode:
    """従来互換の単一ノードモード（ツール無効）。"""

    def test_simple_mode_returns_llm_result(self, monkeypatch):
        """ツール無効時は従来の単一ノード構成で動作する。"""
        mock_result = LLMResult(
            text="テスト応答",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph.call_llm", return_value=mock_result):
            result = run_graph(
                "テスト入力",
                model="test-model",
                provider="openai",
            )

        assert result.text == "テスト応答"
        assert result.model == "test-model"

    def test_simple_mode_passes_system_prompt(self, monkeypatch):
        """システムプロンプトが LLM に渡される。"""
        mock_result = LLMResult(
            text="応答",
            model="m",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph.call_llm", return_value=mock_result) as mock_llm:
            run_graph(
                "入力",
                model="m",
                provider="openai",
                system_prompt="テストプロンプト",
            )

        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("system_prompt") == "テストプロンプト"


class TestRunGraphAgentMode:
    """Agent モード（ツール有効）。"""

    def test_agent_mode_with_no_tools_falls_back(self, monkeypatch):
        """ツール読み込み失敗時は単一ノードにフォールバック。"""
        monkeypatch.setenv("L2_ENABLE_TOOLS", "true")

        mock_result = LLMResult(
            text="フォールバック応答",
            model="test",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph._load_mcp_tools", return_value=[]), \
             patch("lab_lounge.graph.call_llm", return_value=mock_result):
            result = run_graph(
                "テスト",
                model="test",
                provider="openai",
            )

        assert result.text == "フォールバック応答"
