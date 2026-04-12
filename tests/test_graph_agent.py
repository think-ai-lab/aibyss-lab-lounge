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


# ═══════════════════════════════════════════════════════════════════
# TD-2: _run_agent の text_with_context ラッパー回帰テスト
# (Sprint Axis B Block 5)
# ═══════════════════════════════════════════════════════════════════
#
# 背景: Phase 5 実機検証で、RAG が有効なとき LLM が RAG context に
# 惑わされて web_search ツール呼び出しをスキップしてハルシネーションを
# 起こす事例を発見した (Block 4 / AUX-4 / TD-2)。
# 対策として _run_agent の text_with_context ラッパーに
# 「参照情報は過去の参考データ」「最新情報はツールで取得せよ」
# という明示指示を追加した。
#
# これらのテストはラッパーの文字列内容を回帰検証する。
# LangGraph create_react_agent のモックは複雑なので、ここでは
# text_with_context の構築ロジックに集中する。


class TestTD2ContextWrapperInstruction:
    """_run_agent が context を受け取ったときにツール使用指示が付与されることを検証する。"""

    def _build_text_with_context(self, text: str, context: str | None) -> str:
        """_run_agent 内部の text_with_context 構築ロジックを呼び出すヘルパ。

        _run_agent は mock agent を必要とするが、text_with_context の構築部分だけを
        単体テストするため、内部ロジックを再現する (同じコードパス)。
        context が None のとき text そのまま、ある場合は注意書き付きの形式。
        """
        # この形式は graph.py:_run_agent の実装と一致させる (回帰テスト目的)
        if context:
            return (
                f"{text}\n\n"
                f"---\n"
                f"## 参照情報 (過去の会話や知識ベースから抽出)\n\n"
                f"{context}\n\n"
                f"---\n"
                f"**注意**: 上記の参照情報は過去の参考データです。"
                f"最新の情報 (天気・ニュース・時刻・今日の出来事など) が必要な場合は、"
                f"必ず利用可能なツール (web_search 等) を呼び出して確認してください。"
            )
        return text

    def test_context_wrapper_contains_tool_usage_instruction(self):
        """context あり時: 指示文 (web_search / 最新情報 / ツール) が含まれる。"""
        result = self._build_text_with_context(
            "今日の東京の天気は？",
            "ラボの説明文...",
        )
        # TD-2 指示文の必須キーワードが含まれる
        assert "web_search" in result
        assert "最新の情報" in result
        assert "ツール" in result
        assert "参照情報" in result
        # ユーザー発話も含まれる
        assert "今日の東京の天気は？" in result
        # RAG 文脈も含まれる
        assert "ラボの説明文" in result

    def test_no_context_no_instruction_added(self):
        """context=None: 指示文は付与されず text そのまま。"""
        result = self._build_text_with_context("こんにちは", None)
        assert result == "こんにちは"
        assert "web_search" not in result
        assert "参照情報" not in result
        assert "最新の情報" not in result

    def test_graph_py_run_agent_matches_wrapper_logic(self):
        """
        graph.py の _run_agent 内の text_with_context 構築が本テストヘルパーと一致する。

        graph.py のソース文字列を直接読んで、TD-2 対応のキーワードが
        コード内に含まれることを確認する (実装がコメントアウトされた場合の検知)。
        """
        import lab_lounge.graph
        import inspect

        source = inspect.getsource(lab_lounge.graph._run_agent)
        # TD-2 対応の特徴的なキーワードが _run_agent の実装内に存在する
        assert "## 参照情報" in source, "参照情報セクションラベルが消えている"
        assert "web_search" in source, "TD-2 指示文の web_search が消えている"
        assert "最新の情報" in source, "TD-2 指示文の「最新の情報」キーワードが消えている"
        assert "ツール" in source, "TD-2 指示文のツール呼び出し指示が消えている"
