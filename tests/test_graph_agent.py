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
# Sprint Axis D Block 3: tool_routing ガイダンス + BubbleToolCallbackHandler
# ═══════════════════════════════════════════════════════════════════
#
# TD-2 の context wrapper テストは削除。Agent 化により RAG context は
# LLM が自分で retrieve_memory ツールを呼んで取得するため、事前注入が不要に。
# 代わりに tool_routing ガイダンスの存在と BubbleToolCallbackHandler の
# 動作を検証する。


class TestToolRoutingGuidance:
    """Skills 定義ファイルベースのガイダンス生成を検証する。

    Sprint Axis D Block 4: _TOOL_ROUTING_GUIDANCE ハードコードを
    skills/common/tool_routing.md に移行。
    """

    def test_skills_contain_retrieve_memory(self):
        """Skills プロンプトに retrieve_memory ガイダンスが含まれる。"""
        from lab_lounge.skill_loader import build_skills_prompt
        result = build_skills_prompt("mimi")
        assert "retrieve_memory" in result
        assert "web_search" in result

    def test_skills_encourage_retrieve_memory_when_unsure(self):
        """「迷ったら retrieve_memory」の指示が Skills に含まれる。"""
        from lab_lounge.skill_loader import build_skills_prompt
        result = build_skills_prompt("mimi")
        assert "迷った" in result


class TestBubbleToolCallbackHandler:
    """BubbleToolCallbackHandler のツール呼び出し時 bubble 発行を検証する。"""

    def test_retrieve_memory_publishes_searching_bubble(self):
        """retrieve_memory_tool → bubble(searching) が発行される。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("mimi", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "retrieve_memory_tool"}, "test query")

        assert len(published) == 1
        assert published[0]["payload"]["step"] == "searching"
        assert published[0]["payload"]["character"] == "mimi"

    def test_web_search_publishes_searching_bubble(self):
        """web_search_tool → bubble(searching) が発行される (text は web_search キー)。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("chisame", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "web_search_tool"}, "test query")

        assert len(published) == 1
        assert published[0]["payload"]["step"] == "searching"

    def test_unknown_tool_does_not_publish(self):
        """未知のツール名では bubble を発行しない。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("mimi", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "unknown_tool"}, "test")

        assert len(published) == 0


class TestAskCharacterToolRegistration:
    """ask_character ツール登録を検証する (Phase 3)。"""

    def test_ask_character_registered(self):
        """ask_character_tool がツールリストに含まれる。"""
        from lab_lounge.graph import _load_mcp_tools
        tools = _load_mcp_tools()
        tool_names = {t.name for t in tools}
        assert "ask_character_tool" in tool_names

    def test_bubble_handler_has_ask_character(self):
        """BubbleToolCallbackHandler に ask_character のマッピングがある。"""
        from lab_lounge.graph import BubbleToolCallbackHandler
        assert "ask_character_tool" in BubbleToolCallbackHandler.TOOL_MESSAGE_KEY


class TestRetrieveMemoryToolRegistration:
    """L2_ENABLE_RAG による retrieve_memory ツール登録の制御を検証する。"""

    def test_rag_enabled_registers_retrieve_memory(self, monkeypatch):
        """L2_ENABLE_RAG=true で retrieve_memory_tool が登録される。"""
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is True

    def test_rag_disabled_does_not_register(self, monkeypatch):
        """L2_ENABLE_RAG=false で retrieve_memory は登録されない。"""
        monkeypatch.setenv("L2_ENABLE_RAG", "false")
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is False

    def test_rag_default_is_disabled(self, monkeypatch):
        """L2_ENABLE_RAG 未設定ではデフォルト false。"""
        monkeypatch.delenv("L2_ENABLE_RAG", raising=False)
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is False
