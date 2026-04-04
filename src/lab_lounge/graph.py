"""
graph.py — LangGraph state graph（Agent 対応）

責務:
  - utterance text を受け取り、LLMResult を返す
  - ツール無効時: 単一ノード構成（従来互換）
  - ツール有効時: ReAct Agent（LLM がツール使用を自律判断）
  - pipeline.py から run_graph() のみを呼ぶ

【ツールモード】
  L2_ENABLE_TOOLS=true で Agent モードを有効化。
  MCP サーバー（mcp_servers/web_search.py 等）のツールを LLM が自律的に呼び出す。

【前提パッケージ】
  基本:       uv sync --extra llm    → langgraph, langchain-openai
  ツール:     uv sync --extra tools  → fastmcp, langchain-mcp-adapters, langchain-tavily
  Google:     uv sync --extra llm-google
  Anthropic:  uv sync --extra llm-anthropic
"""

import logging
import os
from typing import TypedDict

from .llm import LLMResult, call_llm

logger = logging.getLogger(__name__)


# ─── State 定義 ───────────────────────────────────────────────────

class LLMGraphState(TypedDict):
    """グラフの状態。ノード間で引き継ぐデータを定義する。"""

    text: str
    model: str
    provider: str
    context: str | None          # RAG で取得した参照テキスト（None = non-RAG）
    system_prompt: str | None    # キャラクター別システムプロンプト（None = 従来動作）
    result: LLMResult | None


# ─── ノード実装（従来互換） ──────────────────────────────────────

def _llm_node(state: LLMGraphState) -> LLMGraphState:
    """LLM を呼び出してテキストを生成するノード。"""
    result = call_llm(
        state["text"],
        model=state["model"],
        provider=state["provider"],
        context=state.get("context"),
        system_prompt=state.get("system_prompt"),
    )
    return {**state, "result": result}


# ─── グラフ構築（従来互換: 単一ノード） ──────────────────────────

def _build_simple_graph():
    """
    単一ノード LangGraph（従来互換）。

    langgraph が未インストールの場合は ImportError を送出する。
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph が必要です。"
            " uv sync --extra llm でインストールしてください。"
        ) from exc

    builder = StateGraph(LLMGraphState)
    builder.add_node("llm", _llm_node)
    builder.set_entry_point("llm")
    builder.add_edge("llm", END)
    return builder.compile()


# ─── Agent 構築（ツール有効時）────────────────────────────────────

def _is_tools_enabled() -> bool:
    """ツールモード（Agent）が有効かどうかを返す。"""
    return os.environ.get("L2_ENABLE_TOOLS", "false").lower() in ("true", "1", "yes")


def _get_llm_for_agent(provider: str, model: str):
    """プロバイダーに応じた LangChain Chat モデルを返す。"""
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)


def _load_mcp_tools():
    """MCP サーバーからツールを LangChain ツールとして読み込む。"""
    try:
        from .mcp_servers.web_search import web_search
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def web_search_tool(query: str) -> str:
            """インターネットで情報を検索する。最新のニュース、天気、事実確認など、リアルタイムの情報が必要な場合に使用する。"""
            return web_search(query)

        logger.info("ツール登録完了: web_search")
        return [web_search_tool]

    except ImportError as exc:
        logger.warning("ツール読み込み失敗 (import): %s", exc)
        return []
    except Exception as exc:
        logger.warning("ツール読み込み失敗: %s。ツールなしで続行。", exc)
        return []


def _build_agent_graph(provider: str, model: str, system_prompt: str | None = None):
    """
    ツール付き ReAct Agent グラフを構築する。

    MCP サーバーからツールを読み込み、LLM がツール使用を自律判断する。
    ツール読み込み失敗時は単一ノード構成にフォールバック。
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise ImportError(
            "langgraph が必要です。"
            " uv sync --extra llm でインストールしてください。"
        ) from exc

    tools = _load_mcp_tools()
    if not tools:
        logger.info("ツールなし。単一ノード構成にフォールバック。")
        return None

    llm = _get_llm_for_agent(provider, model)
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )
    logger.info("ReAct Agent 構築完了: tools=%d model=%s", len(tools), model)
    return agent


# ─── 公開 API ────────────────────────────────────────────────────

def run_graph(
    text: str,
    *,
    model: str,
    provider: str = "openai",
    context: str | None = None,
    system_prompt: str | None = None,
    run_metadata: dict | None = None,
) -> LLMResult:
    """
    utterance text を受け取り、LLMResult を返す。

    L2_ENABLE_TOOLS=true の場合は ReAct Agent を使用。
    それ以外は従来の単一ノード構成。

    Args:
        text:          発話テキスト
        model:         使用するモデル名
        provider:      LLM プロバイダ（"openai" / "google" / "anthropic"）
        context:       RAG で取得した参照テキスト（省略時は non-RAG 動作）
        system_prompt: キャラクター別システムプロンプト（省略時は従来動作）
        run_metadata:  LangGraph config["metadata"] に渡す dict (optional)。

    Returns:
        LLMResult

    Raises:
        ImportError:  langgraph が未インストール
        RuntimeError: Graph が result を返さなかった場合
    """
    logger.info(
        "Graph 実行開始: model=%s provider=%s rag=%s tools=%s",
        model, provider, context is not None, _is_tools_enabled(),
    )

    import time

    if _is_tools_enabled():
        agent = _build_agent_graph(provider, model, system_prompt)
        if agent is not None:
            return _run_agent(agent, text, model, context, system_prompt, run_metadata)

    # 従来互換: 単一ノード構成
    graph = _build_simple_graph()
    initial_state: LLMGraphState = {
        "text": text,
        "model": model,
        "provider": provider,
        "context": context,
        "system_prompt": system_prompt,
        "result": None,
    }
    config = {"metadata": run_metadata} if run_metadata else None
    final_state = graph.invoke(initial_state, config)
    result = final_state["result"]
    if result is None:
        raise RuntimeError("Graph が LLMResult を返しませんでした")
    logger.info("Graph 実行完了")
    return result


def _run_agent(
    agent,
    text: str,
    model: str,
    context: str | None,
    system_prompt: str | None,
    run_metadata: dict | None,
) -> LLMResult:
    """ReAct Agent を実行し、LLMResult に変換する。"""
    import time

    t0 = time.monotonic()

    # Agent への入力メッセージを組み立て
    messages = []
    if context:
        text_with_context = f"{text}\n\n---\n## 参照情報\n\n{context}"
    else:
        text_with_context = text

    input_data = {"messages": [{"role": "user", "content": text_with_context}]}
    config = {"metadata": run_metadata} if run_metadata else None

    try:
        result = agent.invoke(input_data, config)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Agent の最終メッセージから応答テキストを抽出
        final_messages = result.get("messages", [])
        response_text = ""
        usage = {}

        if final_messages:
            # 最後の AI メッセージを探す（ToolMessage をスキップ）
            for msg in reversed(final_messages):
                role = getattr(msg, "type", None) or getattr(msg, "role", "")
                if role in ("ai", "assistant"):
                    content = getattr(msg, "content", "")
                    # Gemini はリスト形式で返すことがある
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        response_text = "".join(text_parts)
                    else:
                        response_text = str(content)
                    usage = getattr(msg, "usage_metadata", None) or {}
                    break

        logger.info("Agent 実行完了: latency_ms=%d", latency_ms)
        return LLMResult(
            text=response_text,
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            finish_reason="stop",
        )
    except Exception as exc:
        logger.error("Agent 実行失敗: %s。単一ノードにフォールバック。", exc)
        # フォールバック: 従来の単一ノード構成
        return call_llm(
            text,
            model=model,
            provider="openai",
            context=context,
            system_prompt=system_prompt,
        )
