"""
graph.py — LangGraph 最小 state graph

責務:
  - utterance text を受け取り、LLMResult を返す
  - 1 ノード構成（将来 multi-step / RAG / memory 化を容易にする構造）
  - 実質的に llm.py を呼ぶ薄い graph
  - pipeline.py から run_graph() のみを呼ぶ

【拡張例】
  - ノードを追加して「RAG 検索 → LLM 生成」の 2 ステップに変更する
  - 条件分岐で「応答品質低なら再試行」などのロジックを追加する

【前提パッケージ (real mode)】
  langgraph>=0.2 が必要。
  uv add --optional llm langgraph でインストールしてください。
"""

import logging
from typing import TypedDict

from .llm import LLMResult, call_llm

logger = logging.getLogger(__name__)


# ─── State 定義 ───────────────────────────────────────────────────

class LLMGraphState(TypedDict):
    """グラフの状態。ノード間で引き継ぐデータを定義する。"""

    text: str
    model: str
    provider: str
    result: LLMResult | None


# ─── ノード実装 ───────────────────────────────────────────────────

def _llm_node(state: LLMGraphState) -> LLMGraphState:
    """LLM を呼び出してテキストを生成するノード。"""
    result = call_llm(state["text"], model=state["model"], provider=state["provider"])
    return {**state, "result": result}


# ─── グラフ構築 ───────────────────────────────────────────────────

def _build_graph():
    """
    LangGraph state graph を構築してコンパイルする。

    langgraph が未インストールの場合は ImportError を送出する（lazy import）。
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph が必要です。"
            " uv add --optional llm langgraph でインストールしてください。"
        ) from exc

    builder = StateGraph(LLMGraphState)
    builder.add_node("llm", _llm_node)
    builder.set_entry_point("llm")
    builder.add_edge("llm", END)
    return builder.compile()


# ─── 公開 API ────────────────────────────────────────────────────

def run_graph(
    text: str,
    *,
    model: str,
    provider: str = "openai",
    run_metadata: dict | None = None,
) -> LLMResult:
    """
    utterance text を受け取り、LLMResult を返す。

    Args:
        text:         発話テキスト
        model:        使用するモデル名
        provider:     LLM プロバイダ（"openai" など）
        run_metadata: LangGraph config["metadata"] に渡す dict (optional)。
                      LangSmith が有効なとき trace に添付される。
                      無効のときは渡しても副作用なし。

    Returns:
        LLMResult

    Raises:
        ImportError:  langgraph が未インストール
        RuntimeError: Graph が result を返さなかった場合（通常発生しない）
    """
    logger.info("Graph 実行開始: model=%s provider=%s", model, provider)
    graph = _build_graph()
    initial_state: LLMGraphState = {
        "text": text,
        "model": model,
        "provider": provider,
        "result": None,
    }
    config = {"metadata": run_metadata} if run_metadata else None
    final_state = graph.invoke(initial_state, config)
    result = final_state["result"]
    if result is None:
        raise RuntimeError("Graph が LLMResult を返しませんでした")
    logger.info("Graph 実行完了")
    return result
