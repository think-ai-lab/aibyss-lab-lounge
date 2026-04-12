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
import operator
import os
from typing import Annotated, Any, TypedDict

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
    # TD-2 対策: 参照情報があっても ReAct Agent がツール呼び出しを省略しないよう、
    # 「参照情報は過去の参考データ」「最新情報が必要ならツールを使え」を明示する。
    # 注: キャラクター system_prompt は変更せず、ここでラップすることで
    # キャラクター音声トーンに影響を与えないようにする。
    messages = []
    if context:
        text_with_context = (
            f"{text}\n\n"
            f"---\n"
            f"## 参照情報 (過去の会話や知識ベースから抽出)\n\n"
            f"{context}\n\n"
            f"---\n"
            f"**注意**: 上記の参照情報は過去の参考データです。"
            f"最新の情報 (天気・ニュース・時刻・今日の出来事など) が必要な場合は、"
            f"必ず利用可能なツール (web_search 等) を呼び出して確認してください。"
        )
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

        logger.info("Agent 実行完了: latency_ms=%d text=%s", latency_ms, response_text)
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


# ═══════════════════════════════════════════════════════════════════
# Pipeline Graph (マルチノード) — routing → retrieval → generation → tts
# ═══════════════════════════════════════════════════════════════════


class PipelineGraphState(TypedDict):
    """パイプライングラフの状態。4 ノード間でデータを引き継ぐ。"""

    # 入力
    text: str
    common: dict                   # stream_id, session_id, trace_id
    speaker_hint: str | None
    utterance_meta: dict | None

    # 設定 (orchestrator が初期値を設定, routing ノードがキャラ別に上書き)
    use_real_llm: bool
    llm_provider: str
    llm_model: str
    enable_rag: bool
    rag_top_k: int
    kb_path: str
    use_real_tts: bool
    tts_provider: str
    tts_voice: str
    tts_speaker: str
    tts_output_dir: str
    system_prompt: str | None
    on_tts_chunk_ready: Any

    # ノード出力
    character_slug: str
    rag_context: str | None
    rag_used: bool
    retrieved_doc_ids: list[str]
    retrieval_latency_ms: int
    answer_mode: str
    llm_text: str
    llm_meta: dict
    tts_meta: dict
    events: Annotated[list[dict], operator.add]  # 各ノードで蓄積


# ─── ノード実装 ──────────────────────────────────────────────────


def _routing_node(state: PipelineGraphState) -> dict:
    """ルーティングノード: キャラクター決定 + utterance.final 発行。"""
    from .pipeline import publish, _publish_bubble
    from .router import route
    from .characters import get_character, load_system_prompt
    from .events import build_utterance_final
    from .debug import write_stt_output

    text = state["text"]
    common = state["common"]

    # キャラクター決定
    decision = route(text, name_hint=state["speaker_hint"])
    character = get_character(decision.speaker)
    try:
        system_prompt = load_system_prompt(character)
    except FileNotFoundError:
        logger.warning(
            "システムプロンプトが見つかりません: %s。プロンプトなしで続行。",
            character.system_prompt_file,
        )
        system_prompt = None

    updates: dict = {
        "character_slug": character.slug,
        "system_prompt": system_prompt,
        "tts_provider": character.tts_provider,
        "tts_voice": character.tts_voice,
        "tts_speaker": character.slug,
    }
    # キャラクター固有の LLM 設定があれば上書き
    if getattr(character, "llm_model", ""):
        updates["llm_provider"] = character.llm_provider
        updates["llm_model"] = character.llm_model

    # utterance.final 発行
    utt_kwargs: dict[str, Any] = state["utterance_meta"] or {}
    utt = build_utterance_final(text=text, seq=0, **utt_kwargs, **common)
    publish(utt)
    write_stt_output(text, state["utterance_meta"])

    # bubble: searching
    _publish_bubble("searching", character.slug, common, links=[utt["event_id"]])

    updates["events"] = [utt]
    return updates


def _build_retriever_from_env(
    kb_path: str,
    stream_id: str | None = None,
    exclude_event_ids: list[str] | None = None,
):
    """
    env 変数に応じて Retriever を構築する。

    - L2_USE_C2_RETRIEVER=false (default) → LocalRetriever 単独 (既存動作)
    - L2_USE_C2_RETRIEVER=true →
        CompositeRetriever([
            LocalRetriever,                          # seed corpus
            C2Retriever(primary),                    # LIKE 検索 (L2_C2_URL)
            RecentC2Retriever(primary, scope),       # 時系列リコール (L2_USE_C2_RECENT=true)
            C2Retriever(readonly),                   # LIKE 検索 (L2_C2_URL_READONLY, optional)
        ])

    【プロファイル分離設計】
    dev プロファイルから prod のデータを read-only で参照するための仕組み。
    - prod モード: L2_C2_URL のみ設定 → 書き込む先 = 読む先 = prod C2 (1 本)
    - dev モード:  L2_C2_URL=dev-c2 + L2_C2_URL_READONLY=prod-c2
                   → 書き込みは dev のみ、読みは dev+prod 両方

    【RecentC2Retriever のスコープ】 (L2_C2_RECENT_SCOPE)
    - global (default): stream_id を渡さず全ストリーム横断で直近 N 件を取得。
      新しい run_loop セッションでも過去セッションの会話を参照できる
      (セッションまたぎメモリ)。
    - session: 現ターンの stream_id に限定。同じ run_loop 内の発話のみ参照
      (セッション内メモリ)。

    【RecentC2Retriever は primary のみ】
    RecentC2Retriever は readonly C2 には呼ばない
    (readonly は prod を想定しており、dev セッションのデータが存在しないため)。

    【exclude_event_ids】
    現ターンの utterance.final.event_id を渡すことで、
    自己参照 (ナルシシスティック RAG) を防止する。
    C2Retriever / RecentC2Retriever 両方に同じ list を渡す。

    関連 env:
        L2_USE_C2_RETRIEVER        — C2Retriever を Composite に組み込むか (default: false)
        L2_C2_URL                  — 主 C2 のベース URL (default: http://localhost:8100)
        L2_C2_URL_READONLY         — 副 C2 のベース URL (dev プロファイル用、空なら無効)
        L2_C2_RETRIEVER_TOP_K      — C2Retriever 単独の取得件数 (default: 3)
        L2_C2_RETRIEVER_BUDGET_MS  — C2 HTTP タイムアウト ms (default: 600)
        L2_USE_C2_RECENT           — RecentC2Retriever を追加するか (default: true)
        L2_C2_RECENT_TOP_K         — RecentC2Retriever の取得件数 (default: 5)
        L2_C2_RECENT_SCOPE         — 検索範囲 (default: global)
                                      global  = 全ストリーム横断 (セッションまたぎメモリ)
                                      session = 現ストリームのみ (セッション内メモリ)

    詳細: docs/design/lab-lounge-retriever.md
    """
    from .retriever import LocalRetriever

    use_c2 = os.environ.get("L2_USE_C2_RETRIEVER", "false").lower() == "true"
    if not use_c2:
        return LocalRetriever(kb_path)

    from .retriever import C2Retriever, CompositeRetriever, RecentC2Retriever
    c2_url = os.environ.get("L2_C2_URL", "http://localhost:8100")
    c2_url_readonly = os.environ.get("L2_C2_URL_READONLY", "").strip()
    c2_top_k = int(os.environ.get("L2_C2_RETRIEVER_TOP_K", "3"))
    c2_budget_s = float(os.environ.get("L2_C2_RETRIEVER_BUDGET_MS", "600")) / 1000
    use_recent = os.environ.get("L2_USE_C2_RECENT", "true").lower() == "true"
    recent_top_k = int(os.environ.get("L2_C2_RECENT_TOP_K", "10"))  # TD-8: 5→10 (会話メモリは高リコール必要)
    recent_scope = os.environ.get("L2_C2_RECENT_SCOPE", "global").strip().lower()
    if recent_scope not in ("global", "session"):
        logger.warning(
            "L2_C2_RECENT_SCOPE に未知の値 '%s' を指定、global にフォールバック",
            recent_scope,
        )
        recent_scope = "global"

    retrievers: list = [
        LocalRetriever(kb_path),
        C2Retriever(
            c2_url,
            top_k=c2_top_k,
            timeout_s=c2_budget_s,
            exclude_event_ids=exclude_event_ids,
        ),
    ]

    # RecentC2Retriever のスコープ決定
    if use_recent:
        if recent_scope == "session":
            # セッション内メモリ: stream_id 指定。なければ追加しない
            if stream_id:
                retrievers.append(
                    RecentC2Retriever(
                        c2_url,
                        stream_id=stream_id,
                        top_k=recent_top_k,
                        timeout_s=c2_budget_s,
                        exclude_event_ids=exclude_event_ids,
                    )
                )
                recent_marker = "+Recent(session)"
            else:
                recent_marker = ""
        else:
            # global: stream_id を渡さず全ストリーム横断
            retrievers.append(
                RecentC2Retriever(
                    c2_url,
                    stream_id=None,
                    top_k=recent_top_k,
                    timeout_s=c2_budget_s,
                    exclude_event_ids=exclude_event_ids,
                )
            )
            recent_marker = "+Recent(global)"
    else:
        recent_marker = ""

    if c2_url_readonly:
        retrievers.append(
            C2Retriever(
                c2_url_readonly,
                top_k=c2_top_k,
                timeout_s=c2_budget_s,
                exclude_event_ids=exclude_event_ids,
            )
        )
        # readonly に対する Recent は呼ばない (別 C2 インスタンスのスコープのため)

    readonly_marker = f" + C2(readonly={c2_url_readonly})" if c2_url_readonly else ""
    logger.info(
        "CompositeRetriever 構築: Local + C2(primary=%s%s)%s (top_k=%d/recent=%d budget_s=%.3f)",
        c2_url, recent_marker, readonly_marker, c2_top_k, recent_top_k, c2_budget_s,
    )

    return CompositeRetriever(retrievers)


def _retrieval_node(state: PipelineGraphState) -> dict:
    """検索ノード: RAG 検索 (無効時はスキップ)。"""
    import time
    from .pipeline import _publish_bubble
    from .debug import write_retrieval

    common = state["common"]
    utt_event_id = state["events"][0]["event_id"]

    if not state["enable_rag"]:
        _publish_bubble("thinking", state["character_slug"], common, links=[utt_event_id])
        return {
            "rag_context": None,
            "rag_used": False,
            "retrieved_doc_ids": [],
            "retrieval_latency_ms": 0,
            "answer_mode": "fallback",
        }

    # RAG 有効
    rag_context: str | None = None
    rag_used = False
    retrieved_doc_ids: list[str] = []
    retrieval_latency_ms = 0
    answer_mode = "fallback"

    try:
        # 現ターンの utterance.final.event_id を exclude に渡してナルシシスティック RAG を防ぐ
        retriever = _build_retriever_from_env(
            state["kb_path"],
            stream_id=common.get("stream_id"),
            exclude_event_ids=[utt_event_id] if utt_event_id else None,
        )
        t0 = time.monotonic()
        docs = retriever.retrieve(state["text"], top_k=state["rag_top_k"])
        retrieval_latency_ms = int((time.monotonic() - t0) * 1000)
        if docs:
            rag_context = "\n\n---\n\n".join(d.text for d in docs)
            retrieved_doc_ids = [d.doc_id for d in docs]
            retrieval_scores = [d.score for d in docs]
            rag_used = True
            answer_mode = "grounded"
            logger.info(
                "RAG 検索完了: latency_ms=%d docs=%d ids=%s",
                retrieval_latency_ms, len(docs), retrieved_doc_ids,
            )
        else:
            retrieval_scores = []
        write_retrieval(retrieved_doc_ids, retrieval_scores, retrieval_latency_ms, rag_enabled=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG 検索失敗 (fallback): %s", exc)
        write_retrieval([], [], retrieval_latency_ms, rag_enabled=True)

    _publish_bubble("thinking", state["character_slug"], common, links=[utt_event_id])

    return {
        "rag_context": rag_context,
        "rag_used": rag_used,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieval_latency_ms": retrieval_latency_ms,
        "answer_mode": answer_mode,
    }


def _generation_node(state: PipelineGraphState) -> dict:
    """生成ノード: LLM 呼び出し + llm.final 発行。"""
    from .pipeline import publish, _publish_bubble
    from .events import build_llm_final
    from .observability import build_run_metadata
    from .debug import write_llm_prompt, write_llm_response

    text = state["text"]
    common = state["common"]
    utt_event_id = state["events"][0]["event_id"]

    if state["use_real_llm"]:
        _run_meta = build_run_metadata(
            stream_id=common["stream_id"],
            session_id=common["session_id"],
            trace_id=common["trace_id"],
            rag_used=state["rag_used"],
            answer_mode=state["answer_mode"],
            retrieval_latency_ms=state["retrieval_latency_ms"],
            retrieved_doc_count=len(state["retrieved_doc_ids"]),
            retrieved_doc_ids=state["retrieved_doc_ids"],
        )
        write_llm_prompt(text, state["rag_context"])
        _llm_result = run_graph(
            text,
            model=state["llm_model"],
            provider=state["llm_provider"],
            context=state["rag_context"],
            system_prompt=state["system_prompt"],
            run_metadata=_run_meta,
        )
        write_llm_response(_llm_result.text)
        llm_text = _llm_result.text
        llm_meta: dict[str, Any] = dict(
            model=_llm_result.model,
            input_tokens=_llm_result.input_tokens,
            output_tokens=_llm_result.output_tokens,
            latency_ms=_llm_result.latency_ms,
            finish_reason=_llm_result.finish_reason,
            rag_used=state["rag_used"],
            answer_mode=state["answer_mode"],
            retrieval_latency_ms=state["retrieval_latency_ms"],
            retrieved_doc_count=len(state["retrieved_doc_ids"]),
            retrieved_doc_ids=state["retrieved_doc_ids"],
        )
    else:
        llm_text = f"ダミー応答: {text}"
        llm_meta = {}

    llm = build_llm_final(text=llm_text, seq=1, links=[utt_event_id], **llm_meta, **common)
    publish(llm)

    _publish_bubble("answering", state["character_slug"], common, links=[llm["event_id"]])

    return {
        "llm_text": llm_text,
        "llm_meta": llm_meta,
        "events": [llm],
    }


def _tts_node(state: PipelineGraphState) -> dict:
    """TTS ノード: OBS 立ち絵切り替え + 音声合成 + tts.done 発行。"""
    from .pipeline import publish, _publish_bubble
    from .events import build_tts_done
    from .tts import _parse_voicepeak_json
    from .obs import set_pose

    common = state["common"]
    llm_text = state["llm_text"]
    llm_event_id = state["events"][1]["event_id"]
    character_slug = state["character_slug"]

    # LLM JSON 応答から pose を抽出して OBS 立ち絵を切り替え
    # (tts.done の直前に実行。未設定 or 不明値は set_pose 内で "neutral" にフォールバック)
    _, _, _, pose_value = _parse_voicepeak_json(llm_text)
    set_pose(character_slug, pose_value or "neutral")

    if state["use_real_tts"]:
        from .tts import synthesize as _synthesize
        _tts_result = _synthesize(
            llm_text,
            provider=state["tts_provider"],
            voice=state["tts_voice"],
            speaker=state["tts_speaker"],
            output_dir=state["tts_output_dir"],
            on_chunk_ready=state["on_tts_chunk_ready"],
        )
        tts_meta: dict[str, Any] = dict(
            audio_url=_tts_result.audio_url,
            duration_ms=_tts_result.duration_ms,
            voice=_tts_result.voice,
            format=_tts_result.format,
            sample_rate=_tts_result.sample_rate,
            speaker=_tts_result.speaker,
        )
    else:
        tts_meta = dict(speaker=state["tts_speaker"])

    tts = build_tts_done(text=llm_text, seq=2, links=[llm_event_id], **tts_meta, **common)
    publish(tts)

    _publish_bubble("done", state["character_slug"], common, links=[tts["event_id"]])

    return {
        "tts_meta": tts_meta,
        "events": [tts],
    }


# ─── パイプライングラフ構築 ──────────────────────────────────────


def _build_pipeline_graph():
    """4 ノードパイプライングラフを構築する。

    langgraph が未インストールの場合は ImportError を送出する。
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(PipelineGraphState)
    builder.add_node("routing", _routing_node)
    builder.add_node("retrieval", _retrieval_node)
    builder.add_node("generation", _generation_node)
    builder.add_node("tts", _tts_node)
    builder.set_entry_point("routing")
    builder.add_edge("routing", "retrieval")
    builder.add_edge("retrieval", "generation")
    builder.add_edge("generation", "tts")
    builder.add_edge("tts", END)
    return builder.compile()


def run_pipeline_graph(initial_state: PipelineGraphState) -> PipelineGraphState:
    """パイプライングラフを実行し、最終状態を返す。

    Raises:
        ImportError: langgraph が未インストール
    """
    graph = _build_pipeline_graph()
    return graph.invoke(initial_state)
