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

import functools
import logging
import operator
import os
from typing import Annotated, Any, TypedDict

from .character_status import CharacterStatus, CharacterStatusManager
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
    """LLM を呼び出してテキストを生成するノード。

    旧経路 (= 単一ノードフォールバック)。LLMGraphState は character_slug を持たないため、
    ログ強化 L-3 の caller_slug は None 渡し (= 旧挙動維持、ログ "?" 表示)。
    """
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


def _is_rag_enabled() -> bool:
    """RAG (retrieve_memory ツール) が有効かどうかを返す。"""
    return os.environ.get("L2_ENABLE_RAG", "false").lower() in ("true", "1", "yes")


def _load_mcp_tools(
    character_slug: str | None = None,
    disable_tools: "list[str] | None" = None,
):
    """MCP サーバーからツールを LangChain ツールとして読み込む。

    Args:
        character_slug: 呼出元キャラ slug。ログ強化 L-2 (Phase 0.5-A 後) で追加。
                        ターン毎に Agent 構築されるため、ログを「どのキャラの Agent
                        のためのツール登録か」識別できるようにする。
        disable_tools:  特定ツールを除外するためのリスト (例: ["ask_character"])。
                        Phase 0.5-A 案 W'-3 + バグ 3 修正 (案 A) で追加。
                        実走 logs/runs/run_loop_20260508_181051.log で観察された
                        fallback パス + ask_character + 並行 TTS の deadlock を
                        回避するため、_approved_synthesize_fallback 経路で
                        ["ask_character"] を渡して同ツールを無効化する。
                        通常応答ターン (= bg_result=ready 経路) では None で
                        全ツール有効 (= キャラ間協働応答が動作)。
    """
    disable_set = set(disable_tools or [])
    # ログ強化 L-2: 3 行に分かれていたツール登録ログを 1 行に集約 (冗長削減)。
    # 失敗時のみ warning で個別に出す。
    char_tag = f"[character={character_slug or '?'}]"
    try:
        from langchain_core.tools import tool as lc_tool

        tools = []
        registered: list[str] = []  # ログ用、登録成功したツール名 (1 行集約)

        # web_search ツール
        if "web_search" not in disable_set:
            try:
                from .mcp_servers.web_search import web_search

                @lc_tool
                def web_search_tool(query: str) -> str:
                    """インターネットで情報を検索する。最新のニュース、天気、事実確認など、リアルタイムの情報が必要な場合に使用する。"""
                    return web_search(query)

                tools.append(web_search_tool)
                registered.append("web_search")
            except Exception as exc:
                logger.warning("web_search ツール読み込み失敗 %s: %s", char_tag, exc)

        # retrieve_memory ツール (L2_ENABLE_RAG=true のとき)
        if _is_rag_enabled() and "retrieve_memory" not in disable_set:
            try:
                from .mcp_servers.retrieve_memory import retrieve_memory

                @lc_tool
                def retrieve_memory_tool(query: str) -> str:
                    """過去の会話や知識ベースから関連情報を検索する。「前に話した〜」「さっきの〜」「以前〜」など過去への言及がある場合に使用する。迷ったら使う。"""
                    return retrieve_memory(query)

                tools.append(retrieve_memory_tool)
                registered.append("retrieve_memory(RAG)")
            except Exception as exc:
                logger.warning("retrieve_memory ツール読み込み失敗 %s: %s", char_tag, exc)

        # ask_character ツール (基本は常に登録、disable_tools で fallback パス時に無効化)
        if "ask_character" not in disable_set:
            try:
                from .mcp_servers.ask_character import ask_character

                @lc_tool
                def ask_character_tool(character_slug: str, question: str) -> str:
                    """他のAITuberキャラクターに質問する。自分の専門外の質問や、別の視点が欲しい場合に使用する。character_slug は相手の識別子 (mimi/chisame/sakura/ruka/octamaid)。自分自身には質問しないこと。1 応答で最大 2 回まで。"""
                    return ask_character(character_slug, question)

                tools.append(ask_character_tool)
                registered.append("ask_character")
            except Exception as exc:
                logger.warning("ask_character ツール読み込み失敗 %s: %s", char_tag, exc)

        if not tools:
            logger.warning("有効なツールが 0 件 %s。ツールなしで続行。", char_tag)
        else:
            disabled_log = (
                f" (disabled={sorted(disable_set)})" if disable_set else ""
            )
            logger.info(
                "ツール登録完了 %s: %s%s", char_tag, ", ".join(registered), disabled_log,
            )

        return tools

    except ImportError as exc:
        logger.warning("ツール読み込み失敗 %s (import): %s", char_tag, exc)
        return []
    except Exception as exc:
        logger.warning("ツール読み込み失敗 %s: %s。ツールなしで続行。", char_tag, exc)
        return []


@functools.lru_cache(maxsize=None)
def _get_character_response_schema(character_slug: str):
    """キャラ別の応答 Pydantic スキーマを動的生成して返す (Phase 0.5-A フェーズ 8)。

    create_agent の ``response_format`` に渡すことで、LLM が以下の strict JSON 形式
    で応答することを強制する::

        {"response": "...", "emotion": {<キャラ別キー>: 0-100, ...},
         "speed": 50-200, "pose": "..."}

    Markdown コードブロック (```json ... ```) や絵文字、装飾文字 (`**`、`：` 等)
    の混入を構造的に防ぐ。VOICEPEAK CLI の引数破壊バグ (実走 2026-05-08 で観測)
    の根本対策。

    voicepeak_emotion_keys が空のキャラ (octamaid 等) は emotion フィールド無し
    のスキーマを返す。

    @lru_cache でキャッシュしているのは、生成された Pydantic クラスを LangChain
    側が schema として参照する際に「同一スキーマ ≒ 同一クラス」の同一性を維持
    するため (毎回新規クラスを生成すると参照同一性が崩れる可能性)。

    Args:
        character_slug: キャラクター slug (mimi / chisame / sakura / octamaid / ruka 等)

    Returns:
        Pydantic BaseModel サブクラス (動的生成)
    """
    from pydantic import Field, create_model
    from .characters import get_character

    # 未登録 slug は KeyError を投げる仕様なので try/except で吸収して generic に
    try:
        char = get_character(character_slug)
    except KeyError:
        char = None
    if char is None:
        # 未登録 slug は generic スキーマ (emotion 無し、最小フィールドのみ)
        return create_model(
            "GenericResponse",
            response=(str, Field(..., description="応答テキスト")),
            speed=(int, Field(default=100, ge=50, le=200, description="発話速度")),
            pose=(str, Field(default="neutral", description="OBS 立ち絵 pose")),
        )

    emotion_keys = char.voicepeak_emotion_keys

    if not emotion_keys:
        # emotion 不要キャラ (octamaid / ruka 等、voicevox 等の non-emotion TTS)
        return create_model(
            f"{character_slug.capitalize()}Response",
            response=(str, Field(..., description="ユーザーへの応答テキスト")),
            speed=(int, Field(default=100, ge=50, le=200, description="発話速度")),
            pose=(str, Field(default="neutral", description="OBS 立ち絵 pose")),
        )

    # キャラ別の emotion フィールドを動的構築
    emotion_fields = {
        k: (int, Field(default=0, ge=0, le=100, description=f"{k} 強度 (0-100)"))
        for k in emotion_keys
    }
    emotion_schema = create_model(
        f"{character_slug.capitalize()}Emotion",
        **emotion_fields,
    )

    return create_model(
        f"{character_slug.capitalize()}Response",
        response=(str, Field(..., description="ユーザーへの応答テキスト")),
        emotion=(emotion_schema, Field(..., description="感情パラメータ")),
        speed=(int, Field(default=100, ge=50, le=200, description="発話速度")),
        pose=(str, Field(default="neutral", description="OBS 立ち絵 pose")),
    )


def _build_agent_graph(
    provider: str,
    model: str,
    system_prompt: str | None = None,
    character_slug: str | None = None,
    disable_tools: "list[str] | None" = None,
):
    """
    ツール付き ReAct Agent グラフを構築する。

    MCP サーバーからツールを読み込み、LLM がツール使用を自律判断する。
    ツール読み込み失敗時は単一ノード構成にフォールバック。

    Sprint Axis D Block 4: Skills 定義ファイルから行動判断基準を読み込み、
    system_prompt と結合して Agent に注入する。

    Phase 0.5-A フェーズ 8: 旧 ``langgraph.prebuilt.create_react_agent`` (deprecated)
    から ``langchain.agents.create_agent`` (新 API、LangChain v1) に移行。
    Anthropic Claude 4.x の "assistant message prefill" 制約と衝突して
    response_format=Pydantic 利用時に 400 エラーが出ていた問題を解消する。
    新 API は tool-based structured output (ProviderStrategy) を使うため、
    Markdown コードブロック / 絵文字 / 装飾文字なしの strict JSON が返る
    (実測でレイテンシも -15% 短縮)。

    Args:
        provider:        LLM プロバイダ (openai/anthropic/google)
        model:           モデル名
        system_prompt:   キャラクター別 system prompt
        character_slug:  キャラ slug (= ログ識別 + Skills 読込 + response_format)
        disable_tools:   除外するツール名のリスト (= 案 W'-3 バグ 3 修正、案 A)
    """
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise ImportError(
            "langchain (v1+) が必要です。"
            " uv sync --extra llm でインストールしてください。"
        ) from exc

    # ログ強化 L-2: character_slug を渡して、ツール登録ログにキャラ情報を含める
    # バグ 3 修正: disable_tools を伝播して fallback パスでは ask_character を除外
    tools = _load_mcp_tools(
        character_slug=character_slug,
        disable_tools=disable_tools,
    )
    if not tools:
        logger.info(
            "ツールなし [character=%s]。単一ノード構成にフォールバック。",
            character_slug or "?",
        )
        return None

    # Skills 定義ファイルから行動判断基準を読み込み、system_prompt と結合
    from .skill_loader import build_skills_prompt
    skills_text = build_skills_prompt(character_slug or "") if character_slug else ""

    parts = [p for p in [system_prompt, skills_text] if p]
    combined_prompt = "\n\n".join(parts) if parts else None

    llm = _get_llm_for_agent(provider, model)
    # 並列ツール呼び出しを抑制する (S999 で観測された複数 ask_character の同時発火対策)。
    # parallel_tool_calls=False を bind_tools 時に渡すことで、LLM の 1 応答内で
    # 同時に複数のツールが呼ばれなくなる (= 同一ターンで ask_character を 2 回並列発行
    # しなくなる)。LangChain v0.3+ で OpenAI / Anthropic 共通の引数。
    # Gemini は本パラメータ非対応のため bind_tools せず素通しする (Skills の文言で誘導)。
    bound_llm = (
        llm.bind_tools(tools, parallel_tool_calls=False)
        if provider in ("openai", "anthropic")
        else llm
    )

    # Phase 0.5-A フェーズ 8: キャラ別 Pydantic スキーマで構造化出力を強制。
    # スキーマは voicepeak_emotion_keys から動的生成 (mimi/chisame/sakura で
    # emotion キーが異なる)。response_format=Pydantic で LLM が必ず JSON
    # オブジェクトを返すようになり、Markdown コードブロックや絵文字の混入
    # による VOICEPEAK CLI 引数破壊を構造的に防ぐ。
    response_schema = (
        _get_character_response_schema(character_slug) if character_slug else None
    )

    kwargs: dict = {
        "model": bound_llm,
        "tools": tools,
        "system_prompt": combined_prompt,
    }
    if response_schema is not None:
        kwargs["response_format"] = response_schema

    agent = create_agent(**kwargs)
    # ログ強化 L-2: キャラ識別子を先頭に出して、ターン毎にどのキャラの Agent か判別容易に
    logger.info(
        "Agent 構築完了 (新 API) [character=%s]: tools=%d model=%s skills=%s "
        "parallel_tool_calls=%s structured_output=%s",
        character_slug or "?", len(tools), model, bool(skills_text),
        False if provider in ("openai", "anthropic") else "(provider非対応)",
        response_schema.__name__ if response_schema else "(無効)",
    )
    return agent


# ─── Agent ツール呼び出し時の bubble.update ────────────────────────

class BubbleToolCallbackHandler:
    """Agent がツールを呼んだときに bubble.update を発行する LangChain コールバック。

    Sprint Axis D Block 3: HUD に「記憶検索中」「Web検索中」を動的表示。

    LangChain の CallbackManager が期待する属性 (ignore_chain, raise_error 等) を
    提供するため、必要最小限のプロトコルを実装する。
    BaseCallbackHandler を継承しないのは langchain_core の import を
    グラフ構築時まで遅延させるため。

    step は常に "searching" (V2 STEP_ORDER で統一)。
    text で bubble_messages.json のキャラ別メッセージを使い分ける:
    - retrieve_memory_tool → bubble_messages[character]["searching"]
    - web_search_tool → bubble_messages[character]["web_search"]
    """

    # LangChain CallbackManager が参照する属性 (BaseCallbackHandler 互換)
    ignore_llm = False
    ignore_retry = True
    ignore_chain = True
    ignore_agent = False
    ignore_retriever = True
    ignore_chat_model = False
    raise_error = False
    run_inline = False

    # ツール名 → bubble_messages.json のキー
    TOOL_MESSAGE_KEY: dict[str, str] = {
        "retrieve_memory_tool": "searching",
        "web_search_tool": "web_search",
        "ask_character_tool": "ask_character",
    }

    def __init__(
        self,
        character_slug: str,
        common: dict,
        status_manager: "CharacterStatusManager | None" = None,
    ):
        self._character_slug = character_slug
        self._common = common
        # Phase 0.5-B-α: ツール呼び出し時に Thinking → ToolCalling、終了時に
        # ToolCalling → Thinking を反映する。None 時は status 反映スキップ
        # (= 既存テスト互換、status_manager 未注入時の挙動維持)。
        self._status_manager = status_manager

    # LangChain が呼び出すがこの handler では不要なコールバック (warning 抑制用)
    def on_chain_start(self, *args, **kwargs) -> None: pass  # noqa: E704
    def on_chain_end(self, *args, **kwargs) -> None: pass  # noqa: E704
    def on_chat_model_start(self, *args, **kwargs) -> None: pass  # noqa: E704
    def on_llm_end(self, *args, **kwargs) -> None: pass  # noqa: E704
    def on_llm_start(self, *args, **kwargs) -> None: pass  # noqa: E704

    def on_tool_end(self, *args, **kwargs) -> None:
        """ツール呼び出し終了時の no-op (LangChain CallbackHandler 互換用)。

        【経緯】
        Phase 0.5-B-α commit 5 では「TOOL_CALLING → THINKING」を反映していたが、
        LangChain Agent が web_search 完了後に LLM 応答生成で hang する症状を観察
        (logs/runs/run_loop_20260508_{222342, 224907, 225127}.log)。
        実走 TAKE 1 (web_search 15936 chars) と TAKE 2 (web_search 2217 chars) の
        両方でフィラー無限ループが発生し、Phase 0.5-A 末尾 (= 同コードでこの実装が
        なかった) では同じ web_search サイズで正常応答していたことから、本実装の
        微妙なタイミング差が Agent の internal state machine に影響していると推定。

        【判断】
        commit 5 partial revert として on_tool_end を no-op に戻す
        (= Phase 0.5-A の挙動と同等)。他の状態反映経路 (= on_tool_start で TOOL_CALLING、
        _generation_node 入口で THINKING) は維持。

        【HUD dashboard への影響】
        「TOOL_CALLING のまま → 第 1 chunk 投入で TALKING に直接遷移」となり、
        「Tool 後の Thinking」の中間状態は視覚化されない。ただし TOOL_CALLING 自体が
        「考え中 / 検索中」相当の表示で十分機能するため実用上は問題なし。
        """
        pass

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """ツール呼び出し開始時にbubble.updateを発行する。

        Phase 0.5-B-α: status_manager に ToolCalling 状態を反映する
        (= HUD で「ツール呼出中」表示の根拠データ)。
        """
        from .pipeline import _load_bubble_messages
        from .events import build_bubble_update
        from .bus import publish

        tool_name = serialized.get("name", "")
        msg_key = self.TOOL_MESSAGE_KEY.get(tool_name)
        if not msg_key:
            return  # 未知のツールは無視 (fail-open)

        # Phase 0.5-B-α: Thinking → ToolCalling 反映 (TOOL_MESSAGE_KEY に登録された
        # 既知ツールのみ。未知ツールは無視 = bubble.update も skip)。
        if self._status_manager is not None:
            self._status_manager.set_status(
                self._character_slug, CharacterStatus.TOOL_CALLING,
            )

        char_msgs = _load_bubble_messages().get(self._character_slug, {})
        text = char_msgs.get(msg_key, "検索中…")

        try:
            event = build_bubble_update(
                character=self._character_slug,
                step="searching",
                text=text,
                category="speech",  # Phase 0.5-A 8-10: 通常応答パス (Agent ツール) は speech
                **self._common,
            )
            publish(event)
            logger.info("bubble.update(searching): tool=%s character=%s", tool_name, self._character_slug)
        except Exception as exc:
            logger.warning("bubble.update(searching) 失敗: %s", exc)


# ─── 公開 API ────────────────────────────────────────────────────

def run_graph(
    text: str,
    *,
    model: str,
    provider: str = "openai",
    context: str | None = None,
    system_prompt: str | None = None,
    run_metadata: dict | None = None,
    character_slug: str | None = None,
    common: dict | None = None,
    disable_tools: "list[str] | None" = None,
    status_manager: "CharacterStatusManager | None" = None,
) -> LLMResult:
    """
    utterance text を受け取り、LLMResult を返す。

    L2_ENABLE_TOOLS=true の場合は ReAct Agent を使用。
    それ以外は従来の単一ノード構成。

    Args:
        text:            発話テキスト
        model:           使用するモデル名
        provider:        LLM プロバイダ（"openai" / "google" / "anthropic"）
        context:         RAG で取得した参照テキスト（Agent モードでは未使用、レガシー互換用）
        system_prompt:   キャラクター別システムプロンプト（省略時は従来動作）
        run_metadata:    LangGraph config["metadata"] に渡す dict (optional)
        character_slug:  キャラクター slug (Agent モード時の bubble.update 用)
        common:          stream_id/session_id/trace_id dict (Agent モード時の bubble.update 用)
        disable_tools:   特定ツールを除外するリスト (= 案 W'-3 バグ 3 修正、案 A)。
                         例: ["ask_character"] で ask_character ツールを Agent に
                         登録しない。fallback パス専用 (= deadlock 回避)。

    Returns:
        LLMResult

    Raises:
        ImportError:  langgraph が未インストール
        RuntimeError: Graph が result を返さなかった場合
    """
    # ログ強化 L-2: キャラ識別子を先頭に出して、どのターンの Graph 実行かパッと分かるように
    logger.info(
        "Graph 実行開始 [character=%s]: model=%s provider=%s tools=%s",
        character_slug or "?", model, provider, _is_tools_enabled(),
    )

    if _is_tools_enabled():
        agent = _build_agent_graph(
            provider, model, system_prompt,
            character_slug=character_slug,
            disable_tools=disable_tools,
        )
        if agent is not None:
            return _run_agent(
                agent, text, model,
                run_metadata=run_metadata,
                character_slug=character_slug,
                common=common,
                system_prompt=system_prompt,
                status_manager=status_manager,
            )

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
    # ログ強化 L-2: 単一ノードフォールバック経路でもキャラ識別子を出す
    logger.info("Graph 実行完了 [character=%s]", character_slug or "?")
    return result


def _run_agent(
    agent,
    text: str,
    model: str,
    *,
    run_metadata: dict | None = None,
    character_slug: str | None = None,
    common: dict | None = None,
    system_prompt: str | None = None,
    status_manager: "CharacterStatusManager | None" = None,
) -> LLMResult:
    """ReAct Agent を実行し、LLMResult に変換する。

    Sprint Axis D Block 3:
    - TD-2 context wrapper 削除 (Agent が自分で retrieve_memory ツールを呼ぶため)
    - BubbleToolCallbackHandler でツール呼び出し時に bubble.update を発行
    """
    import time

    t0 = time.monotonic()

    # Agent への入力メッセージ (ツール判断は Agent + system_prompt のガイダンスに委ねる)
    input_data = {"messages": [{"role": "user", "content": text}]}

    # config 構築
    config: dict = {}
    if run_metadata:
        config["metadata"] = run_metadata
    # bubble.update コールバック (Agent がツールを呼んだとき HUD に動的表示)
    # Phase 0.5-B-α: status_manager を Handler に渡し、tool 起動/終了で
    # Thinking ↔ ToolCalling を反映できるようにする。
    if character_slug and common:
        handler = BubbleToolCallbackHandler(
            character_slug, common, status_manager=status_manager,
        )
        config["callbacks"] = [handler]

    try:
        result = agent.invoke(input_data, config or None)
        latency_ms = int((time.monotonic() - t0) * 1000)

        response_text = ""
        usage = {}
        source = "messages"

        # Phase 0.5-A フェーズ 8: 新 API (langchain.agents.create_agent) で
        # response_format=Pydantic を有効化した場合、最終応答が
        # state["structured_response"] に Pydantic インスタンスとして格納される。
        # これを優先的に取得し、JSON 文字列化して LLMResult.text に格納する。
        # 後段の _tts_node / _parse_voicepeak_json は llm_text を JSON として読むため、
        # 既存パイプラインは無変更で互換性を維持できる。
        structured = result.get("structured_response")
        if structured is not None:
            try:
                # Pydantic v2: model_dump_json で ASCII 非エスケープ JSON
                if hasattr(structured, "model_dump_json"):
                    response_text = structured.model_dump_json()
                elif hasattr(structured, "model_dump"):
                    import json as _json
                    response_text = _json.dumps(
                        structured.model_dump(), ensure_ascii=False,
                    )
                elif isinstance(structured, dict):
                    import json as _json
                    response_text = _json.dumps(structured, ensure_ascii=False)
                else:
                    # dump できない (Pydantic でも dict でもない) → messages フォールバック
                    raise TypeError(
                        "structured_response が dump 不可能 (型 ="
                        f" {type(structured).__name__})"
                    )
                source = "structured_response"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "structured_response の JSON 変換失敗 → messages フォールバック: %s",
                    exc,
                )
                structured = None  # フォールバックに進む

        # フォールバック: messages の最後の AI メッセージから応答テキストを抽出
        # (response_format 未指定 / 構造化出力 fail-open / 旧 API 互換のため)
        if structured is None:
            final_messages = result.get("messages", [])
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

        # usage tokens は messages 内の最後の AIMessage から取得 (structured_response
        # 取得時でも同じ messages 配列に含まれている)
        if not usage:
            for msg in reversed(result.get("messages", [])):
                if getattr(msg, "type", None) in ("ai", "assistant") or \
                        getattr(msg, "role", "") in ("ai", "assistant"):
                    usage = getattr(msg, "usage_metadata", None) or {}
                    break

        # ログ強化 L-2: キャラ識別子を先頭に出して、ターン毎にどのキャラの応答か判別容易に。
        # text は全文出していたが、ログが長くなりがちなので冒頭のみ + 文字数で代替表示
        # (= 失敗時の手掛かりとしては十分、payload の text が真の発信内容)。
        text_preview = response_text[:60].replace("\n", " ")
        text_suffix = "..." if len(response_text) > 60 else ""
        logger.info(
            "Agent 実行完了 [character=%s]: latency_ms=%d source=%s text_len=%d text=%r%s",
            character_slug or "?", latency_ms, source, len(response_text),
            text_preview, text_suffix,
        )
        return LLMResult(
            text=response_text,
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            finish_reason="stop",
        )
    except Exception as exc:
        logger.error(
            "Agent 実行失敗 [character=%s]: %s。単一ノードにフォールバック。",
            character_slug or "?", exc,
        )
        # フォールバック: 従来の単一ノード構成 (system_prompt を引き継ぐ)
        # ログ強化 L-3: caller_slug でログにキャラ識別を残す
        return call_llm(
            text,
            model=model,
            provider="openai",
            system_prompt=system_prompt,
            caller_slug=character_slug,
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
    # 配信単位の文脈 (Markdown 本文)。run_loop / run_once 起動時に
    # stream_context.load_stream_context() で読み込まれた値が入る。
    # routing ノードで character の system_prompt にマージされ、
    # generation 以降は state["system_prompt"] 側で配信文脈を含んだ
    # 拡張プロンプトを参照する (この stream_context フィールド自体は
    # 配信中に再利用しない記録目的)。None なら配信文脈なしで動作。
    stream_context: str | None
    on_tts_chunk_ready: Any
    on_pose_ready: Any
    # Phase 0.5-A フェーズ 7: 挙手 BG LLM モードで True にすると、
    # _generation_node 内の bubble.update("answering") 発行を抑制する。
    # 通常応答は run_pipeline 経由で生成するときに graph.py 内で answering bubble
    # を発行するが、挙手の BG 先行生成では「承認時に run_loop が TTS 開始時刻と
    # 同期して bubble.update("answering") を発行する」設計のため、graph 側では
    # 抑制する必要がある (= 二重発行防止)。デフォルト False で既存挙動を維持。
    suppress_bubble_answering: bool
    # Phase 0.5-A 案 W'-3 + バグ 3 修正 (案 A): 特定の Agent ツールを除外する。
    # 例: ["ask_character"] で fallback パス時に ask_character ツールを Agent から
    # 除外し、並行する TTS 再生との deadlock を回避する (logs/runs/run_loop_20260508_181051.log
    # で観察されたハングの対処)。デフォルト None で全ツール有効 = 既存挙動。
    disable_tools: "list[str] | None"
    # Phase 0.5-B-α: 全キャラ状態を一元管理する CharacterStatusManager。
    # _generation_node が「LLM 推論中 = Thinking」、BubbleToolCallbackHandler が
    # 「ツール実行中 = ToolCalling」を反映するため、pipeline → graph で透過渡し。
    # None 時は status 反映スキップ (= 後方互換、テストで未注入時の挙動維持)。
    status_manager: "CharacterStatusManager | None"

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


# 配信文脈をシステムプロンプトに重ねるときの見出し。
# 既存の "## 参照情報" (RAG context、llm.py で結合) と対称に配置することで、
# キャラ素体 → 本日の配信 → 参照情報 という3層の同心円構造を作る。
# 順序の意味: 普遍 (キャラ人格) → 当日の前提 → 今ターンの動的情報。
_STREAM_CONTEXT_HEADING = "## 本日の配信"


def _compose_system_prompt(
    character_prompt: str | None,
    stream_context: str | None,
) -> str | None:
    """
    キャラクター素体プロンプトに配信文脈を重ねた拡張 system_prompt を返す。

    Args:
        character_prompt: load_system_prompt() の戻り値 (キャラ素体)。
        stream_context:   stream_context.load_stream_context() の戻り値。
                          None / 空文字列なら結合せずキャラ素体をそのまま返す。

    Returns:
        結合済みプロンプト。両方 None なら None。

    結合フォーマット:
        <キャラ素体>

        ---

        ## 本日の配信

        <stream_context>

    【WHY: 順序】
        キャラ素体 (上位・不変) → 配信文脈 (中位・配信単位) という
        同心円構造を作る。LLM の attention 順序効果を踏まえると、
        安定した情報を上に置くと一貫した応答になりやすい。
        後段 llm.py で "## 参照情報" (RAG, ターン単位) が末尾に追加されるため、
        最終的に「不変 → 配信単位 → ターン単位」の3層になる。
    """
    if not stream_context:
        return character_prompt
    if not character_prompt:
        # キャラ素体が無い (FileNotFoundError 等) ケース。
        # 配信文脈だけでも LLM の前提に効かせるため、見出し付きで返す。
        return f"{_STREAM_CONTEXT_HEADING}\n\n{stream_context}"
    return (
        f"{character_prompt}"
        f"\n\n---\n\n"
        f"{_STREAM_CONTEXT_HEADING}\n\n{stream_context}"
    )


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
        character_prompt = load_system_prompt(character)
    except FileNotFoundError:
        logger.warning(
            "システムプロンプトが見つかりません: %s。プロンプトなしで続行。",
            character.system_prompt_file,
        )
        character_prompt = None

    # 配信文脈 (run_loop 起動時にロードされ state に乗っている) を重ねる。
    # 配信文脈なし or キャラ素体読み込み失敗時も _compose_system_prompt が
    # 適切に処理する (後方互換)。
    system_prompt = _compose_system_prompt(
        character_prompt, state.get("stream_context")
    )

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

    # bubble: thinking (Sprint Axis D Block 3: "searching" は Agent ツール呼び出し時に動的発行)
    _publish_bubble("thinking", character.slug, common, links=[utt["event_id"]])

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
    """生成ノード: LLM 呼び出し + llm.final 発行。

    Sprint Axis D Block 3:
    - retrieve_memory ツール用の contextvars をセットしてから Agent 実行
    - Agent がツールを自律判断 (retrieve_memory / web_search / なし)
    - rag_used / answer_mode は Agent 実行後に事後判定
    """
    from .pipeline import publish, _publish_bubble
    from .events import build_llm_final
    from .observability import build_run_metadata
    from .debug import write_llm_prompt, write_llm_response

    text = state["text"]
    common = state["common"]
    utt_event_id = state["events"][0]["event_id"]

    if state["use_real_llm"]:
        # Phase 0.5-B-α: LLM 推論開始 → Thinking 反映 (HUD 用)。
        # WHY: graph 内で反映することで run_pipeline (通常応答) と
        # run_pipeline_llm_only (BG LLM) 両経路を 1 箇所でカバー。run_loop 側に
        # 分散させると経路ごとに反映漏れが発生しやすい。state.get で None
        # フォールバック (status_manager 未注入時は no-op、後方互換)。
        _gen_status_manager = state.get("status_manager")
        if _gen_status_manager is not None and state.get("character_slug"):
            _gen_status_manager.set_status(
                state["character_slug"], CharacterStatus.THINKING,
            )

        # Sprint Axis D Block 3: retrieve_memory ツール用のセッションコンテキストをセット
        if _is_rag_enabled():
            from .mcp_servers.retrieve_memory import set_retrieval_context
            set_retrieval_context(
                stream_id=common.get("stream_id"),
                exclude_event_ids=[utt_event_id],
            )

        # Phase 3: ask_character ツール用のコンテキストをセット
        # on_pose_ready は target 応答の pose を target キャラの chunk 1 再生
        # 直前に切り替えるためのコールバック。caller の最終応答 pose 切替と
        # 同じ仕組み (= playback worker の _pending_pose 経由) を target にも
        # 適用する。
        from .mcp_servers.ask_character import set_ask_character_context
        set_ask_character_context(
            on_tts_chunk=state["on_tts_chunk_ready"],
            tts_output_dir=state["tts_output_dir"],
            common=common,
            caller_slug=state["character_slug"],
            on_pose_ready=state.get("on_pose_ready"),
        )

        _run_meta = build_run_metadata(
            stream_id=common["stream_id"],
            session_id=common["session_id"],
            trace_id=common["trace_id"],
        )
        # user message に「ルカからの発言」prefix を付ける。これがないと caller LLM が
        # user message の発話者を「アビスメイト」(視聴者) と推測してしまうケースがある
        # (特に system_prompt でアビスメイト言及が多い sakura で顕著)。明示的に
        # 「ルカからの発言」と渡すことで、各キャラの system_prompt にある
        # 「ルカを『ルカさん』と呼ぶ」等の指示が確実に効くようになる。
        # 例外: text が既に「ルカ」で始まる、もしくは明示的に「アビスメイトから」等と
        # ある場合はそのまま (= ユーザー側で発話者を明示している)。
        if text and not text.startswith("[") and "アビスメイト" not in text[:30]:
            tagged_text = f"[ルカからの発言]\n{text}"
        else:
            tagged_text = text
        write_llm_prompt(tagged_text, None)
        _llm_result = run_graph(
            tagged_text,
            model=state["llm_model"],
            provider=state["llm_provider"],
            system_prompt=state["system_prompt"],
            run_metadata=_run_meta,
            character_slug=state["character_slug"],
            common=common,
            # バグ 3 修正 (案 A): fallback パスでは disable_tools=["ask_character"] が
            # 指定される。state.get で None フォールバック (= 既存挙動互換)。
            disable_tools=state.get("disable_tools"),
            # Phase 0.5-B-α: status_manager を Agent / BubbleToolCallbackHandler に
            # 透過渡し。tool 起動時の Thinking ↔ ToolCalling 反映に使う。
            status_manager=_gen_status_manager,
        )
        write_llm_response(_llm_result.text)
        llm_text = _llm_result.text
        llm_meta: dict[str, Any] = dict(
            model=_llm_result.model,
            input_tokens=_llm_result.input_tokens,
            output_tokens=_llm_result.output_tokens,
            latency_ms=_llm_result.latency_ms,
            finish_reason=_llm_result.finish_reason,
        )
    else:
        llm_text = f"ダミー応答: {text}"
        llm_meta = {}

    # ログ強化 L-2: payload に character を含めて bus ログ + V2 受信側で識別容易に
    llm = build_llm_final(
        text=llm_text, seq=1, links=[utt_event_id],
        character=state["character_slug"],
        **llm_meta, **common,
    )
    publish(llm)

    # Phase 0.5-A フェーズ 7: BG LLM モード (挙手の先行生成) では、承認時に run_loop
    # が TTS 開始時刻と同期して bubble.update("answering") を発行する。graph 側で
    # 発行すると二重発行になるので、suppress_bubble_answering=True で抑制する。
    if not state.get("suppress_bubble_answering", False):
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
    # on_pose_ready コールバックがあれば遅延適用 (本命応答の再生開始タイミングで切替)
    # なければ従来通り即時切替 (run_once.py 等の互換性)
    _, _, _, pose_value = _parse_voicepeak_json(llm_text)
    if state.get("on_pose_ready"):
        state["on_pose_ready"](character_slug, pose_value or "neutral")
    else:
        set_pose(character_slug, pose_value or "neutral")

    if state["use_real_tts"]:
        # caller の最終応答 TTS を投入する前に、ask_character がバックグラウンド
        # 合成中の協働応答 TTS の完了を待つ。これがないと VOICEPEAK FIFO ワーカー
        # に caller TTS と target TTS が同時に投入されて交互合成・再生になる
        # (= 「ちさめ chunk1 → mimi → ちさめ chunk2」のような順序乱れ)。
        # caller LLM 推論は target TTS 合成と並行で進んでいるため、ここで待つ
        # 時間は実用上ゼロ〜数秒に収まることが多い。
        from .mcp_servers.ask_character import wait_bg_tts_complete
        wait_bg_tts_complete(common.get("session_id", ""))

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
            chunk_audio_urls=_tts_result.chunk_audio_urls or None,
            duration_ms=_tts_result.duration_ms,
            voice=_tts_result.voice,
            format=_tts_result.format,
            sample_rate=_tts_result.sample_rate,
            speaker=_tts_result.speaker,
        )
    else:
        tts_meta = dict(speaker=state["tts_speaker"])

    # ログ強化 L-2: payload に character を含めて bus ログ + V2 受信側で識別容易に。
    # speaker (= voicepeak narrator) は既存フィールド、character (= aibyss slug) は新規。
    tts = build_tts_done(
        text=llm_text, seq=2, links=[llm_event_id],
        character=character_slug,
        **tts_meta, **common,
    )
    publish(tts)

    # bubble: done は run_loop.py の _playback_worker が最終チャンク再生 + 5 秒後に発行する
    # (Sprint Axis D Block 1: OBS セリフテロップ表示)

    return {
        "tts_meta": tts_meta,
        "events": [tts],
    }


# ─── パイプライングラフ構築 ──────────────────────────────────────


def _build_pipeline_graph():
    """3 ノードパイプライングラフを構築する。

    Sprint Axis D Block 3: _retrieval_node を削除。
    RAG 検索は _generation_node 内の Agent がツールとして自律呼び出しする。

    langgraph が未インストールの場合は ImportError を送出する。
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(PipelineGraphState)
    builder.add_node("routing", _routing_node)
    builder.add_node("generation", _generation_node)
    builder.add_node("tts", _tts_node)
    builder.set_entry_point("routing")
    builder.add_edge("routing", "generation")
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


# ─── Phase 0.5-A 案 W'-1: LLM-only / TTS-only グラフ ──────────────


def _build_pipeline_graph_llm_only():
    """LLM-only パイプライングラフを構築する (Phase 0.5-A 案 W'-1)。

    routing → generation → END で TTS ノードを除外する。挙手 BG 先行生成で
    「承認確率に賭けて LLM だけ走らせ、TTS は承認後に同期実行」する分離戦略の
    実装基盤。

    【WHY: TTS を分けたい理由】
    案 W' 適用前 (snappy-bentley): bg_runner は run_pipeline 全体 (LLM + 内蔵 TTS)
    を走らせていた。承認が来た瞬間に bg_chunks は完成済で playback worker に流す
    だけで済むのでレイテンシは最小だが、副作用として、却下/lapse 時に TTS で生成
    済の wav が破棄され、VOICEPEAK FIFO の貴重なリソース (= シーケンシャル 1 並列)
    を消費した後に「使わない」結果になる。さらに 3 重発火 (BG (1) + fallback (2) +
    wake (3)) が同時並走すると VOICEPEAK FIFO に 3 ジョブが直列に積まれ、合計
    30 秒以上の遅延が実走で観察された (logs/runs/run_loop_20260508_154959.log)。
    LLM-only に分けることで、却下/lapse 時に「LLM トークンは捨てるが TTS 合成は
    走らせない」状態を作り、VOICEPEAK 競合を緩和する。

    【WHY: グラフ分離 vs フラグ分岐】
    _tts_node に「LLM only モード」フラグを足す案 (= state["skip_tts"]) も検討
    したが、graph 構造を保ったまま条件分岐すると langgraph の compile 結果が
    複雑化し、既存テストの構造仮定が壊れる可能性がある。グラフ自体を別ビルド
    する方が「LLM only と LLM+TTS は別グラフ」と明示的に表現でき、_tts_node の
    ロジック変更を伴わない (= 既存 TTS 経路の回帰リスクを下げる)。

    Raises:
        ImportError: langgraph が未インストール
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(PipelineGraphState)
    builder.add_node("routing", _routing_node)
    builder.add_node("generation", _generation_node)
    builder.set_entry_point("routing")
    builder.add_edge("routing", "generation")
    builder.add_edge("generation", END)
    return builder.compile()


def run_pipeline_graph_llm_only(initial_state: PipelineGraphState) -> PipelineGraphState:
    """LLM-only パイプライングラフを実行し、最終状態を返す (Phase 0.5-A 案 W'-1)。

    final_state["events"] には utterance.final + llm.final の 2 件のみ含まれる
    (tts.done は含まれない)。bubble は routing で thinking、suppress_bubble_answering
    で answering 抑制 (= run_loop が承認時に発行する設計)。

    Raises:
        ImportError: langgraph が未インストール
    """
    graph = _build_pipeline_graph_llm_only()
    return graph.invoke(initial_state)


def _build_pipeline_graph_tts_only():
    """TTS-only パイプライングラフを構築する (Phase 0.5-A 案 W'-1)。

    LLM 結果を initial_state に注入して TTS ノードのみ実行する。挙手承認時に
    BG LLM で先行生成した結果を再利用して TTS を走らせる経路で使う。

    【WHY: tts.synthesize 直接呼出ではなく graph 再利用】
    _tts_node 内には pose 切替 (_parse_voicepeak_json + set_pose) /
    ask_character 協働 TTS 完了待ち (wait_bg_tts_complete) /
    build_tts_done event publish などの周辺ロジックが詰まっている。
    tts.synthesize 直接呼出ではこれらを再実装する必要があり、コード重複と
    保守負荷が増える。グラフ再利用なら新規実装ゼロ、テストも graph レベルで整合。

    Raises:
        ImportError: langgraph が未インストール
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(PipelineGraphState)
    builder.add_node("tts", _tts_node)
    builder.set_entry_point("tts")
    builder.add_edge("tts", END)
    return builder.compile()


def run_pipeline_graph_tts_only(initial_state: PipelineGraphState) -> PipelineGraphState:
    """TTS-only パイプライングラフを実行し、最終状態を返す (Phase 0.5-A 案 W'-1)。

    initial_state には ``llm_text`` + ``character_slug`` + ``events``
    (utterance.final + llm.final) を事前に詰めておく必要がある。詳細は
    ``pipeline.run_pipeline_tts_only`` を参照。

    final_state["events"] は initial_state["events"] + tts.done の追加分。

    Raises:
        ImportError: langgraph が未インストール
    """
    graph = _build_pipeline_graph_tts_only()
    return graph.invoke(initial_state)
