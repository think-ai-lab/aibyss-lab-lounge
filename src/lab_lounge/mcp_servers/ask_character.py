"""
ask_character.py — MCP AITuber 協働ツール

他の AITuber キャラクターに質問し、応答を TTS 合成して再生キューに即座に投入する。
「話しながら考える」設計: 協働先の音声はメイン Agent の最終応答を待たずに再生される。

Phase 3: AITuber 掛け合い

【再帰防止】
  協働先は ReAct Agent として実行されるが、ツールセットから ask_character のみ除外。
  retrieve_memory / web_search は利用可能（品質維持）。
  ask_character → ask_character の再帰は構造的に不可能。

【contextvars】
  on_tts_chunk コールバック等は _generation_node から contextvars で注入される。
  これにより ask_character 内の TTS チャンクが run_loop の再生キューに直接投入され、
  フィラー停止・立ち絵切替・bubble.update が既存メカニズムでそのまま動く。

【使い方】
  graph.py の _load_mcp_tools() が自動的にツール登録する。
"""

import contextvars
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─── セッションコンテキスト (contextvars) ────────────────────────────
# _generation_node がターンごとにセットし、ツール関数が読む。

_on_tts_chunk_var: contextvars.ContextVar[Callable | None] = contextvars.ContextVar(
    "ask_char_on_tts_chunk", default=None,
)
_tts_output_dir_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ask_char_tts_output_dir", default="./data/audio",
)
_common_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "ask_char_common", default={},
)
_caller_slug_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ask_char_caller_slug", default="",
)
# Phase 3: 導入セリフ再生完了同期用。on_tts_chunk → playback_worker が set() する。
_chunk_done_event_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ask_char_chunk_done_event", default=None,
)


def set_ask_character_context(
    *,
    on_tts_chunk: Callable | None = None,
    tts_output_dir: str = "./data/audio",
    common: dict | None = None,
    caller_slug: str = "",
) -> None:
    """Agent 実行前にコンテキストをセットする。graph.py の _generation_node から呼ばれる。"""
    _on_tts_chunk_var.set(on_tts_chunk)
    _tts_output_dir_var.set(tts_output_dir)
    _common_var.set(common or {})
    _caller_slug_var.set(caller_slug)


def reset_ask_character_context() -> None:
    """テスト用: contextvars をデフォルトにリセットする。"""
    _on_tts_chunk_var.set(None)
    _tts_output_dir_var.set("./data/audio")
    _common_var.set({})
    _caller_slug_var.set("")
    _chunk_done_event_var.set(None)


# ─── MCP サーバー ──────────────────────────────────────────────────

_mcp = None


def _get_mcp():
    """FastMCP インスタンスを遅延初期化する。"""
    global _mcp
    if _mcp is None:
        from fastmcp import FastMCP
        _mcp = FastMCP(
            "aibyss-ask-character",
            instructions=(
                "他の AITuber キャラクターに質問するツール。"
                "自分の専門外の質問や、別の視点が欲しい場合に使用してください。"
            ),
        )
        _register_tools(_mcp)
    return _mcp


def _register_tools(mcp):
    """ツールを MCP サーバーに登録する。"""

    @mcp.tool()
    def ask_character(character_slug: str, question: str) -> str:
        """
        他の AITuber キャラクターに質問する。

        自分の専門外の質問や、別の視点が欲しい場合に使用する。
        character_slug は相手のキャラクター識別子:
        - "mimi": ミミ・オクタヴィア（美学・価値観）
        - "chisame": 波心ちさめ（データ・論理）
        - "sakura": 八重笠さくら（感情・心理安全）
        - "ruka": 坂東ルカ（論点整理・全体調整）
        - "octamaid": オクタメイド（補助・進行）

        自分自身の slug は指定しないこと。1 応答で最大 2 回まで。
        """
        return _ask_character_impl(character_slug, question)


def _ask_character_impl(character_slug: str, question: str) -> str:
    """
    協働先キャラクターの Agent を実行し、TTS 合成 + 再生キュー投入を行い、
    応答テキストを返す。

    協働先は ReAct Agent として実行されるが、ツールセットから ask_character のみ除外
    (再帰防止)。retrieve_memory / web_search は利用可能。
    """
    # 遅延 import で循環参照を回避
    from ..characters import get_character, load_system_prompt
    from ..skill_loader import build_skills_prompt

    caller_slug = _caller_slug_var.get()

    # 自己呼出し防止
    if character_slug == caller_slug:
        logger.warning("自分自身 (%s) への ask_character は禁止です。", character_slug)
        return f"エラー: 自分自身 ({character_slug}) には質問できません。別のキャラクターを指定してください。"

    # 1. キャラクター設定の取得
    try:
        target_char = get_character(character_slug)
    except KeyError:
        logger.warning("未知のキャラクター slug: %s", character_slug)
        return f"エラー: キャラクター '{character_slug}' は存在しません。"

    # 呼び出し元キャラの display_name を取得 (質問コンテキスト用)
    try:
        caller_char = get_character(caller_slug) if caller_slug else None
        caller_display = caller_char.display_name if caller_char else "ルカ"
    except KeyError:
        caller_display = "ルカ"

    logger.info(
        "ask_character 実行: caller=%s target=%s question=%r",
        caller_slug, character_slug, question[:80],
    )

    # 質問に「誰からの質問か」コンテキストを追加
    # (協働先の system_prompt がルカ前提のため、実際の発話元を明示する)
    contextualized_question = (
        f"{caller_display}からの質問です。{caller_display}に対して回答してください。\n"
        f"質問: {question}"
    )

    # 2. モデル決定
    model_override = os.environ.get("L2_ASK_CHARACTER_MODEL_OVERRIDE", "")
    if model_override:
        model = model_override
        provider = os.environ.get("L2_LLM_PROVIDER", "openai")
    elif target_char.llm_model:
        model = target_char.llm_model
        provider = target_char.llm_provider
    else:
        provider = os.environ.get("L2_LLM_PROVIDER", "openai")
        model = os.environ.get("L2_LLM_MODEL", "gpt-5.4-mini")

    # 3. システムプロンプト + Skills 読み込み
    try:
        system_prompt = load_system_prompt(target_char)
    except FileNotFoundError:
        system_prompt = None

    skills_text = build_skills_prompt(character_slug)
    parts = [p for p in [system_prompt, skills_text] if p]
    combined_prompt = "\n\n".join(parts) if parts else None

    # 4. 導入セリフ TTS (再生完了待ち) + 協働先 Agent 実行
    import threading
    common = _common_var.get()
    on_tts_chunk = _on_tts_chunk_var.get()
    tts_output_dir = _tts_output_dir_var.get()
    use_real_tts = os.environ.get("L2_USE_REAL_TTS", "false").lower() in ("true", "1", "yes")

    # 4a. 導入セリフを LLM 生成 (emotion JSON 付き) → TTS → 再生完了を待つ
    #     + 並行してちさめ LLM を開始 (待ち時間を最小化)
    #
    #     導入セリフ = ルカへのクッション + ちさめへの問いかけ (キャラの口調で一体生成)
    #     例: 「面白い質問ですわね。ちさめ、AIエージェントの最新動向について教えてちょうだい」

    # 4a-1. 導入セリフ LLM 生成 (軽量モデル、emotion JSON 付き)
    intro_response_text = ""
    if caller_char:
        intro_response_text = _generate_intro(
            caller_char=caller_char,
            target_char=target_char,
            user_question=question,
        )

    # 4a-2. 協働先への質問コンテキストにルカの原文 + 導入セリフを含める
    #        (ちさめは会話の全体像を把握した上で応答できる)
    if intro_response_text:
        contextualized_question = (
            f"ルカからの元の質問: 「{question}」\n\n"
            f"これを受けて{caller_display}があなたにこう語りかけました:\n"
            f"「{intro_response_text}」\n\n"
            f"{caller_display}に対して回答してください。"
        )
    # else: 既存の contextualized_question をそのまま使用

    # 4a-3. 協働先 LLM を別スレッドで並行開始
    collab_result: list[str | None] = [None]
    collab_error: list[Exception | None] = [None]

    def _run_collab():
        try:
            collab_result[0] = _run_collaboration_agent(
                question=contextualized_question,
                model=model,
                provider=provider,
                combined_prompt=combined_prompt,
                character_slug=character_slug,
                common=common,
            )
        except Exception as exc:
            collab_error[0] = exc

    collab_thread = threading.Thread(target=_run_collab, daemon=True)
    collab_thread.start()
    logger.info("ask_character 協働先 LLM 並行開始: %s", character_slug)

    # 4a-4. 導入セリフ TTS → 再生完了を待つ (並行して協働先 LLM が走る)
    if on_tts_chunk and use_real_tts and caller_char and intro_response_text:
        try:
            from ..tts import synthesize as tts_synthesize

            intro_done = threading.Event()
            _chunk_done_event_var.set(intro_done)

            logger.info("ask_character 導入セリフ TTS: [%s] %s", caller_slug, intro_response_text[:60])
            tts_synthesize(
                intro_response_text,
                provider=caller_char.tts_provider,
                voice=caller_char.tts_voice,
                speaker=caller_char.slug,
                output_dir=tts_output_dir,
                on_chunk_ready=on_tts_chunk,
            )

            logger.info("ask_character 導入セリフ再生待ち...")
            intro_done.wait(timeout=120)
            logger.info("ask_character 導入セリフ再生完了")
        except Exception as exc:
            logger.warning("導入セリフ TTS 失敗: %s", exc)

    # 4a-5. 協働先 LLM の完了を待つ (導入再生中に並行実行されていたので大部分は完了済み)
    collab_thread.join(timeout=120)
    if collab_error[0]:
        logger.error("ask_character 協働先 Agent 失敗: %s", collab_error[0])
        response_text = f"エラー: {collab_error[0]}"
    else:
        response_text = collab_result[0] or ""

    # 5. 協働先の応答を TTS 合成 + 再生キュー投入
    if on_tts_chunk and use_real_tts:
        try:
            from ..tts import synthesize as tts_synthesize
            tts_synthesize(
                response_text,
                provider=target_char.tts_provider,
                voice=target_char.tts_voice,
                speaker=target_char.slug,
                output_dir=tts_output_dir,
                on_chunk_ready=on_tts_chunk,
            )
        except Exception as exc:
            logger.warning("ask_character TTS 合成失敗 (%s): %s", character_slug, exc)

    logger.info("ask_character 完了: target=%s response_len=%d", character_slug, len(response_text))

    # Agent への戻り値: 応答元と再生済みであることを明確に伝える
    caller_name = caller_char.display_name if caller_char else "あなた"
    return (
        f"【{target_char.display_name}からの応答】\n"
        f"{response_text}\n\n"
        f"【重要な指示】\n"
        f"- 上記は{target_char.display_name}が話した内容です（ルカからの応答ではありません）。\n"
        f"- この応答はすでに{target_char.display_name}の声で視聴者に直接再生されています。\n"
        f"- 要約や繰り返しは不要です。「聞いてまいりました」「こう言っていました」も不要です。\n"
        f"- {caller_name}として、{target_char.display_name}が話した内容を踏まえた上で、"
        f"あなた自身の視点で補足・感想・次の展開を述べてください。\n"
        f"- {target_char.display_name}がすでに話し終えた前提で、自然に会話を続けてください。"
    )


def _run_collaboration_agent(
    *,
    question: str,
    model: str,
    provider: str,
    combined_prompt: str | None,
    character_slug: str,
    common: dict,
) -> str:
    """
    協働先の ReAct Agent を構築・実行する。

    ask_character を除外したツールセット (retrieve_memory + web_search) を持つ。
    BubbleToolCallbackHandler も接続し、ツール呼び出し時の bubble 表示を維持。
    """
    from ..graph import (
        _load_mcp_tools,
        _get_llm_for_agent,
        _run_agent,
        BubbleToolCallbackHandler,
    )

    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        # LangGraph なし → 単純 LLM 呼出しにフォールバック
        from ..llm import call_llm
        result = call_llm(question, model=model, provider=provider, system_prompt=combined_prompt)
        return result.text

    # ツールセットから ask_character を除外 (再帰防止)
    all_tools = _load_mcp_tools()
    collab_tools = [t for t in all_tools if t.name != "ask_character_tool"]

    if not collab_tools:
        # ツールなし → 単純 LLM 呼出し
        from ..llm import call_llm
        result = call_llm(question, model=model, provider=provider, system_prompt=combined_prompt)
        return result.text

    # retrieve_memory 用の contextvars をセット (協働先も記憶検索できるように)
    from .retrieve_memory import set_retrieval_context
    set_retrieval_context(
        stream_id=common.get("stream_id"),
        exclude_event_ids=[],
    )

    llm = _get_llm_for_agent(provider, model)
    agent = create_react_agent(llm, collab_tools, prompt=combined_prompt)

    # Agent 実行 (BubbleToolCallbackHandler で bubble.update を発行)
    result = _run_agent(
        agent,
        question,
        model,
        character_slug=character_slug,
        common=common,
    )
    return result.text


def _generate_intro(
    caller_char,
    target_char,
    user_question: str,
) -> str:
    """
    導入セリフを軽量 LLM で動的生成する。

    呼び出し元キャラの口調で:
    1. ルカの質問へのコメント (クッション)
    2. 協働先キャラへの問いかけ (質問のキャラ口調での言い換え)
    を一体で生成する。

    emotion/speed/pose 付き JSON で返すため、TTS で感情が反映される。

    Returns:
        生成された JSON テキスト (emotion 付き)。失敗時は空文字。
    """
    from ..llm import call_llm
    from ..characters import load_system_prompt

    # フィラー用の軽量モデルを使用
    filler_model = getattr(caller_char, "filler_model", "") or "gpt-5.4-mini"
    filler_provider = caller_char.llm_provider or "openai"

    # 呼び出し元キャラの system_prompt を使用 (口調・キャラクター性を反映)
    try:
        caller_system_prompt = load_system_prompt(caller_char)
    except FileNotFoundError:
        caller_system_prompt = None

    intro_prompt = (
        f"ルカから「{user_question}」と聞かれました。"
        f"これに対して:\n"
        f"1. まずルカの質問を受け止めるコメントを一言\n"
        f"2. 続けて、{target_char.display_name}に直接語りかけて同じ内容を聞く\n"
        f"を、あなたの口調で自然に繋げて 1〜3 文で返答してください。\n"
        f"通常の応答と同じ JSON フォーマット (emotion/speed/pose/response) で返してください。"
    )

    try:
        result = call_llm(
            intro_prompt,
            model=filler_model,
            provider=filler_provider,
            system_prompt=caller_system_prompt,
        )
        intro = result.text.strip()
        if intro:
            logger.info("導入セリフ生成完了: [%s] %s", caller_char.slug, intro[:80])
            return intro
    except Exception as exc:
        logger.warning("導入セリフ LLM 生成失敗: %s", exc)

    # フォールバック: 固定テキスト
    from ..pipeline import _load_bubble_messages
    fallback = _load_bubble_messages().get(caller_char.slug, {}).get("ask_character", "")
    return fallback


# 直接呼び出し用のエイリアス
ask_character = _ask_character_impl


def get_server():
    """MCP サーバーインスタンスを返す（遅延初期化）。"""
    return _get_mcp()


if __name__ == "__main__":
    server = get_server()
    server.run()
