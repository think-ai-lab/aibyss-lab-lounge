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
import threading
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
# 立ち絵切替予約コールバック (run_loop.py の _on_pose_ready)。target 応答の本応答
# TTS 投入時に呼び出すことで、playback worker が target chunk 1 再生直前に
# target キャラの pose (例: special_doya / special_overdrive) で立ち絵切替する。
# graph.py の _generation_node で _tts_node の on_pose_ready と同じものをここでも
# セットする。
_on_pose_ready_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ask_char_on_pose_ready", default=None,
)
# Phase 0.5-B-β-1 commit 4: target キャラのステータスを HUD dashboard (V2 /status)
# に反映するための CharacterStatusManager 参照。set_ask_character_context で
# graph._generation_node から注入される。bridge filler 投入時に target を THINKING、
# 本応答 chunk 1 投入時に TALKING (metadata: pose + full response_text)、
# bg_tts 合成完了時に READY に反映する。
#
# 【WHY: caller でなく target だけ反映する】
# caller (= mimi が ask_character を起動する側) のステータスは通常応答経路の
# graph._generation_node (THINKING) / BubbleToolCallbackHandler (TOOL_CALLING) /
# graph._tts_node (TALKING) で既に反映される。target (= chisame の音声が ask_character
# 経由で playback queue に入って流れている数十秒) のステータスは Phase 0.5-B-α では
# 未配線 (= ask_character.py に set_status 呼出が 0 件) で、HUD カードが READY のまま
# だった。本 commit でこの穴を埋める。
_status_manager_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ask_char_status_manager", default=None,
)

# ─── ターン状態 (session_id をキーにした module-level dict) ──────────
# contextvars ではなく dict + Lock で管理する理由:
#   LangGraph の ToolNode は asyncio.create_task() などで子 context を生成して
#   ツール関数を実行する。contextvars.set() で書き換えた値は親 context に
#   伝搬しないため、複数回の ask_character_tool 呼出し間でカウントが共有
#   できなかった (S999 で ask_index が常に 1 にリセットされる事象)。
#
# session_id (= run_loop の 1 セッション = 配信 1 回 ≒ 1 ターン群) をキーに
# して、process グローバルな状態を Lock 付きで保持する。set_ask_character_context()
# で session 開始時にリセットする。
_ask_state_lock = threading.Lock()
_ask_counts: dict[str, int] = {}        # session_id → 同一ターン内の呼出し回数
_previous_targets: dict[str, str] = {}  # session_id → 直前 target の display_name
# 同一 session 内の協働応答 TTS バックグラウンドスレッドの完了 event リスト。
# graph.py の _tts_node (caller の最終応答 TTS 投入前) と run_loop.py の playback
# queue close 前で wait_bg_tts_complete() で待つ。これにより:
#   - caller の最終応答 TTS が target TTS と同時に VOICEPEAK ワーカーキューに
#     投入されて交互合成・再生になるのを防ぐ (順序保証)
#   - playback queue の close sentinel が target chunks 投入完了前に送られて
#     残りの chunks が再生されないのを防ぐ (途中切れ防止)
_bg_tts_events: dict[str, list[threading.Event]] = {}


def _reset_session_state(session_id: str) -> None:
    """セッション状態をリセットする (set_ask_character_context から呼ばれる)。"""
    if not session_id:
        return
    with _ask_state_lock:
        _ask_counts[session_id] = 0
        _previous_targets[session_id] = ""


def _next_ask_state(session_id: str) -> tuple[int, str]:
    """カウントをインクリメントし、(新しい ask_index, インクリメント前の直前 target) を返す。

    session_id が空の場合は (1, "") を返し、状態は保持しない。
    """
    if not session_id:
        return (1, "")
    with _ask_state_lock:
        idx = _ask_counts.get(session_id, 0) + 1
        _ask_counts[session_id] = idx
        prev = _previous_targets.get(session_id, "")
    return (idx, prev)


def _record_target(session_id: str, target_display: str) -> None:
    """ask_character 完了後に直前 target の display_name を記録する。"""
    if not session_id:
        return
    with _ask_state_lock:
        _previous_targets[session_id] = target_display


def _register_bg_tts_event(session_id: str, event: threading.Event) -> None:
    """協働応答 TTS バックグラウンドスレッドの完了 event を session に登録する。"""
    if not session_id:
        return
    with _ask_state_lock:
        _bg_tts_events.setdefault(session_id, []).append(event)


def wait_bg_tts_complete(session_id: str, timeout: float = 180.0) -> None:
    """指定 session の全 bg_tts (協働応答 TTS バックグラウンド合成) の完了を待つ。

    graph.py の _tts_node (caller の最終応答 TTS 投入前) と run_loop.py の
    playback queue close 前で呼び出される。完了を待ち終わった events は dict から
    取り除かれるため、複数回呼んでも問題ない (空リストになる)。

    Args:
        session_id: 待機対象の session_id (空文字なら no-op)
        timeout:    1 event あたりの最大待機秒数 (default: 180)
    """
    if not session_id:
        return
    with _ask_state_lock:
        events = _bg_tts_events.pop(session_id, [])
    if not events:
        return
    logger.info(
        "wait_bg_tts_complete: session=%s pending=%d events 待機開始",
        session_id, len(events),
    )
    for ev in events:
        ev.wait(timeout=timeout)
    logger.info("wait_bg_tts_complete: session=%s 全 events 完了", session_id)


def set_ask_character_context(
    *,
    on_tts_chunk: Callable | None = None,
    tts_output_dir: str = "./data/audio",
    common: dict | None = None,
    caller_slug: str = "",
    on_pose_ready: Callable | None = None,
    status_manager: Any = None,
) -> None:
    """Agent 実行前にコンテキストをセットする。graph.py の _generation_node から呼ばれる。

    Args:
        on_pose_ready: target 応答の pose 切替予約コールバック。
                       (slug: str, pose: str) -> None。
                       graph.py の _tts_node が caller の pose 切替で使うものと
                       同じ関数 (_on_pose_ready) を渡す想定。
        status_manager: Phase 0.5-B-β-1 commit 4 で追加。target キャラの HUD
                       ステータス反映に使う CharacterStatusManager。None なら
                       ステータス反映 no-op (= 後方互換、Phase 0.5-B-α 以前と同じ)。
    """
    _on_tts_chunk_var.set(on_tts_chunk)
    _tts_output_dir_var.set(tts_output_dir)
    _common_var.set(common or {})
    _caller_slug_var.set(caller_slug)
    _on_pose_ready_var.set(on_pose_ready)
    _status_manager_var.set(status_manager)
    # ターン開始時に呼出し回数と直前 target をリセット (各ターン独立にカウント)。
    # graph.py の _generation_node がターン開始時に 1 回呼ぶ前提。
    session_id = (common or {}).get("session_id", "")
    _reset_session_state(session_id)


def reset_ask_character_context() -> None:
    """テスト用: contextvars と session 状態をデフォルトにリセットする。"""
    _on_tts_chunk_var.set(None)
    _tts_output_dir_var.set("./data/audio")
    _common_var.set({})
    _caller_slug_var.set("")
    _chunk_done_event_var.set(None)
    _on_pose_ready_var.set(None)
    _status_manager_var.set(None)
    with _ask_state_lock:
        _ask_counts.clear()
        _previous_targets.clear()
        _bg_tts_events.clear()


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
        caller_char = None
        caller_display = "ルカ"

    # ターン内 ask_character 呼出し回数をカウントアップ + 直前の target を読み出す。
    # 2 回目以降は _generate_intro が「ルカへの受け止めコメント」を省くプロンプトに
    # 切り替えるため、ask_index と previous_target_display を後段に渡す。
    # session_id をキーにした module-level dict を使う (contextvars だと
    # LangGraph の ToolNode 子 context で値が伝搬しないため)。
    common_dict = _common_var.get() or {}
    session_id = common_dict.get("session_id", "")

    # 直前の ask_character の協働応答 TTS バックグラウンド合成完了を待つ。
    # これがないと、2 回目の ask_character が始まる際 (caller LLM が次の判断を
    # 出した直後) に「caller の問いかけ TTS」が VOICEPEAK FIFO ワーカーキューに
    # 直前 target の chunks と並行で投入され、再生順序が乱れる
    # (例: 「ちさめ chunk1 → mimi のさくらへの問いかけ → ちさめ chunk2,3 → さくら」)。
    # caller LLM 推論時間が直前 target の TTS 合成と並行で進むため、ここで待つ
    # 時間は実用上ゼロ〜数十秒に収まる。直前 target が再生中なら配信上の空白は
    # 発生しない (= 直前の発話が続いている間に wait する形になる)。
    wait_bg_tts_complete(session_id)

    ask_index, previous_target_display = _next_ask_state(session_id)

    logger.info(
        "ask_character 実行: caller=%s target=%s ask_index=%d previous=%s question=%r",
        caller_slug, character_slug, ask_index, previous_target_display or "(none)",
        question[:80],
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
            ask_index=ask_index,
            previous_target_display=previous_target_display,
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
        from ..pipeline import _publish_bubble

        # 5-a. target の "考え中" bubble を発行する (filler 再生中のテロップ用)。
        # graph.py の _generation_node が caller の thinking bubble を出すのと同じ仕組みで、
        # filler が target の声で再生されている間、V2 HUD には target の thinking テキスト
        # (例: chisame「分析しています」/ sakura「んー……考え中ですよぉ」) を表示する。
        if common:
            try:
                _publish_bubble("thinking", target_char.slug, common)
            except Exception as exc:
                logger.warning(
                    "協働先 bubble.update(thinking) 発行失敗 (%s): %s",
                    target_char.slug, exc,
                )

        # Phase 0.5-B-β-1 commit 4: target キャラの HUD ステータスを THINKING に反映。
        # 上の _publish_bubble("thinking") と対になる V2 SSE event (= /status の
        # CharacterStatusManager 経由) を発火する。bridge filler が再生されている
        # 間 HUD カードが「考え中」(黄色) で表示される。bg_tts 合成失敗時は
        # _bg_tts_synthesize の finally で READY に戻る (= ステータス stuck 防止)。
        _status_manager_for_target = _status_manager_var.get()
        if _status_manager_for_target is not None:
            try:
                from ..character_status import CharacterStatus
                _status_manager_for_target.set_status(
                    target_char.slug, CharacterStatus.THINKING,
                )
            except Exception as exc:
                logger.warning(
                    "ask_character target THINKING 反映失敗 (%s): %s",
                    target_char.slug, exc,
                )

        # 5-b. target の bridge filler を再生キュー投入する (応答 TTS の合成中の空白を埋める)。
        # 「caller の問いかけ完了 → 即 target の応答が始まる」と target の VOICEPEAK 合成
        # (1 チャンク目 ~22 秒) を待つ間に視聴者の耳が空白を感じる。bridge filler
        # (例: chisame の「ええと……」/ sakura の「えっとぉ……」) を 1 つ挟むことで、
        # 受け止めの一言を経て自然に応答へ繋がる。bridge は事前生成キャッシュから取るため、
        # LLM/TTS のレイテンシも上乗せしない。
        try:
            from ..filler import select_filler_path
            bridge_path, _ = select_filler_path(target_char.slug, "bridge")
            if bridge_path:
                # chunk_text を空文字にする理由:
                #   playback worker (run_loop.py:_run_playback_worker) は task["text"] を
                #   bubble.update step="speaking" の表示テキストにそのまま流す。
                #   filler 用の内部識別ラベル (e.g., "(target bridge filler)") をここに
                #   渡すと HUD にそのまま出てしまうため、空文字を渡し、playback worker
                #   側で「空文字なら speaking publish をスキップ」して直前の thinking
                #   テロップを維持させる。
                on_tts_chunk(
                    bridge_path.as_uri(),
                    "",
                    False,  # is_last=False — 本応答が続く
                    target_char.slug,
                )
                logger.info(
                    "ask_character target bridge filler 投入: [%s] %s",
                    target_char.slug, bridge_path.name,
                )
            else:
                logger.debug("target bridge filler なし: %s", target_char.slug)
        except Exception as exc:
            logger.warning("target bridge filler 投入失敗 (%s): %s", target_char.slug, exc)

        # 5-c. 協働応答 TTS 合成 + 再生キュー投入 (バックグラウンド実行)。
        #
        # tts_synthesize は VOICEPEAK の全チャンク合成完了まで同期ブロックする
        # (1 chunk あたり ~20 秒 × 数チャンク = 数十秒)。これを別スレッドに逃すことで、
        # _ask_character_impl は ToolNode 戻り値を即時 return → caller の Agent が
        # すぐ次の LLM 推論 (例: 次の ask_character の呼出し判断) に進める。
        #
        # 完了 event を session_id に紐付けて登録する。graph.py の _tts_node
        # (caller の最終応答 TTS 投入前) と run_loop.py の playback queue close
        # 前で wait_bg_tts_complete() を呼び順序を保証する。これが無いと:
        #   - caller の最終応答 TTS と target TTS が VOICEPEAK FIFO に同時投入されて
        #     交互合成・再生になる (= 「ちさめ chunk1 → mimi → ちさめ chunk2」)
        #   - playback queue の close sentinel が target chunks 投入前に送られて
        #     target が途中で切れる
        #
        # answering bubble は本応答 TTS の最初のチャンクが再生キュー投入される
        # 直前に発火させる (on_chunk_ready ラッパー経由)。これにより:
        #   - bridge filler 再生中は thinking テロップが維持される
        #   - 本応答 TTS の最初のチャンクが鳴り始めるタイミングで answering テロップ
        #     に切替
        bg_tts_done = threading.Event()
        first_chunk_seen = [False]
        # contextvars はバックグラウンドスレッドの context isolation で値が
        # 引き継がれない可能性があるため、ここで値を取得してクロージャ経由で
        # _bg_tts_synthesize に渡す。
        on_pose_ready = _on_pose_ready_var.get()
        # Phase 0.5-B-β-1 commit 4: status_manager も同じく contextvars 引き継ぎ
        # 問題に対処するためメインスレッドで取得し、クロージャ経由で
        # _wrapped_on_chunk_ready / _bg_tts_synthesize から参照する。
        status_manager_capture = _status_manager_var.get()

        # response_text から pose を事前に抽出 (_wrapped_on_chunk_ready で使用)。
        # 本応答 chunk 1 が投入される直前に on_pose_ready を呼ぶことで、
        # _pending_poses に予約するタイミングと _on_tts_chunk で pop されるタイミング
        # の順序が保証される (= bridge filler 投入時には予約がなく neutral、本応答
        # chunk 1 投入時に target_pose が予約されている状態を作る)。
        from ..tts import _parse_voicepeak_json
        _, _, _, target_pose = _parse_voicepeak_json(response_text)

        def _wrapped_on_chunk_ready(url: str, chunk_text: str, is_last: bool, character: str) -> None:
            # 本応答 TTS の最初のチャンクが投入される直前のフック
            if not first_chunk_seen[0]:
                first_chunk_seen[0] = True
                # answering bubble を発行
                if common:
                    try:
                        _publish_bubble("answering", target_char.slug, common)
                    except Exception as exc:
                        logger.warning(
                            "協働先 bubble.update(answering) 発行失敗 (%s): %s",
                            target_char.slug, exc,
                        )
                # target pose 予約 (= _pending_poses[target_char.slug] に格納)。
                # 直後の on_tts_chunk で _pending_poses.pop され、本応答 chunk 1 の
                # task["pose"] にセットされる。playback worker が再生直前に
                # set_pose で立ち絵切替。
                if on_pose_ready and target_pose:
                    try:
                        on_pose_ready(target_char.slug, target_pose)
                        logger.info(
                            "ask_character target pose 予約: %s → %s",
                            target_char.slug, target_pose,
                        )
                    except Exception as exc:
                        logger.warning(
                            "ask_character target pose 予約失敗 (%s): %s",
                            target_char.slug, exc,
                        )
                # Phase 0.5-B-β-1 commit 4: target の HUD ステータスを TALKING に
                # 反映。本応答 chunk 1 が playback queue に投入されるタイミングで
                # 反映 → V2 HUD カードが緑色 + 発話全文 (response_text) と pose
                # を metadata に表示。closure キャプチャ (status_manager_capture)
                # で contextvars の daemon thread 引継ぎ問題を回避済み。
                #
                # metadata の構造は graph._tts_node が通常応答経路で渡す形と統一
                # (= V2 SSE 受信側で同じ shape として扱える):
                #   - pose: target_pose (None なら null、HUD 側で fallback 描画)
                #   - text: response_text (= 協働先 LLM の full レスポンス)
                if status_manager_capture is not None:
                    try:
                        from ..character_status import CharacterStatus
                        talking_metadata: dict[str, Any] = {
                            "pose": target_pose if target_pose else None,
                            "text": response_text,
                        }
                        status_manager_capture.set_status(
                            target_char.slug,
                            CharacterStatus.TALKING,
                            metadata=talking_metadata,
                        )
                    except Exception as exc:
                        logger.warning(
                            "ask_character target TALKING 反映失敗 (%s): %s",
                            target_char.slug, exc,
                        )
            on_tts_chunk(url, chunk_text, is_last, character)

        def _bg_tts_synthesize() -> None:
            try:
                from ..tts import synthesize as tts_synthesize
                tts_synthesize(
                    response_text,
                    provider=target_char.tts_provider,
                    voice=target_char.tts_voice,
                    speaker=target_char.slug,
                    output_dir=tts_output_dir,
                    on_chunk_ready=_wrapped_on_chunk_ready,
                )
            except Exception as exc:
                logger.warning(
                    "ask_character 協働応答 TTS 合成失敗 (%s): %s",
                    character_slug, exc,
                )
            finally:
                # Phase 0.5-B-β-1 commit 4: target の HUD ステータスを READY に
                # 戻す。bg_tts 合成完了 = target の発話が終わった瞬間。例外時も
                # finally で確実に Ready にする (= HUD カードが talking のまま
                # stuck するのを防ぐ、UI 上の整合性を保つ)。
                if status_manager_capture is not None:
                    try:
                        from ..character_status import CharacterStatus
                        status_manager_capture.set_status(
                            target_char.slug, CharacterStatus.READY,
                        )
                    except Exception as exc:
                        logger.warning(
                            "ask_character target READY 反映失敗 (%s): %s",
                            target_char.slug, exc,
                        )
                bg_tts_done.set()

        _register_bg_tts_event(session_id, bg_tts_done)
        threading.Thread(target=_bg_tts_synthesize, daemon=True).start()
        logger.info(
            "ask_character 協働応答 TTS 合成をバックグラウンド開始: target=%s session=%s",
            character_slug, session_id or "(none)",
        )

    logger.info("ask_character 完了: target=%s response_len=%d", character_slug, len(response_text))

    # 直前 target を更新 (次回 ask_character 呼出し時の導入セリフ生成で参照される)
    _record_target(session_id, target_char.display_name)

    # Agent への戻り値: 応答元と再生済みであることを明確に伝える
    caller_name = caller_char.display_name if caller_char else "あなた"
    return (
        f"【{target_char.display_name}からの応答】\n"
        f"{response_text}\n\n"
        f"【重要な指示】\n"
        f"- 上記は{target_char.display_name}が話した内容です（ルカからの応答ではありません）。\n"
        f"- この応答はすでに{target_char.display_name}の声で視聴者に直接再生されています。\n"
        f"- 逐語的な要約や繰り返しは不要です。「聞いてまいりました」「こう言っていました」"
        f"「○○さんによると」のような第三者報告調も不要です。\n"
        f"- まず{target_char.display_name}の発言に直接リアクション（同意・補足・関連付け・異論など）を返してください。\n"
        f"  例: 「そうですわね、構造としてはまさにその通り」"
        f"「{target_char.display_name}の言う『◯◯』、まさに核心ですわね」のような直接的な呼応。\n"
        f"- そのリアクションを起点に、{caller_name}として自分の視点・感想・次の展開を述べてください。\n"
        f"- 振った相手の発言を無視して独白的に締めることは避けてください"
        f"（視聴者には『振った意味がない』と映ります）。\n"
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
        # ログ強化 L-3: caller_slug=character_slug (target = 応答する側) でログ識別
        from ..llm import call_llm
        result = call_llm(
            question, model=model, provider=provider,
            system_prompt=combined_prompt,
            caller_slug=character_slug,
        )
        return result.text

    # ツールセットから ask_character を除外 (再帰防止)
    # ログ強化 L-3: ask_character のターゲット (= target、応答する側) を渡してログ識別
    all_tools = _load_mcp_tools(character_slug=character_slug)
    collab_tools = [t for t in all_tools if t.name != "ask_character_tool"]

    if not collab_tools:
        # ツールなし → 単純 LLM 呼出し
        from ..llm import call_llm
        result = call_llm(
            question, model=model, provider=provider,
            system_prompt=combined_prompt,
            caller_slug=character_slug,
        )
        return result.text

    # retrieve_memory 用の contextvars をセット (協働先も記憶検索できるように)
    from .retrieve_memory import set_retrieval_context
    set_retrieval_context(
        stream_id=common.get("stream_id"),
        exclude_event_ids=[],
    )

    llm = _get_llm_for_agent(provider, model)
    # 並列ツール呼び出しを抑制 (graph.py と同じ理由: S999 対策)。
    # 協働先 Agent は ask_character を呼べないため再帰並列の懸念は無いが、
    # web_search / retrieve_memory も同時実行されると音声レイテンシが乱れるため
    # 抑制しておく。
    bound_llm = (
        llm.bind_tools(collab_tools, parallel_tool_calls=False)
        if provider in ("openai", "anthropic")
        else llm
    )
    agent = create_react_agent(bound_llm, collab_tools, prompt=combined_prompt)

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
    ask_index: int = 1,
    previous_target_display: str = "",
) -> str:
    """
    導入セリフを軽量 LLM で動的生成する。

    呼び出し元キャラの口調で、以下を生成する:

      ask_index == 1 (1 人目への振り):
        1. ルカの質問を受け止めるコメント (クッション)
        2. 協働先キャラへの問いかけ
        を 1〜3 文で一体生成。

      ask_index >= 2 (2 人目以降):
        ルカへのコメントは省略 (1 人目への振りで既に発話済みのため繰り返しを避ける)。
        直前 target (previous_target_display) との対比 / 補完を意識しつつ、
        現 target に直接語りかける 1〜2 文。

    emotion/speed/pose 付き JSON で返すため、TTS で感情が反映される。

    Args:
        caller_char:              呼び出し元キャラ
        target_char:              協働先キャラ
        user_question:            ルカからの元の質問
        ask_index:                同一ターン内の ask_character 呼出し回数 (1 始まり)
        previous_target_display:  直前に呼び出した target の display_name (なければ空文字)

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

    # 直接呼びかけ用の通り名 (nickname) を優先。空なら display_name を使う。
    # フルネーム ("波心ちさめ" / "八重笠さくら") は固いため、 nickname ("ちさめ" /
    # "さくら") で語りかけることで自然な掛け合いになる。
    target_call_name = target_char.nickname or target_char.display_name
    previous_call_name = previous_target_display  # 既に display_name が入っている

    if ask_index <= 1:
        # 1 人目への振り: ルカへの受け止めコメント + target への問いかけ
        intro_prompt = (
            f"ルカから「{user_question}」と聞かれました。"
            f"これに対して:\n"
            f"1. まずルカの質問を受け止めるコメントを一言\n"
            f"2. 続けて、{target_call_name}に直接語りかけて同じ内容を聞く\n"
            f"を、あなたの口調で自然に繋げて 1〜3 文で返答してください。\n"
            f"{target_call_name}には親しみのある呼び方 ({target_call_name}) で語りかけ、"
            f"フルネーム ({target_char.display_name}) は使わないでください。\n"
            f"通常の応答と同じ JSON フォーマット (emotion/speed/pose/response) で返してください。"
        )
    else:
        # 2 人目以降: 「ルカへのコメント」は 1 人目で済ませているため繰り返さない。
        # 直前 target (previous) と現 target (target_char) の対比/補完を意識し、
        # 短く現 target に直接語りかける。冒頭をルカ呼びかけ (例: 「ふふ、ルカ」)
        # で始めない。
        prev_phrase = (
            f"先ほど{previous_call_name}に同じ問いを振りました。"
            if previous_call_name
            else ""
        )
        intro_prompt = (
            f"ルカからの元の質問「{user_question}」について、"
            f"続けて{target_call_name}にも視点を聞きたい場面です。\n"
            f"{prev_phrase}\n"
            f"指示:\n"
            f"- ルカへの受け止めコメントは入れない (1 人目で既に済んでいるため繰り返さない)\n"
            f"- 「ふふ、ルカ」「その問いは…」のようなルカ呼びかけや感想で始めない\n"
            f"- 「では」「次に」「続いて」などの繋ぎ語、または{target_call_name}の名前から始める\n"
            f"- {target_call_name}に直接語りかけて同じ内容を聞く (1〜2 文)\n"
            f"- 親しみのある呼び方 ({target_call_name}) で語りかけ、"
            f"フルネーム ({target_char.display_name}) は使わない\n"
            f"あなたの口調で自然に書いてください。\n"
            f"通常の応答と同じ JSON フォーマット (emotion/speed/pose/response) で返してください。"
        )

    try:
        # ログ強化 L-3: 導入セリフは caller (= 質問する側) が target に対して話すもの
        # なので、caller_slug=caller_char.slug でログ識別する。
        result = call_llm(
            intro_prompt,
            model=filler_model,
            provider=filler_provider,
            system_prompt=caller_system_prompt,
            caller_slug=caller_char.slug,
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
