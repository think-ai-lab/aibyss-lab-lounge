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
import time
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
# Phase 0.5-D-1b: BG LLM 経路 (= run_pipeline_llm_only) では対話 TTS chunks を
# 通常応答 _playback_queue に投入せず、_bg_chunk_buffers に蓄積する defer モード
# フラグ。挙手承認時 (= run_loop.on_handraise_approved、D-2 で配線) に drain して
# 専用 mini playback worker で再生することで「ターン跨ぎ問題」(= 配信事故レベル)
# を構造的に解消する。
#
# 【WHY: 通常応答経路では defer=False で既存挙動維持が必須】
# 通常応答経路 (= 挙手なしターン中の ToolNode 即時呼出) では、ask_character の
# 戻り値文字列「【target からの応答】... 上記は target が話した内容です」を
# caller LLM が読んで「target が既に話した前提でリアクション」を組み立てる。
# 通常応答経路で defer=True にすると、caller のリアクション TTS が target の発話前
# に再生される逆順バグになる。よって本フラグは BG LLM 経路でのみ True。
_defer_chunks_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ask_char_defer_chunks", default=False,
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

# Phase 0.5-B-β-2: 却下/lapse 時の bg_tts キャンセルフラグ。run_loop の
# on_handraise_close callback が cancel_bg_tts(session_id) を呼ぶと、該当
# session の Event が set される。導入セリフ TTS / _wrapped_on_chunk_ready /
# _bg_tts_synthesize の各箇所で is_set() チェックして以降の処理を skip する。
#
# 【WHY: VOICEPEAK 合成中の subprocess は止められない】
# 既に subprocess.run(voicepeak.exe ...) で合成中の chunk は OS レベルで止め
# られない。本フラグで阻止できるのは「未起動の bg_tts thread」「未投入の
# chunk」「導入セリフ TTS の起動」の 3 種類。再生中の chunk は最後まで流れる
# が、バッファ済の未再生 chunks は β-2-3 の playback queue drain で破棄する。
_bg_cancel_flags: dict[str, threading.Event] = {}

# Phase 0.5-D-1a: BG LLM 経路の対話 TTS chunks を session 単位で蓄積する buffer。
#
# 【WHY: ターン跨ぎ問題 (= 配信事故レベル) の根本解決のため】
# Phase 0.5-B-β までは ask_character の対話 TTS chunks は通常応答用の
# `_playback_queue` に直接投入されていた。しかし通常応答ターン中に挙手 BG LLM が
# 動き、承認時に bg_result=none (= LLM 推論未完了) で fallback パスに入ると、
# 通常応答ターン終了 → playback worker が None sentinel で break → その後に
# ようやく完了した旧 BG LLM の ask_character chunks が queue に投入されても
# 物理再生されない (= 数分間の無音、シナリオ 3 take 2 で観察)。
#
# 【解】
# BG LLM 経路 (= run_pipeline_llm_only) では chunks を本 buffer に蓄積し、通常応答
# `_playback_queue` には投入しない。承認時 (= run_loop.on_handraise_approved) で
# `_drain_bg_chunks` で取得して run_pipeline_tts_only の chunks と concat し、
# 専用 mini playback worker (= _spawn_handraise_response_playback) で再生する。
# これにより chunks の lifecycle がターン境界から構造的に独立する。
#
# 【経路分岐 (= D-1b で実装)】
# `set_ask_character_context(defer_chunks=True)` で BG LLM 経路を指定する。
# 通常応答経路 (= defer_chunks=False、default) では本 buffer は使わず、既存挙動
# (= `_playback_queue` 即時投入) を維持する (= 通常応答での ask_character は
# caller LLM の戻り値ベース推論を進めるため即時再生が必要)。
_bg_chunk_buffers: dict[str, list[dict]] = {}

# Phase 0.5-D-2: defer モード (= BG LLM 経路) 用の bg_tts 完了 event 登録 dict。
#
# 【WHY: _bg_tts_events から分離する理由】
# 既存の `wait_bg_tts_complete` (= run_loop.py:1804 のターン終了時に呼ばれる) は
# `_bg_tts_events` の全 event を join する。defer モードの bg_tts thread を同 dict
# に登録すると、通常応答ターン終了時に 188 秒の BG LLM 完了を待ってしまい、次
# ターン開始が遅延する (= 配信品質低下)。defer 経路は承認時に
# `wait_deferred_bg_tts_complete` で別途待つので、通常応答ターン終了側は影響を
# 受けないように分離する。
_deferred_bg_tts_events: dict[str, list[threading.Event]] = {}

# Phase 0.5-D-d-7 (= 中間実走 9 回目 take 9-B 修正): streaming spawn 即時化のため、
# session_id ベースで「approval 後の streaming queue 参照」を保持する dict。
#
# 【WHY: streaming queue 参照を session 単位で持つ理由】
# 旧設計 (D-d-6 まで): bg_tts daemon thread が _wrapped_on_chunk_ready で
# `_append_bg_chunk` を呼んで buffer 蓄積。承認時に `_drain_bg_chunks` で全 chunks
# を取得 → streaming spawn。ただし `wait_deferred_bg_tts_complete` で sakura TTS の
# 全 chunks 合成完了 (= 27 秒) まで待つため、mimi 導入セリフが既に合成済でも
# 再生開始が 27 秒遅延 (= take 9-B で観察、logs/runs/run_loop_20260509_234753.log)。
#
# 【新設計 (D-d-7)】
# 承認後、streaming spawn 起動直前に `set_streaming_queue_ref` で session_id に
# streaming queue を登録。以降、`_append_bg_chunk` が呼ばれる際に buffer 蓄積に
# 加えて streaming queue にも投入することで、合成完了次第 chunks が再生される。
# 順序は VOICEPEAK FIFO で保証される (= 直列合成 → 直列投入)。
#
# 【buffer も継続的に蓄積する理由】
# 既存 `_drain_bg_chunks` テスト + 別経路 (= cancel_bg_tts での drain) との互換性
# 維持。streaming queue.put は追加動作のみ、buffer 蓄積は無変更。
_streaming_queue_refs: dict[str, Any] = {}


def _reset_session_state(session_id: str) -> None:
    """セッション状態をリセットする (set_ask_character_context から呼ばれる)。"""
    if not session_id:
        return
    with _ask_state_lock:
        _ask_counts[session_id] = 0
        _previous_targets[session_id] = ""
        # Phase 0.5-D-1a: BG chunk buffer も同時にリセット。
        # 前ターンの未承認 buffer が残ると、次ターンの通常応答経路には影響しない
        # (= defer_chunks=False で buffer を読まない) が、続く挙手 BG LLM ターンで
        # 想定外の合算が起こりうるため、ターン開始時に明示的にクリアする。
        _bg_chunk_buffers.pop(session_id, None)
        # Phase 0.5-D-2: deferred bg_tts events も同 session 分クリーン
        # (= 前ターンの未消化 event が残らないように)
        _deferred_bg_tts_events.pop(session_id, None)
        # Phase 0.5-D-d-7: streaming queue ref も同 session 分クリーン
        # (= 前ターンの ref が次ターンに漏れて意図しない queue.put を起こさないように)
        _streaming_queue_refs.pop(session_id, None)


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


# ─── Phase 0.5-D-1a: BG chunk buffer 操作 API ──────────────────────
# BG LLM 経路で生成された対話 TTS chunks を session 単位で蓄積/取得/カウントする。
# `_bg_chunk_buffers` dict 直接操作の代わりに本 API 経由で `_ask_state_lock` 配下
# の atomic 操作にする (= 並行 daemon thread からの append と承認 callback からの
# drain の race を防ぐ)。
#
# 【module-private】
# `_` 接頭辞で示す通り module-private API。外部は run_loop 経由で `drain_bg_chunks`
# (D-1b で公開ラッパー追加予定) を呼ぶ。本 phase (D-1a) では未配線のため、テスト
# 以外からは呼ばれない。

def _append_bg_chunk(session_id: str, chunk: dict) -> None:
    """指定 session の BG chunk buffer に chunk dict を末尾追加する。

    `_wrapped_on_chunk_ready` (= ask_character.py 内 closure、D-1b で配線) から
    defer モード時に呼ばれる。chunk dict は `_run_playback_worker` が読む形式
    (`{"url", "text", "is_last", "character", "pose"?}`) を想定。

    【Phase 0.5-D-d-7: streaming queue 直接投入】
    `set_streaming_queue_ref` で session_id に streaming queue が登録されている場合
    (= 承認後、streaming spawn 起動済み)、buffer 蓄積に加えて streaming queue にも
    投入する。これにより合成完了次第 chunks が再生される (= take 9-B の 27 秒遅延
    解消)。streaming queue が未登録 (= 承認前) なら buffer 蓄積のみ (= 既存挙動)。

    順序保証: VOICEPEAK FIFO worker が直列合成 → _wrapped_on_chunk_ready callback も
    直列実行 → buffer / queue への append も直列。これにより合成順 = 投入順 = 再生順
    が自動保証される。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op、後方互換)
        chunk:      playback worker 用 chunk dict
    """
    if not session_id:
        return
    with _ask_state_lock:
        _bg_chunk_buffers.setdefault(session_id, []).append(chunk)
        streaming_queue = _streaming_queue_refs.get(session_id)
    # Phase 0.5-D-d-7: streaming queue.put は Lock 外で呼ぶ (= queue 内部で thread-safe、
    # _ask_state_lock 競合回避)。queue 投入失敗時は warning ログのみで buffer 蓄積は
    # 既に完了しているため後続処理 (= cancel_bg_tts での drain) は影響を受けない。
    if streaming_queue is not None:
        try:
            streaming_queue.put(chunk)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_append_bg_chunk streaming queue.put 失敗 [session=%s]: %s",
                session_id, exc,
            )


def _drain_bg_chunks(session_id: str) -> list[dict]:
    """指定 session の BG chunk buffer を atomic に取得 + クリアする。

    `run_loop.on_handraise_approved` (= 承認時、D-2 で配線) と
    `_approved_synthesize_fallback` (= fallback 起動時、D-4 で配線) と
    `cancel_bg_tts` (= 却下/lapse 時、D-3 で統合) から呼ばれる。

    取得 + クリアを `_ask_state_lock` 配下で atomic に行うため、append 中の
    chunk が「取得後に追加されて drain で見逃される」race は発生しない (=
    drain 時点までの chunks は確実に取得される、それ以降は新 buffer に蓄積)。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op、空 list 返却)

    Returns:
        蓄積された chunks の list (= 順序保証、append 順)。未蓄積なら空 list。
    """
    if not session_id:
        return []
    with _ask_state_lock:
        return _bg_chunk_buffers.pop(session_id, [])


def set_streaming_queue_ref(session_id: str, queue_obj: Any) -> None:
    """指定 session の streaming queue 参照を登録する (Phase 0.5-D-d-7)。

    承認後 (= run_loop._tts_and_play 内、streaming spawn 起動直後) に呼ばれる。
    以降 `_append_bg_chunk` が呼ばれる際、buffer 蓄積に加えて queue_obj.put(chunk)
    も実行される (= take 9-B の 27 秒再生開始遅延を解消)。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op)
        queue_obj:  streaming queue (= `_spawn_handraise_response_playback_streaming`
                    の戻り値 _streaming_queue、queue.Queue 互換 .put(item) を持つ)
    """
    if not session_id:
        return
    with _ask_state_lock:
        _streaming_queue_refs[session_id] = queue_obj


def clear_streaming_queue_ref(session_id: str) -> None:
    """指定 session の streaming queue 参照をクリアする (Phase 0.5-D-d-7)。

    通常は `_reset_session_state` で次ターン開始時にクリアされるが、明示的に
    クリーンする場合 (= 例: cancel_bg_tts 経路、エラー時の cleanup) に呼ぶ。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op)
    """
    if not session_id:
        return
    with _ask_state_lock:
        _streaming_queue_refs.pop(session_id, None)


def drain_and_register_streaming_queue(
    session_id: str, queue_obj: Any,
) -> int:
    """承認後に streaming queue を有効化する atomic API (Phase 0.5-D-d-7)。

    1. Lock 内で `_bg_chunk_buffers` から **既に蓄積済**の chunks を全て取得
    2. Lock 内で取得した chunks を `queue_obj.put()` で順次 streaming queue に投入
    3. Lock 内で `_streaming_queue_refs` に `queue_obj` を登録 (= 以降 `_append_bg_chunk`
       が呼ばれる際に buffer 蓄積 + queue.put 両方を実行する経路を有効化)

    全 step を Lock 内で atomic に実行することで、bg_tts daemon thread の
    `_append_bg_chunk` との順序競合を防ぐ (= drained chunks → 新規合成 chunks の
    順序保証、漏れ + 重複なし)。

    【WHY: 旧 wait_deferred_bg_tts_complete 設計の問題】
    旧設計 (D-d-6 まで) は承認後に sakura TTS 全 chunks 合成完了 (= 27 秒) まで
    待ってから streaming spawn を起動していた。mimi 導入セリフが既に合成済でも
    再生開始が 27 秒遅延する問題 (= take 9-B、run_loop_20260509_234753.log)。

    【新設計 (D-d-7)】
    承認直後に即 streaming spawn 起動 → 本 API で「現時点 buffer」を投入 + 「以降の
    chunks」を queue 直投入経路に切替。これにより合成完了次第 chunks が再生開始
    (= mimi 導入セリフは ~8 秒で再生開始 vs 旧 35 秒)。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op)
        queue_obj:  streaming queue (= `_spawn_handraise_response_playback_streaming`
                    の戻り値 _streaming_queue、queue.Queue 互換 .put(item) を持つ)

    【呼出パターン (run_loop._tts_and_play)】
    旧 _drain_bg_chunks で初回 chunks 取得 (= talking_metadata 構築用) →
    streaming spawn 起動 (= initial_chunks=drained) → 本 API で **再度 drain**
    (= 初回 drain と register の間に到着した race chunks の回収) + register。
    本 API での drained chunks は通常 0 件 (= race 時のみ存在)。

    Args:
        session_id: BG LLM の session_id (空文字なら no-op、空 list 返却)
        queue_obj:  streaming queue (= queue.Queue 互換 .put(item) を持つ)

    Returns:
        Lock 内で drained された chunks の list (= queue.put 済、呼出側で再投入
        不要)。通常空 list (= race 時のみ存在)。ログ用。
    """
    if not session_id:
        return []
    with _ask_state_lock:
        chunks = _bg_chunk_buffers.pop(session_id, [])
        for chunk in chunks:
            try:
                queue_obj.put(chunk)
            except Exception as exc:  # noqa: BLE001
                # queue.put 失敗時は warning ログのみ。他の chunks 投入は続行。
                # queue.Queue() の default は無制限のため、通常 block しない (= Lock
                # 保持時間も短時間)。
                logger.warning(
                    "drain_and_register queue.put 失敗 [session=%s]: %s",
                    session_id, exc,
                )
        _streaming_queue_refs[session_id] = queue_obj
    return chunks


def _peek_bg_chunks_count(session_id: str) -> int:
    """指定 session の BG chunk buffer の chunks 数を返す (debug/テスト用、変更しない)。

    Args:
        session_id: BG LLM の session_id (空文字なら 0)

    Returns:
        蓄積済 chunks の数 (= 未蓄積/未知 session は 0)
    """
    if not session_id:
        return 0
    with _ask_state_lock:
        return len(_bg_chunk_buffers.get(session_id, []))


def _register_bg_tts_event(session_id: str, event: threading.Event) -> None:
    """協働応答 TTS バックグラウンドスレッドの完了 event を session に登録する。

    Phase 0.5-D-2: defer モード (= BG LLM 経路、_defer_chunks_var=True) では
    `_deferred_bg_tts_events` に登録し、通常応答経路 (= False) では既存の
    `_bg_tts_events` に登録する。これにより `wait_bg_tts_complete` (= 通常応答
    ターン終了時に呼ばれる) が defer 経路の event を待たないため、188 秒の
    BG LLM 完了でターン終了が遅延しない。defer 経路の event は
    `wait_deferred_bg_tts_complete` (= 承認時に呼ばれる) で待つ。
    """
    if not session_id:
        return
    target_dict = _deferred_bg_tts_events if _defer_chunks_var.get() else _bg_tts_events
    with _ask_state_lock:
        target_dict.setdefault(session_id, []).append(event)


def wait_deferred_bg_tts_complete(session_id: str, timeout: float = 180.0) -> None:
    """Phase 0.5-D-2: defer モードの bg_tts thread 完了を待つ (承認時用)。

    `wait_bg_tts_complete` (= 通常応答経路用) と同じ実装パターンだが、対象 dict
    が `_deferred_bg_tts_events` に分離されている。`run_loop.on_handraise_approved`
    が drain 直前に本関数を呼び、buffer に全 chunks が蓄積された状態で
    `_drain_bg_chunks` するための同期点を提供する。

    Args:
        session_id: 待機対象の session_id (空文字なら no-op)
        timeout:    1 event あたりの最大待機秒数 (default: 180)
    """
    if not session_id:
        return
    with _ask_state_lock:
        events = _deferred_bg_tts_events.pop(session_id, [])
    if not events:
        return
    logger.info(
        "wait_deferred_bg_tts_complete: session=%s pending=%d events 待機開始",
        session_id, len(events),
    )
    for ev in events:
        ev.wait(timeout=timeout)
    logger.info("wait_deferred_bg_tts_complete: session=%s 全 events 完了", session_id)


def cancel_bg_tts(session_id: str) -> int:
    """指定 session の bg_tts cancel flag を set する (Phase 0.5-B-β-2)。

    run_loop の on_handraise_close callback (= 却下/lapse 時) から呼ばれる。
    本関数で阻止できるのは以下 3 種類:
      - 未起動の _bg_tts_synthesize daemon thread の tts_synthesize 呼出
      - _wrapped_on_chunk_ready の on_tts_chunk 投入
      - 導入セリフ TTS の起動 (set_ask_character_context 後の最初の _ask_character_impl
        呼出より前にキャンセルされた場合)

    既に subprocess.run(voicepeak.exe ...) で合成中の chunk は OS レベルで止め
    られない。再生中の chunk も止められない (playback worker が play_audio_file で
    block 中)。これらは β-2-3 の playback queue drain でも破棄できないが、未投入
    の chunks (= まだ queue に乗っていない、もしくは合成中の VOICEPEAK の次の
    chunk) は本フラグで阻止できる。

    Args:
        session_id: cancel 対象の session_id (空文字なら no-op で 0 返却)

    Returns:
        cancel flag set 時点で session に登録されていた bg_tts event 数
        (= 影響を受ける可能性のある bg_tts thread 数の指標)。0 は「未登録」または
        「対象なし」を示す (= 呼出側が「効果なかった」と判別する用途、現状ログのみ)。
    """
    if not session_id:
        return 0
    with _ask_state_lock:
        flag = _bg_cancel_flags.get(session_id)
        if flag is None:
            return 0
        flag.set()
        # 影響範囲は登録済 bg_tts event 数で示す (= 起動済 bg_tts thread の概数)
        # Phase 0.5-D-2: defer 経路と通常応答経路の両方の event を合算する。
        # cancel は両方の経路の bg_tts thread に対して有効 (= cancel_flag は session
        # 単位で 1 つ、defer モードに関係なく set される)。
        n_normal = len(_bg_tts_events.get(session_id, []))
        n_deferred = len(_deferred_bg_tts_events.get(session_id, []))
        n = n_normal + n_deferred
        # Phase 0.5-D-2-α (= D-3 前倒し): defer モードで buffer に蓄積された chunks
        # も同時に drain (= 完全クリーン)。
        #
        # 【WHY: cancel と drain を統合する】
        # 却下/lapse/fallback 起動時に「(a) 後続 chunks 投入を阻止 (= cancel_flag set)」
        # と「(b) 既蓄積 chunks の破棄 (= buffer drain)」は同じ意味論的タイミングで
        # 発火するべき。これを 2 つの API に分けると呼出側で「片方忘れる」不具合が
        # 起こりやすい (= 実走テスト 2026-05-09 で発見した「fallback パスでは
        # buffer drain されず chunks が蓄積されたまま放置」現象)。1 関数で両方を
        # 統合することで構造的に漏れを防ぐ。
        #
        # 影響範囲: TestHandraiseCloseFlow / TestLlmOnlyAskCharacterDenialDrain は
        # cancel_bg_tts の呼出を spy しているだけなので無変更で pass する想定。
        drained_buffer_chunks = _bg_chunk_buffers.pop(session_id, [])
    logger.info(
        "ask_character cancel_bg_tts: session=%s pending_bg_tts=%d "
        "(normal=%d, deferred=%d) drained_buffer_chunks=%d",
        session_id, n, n_normal, n_deferred, len(drained_buffer_chunks),
    )
    return n


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
    defer_chunks: bool = False,
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
        defer_chunks:  Phase 0.5-D-1b で追加。True (= BG LLM 経路) では対話 TTS
                       chunks を通常応答 _playback_queue に投入せず、
                       _bg_chunk_buffers に蓄積する。承認時 (= D-2 で配線) に
                       drain して再生する。False (= 通常応答経路、default) では
                       既存挙動 (= on_tts_chunk callback で即時 _playback_queue 投入)。
                       BG LLM 経路で True 必須の理由は _defer_chunks_var の docstring
                       (= 通常応答経路では caller LLM の戻り値ベース推論を進める
                       ため即時再生が必要、defer=True にすると逆順バグ)。
    """
    _on_tts_chunk_var.set(on_tts_chunk)
    _tts_output_dir_var.set(tts_output_dir)
    _common_var.set(common or {})
    _caller_slug_var.set(caller_slug)
    _on_pose_ready_var.set(on_pose_ready)
    _status_manager_var.set(status_manager)
    _defer_chunks_var.set(defer_chunks)
    # ターン開始時に呼出し回数と直前 target をリセット (各ターン独立にカウント)。
    # graph.py の _generation_node がターン開始時に 1 回呼ぶ前提。
    session_id = (common or {}).get("session_id", "")
    _reset_session_state(session_id)
    # Phase 0.5-B-β-2: cancel flag を session_id 単位でクリーンに作る。前ターンで
    # set されていた flag を継承すると、今ターンの bg_tts が起動直後に skip され
    # てしまうため、新規 Event で上書きする。session_id 空文字なら no-op (= 後方
    # 互換、テスト等で session_id 未指定パターンに対応)。
    if session_id:
        with _ask_state_lock:
            _bg_cancel_flags[session_id] = threading.Event()
    # Phase 0.5-D-e-2: 前ターン以前の LLM client pool を解放 (= メモリリーク防止)。
    # 新ターン開始のタイミングで active_session_id 以外を一括 cleanup する。
    # 承認/却下/lapse のいずれで前ターンが終わっていても、確実に解放される設計。
    # 同 session_id の client (= 同ターン内 normal/bg) は保持され、ターン中の
    # 連続呼出 (= bg_runner + collab agent + fallback) で再利用される。
    try:
        from ..graph import _cleanup_llm_client_pool_except
        _cleanup_llm_client_pool_except(session_id)
    except Exception as exc:  # noqa: BLE001
        # graph.py の import 失敗等は本流に影響させない (= 防衛的)
        logger.debug("LLM client pool cleanup_except 失敗 (続行): %s", exc)


def reset_ask_character_context() -> None:
    """テスト用: contextvars と session 状態をデフォルトにリセットする。"""
    _on_tts_chunk_var.set(None)
    _tts_output_dir_var.set("./data/audio")
    _common_var.set({})
    _caller_slug_var.set("")
    _chunk_done_event_var.set(None)
    _on_pose_ready_var.set(None)
    _status_manager_var.set(None)
    _defer_chunks_var.set(False)  # Phase 0.5-D-1b
    with _ask_state_lock:
        _ask_counts.clear()
        _previous_targets.clear()
        _bg_tts_events.clear()
        _bg_cancel_flags.clear()
        # Phase 0.5-D-1a: BG chunk buffer も全 session 分クリーン
        # (= テスト間の漏れ防止、後方互換性に影響なし)
        _bg_chunk_buffers.clear()
        # Phase 0.5-D-2: deferred bg_tts events も全 session 分クリーン
        _deferred_bg_tts_events.clear()
    # Phase 0.5-D-e-2: LLM client pool もテスト間で漏れないようクリア。
    # graph.py の import 失敗 (= 単体テストでの軽量環境) は黙殺。
    try:
        from ..graph import _clear_llm_client_pool
        _clear_llm_client_pool()
    except Exception:  # noqa: BLE001
        pass


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

    # Phase 0.5-B-β-2: cancel flag を取得して closure に保持する。run_loop の
    # on_handraise_close callback から cancel_bg_tts(session_id) で set される可能性が
    # ある。導入セリフ TTS / _wrapped_on_chunk_ready / _bg_tts_synthesize の
    # 各箇所で is_set() チェックして以降の処理を skip する。
    # _bg_tts_synthesize は別 thread (= contextvars 引き継ぎ問題あり) で走るため、
    # ここでメインスレッドの dict から取得して closure 経由で渡す。
    with _ask_state_lock:
        cancel_flag = _bg_cancel_flags.get(session_id)

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
    # Phase 0.5-D-d-4 (= 中間実走 7 回目で発見した実装漏れ修正):
    # defer モード (= BG LLM 経路) フラグを冒頭で取得して TTS エントリポイント条件
    # に組み込む。defer モードでは on_tts_chunk callback は使わず buffer 蓄積する
    # 設計のため、on_tts_chunk=None でも TTS を起動する必要がある。
    #
    # 【WHY: IDLE 中挙手で on_tts_chunk=None になる経路】
    # ルカが IDLE 中に発話 → 挙手判定 → bg_runner 起動 (= 別 thread)。
    # bg_runner._body は run_loop の `_current_on_tts_chunk[0]` を ref 経由で取得
    # するが、IDLE 中 (= ターン外) では [0] = 前回ターンの closure or None。
    # その値が run_pipeline_llm_only(on_tts_chunk_ready=...) 経由で graph state に乗り、
    # graph._generation_node の set_ask_character_context(on_tts_chunk=...) で
    # contextvars に set される。結果として ask_character 内で on_tts_chunk=None。
    #
    # 旧条件 `if on_tts_chunk and use_real_tts and ...:` だと defer モードでも
    # on_tts_chunk=None で TTS が完全 skip → buffer 蓄積 0 → 承認時 bg_chunks=0 →
    # mimi 〆セリフのみ再生 (= 中間実走 7 回目 take1/take2 の問題)。
    #
    # 中間実走 1〜6 回目では bg_result=none で fallback パスに行っていたため
    # 発覚せず、案 C リファクタ完了 + D-d で承認パスが正常化したことで初めて表面化。
    defer_chunks = _defer_chunks_var.get()

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
    # Phase 0.5-D-d-4: defer モード (= BG LLM 経路) では on_tts_chunk=None でも
    # TTS を起動する。callback は使わず _wrapped_intro_chunk_ready 内の defer 分岐で
    # _append_bg_chunk により buffer 蓄積される (= D-3-b 設計)。
    if (defer_chunks or on_tts_chunk) and use_real_tts and caller_char and intro_response_text:
        # Phase 0.5-B-β-2: cancel flag set されていれば導入セリフ TTS スキップ。
        # 既に却下/lapse されている場合 (= 稀だが、_generate_intro 中に挙手中
        # キャラが lapse する等) は導入セリフを流す意味がない。
        if cancel_flag is not None and cancel_flag.is_set():
            logger.info(
                "ask_character 導入セリフ TTS skip (cancel flag set): session=%s",
                session_id,
            )
        else:
            try:
                from ..tts import synthesize as tts_synthesize

                # Phase 0.5-D-3-b: intro_done event を削除した。
                # 旧: intro_done = threading.Event() + _chunk_done_event_var.set(intro_done)
                # で物理再生完了同期を取っていたが、Phase 0.5-D-3-a で intro_done.wait
                # 廃止 → event 自体が不要になった。run_loop.py:1842-1846 の
                # task["done_event"] attach 経路は ImportError ガードで安全に残す
                # (= 他用途で将来使う可能性、無害)。

                # Phase 0.5-D-2-α: 導入セリフ TTS の合成完了 chunks を投入する直前で
                # cancel_flag check するラッパー。
                # Phase 0.5-D-3-b: defer モード判定を追加。BG LLM 経路では _bg_chunk_buffers
                # に蓄積してターン跨ぎ漏れを構造的に阻止する。
                #
                # 【WHY: cancel ガードは起動前 check だけでは不十分】
                # tts_synthesize 起動前 (= 上の cancel_flag check) では未 set だった
                # cancel_flag が、subprocess.run(voicepeak.exe...) で合成中に
                # lapse/却下で set されることがある (= 実走テスト 2026-05-09
                # logs/runs/run_loop_20260509_150431.log で観察)。chunks 投入直前
                # (= 合成完了後) でもう一度 check することで漏れを完全に阻止する。
                #
                # 【WHY: defer モードで導入セリフも buffer 経由】
                # 通常応答経路 (= defer=False) では caller LLM の戻り値ベース推論を
                # 進めるため即時再生が必要なので既存挙動を完全維持する。BG LLM 経路
                # (= defer=True) では承認時まで再生を遅延させていいため buffer 蓄積に
                # 切替、これにより通常応答 _playback_queue への投入を完全に止めて
                # ターン跨ぎ漏れを構造的に阻止する。chunk dict には inline metadata
                # (= _pre_play_status / _pre_play_bubble) を付けない:
                # - caller の TALKING は _spawn_handraise_response_playback の起動直前
                #   (= run_loop.py:289-296) で反映済 (= talking_metadata 経由)
                # - bubble は通常 worker の publish_bubble_fn(speaking) で発火する
                def _wrapped_intro_chunk_ready(
                    url: str, chunk_text: str, is_last: bool, character: str,
                ) -> None:
                    if cancel_flag is not None and cancel_flag.is_set():
                        logger.info(
                            "ask_character 導入セリフ chunk skip (cancel flag set): "
                            "session=%s character=%s",
                            session_id, character,
                        )
                        return
                    # Phase 0.5-F-3-fix (= 中間実走 14 シナリオ 1 で発覚した HUD 早期 READY 遷移修正):
                    # 導入セリフ chunk は **caller 応答全体の中の中間 chunk** であって、
                    # 真の最後ではない。VOICEPEAK の合成単位で is_last=True が立つが、
                    # 後続する caller まとめ TTS の最後 chunk が真の caller 応答最終。
                    # is_last=False に強制することで、`_run_playback_worker` の
                    # 「is_last=True chunk 物理再生完了時に set_status(READY)」ロジック
                    # (= run_loop.py:1897、Phase 0.5-B-β-3 commit 2 導入) の誤発火を防ぐ。
                    #
                    # 【WHY: F-3 wiring 切替で顕在化したが本質的に F-3 以前から潜在】
                    # 本バグは Phase 0.5-B-β-3 commit 2 から潜在していたタイミング依存
                    # バグ。callout 経路では caller まとめ LLM が ask_character 完了より
                    # 遅く完了する場合が多く偶然顕在化していなかったが、Gemini 多段階
                    # ask_character の高速ケース (= 中間実走 14 シナリオ 1) で顕在化。
                    # 案 R wiring (F-3) で raisehand 承認経路も callout 経路と同じ
                    # `run_pipeline` を通るようになり、同じ条件で顕在化しやすくなった。
                    is_last_safe = False
                    if _defer_chunks_var.get():
                        # BG LLM 経路: buffer 蓄積 (= 承認時に専用 mini worker で再生)
                        chunk = {
                            "url": url, "text": chunk_text,
                            "is_last": is_last_safe, "character": character,
                        }
                        _append_bg_chunk(session_id, chunk)
                        return
                    # 通常応答経路: 既存挙動完全維持 (= ただし is_last は False 強制)
                    on_tts_chunk(url, chunk_text, is_last_safe, character)

                logger.info("ask_character 導入セリフ TTS: [%s] %s", caller_slug, intro_response_text[:60])
                tts_synthesize(
                    intro_response_text,
                    provider=caller_char.tts_provider,
                    voice=caller_char.tts_voice,
                    speaker=caller_char.slug,
                    output_dir=tts_output_dir,
                    on_chunk_ready=_wrapped_intro_chunk_ready,
                )

                # Phase 0.5-D-3-a: intro_done.wait(timeout=120) を廃止。
                #
                # 【WHY: 物理再生完了同期は不要】
                # 旧設計では「導入セリフ wav の物理再生完了」を待っていたが、これが
                # 実走テスト (logs/runs/run_loop_20260509_165916.log) で 80% latency
                # の主因 (= 149 秒中 120 秒) と判明:
                # - mimi 挙手中に導入セリフ wav が _playback_queue に投入される
                # - しかし通常応答ターン用の playback worker が既に終了済 (or
                #   ターン跨ぎで break) → 物理再生されない → intro_done.set() が
                #   呼ばれない → timeout=120 秒で stuck
                # - ルカ「ミミ様、どうぞ」発話時には bg_result=none で fallback パスに
                #   行き、視聴者が 2 分半待たされる配信事故レベルの UX 不具合
                #
                # 【順序保証は別経路で実現】
                # - VOICEPEAK FIFO worker (= tts.py:_voicepeak_worker_fn) が直列化 →
                #   「導入セリフ → 本応答」の合成完了順は VOICEPEAK 内部で保証
                # - playback queue は FIFO → 投入順 = 物理再生順
                # - 連続 ask_character の順序保証は ask_character.py 冒頭の
                #   wait_bg_tts_complete で維持 (= 既存対処継続)
                #
                # 【効果】
                # ask_character 処理時間: 149 秒 → 約 25 秒 (= 並行 LLM/TTS の最大値)
                # bg_result=ready 確率: 0% → ~80% (= 30 秒以内に承認されれば成立)
                # 案 W'-1 設計意図 (= LLM 先行生成 → 承認時 TTS 再利用) が機能し始める。
                #
                # 【中間状態 (D-3-b 未実装時)】
                # 導入セリフ chunks は依然として通常応答 _playback_queue に流れる
                # (= _wrapped_intro_chunk_ready が defer 分岐していない) ため、
                # ターン跨ぎ漏れの可能性は残る。D-3-b で defer 経路統合により構造的に
                # 解消する。
                #
                # 【intro_done event は保持】
                # `intro_done = threading.Event()` と `_chunk_done_event_var.set(intro_done)`
                # は D-3-a 段階では保持 (= run_loop.py:1842-1846 の task["done_event"]
                # 経路は維持、ImportError ガードで安全)。D-3-b で削除予定。
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
    # Phase 0.5-D-d-4: defer モード (= BG LLM 経路) では on_tts_chunk=None でも
    # TTS を起動する。本応答 TTS chunks は _wrapped_on_chunk_ready 内の defer 分岐
    # で _append_bg_chunk により buffer 蓄積、bridge filler は下の elif _is_defer_mode
    # で chunk dict + inline metadata で buffer 蓄積される (= D-2/D-3-c 設計)。
    if (defer_chunks or on_tts_chunk) and use_real_tts:
        from ..pipeline import _publish_bubble, _load_bubble_messages

        # Phase 0.5-D-3-c: defer モード判定をローカルにキャプチャ。
        # 5-a の即時 thinking 発火 (= 通常モード) と、bridge filler 投入の defer 分岐
        # (= D-3-c) で同じ判定値を使う。
        # Phase 0.5-D-d-4: 上で取得した defer_chunks を再利用しても良いが、本ローカル
        # 変数 _is_defer_mode は周辺コードで複数箇所参照されているため変数名は維持。
        _is_defer_mode = defer_chunks

        # 5-a. target の "考え中" bubble を発行する (filler 再生中のテロップ用)。
        # graph.py の _generation_node が caller の thinking bubble を出すのと同じ仕組みで、
        # filler が target の声で再生されている間、V2 HUD には target の thinking テキスト
        # (例: chisame「分析しています」/ sakura「んー……考え中ですよぉ」) を表示する。
        #
        # Phase 0.5-D-3-c: defer モードでは skip (= 物理再生時に inline metadata で発火)。
        # 「承認前に target THINKING / thinking bubble が表示される」UX 不具合を構造的に
        # 解消する。物理再生発火は bridge filler chunk の `_pre_play_bubble` /
        # `_pre_play_status` (= 下の 5-b で埋込) 経由で worker が発火する。
        if common and not _is_defer_mode:
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
        # Phase 0.5-D-3-c: defer モードでは skip (= 同様、物理再生時に inline metadata
        # 経由で発火、二重発火防止 + UX 不具合解消)。
        _status_manager_for_target = _status_manager_var.get()
        if _status_manager_for_target is not None and not _is_defer_mode:
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
                # Phase 0.5-D-2-α: bridge filler 投入前に cancel_flag check。
                # 却下/lapse 後 or fallback 起動後に bridge filler が
                # _playback_queue に投入されないようにガードする (= 実走テスト
                # 2026-05-09 logs/runs/run_loop_20260509_155109.log で観察された
                # 「fallback パス後に sakura bridge filler が漏れて再生」現象の対処)。
                # bridge filler は _wrapped_on_chunk_ready を経由しない直接呼出のため、
                # 案 C の defer 経路 (= chunk dict 経由 buffer) でもガードされない。
                # ここで明示的に cancel_flag check で阻止する。
                if cancel_flag is not None and cancel_flag.is_set():
                    logger.info(
                        "ask_character target bridge filler skip (cancel flag set): "
                        "[%s] session=%s",
                        target_char.slug, session_id,
                    )
                elif _is_defer_mode:
                    # Phase 0.5-D-3-c: defer モードでは bridge filler chunk も buffer に
                    # 蓄積する (= ターン跨ぎ漏れの構造的阻止)。inline metadata
                    # (`_pre_play_status` + `_pre_play_bubble`) を埋め込んで、物理再生時
                    # に target THINKING + thinking bubble を発火させる経路に統合する
                    # (= 上の 5-a の即時発火を defer モードでは skip 済み、UX 不具合解消)。
                    #
                    # chunk_text="" は通常モードと同じく speaking publish skip 仕組み維持
                    # (= _run_playback_worker で text 空なら speaking publish skip → 直前の
                    # thinking テロップを維持)。`_pre_play_bubble` の text には キャラ別
                    # yaml の "thinking" メッセージ (= 通常モードの _publish_bubble 相当)
                    # を入れる。
                    bubble_msgs = _load_bubble_messages().get(target_char.slug, {})
                    chunk = {
                        "url": bridge_path.as_uri(),
                        "text": "",
                        "is_last": False,
                        "character": target_char.slug,
                        "_pre_play_status": {
                            "slug": target_char.slug,
                            "status": "THINKING",
                        },
                        "_pre_play_bubble": {
                            "slug": target_char.slug,
                            "step": "thinking",
                            "text": bubble_msgs.get("thinking", ""),
                        },
                    }
                    _append_bg_chunk(session_id, chunk)
                    logger.info(
                        "ask_character target bridge filler buffer 蓄積 (defer): [%s] %s",
                        target_char.slug, bridge_path.name,
                    )
                else:
                    # 通常応答経路: 既存挙動完全維持
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
        # Phase 0.5-D-1b: defer モード判定もメインスレッドで closure capture。
        # True なら BG LLM 経路として _bg_chunk_buffers に蓄積、False なら通常応答
        # 経路として既存挙動 (= on_tts_chunk で _playback_queue 即時投入)。
        defer_chunks_capture = _defer_chunks_var.get()

        # response_text から pose と say_text を事前に抽出 (_wrapped_on_chunk_ready
        # で使用)。本応答 chunk 1 が投入される直前に on_pose_ready を呼ぶことで、
        # _pending_poses に予約するタイミングと _on_tts_chunk で pop されるタイミング
        # の順序が保証される (= bridge filler 投入時には予約がなく neutral、本応答
        # chunk 1 投入時に target_pose が予約されている状態を作る)。
        # Phase 0.5-B-β-3 commit 1: say_text も同時に取得し、TALKING metadata.text
        # に渡すことで HUD ダッシュボードに「response 部分のみ」を表示する
        # (= JSON 全文 ({"emotion":..., "response":..., ...}) が HUD に出てしまう
        # 不具合の修正、シナリオ 2 で観察)。通常応答経路の graph._tts_node も
        # 同じ前処理を行っており、metadata 形を統一する。
        from ..tts import _parse_voicepeak_json
        target_say_text, _, _, target_pose = _parse_voicepeak_json(response_text)

        def _wrapped_on_chunk_ready(url: str, chunk_text: str, is_last: bool, character: str) -> None:
            # Phase 0.5-B-β-2: cancel flag set されていれば chunk 投入 skip。
            # bg_tts thread の合成は止められないが、playback queue への投入を阻止
            # することで「却下後に target の応答音声が流れ続ける」状況を防ぐ。
            # answering bubble / pose 予約 / TALKING ステータスも合わせて skip する
            # (= これらは「target が話す」前提の演出なので、cancel 後は不要)。
            if cancel_flag is not None and cancel_flag.is_set():
                return

            # Phase 0.5-D-1b: defer モード分岐 — BG LLM 経路では chunks を buffer に
            # 蓄積し通常応答 _playback_queue には流さない。承認時 (= D-2 で配線) に
            # drain → _spawn_handraise_response_playback で専用 mini worker 再生する
            # ことで、ターン跨ぎ問題 (= 配信事故レベル、シナリオ 3 take 2 で観察) を
            # 構造的に解消する。
            #
            # 【answering bubble / pose 予約 / TALKING 反映は本 phase では skip】
            # buffer 蓄積タイミングで反映すると「承認前に target が TALKING 表示」
            # される UX 不具合が起こる。D-2 で物理再生時 (= playback worker の chunk
            # 再生直前) に発火する設計に移行する (= chunk dict に inline metadata
            # として埋め込む経路)。本 phase ではタイミングずれの中間状態として全部
            # skip し、再生時の発火に集約する準備を進める。
            #
            # 【pose は chunk dict に直接埋め込む】
            # 通常応答経路は on_pose_ready callback 経由で run_loop の _pending_poses
            # にセット → 直後の _on_tts_chunk で pop → chunk dict にセット、という
            # 流れだが、defer 経路ではこの run_loop closure に依存しない (= ターン跨ぎ
            # 独立性のため)。chunk dict の "pose" に直接埋め込み、専用 mini worker
            # (= _spawn_handraise_response_playback の _run_playback_worker) が
            # task["pose"] を読んで OBS 立ち絵切替する経路。
            if defer_chunks_capture:
                chunk: dict[str, Any] = {
                    "url": url, "text": chunk_text,
                    "is_last": is_last, "character": character,
                }
                if not first_chunk_seen[0]:
                    # first chunk: 物理再生時 (= playback worker pop 時) に発火する
                    # metadata を chunk dict に埋め込む。worker 側で
                    # `_pre_play_status` / `_pre_play_bubble` を読んで dispatch する
                    # (= run_loop._run_playback_worker、Phase 0.5-D-2 で配線)。
                    if target_pose:
                        chunk["pose"] = target_pose
                    # _pre_play_status: target キャラの TALKING を物理再生開始時に反映。
                    # 通常応答経路 (= defer=False) では本 closure 内で即時
                    # set_status を呼ぶが、defer 経路では「承認前は再生しない」ため
                    # 「buffer 投入時」ではなく「物理再生開始時」に反映するのが正しい
                    # (= 承認前に target が TALKING 表示される UX 不具合の防止)。
                    if status_manager_capture is not None:
                        chunk["_pre_play_status"] = {
                            "slug": target_char.slug,
                            "status": "TALKING",
                            "metadata": {
                                "pose": target_pose if target_pose else None,
                                "text": target_say_text or response_text,
                            },
                        }
                    # _pre_play_bubble: answering bubble を物理再生開始時に発行。
                    # 通常応答経路の `_publish_bubble("answering", target_char.slug,
                    # common)` 相当の text を _load_bubble_messages から取得する
                    # (= ask_character.py:632 の通常モード経路と同じテキスト生成方法、
                    # キャラ別 yaml の "answering" メッセージ)。
                    if common:
                        try:
                            from ..pipeline import _load_bubble_messages
                            bubble_msgs = _load_bubble_messages().get(target_char.slug, {})
                            chunk["_pre_play_bubble"] = {
                                "slug": target_char.slug,
                                "step": "answering",
                                "text": bubble_msgs.get("answering", ""),
                            }
                        except Exception as exc:
                            logger.warning(
                                "ask_character defer mode bubble messages 取得失敗 (%s): %s",
                                target_char.slug, exc,
                            )
                first_chunk_seen[0] = True
                _append_bg_chunk(session_id, chunk)
                return

            # 通常応答経路 (= defer_chunks=False、既存挙動完全維持)
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
                        # Phase 0.5-B-β-3 commit 1: text は say_text (= JSON parse 後の
                        # response 部分) を優先する。parse 失敗時は元の response_text に
                        # fallback (= 旧挙動、JSON 全文だが視認可能性は維持)。
                        talking_metadata: dict[str, Any] = {
                            "pose": target_pose if target_pose else None,
                            "text": target_say_text or response_text,
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
            # Phase 0.5-D-d-6 (= 中間実走 8 回目 take 1-2 hang 調査用ログ追加):
            # _bg_tts_synthesize daemon thread のライフサイクルを詳細追跡する。
            #
            # take 1-2 で sakura TTS が 50 秒走る間に mimi Agent が hang。本ログで
            # thread 起動 / 完了タイミングを正確に把握 → 「mimi Agent hang と sakura
            # TTS 合成」の時系列の関連性を切り分ける。
            _bg_tts_start = time.monotonic()
            logger.info(
                "_bg_tts_synthesize 開始 [target=%s]: session=%s thread_id=%d "
                "active_threads=%d",
                character_slug, session_id or "(none)",
                threading.get_ident(), threading.active_count(),
            )
            synthesize_failed = False
            cancelled = False
            try:
                # Phase 0.5-B-β-2: cancel flag set されていれば tts_synthesize 起動
                # を skip。VOICEPEAK の subprocess.run は止められないが、起動前なら
                # 完全に阻止できる (= 一番早い cancel タイミング、CPU/GPU 浪費なし)。
                if cancel_flag is not None and cancel_flag.is_set():
                    cancelled = True
                    logger.info(
                        "ask_character 協働応答 TTS skip (cancel flag set): "
                        "target=%s session=%s",
                        character_slug, session_id,
                    )
                    return
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
                synthesize_failed = True
                logger.warning(
                    "ask_character 協働応答 TTS 合成失敗 (%s): %s",
                    character_slug, exc,
                )
            finally:
                # Phase 0.5-B-β-3 commit 2: 正常系の READY 反映は playback worker
                # (= is_last chunk 物理再生完了時、run_loop._run_playback_worker)
                # に移動した。bg_tts 合成完了 != 物理再生完了の不整合を解消する
                # ため (= シナリオ 2 で観察、HUD で発話途中に灰色化する不具合)。
                #
                # ただし例外時 (= 合成失敗) と cancel 時は chunks が playback queue
                # に入らないため、playback worker 経由の READY 反映が走らない。
                # その場合は本 finally で READY を反映して UI stuck を防ぐ
                # (= fallback 経路、HUD カードが talking のまま残らないようにする)。
                if (synthesize_failed or cancelled) and status_manager_capture is not None:
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
                # Phase 0.5-D-d-6: thread 完了タイミングログ。take 1-2 で sakura TTS
                # 完了時刻 (= 本ログ) と mimi Agent 完了時刻のギャップを測るシグナル。
                _bg_tts_latency_ms = int((time.monotonic() - _bg_tts_start) * 1000)
                logger.info(
                    "_bg_tts_synthesize 完了 [target=%s]: session=%s "
                    "latency_ms=%d failed=%s cancelled=%s active_threads=%d",
                    character_slug, session_id or "(none)",
                    _bg_tts_latency_ms,
                    synthesize_failed, cancelled, threading.active_count(),
                )
                bg_tts_done.set()

        _register_bg_tts_event(session_id, bg_tts_done)
        threading.Thread(target=_bg_tts_synthesize, daemon=True).start()
        logger.info(
            "ask_character 協働応答 TTS 合成をバックグラウンド開始: target=%s session=%s",
            character_slug, session_id or "(none)",
        )

    # Phase 0.5-D-d-6: ask_character return 直前の active_threads を記録。
    # take 1-2 で「return 後 mimi Agent が走らない」現象の原因切り分け用 (= thread
    # 数の急増 / Lock contention の兆候を検出するベースライン値)。
    logger.info(
        "ask_character 完了: target=%s response_len=%d active_threads=%d",
        character_slug, len(response_text), threading.active_count(),
    )

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
