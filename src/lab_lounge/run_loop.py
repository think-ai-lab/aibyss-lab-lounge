"""
run_loop.py — ウェイクワード → 録音 → STT → Pipeline → TTS → 再生 の連続ループ

使い方:
    # 基本 (バックエンド自動選択 → 録音 → 応答)
    uv run python -m lab_lounge.run_loop

    # バックエンド指定
    uv run python -m lab_lounge.run_loop --wake-backend speech
    uv run python -m lab_lounge.run_loop --wake-backend porcupine
    uv run python -m lab_lounge.run_loop --wake-backend keyboard

    # 3 ターンで停止
    uv run python -m lab_lounge.run_loop --max-turns 3

    # TTS 再生スキップ
    uv run python -m lab_lounge.run_loop --no-play

前提:
    uv sync --extra wake --extra mic --extra stt --extra llm --extra tts

バックエンド自動選択順序:
    1. L2_PORCUPINE_ACCESS_KEY 設定済み + .ppn ファイルあり → porcupine
    2. sounddevice + openai インストール済み → speech (VAD + STT)
    3. 上記不可 → keyboard (Enter キー)

動作フロー (porcupine / keyboard):
    1. ウェイクワード or Enter キー待機 → speaker_hint 取得
    2. マイク録音 → STT → Pipeline(speaker_hint) → TTS → 再生
    3. ステップ 1 に戻る

動作フロー (speech):
    1. VAD で発話開始検知 → 録音 → STT → キャラクター判定 → transcript 取得
    2. transcript を直接 Pipeline へ (録音・STT を二重実行しない)
    3. ステップ 1 に戻る

フォールバック:
    - 無音 / 録音失敗 → 次のループへ
    - STT / LLM / TTS 失敗 → 警告ログ → 次のループへ
"""

import argparse
import logging
import os
import queue
import threading
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from .audio_io import RecordError, SilenceError, play_audio_file, record_to_file
from .bus import publish
from .emitter import _transcribe_audio
from .events import build_bubble_update
from .log_setup import setup_logging
from .pipeline import PipelineResult, run_pipeline
from .stream_context import load_stream_context

setup_logging(session_name="run_loop")

logger = logging.getLogger(__name__)


def _new_uuid() -> str:
    return str(uuid4())


def _init_listener(
    *,
    backend: str | None,
    audio_device: int | str | None,
    tmp_dir: str | None,
    stt_provider: str = "openai",
) -> tuple:
    """
    ウェイクワードリスナーを初期化する。

    Returns:
        (listener_or_None, effective_backend_name)
        effective_backend_name: "porcupine" | "speech" | "keyboard"
    """
    requested = backend or os.environ.get("L2_WAKE_BACKEND")

    # porcupine バックエンド
    if requested in (None, "porcupine"):
        try:
            from .wake_word import PorcupineListener
            listener = PorcupineListener()
            logger.info("ウェイクワードバックエンド: porcupine")
            return listener, "porcupine"
        except Exception as exc:
            if requested == "porcupine":
                raise
            logger.warning("Porcupine 初期化失敗。speech バックエンドへフォールバック: %s", exc)

    # continuous バックエンド (明示指定のみ)
    if requested == "continuous":
        try:
            from .wake_word import ContinuousListener
            listener = ContinuousListener(
                device=audio_device,
                tmp_dir=tmp_dir,
                stt_provider=stt_provider,
            )
            logger.info("ウェイクワードバックエンド: continuous (常時文字起こし + バッファ)")
            return listener, "continuous"
        except Exception as exc:
            raise

    # bg-continuous バックエンド (Block 0: 録音常時化)
    # 応答パイプライン実行中もマイクを OFF にしないリスナー。
    # BackgroundContinuousListener が別スレッドで録音を継続し、検知した
    # wake_event は Dispatcher (run_loop 本体で生成) の queue に積まれる。
    if requested == "bg-continuous":
        try:
            from .wake_word import BackgroundContinuousListener
            listener = BackgroundContinuousListener(
                device=audio_device,
                tmp_dir=tmp_dir,
                stt_provider=stt_provider,
            )
            logger.info(
                "ウェイクワードバックエンド: bg-continuous (常時録音 + Dispatcher queue)"
            )
            return listener, "bg-continuous"
        except Exception as exc:
            raise

    # sherpa バックエンド (明示指定のみ)
    if requested == "sherpa":
        try:
            from .wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(device=audio_device)
            logger.info("ウェイクワードバックエンド: sherpa (Sherpa-ONNX)")
            return listener, "sherpa"
        except Exception as exc:
            raise

    # speech バックエンド
    if requested in (None, "speech"):
        try:
            from .wake_word import SpeechActivatedListener
            listener = SpeechActivatedListener(
                device=audio_device,
                tmp_dir=tmp_dir,
                stt_provider=stt_provider,
            )
            logger.info("ウェイクワードバックエンド: speech (VAD + STT, provider=%s)", stt_provider)
            return listener, "speech"
        except Exception as exc:
            if requested == "speech":
                raise
            logger.warning("speech バックエンド初期化失敗。keyboard モードへフォールバック: %s", exc)

    # keyboard フォールバック
    logger.info("ウェイクワードバックエンド: keyboard (Enter キー)")
    return None, "keyboard"


def _wait_for_enter() -> None:
    """Enter キー待機（keyboard モード）。"""
    try:
        input("Enter を押して話しかけてください (q で終了): ")
    except EOFError:
        pass


def _is_filler_enabled() -> bool:
    """フィラー音声が有効かどうかを返す。"""
    from .filler import is_filler_enabled
    return is_filler_enabled()


def _run_filler_safe(slug: str, stop_event: threading.Event, user_text: str = "") -> None:
    """フィラー再生ループ（例外を握り潰してログに出す）。"""
    try:
        from .filler import run_filler_loop
        run_filler_loop(slug, stop_event, user_text=user_text)
    except Exception as exc:
        logger.warning("フィラー再生エラー: %s", exc)


def _run_playback_worker(
    q: queue.Queue,
    *,
    publish_bubble_fn,
    play_audio_fn,
    cleanup_audio_fn,
    set_pose_fn=None,
    done_delay_seconds: float = 5.0,
) -> None:
    """
    再生ワーカー: キューから task dict を受け取り、音声再生と bubble publish を調停する。

    動作:
    1. task dict 受信時:
       - キャラクターが変わったら set_pose_fn で立ち絵切替 (再生直前、正確なタイミング)
       - publish_bubble_fn(character, "speaking", chunk_text) を呼ぶ
       - play_audio_fn(url) で再生（ブロッキング）
       - cleanup_audio_fn(url) で後始末
    2. None sentinel 受信時:
       - speaking を 1 回以上 publish 済みなら done_delay_seconds 秒待って
         publish_bubble_fn(last_character, "done", "") を呼んで break
       - 未 publish なら即 break（bubble は V2 安全弁で消える）

    Sprint Axis D Block 1: OBS セリフテロップ表示
    Phase 3: 立ち絵切替を再生時点に移動 (TTS 生成時点ではなく)

    Args:
        q:                   queue.Queue[dict | None] — task dict または None sentinel
            task dict: {"url": str, "text": str, "is_last": bool, "character": str,
                        "pose": str (optional)}
        publish_bubble_fn:   (character: str, step: str, text: str) -> None
        play_audio_fn:       (url: str) -> None  — ブロッキング再生
        cleanup_audio_fn:    (url: str) -> None  — 再生後のファイル削除等
        set_pose_fn:         (character: str, pose: str) -> None  — OBS 立ち絵切替 (optional)
        done_delay_seconds:  最終チャンク再生完了から done publish までの待機秒数
    """
    speaking_published = False
    last_character: str | None = None
    current_pose_character: str | None = None
    current_pose: str | None = None
    while True:
        task = q.get()
        if task is None:
            # 最終チャンク再生完了後: speaking を発行していれば done_delay 秒待って done publish
            if speaking_published and last_character is not None:
                time.sleep(done_delay_seconds)
                publish_bubble_fn(last_character, "done", "")
            break
        # Phase 3: 再生直前にキャラクターが変わったら立ち絵切替 + HUD 通知
        # task["pose"] の値:
        #   - 文字列 (e.g., "special_doya")  → その pose に切替 (chunk 1 など、新規予約時)
        #   - None                           → 切替しない (chunk 2 以降。現在の pose を維持)
        #
        # 「キャラが変わった + pose=None」 のみ neutral にフォールバック。
        # これがないと、chunk 1 で special_doya に切替えた後の chunk 2/3 で neutral
        # に逆戻りしてしまう (chunk 1 だけが pose 値を持ち、後続は None になるため)。
        character = task["character"]
        pose = task.get("pose")  # None or str

        if set_pose_fn:
            should_switch = False
            new_pose: str | None = None
            if character != current_pose_character:
                # キャラ変化 → 必ず切替 (pose 指定があればそれ、無ければ neutral)
                new_pose = pose if pose else "neutral"
                should_switch = True
            elif pose is not None and pose != current_pose:
                # 同キャラだが新しい pose 値が指定された (= bridge filler → 本応答 等)
                new_pose = pose
                should_switch = True
            # else: 同キャラ + pose=None or 同 pose → 切替不要

            if should_switch and new_pose is not None:
                try:
                    set_pose_fn(character, new_pose)
                    logger.info("playback pose 切替: %s → %s", character, new_pose)
                except Exception as exc:
                    logger.warning("playback pose 切替失敗: %s (%s)", character, exc)
                # HUD に立ち絵変更を通知 (publish_bubble_fn 経由)
                # bubble.html は STEP_ORDER 外の "pose_change" を無視するため bubble 表示に影響なし
                publish_bubble_fn(character, "pose_change", f"pose: {new_pose}")
                current_pose_character = character
                current_pose = new_pose
        # speaking publish → 再生 → クリーンアップ → done_event 通知
        # task["text"] が空文字の場合は speaking publish をスキップする。
        # ask_character の target bridge filler が「テロップを出さず直前の
        # thinking テロップを維持したい」ケースでこのパターンを使う
        # (chunk_text="" を on_tts_chunk に渡してくる)。
        chunk_text = task["text"]
        if chunk_text:
            publish_bubble_fn(character, "speaking", chunk_text)
            speaking_published = True
            last_character = character
        play_audio_fn(task["url"])
        cleanup_audio_fn(task["url"])
        # Phase 3: 再生完了通知 (ask_character の導入セリフ同期用)
        done_event = task.get("done_event")
        if done_event is not None:
            done_event.set()


def run_loop(
    *,
    max_turns: int | None = None,
    record_seconds: float = 5.0,
    skip_playback: bool = False,
    wake_timeout: float = 30.0,
    audio_device: int | str | None = None,
    tmp_dir: str | None = None,
    wake_backend: str | None = None,
    stt_provider: str = "openai",
) -> None:
    """
    ウェイクワード → 録音 → 応答 の連続ループ。

    Args:
        max_turns:      最大ターン数 (None = 無制限)
        record_seconds: 録音秒数 (speech バックエンドでは使用しない)
        skip_playback:  TTS 再生をスキップする
        wake_timeout:   ウェイクワード待機タイムアウト秒数
        audio_device:   録音デバイスのインデックスまたは名前
        tmp_dir:        録音一時ファイルの保存先
        wake_backend:   "porcupine" | "speech" | "keyboard" | None (自動選択)
    """
    listener, effective_backend = _init_listener(
        backend=wake_backend,
        audio_device=audio_device,
        tmp_dir=tmp_dir,
        stt_provider=stt_provider,
    )

    # OBS WebSocket 初期化（L2_OBS_WS_URL 未設定時は no-op）
    from .obs import init_obs as _init_obs
    _init_obs()

    mode_msg = {
        "porcupine": "ウェイクワードモード (Porcupine)。キャラクター名を呼んでください。",
        "speech": "音声認識モード (VAD + STT)。キャラクター名を含めて話しかけてください。",
        "sherpa": "音声認識モード (Sherpa-ONNX)。キャラクター名を含めて話しかけてください。",
        "continuous": "常時文字起こしモード。会話の文脈を含めてキャラクター名で呼びかけてください。",
        "bg-continuous": "常時録音モード (BG + Dispatcher)。応答中もマイクが OFF にならず、検知した発話は queue で順次処理されます。",
        "keyboard": "Enter キーモードで起動しました。",
    }
    print(mode_msg.get(effective_backend, "起動しました。"))

    # ─── セッション識別子を run_loop 全体で 1 回だけ生成 ──────────────────
    # stream_id:  音声ストリーム (= この run_loop 実行全体で 1 つ、全ターン共有)
    #             RecentC2Retriever が同一ストリーム内の過去発話を取得するため
    #             ターンごとに再生成すると会話履歴が全て別セッションに分散してしまう
    # session_id: ユーザーセッション (= ここでも同様に run_loop 単位で 1 つ)
    # trace_id:   分散トレース (= ターンごとに再生成、個々のリクエストを追跡)
    session_stream_id = _new_uuid()
    session_id_root = _new_uuid()
    logger.info(
        "run_loop セッション開始: stream_id=%s session_id=%s",
        session_stream_id, session_id_root,
    )

    # ─── 配信文脈の読み込み (起動時1回) ────────────────────────────
    # 「今日の配信内容」(data/stream_context/current.md) を読み込み、
    # 全ターンの run_pipeline() に同じ値を渡す。
    # 配信中にファイルを編集しても反映されない (再起動が必要)。
    # ファイル未存在 / 空時は None で従来通り動作 (後方互換)。
    stream_context = load_stream_context()

    # ─── Dispatcher の生成 (Block 0: bg-continuous バックエンド時のみ) ─────
    # 状態管理 (IDLE / RESPONDING) と wake_event_queue を担う。
    # listener (BackgroundContinuousListener) からの on_wake_detected コールバック
    # で queue に event が積まれ、メインスレッドの dispatcher.wait_for_next_event()
    # で取り出される。応答中も録音は継続される。
    dispatcher = None
    if effective_backend == "bg-continuous":
        from .dispatcher import (
            DRAIN_MAX_AGE_SEC,
            DRAIN_MAX_EVENTS,
            Dispatcher,
        )
        from .events import build_dispatcher_queue_update

        def _publish_queue_update(queue_copy):
            """``dispatcher.queue.update`` イベントを Redis Stream に publish する。

            HUD のデバッグ dashboard で「現在スタックしている応答」を可視化する用途
            (配信画面非表示)。queue 変化時 (add / dequeue / evict) に呼ばれる。
            """
            try:
                now = time.monotonic()
                queue_dicts = [
                    {
                        "character_slug": q.event.character_slug,
                        "keyword": q.event.keyword,
                        "transcript": q.event.transcript,
                        "age_sec": now - q.enqueued_at,
                    }
                    for q in queue_copy
                ]
                event = build_dispatcher_queue_update(
                    queue=queue_dicts,
                    max_size=DRAIN_MAX_EVENTS,
                    ttl_sec=DRAIN_MAX_AGE_SEC,
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    # queue.update イベントは特定ターンに紐付かないため、毎回新 trace_id
                    trace_id=_new_uuid(),
                    state=dispatcher.get_state().value if dispatcher else "idle",
                )
                publish(event)
            except Exception as exc:
                logger.warning("dispatcher.queue.update publish 失敗: %s", exc)

        dispatcher = Dispatcher(on_queue_update=_publish_queue_update)
        # BackgroundContinuousListener を起動。録音スレッドが回り始め、
        # 検知された wake_event は dispatcher.on_wake_detected で queue に積まれる。
        listener.start(on_wake_detected=dispatcher.on_wake_detected)
        logger.info("Block 0: Dispatcher + BackgroundContinuousListener 起動完了")

    def _bg_cleanup_pipeline() -> None:
        """bg-continuous モードでの pipeline 終了処理。

        - listener の routing_paused を解除して wake 判定を再開
        - dispatcher を IDLE 状態に戻し、期限切れ event を破棄、queue 残りがあれば
          メインスレッドを起こす

        Pipeline 成功・失敗を問わず必ず呼ぶ必要がある (routing_paused が True のまま
        放置されると wake 検知が永続的に止まり、wait_for_next_event が無限待機する)。
        """
        if effective_backend == "bg-continuous":
            listener.set_routing_paused(False)
            if dispatcher is not None:
                dispatcher.on_pipeline_complete()

    turn = 0
    try:
        while max_turns is None or turn < max_turns:
            # ─── 1. 起動トリガー ────────────────────────────────
            speaker_hint: str | None = None
            input_text: str | None = None
            utterance_meta = None

            if effective_backend in ("porcupine", "speech", "sherpa", "continuous", "bg-continuous"):
                # bg-continuous は dispatcher の queue から取り出す (応答中も録音継続)
                # その他のバックエンドは listener.listen_once で同期ブロッキング待機
                if effective_backend == "bg-continuous":
                    wake_result = dispatcher.wait_for_next_event(timeout=wake_timeout)
                else:
                    wake_result = listener.listen_once(timeout_seconds=wake_timeout)
                if wake_result is None:
                    # タイムアウト → 再度待機
                    continue
                speaker_hint = wake_result.character_slug
                logger.info("ウェイクワード検知: %s", speaker_hint)
                print(f"ウェイクワード検知: {speaker_hint}")

                # speech / sherpa / continuous / bg-continuous バックエンドは
                # transcript がそのまま発話テキスト (録音 + STT 完了済み)
                if (
                    effective_backend in ("speech", "sherpa", "continuous", "bg-continuous")
                    and wake_result.transcript
                ):
                    input_text = wake_result.transcript
                    logger.info("STT 認識結果: %s", input_text)
                    print(f"認識結果: {input_text}")
            else:
                # keyboard モード
                _wait_for_enter()

            # bg-continuous: pipeline 開始前に listener の routing_paused を True に。
            # これにより応答中は新規 wake_event の通知が止まる (録音とバッファ蓄積は継続)。
            # _bg_cleanup_pipeline() を必ず呼んで False に戻すことを忘れない。
            if effective_backend == "bg-continuous":
                listener.set_routing_paused(True)

            # ─── 2. マイク録音 (porcupine / keyboard のみ) ───────
            if input_text is None:
                logger.info("マイク録音開始 (record_seconds=%d)", record_seconds)
                print("録音中...")
                try:
                    recorded_path = record_to_file(
                        record_seconds,
                        tmp_dir=tmp_dir,
                        device=audio_device,
                    )
                except SilenceError:
                    logger.info("無音検出。次のループへ。")
                    continue
                except RecordError as exc:
                    logger.error("録音失敗: %s", exc)
                    continue

                # ─── 3. STT ─────────────────────────────────────
                try:
                    input_text, utterance_meta = _transcribe_audio(recorded_path)
                except Exception as exc:
                    logger.error("STT 失敗: %s", exc)
                    Path(recorded_path).unlink(missing_ok=True)
                    continue
                finally:
                    Path(recorded_path).unlink(missing_ok=True)

                print(f"認識結果: {input_text}")

            # ─── 4. Pipeline（フィラー + ストリーミング TTS 再生）────
            # 本ターン固有の trace_id を先に生成 (worker 内の bubble.update 発行で使う)
            turn_trace_id = _new_uuid()
            turn_common = {
                "stream_id": session_stream_id,
                "session_id": session_id_root,
                "trace_id": turn_trace_id,
            }

            _chunk_count = [0]
            _playback_queue: queue.Queue[dict | None] | None = None
            _playback_thread: threading.Thread | None = None
            _filler_stop = threading.Event()
            _filler_thread: threading.Thread | None = None

            def _publish_bubble_safe(character: str, step: str, text: str) -> None:
                """bubble.update を publish する。失敗は warning log のみ。"""
                try:
                    event = build_bubble_update(
                        character=character,
                        step=step,
                        text=text,
                        **turn_common,
                    )
                    publish(event)
                except Exception as exc:
                    logger.warning("bubble.update(%s) publish 失敗: %s", step, exc)

            if not skip_playback:
                _playback_queue = queue.Queue()

                def _cleanup_audio(url: str) -> None:
                    """再生済み WAV をディスクから削除する (ディスク節約)。"""
                    try:
                        from .audio_io import _uri_to_path
                        p = Path(_uri_to_path(url))
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass  # 削除失敗は無視 (配信中断を防ぐ)

                # OBS 立ち絵切替関数 (Phase 3: playback worker が再生直前に呼ぶ)
                def _set_pose_safe(character_slug: str, pose: str) -> None:
                    try:
                        from .obs import set_pose
                        set_pose(character_slug, pose)
                    except Exception as exc:
                        logger.warning("OBS pose 切替失敗: %s → %s (%s)", character_slug, pose, exc)

                _playback_thread = threading.Thread(
                    target=_run_playback_worker,
                    args=(_playback_queue,),
                    kwargs={
                        "publish_bubble_fn": _publish_bubble_safe,
                        "play_audio_fn": play_audio_file,
                        "cleanup_audio_fn": _cleanup_audio,
                        "set_pose_fn": _set_pose_safe,
                    },
                    daemon=True,
                )
                _playback_thread.start()

                # フィラー再生開始
                if speaker_hint and _is_filler_enabled():
                    _filler_thread = threading.Thread(
                        target=_run_filler_safe,
                        args=(speaker_hint, _filler_stop, input_text or ""),
                        daemon=True,
                    )
                    _filler_thread.start()

            # OBS 立ち絵 pose の予約 (キャラ別 dict)。パイプラインが on_pose_ready で
            # キャラごとに pose を予約し、対応するキャラの最初の TTS チャンクが
            # 再生キューに入る時に task["pose"] として付与する。
            # Phase 3: 立ち絵切替は playback worker が再生直前に実行 (TTS 生成時ではなく)。
            #
            # 旧設計 (list[1]) では複数キャラの予約が衝突した:
            # - ask_character が target=sakura 用に special_whisper を予約
            # - その後 caller=mimi が最終応答用に special_pondering を予約 → 上書き
            # - sakura chunk 1 再生時に _pending_pose の slug が mimi で一致せず neutral に
            #   フォールバックしてしまう
            # キャラ別 dict にすることで、各キャラの予約が独立に保持される。
            _pending_poses: dict[str, str] = {}

            def _on_pose_ready(slug: str, pose: str) -> None:
                """パイプラインから呼ばれる。pose を予約して実際の切替を遅延させる。"""
                if pose:
                    _pending_poses[slug] = pose

            def _on_tts_chunk(url: str, chunk_text: str, is_last: bool, character: str) -> None:
                # フィラーを停止してから本編を再生
                if _filler_thread is not None and _filler_thread.is_alive():
                    _filler_stop.set()
                    _filler_thread.join(timeout=30)
                    time.sleep(0.5)
                _chunk_count[0] += 1
                _audio_name = url.rsplit("/", 1)[-1] if "/" in url else url
                logger.info(
                    "TTS チャンク再生キュー投入: %s (chunk %d, is_last=%s, len=%d)",
                    _audio_name, _chunk_count[0], is_last, len(chunk_text),
                )
                if _playback_queue is not None:
                    # 該当キャラの pose 予約があれば付与 (playback worker が再生直前に
                    # set_pose する)。一度使ったら dict から削除して、同キャラの後続
                    # チャンクには pose=None を渡す (= 「切替不要、現在の pose を維持」)。
                    # これがないと、chunk 1 で special_doya に切り替わった後、
                    # chunk 2/3 で neutral に逆戻りしてしまう。
                    chunk_pose = _pending_poses.pop(character, None)
                    task = {
                        "url": url,
                        "text": chunk_text,
                        "is_last": is_last,
                        "character": character,
                        "pose": chunk_pose,  # None or pose value
                    }
                    # Phase 3: done_event があれば task に付与 (ask_character 導入セリフ同期用)
                    try:
                        from .mcp_servers.ask_character import _chunk_done_event_var
                        done_event = _chunk_done_event_var.get(None)
                        if done_event is not None and is_last:
                            task["done_event"] = done_event
                            _chunk_done_event_var.set(None)
                    except ImportError:
                        pass
                    _playback_queue.put(task)

            try:
                result = run_pipeline(
                    input_text,
                    # stream_id / session_id は run_loop セッション全体で固定
                    # (RecentC2Retriever が同一ストリーム内の過去発話を引けるように)
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    # trace_id はリクエスト単位 (分散トレース用、個々のターンを識別)
                    trace_id=turn_trace_id,
                    utterance_meta=utterance_meta,
                    speaker_hint=speaker_hint,
                    on_tts_chunk_ready=_on_tts_chunk if not skip_playback else None,
                    on_pose_ready=_on_pose_ready if not skip_playback else None,
                    # 配信文脈は起動時1回ロード済み、全ターン同じ値を渡す
                    stream_context=stream_context,
                )
            except Exception as exc:
                logger.error("Pipeline 失敗: %s", exc)
                _filler_stop.set()
                if _filler_thread is not None:
                    _filler_thread.join(timeout=0.5)
                if _playback_queue is not None:
                    _playback_queue.put(None)
                if _playback_thread is not None:
                    _playback_thread.join(timeout=5)
                # Pipeline 例外時の bubble 閉じは V2 の 30 秒安全弁に委ねる
                # (speaker_hint が不確定なケースもあり、補填 done は見送り)
                # bg-continuous: routing_paused を解除して dispatcher を IDLE に戻す。
                # これがないと次ターンの wait_for_next_event が永続的に blocking する。
                _bg_cleanup_pipeline()
                continue

            # LLM 応答を表示 + ログ記録
            llm_ev = next((ev for ev in result.events if ev["type"] == "llm.final"), None)
            if llm_ev:
                _resp = llm_ev["payload"]["text"]
                logger.info("LLM 応答 [%s]: %s", result.speaker, _resp)
                print(f"[{result.speaker}] {_resp}")

            # ─── 5. TTS 再生完了待機 ────────────────────────────
            # ask_character の協働応答 TTS バックグラウンド合成完了を待つ。
            # これがないと、caller の最終応答 TTS が完了した直後に sentinel が
            # 投げられて、target の chunks がまだ playback queue に投入される前に
            # playback worker が break し、target の発話が途中で切れる
            # (例: 「ちさめ chunk1 だけ再生されて chunks 2-5 が再生されない」)。
            try:
                from .mcp_servers.ask_character import wait_bg_tts_complete
                wait_bg_tts_complete(session_id_root)
            except Exception as exc:
                logger.warning("wait_bg_tts_complete 失敗 (続行): %s", exc)

            if _playback_queue is not None:
                _playback_queue.put(None)
            if _playback_thread is not None:
                _playback_thread.join()

            # ストリーミング未使用時のフォールバック再生（voicevox / edge_tts 等）
            if _chunk_count[0] == 0 and not skip_playback:
                # フィラースレッドの完了を待つ（sounddevice 競合防止）
                if _filler_thread is not None and _filler_thread.is_alive():
                    _filler_stop.set()
                    _filler_thread.join(timeout=30)
                    time.sleep(0.5)

                tts_ev = next((ev for ev in result.events if ev["type"] == "tts.done"), None)
                if tts_ev:
                    audio_url = tts_ev["payload"].get("audio_url", "")
                    fallback_text = tts_ev["payload"].get("text", "")
                    if audio_url:
                        # 非ストリーミング fallback: 再生直前に speaking publish
                        # → 再生 → 5 秒後に done publish
                        _publish_bubble_safe(result.speaker, "speaking", fallback_text)
                        play_audio_file(audio_url)
                        time.sleep(5.0)
                        _publish_bubble_safe(result.speaker, "done", "")
            elif skip_playback:
                # skip_playback=True: worker が走らないので done を直接 publish
                # (pipeline は done を発行しなくなったため)
                _publish_bubble_safe(result.speaker, "done", "")

            # bg-continuous: pipeline 成功 path の終端。
            # routing_paused 解除 + dispatcher 状態を IDLE に戻す + 期限切れ event を破棄。
            # queue に残っている event があれば次の wait_for_next_event で取り出される。
            _bg_cleanup_pipeline()

            turn += 1

    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        if listener is not None:
            listener.cleanup()


# ─── CLI エントリポイント ────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="A.I.byss Lab Lounge — ウェイクワード連続ループ",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="最大ターン数 (省略時は無制限)",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        metavar="N",
        help="録音秒数 (デフォルト 5)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="TTS 再生をスキップする",
    )
    parser.add_argument(
        "--wake-timeout",
        type=float,
        default=None,
        metavar="N",
        help="ウェイクワード待機タイムアウト秒数 (デフォルト 30)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="IDX_OR_NAME",
        help="オーディオデバイスのインデックスまたは名前",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        metavar="DIR",
        help="録音一時ファイルの保存先",
    )
    parser.add_argument(
        "--wake-backend",
        choices=["porcupine", "speech", "sherpa", "continuous", "bg-continuous", "keyboard"],
        default=None,
        metavar="BACKEND",
        help=(
            "ウェイクワードバックエンド: porcupine / speech / sherpa / continuous / "
            "bg-continuous (応答中も録音継続) / keyboard (省略時は自動選択)"
        ),
    )
    parser.add_argument(
        "--stt-provider",
        choices=["openai", "faster-whisper"],
        default=None,
        metavar="PROVIDER",
        help="STT プロバイダ: openai (デフォルト) / faster-whisper (ローカル)",
    )
    args = parser.parse_args(argv)

    record_seconds = args.record_seconds or float(
        os.environ.get("L2_RECORD_SECONDS", "5")
    )
    skip_playback = args.no_play or os.environ.get("L2_NO_PLAY", "false").lower() in (
        "true", "1", "yes"
    )
    wake_timeout = args.wake_timeout or float(
        os.environ.get("L2_WAKE_TIMEOUT", "30")
    )

    device: int | str | None = None
    raw_device = args.device or os.environ.get("L2_AUDIO_DEVICE")
    if raw_device is not None:
        device = int(raw_device) if raw_device.lstrip("-").isdigit() else raw_device

    run_loop(
        max_turns=args.max_turns,
        record_seconds=record_seconds,
        skip_playback=skip_playback,
        wake_timeout=wake_timeout,
        audio_device=device,
        tmp_dir=args.tmp_dir or os.environ.get("L2_TMP_DIR"),
        wake_backend=args.wake_backend or os.environ.get("L2_WAKE_BACKEND"),
        stt_provider=args.stt_provider or os.environ.get("L2_STT_PROVIDER", "openai"),
    )


if __name__ == "__main__":
    main()
