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
from .character_status import CharacterStatus, CharacterStatusManager
from .emitter import _transcribe_audio
from .events import build_bubble_update, build_character_status_update
from .log_setup import setup_logging
from .pipeline import PipelineResult, run_pipeline
from .stream_context import load_stream_context
from .wake_word import WakeWordResult

setup_logging(session_name="run_loop")

logger = logging.getLogger(__name__)


def _new_uuid() -> str:
    return str(uuid4())


# ─── Phase 0.5-B-α: CharacterStatusManager 連携ヘルパー ──────────────


def _extract_llm_response_text(events: list[dict]) -> str:
    """events から llm.final の payload.text を取得し、JSON 形式なら response 部分のみ抽出する。

    Phase 0.5-B-α: character.status.update の metadata.text として送る発話全文を
    取得するためのヘルパー。Phase 0.5-A バグ 4 修正で確立した _parse_voicepeak_json
    による response 抽出ロジックを再利用して、HUD dashboard に生 JSON が出ないように
    する (= bubble.update でも同じ抽出をしている)。

    WHY: llm.final.text は Pydantic JSON 文字列 ({"response":"...","emotion":...,
    "speed":...,"pose":"..."}) なので、character.status.update の text に生 JSON を
    詰めると HUD で読めなくなる。response 部分のみを抽出して人が読める text にする。

    Args:
        events: PipelineResult.events または bg_result.result.events のリスト

    Returns:
        LLM response text。llm.final が無い / text が空 の場合は空文字を返す
        (= caller 側で「if llm_text:」で判定可能)。
    """
    from .tts import _parse_voicepeak_json

    for ev in events:
        if ev.get("type") == "llm.final":
            text_raw = ev.get("payload", {}).get("text", "")
            if text_raw:
                # _parse_voicepeak_json は parse 失敗時 (= JSON でない / response 無し)
                # でも (text, None, None, None) で元テキストをそのまま返す
                response_text, _, _, _ = _parse_voicepeak_json(text_raw)
                return response_text or ""
            break
    return ""


def _build_talking_metadata(
    character: str,
    llm_text: str | None,
    pose: str | None,
) -> dict:
    """Talking 状態の metadata を構築する (Phase 0.5-B-α)。

    HUD dashboard で立ち絵 + 発話全文を表示するため、両方を含める (= ルカ追加要件)。
    どちらも None / 空の場合は metadata から omit する設計:
    - 「空 string や None キー」を含めるより HUD 側で「未取得」を判定しやすい
    - publish payload も小さく抑えられる

    Args:
        character: キャラ slug (= 現状 metadata 内には含めず、log 用識別のみ。
                   character は status.update event の payload top-level にある)
        llm_text:  発話全文 (= _extract_llm_response_text 等で取得した response text)
        pose:      現在の立ち絵 (= chunk task の pose / _pending_poses 等から取得)

    Returns:
        {"pose": str, "text": str} のうち、値がある field のみ含む dict。
        両方 None / 空 の場合は空 dict (= "metadata なし" と区別したい場合は呼出側で
        判定して None を渡す方針)。
    """
    metadata: dict = {}
    if pose:
        metadata["pose"] = pose
    if llm_text:
        metadata["text"] = llm_text
    return metadata


# ─── Phase 0.5-A フェーズ 7: 挙手 BG LLM + handraise 再生統合 ────────


def _get_handraise_responding_padding_sec() -> float:
    """応答完了 → 保留 handraise wav 再生までの空白秒数 (Phase 0.5-A フェーズ 7)。

    RESPONDING 中に挙手したキャラの handraise wav を IDLE 復帰直後に再生すると、
    視聴者には立て続けに 2 つの音声が聞こえて品位を損ねる。間に padding_sec の
    sleep を入れて自然な間を作る。

    環境変数 ``L2_HANDRAISE_RESPONDING_PADDING_SEC`` (default 1.5 秒) で調整可。
    """
    return float(os.environ.get("L2_HANDRAISE_RESPONDING_PADDING_SEC", "1.5"))


def _spawn_handraise_phrase_playback(
    slug: str,
    phrase_path: "Path | None",
    padding_sec: float = 0.0,
    *,
    play_audio_fn=None,
) -> "threading.Thread | None":
    """handraise wav を別 daemon スレッドで再生する (Phase 0.5-A フェーズ 7)。

    sounddevice は global state なので、通常応答 TTS と同時実行すると競合する。
    そのため dispatcher 側で「IDLE 中のみ即再生 / RESPONDING 中は se_pending=True で
    保留 → on_pipeline_complete で release callback 経由で再生」の経路を選択する。
    本関数は実際の物理 IO のみを担当し、排他制御は呼出側の責任。

    Args:
        slug:        挙手キャラ slug (ログ用)
        phrase_path: 再生する wav のパス。None なら no-op (テスト時の安全策)
        padding_sec: 再生前の sleep 秒数。RESPONDING 完了直後の連続再生を回避する
                     用途で 1.5 秒程度の値が渡される (default 0 = IDLE 即時再生時)
        play_audio_fn: テスト差し替え用。None なら ``audio_io.play_audio_file``

    Returns:
        起動した daemon thread。phrase_path=None なら None。
    """
    if phrase_path is None:
        return None
    if play_audio_fn is None:
        play_audio_fn = play_audio_file

    def _body() -> None:
        if padding_sec > 0:
            time.sleep(padding_sec)
        try:
            play_audio_fn(str(phrase_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "handraise phrase 再生失敗: slug=%s err=%s", slug, exc,
            )

    t = threading.Thread(
        target=_body,
        name=f"handraise-phrase-{slug}",
        daemon=True,
    )
    t.start()
    return t


def _spawn_handraise_response_playback_streaming(
    slug: str,
    initial_chunks: list[dict],
    trace_id: str,
    *,
    session_stream_id: str,
    session_id_root: str,
    status_manager: "CharacterStatusManager | None" = None,
    talking_metadata: "dict | None" = None,
) -> "tuple[threading.Thread, queue.Queue]":
    """Phase 0.5-D-3 follow-up 2: 動的 chunks 追加対応の streaming 版 spawn。

    既存 ``_spawn_handraise_response_playback`` と異なり、None sentinel は caller が
    後から投入する。これにより run_pipeline_tts_only の TTS 合成中にも、合成完了した
    chunk から順次 queue に動的追加できる。

    【WHY: bg_chunks 即時再生 + tts_only 並行投入】
    実走テスト 2026-05-09 logs/runs/run_loop_20260509_190927.log で観察された
    「approval 後 18.5 秒の空白」(= run_pipeline_tts_only が同期 blocking で全
    tts_only chunks 蓄積完了まで待ってから spawn) の根本対処。bg_chunks (= 既に
    合成済の導入セリフ + bridge filler + sakura 本応答 = 4 件) を即時再生開始し、
    tts_only chunks (= caller の〆セリフ) は合成完了次第 queue に投入することで、
    視聴者の体感を「approval → 即「ルカ、その問いは…」」に近づける (= 案 W'-1 の
    本来の設計意図)。

    Args:
        slug:               挙手キャラ slug
        initial_chunks:     起動時に投入する chunks (= bg_chunks 等の既蓄積分)
        trace_id:           bubble.update 発行時の handraise 単位 trace_id
        session_stream_id:  bubble.update の stream_id
        session_id_root:    bubble.update の session_id
        status_manager:     CharacterStatusManager
        talking_metadata:   Talking 反映時の metadata

    Returns:
        (起動した daemon thread, 投入用 queue) のタプル。caller は queue.put(chunk) で
        chunks を動的追加し、最後に queue.put(None) で worker 終了通知する。
    """
    q: queue.Queue = queue.Queue()
    for chunk in initial_chunks:
        q.put(chunk)
    # 注: None sentinel は caller が後から投入する (= 既存 _spawn_handraise_response_playback
    # との挙動の違い)。run_pipeline_tts_only 完了後に caller が queue.put(None) を呼ぶ。

    def _publish_bubble_for_handraise(character: str, step: str, text: str) -> None:
        """既存 _spawn_handraise_response_playback と同じ bubble 発行ロジック。"""
        try:
            event = build_bubble_update(
                character=character,
                step=step,
                text=text,
                stream_id=session_stream_id,
                session_id=session_id_root,
                trace_id=_new_uuid(),
                category="speech",
            )
            publish(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "handraise playback (streaming) bubble.update(%s) publish 失敗: %s", step, exc,
            )

    def _cleanup_audio(url: str) -> None:
        try:
            from .audio_io import _uri_to_path
            p = Path(_uri_to_path(url))
            if p.is_file():
                p.unlink()
        except Exception:
            pass

    def _set_pose_safe(character: str, pose: str) -> None:
        try:
            from .obs import set_pose
            set_pose(character, pose)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "handraise playback (streaming) set_pose 失敗 (無視): %s", exc,
            )

    if status_manager is not None:
        status_manager.set_status(
            slug, CharacterStatus.TALKING, metadata=talking_metadata,
        )

    def _worker_with_status_tracking() -> None:
        try:
            _run_playback_worker(
                q,
                publish_bubble_fn=_publish_bubble_for_handraise,
                play_audio_fn=play_audio_file,
                cleanup_audio_fn=_cleanup_audio,
                set_pose_fn=_set_pose_safe,
                status_manager=status_manager,
            )
        finally:
            if status_manager is not None:
                status_manager.set_status(slug, CharacterStatus.READY)

    t = threading.Thread(
        target=_worker_with_status_tracking,
        daemon=True,
        name=f"handraise-playback-streaming-{slug}",
    )
    t.start()
    return t, q


def _create_handraise_runner_and_callbacks(
    *,
    session_stream_id: str,
    session_id_root: str,
    stream_context: "str | None",
    status_manager: "CharacterStatusManager | None" = None,
    on_tts_chunk_ready_ref: "list | None" = None,
    on_pose_ready_ref: "list | None" = None,
    playback_queue_ref: "list | None" = None,
):
    """Phase 0.5-A フェーズ 7: 4 つの callback と bg_runner を生成する factory。

    run_loop の closure 内変数 (session_stream_id 等) を捕捉した上で、
    ``Dispatcher`` コンストラクタに渡せる callable 群を返す。module-level の
    factory として書くことで、テストから直接 factory を呼んで返り値を検証できる。

    Args:
        session_stream_id: 当該セッションの stream_id (handraise イベントに付与)
        session_id_root:   当該セッションの session_id
        stream_context:    配信文脈 markdown (run_pipeline に渡す)
        status_manager:    Phase 0.5-B-α で追加。挙手承認応答経路で Talking +
                           Ready を反映するために bg_runner / approved callback
                           で参照する。
        on_tts_chunk_ready_ref: Phase 0.5-B-β-1 commit 3 で追加。
                           ``list[Callable | None]`` (= 1 要素の mutable list)
                           で渡す。bg_runner._body は ``ref[0]`` 経由で最新の
                           ``_on_tts_chunk`` closure を取得し、
                           ``run_pipeline_llm_only(on_tts_chunk_ready=ref[0])``
                           に渡す。これにより BG LLM 経路の ask_character ツール
                           で対話 TTS chunk が playback queue に届くようになる
                           (= A1 修正の最終配線)。
        on_pose_ready_ref: 同様に target キャラの pose 切替予約 callback の
                           mutable list。
        playback_queue_ref: Phase 0.5-B-β-2 commit 3 で追加。
                           ``list[queue.Queue | None]`` で渡す。on_handraise_close
                           callback が ``ref[0].put({"_drain": True})`` で playback
                           queue に drain task を投入し、却下/lapse 時の残 chunks を
                           破棄する経路。on_tts_chunk_ready_ref / on_pose_ready_ref
                           と同じ late binding パターンで、ターン毎に再構築される
                           _playback_queue を最新参照する。

    【WHY: mutable list ラッパー (= late binding) を使う】
    factory はセッション開始時 1 回だけ呼ばれる。一方 ``_on_tts_chunk`` /
    ``_on_pose_ready`` は run_loop の **ターン毎の内側 closure として再構築**
    される (= run_loop.py:1509, 1514)。factory が直接 callable を受け取ると、
    最初のターン以降の closure 更新を反映できない (= 古い closure を保持し続ける)。
    mutable list の [0] 経由で参照すれば、run_loop がターンごとに ``ref[0]`` を
    上書きすることで、bg_runner._body は常に最新の closure を見られる。

    Returns:
        (bg_runner, on_handraise_started, on_handraise_phrase_pending_release,
         on_handraise_approved) の tuple。``Dispatcher(**kwargs)`` に渡す形。
    """

    def bg_runner(
        *,
        target_slug: str,
        transcript_snapshot,
        cancel_event: "threading.Event",
        on_complete,
    ) -> "threading.Thread":
        """BG LLM スレッドを起動して thread を返す (Phase 0.5-A 案 W'-1)。

        承認確率に賭けて LLM のみ先行生成する分離設計。TTS は承認時に
        ``on_handraise_approved`` 内で同期実行する (= ``run_pipeline_tts_only``
        経由で graph._tts_node を再利用)。これにより却下/lapse 時の VOICEPEAK
        FIFO 投入を回避し、3 重発火時の VOICEPEAK 競合を緩和する。

        【WHY: cancel_aware_chunk_hook が無くなった】
        旧 bg_runner は run_pipeline (LLM + 内蔵 TTS) を走らせ、TTS chunk 投入時
        に cancel_event を観察して途中破棄していた。LLM-only 化により TTS 自体
        が走らず、chunk 投入経路が無くなった。代わりに cancel_event は run_loop
        の承認パス側で「BG が走り終わる前に lapse/denied → state pop 後は
        run_loop が bg_result を受け取らないので結果的に無視」という形で間接的
        に機能する (= dispatcher._bg_set_result で state None ガード済)。

        【WHY: trace_id を新規発番】
        BG LLM の trace_id は「BG が独自に走らせた LLM 推論」を識別する単位。
        承認時に run_loop は bg_result.trace_id を引き継いで TTS / playback /
        bubble の trace_id を統一する (= 既存の HandraiseBgResult.trace_id 引継ぎ
        ロジック維持)。

        Args:
            target_slug:        挙手中キャラ slug
            transcript_snapshot: 挙手判定時の TranscriptBuffer 文字列
            cancel_event:       却下/lapse 時に set される threading.Event
            on_complete:        BG LLM 完了時に呼ばれる callback。
                                ``on_complete(HandraiseBgResult)`` の形。

        Returns:
            BG LLM スレッド (daemon)。run_loop main thread はこれを join しない。
        """
        from .dispatcher import HandraiseBgResult
        from .pipeline import run_pipeline_llm_only

        bg_trace_id = _new_uuid()

        def _body() -> None:
            # Phase 0.5-D-d-6 (= 中間実走 8 回目 take 1-2 hang 調査用ログ追加):
            # bg_runner._body のライフサイクルを詳細追跡する。中間実走 9 回目以降
            # で hang 再現時に「どこで止まっているか」を切り分け可能にする。
            #
            # 観察ポイント:
            # - bg_runner._body 開始 (= thread spawn 直後)
            # - run_pipeline_llm_only 完了 (= mimi Agent.invoke 完了タイミング)
            # - on_complete callback 呼出 (= dispatcher._bg_set_result トリガー)
            #
            # Take 1-2 では「run_pipeline_llm_only 完了」ログが出ないはず (= mimi
            # Agent が hang)。出ていれば原因は別経路 (= dispatcher / 後続経路)。
            _bg_body_start = time.monotonic()
            _tts_cb_repr = (
                "set" if (on_tts_chunk_ready_ref and on_tts_chunk_ready_ref[0])
                else "None"
            )
            logger.info(
                "bg_runner._body 開始 [target=%s]: trace_id=%s thread_id=%d "
                "tts_cb=%s active_threads=%d",
                target_slug, bg_trace_id, threading.get_ident(),
                _tts_cb_repr, threading.active_count(),
            )
            try:
                # Phase 0.5-B-β-1 commit 3: ask_character の対話 TTS を BG LLM
                # 経路でも playback queue に届けるため、最新ターンの _on_tts_chunk
                # / _on_pose_ready を mutable list ref 経由で取得して渡す。
                #
                # 参照タイミング: bg_runner が起動するのは挙手検知時 (= ターン中、
                # _playback_queue が有効) なので ref[0] は通常 non-None。IDLE 中の
                # 挙手では ref[0] が None (= 前回ターン終了時に明示クリアされていない
                # 場合も既存実装の `if _playback_queue is not None:` で安全)。
                _tts_cb = (
                    on_tts_chunk_ready_ref[0]
                    if on_tts_chunk_ready_ref else None
                )
                _pose_cb = (
                    on_pose_ready_ref[0]
                    if on_pose_ready_ref else None
                )
                result = run_pipeline_llm_only(
                    transcript_snapshot if isinstance(transcript_snapshot, str)
                    else str(transcript_snapshot or ""),
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    trace_id=bg_trace_id,
                    speaker_hint=target_slug,
                    stream_context=stream_context,
                    # Phase 0.5-B-α: BG LLM 経路でも graph._generation_node が
                    # Thinking 反映 (= 挙手中キャラの thinking が HUD に出る)。
                    status_manager=status_manager,
                    # Phase 0.5-B-β-1 commit 3: ask_character の対話 TTS を BG LLM
                    # 経路でも playback queue に届けるための callback。
                    # graph._tts_node は LLM-only graph 不在のため呼ばれず、案 W'-1
                    # の最終応答 TTS suppress は構造的に維持される。
                    on_tts_chunk_ready=_tts_cb,
                    on_pose_ready=_pose_cb,
                )
                # Phase 0.5-D-d-6: hang 調査用ログ。本ログが出ない場合 mimi Agent
                # (= run_pipeline_llm_only 内の Agent.invoke) で hang していることを
                # 示す重要なシグナル。
                _llm_only_latency_ms = int(
                    (time.monotonic() - _bg_body_start) * 1000
                )
                logger.info(
                    "bg_runner._body run_pipeline_llm_only 完了 [target=%s]: "
                    "trace_id=%s latency_ms=%d active_threads=%d",
                    target_slug, bg_trace_id, _llm_only_latency_ms,
                    threading.active_count(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "BG LLM 失敗 [character=%s]: trace_id=%s err=%s",
                    target_slug, bg_trace_id, exc,
                )
                # Phase 0.5-D-d-6: 失敗パスでも on_complete callback 呼出を明記
                logger.info(
                    "bg_runner._body on_complete callback 呼出 [target=%s]: "
                    "result=None (失敗) trace_id=%s",
                    target_slug, bg_trace_id,
                )
                on_complete(HandraiseBgResult(
                    chunks=[], result=None, trace_id=bg_trace_id,
                ))
                return

            if cancel_event.is_set():
                # 完了直後 cancel 観察ログ。state は既に pop 済 → on_handraise_approved
                # に乗らないので結果は無視されるが、調査のためログだけ残す。
                logger.info(
                    "BG LLM 完了後に cancel 観察 [character=%s]: trace_id=%s",
                    target_slug, bg_trace_id,
                )

            # Phase 0.5-D-d-6: 正常完了時の on_complete callback 呼出を明記。
            # dispatcher._bg_set_result が呼ばれて bg_completed.set される起点。
            logger.info(
                "bg_runner._body on_complete callback 呼出 [target=%s]: "
                "result=ok trace_id=%s",
                target_slug, bg_trace_id,
            )
            on_complete(HandraiseBgResult(
                # WHY: chunks は LLM-only モードでは常に空 (= TTS が走らないため)。
                # 承認時に on_handraise_approved が run_pipeline_tts_only を呼び、
                # その中で chunks を生成する。
                chunks=[],
                result=result,
                trace_id=bg_trace_id,
            ))

        thread = threading.Thread(
            target=_body,
            name=f"BG-LLM-only-{target_slug}",
            daemon=True,
        )
        thread.start()
        return thread

    def on_handraise_started(slug, phrase_path, se_pending) -> None:
        """挙手開始通知。IDLE 中は即再生、RESPONDING 中は保留 (se_pending=True)。"""
        if se_pending:
            # RESPONDING 中: on_pipeline_complete 内で release callback 発火を待つ
            logger.info(
                "handraise wav 保留 (se_pending=True、応答完了後に再生): %s", slug,
            )
            return
        _spawn_handraise_phrase_playback(slug, phrase_path, padding_sec=0.0)

    def on_handraise_phrase_pending_release(slug, phrase_path) -> None:
        """RESPONDING → IDLE 遷移時、保留していた wav をリリース再生。"""
        padding = _get_handraise_responding_padding_sec()
        _spawn_handraise_phrase_playback(slug, phrase_path, padding_sec=padding)

    def on_handraise_approved(slug, bg_result, transcript_snapshot, trace_id) -> None:
        """承認時の応答開始 (Phase 0.5-A 案 W'-1: TTS-only graph で LLM 結果再利用)。

        bg_result.result が ready なら ``run_pipeline_tts_only(bg_result.result)``
        を別 daemon thread で呼び、生成された chunks を ``_spawn_handraise_response_playback``
        で再生する。bg_result が None / result が None / llm.final.text が空 なら
        ``_approved_synthesize_fallback`` で同期再生成にフォールバック (LLM 失敗
        時の救済として残す)。

        【WHY: daemon thread 化】
        dispatcher.on_approval_granted は本 callback を Lock 解放後に同期呼び出し
        する。この callback で同期 TTS (数秒〜十数秒) を実行すると dispatcher の
        callback 戻り遅延が長くなり、wait_for_next_event の通知タイミングや次の
        on_segment_added 受領 (= 録音継続スレッド) に遅延を波及させる懸念がある。
        daemon thread に切り出して即座に return するのが安全。

        【WHY: fallback パスを残す】
        LLM 失敗 / result=None / llm_text 空 / TTS 失敗 / chunks 空 のケースで
        ``_approved_synthesize_fallback`` (run_pipeline 同期再実行) で救済する。
        BG LLM-only に分離した結果、bg_result が高確率で ready になるが、API
        呼出失敗 / Gemini 異常応答 / 例外などの稀ケースで救済が必要。
        """
        from .pipeline import run_pipeline_tts_only

        # bg_result / result が無効ならフォールバック (LLM 失敗時の救済)
        if bg_result is None or bg_result.result is None:
            # Phase 0.5-D-d-2: fallback 進入の状態を詳細ログ。
            # progressing 経路で 30s 待ったが間に合わなかったケース、bg_runner 例外、
            # その他 (= LLM API エラー) を区別できるようにする (= ログから根本原因を
            # 特定する材料を増やす、配信事故時の調査効率向上)。
            bg_result_state = (
                "bg_result=None (BG 未起動 or 30s timeout)" if bg_result is None
                else "result=None (BG LLM 失敗 or API エラー)"
            )
            logger.info(
                "挙手承認 [character=%s]: %s → fallback 同期再生成 (真の救済パス)",
                slug, bg_result_state,
            )
            threading.Thread(
                target=lambda: _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                    status_manager=status_manager,
                ),
                name=f"approved-fallback-{slug}",
                daemon=True,
            ).start()
            return

        # bg_trace_id を引き継ぐ (handraise 単位の trace_id 一貫性)
        bg_trace_id = bg_result.trace_id or trace_id or _new_uuid()

        # answering bubble の text を pipeline_result.events から抽出
        # WHY (バグ 4 修正、実走 logs/runs/run_loop_20260508_184930.log で発覚):
        # llm.final.text は Pydantic JSON 文字列 ({"response":"...","emotion":{...},
        # "speed":...,"pose":"..."}) なので、bubble.text に生 JSON が表示される
        # のを防ぐため response 部分のみを取り出す。通常応答パスでは graph._tts_node
        # 内で _parse_voicepeak_json により処理されるが、案 W'-1 で run_loop が
        # bubble.update を発行する経路では、ここで明示的に parse する必要がある。
        from .tts import _parse_voicepeak_json
        answering_text = ""
        try:
            for ev in bg_result.result.events:
                if ev.get("type") == "llm.final":
                    llm_text_raw = ev.get("payload", {}).get("text", "")
                    if llm_text_raw:
                        # _parse_voicepeak_json は parse 失敗時 (= JSON でない /
                        # response key 無し) でも (text, None, None, None) で
                        # 元テキストをそのまま返す → fallback ロジック不要
                        answering_text, _, _, _ = _parse_voicepeak_json(llm_text_raw)
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("answering text 抽出失敗 (空文字で続行): %s", exc)

        if not answering_text:
            # llm.final が無い or text が空 → fallback パスで救済
            logger.warning(
                "挙手承認 [character=%s]: llm.final.text 空 → fallback パス",
                slug,
            )
            threading.Thread(
                target=lambda: _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                    status_manager=status_manager,
                ),
                name=f"approved-fallback-{slug}",
                daemon=True,
            ).start()
            return

        # Phase 0.5-D-3 follow-up: 事前 bubble.update("answering") publish 廃止。
        #
        # 【WHY: 物理再生時に発火する設計に移行】
        # 旧設計では approval 直後 (= TTS 合成 + chunks 投入の 21 秒前) に answering
        # bubble を publish していた。中間実走 3 回目 (logs/runs/run_loop_20260509_184653.log)
        # で「approval 直後に〆セリフ text が bubble 表示」(= 物理再生は導入セリフから
        # 始まるのに) フライング UX 不具合として観察された。
        #
        # 【新設計】
        # tts_only chunks (= caller の〆セリフ) の first chunk に `_pre_play_bubble`
        # inline metadata を埋め込み、物理再生開始時 (= playback worker が pop した
        # 直後) に発火する。これにより「視聴者が caller の〆セリフ text を見る瞬間」
        # と「実際に caller が〆セリフを話し始める瞬間」が同期する (= D-3-c で
        # bridge filler に対して同様に行った設計を caller の〆 chunks にも適用)。
        #
        # bg_chunks (= 導入セリフ + bridge + 本応答) は ask_character 内の inline
        # metadata 経路で発火される (= D-3-b/D-3-c)。

        # TTS-only graph で TTS 実行 + chunks 蓄積 → playback worker (daemon thread)
        def _tts_and_play() -> None:
            # Phase 0.5-D-3 follow-up 2: streaming spawn 設計に変更。
            #
            # 【WHY: 旧設計の問題】
            # 旧設計 (D-2/D-3 まで) は run_pipeline_tts_only を同期 blocking で呼び、
            # 全 tts_only chunks 蓄積完了を待ってから _spawn_handraise_response_playback
            # を起動していた。実走テスト 2026-05-09 logs/runs/run_loop_20260509_190927.log
            # で「approval 後 18.5 秒の空白」(= mimi 〆セリフ chunk 1 の VOICEPEAK 合成
            # 18 秒 + chunk 2 合成 18 秒 = 直列 36 秒) 中、視聴者は何も聞こえない
            # フライング待ちが発生していた。
            #
            # 【新設計】
            # 1. bg_chunks (= 既に合成済の導入セリフ + bridge filler + 本応答 = 通常 4 件)
            #    を _spawn_handraise_response_playback_streaming で即時 spawn → 物理再生開始
            # 2. run_pipeline_tts_only を並行で同期実行、各 tts_only chunk が合成完了次第
            #    on_chunk_ready callback で queue に動的追加
            # 3. worker は bg_chunks 再生中 (= 約 30 秒) に tts_only chunks が queue に
            #    流れてくるので、バッファ切れなしで連続再生可能
            # 4. run_pipeline_tts_only 完了後、queue.put(None) で worker 終了通知
            #
            # 視聴者体感: 「approval → 即「ルカ、その問いは…」(= 導入セリフ) → 各キャラ
            # 応答 → 自然に〆セリフ」のシームレスな対話が、合成時間に関わらず実現する。
            # Phase 0.5-D-d-7 (= 中間実走 9 回目 take 9-B 修正、再生開始遅延解消):
            # 旧 wait_deferred_bg_tts_complete + _drain_bg_chunks の経路は、sakura TTS
            # 全 chunks 合成完了 (= 27 秒) まで待ってから streaming spawn 起動するため、
            # mimi 導入セリフが既に合成済でも再生開始が 27 秒遅延する問題があった
            # (= run_loop_20260509_234753.log で観察)。
            #
            # 新設計 (3 step):
            # 1. 初回 _drain_bg_chunks (= talking_metadata 構築用 + initial_chunks 取得)
            # 2. _spawn_handraise_response_playback_streaming で起動 (= queue 取得)
            # 3. drain_and_register_streaming_queue (= race 漏れ回収 + queue ref register)
            #
            # これにより、合成完了次第 chunks が streaming queue に投入される (= mimi
            # 導入セリフは ~8 秒で再生開始 vs 旧 35 秒、約 27 秒短縮)。
            from .mcp_servers.ask_character import (
                _drain_bg_chunks,
                drain_and_register_streaming_queue,
            )

            # Step 1: 初回 drain (= talking_metadata 構築用、現時点で合成済の chunks)
            bg_chunks = _drain_bg_chunks(session_id_root)
            if bg_chunks:
                logger.info(
                    "挙手承認 BG buffer 初回 drain [character=%s]: chunks=%d (= 即時投入用)",
                    slug, len(bg_chunks),
                )

            # Talking metadata 構築 (= caller の HUD 表示用)。
            # tts_only chunks の first pose は合成完了前に取得不可なので、bg_chunks
            # の最初 (= 通常 caller の導入セリフ first chunk) の pose を一旦使う。
            # bg_chunks 空時は pose=None で起動 (= text のみの metadata)。
            #
            # Phase 0.5-D-d-5 (= 中間実走 8 回目 take 2-2 修正):
            # talking_metadata.text は bg_chunks[0] の text (= caller 導入セリフ) を
            # 使う。〆セリフ text への切り替えは tts_only first chunk の inline metadata
            # (= _pre_play_status、D-d-3 で実装済) で物理再生時に発火する。
            #
            # 【WHY: 旧設計の不具合】
            # 旧設計では talking_metadata.text = answering_text (= 〆セリフ) で
            # streaming spawn 起動時に publish していたが、bg_chunks の最初は caller
            # 導入セリフから再生開始するため、HUD で 〆セリフ text が表示されながら
            # 音声は導入セリフが流れる UX 不整合 (= 中間実走 8 回目 take 2-2、
            # logs/runs/run_loop_20260509_230651.log で観察)。
            #
            # 【bg_chunks 空時のフォールバック】
            # bg_chunks 空時 / first chunk が caller 以外の場合は caller_initial_text=""
            # → answering_text (= 〆セリフ) にフォールバック (= 旧挙動と整合)。これは
            # 実装漏れ等で bg_chunks=0 になった場合 (= 中間実走 7 回目で観察) の救済。
            caller_initial_pose: "str | None" = None
            caller_initial_text = ""
            if bg_chunks:
                # bg_chunks の最初は caller (= slug) の導入セリフ first chunk
                first_bg = bg_chunks[0]
                if first_bg.get("character") == slug:
                    caller_initial_pose = first_bg.get("pose")
                    caller_initial_text = first_bg.get("text", "")
            display_text = caller_initial_text or answering_text
            talking_metadata = _build_talking_metadata(
                slug, display_text, caller_initial_pose,
            )

            # Phase 0.5-D-3 follow-up 2: streaming spawn (= bg_chunks 即時再生開始)
            logger.info(
                "挙手承認 streaming spawn [character=%s]: bg_chunks=%d (= 即時再生開始)",
                slug, len(bg_chunks),
            )
            _streaming_thread, _streaming_queue = (
                _spawn_handraise_response_playback_streaming(
                    slug,
                    bg_chunks,
                    bg_trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    status_manager=status_manager,
                    talking_metadata=talking_metadata if talking_metadata else None,
                )
            )

            # Phase 0.5-D-d-7 Step 3: race 漏れ回収 + queue ref register
            # streaming spawn 起動 (Step 2) と本 step の間に bg_tts daemon thread が
            # `_append_bg_chunk` を呼んでいた場合、その chunks を回収して queue に
            # 投入する (= 通常 0 件、race 時のみ存在)。同時に queue ref を register
            # することで、以降の `_append_bg_chunk` は buffer 蓄積 + queue.put 両方を
            # 実行する経路に切り替わる (= 合成完了次第 chunks が再生される、これで
            # take 9-B の 27 秒再生開始遅延を構造的に解消)。
            late_chunks = drain_and_register_streaming_queue(
                session_id_root, _streaming_queue,
            )
            if late_chunks:
                logger.info(
                    "挙手承認 BG buffer 再 drain [character=%s]: chunks=%d (= race 漏れ回収)",
                    slug, len(late_chunks),
                )

            # tts_only chunks の動的追加用 closure 変数
            _first_tts_only_chunk_seen = [False]
            _tts_only_count = [0]

            def on_chunk(url, chunk_text, is_last, character, pose=None):
                """on_tts_chunk_ready 用、tts_only chunks を queue に動的追加。

                Phase 0.5-D-3 follow-up: first tts_only chunk に caller answering bubble
                を inline metadata で埋込。物理再生開始時に worker が発火する経路。
                """
                chunk: dict = {
                    "url": url,
                    "text": chunk_text,
                    "is_last": is_last,
                    "character": character,
                }
                if pose is not None:
                    chunk["pose"] = pose

                # first tts_only chunk のみに caller answering bubble inline 埋込
                if not _first_tts_only_chunk_seen[0]:
                    _first_tts_only_chunk_seen[0] = True
                    chunk["_pre_play_bubble"] = {
                        "slug": slug,
                        "step": "answering",
                        "text": answering_text,
                    }
                    # Phase 0.5-D-d-3 (= A4): caller の TALKING を物理再生開始時に発火する
                    # inline metadata。bg_chunks 経路 (= ask_character target chunks) と
                    # 同じ設計を caller 側にも適用 (= ask_character.py:1004-1010 の target
                    # chunks pattern を caller chunks に複製)。
                    #
                    # 【WHY: 物理再生時に発火する理由】
                    # caller の TALKING は streaming spawn 起動直前に talking_metadata で
                    # 1 回反映されている (= run_loop.py:957) が、これは bg_chunks 即時再生と
                    # 同期した「caller が話し始める瞬間」を視聴者に届ける。tts_only chunks の
                    # first (= caller の〆セリフ) が再生される瞬間に再度 TALKING を発火する
                    # ことで、caller の〆セリフ pose と text を最新値で HUD に反映する
                    # (= bg_chunks の first pose は導入セリフのもの、〆セリフは別 pose の可能性)。
                    #
                    # 【冪等性】set_status は同 status + 同 metadata なら no-op、status 一致でも
                    # metadata (= pose / text) が異なれば publish される (= 仕様通り、
                    # character_status.py:208-209)。pose=None + text 空文字列のケースでは
                    # inline metadata 自体を omit (= 既存の TALKING 反映を維持、HUD 不変)。
                    if pose is not None or answering_text:
                        chunk["_pre_play_status"] = {
                            "slug": slug,
                            "status": "TALKING",
                            "metadata": {
                                "pose": pose if pose else None,
                                "text": answering_text,
                            },
                        }

                # streaming queue に動的追加 (= worker が並行で再生)
                _streaming_queue.put(chunk)
                _tts_only_count[0] += 1

            # Phase 0.5-D-d-8 (= 中間実走 10 回目 take 2/3 修正、VOICEPEAK FIFO 順序
            # 乱れ解消): mimi 〆セリフ TTS の VOICEPEAK 投入前に sakura TTS 全 chunks
            # 合成完了を待つ。
            #
            # 【WHY: VOICEPEAK FIFO 順序乱れ問題】
            # D-d-7 で wait_deferred_bg_tts_complete を削除した結果、mimi 〆セリフ TTS
            # が sakura TTS の途中で並行投入され、VOICEPEAK FIFO worker の合成順 (=
            # 投入順) で交錯が発生:
            #   旧 (D-d-7): 投入順 = mimi 導入 → sakura 1 → sakura 2 → mimi 〆 1 →
            #               sakura 3 → mimi 〆 2 (= take 2/3 で「ミミ → さくら → ミミ
            #               → さくら → ミミ」の交錯を観察、logs/runs/run_loop_20260510_001209.log)
            #
            # VOICEPEAK は process global で 1 個直列のため、複数キャラの並行投入が
            # 順序乱れを引き起こす根本制約。
            #
            # 【新設計 (D-d-8) のバランス】
            # - mimi 導入セリフ + sakura chunks: 即時再生開始 (= D-d-7 の利点維持、
            #   bg_chunks の蓄積分は streaming queue 直投入で再生)
            # - mimi 〆セリフ chunks: sakura TTS 完了後に VOICEPEAK 投入 (= 順序保証、
            #   ミミ → さくら → ミミ の自然な対話順)
            # - mimi 〆セリフ再生開始は旧設計と同じタイミング (= sakura 完了後)
            from .mcp_servers.ask_character import wait_deferred_bg_tts_complete
            wait_deferred_bg_tts_complete(session_id_root)

            logger.info(
                "挙手承認 TTS 同期実行 開始 [character=%s]: trace_id=%s text_len=%d",
                slug, bg_trace_id, len(answering_text),
            )
            try:
                run_pipeline_tts_only(
                    bg_result.result,
                    on_tts_chunk_ready=on_chunk,
                    # WHY: pose は _tts_node 内で即時 set_pose に倒す (= on_pose_ready=None)。
                    # 専用 mini playback worker は pending pose dict を持たないため、
                    # chunk task に pose 埋め込む経路が無い。即時 set_pose で十分。
                    on_pose_ready=None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "挙手承認 TTS 同期実行失敗 [character=%s]: %s → fallback パス",
                    slug, exc,
                )
                # 例外時: worker 終了通知 (= bg_chunks 再生は継続済み or 完了済み)
                _streaming_queue.put(None)
                # bg_chunks 空なら fallback で救済、bg_chunks ありなら既に再生中で完結
                if not bg_chunks:
                    _approved_synthesize_fallback(
                        slug, transcript_snapshot, trace_id,
                        session_stream_id=session_stream_id,
                        session_id_root=session_id_root,
                        stream_context=stream_context,
                        status_manager=status_manager,
                    )
                return

            # tts_only 完了 → queue に None sentinel 投入で worker 終了通知
            # (= bg_chunks + tts_only chunks 全て再生完了後 worker break)
            _streaming_queue.put(None)

            # 全 chunks 空 (= bg_chunks 空 + tts_only 空) のみ fallback 必要
            # bg_chunks 4 件あれば再生される、tts_only 空でも導入 + bridge + 本応答は OK
            if not bg_chunks and _tts_only_count[0] == 0:
                logger.warning(
                    "挙手承認 TTS [character=%s]: 全 chunks 空 → fallback パス",
                    slug,
                )
                _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                    status_manager=status_manager,
                )
                return

            # ログメッセージは既存形式を維持 (= W'-3 ログ強化テストとの後方互換性)。
            # streaming 設計でも「同期実行 完了 → playback worker 起動」のフローを
            # ログ上は維持し、grep パターンでの追跡互換性を保つ。
            logger.info(
                "挙手承認 TTS 同期実行 完了 [character=%s]: bg=%d + tts_only=%d = total=%d (streaming)",
                slug, len(bg_chunks), _tts_only_count[0],
                len(bg_chunks) + _tts_only_count[0],
            )
            logger.info(
                "挙手承認 playback worker 起動 [character=%s] (streaming)",
                slug,
            )

        threading.Thread(
            target=_tts_and_play,
            name=f"approved-tts-{slug}",
            daemon=True,
        ).start()

    def on_handraise_close(slug: str, reason: str) -> None:
        """却下/lapse 時の close 通知 (Phase 0.5-B-β-2 commit 3)。

        dispatcher.on_approval_denied / on_lapse_timeout から発火される
        ``Dispatcher.on_handraise_close`` callback の本体。以下を実行することで
        案 A の音声漏れ (= 挙手中に流れた ask_character の対話 TTS) を最小化する:

        1. ``cancel_bg_tts(session_id)`` で ask_character の bg_tts daemon thread
           + 導入セリフ TTS + on_chunk 投入を阻止する。
        2. ``playback_queue_ref[0].put({"_drain": True})`` で playback queue 内の
           未処理 chunks を破棄する (= _run_playback_worker の drain 経路)。

        【WHY: 既に subprocess 中 / 再生中の chunks は止められない】
        VOICEPEAK 合成中の subprocess.run と play_audio_fn block 中の chunk は
        OS レベルで止められない。実用上、却下後の音声漏れは「再生中の 1 chunk +
        合成済キュー先頭の 1 chunk」程度まで縮小する (= 通常応答 TTS 数十秒分が
        漏れる Phase 0.5-A 状態と比べると劇的に改善)。

        【WHY: ask_character の cancel と queue drain を両方走らせる】
        cancel_bg_tts だけでは「合成済で playback queue に乗っている chunks」が
        残る。queue drain だけでは「これから合成される bg_tts thread が打ち消されない」。
        両方を走らせることで、両系統の音声漏れ source を阻止できる。

        Args:
            slug:   close 対象 (= 却下/lapse された) target キャラ slug
            reason: ``"denied"`` (= ルカが却下) or ``"lapsed"`` (= タイムアウト)
        """
        from .mcp_servers.ask_character import cancel_bg_tts

        n = cancel_bg_tts(session_id_root)

        # playback queue に drain task 投入。late binding で最新の queue 参照を
        # 取得 (= ターン外なら ref[0] が None で no-op、ターン中なら queue にアクセス)。
        drained = False
        if playback_queue_ref is not None:
            current_q = playback_queue_ref[0]
            if current_q is not None:
                try:
                    current_q.put({"_drain": True})
                    drained = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "挙手 close drain task 投入失敗 [character=%s reason=%s]: %s",
                        slug, reason, exc,
                    )

        logger.info(
            "挙手 close [character=%s reason=%s]: bg_tts cancel=%d drain=%s",
            slug, reason, n, drained,
        )

    def on_approval_progressing(slug: str) -> None:
        """承認時 BG LLM 未完了 → bridge filler 即時再生 (Phase 0.5-D-d-2)。

        【WHY: PROGRESSING 中の bridge filler 即時再生】
        実走テスト 6 回目 (logs/runs/run_loop_20260509_195508.log) で、ルカが
        「ミミ様、どうぞ」承認時に BG LLM 未完了 → fallback で 41 秒沈黙の問題を観察。
        bg_completed.wait(timeout=30s) 中の沈黙を target キャラの bridge filler
        (4-8 秒程度の繋ぎセリフ) で埋めることで:
          - 承認音声 → 即「ええと…」(= 自然な「呼ばれた感」のレスポンス)
          - その間 BG LLM 完了待ち、完了後 streaming spawn で本応答開始
          無音時間が 30 秒 → 4-8 秒に短縮 (= 配信事故レベルからの脱却)。

        ★UX 違和感のトレードオフ: 「approval 前に target が考え始めた?」と視聴者が
        感じる可能性あり。ただし 30 秒沈黙の配信事故レベルリスクと比較すると許容範囲。

        ★R4 (= 通常応答 TTS 再生中の 3 重音声重なり防止) は dispatcher 側で
        ``state.se_pending=True`` のキャラには callback を発火しないガードで実現済。
        本関数は呼ばれた時点で「IDLE 中の挙手 progressing」として安全。

        Args:
            slug: 承認された target キャラ slug (= bridge filler を再生する対象)
        """
        try:
            from .filler import select_filler_path
            bridge_path, _ = select_filler_path(slug, "bridge")
            if bridge_path is None:
                logger.debug(
                    "on_approval_progressing: bridge filler なし [character=%s]",
                    slug,
                )
                return
            # 直接 _spawn_handraise_phrase_playback で再生 (= 専用 mini playback worker、
            # padding=0 で即時再生開始)。承認音声と TTS の重なりは許容 (= 案 W'-1
            # 設計と整合、Lock 不要)。
            _spawn_handraise_phrase_playback(slug, bridge_path, padding_sec=0.0)
            logger.info(
                "on_approval_progressing bridge filler 即時再生 [character=%s]: %s",
                slug, bridge_path.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "on_approval_progressing bridge filler 起動失敗 [character=%s]: %s",
                slug, exc,
            )

    return (
        bg_runner,
        on_handraise_started,
        on_handraise_phrase_pending_release,
        on_handraise_approved,
        on_handraise_close,
        on_approval_progressing,
    )


# ─── Phase 0.5-F-2: 案 R 用 callback factory (raisehand → callout 統合) ────


def _create_replay_runner_and_callbacks(
    *,
    session_stream_id: str,
    session_id_root: str,
    stream_context: "str | None" = None,
    status_manager: "CharacterStatusManager | None" = None,
    dispatcher_ref: "list | None" = None,
):
    """Phase 0.5-F-2: 案 R 用の 4 つの callback を生成する factory。

    ``_create_handraise_runner_and_callbacks`` の置き換え候補。中間実走 13
    シナリオ γ (= raisehand 承認後の沈黙 5 分以上、配信事故レベル) への根本対処。

    【WHY: 案 R = raisehand を callout 経路に統合】
    Phase 0.5-A 案 W'-1 の「LLM 先行計算 (= bg_runner)」設計が、多段階 ask_character
    で破綻 (= 60s 超 latency で fallback 起動 → TTS 4 系統廃棄経路)。callout 経路
    では同じワークロードが廃棄経路を持たず正常動作するため、raisehand 承認時に
    callout 経路と同じ ``run_pipeline`` (= LLM + TTS 一体) を起動する設計に統合。

    【F-2 では wiring しない】
    本 factory は **opt-in 用** に併設するだけで、F-2 commit では既存
    ``_create_handraise_runner_and_callbacks`` を使い続ける。F-3 commit で
    ``run_loop`` 内の ``Dispatcher(...)`` 引数を切替えると初めて案 R 経路が
    起動する。これにより partial revert の柔軟性を最大化 (= F-3 wiring 戻しで
    完全復元可能)。

    【返り値の signature 設計】
    既存 factory の 6 callable から:
      - ``bg_runner`` 削除 (= 案 R で LLM 先行計算なし、Dispatcher に None 渡す)
      - ``on_handraise_approved`` 削除 (= ``on_approval_replay`` で代替)
      - ``on_approval_progressing`` 削除 (= R-1-b: 通常応答 filler に統合、
        ``run_pipeline`` 経由で filler が自然に起動するため不要)
      - ``on_handraise_started`` / ``on_handraise_phrase_pending_release`` 維持
      - ``on_handraise_close`` 維持 (ただし内部は no-op、bg_runner 起動なしのため)
      - ``on_approval_replay`` 新規追加 (= F-1 で導入した dispatcher 側 callback)

    Args:
        session_stream_id: 当該セッションの stream_id (= 既存 factory と互換)
        session_id_root:   当該セッションの session_id (= 同上)
        stream_context:    配信文脈 markdown (案 R では未使用、互換のため受領)
        status_manager:    ステータス管理 (案 R では未使用、互換のため受領)
        dispatcher_ref:    ``list[Dispatcher | None]`` の mutable list (= late binding)。
                           ``on_approval_replay`` 内で ``ref[0].on_wake_detected(...)``
                           で queue に inject する用。F-3 wiring 時に
                           ``dispatcher_ref[0] = dispatcher`` を設定する。

    Returns:
        (on_handraise_started, on_handraise_phrase_pending_release,
         on_handraise_close, on_approval_replay) の 4-tuple。
    """

    def on_handraise_started(
        slug: str, phrase_path: "Path | None", se_pending: bool,
    ) -> None:
        """挙手検知時の handraise wav 即再生 (= 既存と同等の演出)。

        IDLE 中なら即再生、RESPONDING 中 (= se_pending=True) は IDLE 復帰時の
        ``on_handraise_phrase_pending_release`` を待つ。
        """
        if se_pending:
            # 通常応答中なので handraise wav は保留 → on_pipeline_complete 後に release
            logger.info(
                "挙手検知 [character=%s]: RESPONDING 中のため wav 保留 (= phrase_pending_release で再生)",
                slug,
            )
            return
        if phrase_path is None:
            logger.debug("挙手検知 [character=%s]: phrase wav なし (= filler 未生成)", slug)
            return
        try:
            _spawn_handraise_phrase_playback(slug, phrase_path)
            logger.info(
                "挙手検知 handraise wav 即再生 [character=%s]: %s",
                slug, phrase_path.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "挙手検知 handraise wav 起動失敗 [character=%s]: %s",
                slug, exc,
            )

    def on_handraise_phrase_pending_release(
        slug: str, phrase_path: "Path | None",
    ) -> None:
        """RESPONDING → IDLE 遷移時の保留 wav リリース (= 既存と同等)。"""
        if phrase_path is None:
            return
        try:
            _spawn_handraise_phrase_playback(slug, phrase_path)
            logger.info(
                "挙手 phrase pending release [character=%s]: %s",
                slug, phrase_path.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "挙手 phrase pending release 起動失敗 [character=%s]: %s",
                slug, exc,
            )

    def on_handraise_close(slug: str, reason: str) -> None:
        """却下/lapse 時の close 通知 (Phase 0.5-F-2、no-op 化)。

        【WHY: no-op】
        案 R では bg_runner._body が起動しない (= LLM 先行計算なし) ため、
        cancel 対象の bg_tts thread / buffer が **存在しない**。よって
        ``cancel_bg_tts(session_id)`` 呼出は無効、playback queue drain も
        通常応答経路の TTS を誤って drain する害があるため呼ばない。

        ログのみ残して却下/lapse の事実を記録する (= 配信事故再発時のトレース用)。
        """
        logger.info(
            "挙手 close [character=%s reason=%s]: 案 R 経路では no-op "
            "(bg_runner 起動なしのため cancel 不要)",
            slug, reason,
        )

    def on_approval_replay(slug: str, transcript_snapshot) -> None:
        """承認時に callout 経路に inject する (Phase 0.5-F-2 案 R の中核)。

        ``dispatcher.on_wake_detected(WakeWordResult(transcript=...))`` で wake_event
        queue にエンキューし、run_loop の ``wait_for_next_event`` が次ターンとして
        取り出す。次ターンでは callout 経路と同じ ``run_pipeline`` (= LLM + TTS
        一体) が走り、視聴者に音声が届く。

        【WHY: queue 経由の inject】
        - 通常応答ターン中の挙手承認 → IDLE 復帰時に自然に次ターンとして処理される
        - VOICEPEAK FIFO 順序が保たれる (= 1 ターン内で 1 直列)
        - callout 経路と完全に同じコードパスを通る (= 廃棄経路の構造的消滅)

        Args:
            slug:                承認された target キャラ slug
            transcript_snapshot: 挙手判定時の TranscriptBuffer 文字列
                                 (= 通常 callout の wake_word 転写と同じ役割)
        """
        if dispatcher_ref is None or dispatcher_ref[0] is None:
            logger.warning(
                "on_approval_replay: dispatcher_ref 未設定 [character=%s] "
                "(= F-3 wiring 未完了の可能性、no-op で skip)",
                slug,
            )
            return
        # transcript_snapshot は文字列前提 (= 通常 callout 経路の transcript と同型)。
        # None / 空文字なら空文字でキューに inject (= run_pipeline は空文字でも処理可能)。
        transcript_text = str(transcript_snapshot) if transcript_snapshot else ""
        wake_result = WakeWordResult(
            keyword="<approval>",  # 専用識別子 (= 通常 wake と区別、ログで判別容易)
            character_slug=slug,
            keyword_index=-1,
            transcript=transcript_text,
        )
        try:
            dispatcher_ref[0].on_wake_detected(wake_result)
            logger.info(
                "on_approval_replay: callout 経路 inject 完了 "
                "[character=%s] transcript_len=%d",
                slug, len(transcript_text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "on_approval_replay: on_wake_detected inject 失敗 [character=%s]: %s",
                slug, exc,
            )

    return (
        on_handraise_started,
        on_handraise_phrase_pending_release,
        on_handraise_close,
        on_approval_replay,
    )


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
    on_last_chunk_played=None,
    status_manager: "CharacterStatusManager | None" = None,
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
       - speaking を 1 回以上 publish 済みなら:
         - on_last_chunk_played() callback を発火 (Phase 0.5-A 8-11、done_delay の前)
         - done_delay_seconds 秒待って publish_bubble_fn(last_character, "done", "") を呼ぶ
         - break
       - 未 publish なら即 break（bubble は V2 安全弁で消える）

    Sprint Axis D Block 1: OBS セリフテロップ表示
    Phase 3: 立ち絵切替を再生時点に移動 (TTS 生成時点ではなく)
    Phase 0.5-A 8-11: on_last_chunk_played callback で挙手 wav 早期再生を実現

    Args:
        q:                   queue.Queue[dict | None] — task dict または None sentinel
            task dict: {"url": str, "text": str, "is_last": bool, "character": str,
                        "pose": str (optional)}
        publish_bubble_fn:   (character: str, step: str, text: str) -> None
        play_audio_fn:       (url: str) -> None  — ブロッキング再生
        cleanup_audio_fn:    (url: str) -> None  — 再生後のファイル削除等
        set_pose_fn:         (character: str, pose: str) -> None  — OBS 立ち絵切替 (optional)
        done_delay_seconds:  最終チャンク再生完了から done publish までの待機秒数
        on_last_chunk_played: () -> None  — Phase 0.5-A 8-11: 最終 chunk 物理再生完了直後
                              (done_delay sleep の前) に発火する callback。
                              dispatcher.flush_pending_handraise_releases を渡すことで
                              RESPONDING 中保留された挙手 wav を 5 秒早く release する。
                              None (default) なら従来挙動 (= done_delay → done bubble の直列)。
        status_manager:       Phase 0.5-B-β-3 commit 2 で追加。is_last=True chunk の
                              **物理再生完了時** に ``set_status(character, READY)`` を
                              呼ぶ。HUD UX で「発話完了 = READY」を視覚化するため
                              (= シナリオ 2 で観察した「発話途中で灰色化する」不具合の
                              修正、ask_character target キャラ用に追加)。冪等性
                              により、通常応答 caller の最終応答経路では既存の
                              ``_bg_cleanup_pipeline`` 経由 READY 反映と二重発火するが
                              CharacterStatusManager.set_status の同 status no-op で
                              害なし。None (default) なら反映スキップ (= 後方互換)。
    """
    speaking_published = False
    last_character: str | None = None
    current_pose_character: str | None = None
    current_pose: str | None = None
    # Phase 0.5-D-3 follow-up 3: 話者切替時に間を入れるための追跡変数。
    # last_character は speaking publish 時のみ更新 (= text 非空 chunks のみ、done bubble 用)
    # だが、本変数は物理再生完了時に毎回更新する (= text="" の bridge filler 含む)。
    # これにより「caller chunk → bridge filler → 本応答 chunks」で正しくキャラ切替
    # 判定ができる。
    prev_played_character: str | None = None
    while True:
        task = q.get()
        # Phase 0.5-B-β-2 commit 3: drain task の処理 (= 却下/lapse 時の残 chunks 破棄)。
        # run_loop の on_handraise_close callback が q.put({"_drain": True}) で
        # 投入する。q.queue (= 内部 deque) を q.mutex 保護下で iterate し、None
        # sentinel と _drain task 自体は維持して、通常 task のみ破棄する。
        # 「現在再生中の chunk」(= play_audio_fn block 中) は止められないが、
        # まだ playback worker が pop していない queue 内の未再生 chunks は本処理で
        # 破棄できる。drain 後 continue で通常処理続行 (= ターン全体は続ける、
        # 後続の None sentinel まで待つ)。
        if isinstance(task, dict) and task.get("_drain"):
            discarded = 0
            with q.mutex:
                # 内部 deque を直接操作。queue.Queue の標準的な「peek/iter」イディオム。
                # collections.deque と異なり queue.Queue 自体は iter API を提供しない
                # ため、mutex 保護下で popleft → 維持 list に振り分けて再構築する。
                kept: list = []
                while q.queue:
                    item = q.queue.popleft()
                    if item is None or (
                        isinstance(item, dict) and item.get("_drain")
                    ):
                        # sentinel と他の drain task は維持 (= 後続の cleanup 経路で消費)
                        kept.append(item)
                    else:
                        discarded += 1
                q.queue.extend(kept)
            logger.info(
                "playback worker drain: discarded=%d remaining=%d",
                discarded, len(q.queue),
            )
            continue
        if task is None:
            # 最終チャンク再生完了後: speaking を発行していれば done_delay 秒待って done publish
            if speaking_published and last_character is not None:
                # Phase 0.5-A 8-11: done_delay の "前" に flush callback を発火する。
                # 通常応答の最終 chunk 物理再生完了直後に挙手 wav を release することで、
                # 視聴者体感を「応答終了 → 6.5 秒空 → 挙手 wav」から「応答終了 → 1.5 秒
                # → 挙手 wav」へ短縮する (実走 C-1 で観察した遅延の解消)。
                # done bubble の発行タイミング (5 秒後) は維持し、V2 HUD UX への影響なし。
                if on_last_chunk_played is not None:
                    try:
                        on_last_chunk_played()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "_run_playback_worker on_last_chunk_played failed: %s", exc,
                        )
                time.sleep(done_delay_seconds)
                publish_bubble_fn(last_character, "done", "")
            break

        # Phase 0.5-D-2: defer 経路 chunk の物理再生直前 metadata 発火。
        # ask_character の defer モード _wrapped_on_chunk_ready が chunk dict に
        # 埋め込んだ `_pre_play_status` / `_pre_play_bubble` を、worker が pop した
        # 直後 (= 物理再生開始の直前) に発火する。
        #
        # 【WHY: chunk 投入時ではなく物理再生時に発火する理由】
        # 案 C リファクタで挙手中の対話 TTS chunks は buffer に蓄積されてから承認時
        # に専用 mini playback worker で再生される。「buffer 投入時」に TALKING /
        # answering bubble を発火すると、視聴者から「承認前なのに target が話して
        # いる」ように見える UX 不具合が起こる。物理再生開始時に発火させることで
        # 「実際に音声が流れ始める瞬間」と HUD 表示が同期する。
        #
        # 通常応答経路 (= defer=False) の chunks にはこのキーが含まれないため
        # (= ask_character.py の defer 分岐でのみ埋め込み)、本 dispatch の影響なし
        # = 既存挙動完全維持。
        if isinstance(task, dict):
            pre_play_status = task.get("_pre_play_status")
            if pre_play_status and status_manager is not None:
                try:
                    status_manager.set_status(
                        pre_play_status["slug"],
                        CharacterStatus[pre_play_status["status"]],
                        metadata=pre_play_status.get("metadata"),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "playback worker pre_play_status 反映失敗 [character=%s]: %s",
                        pre_play_status.get("slug"), exc,
                    )
            pre_play_bubble = task.get("_pre_play_bubble")
            if pre_play_bubble:
                try:
                    publish_bubble_fn(
                        pre_play_bubble["slug"],
                        pre_play_bubble["step"],
                        pre_play_bubble.get("text", ""),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "playback worker pre_play_bubble 発行失敗 [character=%s]: %s",
                        pre_play_bubble.get("slug"), exc,
                    )

        # Phase 0.5-D-3 follow-up 3: 話者切替時に 0.5 秒の間を挿入。
        #
        # 【WHY】
        # 中間実走 5 回目 (logs/runs/run_loop_20260509_194304.log) で「フローが
        # 滑らかすぎて畳み掛けられている感じ」が観察された。streaming spawn 設計
        # (= follow-up 2) で「caller 導入セリフ → bridge filler → target 応答 →
        # caller 〆」の連続再生が実現したが、話者切替境界に間がないため、視聴者の
        # 認知的に「会話のターン交代」が知覚しにくくなっていた。0.5 秒の間は人間の
        # 自然な会話の間 (= 互いの発言を受け止める無音時間) に近い値で、聞きやすさを
        # 向上させる。
        #
        # 初回 chunk (= prev_played_character is None) では sleep しない (=
        # approval → 即時音声再生の効果を維持、follow-up 2 設計の趣旨を壊さない)。
        # 同一キャラ連続 chunks (= caller の長い応答を分割した chunk 2, 3, ...) では
        # 間を入れない (= 同じキャラが続けて話している自然な流れを保つ)。
        character = task["character"]
        if prev_played_character is not None and character != prev_played_character:
            time.sleep(0.5)

        # Phase 3: 再生直前にキャラクターが変わったら立ち絵切替 + HUD 通知
        # task["pose"] の値:
        #   - 文字列 (e.g., "special_doya")  → その pose に切替 (chunk 1 など、新規予約時)
        #   - None                           → 切替しない (chunk 2 以降。現在の pose を維持)
        #
        # 「キャラが変わった + pose=None」 のみ neutral にフォールバック。
        # これがないと、chunk 1 で special_doya に切替えた後の chunk 2/3 で neutral
        # に逆戻りしてしまう (chunk 1 だけが pose 値を持ち、後続は None になるため)。
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

        # Phase 0.5-D-3 follow-up 3: 物理再生完了後に prev_played_character を更新。
        # text 有無 (= bridge filler 含む) に関わらず毎回更新することで、次の chunk
        # で正しくキャラ切替判定 (= 上の time.sleep(0.5)) ができる。
        prev_played_character = character

        # Phase 0.5-B-β-3 commit 2: is_last=True chunk の **物理再生完了時** に
        # status_manager で READY 反映。bg_tts 合成完了 != 再生完了の不整合を
        # 解消する (= シナリオ 2 で観察、HUD で発話途中に灰色化する不具合)。
        # status_manager.set_status の冪等性 (同 status no-op) により、通常応答
        # caller の最終応答経路で _bg_cleanup_pipeline と二重発火しても害なし。
        # ask_character target キャラ用に追加した経路がメインの想定。
        if task.get("is_last") and status_manager is not None:
            try:
                status_manager.set_status(
                    character, CharacterStatus.READY,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "playback worker is_last READY 反映失敗 [character=%s]: %s",
                    character, exc,
                )
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
    # Phase 0.5-B-α: 全キャラ状態を一元管理する CharacterStatusManager は
    # bg-continuous モードのみで生成。他 backend (porcupine / speech / sherpa /
    # continuous / keyboard) では None で動作 (= status 反映スキップ、後方互換)。
    # WHY: 挙手機能 (= Raisehand 状態) や bg LLM (= Thinking) は bg-continuous
    # でしか動かないため、Manager 自体も bg-continuous 限定。それ以外の backend で
    # 「Talking 状態を出したい」要件が出たら別途対応。
    status_manager: "CharacterStatusManager | None" = None
    if effective_backend == "bg-continuous":
        from .dispatcher import (
            DRAIN_MAX_AGE_SEC,
            DRAIN_MAX_EVENTS,
            Dispatcher,
        )
        from .events import (
            build_dispatcher_handraise_update,
            build_dispatcher_queue_update,
        )

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

        # ─── Phase 0.5-A フェーズ 6: handraise 関連の publish helper ───
        # Dispatcher は状態機械として純粋に保つため、bus.publish への接続は run_loop の
        # closure 側で実装する。queue.update と同じく、handraise 関連イベントは特定の
        # ターンに紐付かないため毎回新 trace_id を生成する。
        def _publish_handraise_update(states_copy, cooldowns_copy):
            """``dispatcher.handraise.update`` イベントを Redis Stream に publish する。

            Phase 0.5-A の挙手機能で、HandraiseState / CooldownState の dict が
            変化した瞬間 (start / approval / denial / lapse) に呼ばれる。HUD の
            デバッグ dashboard で「現在挙手中のキャラ」「連続却下回数」を可視化する
            (配信画面非表示)。

            Args:
                states_copy:    {target_slug: HandraiseState} の shallow copy
                cooldowns_copy: {target_slug: CooldownState} の shallow copy
            """
            try:
                now = time.monotonic()
                states_payload = [
                    {
                        "target_slug": s.target_slug,
                        "started_at_age_sec": now - s.started_at,
                        "phrase": s.phrase,
                        "se_pending": s.se_pending,
                        "utterance_count_since": s.utterance_count_since,
                        "trace_id": s.trace_id,
                    }
                    for s in states_copy.values()
                ]
                cooldowns_payload = {
                    slug: {
                        "cooldown_until": cd.cooldown_until,
                        "consecutive_denials": cd.consecutive_denials,
                        "threshold_multiplier": cd.threshold_multiplier,
                    }
                    for slug, cd in cooldowns_copy.items()
                }
                event = build_dispatcher_handraise_update(
                    handraise_states=states_payload,
                    cooldowns=cooldowns_payload,
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    trace_id=_new_uuid(),
                )
                publish(event)
            except Exception as exc:
                logger.warning("dispatcher.handraise.update publish 失敗: %s", exc)

        def _publish_bubble_from_dispatcher(character, step, text, ttl_ms, category):
            """Dispatcher 発の ``bubble.update`` を Redis Stream に publish する。

            通常応答中の bubble.update (ターン毎 trace_id) と区別するため、
            handraise 関連の bubble は毎回新 trace_id で発行する (= ターンに紐付か
            ない概念)。Dispatcher 側から ``handraise / denied / lapsed`` の 3 種類が
            発行される。``answering`` は run_loop が承認時に別経路で発行する
            (フェーズ 7 で接続)。

            Phase 0.5-A 8-10: ``category`` は dispatcher 側で "handraise" を明示して
            渡してくる。本 callback はそのまま forward する (ロジック非介入)。
            """
            try:
                event = build_bubble_update(
                    character=character,
                    step=step,
                    text=text,
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    trace_id=_new_uuid(),
                    ttl_ms=ttl_ms,
                    category=category,
                )
                publish(event)
            except Exception as exc:
                logger.warning("bubble.update(%s) from dispatcher publish 失敗: %s", step, exc)

        # ─── Phase 0.5-B-α: CharacterStatusManager 生成 + publish closure 接続 ──
        # 全キャラ状態 (Ready/Thinking/ToolCalling/Raisehand/Talking) を一元管理。
        # subscribe で character.status.update event を Redis Stream に publish し、
        # HUD dashboard (= V2 nautilus-v2、別リポ) が状態を可視化する土台を作る。
        status_manager = CharacterStatusManager()

        def _publish_character_status(
            slug: str,
            new_status: "CharacterStatus",
            old_status: "CharacterStatus",
            metadata: "dict | None",
        ) -> None:
            """状態変化時に character.status.update event を bus に publish する。

            queue.update / handraise.update と同じく、状態変化は特定ターンに紐付か
            ないため毎回新 trace_id を生成。失敗は warning ログのみで状態管理本体は
            続行 (= bus 障害でアプリが止まらない fail-open)。

            Args:
                slug:        キャラ slug
                new_status:  遷移後の CharacterStatus
                old_status:  遷移前の CharacterStatus
                metadata:    Talking 時の {"pose": str, "text": str} 等。他状態は None。
            """
            try:
                event = build_character_status_update(
                    character=slug,
                    status=new_status.value,
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    trace_id=_new_uuid(),
                    previous_status=old_status.value,
                    metadata=metadata,
                )
                publish(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "character.status.update publish 失敗 [character=%s]: %s",
                    slug, exc,
                )

        status_manager.subscribe(_publish_character_status)

        # Phase 0.5-A フェーズ 7: BG LLM 起動 + handraise 物理通知の 4 callback を
        # factory 経由で生成 (session 識別子 + 配信文脈を closure として捕捉)。
        # Phase 0.5-B-α: status_manager も factory に注入し、挙手承認応答経路で
        # Talking + Ready を反映できるようにする。
        # Phase 0.5-B-β-1 commit 3: ask_character の対話 TTS を BG LLM 経路でも
        # Phase 0.5-F-3: 案 R wiring 切替 ★大きな挙動変更点★
        # 既存 `_create_handraise_runner_and_callbacks` (= 6-callable factory) から
        # `_create_replay_runner_and_callbacks` (= 4-callable factory) に切替える。
        #
        # 【WHY: 案 R に切替えると何が変わる】
        # - bg_runner=None: 挙手中の LLM 先行計算を停止 (= Phase 0.5-A 案 W'-1 の破棄)
        # - on_handraise_approved=None: 承認時の専用経路を廃止
        # - on_approval_replay=replay_callback: 承認時に dispatcher.on_wake_detected で
        #   wake_event queue に inject → 次ターンとして callout 経路 (= run_pipeline) で
        #   処理 → 視聴者に音声が確実に届く
        # - on_approval_progressing=None: bridge filler 専用経路を廃止 (= R-1-b、
        #   通常応答 filler に統合される、F-8 で詳細実装予定)
        #
        # この commit 以降、raisehand 承認の挙動は callout 経路と完全統合される。
        # 中間実走 13 シナリオ γ で観察された「TTS 4 系統廃棄経路 → 5 分以上沈黙」は
        # 構造的に解消される (= callout 経路は廃棄経路を持たない)。
        #
        # 【既存 mutable list ラッパー (= _current_on_tts_chunk 等) の維持】
        # F-3 では削除しない (= F-4 で旧 factory + bg_runner / on_handraise_approved
        # 経路の dead code を一括削除する設計)。F-3 完了時点では旧 factory も module
        # に残るが、wiring されないので呼ばれない。partial revert は wiring 戻しで瞬時。
        _current_on_tts_chunk: list = [None]
        _current_on_pose_ready: list = [None]
        _current_playback_queue: list = [None]
        # 案 R 用 dispatcher_ref (= late binding、Dispatcher 構築直後に [0] にセット)
        _dispatcher_ref: list = [None]
        (
            _on_handraise_started,
            _on_handraise_phrase_pending_release,
            _on_handraise_close,
            _on_approval_replay,
        ) = _create_replay_runner_and_callbacks(
            session_stream_id=session_stream_id,
            session_id_root=session_id_root,
            stream_context=stream_context,
            status_manager=status_manager,
            dispatcher_ref=_dispatcher_ref,
        )

        dispatcher = Dispatcher(
            on_queue_update=_publish_queue_update,
            on_handraise_update=_publish_handraise_update,
            on_bubble_update=_publish_bubble_from_dispatcher,
            # Phase 0.5-F-3: 案 R wiring (= 廃止経路を None に、新経路を有効化)
            bg_runner=None,                      # ★ LLM 先行計算なし
            on_handraise_started=_on_handraise_started,
            on_handraise_phrase_pending_release=_on_handraise_phrase_pending_release,
            on_handraise_approved=None,          # ★ 承認専用経路を廃止
            on_handraise_close=_on_handraise_close,
            on_approval_progressing=None,        # ★ bridge filler 専用経路を廃止 (R-1-b)
            on_approval_replay=_on_approval_replay,  # ★ 案 R の中核 (= callout 経路統合)
            status_manager=status_manager,
        )
        # 案 R: dispatcher_ref に Dispatcher 自身を late binding で設定。
        # on_approval_replay 内で `_dispatcher_ref[0].on_wake_detected(...)` 呼出に使う。
        _dispatcher_ref[0] = dispatcher
        # BackgroundContinuousListener を起動。録音スレッドが回り始め、
        # 検知された wake_event は dispatcher.on_wake_detected で queue に積まれる。
        # Phase 0.5-A フェーズ 6: 全 segment を dispatcher.on_segment_added に流して
        # 挙手判定 (interjection_candidate) と承認判定 (check_approval) を担わせる。
        listener.start(
            on_wake_detected=dispatcher.on_wake_detected,
            on_segment_added=dispatcher.on_segment_added,
        )
        logger.info("Block 0 + Phase 0.5-A: Dispatcher + BackgroundContinuousListener 起動完了")

    def _bg_cleanup_pipeline(completed_slug: str | None = None) -> None:
        """bg-continuous モードでの pipeline 終了処理。

        - listener の routing_paused を解除して wake 判定を再開
        - dispatcher を IDLE 状態に戻し、期限切れ event を破棄、queue 残りがあれば
          メインスレッドを起こす

        Pipeline 成功・失敗を問わず必ず呼ぶ必要がある (routing_paused が True のまま
        放置されると wake 検知が永続的に止まり、wait_for_next_event が無限待機する)。

        Args:
            completed_slug: 完了したターンのキャラ slug。ログ強化 L-4 で追加。
                            dispatcher.on_pipeline_complete のログに渡して識別容易に。
                            失敗 path で slug 不明な場合は None で OK。
        """
        if effective_backend == "bg-continuous":
            listener.set_routing_paused(False)
            if dispatcher is not None:
                dispatcher.on_pipeline_complete(completed_slug=completed_slug)
        # Phase 0.5-B-α: 通常応答完了 → Ready (Thinking / Talking のいずれからでも)
        # bg-continuous 以外の backend (porcupine 等) では status_manager=None で no-op。
        # WHY: completed_slug 不明時 (= pipeline 例外で speaker 不明) は status 反映
        # スキップ (= 誤って他キャラを Ready にするのを防ぐ fail-safe)。
        if status_manager is not None and completed_slug:
            status_manager.set_status(completed_slug, CharacterStatus.READY)

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
                """bubble.update を publish する。失敗は warning log のみ。

                Phase 0.5-A 8-10: 通常応答 playback worker から呼ばれる (speaking/done/
                pose_change)。すべて category="speech" 固定。挙手系の bubble はこの
                経路を通らない (dispatcher 経由)。
                """
                try:
                    event = build_bubble_update(
                        character=character,
                        step=step,
                        text=text,
                        category="speech",
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

                # Phase 0.5-A 8-11: bg-continuous モード時のみ flush callback を渡す。
                # WHY: dispatcher が存在しない backend (continuous / speech-activated) では
                # 挙手機能自体が動かないため flush 対象が無く、None でよい (= 従来挙動)。
                _on_last_chunk_played_cb = (
                    dispatcher.flush_pending_handraise_releases
                    if dispatcher is not None
                    else None
                )
                _playback_thread = threading.Thread(
                    target=_run_playback_worker,
                    args=(_playback_queue,),
                    kwargs={
                        "publish_bubble_fn": _publish_bubble_safe,
                        "play_audio_fn": play_audio_file,
                        "cleanup_audio_fn": _cleanup_audio,
                        "set_pose_fn": _set_pose_safe,
                        "on_last_chunk_played": _on_last_chunk_played_cb,
                        # Phase 0.5-B-β-3 commit 2: ask_character target キャラの
                        # READY 反映を playback worker 経由に移すため status_manager を渡す。
                        # caller の最終応答経路では _bg_cleanup_pipeline が READY 反映する
                        # 経路と二重になるが、status_manager の冪等性で害なし。
                        "status_manager": status_manager,
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
                # ログ強化 L-4: character を含めて、ask_character や挙手承認応答中に
                # 並行する複数 chunk のうちどのキャラのか識別容易に
                logger.info(
                    "TTS チャンク再生キュー投入 [character=%s]: %s (chunk %d, is_last=%s, len=%d)",
                    character, _audio_name, _chunk_count[0], is_last, len(chunk_text),
                )
                if _playback_queue is not None:
                    # 該当キャラの pose 予約があれば付与 (playback worker が再生直前に
                    # set_pose する)。一度使ったら dict から削除して、同キャラの後続
                    # チャンクには pose=None を渡す (= 「切替不要、現在の pose を維持」)。
                    # これがないと、chunk 1 で special_doya に切り替わった後、
                    # chunk 2/3 で neutral に逆戻りしてしまう。
                    chunk_pose = _pending_poses.pop(character, None)
                    # Phase 0.5-B-α (commit 9): 通常応答経路の Talking 反映は graph._tts_node
                    # 内に集約 (= full LLM response を metadata.text に含めるため)。
                    # 旧 commit 4 ではここで chunk_count == 1 時に reflect していたが、
                    # chunk_text のみで full text を含められなかったため graph 側に移動。
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

            # Phase 0.5-B-β-1 commit 3: 最新ターンの closure を mutable list ref に
            # 書き込み、bg_runner._body から ref[0] 経由で参照可能にする。
            # これにより通常応答ターンと並行する挙手 BG LLM の ask_character ツールが
            # 同じ _playback_queue (= 通常応答用) に対話 TTS chunk を投入できる。
            # Phase 0.5-B-β-2 commit 3: 同パターンで _current_playback_queue も
            # 更新。on_handraise_close callback が drain task を投入する先となる。
            _current_on_tts_chunk[0] = _on_tts_chunk
            _current_on_pose_ready[0] = _on_pose_ready
            _current_playback_queue[0] = _playback_queue

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
                    # Phase 0.5-B-α: graph._generation_node + BubbleToolCallbackHandler
                    # で Thinking / ToolCalling 反映に使う (= 通常応答経路)。
                    # bg-continuous 以外の backend では status_manager=None で no-op。
                    status_manager=status_manager,
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
                # ログ強化 L-4: 失敗 path でも speaker_hint があれば slug context として渡す
                _bg_cleanup_pipeline(completed_slug=speaker_hint)
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
            # ログ強化 L-4: result.speaker を slug context として渡す (= 完了応答の識別)
            _bg_cleanup_pipeline(completed_slug=result.speaker)

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

    # Phase 0.5-D-e-4-2 (= 中間実走 12 検分):
    # 起動時に L2_FALLBACK_TIMEOUT_SEC の解決値を INFO 出力。
    # 環境変数 typo / 不正値 fallback を実走で早期発見できるようにする
    # (= dispatcher.py の bg_completed_timeout ログと対応する fallback 側ログ)。
    _fb_timeout_env = os.environ.get("L2_FALLBACK_TIMEOUT_SEC", "(未設定)")
    try:
        _fb_timeout_resolved = (
            float(_fb_timeout_env) if _fb_timeout_env != "(未設定)" else 30.0
        )
    except ValueError:
        _fb_timeout_resolved = 30.0
    logger.info(
        "run_loop 設定: fallback_timeout=%.1fs (env L2_FALLBACK_TIMEOUT_SEC=%s)",
        _fb_timeout_resolved, _fb_timeout_env,
    )

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
