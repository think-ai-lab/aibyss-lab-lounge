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


def _spawn_handraise_response_playback(
    slug: str,
    chunks: list[dict],
    trace_id: str,
    *,
    session_stream_id: str,
    session_id_root: str,
) -> "threading.Thread":
    """承認後の TTS chunks を専用 mini playback worker で再生する (Phase 0.5-A フェーズ 7)。

    通常応答用の ``_playback_queue`` は「ターン終了時に sentinel で閉じる」設計のため
    ターン外 (= 挙手応答) では再利用できない。独立した queue + worker thread を起動する。

    chunks は dispatcher の HandraiseBgResult.chunks (各 dict は ``_run_playback_worker``
    が読む形式: ``{"url", "text", "is_last", "character", "pose"?}``)。

    Args:
        slug:               挙手キャラ slug (worker 名 + ログ用)
        chunks:             再生する TTS chunks の list (BG LLM 先行生成済)
        trace_id:           bubble.update 発行時の handraise 単位 trace_id
        session_stream_id:  bubble.update の stream_id
        session_id_root:    bubble.update の session_id

    Returns:
        起動した daemon thread。
    """
    q: queue.Queue = queue.Queue()
    for chunk in chunks:
        q.put(chunk)
    q.put(None)  # sentinel — 最終 chunk 再生後 worker が done bubble を発行して終了

    def _publish_bubble_for_handraise(character: str, step: str, text: str) -> None:
        """handraise 応答中の bubble.update を毎回新 trace_id で発行。

        Phase 0.5-A 8-10 (A2 確定): 承認後応答 (answering/speaking/done) は category="speech"。
        WHY: 挙手バブル (category="handraise") は承認時に消費され、応答は通常応答エリア
        で表示する設計 (UI 一貫性優先)。HUD 側の switch 文を単純に保てる。
        """
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
                "handraise playback bubble.update(%s) publish 失敗: %s", step, exc,
            )

    def _cleanup_audio(url: str) -> None:
        """再生済 wav の一時ファイルを削除 (ベストエフォート)。"""
        try:
            from .audio_io import _uri_to_path
            p = Path(_uri_to_path(url))
            if p.is_file():
                p.unlink()
        except Exception:
            pass  # cleanup 失敗は無視 (再生自体は完了済)

    def _set_pose_safe(character: str, pose: str) -> None:
        """OBS 立ち絵切替 (失敗しても続行)。"""
        try:
            from .obs import set_pose
            set_pose(character, pose)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "handraise playback set_pose 失敗 (無視): %s", exc,
            )

    t = threading.Thread(
        target=_run_playback_worker,
        args=(q,),
        kwargs={
            "publish_bubble_fn": _publish_bubble_for_handraise,
            "play_audio_fn": play_audio_file,
            "cleanup_audio_fn": _cleanup_audio,
            "set_pose_fn": _set_pose_safe,
        },
        daemon=True,
        name=f"handraise-playback-{slug}",
    )
    t.start()
    return t


def _approved_synthesize_fallback(
    slug: str,
    transcript_snapshot,
    trace_id: str,
    *,
    session_stream_id: str,
    session_id_root: str,
    stream_context: "str | None" = None,
) -> None:
    """挙手承認時に bg_result が無効だった場合の同期 fallback (Phase 0.5-A フェーズ 7)。

    BG LLM が起動失敗 / cancel / 例外で chunks 空の場合に呼ばれる。本関数は daemon
    スレッド内で同期的に ``run_pipeline`` を呼び、TTS chunks を蓄積してから
    ``_spawn_handraise_response_playback`` で再生する。

    通常応答パスは run_loop closure 内で ``on_tts_chunk_ready`` 経由で
    ``_playback_queue`` に投入されるが、fallback はターン外 daemon thread で動くため
    その経路に乗せられない。代わりに bg_result 経路と同じ専用 mini playback worker
    (``_spawn_handraise_response_playback``) を再利用する。

    graph 側の ``_publish_bubble("answering")`` は ``suppress_bubble_answering=False``
    で発行される (= 通常応答パスと同じ流れ。bg_result 経路は run_loop 側で発行
    していたが、fallback では graph 経路の方が自然)。

    呼出スレッドは別 daemon thread の ``approved-fallback-<slug>`` (run_loop の
    callback factory が起動する) なので、run_pipeline blocking してもメインループや
    録音スレッドに影響しない。

    Args:
        slug:              挙手キャラ slug
        transcript_snapshot: 挙手判定時の TranscriptBuffer 文字列 (BG LLM の入力)
        trace_id:          handraise 単位の trace_id (空なら新規生成)
        session_stream_id: stream_id
        session_id_root:   session_id
        stream_context:    配信文脈 markdown (起動時に load_stream_context() で取得)
    """
    text = transcript_snapshot or ""
    if not isinstance(text, str):
        text = str(text)

    fallback_trace_id = trace_id or _new_uuid()
    chunks: list[dict] = []

    def on_chunk(url, chunk_text, is_last, character, pose=None):
        """TTS chunk 蓄積 hook。run_pipeline 完了後に playback worker へまとめて渡す。"""
        chunk: dict = {
            "url": url,
            "text": chunk_text,
            "is_last": is_last,
            "character": character,
        }
        if pose is not None:
            chunk["pose"] = pose
        chunks.append(chunk)

    # ログ強化 W'-3: fallback パス進入を実走で識別容易にする。W'-1 適用後は
    # fallback は「LLM 失敗時の救済」のみに縮小されるため、このログ自体が稀。
    # シナリオ B/C 再走時に出ていれば「BG LLM が失敗した」とすぐ判別可能。
    logger.info(
        "fallback 同期再生成 開始 [character=%s]: trace_id=%s text_len=%d",
        slug, fallback_trace_id, len(text),
    )

    try:
        run_pipeline(
            text,
            stream_id=session_stream_id,
            session_id=session_id_root,
            trace_id=fallback_trace_id,
            speaker_hint=slug,
            on_tts_chunk_ready=on_chunk,
            stream_context=stream_context,
            suppress_bubble_answering=False,  # graph 側で answering bubble 発行
            # WHY (バグ 3 修正、案 A、実走 logs/runs/run_loop_20260508_181051.log で発覚):
            # fallback パスでは ask_character ツールを Agent から除外する。
            # 通常応答ターン (= chisame の callout 応答) が並行 TTS 再生中に
            # mimi の挙手承認 → fallback パスで run_pipeline 起動 → LLM が
            # ask_character ツールを呼ぶ → 導入セリフ TTS が VOICEPEAK FIFO に
            # 投入されるが、現在再生中の chisame TTS の完了待ちで
            # 「ask_character 導入セリフ再生待ち...」のままハング、という
            # deadlock を観察した。fallback パスを「単独応答」に限定してこの
            # 経路を断ち切る。bg_result=ready 経路 (= 通常パス) では disable_tools=None
            # のままなので ask_character の協働応答は通常通り使える。
            disable_tools=["ask_character"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "挙手承認 fallback (run_pipeline 同期再生成) 失敗 [character=%s]: err=%s",
            slug, exc,
        )
        return

    if not chunks:
        # TTS 出力なし (ダミー TTS モード or TTS 失敗)。再生はスキップするが、
        # graph 側で llm.final / tts.done は発行済 (bubble は thinking → answering で停止)。
        logger.warning(
            "挙手承認 fallback [character=%s]: chunks 空 (TTS 出力なし) → 再生スキップ",
            slug,
        )
        return

    # ログ強化 W'-3: chunks 蓄積完了をログに残す。W'-1 適用後は通常応答ターンの
    # _on_tts_chunk 経由で「TTS チャンク再生キュー投入」ログが出るが、fallback は
    # 専用 mini worker 経路なのでこのログがないと chunks 蓄積を直接観察できない。
    total_chars = sum(len(c.get("text", "")) for c in chunks)
    logger.info(
        "fallback chunks 蓄積完了 [character=%s]: count=%d total_text_chars=%d",
        slug, len(chunks), total_chars,
    )

    # 蓄積した chunks を専用 mini playback worker で再生 (bg_result 経路と同じ仕組み)。
    # worker が speaking → done の bubble.update を発行しつつ、wav を順次再生する。
    _spawn_handraise_response_playback(
        slug,
        chunks,
        fallback_trace_id,
        session_stream_id=session_stream_id,
        session_id_root=session_id_root,
    )


def _create_handraise_runner_and_callbacks(
    *,
    session_stream_id: str,
    session_id_root: str,
    stream_context: "str | None",
):
    """Phase 0.5-A フェーズ 7: 4 つの callback と bg_runner を生成する factory。

    run_loop の closure 内変数 (session_stream_id 等) を捕捉した上で、
    ``Dispatcher`` コンストラクタに渡せる callable 群を返す。module-level の
    factory として書くことで、テストから直接 factory を呼んで返り値を検証できる。

    Args:
        session_stream_id: 当該セッションの stream_id (handraise イベントに付与)
        session_id_root:   当該セッションの session_id
        stream_context:    配信文脈 markdown (run_pipeline に渡す)

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
            try:
                result = run_pipeline_llm_only(
                    transcript_snapshot if isinstance(transcript_snapshot, str)
                    else str(transcript_snapshot or ""),
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    trace_id=bg_trace_id,
                    speaker_hint=target_slug,
                    stream_context=stream_context,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "BG LLM 失敗 [character=%s]: trace_id=%s err=%s",
                    target_slug, bg_trace_id, exc,
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
            logger.info(
                "挙手承認 [character=%s]: bg_result %s → fallback パス",
                slug,
                "未生成" if bg_result is None else "result=None",
            )
            threading.Thread(
                target=lambda: _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                ),
                name=f"approved-fallback-{slug}",
                daemon=True,
            ).start()
            return

        # bg_trace_id を引き継ぐ (handraise 単位の trace_id 一貫性)
        bg_trace_id = bg_result.trace_id or trace_id or _new_uuid()

        # answering bubble の text を pipeline_result.events から抽出
        answering_text = ""
        try:
            for ev in bg_result.result.events:
                if ev.get("type") == "llm.final":
                    answering_text = ev.get("payload", {}).get("text", "")
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
                ),
                name=f"approved-fallback-{slug}",
                daemon=True,
            ).start()
            return

        # bubble.update("answering") を発行 (TTS 開始直前タイミング)
        # Phase 0.5-A 8-10 (A2 確定): 承認後応答は category="speech" (UI 一貫性優先)。
        try:
            event = build_bubble_update(
                character=slug,
                step="answering",
                text=answering_text,
                stream_id=session_stream_id,
                session_id=session_id_root,
                trace_id=bg_trace_id,
                category="speech",
            )
            publish(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bubble.update(answering) from approval publish 失敗: %s", exc,
            )

        # TTS-only graph で TTS 実行 + chunks 蓄積 → playback worker (daemon thread)
        def _tts_and_play() -> None:
            chunks: list[dict] = []

            def on_chunk(url, chunk_text, is_last, character, pose=None):
                """on_tts_chunk_ready 用、chunks 蓄積。"""
                chunk: dict = {
                    "url": url,
                    "text": chunk_text,
                    "is_last": is_last,
                    "character": character,
                }
                if pose is not None:
                    chunk["pose"] = pose
                chunks.append(chunk)

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
                _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                )
                return

            if not chunks:
                # ダミー TTS モード or TTS 出力なし → fallback パスで救済
                logger.warning(
                    "挙手承認 TTS [character=%s]: chunks 空 → fallback パス",
                    slug,
                )
                _approved_synthesize_fallback(
                    slug, transcript_snapshot, trace_id,
                    session_stream_id=session_stream_id,
                    session_id_root=session_id_root,
                    stream_context=stream_context,
                )
                return

            logger.info(
                "挙手承認 TTS 同期実行 完了 [character=%s]: chunks=%d",
                slug, len(chunks),
            )
            # ログ強化 W'-3: playback worker 起動を明示。実走時に「TTS は完了
            # したが playback まで到達したか」を 1 行 grep で追跡可能にする。
            logger.info(
                "挙手承認 playback worker 起動 [character=%s]",
                slug,
            )
            _spawn_handraise_response_playback(
                slug,
                chunks,
                bg_trace_id,
                session_stream_id=session_stream_id,
                session_id_root=session_id_root,
            )

        threading.Thread(
            target=_tts_and_play,
            name=f"approved-tts-{slug}",
            daemon=True,
        ).start()

    return (
        bg_runner,
        on_handraise_started,
        on_handraise_phrase_pending_release,
        on_handraise_approved,
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

        # Phase 0.5-A フェーズ 7: BG LLM 起動 + handraise 物理通知の 4 callback を
        # factory 経由で生成 (session 識別子 + 配信文脈を closure として捕捉)。
        (
            _bg_runner,
            _on_handraise_started,
            _on_handraise_phrase_pending_release,
            _on_handraise_approved,
        ) = _create_handraise_runner_and_callbacks(
            session_stream_id=session_stream_id,
            session_id_root=session_id_root,
            stream_context=stream_context,
        )

        dispatcher = Dispatcher(
            on_queue_update=_publish_queue_update,
            on_handraise_update=_publish_handraise_update,
            on_bubble_update=_publish_bubble_from_dispatcher,
            bg_runner=_bg_runner,
            on_handraise_started=_on_handraise_started,
            on_handraise_phrase_pending_release=_on_handraise_phrase_pending_release,
            on_handraise_approved=_on_handraise_approved,
        )
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
