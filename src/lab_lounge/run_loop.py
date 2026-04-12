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
from .emitter import _transcribe_audio
from .log_setup import setup_logging
from .pipeline import PipelineResult, run_pipeline

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

    turn = 0
    try:
        while max_turns is None or turn < max_turns:
            # ─── 1. 起動トリガー ────────────────────────────────
            speaker_hint: str | None = None
            input_text: str | None = None
            utterance_meta = None

            if effective_backend in ("porcupine", "speech", "sherpa", "continuous"):
                wake_result = listener.listen_once(timeout_seconds=wake_timeout)
                if wake_result is None:
                    # タイムアウト → 再度待機
                    continue
                speaker_hint = wake_result.character_slug
                logger.info("ウェイクワード検知: %s", speaker_hint)
                print(f"ウェイクワード検知: {speaker_hint}")

                # speech / sherpa / continuous バックエンドは transcript がそのまま発話テキスト
                if effective_backend in ("speech", "sherpa", "continuous") and wake_result.transcript:
                    input_text = wake_result.transcript
                    logger.info("STT 認識結果: %s", input_text)
                    print(f"認識結果: {input_text}")
            else:
                # keyboard モード
                _wait_for_enter()

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
            _chunk_count = [0]
            _playback_queue: queue.Queue[str | None] | None = None
            _playback_thread: threading.Thread | None = None
            _filler_stop = threading.Event()
            _filler_thread: threading.Thread | None = None

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

                def _playback_worker(q: queue.Queue) -> None:
                    while True:
                        url = q.get()
                        if url is None:
                            break
                        play_audio_file(url)
                        _cleanup_audio(url)

                _playback_thread = threading.Thread(
                    target=_playback_worker, args=(_playback_queue,), daemon=True,
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

            # OBS 立ち絵を本命応答の再生開始タイミングに同期させるための
            # 遅延 pose 適用。パイプラインが on_pose_ready で pose を予約し、
            # 最初のチャンク投入時に _on_tts_chunk 内で set_pose を実行する。
            _pending_pose: list[str | None] = [None, None]  # [character_slug, pose_value]

            def _on_pose_ready(slug: str, pose: str) -> None:
                """パイプラインから呼ばれる。pose を予約して実際の切替を遅延させる。"""
                _pending_pose[:] = [slug, pose]

            def _on_tts_chunk(url: str) -> None:
                # フィラーを停止してから本編を再生
                # フレーズ完了まで待ち、間を持たせてから本編を開始
                if _filler_thread is not None and _filler_thread.is_alive():
                    _filler_stop.set()
                    _filler_thread.join(timeout=30)
                    # フィラーと本編の間に少し間を持たせる
                    import time
                    time.sleep(0.5)
                # 最初のチャンクで OBS 立ち絵を切り替え (本命応答の再生開始と同期)
                if _pending_pose[0] is not None:
                    from .obs import set_pose
                    set_pose(_pending_pose[0], _pending_pose[1] or "neutral")
                    _pending_pose[0] = None
                _chunk_count[0] += 1
                _audio_name = url.rsplit("/", 1)[-1] if "/" in url else url
                logger.info("TTS チャンク再生キュー投入: %s (chunk %d)", _audio_name, _chunk_count[0])
                if _playback_queue is not None:
                    _playback_queue.put(url)

            try:
                result = run_pipeline(
                    input_text,
                    # stream_id / session_id は run_loop セッション全体で固定
                    # (RecentC2Retriever が同一ストリーム内の過去発話を引けるように)
                    stream_id=session_stream_id,
                    session_id=session_id_root,
                    # trace_id はリクエスト単位 (分散トレース用、個々のターンを識別)
                    trace_id=_new_uuid(),
                    utterance_meta=utterance_meta,
                    speaker_hint=speaker_hint,
                    on_tts_chunk_ready=_on_tts_chunk if not skip_playback else None,
                    on_pose_ready=_on_pose_ready if not skip_playback else None,
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
                continue

            # LLM 応答を表示 + ログ記録
            llm_ev = next((ev for ev in result.events if ev["type"] == "llm.final"), None)
            if llm_ev:
                _resp = llm_ev["payload"]["text"]
                logger.info("LLM 応答 [%s]: %s", result.speaker, _resp)
                print(f"[{result.speaker}] {_resp}")

            # ─── 5. TTS 再生完了待機 ────────────────────────────
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
                    import time
                    time.sleep(0.5)

                tts_ev = next((ev for ev in result.events if ev["type"] == "tts.done"), None)
                if tts_ev:
                    audio_url = tts_ev["payload"].get("audio_url", "")
                    if audio_url:
                        play_audio_file(audio_url)

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
        choices=["porcupine", "speech", "sherpa", "continuous", "keyboard"],
        default=None,
        metavar="BACKEND",
        help="ウェイクワードバックエンド: porcupine / speech / keyboard (省略時は自動選択)",
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
