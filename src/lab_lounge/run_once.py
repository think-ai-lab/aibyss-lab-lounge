"""
run_once.py — 1 回録音 → STT → LLM → TTS → 再生 の最小ランタイム

使い方:
    # デフォルト (5 秒録音)
    uv run python -m lab_lounge.run_once

    # マイク入力モードを明示 (動作は同じ)
    uv run python -m lab_lounge.run_once --mic

    # 録音秒数を指定
    uv run python -m lab_lounge.run_once --record-seconds 10

    # TTS 再生をスキップ (ファイルパスのみ表示)
    uv run python -m lab_lounge.run_once --no-play

前提:
    uv sync --extra mic --extra stt --extra llm --extra tts

動作フロー:
    1. マイク録音       → 一時 WAV ファイルに保存 (audio_io.record_to_file)
    2. STT             → テキスト変換 (emitter._transcribe_audio)
    3. pipeline 実行   → utterance.final / llm.final / tts.done を Event Bus に publish
    4. TTS 再生        → tts.done の audio_url を再生 (audio_io.play_audio_file)

フォールバック:
    - 無音        → 処理中断 (SilenceError)
    - 録音失敗    → 処理中断 (RecordError)
    - STT 失敗    → llm / tts / 再生に進まない
    - 再生失敗    → 警告ログのみ (PipelineResult は返す)

LangSmith 観測:
    LANGSMITH_TRACING=true + LANGSMITH_API_KEY を設定すると
    pipeline 内部の LLM run が LangSmith に記録される。
    run_metadata には stream_id / session_id / trace_id が付与される。
"""

import argparse
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from .audio_io import RecordError, SilenceError, play_audio_file, record_to_file
from .emitter import _transcribe_audio
from .pipeline import PipelineResult, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _new_uuid() -> str:
    return str(uuid4())


# ─── コア関数 ────────────────────────────────────────────────────

def run_once(
    record_seconds: float = 5.0,
    *,
    stream_id: str | None = None,
    tmp_dir: str | None = None,
    skip_playback: bool = False,
    audio_device: int | str | None = None,
) -> PipelineResult | None:
    """
    1 回録音して STT → pipeline (LLM + TTS) → 再生 まで行う。

    Args:
        record_seconds: 録音秒数 (デフォルト 5.0)
        stream_id:      固定 stream_id (省略時は UUIDv4 自動生成)
        tmp_dir:        録音一時ファイルの保存先ディレクトリ (省略時は OS tmpdir)
        skip_playback:  True のとき TTS 再生をスキップする
        audio_device:   録音デバイスのインデックスまたは名前

    Returns:
        PipelineResult (成功時)
        None           (無音検出 / 録音失敗 / STT 失敗時)
    """
    t0_total = time.monotonic()

    # ─── 1. マイク録音 ───────────────────────────────────────────
    t0_rec = time.monotonic()
    try:
        recorded_path = record_to_file(
            record_seconds,
            tmp_dir=tmp_dir,
            device=audio_device,
        )
    except SilenceError as exc:
        logger.warning("無音検出のため中断します: %s", exc)
        return None
    except RecordError as exc:
        logger.error("録音失敗のため中断します: %s", exc)
        return None
    rec_ms = int((time.monotonic() - t0_rec) * 1000)
    logger.info("timing: record=%d ms", rec_ms)

    # ─── 2. STT (完了後に一時ファイル削除) ──────────────────────
    t0_stt = time.monotonic()
    stt_exc: Exception | None = None
    input_text = ""
    utterance_meta: dict = {}
    try:
        input_text, utterance_meta = _transcribe_audio(recorded_path)
    except Exception as exc:
        stt_exc = exc
    finally:
        # 成功・失敗どちらでも一時ファイルを削除する
        try:
            Path(recorded_path).unlink(missing_ok=True)
        except Exception:
            pass

    if stt_exc is not None:
        logger.error("STT 失敗のため中断します: %s", stt_exc)
        return None

    stt_ms = int((time.monotonic() - t0_stt) * 1000)
    logger.info("timing: stt=%d ms | text=%r", stt_ms, input_text[:50])

    # ─── 3. pipeline (LLM + TTS イベント publish) ───────────────
    _sid = stream_id or _new_uuid()
    _sess = _new_uuid()
    _trace = _new_uuid()

    t0_pipe = time.monotonic()
    result = run_pipeline(
        input_text,
        stream_id=_sid,
        session_id=_sess,
        trace_id=_trace,
        utterance_meta=utterance_meta,
    )
    pipe_ms = int((time.monotonic() - t0_pipe) * 1000)

    # LLM / TTS の latency をイベント payload から取得してログ出力
    llm_ev = next((ev for ev in result.events if ev["type"] == "llm.final"), None)
    tts_ev = next((ev for ev in result.events if ev["type"] == "tts.done"), None)
    llm_lat = (llm_ev or {}).get("payload", {}).get("latency_ms", "?")
    tts_dur = (tts_ev or {}).get("payload", {}).get("duration_ms", "?")
    logger.info(
        "timing: llm=%s ms | tts_duration=%s ms | pipe_total=%d ms",
        llm_lat, tts_dur, pipe_ms,
    )

    # ─── 4. TTS 再生 ─────────────────────────────────────────────
    if not skip_playback and tts_ev:
        audio_url = tts_ev["payload"].get("audio_url", "")
        if audio_url:
            t0_play = time.monotonic()
            success = play_audio_file(audio_url)
            play_ms = int((time.monotonic() - t0_play) * 1000)
            if success:
                logger.info("timing: play=%d ms", play_ms)
            else:
                # ファイルが存在しない場合 (ダミー TTS) は DEBUG で済む
                from pathlib import Path as _Path
                from .audio_io import _uri_to_path as _u2p
                if _Path(_u2p(audio_url)).exists():
                    logger.warning("再生できませんでした。TTS ファイル: %s", audio_url)
                else:
                    logger.debug("再生スキップ (ダミー TTS): %s", audio_url)

    total_ms = int((time.monotonic() - t0_total) * 1000)
    logger.info("timing: total=%d ms", total_ms)

    return result


# ─── CLI エントリポイント ────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="A.I.byss Lab Lounge — 1 回録音して 1 往復会話する最小ランタイム",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="マイク入力を使う (デフォルト動作。省略可)",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        metavar="N",
        help="録音秒数 (環境変数 L2_RECORD_SECONDS でも設定可。デフォルト 5)",
    )
    parser.add_argument(
        "--stream-id",
        default=None,
        help="ストリーム ID (省略時は UUIDv4 で自動生成)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="TTS 再生をスキップする",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        metavar="DIR",
        help="録音一時ファイルの保存先ディレクトリ (省略時は OS tmpdir)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="IDX_OR_NAME",
        help="オーディオデバイスのインデックスまたは名前 (省略時はデフォルト)",
    )
    args = parser.parse_args(argv)

    # record_seconds 解決: CLI 引数 > 環境変数 > デフォルト (5 秒)
    if args.record_seconds is not None:
        record_seconds = args.record_seconds
    else:
        record_seconds = float(os.environ.get("L2_RECORD_SECONDS", "5"))

    skip_playback = args.no_play or os.environ.get("L2_NO_PLAY", "false").lower() in (
        "true", "1", "yes"
    )
    tmp_dir = args.tmp_dir or os.environ.get("L2_TMP_DIR")

    # device: CLI 引数 > 環境変数 > None (デフォルトデバイス)
    device: int | str | None = None
    raw_device = args.device or os.environ.get("L2_AUDIO_DEVICE")
    if raw_device is not None:
        device = int(raw_device) if raw_device.lstrip("-").isdigit() else raw_device

    result = run_once(
        record_seconds=record_seconds,
        stream_id=args.stream_id,
        tmp_dir=tmp_dir,
        skip_playback=skip_playback,
        audio_device=device,
    )

    if result is None:
        print("処理を中断しました。")
        return

    print(f"stream_id  : {result.stream_id}")
    print(f"session_id : {result.session_id}")
    print(f"trace_id   : {result.trace_id}")
    for ev in result.events:
        print(f"  published: type={ev['type']} event_id={ev['event_id']}")


if __name__ == "__main__":
    main()
