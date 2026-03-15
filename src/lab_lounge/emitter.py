"""
emitter.py — 開発用テキスト / 音声ファイルエミッタ CLI エントリポイント

使い方:
    # テキスト入力
    uv run python -m lab_lounge.emitter "今日の天気を教えて"
    uv run python -m lab_lounge.emitter "テキスト" --stream-id my-stream-001

    # 音声ファイル入力 (L2_USE_REAL_STT=true が必要)
    uv run python -m lab_lounge.emitter --audio-file samples/q1.wav

    # text と --audio-file は排他。同時指定はエラー。

出力:
    publish 完了後に stream_id / session_id / trace_id を標準出力する

STT モード:
    L2_USE_REAL_STT=true  のとき transcribe_audio_file() を呼ぶ
    L2_USE_REAL_STT=false のとき --audio-file が指定されても dummy text を使う
"""

import argparse
import logging
import os
import sys
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from .pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _new_uuid() -> str:
    return str(uuid4())


def _transcribe_audio(audio_path: str) -> tuple[str, dict[str, Any]]:
    """
    音声ファイルを文字起こしし (text, utterance_meta) を返す。

    L2_USE_REAL_STT=true  のとき: stt.transcribe_audio_file() を呼ぶ
    L2_USE_REAL_STT=false のとき: 警告を出してダミーテキストを返す

    Returns:
        (text, utterance_meta)
          utterance_meta は build_utterance_final() に **kwargs で渡す dict
    """
    use_real = os.environ.get("L2_USE_REAL_STT", "false").lower() in ("true", "1", "yes")

    if use_real:
        from .stt import transcribe_audio_file as _transcribe
        provider = os.environ.get("L2_STT_PROVIDER", "openai")
        lang = os.environ.get("L2_STT_LANG", "ja")
        model = os.environ.get("L2_STT_MODEL", "whisper-1")
        result = _transcribe(audio_path, provider=provider, lang=lang, model=model)
        meta: dict[str, Any] = {
            "lang": result.lang,
            "confidence": result.confidence if result.confidence is not None else 0.0,
            "duration_ms": result.duration_ms,
        }
        if result.words is not None:
            meta["words"] = result.words
        return result.text, meta
    else:
        logger.warning(
            "L2_USE_REAL_STT=false のため '%s' はダミートランスクリプトを使用します。"
            " real STT を使うには L2_USE_REAL_STT=true を設定してください。",
            audio_path,
        )
        return f"音声ファイル入力（ダミー）: {audio_path}", {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="A.I.byss Lab Lounge — 開発用テキスト / 音声ファイルエミッタ",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="発話テキスト（utterance.final の payload.text）。--audio-file と排他。",
    )
    parser.add_argument(
        "--audio-file",
        metavar="PATH",
        default=None,
        help="音声ファイルパス。STT を経由してテキストに変換する。text と排他。",
    )
    parser.add_argument(
        "--stream-id",
        default=None,
        help="ストリーム ID（省略時は UUIDv4 で自動生成）",
    )
    args = parser.parse_args(argv)

    # text / --audio-file の排他チェック
    if args.text is None and args.audio_file is None:
        parser.error("text または --audio-file のどちらかを指定してください")
    if args.text is not None and args.audio_file is not None:
        parser.error("text と --audio-file は同時に指定できません")

    # 入力解決
    if args.audio_file is not None:
        input_text, utterance_meta = _transcribe_audio(args.audio_file)
    else:
        input_text = args.text
        utterance_meta = None

    stream_id = args.stream_id or _new_uuid()
    session_id = _new_uuid()
    trace_id = _new_uuid()  # TODO: 将来 W3C Trace Context 形式に変更 (Guardrail G-7)

    result = run_pipeline(
        input_text,
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        utterance_meta=utterance_meta,
    )

    print(f"stream_id  : {result.stream_id}")
    print(f"session_id : {result.session_id}")
    print(f"trace_id   : {result.trace_id}")
    for ev in result.events:
        print(f"  published: type={ev['type']} event_id={ev['event_id']}")


if __name__ == "__main__":
    main()

