"""
emitter.py — 開発用テキストエミッタ CLI エントリポイント

使い方:
    uv run python -m lab_lounge.emitter "今日の天気を教えて"
    uv run python -m lab_lounge.emitter "テキスト" --stream-id my-stream-001

出力:
    publish 完了後に stream_id / session_id / trace_id を標準出力する

非スコープ:
    マイク入力 / 本物の STT / LLM / TTS は呼び出さない
"""

import argparse
import logging
import sys
from uuid import uuid4

from .pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


def _new_uuid() -> str:
    return str(uuid4())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="A.I.byss Lab Lounge — 開発用テキストエミッタ",
    )
    parser.add_argument("text", help="発話テキスト（utterance.final の payload.text）")
    parser.add_argument(
        "--stream-id",
        default=None,
        help="ストリーム ID（省略時は UUIDv4 で自動生成）",
    )
    args = parser.parse_args(argv)

    stream_id = args.stream_id or _new_uuid()
    session_id = _new_uuid()
    trace_id = _new_uuid()  # TODO: 将来 W3C Trace Context 形式に変更 (Guardrail G-7)

    result = run_pipeline(
        args.text,
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
    )

    print(f"stream_id  : {result.stream_id}")
    print(f"session_id : {result.session_id}")
    print(f"trace_id   : {result.trace_id}")
    for ev in result.events:
        print(f"  published: type={ev['type']} event_id={ev['event_id']}")


if __name__ == "__main__":
    main()
