"""
generate_smoke_wav.py — セマンティックスモーク用 WAV 生成スクリプト

VOICEVOX Engine を使って短い実発話 WAV を生成し、
samples/test_greeting.wav として保存する。

【使い方】
  # VOICEVOX Engine を起動してから（別ターミナル）:
  # docker run -p 50021:50021 voicevox/voicevox_engine:latest

  uv run python scripts/generate_smoke_wav.py

  # スピーカー ID / テキスト / 出力先を変更する場合:
  uv run python scripts/generate_smoke_wav.py --speaker 3 --text "こんにちは" --output samples/my_wav.wav

【STT 検証への使い方】
  生成した WAV を emitter に渡して STT が正しく認識できるか確認する:

  uv run python -m lab_lounge.emitter \\
      --audio-file samples/test_greeting.wav \\
      --stream-id semantic-smoke

  STT 結果（utterance.final.payload.text）が --expected-text の内容と
  おおよそ一致すれば OK とする（完全一致は不要）。
"""

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path


DEFAULT_TEXT = "こんにちは、私はAITuberのヴォイドルです。よろしくお願いします。"
DEFAULT_SPEAKER = 89  # Voidoll
DEFAULT_OUTPUT = Path(__file__).parent.parent / "samples" / "test_greeting.wav"
DEFAULT_VOICEVOX_URL = "http://localhost:50021"


def generate_wav(
    text: str,
    *,
    speaker_id: int,
    output_path: Path,
    voicevox_url: str,
) -> None:
    """VOICEVOX Engine で音声合成して WAV に保存する。"""
    print(f"VOICEVOX URL : {voicevox_url}")
    print(f"スピーカー ID : {speaker_id}")
    print(f"テキスト      : {text}")
    print(f"出力先        : {output_path}")
    print()

    # 1. audio_query
    print("1/2 audio_query ...")
    query_params = urllib.parse.urlencode({"text": text, "speaker": speaker_id})
    query_url = f"{voicevox_url}/audio_query?{query_params}"
    with urllib.request.urlopen(
        urllib.request.Request(query_url, method="POST")
    ) as resp:
        audio_query = json.loads(resp.read().decode("utf-8"))

    # 2. synthesis
    print("2/2 synthesis ...")
    synth_url = f"{voicevox_url}/synthesis?speaker={speaker_id}"
    body = json.dumps(audio_query).encode("utf-8")
    req = urllib.request.Request(
        synth_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    with urllib.request.urlopen(req) as resp:
        wav_bytes = resp.read()

    # duration 確認
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)

    print()
    print(f"✅ 生成完了: {output_path}  ({duration_ms} ms)")
    print()
    print("STT 検証コマンド:")
    print(
        f"  uv run python -m lab_lounge.emitter "
        f"--audio-file {output_path} --stream-id semantic-smoke"
    )
    print()
    print("期待 STT テキスト（おおよそ一致すれば OK）:")
    print(f"  {text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VOICEVOX で test_greeting.wav を生成する"
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"合成テキスト (デフォルト: {DEFAULT_TEXT!r})",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=DEFAULT_SPEAKER,
        help=f"VOICEVOX スピーカー ID (デフォルト: {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"出力ファイルパス (デフォルト: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--voicevox-url",
        default=DEFAULT_VOICEVOX_URL,
        help=f"VOICEVOX Engine URL (デフォルト: {DEFAULT_VOICEVOX_URL})",
    )
    args = parser.parse_args()

    try:
        generate_wav(
            args.text,
            speaker_id=args.speaker,
            output_path=args.output,
            voicevox_url=args.voicevox_url,
        )
    except OSError as exc:
        print(f"❌ VOICEVOX Engine に接続できません: {exc}", file=sys.stderr)
        print(
            "   VOICEVOX Engine を起動してから再実行してください:\n"
            "   docker run -p 50021:50021 voicevox/voicevox_engine:latest",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
