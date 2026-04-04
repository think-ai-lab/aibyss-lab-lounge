"""
poc_sherpa_streaming.py -- Sherpa-ONNX ストリーミング STT PoC

Sherpa-ONNX + 日本語 Zipformer モデルでマイクからリアルタイム文字起こしする。
環境（Windows / RTX 5090 / CUDA）での動作確認用スタンドアロンスクリプト。

【事前準備】
  1. uv sync --extra stt-sherpa --extra mic
  2. 日本語モデルをダウンロード・展開:
     curl -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2 | tar xjf -
     mv sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01 sherpa-models/

【使い方】
  uv run python scripts/poc_sherpa_streaming.py
  uv run python scripts/poc_sherpa_streaming.py --provider cpu
  uv run python scripts/poc_sherpa_streaming.py --model-dir path/to/model
  uv run python scripts/poc_sherpa_streaming.py --device 2

【出力】
  リアルタイムで部分認識結果と確定結果をコンソールに表示。
  Ctrl+C で終了。

【注意】
  日本語 Zipformer モデルは Offline (非ストリーミング) モデルのため、
  sherpa_onnx.OfflineRecognizer + VAD で擬似リアルタイム処理を行う。
  発話区間の終了を検知した時点で即座に認識を実行する。
"""

import argparse
import os
import sys
from pathlib import Path


_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "sherpa-models"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sherpa-ONNX ストリーミング STT PoC",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="モデルファイルの配置先 (デフォルト: ./sherpa-models/)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="ONNX Runtime provider: cuda / cpu (デフォルト: cuda)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="マイクデバイスのインデックスまたは名前",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        default=True,
        help="int8 量子化モデルを使用 (デフォルト: True)",
    )
    parser.add_argument(
        "--no-quantized",
        action="store_false",
        dest="quantized",
        help="fp32 モデルを使用",
    )
    args = parser.parse_args()

    # -- 依存チェック --
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        print(
            "sounddevice / numpy が未インストールです。\n"
            "  uv sync --extra mic でインストールしてください。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        import sherpa_onnx
    except ImportError as exc:
        print(
            "sherpa-onnx が未インストールです。\n"
            "  uv sync --extra stt-sherpa でインストールしてください。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # -- モデルパス解決 --
    model_dir = Path(
        args.model_dir
        or os.environ.get("L2_SHERPA_MODEL_DIR")
        or str(_DEFAULT_MODEL_DIR)
    )

    # サブディレクトリにモデルが配置されている場合を自動検出
    if not (model_dir / "tokens.txt").is_file():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir() and (d / "tokens.txt").is_file()]
        if len(subdirs) == 1:
            model_dir = subdirs[0]
            print(f"モデルサブディレクトリ検出: {model_dir.name}")

    suffix = ".int8" if args.quantized else ""
    pattern = f"encoder-epoch-*-avg-*{suffix}.onnx"
    encoder_candidates = sorted(model_dir.glob(pattern))

    if not encoder_candidates:
        for fallback in (".fp16", ""):
            fb_pattern = f"encoder-epoch-*-avg-*{fallback}.onnx"
            encoder_candidates = sorted(model_dir.glob(fb_pattern))
            if encoder_candidates:
                suffix = fallback
                break

    if encoder_candidates:
        encoder = encoder_candidates[0]
        stem = encoder.name.replace("encoder-", "").replace(".onnx", "")
        decoder = model_dir / f"decoder-{stem}.onnx"
        joiner = model_dir / f"joiner-{stem}.onnx"
    else:
        encoder = model_dir / f"encoder-epoch-99-avg-1{suffix}.onnx"
        decoder = model_dir / f"decoder-epoch-99-avg-1{suffix}.onnx"
        joiner = model_dir / f"joiner-epoch-99-avg-1{suffix}.onnx"

    tokens = model_dir / "tokens.txt"

    for p in (encoder, decoder, joiner, tokens):
        if not p.is_file():
            print(f"モデルファイルが見つかりません: {p}", file=sys.stderr)
            print()
            print("sherpa-models/ にモデルを配置してください。")
            print("例: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models")
            raise SystemExit(1)

    provider = args.provider or os.environ.get("L2_SHERPA_PROVIDER", "cuda")

    # -- Recognizer 初期化 --
    print(f"モデル読み込み中... (provider={provider})")
    print(f"  encoder: {encoder.name}")
    print(f"  decoder: {decoder.name}")
    print(f"  joiner:  {joiner.name}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        provider=provider,
    )
    print("Recognizer 初期化完了")

    # -- マイクデバイス --
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass

    devices = sd.query_devices()
    default_idx = sd.default.device[0]
    print(f"マイク: {devices[default_idx]['name']}")
    print()
    print("=" * 60)
    print("話しかけてください。Ctrl+C で終了。")
    print("=" * 60)
    print()

    # -- VAD パラメータ --
    sample_rate = 16000
    frame_samples = 512         # ~32ms
    vad_threshold = float(os.environ.get("L2_SILENCE_THRESHOLD", "0.005"))
    onset_hold = 8              # 発話開始に必要な連続フレーム数
    silence_frames = 24         # 無音終了に必要な連続フレーム数

    onset_count = 0
    silence_count = 0
    recording: list = []
    recording_started = False
    utterance_num = 0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            device=device,
        ) as stream:
            while True:
                pcm, overflowed = stream.read(frame_samples)
                frame = pcm[:, 0]
                rms = float(np.sqrt(np.mean(frame ** 2)))

                if not recording_started:
                    if rms >= vad_threshold:
                        onset_count += 1
                    else:
                        onset_count = 0

                    if onset_count >= onset_hold:
                        recording_started = True
                        silence_count = 0
                        recording = []
                        print("[VAD] 発話開始検知", end="", flush=True)
                else:
                    recording.append(frame.copy())

                    if rms < vad_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if silence_count >= silence_frames or len(recording) > 15000:
                        # 発話終了 → 認識実行
                        audio_data = np.concatenate(recording)
                        duration_ms = len(audio_data) / sample_rate * 1000

                        print(
                            f" → 発話終了 ({duration_ms:.0f}ms, "
                            f"{len(recording)} frames)"
                        )

                        # Sherpa-ONNX で認識
                        s = recognizer.create_stream()
                        s.accept_waveform(sample_rate, audio_data)
                        recognizer.decode_stream(s)
                        result = s.result.text.strip()

                        utterance_num += 1
                        if result:
                            print(f"  [{utterance_num}] {result}")
                        else:
                            print(f"  [{utterance_num}] (認識結果なし)")
                        print()

                        # リセット
                        recording_started = False
                        recording = []
                        onset_count = 0
                        silence_count = 0

    except KeyboardInterrupt:
        print("\n終了します。")


if __name__ == "__main__":
    main()
