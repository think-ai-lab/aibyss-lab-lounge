"""
calibrate_silence.py — L2_SILENCE_THRESHOLD 校正スクリプト

環境の背景ノイズを測定し、無音判定しきい値の推奨値を出力する。

【使い方】
  # mic extra をインストール済みの場合:
  uv run python scripts/calibrate_silence.py

  # 測定秒数を変更する場合:
  uv run python scripts/calibrate_silence.py --seconds 5

  # 別のマイクデバイスを指定する場合:
  uv run python scripts/calibrate_silence.py --device 2

【出力例】
  背景ノイズを 3 秒間測定します...
  ─────────────────────────────────
  RMS (平均) : 0.0012
  RMS (最大) : 0.0021
  ─────────────────────────────────
  推奨 L2_SILENCE_THRESHOLD: 0.003

  .env に以下を追加してください:
  L2_SILENCE_THRESHOLD=0.003

【しきい値の考え方】
  - 推奨値 = 背景ノイズ RMS 最大値 × 1.5（余裕係数）
  - 静かな環境: 0.001〜0.003
  - 通常環境:   0.003〜0.007
  - 騒がしい環境: 0.007 以上
  - デフォルト値: 0.005

  「無音検知で処理が中断する」場合は推奨値より低い値を試してください。
  「声を入れても反応しない」場合はしきい値が高すぎる可能性があります。
"""

import argparse
import sys


def calibrate(
    seconds: float,
    *,
    device: int | str | None,
    sample_rate: int = 16000,
) -> None:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        print(
            "❌ sounddevice / numpy が未インストールです。\n"
            "   uv sync --extra mic でインストールしてください。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"背景ノイズを {seconds:.0f} 秒間測定します...")
    print("  マイクに向かって話さず、静かにしてください。")
    print()

    try:
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:
        print(f"❌ 録音デバイスエラー: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    rms_mean = float(np.sqrt(np.mean(audio ** 2)))
    # フレームごとの RMS 最大値（ピーク検出）
    frame_size = sample_rate // 10  # 100ms フレーム
    n_frames = len(audio) // frame_size
    if n_frames > 0:
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        rms_max = float(frame_rms.max())
    else:
        rms_max = rms_mean

    # 推奨値 = ピーク RMS × 1.5（余裕係数）、小数点 4 桁に丸める
    recommended = round(rms_max * 1.5, 4)
    # 最低値 0.001 に切り上げ
    recommended = max(recommended, 0.001)

    print("─────────────────────────────────")
    print(f"  RMS (平均)  : {rms_mean:.4f}")
    print(f"  RMS (最大)  : {rms_max:.4f}")
    print("─────────────────────────────────")
    print(f"  推奨 L2_SILENCE_THRESHOLD: {recommended}")
    print()
    print("  .env に以下を追加してください:")
    print(f"    L2_SILENCE_THRESHOLD={recommended}")
    print()

    # 環境判定コメント
    if recommended < 0.003:
        print("  📊 静かな環境です。デフォルト (0.005) より低い値を推奨します。")
    elif recommended < 0.007:
        print("  📊 通常の環境です。推奨値をそのまま使えます。")
    else:
        print("  📊 騒がしい環境です。マイクの感度調整も検討してください。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="背景ノイズを測定して L2_SILENCE_THRESHOLD の推奨値を出力する"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="測定秒数 (デフォルト: 3)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="録音デバイスのインデックスまたは名前 (省略時はデフォルトデバイス)",
    )
    args = parser.parse_args()

    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # 名前指定のままにする

    calibrate(args.seconds, device=device)


if __name__ == "__main__":
    main()
