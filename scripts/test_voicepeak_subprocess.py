"""
test_voicepeak_subprocess.py — VOICEPEAK の subprocess 挙動を診断する。

目的:
  - subprocess.run が即座に戻るか、WAV 生成まで待つかを計測
  - parent exit のタイミングと WAV ファイル生成のタイミングを比較
  - stderr/stdout の内容を確認

実行:
  uv run python scripts/test_voicepeak_subprocess.py
"""

import os
import subprocess
import time
from pathlib import Path


def _test(label: str, cmd_str: str, out_path: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{label}]")
    print(f"cmd: {cmd_str[:120]}{'...' if len(cmd_str) > 120 else ''}")

    # 既存ファイル削除
    if out_path.exists():
        out_path.unlink()

    t0 = time.monotonic()
    result = subprocess.run(cmd_str, capture_output=True, shell=True)
    t1 = time.monotonic()
    subprocess_elapsed = t1 - t0

    # subprocess 完了直後に WAV 存在チェック
    exists_immediately = out_path.exists()
    size_immediately = out_path.stat().st_size if exists_immediately else 0

    print(f"subprocess.run elapsed: {subprocess_elapsed:.3f}s")
    print(f"returncode: {result.returncode}")
    print(f"stderr ({len(result.stderr)} bytes): {result.stderr[:200]!r}")
    print(f"stdout ({len(result.stdout)} bytes): {result.stdout[:200]!r}")
    print(f"WAV exists immediately: {exists_immediately} (size={size_immediately})")

    # もし存在しなければポーリング
    if not exists_immediately or size_immediately < 1024:
        print("Polling for WAV file...")
        for i in range(40):  # 最大 20 秒
            time.sleep(0.5)
            if out_path.exists() and out_path.stat().st_size > 1024:
                elapsed_since_subprocess = time.monotonic() - t1
                final_size = out_path.stat().st_size
                print(
                    f"  WAV ready after {elapsed_since_subprocess:.1f}s "
                    f"since subprocess.run returned (size={final_size})"
                )
                break
        else:
            print("  WAV never appeared (20s timeout)")


def main() -> None:
    voicepeak = r"C:\Program Files\VOICEPEAK\voicepeak.exe"
    audio_dir = Path(__file__).resolve().parent.parent / "data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: 短いテキスト + emotion なし
    out1 = audio_dir / "subproc_test1.wav"
    cmd1 = (
        f'"{voicepeak}" --say "テストですわ"'
        f' --narrator "Asumi Ririse"'
        f' --out "{out1}"'
    )
    _test("Test 1: short text, no emotion", cmd1, out1)

    # Test 2: 長いテキスト + emotion
    out2 = audio_dir / "subproc_test2.wav"
    long_text = (
        "まあ、それはなんて素敵なお知らせですの。"
        "わたくしまで胸がふわりと明るくなりましたわ。"
        "もしよろしければ、その嬉しさを少し分けてくださるかしら。"
        "アビスの底で見つけるひかりのようなお話、ぜひ伺いたいですわ。"
    )
    cmd2 = (
        f'"{voicepeak}" --say "{long_text}"'
        f' --narrator "Asumi Ririse"'
        f' --out "{out2}"'
        f' --speed 95'
        f' --emotion happy=88,fun=42,angry=0,sad=0,sulky=0'
    )
    _test("Test 2: long text + emotion (error repro)", cmd2, out2)

    # Test 3: 中くらいのテキスト + emotion なし
    out3 = audio_dir / "subproc_test3.wav"
    cmd3 = (
        f'"{voicepeak}" --say "{long_text}"'
        f' --narrator "Asumi Ririse"'
        f' --out "{out3}"'
    )
    _test("Test 3: long text, no emotion", cmd3, out3)

    # クリーンアップ（コメントアウト: ログ確認後に手動削除可）
    # for p in [out1, out2, out3]:
    #     if p.exists():
    #         p.unlink()


if __name__ == "__main__":
    main()
