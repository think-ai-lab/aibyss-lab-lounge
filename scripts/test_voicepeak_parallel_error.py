"""
test_voicepeak_parallel_error.py — VOICEPEAK 並列実行エラーを再現する診断スクリプト

目的:
  VOICEPEAK の独特な出力方式 (親プロセス即 exit、子プロセスが非同期で stdout/stderr 出力)
  を踏まえて、並列実行エラー時に何が記録されるかを調べる。

  2 つ以上の voicepeak.exe を同時起動して、2 つ目以降がどんなエラーを返すかを観察する。

実行:
  uv run python scripts/test_voicepeak_parallel_error.py
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


VOICEPEAK = r"C:\Program Files\VOICEPEAK\voicepeak.exe"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"


def _build_cmd(text: str, out_filename: str) -> str:
    """VOICEPEAK コマンド文字列を組み立てる。"""
    out_path = AUDIO_DIR / out_filename
    return (
        f'"{VOICEPEAK}" --say "{text}"'
        f' --narrator "Asumi Ririse"'
        f' --out "{out_path}"'
        f' --speed 100'
    )


def _decode(b: bytes) -> str:
    """bytes を CP932 → UTF-8 の順で安全に decode。"""
    if not b:
        return "(empty)"
    for enc in ("cp932", "utf-8"):
        try:
            return b.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace").strip()


def run_voicepeak(label: str, text: str, out_filename: str) -> dict:
    """1 つの VOICEPEAK プロセスを実行し、結果を辞書で返す。"""
    cmd = _build_cmd(text, out_filename)
    out_path = AUDIO_DIR / out_filename

    if out_path.exists():
        out_path.unlink()

    print(f"[{label}] start: {time.strftime('%H:%M:%S')}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, shell=True)
    elapsed = time.monotonic() - t0

    return {
        "label": label,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stderr_raw": result.stderr,
        "stdout_raw": result.stdout,
        "stderr": _decode(result.stderr),
        "stdout": _decode(result.stdout),
        "wav_exists": out_path.exists(),
        "wav_size": out_path.stat().st_size if out_path.exists() else 0,
    }


def _print_result(result: dict) -> None:
    print(f"\n--- [{result['label']}] ---")
    print(f"  elapsed:    {result['elapsed']:.2f}s")
    print(f"  returncode: {result['returncode']}")
    print(f"  stderr:     {result['stderr']!r}")
    print(f"  stdout:     {result['stdout']!r}")
    print(f"  stderr len: {len(result['stderr_raw'])} bytes")
    print(f"  stdout len: {len(result['stdout_raw'])} bytes")
    print(f"  wav exists: {result['wav_exists']} (size={result['wav_size']})")


def test_sequential() -> None:
    """ベースライン: 順次実行 (両方成功するはず)。"""
    print("\n" + "=" * 60)
    print("[Test 1] 順次実行 (ベースライン)")
    print("=" * 60)

    r1 = run_voicepeak("seq-1", "順次テスト1ですわ", "parallel_seq1.wav")
    r2 = run_voicepeak("seq-2", "順次テスト2ですわ", "parallel_seq2.wav")

    _print_result(r1)
    _print_result(r2)


def test_parallel_2() -> None:
    """同時に 2 プロセス起動。2 つ目がエラーを返すか観察。"""
    print("\n" + "=" * 60)
    print("[Test 2] 並列実行 (2 プロセス)")
    print("=" * 60)

    results: list[dict] = [None, None]  # type: ignore

    def runner(idx: int, label: str, text: str, filename: str) -> None:
        results[idx] = run_voicepeak(label, text, filename)

    t1 = threading.Thread(
        target=runner, args=(0, "par-1", "並列テスト1ですわ", "parallel_par1.wav"),
    )
    t2 = threading.Thread(
        target=runner, args=(1, "par-2", "並列テスト2ですわ", "parallel_par2.wav"),
    )

    t1.start()
    time.sleep(0.05)  # ほぼ同時起動 (50ms 差)
    t2.start()

    t1.join()
    t2.join()

    _print_result(results[0])
    _print_result(results[1])


def test_parallel_3() -> None:
    """同時に 3 プロセス起動。1 つは成功、2 つは失敗を期待。"""
    print("\n" + "=" * 60)
    print("[Test 3] 並列実行 (3 プロセス)")
    print("=" * 60)

    results: list[dict] = [None, None, None]  # type: ignore

    def runner(idx: int, label: str, text: str, filename: str) -> None:
        results[idx] = run_voicepeak(label, text, filename)

    threads = [
        threading.Thread(target=runner, args=(0, "par3-1", "並列3テスト1", "parallel3_1.wav")),
        threading.Thread(target=runner, args=(1, "par3-2", "並列3テスト2", "parallel3_2.wav")),
        threading.Thread(target=runner, args=(2, "par3-3", "並列3テスト3", "parallel3_3.wav")),
    ]

    for t in threads:
        t.start()
        time.sleep(0.05)

    for t in threads:
        t.join()

    for r in results:
        _print_result(r)


def test_back_to_back_no_cooldown() -> None:
    """連続実行: 1 つ目完了直後に 2 つ目を起動 (クールダウンなし)。"""
    print("\n" + "=" * 60)
    print("[Test 4] 連続実行 (クールダウンなし、即連投)")
    print("=" * 60)

    r1 = run_voicepeak("b2b-1", "連投テスト1ですわ", "back_to_back_1.wav")
    # 一切待たずに 2 つ目を投入
    r2 = run_voicepeak("b2b-2", "連投テスト2ですわ", "back_to_back_2.wav")

    _print_result(r1)
    _print_result(r2)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    test_sequential()
    print("\n--- 5 秒待機 ---\n")
    time.sleep(5)

    test_back_to_back_no_cooldown()
    print("\n--- 5 秒待機 ---\n")
    time.sleep(5)

    test_parallel_2()
    print("\n--- 5 秒待機 ---\n")
    time.sleep(5)

    test_parallel_3()

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
