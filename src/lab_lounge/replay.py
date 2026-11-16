"""
replay.py — 判定のリプレイ（固定のセットに本物の判定の関数をかけ、答えを残して比べる）（計測の道具）

責務:
  - run:    セット（replay_set が作った data/replay/set.jsonl から除外を引いたもの + 合成の行）に、
            本物の判定の関数（check_approval / check_intent / _route_by_llm）をかけ、答えと時間を 1 行ずつ残す
  - report: 1 回の実行の、種類ごとの自己一致率（Wilson の 95% 区間）、unknown とエラーの数、混同行列、
            合成の行の正答数、当時の答えとの一致率を出す

本番の経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない。

【WHY 本物の関数を呼ぶか】
  判定の論理（system prompt の組み立て、答えの解釈、候補外の扱い）を複製すると、複製と本番がずれたときに
  リプレイが本番を測らなくなる。差し替えるのは LLM を呼ぶ 1 点（router._call_router_llm）だけで、
  そこで生の答え、時間、system prompt のハッシュを記録する。

【使い方】
  # 少数で試す（単価を実測し、全体の費用を見積もる）
  uv run --frozen --all-extras python -m lab_lounge.replay run --profile dev --label canary --canary 5 --reps 1 --max-usd 0.05

  # 本番の比べる基準を 2 回流す
  uv run --frozen --all-extras python -m lab_lounge.replay run --profile dev --label <名前> --reps 2 --max-usd 1.0

  # 集計
  uv run --frozen --all-extras python -m lab_lounge.replay report --run <名前>

【出力】（data/replay/runs/、gitignore 済み）
  <名前>.jsonl:        1 判定 1 行（id、rep、attempt、生の答え、result_key、時間、エラーの型、prompt のハッシュ、
                       応答のモデル名）。入力の本文は書かない（id でセットと結ぶ）
  <名前>.meta.json:    commit、uv.lock のハッシュ、SDK の版、モデル、フラグ、セットのハッシュ、件数
  <名前>.problems.log: 実行中の WARNING 以上のログ（判定の関数の警告は答えや本文の断片を含みうる）

【出力の約束】
  画面には、件数、率、表、時間、費用だけを出す。発話の本文と生の答えは出さない。
  判定ログ（L2_DECISION_LOG）は止め、止まっていることを確かめてから流す
  （リプレイの発話が、次のリプレイの入力に混ざらないように）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import decision_log, replay_set, router, spend
from .spend import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, run_tool

EXIT_REFUSED = 2

# 1 回の実行で既定の並列の数。WHY 4: API の待ちが大半なので並べると速いが、増やすと上限（rate limit）と
# 予約の見積もりが重なって止まりやすくなる。
DEFAULT_WORKERS = 4

# 判定に効く環境変数。meta に値を残し、再開のときに同じであることを確かめる。
_ENV_IN_META = ("L2_LLM_ROUTER_MODEL", "L2_INTENT_GATE_MODEL", "L2_USE_LLM_ROUTER", "L2_USE_INTENT_GATE",
                "L2_ROUTER_LLM_TIMEOUT_SEC", "L2_LLM_MAX_RETRIES")
_SDKS = ("anthropic", "openai", "google-genai", "httpx", "httpx2")

# 再開のときに一致を求める meta の鍵。
_RESUME_KEYS = ("uv_lock_sha", "models", "set_sha", "env")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── 記録の包み ──────────────────────────────────────────────────

_capture = threading.local()


@dataclass
class CallRecord:
    """1 回の LLM 呼び出しの記録（本文なし）。"""

    raw: str | None
    elapsed_ms: int
    error: str | None
    prompt_sha: str
    model: str


class RecordingRouter:
    """with の中だけ、router._call_router_llm を「呼んで記録する」包みに差し替える。出るときに戻す。

    記録はスレッドごとに分ける（並列の行が混ざらないように）。
    """

    def __enter__(self) -> RecordingRouter:
        self.original = router._call_router_llm
        original = self.original

        def call(model: str, system_prompt: str, user_text: str) -> str | None:
            start = time.monotonic()
            answer = original(model, system_prompt, user_text)
            elapsed = int((time.monotonic() - start) * 1000)
            records = getattr(_capture, "records", None)
            if records is not None:
                records.append(CallRecord(answer, elapsed, getattr(router._router_call_state, "error", None),
                                          decision_log.prompt_hash(system_prompt), model))
            return answer

        router._call_router_llm = call
        return self

    def __exit__(self, *exc_info: object) -> None:
        router._call_router_llm = self.original


def run_row(row: dict[str, Any], rep: int, attempt: int) -> dict[str, Any]:
    """1 行を本物の判定の関数で判定し、結果の行（本文なし）を返す。RecordingRouter の中で呼ぶ。"""
    _capture.records = []
    spend.last_call()  # 前の行の名残を消す
    try:
        value = replay_set.judge(row)
        key = replay_set.result_key(row["kind"], value)
        failure = None
    except spend.SpendStop:
        raise
    except Exception as exc:  # noqa: BLE001 - 判定の関数の外の失敗も、行の失敗として残す（型だけ）
        key, failure = None, type(exc).__name__
    records: list[CallRecord] = _capture.records
    _capture.records = None
    last = records[-1] if records else None
    error = failure or (last.error if last else "NoCall")
    if error:
        key = f"error:{error}"
    wire = spend.last_call()
    return {
        "id": row["id"], "rep": rep, "attempt": attempt, "raw": last.raw if last else None, "result_key": key,
        "elapsed_ms": last.elapsed_ms if last else None, "error": error,
        "prompt_sha": last.prompt_sha if last else None, "model": last.model if last else None,
        "model_on_wire": wire.get("model_on_wire") if wire else None, "calls": len(records),
    }


# ─── セットと meta ───────────────────────────────────────────────

def load_rows(set_dir: Path, synthetic: Path | None) -> tuple[list[dict[str, Any]], int, str | None]:
    """流す行（セットから除外を引いたもの + 合成の行）を、id の順で返す。

    Returns:
        (行, 除外した数, 除外の一覧が古いときの警告)
    """
    rows = replay_set.load_set(set_dir)
    excluded, warning = replay_set.load_exclusions(set_dir)
    kept = [row for row in rows if row["id"] not in excluded]
    dropped = len(rows) - len(kept)
    kept.extend(replay_set.load_synthetic(synthetic))
    ids = [row["id"] for row in kept]
    if len(ids) != len(set(ids)):
        raise ValueError("セットと合成の行で id が重なっています")
    return sorted(kept, key=lambda r: r["id"]), dropped, warning


def set_sha(rows: Iterable[dict[str, Any]]) -> str:
    """流す行の中身（id、種類、入力、対象、候補、期待値）のハッシュ。セットが変わったことを見分ける。"""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        part = [row["id"], row["kind"], row["input"], row.get("character"), row.get("candidates"),
                row.get("expected"), row.get("alt_ok")]
        digest.update(json.dumps(part, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
    return digest.hexdigest()[:16]


def canary_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """種類（区分）を順にめぐって count 行を選ぶ。固定の順なので、同じセットなら同じ行になる。"""
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        queues[replay_set.bucket(row)].append(row)
    order = [name for name in replay_set.TARGETS if queues[name]]
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count and any(queues[name] for name in order):
        for name in order:
            if queues[name] and len(chosen) < count:
                chosen.append(queues[name].pop(0))
    return chosen


def _git_commit() -> str | None:
    try:
        done = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                              capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


def build_meta(label: str, rows: list[dict[str, Any]], *, reps: int, excluded: int) -> dict[str, Any]:
    """実行の meta。値は設定と版と数だけ（本文なし）。"""
    versions = {}
    for name in _SDKS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "label": label, "started": datetime.now(UTC).isoformat(timespec="seconds"), "commit": _git_commit(),
        "uv_lock_sha": _file_sha(_REPO_ROOT / "uv.lock"), "sdk": versions,
        "models": {
            "router": os.environ.get("L2_LLM_ROUTER_MODEL", "claude-haiku-4-5"),
            "intent_gate": os.environ.get("L2_INTENT_GATE_MODEL", "claude-haiku-4-5"),
        },
        "env": {name: os.environ.get(name) for name in _ENV_IN_META},
        "set_sha": set_sha(rows), "rows": len(rows), "reps": reps, "excluded": excluded,
        "by_bucket": dict(Counter(replay_set.bucket(r) for r in rows)),
    }


# ─── 実行 ────────────────────────────────────────────────────────

class RunFile:
    """結果の行を 1 行ずつ追記する（並列のスレッドから呼ぶ）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def lines(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, line: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def _final_by_key(lines: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """(id, rep) ごとに、いちばん後の試みの行を返す。"""
    final: dict[tuple[str, int], dict[str, Any]] = {}
    for line in lines:
        key = (line["id"], line["rep"])
        if key not in final or line["attempt"] >= final[key]["attempt"]:
            final[key] = line
    return final


def execute(rows: list[dict[str, Any]], run_file: RunFile, *, reps: int, workers: int,
            meter: spend.UsageMeter | None = None,
            progress: Callable[[str], None] = emit) -> Counter:
    """rep 1 を全部流してから rep 2 …。失敗の行は最後に 1 回だけやり直す。済んだ (id, rep) は飛ばす（再開）。

    Raises:
        spend.SpendStop: 上限などで止めた（残りの行は流さない）
        RuntimeError: 判定ログがオン
    """
    if decision_log.is_enabled():
        raise RuntimeError("判定ログがオンのままです（リプレイの発話が次の入力に混ざる）")
    done = _final_by_key(run_file.lines())
    stats: Counter = Counter()
    calls_made = 0

    def run_batch(batch: list[tuple[dict[str, Any], int, int]], title: str) -> None:
        nonlocal calls_made
        if not batch:
            return
        with RecordingRouter(), concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_row, row, rep, attempt) for row, rep, attempt in batch]
            try:
                for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    line = future.result()
                    run_file.append(line)
                    stats["error" if line["error"] else "ok"] += 1
                    calls_made += 0 if line["error"] else line["calls"]
                    if count % 50 == 0 or count == len(batch):
                        progress(f"{title}: {count} / {len(batch)}")
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
        if meter is not None:
            meter.check_capture(calls_made)

    for rep in range(1, reps + 1):
        run_batch([(row, rep, 1) for row in rows if (row["id"], rep) not in done], f"rep {rep}")
    done = _final_by_key(run_file.lines())
    retry = [(row, rep, 2) for rep in range(1, reps + 1) for row in rows
             if (row["id"], rep) in done and done[(row["id"], rep)]["error"] and done[(row["id"], rep)]["attempt"] == 1]
    run_batch(retry, "やり直し")
    return stats


# ─── 集計 ────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項の割合の Wilson の区間（既定は 95%）。n=0 なら (0, 1)。

    WHY Wilson: 一致率が 100% に近く件数が少ないとき、正規近似の区間は 1 を超えたり幅 0 になったりする。
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class Report:
    """1 回の実行の集計（数と結果の鍵だけ）。"""

    label: str
    rows: int
    reps: int
    agreement: dict[str, tuple[int, int]]
    unknown: Counter
    errors: Counter
    confusion: dict[str, Counter]
    synthetic: dict[str, tuple[int, int]]
    historical: dict[str, tuple[int, int]]
    missing: int


def make_report(label: str, rows: list[dict[str, Any]], lines: list[dict[str, Any]], reps: int) -> Report:
    """実行の行から集計を作る。"""
    final = _final_by_key(lines)
    agreement: dict[str, tuple[int, int]] = {}
    unknown: Counter = Counter()
    errors: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    synthetic: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    historical: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pairs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    missing = 0
    for row in rows:
        name = replay_set.bucket(row)
        keys = [final.get((row["id"], rep), {}).get("result_key") for rep in range(1, reps + 1)]
        missing += sum(key is None for key in keys)
        for key in keys:
            if key is None:
                continue
            if key.startswith("error:"):
                errors[name] += 1
            elif key == "unknown":
                unknown[name] += 1
        first = keys[0] if keys else None
        if reps >= 2 and all(k is not None and not k.startswith("error:") for k in keys[:2]):
            pairs[name][1] += 1
            pairs[name][0] += keys[0] == keys[1]
        if first is None or first.startswith("error:"):
            continue
        if row.get("source") == "synthetic":
            synthetic[name][1] += 1
            synthetic[name][0] += first == row["expected"] or first in row.get("alt_ok", [])
            confusion[name][f"期待 {row['expected']} → {first}"] += 1
        elif row.get("hist_key") is not None:
            historical[name][1] += 1
            historical[name][0] += first == row["hist_key"]
            confusion[name][f"当時 {row['hist_key']} → {first}"] += 1
        else:
            confusion[name][f"当時なし → {first}"] += 1
    return Report(label, len(rows), reps, {k: (v[0], v[1]) for k, v in pairs.items()}, unknown, errors,
                  dict(confusion), {k: (v[0], v[1]) for k, v in synthetic.items()},
                  {k: (v[0], v[1]) for k, v in historical.items()}, missing)


def print_report(report: Report) -> None:
    """集計を画面へ出す（区分、数、率、結果の鍵だけ）。"""
    emit(f"実行 {report.label}: {report.rows} 行 × {report.reps} 回（結果の無い組 {report.missing}）")
    emit("自己一致率（rep 1 と rep 2、エラーを除く。Wilson の 95% 区間）:")
    for name in replay_set.TARGETS:
        if name in report.agreement:
            k, n = report.agreement[name]
            low, high = wilson(k, n)
            emit(f"  {name}: {k} / {n} = {k / n if n else 0:.1%}（{low:.1%}〜{high:.1%}）")
    emit("unknown / エラー（全 rep の合計）:")
    for name in replay_set.TARGETS:
        emit(f"  {name}: unknown {report.unknown[name]}  エラー {report.errors[name]}")
    emit("合成の行の正答（rep 1）:")
    for name, (k, n) in sorted(report.synthetic.items()):
        emit(f"  {name}: {k} / {n}")
    emit("当時の答えとの一致（rep 1、参考）:")
    for name, (k, n) in sorted(report.historical.items()):
        emit(f"  {name}: {k} / {n}")
    emit("混同行列（rep 1）:")
    for name in replay_set.TARGETS:
        for cell, count in sorted(report.confusion.get(name, Counter()).items()):
            emit(f"  {name}: {cell}: {count}")


# ─── コマンドライン ──────────────────────────────────────────────

def _runs_dir(set_dir: Path) -> Path:
    return set_dir / "runs"


def _cmd_run(args: argparse.Namespace) -> int:
    from .preflight import _captured_logs, prepare_env, silence_side_outputs

    set_dir = args.set_dir or replay_set.default_out_dir()
    runs = _runs_dir(set_dir)
    run_path, meta_path = runs / f"{args.label}.jsonl", runs / f"{args.label}.meta.json"
    if run_path.exists() and not args.resume:
        emit(f"実行 {args.label} はすでにあります（続けるなら --resume、別の名前なら --label）")
        return EXIT_REFUSED

    prepare_env(args.profile, _REPO_ROOT)
    silence_side_outputs()
    if decision_log.is_enabled():
        emit("判定ログを止められませんでした")
        return EXIT_REFUSED

    rows, excluded, warning = load_rows(set_dir, args.synthetic)
    if warning:
        emit(f"[警告] {warning}")
    if args.canary:
        rows = canary_rows(rows, args.canary)
    meta = build_meta(args.label, rows, reps=args.reps, excluded=excluded)
    if args.resume and meta_path.is_file():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = [key for key in _RESUME_KEYS if previous.get(key) != meta.get(key)]
        if changed:
            emit(f"再開できません（前の回と違う: {', '.join(changed)}）")
            return EXIT_REFUSED
        meta = previous
    else:
        runs.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    overrides = dict(spend.parse_price_override(text) for text in args.price)
    ledger = spend.SpendLedger(args.spend_dir or spend.default_spend_dir())
    emit(f"実行 {args.label}: {len(rows)} 行 × {args.reps} 回、並列 {args.workers}、上限 ${args.max_usd:.2f}"
         f"（除外 {excluded}）")
    started = time.monotonic()
    with _captured_logs(runs / f"{args.label}.problems.log"), \
            spend.UsageMeter(ledger, args.label, max_usd=args.max_usd, overrides=overrides) as meter:
        before = ledger.run_usd(args.label)
        stats = execute(rows, RunFile(run_path), reps=args.reps, workers=args.workers, meter=meter)
        spent = ledger.run_usd(args.label) - before
    wire = Counter(json.loads(line).get("model_on_wire") for line in
                   ledger.calls_path.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("run") == args.label)
    meta["model_on_wire"] = {str(k): v for k, v in wire.items()}
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    judged = stats["ok"] + stats["error"]
    emit(f"判定 {judged} 件（エラー {stats['error']}）、{time.monotonic() - started:.0f} 秒、"
         f"今回 ${spent:.4f}（1 件あたり ${spent / judged if judged else 0:.5f}）、"
         f"usage を拾えず見積もりで数えた呼び出し {meter.stats.uncounted}")
    emit("応答のモデル名: " + ", ".join(f"{spend._showable(k)} {v}" for k, v in sorted(wire.items(), key=str)))
    print_report(make_report(args.label, rows, RunFile(run_path).lines(), args.reps))
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    set_dir = args.set_dir or replay_set.default_out_dir()
    runs = _runs_dir(set_dir)
    meta_path = runs / f"{args.run}.meta.json"
    if not meta_path.is_file():
        emit(f"実行 {args.run} の meta がありません")
        return EXIT_FAILED
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows, _, _ = load_rows(set_dir, args.synthetic)
    lines = RunFile(runs / f"{args.run}.jsonl").lines()
    ran = {line["id"] for line in lines}
    rows = [row for row in rows if row["id"] in ran]
    # 途中で止まった実行は流した行が少ないので、行の数がそろうときだけ中身を比べる。
    if len(rows) != meta["rows"]:
        emit(f"[警告] 結果のある行 {len(rows)} / 実行のときの行 {meta['rows']}（途中で止まったか、セットが変わった）")
    elif set_sha(rows) != meta["set_sha"]:
        emit("[警告] セットの中身が実行のときと違います")
    print_report(make_report(args.run, rows, lines, int(meta["reps"])))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab_lounge.replay", description="判定のリプレイ")
    parser.add_argument("--set-dir", type=Path, default=None, help="セットの場所（既定は data/replay/）")
    parser.add_argument("--synthetic", type=Path, default=None, help="合成の行（既定は measure_sets/ の中）")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="セットに本物の判定の関数をかける（課金あり）")
    run.add_argument("--profile", choices=("dev", "prod"), required=True)
    run.add_argument("--label", required=True, help="実行の名前（英数字と _.- で 64 字まで）")
    run.add_argument("--reps", type=int, default=2)
    run.add_argument("--canary", type=int, default=0, help="種類をめぐって N 行だけ流す")
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--max-usd", type=float, required=True, help="この実行の上限（同じ名前の過去の分を含む）")
    run.add_argument("--resume", action="store_true", help="済んだ行を飛ばして続ける")
    run.add_argument("--price", action="append", default=[], help="model=in,out[,cache_read[,cache_write]]")
    run.add_argument("--spend-dir", type=Path, default=None, help=argparse.SUPPRESS)
    report = sub.add_parser("report", help="1 回の実行の集計")
    report.add_argument("--run", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.replay <run|report>``。"""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.command == "run":
        if args.reps < 1 or args.workers < 1 or args.canary < 0:
            emit("--reps と --workers は 1 以上、--canary は 0 以上にしてください")
            return EXIT_USAGE
        return _cmd_run(args)
    return _cmd_report(args)


if __name__ == "__main__":
    run_tool(main, name="replay")
