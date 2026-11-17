"""
c2_probe.py — C2（記憶検索）の時間の物差し（計測の道具）

責務:
  - latency:  合成の問い合わせを、C2 の /retrieve（と /retrieve/recent）へ、本番の retriever と同じ形の要求で
              送り、cold と warm の分布、しきい値を超えた割合、予算の目安を出す（読むだけ。書き込まない）
  - timeouts: 実走ログの「C2Retriever タイムアウト」と「記憶検索実行」を月ごとに数える。
              2026-10-31（接続先を 127.0.0.1 に直した日）の前後でも分ける

本番の経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない
（本番のモジュールがこれを import しないことはテストで固定している）。課金はない（C2 の側の埋め込みの費用は
C2 が呼ぶので、ここでは問い合わせの数だけを meta に残す）。

【WHY 本番の retriever と同じ形の要求か】
  C2Retriever は GET {base}/retrieve?q=..&k=.. を、リクエストごとに新しい httpx.Client で送る（接続を使い回さない）。
  RecentC2Retriever は GET {base}/retrieve/recent?k=..。形が違うと、本番で起きる遅さを測れない。
  要求の形はテストで、本物の retriever が偽サーバへ送るものと一致することを確かめている。

【使い方】
  uv run --frozen --all-extras python -m lab_lounge.c2_probe latency --profile dev --label <名前>
  uv run --frozen --all-extras python -m lab_lounge.c2_probe timeouts --label <名前>

  latency の送り先は、--profile の env（L2_C2_URL と、あれば L2_C2_URL_READONLY）。ポート 8100 を prod、8101 を dev と
  呼ぶ（run スクリプトの規約）。最初に /health を見て、届かないものがあれば何も送らずに終わる（サービスは起動しない）。
  問い合わせは measure_sets/c2_queries.jsonl（--queries で差し替え）。

【終了コード】
  0: 正常 / 1: 失敗（使える実走ログが無い、など）/ 2: 断った（問い合わせが無い）/ 3: 前提サービスなし（/health が届かない）
  64: コマンドラインの誤り / 130: Ctrl+C

【出力】（data/probe/、gitignore 済み）
  c2_latency_<名前>.jsonl:      1 要求 1 行（送り先の名前、道、問い合わせの id、周、時間、結果の数）。問い合わせの本文なし
  c2_latency_<名前>.meta.json:  設定、問い合わせの数、集計、予算の目安
  c2_timeouts_<名前>.json:      月ごと、10-31 の前後ごとの数

【出力の約束】
  画面には、ホストとポート、件数、時間、割合だけを出す。URL の全体、鍵、問い合わせの本文は出さない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import probe, replay, replay_set, spend
from .spend import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, run_tool

EXIT_REFUSED = 2
EXIT_MISSING_SERVICE = 3  # preflight の EXIT_MISSING_SERVICE と同じ値

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PASSES = 5
# WHY 10 秒: 本番の予算（既定 600 ms）で打ち切ると、遅い要求の本当の長さが見えない。分布を見るために長く待つ。
DEFAULT_TIMEOUT_SEC = 10.0
# このしきい値（ms）を超えた割合を出す。600 は本番の既定の予算。
THRESHOLDS_MS = (200, 300, 400, 600, 800, 1000, 1500)
HEALTH_TIMEOUT_SEC = 2.0

# 本番の組み立て（graph._build_retriever_from_env / retrieve_memory）の既定。
DEFAULT_RAG_TOP_K = 5
DEFAULT_RECENT_TOP_K = 10

# 127.0.0.1 に直した日。これより前のログは localhost（::1 を先に試して約 2 秒失う）の影響を含みうる。
CUTOFF_DAY = "2026-10-31"

_PORT_ROLES = {8100: "prod", 8101: "dev"}
_LOOPBACK_NAMES = ("127.0.0.1", "localhost")
_QUERY_SHAPES = ("short", "long")
_QUERY_LIKELY = ("hit", "miss")


# ─── 送り先と要求の形 ────────────────────────────────────────────

@dataclass(frozen=True)
class Target:
    """送り先の C2。url は鍵を含みうるので画面へ出さない（shown だけを出す）。"""

    role: str   # primary / readonly（L2_C2_URL / L2_C2_URL_READONLY）
    url: str

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""

    @property
    def port(self) -> int | None:
        try:
            return urlsplit(self.url).port
        except ValueError:
            return None

    @property
    def label(self) -> str:
        """prod / dev / other（ポートで決める）と役割。"""
        return f"{_PORT_ROLES.get(self.port or 0, 'other')}({self.role})"

    @property
    def shown(self) -> str:
        """画面へ出してよい形（ホストとポートだけ）。"""
        return f"{self.host}:{self.port}" if self.port else self.host

    def with_host(self, host: str) -> Target:
        """ホスト名だけを差し替えた送り先（localhost と 127.0.0.1 の比較用）。"""
        parts = urlsplit(self.url)
        netloc = host if parts.port is None else f"{host}:{parts.port}"
        return Target(self.role, urlunsplit((parts.scheme, netloc, parts.path, "", "")))


def targets_from_env() -> list[Target]:
    """env（L2_C2_URL、L2_C2_URL_READONLY）から送り先を作る。既定は graph と同じ。"""
    primary = os.environ.get("L2_C2_URL", "http://127.0.0.1:8100").rstrip("/")
    found = [Target("primary", primary)]
    readonly = os.environ.get("L2_C2_URL_READONLY", "").strip().rstrip("/")
    if readonly:
        found.append(Target("readonly", readonly))
    return found


def sub_top_k() -> int:
    """retrieve_memory → CompositeRetriever が各 retriever へ渡す k（max(top_k * 3, 10)）。"""
    return max(int(os.environ.get("L2_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))) * 3, 10)


def request_for(kind: str, query: str, k: int) -> tuple[str, dict[str, str | int]]:
    """本番の retriever と同じ形の要求（道、引数）を返す。

    - retrieve: C2Retriever。GET /retrieve?q=<query>&k=<k>
    - recent:   RecentC2Retriever（global）。GET /retrieve/recent?k=<max(k, 既定の件数)>（query は使わない）
    exclude_event_ids と stream_id は付けない（計測の要求に、現ターンの event も stream も無い）。
    """
    if kind == "retrieve":
        return "/retrieve", {"q": query, "k": k}
    if kind == "recent":
        recent = int(os.environ.get("L2_C2_RECENT_TOP_K", str(DEFAULT_RECENT_TOP_K)))
        return "/retrieve/recent", {"k": max(k, recent)}
    raise ValueError(f"知らない道: {kind}")


def c2_retriever_count() -> int:
    """1 回の記憶検索で順に呼ばれる C2 の retriever の数（graph._build_retriever_from_env の組み立てと同じ）。"""
    count = 1  # C2Retriever(primary)
    if os.environ.get("L2_USE_C2_RECENT", "true").lower() == "true":
        count += 1
    if os.environ.get("L2_C2_URL_READONLY", "").strip():
        count += 1
    return count


# ─── 問い合わせ（合成） ──────────────────────────────────────────

@dataclass(frozen=True)
class Query:
    id: str
    text: str
    shape: str
    likely: str


def default_queries_path() -> Path:
    return probe.measure_sets_dir() / "c2_queries.jsonl"


def load_queries(path: Path | None = None) -> list[Query]:
    """合成の問い合わせを読む。形: ``{"id": "q01", "query": "...", "shape": "short|long", "likely": "hit|miss"}``。

    Raises:
        MeasureInputError: ファイルが無い、形が違う、id が重なる（行番号だけを知らせる。本文は出さない）
    """
    path = path or default_queries_path()
    if not path.is_file():
        raise probe.MeasureInputError(f"問い合わせ（{path.name}）がありません")
    items: list[Query] = []
    bad: list[int] = []
    seen: set[str] = set()
    for number, line in enumerate(probe.read_text_checked(path).split("\n"), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            bad.append(number)
            continue
        valid = (isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] not in seen
                 and isinstance(item.get("query"), str) and item["query"].strip()
                 and item.get("shape") in _QUERY_SHAPES and item.get("likely") in _QUERY_LIKELY)
        if not valid:
            bad.append(number)
            continue
        seen.add(item["id"])
        items.append(Query(item["id"], item["query"], item["shape"], item["likely"]))
    if bad:
        raise probe.MeasureInputError(f"問い合わせの形が違います（{path.name} の {', '.join(map(str, bad))} 行目）")
    if not items:
        raise probe.MeasureInputError(f"問い合わせ（{path.name}）が空です")
    return items


# ─── 計測 ────────────────────────────────────────────────────────

@dataclass
class Sample:
    """1 要求の記録。本文は持たない。"""

    group: str       # 送り先の名前（prod(primary) など。ホスト比較は @ホスト名つき）
    kind: str        # retrieve / recent
    query_id: str
    pass_no: int     # 1 周目 = cold、2 周目以降 = warm
    elapsed_ms: float
    ok: bool
    error: str | None
    results: int | None

    @property
    def phase(self) -> str:
        return "cold" if self.pass_no == 1 else "warm"

    def line(self) -> dict[str, Any]:
        return {"group": self.group, "kind": self.kind, "query_id": self.query_id, "pass": self.pass_no,
                "ms": round(self.elapsed_ms, 1), "ok": self.ok, "error": self.error, "results": self.results}


def get_once(target: Target, kind: str, query: Query, k: int, timeout_sec: float, group: str, pass_no: int) -> Sample:
    """要求を 1 回送って時間を測る。本番と同じく、リクエストごとに新しい httpx.Client（接続を使い回さない）。

    失敗は型の名前だけを残す（例外の文言に URL が入りうる）。時間切れは待った時間がそのまま入る。
    """
    import httpx

    path, params = request_for(kind, query.text, k)
    started = time.monotonic()
    results: int | None = None
    error: str | None = None
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.get(f"{target.url}{path}", params=params)
            response.raise_for_status()
            data = response.json()
        results = len(data.get("results", [])) if isinstance(data, dict) else 0
    except Exception as exc:  # noqa: BLE001 - 失敗の型だけを残す
        error = type(exc).__name__
    elapsed = (time.monotonic() - started) * 1000
    return Sample(group, kind, query.id, pass_no, elapsed, error is None, error, results)


def check_health(target: Target, timeout_sec: float = HEALTH_TIMEOUT_SEC) -> str | None:
    """/health を見る。届いて 2xx なら None、そうでなければ理由（状態コードか例外の型）を返す。"""
    import httpx

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.get(f"{target.url}/health")
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    return None if 200 <= response.status_code < 300 else f"HTTP-{response.status_code}"


def percentile_table(values: list[float]) -> dict[str, float | None]:
    return {name: probe.percentile(values, q) for name, q in
            (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))} | {"max": max(values) if values else None}


def summarize(samples: list[Sample]) -> dict[str, Any]:
    """分布としきい値を超えた割合。

    分布（p50〜最大）は応答を受けた要求だけで数える。しきい値を超えた割合は、失敗（時間切れや接続の失敗）も
    「超えた」に数える（待たされた側から見れば、どのしきい値も超えているため）。
    """
    ok = [s.elapsed_ms for s in samples if s.ok]
    failed = len(samples) - len(ok)
    over = {}
    for limit in THRESHOLDS_MS:
        count = sum(1 for s in samples if (not s.ok) or s.elapsed_ms > limit)
        over[str(limit)] = count / len(samples) if samples else None
    return {"n": len(samples), "failed": failed, **percentile_table(ok), "over": over,
            "errors": dict(Counter(s.error for s in samples if s.error))}


def budget_hint(warm_ok_ms: list[float]) -> int | None:
    """warm の 99% が収まる最小の値（100 ms 刻みで切り上げ）。データが無ければ None。"""
    p99 = probe.percentile(warm_ok_ms, 0.99)
    return None if p99 is None else int(math.ceil(p99 / 100.0) * 100) or 100


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def print_summary(group: str, kind: str, phase: str, summary: dict[str, Any]) -> None:
    over = " ".join(f"{limit}:{'-' if v is None else f'{v * 100:.0f}%'}" for limit, v in summary["over"].items())
    emit(f"  {group} {kind} {phase}: n={summary['n']} 失敗={summary['failed']}  p50={_fmt(summary['p50'])} "
         f"p90={_fmt(summary['p90'])} p95={_fmt(summary['p95'])} p99={_fmt(summary['p99'])} 最大={_fmt(summary['max'])} ms  "
         f"超えた割合(ms) {over}")


def _selected_targets(args: argparse.Namespace, found: list[Target]) -> list[Target]:
    wanted = {name for name in args.targets.split(",") if name}
    return [t for t in found if t.role in wanted]


def _cmd_latency(args: argparse.Namespace) -> int:
    from .preflight import prepare_env

    if not spend._RUN_NAME.fullmatch(args.label):
        emit("--label は英数字と _.- で 64 字までにしてください")
        return EXIT_USAGE
    out_dir = args.out_dir or spend.default_spend_dir()
    run_path = out_dir / f"c2_latency_{args.label}.jsonl"
    if run_path.exists():
        emit(f"実行 {args.label} はすでにあります（別の名前で --label）")
        return EXIT_REFUSED
    try:
        queries = load_queries(args.queries)
    except probe.MeasureInputError as exc:  # 自分で作った文言（行番号と名前だけ）
        emit(f"[断りました] {exc}")
        return EXIT_REFUSED

    prepare_env(args.profile, _REPO_ROOT)
    targets = _selected_targets(args, targets_from_env())
    if not targets:
        emit("送り先がありません（--targets は primary / readonly のいくつか。readonly は L2_C2_URL_READONLY が要ります）")
        return EXIT_USAGE
    # /health: 届かないものがあれば、問い合わせを 1 本も送らずに止める（サービスは起動しない）。
    unhealthy = [(t, reason) for t in targets if (reason := check_health(t)) is not None]
    for target, reason in unhealthy:
        emit(f"[前提サービスなし] {target.label} {target.shown} の /health が届きません（{spend._showable(reason)}）")
    if unhealthy:
        emit("何も送らずに止めました。C2 を起動してから流し直してください（この道具はサービスを起動しません）")
        return EXIT_MISSING_SERVICE

    k = args.k or sub_top_k()
    writer = replay.RunFile(run_path)
    emit(f"c2 latency {args.label}: 送り先 " + "、".join(f"{t.label} {t.shown}" for t in targets)
         + f"、問い合わせ {len(queries)} 本 × {args.passes} 周、k={k}、時間切れ {args.timeout_sec:.0f} 秒")

    samples: list[Sample] = []

    def record(sample: Sample) -> None:
        samples.append(sample)
        writer.append(sample.line())

    with probe.prevent_sleep():
        # 周 → 問い合わせ → 送り先の順に交互に送る（時刻の偏りが送り先の差に出ないように）。
        for pass_no in range(1, args.passes + 1):
            for query in queries:
                for target in targets:
                    record(get_once(target, "retrieve", query, k, args.timeout_sec, target.label, pass_no))
                    if target.role == "primary":  # 本番は readonly に対して recent を呼ばない
                        record(get_once(target, "recent", query, k, args.timeout_sec, target.label, pass_no))
        if not args.no_host_compare:
            _compare_hosts(targets, queries, k, args.timeout_sec, record)

    summaries: dict[str, dict[str, Any]] = {}
    hints: dict[str, int | None] = {}
    for group in sorted({s.group for s in samples}):
        for kind in ("retrieve", "recent"):
            for phase in ("cold", "warm"):
                chosen = [s for s in samples if (s.group, s.kind, s.phase) == (group, kind, phase)]
                if not chosen:
                    continue
                key = f"{group}/{kind}/{phase}"
                summaries[key] = summarize(chosen)
                print_summary(group, kind, phase, summaries[key])
            if "@" not in group:
                warm_ok = [s.elapsed_ms for s in samples if (s.group, s.kind, s.phase, s.ok) == (group, kind, "warm", True)]
                if warm_ok:
                    hints[f"{group}/{kind}"] = budget_hint(warm_ok)
    retrievers = c2_retriever_count()
    for key, hint in hints.items():
        if hint is not None:
            emit(f"  予算の目安 {key}: warm の p99 を収める最小 {hint} ms（100 ms 刻み）。"
                 f"retriever {retrievers} 本が順に最悪の時間を使うと {hint * retrievers} ms")
    meta = {
        "tool": "c2_latency", "label": args.label, "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": probe.git_commit(), "profile": args.profile,
        "targets": [{"label": t.label, "host": t.host, "port": t.port} for t in targets],
        "queries": {"n": len(queries), "sha": probe.digest([(q.id, q.text, q.shape, q.likely) for q in queries])},
        "passes": args.passes, "k": k, "timeout_sec": args.timeout_sec, "thresholds_ms": list(THRESHOLDS_MS),
        "host_compare": not args.no_host_compare, "retrievers_per_search": retrievers,
        # C2 が埋め込み（有料）を呼ぶ /retrieve の数。支出の記録の外なので、数だけを残す。
        "embedding_calls_estimate": sum(1 for s in samples if s.kind == "retrieve"),
        "summary": summaries, "budget_hint_ms": hints,
        "notes": ["cold = 1 周目（その問い合わせが初めて）、warm = 2 周目以降", "リクエストごとに新しい httpx.Client（本番と同じ）"],
    }
    probe._write_json(out_dir / f"c2_latency_{args.label}.meta.json", meta)
    failed = sum(1 for s in samples if not s.ok)
    emit(f"合計 {len(samples)} 要求、失敗 {failed}、C2 の埋め込みの呼び出し（概算）{meta['embedding_calls_estimate']}")
    return EXIT_OK


def _compare_hosts(targets: list[Target], queries: list[Query], k: int, timeout_sec: float,
                   record: Any) -> None:
    """loopback の送り先を、localhost と 127.0.0.1 で 1 周ずつ比べる（::1 を先に試して遅くなる件の確認）。"""
    for target in targets:
        if target.host not in _LOOPBACK_NAMES:
            continue
        for name in _LOOPBACK_NAMES:
            variant = target.with_host(name)
            for query in queries:
                record(get_once(variant, "retrieve", query, k, timeout_sec, f"{target.label}@{name}", 1))


# ─── timeouts ────────────────────────────────────────────────────

_TIMEOUT_LINE = re.compile(r"^(?P<who>Recent)?C2Retriever タイムアウト \(timeout_s=(?P<t>[0-9.]+)\)")
_HTTP_ERROR_LINE = re.compile(r"^(?P<who>Recent)?C2Retriever HTTP エラー")
_SEARCH_PREFIX = "記憶検索実行: "


def timeouts_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """実走ログから、月ごと（YYYY-MM）と 10-31 の前後（before / after）ごとに数える。

    読むのは各行の先頭の形だけ（記憶検索実行の行の問い合わせの本文は見ない）。
    """
    buckets: dict[str, Counter] = {}
    for path in paths:
        month, day = probe.log_month(path), probe.log_day(path)
        if month is None or day is None:
            continue
        side = "before" if day < CUTOFF_DAY else "after"
        for message in probe.log_messages(path):
            if message.startswith(_SEARCH_PREFIX):
                counts = {"tool_calls": 1}
            elif "タイムアウト" in message[:40]:
                found = _TIMEOUT_LINE.match(message)
                if not found:
                    continue
                kind = "recent" if found["who"] else "c2"
                counts = {f"timeout_{kind}": 1, f"timeout_{kind}_{found['t']}s": 1}
            elif "HTTP エラー" in message[:40]:
                found = _HTTP_ERROR_LINE.match(message)
                if not found:
                    continue
                counts = {"http_error": 1}
            else:
                continue
            for key in (f"month:{month}", f"side:{side}"):
                buckets.setdefault(key, Counter()).update(counts)
    return {key: dict(counter) for key, counter in sorted(buckets.items())}


TIMEOUT_NOTE = ("分母（記憶検索実行）は retrieve_memory の道具の呼び出しの数、分子は retriever ごとのタイムアウトの行の数。"
                "1 回の検索で C2Retriever、RecentC2Retriever、（dev なら）readonly の C2Retriever が順に呼ばれ、"
                "それぞれが別にタイムアウトしうるので、比は『道具の呼び出しあたりのタイムアウトの行数』で確率ではない"
                "（最大で retriever の数まで超えうる）。道具を通らない RAG の経路のタイムアウトも分子に入る。")


def print_timeouts(stats: dict[str, dict[str, Any]]) -> None:
    for key, counts in stats.items():
        calls = counts.get("tool_calls", 0)
        c2, recent = counts.get("timeout_c2", 0), counts.get("timeout_recent", 0)
        ratio = "-" if not calls else f"{(c2 + recent) / calls * 100:.1f}%"
        by_value = " ".join(f"{k[len('timeout_'):]}={v}" for k, v in sorted(counts.items())
                            if re.fullmatch(r"timeout_(?:c2|recent)_[0-9.]+s", k))
        if key.startswith("side:"):
            title = f"{CUTOFF_DAY} より前" if key == "side:before" else f"{CUTOFF_DAY} 以降"
        else:
            title = key[len("month:"):]
        emit(f"  {title}: 検索 {calls}  タイムアウト c2={c2} recent={recent}（{by_value or '-'}）  "
             f"HTTP エラー={counts.get('http_error', 0)}  行数/検索={ratio}")


def _cmd_timeouts(args: argparse.Namespace) -> int:
    runs_dir = args.logs_dir or probe.default_runs_dir()
    if not runs_dir.is_dir():
        emit(f"実走ログの場所がありません: {runs_dir.name}")
        return EXIT_FAILED
    paths, by_rule = probe.select_run_logs(runs_dir, args.since)
    emit(f"実走ログ: 使う {by_rule.get('使う', 0)} 本（" + "、".join(
        f"{rule} {count}" for rule, count in sorted(by_rule.items()) if rule != "使う") + "）")
    if not paths:
        emit("使える実走ログがありません")
        return EXIT_FAILED
    stats = timeouts_stats(paths)
    print_timeouts(stats)
    emit("注記: " + TIMEOUT_NOTE)
    out_dir = args.out_dir or spend.default_spend_dir()
    probe._write_json(out_dir / f"c2_timeouts_{args.label}.json", {
        "tool": "c2_timeouts", "label": args.label, "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "since": args.since.isoformat(), "cutoff_day": CUTOFF_DAY, "files": {"used": len(paths), "by_rule": dict(by_rule)},
        "counts": stats, "note": TIMEOUT_NOTE,
    })
    emit(f"書き出し: c2_timeouts_{args.label}.json")
    return EXIT_OK


# ─── コマンドライン ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab_lounge.c2_probe", description="C2（記憶検索）の時間の物差し")
    sub = parser.add_subparsers(dest="command", required=True)
    latency = sub.add_parser("latency", help="合成の問い合わせを C2 へ送り、cold と warm の分布を出す（読むだけ）")
    latency.add_argument("--profile", choices=("dev", "prod"), required=True, help="送り先の env を読むプロファイル")
    latency.add_argument("--label", required=True, help="実行の名前（英数字と _.- で 64 字まで）")
    latency.add_argument("--passes", type=int, default=DEFAULT_PASSES, help=f"問い合わせを何周送るか（既定 {DEFAULT_PASSES}）")
    latency.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    latency.add_argument("--k", type=int, default=0, help="取得件数（既定は本番と同じ max(L2_RAG_TOP_K × 3, 10)）")
    latency.add_argument("--targets", default="primary,readonly", help="送り先の役割（コンマ区切り）")
    latency.add_argument("--no-host-compare", action="store_true", help="localhost と 127.0.0.1 の比較を省く")
    latency.add_argument("--queries", type=Path, default=None, help="問い合わせ（既定は measure_sets/ の中）")
    latency.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    timeouts = sub.add_parser("timeouts", help="実走ログのタイムアウトを月ごと、10-31 の前後ごとに数える")
    timeouts.add_argument("--label", default="timeouts")
    timeouts.add_argument("--since", type=date.fromisoformat, default=replay_set.DEFAULT_SINCE)
    timeouts.add_argument("--logs-dir", type=Path, default=None, help="logs/runs/ の場所")
    timeouts.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.c2_probe <latency|timeouts>``。"""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.command == "latency":
        if args.passes < 1 or args.timeout_sec <= 0 or args.k < 0:
            emit("--passes は 1 以上、--timeout-sec は正の数、--k は 0 以上にしてください")
            return EXIT_USAGE
        return _cmd_latency(args)
    if not spend._RUN_NAME.fullmatch(args.label):
        emit("--label は英数字と _.- で 64 字までにしてください")
        return EXIT_USAGE
    return _cmd_timeouts(args)


if __name__ == "__main__":
    run_tool(main, name="c2_probe")
