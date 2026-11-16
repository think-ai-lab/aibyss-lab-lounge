"""
probe.py — 本応答と判定まわりの時間と費用の物差し（計測の道具）

責務:
  - raw:     本物の判定の関数（挙手の候補と承認）と、filler の LLM 呼び出し（キャラごとのモデル）を、
             合成の入力で n 回ずつ流し、p50 / p90 / 最大、失敗の数、tokens、費用を出す

本番の経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない
（本番のモジュールがこれを import しないことはテストで固定している）。

【WHY 本物の関数を呼ぶか】
  判定の system prompt や filler の組み立てを複製すると、複製と本番がずれたときに本番を測らなくなる。
  差し替えるのは LLM を呼ぶ 1 点（router._call_router_llm と filler._call_filler_llm）だけで、
  そこで時間と失敗を記録する。tokens と費用は spend の HTTP の層の写しから取る（SDK の戻り値には出ない）。

【WHY 後継のモデルと比べやすい形にするか】
  新しいモデルが出たら、同じ入力・同じ集計の形で流して並べたい。
  そのため、モデルは環境変数（L2_LLM_ROUTER_MODEL / L2_INTENT_GATE_MODEL / L2_LLM_FILLER_MODEL と
  キャラごとの filler_model）の値のまま使い、出力の 1 行 1 呼び出しにモデル名（要求と応答）と
  cache_read / cache_creation を必ず残す。モデルの行を 1 つ足して同じコマンドを流せば比べられる。

【使い方】
  uv run --frozen --all-extras python -m lab_lounge.probe raw --profile dev --label <名前> --max-usd 0.5

  入力は合成の行だけ（実発話は使わない）:
    - 挙手の候補と承認: measure_sets/replay_synthetic.jsonl（--synthetic で差し替え）
    - filler:          measure_sets/agent_utterances.jsonl（--utterances で差し替え）。{name} は
                       そのキャラの wake_word、{other} は別のパネリストの wake_word に置き換える

【終了コード】
  0: 正常 / 1: 呼び出しの失敗がある、または想定外のエラー / 2: 断った（同じ名前の回がある、入力が足りない）
  4: 上限、価格表にないモデル、usage の拾い損ねで止めた（spend）/ 64: コマンドラインの誤り / 130: Ctrl+C

【出力】（data/probe/、gitignore 済み）
  raw_<名前>.jsonl:        1 呼び出し 1 行（グループ、入力の id、モデル、時間、失敗の型、費用、tokens）。本文なし
  raw_<名前>.meta.json:    commit、uv.lock のハッシュ、SDK の版、モデル、入力の数とハッシュ
  raw_<名前>.problems.log: 実行中の WARNING 以上のログ（例外の文言を含みうるので Claude は開かない）

【出力の約束】
  画面には、件数、時間、tokens、費用だけを出す。発話の本文と、モデルの答えは出さず、ファイルにも書かない。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import decision_log, replay, replay_set, router, spend
from .spend import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, run_tool

EXIT_REFUSED = 2

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 1 グループで流す回数の既定。WHY 30: p90 を 3 件目に見られる最小の数。これ未満では最大と区別がつかない。
DEFAULT_N = 30

# filler を測るキャラ。filler_model がキャラごとに違う（gpt-5.4-mini、gemini-3.1-flash-lite、claude-haiku-4-5、
# 既定の gpt-5.4-nano）ので、4 社のモデルがそろう。
FILLER_CHARACTERS = ("mimi", "chisame", "sakura", "octamaid")

# {other}（別のパネリストの呼びかけ）の相手。キャラごとに固定にして、回をまたいで同じ入力にする。
OTHER_PANELIST = {"mimi": "chisame", "chisame": "sakura", "sakura": "mimi", "octamaid": "chisame"}

UTTERANCE_CATEGORIES = ("greeting", "explain", "deep", "search", "collab")

_SDKS = ("anthropic", "openai", "google-genai", "langchain-anthropic", "langchain-openai",
         "langchain-google-genai", "langgraph", "httpx", "httpx2")
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_ENV_IN_META = ("L2_LLM_ROUTER_MODEL", "L2_INTENT_GATE_MODEL", "L2_LLM_FILLER_MODEL", "L2_LLM_MAX_RETRIES",
                "L2_ROUTER_LLM_TIMEOUT_SEC", "L2_LLM_MAX_OUTPUT_TOKENS")


def measure_sets_dir() -> Path:
    """合成の行（commit する）の置き場所。"""
    return Path(__file__).resolve().parent / "measure_sets"


def default_utterances_path() -> Path:
    """計測の発話（合成）の既定の場所。"""
    return measure_sets_dir() / "agent_utterances.jsonl"


# ─── 小さな部品 ──────────────────────────────────────────────────

def percentile(values: list[float], q: float) -> float | None:
    """最近順位法（nearest-rank）の q 分位。空なら None。

    WHY 最近順位法: 補間をしないので、出た値のどれかがそのまま答えになる（n が小さいときに
         「実際に起きた時間」として読める）。
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


@contextlib.contextmanager
def prevent_sleep() -> Iterator[None]:
    """with の中だけ、OS のスリープを止める（このプロセスの要求だけ。設定は変えない）。

    WHY: 夜の配信の時間帯などに、画面を見ずに流す計測がスリープで途切れると、時間の分布が壊れる。
         SetThreadExecutionState は呼んだスレッドの要求で、プロセスが終われば消える。
         Windows 以外では何もしない。失敗しても計測は止めない。
    """
    if not _set_execution_state(prevent=True):
        yield
        return
    try:
        yield
    finally:
        _set_execution_state(prevent=False)


def _set_execution_state(*, prevent: bool) -> bool:
    """スリープの抑止を入れる / 戻す。入れられたときだけ True（テストで差し替える）。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        continuous, system_required = 0x80000000, 0x00000001
        flags = continuous | system_required if prevent else continuous
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def git_commit() -> str | None:
    return replay._git_commit()


def file_sha(path: Path) -> str | None:
    return replay._file_sha(path)


def sdk_versions(names: tuple[str, ...] = _SDKS) -> dict[str, str | None]:
    """関係するライブラリの版（無ければ None）。"""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def digest(parts: Any) -> str:
    """JSON にできる値のハッシュ（先頭 16 桁）。入力の集まりが変わったことを見分ける。"""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ─── 入力（合成） ────────────────────────────────────────────────

@dataclass(frozen=True)
class RouterItem:
    """判定の入力 1 件。本文は計測の間だけメモリにある。"""

    id: str
    text: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Utterance:
    """計測の発話（合成）1 本。{name} と {other} はまだ置き換えていない。"""

    id: str
    category: str
    text: str


def load_router_pool(path: Path | None) -> tuple[list[RouterItem], list[RouterItem]]:
    """合成の行から、挙手の候補の入力と、承認の入力を取り出す。

    挙手の候補は、interjection と intent（呼びかけ、言及）の入力を使う。本番では、挙手の候補の判定は
    背景の発話の全部にかかるので、呼びかけを含む発話も入力としてふさわしい。

    Raises:
        ValueError: 合成の行の形が違う、または種類が足りない（行番号と名前だけを知らせる）
    """
    rows = replay_set.load_synthetic(path)
    interjection = [RouterItem(r["id"], r["input"], ()) for r in rows if r["kind"] in ("interjection", "intent")]
    approval = [RouterItem(r["id"], r["input"], tuple(r["candidates"])) for r in rows
                if r["kind"] == "approval" and r["candidates"]]
    if not interjection or not approval:
        raise ValueError("合成の行に、挙手の候補（interjection / intent）か承認（approval）の入力がありません")
    return interjection, approval


def load_utterances(path: Path | None = None) -> list[Utterance]:
    """計測の発話（合成）を読む。

    形: ``{"id": "u01", "category": "greeting|explain|deep|search|collab", "text": "{name}、..."}``

    Raises:
        ValueError: ファイルが無い、形が違う行がある、id が重なる、未知の置き換え（行番号だけを知らせる。本文は出さない）
    """
    path = path or default_utterances_path()
    if not path.is_file():
        raise ValueError(f"計測の発話（{path.name}）がありません")
    items: list[Utterance] = []
    bad: list[int] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            bad.append(number)
            continue
        valid = (isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] not in seen
                 and item.get("category") in UTTERANCE_CATEGORIES
                 and isinstance(item.get("text"), str) and item["text"].strip()
                 and set(_PLACEHOLDER.findall(item["text"])) <= {"name", "other"})
        if not valid:
            bad.append(number)
            continue
        seen.add(item["id"])
        items.append(Utterance(item["id"], item["category"], item["text"]))
    if bad:
        raise ValueError(f"計測の発話の形が違います（{path.name} の {', '.join(map(str, bad))} 行目）")
    if not items:
        raise ValueError(f"計測の発話（{path.name}）が空です")
    return items


def wake_word(slug: str) -> str:
    """キャラの呼びかけの名前（characters.py の wake_word）。"""
    from .characters import get_character

    word = get_character(slug).wake_word
    if not word:
        raise ValueError(f"キャラクター {slug} に wake_word がありません")
    return word


def expand_utterance(text: str, slug: str) -> str:
    """{name} をそのキャラの wake_word に、{other} を固定の相手の wake_word に置き換える。"""
    other = OTHER_PANELIST[slug]
    return text.replace("{name}", wake_word(slug)).replace("{other}", wake_word(other))


def cycle(pool: list[Any], n: int) -> list[Any]:
    """pool を順にめぐって n 件にする（乱数を使わない。同じ pool なら同じ並びになる）。"""
    return [pool[i % len(pool)] for i in range(n)]


# ─── 計測の 1 呼び出し ───────────────────────────────────────────

@dataclass
class Sample:
    """1 呼び出しの記録。本文もモデルの答えも持たない。"""

    group: str
    index: int
    input_id: str
    model: str | None
    model_on_wire: str | None
    elapsed_ms: int | None
    error: str | None
    usd: float
    tokens: dict[str, int] = field(default_factory=dict)

    def line(self) -> dict[str, Any]:
        return {"group": self.group, "i": self.index, "input_id": self.input_id, "model": self.model,
                "model_on_wire": self.model_on_wire, "elapsed_ms": self.elapsed_ms, "error": self.error,
                "usd": round(self.usd, 9), "tokens": self.tokens}


@dataclass
class Outcome:
    """1 回の LLM 呼び出しを、本物の関数の中で見た結果。"""

    model: str | None
    elapsed_ms: int | None
    error: str | None


_TOKEN_KEYS = ("input", "output", "reasoning", "cache_read", "cache_creation")

_filler_capture = threading.local()


class RecordingFiller:
    """with の中だけ、filler._call_filler_llm を「呼んで時間と失敗を記録する」包みに差し替える。出るときに戻す。

    WHY: _call_filler_llm は例外を握って None を返すので、呼び出し側は失敗を知れない。None を失敗として数える
         （空の応答も同じ None になるので、原因は問題ログを見る。画面には出さない）。
    """

    def __enter__(self) -> RecordingFiller:
        from . import filler

        self._filler = filler
        self.original = filler._call_filler_llm
        original = self.original

        def call(model: str, system_prompt: str, user_text: str) -> str | None:
            start = time.monotonic()
            answer = original(model, system_prompt, user_text)
            elapsed = int((time.monotonic() - start) * 1000)
            _filler_capture.outcome = Outcome(model, elapsed, None if answer else "NoText")
            return answer

        filler._call_filler_llm = call
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._filler._call_filler_llm = self.original


def _router_outcome(call: Callable[[], object]) -> Outcome:
    """本物の判定の関数を呼び、その中の LLM 呼び出しの時間と失敗を返す（RecordingRouter の中で）。"""
    replay._capture.records = []
    try:
        call()
    finally:
        records = replay._capture.records
        replay._capture.records = None
    last = records[-1] if records else None
    return Outcome(last.model if last else None, last.elapsed_ms if last else None, last.error if last else "NoCall")


def _filler_outcome(slug: str, text: str) -> Outcome:
    """本物の filler の生成を呼び、その中の LLM 呼び出しの時間と失敗を返す（RecordingFiller の中で）。"""
    from . import filler

    _filler_capture.outcome = None
    filler._generate_filler_text(slug, user_text=text)
    return _filler_capture.outcome or Outcome(None, None, "NoCall")


# ─── raw ─────────────────────────────────────────────────────────

class RawGroup:
    """1 グループ（同じ関数、同じモデルの呼び出しの束）。"""

    def __init__(self, name: str, calls: list[tuple[str, Callable[[], Outcome]]]) -> None:
        self.name = name
        self.calls = calls  # (入力の id, 呼び出し) の並び


def build_groups(*, n: int, only: str, interjection: list[RouterItem], approval: list[RouterItem],
                 utterances: list[Utterance]) -> list[RawGroup]:
    """流すグループを組み立てる。入力は pool を順にめぐって n 件にする。"""
    groups: list[RawGroup] = []
    if only in ("all", "router"):
        groups.append(RawGroup("router:interjection", [
            (item.id, lambda item=item: _router_outcome(lambda: router.check_intent(item.text, None)))
            for item in cycle(interjection, n)]))
        groups.append(RawGroup("router:approval", [
            (item.id, lambda item=item: _router_outcome(lambda: router.check_approval(item.text, list(item.candidates))))
            for item in cycle(approval, n)]))
    if only in ("all", "filler"):
        for slug in FILLER_CHARACTERS:
            groups.append(RawGroup(f"filler:{slug}", [
                (utt.id, lambda slug=slug, utt=utt: _filler_outcome(slug, expand_utterance(utt.text, slug)))
                for utt in cycle(utterances, n)]))
    return groups


def measure_group(group: RawGroup, ledger: spend.SpendLedger, label: str, *, gap_sec: float,
                  sink: Callable[[Sample], None], sleep: Callable[[float], None] = time.sleep) -> list[Sample]:
    """グループの呼び出しを順に流し、1 回ごとに Sample を sink へ渡す。

    費用は、呼び出しの前後の支出の記録の差（SDK の再試行の分も入る）。tokens とモデル名は、最後に数えた
    HTTP の応答（spend.last_call）から取る。並列にしない: 時間の分布に自分の並列が混ざらないように。
    """
    samples: list[Sample] = []
    for index, (input_id, call) in enumerate(group.calls):
        if index and gap_sec > 0:
            sleep(gap_sec)
        spend.last_call()  # 前の呼び出しの名残を消す
        before = ledger.run_usd(label)
        outcome = call()
        usd = ledger.run_usd(label) - before
        wire = spend.last_call() or {}
        tokens = {key: int((wire.get("tokens") or {}).get(key, 0)) for key in _TOKEN_KEYS}
        sample = Sample(group.name, index, input_id, outcome.model, wire.get("model_on_wire"), outcome.elapsed_ms,
                        outcome.error, usd, tokens)
        samples.append(sample)
        sink(sample)
    return samples


def summarize(samples: list[Sample]) -> dict[str, Any]:
    """グループの集計。時間は成功した呼び出しだけで数える（失敗の待ち時間が分布を歪めないように）。"""
    ok = [s for s in samples if not s.error and s.elapsed_ms is not None]
    times = [float(s.elapsed_ms) for s in ok if s.elapsed_ms is not None]
    total = {key: sum(s.tokens.get(key, 0) for s in samples) for key in _TOKEN_KEYS}
    models = Counter(s.model for s in samples if s.model)
    wire = Counter(s.model_on_wire for s in samples if s.model_on_wire)
    return {
        "n": len(samples), "failed": len(samples) - len(ok),
        "p50": percentile(times, 0.5), "p90": percentile(times, 0.9), "max": max(times) if times else None,
        "tokens": total, "usd": sum(s.usd for s in samples),
        "models": dict(models), "model_on_wire": dict(wire),
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    """グループの 1 行を画面へ（件数、時間、tokens、費用だけ）。"""
    models = "/".join(spend._showable(m) for m in sorted(summary["models"])) or "-"
    wire = "/".join(spend._showable(m) for m in sorted(summary["model_on_wire"])) or "-"
    t = summary["tokens"]
    emit(f"  {name}: モデル {models}（応答 {wire}）  n={summary['n']} 失敗={summary['failed']}  "
         f"p50={_ms(summary['p50'])}ms p90={_ms(summary['p90'])}ms 最大={_ms(summary['max'])}ms  "
         f"tokens in={t['input']} out={t['output']} reasoning={t['reasoning']} "
         f"cache_read={t['cache_read']} cache_creation={t['cache_creation']}  ${summary['usd']:.4f}")


def raw_meta(label: str, args: argparse.Namespace, *, pools: dict[str, int], inputs_sha: str) -> dict[str, Any]:
    """raw の meta。値は設定と版と数だけ（本文なし）。"""
    from .characters import get_character

    filler_models = {}
    for slug in FILLER_CHARACTERS:
        char = get_character(slug)
        filler_models[slug] = char.filler_model or os.environ.get("L2_LLM_FILLER_MODEL", "gpt-5.4-nano")
    return {
        "tool": "raw", "label": label, "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(), "uv_lock_sha": file_sha(_REPO_ROOT / "uv.lock"), "sdk": sdk_versions(),
        "models": {"router": os.environ.get("L2_LLM_ROUTER_MODEL", "claude-haiku-4-5"),
                   "intent_gate": os.environ.get("L2_INTENT_GATE_MODEL", "claude-haiku-4-5"),
                   "filler": filler_models},
        "env": {name: os.environ.get(name) for name in _ENV_IN_META},
        "n": args.n, "only": args.only, "gap_sec": args.gap_sec, "profile": args.profile,
        "pools": pools, "inputs_sha": inputs_sha,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=1, sort_keys=True)


def _cmd_raw(args: argparse.Namespace) -> int:
    from .preflight import _captured_logs, prepare_env, silence_side_outputs

    if not spend._RUN_NAME.fullmatch(args.label):
        emit("--label は英数字と _.- で 64 字までにしてください")
        return EXIT_USAGE
    out_dir = args.out_dir or spend.default_spend_dir()
    run_path = out_dir / f"raw_{args.label}.jsonl"
    if run_path.exists():
        emit(f"実行 {args.label} はすでにあります（別の名前で --label）")
        return EXIT_REFUSED
    try:
        interjection, approval = load_router_pool(args.synthetic) if args.only != "filler" else ([], [])
        utterances = load_utterances(args.utterances) if args.only != "router" else []
    except ValueError as exc:  # 自分の読み込みの例外。文言は行番号と名前だけ
        emit(f"[断りました] {exc}")
        return EXIT_REFUSED

    prepare_env(args.profile, _REPO_ROOT)
    silence_side_outputs()
    if decision_log.is_enabled():
        emit("判定ログを止められませんでした")
        return EXIT_REFUSED

    groups = build_groups(n=args.n, only=args.only, interjection=interjection, approval=approval,
                          utterances=utterances)
    pools = {"interjection": len(interjection), "approval": len(approval), "filler": len(utterances)}
    inputs_sha = digest([[(i.id, i.text, i.candidates) for i in interjection],
                         [(i.id, i.text, i.candidates) for i in approval],
                         [(u.id, u.category, u.text) for u in utterances]])
    meta = raw_meta(args.label, args, pools=pools, inputs_sha=inputs_sha)

    overrides = dict(spend.parse_price_override(text) for text in args.price)
    ledger = spend.SpendLedger(args.spend_dir or spend.default_spend_dir())
    emit(f"raw {args.label}: {len(groups)} グループ × n={args.n}、上限 ${args.max_usd:.2f}"
         f"（入力 挙手の候補 {pools['interjection']} / 承認 {pools['approval']} / filler {pools['filler']} 本をめぐる）")
    writer = replay.RunFile(run_path)
    summaries: dict[str, dict[str, Any]] = {}
    calls_made = 0
    with prevent_sleep(), _captured_logs(out_dir / f"raw_{args.label}.problems.log"), \
            spend.UsageMeter(ledger, args.label, max_usd=args.max_usd, overrides=overrides) as meter, \
            replay.RecordingRouter(), RecordingFiller():
        for group in groups:
            samples = measure_group(group, ledger, args.label, gap_sec=args.gap_sec,
                                    sink=lambda sample: writer.append(sample.line()))
            calls_made += sum(1 for s in samples if not s.error)
            meter.check_capture(calls_made)
            summaries[group.name] = summarize(samples)
            print_summary(group.name, summaries[group.name])
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["summary"] = summaries
    _write_json(out_dir / f"raw_{args.label}.meta.json", meta)
    total = sum(s["usd"] for s in summaries.values())
    failed = sum(s["failed"] for s in summaries.values())
    emit(f"合計 ${total:.4f}、失敗 {failed}、usage を拾えず見積もりで数えた呼び出し {meter.stats.uncounted}")
    return EXIT_FAILED if failed else EXIT_OK


# ─── コマンドライン ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab_lounge.probe", description="本応答と判定まわりの時間と費用の物差し")
    sub = parser.add_subparsers(dest="command", required=True)
    raw = sub.add_parser("raw", help="判定（挙手の候補と承認）と filler の LLM 呼び出しの時間と費用（課金あり）")
    raw.add_argument("--profile", choices=("dev", "prod"), required=True)
    raw.add_argument("--label", required=True, help="実行の名前（英数字と _.- で 64 字まで）")
    raw.add_argument("--max-usd", type=float, required=True, help="この実行の上限（同じ名前の過去の分を含む）")
    raw.add_argument("--n", type=int, default=DEFAULT_N, help=f"グループごとの回数（既定 {DEFAULT_N}）")
    raw.add_argument("--only", choices=("all", "router", "filler"), default="all")
    raw.add_argument("--gap-sec", type=float, default=0.0, help="呼び出しの間（秒）")
    raw.add_argument("--synthetic", type=Path, default=None, help="合成の行（既定は measure_sets/ の中）")
    raw.add_argument("--utterances", type=Path, default=None, help="計測の発話（既定は measure_sets/ の中）")
    raw.add_argument("--price", action="append", default=[], help="model=in,out[,cache_read[,cache_write]]")
    raw.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    raw.add_argument("--spend-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.probe <raw>``。"""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.command == "raw":
        if args.n < 1 or args.gap_sec < 0:
            emit("--n は 1 以上、--gap-sec は 0 以上にしてください")
            return EXIT_USAGE
        return _cmd_raw(args)
    return EXIT_USAGE


if __name__ == "__main__":
    run_tool(main, name="probe")
