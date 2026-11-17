"""
probe.py — 本応答と判定まわりの時間と費用の物差し（計測の道具）

責務:
  - raw:     本物の判定の関数（挙手の候補と承認）と、filler の LLM 呼び出し（キャラごとのモデル）を、
             合成の入力で n 回ずつ流し、p50 / p90 / 最大、失敗の数、tokens、費用を出す
  - history: 実走ログから、キャラごと・月ごとの Agent の全体の時間と step の時間を数える（数字だけ。課金なし）
  - agent:   本番と同じ Agent（道具つき）の組み立てで、キャラごとに合成の発話を流し、全体の時間、step ごとの
             時間と tokens（キャッシュの読み書きを含む）、道具の発火、費用を測る。道具は長さを固定した定型の応答
             に差し替える（外の検索や記憶や他キャラの TTS を呼ばない）。返答の本文は聴き比べ用に別ファイルへ残す

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
  uv run --frozen --all-extras python -m lab_lounge.probe agent --profile dev --label <名前> --max-usd 3
  #   少数で試す: --limit 2 --characters mimi / 別のモデルで全キャラ: --llm anthropic:claude-sonnet-4-6
  uv run --frozen --all-extras python -m lab_lounge.probe history --label <名前>   # 実走ログの月ごとの時間（課金なし）

  入力は合成の行だけ（実発話は使わない）:
    - 挙手の候補と承認: measure_sets/replay_synthetic.jsonl（--synthetic で差し替え）
    - filler と agent: measure_sets/agent_utterances.jsonl（--utterances で差し替え）。{name} は
                       そのキャラの wake_word、{other} は別のパネリストの wake_word に置き換える

【終了コード】
  0: 正常 / 1: 呼び出しの失敗がある、時間切れ、または想定外のエラー / 2: 断った（同じ名前の回がある、入力が足りない、
  ツールが無効）
  4: 上限、価格表にないモデル、usage の拾い損ねで止めた（spend）/ 64: コマンドラインの誤り / 130: Ctrl+C

【出力】（data/probe/、gitignore 済み）
  raw_<名前>.jsonl:        1 呼び出し 1 行（グループ、入力の id、モデル、時間、失敗の型、費用、tokens）。本文なし
  raw_<名前>.meta.json:    commit、uv.lock のハッシュ、SDK の版、モデル、入力の数とハッシュ
  raw_<名前>.problems.log: 実行中の WARNING 以上のログ（例外の文言を含みうるので Claude は開かない）
  history_<名前>.json:         月ごとの時間の表（数字だけ）
  agent_<名前>.jsonl:          1 呼び出し 1 行（数字 + 返答の本文）。返答の本文はこのファイルだけに書く（Claude は開かない）
  agent_<名前>.metrics.jsonl:  上から本文を除いたもの（数字だけ）
  agent_<名前>.meta.json:      commit、モデル、配信文脈と人格のハッシュ、集計
  stream_context_snapshot.md:  最初の回の配信文脈の写し。後の回も同じものを使う（--refresh-snapshot で取り直す）

【出力の約束】
  画面には、件数、時間、tokens、費用だけを出す。発話の本文と、モデルの答えは出さず、ファイルにも書かない。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import contextvars
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
from datetime import UTC, date, datetime
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

class MeasureInputError(ValueError):
    """計測の入力（合成の行、引数）の形が違う。

    WHY 専用の型: 画面へ出してよい文言は、自分で作ったもの（ファイル名と行番号だけ）に限る。ValueError 全般を
         捕まえて出すと、JSON や変換の例外（入力の断片を含みうる）まで画面へ出てしまう。この型だけを捕まえて出す。
    """


def read_text_checked(path: Path) -> str:
    """UTF-8 のファイルを読む。読めなければ、ファイル名だけを含む MeasureInputError にする。"""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise MeasureInputError(f"{path.name} を読めません（開けない、または UTF-8 でない）") from None


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
        MeasureInputError: 合成の行の形が違う、または種類が足りない（行番号と名前だけを知らせる）
    """
    try:
        rows = replay_set.load_synthetic(path)
    except UnicodeDecodeError:
        raise MeasureInputError("合成の行を読めません（UTF-8 でない）") from None
    except ValueError as exc:  # replay_set 自身の文言（ファイル名と行番号だけ）
        raise MeasureInputError(str(exc)) from None
    interjection = [RouterItem(r["id"], r["input"], ()) for r in rows if r["kind"] in ("interjection", "intent")]
    approval = [RouterItem(r["id"], r["input"], tuple(r["candidates"])) for r in rows
                if r["kind"] == "approval" and r["candidates"]]
    if not interjection or not approval:
        raise MeasureInputError("合成の行に、挙手の候補（interjection / intent）か承認（approval）の入力がありません")
    return interjection, approval


def load_utterances(path: Path | None = None) -> list[Utterance]:
    """計測の発話（合成）を読む。

    形: ``{"id": "u01", "category": "greeting|explain|deep|search|collab", "text": "{name}、..."}``

    Raises:
        MeasureInputError: ファイルが無い、形が違う行がある、id が重なる、未知の置き換え（行番号だけを知らせる。本文は出さない）
    """
    path = path or default_utterances_path()
    if not path.is_file():
        raise MeasureInputError(f"計測の発話（{path.name}）がありません")
    items: list[Utterance] = []
    bad: list[int] = []
    seen: set[str] = set()
    # WHY split("\n"): JSONL は \n だけで区切る。splitlines は本文の U+2028 などでも割り、1 件が壊れる。
    for number, line in enumerate(read_text_checked(path).split("\n"), start=1):
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
        raise MeasureInputError(f"計測の発話の形が違います（{path.name} の {', '.join(map(str, bad))} 行目）")
    if not items:
        raise MeasureInputError(f"計測の発話（{path.name}）が空です")
    return items


def wake_word(slug: str) -> str:
    """キャラの呼びかけの名前（characters.py の wake_word）。"""
    from .characters import get_character

    word = get_character(slug).wake_word
    if not word:
        raise MeasureInputError(f"キャラクター {slug} に wake_word がありません")
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
    except MeasureInputError as exc:  # 自分で作った文言（行番号と名前だけ）
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


# ─── agent ───────────────────────────────────────────────────────

AGENT_CHARACTERS = ("mimi", "chisame", "sakura", "octamaid")

# 道具の定型の応答の長さ（文字数）の既定。WHY: 道具の応答の長さは、次の step の入力 tokens と時間に効く。
# 実の検索（web_search）は長く、記憶（retrieve_memory）は中くらい、他キャラの返答（ask_character）は短い。
# 回をまたいで同じ長さにして、モデルや設定の違いだけが数字に出るようにする。
DEFAULT_TOOL_CHARS = {"web_search": 1200, "retrieve_memory": 600, "ask_character": 300}

# 定型の応答の素。深海の一般的な説明（中身に意味はなく、毎回同じ文を繰り返して長さをそろえる）。
_CANNED_BASE = ("深海には光がほとんど届かず、多くの生き物が自ら光を出して暮らしています。"
                "水圧は水深とともに増し、千メートルでは地上の約百倍になります。"
                "深海の調査は、無人探査機と音響の観測を組み合わせて進められています。")

# meta に値を残す環境変数（Agent の組み立てと時間に効くもの）。
_AGENT_ENV_IN_META = (*_ENV_IN_META, "L2_ENABLE_TOOLS", "L2_ENABLE_RAG", "L2_AGENT_INVOKE_TIMEOUT_SEC",
                      "L2_AGENT_RECURSION_LIMIT", "L2_LLM_REASONING_EFFORT", "L2_LLM_TIMEOUT_SEC")

_TOOL_NAME_OK = re.compile(r"[A-Za-z0-9_\-]{1,64}")


def canned_text(tool: str, chars: int) -> str:
    """道具の定型の応答。ちょうど chars 文字で、同じ引数なら同じ文字列になる。"""
    prefix = f"[{tool} の定型の応答]\n"
    body = (_CANNED_BASE * (chars // len(_CANNED_BASE) + 1))[: max(0, chars - len(prefix))]
    return (prefix + body)[:chars]


@contextlib.contextmanager
def canned_tools(lengths: dict[str, int], fires: Counter) -> Iterator[None]:
    """with の中だけ、3 つの道具の本体を、長さを固定した定型の応答を返す関数に差し替える。出るときに戻す。

    差し替えるのは mcp_servers の関数（web_search / retrieve_memory / ask_character）だけで、LangChain に渡す
    道具のスキーマ（名前、引数、説明）は graph._load_mcp_tools が作る本物のまま。graph は道具を組み立てる
    たびにこれらの関数を import するので、モジュールの属性を差し替えれば効く。
    WHY: 外の検索（費用と時間のばらつき）、記憶検索（C2）、他キャラの生成と TTS を呼ばずに、本応答の LLM の
         step だけを測るため。発火した数は fires に数える。
    """
    from .mcp_servers import ask_character, retrieve_memory, web_search

    targets = [(web_search, "web_search"), (retrieve_memory, "retrieve_memory"), (ask_character, "ask_character")]
    lock = threading.Lock()
    originals = []

    def make(name: str) -> Callable[..., str]:
        text = canned_text(name, lengths[name])

        def canned(*_args: Any, **_kwargs: Any) -> str:
            with lock:
                fires[name] += 1
            return text

        return canned

    for module, name in targets:
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, make(name))
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def tag_speaker(text: str) -> str:
    """本番の user の前置きの規則（発話者がルカだと LLM に明示する）。

    graph._generation_node のなかの規則の写し。関数に切り出されていないので、写してテストで固定する
    （tests/test_probe.py が graph.py の該当の行を読み、変わったら赤くなる）。
    WHY 写す: これが無いと、user message の発話者を LLM が視聴者と推測して、呼び方や口調が本番とずれ、
         出力の長さ（= 時間と tokens）まで変わりうる。本番の graph.py の経路は変えない方針なので、複製する。
    """
    if text and not text.startswith("[") and "アビスメイト" not in text[:30]:
        return f"[ルカからの発言]\n{text}"
    return text


def _safe_tool_name(name: object) -> str:
    return name if isinstance(name, str) and _TOOL_NAME_OK.fullmatch(name) else "<不正な名前>"


def make_recorder() -> Any:
    """step ごとの時間と tokens、道具の時間を、run_id で開始と終了を組んで記録するコールバックを作る。

    WHY BaseCallbackHandler を継承: LangChain が handler に求める属性が更新で増えても、属性の不足で例外が握られて
         黙って記録しなくなるのを避ける。関数の中で import するのは、langchain_core の import が壊れた環境でも
         モジュールの読み込みを止めないため。
    WHY run_id で組む: 並列の step や道具の中の LLM があっても、開始と終了を取り違えない。
    WHY 道具の名前を検査して残す: 名前はモデルの出力から来る。本文は残さない。
    """
    from langchain_core.callbacks import BaseCallbackHandler

    from .graph import _extract_llm_usage, _first_generation_message

    class _Recorder(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.steps: list[dict[str, Any]] = []
            self.tools: list[dict[str, Any]] = []
            self._llm_open: dict[Any, float] = {}
            self._tool_open: dict[Any, tuple[str, float]] = {}

        def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any, **_kwargs: Any) -> None:
            self._llm_open[run_id] = time.monotonic()

        def on_llm_end(self, response: Any, *, run_id: Any, **_kwargs: Any) -> None:
            started = self._llm_open.pop(run_id, None)
            info = _extract_llm_usage(response)
            message = _first_generation_message(response)
            calls = [_safe_tool_name(call.get("name")) for call in (getattr(message, "tool_calls", None) or [])
                     if isinstance(call, dict)]
            self.steps.append({
                "n": len(self.steps) + 1,
                "ms": None if started is None else int((time.monotonic() - started) * 1000),
                **{key: int(info[key] or 0) for key in _TOKEN_KEYS},
                "finish": info["finish"] if isinstance(info["finish"], str) else None,
                "model": info["model"] if isinstance(info["model"], str) else None,
                "tool_calls": calls, "error": None,
            })

        def _llm_failed(self, error: BaseException, run_id: Any) -> None:
            started = self._llm_open.pop(run_id, None)
            self.steps.append({
                "n": len(self.steps) + 1,
                "ms": None if started is None else int((time.monotonic() - started) * 1000),
                **dict.fromkeys(_TOKEN_KEYS, 0), "finish": None, "model": None, "tool_calls": [],
                "error": type(error).__name__,
            })

        def on_llm_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
            self._llm_failed(error, run_id)

        def on_chat_model_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
            self._llm_failed(error, run_id)

        def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any, **kwargs: Any) -> None:
            name = (serialized or {}).get("name") if isinstance(serialized, dict) else None
            self._tool_open[run_id] = (_safe_tool_name(name or kwargs.get("name")), time.monotonic())

        def _tool_done(self, run_id: Any, error: BaseException | None) -> None:
            name, started = self._tool_open.pop(run_id, ("<不明>", time.monotonic()))
            self.tools.append({"name": name, "ms": int((time.monotonic() - started) * 1000),
                               "error": None if error is None else type(error).__name__})

        def on_tool_end(self, output: Any, *, run_id: Any, **_kwargs: Any) -> None:
            self._tool_done(run_id, None)

        def on_tool_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
            self._tool_done(run_id, error)

    return _Recorder()


class AgentTimeout(Exception):
    """agent.invoke が時間切れになった。呼び出し側が記録してから hard_exit する。

    Attributes:
        line: 時間切れまでに取れた分（step、道具、費用）の行。measure_agent_call が入れる
    """

    line: dict[str, Any] | None = None


def resolve_stream_context(snapshot: Path, *, refresh: bool,
                           load_live: Callable[[], str | None]) -> tuple[str | None, dict[str, Any]]:
    """配信文脈を決める。写しがあればそれを使い、無ければ今のものを読んで写しを作る。

    WHY 写す: 配信文脈は日々書き換わるので、そのまま読むと回ごとに system prompt が変わり、モデルや設定の
         違いと見分けられない。最初の回の写しを `data/probe/` に残し、後の回も同じものを使う。
    meta には、由来、文字数、ハッシュだけを残す（中身は書かない）。
    """
    origin = "snapshot"
    text: str | None = None
    if snapshot.is_file() and not refresh:
        text = snapshot.read_text(encoding="utf-8").strip() or None
    if text is None:
        origin = "live"
        text = load_live()
        if text:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text + "\n")
    if not text:
        return None, {"origin": "none", "chars": 0, "sha": None}
    return text, {"origin": origin, "chars": len(text), "sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def persona_hashes(slug: str) -> dict[str, str | None]:
    """人格ファイルと Skills のハッシュ（再現のため。中身は残さない）。"""
    from .characters import get_character, load_system_prompt
    from .skill_loader import build_skills_prompt

    def sha(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    try:
        persona = sha(load_system_prompt(get_character(slug)))
    except FileNotFoundError:
        persona = None
    return {"persona_sha": persona, "skills_sha": sha(build_skills_prompt(slug))}


def response_text(result: dict[str, Any]) -> str:
    """agent の戻り値から返答の本文を取る（構造化出力を優先し、無ければ最後の AI の本文）。聴き比べ用。"""
    structured = result.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            return structured.model_dump_json()
        if isinstance(structured, dict):
            return json.dumps(structured, ensure_ascii=False)
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", "") in ("ai", "assistant"):
            content = getattr(message, "content", "")
            if isinstance(content, list):
                return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            return str(content)
    return ""


def invoke_agent(agent: Any, tagged_text: str, recorder: Any, *, timeout_sec: float, recursion_limit: int) -> dict[str, Any]:
    """agent.invoke を別スレッドで呼び、壁時計の時間切れで打ち切る。

    graph._invoke_agent_with_timeout は使わない: 打ち切りの後始末（協働先の TTS を止める）は計測では要らず、
    打ち切ったら記録してプロセスごと終える（orphan のスレッドの課金を止めるため）。
    contextvars は呼び出し元のものを引き継ぐ（本番と同じ）。

    Raises:
        AgentTimeout: timeout_sec を超えて、まだ終わっていない（agent.invoke が自分で投げたタイムアウトは含まない）
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    context = contextvars.copy_context()
    config = {"recursion_limit": recursion_limit, "callbacks": [recorder]}
    future = executor.submit(context.run, agent.invoke, {"messages": [{"role": "user", "content": tagged_text}]}, config)
    try:
        return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        # WHY done() を見る: 3.11 以降、組み込みの TimeoutError と concurrent.futures.TimeoutError は同じ型。
        #      SDK の中の別のタイムアウト（agent.invoke が投げた例外）を、壁時計の時間切れと取り違えて
        #      プロセスごと止めると、残りのキャラの計測が失われる。終わっていれば、中身の例外は普通の失敗として返す。
        if future.done():
            return future.result(timeout=0)
        raise AgentTimeout(f"{timeout_sec:.0f} 秒") from None
    finally:
        executor.shutdown(wait=False)


@dataclass
class AgentSettings:
    """1 回の計測で共通の設定。"""

    stream_context: str | None
    timeout_sec: float
    recursion_limit: int
    llm_override: tuple[str, str] | None = None  # (provider, model)


def measure_agent_call(slug: str, utterance: Utterance, index: int, settings: AgentSettings,
                       ledger: spend.SpendLedger, label: str) -> tuple[dict[str, Any], str]:
    """1 回の Agent の呼び出しを測り、(数字の行, 返答の本文) を返す。

    組み立ては preflight と同じ（_agent_prompt、_build_agent_graph、agent.invoke を直に）。_run_agent は使わない:
    失敗をフォールバックで隠し、finish を固定し、tokens を最後の 1 回分しか数えないため。
    Agent は本番と同じくターンごとに作る（build_ms に出る。LLM のクライアントの作り直しも含む）。

    Raises:
        AgentTimeout: 時間切れ（呼び出し側が記録して終える）
        spend.SpendStop: 上限などで止めた
    """
    from . import graph
    from .characters import get_character
    from .pipeline import _get_llm_mode
    from .preflight import _agent_prompt, _summarize_agent_result

    character = get_character(slug)
    _, provider, model = _get_llm_mode(character)
    if settings.llm_override:
        provider, model = settings.llm_override
    spend.last_call()
    usd_before, calls_before = ledger.run_usd(label), ledger.data["by_run"].get(label, {}).get("calls", 0)
    line: dict[str, Any] = {
        "i": index, "character": slug, "utterance_id": utterance.id, "category": utterance.category,
        "cold": index == 0, "provider": provider, "model": model,
        "build_ms": None, "total_ms": None, "error": None, "source": None, "response_chars": 0,
        "steps": [], "tools": [], "tools_fired": [], "finish": None, "model_on_wire": [],
    }
    recorder = make_recorder()
    body = ""
    started = time.monotonic()
    try:
        prompt = _agent_prompt(character, settings.stream_context)
        agent = graph._build_agent_graph(provider, model, prompt, character_slug=slug)
        line["build_ms"] = int((time.monotonic() - started) * 1000)
        if agent is None:
            line["error"] = "NoAgent"
        else:
            invoked = time.monotonic()
            try:
                result = invoke_agent(agent, tag_speaker(expand_utterance(utterance.text, slug)), recorder,
                                      timeout_sec=settings.timeout_sec, recursion_limit=settings.recursion_limit)
            finally:
                line["total_ms"] = int((time.monotonic() - invoked) * 1000)
                line["steps"], line["tools"] = recorder.steps, recorder.tools
            summary = _summarize_agent_result(result)
            body = response_text(result)
            # source が structured_response でないとき（本番はメッセージの本文へ落ちる）は、失敗にせず source に出す。
            line.update(source=summary["source"], response_chars=summary["response_len"], finish=summary["finish"],
                        tools_fired=[_safe_tool_name(name) for name in summary["tools"]])
    except AgentTimeout as exc:
        line["error"] = "Timeout"
        exc.line = line  # finally が費用を書き足した後の同じ行を、呼び出し側が記録する
        raise
    except spend.SpendStop:
        raise
    except Exception as exc:  # noqa: BLE001 - 失敗の型だけを残す（文言には本文が入りうる）
        line["error"] = type(exc).__name__
    finally:
        line["usd"] = ledger.run_usd(label) - usd_before
        line["http_calls"] = ledger.data["by_run"].get(label, {}).get("calls", 0) - calls_before
    return line, body


def summarize_agent(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """1 キャラの集計。時間は成功した呼び出しだけ、cold は最初の 1 回、warm は 2 回目以降。"""
    ok = [line for line in lines if not line["error"] and line["total_ms"] is not None]
    cold = [float(line["total_ms"]) for line in ok if line["cold"]]
    warm = [float(line["total_ms"]) for line in ok if not line["cold"]]
    steps = [step for line in ok for step in line["steps"]]
    step_ms = [float(step["ms"]) for step in steps if step["ms"] is not None and not step["error"]]
    tokens = {key: sum(step[key] for line in lines for step in line["steps"]) for key in _TOKEN_KEYS}
    fired: Counter = Counter()
    for line in ok:
        fired.update(line["tools_fired"])
    return {
        "n": len(lines), "failed": len(lines) - len(ok), "cold_ms": cold[0] if cold else None,
        "warm_p50": percentile(warm, 0.5), "warm_p90": percentile(warm, 0.9), "warm_max": max(warm) if warm else None,
        "step_p50": percentile(step_ms, 0.5), "step_p90": percentile(step_ms, 0.9),
        "steps_mean": (len(steps) / len(ok)) if ok else None, "tokens": tokens,
        "tools_fired": dict(fired), "source": dict(Counter(line["source"] for line in lines if line["source"])),
        "usd": sum(line.get("usd", 0.0) for line in lines),
        "models": dict(Counter(line["model"] for line in lines if line["model"])),
        "model_on_wire": dict(Counter(step["model"] for line in lines for step in line["steps"] if step["model"])),
        "finish": dict(Counter(line["finish"] for line in lines if line["finish"])),
    }


def print_agent_summary(slug: str, summary: dict[str, Any]) -> None:
    """キャラごとの 2 行を画面へ（時間、steps、tokens、道具、source、費用だけ）。"""
    models = "/".join(spend._showable(m) for m in sorted(summary["models"])) or "-"
    wire = "/".join(spend._showable(m) for m in sorted(summary["model_on_wire"])) or "-"
    n_ok = max(summary["n"] - summary["failed"], 1)
    t = summary["tokens"]
    fired = " ".join(f"{spend._showable(k)}×{v}" for k, v in sorted(summary["tools_fired"].items())) or "なし"
    source = " ".join(f"{spend._showable(k)}×{v}" for k, v in sorted(summary["source"].items())) or "-"
    emit(f"  {slug}: モデル {models}（応答 {wire}）  n={summary['n']} 失敗={summary['failed']}  "
         f"cold={_ms(summary['cold_ms'])}ms  warm p50={_ms(summary['warm_p50'])}ms p90={_ms(summary['warm_p90'])}ms "
         f"最大={_ms(summary['warm_max'])}ms  step p50={_ms(summary['step_p50'])}ms p90={_ms(summary['step_p90'])}ms")
    emit(f"      steps 平均 {summary['steps_mean'] if summary['steps_mean'] is None else round(summary['steps_mean'], 2)}  "
         f"tokens/回 in={t['input'] // n_ok} out={t['output'] // n_ok} reasoning={t['reasoning'] // n_ok} "
         f"cache_read={t['cache_read'] // n_ok} cache_creation={t['cache_creation'] // n_ok}  "
         f"道具 {fired}  source {source}  ${summary['usd']:.4f}")


def parse_tool_chars(texts: list[str]) -> dict[str, int]:
    """``--tool-chars web_search=800`` の列を、既定に重ねた長さの表にする。

    Raises:
        MeasureInputError: 名前が道具にない、または長さが 1 以上の整数でない
    """
    lengths = dict(DEFAULT_TOOL_CHARS)
    for text in texts:
        name, _, value = text.partition("=")
        if name not in lengths or not value.isdecimal() or int(value) < 1:
            raise MeasureInputError("--tool-chars は 道具名=文字数（web_search / retrieve_memory / ask_character）の形にしてください")
        lengths[name] = int(value)
    return lengths


def parse_llm_override(text: str | None) -> tuple[str, str] | None:
    """``--llm anthropic:claude-sonnet-4-6`` を (provider, model) にする。

    Raises:
        MeasureInputError: provider:model の形でない、または provider が違う
    """
    if not text:
        return None
    provider, _, model = text.partition(":")
    if provider not in ("openai", "anthropic", "google") or not model:
        raise MeasureInputError("--llm は provider:model（provider は openai / anthropic / google）の形にしてください")
    return provider, model


def agent_meta(label: str, args: argparse.Namespace, *, characters: list[str], settings: AgentSettings,
               context_meta: dict[str, Any], lengths: dict[str, int], utterances: list[Utterance]) -> dict[str, Any]:
    """agent の meta。値は設定と版とハッシュと数だけ（本文なし）。"""
    from .characters import get_character
    from .pipeline import _get_llm_mode

    models = {}
    for slug in characters:
        _, provider, model = _get_llm_mode(get_character(slug))
        models[slug] = f"{settings.llm_override[0]}:{settings.llm_override[1]}" if settings.llm_override else f"{provider}:{model}"
    return {
        "tool": "agent", "label": label, "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(), "uv_lock_sha": file_sha(_REPO_ROOT / "uv.lock"), "sdk": sdk_versions(),
        "profile": args.profile, "characters": characters, "models": models,
        "llm_override": args.llm, "env": {name: os.environ.get(name) for name in _AGENT_ENV_IN_META},
        "timeout_sec": settings.timeout_sec, "recursion_limit": settings.recursion_limit,
        "stream_context": context_meta, "personas": {slug: persona_hashes(slug) for slug in characters},
        "tool_chars": lengths, "gap_sec": args.gap_sec,
        "utterances": {"n": len(utterances), "sha": digest([(u.id, u.category, u.text) for u in utterances])},
    }


def _cmd_agent(args: argparse.Namespace) -> int:
    from . import graph
    from .preflight import _captured_logs, prepare_env, silence_side_outputs
    from .stream_context import load_stream_context

    if not spend._RUN_NAME.fullmatch(args.label):
        emit("--label は英数字と _.- で 64 字までにしてください")
        return EXIT_USAGE
    characters = [c for c in args.characters.split(",") if c] if args.characters else list(AGENT_CHARACTERS)
    if not characters or any(c not in AGENT_CHARACTERS for c in characters):
        emit(f"--characters は {', '.join(AGENT_CHARACTERS)} のいくつかをコンマ区切りで指定してください")
        return EXIT_USAGE
    try:
        lengths = parse_tool_chars(args.tool_chars)
        llm_override = parse_llm_override(args.llm)
    except MeasureInputError as exc:  # 自分で作った文言（入力の値は含まない）
        emit(str(exc))
        return EXIT_USAGE
    out_dir = args.out_dir or spend.default_spend_dir()
    run_path = out_dir / f"agent_{args.label}.jsonl"
    if run_path.exists():
        emit(f"実行 {args.label} はすでにあります（別の名前で --label）")
        return EXIT_REFUSED
    try:
        utterances = load_utterances(args.utterances)
    except MeasureInputError as exc:
        emit(f"[断りました] {exc}")
        return EXIT_REFUSED
    if args.limit:
        utterances = utterances[: args.limit]

    prepare_env(args.profile, _REPO_ROOT)
    silence_side_outputs()
    if decision_log.is_enabled():
        emit("判定ログを止められませんでした")
        return EXIT_REFUSED
    if not graph._is_tools_enabled():
        # WHY 断る: ツールが無効だと、本番は Agent を通らず単一ノードで答える。Agent を測っても本番の物差しにならない。
        emit("[断りました] L2_ENABLE_TOOLS が無効です（本番は Agent を通りません）")
        return EXIT_REFUSED
    if not graph._is_rag_enabled():
        emit("[警告] L2_ENABLE_RAG が無効で、retrieve_memory は道具に入りません（本番と同じ設定です）")

    stream_context, context_meta = resolve_stream_context(
        out_dir / "stream_context_snapshot.md", refresh=args.refresh_snapshot, load_live=load_stream_context)
    settings = AgentSettings(stream_context, graph._get_agent_invoke_timeout(), graph._get_agent_recursion_limit(),
                             llm_override)
    meta = agent_meta(args.label, args, characters=characters, settings=settings, context_meta=context_meta,
                      lengths=lengths, utterances=utterances)
    meta_path = out_dir / f"agent_{args.label}.meta.json"
    _write_json(meta_path, meta)

    overrides = dict(spend.parse_price_override(text) for text in args.price)
    ledger = spend.SpendLedger(args.spend_dir or spend.default_spend_dir())
    emit(f"agent {args.label}: {len(characters)} キャラ × {len(utterances)} 発話、上限 ${args.max_usd:.2f}、"
         f"時間切れ {settings.timeout_sec:.0f} 秒、recursion_limit {settings.recursion_limit}、"
         f"配信文脈 {context_meta['origin']}（{context_meta['chars']} 字）")
    full, metrics = replay.RunFile(run_path), replay.RunFile(out_dir / f"agent_{args.label}.metrics.jsonl")
    summaries: dict[str, dict[str, Any]] = {}
    fires: Counter = Counter()
    done = 0
    with prevent_sleep(), _captured_logs(out_dir / f"agent_{args.label}.problems.log"), \
            spend.UsageMeter(ledger, args.label, max_usd=args.max_usd, overrides=overrides) as meter, \
            canned_tools(lengths, fires):
        for slug in characters:
            lines: list[dict[str, Any]] = []
            for index, utterance in enumerate(utterances):
                if index and args.gap_sec > 0:
                    time.sleep(args.gap_sec)
                try:
                    line, body = measure_agent_call(slug, utterance, index, settings, ledger, args.label)
                except AgentTimeout as timeout:
                    # 時間切れ: 取れた分（step、道具、費用）を残してから、残りのスレッドを待たずに終える。
                    line = timeout.line or {"i": index, "character": slug, "utterance_id": utterance.id,
                                            "category": utterance.category, "cold": index == 0, "error": "Timeout",
                                            "total_ms": None, "steps": [], "tools": [], "tools_fired": [],
                                            "source": None, "model": None, "usd": 0.0}
                    metrics.append(line)
                    full.append({**line, "response": ""})
                    lines.append(line)
                    summaries[slug] = summarize_agent(lines)
                    meta.update(finished=datetime.now(UTC).isoformat(timespec="seconds"), timed_out=True,
                                summary=summaries)
                    _write_json(meta_path, meta)
                    emit(f"[時間切れ] {slug} の {utterance.id} が {settings.timeout_sec:.0f} 秒を超えました。"
                         f"記録して終えます（取り残したスレッドの課金を止めるため）")
                    spend.hard_exit(EXIT_FAILED)
                lines.append(line)
                metrics.append(line)
                full.append({**line, "response": body})
                done += 0 if line["error"] else 1
            meter.check_capture(done)
            summaries[slug] = summarize_agent(lines)
            print_agent_summary(slug, summaries[slug])
    meta.update(finished=datetime.now(UTC).isoformat(timespec="seconds"), timed_out=False, summary=summaries,
                tools_fired_total=dict(fires))
    _write_json(meta_path, meta)
    failed = sum(s["failed"] for s in summaries.values())
    emit(f"合計 ${sum(s['usd'] for s in summaries.values()):.4f}、失敗 {failed}、"
         f"道具の発火（定型の応答を返した数）{dict(fires)}、usage を拾えず見積もりで数えた呼び出し {meter.stats.uncounted}")
    return EXIT_FAILED if failed else EXIT_OK


# ─── history ─────────────────────────────────────────────────────

# 実走ログの書式がいつから出ているか（git の履歴で確かめた日付。graph.py の logger.info の文面）。
# WHY: 書式の無い期間は、0 件ではなく「データなし」と出して、数字が無いことを誤読させない。
#   - Agent LLM step N 完了 [character=..] latency_ms=..:  2026-05-11 から（commit e2651e8）
#   - Agent 実行完了 [character=..]: latency_ms=.. source=..: 2026-05-08 から（commit 2333898。
#     それ以前は [character=..] が無く、キャラごとに分けられない）
#   - Agent 実行失敗 [character=..]:                         2026-05-08 から（同上）
LOG_FORMATS_SINCE = {"step": "2026-05-11", "total": "2026-05-08", "failed": "2026-05-08"}

_STEP_LINE = re.compile(r"^Agent LLM step (?P<n>\d+) 完了 \[character=(?P<slug>[^\]]+)\] latency_ms=(?P<ms>-?\d+)")
_TOTAL_LINE = re.compile(r"^Agent 実行完了 \[character=(?P<slug>[^\]]+)\]: latency_ms=(?P<ms>\d+) source=")
_FAILED_LINE = re.compile(r"^Agent 実行失敗 \[character=(?P<slug>[^\]]+)\]")
_SLUG_OK = re.compile(r"[a-z0-9_]{1,24}")
_LOG_DATE = re.compile(r"^run_loop_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_\d{6}\.log$")


def default_runs_dir() -> Path:
    """実走ログの場所（リポジトリの logs/runs/、gitignore 済み）。"""
    return replay_set.default_logs_dir() / "runs"


def select_run_logs(runs_dir: Path, since: date) -> tuple[list[Path], Counter]:
    """計測に使う実走ログを選ぶ（replay_set と同じ規則: pytest の残りを除き、本物の配信の回だけ）。

    Returns:
        (ファイル, 規則ごとの本数)。本数は画面に出してよい
    """
    report = replay_set.BuildReport()
    paths = replay_set.select_files(runs_dir, since, report)
    return paths, report.files_by_rule


def log_month(path: Path) -> str | None:
    """ファイル名の日付の月（YYYY-MM）。名前の形が違えば None。"""
    match = _LOG_DATE.match(path.name)
    return f"{match['y']}-{match['m']}" if match else None


def log_day(path: Path) -> str | None:
    """ファイル名の日付（YYYY-MM-DD）。名前の形が違えば None。"""
    match = _LOG_DATE.match(path.name)
    return f"{match['y']}-{match['m']}-{match['d']}" if match else None


def log_messages(path: Path) -> Iterator[str]:
    """実走ログの 1 行ごとの本文（行頭の形が無い続きの行は飛ばす）。本文は呼び出し側が先頭の形だけを見る。"""
    for _name, message in replay_set._messages(path.read_text(encoding="utf-8", errors="replace")):
        yield message


def _slug_of(value: str) -> str:
    return value if _SLUG_OK.fullmatch(value) else "?"


def history_stats(paths: list[Path]) -> dict[str, Any]:
    """実走ログから、キャラごと、月ごとに Agent の全体の時間と step の時間を数える（数字だけ）。

    全体の時間は Agent 実行完了の latency_ms（_run_agent の開始から完了まで）、step の時間は
    Agent LLM step N 完了の latency_ms。行の本文（text=...）は読まない（先頭の形だけを見る）。
    """
    totals: dict[tuple[str, str], list[float]] = {}
    steps: dict[tuple[str, str], list[float]] = {}
    failed: Counter = Counter()
    for path in paths:
        month = log_month(path)
        if month is None:
            continue
        for message in log_messages(path):
            if message.startswith("Agent LLM step "):
                found = _STEP_LINE.match(message)
                if found and int(found["ms"]) >= 0:
                    steps.setdefault((_slug_of(found["slug"]), month), []).append(float(found["ms"]))
            elif message.startswith("Agent 実行完了 "):
                found = _TOTAL_LINE.match(message)
                if found:
                    totals.setdefault((_slug_of(found["slug"]), month), []).append(float(found["ms"]))
            elif message.startswith("Agent 実行失敗 "):
                found = _FAILED_LINE.match(message)
                if found:
                    failed[(_slug_of(found["slug"]), month)] += 1
    return {"totals": totals, "steps": steps, "failed": failed}


def _dist(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "max": max(values) if values else None}


def months_between(first: str, last: str) -> list[str]:
    """YYYY-MM の first から last までの月を並べる。"""
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def history_report(stats: dict[str, Any], months: list[str]) -> dict[str, Any]:
    """history_stats の結果を、キャラ → 月 → {total, steps, failed} の表にする。書式の無い月は None（データなし）。"""
    slugs = sorted({key[0] for key in (*stats["totals"], *stats["steps"], *stats["failed"])})
    table: dict[str, dict[str, Any]] = {}
    for slug in slugs:
        table[slug] = {}
        for month in months:
            key = (slug, month)
            cell: dict[str, Any] = {
                "total": _dist(stats["totals"].get(key, [])) if month >= LOG_FORMATS_SINCE["total"][:7] else None,
                "steps": _dist(stats["steps"].get(key, [])) if month >= LOG_FORMATS_SINCE["step"][:7] else None,
                "failed": stats["failed"].get(key, 0) if month >= LOG_FORMATS_SINCE["failed"][:7] else None,
            }
            table[slug][month] = cell
    return table


def _dist_text(cell: dict[str, Any] | None) -> str:
    if cell is None:
        return "書式なし"
    if cell["n"] == 0:
        return "データなし"
    return f"n={cell['n']} p50={_ms(cell['p50'])} p90={_ms(cell['p90'])}"


def print_history(table: dict[str, dict[str, Any]]) -> None:
    """キャラごと、月ごとの 1 行を画面へ（件数と ms だけ）。"""
    for slug, by_month in table.items():
        emit(f"  {slug}:")
        for month, cell in by_month.items():
            emit(f"    {month}  全体 {_dist_text(cell['total'])}  |  step {_dist_text(cell['steps'])}  |  "
                 f"失敗 {'-' if cell['failed'] is None else cell['failed']}")


def _cmd_history(args: argparse.Namespace) -> int:
    from .preflight import silence_side_outputs

    silence_side_outputs()
    runs_dir = args.logs_dir or default_runs_dir()
    if not runs_dir.is_dir():
        emit(f"実走ログの場所がありません: {runs_dir.name}")
        return EXIT_FAILED
    paths, by_rule = select_run_logs(runs_dir, args.since)
    emit(f"実走ログ: 使う {by_rule.get('使う', 0)} 本（" + "、".join(
        f"{rule} {count}" for rule, count in sorted(by_rule.items()) if rule != "使う") + "）")
    if not paths:
        emit("使える実走ログがありません")
        return EXIT_FAILED
    months_present = sorted({m for m in map(log_month, paths) if m})
    months = months_between(months_present[0], months_present[-1])
    table = history_report(history_stats(paths), months)
    print_history(table)
    out_dir = args.out_dir or spend.default_spend_dir()
    _write_json(out_dir / f"history_{args.label}.json", {
        "tool": "history", "label": args.label, "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "since": args.since.isoformat(), "files": {"used": len(paths), "by_rule": dict(by_rule)},
        "formats_since": LOG_FORMATS_SINCE, "months": months, "table": table,
        "notes": ["全体 = Agent 実行完了の latency_ms（_run_agent の開始から完了まで）、step = Agent LLM step N 完了の latency_ms",
                  "月は実走ログのファイル名の日付。書式なし = その書式のログが出る前の月、データなし = 書式はあるが該当の行が 0 件"],
    })
    emit(f"書き出し: history_{args.label}.json")
    return EXIT_OK


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
    history = sub.add_parser("history", help="実走ログから、キャラごと・月ごとの Agent の時間を数える（課金なし）")
    history.add_argument("--label", default="history", help="出力のファイル名（history_<名前>.json）")
    history.add_argument("--since", type=date.fromisoformat, default=replay_set.DEFAULT_SINCE,
                         help="この日以降のログを使う（既定は判定の本文が残る日）")
    history.add_argument("--logs-dir", type=Path, default=None, help="logs/runs/ の場所（既定はリポジトリの logs/runs/）")
    history.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    agent = sub.add_parser("agent", help="本番と同じ Agent の時間、step ごとの tokens、道具の発火、費用（課金あり）")
    agent.add_argument("--profile", choices=("dev", "prod"), required=True)
    agent.add_argument("--label", required=True, help="実行の名前（英数字と _.- で 64 字まで）")
    agent.add_argument("--max-usd", type=float, required=True, help="この実行の上限（同じ名前の過去の分を含む）")
    agent.add_argument("--characters", default="", help=f"コンマ区切り（既定は {','.join(AGENT_CHARACTERS)}）")
    agent.add_argument("--limit", type=int, default=0, help="発話を先頭から N 本だけ流す（少数で試す用）")
    agent.add_argument("--gap-sec", type=float, default=2.0, help="呼び出しの間（秒）。既定 2")
    agent.add_argument("--llm", default=None, help="全キャラのモデルを provider:model で差し替える（食べ比べ用）")
    agent.add_argument("--tool-chars", action="append", default=[], help="道具の定型の応答の長さ 道具名=文字数")
    agent.add_argument("--utterances", type=Path, default=None, help="計測の発話（既定は measure_sets/ の中）")
    agent.add_argument("--refresh-snapshot", action="store_true", help="配信文脈の写しを今のもので取り直す")
    agent.add_argument("--price", action="append", default=[], help="model=in,out[,cache_read[,cache_write]]")
    agent.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    agent.add_argument("--spend-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.probe <raw|agent|history>``。"""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.command == "raw":
        if args.n < 1 or args.gap_sec < 0:
            emit("--n は 1 以上、--gap-sec は 0 以上にしてください")
            return EXIT_USAGE
        return _cmd_raw(args)
    if args.command == "history":
        if not spend._RUN_NAME.fullmatch(args.label):
            emit("--label は英数字と _.- で 64 字までにしてください")
            return EXIT_USAGE
        return _cmd_history(args)
    if args.command == "agent":
        if args.limit < 0 or args.gap_sec < 0:
            emit("--limit と --gap-sec は 0 以上にしてください")
            return EXIT_USAGE
        return _cmd_agent(args)
    return EXIT_USAGE


if __name__ == "__main__":
    run_tool(main, name="probe")
