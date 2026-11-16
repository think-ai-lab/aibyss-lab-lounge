"""
replay_set.py — 判定のリプレイに使う固定のセットを作る（計測の道具）

責務:
  - 実走ログ（logs/runs/）から、判定（承認、挙手の候補、意図ゲート、複数名の振り分け）と callout の入力を
    全文で組み直し、照合できた行だけを data/replay/set.jsonl へ書く
  - 判定ログ（logs/decisions/、L2_DECISION_LOG=true のときだけ書かれる）の行も同じ形で混ぜる
  - 人が外したい行に印を付ける確認のファイル（data/replay/review.md）を作り、印を除外の一覧へ移す
  - 合成の行（measure_sets/replay_synthetic.jsonl）を読む
  - 判定の種類ごとに、本物の判定の関数の呼び方と、答えを 1 つの文字列（result_key）にそろえる方法を持つ

本番の経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない。

【WHY 実走ログから組み直せるか】
  判定のログ（INFO）は入力の先頭 30 字しか残さない（router._text_preview）。それでも全文に戻せる:
    - 背景の聞き取りは、セグメントを足すたびに ``BG buffer 蓄積: <全文> (segments=N chars=M)`` を出す
    - バッファは末尾に足し、先頭からしか消さない（transcript_buffer.py）。判定はその直後に同じスレッドで、
      バッファの全文（full_text、意図ゲートは extract_context）にかかる
    - だから、直前の BG の行の N を使い、それまでのセグメントの末尾 N 個を改行でつなげば入力に戻る
  外れを混ぜないよう、2 つで照合する: 末尾 N 個の長さの合計 == M、_text_preview(組み直した入力) == ログの text=。
  どちらかが合わない行は捨て、理由ごとに数だけ数える。

【使い方】
  # セットを作る（画面には件数だけ。本文は data/replay/ のファイルにだけ書く）
  uv run --frozen --all-extras python -m lab_lounge.replay_set build

  # review.md の [x] を除外の一覧（data/replay/exclude.txt、id だけ）へ移す
  uv run --frozen --all-extras python -m lab_lounge.replay_set exclude

【出力の約束】
  画面には、ファイルの数、種類ごとの件数、捨てた理由ごとの件数だけを出す。発話の本文は出さない。
  判定の関数が出すログ（30 字のプレビューを含む）は捕まえて、画面へ流さない。
  例外は型と行番号だけを出す（spend.run_tool）。
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import decision_log, router
from .spend import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, run_tool
from .transcript_buffer import TranscriptBuffer, TranscriptSegment

# ─── 定数 ─────────────────────────────────────────────────────────

KINDS = ("approval", "intent", "interjection", "route")

# 種類ごと（intent は callout と mention に分ける）に、セットへ入れる目標の数。
TARGETS = {"approval": 100, "interjection": 120, "callout": 70, "mention": 40, "route": 20}

# 1 つの塊（同じ会話のひとまとまり）から、区分ごと・「決めた / 決めなかった」ごとに取る上限。
# WHY: 伸びていくバッファの判定は、ほとんど同じ入力が続く。上限が無いと、1 つの長い会話がセットを占める。
PER_EPOCH = 2

# 塊の区切り: バッファが空から始まった（segments=1）か、前の行から この秒数より空いたとき。
# WHY: 満杯のバッファは古い方から消えていくので、セグメントの数は増えない。数の増減で区切ると、
#      長い会話では毎行が新しい塊になり、上限が効かない。30 秒は本番のバッファの窓（L2_CONTEXT_WINDOW_SEC）の既定。
EPOCH_GAP_SEC = 30.0

# 区分ごとに、「何も決めなかった行」（当時の答えが none / unknown）に取っておく割合。足りなければある分だけ。
# WHY: 本番で一番多いのは「誰も手を挙げない」「挙手中の無関係な発話」。決めた行だけを集めると、新しいモデルが
#      余計に手を挙げる、関係ない発話で許可する、という変化に気づけない。
UNDECIDED_SHARE = {"approval": 0.3, "interjection": 0.5, "route": 0.3}

# 判定の本文がログに入り始めた日（router._text_preview の導入）。これより前の実走ログは組み直せない。
DEFAULT_SINCE = date(2026, 5, 8)

# 意図ゲートの入力（extract_context）を作るときの、文字数の上限。本番の既定（L2_CONTEXT_MAX_CHARS）と同じ。
DEFAULT_CONTEXT_MAX_CHARS = 2000

DEFAULT_SEED = 20261116

# ログの書式（log_setup._LOG_FORMAT: "%(asctime)s %(levelname)-8s %(name)s: %(message)s"）。
_LINE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<level>[A-Z]+) +(?P<name>[\w.]+): (?P<msg>.*)$")

# 組み直しに使うログの文言。本番のソースから消えたらテストが赤くなる（番人）。
BG_BUFFER_FORMAT = "BG buffer 蓄積: %r (segments=%d chars=%d)"
BG_WAKE_FORMAT = "BG wake 検知: %s (context=%d chars)"
AGENT_INVOKE_FORMAT = "Agent.invoke 開始 [character=%s] text_len=%d text=%r"
ROUTE_DELEGATE_PREFIX = "複数キャラクター検出: "
JUDGMENT_PREFIXES = {
    "interjection_candidate 応答: ": "interjection",
    "check_approval 応答: ": "approval",
    "意図ゲート応答: ": "intent",
    "LLM ルーター応答: ": "route",
}
# Agent.invoke の本文の前置き（graph._generation_node が付ける）。callout の入力はこれを外した後。
CALLOUT_PREFIX = "[ルカからの発言]\n"

_BG_BUFFER = re.compile(r"^BG buffer 蓄積: (?P<text>.*) \(segments=(?P<n>\d+) chars=(?P<chars>\d+)\)$")
_BG_WAKE = re.compile(r"^BG wake 検知: (?P<slug>[\w-]+) \(context=(?P<len>\d+) chars\)$")
_AGENT_INVOKE = re.compile(r"^Agent\.invoke 開始 \[character=(?P<slug>[^\]]+)\] text_len=(?P<len>\d+) text=(?P<text>.*)$")
_ROUTE_DELEGATE = re.compile(r"^複数キャラクター検出: (?P<list>\[[^\]]*\]) → LLM ルーターへ委譲")
_FILE_NAME = re.compile(r"^run_loop_(?P<date>\d{8})_(?P<time>\d{6})\.log$")
_ID = re.compile(r"^(?:[0-9a-f]{12}|s\d{3,})$")
_SLUG = r"[a-z0-9_]+"
_EXPECTED = {
    "approval": re.compile(rf"^(?:granted:{_SLUG}|denied:{_SLUG}|none)$"),
    "intent": re.compile(r"^(?:callout|mention|unknown)$"),
    "interjection": re.compile(rf"^(?:{_SLUG})$"),
    "route": re.compile(rf"^(?:{_SLUG})$"),
}

# pytest の残りの実走ログを、中身で見分ける規則。
# WHY: 開発中は pytest も logs/runs/ へ書いていたので、名前だけでは見分けられない。
_REAL_RUN_MARK = "Agent LLM step 1 完了"
_TEST_RUN_MARKS = ("invalid_char", "boom", "unknown_char")
_TEST_MARK = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(_TEST_RUN_MARKS) + r")(?![A-Za-z0-9_])")
# 発話の本文をそのまま含む行。テストの印はここでは照らさない（発話の中の「boom」で本物の回を落とさない）。
_UTTERANCE_PREFIXES = ("STT 認識結果", "BG buffer 蓄積", "Agent.invoke 開始", "BG 名前ゲート", "BG: on_segment_added",
                       *JUDGMENT_PREFIXES)
# repr で書かれた文字列（発話の断片を含みうる）。印を照らす前に取り除く。
_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def default_out_dir() -> Path:
    """セットと確認のファイルを置く場所（リポジトリの data/replay/、gitignore 済み）。"""
    return _repo_root() / "data" / "replay"


def default_logs_dir() -> Path:
    """実走ログと判定ログの親（リポジトリの logs/、gitignore 済み）。"""
    return _repo_root() / "logs"


def default_synthetic_path() -> Path:
    """合成の行のファイル（commit する。本文はすべて合成）。"""
    return Path(__file__).resolve().parent / "measure_sets" / "replay_synthetic.jsonl"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ─── 判定の呼び方と、答えのそろえ方 ──────────────────────────────

def judge(row: dict[str, Any]) -> Any:
    """行の種類に合わせて、本物の判定の関数を呼ぶ。判定の論理は複製しない。

    Returns:
        判定の関数の戻り値（ApprovalResult / IntentResult / CharacterConfig / "none" / None）
    """
    from .characters import get_character

    kind, text = row["kind"], row["input"]
    if kind == "approval":
        return router.check_approval(text, list(row["candidates"]))
    if kind == "intent":
        return router.check_intent(text, row["character"])
    if kind == "interjection":
        return router.check_intent(text, None)
    if kind == "route":
        return router._route_by_llm(text, [get_character(slug) for slug in row["candidates"]])
    raise ValueError(f"知らない判定の種類です: {kind}")


def result_key(kind: str, value: Any) -> str:
    """判定の戻り値を、比べやすい 1 つの文字列にそろえる。

    - approval:     granted:<slug> / denied:<slug> / none
    - intent:       callout / mention / unknown
    - interjection: <slug> / unknown
    - route:        <slug> / none / unknown（答えが候補に無い）
    呼び出しの失敗（error:<型>）は、記録の包みの側で分ける。ここでは数えない。
    """
    if kind == "approval":
        if value is None:
            return "none"
        return f"{'granted' if value.granted else 'denied'}:{value.target_slug}"
    if kind == "intent":
        return value.intent
    if kind == "interjection":
        return value.target_slug if value.intent == "interjection_candidate" and value.target_slug else "unknown"
    if kind == "route":
        if value is None:
            return "unknown"
        return value if isinstance(value, str) else value.slug
    raise ValueError(f"知らない判定の種類です: {kind}")


@contextlib.contextmanager
def scripted_router(answer: Callable[[str, str, str], str | None]) -> Iterator[None]:
    """router._call_router_llm を、決めた答えを返す関数に差し替える。出るときに戻す。

    WHY: 当時の生の答え（ログの先頭の %r）を、本物の判定の関数に解釈させるため（解釈の規則を複製しない）。
         API は呼ばない。
    """
    original = router._call_router_llm
    router._call_router_llm = answer
    try:
        yield
    finally:
        router._call_router_llm = original


def interpret(row: dict[str, Any], raw: str | None) -> str:
    """生の答えを、本物の判定の関数で解釈した result_key を返す。"""
    with scripted_router(lambda _model, _system, _text: raw):
        return result_key(row["kind"], judge(row))


def bucket(row: dict[str, Any]) -> str:
    """目標の数を数える区分。intent は callout と mention に分ける（期待値か当時の答えで）。"""
    if row["kind"] != "intent":
        return row["kind"]
    label = row.get("expected") or row.get("hist_key") or "callout"
    return "mention" if label == "mention" else "callout"


def row_id(kind: str, character: str | None, candidates: Iterable[str], text: str) -> str:
    """行の id。同じ入力の同じ判定は同じ id になる（重複を 1 つにする鍵）。"""
    key = "\0".join([kind, character or "", ",".join(candidates), text])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# ─── 実走ログの読み方 ────────────────────────────────────────────

@dataclass
class BuildReport:
    """build の集計。画面に出してよい（数だけ）。notes は数の読み方の注記（偏りなど）。"""

    files_seen: int = 0
    notes: list[str] = field(default_factory=list)
    files_by_rule: Counter = field(default_factory=Counter)
    found: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)
    kept: Counter = field(default_factory=Counter)
    informative: Counter = field(default_factory=Counter)
    undecided: Counter = field(default_factory=Counter)
    epochs: int = 0
    decision_rows: int = 0


def test_mark_in(text: str) -> str | None:
    """発話でない行に、pytest の目印（invalid_char / boom / unknown_char）があれば、その印を返す。

    WHY 発話でない行だけ: 発話や判定のプレビューの中の「boom」で、本物の配信の回を丸ごと落とさないため。
         見るのは行頭の形がある行だけで、発話を含む行（STT、BG のバッファ、Agent の入力、判定）は飛ばし、
         ほかの行も repr で書かれた文字列を除いてから照らす。
    """
    for _ts, name, msg in _records(text):
        if msg.startswith(_UTTERANCE_PREFIXES):
            continue
        found = _TEST_MARK.search(name) or _TEST_MARK.search(_QUOTED.sub("", msg))
        if found:
            return found[1]
    return None


def select_files(runs_dir: Path, since: date, report: BuildReport, *, require_agent: bool = True) -> list[Path]:
    """組み直しに使う実走ログを選ぶ。規則ごとに外した本数を report.files_by_rule に数える。

    規則:
        - 名前の形（run_loop_YYYYMMDD_HHMMSS.log）と日付（since 以降）
        - 名前の規則: 2026-10-31 の全部と、2026-11-01 の 02:54:05 までは pytest の残り
        - 中身の規則: 発話でない行に pytest の目印が無い（印ごとに数える）
        - require_agent=True なら、Agent の応答が 1 度はあった（Agent LLM step 1 完了）回だけ

    Args:
        require_agent: False にすると、Agent が 1 度も完了しなかった回も使う。
                       WHY: Agent の時間やタイムアウトを数えるとき、最も悪い回（1 度も答えられなかった回）を落とさないため。
                       True のときは、外れた回の数を偏りの注記として report.notes に残す
    """
    chosen = []
    without_agent = 0
    for path in sorted(runs_dir.rglob("run_loop_*.log")):
        report.files_seen += 1
        match = _FILE_NAME.match(path.name)
        if match is None:
            report.files_by_rule["名前の形が違う"] += 1
            continue
        day = date(int(match["date"][:4]), int(match["date"][4:6]), int(match["date"][6:]))
        if day < since:
            report.files_by_rule["日付が古い（本文のプレビューが無い）"] += 1
            continue
        if match["date"] == "20261031" or (match["date"] == "20261101" and match["time"] <= "025405"):
            report.files_by_rule["名前の規則（pytest の残り）"] += 1
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        mark = test_mark_in(content)
        if mark is not None:
            report.files_by_rule[f"中身の規則（pytest の目印 {mark}）"] += 1
            continue
        if _REAL_RUN_MARK not in content:
            without_agent += 1
            if require_agent:
                report.files_by_rule["Agent の応答が 1 度も無い回"] += 1
                continue
        report.files_by_rule["使う"] += 1
        chosen.append(path)
    if require_agent and without_agent:
        report.notes.append(f"Agent の応答が 1 度も無かった回 {without_agent} 本は入っていません（最も悪い回が抜ける偏り）")
    elif without_agent:
        report.notes.append(f"Agent の応答が 1 度も無かった回 {without_agent} 本を含みます")
    return chosen


def _records(text: str) -> Iterator[tuple[str, str, str]]:
    """ログの行を (時刻, logger 名, 本文) で返す。行頭の形が無い行（複数行の続き）は飛ばす。

    WHY splitlines: ログは Windows では CRLF で書かれる。JSONL と違い、本文の中の改行は repr で
         エスケープされている（STT の行の続きは、行頭の形が無いので飛ばす）。
    """
    for line in text.splitlines():
        match = _LINE.match(line)
        if match:
            yield match["ts"], match["name"], match["msg"]


def _messages(text: str) -> Iterator[tuple[str, str]]:
    """ログの行を (logger 名, 本文) で返す。行頭の形が無い行（複数行の続き）は飛ばす。"""
    for _ts, name, msg in _records(text):
        yield name, msg


def _literal_str(text: str) -> str | None:
    """%r で書かれた文字列を戻す。戻せなければ None（例外の文言は本文を含みうるので使わない）。"""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, str) else None


@dataclass
class _Judgment:
    kind: str
    answer: str
    model: str | None
    candidates: list[str]
    character: str | None
    preview: str


def _parse_judgment(kind: str, rest: str) -> _Judgment | None:
    """判定の行の本文（前置きの後）を読む。形が違えば None。

    本文の形: ``<答えの repr> (model=<m> [candidates=[...] | char=<slug>] text=<プレビュー>)``。
    答えの repr とプレビューには ``)`` や `` (model=`` が入りうるので、答えは literal として読める最短の
    切れ目を探し、プレビューは行末の ``)`` から逆に取る。
    """
    if not rest.endswith(")"):
        return None
    for match in re.finditer(r" \(model=", rest):
        answer = _literal_str(rest[: match.start()])
        if answer is None:
            continue
        tail = rest[match.end(): -1]
        model, _, tail = tail.partition(" ")
        candidates: list[str] = []
        character = None
        if tail.startswith("candidates="):
            found = re.match(r"candidates=(\[[^\]]*\]) ", tail)
            if not found:
                return None
            parsed = ast.literal_eval(found[1]) if re.fullmatch(r"\[[\w', ]*\]", found[1]) else None
            if not isinstance(parsed, list):
                return None
            candidates = [str(slug) for slug in parsed]
            tail = tail[found.end():]
        elif tail.startswith("char="):
            character, _, tail = tail[len("char="):].partition(" ")
        if not tail.startswith("text="):
            return None
        return _Judgment(kind, answer, model or None, candidates, character, tail[len("text="):])
    return None


@dataclass
class _FileState:
    """1 本の実走ログを読む間の状態。"""

    stem: str
    day: str
    segments: list[str] = field(default_factory=list)
    last_bg: tuple[int, int] | None = None
    last_bg_at: datetime | None = None
    epoch: int = 0
    pending_wake: tuple[str, int] | None = None
    pending_route: list[str] | None = None


def rows_from_runlog(path: Path, report: BuildReport, *, context_max_chars: int) -> list[dict[str, Any]]:
    """1 本の実走ログから、照合できた判定の行と callout の行を作る。"""
    match = _FILE_NAME.match(path.name)
    assert match is not None
    day = f"{match['date'][:4]}-{match['date'][4:6]}-{match['date'][6:]}"
    state = _FileState(stem=path.stem, day=day)
    rows: list[dict[str, Any]] = []
    for stamp, _name, msg in _records(path.read_text(encoding="utf-8", errors="replace")):
        if msg.startswith("BG buffer 蓄積: "):
            _on_bg(state, msg, report, at=_log_time(stamp))
        elif msg.startswith("BG wake 検知: "):
            found = _BG_WAKE.match(msg)
            state.pending_wake = (found["slug"], int(found["len"])) if found else None
        elif msg.startswith("Agent.invoke 開始 "):
            row = _on_agent(state, msg, report)
            if row:
                rows.append(row)
        elif msg.startswith(ROUTE_DELEGATE_PREFIX):
            found = _ROUTE_DELEGATE.match(msg)
            parsed = ast.literal_eval(found["list"]) if found and re.fullmatch(r"\[[\w', ]*\]", found["list"]) else None
            state.pending_route = [str(s) for s in parsed] if isinstance(parsed, list) else None
        else:
            for prefix, kind in JUDGMENT_PREFIXES.items():
                if msg.startswith(prefix):
                    row = _on_judgment(state, kind, msg[len(prefix):], report, context_max_chars)
                    if row:
                        rows.append(row)
                    break
    report.epochs += state.epoch
    return rows


def _log_time(stamp: str) -> datetime | None:
    """ログの時刻（2026-06-01 20:00:00,123）を読む。読めなければ None。"""
    try:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def _new_epoch(previous: datetime | None, now: datetime | None) -> bool:
    """前の行から EPOCH_GAP_SEC より空いたか（時刻が読めなければ区切る）。"""
    return previous is None or now is None or (now - previous).total_seconds() > EPOCH_GAP_SEC


def _on_bg(state: _FileState, msg: str, report: BuildReport, *, at: datetime | None = None) -> None:
    found = _BG_BUFFER.match(msg)
    text = _literal_str(found["text"]) if found else None
    if found is None or text is None:
        # 読めないセグメントがあると、以後の末尾 N 個がずれる。このファイルの残りは組み直さない。
        report.dropped["BG の行を読めない（以後のファイルの残りを使わない）"] += 1
        state.segments.append("\0unreadable")
        state.last_bg = None
        return
    n, chars = int(found["n"]), int(found["chars"])
    if state.last_bg is None or n == 1 or _new_epoch(state.last_bg_at, at):
        state.epoch += 1
    state.segments.append(text)
    state.last_bg = (n, chars)
    state.last_bg_at = at


def _buffer_tail(state: _FileState, report: BuildReport) -> list[str] | None:
    """直前の BG の行が示すバッファの中身（末尾 N 個）を返す。照合できなければ理由を数えて None。"""
    if state.last_bg is None:
        report.dropped["直前の BG の行が無い"] += 1
        return None
    n, chars = state.last_bg
    if n > len(state.segments) or n == 0:
        report.dropped["セグメントの数が足りない"] += 1
        return None
    tail = state.segments[-n:]
    if any(seg.startswith("\0") for seg in tail):
        report.dropped["読めないセグメントを含む"] += 1
        return None
    if sum(len(seg) for seg in tail) != chars:
        report.dropped["文字数が合わない"] += 1
        return None
    return tail


def _context_of(tail: list[str], context_max_chars: int) -> str:
    """意図ゲートの入力（extract_context）を、本物の TranscriptBuffer に同じセグメントを入れて作る。

    WHY 本物を使う: 切り詰めの規則（改行で揃える）を複製しない。時間で消えないよう window は大きくする。
    """
    buffer = TranscriptBuffer(window_sec=1e12, max_chars=context_max_chars)
    for text in tail:
        buffer.add(TranscriptSegment(text=text, timestamp=0.0, duration_ms=0))
    return buffer.extract_context()


def _on_judgment(state: _FileState, kind: str, rest: str, report: BuildReport,
                 context_max_chars: int) -> dict[str, Any] | None:
    report.found[kind] += 1
    parsed = _parse_judgment(kind, rest)
    route_candidates, state.pending_route = state.pending_route, None
    if parsed is None:
        report.dropped["判定の行の形が違う"] += 1
        return None
    tail = _buffer_tail(state, report)
    if tail is None:
        return None
    text = _context_of(tail, context_max_chars) if kind == "intent" else "\n".join(tail)
    if router._text_preview(text) != parsed.preview:
        report.dropped["プレビューが合わない"] += 1
        return None
    candidates = parsed.candidates
    if kind == "route":
        if not route_candidates:
            report.dropped["振り分けの候補が分からない"] += 1
            return None
        candidates = route_candidates
    if kind == "intent" and not parsed.character:
        report.dropped["判定の行の形が違う"] += 1
        return None
    return _row(kind=kind, text=text, character=parsed.character if kind == "intent" else None,
                candidates=candidates, source="runlog", hist_raw=parsed.answer, hist_model=parsed.model,
                epoch=f"{state.stem}#{state.epoch}", day=state.day)


def _on_agent(state: _FileState, msg: str, report: BuildReport) -> dict[str, Any] | None:
    """Agent.invoke の行が、直前の BG wake 検知の発話そのものなら callout の行にする。

    WHY wake と組にする: Agent は挙手の承認や協働でも呼ばれ、そのときの本文はルカの呼びかけではない。
         直前の wake のキャラと文脈の長さが一致したものだけを、呼びかけとして使う。
    """
    found = _AGENT_INVOKE.match(msg)
    wake, state.pending_wake = state.pending_wake, None
    text = _literal_str(found["text"]) if found else None
    if found is None or text is None:
        report.dropped["Agent の行を読めない"] += 1
        return None
    text = text.removeprefix(CALLOUT_PREFIX)
    if wake is None or wake != (found["slug"], len(text)):
        report.dropped["Agent の行が wake の発話でない"] += 1
        return None
    report.found["callout"] += 1
    return _row(kind="intent", text=text, character=found["slug"], candidates=[], source="runlog",
                hist_raw=None, hist_model=None, epoch=f"{state.stem}#{state.epoch}", day=state.day)


def _row(*, kind: str, text: str, character: str | None, candidates: list[str], source: str,
         hist_raw: str | None, hist_model: str | None, epoch: str, day: str) -> dict[str, Any]:
    return {
        "id": row_id(kind, character, candidates, text), "kind": kind, "input": text, "character": character,
        "candidates": candidates, "source": source, "expected": None, "alt_ok": [], "hist_raw": hist_raw,
        "hist_model": hist_model, "epoch": epoch, "date": day,
    }


def rows_from_decision_log(directory: Path, since: date, report: BuildReport) -> list[dict[str, Any]]:
    """判定ログ（decisions_YYYYMMDD.jsonl）の行を、セットの行の形にする。

    入力は全文なので照合は要らない。呼び出しに失敗した行（error あり、raw が null）は判定ではないので捨てる。
    """
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("decisions_*.jsonl")):
        stamp = path.stem.removeprefix("decisions_")
        if not re.fullmatch(r"\d{8}", stamp) or date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])) < since:
            continue
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        # 判定ログの塊は時刻で区切る（EPOCH_GAP_SEC より空いたら次の塊）。
        epoch, previous = 0, None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                report.dropped["判定ログの行を読めない"] += 1
                continue
            kind = item.get("kind") if isinstance(item, dict) else None
            if kind not in KINDS:
                report.dropped["判定ログの種類が違う"] += 1
                continue
            report.found[kind] += 1
            if item.get("error") or item.get("raw") is None or not isinstance(item.get("input"), str):
                report.dropped["判定ログの呼び出しが失敗した行"] += 1
                continue
            candidates = [str(s) for s in item.get("candidates") or []]
            character = item.get("character") if kind == "intent" else None
            if kind == "intent" and not character:
                report.dropped["判定ログの種類が違う"] += 1
                continue
            report.decision_rows += 1
            try:
                at = datetime.fromisoformat(str(item.get("ts"))).replace(tzinfo=None)
            except ValueError:
                at = None
            if _new_epoch(previous, at):
                epoch += 1
            previous = at
            rows.append(_row(kind=kind, text=item["input"], character=character, candidates=candidates,
                             source="decision_log", hist_raw=str(item["raw"]), hist_model=item.get("model"),
                             epoch=f"{path.stem}#{epoch}", day=day))
    return rows


# ─── 選び方 ──────────────────────────────────────────────────────

def _length_band(text: str) -> int:
    return 0 if len(text) <= 30 else 1 if len(text) <= 200 else 2


def _informative(row: dict[str, Any]) -> bool:
    """当時の答えが「何かを決めた」答えか（承認の granted / denied、候補の slug、callout / mention など）。

    WHY 先に取る: 承認の判定はほとんどが none。そのまま無作為に取ると、granted / denied の行が落ちる。
    """
    key = row.get("hist_key")
    return key is not None and key not in ("none", "unknown")


def _take(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """長さと日付の層を順にめぐって limit 行まで取る。(取った行, 残り) を返す。"""
    strata: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(_length_band(row["input"]), row["date"] or "")].append(row)
    queues = [strata[key] for key in sorted(strata)]
    for queue in queues:
        rng.shuffle(queue)
    taken: list[dict[str, Any]] = []
    while len(taken) < limit and any(queues):
        for queue in queues:
            if queue and len(taken) < limit:
                taken.append(queue.pop())
    return taken, [row for queue in queues for row in queue]


def select_rows(rows: list[dict[str, Any]], report: BuildReport, *, seed: int) -> list[dict[str, Any]]:
    """重複を除き、塊ごとの上限をかけ、区分ごとに「何も決めなかった行」の枠を取ってから、長さと日付の層に分けて
    目標の数まで取る。固定の seed で決まる。"""
    rng = random.Random(seed)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["id"] in unique:
            report.dropped["同じ入力の重複"] += 1
        else:
            unique[row["id"]] = row

    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        groups[(row["epoch"], bucket(row), _informative(row))].append(row)
    capped: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        capped.extend(group[:PER_EPOCH])
        if len(group) > PER_EPOCH:
            report.dropped["同じ塊の上限を超えた"] += len(group) - PER_EPOCH

    chosen: list[dict[str, Any]] = []
    for name, target in TARGETS.items():
        pool = [r for r in capped if bucket(r) == name]
        decided = [r for r in pool if _informative(r)]
        undecided = [r for r in pool if not _informative(r)]
        quota = round(target * UNDECIDED_SHARE.get(name, 0.0))
        first, undecided = _take(undecided, quota, rng)
        second, _ = _take(decided, target - len(first), rng)
        third, _ = _take(undecided, target - len(first) - len(second), rng)  # 決めた行が足りなければ埋める
        for row in [*first, *second, *third]:
            chosen.append(row)
            report.kept[name] += 1
            report.informative[name] += _informative(row)
            report.undecided[name] += not _informative(row)
        if len(pool) > report.kept[name]:
            report.dropped["目標の数を超えた"] += len(pool) - report.kept[name]
    return chosen


# ─── 書き出しと読み込み ──────────────────────────────────────────

def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _one_line(text: str) -> str:
    """確認のファイルの 1 行に収める（改行は ⏎、タブは空白）。"""
    return text.replace("\r\n", "⏎").replace("\n", "⏎").replace("\r", "⏎").replace("\t", " ")


_REVIEW_HEADER = """\
# 除外の確認

外したい行の `[ ]` を `[x]` にして保存し、`python -m lab_lounge.replay_set exclude` を流してください。
外す目安: 判定の入力として成り立たない（認識の崩れ、雑音、別の人の声）、公開の場で使いたくない、など。
行の形は `- [ ] <id> <種類> <候補または対象> | <本文>`。本文の改行は ⏎ で表しています。

"""


def review_line(row: dict[str, Any]) -> str:
    target = row["character"] if row["kind"] == "intent" else ",".join(row["candidates"]) or "-"
    return f"- [ ] {row['id']} {row['kind']} {target} | {_one_line(row['input'])}"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, report: BuildReport, *, seed: int) -> None:
    """set.jsonl、set.meta.json、review.md を書く。"""
    ordered = sorted(rows, key=lambda r: (KINDS.index(r["kind"]), r["date"], r["id"]))
    _write_atomic(out_dir / "set.jsonl", "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                  for r in ordered))
    meta = {
        "seed": seed, "targets": TARGETS, "per_epoch": PER_EPOCH, "epoch_gap_sec": EPOCH_GAP_SEC,
        "undecided_share": UNDECIDED_SHARE, "kept": dict(report.kept), "informative": dict(report.informative),
        "undecided": dict(report.undecided), "notes": report.notes, "files": dict(report.files_by_rule),
        "dropped": dict(report.dropped), "found": dict(report.found), "epochs": report.epochs,
    }
    _write_atomic(out_dir / "set.meta.json", json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True))
    _write_atomic(out_dir / "review.md", _REVIEW_HEADER + "".join(review_line(r) + "\n" for r in ordered))


def load_set(out_dir: Path) -> list[dict[str, Any]]:
    """set.jsonl を読む。"""
    path = out_dir / "set.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_synthetic(path: Path | None = None) -> list[dict[str, Any]]:
    """合成の行を読み、セットの行の形にそろえる。ファイルが無ければ空。

    Raises:
        ValueError: 形が違う行がある（行番号だけを知らせる。本文は出さない）
    """
    path = path or default_synthetic_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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
        if not _valid_synthetic(item) or item["id"] in seen:
            bad.append(number)
            continue
        seen.add(item["id"])
        rows.append({
            "id": item["id"], "kind": item["kind"], "input": item["input"],
            "character": item.get("character") if item["kind"] == "intent" else None,
            "candidates": list(item.get("candidates") or []), "source": "synthetic",
            "expected": item["expected"], "alt_ok": list(item.get("alt_ok") or []),
            "hist_raw": None, "hist_model": None, "epoch": f"synthetic#{item['id']}", "date": None,
        })
    if bad:
        raise ValueError(f"合成の行の形が違います（{path.name} の {', '.join(map(str, bad))} 行目）")
    return rows


def _valid_synthetic(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    kind = item.get("kind")
    if not (isinstance(item.get("id"), str) and re.fullmatch(r"s\d{3,}", item["id"]) and kind in KINDS):
        return False
    if not (isinstance(item.get("input"), str) and item["input"].strip()):
        return False
    expected = item.get("expected")
    if not (isinstance(expected, str) and _EXPECTED[kind].fullmatch(expected)):
        return False
    candidates, alt_ok = item.get("candidates") or [], item.get("alt_ok") or []
    if not (isinstance(candidates, list) and isinstance(alt_ok, list)
            and all(isinstance(s, str) and re.fullmatch(_SLUG, s) for s in [*candidates, *alt_ok])):
        return False
    if kind == "intent":
        return isinstance(item.get("character"), str) and bool(re.fullmatch(_SLUG, item["character"]))
    if kind in ("approval", "route"):
        return bool(candidates)
    return True


@dataclass
class ExcludeResult:
    excluded: list[str]
    malformed_lines: list[int]
    unknown_ids: int


def read_review(review_path: Path, known_ids: set[str]) -> ExcludeResult:
    """review.md の印を読む。``- [x]`` の id を除外にする。形の崩れた行は行番号だけを返す。"""
    excluded: list[str] = []
    malformed: list[int] = []
    unknown = 0
    for number, line in enumerate(review_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("- ["):
            continue
        match = re.match(r"^- \[(?P<mark>.)\] (?P<id>\S+) ", line)
        if match is None or match["mark"] not in (" ", "x", "X") or not _ID.match(match["id"]):
            malformed.append(number)
            continue
        if match["mark"] in ("x", "X"):
            if match["id"] not in known_ids:
                unknown += 1
                continue
            excluded.append(match["id"])
    return ExcludeResult(sorted(set(excluded)), malformed, unknown)


def load_exclusions(out_dir: Path) -> tuple[set[str], str | None]:
    """除外の id と、古いままのときの警告の文（無ければ None）を返す。

    WHY 警告: review.md に印を付けたのに exclude を流し忘れると、外したはずの行がリプレイに混ざる。
    """
    exclude, review, set_path = out_dir / "exclude.txt", out_dir / "review.md", out_dir / "set.jsonl"
    ids = set(exclude.read_text(encoding="utf-8").split()) if exclude.is_file() else set()
    newest = max((p.stat().st_mtime for p in (review, set_path) if p.is_file()), default=None)
    if newest is not None and (not exclude.is_file() or exclude.stat().st_mtime < newest):
        return ids, "除外の一覧が review.md / set.jsonl より古いか、ありません（replay_set exclude を流してください）"
    return ids, None


# ─── コマンドライン ──────────────────────────────────────────────

def build(logs_dir: Path, out_dir: Path, *, since: date = DEFAULT_SINCE, seed: int = DEFAULT_SEED,
          context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS, force: bool = False,
          dry_run: bool = False) -> BuildReport:
    """セットを作って書く（dry_run なら数えるだけで書かない）。判定ログ（L2_DECISION_LOG）がオフであることを前提にする。

    Raises:
        RuntimeError: 判定ログがオン（当時の答えを解釈する呼び出しが、判定ログへ書き足してしまう）
        FileExistsError: review.md がすでにある（印を上書きで消さないため。force で上書き）
    """
    if decision_log.is_enabled():
        raise RuntimeError("判定ログがオンのままです")
    if (out_dir / "review.md").exists() and not force and not dry_run:
        raise FileExistsError("review.md がすでにあります（印を消さないよう止めました。--force で作り直し）")
    report = BuildReport()
    rows: list[dict[str, Any]] = []
    for path in select_files(logs_dir / "runs", since, report):
        rows.extend(rows_from_runlog(path, report, context_max_chars=context_max_chars))
    rows.extend(rows_from_decision_log(logs_dir / "decisions", since, report))
    usable = []
    for row in rows:
        try:
            row["hist_key"] = interpret(row, row["hist_raw"]) if row["hist_raw"] is not None else None
        except KeyError:
            # 今の登録に無いキャラ（名前の変わった候補など）は、本物の関数で呼べないので使わない。
            report.dropped["今の登録に無いキャラ"] += 1
            continue
        usable.append(row)
    chosen = select_rows(usable, report, seed=seed)
    if not dry_run:
        write_outputs(chosen, out_dir, report, seed=seed)
    return report


def _print_build(report: BuildReport) -> None:
    emit(f"実走ログ: {report.files_seen} 本")
    for rule, count in sorted(report.files_by_rule.items()):
        emit(f"  {rule}: {count}")
    for note in report.notes:
        emit(f"  注: {note}")
    emit(f"判定ログの行: {report.decision_rows}  塊: {report.epochs}")
    emit("見つけた判定: " + ", ".join(f"{k} {v}" for k, v in sorted(report.found.items())))
    emit("捨てた理由:")
    for reason, count in sorted(report.dropped.items()):
        emit(f"  {reason}: {count}")
    emit("セットに入れた行（当時の答えが何かを決めた行 / 何も決めなかった行）:")
    for name, target in TARGETS.items():
        kept = report.kept[name]
        share = report.undecided[name] / kept if kept else 0.0
        goal = f"、枠 {UNDECIDED_SHARE[name]:.0%}" if name in UNDECIDED_SHARE else ""
        emit(f"  {name}: {kept} / 目標 {target}（決めた {report.informative[name]} / 決めなかった {report.undecided[name]}"
             f" = {share:.0%}{goal}）")


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.replay_set <build|exclude>``。"""
    parser = argparse.ArgumentParser(prog="python -m lab_lounge.replay_set", description="判定のリプレイのセット")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="実走ログと判定ログからセットを作る（画面は件数だけ）")
    b.add_argument("--logs", type=Path, default=None, help="logs/ の場所（既定はリポジトリの logs/）")
    b.add_argument("--out", type=Path, default=None, help="出力の場所（既定は data/replay/）")
    b.add_argument("--since", type=date.fromisoformat, default=DEFAULT_SINCE, help="この日以降のログを使う")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--context-max-chars", type=int, default=DEFAULT_CONTEXT_MAX_CHARS)
    b.add_argument("--force", action="store_true", help="review.md があっても作り直す（付けた印は消える）")
    b.add_argument("--dry-run", action="store_true", help="数えるだけで、ファイルを書かない")
    e = sub.add_parser("exclude", help="review.md の [x] を exclude.txt へ移す（id だけ）")
    e.add_argument("--out", type=Path, default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    out_dir = args.out or default_out_dir()

    if args.command == "exclude":
        review = out_dir / "review.md"
        if not review.is_file():
            emit("review.md がありません（先に build を流してください）")
            return EXIT_FAILED
        result = read_review(review, {row["id"] for row in load_set(out_dir)})
        _write_atomic(out_dir / "exclude.txt", "".join(f"{i}\n" for i in result.excluded))
        emit(f"除外: {len(result.excluded)} 行")
        if result.unknown_ids:
            emit(f"セットに無い id の印: {result.unknown_ids} 行（無視しました）")
        if result.malformed_lines:
            emit(f"形の崩れた行（直してから流し直してください）: {', '.join(map(str, result.malformed_lines))} 行目")
            return EXIT_FAILED
        return EXIT_OK

    from .preflight import _captured_logs, silence_side_outputs

    silence_side_outputs()
    with _captured_logs():
        report = build(args.logs or default_logs_dir(), out_dir, since=args.since, seed=args.seed,
                       context_max_chars=args.context_max_chars, force=args.force, dry_run=args.dry_run)
    _print_build(report)
    if args.dry_run:
        emit("--dry-run: ファイルは書いていません")
    return EXIT_OK


if __name__ == "__main__":
    run_tool(main, name="replay_set")
