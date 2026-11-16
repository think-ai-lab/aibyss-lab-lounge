"""
replay_set.py（実走ログからの組み直し、判定ログの混ぜ方、選び方、確認のファイル、合成の読み込み）のテスト

実走ログは、本物の logger と書式（log_setup._LOG_FORMAT）で書く。判定の行は本物の router の関数に
決めた答えを返させて出させる（ログの文言を複製しない）。BG の行と Agent の行は、replay_set が頼る
書式の定数で書き、その定数が本番のソースにそのまま在ることを番人のテストで確かめる。

本文には目印の文字列だけを使い、画面に出ないことを確かめる。API は呼ばない。
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from lab_lounge import log_setup, preflight, replay_set, router
from lab_lounge.characters import get_character
from lab_lounge.transcript_buffer import TranscriptBuffer, TranscriptSegment

_SENTINEL = "SENTINEL-replay-set-5d1e"
_SRC = Path(__file__).resolve().parents[1] / "src" / "lab_lounge"


def _name(slug: str) -> str:
    """発話に入れるキャラの呼び名（登録から取る。名前を複製しない）。"""
    config = get_character(slug)
    return config.nickname or config.display_name


class RunLog:
    """本物の logger と書式で、1 本の実走ログを書く。"""

    def __init__(self, path: Path, *, max_chars: int = 2000) -> None:
        self.path = path
        self.buffer = TranscriptBuffer(window_sec=1e9, max_chars=max_chars)
        self.handler = logging.FileHandler(path, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter(log_setup._LOG_FORMAT))
        self.root = logging.getLogger("lab_lounge")
        self.previous_level = self.root.level
        self.root.addHandler(self.handler)
        self.root.setLevel(logging.INFO)
        self.clock = 0.0
        logging.getLogger("lab_lounge.graph").info("Agent LLM step 1 完了 [character=mimi] latency_ms=1")

    def close(self) -> None:
        self.root.removeHandler(self.handler)
        self.root.setLevel(self.previous_level)
        self.handler.close()

    def bg(self, text: str, *, chars: int | None = None) -> None:
        self.clock += 1
        self.buffer.add(TranscriptSegment(text=text, timestamp=self.clock, duration_ms=0))
        logging.getLogger("lab_lounge.wake_word").info(
            replay_set.BG_BUFFER_FORMAT, text, len(self.buffer), self.buffer.total_chars if chars is None else chars)

    def judge(self, kind: str, answer: str, *, text: str | None = None, character: str | None = None,
              candidates: list[str] | None = None, monkeypatch=None) -> None:
        """本物の判定の関数を呼び、本物のログの行を出させる。"""
        full = self.buffer.full_text() if text is None else text
        with replay_set.scripted_router(lambda _m, _s, _t: answer):
            if kind == "approval":
                router.check_approval(full, candidates or ["chisame"])
            elif kind == "interjection":
                router.check_intent(full, None)
            elif kind == "intent":
                router.check_intent(self.buffer.extract_context() if text is None else text, character)
            elif kind == "route":
                monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")
                router.route(full)

    def callout(self, slug: str, *, context: str | None = None, wake_len: int | None = None) -> None:
        context = self.buffer.extract_context() if context is None else context
        logging.getLogger("lab_lounge.wake_word").info(
            replay_set.BG_WAKE_FORMAT, slug, len(context) if wake_len is None else wake_len)
        tagged = replay_set.CALLOUT_PREFIX + context
        logging.getLogger("lab_lounge.graph").info(replay_set.AGENT_INVOKE_FORMAT, slug, len(tagged), tagged)

    def raw(self, line: str) -> None:
        self.handler.stream.write(line + "\n")
        self.handler.flush()


@pytest.fixture
def logs(tmp_path):
    (tmp_path / "logs" / "runs").mkdir(parents=True)
    return tmp_path / "logs"


@pytest.fixture
def runlog(logs):
    opened: list[RunLog] = []

    def make(name: str = "run_loop_20260601_200000.log", **kwargs) -> RunLog:
        log = RunLog(logs / "runs" / name, **kwargs)
        opened.append(log)
        return log

    yield make
    for log in opened:
        log.close()


def _rows(logs: Path, **kwargs) -> tuple[list[dict], replay_set.BuildReport]:
    report = replay_set.BuildReport()
    rows = []
    for path in replay_set.select_files(logs / "runs", kwargs.pop("since", date(2026, 5, 8)), report):
        rows.extend(replay_set.rows_from_runlog(path, report, context_max_chars=2000))
    for row in rows:
        row["hist_key"] = replay_set.interpret(row, row["hist_raw"]) if row["hist_raw"] is not None else None
    return rows, report


# ─── 番人 ────────────────────────────────────────────────────────


class TestFormatsStillExistInTheSource:
    """組み直しが頼るログの文言が、本番のソースから消えたら赤くする。"""

    def test_log_line_format(self):
        assert log_setup._LOG_FORMAT == "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    @pytest.mark.parametrize(("module", "text"), [
        ("wake_word.py", f'"{replay_set.BG_BUFFER_FORMAT}"'),
        ("wake_word.py", f'"{replay_set.BG_WAKE_FORMAT}"'),
        ("graph.py", f'"{replay_set.AGENT_INVOKE_FORMAT}"'),
        ("graph.py", 'tagged_text = f"[ルカからの発言]\\n{text}"'),
        ("router.py", f'"{replay_set.ROUTE_DELEGATE_PREFIX}%s → LLM ルーターへ委譲 (text=%s)"'),
        ("router.py", '"interjection_candidate 応答: %r (model=%s candidates=%s text=%s)"'),
        ("router.py", '"check_approval 応答: %r (model=%s candidates=%s text=%s)"'),
        ("router.py", '"意図ゲート応答: %r (model=%s char=%s text=%s)"'),
        ("router.py", '"LLM ルーター応答: %r (model=%s text=%s)"'),
    ])
    def test_source_has_the_format(self, module, text):
        assert text in (_SRC / module).read_text(encoding="utf-8")

    def test_wake_logs_the_buffer_before_judging(self):
        """BG の行の直後に、同じバッファの全文で判定する順番（組み直しの前提）。"""
        source = (_SRC / "wake_word.py").read_text(encoding="utf-8")
        start = source.index(f'"{replay_set.BG_BUFFER_FORMAT}"')
        assert source.index("self._buffer.add(segment)") < start
        assert start < source.index("self._on_segment_added(segment, self._buffer.full_text())", start)

    def test_judgment_prefixes_are_known_kinds(self):
        assert set(replay_set.JUDGMENT_PREFIXES.values()) == set(replay_set.KINDS)


# ─── 組み直し ────────────────────────────────────────────────────


class TestRebuild:
    def test_each_kind_is_rebuilt_in_full(self, runlog, logs, monkeypatch):
        log = runlog()
        first, second = f"{_SENTINEL} 一つめ" + "あ" * 40, f"{_name('chisame')}、{_SENTINEL} 二つめ"
        log.bg(first)
        log.bg(second)
        log.judge("approval", "granted:chisame", candidates=["chisame"])
        log.judge("interjection", "sakura")
        log.judge("intent", "mention", character="chisame")
        log.callout("chisame")
        rows, report = _rows(logs)

        full = f"{first}\n{second}"
        by_kind = {(r["kind"], r["hist_raw"]): r for r in rows}
        approval = by_kind[("approval", "granted:chisame")]
        assert (approval["input"], approval["candidates"], approval["hist_key"]) == (full, ["chisame"], "granted:chisame")
        assert by_kind[("interjection", "sakura")]["hist_key"] == "sakura"
        mention = by_kind[("intent", "mention")]
        assert (mention["input"], mention["character"], replay_set.bucket(mention)) == (full, "chisame", "mention")
        callout = by_kind[("intent", None)]
        assert (callout["input"], callout["character"], replay_set.bucket(callout)) == (full, "chisame", "callout")
        assert not report.dropped

    def test_route_takes_candidates_from_the_delegation_line(self, runlog, logs, monkeypatch):
        log = runlog()
        log.bg(f"{_name('mimi')}と{_name('chisame')}、{_SENTINEL}")
        log.judge("route", "chisame", monkeypatch=monkeypatch)
        rows, _ = _rows(logs)
        (row,) = [r for r in rows if r["kind"] == "route"]
        assert set(row["candidates"]) == {"mimi", "chisame"}
        assert row["hist_key"] == "chisame"

    def test_eviction_from_the_head_is_followed(self, runlog, logs):
        log = runlog(max_chars=50)
        for i in range(5):
            log.bg(f"{_SENTINEL}-{i}-" + "い" * 10)
        log.judge("approval", "none")
        (row,) = [r for r in _rows(logs)[0] if r["kind"] == "approval"]
        assert row["input"] == log.buffer.full_text()
        assert row["input"].count("\n") < 4

    def test_clear_starts_over(self, runlog, logs):
        log = runlog()
        log.bg(f"{_SENTINEL} 消える")
        log.buffer.clear()
        log.bg(f"{_SENTINEL} 残る")
        log.judge("approval", "denied:chisame")
        (row,) = [r for r in _rows(logs)[0] if r["kind"] == "approval"]
        assert row["input"] == f"{_SENTINEL} 残る"

    def test_chars_mismatch_is_dropped(self, runlog, logs):
        log = runlog()
        log.bg(f"{_SENTINEL} 一", chars=999)
        log.judge("approval", "none")
        rows, report = _rows(logs)
        assert rows == []
        assert report.dropped["文字数が合わない"] == 1

    def test_preview_mismatch_is_dropped(self, runlog, logs):
        log = runlog()
        log.bg(f"{_SENTINEL} 一")
        log.judge("approval", "none", text="別の発話")
        rows, report = _rows(logs)
        assert rows == []
        assert report.dropped["プレビューが合わない"] == 1

    def test_judgment_without_a_buffer_is_dropped(self, runlog, logs):
        log = runlog()
        log.judge("approval", "none", text=_SENTINEL)
        assert _rows(logs)[1].dropped["直前の BG の行が無い"] == 1

    def test_agent_not_following_a_wake_is_not_a_callout(self, runlog, logs):
        log = runlog()
        log.bg(f"{_SENTINEL} 一")
        log.callout("chisame", wake_len=3)
        rows, report = _rows(logs)
        assert rows == []
        assert report.dropped["Agent の行が wake の発話でない"] == 1

    def test_unreadable_bg_line_stops_the_rest_of_the_file(self, runlog, logs):
        log = runlog()
        log.bg(f"{_SENTINEL} 一")
        log.raw(f"2026-06-01 20:00:00,000 INFO     lab_lounge.wake_word: BG buffer 蓄積: '{_SENTINEL} (segments=2 chars=9)")
        log.judge("approval", "none", text=f"{_SENTINEL} 一")
        rows, report = _rows(logs)
        assert rows == []
        assert report.dropped["BG の行を読めない（以後のファイルの残りを使わない）"] == 1

    def test_answers_with_brackets_and_markers_are_parsed(self):
        rest = "'none (model=x) )' (model=claude-haiku-4-5 candidates=['chisame'] text='a)b'...)"
        parsed = replay_set._parse_judgment("approval", rest)
        assert (parsed.answer, parsed.model, parsed.candidates, parsed.preview) == (
            "none (model=x) )", "claude-haiku-4-5", ["chisame"], "'a)b'...")

    def test_continuation_lines_are_skipped(self, runlog, logs):
        log = runlog()
        log.raw("2026-06-01 20:00:00,000 INFO     lab_lounge.stt: STT 認識結果: 一行め")
        log.raw("BG buffer 蓄積: 'x' (segments=1 chars=1)")
        log.bg(f"{_SENTINEL}")
        log.judge("approval", "none")
        assert len(_rows(logs)[0]) == 1


class TestFiles:
    @pytest.mark.parametrize(("name", "used"), [
        ("run_loop_20260507_235959.log", False),
        ("run_loop_20260508_000000.log", True),
        ("run_loop_20261031_120000.log", False),
        ("run_loop_20261101_025405.log", False),
        ("run_loop_20261101_025406.log", True),
        ("run_loop_x.log", False),
    ])
    def test_name_rules(self, runlog, logs, name, used):
        runlog(name)
        report = replay_set.BuildReport()
        assert bool(replay_set.select_files(logs / "runs", date(2026, 5, 8), report)) is used
        assert report.files_seen == 1

    def test_nested_month_folder_is_read(self, logs):
        (logs / "runs" / "202605").mkdir()
        path = logs / "runs" / "202605" / "run_loop_20260509_000000.log"
        path.write_text("Agent LLM step 1 完了\n", encoding="utf-8")
        assert replay_set.select_files(logs / "runs", date(2026, 5, 8), replay_set.BuildReport()) == [path]

    @pytest.mark.parametrize("text", ["応答なし\n", "Agent LLM step 1 完了\nboom\n", "Agent LLM step 1 完了 unknown_char\n"])
    def test_content_rules(self, logs, text):
        (logs / "runs" / "run_loop_20260601_000000.log").write_text(text, encoding="utf-8")
        report = replay_set.BuildReport()
        assert replay_set.select_files(logs / "runs", date(2026, 5, 8), report) == []
        assert report.files_by_rule["中身の規則（pytest の残り、または応答の無い回）"] == 1


# ─── 判定ログと選び方 ────────────────────────────────────────────


def _decision(kind: str, text: str, raw: str | None = "none", **extra) -> str:
    row = {"ts": "2026-11-20T21:00:00.000+09:00", "kind": kind, "model": "claude-haiku-4-5", "character": None,
           "candidates": ["chisame"], "input": text, "raw": raw, "result": None, "elapsed_ms": 1, "error": None,
           "system_prompt_sha": "0" * 16, **extra}
    return json.dumps(row, ensure_ascii=False)


class TestDecisionLog:
    def test_rows_are_mixed_and_failures_dropped(self, logs):
        (logs / "decisions").mkdir()
        (logs / "decisions" / "decisions_20261120.jsonl").write_text("\n".join([
            _decision("approval", f"{_SENTINEL} 一", "granted:chisame"),
            _decision("approval", f"{_SENTINEL} 二", None),
            _decision("approval", f"{_SENTINEL} 三", "none", error="APITimeoutError"),
            _decision("intent", f"{_SENTINEL} 四", "callout", character="mimi"),
            "{壊れた行",
        ]) + "\n", encoding="utf-8")
        report = replay_set.BuildReport()
        rows = replay_set.rows_from_decision_log(logs / "decisions", date(2026, 5, 8), report)
        assert [(r["kind"], r["hist_raw"], r["source"]) for r in rows] == [
            ("approval", "granted:chisame", "decision_log"), ("intent", "callout", "decision_log")]
        assert rows[1]["character"] == "mimi"
        assert report.dropped["判定ログの呼び出しが失敗した行"] == 2
        assert report.dropped["判定ログの行を読めない"] == 1

    def test_same_input_as_the_runlog_has_the_same_id(self):
        a = replay_set.row_id("approval", None, ["chisame"], "x")
        assert a == replay_set.row_id("approval", None, ["chisame"], "x")
        assert a != replay_set.row_id("approval", None, ["mimi"], "x")


class TestSelect:
    def _row(self, i: int, *, epoch: str, hist: str = "none", kind: str = "approval") -> dict:
        text = f"{_SENTINEL}-{i}"
        return {"id": replay_set.row_id(kind, None, ["chisame"], text), "kind": kind, "input": text,
                "character": None, "candidates": ["chisame"], "hist_raw": hist, "hist_key": hist,
                "epoch": epoch, "date": "2026-06-01"}

    def test_epoch_cap_prefers_informative_answers(self):
        rows = [self._row(i, epoch="a#1") for i in range(5)] + [self._row(9, epoch="a#1", hist="denied:chisame")]
        report = replay_set.BuildReport()
        chosen = replay_set.select_rows(rows, report, seed=1)
        assert len(chosen) == replay_set.PER_EPOCH
        assert "denied:chisame" in {r["hist_key"] for r in chosen}
        assert report.dropped["同じ塊の上限を超えた"] == 4

    def test_targets_and_determinism(self, monkeypatch):
        monkeypatch.setitem(replay_set.TARGETS, "approval", 3)
        rows = [self._row(i, epoch=f"e#{i}") for i in range(10)]
        first = replay_set.select_rows([dict(r) for r in rows], replay_set.BuildReport(), seed=7)
        second = replay_set.select_rows([dict(r) for r in rows], replay_set.BuildReport(), seed=7)
        assert len(first) == 3
        assert [r["id"] for r in first] == [r["id"] for r in second]

    def test_duplicates_are_merged(self):
        row = self._row(1, epoch="a#1")
        report = replay_set.BuildReport()
        assert len(replay_set.select_rows([row, dict(row, epoch="b#1")], report, seed=1)) == 1
        assert report.dropped["同じ入力の重複"] == 1


# ─── 確認のファイルと除外 ────────────────────────────────────────


def _silence(monkeypatch):
    """main が触る環境変数を monkeypatch に覚えさせ、テストの後に戻す。"""
    for name in ("L2_DEBUG_ARTIFACTS", "L2_DECISION_LOG", *preflight._TRACING_VARS):
        monkeypatch.setenv(name, "false")


class TestCommandLine:
    def _build(self, runlog, logs, out, monkeypatch, capsys):
        log = runlog()
        log.bg(f"{_SENTINEL} 一行め\nと改行\tとタブ")
        log.judge("approval", "granted:chisame")
        log.bg(f"{_name('mimi')}、{_SENTINEL} 二行め")
        log.judge("interjection", "mimi")
        _silence(monkeypatch)
        code = replay_set.main(["build", "--logs", str(logs), "--out", str(out)])
        return code, capsys.readouterr()

    def test_build_writes_files_and_prints_counts_only(self, runlog, logs, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        code, captured = self._build(runlog, logs, out, monkeypatch, capsys)
        assert code == 0
        assert _SENTINEL not in captured.out + captured.err
        assert "approval: 1 / 目標 100" in captured.out
        rows = replay_set.load_set(out)
        assert {r["kind"] for r in rows} == {"approval", "interjection"}
        review = (out / "review.md").read_text(encoding="utf-8")
        line = next(x for x in review.splitlines() if " approval " in x)
        assert line.startswith("- [ ] ") and "⏎" in line and "\t" not in line
        assert json.loads((out / "set.meta.json").read_text(encoding="utf-8"))["kept"]["approval"] == 1

    def test_dry_run_writes_nothing(self, runlog, logs, tmp_path, monkeypatch, capsys):
        log = runlog()
        log.bg(_SENTINEL)
        log.judge("approval", "none")
        _silence(monkeypatch)
        assert replay_set.main(["build", "--logs", str(logs), "--out", str(tmp_path / "out"), "--dry-run"]) == 0
        assert not (tmp_path / "out").exists()
        assert _SENTINEL not in capsys.readouterr().out

    def test_build_refuses_to_overwrite_marks(self, runlog, logs, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        self._build(runlog, logs, out, monkeypatch, capsys)
        with pytest.raises(FileExistsError):
            replay_set.main(["build", "--logs", str(logs), "--out", str(out)])

    def test_build_refuses_while_the_decision_log_is_on(self, logs, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_DECISION_LOG", "true")
        with pytest.raises(RuntimeError):
            replay_set.build(logs, tmp_path / "out")

    def test_exclude_round_trip(self, runlog, logs, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        self._build(runlog, logs, out, monkeypatch, capsys)
        ids, warning = replay_set.load_exclusions(out)
        assert ids == set() and warning is not None

        review = out / "review.md"
        lines = review.read_text(encoding="utf-8").splitlines()
        target = next(i for i, x in enumerate(lines) if " approval " in x)
        excluded_id = lines[target].split()[3]
        lines[target] = lines[target].replace("- [ ]", "- [x]", 1)
        review.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert replay_set.main(["exclude", "--out", str(out)]) == 0
        captured = capsys.readouterr()
        assert "除外: 1 行" in captured.out and _SENTINEL not in captured.out
        assert (out / "exclude.txt").read_text(encoding="utf-8") == f"{excluded_id}\n"
        assert replay_set.load_exclusions(out) == ({excluded_id}, None)

    def test_exclude_reports_malformed_lines_by_number(self, runlog, logs, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        self._build(runlog, logs, out, monkeypatch, capsys)
        review = out / "review.md"
        lines = review.read_text(encoding="utf-8").splitlines()
        lines.append(f"- [?] 0123456789ab approval chisame | {_SENTINEL}")
        lines.append(f"- [x] {_SENTINEL} approval chisame | x")
        lines.append("- [x] ffffffffffff approval chisame | セットに無い")
        review.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert replay_set.main(["exclude", "--out", str(out)]) == 1
        captured = capsys.readouterr()
        assert f"{len(lines) - 2}, {len(lines) - 1} 行目" in captured.out
        assert "セットに無い id の印: 1 行" in captured.out
        assert _SENTINEL not in captured.out + captured.err


# ─── 合成の行 ────────────────────────────────────────────────────


class TestSynthetic:
    def _write(self, tmp_path, *rows) -> Path:
        path = tmp_path / "replay_synthetic.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return path

    def _base(self, **extra) -> dict:
        return {"id": "s001", "kind": "approval", "input": _SENTINEL, "character": None, "candidates": ["chisame"],
                "expected": "denied:chisame", "alt_ok": [], "note": "合成", **extra}

    def test_rows_are_loaded_in_the_set_shape(self, tmp_path):
        path = self._write(tmp_path, self._base(), self._base(
            id="s045", kind="interjection", candidates=[], expected="sakura", alt_ok=["mimi"]),
            self._base(id="s017", kind="intent", character="chisame", candidates=[], expected="mention"))
        rows = replay_set.load_synthetic(path)
        assert [(r["id"], r["source"], r["expected"]) for r in rows] == [
            ("s001", "synthetic", "denied:chisame"), ("s045", "synthetic", "sakura"), ("s017", "synthetic", "mention")]
        assert rows[1]["alt_ok"] == ["mimi"]
        assert replay_set.bucket(rows[2]) == "mention"

    def test_missing_file_is_empty(self, tmp_path):
        assert replay_set.load_synthetic(tmp_path / "none.jsonl") == []

    @pytest.mark.parametrize("bad", [
        {"expected": "granted"}, {"kind": "nope"}, {"id": "x1"}, {"input": ""}, {"candidates": []},
        {"kind": "intent", "character": None, "expected": "mention"}, {"alt_ok": [_SENTINEL]},
    ])
    def test_bad_rows_are_reported_by_line_number_only(self, tmp_path, bad):
        path = self._write(tmp_path, self._base(id="s002"), self._base(**bad))
        with pytest.raises(ValueError) as caught:
            replay_set.load_synthetic(path)
        assert "2 行目" in str(caught.value)
        assert _SENTINEL not in str(caught.value)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            replay_set.load_synthetic(self._write(tmp_path, self._base(), self._base()))

    def test_committed_rows_load(self):
        """commit した合成の行（measure_sets/）が、読み込みの形に合っている。"""
        if not replay_set.default_synthetic_path().is_file():
            pytest.skip("合成の行がまだ無い")
        assert len(replay_set.load_synthetic()) > 0


class TestJudge:
    @pytest.mark.parametrize(("row", "raw", "key"), [
        ({"kind": "approval", "input": "x", "candidates": ["chisame"]}, "granted:chisame", "granted:chisame"),
        ({"kind": "approval", "input": "x", "candidates": ["chisame"]}, "denied:chisame", "denied:chisame"),
        ({"kind": "approval", "input": "x", "candidates": ["chisame"]}, "none", "none"),
        ({"kind": "approval", "input": "x", "candidates": ["chisame"]}, None, "none"),
        ({"kind": "intent", "input": "x", "character": "mimi"}, "mention", "mention"),
        ({"kind": "intent", "input": "x", "character": "mimi"}, "zzz", "unknown"),
        ({"kind": "interjection", "input": "x"}, "sakura", "sakura"),
        ({"kind": "interjection", "input": "x"}, "none", "unknown"),
        ({"kind": "route", "input": "x", "candidates": ["mimi", "chisame"]}, "chisame", "chisame"),
        ({"kind": "route", "input": "x", "candidates": ["mimi", "chisame"]}, "none", "none"),
        ({"kind": "route", "input": "x", "candidates": ["mimi", "chisame"]}, "sakura", "unknown"),
    ])
    def test_raw_answers_are_read_by_the_real_functions(self, row, raw, key):
        assert replay_set.interpret(row, raw) == key

    def test_scripted_router_is_restored(self):
        original = router._call_router_llm
        with pytest.raises(ZeroDivisionError), replay_set.scripted_router(lambda *_: "none"):
            raise ZeroDivisionError
        assert router._call_router_llm is original
