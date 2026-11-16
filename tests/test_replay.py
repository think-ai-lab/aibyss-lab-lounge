"""
replay.py の run / report のテスト（API は呼ばない）

LLM を呼ぶ 1 点（router._call_router_llm）を、入力ごとに答えを決める関数へ差し替え、本物の判定の関数
（check_approval / check_intent / _route_by_llm）を通して流す。本文には目印の文字列だけを使う。
本物の SDK と支出の記録を通す確かめは tests/contract/test_contract_replay.py にある。
"""

from __future__ import annotations

import json
import threading

import pytest

from lab_lounge import replay, replay_set, router, spend

_SENTINEL = "SENTINEL-replay-3b7c"


def _row(i: int, kind: str = "approval", *, hist: str | None = None, source: str = "runlog", **extra) -> dict:
    text = f"{_SENTINEL}-{kind}-{i}"
    base = {"kind": kind, "input": text, "character": "mimi" if kind == "intent" else None,
            "candidates": {"approval": ["chisame"], "route": ["mimi", "chisame"]}.get(kind, []),
            "source": source, "expected": None, "alt_ok": [], "hist_raw": hist, "hist_key": hist,
            "epoch": f"e#{i}", "date": "2026-06-01"}
    base.update(extra)
    base["id"] = extra.get("id") or replay_set.row_id(kind, base["character"], base["candidates"], text)
    return base


class Script:
    """入力ごとに答えを返す。呼ばれた回数も数える。"""

    def __init__(self, answers: dict[str, list]) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, model: str, system_prompt: str, user_text: str):
        with self.lock:
            self.calls.append(user_text)
            queue = self.answers[user_text]
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, tuple):  # (None, 例外の型名): _call_router_llm が失敗を握った形
            router._router_call_state.error = answer[1]
            return None
        return answer


@pytest.fixture
def run_file(tmp_path) -> replay.RunFile:
    return replay.RunFile(tmp_path / "runs" / "t.jsonl")


def _execute(rows, run_file, script, **kwargs):
    with replay_set.scripted_router(script):
        return replay.execute(rows, run_file, progress=lambda _line: None, **{"reps": 2, "workers": 4, **kwargs})


class TestExecute:
    def test_every_kind_runs_through_the_real_functions(self, run_file):
        rows = [_row(1, "approval"), _row(2, "intent"), _row(3, "interjection"), _row(4, "route")]
        answers = {rows[0]["input"]: ["denied:chisame"], rows[1]["input"]: ["mention"],
                   rows[2]["input"]: ["sakura"], rows[3]["input"]: ["chisame"]}
        stats = _execute(rows, run_file, Script(answers))
        assert stats == {"ok": 8}
        final = replay._final_by_key(run_file.lines())
        assert {final[(r["id"], 1)]["result_key"] for r in rows} == {"denied:chisame", "mention", "sakura", "chisame"}
        line = final[(rows[0]["id"], 2)]
        assert (line["calls"], line["error"], len(line["prompt_sha"]), line["model"]) == (1, None, 16, "claude-haiku-4-5")
        assert _SENTINEL not in run_file.path.read_text(encoding="utf-8")

    def test_rep_one_finishes_before_rep_two(self, run_file):
        rows = [_row(i) for i in range(6)]
        _execute(rows, run_file, Script({r["input"]: ["none"] for r in rows}))
        reps = [line["rep"] for line in run_file.lines()]
        assert reps == sorted(reps)

    def test_resume_skips_done_pairs(self, run_file):
        rows = [_row(i) for i in range(3)]
        script = Script({r["input"]: ["none"] for r in rows})
        _execute(rows, run_file, script, reps=1)
        assert len(script.calls) == 3
        _execute(rows, run_file, script, reps=2)
        assert len(script.calls) == 6
        assert len(run_file.lines()) == 6

    def test_failures_are_retried_once_at_the_end(self, run_file):
        rows = [_row(1), _row(2)]
        script = Script({rows[0]["input"]: [(None, "APITimeoutError"), "granted:chisame"],
                         rows[1]["input"]: [(None, "APITimeoutError")]})
        stats = _execute(rows, run_file, script, reps=1, workers=1)
        final = replay._final_by_key(run_file.lines())
        assert (final[(rows[0]["id"], 1)]["attempt"], final[(rows[0]["id"], 1)]["result_key"]) == (2, "granted:chisame")
        assert final[(rows[1]["id"], 1)]["result_key"] == "error:APITimeoutError"
        assert stats == {"error": 3, "ok": 1}
        # 2 回目の実行では、やり直し済みの行をもう一度流さない
        before = len(script.calls)
        _execute(rows, run_file, script, reps=1, workers=1)
        assert len(script.calls) == before

    def test_exceptions_outside_the_call_are_kept_as_type_only(self, run_file):
        row = _row(1)
        _execute([row], run_file, Script({row["input"]: [ValueError(_SENTINEL)]}), reps=1)
        (line, retry) = run_file.lines()
        assert line["result_key"] == "error:ValueError" and retry["attempt"] == 2
        assert _SENTINEL not in run_file.path.read_text(encoding="utf-8")

    def test_spend_stop_halts_the_run(self, run_file):
        rows = [_row(i) for i in range(20)]
        answers = {r["input"]: ["none"] for r in rows}
        answers[rows[0]["input"]] = [spend.BudgetExceeded("x")]
        with pytest.raises(spend.BudgetExceeded):
            _execute(rows, run_file, Script(answers), reps=1, workers=1)
        assert run_file.lines() == []

    def test_refuses_while_the_decision_log_is_on(self, run_file, monkeypatch):
        monkeypatch.setenv("L2_DECISION_LOG", "true")
        with pytest.raises(RuntimeError):
            _execute([_row(1)], run_file, Script({}))

    def test_recording_router_is_restored(self):
        original = router._call_router_llm
        with pytest.raises(ZeroDivisionError), replay.RecordingRouter():
            assert router._call_router_llm is not original
            raise ZeroDivisionError
        assert router._call_router_llm is original


class TestReport:
    def _lines(self, rows, keys_by_rep):
        return [{"id": r["id"], "rep": rep, "attempt": 1, "result_key": keys[i], "error": None}
                for rep, keys in keys_by_rep.items() for i, r in enumerate(rows)]

    def test_agreement_unknown_errors_and_history(self):
        rows = [_row(i, hist="none") for i in range(4)]
        lines = self._lines(rows, {1: ["none", "granted:chisame", "none", "error:X"],
                                   2: ["none", "denied:chisame", "none", "none"]})
        report = replay.make_report("t", rows, lines, 2)
        assert report.agreement["approval"] == (2, 3)
        assert report.errors["approval"] == 1
        assert report.historical["approval"] == (2, 3)
        assert report.confusion["approval"]["当時 none → granted:chisame"] == 1

    def test_synthetic_counts_alt_ok(self):
        rows = [_row(1, "interjection", source="synthetic", expected="sakura", alt_ok=["mimi"], id="s001"),
                _row(2, "interjection", source="synthetic", expected="sakura", id="s002"),
                _row(3, "intent", source="synthetic", expected="mention", id="s003")]
        lines = self._lines(rows, {1: ["mimi", "chisame", "unknown"]})
        report = replay.make_report("t", rows, lines, 1)
        assert report.synthetic["interjection"] == (1, 2)
        assert report.synthetic["mention"] == (0, 1)
        assert report.unknown["mention"] == 1

    def test_missing_pairs_are_counted(self):
        rows = [_row(1), _row(2)]
        report = replay.make_report("t", rows, self._lines(rows[:1], {1: ["none"]}), 2)
        assert report.missing == 3

    @pytest.mark.parametrize(("k", "n", "low", "high"), [(10, 10, 0.722, 1.0), (0, 10, 0.0, 0.278), (5, 10, 0.237, 0.763)])
    def test_wilson(self, k, n, low, high):
        got = replay.wilson(k, n)
        assert got == pytest.approx((low, high), abs=0.001)

    def test_print_has_no_text(self, capsys):
        rows = [_row(1, hist="none")]
        replay.print_report(replay.make_report("t", rows, self._lines(rows, {1: ["none"], 2: ["none"]}), 2))
        out = capsys.readouterr().out
        assert "approval: 1 / 1" in out and _SENTINEL not in out


class TestSetHandling:
    def _write_set(self, directory, rows):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")

    def test_exclusions_and_synthetic_are_applied(self, tmp_path):
        rows = [_row(i) for i in range(3)]
        self._write_set(tmp_path, rows)
        (tmp_path / "review.md").write_text("x", encoding="utf-8")
        (tmp_path / "exclude.txt").write_text(rows[0]["id"] + "\n", encoding="utf-8")
        synthetic = tmp_path / "synthetic.jsonl"
        synthetic.write_text(json.dumps({"id": "s001", "kind": "approval", "input": _SENTINEL, "character": None,
                                         "candidates": ["chisame"], "expected": "none", "alt_ok": [], "note": ""}) + "\n",
                             encoding="utf-8")
        loaded, excluded, warning = replay.load_rows(tmp_path, synthetic)
        assert excluded == 1 and warning is None
        assert {r["id"] for r in loaded} == {rows[1]["id"], rows[2]["id"], "s001"}

    def test_set_sha_changes_with_the_input(self):
        row = _row(1)
        assert replay.set_sha([row]) != replay.set_sha([dict(row, input=row["input"] + "x")])

    def test_canary_rotates_over_kinds(self):
        rows = [_row(i, "approval") for i in range(3)] + [_row(i, "interjection") for i in range(3)]
        picked = replay.canary_rows(rows, 3)
        assert [replay_set.bucket(r) for r in picked] == ["approval", "interjection", "approval"]
