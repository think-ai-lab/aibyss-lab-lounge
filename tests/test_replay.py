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
        (directory / "private").mkdir(parents=True, exist_ok=True)
        (directory / "private" / "set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")

    def test_exclusions_and_synthetic_are_applied(self, tmp_path):
        rows = [_row(i) for i in range(3)]
        self._write_set(tmp_path, rows)
        (tmp_path / "private" / "review.md").write_text("x", encoding="utf-8")
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


# ─── compare / review / score ────────────────────────────────────


def _lines_for(rows, keys_by_rep):
    return [{"id": r["id"], "rep": rep, "attempt": 1, "result_key": keys[i], "error": None}
            for rep, keys in keys_by_rep.items() for i, r in enumerate(rows) if keys[i] is not None]


class TestCompare:
    def test_flips_unstable_rows_and_transitions(self):
        rows = [_row(1), _row(2), _row(3), _row(4, "intent"), _row(5, "intent"), _row(6)]
        base = _lines_for(rows, {1: ["granted:chisame", "granted:chisame", "none", "callout", "unknown", "error:X"],
                                 2: ["granted:chisame", "denied:chisame", "none", "callout", "unknown", "none"]})
        new = _lines_for(rows, {1: ["denied:chisame", "denied:chisame", "none", "mention", "callout", "none"]})
        result = replay.compare_runs(rows, base, new, base_reps=2)
        assert (result.compared, result.flips, result.flips_unstable, result.unstable) == (5, 1, 1, 1)
        assert (result.unknown_base, result.unknown_run, result.errors_base) == (1, 0, 1)
        assert result.transitions["callout: callout → mention"] == 1
        assert {(d.base, d.new, d.unstable) for d in result.diffs} == {
            ("granted:chisame", "denied:chisame", False), ("granted:chisame", "denied:chisame", True),
            ("callout", "mention", False), ("unknown", "callout", False)}

    def test_single_rep_base_has_no_unstable_rows(self):
        rows = [_row(1)]
        result = replay.compare_runs(rows, _lines_for(rows, {1: ["granted:chisame"]}),
                                     _lines_for(rows, {1: ["denied:chisame"]}), base_reps=1)
        assert (result.flips, result.unstable) == (1, 0)

    def test_missing_results_are_counted(self):
        rows = [_row(1), _row(2)]
        result = replay.compare_runs(rows, _lines_for(rows, {1: ["none", "none"]}),
                                     _lines_for(rows, {1: ["none", None]}), base_reps=1)
        assert (result.compared, result.missing) == (1, 1)


@pytest.fixture
def pair(tmp_path):
    """基準の回（2 回）と新しい回（1 回）を、セットと一緒に置く。"""
    rows = [_row(1), _row(2), _row(3, "intent"), _row(4, "interjection"), _row(5)]
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                        encoding="utf-8")
    (tmp_path / "exclude.txt").write_text("", encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()

    def put(label, reps, keys_by_rep):
        (runs / f"{label}.meta.json").write_text(json.dumps({"reps": reps, "set_sha": "x", "rows": len(rows)}),
                                                 encoding="utf-8")
        (runs / f"{label}.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in _lines_for(rows, keys_by_rep)), encoding="utf-8")

    put("base", 2, {1: ["none", "granted:chisame", "callout", "mimi", "none"],
                    2: ["none", "granted:chisame", "callout", "mimi", "none"]})
    put("new", 1, {1: ["none", "granted:chisame", "mention", "sakura", "granted:chisame"]})
    put("flip", 1, {1: ["none", "denied:chisame", "callout", "mimi", "none"]})
    return tmp_path, rows


def _cli(set_dir, *args):
    return replay.main(["--set-dir", str(set_dir), "--synthetic", str(set_dir / "none.jsonl"), *args])


def _mark(path, marks: dict[str, str]):
    """確認のファイルの行に印を付ける（id → 印）。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        for row_id, mark in marks.items():
            if line.startswith(f"- [ ] {row_id} "):
                line = line.replace("- [ ]", f"- [{mark}]", 1)
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


class TestCompareCommands:
    def test_compare_exit_code_follows_flips(self, pair, capsys):
        set_dir, _ = pair
        assert _cli(set_dir, "compare", "--base", "base", "--run", "new") == 0
        assert _cli(set_dir, "compare", "--base", "base", "--run", "flip") == 1
        out = capsys.readouterr().out
        assert "承認の反転（granted ⇔ denied、揺れない行）: 1" in out and _SENTINEL not in out

    def test_review_score_round_trip(self, pair, capsys):
        set_dir, rows = pair
        assert _cli(set_dir, "review", "--base", "base", "--run", "new") == 0
        review = set_dir / "private" / "review_new_vs_base.md"
        text = review.read_text(encoding="utf-8")
        assert text.count("\n- [ ] ") == 3 and "callout → mention" in text
        assert _cli(set_dir, "review", "--base", "base", "--run", "new") == replay.EXIT_REFUSED

        assert _cli(set_dir, "score", "--base", "base", "--run", "new") == replay.EXIT_UNDECIDED
        _mark(review, {rows[2]["id"]: "x", rows[3]["id"]: "a", rows[4]["id"]: "o"})
        assert _cli(set_dir, "score", "--base", "base", "--run", "new") == 0
        out = capsys.readouterr().out
        assert "正しい 1、誤り 1、別の候補も妥当 1、未判定 0" in out
        assert "PASS (b) callout → mention の誤り: 1 / 上限 1" in out
        assert _SENTINEL not in out

        # 付けた印は覚えていて、作り直した確認のファイルには出さない
        review.unlink()
        assert _cli(set_dir, "review", "--base", "base", "--run", "new") == 0
        assert "3 行は飛ばしました" in capsys.readouterr().out
        assert _cli(set_dir, "score", "--base", "base", "--run", "new") == 0

    def test_score_fails_on_too_many_wrong(self, pair, capsys, monkeypatch):
        set_dir, rows = pair
        monkeypatch.setattr(replay, "MAX_WRONG", 0)
        _cli(set_dir, "review", "--base", "base", "--run", "new")
        _mark(set_dir / "private" / "review_new_vs_base.md", {rows[2]["id"]: "o", rows[3]["id"]: "o", rows[4]["id"]: "x"})
        assert _cli(set_dir, "score", "--base", "base", "--run", "new") == 1
        assert "FAIL (b) 誤りとした行" in capsys.readouterr().out

    def test_score_reports_bad_marks_by_line_number(self, pair, capsys):
        set_dir, rows = pair
        _cli(set_dir, "review", "--base", "base", "--run", "new")
        review = set_dir / "private" / "review_new_vs_base.md"
        _mark(review, {rows[4]["id"]: "a"})  # [a] は挙手候補だけ
        with review.open("a", encoding="utf-8") as stream:
            stream.write(f"- [?] {rows[2]['id']} intent callout → mention | {_SENTINEL}\n")
            stream.write(f"- [o] {rows[3]['id']} interjection mimi → chisame | {_SENTINEL}\n")  # 今の違いと合わない
        assert _cli(set_dir, "score", "--base", "base", "--run", "new") == 1
        out = capsys.readouterr().out
        total = len(review.read_text(encoding="utf-8").splitlines())
        bad_a = next(i for i, line in enumerate(review.read_text(encoding="utf-8").splitlines(), start=1)
                     if line.startswith("- [a]"))
        assert f"{bad_a}, {total - 1}, {total} 行目" in out
        assert _SENTINEL not in out
        assert not (set_dir / "judged.jsonl").exists()

    def test_missing_meta(self, pair, capsys):
        set_dir, _ = pair
        assert _cli(set_dir, "compare", "--base", "base", "--run", "nope") == 1


@pytest.mark.parametrize("extra", [["--max-usd", "0"], ["--max-usd", "nan"], ["--max-usd", "1", "--price", "gpt-9=a,b"],
                                   ["--max-usd", "1", "--reps", "0"]])
def test_run_rejects_bad_arguments_before_anything(tmp_path, extra, capsys):
    code = replay.main(["--set-dir", str(tmp_path), "run", "--profile", "dev", "--label", "x", *extra])
    assert code == replay.EXIT_USAGE
    assert not (tmp_path / "runs").exists()


class TestPrivateAnswers:
    def test_only_canonical_answers_stay_in_the_numbers_file(self, tmp_path):
        rows = [_row(1), _row(2), _row(3)]
        answers = {rows[0]["input"]: ["granted:chisame"], rows[1]["input"]: [f"none、{_SENTINEL}って言ってたので"],
                   rows[2]["input"]: [(None, "APITimeoutError")]}
        run_file = replay.RunFile(tmp_path / "runs" / "t.jsonl")
        private = replay.RunFile(tmp_path / "private" / "runs" / "t.answers.jsonl")
        with replay_set.scripted_router(Script(answers)):
            replay.execute(rows, run_file, reps=1, workers=1, answers=private, progress=lambda _l: None)
        lines = {line["id"]: line for line in run_file.lines()}
        assert lines[rows[0]["id"]]["raw"] == "granted:chisame"
        quoted = lines[rows[1]["id"]]
        assert quoted["raw"] is None and quoted["raw_len"] > 0 and len(quoted["raw_sha"]) == 12
        assert _SENTINEL not in run_file.path.read_text(encoding="utf-8")
        assert _SENTINEL in private.path.read_text(encoding="utf-8")
        assert {a["id"] for a in private.lines()} == {rows[0]["id"], rows[1]["id"]}

    def test_canonical_answers_cover_every_slug(self):
        from lab_lounge.characters import get_all_characters

        canonical = replay.canonical_answers()
        for config in get_all_characters():
            assert {config.slug, f"granted:{config.slug}", f"denied:{config.slug}"} <= canonical


class TestPrerequisites:
    def _put(self, set_dir, label, reps, keys_by_rep, rows):
        runs = set_dir / "runs"
        (runs / f"{label}.meta.json").write_text(json.dumps({"reps": reps, "set_sha": "x", "rows": len(rows)}),
                                                 encoding="utf-8")
        (runs / f"{label}.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in _lines_for(rows, keys_by_rep)), encoding="utf-8")

    def test_incomplete_run_fails_compare_and_score(self, pair, capsys):
        set_dir, rows = pair
        self._put(set_dir, "partial", 1, {1: ["none", "granted:chisame", "callout", "mimi", None]}, rows)
        assert _cli(set_dir, "compare", "--base", "base", "--run", "partial") == 1
        assert "FAIL (前提) 全部の行に結果がある" in capsys.readouterr().out
        assert _cli(set_dir, "score", "--base", "base", "--run", "partial") == 1

    def test_error_heavy_run_fails(self, pair, capsys):
        set_dir, rows = pair
        self._put(set_dir, "broken", 1, {1: ["error:APITimeoutError"] * 4 + ["none"]}, rows)
        assert _cli(set_dir, "compare", "--base", "base", "--run", "broken") == 1
        out = capsys.readouterr().out
        assert "FAIL (前提) エラーの行: 4 / 上限 3" in out
        # 失敗だらけでも違う行は 0 になる。前提が無いと合格に見えていた
        assert "違う行: 0" in out

    def test_flips_that_do_not_repeat_in_rep_two_are_shown(self, pair, capsys):
        set_dir, rows = pair
        self._put(set_dir, "twice", 2, {1: ["none", "denied:chisame", "callout", "mimi", "none"],
                                        2: ["none", "granted:chisame", "callout", "mimi", "none"]}, rows)
        assert _cli(set_dir, "compare", "--base", "base", "--run", "twice") == 1
        assert "基準と同じに戻った反転 1" in capsys.readouterr().out

    def test_run_errors_limit(self):
        lines = [{"id": f"r{i}", "rep": 1, "attempt": 2, "error": "X" if i < 4 else None} for i in range(10)]
        assert replay.run_errors(lines, 10) == (4, 3)
        assert replay.run_errors(lines, 1000) == (4, 10)


@pytest.mark.parametrize("args", [
    ["run", "--profile", "dev", "--label", "../escape", "--max-usd", "1"],
    ["report", "--run", r"..\x"],
    ["compare", "--base", "a b", "--run", "x"],
    ["score", "--base", "x", "--run", "x" * 65],
])
def test_run_names_are_checked_before_use_as_paths(tmp_path, args):
    assert replay.main(["--set-dir", str(tmp_path), *args]) == replay.EXIT_USAGE
    assert list(tmp_path.iterdir()) == []


def test_compare_names_kinds_whose_prompt_changed(pair, capsys):
    set_dir, _ = pair
    for label, sha in (("base", "aaa"), ("new", "bbb")):
        path = set_dir / "runs" / f"{label}.meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["prompt_shas"] = {"approval": [sha], "intent": ["same"]}
        path.write_text(json.dumps(meta), encoding="utf-8")
    _cli(set_dir, "compare", "--base", "base", "--run", "new")
    assert "system prompt が基準と違う種類: approval" in capsys.readouterr().out


def test_judge_code_sha_is_stable():
    assert replay.judge_code_sha() == replay.judge_code_sha()
    assert len(replay.judge_code_sha()) == 16


# ─── system prompt のまとまり（種類 × キャラ × 候補の並び） ───────────────────

def _grp_row(kind, character=None, candidates=(), text="x"):
    return {"id": f"{kind}-{character}-{','.join(candidates)}-{text}", "kind": kind, "character": character,
            "candidates": list(candidates)}


def _grp_line(row, sha):
    return {"id": row["id"], "prompt_sha": sha}


class TestPromptGroups:
    def test_the_key_is_kind_character_and_candidate_order(self):
        assert replay.prompt_group(_grp_row("intent", "mimi")) == "intent|mimi|"
        assert replay.prompt_group(_grp_row("approval", None, ["chisame", "mimi"])) == "approval||chisame,mimi"
        assert replay.prompt_group(_grp_row("approval", None, ["mimi", "chisame"])) == "approval||mimi,chisame"
        assert replay.prompt_group(_grp_row("interjection")) == "interjection||"

    def test_labels_show_slugs_only(self):
        assert replay.group_label("intent|mimi|") == "intent[mimi]"
        assert replay.group_label("approval||chisame") == "approval[chisame]"
        assert replay.group_label("interjection||") == "interjection"

    def test_intents_for_different_characters_are_not_a_mixed_prompt(self):
        rows = [_grp_row("intent", "mimi", text="a"), _grp_row("intent", "mimi", text="b"),
                _grp_row("intent", "chisame", text="c")]
        lines = [_grp_line(rows[0], "sha-mimi"), _grp_line(rows[1], "sha-mimi"), _grp_line(rows[2], "sha-chisame")]
        shas = replay.prompt_shas(rows, lines)
        assert shas == {"intent|chisame|": ["sha-chisame"], "intent|mimi|": ["sha-mimi"]}
        assert replay.mixed_prompt_groups(shas) == []  # 種類ごとだと 2 つになり、警告が出ていた

    def test_approvals_with_different_candidates_are_not_mixed_either(self):
        rows = [_grp_row("approval", None, ["chisame"]), _grp_row("approval", None, ["mimi", "sakura"])]
        shas = replay.prompt_shas(rows, [_grp_line(rows[0], "a"), _grp_line(rows[1], "b")])
        assert replay.mixed_prompt_groups(shas) == []

    def test_two_prompts_in_the_same_group_are_warned(self):
        rows = [_grp_row("intent", "mimi", text="a"), _grp_row("intent", "mimi", text="b")]
        shas = replay.prompt_shas(rows, [_grp_line(rows[0], "one"), _grp_line(rows[1], "two")])
        assert replay.mixed_prompt_groups(shas) == ["intent[mimi]"]

    def test_lines_without_a_row_or_a_hash_are_skipped(self):
        row = _grp_row("intent", "mimi")
        shas = replay.prompt_shas([row], [{"id": "unknown", "prompt_sha": "x"}, {"id": row["id"], "prompt_sha": None}])
        assert shas == {}


class TestChangedPromptGroups:
    def test_new_form_compares_per_group(self):
        base = {"intent|mimi|": ["a"], "intent|chisame|": ["b"]}
        assert replay.changed_prompt_groups(base, dict(base)) == []
        assert replay.changed_prompt_groups(base, {"intent|mimi|": ["a"], "intent|chisame|": ["c"]}) == [
            "intent[chisame]"]

    def test_a_group_that_appears_or_disappears_is_a_change(self):
        assert replay.changed_prompt_groups({"intent|mimi|": ["a"]}, {}) == ["intent[mimi]"]

    def test_an_old_form_meta_is_still_readable_against_a_new_one(self):
        old = {"approval": ["a"], "intent": ["m", "c"]}  # 種類だけの鍵（古い meta）
        same = {"approval||chisame": ["a"], "intent|mimi|": ["m"], "intent|chisame|": ["c"]}
        assert replay.changed_prompt_groups(old, same) == []
        changed = {"approval||chisame": ["a"], "intent|mimi|": ["m2"], "intent|chisame|": ["c"]}
        assert replay.changed_prompt_groups(old, changed) == ["intent"]

    def test_two_old_form_metas_compare_by_kind_as_before(self):
        assert replay.changed_prompt_groups({"approval": ["a"], "intent": ["s"]},
                                            {"approval": ["b"], "intent": ["s"]}) == ["approval"]

    def test_empty_metas(self):
        assert replay.changed_prompt_groups({}, {}) == []

    def test_by_kind_merges_the_new_keys(self):
        assert replay.by_kind({"intent|mimi|": ["a"], "intent|chisame|": ["b"], "approval||x": ["c"]}) == {
            "approval": ["c"], "intent": ["a", "b"]}


def test_compare_says_which_group_changed_in_new_form_metas(pair, capsys):
    set_dir, _ = pair
    for label, sha in (("base", "aaa"), ("new", "bbb")):
        path = set_dir / "runs" / f"{label}.meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["prompt_shas"] = {"intent|mimi|": ["same"], "intent|chisame|": [sha]}
        path.write_text(json.dumps(meta), encoding="utf-8")
    _cli(set_dir, "compare", "--base", "base", "--run", "new")
    out = capsys.readouterr().out
    assert "system prompt が基準と違う種類: intent[chisame]" in out and "intent[mimi]" not in out
