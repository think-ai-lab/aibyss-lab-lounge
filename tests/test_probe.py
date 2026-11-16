"""
probe.py のテスト（API は呼ばない）

LLM を呼ぶ 2 点（router._call_router_llm と filler._call_filler_llm）を、決めた答えを返す関数へ差し替え、
本物の判定の関数と filler の生成を通して流す。本文には目印の文字列だけを使い、画面とファイルに出ないことを確かめる。
本物の SDK と支出の記録を通す確かめは tests/contract/test_contract_probe.py にある。
"""

from __future__ import annotations

import json

import pytest

from lab_lounge import characters, filler, probe, replay, router, spend

_SENTINEL = "SENTINEL-probe-5d2e"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _synthetic_rows():
    return [
        {"id": "s001", "kind": "approval", "input": f"{_SENTINEL}-a1", "character": None, "candidates": ["chisame"],
         "expected": "granted:chisame", "alt_ok": [], "note": ""},
        {"id": "s002", "kind": "approval", "input": f"{_SENTINEL}-a2", "character": None, "candidates": ["mimi"],
         "expected": "denied:mimi", "alt_ok": [], "note": ""},
        {"id": "s003", "kind": "interjection", "input": f"{_SENTINEL}-j1", "character": None, "candidates": [],
         "expected": "sakura", "alt_ok": [], "note": ""},
        {"id": "s004", "kind": "intent", "input": f"{_SENTINEL}-i1", "character": "mimi", "candidates": [],
         "expected": "mention", "alt_ok": [], "note": ""},
        {"id": "s005", "kind": "route", "input": f"{_SENTINEL}-r1", "character": None,
         "candidates": ["mimi", "chisame"], "expected": "chisame", "alt_ok": [], "note": ""},
    ]


def _synthetic(tmp_path, rows=None):
    return _write_jsonl(tmp_path / "replay_synthetic.jsonl", rows or _synthetic_rows())


def _utterances(tmp_path, texts=None):
    texts = texts or [f"{{name}}、{_SENTINEL}-u1", f"{{name}}と{{other}}で{_SENTINEL}-u2"]
    rows = [{"id": f"u{i + 1:02d}", "category": probe.UTTERANCE_CATEGORIES[i % 5], "text": text}
            for i, text in enumerate(texts)]
    return _write_jsonl(tmp_path / "agent_utterances.jsonl", rows)


class TestPercentile:
    def test_nearest_rank(self):
        values = [float(v) for v in range(1, 11)]
        assert probe.percentile(values, 0.5) == 5.0
        assert probe.percentile(values, 0.9) == 9.0
        assert probe.percentile(values, 1.0) == 10.0

    def test_single_and_empty(self):
        assert probe.percentile([7.0], 0.9) == 7.0
        assert probe.percentile([], 0.5) is None

    def test_does_not_mutate_input(self):
        values = [3.0, 1.0, 2.0]
        probe.percentile(values, 0.5)
        assert values == [3.0, 1.0, 2.0]


class TestInputs:
    def test_router_pool_splits_kinds(self, tmp_path):
        interjection, approval = probe.load_router_pool(_synthetic(tmp_path))
        assert [i.id for i in interjection] == ["s003", "s004"]
        assert [(i.id, i.candidates) for i in approval] == [("s001", ("chisame",)), ("s002", ("mimi",))]

    def test_router_pool_needs_both(self, tmp_path):
        rows = [r for r in _synthetic_rows() if r["kind"] != "approval"]
        with pytest.raises(ValueError, match="入力がありません"):
            probe.load_router_pool(_synthetic(tmp_path, rows))

    def test_utterances_load(self, tmp_path):
        items = probe.load_utterances(_utterances(tmp_path))
        assert [i.id for i in items] == ["u01", "u02"]

    def test_utterances_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="がありません"):
            probe.load_utterances(tmp_path / "none.jsonl")

    @pytest.mark.parametrize("row", [
        {"id": "u01", "category": "nope", "text": "x"},
        {"id": "u01", "category": "deep", "text": ""},
        {"id": "u01", "category": "deep", "text": "{who}、こんにちは"},
        {"id": 1, "category": "deep", "text": "x"},
    ])
    def test_utterances_bad_row_reports_only_the_line_number(self, tmp_path, row):
        row = {**row, "text": row["text"] + _SENTINEL if row["text"] else row["text"]}
        path = _write_jsonl(tmp_path / "u.jsonl", [row])
        with pytest.raises(ValueError) as caught:
            probe.load_utterances(path)
        assert "1 行目" in str(caught.value) and _SENTINEL not in str(caught.value)

    def test_utterances_duplicate_id_and_broken_json(self, tmp_path):
        path = tmp_path / "u.jsonl"
        path.write_text(
            json.dumps({"id": "u01", "category": "deep", "text": "a"}) + "\n"
            + json.dumps({"id": "u01", "category": "deep", "text": "b"}) + "\n"
            + "{" + _SENTINEL + "\n", encoding="utf-8")
        with pytest.raises(ValueError) as caught:
            probe.load_utterances(path)
        assert "2, 3 行目" in str(caught.value) and _SENTINEL not in str(caught.value)

    def test_expand_uses_wake_words_and_fixed_partner(self):
        for slug, partner in probe.OTHER_PANELIST.items():
            text = probe.expand_utterance("{name}、{other}はどう思う？", slug)
            mine = characters.get_character(slug).wake_word
            theirs = characters.get_character(partner).wake_word
            assert text == f"{mine}、{theirs}はどう思う？"

    def test_other_panelist_is_always_someone_else_who_has_a_wake_word(self):
        for slug, partner in probe.OTHER_PANELIST.items():
            assert partner != slug and characters.get_character(partner).wake_word

    def test_cycle_is_deterministic(self):
        assert probe.cycle(["a", "b", "c"], 7) == ["a", "b", "c", "a", "b", "c", "a"]


class TestSleep:
    def test_prevents_then_restores(self, monkeypatch):
        calls = []
        monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: calls.append(prevent) or True)
        with probe.prevent_sleep():
            assert calls == [True]
        assert calls == [True, False]

    def test_restores_after_an_error(self, monkeypatch):
        calls = []
        monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: calls.append(prevent) or True)
        with pytest.raises(ZeroDivisionError), probe.prevent_sleep():
            1 / 0
        assert calls == [True, False]

    def test_unavailable_is_a_no_op(self, monkeypatch):
        calls = []
        monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: calls.append(prevent) or False)
        with probe.prevent_sleep():
            pass
        assert calls == [True]


class TestRecordingFiller:
    def test_records_time_and_failure_and_restores(self, monkeypatch):
        answers = iter([f"{_SENTINEL}-独り言", None])
        monkeypatch.setattr(filler, "_call_filler_llm", lambda model, system, user: next(answers))
        patched = filler._call_filler_llm
        with probe.RecordingFiller():
            ok = probe._filler_outcome("mimi", f"{_SENTINEL}-q")
            failed = probe._filler_outcome("mimi", f"{_SENTINEL}-q")
            assert filler._call_filler_llm is not patched
        assert filler._call_filler_llm is patched
        assert (ok.model, ok.error) == ("gpt-5.4-mini", None) and ok.elapsed_ms is not None
        assert failed.error == "NoText"

    def test_unknown_character_is_no_call(self):
        with probe.RecordingFiller():
            assert probe._filler_outcome("nobody", "x").error == "NoCall"


class TestRouterOutcome:
    def test_records_error_type_from_the_real_function(self, monkeypatch):
        def failing(model, system_prompt, user_text):
            router._router_call_state.error = "APITimeoutError"
            return None

        monkeypatch.setattr(router, "_call_router_llm", failing)
        with replay.RecordingRouter():
            outcome = probe._router_outcome(lambda: router.check_approval(_SENTINEL, ["chisame"]))
        assert (outcome.error, outcome.model) == ("APITimeoutError", "claude-haiku-4-5")

    def test_no_llm_call_is_reported(self):
        outcome = probe._router_outcome(lambda: None)
        assert outcome.error == "NoCall" and outcome.elapsed_ms is None


class TestMeasureGroup:
    def _ledger(self, tmp_path):
        return spend.SpendLedger(tmp_path / "spend")

    def test_runs_in_order_with_gaps_and_collects_samples(self, tmp_path):
        ledger = self._ledger(tmp_path)
        seen, gaps = [], []

        def call(i):
            def run():
                seen.append(i)
                return probe.Outcome("claude-haiku-4-5", 100 + i, None)
            return run

        group = probe.RawGroup("g", [(f"id{i}", call(i)) for i in range(3)])
        sink: list[probe.Sample] = []
        samples = probe.measure_group(group, ledger, "unit", gap_sec=2.0, sink=sink.append, sleep=gaps.append)
        assert seen == [0, 1, 2] and gaps == [2.0, 2.0]
        assert [s.elapsed_ms for s in samples] == [100, 101, 102] and sink == samples

    def test_cost_and_tokens_come_from_the_spend_layer(self, tmp_path):
        ledger = self._ledger(tmp_path)

        def run():
            ledger.add(run="unit", model="claude-haiku-4-5", usd=0.002, usage=None, status="no_usage")
            spend._THREAD.last = {"model": "claude-haiku-4-5", "model_on_wire": "claude-haiku-4-5-20251001",
                                  "status": "usage", "usd": 0.002,
                                  "tokens": {"input": 10, "output": 2, "reasoning": 0, "cache_read": 5,
                                             "cache_creation": 7}}
            return probe.Outcome("claude-haiku-4-5", 50, None)

        (sample,) = probe.measure_group(probe.RawGroup("g", [("x", run)]), ledger, "unit", gap_sec=0,
                                        sink=lambda _s: None)
        assert sample.usd == pytest.approx(0.002) and sample.model_on_wire == "claude-haiku-4-5-20251001"
        assert sample.tokens == {"input": 10, "output": 2, "reasoning": 0, "cache_read": 5, "cache_creation": 7}

    def test_a_stale_last_call_is_not_reused(self, tmp_path):
        ledger = self._ledger(tmp_path)
        spend._THREAD.last = {"model": "x", "model_on_wire": "stale", "status": "usage", "usd": 1.0,
                              "tokens": {"input": 99}}
        group = probe.RawGroup("g", [("x", lambda: probe.Outcome(None, None, "NoCall"))])
        (sample,) = probe.measure_group(group, ledger, "unit", gap_sec=0, sink=lambda _s: None)
        assert sample.model_on_wire is None and sample.tokens["input"] == 0

    def test_sample_line_has_no_text(self):
        sample = probe.Sample("g", 0, "s001", "m", "w", 10, None, 0.1, {"input": 1})
        assert set(sample.line()) == {"group", "i", "input_id", "model", "model_on_wire", "elapsed_ms", "error",
                                      "usd", "tokens"}


class TestSummarize:
    def _sample(self, ms, error=None, usd=0.001, **tokens):
        return probe.Sample("g", 0, "x", "m", "w", ms, error, usd, {"input": 1, "output": 1, **tokens})

    def test_times_use_successes_only(self):
        samples = [self._sample(100), self._sample(200), self._sample(90000, error="APITimeoutError")]
        summary = probe.summarize(samples)
        assert (summary["n"], summary["failed"], summary["max"]) == (3, 1, 200.0)
        assert summary["p50"] == 100.0 and summary["usd"] == pytest.approx(0.003)

    def test_cache_tokens_are_summed(self):
        summary = probe.summarize([self._sample(1, cache_read=4, cache_creation=2),
                                   self._sample(1, cache_read=6, cache_creation=0)])
        assert summary["tokens"]["cache_read"] == 10 and summary["tokens"]["cache_creation"] == 2

    def test_all_failed(self):
        summary = probe.summarize([self._sample(None, error="NoCall")])
        assert summary["p50"] is None and summary["failed"] == 1

    def test_print_summary_shows_numbers_only(self, capsys):
        probe.print_summary("filler:mimi", probe.summarize([self._sample(120)]))
        out = capsys.readouterr().out
        assert "p50=120ms" in out and "cache_creation=0" in out


class TestBuildGroups:
    def _pools(self, tmp_path):
        interjection, approval = probe.load_router_pool(_synthetic(tmp_path))
        return interjection, approval, probe.load_utterances(_utterances(tmp_path))

    def test_groups_by_only(self, tmp_path):
        interjection, approval, utterances = self._pools(tmp_path)

        def names(only):
            groups = probe.build_groups(n=3, only=only, interjection=interjection, approval=approval,
                                        utterances=utterances)
            return [g.name for g in groups]

        assert names("router") == ["router:interjection", "router:approval"]
        assert names("filler") == [f"filler:{s}" for s in probe.FILLER_CHARACTERS]
        assert len(names("all")) == 6

    def test_each_group_has_n_calls_cycling_the_pool(self, tmp_path):
        interjection, approval, utterances = self._pools(tmp_path)
        groups = probe.build_groups(n=5, only="all", interjection=interjection, approval=approval,
                                    utterances=utterances)
        assert all(len(g.calls) == 5 for g in groups)
        assert [i for i, _ in groups[0].calls] == ["s003", "s004", "s003", "s004", "s003"]


class TestCommandLine:
    def test_bad_arguments(self):
        assert probe.main(["raw", "--profile", "dev", "--label", "x", "--max-usd", "1", "--n", "0"]) == probe.EXIT_USAGE
        assert probe.main(["nope"]) == probe.EXIT_USAGE
        assert probe.main(["raw"]) == probe.EXIT_USAGE

    def test_bad_label(self):
        assert probe.main(["raw", "--profile", "dev", "--label", "a b", "--max-usd", "1"]) == probe.EXIT_USAGE

    def test_missing_inputs_are_refused_before_any_call(self, tmp_path, capsys):
        code = probe.main(["raw", "--profile", "dev", "--label", "x", "--max-usd", "1", "--out-dir", str(tmp_path),
                           "--synthetic", str(tmp_path / "none.jsonl"), "--utterances", str(tmp_path / "none2.jsonl")])
        assert code == probe.EXIT_REFUSED
        assert "断りました" in capsys.readouterr().out
