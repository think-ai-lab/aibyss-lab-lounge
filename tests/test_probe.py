"""
probe.py のテスト（API は呼ばない）

LLM を呼ぶ 2 点（router._call_router_llm と filler._call_filler_llm）を、決めた答えを返す関数へ差し替え、
本物の判定の関数と filler の生成を通して流す。本文には目印の文字列だけを使い、画面とファイルに出ないことを確かめる。
本物の SDK と支出の記録を通す確かめは tests/contract/test_contract_probe.py にある。
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from lab_lounge import characters, filler, graph, probe, replay, router, spend

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
        samples = probe.measure_rounds([group], ledger, "unit", gap_sec=2.0, sink=sink.append, sleep=gaps.append)["g"]
        assert seen == [0, 1, 2] and gaps == [2.0, 2.0]
        assert [s.elapsed_ms for s in samples] == [100, 101, 102] and sink == samples

    def test_cost_and_tokens_come_from_the_spend_layer(self, tmp_path):
        ledger = self._ledger(tmp_path)

        def run():
            ledger.add(run="unit", model="claude-haiku-4-5", usd=0.002, usage=None, status="no_usage")
            spend._THREAD.last = {"model": "claude-haiku-4-5", "model_on_wire": "claude-haiku-4-5-20251001",
                                  "status": "usage", "usd": 0.002,
                                  "tokens": {"input_total": 22, "input_uncached": 10, "output": 2, "reasoning": 0, "cache_read": 5,
                                             "cache_creation": 7}}
            return probe.Outcome("claude-haiku-4-5", 50, None)

        (sample,) = probe.measure_rounds([probe.RawGroup("g", [("x", run)])], ledger, "unit", gap_sec=0,
                                         sink=lambda _s: None)["g"]
        assert sample.usd == pytest.approx(0.002) and sample.model_on_wire == "claude-haiku-4-5-20251001"
        assert sample.tokens == {"input_total": 22, "input_uncached": 10, "output": 2, "reasoning": 0, "cache_read": 5,
                                 "cache_creation": 7}

    def test_a_stale_last_call_is_not_reused(self, tmp_path):
        ledger = self._ledger(tmp_path)
        spend._THREAD.last = {"model": "x", "model_on_wire": "stale", "status": "usage", "usd": 1.0,
                              "tokens": {"input_total": 99, "input_uncached": 99}}
        group = probe.RawGroup("g", [("x", lambda: probe.Outcome(None, None, "NoCall"))])
        (sample,) = probe.measure_rounds([group], ledger, "unit", gap_sec=0, sink=lambda _s: None)["g"]
        assert sample.model_on_wire is None and sample.tokens["input_total"] == 0

    def test_sample_line_has_no_text(self):
        sample = probe.Sample("g", 0, "s001", "m", "w", 10, None, 0.1, {"input_total": 1})
        assert set(sample.line()) == {"group", "i", "input_id", "model", "provider", "model_on_wire", "elapsed_ms",
                                      "error", "first_in_process", "first_for_provider", "usd", "tokens"}


class TestSummarize:
    def _sample(self, ms, error=None, usd=0.001, **tokens):
        return probe.Sample("g", 0, "x", "m", "w", ms, error, usd, {"input_total": 1, "output": 1, **tokens})

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


# ─── agent ───────────────────────────────────────────────────────

_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("probe_test_var", default="unset")


class TestTagSpeaker:
    @pytest.mark.parametrize("text, tagged", [
        ("今日の天気は？", True),
        ("[ルカからの発言]\nすでに付いている", False),
        ("[システム] 通知", False),
        ("アビスメイトからの質問です", False),
        ("あ" * 30 + "アビスメイトのみなさん", True),  # 先頭 30 字より後ろの「アビスメイト」は無視される（本番と同じ）
        ("", False),
    ])
    def test_matches_the_production_rule(self, text, tagged):
        assert probe.tag_speaker(text) == (f"[ルカからの発言]\n{text}" if tagged else text)

    def test_production_source_still_has_the_rule_we_copied(self):
        """graph._generation_node の規則が変わったら赤くなる（規則は関数でなく本文の中にあるため、字面で固定する）。"""
        source = Path(graph.__file__).read_text(encoding="utf-8")
        assert 'if text and not text.startswith("[") and "アビスメイト" not in text[:30]:' in source
        assert 'tagged_text = f"[ルカからの発言]\\n{text}"' in source
        assert "run_graph(\n            tagged_text," in source  # run_graph へ渡すのはこの値


class TestCannedTools:
    @pytest.mark.parametrize("chars", [1, 20, 300, 1200, 5000])
    def test_exact_length_and_deterministic(self, chars):
        text = probe.canned_text("web_search", chars)
        assert len(text) == chars and text == probe.canned_text("web_search", chars)

    def test_patches_the_module_functions_and_restores(self):
        from lab_lounge.mcp_servers import ask_character, retrieve_memory, web_search

        originals = (web_search.web_search, retrieve_memory.retrieve_memory, ask_character.ask_character)
        fires: Counter = Counter()
        with probe.canned_tools(probe.DEFAULT_TOOL_CHARS, fires):
            assert len(web_search.web_search("q")) == probe.DEFAULT_TOOL_CHARS["web_search"]
            assert len(retrieve_memory.retrieve_memory("q")) == probe.DEFAULT_TOOL_CHARS["retrieve_memory"]
            assert len(ask_character.ask_character("mimi", "q")) == probe.DEFAULT_TOOL_CHARS["ask_character"]
        assert fires == {"web_search": 1, "retrieve_memory": 1, "ask_character": 1}
        assert (web_search.web_search, retrieve_memory.retrieve_memory, ask_character.ask_character) == originals

    def test_restores_after_an_error(self):
        from lab_lounge.mcp_servers import web_search

        original = web_search.web_search
        with pytest.raises(ZeroDivisionError), probe.canned_tools(probe.DEFAULT_TOOL_CHARS, Counter()):
            1 / 0
        assert web_search.web_search is original

    def test_tool_arguments_do_not_leak_into_the_response(self):
        from lab_lounge.mcp_servers import ask_character

        with probe.canned_tools(probe.DEFAULT_TOOL_CHARS, Counter()):
            assert _SENTINEL not in ask_character.ask_character("mimi", _SENTINEL)

    def test_parse_tool_chars(self):
        assert probe.parse_tool_chars(["web_search=800"])["web_search"] == 800
        assert probe.parse_tool_chars([])["ask_character"] == probe.DEFAULT_TOOL_CHARS["ask_character"]
        for bad in ("nope=10", "web_search=0", "web_search=x", "web_search"):
            with pytest.raises(ValueError):
                probe.parse_tool_chars([bad])


class TestLlmOverride:
    def test_parse(self):
        assert probe.parse_llm_override("anthropic:claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")
        assert probe.parse_llm_override(None) is None
        for bad in ("claude-sonnet-4-6", "mistral:x", "openai:"):
            with pytest.raises(ValueError):
                probe.parse_llm_override(bad)


class TestRecorder:
    def _response(self, *, tool_calls=(), usage=None, meta=None):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        message = AIMessage(content=_SENTINEL, tool_calls=list(tool_calls), usage_metadata=usage,
                            response_metadata=meta or {})
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    def test_step_gets_time_tokens_cache_finish_and_model(self):
        recorder = probe.make_recorder()
        run = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run)
        recorder.on_llm_end(self._response(
            tool_calls=[{"name": "web_search_tool", "args": {"query": _SENTINEL}, "id": "c1"}],
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                   "input_token_details": {"cache_read": 60, "cache_creation": 30},
                   "output_token_details": {"reasoning": 5}},
            meta={"stop_reason": "tool_use", "model_name": "claude-haiku-4-5-20251001"}), run_id=run)
        (step,) = recorder.steps
        assert step["n"] == 1 and step["ms"] is not None and step["error"] is None
        # LangChain の input_tokens は合計（キャッシュの読み書きを含む）。キャッシュ外はそこから引く。
        assert (step["input_total"], step["input_uncached"], step["output"], step["reasoning"], step["cache_read"],
                step["cache_creation"]) == (100, 10, 20, 5, 60, 30)
        assert (step["finish"], step["model"], step["tool_calls"]) == (
            "tool_use", "claude-haiku-4-5-20251001", ["web_search_tool"])
        assert _SENTINEL not in json.dumps(recorder.steps)

    def test_two_overlapping_runs_are_paired_by_run_id(self):
        recorder = probe.make_recorder()
        first, second = uuid.uuid4(), uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=first)
        recorder.on_chat_model_start({}, [[]], run_id=second)
        recorder.on_llm_end(self._response(usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
                            run_id=second)
        recorder.on_llm_end(self._response(usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}),
                            run_id=first)
        assert [s["input_total"] for s in recorder.steps] == [1, 2] and not recorder._llm_open

    def test_error_step_keeps_only_the_type(self):
        recorder = probe.make_recorder()
        run = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run)
        recorder.on_chat_model_error(RuntimeError(_SENTINEL), run_id=run)
        (step,) = recorder.steps
        assert step["error"] == "RuntimeError" and _SENTINEL not in json.dumps(recorder.steps)

    def test_missing_usage_is_zero_not_a_crash(self):
        recorder = probe.make_recorder()
        run = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run)
        recorder.on_llm_end(self._response(), run_id=run)
        assert recorder.steps[0]["input_total"] == 0 and recorder.steps[0]["cache_read"] == 0

    def test_tool_name_and_time_are_recorded_and_odd_names_are_hidden(self):
        recorder = probe.make_recorder()
        ok, odd, failed = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        recorder.on_tool_start({"name": "web_search_tool"}, _SENTINEL, run_id=ok)
        recorder.on_tool_end(_SENTINEL, run_id=ok)
        recorder.on_tool_start({"name": f"bad name {_SENTINEL}"}, "x", run_id=odd)
        recorder.on_tool_end("x", run_id=odd)
        recorder.on_tool_start({}, "x", run_id=failed, name="ask_character_tool")
        recorder.on_tool_error(ValueError(_SENTINEL), run_id=failed)
        assert [t["name"] for t in recorder.tools] == ["web_search_tool", "<不正な名前>", "ask_character_tool"]
        assert recorder.tools[2]["error"] == "ValueError" and _SENTINEL not in json.dumps(recorder.tools)


class TestStreamContext:
    def test_first_run_snapshots_and_later_runs_reuse_it(self, tmp_path):
        snapshot = tmp_path / "ctx.md"
        text, meta = probe.resolve_stream_context(snapshot, refresh=False, load_live=lambda: f"今日の配信 {_SENTINEL}")
        assert meta["origin"] == "live" and meta["chars"] == len(text) and snapshot.is_file()
        text2, meta2 = probe.resolve_stream_context(snapshot, refresh=False, load_live=lambda: "変わった内容")
        assert text2 == text and meta2["origin"] == "snapshot" and meta2["sha"] == meta["sha"]

    def test_refresh_takes_the_live_one(self, tmp_path):
        snapshot = tmp_path / "ctx.md"
        probe.resolve_stream_context(snapshot, refresh=False, load_live=lambda: "古い")
        text, meta = probe.resolve_stream_context(snapshot, refresh=True, load_live=lambda: "新しい")
        assert text == "新しい" and meta["origin"] == "live"
        assert probe.resolve_stream_context(snapshot, refresh=False, load_live=lambda: None)[0] == "新しい"

    def test_no_context_at_all(self, tmp_path):
        text, meta = probe.resolve_stream_context(tmp_path / "ctx.md", refresh=False, load_live=lambda: None)
        assert text is None and meta == {"origin": "none", "chars": 0, "sha": None}
        assert not (tmp_path / "ctx.md").exists()

    def test_meta_never_has_the_text(self, tmp_path):
        _, meta = probe.resolve_stream_context(tmp_path / "ctx.md", refresh=False, load_live=lambda: _SENTINEL)
        assert _SENTINEL not in json.dumps(meta)


def test_persona_hashes_exist_for_every_measured_character():
    for slug in probe.AGENT_CHARACTERS:
        hashes = probe.persona_hashes(slug)
        assert len(hashes["skills_sha"]) == 16 and (hashes["persona_sha"] is None or len(hashes["persona_sha"]) == 16)


class TestInvokeAgent:
    class _Agent:
        def __init__(self, work):
            self.work, self.seen = work, {}

        def invoke(self, payload, config):
            self.seen = {"payload": payload, "config": config, "context": _VAR.get()}
            return self.work()

    def test_passes_the_message_config_and_contextvars(self):
        _VAR.set("from-caller")
        agent = self._Agent(lambda: {"ok": True})
        recorder = object()
        assert probe.invoke_agent(agent, "こんにちは", recorder, timeout_sec=5, recursion_limit=25) == {"ok": True}
        assert agent.seen["payload"] == {"messages": [{"role": "user", "content": "こんにちは"}]}
        assert agent.seen["config"] == {"recursion_limit": 25, "callbacks": [recorder]}
        assert agent.seen["context"] == "from-caller"

    def test_timeout_raises_without_waiting_for_the_worker(self):
        release = threading.Event()
        agent = self._Agent(lambda: release.wait(5))
        started = time.monotonic()
        with pytest.raises(probe.AgentTimeout):
            probe.invoke_agent(agent, "x", object(), timeout_sec=0.05, recursion_limit=25)
        assert time.monotonic() - started < 2
        release.set()

    def test_worker_exceptions_propagate_including_spend_stops(self):
        def boom():
            raise spend.BudgetExceeded("上限")

        with pytest.raises(spend.BudgetExceeded):
            probe.invoke_agent(self._Agent(boom), "x", object(), timeout_sec=5, recursion_limit=25)


class TestSummarizeAgent:
    def _line(self, ms, *, cold=False, error=None, steps=2, tools=(), usd=0.01, source="structured_response"):
        step = {"ms": 100, "input_total": 16, "input_uncached": 10, "output": 5, "reasoning": 1, "cache_read": 4, "cache_creation": 2,
                "finish": "stop", "model": "m-wire", "error": None, "tool_calls": []}
        return {"cold": cold, "error": error, "total_ms": ms, "steps": [step] * steps, "tools_fired": list(tools),
                "source": source, "usd": usd, "model": "m", "finish": "stop"}

    def test_splits_cold_and_warm_and_ignores_failures_in_times(self):
        lines = [self._line(5000, cold=True), self._line(1000), self._line(2000), self._line(3000),
                 self._line(90000, error="Timeout")]
        summary = probe.summarize_agent(lines)
        assert summary["cold_ms"] == 5000.0 and summary["warm_p50"] == 2000.0 and summary["warm_max"] == 3000.0
        assert (summary["n"], summary["failed"], summary["steps_mean"]) == (5, 1, 2.0)
        assert summary["usd"] == pytest.approx(0.05)

    def test_tokens_tools_and_sources(self):
        summary = probe.summarize_agent([self._line(1, tools=["web_search_tool"]),
                                         self._line(1, tools=["web_search_tool", "ask_character_tool"],
                                                    source="messages")])
        assert summary["tokens"]["cache_read"] == 16 and summary["tokens"]["cache_creation"] == 8
        assert summary["tools_fired"] == {"web_search_tool": 2, "ask_character_tool": 1}
        assert summary["source"] == {"structured_response": 1, "messages": 1}

    def test_print_shows_numbers_only(self, capsys):
        probe.print_agent_summary("mimi", probe.summarize_agent([self._line(1200, cold=True), self._line(900)]))
        out = capsys.readouterr().out
        assert "cold=1200ms" in out and "warm p50=900ms" in out and "cache_creation=" in out


class TestAgentCommandLine:
    def test_bad_arguments(self):
        base = ["agent", "--profile", "dev", "--label", "x", "--max-usd", "1"]
        assert probe.main([*base, "--characters", "ruka"]) == probe.EXIT_USAGE
        assert probe.main([*base, "--llm", "bad"]) == probe.EXIT_USAGE
        assert probe.main([*base, "--tool-chars", "nope=1"]) == probe.EXIT_USAGE
        assert probe.main([*base, "--limit", "-1"]) == probe.EXIT_USAGE

    def test_missing_utterances_are_refused(self, tmp_path, capsys):
        code = probe.main(["agent", "--profile", "dev", "--label", "x", "--max-usd", "1", "--out-dir", str(tmp_path),
                           "--utterances", str(tmp_path / "none.jsonl")])
        assert code == probe.EXIT_REFUSED and "断りました" in capsys.readouterr().out


# ─── history ─────────────────────────────────────────────────────

def _log_line(message: str, name: str = "lab_lounge.graph", stamp: str = "2026-06-01 12:00:00,123") -> str:
    return f"{stamp} INFO     {name}: {message}\n"


def _step(n: int, slug: str, ms: int) -> str:
    return _log_line("Agent LLM step %d 完了 [character=%s] latency_ms=%d tokens=in:1/out:1 finish=stop model=m"
                     % (n, slug, ms))


def _total(slug: str, ms: int, text: str = _SENTINEL) -> str:
    return _log_line("Agent 実行完了 [character=%s]: latency_ms=%d source=structured_response text_len=%d text=%r"
                     % (slug, ms, len(text), text))


def _write_log(runs, name, lines):
    runs.mkdir(parents=True, exist_ok=True)
    (runs / name).write_text("".join(lines), encoding="utf-8")


class TestHistory:
    def _runs(self, tmp_path):
        runs = tmp_path / "runs"
        _write_log(runs, "run_loop_20260601_120000.log", [
            _step(1, "mimi", 1000), _step(2, "mimi", 3000), _total("mimi", 4500),
            _step(1, "sakura", 500), _total("sakura", 700, _SENTINEL + "\n続きの行"),
            "続きの行（行頭の形なし）\n",
            _log_line("Agent 実行失敗 [character=mimi]: " + _SENTINEL + "。単一ノードにフォールバック。"),
        ])
        _write_log(runs, "run_loop_20260605_120000.log", [_step(1, "mimi", 2000), _total("mimi", 2500)])
        _write_log(runs, "run_loop_20260705_120000.log", [_step(1, "mimi", 800), _total("mimi", 900)])
        # 使わない: pytest の残り（名前）、テストの目印、応答の無い回
        _write_log(runs, "run_loop_20261031_100000.log", [_step(1, "mimi", 99999), _total("mimi", 99999)])
        # 目印は発話でない行（行頭の形がある行）だけで照らす（replay_set.test_mark_in）
        _write_log(runs, "run_loop_20260610_120000.log", [_step(1, "mimi", 99999),
                                                          _log_line("失敗: boom", name="lab_lounge.pipeline")])
        # Agent が 1 度も step を完了しなかった回も、時間の数字には含める（最も悪い回を落とさない）
        _write_log(runs, "run_loop_20260611_120000.log", [_total("sakura", 30000)])
        return runs

    def test_selects_like_the_replay_set_and_counts_by_month(self, tmp_path):
        runs = self._runs(tmp_path)
        paths, by_rule, notes = probe.select_run_logs(runs, date(2026, 5, 8))
        assert by_rule["使う"] == 4 and sum(by_rule.values()) == 6
        assert any("1 度も無かった回 1 本を含みます" in note for note in notes)
        stats = probe.history_stats(paths)
        assert sorted(stats["totals"][("mimi", "2026-06")]) == [2500.0, 4500.0]
        assert sorted(stats["steps"][("mimi", "2026-06")]) == [1000.0, 2000.0, 3000.0]
        assert sorted(stats["totals"][("sakura", "2026-06")]) == [700.0, 30000.0]
        assert stats["failed"][("mimi", "2026-06")] == 1
        assert max(stats["totals"][("mimi", "2026-07")]) == 900.0  # 使わない回の 99999 は入らない

    def test_report_marks_months_before_the_format_and_empty_months(self):
        stats = {"totals": {("mimi", "2026-06"): [1.0, 2.0, 3.0]}, "steps": {}, "failed": {}}
        table = probe.history_report(stats, ["2026-04", "2026-05", "2026-06", "2026-07"])
        assert table["mimi"]["2026-04"] == {"total": None, "steps": None, "failed": None}
        assert table["mimi"]["2026-05"]["steps"]["n"] == 0 and table["mimi"]["2026-05"]["total"]["n"] == 0
        assert table["mimi"]["2026-06"]["total"] == {"n": 3, "p50": 2.0, "p90": 3.0, "max": 3.0}
        assert probe._dist_text(table["mimi"]["2026-04"]["total"]) == "書式なし"
        assert probe._dist_text(table["mimi"]["2026-07"]["total"]) == "データなし"

    def test_months_between(self):
        assert probe.months_between("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]

    def test_command_prints_numbers_only_and_writes_json(self, tmp_path, capsys):
        runs = self._runs(tmp_path)
        out_dir = tmp_path / "out"
        assert probe.main(["history", "--label", "t", "--logs-dir", str(runs), "--out-dir", str(out_dir)]) == 0
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err
        assert "n=2 p50=2500 p90=4500" in captured.out and "データなし" in captured.out
        saved = json.loads((out_dir / "history_t.json").read_text(encoding="utf-8"))
        assert _SENTINEL not in json.dumps(saved) and saved["formats_since"]["step"] == "2026-05-11"
        assert saved["table"]["mimi"]["2026-06"]["steps"]["n"] == 3
        assert saved["files"]["used"] == 4 and any("含みます" in note for note in saved["notes"])
        assert "注: " in captured.out

    def test_no_logs_is_a_failure(self, tmp_path, capsys):
        (tmp_path / "runs").mkdir()
        assert probe.main(["history", "--logs-dir", str(tmp_path / "runs"), "--out-dir", str(tmp_path)]) == 1
        assert probe.main(["history", "--logs-dir", str(tmp_path / "missing"), "--out-dir", str(tmp_path)]) == 1

    def test_the_log_formats_we_read_still_exist_in_graph(self):
        """graph.py の文面が変わったら赤くなる（history は先頭の形だけを読むので、変わると黙って 0 件になる）。"""
        source = Path(graph.__file__).read_text(encoding="utf-8")
        assert '"Agent LLM step %d 完了 [character=%s] latency_ms=%d%s"' in source
        assert '"Agent 実行完了 [character=%s]: latency_ms=%d source=%s text_len=%d text=%r"' in source
        assert '"Agent 実行失敗 [character=%s]: %s。単一ノードにフォールバック。"' in source


# ─── 入力のエラー ────────────────────────────────────────────────

class TestMeasureInputError:
    def test_it_is_a_value_error_with_only_our_own_wording(self):
        assert issubclass(probe.MeasureInputError, ValueError)

    def test_a_file_that_is_not_utf8_is_reported_by_name_only(self, tmp_path):
        path = tmp_path / "agent_utterances.jsonl"
        path.write_bytes(_SENTINEL.encode() + bytes([0xFF, 0xFE]))
        with pytest.raises(probe.MeasureInputError) as caught:
            probe.load_utterances(path)
        assert "agent_utterances.jsonl" in str(caught.value) and _SENTINEL not in str(caught.value)

    def test_a_synthetic_file_that_is_not_utf8_is_refused_cleanly(self, tmp_path, capsys):
        bad = tmp_path / "replay_synthetic.jsonl"
        bad.write_bytes(_SENTINEL.encode() + bytes([0xFF, 0xFE]))
        code = probe.main(["raw", "--profile", "dev", "--label", "x", "--max-usd", "1", "--out-dir", str(tmp_path),
                           "--synthetic", str(bad), "--utterances", str(_utterances(tmp_path))])
        out = capsys.readouterr().out
        assert code == probe.EXIT_REFUSED and "断りました" in out and _SENTINEL not in out

    def test_line_and_paragraph_separators_inside_a_text_do_not_split_a_row(self, tmp_path):
        text = "{name}、" + chr(0x2028) + "続き" + chr(0x85) + "さらに"
        items = probe.load_utterances(_utterances(tmp_path, [text, "{name}、二つ目"]))
        assert [i.id for i in items] == ["u01", "u02"] and items[0].text == text

    def test_a_crlf_file_is_read(self, tmp_path):
        path = _utterances(tmp_path)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        assert len(probe.load_utterances(path)) == 2

    def test_only_our_own_errors_are_printed(self, tmp_path, monkeypatch, capsys):
        def boom(_path):
            raise ValueError(_SENTINEL)  # 自分の文言ではない ValueError（入力の断片を含みうる）

        monkeypatch.setattr(probe, "load_utterances", boom)
        with pytest.raises(ValueError):
            probe.main(["raw", "--profile", "dev", "--label", "x", "--max-usd", "1", "--out-dir", str(tmp_path),
                        "--synthetic", str(_synthetic(tmp_path)), "--utterances", str(tmp_path / "u.jsonl")])
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err


class TestInnerTimeoutIsNotTheWallClock:
    """agent.invoke の中（SDK など）が自分で投げた TimeoutError は、壁時計の時間切れではない。"""

    class _Agent:
        def __init__(self, error=None, result=None):
            self.error, self.result = error, result

        def invoke(self, payload, config):
            if self.error is not None:
                raise self.error
            return self.result

    def test_a_timeout_raised_inside_the_agent_comes_back_as_it_is(self):
        agent = self._Agent(error=TimeoutError("inner"))
        with pytest.raises(TimeoutError) as caught:
            probe.invoke_agent(agent, "x", object(), timeout_sec=5, recursion_limit=25)
        assert not isinstance(caught.value, probe.AgentTimeout)

    def test_the_wall_clock_timeout_is_still_detected(self):
        release = threading.Event()

        class Slow(self._Agent):
            def invoke(self, payload, config):
                release.wait(5)

        with pytest.raises(probe.AgentTimeout):
            probe.invoke_agent(Slow(), "x", object(), timeout_sec=0.05, recursion_limit=25)
        release.set()

    def test_measure_records_it_as_an_ordinary_failure_and_goes_on(self, tmp_path, monkeypatch):
        monkeypatch.setattr(graph, "_build_agent_graph", lambda *a, **k: self._Agent(error=TimeoutError(_SENTINEL)))
        ledger = spend.SpendLedger(tmp_path / "spend")
        settings = probe.AgentSettings(None, 5.0, 25)
        utterance = probe.Utterance("u01", "greeting", "{name}、こんにちは")
        line, body = probe.measure_agent_call("sakura", utterance, 0, settings, ledger, "unit")
        assert line["error"] == "TimeoutError" and body == "" and _SENTINEL not in json.dumps(line)


class TestToolsOfFailedCalls:
    """失敗した回でも、失敗の前に定型の応答を返した道具を数える（成功の回の数とは分けて出す）。"""

    class _ToolThenFail:
        """道具を 1 回呼んだことをコールバックへ知らせてから、失敗する Agent。"""

        def __init__(self, error):
            self.error = error

        def invoke(self, payload, config):
            (recorder,) = config["callbacks"]
            run = uuid.uuid4()
            recorder.on_tool_start({"name": "retrieve_memory_tool"}, "", run_id=run)
            recorder.on_tool_end("", run_id=run)
            raise self.error

    def test_a_failed_call_keeps_the_tools_that_returned_before_the_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(graph, "_build_agent_graph", lambda *a, **k: self._ToolThenFail(RuntimeError(_SENTINEL)))
        ledger = spend.SpendLedger(tmp_path / "spend")
        settings = probe.AgentSettings(None, 5.0, 25)
        utterance = probe.Utterance("u01", "explain", "{name}、教えて")
        line, body = probe.measure_agent_call("mimi", utterance, 1, settings, ledger, "unit")
        assert line["error"] == "RuntimeError" and body == ""
        assert line["tools_fired"] == ["retrieve_memory_tool"]
        assert _SENTINEL not in json.dumps(line)

    def test_a_wall_clock_timeout_also_keeps_them(self, tmp_path, monkeypatch):
        release = threading.Event()

        class ToolThenHang:
            def invoke(self, payload, config):
                (recorder,) = config["callbacks"]
                run = uuid.uuid4()
                recorder.on_tool_start({"name": "ask_character_tool"}, "", run_id=run)
                recorder.on_tool_end("", run_id=run)
                release.wait(5)

        monkeypatch.setattr(graph, "_build_agent_graph", lambda *a, **k: ToolThenHang())
        ledger = spend.SpendLedger(tmp_path / "spend")
        settings = probe.AgentSettings(None, 0.3, 25)
        utterance = probe.Utterance("u01", "collab", "{name}、聞いて")
        try:
            with pytest.raises(probe.AgentTimeout) as caught:
                probe.measure_agent_call("mimi", utterance, 1, settings, ledger, "unit")
        finally:
            release.set()
        assert caught.value.line["error"] == "Timeout"
        assert caught.value.line["tools_fired"] == ["ask_character_tool"]

    def test_the_summary_counts_them_apart_from_the_successful_calls(self, capsys):
        ok = {"cold": False, "error": None, "total_ms": 10, "steps": [], "tools": [],
              "tools_fired": ["web_search_tool"], "source": "x", "usd": 0.0, "model": "m", "finish": "stop"}
        # 直す前に書いた行の形（失敗の回は tools_fired が空で、コールバックの記録 tools だけがある）
        failed = {**ok, "error": "APITimeoutError", "total_ms": 80000, "tools_fired": [],
                  "tools": [{"name": "retrieve_memory_tool", "ms": 1, "error": None},
                            {"name": "web_search_tool", "ms": 1, "error": "ToolException"}]}
        summary = probe.summarize_agent([ok, failed])
        assert summary["tools_fired"] == {"web_search_tool": 1}  # 成功の回の数の意味は変えない
        assert summary["tools_fired_failed"] == {"retrieve_memory_tool": 1}  # エラーで終わった道具は数えない
        probe.print_agent_summary("mimi", summary)
        assert "道具 web_search_tool×1（失敗の回 retrieve_memory_tool×1）" in capsys.readouterr().out

    def test_the_summary_line_stays_the_same_without_failed_tools(self, capsys):
        ok = {"cold": False, "error": None, "total_ms": 10, "steps": [], "tools": [],
              "tools_fired": [], "source": "x", "usd": 0.0, "model": "m", "finish": "stop"}
        probe.print_agent_summary("mimi", probe.summarize_agent([ok]))
        out = capsys.readouterr().out
        assert "道具 なし  source" in out and "失敗の回" not in out


class TestProviderCapture:
    def _s(self, model, error=None, provider=None):
        return probe.Sample("g", 0, "x", model, None, 10, error, 0.0, {}, provider)

    def test_counts_successful_calls_by_provider_from_the_requested_model(self):
        samples = [self._s("claude-haiku-4-5"), self._s("claude-haiku-4-5"), self._s("gpt-5.4-mini"),
                   self._s("gemini-3.1-flash-lite", error="NoText"), self._s(None)]
        assert probe.expected_by_provider(samples) == {"anthropic": 2, "openai": 1}

    def test_an_unpriced_model_is_not_counted_instead_of_crashing(self):
        assert probe.expected_by_provider([self._s("some-unknown-model")]) == {}

    def test_provider_for_prefers_what_the_http_layer_saw(self):
        assert probe.provider_for("gpt-5.4-mini", {"provider": "google"}) == "google"
        assert probe.provider_for("gpt-5.4-mini", {}) == "openai"
        assert probe.provider_for(None, None) is None

    def test_a_step_without_any_counted_http_call_stops(self):
        step = {"error": None}
        with pytest.raises(spend.UsageCaptureLost):
            probe.check_steps_were_metered({"character": "mimi", "steps": [step], "http_calls": 0})
        probe.check_steps_were_metered({"character": "mimi", "steps": [step], "http_calls": 1})
        probe.check_steps_were_metered({"character": "mimi", "steps": [], "http_calls": 0})  # step が無いなら対象外
        probe.check_steps_were_metered({"character": "mimi", "steps": [{"error": "APIError"}], "http_calls": 0})


class TestTokensAndLabels:
    def _line(self, *, error=None, total=100, uncached=40, cache_read=50, cache_creation=10):
        step = {"ms": 5, "input_total": total, "input_uncached": uncached, "output": 7, "reasoning": 0,
                "cache_read": cache_read, "cache_creation": cache_creation, "finish": "stop", "model": "m",
                "error": None, "tool_calls": []}
        return {"cold": False, "error": error, "total_ms": 10, "steps": [step], "tools_fired": [], "source": "x",
                "usd": 0.0, "model": "m", "finish": "stop"}

    def test_the_two_input_names_mean_total_and_not_cached(self):
        recorder = probe.make_recorder()
        run = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run)
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        message = AIMessage(content="x", usage_metadata={
            "input_tokens": 1000, "output_tokens": 5, "total_tokens": 1005,
            "input_token_details": {"cache_read": 700, "cache_creation": 100}})
        recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)
        (step,) = recorder.steps
        assert step["input_total"] == 1000 and step["input_uncached"] == 200  # 1000 - 700 - 100
        assert step["input_total"] == step["input_uncached"] + step["cache_read"] + step["cache_creation"]

    def test_per_call_tokens_use_only_the_successful_calls(self, capsys):
        lines = [self._line(), self._line(), self._line(error="APIError", total=9999, uncached=9999)]
        summary = probe.summarize_agent(lines)
        assert summary["tokens"]["input_total"] == 200 and summary["tokens"]["input_uncached"] == 80
        probe.print_agent_summary("mimi", summary)
        out = capsys.readouterr().out
        assert "成功 2 回の平均" in out and "in_total=100" in out and "in_uncached=40" in out
        assert "cache_read=50" in out and "cache_creation=10" in out

    def test_raw_summary_names_both_input_counts(self, capsys):
        sample = probe.Sample("g", 0, "x", "m", "w", 10, None, 0.0,
                              {"input_total": 12, "input_uncached": 2, "output": 1, "reasoning": 0, "cache_read": 10,
                               "cache_creation": 0})
        probe.print_summary("g", probe.summarize([sample]))
        out = capsys.readouterr().out
        assert "in_total=12" in out and "in_uncached=2" in out and "合計" in out

    @pytest.mark.parametrize("value, expected", [
        ("end_turn", "end_turn"), ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
        ("answer with spaces " + _SENTINEL, None), (None, None), ("", None), (5, None)])
    def test_provider_supplied_labels_are_shown_only_when_they_look_like_names(self, value, expected):
        shown = probe.label_of(value)
        if expected is None and isinstance(value, str) and value:
            assert shown is not None and _SENTINEL not in shown and shown.startswith("<")
        else:
            assert shown == expected

    def test_a_step_with_an_odd_finish_or_model_is_stored_as_a_length(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        recorder = probe.make_recorder()
        run = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run)
        message = AIMessage(content="x", response_metadata={"stop_reason": "has space " + _SENTINEL,
                                                            "model_name": "m " + _SENTINEL})
        recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)
        assert _SENTINEL not in json.dumps(recorder.steps)


class TestInterleaving:
    def _group(self, name, model, count, log):
        def call(i):
            def run():
                log.append((name, i))
                return probe.Outcome(model, 10, None)
            return run
        return probe.RawGroup(name, [(f"{name}{i}", call(i)) for i in range(count)])

    def test_groups_take_turns_by_input_number(self, tmp_path):
        log: list = []
        groups = [self._group("a", "claude-haiku-4-5", 3, log), self._group("b", "gpt-5.4-mini", 3, log)]
        probe.measure_rounds(groups, spend.SpendLedger(tmp_path / "s"), "unit", gap_sec=0, sink=lambda _s: None)
        assert log == [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)]

    def test_groups_of_different_lengths_and_the_gap_between_every_call(self, tmp_path):
        log, gaps = [], []
        groups = [self._group("a", "claude-haiku-4-5", 1, log), self._group("b", "gpt-5.4-mini", 2, log)]
        collected = probe.measure_rounds(groups, spend.SpendLedger(tmp_path / "s"), "unit", gap_sec=2.0,
                                         sink=lambda _s: None, sleep=gaps.append)
        assert log == [("a", 0), ("b", 0), ("b", 1)] and gaps == [2.0, 2.0]
        assert {name: len(samples) for name, samples in collected.items()} == {"a": 1, "b": 2}

    def test_after_round_is_called_once_per_round_with_that_rounds_samples(self, tmp_path):
        log, rounds = [], []
        groups = [self._group("a", "claude-haiku-4-5", 2, log), self._group("b", "gpt-5.4-mini", 2, log)]
        probe.measure_rounds(groups, spend.SpendLedger(tmp_path / "s"), "unit", gap_sec=0, sink=lambda _s: None,
                             after_round=lambda samples: rounds.append([s.group for s in samples]))
        assert rounds == [["a", "b"], ["a", "b"]]

    def test_first_call_marks(self, tmp_path):
        log: list = []
        groups = [self._group("a", "claude-haiku-4-5", 2, log), self._group("b", "gpt-5.4-mini", 2, log),
                  self._group("c", "claude-haiku-4-5", 1, log)]
        collected = probe.measure_rounds(groups, spend.SpendLedger(tmp_path / "s"), "unit", gap_sec=0,
                                         sink=lambda _s: None)
        flat = [(s.group, s.index, s.first_in_process, s.first_for_provider)
                for index in range(2) for name in "abc" for s in collected.get(name, []) if s.index == index]
        assert flat == [("a", 0, True, True), ("b", 0, False, True), ("c", 0, False, False),
                        ("a", 1, False, False), ("b", 1, False, False)]

    def test_collected_survives_a_stop_in_the_middle(self, tmp_path):
        log: list = []
        collected: dict = {}
        group = self._group("a", "claude-haiku-4-5", 3, log)

        def stop(samples):
            if len(collected["a"]) == 2:
                raise spend.BudgetExceeded("上限")

        with pytest.raises(spend.BudgetExceeded):
            probe.measure_rounds([group], spend.SpendLedger(tmp_path / "s"), "unit", gap_sec=0, sink=lambda _s: None,
                                 collected=collected, after_round=stop)
        assert len(collected["a"]) == 2  # 止まるまでの分は、外から渡した入れ物に残る

    def test_mark_first_calls_for_the_agent_lines(self):
        seen: set[str] = set()
        one, two, three = {"provider": "openai"}, {"provider": "openai"}, {"provider": "google"}
        probe.mark_first_calls(one, True, seen)
        probe.mark_first_calls(two, False, seen)
        probe.mark_first_calls(three, False, seen)
        assert [(x["first_in_process"], x["first_for_provider"]) for x in (one, two, three)] == [
            (True, True), (False, False), (False, True)]
