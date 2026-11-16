"""
test_turn_summary.py — TURN SUMMARY (TurnStats と format_turn_summary) の単体テスト

偽サーバで 1 ターン流す確認は tests/contract/test_contract_turn_summary.py。ここは集計と整形の約束だけ。
"""

from __future__ import annotations

import logging
import re
import threading
import uuid

from lab_lounge.graph import (
    BubbleToolCallbackHandler,
    TurnStats,
    begin_turn_stats,
    current_turn_stats,
    end_turn_stats,
)
from lab_lounge.llm import LLMResult
from lab_lounge.pipeline import PipelineResult, format_turn_summary, run_pipeline

_INFO = {"input": 12, "output": 3, "reasoning": 2, "cache_read": 4, "cache_creation": 0,
         "finish": "stop", "model": "m"}


def _line(stats: TurnStats, **overrides) -> str:
    args = dict(speaker=None, trace_id="t-1", outcome="ok", total_ms=1500)
    args.update(overrides)
    return format_turn_summary(stats, **args)


class TestFormat:
    def test_is_one_line_of_space_separated_key_values(self):
        """後で機械で読むため、1 行で、先頭以外は key=value、値に空白を含まない。"""
        stats = TurnStats(speaker="mimi", path="agent", source="messages")
        stats.add_tool("web_search_tool")
        stats.add_tool("web_search_tool")
        stats.add_tool("ask_character_tool")
        stats.add_step_start()
        stats.add_step_end(_INFO, 800)

        line = _line(stats)

        assert "\n" not in line and line.startswith("TURN SUMMARY ")
        pairs = line.removeprefix("TURN SUMMARY ").split(" ")
        assert all(re.fullmatch(r"\w+=\S+", p) for p in pairs), pairs
        row = dict(p.split("=", 1) for p in pairs)
        assert row["tools"] == "web_search_tool:2,ask_character_tool:1"
        assert row["tokens"] == "in:12/out:3"
        assert (row["reasoning"], row["cache_read"], row["steps"]) == ("2", "4", "1")
        assert (row["llm_ms"], row["total_ms"], row["result"]) == ("800", "1500", "ok")
        assert (row["fallback"], row["fallback_ms"]) == ("in:0/out:0", "0")

    def test_empty_turn_has_placeholders_not_blanks(self):
        row = dict(re.findall(r"(\w+)=(\S+)", _line(TurnStats(), outcome="TimeoutError")))
        assert (row["speaker"], row["path"], row["source"], row["tools"]) == ("?", "-", "-", "-")
        assert row["result"] == "TimeoutError"

    def test_speaker_argument_wins_over_recorded_one(self):
        assert "speaker=sakura " in _line(TurnStats(speaker="mimi"), speaker="sakura")
        assert "speaker=mimi " in _line(TurnStats(speaker="mimi"))

    def test_values_never_contain_spaces_or_equals(self):
        """空白や = を含む名前でも、行が割れない。空は - にする。区切りに使う , と : は道具の名前の中だけ置き換える。"""
        stats = TurnStats(speaker="mi mi=x", path="", source=None)
        stats.add_tool("web search=tool,v2:beta")

        line = _line(stats, outcome="Weird Error=1", trace_id="t 1")

        pairs = line.removeprefix("TURN SUMMARY ").split(" ")
        assert all(re.fullmatch(r"\w+=\S+", p) for p in pairs), pairs
        row = dict(p.split("=", 1) for p in pairs)
        assert row["speaker"] == "mi_mi_x" and row["trace"] == "t_1" and row["result"] == "Weird_Error_1"
        assert row["path"] == "-" and row["source"] == "-"
        assert row["tools"] == "web_search_tool_v2_beta:1"


class TestTurnStats:
    def test_aux_is_kept_apart_from_steps_and_main_tokens(self):
        stats = TurnStats()
        stats.add_aux_start()
        stats.add_aux_end(_INFO)
        assert (stats.steps, stats.input_tokens, stats.aux_calls, stats.aux_input_tokens) == (0, 0, 1, 12)

    def test_call_llm_result_adds_tokens_and_latency(self):
        """単一ノード / フォールバックは handler を通らないので、call_llm の結果を別に足す。"""
        stats = TurnStats()
        stats.add_call_llm_result(LLMResult("x", "m", 20, 5, 700, "stop"))
        assert (stats.input_tokens, stats.output_tokens, stats.llm_ms) == (20, 5, 700)

    def test_fallback_result_is_kept_apart_from_the_main_answer(self):
        """Agent の失敗後のフォールバックは、本応答の tokens / llm_ms に混ざらない。"""
        stats = TurnStats(path="fallback")
        stats.add_step_start()  # 失敗した Agent の step
        stats.add_fallback_result(LLMResult("x", "m", 20, 5, 700, "stop"))

        assert (stats.input_tokens, stats.output_tokens, stats.llm_ms) == (0, 0, 0)
        row = dict(re.findall(r"(\w+)=(\S+)", _line(stats)))
        assert (row["steps"], row["tokens"]) == ("1", "in:0/out:0")
        assert (row["fallback"], row["fallback_ms"]) == ("in:20/out:5", "700")

    def test_negative_latency_is_not_added(self):
        """開始が記録されていない step の latency (-1) は、合計を減らさない。"""
        stats = TurnStats()
        stats.add_step_end(_INFO, -1)
        assert stats.llm_ms == 0

    def test_concurrent_adds_do_not_lose_counts(self):
        stats = TurnStats()

        def work():
            for _ in range(500):
                stats.add_step_start()
                stats.add_step_end(_INFO, 1)

        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert stats.steps == 4000 and stats.input_tokens == 4000 * 12


class TestContextVar:
    def test_begin_end_restores_previous(self):
        assert current_turn_stats() is None
        stats, token = begin_turn_stats()
        assert current_turn_stats() is stats
        end_turn_stats(token)
        assert current_turn_stats() is None

    def test_other_thread_without_copied_context_does_not_see_it(self):
        """協働先のスレッドは context を写さないので、集計は見えない (= 協働先の分は入らない)。"""
        stats, token = begin_turn_stats()
        seen = []
        t = threading.Thread(target=lambda: seen.append(current_turn_stats()))
        t.start()
        t.join()
        end_turn_stats(token)
        assert seen == [None]


def _result_with_usage():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult as LCResult

    message = AIMessage(content="x", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
    return LCResult(generations=[[ChatGeneration(message=message)]])


class TestHandlerDetach:
    """_run_agent が戻った後に、残った Agent のスレッドが集計へ足さない。"""

    def test_detached_handler_stops_adding_but_still_logs(self, caplog):
        caplog.set_level(logging.INFO, logger="lab_lounge.graph")
        stats, token = begin_turn_stats()
        try:
            handler = BubbleToolCallbackHandler("mimi", {}, model="m", provider="openai")
            run = uuid.uuid4()
            handler.on_chat_model_start({}, [[1]], run_id=run, parent_run_id=None)
            assert stats.steps == 1

            handler.detach_stats()
            handler.on_tool_start({"name": "late_tool"}, "q", run_id=uuid.uuid4())
            handler.on_llm_end(_result_with_usage(), run_id=run)
            handler.on_chat_model_start({}, [[1]], run_id=uuid.uuid4(), parent_run_id=None)
        finally:
            end_turn_stats(token)

        assert (stats.steps, stats.input_tokens, stats.tools) == (1, 0, {})
        assert any("Agent LLM step 1 完了" in r.getMessage() for r in caplog.records)  # ログ行は出る


class TestRunPipelineRobustness:
    def test_reset_failure_does_not_replace_the_result(self, monkeypatch, caplog):
        """集計を外す処理の例外で、ターンの戻り値を置き換えない。TURN SUMMARY は出る。"""
        caplog.set_level(logging.INFO, logger="lab_lounge.pipeline")
        sentinel = PipelineResult("s", "sess", "t", "mimi", [])
        monkeypatch.setattr("lab_lounge.pipeline._run_pipeline_graph", lambda *a, **k: sentinel)

        def broken_reset(_token):
            raise ValueError("reset failed")

        monkeypatch.setattr("lab_lounge.graph.end_turn_stats", broken_reset)

        result = run_pipeline("テスト", stream_id="s", session_id="sess", trace_id="t")

        assert result is sentinel
        assert any(r.getMessage().startswith("TURN SUMMARY ") for r in caplog.records)
