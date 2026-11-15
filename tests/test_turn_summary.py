"""
test_turn_summary.py — TURN SUMMARY (TurnStats と format_turn_summary) の単体テスト

偽サーバで 1 ターン流す確認は tests/contract/test_contract_turn_summary.py。ここは集計と整形の約束だけ。
"""

from __future__ import annotations

import re
import threading

from lab_lounge.graph import (
    TurnStats,
    begin_turn_stats,
    current_turn_stats,
    end_turn_stats,
)
from lab_lounge.llm import LLMResult
from lab_lounge.pipeline import format_turn_summary

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

    def test_empty_turn_has_placeholders_not_blanks(self):
        row = dict(re.findall(r"(\w+)=(\S+)", _line(TurnStats(), outcome="TimeoutError")))
        assert (row["speaker"], row["path"], row["source"], row["tools"]) == ("?", "-", "-", "-")
        assert row["result"] == "TimeoutError"

    def test_speaker_argument_wins_over_recorded_one(self):
        assert "speaker=sakura " in _line(TurnStats(speaker="mimi"), speaker="sakura")
        assert "speaker=mimi " in _line(TurnStats(speaker="mimi"))

    def test_does_not_contain_utterance_text(self):
        """実発話の全文を含めない (件数と名前と時間だけ)。tools の名前以外の自由文が入る場所が無い。"""
        stats = TurnStats(speaker="mimi", path="agent", source="structured_response")
        assert "こんにちは" not in _line(stats)


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
