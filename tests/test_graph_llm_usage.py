"""
test_graph_llm_usage.py — Agent の step ログ (tokens / finish / モデル名 / 補助) のテスト

対象: graph.py の _extract_llm_usage / _format_llm_usage と BubbleToolCallbackHandler の
on_chat_model_start / on_llm_end / on_llm_error / on_tool_start。

本物の langchain_core のメッセージ型 (AIMessage / ChatGeneration / LLMResult) で、4 つの
プロバイダの形を作る。偽サーバ越しに SDK を通す確認は tests/contract 側 (step ログの tokens は
test_contract_ask_character の補助の行) にあり、ここは「読み方」の単体の固定。
"""

from __future__ import annotations

import logging
import uuid

import pytest

from lab_lounge.graph import (
    BubbleToolCallbackHandler,
    _extract_llm_usage,
    _format_llm_usage,
)

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402


def _result(*, usage=None, meta=None, llm_output=None) -> LLMResult:
    """1 回ぶんの応答 (LLMResult) を作る。usage / meta は message 側、llm_output は LLMResult 側。"""
    message = AIMessage(content="x", usage_metadata=usage, response_metadata=meta or {})
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


# 4 つの形。usage_metadata は LangChain が 3 社で揃えた形、response_metadata はプロバイダごとに違う。
OPENAI_CHAT = _result(
    usage={
        "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
        "input_token_details": {"cache_read": 4},
        "output_token_details": {"reasoning": 2},
    },
    meta={"model_name": "gpt-5.5-2026-01-01", "finish_reason": "stop"},
)
# Responses API は finish_reason を持たず、status だけを持つ。
OPENAI_RESPONSES = _result(
    usage={
        "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
        "output_token_details": {"reasoning": 2},
    },
    meta={"model_name": "gpt-5.5-2026-01-01", "status": "completed"},
)
# Anthropic は stop_reason。キャッシュ書き込みも持つ。
ANTHROPIC = _result(
    usage={
        "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
        "input_token_details": {"cache_read": 5, "cache_creation": 7},
    },
    meta={"model_name": "claude-sonnet-5-5", "stop_reason": "end_turn"},
)
# Gemini は model_version。finish_reason は大文字の列挙名。
GEMINI = _result(
    usage={
        "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
        "output_token_details": {"reasoning": 3},
    },
    meta={"model_version": "gemini-3-flash", "finish_reason": "MAX_TOKENS"},
)


class TestExtractLlmUsage:
    """usage_metadata と response_metadata から、5 つの数字と finish とモデル名を読む。"""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            (OPENAI_CHAT, dict(input=12, output=3, reasoning=2, cache_read=4, cache_creation=0,
                               finish="stop", model="gpt-5.5-2026-01-01")),
            (OPENAI_RESPONSES, dict(input=12, output=3, reasoning=2, cache_read=0, cache_creation=0,
                                    finish="completed", model="gpt-5.5-2026-01-01")),
            (ANTHROPIC, dict(input=12, output=3, reasoning=0, cache_read=5, cache_creation=7,
                             finish="end_turn", model="claude-sonnet-5-5")),
            (GEMINI, dict(input=12, output=3, reasoning=3, cache_read=0, cache_creation=0,
                          finish="MAX_TOKENS", model="gemini-3-flash")),
        ],
        ids=["openai-chat", "openai-responses", "anthropic", "gemini"],
    )
    def test_four_provider_shapes(self, response, expected):
        assert _extract_llm_usage(response) == expected

    def test_falls_back_to_llm_output_without_usage_metadata(self):
        """usage_metadata が無い (旧形式の) 応答は、llm_output の token_usage を読む。"""
        response = _result(llm_output={
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 5},
            "model_name": "gpt-4o",
        })
        info = _extract_llm_usage(response)
        assert (info["input"], info["output"]) == (20, 5)
        assert info["model"] == "gpt-4o"
        assert info["finish"] is None

    def test_nothing_readable_does_not_raise(self):
        """空の応答や壊れた形でも例外にならず、全部 None で返る (ログのために配信を止めない)。"""
        assert all(v is None for v in _extract_llm_usage(LLMResult(generations=[])).values())
        assert all(v is None for v in _extract_llm_usage(object()).values())


class TestFormatLlmUsage:
    """step ログの末尾の整形。tokens=in:X/out:Y の形は先頭に残る。"""

    def test_keeps_legacy_tokens_shape_first(self):
        text = _format_llm_usage(_extract_llm_usage(ANTHROPIC))
        assert text.startswith(" tokens=in:12/out:3 ")
        assert text == (
            " tokens=in:12/out:3 reasoning=0 cache_read=5 cache_creation=7"
            " finish=end_turn model=claude-sonnet-5-5"
        )

    def test_missing_usage_omits_tokens_and_marks_unknowns(self):
        assert _format_llm_usage(_extract_llm_usage(_result())) == " finish=- model=-"


class _LogCapture:
    """handler の logger (lab_lounge.graph) の INFO 以上を、文字列の列で受ける。"""

    def __init__(self, caplog):
        caplog.set_level(logging.INFO, logger="lab_lounge.graph")
        self._caplog = caplog

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self._caplog.records if r.name == "lab_lounge.graph"]


class TestHandlerStepLogs:
    """handler が step 行に tokens / finish / モデルを出し、道具の子の LLM を補助として分ける。"""

    def _handler(self) -> BubbleToolCallbackHandler:
        return BubbleToolCallbackHandler("mimi", {}, model="m", provider="openai")

    def test_step_line_has_tokens_finish_and_model(self, caplog):
        log = _LogCapture(caplog)
        handler = self._handler()
        run = uuid.uuid4()

        handler.on_chat_model_start({}, [[1, 2]], run_id=run, parent_run_id=None)
        handler.on_llm_end(GEMINI, run_id=run)

        start, end = log.messages()
        assert start.startswith("Agent LLM step 1 開始 [character=mimi] msgs=2")
        assert end.startswith("Agent LLM step 1 完了 [character=mimi] latency_ms=")
        assert "tokens=in:12/out:3 reasoning=3" in end
        assert "finish=MAX_TOKENS model=gemini-3-flash" in end

    def test_child_of_tool_is_aux_and_does_not_advance_step(self, caplog):
        """道具の run の子の LLM (導入セリフ) は、補助として別の行に出し、step 番号を進めない。"""
        log = _LogCapture(caplog)
        handler = self._handler()
        tool_run, aux_run, main_run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        handler.on_chat_model_start({}, [[1]], run_id=uuid.uuid4(), parent_run_id=None)  # step 1
        handler.on_tool_start({"name": "some_tool"}, "q", run_id=tool_run)
        handler.on_chat_model_start({}, [[1]], run_id=aux_run, parent_run_id=tool_run)    # 補助
        handler.on_llm_end(OPENAI_CHAT, run_id=aux_run)
        handler.on_chat_model_start({}, [[1]], run_id=main_run, parent_run_id=None)      # step 2

        text = "\n".join(log.messages())
        assert "Agent 補助 LLM 開始" in text
        assert "Agent 補助 LLM 完了 [character=mimi] latency_ms=" in text
        assert "Agent LLM step 2 開始" in text and "step 3" not in text

    def test_unknown_tool_run_id_is_remembered_too(self, caplog):
        """未知の道具 (bubble を出さない) の子の LLM も、補助として扱う。"""
        log = _LogCapture(caplog)
        handler = self._handler()
        tool_run, aux_run = uuid.uuid4(), uuid.uuid4()

        handler.on_tool_start({"name": "some_unknown_tool"}, "q", run_id=tool_run)
        handler.on_chat_model_start({}, [[1]], run_id=aux_run, parent_run_id=tool_run)

        assert any("Agent 補助 LLM 開始" in m for m in log.messages())

    def test_aux_end_does_not_consume_main_step_timer(self, caplog):
        """補助の完了が、本応答の step の開始時刻を上書きしたり消したりしない。"""
        _LogCapture(caplog)
        handler = self._handler()
        tool_run, aux_run = uuid.uuid4(), uuid.uuid4()
        handler.on_chat_model_start({}, [[1]], run_id=uuid.uuid4(), parent_run_id=None)
        t0 = handler._llm_step_t0
        handler.on_tool_start({"name": "x"}, "q", run_id=tool_run)
        handler.on_chat_model_start({}, [[1]], run_id=aux_run, parent_run_id=tool_run)
        handler.on_llm_end(OPENAI_CHAT, run_id=aux_run)
        assert handler._llm_step_t0 == t0

    def test_aux_error_is_separate_warning(self, caplog):
        caplog.set_level(logging.INFO, logger="lab_lounge.graph")
        handler = self._handler()
        tool_run, aux_run = uuid.uuid4(), uuid.uuid4()
        handler.on_tool_start({"name": "x"}, "q", run_id=tool_run)
        handler.on_chat_model_start({}, [[1]], run_id=aux_run, parent_run_id=tool_run)
        handler.on_llm_error(TimeoutError("boom"), run_id=aux_run)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and warnings[0].startswith("Agent 補助 LLM エラー")
