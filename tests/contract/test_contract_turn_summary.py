"""
TURN SUMMARY（run_pipeline の終わりの 1 行）を、偽の LLM サーバで 1 ターン流して確かめる

本物の LangChain と SDK を通し、step の数・tokens・道具・補助が、実際の往復どおりに集計されることを
固定する。数字の期待は偽サーバの usage（1 回の応答が in 12 / out 3）から決まる。
"""

from __future__ import annotations

import logging
import re

from lab_lounge.characters import get_character
from lab_lounge.mcp_servers.ask_character import set_ask_character_context
from lab_lounge.pipeline import run_pipeline

_COMMON = dict(stream_id="s-1", session_id="sess-1", trace_id="trace-1")


def _summaries(caplog) -> list[dict[str, str]]:
    """ログの TURN SUMMARY の行を、key=value の dict にして返す。"""
    rows = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("TURN SUMMARY "):
            rows.append(dict(re.findall(r"(\w+)=(\S+)", message)))
    return rows


def _enable_real_llm(monkeypatch) -> None:
    monkeypatch.setenv("L2_USE_REAL_LLM", "true")


def test_one_turn_with_a_tool_call_is_summarized(agent_env, caplog, monkeypatch):
    """道具を 1 回呼んで締めるターン: step 2 回、補助 1 回 (導入セリフ)、協働先の分は含まれない。"""
    _enable_real_llm(monkeypatch)
    caplog.set_level(logging.INFO, logger="lab_lounge.pipeline")
    caller = get_character("mimi")
    target = get_character("sakura")
    set_ask_character_context(caller_slug="mimi", common=agent_env.common)
    agent_env.llm.script(caller.llm_model, [
        ("tool_call", {"name": "ask_character_tool", "args": {"character_slug": "sakura", "question": "どう思う？"}}),
    ])
    agent_env.llm.script(caller.filler_model, [("text", {"text": "ミミの導入"})])
    agent_env.llm.scene(caller.llm_provider, "ok", text="まとめ")
    agent_env.llm.scene(target.llm_provider, "ok", text="さくらの答え")

    result = run_pipeline("さくらにも聞いてみて", speaker_hint="mimi", **_COMMON)

    assert result.speaker == "mimi"
    (row,) = _summaries(caplog)
    assert row["trace"] == "trace-1"
    assert row["speaker"] == "mimi"
    assert row["path"] == "agent"
    assert row["source"] == "structured_response"
    assert row["tools"] == "ask_character_tool:1"
    # caller の本応答は 2 回 (道具を呼ぶ、締める)。偽サーバの 1 回は in 12 / out 3。
    assert row["steps"] == "2"
    assert row["tokens"] == "in:24/out:6"
    # 導入セリフは補助。協働先の Agent の分は集計に入らない (入れば steps / tokens が増える)。
    assert row["aux"] == "1"
    assert row["aux_tokens"] == "in:12/out:3"
    assert row["result"] == "ok"
    assert int(row["total_ms"]) >= int(row["llm_ms"]) >= 0


def test_failed_turn_still_emits_summary_and_reraises(agent_env, caplog, monkeypatch):
    """例外で終わるターンも 1 行出し、result に例外の型を書き、例外は呼び出し元へ再送出する。"""
    import pytest

    _enable_real_llm(monkeypatch)
    caplog.set_level(logging.INFO, logger="lab_lounge.pipeline")

    def boom(_state):
        raise RuntimeError("generation exploded")

    monkeypatch.setattr("lab_lounge.graph.run_pipeline_graph", boom)

    with pytest.raises(RuntimeError, match="generation exploded"):
        run_pipeline("テスト", speaker_hint="mimi", **_COMMON)

    (row,) = _summaries(caplog)
    assert row["result"] == "RuntimeError"
    assert row["steps"] == "0"
    assert row["path"] == "-"
