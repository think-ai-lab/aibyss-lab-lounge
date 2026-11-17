"""
probe raw の契約テスト（本物の SDK と支出の記録を、偽の LLM サーバで通す）

``python -m lab_lounge.probe raw`` の入口（main）を、環境の読み込み（prepare_env）だけ止めて流す。
router は Anthropic、filler は 4 キャラ分の 3 社（OpenAI 2、Google、Anthropic）へ向かう。
本文には目印の文字列だけを使い、画面、実行の記録、支出の記録に出ないことを確かめる。
"""

from __future__ import annotations

import json

import pytest

from lab_lounge import preflight, probe, spend
from lab_lounge.characters import get_character

_SENTINEL = "SENTINEL-probe-contract-7c19"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def probe_env(fake_llm, tmp_path, monkeypatch):
    """合成の入力と、環境を読まない入口を用意する。"""
    synthetic = _write_jsonl(tmp_path / "replay_synthetic.jsonl", [
        {"id": "s001", "kind": "approval", "input": f"{_SENTINEL}-a", "character": None, "candidates": ["chisame"],
         "expected": "granted:chisame", "alt_ok": [], "note": ""},
        {"id": "s002", "kind": "interjection", "input": f"{_SENTINEL}-j", "character": None, "candidates": [],
         "expected": "sakura", "alt_ok": [], "note": ""},
    ])
    utterances = _write_jsonl(tmp_path / "agent_utterances.jsonl", [
        {"id": "u01", "category": "greeting", "text": f"{{name}}、{_SENTINEL}-u1"},
        {"id": "u02", "category": "collab", "text": f"{{name}}と{{other}}で{_SENTINEL}-u2"},
    ])
    monkeypatch.setattr(preflight, "prepare_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: False)
    for name in ("L2_DEBUG_ARTIFACTS", "L2_DECISION_LOG", *preflight._TRACING_VARS):
        monkeypatch.setenv(name, "false")
    for provider in ("anthropic", "openai", "google"):
        fake_llm.scene(provider, "ok", text="none")
    return tmp_path / "out", tmp_path / "spend", synthetic, utterances


def _run(env, *extra, label="contract"):
    out_dir, spend_dir, synthetic, utterances = env
    return probe.main(["raw", "--profile", "dev", "--label", label, "--max-usd", "1", "--n", "3",
                       "--out-dir", str(out_dir), "--spend-dir", str(spend_dir), "--synthetic", str(synthetic),
                       "--utterances", str(utterances), *extra])


def _lines(out_dir, label="contract"):
    path = out_dir / f"raw_{label}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_raw_measures_every_group_without_text(probe_env, fake_llm, capsys):
    out_dir, spend_dir, _synthetic, _utterances = probe_env

    assert _run(probe_env) == 0

    captured = capsys.readouterr()
    assert _SENTINEL not in captured.out + captured.err
    lines = _lines(out_dir)
    assert len(lines) == 18  # 6 グループ × n=3
    by_group: dict[str, list[dict]] = {}
    for line in lines:
        by_group.setdefault(line["group"], []).append(line)
    assert sorted(by_group) == ["filler:chisame", "filler:mimi", "filler:octamaid", "filler:sakura",
                                "router:approval", "router:interjection"]
    # モデルは本物の関数が選んだもの。キャラごとの filler_model と、router の既定。
    assert {g: {line["model"] for line in rows} for g, rows in by_group.items()} == {
        "router:interjection": {"claude-haiku-4-5"}, "router:approval": {"claude-haiku-4-5"},
        "filler:mimi": {"gpt-5.4-mini"}, "filler:chisame": {"gemini-3.1-flash-lite"},
        "filler:sakura": {"claude-haiku-4-5"}, "filler:octamaid": {"gpt-5.4-nano"},
    }
    assert all(line["error"] is None and line["elapsed_ms"] is not None for line in lines)
    # tokens は HTTP の層の写し。3 社とも拾えて、費用が付く。
    assert all(line["tokens"]["input_total"] > 0 and line["tokens"]["input_uncached"] > 0 and line["tokens"]["output"] > 0 and line["usd"] > 0 for line in lines)
    assert all(set(line["tokens"]) == {"input_total", "input_uncached", "output", "reasoning", "cache_read", "cache_creation"} for line in lines)
    assert all(line["model_on_wire"] for line in lines)

    meta = json.loads((out_dir / "raw_contract.meta.json").read_text(encoding="utf-8"))
    assert meta["n"] == 3 and meta["models"]["filler"]["octamaid"] == "gpt-5.4-nano"
    assert meta["pools"] == {"interjection": 1, "approval": 1, "filler": 2}
    assert set(meta["summary"]) == set(by_group)
    spent = json.loads((spend_dir / "spent.json").read_text(encoding="utf-8"))
    assert spent["by_run"]["contract"]["calls"] == 18
    for path in [*out_dir.iterdir(), *spend_dir.iterdir()]:
        if path.suffix != ".log":
            assert _SENTINEL not in path.read_text(encoding="utf-8"), path.name


def test_filler_inputs_use_the_wake_words(probe_env, fake_llm):
    assert _run(probe_env, "--only", "filler") == 0
    bodies = " ".join(json.dumps(req.body, ensure_ascii=False) for req in fake_llm.requests_for())
    assert "ミミ様、" in bodies and "ちさめさんと" in bodies  # {other}（mimi の相手）= ちさめさん
    assert "{name}" not in bodies and "{other}" not in bodies


def test_only_router_sends_nothing_to_the_filler_models(probe_env, fake_llm):
    assert _run(probe_env, "--only", "router") == 0
    assert len(fake_llm.requests_for("anthropic")) == 6
    assert fake_llm.requests_for("openai") == [] and fake_llm.requests_for("google") == []


def test_failed_calls_are_counted_and_exit_1(probe_env, fake_llm, capsys):
    fake_llm.scene("anthropic", "http_error", status=400)
    assert _run(probe_env, "--only", "router") == 1
    out = capsys.readouterr().out
    assert "失敗=3" in out and _SENTINEL not in out
    assert all(line["error"] for line in _lines(probe_env[0]))


def test_existing_label_is_refused(probe_env, capsys):
    assert _run(probe_env, "--only", "router") == 0
    assert _run(probe_env, "--only", "router") == probe.EXIT_REFUSED
    assert "すでにあります" in capsys.readouterr().out


def test_budget_stop_sends_nothing(probe_env, fake_llm):
    out_dir, spend_dir, synthetic, utterances = probe_env
    with pytest.raises(spend.BudgetExceeded):
        probe.main(["raw", "--profile", "dev", "--label", "tiny", "--max-usd", "0.0000001", "--n", "3",
                    "--out-dir", str(out_dir), "--spend-dir", str(spend_dir), "--synthetic", str(synthetic),
                    "--utterances", str(utterances), "--only", "router"])
    assert fake_llm.requests_for("anthropic") == []


# ─── agent ───────────────────────────────────────────────────────

_CONTEXT_SENTINEL = "SENTINEL-probe-context-3e55"
_ANSWER_SENTINEL = "SENTINEL-probe-answer-91ab"
_AGENT_SLUGS = ("mimi", "chisame", "sakura", "octamaid")


class _Exited(Exception):
    """hard_exit の代わり。本物はプロセスを終えるので、テストでは例外にして止める。"""


@pytest.fixture
def agent_probe_env(agent_env, tmp_path, monkeypatch):
    """Agent の経路を偽サーバだけで流せる環境と、合成の発話、配信文脈を用意する。"""
    utterances = _write_jsonl(tmp_path / "agent_utterances.jsonl", [
        {"id": "u01", "category": "greeting", "text": f"{{name}}、{_SENTINEL}-u1"},
        {"id": "u02", "category": "collab", "text": f"{{name}}と{{other}}で{_SENTINEL}-u2"},
        {"id": "u03", "category": "deep", "text": f"{{name}}、{_SENTINEL}-u3"},
    ])
    context = tmp_path / "current.md"
    context.write_text(f"今日の配信 {_CONTEXT_SENTINEL}\n", encoding="utf-8")
    monkeypatch.setenv("L2_STREAM_CONTEXT_FILE", str(context))
    monkeypatch.setattr(preflight, "prepare_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: False)
    for name in ("L2_DEBUG_ARTIFACTS", "L2_DECISION_LOG", *preflight._TRACING_VARS):
        monkeypatch.setenv(name, "false")
    for provider in ("anthropic", "openai", "google"):
        agent_env.llm.scene(provider, "ok", text=_ANSWER_SENTINEL)
    return agent_env, tmp_path / "out", tmp_path / "spend", utterances, context


def _agent_run(env, *extra, label="agt", limit="2"):
    _agent, out_dir, spend_dir, utterances, _context = env
    return probe.main(["agent", "--profile", "dev", "--label", label, "--max-usd", "1", "--limit", limit,
                       "--gap-sec", "0", "--out-dir", str(out_dir), "--spend-dir", str(spend_dir),
                       "--utterances", str(utterances), *extra])


def _metrics(out_dir, label="agt"):
    path = out_dir / f"agent_{label}.metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _llm_of(slug):
    char = get_character(slug)
    return char.llm_provider, char.llm_model or "gpt-5.4-mini"


def _tool_names(body):
    return {t.get("function", {}).get("name") or t.get("name") for t in body["tools"]}


def test_agent_measures_steps_tools_and_cache_fields_without_text(agent_probe_env, capsys):
    agent, out_dir, spend_dir, _utterances, _context = agent_probe_env
    provider, model = _llm_of("mimi")
    agent.llm.script(model, [("tool_call", {"name": "web_search_tool", "args": {"query": "q"}})])

    assert _agent_run(agent_probe_env) == 0

    captured = capsys.readouterr()
    lines = _metrics(out_dir)
    assert len(lines) == 8
    # 発話の番号ごとに、キャラを交互に流す（時間帯の差が後のキャラに偏らないように）。
    assert [(x["character"], x["i"], x["cold"]) for x in lines][:5] == [
        ("mimi", 0, True), ("chisame", 0, True), ("sakura", 0, True), ("octamaid", 0, True), ("mimi", 1, False)]
    assert [x["first_in_process"] for x in lines] == [True] + [False] * 7
    assert [x["first_for_provider"] for x in lines][:4] == [True, True, True, False]  # openai, google, anthropic, openai
    first = lines[0]
    assert (first["error"], first["source"], first["tools_fired"]) == (None, "structured_response", ["web_search_tool"])
    assert len(first["steps"]) >= 2 and first["steps"][0]["tool_calls"] == ["web_search_tool"]
    assert [t["name"] for t in first["tools"]] == ["web_search_tool"]
    for line in lines:
        assert line["total_ms"] is not None and line["build_ms"] is not None and line["usd"] > 0
        for step in line["steps"]:
            assert step["input_total"] > 0 and step["input_uncached"] > 0 and step["output"] > 0 and step["ms"] is not None and step["model"]
            assert {"reasoning", "cache_read", "cache_creation", "finish"} <= set(step)
    assert "response" not in first
    # 道具は定型の応答で、スキーマは本物のまま: 2 回目の要求に定型の応答が載り、1 回目の要求に 3 つの道具の名前がある。
    sent = agent.llm.requests_for(provider, model)
    assert "web_search の定型の応答" in json.dumps(sent[1].body, ensure_ascii=False)
    assert {"web_search_tool", "ask_character_tool"} <= _tool_names(sent[0].body)
    # 本番と同じ user の前置きと、配信文脈（写し）が system prompt に入る。
    body_text = json.dumps(sent[0].body, ensure_ascii=False)
    assert "[ルカからの発言]" in body_text and "ミミ様" in body_text and _CONTEXT_SENTINEL in body_text
    # 画面と数字のファイルには、発話も返答も配信文脈も出ない。返答は 1 つのファイルだけ。
    assert all(token not in captured.out + captured.err for token in (_SENTINEL, _ANSWER_SENTINEL, _CONTEXT_SENTINEL))
    assert _ANSWER_SENTINEL in (out_dir / "agent_agt.jsonl").read_text(encoding="utf-8")
    for path in [*out_dir.iterdir(), *spend_dir.iterdir()]:
        if path.suffix == ".log" or path.name in ("agent_agt.jsonl", "stream_context_snapshot.md"):
            continue
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in (_SENTINEL, _ANSWER_SENTINEL, _CONTEXT_SENTINEL)), path.name
    meta = json.loads((out_dir / "agent_agt.meta.json").read_text(encoding="utf-8"))
    assert meta["stream_context"]["origin"] == "live" and meta["stream_context"]["chars"] > 0
    assert set(meta["personas"]) == set(_AGENT_SLUGS) and meta["timed_out"] is False
    assert meta["tools_fired_total"] == {"web_search": 1}
    assert set(meta["summary"]) == set(_AGENT_SLUGS)
    assert "cold=" in captured.out and "warm p50=" in captured.out and "cache_creation=" in captured.out


def test_agent_tools_are_restored_after_the_run(agent_probe_env):
    from lab_lounge.mcp_servers import web_search

    original = web_search.web_search
    assert _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "1") == 0
    assert web_search.web_search is original


def test_later_runs_reuse_the_stream_context_snapshot(agent_probe_env):
    agent, out_dir, _spend_dir, _utterances, context = agent_probe_env
    assert _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "1") == 0
    context.write_text("まったく違う配信", encoding="utf-8")
    assert _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "1", label="agt2") == 0
    first = json.loads((out_dir / "agent_agt.meta.json").read_text(encoding="utf-8"))["stream_context"]
    second = json.loads((out_dir / "agent_agt2.meta.json").read_text(encoding="utf-8"))["stream_context"]
    assert second["origin"] == "snapshot" and second["sha"] == first["sha"]
    assert "まったく違う配信" not in json.dumps([r.body for r in agent.llm.requests_for()], ensure_ascii=False)


def test_llm_override_sends_every_character_to_one_model(agent_probe_env):
    agent, out_dir, _spend_dir, _utterances, _context = agent_probe_env
    assert _agent_run(agent_probe_env, "--llm", "anthropic:claude-haiku-4-5", "--limit", "1") == 0
    assert agent.llm.requests_for("openai") == [] and agent.llm.requests_for("google") == []
    assert len(agent.llm.requests_for("anthropic")) == 4
    meta = json.loads((out_dir / "agent_agt.meta.json").read_text(encoding="utf-8"))
    assert set(meta["models"].values()) == {"anthropic:claude-haiku-4-5"}
    assert {line["model"] for line in _metrics(out_dir)} == {"claude-haiku-4-5"}


def test_failed_agent_calls_are_recorded_by_type_and_exit_1(agent_probe_env, capsys):
    agent, out_dir, _spend_dir, _utterances, _context = agent_probe_env
    agent.llm.scene("anthropic", "http_error", status=400)
    assert _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "2") == 1
    lines = _metrics(out_dir)
    assert all(line["error"] for line in lines) and all(line["error"].isidentifier() for line in lines)
    captured = capsys.readouterr()
    assert "失敗=2" in captured.out and _SENTINEL not in captured.out + captured.err


def test_timeout_is_recorded_before_exiting(agent_probe_env, monkeypatch, capsys):
    _agent, out_dir, _spend_dir, _utterances, _context = agent_probe_env
    codes = []

    def hard_exit(code):
        codes.append(code)
        raise _Exited

    def too_slow(*_args, **_kwargs):
        raise probe.AgentTimeout("90 秒")

    monkeypatch.setattr(spend, "hard_exit", hard_exit)
    monkeypatch.setattr(probe, "invoke_agent", too_slow)
    with pytest.raises(_Exited):
        _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "2")
    assert codes == [probe.EXIT_FAILED]
    (line,) = _metrics(out_dir)
    assert line["error"] == "Timeout" and line["character"] == "sakura"
    meta = json.loads((out_dir / "agent_agt.meta.json").read_text(encoding="utf-8"))
    assert meta["timed_out"] is True and meta["summary"]["sakura"]["failed"] == 1
    out = capsys.readouterr().out
    assert "時間切れ" in out and _SENTINEL not in out


def test_agent_is_refused_when_tools_are_disabled(agent_probe_env, monkeypatch, capsys):
    monkeypatch.setenv("L2_ENABLE_TOOLS", "false")
    assert _agent_run(agent_probe_env) == probe.EXIT_REFUSED
    assert "L2_ENABLE_TOOLS" in capsys.readouterr().out
    assert agent_probe_env[0].llm.requests_for() == []


def test_agent_budget_stop_sends_nothing(agent_probe_env):
    agent, out_dir, spend_dir, utterances, _context = agent_probe_env
    with pytest.raises(spend.BudgetExceeded):
        probe.main(["agent", "--profile", "dev", "--label", "tiny", "--max-usd", "0.0000001", "--limit", "1",
                    "--gap-sec", "0", "--out-dir", str(out_dir), "--spend-dir", str(spend_dir),
                    "--utterances", str(utterances)])
    assert agent.llm.requests_for() == []


# ─── 提供元ごとの拾い損ねの確かめ ─────────────────────────────────

def test_raw_stops_when_one_provider_was_never_metered(probe_env, monkeypatch):
    """google の呼び出しが成功したのに、HTTP の層が 1 件も数えていなければ、止める（全体では別の社が数えていても）。"""
    out_dir, spend_dir, synthetic, utterances = probe_env

    def fake_rounds(groups, ledger, label, *, after_round=None, **_kwargs):
        samples = [probe.Sample(groups[0].name, 0, "u01", "gemini-3.1-flash-lite", None, 10, None, 0.0, {}, "google")]
        after_round(samples)
        return {groups[0].name: samples}

    monkeypatch.setattr(probe, "measure_rounds", fake_rounds)
    with pytest.raises(spend.UsageCaptureLost):
        _run(probe_env, "--only", "filler")


def test_agent_stops_when_steps_ended_but_no_http_call_was_counted(agent_probe_env, monkeypatch):
    def fake_call(slug, utterance, index, settings, ledger, label):
        line = {"i": index, "character": slug, "utterance_id": utterance.id, "category": utterance.category,
                "cold": index == 0, "provider": "anthropic", "model": "claude-haiku-4-5", "error": None,
                "total_ms": 5, "steps": [{"ms": 3, "input_total": 1, "output": 1, "reasoning": 0, "cache_read": 0,
                                          "cache_creation": 0, "error": None, "tool_calls": [], "finish": "stop",
                                          "model": "m"}],
                "tools": [], "tools_fired": [], "source": "structured_response", "usd": 0.0, "http_calls": 0}
        return line, ""

    monkeypatch.setattr(probe, "measure_agent_call", fake_call)
    with pytest.raises(spend.UsageCaptureLost):
        _agent_run(agent_probe_env, "--characters", "sakura", "--limit", "1")
