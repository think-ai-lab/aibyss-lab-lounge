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
    assert all(line["tokens"]["input"] > 0 and line["tokens"]["output"] > 0 and line["usd"] > 0 for line in lines)
    assert all(set(line["tokens"]) == {"input", "output", "reasoning", "cache_read", "cache_creation"} for line in lines)
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
