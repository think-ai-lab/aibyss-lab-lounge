"""
replay run / report の契約テスト（本物の SDK と支出の記録を、偽の LLM サーバで通す）

``python -m lab_lounge.replay run`` の入口（main）を、環境の読み込み（prepare_env）だけ止めて流す。
本文には目印の文字列だけを使い、画面、実行の記録、支出の記録に出ないことを確かめる。
"""

from __future__ import annotations

import json

import pytest

from lab_lounge import preflight, replay, replay_set, spend

_SENTINEL = "SENTINEL-replay-contract-a41f"


def _rows() -> list[dict]:
    rows = []
    for i, (kind, character, candidates) in enumerate([
        ("approval", None, ["chisame"]), ("intent", "mimi", []), ("interjection", None, []),
        ("route", None, ["mimi", "chisame"]),
    ]):
        text = f"{_SENTINEL}-{i}"
        rows.append({"id": replay_set.row_id(kind, character, candidates, text), "kind": kind, "input": text,
                     "character": character, "candidates": candidates, "source": "runlog", "expected": None,
                     "alt_ok": [], "hist_raw": "none", "hist_key": None, "epoch": f"e#{i}", "date": "2026-06-01"})
    return rows


@pytest.fixture
def replay_env(fake_llm, tmp_path, monkeypatch):
    """セットと、環境を読まない入口を用意する。"""
    set_dir = tmp_path / "replay"
    (set_dir / "private").mkdir(parents=True)
    (set_dir / "private" / "set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows()),
                                       encoding="utf-8")
    (set_dir / "exclude.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(preflight, "prepare_env", lambda *_args, **_kwargs: [])
    for name in ("L2_DEBUG_ARTIFACTS", "L2_DECISION_LOG", *preflight._TRACING_VARS):
        monkeypatch.setenv(name, "false")
    fake_llm.scene("anthropic", "ok", text="none")
    return set_dir, tmp_path / "probe", tmp_path / "none.jsonl"


def _run(set_dir, spend_dir, synthetic, *extra):
    return replay.main(["--set-dir", str(set_dir), "--synthetic", str(synthetic), "run", "--profile", "dev",
                        "--label", "contract", "--max-usd", "1", "--spend-dir", str(spend_dir), *extra])


def test_run_records_results_and_spend_without_text(replay_env, fake_llm, capsys):
    set_dir, spend_dir, synthetic = replay_env

    assert _run(set_dir, spend_dir, synthetic) == 0

    captured = capsys.readouterr()
    assert _SENTINEL not in captured.out + captured.err
    assert "判定 8 件（エラー 0）" in captured.out
    lines = replay.RunFile(set_dir / "runs" / "contract.jsonl").lines()
    assert len(lines) == 8 and {line["model_on_wire"] for line in lines} == {"claude-haiku-4-5"}
    assert len(fake_llm.requests_for("anthropic")) == 8
    meta = json.loads((set_dir / "runs" / "contract.meta.json").read_text(encoding="utf-8"))
    assert (meta["rows"], meta["reps"], meta["model_on_wire"]) == (4, 2, {"claude-haiku-4-5": 8})
    spent = json.loads((spend_dir / "spent.json").read_text(encoding="utf-8"))
    assert spent["by_run"]["contract"]["calls"] == 8
    # 数字のファイル（private/ の外）には、本文も生の答えも無い
    for path in [*set_dir.joinpath("runs").iterdir(), *spend_dir.iterdir()]:
        assert _SENTINEL not in path.read_text(encoding="utf-8"), path.name
    assert (set_dir / "private" / "runs" / "contract.answers.jsonl").is_file()


def test_existing_run_needs_resume_and_the_same_settings(replay_env, monkeypatch, capsys):
    set_dir, spend_dir, synthetic = replay_env
    assert _run(set_dir, spend_dir, synthetic, "--reps", "1") == 0
    assert _run(set_dir, spend_dir, synthetic) == replay.EXIT_REFUSED
    monkeypatch.setenv("L2_INTENT_GATE_MODEL", "claude-sonnet-4-6")
    assert _run(set_dir, spend_dir, synthetic, "--resume") == replay.EXIT_REFUSED
    assert "再開できません" in capsys.readouterr().out


def test_report_prints_counts(replay_env, capsys):
    set_dir, spend_dir, synthetic = replay_env
    assert _run(set_dir, spend_dir, synthetic) == 0
    capsys.readouterr()
    assert replay.main(["--set-dir", str(set_dir), "--synthetic", str(synthetic), "report", "--run", "contract"]) == 0
    out = capsys.readouterr().out
    assert "approval: 1 / 1" in out and _SENTINEL not in out


def test_canary_limits_the_rows(replay_env, fake_llm):
    set_dir, spend_dir, synthetic = replay_env
    assert _run(set_dir, spend_dir, synthetic, "--canary", "2", "--reps", "1") == 0
    assert len(fake_llm.requests_for("anthropic")) == 2


def test_budget_stop_leaves_partial_results_and_spend(replay_env, fake_llm):
    set_dir, spend_dir, synthetic = replay_env
    with pytest.raises(spend.BudgetExceeded):
        _run(set_dir, spend_dir, synthetic, "--max-usd", "0.0000001")
    assert fake_llm.requests_for("anthropic") == []
