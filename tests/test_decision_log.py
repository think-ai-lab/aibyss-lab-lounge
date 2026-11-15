"""
test_decision_log.py — 判定ログ (decision_log) と、router.py の 4 か所の書き込みのテスト

書き先 logs/decisions は作業ディレクトリからの相対なので、どのテストも tmp_path へ chdir してから流す
(実リポの logs/ へは書かない)。LLM は呼ばない: _call_router_llm を差し替えるか、鍵が無い環境で
SDK が例外を出す経路を使う。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_lounge import decision_log, router
from lab_lounge.characters import get_all_characters


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    """作業ディレクトリを一時へ移し、モジュールの「掃除済み」「警告済み」の状態を毎回戻す。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(decision_log, "_cleaned_for", None)
    monkeypatch.setattr(decision_log, "_warned", False)
    return tmp_path


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("L2_DECISION_LOG", "true")


def _rows(root: Path) -> list[dict]:
    """logs/decisions の全ファイルの全行を読む。"""
    rows = []
    for path in sorted((root / "logs" / "decisions").glob("decisions_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows


def _record(**overrides) -> None:
    args = dict(
        model="m", system_prompt="sys", input_text="こんにちは", raw="mimi",
        result="mimi", elapsed_ms=12,
    )
    args.update(overrides)
    decision_log.record("route", **args)


# ─── 既定ではファイルを作らない ───────────────────────────────────


def test_default_off_creates_no_file(in_tmp):
    _record()
    assert not (in_tmp / "logs").exists()


@pytest.mark.parametrize("value", ["false", "0", "", "ture"])
def test_non_true_values_stay_off(in_tmp, monkeypatch, value):
    monkeypatch.setenv("L2_DECISION_LOG", value)
    _record()
    assert not (in_tmp / "logs").exists()


# ─── オンにすると、1 判定 1 行 ────────────────────────────────────


def test_enabled_appends_one_line_per_decision(in_tmp, enabled):
    _record(input_text="一つ目")
    _record(input_text="二つ目", result=None, raw=None, error="TimeoutError")

    first, second = _rows(in_tmp)
    assert (first["kind"], first["input"], first["raw"], first["result"]) == ("route", "一つ目", "mimi", "mimi")
    assert first["elapsed_ms"] == 12 and first["error"] is None
    assert (second["input"], second["raw"], second["result"], second["error"]) == ("二つ目", None, None, "TimeoutError")
    # 1 日 1 ファイル、名前は日付。
    (name,) = [p.name for p in (in_tmp / "logs" / "decisions").iterdir()]
    assert name == f"decisions_{date.today():%Y%m%d}.jsonl"


def test_row_keeps_full_input_but_only_hash_of_prompt(in_tmp, enabled):
    long_text = "あ" * 500
    _record(input_text=long_text, system_prompt="秘密ではないが長い system prompt")
    (row,) = _rows(in_tmp)
    assert row["input"] == long_text  # 30 文字に切らない
    assert row["system_prompt_sha"] == decision_log.prompt_hash("秘密ではないが長い system prompt")
    assert len(row["system_prompt_sha"]) == 16
    assert "system prompt" not in json.dumps(row, ensure_ascii=False)
    assert datetime.fromisoformat(row["ts"]).utcoffset() is not None


def test_dataclass_result_is_written_as_dict(in_tmp, enabled):
    _record(result=router.IntentResult(intent="callout", target_slug="mimi", confidence=1.0))
    (row,) = _rows(in_tmp)
    assert row["result"] == {"intent": "callout", "target_slug": "mimi", "confidence": 1.0}


# ─── 古いファイルを消す ───────────────────────────────────────────


def _touch_day(root: Path, days_ago: int) -> Path:
    directory = root / "logs" / "decisions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"decisions_{date.today() - timedelta(days=days_ago):%Y%m%d}.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_old_files_are_removed_on_first_write_of_the_day(in_tmp, enabled, monkeypatch):
    monkeypatch.setenv("L2_DECISION_LOG_RETENTION_DAYS", "5")
    old = _touch_day(in_tmp, 6)
    edge = _touch_day(in_tmp, 5)
    recent = _touch_day(in_tmp, 1)
    foreign = in_tmp / "logs" / "decisions" / "notes.txt"
    foreign.write_text("keep", encoding="utf-8")

    _record()

    assert not old.exists()
    assert edge.exists() and recent.exists()  # 保持日数ちょうどは残す
    assert foreign.exists()  # 名前の形に合わないものは触らない


def test_cleanup_runs_only_once_per_day(in_tmp, enabled, monkeypatch):
    monkeypatch.setenv("L2_DECISION_LOG_RETENTION_DAYS", "5")
    _record()  # 1 回目で掃除済みになる
    stale = _touch_day(in_tmp, 30)
    _record()
    assert stale.exists()


def test_default_retention_is_30_days(in_tmp, enabled):
    keep = _touch_day(in_tmp, 30)
    drop = _touch_day(in_tmp, 31)
    _record()
    assert keep.exists() and not drop.exists()


@pytest.mark.parametrize("value", ["abc", "-3", ""])
def test_bad_retention_falls_back_to_default(in_tmp, enabled, monkeypatch, value):
    """読めない値で全消しにならない。"""
    monkeypatch.setenv("L2_DECISION_LOG_RETENTION_DAYS", value)
    keep = _touch_day(in_tmp, 10)
    _record()
    assert keep.exists()


# ─── 書けなくても止めない ─────────────────────────────────────────


def test_unwritable_directory_does_not_raise_and_warns_once(in_tmp, enabled, caplog):
    (in_tmp / "logs").write_text("これはファイル。ディレクトリを作れない", encoding="utf-8")
    caplog.set_level(logging.DEBUG, logger="lab_lounge.decision_log")

    _record()
    _record()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "判定は続けます" in warnings[0].getMessage()


# ─── router.py の 4 か所 ──────────────────────────────────────────


def _answer(value):
    """_call_router_llm を、決まった応答を返す関数に差し替える。"""
    return patch.object(router, "_call_router_llm", lambda model, system, text: value)


def test_route_by_llm_logs_candidates_and_result(in_tmp, enabled):
    candidates = get_all_characters()[:3]
    slug = candidates[0].slug
    with _answer(slug):
        picked = router._route_by_llm("ミミ、聞いて", candidates)
    assert picked.slug == slug
    (row,) = _rows(in_tmp)
    assert (row["kind"], row["input"], row["raw"], row["result"]) == ("route", "ミミ、聞いて", slug, slug)
    assert row["candidates"] == [c.slug for c in candidates]
    assert row["character"] is None


def test_route_by_llm_logs_unmatched_answer_as_null_result(in_tmp, enabled):
    with _answer("someone_else"):
        assert router._route_by_llm("x", get_all_characters()[:2]) is None
    (row,) = _rows(in_tmp)
    assert (row["raw"], row["result"]) == ("someone_else", None)


def test_intent_for_character_logs_character_and_result(in_tmp, enabled):
    with _answer("callout"):
        result = router.check_intent("ミミ、教えて", "mimi")
    assert result.intent == "callout"
    (row,) = _rows(in_tmp)
    assert (row["kind"], row["character"], row["raw"]) == ("intent", "mimi", "callout")
    assert row["result"]["intent"] == "callout"


def test_interjection_logs_candidates(in_tmp, enabled):
    with _answer("none"):
        result = router.check_intent("今日は暑いね")
    assert result.intent == "unknown"
    (row,) = _rows(in_tmp)
    assert (row["kind"], row["raw"], row["character"]) == ("interjection", "none", None)
    assert row["candidates"] and "octamaid" not in row["candidates"]


def test_approval_logs_granted_and_denied_and_none(in_tmp, enabled):
    with _answer("granted:mimi"):
        assert router.check_approval("ミミ、どうぞ", ["mimi", "sakura"]).granted is True
    with _answer("none"):
        assert router.check_approval("別の話", ["mimi"]) is None
    first, second = _rows(in_tmp)
    assert first["kind"] == "approval" and first["candidates"] == ["mimi", "sakura"]
    assert first["result"] == {"granted": True, "target_slug": "mimi", "confidence": 1.0}
    assert second["result"] is None and second["raw"] == "none"


def test_call_failure_is_logged_with_exception_type(in_tmp, enabled):
    """鍵が無い環境では SDK が例外を出す。_call_router_llm は握って None を返し、型だけが判定ログに残る。"""
    result = router.check_approval("ミミ、どうぞ", ["mimi"])  # 既定のモデル。呼び出しは SDK の手前で失敗する
    assert result is None
    (row,) = _rows(in_tmp)
    assert row["raw"] is None and row["result"] is None
    assert row["error"]  # 例外の型名 (SDK / プロバイダに依らず空でない)
    assert row["elapsed_ms"] >= 0


def test_unwritable_log_does_not_change_the_decision(in_tmp, enabled):
    (in_tmp / "logs").write_text("file", encoding="utf-8")
    with _answer("denied:mimi"):
        result = router.check_approval("やめて", ["mimi"])
    assert (result.granted, result.target_slug) == (False, "mimi")


def test_off_leaves_decisions_untouched_and_writes_nothing(in_tmp):
    with _answer("callout"):
        assert router.check_intent("ミミ、教えて", "mimi").intent == "callout"
    assert not (in_tmp / "logs").exists()
