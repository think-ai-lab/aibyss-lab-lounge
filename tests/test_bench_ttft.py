"""
tests/bench_ttft.py（モデル比較の補助の道具）のテスト。API は呼ばない。

確かめること:
  - Claude のモデルには temperature を送らない（新しいモデルでは API が 400 にする。後継のモデルを足しても同じ）
  - [skip]、ウォームアップの失敗、試行のエラーのどれか 1 つでもあれば終了コード 1（黙って表が欠けたまま緑にならない）
  - 冒頭に、補助の道具で採否の根拠にしない旨がある
"""

from __future__ import annotations

import sys
import types

import pytest

import bench_ttft


@pytest.mark.parametrize("provider, model, expected", [
    ("anthropic", "claude-haiku-4-5", False),
    ("anthropic", "claude-sonnet-4-6", False),
    ("anthropic", "claude-opus-4-8", False),
    ("anthropic", "claude-some-future-model", False),  # 後継のモデルでも、表の更新漏れで落ちない
    ("openai", "gpt-5.5", False),
    ("openai", "gpt-5.4-mini", False),
    ("openai", "gpt-4.1", True),
    ("google", "gemini-3.1-pro-preview", True),
])
def test_which_models_get_temperature(provider, model, expected):
    assert bench_ttft.sends_temperature(provider, model) is expected


@pytest.mark.parametrize("provider, module, cls, model, sent", [
    ("anthropic", "langchain_anthropic", "ChatAnthropic", "claude-haiku-4-5", False),
    ("anthropic", "langchain_anthropic", "ChatAnthropic", "claude-sonnet-4-6", False),
    ("google", "langchain_google_genai", "ChatGoogleGenerativeAI", "gemini-3.5-flash", True),
])
def test_build_llm_passes_temperature_only_where_allowed(monkeypatch, provider, module, cls, model, sent):
    seen: dict = {}
    fake = types.ModuleType(module)
    setattr(fake, cls, lambda **kwargs: seen.update(kwargs))
    monkeypatch.setitem(sys.modules, module, fake)
    bench_ttft.build_llm(provider, model)
    assert ("temperature" in seen) is sent


def test_the_header_says_it_is_an_auxiliary_tool():
    assert "補助の道具" in bench_ttft.__doc__ and "採否の根拠にしない" in bench_ttft.__doc__


def _run_main(monkeypatch, tmp_path, *, build, trial):
    monkeypatch.setattr(sys, "argv", ["bench_ttft.py", "--only", "haiku", "--trials", "2", "--prompt", "greeting",
                                      "--csv", str(tmp_path / "out.csv")])
    monkeypatch.setattr(bench_ttft, "load_env", lambda _path=None: None)
    monkeypatch.setattr(bench_ttft, "check_keys", lambda _providers: None)
    monkeypatch.setattr(bench_ttft.time, "sleep", lambda _s: None)
    monkeypatch.setattr(bench_ttft, "build_llm", build)
    monkeypatch.setattr(bench_ttft, "run_trial", trial)
    return bench_ttft.main()


def _ok(llm, model, kind, prompt):
    return bench_ttft.Trial(model, kind, 100.0, 90.0, 500.0, 10)


def _failed(llm, model, kind, prompt):
    return bench_ttft.Trial(model, kind, None, None, None, 0, error="APIError()")


def test_exit_0_when_everything_works(monkeypatch, tmp_path, capsys):
    assert _run_main(monkeypatch, tmp_path, build=lambda p, m: object(), trial=_ok) == 0
    assert (tmp_path / "out.csv").is_file() and "失敗が" not in capsys.readouterr().out


def test_skip_is_a_failure(monkeypatch, tmp_path, capsys):
    def build(provider, model):
        raise RuntimeError("no client")

    assert _run_main(monkeypatch, tmp_path, build=build, trial=_ok) == 1
    out = capsys.readouterr().out
    assert "[skip]" in out and "失敗が 1 件" in out


def test_trial_errors_are_a_failure(monkeypatch, tmp_path, capsys):
    assert _run_main(monkeypatch, tmp_path, build=lambda p, m: object(), trial=_failed) == 1
    assert "失敗が 3 件" in capsys.readouterr().out  # ウォームアップ 1 + 試行 2


def test_a_failed_warmup_alone_is_a_failure(monkeypatch, tmp_path):
    def trial(llm, model, kind, prompt):
        return _failed(llm, model, kind, prompt) if kind == "warmup" else _ok(llm, model, kind, prompt)

    assert _run_main(monkeypatch, tmp_path, build=lambda p, m: object(), trial=trial) == 1


_REAL_RUN_TRIAL = bench_ttft.run_trial  # _run_main が差し替える前の本物


class _Chunk:
    def __init__(self, content):
        self.content = content


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def stream(self, _messages):
        yield from self.chunks


def test_a_trial_with_no_visible_text_is_a_failure_not_a_crash():
    """考えるだけで終わった試行（TTFT なし）は、失敗として数える。成功に数えると表の書式が落ちる。"""
    thinking_only = _Stream([_Chunk([{"type": "thinking", "thinking": "..."}]), _Chunk([])])
    trial = bench_ttft.run_trial(thinking_only, "m", "greeting", "p")
    assert trial.error == "NoVisibleText" and trial.ttft_ms is None and trial.first_chunk_ms is not None
    rows = bench_ttft.summarize([trial, bench_ttft.Trial("m", "greeting", 100.0, 90.0, 500.0, 10)])
    bench_ttft.print_table(rows)  # None を含んでも落ちない
    assert (rows[0]["n_ok"], rows[0]["n_err"]) == (1, 1)


def test_visible_text_gives_a_ttft():
    trial = bench_ttft.run_trial(_Stream([_Chunk([{"type": "thinking"}]), _Chunk("こんにちは")]), "m", "g", "p")
    assert trial.error == "" and trial.ttft_ms is not None and trial.out_chars == 5


def test_main_survives_a_no_visible_text_trial_and_still_prints_the_table(monkeypatch, tmp_path, capsys):
    def trial(llm, model, kind, prompt):
        return _REAL_RUN_TRIAL(_Stream([_Chunk([{"type": "thinking"}])]), model, kind, prompt)

    assert _run_main(monkeypatch, tmp_path, build=lambda p, m: object(), trial=trial) == 1
    out = capsys.readouterr().out
    assert "===== 集計 =====" in out and "NoVisibleText" in out


def test_the_csv_keeps_the_trials_done_before_a_crash(monkeypatch, tmp_path):
    calls = []

    def trial(llm, model, kind, prompt):
        calls.append(kind)
        if len(calls) == 3:  # ウォームアップ 1 + 試行 1 の後（2 試行目で落ちる）
            raise KeyboardInterrupt
        return _ok(llm, model, kind, prompt)

    with pytest.raises(KeyboardInterrupt):
        _run_main(monkeypatch, tmp_path, build=lambda p, m: object(), trial=trial)
    lines = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 1  # 見出し + 完了した 1 試行
