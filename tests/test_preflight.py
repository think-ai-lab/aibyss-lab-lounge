"""
test_preflight.py — 配信前の点検のテスト

点検そのものが「壊れているのに緑」を返すと、無音の障害を見逃す道具になる。そのため
None と例外が必ず FAIL になること、経路の配線と必須キャラの振り分け、秘匿値が出力に現れない
こと、本番ストリームで止まること、--static が課金しないことを固定する。
LLM はすべてモックで、実 API は呼ばない。
"""

import dataclasses
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from dotenv import dotenv_values

from lab_lounge import preflight
from lab_lounge.characters import get_all_characters, get_character

# 秘匿値のセンチネル。どの出力にも現れてはいけない。
SECRET = "SENTINEL-not-a-real-secret-0123456789"
BODY = "BODYSENTINEL これは実発話です"
# redact の形（キーや URL の資格情報）に当たらない例外文。当たる形だと、_cli が表示しても redact が隠し、
# 値を探す assert が落ちない。短い印の SENTINELpw を探す。
_TEXT_REDACT_CANNOT_HIDE = "SENTINELpw plain text of the exception"

# 値を表示しないことを確かめる短い印。切り詰められても見つかるように短くする。
_MARK = "zqx"

_REPO_ROOT = Path(__file__).resolve().parents[1]

_C2_ON = {"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"}
_REAL_TTS = {"L2_USE_REAL_TTS": "true"}
_RAW_LLM_OFF = {"L2_USE_LLM_ROUTER": "false", "L2_USE_INTENT_GATE": "false", "L2_USE_HANDRAISE": "false", "L2_FILLER_ENABLED": "false"}

_LLM_ENTRY_POINTS = (
    "lab_lounge.router._call_router_llm",
    "lab_lounge.filler._call_filler_llm",
    "lab_lounge.graph._build_agent_graph",
    "lab_lounge.llm.call_llm",
    "lab_lounge.retriever._embed",
    "lab_lounge.pipeline.run_pipeline",
    "langchain_tavily.TavilySearch",
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """このファイルのテストからは、どこへも接続させない。

    WHY: 点検は実サービスの死活を見るので、モックが漏れると起動中の実 VOICEVOX や実 Redis に届く。
    """
    def refuse(self, address):
        raise ConnectionRefusedError(f"blocked in test: {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """実 .env と実サービスから切り離した環境を作り、一時ディレクトリを repo_root として返す。

    prepare_env は os.environ へ直接書くので、丸ごと退避して元へ戻す。
    """
    with patch.dict(os.environ):
        for name in list(os.environ):
            if name.startswith(("L2_", "OPENAI", "ANTHROPIC", "GOOGLE", "GEMINI", "TAVILY", "LANGSMITH", "LANGCHAIN", "REDIS")):
                del os.environ[name]
        os.environ.pop("PYTHON_DOTENV_DISABLED", None)
        # 閉じたポート。消すと既定の Redis へ届いてしまうので、明示的に向け先を潰す。
        os.environ["REDIS_URL"] = "redis://127.0.0.1:6"
        # 実リポの current.md を読まないようにする。
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(tmp_path / "no-such-context.md")
        # voices.json、読み辞書、filler の WAV も、既定のパスだと手元の実物を見てしまう。
        os.environ["L2_IRODORI_VOICES_JSON"] = str(tmp_path / "no-such-voices.json")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(tmp_path / "no-such-readings.json")
        monkeypatch.setattr("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path / "no-such-filler-cache")
        # aibyss-workspace を併置しない clone でも、終了コードを見るテストが schema の行で変わらないようにする。
        schema = tmp_path / "minimal-schema.json"
        schema.write_text('{"type": "object"}', encoding="utf-8")
        os.environ["AIBYSS_SCHEMA_PATH"] = str(schema)
        monkeypatch.chdir(tmp_path)
        yield tmp_path


@pytest.fixture
def capture():
    return preflight._LogCapture()


@pytest.fixture
def voices(isolated):
    """irodori の voices.json を一時ディレクトリに作る（参照音声なし）。"""
    path = isolated / "voices.json"
    keys = {c.tts_voice: {"caption": "x"} for c in get_all_characters() if c.tts_provider == "irodori_vd"}
    path.write_text(json.dumps({"voices": keys}), encoding="utf-8")
    os.environ["L2_IRODORI_VOICES_JSON"] = str(path)
    return path


def _write_real_env(root: Path, **extra: str) -> None:
    """本番相当のフラグとダミーのキーを持つ .env を書く。"""
    values = {
        "REDIS_STREAM_KEY": "aibyss:events-dev",
        "L2_USE_REAL_LLM": "true",
        "L2_USE_REAL_TTS": "true",
        "L2_ENABLE_TOOLS": "true",
        "OPENAI_API_KEY": "dummy-openai",
        "ANTHROPIC_API_KEY": "dummy-anthropic",
        "GOOGLE_API_KEY": "dummy-google",
        "TAVILY_API_KEY": "dummy-tavily",
        **extra,
    }
    (root / ".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def _guard_entry_points():
    """課金する入口をすべて MagicMock に差し替える。呼ばれたかどうかは assert で確かめる。

    例外で止める形にすると、点検の except Exception に飲まれて FAIL* の 1 行になり、気づけない。
    """
    return [patch(name, MagicMock(name=name)) for name in _LLM_ENTRY_POINTS]


def _status(report: preflight.Report, name: str) -> preflight.Check:
    return next(c for c in report.checks if c.name == name)


def _log_record(name: str, level: int, msg: str, args: tuple = ()) -> logging.LogRecord:
    return logging.getLogger(name).makeRecord(name, level, __file__, 1, msg, args, None)


class _FakeApiError(Exception):
    """API の応答に由来する例外。本文に入力のエコーが入る。"""

    status_code = 400
    type = "invalid_request_error"
    param = "temperature"


class TestRawLlmChecks:
    """_call_router_llm は失敗を None に変える。点検ではそれを FAIL にする。"""

    @pytest.mark.parametrize("value", [None, "", "   \n"])
    def test_none_and_blank_are_fail(self, isolated, capture, value):
        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", lambda: value)
        assert _status(report, "router").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_exception_is_fail(self, isolated, capture):
        def boom():
            raise TypeError("unexpected keyword argument 'temperature'")

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", boom)
        check = _status(report, "router")
        assert check.status == "FAIL"
        assert "temperature" in check.detail

    def test_text_is_pass(self, isolated, capture):
        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", lambda: "mimi")
        assert _status(report, "router").status == "PASS"
        assert report.exit_code() == preflight.EXIT_OK

    def test_none_reports_the_logged_reason(self, isolated, capture):
        """None を返す関数の失敗理由はログにしかないので、FAIL の説明に拾う。"""
        def fail_quietly():
            capture.emit(_log_record("lab_lounge.router", logging.ERROR, "LLM ルーター呼び出し失敗: %s", ("status=400",)))
            return None

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", fail_quietly)
        assert "status=400" in _status(report, "router").detail

    def test_router_uses_the_real_entry_point_and_skips_when_disabled(self, isolated, capture):
        os.environ["L2_USE_INTENT_GATE"] = "true"
        with patch("lab_lounge.router._call_router_llm", return_value=None) as call:
            report = preflight.Report()
            preflight.check_router(report, capture)
        assert _status(report, "router").status == "SKIP"
        assert _status(report, "intent").status == "FAIL"
        call.assert_called_once()

    @pytest.mark.parametrize("answer,status", [("mimi", "PASS"), ("none", "PASS"), ("mi", "FAIL"), ("申し訳ありません", "FAIL")])
    def test_router_rejects_answers_production_would_discard(self, isolated, capture, answer, status):
        """候補にない応答は本番の _route_by_llm で捨てられるので、非空でも FAIL にする。"""
        os.environ["L2_USE_LLM_ROUTER"] = "true"
        os.environ["L2_USE_HANDRAISE"] = "false"
        with patch("lab_lounge.router._call_router_llm", return_value=answer):
            report = preflight.Report()
            preflight.check_router(report, capture)
        assert _status(report, "router").status == status

    def test_intent_runs_for_handraise_even_when_the_gate_is_off(self, isolated, capture):
        """挙手と承認はゲートを見ずに同じモデルを呼ぶので、既定の設定でも点検する。"""
        with patch("lab_lounge.router._call_router_llm", return_value=None) as call:
            report = preflight.Report()
            preflight.check_router(report, capture)
        assert _status(report, "intent").status == "FAIL"
        call.assert_called_once()

    def test_intent_is_skipped_only_when_both_are_off(self, isolated, capture):
        os.environ["L2_USE_HANDRAISE"] = "false"
        with patch("lab_lounge.router._call_router_llm") as call:
            report = preflight.Report()
            preflight.check_router(report, capture)
        assert _status(report, "intent").status == "SKIP"
        call.assert_not_called()

    def test_non_required_failure_does_not_change_exit_code(self, isolated, capture):
        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "filler/ruka", "m", lambda: None, required=False)
        assert _status(report, "filler/ruka").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK


class TestFillerCheck:
    def test_calls_the_real_entry_point_with_the_real_prompt(self, isolated, capture):
        character = get_character("octamaid")
        with patch("lab_lounge.filler._call_filler_llm", return_value="つぶやき") as call, \
                patch("lab_lounge.filler._build_filler_prompt", return_value="PROMPT") as build:
            report = preflight.Report()
            preflight.check_filler(report, capture, character, required=True)
        build.assert_called_once_with("octamaid")
        assert call.call_args.args[0] == preflight.effective_filler_model(character)
        assert call.call_args.args[1] == "PROMPT"
        assert _status(report, "filler/octamaid").status == "PASS"

    def test_none_is_fail(self, isolated, capture):
        with patch("lab_lounge.filler._call_filler_llm", return_value=None):
            report = preflight.Report()
            preflight.check_filler(report, capture, get_character("octamaid"), required=True)
        assert _status(report, "filler/octamaid").status == "FAIL"

    def test_disabled_is_skip_and_does_not_call(self, isolated, capture):
        os.environ["L2_FILLER_ENABLED"] = "false"
        with patch("lab_lounge.filler._call_filler_llm") as call:
            report = preflight.Report()
            preflight.check_filler(report, capture, get_character("octamaid"), required=True)
        assert _status(report, "filler/octamaid").status == "SKIP"
        call.assert_not_called()

    @pytest.mark.parametrize("text,status", [
        ('{"response": "深海の灯りが', "FAIL"),  # max_tokens で切れた JSON
        ('```json\n{"response": "深海の', "FAIL"),
        ('{"pose": "happy"}', "FAIL"),
        ('{"response": "   "}', "FAIL"),
        ('{"response": "こんにちは"}', "PASS"),
        ('```json\n{"response": "こんにちは"}\n```', "PASS"),  # 本番の parser も読める
        ("つぶやき", "PASS"),
        ("{不思議}ですわね", "PASS"),  # JSON ではないプレーンな文
    ])
    def test_only_broken_json_is_fail(self, isolated, capture, text, status):
        """本番の _parse_voicepeak_json が読める応答を拒否しない。切れた JSON だけを赤くする。"""
        character = SimpleNamespace(slug="x", filler_model="m")
        with patch("lab_lounge.filler._call_filler_llm", return_value=text), \
                patch("lab_lounge.filler._build_filler_prompt", return_value="PROMPT"):
            report = preflight.Report()
            preflight.check_filler(report, capture, character, required=True)
        assert _status(report, "filler/x").status == status


def _agent_result(structured=None, finish="stop", tool_calls=(), steps=1, finishes=None, meta_key="finish_reason"):
    """agent.invoke の戻り値を作る。finishes を渡すと段ごとに終わり方を変えられる。"""
    messages = [
        SimpleNamespace(
            type="ai",
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
            response_metadata={meta_key: value},
            tool_calls=list(tool_calls),
        )
        for value in (finishes or [finish] * steps)
    ]
    return {"structured_response": structured, "messages": messages}


class TestAgentCheck:
    """_run_agent のフォールバックに隠されないことを確かめる。"""

    def _run(self, capture, *, agent=None, invoke_side_effect=None, result=None, build_side_effect=None):
        character = get_character("mimi")
        invoke = MagicMock(return_value=result, side_effect=invoke_side_effect)
        build = MagicMock(return_value=agent, side_effect=build_side_effect)
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", build), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", invoke):
            preflight.check_agent(report, capture, character, None, required=True)
        return _status(report, "main/mimi"), report, invoke

    def test_no_tools_is_fail(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _, _ = self._run(capture, agent=None)
        assert check.status == "FAIL"

    def test_build_exception_is_fail(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _, _ = self._run(capture, build_side_effect=ImportError("langchain"))
        assert check.status == "FAIL"

    def test_invoke_exception_is_fail(self, isolated, capture):
        """強制 tool_choice の 400 はここに来る。本番ならツールなしへ無音で落ちる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _, _ = self._run(capture, agent=object(), invoke_side_effect=_FakeApiError(BODY))
        assert check.status == "FAIL"
        assert "status=400" in check.detail

    def test_without_structured_response_is_fail(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _, _ = self._run(capture, agent=object(), result=_agent_result(structured=None))
        assert check.status == "FAIL"
        assert "source=messages" in check.detail

    @pytest.mark.parametrize("response", ["", "   "])
    def test_blank_response_is_fail(self, isolated, capture, response):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _, _ = self._run(capture, agent=object(), result=_agent_result(SimpleNamespace(response=response)))
        assert check.status == "FAIL"

    @pytest.mark.parametrize("finish", ["length", "max_tokens", "MAX_TOKENS"])
    def test_truncated_output_is_fail(self, isolated, capture, finish):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        result = _agent_result(SimpleNamespace(response="こんにちは"), finish=finish)
        check, _, _ = self._run(capture, agent=object(), result=result)
        assert check.status == "FAIL"

    def test_structured_response_is_pass_and_hides_the_body(self, isolated, capture, capsys):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, report, _ = self._run(capture, agent=object(), result=_agent_result(SimpleNamespace(response=BODY)))
        assert check.status == "PASS"
        assert "source=structured_response" in check.detail
        assert report.input_tokens == 10
        assert "BODYSENTINEL" not in capsys.readouterr().out

    def test_invoke_has_a_recursion_limit(self, isolated, capture):
        """上限がないと、構造化出力の検証を通らない応答で再試行が止まらず、課金が続く。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _, _, invoke = self._run(capture, agent=object(), result=_agent_result(SimpleNamespace(response="x")))
        assert invoke.call_args.args[2]["recursion_limit"] == preflight._AGENT_RECURSION_LIMIT

    def test_user_text_has_the_production_tag(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _, _, invoke = self._run(capture, agent=object(), result=_agent_result(SimpleNamespace(response="x")))
        content = invoke.call_args.args[1]["messages"][0]["content"]
        assert content.startswith("[ルカからの発言]\n")

    def test_returns_whether_it_passed(self, isolated, capture):
        """run_paid はこれを見て、同じ Agent を通すパイプラインを省く。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        character = get_character("mimi")
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(SimpleNamespace(response="x"))):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is True
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(None)):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is False
        with patch("lab_lounge.graph._build_agent_graph", return_value=None):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is False
        with patch("lab_lounge.graph._build_agent_graph", side_effect=RuntimeError("x")):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is False
        os.environ["L2_ENABLE_TOOLS"] = "false"
        fake = SimpleNamespace(text="こんにちは", input_tokens=1, output_tokens=1, finish_reason="stop")
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is True
        with patch("lab_lounge.llm.call_llm", return_value=SimpleNamespace(**{**vars(fake), "text": ""})):
            assert preflight.check_agent(preflight.Report(), capture, character, None, required=True) is False

    def test_lost_tool_is_fail(self, isolated, capture):
        """ask_character だけが消えても agent は作れるので、ログで捕まえる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"

        def build(*args, **kwargs):
            capture.emit(_log_record("lab_lounge.graph", logging.WARNING, "ask_character ツール読み込み失敗 %s: %s", ("[x]", "boom")))
            return object()

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", build), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(SimpleNamespace(response="x"))):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == "FAIL"

    def test_structured_output_tool_is_not_counted_as_a_tool(self, isolated):
        result = _agent_result(
            SimpleNamespace(response="x"),
            tool_calls=[{"name": "MimiResponse"}, {"name": "web_search_tool"}],
        )
        assert preflight._summarize_agent_result(result)["tools"] == ["web_search_tool"]

    @pytest.mark.parametrize("text,finish,status", [
        ("[{'type': 'thinking', 'thinking': '...'}]", "stop", "FAIL"),  # str(response.content) の repr
        ("[]", "stop", "FAIL"),
        ("['y']", "stop", "FAIL"),
        ("   ", "stop", "FAIL"),
        ("", "stop", "FAIL"),
        ("こんにちは", "length", "FAIL"),
        ("[笑] こんにちは", "stop", "PASS"),
        ("こんにちは", "stop", "PASS"),
    ])
    def test_single_node_path(self, isolated, capture, text, finish, status):
        fake = SimpleNamespace(text=text, input_tokens=1, output_tokens=1, finish_reason=finish)
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == status


class TestServiceChecks:
    def _fake_tavily_response(self, status_code, body):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = body
        response.reason = "reason"
        return response

    @pytest.mark.parametrize("status_code,body,expected", [
        (401, {"detail": {"error": "Unauthorized"}}, "FAIL"),  # キー切れ
        (432, {"detail": {"error": "plan limit"}}, "FAIL"),  # クレジット切れ
        (500, {"detail": {"error": "boom"}}, "FAIL"),
        (200, {"results": []}, "FAIL"),
        (200, {"results": [{"title": "t", "url": "u", "content": "c"}]}, "PASS"),
    ])
    def test_tavily_judges_the_result_with_the_real_library(self, isolated, capture, status_code, body, expected):
        """langchain_tavily は HTTP エラーを例外にしないので、文字列の非空で判定すると PASS になる。"""
        pytest.importorskip("langchain_tavily")
        os.environ["L2_ENABLE_TOOLS"] = "true"
        os.environ["TAVILY_API_KEY"] = "dummy"
        report = preflight.Report()
        with patch("requests.post", return_value=self._fake_tavily_response(status_code, body)):
            preflight.check_tavily(report, capture)
        assert _status(report, "tavily").status == expected

    def test_tavily_connection_error_is_fail(self, isolated, capture):
        pytest.importorskip("langchain_tavily")
        import requests

        os.environ["L2_ENABLE_TOOLS"] = "true"
        os.environ["TAVILY_API_KEY"] = "dummy"
        report = preflight.Report()
        with patch("requests.post", side_effect=requests.ConnectionError("down")):
            preflight.check_tavily(report, capture)
        assert _status(report, "tavily").status == "FAIL"

    def test_tavily_skips_without_tools(self, isolated, capture):
        with patch("langchain_tavily.TavilySearch") as search:
            report = preflight.Report()
            preflight.check_tavily(report, capture)
        assert _status(report, "tavily").status == "SKIP"
        search.assert_not_called()

    def test_embeddings(self, isolated, capture):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        with patch("lab_lounge.retriever._embed", return_value=[0.1, 0.2]):
            report = preflight.Report()
            preflight.check_embeddings(report, capture)
        assert _status(report, "embeddings").status == "PASS"
        with patch("lab_lounge.retriever._embed", side_effect=RuntimeError("no key")):
            report = preflight.Report()
            preflight.check_embeddings(report, capture)
        assert _status(report, "embeddings").status == "FAIL"
        with patch("lab_lounge.retriever._embed", return_value=[]):
            report = preflight.Report()
            preflight.check_embeddings(report, capture)
        assert _status(report, "embeddings").status == "FAIL"

    def test_embeddings_skip_without_rag(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        with patch("lab_lounge.retriever._embed") as embed:
            report = preflight.Report()
            preflight.check_embeddings(report, capture)
        assert _status(report, "embeddings").status == "SKIP"
        embed.assert_not_called()

    def test_tts_engine_row_reflects_the_engine(self, isolated):
        character = get_character("octamaid")
        with patch.object(preflight, "_http_ok", return_value=False):
            report = preflight.Report()
            assert preflight.check_tts_engine(report, character, required=True) is False
        assert _status(report, "tts/octamaid").status == "FAIL"
        with patch.object(preflight, "_http_ok", return_value=True):
            report = preflight.Report()
            assert preflight.check_tts_engine(report, character, required=True) is True
        assert _status(report, "tts/octamaid").status == "PASS"

    def test_tts_engine_status_uses_the_right_endpoint(self, isolated):
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:15021/"
        os.environ["L2_TTS_IRODORI_URL"] = "http://127.0.0.1:18080"
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            assert preflight.tts_engine_status(SimpleNamespace(tts_provider="voicevox")) == (True, "voicevox")
            assert preflight.tts_engine_status(SimpleNamespace(tts_provider="irodori_vd")) == (True, "irodori")
        assert [c.args[0] for c in ok.call_args_list] == [
            "http://127.0.0.1:15021/version", "http://127.0.0.1:18080/health",
        ]

    def test_voicepeak_status_uses_the_configured_path(self, isolated):
        os.environ["L2_TTS_VOICEPEAK_PATH"] = str(isolated / "no-such-voicepeak.exe")
        ok, engine = preflight.tts_engine_status(SimpleNamespace(tts_provider="voicepeak"))
        assert (ok, engine) == (False, "voicepeak")

    def test_c2_like_mode_is_a_warning(self, isolated):
        """意味検索が語句一致に退化していても応答は続くので、WARN で知らせる。"""
        response = MagicMock()
        response.read.return_value = b'{"search_mode": "like"}'
        response.__enter__.return_value = response
        report = preflight.Report()
        with patch("urllib.request.urlopen", return_value=response):
            preflight._check_c2_endpoint(report, "c2", "http://127.0.0.1:8100", required=True)
        assert _status(report, "c2").status == "WARN"

    def test_c2_semantic_is_pass(self, isolated):
        response = MagicMock()
        response.read.return_value = b'{"search_mode": "semantic"}'
        response.__enter__.return_value = response
        report = preflight.Report()
        with patch("urllib.request.urlopen", return_value=response):
            preflight._check_c2_endpoint(report, "c2", "http://127.0.0.1:8100", required=True)
        assert _status(report, "c2").status == "PASS"

    def test_c2_unreachable_is_fail_only_when_required(self, isolated):
        report = preflight.Report()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            preflight._check_c2_endpoint(report, "c2", "http://127.0.0.1:8100", required=True)
            preflight._check_c2_endpoint(report, "c2/readonly", "http://127.0.0.1:8100", required=False)
        assert _status(report, "c2").status == "FAIL"
        assert _status(report, "c2/readonly").status == "WARN"

    def test_c2_uses_the_same_rule_as_graph(self, isolated):
        """graph.py は "true" だけを真と読む。1 では本番が使わない C2 を測らない。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "1"})
        report = preflight.Report()
        preflight.check_c2(report)
        assert _status(report, "c2").status == "SKIP"
        assert preflight._c2_retriever_enabled() is False

    def test_missing_services(self, isolated):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"})
        with patch.object(preflight, "_http_ok", return_value=False):
            missing = preflight.missing_services(get_all_characters(), with_pipeline=True)
        assert set(missing) == {"C2", "Redis", "irodori", "voicevox"}

    def test_missing_services_ignores_tts_of_non_required(self, isolated):
        with patch.object(preflight, "tts_engine_status", return_value=(False, "voicepeak")), \
                patch("redis.Redis.from_url"):
            assert preflight.missing_services([get_character("ruka")], with_pipeline=True) == []

    def test_missing_services_without_pipeline_needs_no_redis(self, isolated):
        with patch("redis.Redis.from_url") as from_url:
            assert preflight.missing_services(get_all_characters(), with_pipeline=False) == []
        from_url.assert_not_called()

    def test_missing_services_survives_a_malformed_redis_url(self, isolated):
        os.environ["REDIS_URL"] = "127.0.0.1:6379"
        assert "Redis" in preflight.missing_services([], with_pipeline=True)


def _pipeline_result(speaker="mimi", events=None):
    if events is None:
        events = [{"type": "tts.done", "payload": {"text": '{"response": "こんにちは", "speed": 100, "pose": "neutral"}'}}]
    return SimpleNamespace(speaker=speaker, events=events)


class TestPipelineCheck:
    """本番と同じ JSON の形で、1 条件ずつ崩して確かめる。"""

    def _run(self, capture, result, logs=("Agent 実行完了 source=structured_response",), level=logging.INFO, side_effect=None):
        def fake_pipeline(*args, **kwargs):
            for message in logs:
                capture.emit(_log_record("lab_lounge.graph", level, message.replace("%", "%%")))
            if side_effect:
                raise side_effect
            return result

        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", fake_pipeline):
            preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
        return _status(report, "pipeline/mimi"), report

    def test_clean_turn_is_pass(self, isolated, capture):
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _ = self._run(capture, _pipeline_result())
        assert check.status == "PASS"

    def test_exception_is_fail(self, isolated, capture):
        check, _ = self._run(capture, None, side_effect=RuntimeError("irodori down"))
        assert check.status == "FAIL"

    def test_other_speaker_is_fail(self, isolated, capture):
        check, _ = self._run(capture, _pipeline_result(speaker="octamaid"))
        assert check.status == "FAIL"

    @pytest.mark.parametrize("message,level,fragment", [
        ("Agent 実行失敗 [character=mimi]: 400", logging.ERROR, "フォールバック"),
        ("ask_character 協働先 Agent 失敗: 404", logging.ERROR, "協働先の Agent"),
        # 本番は WARNING で出すので、ログのレベルだけでは FAIL にならない。文言で捕まえる必要がある。
        ("ask_character ツール読み込み失敗 [x]: boom", logging.WARNING, "ツールの読み込み"),
    ])
    def test_fallback_logs_are_fail_even_when_voice_was_produced(self, isolated, capture, message, level, fragment):
        """声が出ていても、途中で握りつぶしていたら FAIL にする。行だけで理由が読めるようにする。"""
        check, _ = self._run(capture, _pipeline_result(), logs=[message, "source=structured_response"], level=level)
        assert check.status == "FAIL"
        assert fragment in check.detail

    def test_missing_structured_response_is_fail(self, isolated, capture):
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_ENABLE_TOOLS"] = "true"
        check, _ = self._run(capture, _pipeline_result(), logs=["Agent 実行完了 source=messages"])
        assert check.status == "FAIL"

    def test_missing_tts_done_is_fail(self, isolated, capture):
        check, _ = self._run(capture, _pipeline_result(events=[{"type": "llm.final", "payload": {}}]))
        assert check.status == "FAIL"

    @pytest.mark.parametrize("text", [
        '{"response": "エラー: 404 model not found", "speed": 100, "pose": "neutral"}',  # 本番の Agent の形
        "エラー: 404",  # 単一ノードの形
        '{"response": "   ", "speed": 100, "pose": "neutral"}',
    ])
    def test_error_or_blank_voice_is_fail(self, isolated, capture, text):
        events = [{"type": "tts.done", "payload": {"text": text}}]
        check, _ = self._run(capture, _pipeline_result(events=events))
        assert check.status == "FAIL"

    def test_tokens_from_llm_final_are_counted(self, isolated, capture):
        events = [
            {"type": "llm.final", "payload": {"input_tokens": 7, "output_tokens": 3}},
            *_pipeline_result().events,
        ]
        _, report = self._run(capture, _pipeline_result(events=events))
        assert (report.input_tokens, report.output_tokens) == (7, 3)

    def test_ruka_is_skipped_because_routing_excludes_it(self, isolated, capture):
        with patch("lab_lounge.pipeline.run_pipeline") as run:
            report = preflight.Report()
            preflight.check_pipeline(report, capture, get_character("ruka"), "preflight-x", None, required=False)
        assert _status(report, "pipeline/ruka").status == "SKIP"
        run.assert_not_called()

    def test_does_not_touch_obs_and_routes_by_hint(self, isolated, capture):
        """on_pose_ready が None だと _tts_node が OBS の立ち絵を即時に切り替える。"""
        with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result()) as run:
            report = preflight.Report()
            preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
        kwargs = run.call_args.kwargs
        assert callable(kwargs["on_pose_ready"])
        assert kwargs["speaker_hint"] == "mimi"
        assert kwargs["stream_id"] == "preflight-x"


_COLLAB_OK_LOGS = (
    "導入セリフ生成完了: [sakura] こんにちは",
    "Agent 実行完了 [character=chisame]: latency_ms=1 source=messages",
)


class TestCollabCheck:
    """協働は失敗を固定文やツールなしの呼び出しに変えるので、通ったことを示すログで判定する。"""

    def test_clean_logs_have_no_problem(self):
        assert preflight._collab_problems(list(_COLLAB_OK_LOGS), "chisame") == []

    @pytest.mark.parametrize("intro,is_repr", [
        ("[{'type': 'thinking'}]", True),
        ("['y']", True),
        ("[]", True),
        ("[笑] ちさめ、聞かせて", False),  # ふつうの文を赤くしない
        ("ちさめ、聞かせて。[笑]", False),
    ])
    def test_intro_repr_is_told_from_a_normal_sentence(self, intro, is_repr):
        logs = [f"導入セリフ生成完了: [sakura] {intro}", _COLLAB_OK_LOGS[1]]
        problems = preflight._collab_problems(logs, "chisame")
        assert any("repr" in problem for problem in problems) is is_repr

    @pytest.mark.parametrize("chunks,fragment", [
        ([("chisame", "こんにちは"), ("sakura", '{"response": "チサメ、')], "JSON の断片"),  # 切れた導入
        ([("chisame", '```json\n{"response": "深海の')], "JSON の断片"),  # 切れた協働先
        ([("chisame", "[]")], "JSON の断片"),
        ([("chisame", "   ")], "空"),
        ([("chisame", "エラー: Error code: 404")], "エラー文"),
        ([("sakura", "導入だけ")], "1 つも届きません"),
        ([], "1 つも届きません"),
    ])
    def test_spoken_chunks_are_inspected(self, chunks, fragment):
        """_run_agent は finish_reason を stop に固定するので、切れた応答は中身を見ないと分からない。"""
        assert any(fragment in problem for problem in preflight._chunk_problems(chunks, "chisame"))

    def test_clean_chunks_have_no_problem(self):
        assert preflight._chunk_problems([("sakura", "ちさめ、聞かせて"), ("chisame", "こんにちは")], "chisame") == []

    @pytest.mark.parametrize("logs,fragment", [
        ([*_COLLAB_OK_LOGS, "ask_character 協働先 Agent 失敗: 404"], "協働先の Agent"),
        (["導入セリフ LLM 生成失敗: boom", _COLLAB_OK_LOGS[1]], "導入セリフの生成"),
        ([_COLLAB_OK_LOGS[1]], "固定文"),  # 生成成功のログがない
        (["導入セリフ生成完了: [sakura] [{'type': 'thinking'}]", _COLLAB_OK_LOGS[1]], "repr"),
        ([_COLLAB_OK_LOGS[0]], "Agent を通っていません"),  # create_react_agent の消失
        ([*_COLLAB_OK_LOGS, "Agent 実行失敗 [character=chisame]: 400"], "フォールバック"),
        ([*_COLLAB_OK_LOGS, "導入セリフ TTS 失敗: boom"], "TTS"),
        ([*_COLLAB_OK_LOGS, "ask_character 協働応答 TTS 合成失敗 (chisame): boom"], "TTS"),
    ])
    def test_each_silent_failure_is_named(self, logs, fragment):
        problems = preflight._collab_problems(logs, "chisame")
        assert any(fragment in problem for problem in problems)

    def _run(self, capture, *, tools=("ask_character_tool",), logs=_COLLAB_OK_LOGS, structured=True, chunks=True, invoke_error=None, agent=object()):
        from lab_lounge.mcp_servers import ask_character as ac

        def fake_invoke(*args, **kwargs):
            if invoke_error:
                raise invoke_error
            # 本物の contextvar に入ったコールバックを呼んで、再生キューへの投入を模す。
            on_chunk = ac._on_tts_chunk_var.get()
            if on_chunk and chunks:
                on_chunk("file:///bridge.wav", "", False, "chisame")
                on_chunk("file:///a.wav", "こんにちは", True, "chisame")
            for message in logs:
                capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
            return _agent_result(
                SimpleNamespace(response="まとめ") if structured else None,
                tool_calls=[{"name": name} for name in tools],
                steps=2,
            )

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=agent), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke), \
                patch("lab_lounge.mcp_servers.ask_character._ask_character_impl", return_value="応答") as direct:
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "preflight-x", None)
        return _status(report, "collab/sakura->chisame"), direct

    def test_pass(self, isolated, capture):
        check, direct = self._run(capture)
        assert check.status == "PASS"
        direct.assert_not_called()

    def test_no_voice_from_target_is_fail(self, isolated, capture):
        check, _ = self._run(capture, chunks=False)
        assert check.status == "FAIL"

    def test_bridge_filler_alone_does_not_count_as_voice(self, isolated, capture):
        from lab_lounge.mcp_servers import ask_character as ac

        def fake_invoke(*args, **kwargs):
            ac._on_tts_chunk_var.get()("file:///bridge.wav", "", False, "chisame")
            for message in _COLLAB_OK_LOGS:
                capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
            return _agent_result(SimpleNamespace(response="x"), tool_calls=[{"name": "ask_character_tool"}])

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke):
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        assert _status(report, "collab/sakura->chisame").status == "FAIL"

    def test_model_not_calling_the_tool_falls_back_to_a_direct_call(self, isolated, capture):
        """モデルの判断には揺れがあるので、協働の経路だけは直接呼んで確かめ、WARN で知らせる。"""
        check, direct = self._run(capture, tools=())
        assert check.status == "WARN"
        direct.assert_called_once()

    def test_silent_collab_failure_is_fail(self, isolated, capture):
        check, _ = self._run(capture, logs=(_COLLAB_OK_LOGS[0],))
        assert check.status == "FAIL"

    def test_second_step_without_structured_response_is_fail(self, isolated, capture):
        check, _ = self._run(capture, structured=False)
        assert check.status == "FAIL"

    def test_invoke_exception_is_fail(self, isolated, capture):
        check, _ = self._run(capture, invoke_error=_FakeApiError(BODY))
        assert check.status == "FAIL"
        assert "BODYSENTINEL" not in check.detail

    def test_no_tools_is_fail(self, isolated, capture):
        check, _ = self._run(capture, agent=None)
        assert check.status == "FAIL"

    def test_context_is_reset_afterwards(self, isolated, capture):
        from lab_lounge.mcp_servers import ask_character as ac

        self._run(capture)
        assert ac._caller_slug_var.get() == ""
        assert ac._on_tts_chunk_var.get() is None

    def test_pairs(self):
        everyone = get_all_characters()
        assert [(a.slug, b.slug) for a, b in preflight._collab_pairs(everyone, "one")] == [("sakura", "chisame")]
        assert len(preflight._collab_pairs(everyone, "all")) == 3
        assert preflight._collab_pairs([get_character("mimi")], "all") == []


class TestRunPaidWiring:
    """配線が外れると、点検が何も流さず exit 0 を返す。"""

    def _patches(self):
        names = ("check_router", "check_filler", "check_tavily", "check_embeddings", "check_c2", "check_agent", "check_pipeline", "check_collab")
        return {name: patch.object(preflight, name) for name in names}

    def _run(self, capture, **kwargs):
        patches = self._patches()
        mocks = {name: p.start() for name, p in patches.items()}
        try:
            report = preflight.Report()
            with patch.object(preflight, "check_tts_engine", return_value=kwargs.pop("engine_ok", True)) as engine:
                mocks["check_tts_engine"] = engine
                preflight.run_paid(report, capture, kwargs.pop("characters", get_all_characters()), **kwargs)
        finally:
            for p in patches.values():
                p.stop()
        return mocks, report

    def test_failed_agent_skips_its_pipeline(self, isolated, capture):
        """壊れた応答で流すと、_run_agent の recursion_limit (既定 25) まで API を呼んでから落ちるので、課金が増える。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        patches = self._patches()
        mocks = {name: p.start() for name, p in patches.items()}
        mocks["check_agent"].side_effect = lambda report, capture, character, ctx, *, required: character.slug != "mimi"
        report = preflight.Report()
        try:
            with patch.object(preflight, "check_tts_engine", return_value=True):
                preflight.run_paid(report, capture, get_all_characters(), with_pipeline=True)
        finally:
            for p in patches.values():
                p.stop()
        assert "mimi" not in {call.args[2].slug for call in mocks["check_pipeline"].call_args_list}
        assert _status(report, "pipeline/mimi").status == "SKIP"
        assert _status(report, "pipeline/mimi").required is True

    def test_collab_is_skipped_when_tools_are_off(self, isolated, capture):
        """ツールが無効なら本番は ask_character を使わない。本番にない経路で課金しない。"""
        mocks, report = self._run(capture, with_pipeline=True)
        mocks["check_collab"].assert_not_called()
        assert _status(report, "collab").status == "SKIP"

    def test_every_path_runs_and_required_is_split(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        mocks, _ = self._run(capture, with_pipeline=True)
        for name in ("check_router", "check_tavily", "check_embeddings", "check_c2"):
            mocks[name].assert_called_once()
        required = {call.args[2].slug: call.kwargs["required"] for call in mocks["check_agent"].call_args_list}
        assert required == {
            "mimi": True, "chisame": True, "sakura": True, "octamaid": True, "ruka": False, "aruka": False,
        }
        assert {call.args[2].slug: call.kwargs["required"] for call in mocks["check_filler"].call_args_list} == required
        assert {call.args[2].slug: call.kwargs["required"] for call in mocks["check_pipeline"].call_args_list} == required
        mocks["check_collab"].assert_called_once()

    def test_no_pipeline_skips_pipeline_and_collab(self, isolated, capture):
        mocks, report = self._run(capture, with_pipeline=False)
        mocks["check_pipeline"].assert_not_called()
        mocks["check_collab"].assert_not_called()
        assert _status(report, "collab").status == "SKIP"

    def test_collab_modes(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        mocks, _ = self._run(capture, with_pipeline=True, collab="all")
        assert mocks["check_collab"].call_count == 3
        mocks, report = self._run(capture, with_pipeline=True, collab="none")
        mocks["check_collab"].assert_not_called()
        assert _status(report, "collab").status == "WARN"

    def test_pipeline_is_skipped_only_for_non_required_with_a_dead_engine(self, isolated, capture):
        """別枠のキャラの TTS が落ちていると分かっているなら課金しない。必須キャラは流して赤くする。"""
        mocks, report = self._run(capture, with_pipeline=True, engine_ok=False)
        slugs = {call.args[2].slug for call in mocks["check_pipeline"].call_args_list}
        assert slugs == {"mimi", "chisame", "sakura", "octamaid"}
        assert _status(report, "pipeline/aruka").status == "SKIP"

    def test_stream_id_is_marked_as_preflight(self, isolated, capture):
        mocks, _ = self._run(capture, with_pipeline=True)
        stream_id = mocks["check_pipeline"].call_args_list[0].args[3]
        assert re.fullmatch(r"preflight-\d{8}-\d{6}", stream_id)


class TestMainPaidPath:
    """--paid を最後まで通して、終了コードまで確かめる。"""

    def _main(self, root, argv=("--paid", "--profile", "dev"), failing=None):
        def fake_agent(report, capture, character, ctx, *, required):
            status = "FAIL" if character.slug == failing else "PASS"
            report.add(f"main/{character.slug}", status, required=required)
            return status == "PASS"

        with patch.object(preflight, "missing_services", return_value=[]), \
                patch.object(preflight, "check_router"), patch.object(preflight, "check_filler"), \
                patch.object(preflight, "check_tavily"), patch.object(preflight, "check_embeddings"), \
                patch.object(preflight, "check_c2"), patch.object(preflight, "check_pipeline"), \
                patch.object(preflight, "check_collab"), patch.object(preflight, "check_tts_engine", return_value=True), \
                patch.object(preflight, "check_agent", side_effect=fake_agent) as agent:
            code = preflight.main(list(argv), repo_root=root)
        return code, agent

    def test_all_pass_is_zero_and_really_runs(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        code, agent = self._main(isolated)
        assert code == preflight.EXIT_OK
        assert agent.call_count == 6

    def test_required_failure_is_one(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        code, _ = self._main(isolated, failing="sakura")
        assert code == preflight.EXIT_FAILED

    def test_non_required_failure_is_zero(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        code, _ = self._main(isolated, failing="aruka")
        assert code == preflight.EXIT_OK

    def test_characters_filter(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        _, agent = self._main(isolated, argv=("--paid", "--profile", "dev", "--characters", "mimi,sakura"))
        assert [call.args[2].slug for call in agent.call_args_list] == ["mimi", "sakura"]

    def test_no_pipeline_warns_it_cannot_pass_the_gate(self, isolated, voices, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        code, _ = self._main(isolated, argv=("--paid", "--profile", "dev", "--no-pipeline"))
        assert code == preflight.EXIT_OK
        assert "配信前の合格には使えません" in capsys.readouterr().out

    def test_paid_silences_tracing_and_debug_artifacts(self, isolated, voices):
        _write_real_env(
            isolated, L2_IRODORI_VOICES_JSON=str(voices),
            LANGSMITH_TRACING_V2="true", L2_DEBUG_ARTIFACTS="true",
        )
        self._main(isolated)
        assert preflight._tracing_state() == "無効"
        from lab_lounge.debug import is_debug_enabled

        assert is_debug_enabled() is False

    def test_trace_flag_keeps_tracing(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), LANGSMITH_TRACING="true")
        self._main(isolated, argv=("--paid", "--profile", "dev", "--trace"))
        assert os.environ["LANGSMITH_TRACING"] == "true"


class TestNoPipelineWithoutTts:
    """--no-pipeline は TTS を通らないので、TTS の設定で課金の前に止めない。"""

    @staticmethod
    def _main(root: Path, *argv: str) -> tuple[int, MagicMock]:
        with patch.object(preflight, "missing_services", return_value=[]), \
                patch.object(preflight, "run_paid") as run:
            return preflight.main(list(argv), repo_root=root), run

    @pytest.mark.parametrize("flags,called,code", [
        (("--no-pipeline",), True, preflight.EXIT_OK),
        ((), False, preflight.EXIT_FAILED),
    ], ids=["no-pipeline", "full"])
    def test_real_tts_flag_stops_only_the_run_that_uses_tts(self, isolated, capsys, flags, called, code):
        _write_real_env(isolated, L2_USE_REAL_TTS="false")
        got, run = self._main(isolated, "--paid", "--profile", "dev", *flags)
        assert (got, run.called) == (code, called)
        assert ("mode/L2_USE_REAL_TTS" in capsys.readouterr().out) is not called

    @pytest.mark.parametrize("flags,called,code,mark", [
        (("--no-pipeline",), True, preflight.EXIT_OK, "[FAIL*] tts-config/mimi"),
        ((), False, preflight.EXIT_FAILED, "[FAIL ] tts-config/mimi"),
    ], ids=["no-pipeline", "full"])
    def test_missing_voices_stop_only_the_run_that_uses_tts(self, isolated, capsys, flags, called, code, mark):
        """行は残して、合否にだけ数えない。"""
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(isolated / "no-such-voices.json"))
        got, run = self._main(isolated, "--paid", "--profile", "dev", *flags)
        assert (got, run.called) == (code, called)
        assert mark in capsys.readouterr().out

    def test_real_llm_flag_still_stops_it(self, isolated, capsys):
        """LLM の経路は通るので、ダミーのままなら止める。"""
        _write_real_env(isolated, L2_USE_REAL_LLM="false")
        got, run = self._main(isolated, "--paid", "--profile", "dev", "--no-pipeline")
        assert (got, run.called) == (preflight.EXIT_FAILED, False)
        assert "[FAIL ] mode/L2_USE_REAL_LLM" in capsys.readouterr().out

    def test_static_gate_is_not_loosened(self, isolated, capsys):
        """--no-pipeline は --paid の省略。--static に付けても、起動前のゲートから TTS を外さない。"""
        _write_real_env(isolated, REDIS_STREAM_KEY=preflight.PROD_STREAM, L2_USE_REAL_TTS="false")
        got, _ = self._main(isolated, "--static", "--profile", "prod", "--no-pipeline")
        assert got == preflight.EXIT_FAILED
        assert "[FAIL ] mode/L2_USE_REAL_TTS" in capsys.readouterr().out

    @pytest.mark.parametrize("flags,called,code,mark", [
        (("--no-pipeline",), True, preflight.EXIT_OK, "[FAIL*] tts-config/readings"),
        ((), False, preflight.EXIT_FAILED, "[FAIL ] tts-config/readings"),
    ], ids=["no-pipeline", "full"])
    def test_broken_readings_stop_only_the_run_that_uses_tts(self, isolated, voices, capsys, flags, called, code, mark):
        """読み辞書も TTS の設定なので、tts-config のキャラの行と同じ扱いにする。"""
        readings = isolated / "broken-readings.json"
        readings.write_text("{broken", encoding="utf-8")
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), L2_IRODORI_READINGS_JSON=str(readings))
        got, run = self._main(isolated, "--paid", "--profile", "dev", *flags)
        assert (got, run.called) == (code, called)
        assert mark in capsys.readouterr().out


class TestLogCapture:
    def test_collects_real_logger_and_worker_threads_and_restores_root(self, isolated):
        root = logging.getLogger()
        before_handlers, before_level = list(root.handlers), root.level
        with preflight._captured_logs() as capture:
            logging.getLogger("lab_lounge.graph").info("from main")
            thread = threading.Thread(target=lambda: logging.getLogger("lab_lounge.graph").warning("from worker"))
            thread.start()
            thread.join()
            logging.getLogger("httpx").warning("ignored")
            messages = capture.messages_since(0)
        assert messages == ["from main", "from worker"]
        assert list(root.handlers) == before_handlers
        assert root.level == before_level

    def test_logs_do_not_fall_through_to_last_resort(self, isolated, capsys):
        """本番と同じく root に handler がない状態で、ログが lastResort から stderr へ落ちないこと。

        pytest の logging の handler は画面へ書かないので、root をそのままにすると capsys の検査が効かない。
        """
        with patch.object(logging.getLogger(), "handlers", []), \
                patch.object(logging, "lastResort", logging.StreamHandler(sys.stderr)), \
                preflight._captured_logs():
            logging.getLogger("lab_lounge.graph").error("Agent 実行完了 text=%r", BODY)
            logging.getLogger("httpx").error("text=%r", BODY)
        captured = capsys.readouterr()
        assert not _shows_answer(captured.out + captured.err)

    def test_warnings_are_routed_away_and_restored(self, isolated):
        """pydantic の警告は input_value= に値を含む。pytest が横取りするので capsys では検査にならない。"""
        before = logging.getLogger().level
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with preflight._captured_logs():
                warnings.warn(f"input_value={BODY}", UserWarning, stacklevel=1)
            assert seen == []
            warnings.warn("after", UserWarning, stacklevel=1)
            assert len(seen) == 1
        assert logging.getLogger().level == before

    def test_existing_root_handlers_do_not_receive_the_body(self, isolated, capsys):
        root = logging.getLogger()
        console = logging.StreamHandler(sys.stderr)
        root.addHandler(console)
        try:
            with preflight._captured_logs():
                logging.getLogger("lab_lounge.graph").error("text=%r", BODY)
        finally:
            root.removeHandler(console)
        assert "BODYSENTINEL" not in capsys.readouterr().err

    def test_badly_formatted_log_does_not_break_the_caller(self, isolated, capture):
        """本番の logger 呼び出しの場所で例外を出すと、点検の結果まで変わる。"""
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "n=%d", ("notint",)))
        assert capture.messages_since(0) == ["n=%d"]
        assert "n=" in capture.last_problem(0)

    def test_problem_log_oserror_does_not_stop_the_check(self, isolated):
        capture = preflight._LogCapture(isolated / "dir")
        (isolated / "dir").mkdir()
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "x"))
        assert capture.wrote_problems is False

    def test_window_and_level_of_last_problem(self, capture):
        capture.emit(_log_record("lab_lounge.x", logging.ERROR, "古い問題"))
        mark = capture.mark()
        capture.emit(_log_record("lab_lounge.x", logging.INFO, "情報"))
        assert capture.last_problem(mark) == ""
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "一つ目"))
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "二つ目"))
        capture.emit(_log_record("lab_lounge.x", logging.INFO, "後の情報"))
        assert capture.last_problem(mark) == "二つ目"
        assert capture.messages_since(mark)[0] == "情報"

    def test_problem_log_keeps_the_full_text_and_redacts_secrets(self, isolated):
        os.environ["OPENAI_API_KEY"] = SECRET
        path = isolated / "logs" / "preflight" / "p.log"
        with preflight._captured_logs(path) as capture:
            logging.getLogger("lab_lounge.router").error("失敗: %s", f"{BODY} {SECRET}")
            logging.getLogger("lab_lounge.router").info("本文 text=%r", "INFO-BODY")
            capture.note_exception("main/mimi", RuntimeError("full detail"))
        text = path.read_text(encoding="utf-8")
        assert "BODYSENTINEL" in text
        assert "full detail" in text
        assert "INFO-BODY" not in text
        assert SECRET not in text
        assert capture.wrote_problems is True


class TestSecrets:
    """API キーの値は set / unset 以外の形で出さない。"""

    def test_static_output_has_no_secret(self, isolated, capsys, monkeypatch):
        for name in preflight._SECRET_ENV_NAMES:
            os.environ[name] = SECRET
        os.environ["L2_OBS_WS_URL"] = "ws://192.168.0.200:4455"
        fake_obs = MagicMock()
        fake_obs.ReqClient.side_effect = ConnectionRefusedError(f"192.168.0.200:4455 auth={SECRET}")
        monkeypatch.setitem(sys.modules, "obsws_python", fake_obs)
        preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        captured = capsys.readouterr()
        assert SECRET not in captured.out + captured.err
        assert "192.168.0.200" not in captured.out + captured.err

    def test_local_exception_text_is_redacted(self, isolated, capsys, capture):
        os.environ["OPENAI_API_KEY"] = SECRET

        def boom():
            raise RuntimeError(f"401 bad key {SECRET} and sk-abcdefghijkl and AIzaSyABCDEFG")

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", boom)
        out = capsys.readouterr().out
        assert SECRET not in out
        assert "sk-abcdefghijkl" not in out
        assert "AIzaSyABCDEFG" not in out
        assert "[REDACTED]" in out

    def test_api_error_body_is_not_shown(self, isolated, capsys, capture):
        """モデレーションや入力検証のエラーは、発話や prompt の一部を本文に含めて返す。"""
        def boom():
            raise _FakeApiError(f"Invalid prompt: {BODY}")

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", boom)
        out = capsys.readouterr().out
        assert "BODYSENTINEL" not in out
        assert "status=400" in out
        assert "param=temperature" in out

    def test_logged_api_error_body_is_not_shown(self, isolated, capture):
        record = _log_record("lab_lounge.router", logging.ERROR, "LLM ルーター呼び出し失敗: %s", (_FakeApiError(BODY),))
        capture.emit(record)
        reason = capture.last_problem(0)
        assert "BODYSENTINEL" not in reason
        assert "status=400" in reason

    def test_long_log_argument_is_replaced_by_its_length(self, isolated, capture):
        capture.emit(_log_record("lab_lounge.router", logging.WARNING, "応答: %s", (BODY * 5,)))
        assert "BODYSENTINEL" not in capture.last_problem(0)

    def test_secret_across_the_truncation_boundary_leaves_no_prefix(self, isolated):
        """先に切り詰めると、途中までの秘匿値が全文一致に当たらず先頭が残る。"""
        os.environ["L2_PORCUPINE_ACCESS_KEY"] = "PorcupineAccessKeyAAAAAAAAAAAAAAAAAAAA"
        described = preflight._describe_exc(ValueError("x" * 180 + " PorcupineAccessKeyAAAAAAAAAAAAAAAAAAAA"))
        assert "Porcupine" not in described

    @pytest.mark.parametrize("text,hidden", [
        ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
        ("x-api-key: abcdef123", "abcdef123"),
        ("api_key=abcdef123", "abcdef123"),
        ("password=abcdef123", "abcdef123"),
        ("redis://:abcdef123@host:6379", "abcdef123"),
    ])
    def test_redact_hides_secret_contexts(self, text, hidden):
        assert hidden not in preflight.redact(text)

    def test_redact_hides_encoded_values(self, isolated):
        os.environ["L2_PORCUPINE_ACCESS_KEY"] = "abc+def/ghi=="
        assert "abc%2Bdef" not in preflight.redact("key=abc%2Bdef%2Fghi%3D%3D")

    def test_redact_hides_lan_addresses_but_not_loopback(self):
        assert preflight.redact("ws://192.168.0.200:4455") == "ws://[LAN]:4455"
        assert preflight.redact("redis://127.0.0.1:6379") == "redis://127.0.0.1:6379"

    def test_keys_are_reported_as_set_or_unset(self, isolated, capsys):
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["OPENAI_API_KEY"] = SECRET
        report = preflight.Report()
        preflight._check_keys(report, {"openai", "anthropic"}, set())
        assert _status(report, "key/openai").detail == "set"
        assert _status(report, "key/anthropic").status == "FAIL"
        assert SECRET not in capsys.readouterr().out

    def test_inherited_key_is_a_warning(self, isolated):
        """継承したキーで緑になると、Explorer から起動する本番では落ちる。"""
        os.environ["OPENAI_API_KEY"] = SECRET
        report = preflight.Report()
        preflight._check_keys(report, {"openai"}, {"SOMETHING_ELSE"})
        assert _status(report, "key/openai").status == "WARN"
        report = preflight.Report()
        preflight._check_keys(report, {"openai"}, {"OPENAI_API_KEY"})
        assert _status(report, "key/openai").status == "PASS"

    @pytest.mark.parametrize("url", [
        "redis://:SENTINELpw/abc@127.0.0.1:6379",
        "redis://:SENTINELpw#abc@127.0.0.1:6379",
    ])
    def test_malformed_url_is_reported_without_the_password(self, isolated, capsys, url):
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "FAIL"
        assert "SENTINELpw" not in capsys.readouterr().out

    def test_remote_host_is_not_shown(self, isolated, capsys):
        os.environ["L2_C2_URL_READONLY"] = "http://obs-pc.local:8100"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_C2_URL_READONLY").detail == "remote:8100"
        assert "obs-pc" not in capsys.readouterr().out

    def test_malformed_url_run_ends_normally_without_the_password(self, tmp_path):
        """--static の実行全体で、パスワードを含む URL の値が出ないことを確かめる。

        _cli は想定外の例外も exit 1 と Traceback なしにするので、終了コードだけでは FAIL の 1 行と見分けられない。
        行と要約まで届いたことも確かめる。配信文脈は実リポの current.md を開かないように一時ディレクトリへ向ける。
        """
        env = {k: v for k, v in os.environ.items() if not k.startswith(("L2_", "OPENAI", "ANTHROPIC", "GOOGLE", "TAVILY", "LANGSMITH"))}
        env.update(PYTHON_DOTENV_DISABLED="1", REDIS_URL="redis://:SENTINELpw/abc@127.0.0.1:6", PYTHONIOENCODING="utf-8",
                   L2_STREAM_CONTEXT_FILE=str(tmp_path / "none.md"))
        done = subprocess.run(
            [sys.executable, "-m", "lab_lounge.preflight", "--static", "--profile", "dev"],
            capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path, timeout=120, check=False,
        )
        assert done.returncode == preflight.EXIT_FAILED
        assert "SENTINELpw" not in done.stdout + done.stderr
        assert "Traceback" not in done.stderr
        assert "想定外のエラー" not in done.stdout
        assert "[FAIL ] url/REDIS_URL" in done.stdout
        assert "  FAIL url/REDIS_URL" in done.stdout  # 要約まで届いた

    @pytest.mark.parametrize("url", ["redis://:SENTINELpw/abc@127.0.0.1:6379", "redis://:SENTINELpw#abc@127.0.0.1:6379"])
    def test_malformed_url_detail_has_no_password_before_redact(self, isolated, url):
        """画面の出力は redact を通るので、# の形は行に値を足しても隠れてしまう。redact の前の detail で確かめる。"""
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert "SENTINELpw" not in _status(report, "url/REDIS_URL").detail


class TestRefusals:
    """課金や本番への書き込みの前に止まることを確かめる。"""

    def test_paid_refuses_the_prod_stream(self, isolated):
        _write_real_env(isolated, REDIS_STREAM_KEY="aibyss:events")
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        try:
            assert preflight.main(["--paid", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_REFUSED
        finally:
            for p in patches:
                p.stop()
        for mock in mocks:
            mock.assert_not_called()

    def test_allow_prod_passes_the_stream_guard(self, isolated, voices):
        _write_real_env(isolated, REDIS_STREAM_KEY="aibyss:events", L2_IRODORI_VOICES_JSON=str(voices))
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        try:
            with patch.object(preflight, "missing_services", return_value=["Redis"]):
                code = preflight.main(["--paid", "--profile", "prod", "--allow-prod"], repo_root=isolated)
        finally:
            for p in patches:
                p.stop()
        assert code == preflight.EXIT_MISSING_SERVICE
        for mock in mocks:
            mock.assert_not_called()

    def test_paid_stops_before_billing_when_services_are_missing(self, isolated, voices, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        try:
            code = preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        finally:
            for p in patches:
                p.stop()
        assert code == preflight.EXIT_MISSING_SERVICE
        assert "Redis" in capsys.readouterr().out
        for mock in mocks:
            mock.assert_not_called()

    def test_paid_stops_before_billing_when_static_fails(self, isolated, voices):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), OPENAI_API_KEY="")
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        try:
            assert preflight.main(["--paid", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_FAILED
        finally:
            for p in patches:
                p.stop()
        for mock in mocks:
            mock.assert_not_called()

    @pytest.mark.parametrize("name", ["L2_USE_REAL_LLM", "L2_USE_REAL_TTS"])
    @pytest.mark.parametrize("value", ["false", "ture"])
    def test_paid_refuses_dummy_mode(self, isolated, voices, name, value):
        """ダミーのままだと、課金だけして本番の経路を通らず緑になる。"""
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), **{name: value})
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        try:
            assert preflight.main(["--paid", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_FAILED
        finally:
            for p in patches:
                p.stop()
        for mock in mocks:
            mock.assert_not_called()

    def test_static_never_calls_a_billed_path_even_with_every_flag_on(self, isolated, voices, monkeypatch):
        """フラグをすべて有効にしないと、課金の経路が混入しても SKIP で素通りしてしまう。"""
        _write_real_env(
            isolated,
            L2_IRODORI_VOICES_JSON=str(voices),
            L2_USE_LLM_ROUTER="true", L2_USE_INTENT_GATE="true", L2_ENABLE_RAG="true",
            L2_USE_C2_RETRIEVER="true", L2_FILLER_ENABLED="true",
        )
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        connect = MagicMock(side_effect=OSError("blocked"))
        monkeypatch.setattr(socket.socket, "connect", connect)
        try:
            preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        finally:
            for p in patches:
                p.stop()
        for mock in mocks:
            mock.assert_not_called()
        connect.assert_not_called()

    def test_static_may_read_the_prod_stream(self, isolated):
        """本番の起動前にも流すので、--static は本番ストリームでも止まらない。"""
        _write_real_env(isolated, REDIS_STREAM_KEY="aibyss:events", L2_USE_REAL_TTS="false", L2_USE_REAL_LLM="false")
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_OK

    def test_static_prod_requires_real_modes(self, isolated):
        """本番ストリームでダミーが正当な場面はない。"""
        _write_real_env(isolated, REDIS_STREAM_KEY="aibyss:events", L2_USE_REAL_LLM="false", L2_USE_REAL_TTS="false")
        assert preflight.main(["--static", "--profile", "prod"], repo_root=isolated) == preflight.EXIT_FAILED

    def test_static_dev_allows_dummy_mode(self, isolated):
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_OK

    @pytest.mark.parametrize("argv", [
        ["--static"],  # --profile がない
        ["--profile", "dev"],  # モードがない
        ["--static", "--paid", "--profile", "dev"],
        ["--static", "--profile", "dev", "--characters", "nobody"],
    ])
    def test_usage_errors_are_distinct_from_refusals(self, isolated, argv):
        with pytest.raises(SystemExit) as excinfo:
            preflight.main(argv, repo_root=isolated)
        assert excinfo.value.code == preflight.EXIT_USAGE

    def test_dev_points_at_env_dev_first(self, isolated, capsys):
        code = preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_REFUSED
        assert out.index(".env.dev") < out.index("--allow-prod")
        assert "REDIS_STREAM_KEY" in out
        assert "本当に流すなら" not in out

    def test_prod_names_only_allow_prod(self, isolated, capsys):
        code = preflight.main(["--paid", "--profile", "prod"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_REFUSED
        assert "本当に流すなら --allow-prod" in out
        assert ".env.dev" not in out


class TestEnvPreparation:
    """継承した接続先で本番と違う相手を測らないことを確かめる。"""

    def test_inherited_endpoint_is_removed_and_named(self, isolated, capsys):
        os.environ["ANTHROPIC_BASE_URL"] = "http://proxy.invalid/secret-path"
        removed = preflight.clean_llm_env()
        out = capsys.readouterr().out
        assert removed == ["ANTHROPIC_BASE_URL"]
        assert "ANTHROPIC_BASE_URL" not in os.environ
        assert "ANTHROPIC_BASE_URL" in out
        assert "secret-path" not in out

    def test_endpoint_from_env_file_is_refused(self, isolated):
        env_file = isolated / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://elsewhere.invalid\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            preflight.clean_llm_env([env_file])
        assert excinfo.value.code == preflight.EXIT_REFUSED

    def test_main_refuses_endpoint_from_env_file(self, isolated):
        (isolated / ".env").write_text("ANTHROPIC_BASE_URL=http://elsewhere.invalid\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        assert excinfo.value.code == preflight.EXIT_REFUSED

    def test_loopback_is_kept_only_when_asked_under_pytest(self, isolated):
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9999"
        os.environ["ANTHROPIC_BASE_URL"] = "http://proxy.invalid"
        preflight.clean_llm_env(allow_loopback=True)
        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:9999"
        assert "ANTHROPIC_BASE_URL" not in os.environ

        preflight.clean_llm_env()
        assert "OPENAI_BASE_URL" not in os.environ

    def test_loopback_flag_is_ignored_outside_pytest(self, isolated):
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9999"
        with patch.dict(os.environ):
            del os.environ["PYTEST_CURRENT_TEST"]
            preflight.clean_llm_env(allow_loopback=True)
            assert "OPENAI_BASE_URL" not in os.environ

    def test_dev_profile_overlays_env_dev_and_files_win(self, isolated):
        os.environ["REDIS_STREAM_KEY"] = "left-over-from-a-previous-run"
        (isolated / ".env").write_text("REDIS_STREAM_KEY=aibyss:events\nL2_LLM_MODEL=base\n", encoding="utf-8")
        (isolated / ".env.dev").write_text("REDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        preflight.prepare_env("dev", isolated)
        assert os.environ["REDIS_STREAM_KEY"] == "aibyss:events-dev"
        assert os.environ["L2_LLM_MODEL"] == "base"

    def test_prod_profile_ignores_env_dev(self, isolated):
        (isolated / ".env").write_text("REDIS_STREAM_KEY=aibyss:events\n", encoding="utf-8")
        (isolated / ".env.dev").write_text("REDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        preflight.prepare_env("prod", isolated)
        assert os.environ["REDIS_STREAM_KEY"] == "aibyss:events"

    @pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"])
    def test_every_tracing_variable_is_silenced(self, isolated, name):
        """定数でパラメータ化すると、一覧から消した名前はテストごと消える。名前は直書きする。"""
        os.environ[name] = "true"
        assert "有効" in preflight._tracing_state()
        preflight.silence_side_outputs()
        assert os.environ[name] == "false"
        assert preflight._tracing_state() == "無効"

    @pytest.mark.parametrize("name", [
        "ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL", "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_BASE_URL", "OPENAI_API_BASE", "GOOGLE_GEMINI_BASE_URL",
    ])
    def test_every_inherited_endpoint_is_removed(self, isolated, name):
        os.environ[name] = "x"
        preflight.clean_llm_env()
        assert name not in os.environ

    def test_refusal_does_not_print_the_endpoint(self, isolated, capsys):
        env_file = isolated / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://elsewhere.invalid\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            preflight.clean_llm_env([env_file])
        assert "elsewhere" not in capsys.readouterr().out

    def test_langsmith_itself_agrees_tracing_is_off(self, isolated):
        """langsmith は *_TRACING_V2 を先に見る。実物の判定で確かめる。"""
        ls_utils = pytest.importorskip("langsmith.utils")
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        preflight.silence_side_outputs()
        ls_utils.get_env_var.cache_clear()
        try:
            assert ls_utils.tracing_is_enabled() is False
        finally:
            ls_utils.get_env_var.cache_clear()

    def test_trace_keeps_tracing_but_not_debug_artifacts(self, isolated):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["L2_DEBUG_ARTIFACTS"] = "true"
        preflight.silence_side_outputs(trace=True)
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["L2_DEBUG_ARTIFACTS"] == "false"

    def test_dotenv_disabled_reads_nothing(self, isolated):
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        (isolated / ".env").write_text("L2_LLM_MODEL=from-file\n", encoding="utf-8")
        assert preflight.prepare_env("dev", isolated) == []
        assert "L2_LLM_MODEL" not in os.environ


class TestPaidForcesDecisionLogOff:
    """--paid は判定ログを強制でオフにする (点検の試験の発話が、判定のリプレイの入力に混ざらないように)。"""

    def test_silence_side_outputs_turns_it_off_and_returns_the_production_value(self, isolated):
        os.environ["L2_DECISION_LOG"] = "true"
        production = preflight.silence_side_outputs()
        assert production["L2_DECISION_LOG"] == "true"
        assert os.environ["L2_DECISION_LOG"] == "false"

    def test_decision_log_module_sees_the_forced_off(self, isolated):
        """decision_log は毎回環境変数を読むので、import 済みのプロセスでも切り替えが効く。"""
        from lab_lounge import decision_log

        os.environ["L2_DECISION_LOG"] = "true"
        assert decision_log.is_enabled() is True
        preflight.silence_side_outputs()
        assert decision_log.is_enabled() is False

    def test_row_reports_production_value_but_not_when_it_was_off(self, isolated):
        report = preflight.Report()
        preflight._report_silenced_outputs(report, {"L2_DECISION_LOG": "true"})
        row = _status(report, "flag/L2_DECISION_LOG")
        assert row.status == "WARN" and "この実行では止めています" in row.detail

        report = preflight.Report()
        preflight._report_silenced_outputs(report, {"L2_DECISION_LOG": "false"})
        assert not any(c.name == "flag/L2_DECISION_LOG" for c in report.checks)

    def test_paid_main_path_stops_it(self, isolated, capsys):
        (isolated / ".env").write_text(
            "L2_DECISION_LOG=true\nREDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8",
        )
        with patch.object(preflight, "run_paid"):
            preflight.main(["--paid", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated)
        out = capsys.readouterr().out
        assert "[WARN ] flag/L2_DECISION_LOG: 本番は true" in out
        assert os.environ["L2_DECISION_LOG"] == "false"


class TestSilencedOutputsRow:
    """--paid は実発話が外へ出る設定を止めるが、本番でその設定が有効なことは表から消さない。"""

    _ENV = "L2_DEBUG_ARTIFACTS=true\nLANGSMITH_TRACING=true\nREDIS_STREAM_KEY=aibyss:events-dev\n"

    def _run(self, root, capsys, *flags):
        (root / ".env").write_text(self._ENV, encoding="utf-8")
        # ダミーの設定なので課金の前に止まる。
        with patch.object(preflight, "run_paid") as run:
            preflight.main(["--paid", "--profile", "dev", "--characters", "octamaid", *flags], repo_root=root)
        run.assert_not_called()
        return capsys.readouterr().out

    def test_paid_reports_what_production_will_do(self, isolated, capsys):
        out = self._run(isolated, capsys)
        assert "[WARN ] flag/L2_DEBUG_ARTIFACTS: 本番は true" in out
        assert "本番は有効 (LANGSMITH_TRACING)" in out
        assert out.count("この実行では止めています") == 2

    def test_this_run_still_stays_silent(self, isolated, capsys):
        self._run(isolated, capsys)
        assert os.environ["L2_DEBUG_ARTIFACTS"] == "false"
        assert preflight._tracing_state() == "無効"

    def test_trace_keeps_tracing_so_it_does_not_claim_to_have_stopped_it(self, isolated, capsys):
        out = self._run(isolated, capsys, "--trace")
        assert "flag/LANGSMITH: 有効 (LANGSMITH_TRACING)" in out
        assert "本番は有効" not in out
        assert "flag/L2_DEBUG_ARTIFACTS: 本番は true" in out

    def test_static_rows_are_unchanged(self, isolated, capsys):
        (isolated / ".env").write_text(self._ENV, encoding="utf-8")
        preflight.main(["--static", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated)
        out = capsys.readouterr().out
        assert "flag/L2_DEBUG_ARTIFACTS: true" in out
        assert "flag/LANGSMITH: 有効" in out
        assert "止めています" not in out

    def test_production_off_adds_no_rows(self, isolated):
        os.environ["LANGSMITH_TRACING"] = "false"
        report = preflight.Report()
        preflight._report_silenced_outputs(report, preflight.silence_side_outputs())
        assert report.checks == []

    def test_reports_before_the_paid_run_starts(self, isolated, voices, capsys):
        """合格する設定でも、課金の前に本番の設定が読める。"""
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), L2_DEBUG_ARTIFACTS="true")

        def check_output_so_far(*_args, **_kwargs):
            assert "flag/L2_DEBUG_ARTIFACTS: 本番は true" in capsys.readouterr().out

        with patch.object(preflight, "missing_services", return_value=[]), \
                patch.object(preflight, "run_paid", side_effect=check_output_so_far) as run:
            preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        run.assert_called_once()


_LOCK = """
[[package]]
name = "lab-lounge"
version = "0.1.0"
source = { editable = "." }
dependencies = [{ name = "needed" }, { name = "windows-only", marker = "sys_platform == 'nosuchplatform'" }]

[package.optional-dependencies]
extra-one = [{ name = "from-extra" }]

[[package]]
name = "needed"
version = "1.0"
dependencies = [{ name = "transitive" }]

[[package]]
name = "from-extra"
version = "2.0"

[[package]]
name = "transitive"
version = "3.0"

[[package]]
name = "windows-only"
version = "4.0"

[[package]]
name = "not-reachable"
version = "5.0"
"""


class TestVersionCheck:
    """素の uv sync で extras が消えた環境を、起動前に赤くする。"""

    def test_closure_follows_extras_and_skips_markers_and_unreachable(self):
        wanted = preflight.locked_closure(tomllib.loads(_LOCK))
        assert wanted == {"needed": {"1.0"}, "from-extra": {"2.0"}, "transitive": {"3.0"}}

    def test_real_lock_is_readable(self):
        """uv.lock の形式が変わると、検知器が何も見なくなる。実物で固定する。"""
        lock = _REPO_ROOT / "uv.lock"
        if not lock.is_file():
            pytest.skip("uv.lock がありません")
        wanted = preflight.locked_closure(tomllib.loads(lock.read_text(encoding="utf-8")))
        assert {"redis", "langchain", "langgraph", "openai", "anthropic", "google-genai"} <= set(wanted)

    def test_missing_and_mismatch_are_fail(self, isolated):
        (isolated / "uv.lock").write_text(_LOCK, encoding="utf-8")
        installed = {"needed": "1.0", "transitive": "9.9"}

        def fake_version(name):
            if name in installed:
                return installed[name]
            raise preflight.metadata.PackageNotFoundError(name)

        report = preflight.Report()
        with patch.object(preflight.metadata, "version", fake_version):
            preflight._check_versions(report, isolated)
        check = _status(report, "deps/uv.lock")
        assert check.status == "FAIL"
        assert "from-extra" in check.detail
        assert "transitive 9.9" in check.detail

    def test_matching_lock_is_pass(self, isolated):
        (isolated / "uv.lock").write_text(_LOCK, encoding="utf-8")
        versions = {"needed": "1.0", "from-extra": "2.0", "transitive": "3.0"}
        report = preflight.Report()
        with patch.object(preflight.metadata, "version", versions.__getitem__):
            preflight._check_versions(report, isolated)
        assert _status(report, "deps/uv.lock").status == "PASS"

    def test_no_lock_is_a_warning(self, isolated):
        report = preflight.Report()
        preflight._check_versions(report, isolated)
        assert _status(report, "deps/uv.lock").status == "WARN"

    @pytest.mark.parametrize("lock", ["version = 1\n", '[[package]]\nname = "x"\nversion = "1"\nsource = { virtual = "." }\n'])
    def test_unreadable_lock_is_fail_not_zero_matches(self, isolated, lock):
        """0 件のまま PASS にすると、検知器が何も見ていない状態を見逃す。"""
        (isolated / "uv.lock").write_text(lock, encoding="utf-8")
        report = preflight.Report()
        preflight._check_versions(report, isolated)
        assert _status(report, "deps/uv.lock").status == "FAIL"

    def test_static_fails_when_the_environment_lost_packages(self, isolated):
        """素の uv sync で依存が消えた環境は赤くなること。"""
        (isolated / "uv.lock").write_text(_LOCK, encoding="utf-8")
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_FAILED


class TestImports:
    """版が一致していても import できないことがある（Windows のパス長、API の消失）。"""

    def test_failed_import_is_fail(self, isolated):
        def fake_import(name):
            if name == "anthropic":
                raise ModuleNotFoundError("No module named anthropic.types.beta.very_long_name")
            return MagicMock()

        report = preflight.Report()
        with patch("importlib.import_module", fake_import):
            preflight._check_imports(report, {"anthropic"})
        assert _status(report, "import/anthropic").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_all_imported_is_pass(self, isolated):
        report = preflight.Report()
        with patch("importlib.import_module", return_value=MagicMock()):
            preflight._check_imports(report, {"openai", "anthropic", "google"})
        assert _status(report, "import").status == "PASS"

    def test_tools_require_create_react_agent(self, isolated):
        """create_react_agent が消えると、協働先がログなしでツールなしの呼び出しへ落ちる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"

        def fake_import(name):
            if name == "langgraph.prebuilt":
                return SimpleNamespace()
            return MagicMock()

        report = preflight.Report()
        with patch("importlib.import_module", fake_import):
            preflight._check_imports(report, set())
        assert _status(report, "import/langgraph.prebuilt").status == "FAIL"


class TestStaticFindings:
    def test_stale_stream_context_warns_without_showing_the_body(self, isolated, capsys):
        path = isolated / "current.md"
        path.write_text("配信外のメモ", encoding="utf-8")
        old = 1_000_000_000
        os.utime(path, (old, old))
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        assert _status(report, "stream_context").status == "WARN"
        assert "配信外のメモ" not in capsys.readouterr().out

    def test_fresh_stream_context_is_not_a_warning(self, isolated):
        path = isolated / "current.md"
        path.write_text("本日の配信", encoding="utf-8")
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        assert _status(report, "stream_context").status == "INFO"

    def test_empty_old_stream_context_is_not_a_warning(self, isolated):
        """空にした current.md は prompt に足されないので、警告しない。"""
        path = isolated / "current.md"
        path.write_text("  \n", encoding="utf-8")
        old = 1_000_000_000
        os.utime(path, (old, old))
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        assert _status(report, "stream_context").status == "INFO"

    def test_localhost_and_excluded_port_warn(self, isolated, monkeypatch):
        monkeypatch.setattr(preflight.sys, "platform", "win32")
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:50021"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "WARN"
        assert _status(report, "url/L2_TTS_VOICEVOX_URL").status == "WARN"

    def test_stream_mismatch_warns(self, isolated):
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events"
        report = preflight.Report()
        preflight._check_stream(report, "dev")
        assert _status(report, "stream").status == "WARN"

    def test_misspelled_flag_warns(self, isolated):
        os.environ["L2_USE_REAL_LLM"] = "ture"
        report = preflight.Report()
        preflight._check_flags(report)
        assert _status(report, "flag/L2_USE_REAL_LLM").status == "WARN"

    def test_c2_retriever_value_graph_does_not_accept_warns(self, isolated):
        os.environ["L2_USE_C2_RETRIEVER"] = "1"
        report = preflight.Report()
        preflight._check_flags(report)
        assert any(c.name == "flag/L2_USE_C2_RETRIEVER" and c.status == "WARN" for c in report.checks)

    def test_debug_artifacts_warn(self, isolated):
        os.environ["L2_DEBUG_ARTIFACTS"] = "true"
        report = preflight.Report()
        preflight._check_flags(report)
        assert _status(report, "flag/L2_DEBUG_ARTIFACTS").status == "WARN"

    def test_decision_log_warns_only_when_on(self, isolated):
        """判定ログは実発話を平文で残すので、オンのときだけ WARN を出す (既定の false では行を出さない)。"""
        report = preflight.Report()
        preflight._check_flags(report)
        assert not any(c.name == "flag/L2_DECISION_LOG" for c in report.checks)

        os.environ["L2_DECISION_LOG"] = "true"
        report = preflight.Report()
        preflight._check_flags(report)
        row = _status(report, "flag/L2_DECISION_LOG")
        assert row.status == "WARN"
        # 書き先は cwd からの相対なので、絶対パスで示す。
        assert str(Path.cwd() / "logs" / "decisions") in row.detail

    def test_debug_artifacts_warn_does_not_call_logs_tracked(self, isolated):
        """logs/ は gitignore 済み。「追跡済み」と書くと、公開リポジトリに入ると誤解させる。"""
        os.environ["L2_DEBUG_ARTIFACTS"] = "true"
        report = preflight.Report()
        preflight._check_flags(report)
        detail = _status(report, "flag/L2_DEBUG_ARTIFACTS").detail
        assert "gitignore 済み" in detail and "追跡済み" not in detail

    def test_broken_numeric_env_is_fail_and_hides_the_value(self, isolated, capsys):
        os.environ["L2_LLM_MAX_RETRIES"] = "abc-value"
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, "flag/L2_LLM_MAX_RETRIES").status == "FAIL"
        assert "abc-value" not in capsys.readouterr().out

    def test_missing_persona_falls_back_with_a_warning(self, isolated):
        character = SimpleNamespace(slug="mimi", system_prompt_file="no-such.txt", sample_fallback="sample_empath.md")
        report = preflight.Report()
        preflight._check_personas(report, [character])
        assert _status(report, "persona/mimi").status == "WARN"

    def test_missing_persona_and_sample_is_fail(self, isolated):
        character = SimpleNamespace(slug="mimi", system_prompt_file="no-such.txt", sample_fallback="no-such-sample.md")
        report = preflight.Report()
        preflight._check_personas(report, [character])
        assert _status(report, "persona/mimi").status == "FAIL"

    def test_missing_voice_is_fail_for_required(self, isolated, voices):
        os.environ["L2_USE_REAL_TTS"] = "true"
        character = SimpleNamespace(slug="mimi", tts_provider="irodori_vd", tts_voice="no-such-voice")
        report = preflight.Report()
        preflight._check_tts_config(report, [character])
        assert _status(report, "tts-config/mimi").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_missing_voices_file_is_fail(self, isolated):
        os.environ["L2_USE_REAL_TTS"] = "true"
        os.environ["L2_IRODORI_VOICES_JSON"] = str(isolated / "no-such.json")
        character = SimpleNamespace(slug="mimi", tts_provider="irodori_vd", tts_voice="mimi")
        report = preflight.Report()
        preflight._check_tts_config(report, [character])
        assert _status(report, "tts-config/mimi").status == "FAIL"

    def test_missing_reference_wav_is_fail(self, isolated):
        os.environ["L2_USE_REAL_TTS"] = "true"
        path = isolated / "with-ref.json"
        path.write_text(json.dumps({"voices": {"mimi": {"ref_wav": "no-such.wav"}}}), encoding="utf-8")
        os.environ["L2_IRODORI_VOICES_JSON"] = str(path)
        character = SimpleNamespace(slug="mimi", tts_provider="irodori_vd", tts_voice="mimi")
        report = preflight.Report()
        preflight._check_tts_config(report, [character])
        assert _status(report, "tts-config/mimi").status == "FAIL"

    def test_missing_voicepeak_of_non_required_does_not_fail_the_run(self, isolated):
        os.environ["L2_USE_REAL_TTS"] = "true"
        os.environ["L2_TTS_VOICEPEAK_PATH"] = str(isolated / "no-such.exe")
        report = preflight.Report()
        preflight._check_tts_config(report, [get_character("ruka")])
        assert _status(report, "tts-config/ruka").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK

    def test_tts_config_is_skipped_for_dummy_mode(self, isolated):
        report = preflight.Report()
        preflight._check_tts_config(report, get_all_characters())
        assert report.checks == []

    def test_obs_failure_is_only_a_warning(self, isolated, monkeypatch):
        """OBS がなくても配信の主機能は動くので、起動を止めない。"""
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        fake_obs = MagicMock()
        fake_obs.ReqClient.side_effect = ConnectionRefusedError("refused")
        monkeypatch.setitem(sys.modules, "obsws_python", fake_obs)
        report = preflight.Report()
        preflight._check_obs(report, get_all_characters())
        assert _status(report, "obs").status == "WARN"
        assert report.exit_code() == preflight.EXIT_OK

    def test_obs_missing_group_is_named(self, isolated, monkeypatch):
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        client = MagicMock()

        def groups(slug):
            if slug == "mimi":
                return SimpleNamespace(scene_items=[{"sourceName": "mimi_neutral", "sceneItemId": 1}])
            raise RuntimeError("not found")

        client.get_group_scene_item_list.side_effect = groups
        fake_obs = MagicMock()
        fake_obs.ReqClient.return_value = client
        monkeypatch.setitem(sys.modules, "obsws_python", fake_obs)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi"), get_character("sakura")])
        check = _status(report, "obs")
        assert check.status == "WARN"
        assert "sakura" in check.detail
        assert "mimi" not in check.detail
        client.disconnect.assert_called_once()


class TestFillerModelDrift:
    """filler.py と同じモデルを見ていること。ずれると別のモデルを点検してしまう。"""

    @pytest.mark.parametrize("character", get_all_characters(), ids=lambda c: c.slug)
    def test_matches_generate_filler_text(self, isolated, character):
        os.environ["L2_LLM_FILLER_MODEL"] = "fallback-model"
        from lab_lounge import filler

        with patch.object(filler, "_call_filler_llm", return_value="text") as call:
            filler._generate_filler_text(character.slug)
        assert call.call_args.args[0] == preflight.effective_filler_model(character)


class TestFillerCacheRow:
    """opener の WAV がないと本番は LLM を呼ぶ前に終わる。WAV は git に入らないので新しい環境で必ず起きる。"""

    def _report(self, *slugs):
        os.environ["L2_USE_REAL_TTS"] = "true"
        report = preflight.Report()
        preflight._check_filler_cache(report, [get_character(slug) for slug in slugs])
        return report

    def _put_opener(self, slug):
        from lab_lounge import filler

        path = filler._FILLER_CACHE_DIR / slug / "opener_00.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_production_gives_up_before_the_llm_without_an_opener(self, isolated):
        """WARN の根拠。本番が opener なしでも LLM のフィラーへ進むように直ったら、見直す。"""
        from lab_lounge import filler

        with patch.object(filler, "_generate_filler_text") as generate, patch("lab_lounge.audio_io.play_audio_file") as play:
            filler.run_filler_loop("mimi", threading.Event())
        generate.assert_not_called()
        play.assert_not_called()

    def test_opener_present_has_no_row_and_production_goes_on(self, isolated):
        """行が消える入力で、本番が opener の再生と LLM のフィラーまで進むこと。"""
        from lab_lounge import filler

        self._put_opener("mimi")
        assert self._report("mimi").checks == []
        reached_llm = threading.Event()
        stop = threading.Event()
        stop.set()  # opener の再生の直後に終わらせる。

        def generate(*_args, **_kwargs):
            reached_llm.set()

        with patch.object(filler, "_generate_filler_text", side_effect=generate), \
                patch("lab_lounge.audio_io.play_audio_file") as play:
            filler.run_filler_loop("mimi", stop)
            assert reached_llm.wait(5)
        play.assert_called_once()

    def test_missing_opener_only_warns_and_names_the_characters(self, isolated):
        self._put_opener("sakura")
        report = self._report("mimi", "sakura", "octamaid")
        check = _status(report, "filler-cache")
        assert check.status == "WARN"
        assert report.exit_code() == preflight.EXIT_OK
        assert "mimi" in check.detail and "octamaid" in check.detail and "sakura" not in check.detail
        assert "generate_filler_cache" in check.detail

    def test_the_hinted_script_exists(self):
        assert (_REPO_ROOT / "scripts" / "generate_filler_cache.py").is_file()

    @pytest.mark.parametrize("slug", ["ruka", "aruka"])
    def test_non_required_characters_are_not_reported(self, isolated, slug):
        """ルーティングされないか、フレーズがなくて作れないキャラの行が毎回並ぶと、本物の警告が埋もれる。"""
        assert self._report(slug).checks == []

    def test_disabled_filler_has_no_row(self, isolated):
        os.environ["L2_FILLER_ENABLED"] = "false"
        assert self._report("mimi").checks == []

    def test_dummy_tts_has_no_row(self, isolated):
        """ダミーの TTS は音声を作らないので、WAV がないことは何も変えない。"""
        report = preflight.Report()
        preflight._check_filler_cache(report, [get_character("mimi")])
        assert report.checks == []


class TestHandraiseFlagDrift:
    """dispatcher.py と同じ規則で読むこと。"""

    @pytest.mark.parametrize("value", [None, "true", "1", "yes", "false", "0", ""])
    def test_matches_dispatcher(self, isolated, value):
        from lab_lounge import dispatcher

        if value is not None:
            os.environ["L2_USE_HANDRAISE"] = value
        assert preflight._handraise_enabled() == dispatcher._get_handraise_config()["use_handraise"]


# ─── 変異テストで見つかった穴 ─────────────────────────────────────
#
# 以下は、preflight.py へ故障を 1 つずつ入れても緑のままだった箇所を固定するテスト。
# 点検は本番の代わりに壊れたことを知らせる道具なので、配線や判定が静かに外れると検知が消える。


def test_exit_codes_are_pinned_as_numbers():
    """数値は README の表に書いた約束で、スクリプトの分岐にも使える。定数同士の比較では EXIT_FAILED=0 でも緑になる。"""
    assert (preflight.EXIT_OK, preflight.EXIT_FAILED, preflight.EXIT_REFUSED,
            preflight.EXIT_MISSING_SERVICE, preflight.EXIT_USAGE) == (0, 1, 2, 3, 64)


class TestMainWiring:
    @pytest.mark.parametrize("flags,with_pipeline,collab", [
        ((), True, "one"),
        (("--no-pipeline",), False, "one"),
        (("--collab", "all"), True, "all"),
        (("--collab", "none"), True, "none"),
    ])
    def test_forwards_pipeline_and_collab(self, isolated, voices, flags, with_pipeline, collab):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        with patch.object(preflight, "missing_services", return_value=[]) as missing, \
                patch.object(preflight, "run_paid") as run:
            preflight.main(["--paid", "--profile", "dev", *flags], repo_root=isolated)
        assert run.call_args.kwargs == {"with_pipeline": with_pipeline, "collab": collab}
        assert missing.call_args.kwargs == {"with_pipeline": with_pipeline}

    def test_prod_profile_is_measured_without_env_dev(self, isolated, voices, capsys):
        """--profile prod で .env.dev を読むと、本番の起動前に dev の設定を測って緑を返す。

        voices.json は一時ファイルへ向け、FAIL の行の名前まで確かめる。既定のパスのままだと、voices.json の
        ない環境では tts-config の FAIL が別の理由で exit 1 を作り、.env.dev を読んでも落ちない。
        """
        _write_real_env(isolated, REDIS_STREAM_KEY="aibyss:events", L2_USE_REAL_LLM="false", L2_USE_REAL_TTS="false",
                        L2_IRODORI_VOICES_JSON=str(voices))
        (isolated / ".env.dev").write_text("L2_USE_REAL_LLM=true\nL2_USE_REAL_TTS=true\n", encoding="utf-8")
        assert preflight.main(["--static", "--profile", "prod"], repo_root=isolated) == 1
        assert "[FAIL ] mode/L2_USE_REAL_LLM" in capsys.readouterr().out

    def test_changes_to_the_repo_root(self, isolated, tmp_path_factory, monkeypatch):
        monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
        preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        assert Path.cwd().resolve() == isolated.resolve()

    def test_passes_env_file_names_so_an_inherited_key_warns(self, isolated, capsys):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            os.environ[name] = "inherited-value"
        (isolated / ".env").write_text("REDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        assert "親プロセスから継承" in capsys.readouterr().out

    def test_static_does_not_silence_what_it_reports(self, isolated, capsys):
        (isolated / ".env").write_text("LANGSMITH_TRACING=true\nL2_DEBUG_ARTIFACTS=true\n", encoding="utf-8")
        preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        out = capsys.readouterr().out
        assert "flag/LANGSMITH: 有効" in out
        assert "flag/L2_DEBUG_ARTIFACTS" in out

    def test_problem_log_is_written_under_logs_preflight(self, isolated, voices, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))

        def noisy(*args, **kwargs):
            logging.getLogger("lab_lounge.x").warning("注意: %s", BODY)

        with patch.object(preflight, "missing_services", return_value=[]), patch.object(preflight, "run_paid", noisy):
            preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        logs = list((isolated / "logs" / "preflight").glob("preflight_*.log"))
        assert logs and BODY in logs[0].read_text(encoding="utf-8")
        assert "詳細:" in capsys.readouterr().out

    def test_inherited_real_mode_warns(self, isolated, voices):
        """継承した L2_USE_REAL_LLM で緑になると、Explorer から起動する本番ではダミーになる。"""
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_USE_REAL_TTS"] = "true"
        report = preflight.Report()
        preflight._check_real_modes(report, "prod", {"SOMETHING_ELSE"}, paid=False)
        assert _status(report, "mode/L2_USE_REAL_LLM").status == "WARN"
        report = preflight.Report()
        preflight._check_real_modes(report, "prod", {"L2_USE_REAL_LLM", "L2_USE_REAL_TTS"}, paid=False)
        assert report.checks == []


class TestRunStaticWiring:
    @pytest.mark.parametrize("name", [
        "_check_versions", "_check_imports", "_check_keys", "_check_flags", "_check_env_quoting", "_check_numeric_env",
        "_check_real_modes", "_check_personas", "_check_tts_config", "_check_readings", "_check_filler_cache",
        "_check_kb_index", "_check_endpoints", "_check_stream", "_check_run_script_refusals", "_check_event_schema",
        "_check_stream_context", "_check_obs",
    ])
    def test_calls_every_check(self, isolated, name):
        with patch.object(preflight, name) as check:
            preflight.run_static(preflight.Report(), isolated, "dev", get_all_characters(), set())
        check.assert_called_once()

    def test_feeds_providers_to_imports_and_keys(self, isolated):
        with patch.object(preflight, "_check_models", return_value={"anthropic"}), \
                patch.object(preflight, "_check_imports") as imports, patch.object(preflight, "_check_keys") as keys:
            preflight.run_static(preflight.Report(), isolated, "dev", [], {"X"})
        assert imports.call_args.args[1] == {"anthropic"}
        assert keys.call_args.args[1:] == ({"anthropic"}, {"X"})

    def test_reports_tracing_state(self, isolated):
        report = preflight.Report()
        preflight.run_static(report, isolated, "dev", [], set())
        assert _status(report, "flag/LANGSMITH")

    def test_broken_numeric_env_fails_the_whole_run(self, isolated):
        os.environ["L2_LLM_MAX_RETRIES"] = "abc"
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == 1

    @pytest.mark.parametrize("name,bad,env", [
        ("L2_LLM_TIMEOUT_SEC", "x", {"L2_USE_REAL_LLM": "true"}),
        ("L2_ROUTER_LLM_TIMEOUT_SEC", "x", {}),
        ("L2_LLM_MAX_RETRIES", "1.5", {}),
        ("L2_AGENT_INVOKE_TIMEOUT_SEC", "x", {"L2_USE_REAL_LLM": "true", "L2_ENABLE_TOOLS": "true"}),
        ("L2_C2_RETRIEVER_TOP_K", "1.5", {"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"}),
        ("L2_C2_RETRIEVER_BUDGET_MS", "x", {"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"}),
        ("L2_C2_RECENT_TOP_K", "1.5", {"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"}),
        ("L2_RAG_TOP_K", "1.5", {}),
    ])
    def test_numeric_table(self, isolated, name, bad, env):
        os.environ.update({name: bad, **env})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, f"flag/{name}").status == "FAIL"


def _production_sources() -> dict[Path, str]:
    """preflight.py を除く本番のソースを返す。"""
    import lab_lounge

    sources: dict[Path, str] = {}
    for base in lab_lounge.__path__:
        for path in Path(base).rglob("*.py"):
            if path.name != "preflight.py":
                sources[path] = path.read_text(encoding="utf-8")
    return sources


class TestNumericEnvCoverage:
    """壊れた数値は本番を例外で止める。読む経路が今の設定で有効なら FAIL、無効なら WARN にする。"""

    def test_table_matches_what_production_reads(self):
        """表から漏れると --static が緑のまま本番が落ちる。int と float を取り違えると、正しい値を赤くする。"""
        sources = _production_sources()
        found: dict[str, set[str]] = {}
        for text in sources.values():
            for kind, name in re.findall(r'\b(int|float)\(\s*os\.environ\.get\(\s*"([A-Z0-9_]+)"', text):
                found.setdefault(name, set()).add(kind)
        # 読む関数に呼び出し元がないので、壊れていても本番は落ちない。配線したら表へ足す。
        unread = {"L2_HANDRAISE_RESPONDING_PADDING_SEC"}
        assert sum(text.count("_get_handraise_responding_padding_sec") for text in sources.values()) == 1
        # 一度変数へ受けてから int() するので、上の正規表現では見つからない。
        split_read = {"L2_TTS_IRODORI_SEED"}
        table = {name: parse.__name__ for name, parse in preflight._NUMERIC_ENV}
        assert set(table) == (set(found) - unread) | split_read
        assert {name: kinds for name, kinds in found.items() if name not in unread} == {
            name: {kind} for name, kind in table.items() if name not in split_read
        }

    @pytest.mark.parametrize("name,env", [
        ("L2_LLM_TIMEOUT_SEC", {"L2_USE_REAL_LLM": "true"}),
        ("L2_ROUTER_LLM_TIMEOUT_SEC", {}),  # 既定で挙手と filler が有効
        ("L2_LLM_MAX_RETRIES", {}),
        ("L2_LLM_MAX_RETRIES", {**_RAW_LLM_OFF, "L2_USE_REAL_LLM": "true"}),
        ("L2_AGENT_INVOKE_TIMEOUT_SEC", {"L2_USE_REAL_LLM": "true", "L2_ENABLE_TOOLS": "true"}),
        ("L2_AGENT_RECURSION_LIMIT", {"L2_USE_REAL_LLM": "true", "L2_ENABLE_TOOLS": "true"}),
        ("L2_C2_RETRIEVER_TOP_K", _C2_ON),
        ("L2_C2_RETRIEVER_BUDGET_MS", _C2_ON),
        ("L2_C2_RECENT_TOP_K", _C2_ON),
        ("L2_RAG_TOP_K", {}),  # RAG が無効でも _get_rag_mode が毎ターン読む
        ("L2_TTS_IRODORI_SHORT_CHARS", {}),  # ダミー TTS でも tts.py の import で読む
        ("L2_TTS_IRODORI_SEC_PER_CHAR", {}),
        ("L2_TTS_IRODORI_MIN_SEC", {}),
        ("L2_TTS_IRODORI_TIMEOUT_SEC", _REAL_TTS),
        ("L2_TTS_IRODORI_NUM_STEPS", _REAL_TTS),
        ("L2_TTS_IRODORI_SEED", _REAL_TTS),
        ("L2_RECORD_SECONDS", {}),  # run_loop の起動で読む
        ("L2_WAKE_TIMEOUT", {}),
        ("L2_SILENCE_THRESHOLD", {"L2_VAD_BACKEND": "webrtc"}),  # .env.example の説明と違い、VAD によらず float() する
        ("L2_SILENCE_THRESHOLD", {"L2_VAD_BACKEND": "rms"}),
        ("L2_VAD_AGGRESSIVENESS", {"L2_VAD_BACKEND": "webrtc"}),
        ("L2_CONTEXT_WINDOW_SEC", {}),  # 起動スクリプトの既定の continuous で読む
        ("L2_CONTEXT_MAX_CHARS", {}),
        ("L2_HANDRAISE_LAPSE_SEC", {}),
        ("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", {}),
        ("L2_STT_COMPRESSION_RATIO_THRESHOLD", {"L2_STT_PROVIDER": "faster-whisper"}),
        ("L2_STT_LOG_PROB_THRESHOLD", {"L2_STT_PROVIDER": "faster-whisper"}),
        ("L2_STT_NO_SPEECH_THRESHOLD", {"L2_STT_PROVIDER": "faster-whisper"}),
        ("L2_BUS_CONNECT_TIMEOUT_SEC", {}),  # ダミーの構成でも publish のたびに読む
        ("L2_BUS_SOCKET_TIMEOUT_SEC", {}),
        ("L2_BUS_RETRY_AFTER_SEC", {}),
    ])
    def test_broken_value_on_a_live_path_is_fail_and_hides_the_value(self, isolated, capsys, name, env):
        os.environ.update({name: _MARK, **env})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, f"flag/{name}").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_FAILED
        assert _MARK not in capsys.readouterr().out

    @pytest.mark.parametrize("name,env", [
        ("L2_LLM_TIMEOUT_SEC", {}),  # ダミー LLM では _llm_node が呼ばない
        ("L2_ROUTER_LLM_TIMEOUT_SEC", _RAW_LLM_OFF),
        ("L2_LLM_MAX_RETRIES", _RAW_LLM_OFF),
        ("L2_AGENT_INVOKE_TIMEOUT_SEC", {"L2_USE_REAL_LLM": "true"}),  # ツールが無効
        ("L2_AGENT_INVOKE_TIMEOUT_SEC", {"L2_ENABLE_TOOLS": "true"}),  # ダミー LLM
        ("L2_AGENT_RECURSION_LIMIT", {"L2_USE_REAL_LLM": "true"}),  # ツールが無効なら Agent を作らない
        ("L2_AGENT_RECURSION_LIMIT", {"L2_ENABLE_TOOLS": "true"}),  # ダミー LLM
        ("L2_C2_RETRIEVER_TOP_K", {**_C2_ON, "L2_USE_C2_RETRIEVER": "false"}),
        ("L2_C2_RETRIEVER_BUDGET_MS", {**_C2_ON, "L2_ENABLE_RAG": "false"}),
        ("L2_C2_RECENT_TOP_K", {**_C2_ON, "L2_USE_C2_RETRIEVER": "1"}),  # graph.py は "true" だけを真と読む
        ("L2_TTS_IRODORI_TIMEOUT_SEC", {}),  # ダミー TTS
        ("L2_TTS_IRODORI_NUM_STEPS", {}),
        ("L2_TTS_IRODORI_SEED", {}),
        ("L2_VOICEPEAK_MAX_RETRIES", {}),
        ("L2_VOICEPEAK_OUTPUT_RETRY_MAX_ATTEMPTS", {}),
        ("L2_VAD_AGGRESSIVENESS", {"L2_VAD_BACKEND": "rms"}),
        ("L2_VAD_AGGRESSIVENESS", {}),  # 本番の既定は rms
        ("L2_STT_NO_SPEECH_THRESHOLD", {"L2_STT_PROVIDER": "openai"}),
        ("L2_STT_NO_SPEECH_THRESHOLD", {}),  # 本番の既定は openai
    ])
    def test_broken_value_on_an_unused_path_warns_and_does_not_stop(self, isolated, capsys, name, env):
        """使わない経路の値で配信の起動を止めない。誤りそのものは知らせる。"""
        os.environ.update({name: _MARK, **env})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, f"flag/{name}").status == "WARN"
        assert report.exit_code() == preflight.EXIT_OK
        assert _MARK not in capsys.readouterr().out

    @pytest.mark.parametrize("name", ["L2_TTS_IRODORI_TIMEOUT_SEC", "L2_TTS_IRODORI_NUM_STEPS", "L2_TTS_IRODORI_SEED"])
    @pytest.mark.parametrize("slug,status", [("mimi", "FAIL"), ("octamaid", "WARN")])
    def test_irodori_values_follow_the_characters_being_checked(self, isolated, name, slug, status):
        os.environ.update({name: _MARK, **_REAL_TTS})
        report = preflight.Report()
        preflight._check_numeric_env(report, [get_character(slug)])
        assert _status(report, f"flag/{name}").status == status

    @pytest.mark.parametrize("name", [
        "L2_VOICEPEAK_RETRY_WAIT_SEC", "L2_VOICEPEAK_MAX_RETRIES",
        "L2_VOICEPEAK_OUTPUT_RETRY_INTERVAL_SEC", "L2_VOICEPEAK_OUTPUT_RETRY_MAX_ATTEMPTS",
    ])
    def test_voicepeak_values_do_not_fail_the_run_for_a_character_off_the_air(self, isolated, name):
        """VOICEPEAK を使うのは配信に出ない ruka だけなので、別枠の FAIL に留める。"""
        os.environ.update({name: _MARK, **_REAL_TTS})
        everyone = preflight.Report()
        preflight._check_numeric_env(everyone, get_all_characters())
        check = _status(everyone, f"flag/{name}")
        assert (check.status, check.required) == ("FAIL", False)
        assert everyone.exit_code() == preflight.EXIT_OK
        only_mimi = preflight.Report()
        preflight._check_numeric_env(only_mimi, [get_character("mimi")])
        assert _status(only_mimi, f"flag/{name}").status == "WARN"

    @pytest.mark.parametrize("name", ["L2_HANDRAISE_LAPSE_SEC", "L2_HANDRAISE_LAPSE_UTTERANCE_COUNT"])
    def test_handraise_values_fail_even_with_handraise_off(self, isolated, name):
        """Dispatcher の生成で L2_USE_HANDRAISE によらず読むので、このフラグで WARN へ下げると緑のまま起動できなくなる。"""
        from lab_lounge.dispatcher import _get_handraise_config

        os.environ.update({name: _MARK, "L2_USE_HANDRAISE": "false"})
        with pytest.raises(ValueError):
            _get_handraise_config()
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, f"flag/{name}").status == "FAIL"

    def test_rag_top_k_is_read_with_rag_off(self, isolated):
        """FAIL の根拠。無効なら読まないように本番が直ったら、このテストが落ちるので判定を見直す。"""
        from lab_lounge.pipeline import _get_rag_mode

        os.environ.update({"L2_RAG_TOP_K": _MARK, "L2_ENABLE_RAG": "false"})
        with pytest.raises(ValueError):
            _get_rag_mode()
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, "flag/L2_RAG_TOP_K").status == "FAIL"

    @pytest.mark.parametrize("name", ["L2_TTS_IRODORI_SHORT_CHARS", "L2_TTS_IRODORI_SEC_PER_CHAR", "L2_TTS_IRODORI_MIN_SEC"])
    def test_values_tts_reads_at_import_stop_the_module(self, isolated, monkeypatch, name):
        """FAIL の根拠。_tts_node はダミー TTS でも tts を import するので、全ターンが落ちる。"""
        import importlib

        os.environ[name] = _MARK
        monkeypatch.delitem(sys.modules, "lab_lounge.tts", raising=False)
        with pytest.raises(ValueError):
            importlib.import_module("lab_lounge.tts")
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert _status(report, f"flag/{name}").status == "FAIL"

    @pytest.mark.parametrize("name", ["L2_TTS_IRODORI_SHORT_CHARS", "L2_HANDRAISE_LAPSE_SEC"])
    def test_main_exits_one_with_dummy_modes(self, isolated, capsys, name):
        """ダミー起動の --static でも緑にしない。実 LLM と実 TTS を問わず本番が止まる値なので。"""
        (isolated / ".env").write_text(f"REDIS_STREAM_KEY=aibyss:events-dev\n{name}={_MARK}\n", encoding="utf-8")
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert f"[FAIL ] flag/{name}:" in out
        assert _MARK not in out

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_seed_means_the_value_in_voices_json(self, isolated, value):
        """.env.example が「空=キャラ毎固定値」と説明している。int("") で赤くすると、正常な設定で配信を止める。"""
        os.environ.update({"L2_TTS_IRODORI_SEED": value, **_REAL_TTS})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert report.checks == []

    def test_setting_that_nothing_reads_has_no_row(self, isolated):
        os.environ["L2_HANDRAISE_RESPONDING_PADDING_SEC"] = _MARK
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert report.checks == []

    def test_uncommenting_the_examples_never_breaks_a_number(self, isolated):
        """.env.example のコメントを外して使う人を、数値の点検で止めない。"""
        names = {name for name, _ in preflight._NUMERIC_ENV}
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        seen = 0
        for name, value in re.findall(r"(?m)^#?\s*(L2_[A-Z0-9_]+)=(.*)$", text):
            if name in names:
                os.environ[name] = value.strip()
                seen += 1
        assert seen >= 10
        os.environ.update(_C2_ON)
        os.environ.update({"L2_USE_REAL_LLM": "true", "L2_USE_REAL_TTS": "true", "L2_STT_PROVIDER": "faster-whisper"})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert report.checks == []


class TestStaticKeepsGoing:
    """1 つの点検の例外で、残りの行と要約を失わない。起動前のゲートは、何が壊れたかを読めることが要る。"""

    @pytest.mark.parametrize("check,row", [
        ("_check_versions", "deps/uv.lock"),
        ("_check_models", "model"),
        ("_check_imports", "import"),
        ("_check_keys", "key"),
        ("_check_flags", "flag"),
        ("_check_env_quoting", "env/quoting"),
        ("_check_numeric_env", "flag/numeric"),
        ("_check_real_modes", "mode"),
        ("_check_personas", "persona"),
        ("_check_tts_config", "tts-config"),
        ("_check_readings", "tts-config/readings"),
        ("_check_filler_cache", "filler-cache"),
        ("_check_kb_index", "rag/index"),
        ("_check_endpoints", "url"),
        ("_check_stream", "stream"),
        ("_check_run_script_refusals", "run-script"),
        ("_check_loopback_proxy", "env/proxy"),
        ("_check_event_schema", "schema"),
        ("_check_stream_context", "stream_context"),
        ("_check_obs", "obs"),
    ])
    def test_a_check_that_raises_is_one_fail_row(self, isolated, check, row):
        report = preflight.Report()
        with patch.object(preflight, check, side_effect=RuntimeError(_MARK)):
            preflight.run_static(report, isolated, "dev", [get_character("mimi")], set())
        found = _status(report, row)
        assert (found.status, found.required) == ("FAIL", True)
        assert _MARK not in found.detail
        assert "RuntimeError" in found.detail
        # 例外の後ろの点検も流れている。
        assert _status(report, "flag/LANGSMITH")
        assert any(c.name == "obs" for c in report.checks)
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_main_shows_the_summary_and_keeps_the_full_text_in_the_log(self, isolated, capsys):
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "_check_models", side_effect=AttributeError(_MARK)):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert "[FAIL ] model: 点検を続けられません (AttributeError)" in out
        assert "  FAIL model" in out
        assert "[INFO ] stream:" in out
        assert _MARK not in out
        assert "詳細: logs/preflight/" in out
        logs = list((isolated / "logs" / "preflight").glob("*.log"))
        assert any(_MARK in path.read_text(encoding="utf-8") for path in logs)

    def test_models_failure_looks_at_every_provider(self, isolated):
        """どの SDK が原因かを import の行で見せるため。"""
        with patch.object(preflight, "_check_models", side_effect=AttributeError("x")), \
                patch.object(preflight, "_check_imports") as imports, patch.object(preflight, "_check_keys") as keys:
            preflight.run_static(preflight.Report(), isolated, "dev", [], {"X"})
        assert imports.call_args.args[1] == {"openai", "anthropic", "google"}
        assert keys.call_args.args[1:] == ({"openai", "anthropic", "google"}, {"X"})

    @pytest.mark.parametrize("error", [AttributeError, TypeError])
    def test_sdk_that_breaks_at_import_is_named_on_its_row(self, isolated, monkeypatch, capsys, error):
        """filler.py と router.py は ImportError しか受けない。SDK の更新で別の例外になると、_check_models で落ちる。"""
        import importlib.abc

        class Broken(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "google.genai" or name.startswith("google.genai."):
                    raise error(_MARK)
                return None

        for name in list(sys.modules):
            if name == "google.genai" or name.startswith("google.genai.") or name in ("lab_lounge.filler", "lab_lounge.router"):
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert f"[FAIL ] model: 点検を続けられません ({error.__name__})" in out
        assert "[FAIL ] import/google.genai:" in out
        assert "  FAIL model" in out
        assert "[INFO ] stream:" in out

    def test_value_that_stops_the_tts_import_keeps_the_rest(self, isolated, voices, monkeypatch, capsys):
        """実 TTS では _check_tts_config の import が ValueError になる。原因の名前は数値の行で読める。"""
        monkeypatch.delitem(sys.modules, "lab_lounge.tts", raising=False)
        _write_real_env(isolated, L2_TTS_IRODORI_SHORT_CHARS=_MARK, L2_IRODORI_VOICES_JSON=str(voices))
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert "[FAIL ] flag/L2_TTS_IRODORI_SHORT_CHARS:" in out
        assert "[FAIL ] tts-config: 点検を続けられません (ValueError)" in out
        assert "url/REDIS_URL" in out
        assert "  FAIL tts-config" in out
        assert _MARK not in out

    def test_paid_stops_before_any_charge_when_a_check_raises(self, isolated):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(isolated / "none.json"))
        with patch.object(preflight, "_check_personas", side_effect=RuntimeError("x")), \
                patch.object(preflight, "missing_services") as services, patch.object(preflight, "run_paid") as paid:
            code = preflight.main(["--paid", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        services.assert_not_called()
        paid.assert_not_called()

    @pytest.mark.parametrize("body", [
        "[]",
        '{"voices": []}',
        '{"voices": {"mimi": "x"}}',
        '{"common": [], "voices": {"mimi": {}}}',
        '{"common": {"realtime": []}, "voices": {"mimi": {}}}',
        "{broken",
    ])
    def test_voices_json_of_the_wrong_shape_stays_on_the_character_row(self, isolated, body):
        path = isolated / "voices-shape.json"
        path.write_text(body, encoding="utf-8")
        os.environ.update({"L2_USE_REAL_TTS": "true", "L2_IRODORI_VOICES_JSON": str(path)})
        report = preflight.Report()
        preflight._check_tts_config(report, [get_character("mimi"), get_character("aruka")])
        assert _status(report, "tts-config/mimi").status == "FAIL"
        # 配信に出ないキャラの行は別枠のまま。
        assert _status(report, "tts-config/aruka").required is False
        assert report.exit_code() == preflight.EXIT_FAILED


class TestModelsImportsKeys:
    def test_models_collect_every_provider_and_honour_env(self, isolated):
        os.environ.update({
            "L2_USE_LLM_ROUTER": "true", "L2_USE_INTENT_GATE": "true",
            "L2_LLM_ROUTER_MODEL": "gemini-router", "L2_INTENT_GATE_MODEL": "claude-intent",
        })
        character = SimpleNamespace(slug="x", filler_model="claude-filler")
        report = preflight.Report()
        with patch("lab_lounge.pipeline._get_llm_mode", return_value=(True, "openai", "gpt-main")):
            providers = preflight._check_models(report, [character])
        assert providers == {"openai", "anthropic", "google"}
        assert "gemini-router" in _status(report, "model/router").detail
        assert "claude-intent" in _status(report, "model/intent").detail

    def test_router_and_intent_add_their_own_provider(self, isolated):
        os.environ.update({"L2_USE_LLM_ROUTER": "true", "L2_LLM_ROUTER_MODEL": "gemini-router", "L2_USE_HANDRAISE": "false"})
        assert preflight._check_models(preflight.Report(), []) == {"google"}
        os.environ.update({"L2_USE_LLM_ROUTER": "false", "L2_USE_HANDRAISE": "true", "L2_INTENT_GATE_MODEL": "claude-intent"})
        assert preflight._check_models(preflight.Report(), []) == {"anthropic"}

    def test_imports_cover_every_sdk(self, isolated):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        seen = []

        def fake(name):
            seen.append(name)
            return MagicMock()

        with patch("importlib.import_module", fake):
            preflight._check_imports(preflight.Report(), {"openai", "anthropic", "google"})
        assert set(seen) == {
            "openai", "langchain_openai", "anthropic", "langchain_anthropic", "google.genai", "langchain_google_genai",
            "langchain.agents", "langchain_tavily", "langgraph.prebuilt",
        }

    def test_failed_import_has_no_pass_row(self, isolated):
        def fake(name):
            raise ImportError("x")

        report = preflight.Report()
        with patch("importlib.import_module", fake):
            preflight._check_imports(report, {"openai"})
        assert not any(c.name == "import" for c in report.checks)

    def test_tavily_key_is_required_when_tools_are_on(self, isolated):
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        assert _status(report, "key/tavily").status == "FAIL"

    def test_gemini_key_satisfies_google(self, isolated):
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["GEMINI_API_KEY"] = "x"
        report = preflight.Report()
        preflight._check_keys(report, {"google"}, set())
        assert _status(report, "key/google").status == "PASS"


class TestLiveness:
    def test_http_ok_judges_status_and_failure(self):
        response = MagicMock(status=500)
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response):
            assert preflight._http_ok("http://127.0.0.1:1/x") is False
        response.status = 200
        with patch("urllib.request.urlopen", return_value=response):
            assert preflight._http_ok("http://127.0.0.1:1/x") is True
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert preflight._http_ok("http://127.0.0.1:1/x") is False

    def test_tts_status_honours_the_irodori_url(self, isolated):
        os.environ["L2_TTS_IRODORI_URL"] = "http://127.0.0.1:9"
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.tts_engine_status(SimpleNamespace(tts_provider="irodori_vd"))
        assert ok.call_args.args[0] == "http://127.0.0.1:9/health"

    def test_missing_services_honours_the_c2_url(self, isolated):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"})
        os.environ["L2_C2_URL"] = "http://127.0.0.1:9"
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.missing_services([], with_pipeline=False)
        assert ok.call_args.args[0] == "http://127.0.0.1:9/health"

    def test_missing_services_does_not_double_the_slash_of_the_c2_url(self, isolated):
        os.environ.update(_C2_ON)
        os.environ["L2_C2_URL"] = "http://127.0.0.1:9/"
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.missing_services([], with_pipeline=False)
        assert ok.call_args.args[0] == "http://127.0.0.1:9/health"

    def test_c2_that_listens_but_reports_503_is_missing(self, isolated):
        """consumer が止まっても C2 は listen を続ける。接続できるだけで「いる」と言うと、記録されないまま課金が始まる。"""
        import urllib.error

        os.environ.update(_C2_ON)
        error = urllib.error.HTTPError("http://127.0.0.1:8100/health", 503, "Service Unavailable", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            assert preflight.missing_services([], with_pipeline=False) == ["C2"]

    def test_c2_that_reports_200_is_not_missing(self, isolated):
        os.environ.update(_C2_ON)
        response = MagicMock(status=200)
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response):
            assert preflight.missing_services([], with_pipeline=False) == []

    def test_check_c2_requires_primary_and_not_readonly(self, isolated):
        os.environ.update({
            "L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true",
            "L2_C2_URL": "http://a:1", "L2_C2_URL_READONLY": "http://b:2",
        })
        report = preflight.Report()
        with patch.object(preflight, "_check_c2_endpoint") as endpoint:
            preflight.check_c2(report)
        assert [(c.args[1], c.args[2], c.kwargs["required"]) for c in endpoint.call_args_list] == [
            ("c2", "http://a:1", True), ("c2/readonly", "http://b:2", False),
        ]

    def test_c2_wrong_shape_is_a_row_not_a_crash(self, isolated):
        """想定外の応答で点検全体が止まると、要約も出ない。"""
        response = MagicMock()
        response.read.return_value = b"[]"
        response.__enter__.return_value = response
        report = preflight.Report()
        with patch("urllib.request.urlopen", return_value=response):
            preflight._check_c2_endpoint(report, "c2", "http://x:1", required=True)
            preflight._check_c2_endpoint(report, "c2/readonly", "http://x:1", required=False)
        assert _status(report, "c2").status == "FAIL"
        assert _status(report, "c2/readonly").status == "WARN"


class TestMeasuresWhatProductionUses:
    """本番と同じモデルと prompt で呼んでいること。ずれると本番の破損を緑で見逃す。"""

    def test_agent_is_built_with_the_effective_model_and_persona(self, isolated, capture):
        from lab_lounge.characters import load_system_prompt
        from lab_lounge.graph import _get_agent_invoke_timeout
        from lab_lounge.pipeline import _get_llm_mode

        os.environ["L2_ENABLE_TOOLS"] = "true"
        character = get_character("mimi")
        _, provider, model = _get_llm_mode(character)
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()) as build, \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(SimpleNamespace(response="x"))) as invoke:
            preflight.check_agent(preflight.Report(), capture, character, "配信文脈マーカー", required=True)
        assert build.call_args.args[:2] == (provider, model)
        prompt = build.call_args.args[2]
        assert "配信文脈マーカー" in prompt
        assert load_system_prompt(character).strip()[:40] in prompt
        assert invoke.call_args.args[1]["messages"][0]["role"] == "user"
        assert invoke.call_args.kwargs["timeout_sec"] == _get_agent_invoke_timeout()
        # 挨拶は 2 段階、協働の 1 往復は 4 段階で足りる。上げると検証に落ち続けたときの課金が増える。
        assert 4 <= invoke.call_args.args[2]["recursion_limit"] <= 10

    def test_single_node_uses_the_effective_model_and_persona(self, isolated, capture):
        from lab_lounge.pipeline import _get_llm_mode

        character = get_character("mimi")
        _, provider, model = _get_llm_mode(character)
        fake = SimpleNamespace(text="こんにちは", input_tokens=3, output_tokens=2, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake) as call:
            preflight.check_agent(report, capture, character, "配信文脈マーカー", required=True)
        assert call.call_args.kwargs["model"] == model
        assert call.call_args.kwargs["provider"] == provider
        assert "配信文脈マーカー" in call.call_args.kwargs["system_prompt"]
        assert (report.input_tokens, report.output_tokens) == (3, 2)

    @pytest.mark.parametrize("env,model", [
        ({"L2_USE_LLM_ROUTER": "true", "L2_USE_HANDRAISE": "false", "L2_LLM_ROUTER_MODEL": "router-override"}, "router-override"),
        ({"L2_INTENT_GATE_MODEL": "intent-override"}, "intent-override"),
    ])
    def test_router_probes_the_model_from_env(self, isolated, capture, env, model):
        os.environ.update(env)
        with patch("lab_lounge.router._call_router_llm", return_value="mimi") as call:
            preflight.check_router(preflight.Report(), capture)
        assert call.call_args.args[0] == model

    def test_run_paid_forwards_stream_context_everywhere(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        names = ("check_router", "check_filler", "check_tavily", "check_embeddings", "check_c2", "check_agent", "check_pipeline", "check_collab")
        patches = {n: patch.object(preflight, n) for n in names}
        mocks = {n: p.start() for n, p in patches.items()}
        try:
            with patch.object(preflight, "check_tts_engine", return_value=True) as engine, \
                    patch("lab_lounge.stream_context.load_stream_context", return_value="CTX"):
                preflight.run_paid(preflight.Report(), capture, get_all_characters(), with_pipeline=True)
        finally:
            for p in patches.values():
                p.stop()
        assert all(c.args[3] == "CTX" for c in mocks["check_agent"].call_args_list)
        assert all(c.args[4] == "CTX" for c in mocks["check_pipeline"].call_args_list)
        assert all(c.args[5] == "CTX" for c in mocks["check_collab"].call_args_list)
        required = {c.args[1].slug: c.kwargs["required"] for c in engine.call_args_list}
        assert required["aruka"] is False
        assert required["mimi"] is True

    def test_run_paid_says_when_no_collab_pair_exists(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        mocks, report = _run_paid_with_fakes(capture, with_pipeline=True, characters=[get_character("mimi")])
        mocks["check_collab"].assert_not_called()
        assert _status(report, "collab").status == "SKIP"

    def test_pipeline_forwards_stream_context(self, isolated, capture):
        with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result()) as run:
            preflight.check_pipeline(preflight.Report(), capture, get_character("mimi"), "s", "CTX", required=True)
        assert run.call_args.kwargs["stream_context"] == "CTX"

    def test_collab_pairs_cover_every_provider_on_both_sides(self):
        from lab_lounge.pipeline import _get_llm_mode

        provider = {c.slug: _get_llm_mode(c)[1] for c in get_all_characters()}
        pairs = preflight._collab_pairs(get_all_characters(), "all")
        assert {provider[a.slug] for a, _ in pairs} == {provider[b.slug] for _, b in pairs} == {"openai", "anthropic", "google"}


# ─── モックに隠れていた穴（呼び出しの形、接続先、本番との読み方のずれ） ───────
#
# MagicMock は引数を何でも受け、戻り値も決まっているので、接続先や引数を取り違えても緑のままになる。
# --paid は課金が走ってから初めて本物を呼ぶので、ここで本番のシグネチャと本番の関数が読む値に照合する。

class TestCallShapesMatchProduction:
    """呼び出しの引数を本番のシグネチャで固定する。

    autospec にすると、本番と合わない呼び出しは TypeError になり、点検の except で FAIL の行に変わる。
    そのため PASS の行が出れば、引数が本番と合っている。--paid の最初の実行で課金してから落ちるのを防ぐ。
    ReqClient は **kwargs を受け、TavilySearch は知らない引数を黙って無視するので、autospec では見つからない。
    TestConnectionTargets と test_tavily_arguments_match_production で補う。
    """

    def test_router(self, isolated, capture):
        os.environ.update({"L2_USE_LLM_ROUTER": "true", "L2_USE_HANDRAISE": "false"})
        report = preflight.Report()
        with patch("lab_lounge.router._call_router_llm", autospec=True, return_value="mimi"):
            preflight.check_router(report, capture)
        assert _status(report, "router").status == "PASS", _status(report, "router").detail

    def test_filler(self, isolated, capture):
        report = preflight.Report()
        with patch("lab_lounge.filler._call_filler_llm", autospec=True, return_value="つぶやき"), \
                patch("lab_lounge.filler._build_filler_prompt", autospec=True, return_value="P"):
            preflight.check_filler(report, capture, get_character("mimi"), required=True)
        assert _status(report, "filler/mimi").status == "PASS", _status(report, "filler/mimi").detail

    def test_agent_path(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", autospec=True, return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", autospec=True,
                      return_value=_agent_result(SimpleNamespace(response="x"))):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == "PASS", _status(report, "main/mimi").detail

    def test_single_node_path(self, isolated, capture):
        fake = SimpleNamespace(text="こんにちは", input_tokens=1, output_tokens=1, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", autospec=True, return_value=fake):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == "PASS", _status(report, "main/mimi").detail

    def test_pipeline(self, isolated, capture):
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", autospec=True, return_value=_pipeline_result()):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "PASS", _status(report, "pipeline/mimi").detail

    def test_embeddings(self, isolated, capture):
        os.environ.update({"L2_ENABLE_RAG": "true", "L2_ENABLE_TOOLS": "true"})
        report = preflight.Report()
        with patch("lab_lounge.retriever._embed", autospec=True, return_value=[0.1]):
            preflight.check_embeddings(report, capture)
        assert _status(report, "embeddings").status == "PASS", _status(report, "embeddings").detail

    def test_collab(self, isolated, capture):
        """モデルがツールを呼ばない形で流すと、Agent の 2 つの入口と _ask_character_impl の直接呼びを一度に通せる。"""
        from lab_lounge.mcp_servers import ask_character as ac

        def fake_invoke(*_args, **_kwargs):
            ac._on_tts_chunk_var.get()("u", "こんにちは", True, "chisame")
            for message in _COLLAB_OK_LOGS:
                capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
            return _agent_result(SimpleNamespace(response="x"))

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", autospec=True, return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", autospec=True, side_effect=fake_invoke), \
                patch("lab_lounge.mcp_servers.ask_character._ask_character_impl", autospec=True, return_value="応答") as direct:
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        direct.assert_called_once()
        check = _status(report, "collab/sakura->chisame")
        assert check.status == "WARN", check.detail

    def test_obs_client(self, isolated):
        """本物のクラスを spec にする。SDK の更新でメソッドの名前や引数が変わると、ここで行が WARN に変わる。"""
        pytest.importorskip("obsws_python")
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        report = preflight.Report()
        with patch("obsws_python.ReqClient", autospec=True) as client:
            client.return_value.get_group_scene_item_list.return_value = SimpleNamespace(
                scene_items=[{"sourceName": "mimi_neutral", "sceneItemId": 1}],
            )
            preflight._check_obs(report, [get_character("mimi")])
        assert _status(report, "obs").status == "PASS", _status(report, "obs").detail


def test_tavily_arguments_match_production(isolated, capture):
    """check_tavily の docstring の「本番と同じ引数」を、本番の関数と比べて確かめる。

    TavilySearch は知らない引数を黙って無視するので、綴りを誤ると本番と別の条件を測ったまま緑になる。
    """
    from lab_lounge.mcp_servers.web_search import _web_search_impl

    os.environ.update({"L2_ENABLE_TOOLS": "true", "TAVILY_API_KEY": "dummy"})
    with patch("langchain_tavily.TavilySearch") as search:
        search.return_value.invoke.return_value = {"results": [{"title": "t"}]}
        _web_search_impl("q", 1)
        preflight.check_tavily(preflight.Report(), capture)
    # 2 回呼ばれたことの確認も兼ねる。点検が SKIP すると 1 回になり、ここで落ちる。
    production, probe = search.call_args_list
    assert probe == production


class TestConnectionTargets:
    """接続先を読み違えると、本番と別の相手を測って緑を返す。戻り値だけを見るモックでは気づけない。"""

    def test_missing_services_honours_the_redis_url(self, isolated):
        """dev の Redis が別ポートでも、既定ポートの Redis が動いていれば前提の確認を通ってしまう。"""
        os.environ["REDIS_URL"] = "redis://127.0.0.1:16379"
        with patch("redis.Redis.from_url") as from_url:
            preflight.missing_services([], with_pipeline=True)
        assert from_url.call_args.args[0] == "redis://127.0.0.1:16379"

    def test_obs_connects_to_the_url_it_was_given(self, isolated, monkeypatch):
        """OBS は別の機械にあることがある。手元へ繋ぎに行くと、配信で使う OBS を測れない。"""
        os.environ["L2_OBS_WS_URL"] = "ws://192.0.2.10:4466"
        fake = MagicMock()
        fake.ReqClient.return_value.get_group_scene_item_list.return_value = SimpleNamespace(
            scene_items=[{"sourceName": "mimi_neutral", "sceneItemId": 1}],
        )
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        preflight._check_obs(preflight.Report(), [get_character("mimi")])
        kwargs = fake.ReqClient.call_args.kwargs
        assert (kwargs["host"], kwargs["port"]) == ("192.0.2.10", 4466)

    def test_c2_endpoint_calls_what_the_retriever_calls(self, isolated):
        """本番の C2Retriever は httpx、点検は urllib で呼ぶので、URL の組み立てが別々にある。接続先、パス、クエリの名前を比べる。

        既定とは別のポートを渡すので、渡された URL を使わない誤りも見逃さない。
        """
        from urllib.parse import parse_qs, urlsplit

        from lab_lounge.retriever import C2Retriever

        base = "http://127.0.0.1:18100/"
        with patch("httpx.Client") as client:
            get = client.return_value.__enter__.return_value.get
            get.return_value.json.return_value = {"results": []}
            C2Retriever(base).retrieve("q", top_k=1)
        response = MagicMock()
        response.read.return_value = b'{"search_mode": "semantic"}'
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            preflight._check_c2_endpoint(preflight.Report(), "c2", base, required=True)
        probe = urlsplit(urlopen.call_args.args[0])
        assert f"{probe.scheme}://{probe.netloc}{probe.path}" == get.call_args.args[0]
        assert set(parse_qs(probe.query)) == set(get.call_args.kwargs["params"])

    def test_voicepeak_status_finds_the_configured_executable(self, isolated):
        """PATH の外に置いた VOICEPEAK を設定で指しても、前提サービスなしで --paid が止まらないこと。

        PATH を空にするので、VOICEPEAK の入った機械でも、設定を読まない誤りを見逃さない。
        """
        os.environ["PATH"] = ""
        os.environ["L2_TTS_VOICEPEAK_PATH"] = sys.executable
        assert preflight.tts_engine_status(SimpleNamespace(tts_provider="voicepeak")) == (True, "voicepeak")


class TestFlagRulesDriftFromProduction:
    """真偽の読み方が本番とずれると、本番がダミーや記憶なしで動く設定を緑にする。本番の関数と比べて固定する。"""

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "YES", "on", "false", "0", "", " true"])
    def test_is_truthy_matches_every_reader(self, isolated, value):
        from lab_lounge.debug import is_debug_enabled
        from lab_lounge.graph import _is_rag_enabled, _is_tools_enabled
        from lab_lounge.pipeline import _get_llm_mode, _get_rag_mode, _get_tts_mode

        os.environ.update({
            "L2_ENABLE_TOOLS": value, "L2_ENABLE_RAG": value, "L2_USE_REAL_LLM": value,
            "L2_USE_REAL_TTS": value, "L2_DEBUG_ARTIFACTS": value,
        })
        expected = preflight._is_truthy(value)
        assert _is_tools_enabled() is expected
        assert _is_rag_enabled() is expected
        assert _get_rag_mode()[0] is expected
        assert _get_llm_mode()[0] is expected
        assert _get_tts_mode()[0] is expected
        assert is_debug_enabled() is expected

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "false", "", " true"])
    def test_c2_flag_matches_build_retriever(self, isolated, value):
        """graph.py の実物で確かめる。読み方が変わると、本番が使う C2 を点検が SKIP する。"""
        from lab_lounge.graph import _build_retriever_from_env

        os.environ.update({
            "L2_ENABLE_RAG": "true", "L2_ENABLE_TOOLS": "true", "L2_USE_C2_RETRIEVER": value, "L2_USE_C2_RECENT": "false",
        })
        # LocalRetriever は索引のファイルを読むので差し替える。単独で返れば C2 を使わない構成。
        with patch("lab_lounge.retriever.LocalRetriever") as local:
            built = _build_retriever_from_env("unused")
        assert preflight._c2_retriever_enabled() is (built is not local.return_value)


def _production_voicevox_url(output_dir):
    """本番の _call_voicevox が接続する URL を、合成せずに取り出す。"""
    from lab_lounge import tts

    with patch.object(tts, "_generate_voicevox_single_file", return_value=(1, 24000)) as generate:
        tts._call_voicevox("こんにちは", voice="3", speaker="x", output_dir=str(output_dir))
    return generate.call_args.kwargs["voicevox_url"]


def _production_irodori_url():
    """本番の _irodori_url から、/synthesize を除いた土台の URL を取り出す。"""
    from lab_lounge.tts import _irodori_url

    return _irodori_url().removesuffix("/synthesize")


def _production_c2_url():
    """本番の _build_retriever_from_env が C2Retriever へ渡す URL を取り出す。L2_USE_C2_RETRIEVER を真にして呼ぶ。"""
    from lab_lounge.graph import _build_retriever_from_env

    with patch("lab_lounge.retriever.LocalRetriever"), patch("lab_lounge.retriever.C2Retriever") as c2:
        _build_retriever_from_env("unused")
    return c2.call_args.args[0]


class TestDefaultUrlsMatchProduction:
    """既定の URL は点検と本番に別々に書いてある。本番だけ変えると、点検は古いポートを測り続ける。"""

    def test_voicevox_liveness(self, isolated):
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.tts_engine_status(SimpleNamespace(tts_provider="voicevox"))
        assert ok.call_args.args[0] == _production_voicevox_url(isolated).rstrip("/") + "/version"

    def test_irodori_liveness(self, isolated):
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.tts_engine_status(SimpleNamespace(tts_provider="irodori_vd"))
        assert ok.call_args.args[0] == _production_irodori_url() + "/health"

    def test_c2_liveness_and_probe(self, isolated):
        os.environ.update({"L2_ENABLE_RAG": "true", "L2_ENABLE_TOOLS": "true", "L2_USE_C2_RETRIEVER": "true"})
        expected = _production_c2_url()
        with patch.object(preflight, "_http_ok", return_value=True) as ok:
            preflight.missing_services([], with_pipeline=False)
        assert ok.call_args.args[0] == f"{expected}/health"
        with patch.object(preflight, "_check_c2_endpoint") as endpoint:
            preflight.check_c2(preflight.Report())
        assert endpoint.call_args.args[2] == expected

    def test_endpoint_rows(self, isolated):
        """url/ の行に出るポートも、本番が既定で繋ぐ先と同じであること。読み手はこの行で接続先を確かめる。"""
        from urllib.parse import urlsplit

        from lab_lounge.bus import _redis_url

        os.environ.pop("REDIS_URL")
        os.environ["L2_USE_C2_RETRIEVER"] = "true"
        production = {
            "REDIS_URL": _redis_url(),
            "L2_C2_URL": _production_c2_url(),
            "L2_TTS_VOICEVOX_URL": _production_voicevox_url(isolated),
            "L2_TTS_IRODORI_URL": _production_irodori_url(),
        }
        report = preflight.Report()
        preflight._check_endpoints(report)
        for name, url in production.items():
            assert _status(report, f"url/{name}").detail == f"loopback:{urlsplit(url).port}", name


class TestTruncation:
    def test_anthropic_stop_reason_is_seen(self):
        summary = preflight._summarize_agent_result(
            _agent_result(SimpleNamespace(response="x"), finish="max_tokens", meta_key="stop_reason"),
        )
        assert any("切れました" in p for p in preflight._agent_problems(summary, []))

    def test_judged_on_the_last_step_and_tokens_add_up(self):
        summary = preflight._summarize_agent_result(
            _agent_result(SimpleNamespace(response="x"), finishes=["tool_calls", "length"]),
        )
        assert any("切れました" in p for p in preflight._agent_problems(summary, []))
        assert summary["input_tokens"] == 20
        assert summary["output_tokens"] == 10


class TestLogWindow:
    """前の点検のログを、後の点検の証拠に流用しないこと。"""

    def test_pipeline_does_not_borrow_success_logs(self, isolated, capture):
        os.environ["L2_USE_REAL_LLM"] = os.environ["L2_ENABLE_TOOLS"] = "true"
        capture.emit(_log_record("lab_lounge.graph", logging.INFO, "Agent 実行完了 [character=mimi]: source=structured_response"))
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result()):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "FAIL"

    def test_pipeline_does_not_inherit_failures(self, isolated, capture):
        capture.emit(_log_record("lab_lounge.graph", logging.ERROR, "Agent 実行失敗 [character=chisame]: 400"))
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result()):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "PASS"

    def test_agent_does_not_inherit_failures(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        capture.emit(_log_record("lab_lounge.graph", logging.WARNING, "ask_character ツール読み込み失敗 %s", ("x",)))
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(SimpleNamespace(response="x"))):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == "PASS"

    def _collab_with_logs(self, capture, new_logs):
        from lab_lounge.mcp_servers import ask_character as ac

        def fake_invoke(*args, **kwargs):
            ac._on_tts_chunk_var.get()("u", "こんにちは", True, "chisame")
            for message in new_logs:
                capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
            return _agent_result(SimpleNamespace(response="x"), tool_calls=[{"name": "ask_character_tool"}])

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke):
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        return _status(report, "collab/sakura->chisame").status

    def test_collab_does_not_borrow_success_logs(self, isolated, capture):
        """run_paid の実際の順序: pipeline/chisame の成功の後に、協働先が Agent を通らない。"""
        capture.emit(_log_record("lab_lounge.graph", logging.INFO, _COLLAB_OK_LOGS[1]))
        assert self._collab_with_logs(capture, [_COLLAB_OK_LOGS[0]]) == "FAIL"

    def test_collab_does_not_inherit_failures(self, isolated, capture):
        capture.emit(_log_record("lab_lounge.graph", logging.ERROR, "Agent 実行失敗 [character=mimi]: 400"))
        assert self._collab_with_logs(capture, _COLLAB_OK_LOGS) == "PASS"


class TestPipelineEvents:
    def test_other_events_are_ignored_and_missing_text_is_fail(self, isolated, capture):
        real_shape = [{"type": "llm.final", "payload": {}}, *_pipeline_result().events]
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=SimpleNamespace(speaker="mimi", events=real_shape)):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "PASS"
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=SimpleNamespace(speaker="mimi", events=[{"type": "tts.done", "payload": {}}])):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "FAIL"

    @pytest.mark.parametrize("env", [{"L2_USE_REAL_LLM": "true"}, {}])
    def test_structured_is_required_only_on_the_agent_path(self, isolated, capture, env):
        os.environ.update(env)
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result()):
            preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=True)
        assert _status(report, "pipeline/mimi").status == "PASS"


class TestCollabWiring:
    def _collab(self, capture, *, emit=None, tools=("ask_character_tool",)):
        from lab_lounge.mcp_servers import ask_character as ac

        seen = {}

        def fake_invoke(*args, **kwargs):
            seen["session_id"] = kwargs["session_id"]
            seen["content"] = args[1]["messages"][0]["content"]
            on_chunk = ac._on_tts_chunk_var.get()
            for chunk in emit if emit is not None else [("u", "こんにちは", True, "chisame")]:
                on_chunk(*chunk)
            for message in _COLLAB_OK_LOGS:
                capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
            return _agent_result(SimpleNamespace(response="まとめ"), tool_calls=[{"name": n} for n in tools])

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()) as build, \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke), \
                patch("lab_lounge.mcp_servers.ask_character._ask_character_impl", return_value="応答") as direct, \
                patch("lab_lounge.mcp_servers.ask_character.wait_bg_tts_complete") as wait, \
                patch("lab_lounge.mcp_servers.ask_character.set_ask_character_context", wraps=ac.set_ask_character_context) as context:
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "preflight-x", "CTX")
        return _status(report, "collab/sakura->chisame"), SimpleNamespace(seen=seen, direct=direct, wait=wait, context=context, build=build)

    def test_caller_chunks_do_not_count_as_target_voice(self, isolated, capture):
        check, _ = self._collab(capture, emit=[("u", "導入", False, "sakura")])
        assert check.status == "FAIL"

    def test_broken_chunk_is_fail(self, isolated, capture):
        check, _ = self._collab(capture, emit=[("u", '{"response": "深海の', True, "chisame")])
        assert check.status == "FAIL"

    def test_context_session_and_wait(self, isolated, capture):
        os.environ["L2_TTS_OUTPUT_DIR"] = "./preflight-audio"
        _, env = self._collab(capture)
        kwargs = env.context.call_args.kwargs
        assert kwargs["caller_slug"] == "sakura"
        assert kwargs["common"]["stream_id"] == "preflight-x"
        assert kwargs["tts_output_dir"] == "./preflight-audio"
        env.wait.assert_called_once_with(kwargs["common"]["session_id"])
        # 打ち切ったときに本番と同じく協働先の TTS を止めるため、session_id を渡す。
        assert env.seen["session_id"] == kwargs["common"]["session_id"]
        assert "ask_character" in env.seen["content"] and "chisame" in env.seen["content"]
        assert "CTX" in env.build.call_args.args[2]

    def test_other_tool_is_not_treated_as_the_collab_tool(self, isolated, capture):
        _, env = self._collab(capture, tools=("web_search_tool",))
        env.direct.assert_called_once()
        assert env.direct.call_args.args[0] == "chisame"

    def test_proof_must_name_the_target(self):
        logs = ["導入セリフ生成完了: [sakura] x", "Agent 実行完了 [character=sakura]: source=messages"]
        assert any("Agent を通っていません" in p for p in preflight._collab_problems(logs, "chisame"))


class TestRequiredSplit:
    """別枠のキャラの故障で配信前の判定が赤くならず、必須のキャラの故障は必ず赤くなること。"""

    def test_filler(self, isolated, capture):
        report = preflight.Report()
        with patch("lab_lounge.filler._call_filler_llm", return_value=None), patch("lab_lounge.filler._build_filler_prompt", return_value="P"):
            preflight.check_filler(report, capture, get_character("aruka"), required=False)
        assert _status(report, "filler/aruka").status == "FAIL"
        assert report.exit_code() == 0

    def test_raw_exception_and_validate(self, isolated, capture):
        def boom():
            raise TypeError("x")

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "a", "m", boom, required=False)
        preflight._check_raw_llm(report, capture, "b", "m", lambda: "x", required=False, validate=lambda _t: "bad")
        assert [c.status for c in report.checks] == ["FAIL", "FAIL"]
        assert report.exit_code() == 0

    def test_agent(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", side_effect=RuntimeError("x")):
            preflight.check_agent(report, capture, get_character("aruka"), None, required=False)
        assert report.exit_code() == 0
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(None)):
            preflight.check_agent(report, capture, get_character("aruka"), None, required=False)
        assert _status(report, "main/aruka").status == "FAIL"
        assert report.exit_code() == 0

    def test_pipeline(self, isolated, capture):
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", side_effect=RuntimeError("x")):
            preflight.check_pipeline(report, capture, get_character("aruka"), "s", None, required=False)
        assert report.exit_code() == 0
        report = preflight.Report()
        with patch("lab_lounge.pipeline.run_pipeline", return_value=SimpleNamespace(speaker="other", events=[])):
            preflight.check_pipeline(report, capture, get_character("aruka"), "s", None, required=False)
        assert _status(report, "pipeline/aruka").status == "FAIL"
        assert report.exit_code() == 0

    @pytest.mark.parametrize("slug,code", [("mimi", 1), ("aruka", 0)])
    def test_missing_persona(self, isolated, slug, code):
        character = SimpleNamespace(slug=slug, system_prompt_file="no-such.txt", sample_fallback="no-such-sample.md")
        report = preflight.Report()
        preflight._check_personas(report, [character])
        assert report.exit_code() == code

    def test_present_persona_has_no_row(self, isolated):
        report = preflight.Report()
        with patch("lab_lounge.characters._PROMPTS_DIR", Path(isolated)):
            (isolated / "ok.md").write_text("x", encoding="utf-8")
            preflight._check_personas(report, [SimpleNamespace(slug="mimi", system_prompt_file="ok.md", sample_fallback="no")])
        assert report.checks == []

    def test_reference_wav_of_non_required(self, isolated):
        os.environ["L2_USE_REAL_TTS"] = "true"
        path = isolated / "v.json"
        path.write_text(json.dumps({"voices": {"a": {"ref_wav": "no-such.wav"}}}), encoding="utf-8")
        os.environ["L2_IRODORI_VOICES_JSON"] = str(path)
        report = preflight.Report()
        preflight._check_tts_config(report, [SimpleNamespace(slug="aruka", tts_provider="irodori_vd", tts_voice="a")])
        assert report.exit_code() == 0


class TestTavilyShapes:
    def test_constructor_failure_is_fail(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        with patch("langchain_tavily.TavilySearch", side_effect=ValueError("TAVILY_API_KEY is required")):
            preflight.check_tavily(report, capture)
        assert _status(report, "tavily").status == "FAIL"

    @pytest.mark.parametrize("value", [{"error": "quota"}, "No search results"])
    def test_non_exception_shapes_are_fail(self, isolated, capture, value):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        search = MagicMock()
        search.return_value.invoke.return_value = value
        report = preflight.Report()
        with patch("langchain_tavily.TavilySearch", search):
            preflight.check_tavily(report, capture)
        assert _status(report, "tavily").status == "FAIL"


class TestRedactionDetails:
    @pytest.mark.parametrize("name", [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY",
        "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "VOYAGE_API_KEY", "L2_PORCUPINE_ACCESS_KEY", "L2_OBS_WS_PASSWORD",
    ])
    def test_each_secret_name_is_redacted_by_value(self, isolated, name):
        """定数をループすると、一覧から消えた名前はテストからも消える。名前は直書きする。"""
        os.environ[name] = "plain-looking-value"
        assert "plain-looking-value" not in preflight.redact("x plain-looking-value y")

    @pytest.mark.parametrize("text,hidden", [
        ("tvly-abcdefgh", "abcdefgh"), ("lsv2_pt_abcdefgh", "abcdefgh"), ("pa-abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
        ("sk-abc12", "abc12"), ("ghp_abcdefgh", "abcdefgh"), ("eyJhbGciOiJI.eyJzdWIi.abc", "eyJhbGciOiJI"),
    ])
    def test_key_shapes(self, text, hidden):
        assert hidden not in preflight.redact(text)

    def test_ordinary_words_are_not_hidden(self):
        assert preflight.redact("task-abcd max token count") == "task-abcd max token count"

    @pytest.mark.parametrize("text", ["10.1.2.3", "172.16.1.2", "172.31.1.2", "192.168.1.2", "100.100.1.2"])
    def test_private_ranges(self, text):
        assert preflight.redact(text) == "[LAN]"

    def test_home_is_hidden_and_short_values_are_left_alone(self, isolated, monkeypatch):
        os.environ["L2_OBS_WS_PASSWORD"] = "abc"
        assert preflight.redact("abc") == "abc"
        # 実物のホームに頼ると、HOME=/ の環境やホームが決まらない環境でこのテストだけが赤くなる。
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: "/home/dummyuser"))
        assert preflight.redact("/home/dummyuser/x") == "~/x"

    @pytest.mark.parametrize("value,shown", [
        ("  padded-value  ", "padded-value"), ("a b/c+d", "a%20b%2Fc%2Bd"), ("a b/c+d", "a+b%2Fc%2Bd"),
    ])
    def test_value_variants(self, isolated, value, shown):
        os.environ["L2_OBS_WS_PASSWORD"] = value
        assert shown not in preflight.redact(f"x {shown} y")

    def test_secret_with_a_newline_does_not_survive_the_one_liner(self, isolated):
        os.environ["L2_OBS_WS_PASSWORD"] = "line1secret\nline2secret"
        assert "line1secret" not in preflight._one_line("bad line1secret\nline2secret")

    def test_one_line_format(self):
        text = preflight._one_line("a\nb " + "x" * 500)
        assert "\n" not in text and len(text) <= preflight._MAX_DETAIL_CHARS

    def test_emit_redacts_on_its_own(self, isolated, capsys):
        os.environ["OPENAI_API_KEY"] = SECRET
        preflight._emit(f"line {SECRET}")
        assert SECRET not in capsys.readouterr().out

    def test_key_row_never_holds_the_value(self, isolated):
        os.environ["L2_OBS_WS_PASSWORD"] = SECRET
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        assert _status(report, "key/L2_OBS_WS_PASSWORD").detail == "set"

    @pytest.mark.parametrize("module", [
        "openai", "anthropic", "google", "httpx", "httpx2", "httpcore", "httpcore2", "requests", "tavily",
        "langchain_openai", "langchain_anthropic", "langchain_google_genai", "langchain_tavily",
    ])
    def test_module_only_api_errors_hide_the_body(self, module):
        """接続系の例外は status_code を持たない。本文には URL や入力が入りうる。"""
        error = type("Err", (Exception,), {"__module__": f"{module}._x"})(BODY)
        assert "BODYSENTINEL" not in preflight._describe_exc(error)

    def test_response_only_api_errors_hide_the_body(self):
        error = type("Err", (Exception,), {"response": object()})(BODY)
        assert "BODYSENTINEL" not in preflight._describe_exc(error)

    def test_describe_reads_the_error_body_and_drops_long_or_odd_values(self):
        error = type("Err", (Exception,), {"status_code": 400, "body": {"error": {"code": "bad", "param": "p" * 100, "type": object()}}})("x")
        described = preflight._describe_exc(error)
        assert "code=bad" in described and "ppp" not in described and "type=" not in described

    def test_obs_failure_text_is_not_shown(self, isolated, monkeypatch, capsys):
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        fake = MagicMock()
        fake.ReqClient.side_effect = ConnectionRefusedError("obs-pc.local:4455")
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        preflight._check_obs(preflight.Report(), [])
        assert "obs-pc" not in capsys.readouterr().out

    def test_obs_uses_the_password_and_flags_empty_groups(self, isolated, monkeypatch):
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        os.environ["L2_OBS_WS_PASSWORD"] = "pw"
        fake = MagicMock()
        fake.ReqClient.return_value.get_group_scene_item_list.return_value = SimpleNamespace(scene_items=[])
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        assert fake.ReqClient.call_args.kwargs["password"] == "pw"
        assert "mimi" in _status(report, "obs").detail

    def test_display_path_hides_outside_paths(self, isolated, tmp_path_factory):
        other = tmp_path_factory.mktemp("o") / "current.md"
        assert preflight._display_path(other, isolated) == "current.md"
        assert preflight._display_path(isolated / "a" / "b.md", isolated) == "a/b.md"


class TestCli:
    """_cli は os._exit で終わるので、別プロセスで確かめる。実サービスへは接続しない。"""

    _DROP = ("L2_", "OPENAI", "ANTHROPIC", "GOOGLE", "TAVILY", "LANGSMITH")

    def _env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(self._DROP)}
        env.update(PYTHON_DOTENV_DISABLED="1", REDIS_URL="redis://127.0.0.1:6", PYTHONIOENCODING="utf-8", **extra)
        return env

    def _run(self, code, cwd):
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                              env=self._env(), cwd=cwd, timeout=60, check=False)

    def test_catches_an_unexpected_exception_without_the_value(self, tmp_path):
        code = (
            "from lab_lounge import preflight as p\n"
            f"def boom(*a, **k):\n    raise RuntimeError({_TEXT_REDACT_CANNOT_HIDE!r})\n"
            "p.main = boom\np._cli()\n"
        )
        done = self._run(code, tmp_path)
        assert done.returncode == 1
        assert "SENTINELpw" not in done.stdout + done.stderr
        assert "Traceback" not in done.stderr
        assert "想定外のエラー: RuntimeError" in done.stdout

    def test_exits_without_joining_workers(self, tmp_path):
        """打ち切った Agent の worker が残っていても、結果を出したら終わる。"""
        code = (
            "import time\n"
            "from concurrent.futures import ThreadPoolExecutor\n"
            "from lab_lounge import preflight as p\n"
            "ex = ThreadPoolExecutor(1)\nex.submit(time.sleep, 300)\n"
            "p.main = lambda *a, **k: 0\np._cli()\n"
        )
        assert self._run(code, tmp_path).returncode == 0

    def test_passes_exit_codes_and_silences_late_logs(self, tmp_path):
        code = (
            "import logging\nfrom lab_lounge import preflight as p\n"
            "def m(*a, **k):\n    raise SystemExit(64)\n"
            "p.main = m\n"
            "import os\n_real = os._exit\n"
            "def _late(code):\n    logging.getLogger('lab_lounge.x').error('LATEBODY')\n    import sys; sys.stderr.flush(); _real(code)\n"
            "os._exit = _late\np._cli()\n"
        )
        done = self._run(code, tmp_path)
        assert done.returncode == 64
        assert "LATEBODY" not in done.stderr

    def test_keyboard_interrupt_is_130(self, tmp_path):
        code = "from lab_lounge import preflight as p\ndef m(*a, **k):\n    raise KeyboardInterrupt\np.main = m\np._cli()\n"
        assert self._run(code, tmp_path).returncode == 130

    def test_module_entry_starts_and_returns_the_usage_code(self, tmp_path):
        """`python -m lab_lounge.preflight` が起動でき、使い方の誤りが 64 で終わること。

        _cli を通ることは test_module_entry_hides_an_unexpected_exception、cp932 は TestStdoutEncoding が確かめる。
        """
        env = self._env()
        env["PYTHONIOENCODING"] = "cp932"
        done = subprocess.run(
            [sys.executable, "-m", "lab_lounge.preflight", "--static", "--profile", "dev", "--characters", "nobody"],
            capture_output=True, text=True, encoding="cp932", errors="replace", env=env, cwd=tmp_path, timeout=120, check=False,
        )
        assert done.returncode == 64
        assert "Traceback" not in done.stderr

    def test_cli_sentinel_is_not_hidden_by_redact(self):
        """センチネルが redact で消える形だと、_cli が例外文を表示しても、値を探すテストが落ちない。"""
        assert "SENTINELpw" in preflight.redact(_TEXT_REDACT_CANNOT_HIDE)


_LOCK_WITH_EXTRAS = """
[[package]]
name = "lab-lounge"
version = "0.1.0"
source = { editable = "." }
dependencies = [{ name = "a", extra = ["fast"] }, { name = "dual", version = "2.0" }]

[[package]]
name = "a"
version = "1.0"

[package.optional-dependencies]
fast = [{ name = "wanted" }]
slow = [{ name = "unwanted" }]

[[package]]
name = "wanted"
version = "1.0"

[[package]]
name = "unwanted"
version = "1.0"

[[package]]
name = "dual"
version = "1.0"

[[package]]
name = "dual"
version = "2.0"
"""


def test_closure_follows_only_the_requested_extra_and_pins_the_version():
    wanted = preflight.locked_closure(tomllib.loads(_LOCK_WITH_EXTRAS))
    assert wanted == {"a": {"1.0"}, "wanted": {"1.0"}, "dual": {"2.0"}}


class TestSmallChecks:
    @pytest.mark.parametrize("value", ["TRUE", "Yes", ""])
    def test_valid_spellings_do_not_warn(self, isolated, value):
        os.environ["L2_USE_HANDRAISE"] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert _status(report, "flag/L2_USE_HANDRAISE").status == "INFO"

    def test_porcupine_without_a_key_warns(self, isolated):
        os.environ["L2_WAKE_BACKEND"] = "porcupine"
        report = preflight.Report()
        preflight._check_flags(report)
        assert any(c.name == "flag/L2_WAKE_BACKEND" and c.status == "WARN" for c in report.checks)

    def test_prod_profile_on_the_dev_stream_warns(self, isolated):
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        report = preflight.Report()
        preflight._check_stream(report, "prod")
        assert _status(report, "stream").status == "WARN"

    @pytest.mark.parametrize("name", ["REDIS_URL", "L2_C2_URL", "L2_C2_URL_READONLY", "L2_TTS_VOICEVOX_URL", "L2_TTS_IRODORI_URL"])
    def test_every_endpoint_is_checked_and_localhost_warns(self, isolated, name):
        os.environ[name] = "redis://localhost:1234" if name == "REDIS_URL" else "http://localhost:1234"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, f"url/{name}").status == "WARN"

    @pytest.mark.parametrize("value", ["http://127.0.0.1:8100", "http://[::1]:8100"])
    def test_loopback_kind_is_reported(self, isolated, value):
        os.environ["L2_C2_URL"] = value
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_C2_URL").detail == "loopback:8100"

    def test_url_without_a_host_is_fail(self, isolated):
        os.environ.update(L2_C2_URL="not-a-url", L2_ENABLE_TOOLS="true", L2_ENABLE_RAG="true", L2_USE_C2_RETRIEVER="true")
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_C2_URL").status == "FAIL"

    def test_port_50021_warns_only_for_voicevox(self, isolated):
        os.environ["L2_C2_URL"] = "http://127.0.0.1:50021"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_C2_URL").status == "INFO"


@pytest.mark.parametrize("path,fragment", [
    ("graph.py", '"Agent 実行失敗 [character=%s]'),
    ("graph.py", '"Agent 実行完了 [character=%s]'),
    ("graph.py", "source=%s"),
    ("graph.py", '"ask_character ツール読み込み失敗 %s'),
    ("graph.py", '"web_search ツール読み込み失敗 %s'),
    ("graph.py", '"retrieve_memory ツール読み込み失敗 %s'),
    ("graph.py", '"ツール読み込み失敗 %s (import)'),
    ("graph.py", '"ツール読み込み失敗 %s: %s。'),
    ("mcp_servers/ask_character.py", '"ask_character 協働先 Agent 失敗: %s"'),
    ("mcp_servers/ask_character.py", '"導入セリフ LLM 生成失敗: %s"'),
    ("mcp_servers/ask_character.py", '"導入セリフ生成完了: [%s] %s"'),
    ("mcp_servers/ask_character.py", '"導入セリフ TTS 失敗: %s"'),
    ("mcp_servers/ask_character.py", '"ask_character 協働応答 TTS 合成失敗 (%s): %s"'),
])
def test_production_log_wording_matches_what_preflight_looks_for(path, fragment):
    """点検はこの文言をログから探す。本番側で変わると、失敗の判定が無言で死ぬ。"""
    import lab_lounge

    for base in lab_lounge.__path__:
        target = Path(base) / path
        if target.is_file():
            assert fragment in target.read_text(encoding="utf-8")
            return
    pytest.fail(f"{path} が見つかりません")


# ─── 表の行、要約、例外の要約、ログの取り込み ─────────────────────
#
# ここも変異テストですり抜けた箇所を固定する。利用者は判定をこの表と要約で読むので、
# 印や一覧が静かに変わると、止めるべき故障を読み飛ばしたり、緑の点検を赤と読んだりする。


def _api_error_with(**attrs):
    """status_code を持つ、API の応答に由来する例外を作る。属性で SDK ごとの違いを表す。"""
    return type("Err", (Exception,), {"status_code": 400, **attrs})("本文")


def _summary_lines(report, capsys):
    """それまでの表の行を捨てて、print_summary の出力だけを行のリストで返す。"""
    capsys.readouterr()
    report.print_summary()
    return capsys.readouterr().out.splitlines()


class TestReportRows:
    """1 行ずつ出す表の形。README の「期待される出力」もこの形を示している。"""

    @pytest.mark.parametrize("status,required,starred", [
        ("FAIL", True, False),
        ("FAIL", False, True),
        ("PASS", True, False),
        ("PASS", False, False),
        ("WARN", False, False),
        ("SKIP", False, False),
    ])
    def test_only_a_non_required_failure_is_starred(self, isolated, capsys, status, required, starred):
        """* は「FAIL だが終了コードには影響しない」という印。

        必須の FAIL に付くと配信を止めるべき故障を読み飛ばし、別枠の FAIL に付かないと、
        終了コードが 0 なのに赤い行が残る理由が読めなくなる。
        """
        preflight.Report().add("main/x", status, "detail", required=required)
        row = capsys.readouterr().out
        assert status in row
        assert ("*" in row) is starred

    def test_row_without_detail_has_no_dangling_separator(self, isolated, capsys):
        """値が空のフラグなどは detail が空になる。区切りだけ残ると、表示が途中で切れたように読める。"""
        preflight.Report().add("flag/x", "INFO", "")
        assert capsys.readouterr().out.rstrip().endswith("flag/x")


class TestSummary:
    """print_summary は利用者が最後に読む結論。長い表を見返さず、ここで配信の可否を判断する。"""

    def test_each_count_sits_next_to_its_own_label(self, isolated, capsys):
        """FAIL の列に別の数が入ると、壊れた環境を FAIL 0 と読んだり、緑の点検を赤と読んだりする。

        件数をすべて変えて、どの数がどの列に入っても見分けられるようにしている。INFO は数えない。
        """
        report = preflight.Report()
        for status, count in (("PASS", 4), ("FAIL", 3), ("WARN", 2), ("SKIP", 1)):
            for index in range(count):
                report.add(f"path/{status.lower()}{index}", status)
        report.add("path/info", "INFO")
        out = "\n".join(_summary_lines(report, capsys))
        assert dict(re.findall(r"\b(PASS|FAIL|WARN|SKIP)\s+(\d+)", out)) == {"PASS": "4", "FAIL": "3", "WARN": "2", "SKIP": "1"}

    def test_token_totals_are_shown(self, isolated, capsys):
        """--paid で使った量の合計を読める場所はここだけ。消えると課金の増え方に気づきにくくなる。"""
        report = preflight.Report()
        report.input_tokens, report.output_tokens = 1234, 567
        out = "\n".join(_summary_lines(report, capsys))
        assert "1234" in out
        assert "567" in out

    def test_lists_the_failures_and_nothing_else(self, isolated, capsys):
        """一覧が消えると長い表から赤い行を探すことになり、他の行が混ざると緑の経路が壊れて見える。"""
        report = preflight.Report()
        for status in ("PASS", "WARN", "SKIP", "INFO", "FAIL"):
            report.add(f"path/{status.lower()}", status)
        listed = [line for line in _summary_lines(report, capsys) if "path/" in line]
        assert len(listed) == 1
        assert "FAIL" in listed[0]
        assert "path/fail" in listed[0]

    def test_only_non_required_failures_are_marked_in_the_list(self, isolated, capsys):
        """必須の FAIL に別枠の印が付くと終了コード 1 の原因を読み飛ばし、別枠に印がないと配信を止める故障と誤解する。"""
        report = preflight.Report()
        report.add("main/mimi", "FAIL", required=True)
        report.add("main/aruka", "FAIL", required=False)
        lines = _summary_lines(report, capsys)
        assert "別枠" not in next(line for line in lines if "main/mimi" in line)
        assert "別枠" in next(line for line in lines if "main/aruka" in line)


class TestDescribeExc:
    """FAIL の行に出す例外の要約。本文を出さない分、型と status などで原因を読めないと困る。"""

    def test_exception_without_text_has_no_dangling_separator(self):
        """raise NotImplementedError や TimeoutError() は本文が空。区切りだけ残ると本文が表示から漏れたように読める。"""
        described = preflight._describe_exc(TimeoutError())
        assert "TimeoutError" in described
        assert not described.rstrip().endswith(":")

    def test_missing_status_is_left_out(self):
        """接続の失敗には応答がなく status もない。status=None と出すと、応答が返ったように読める。"""
        error = type("Err", (Exception,), {"__module__": "httpx._x"})("x")
        assert "status" not in preflight._describe_exc(error)

    def test_attribute_is_kept_when_the_body_does_not_have_it(self):
        """SDK が属性で持つ値を、形の違う body で消さないこと。param は 400 の原因を読む手がかり。"""
        described = preflight._describe_exc(_api_error_with(param="temperature", body={"message": "x"}))
        assert "param=temperature" in described

    @pytest.mark.parametrize("attrs", [
        {"code": "bad\nrequest"},  # 応答に由来する値。改行が入ると表の行が割れる
        {"type": "t" * 60, "code": "c" * 60, "param": "p" * 60},  # 1 つずつは上限内でも、合わせると長い
    ], ids=["newline", "long"])
    def test_api_summary_fits_one_row(self, attrs):
        """行が割れたり伸びたりすると、配信中のコンソールで他の経路の行と混ざって読めなくなる。"""
        described = preflight._describe_exc(_api_error_with(**attrs))
        assert "\n" not in described
        assert len(described) <= preflight._MAX_DETAIL_CHARS

    def test_status_under_the_response_is_shown_without_the_url_or_body(self):
        """HTTPStatusError は status を response の下に持つ。本文の URL には q= の問いと host が入る。"""
        import httpx

        request = httpx.Request("GET", "http://c2.invalid:8100/retrieve?q=BODYSENTINEL")
        error = httpx.HTTPStatusError(f"error for {request.url}", request=request, response=httpx.Response(422, request=request))
        described = preflight._describe_exc(error)
        assert "status=422" in described
        assert "c2.invalid" not in described
        assert not _shows_answer(described)

    def test_response_without_a_status_adds_nothing(self):
        error = type("Err", (Exception,), {"response": object()})(BODY)
        assert "status=" not in preflight._describe_exc(error)


class TestSafeLogMessage:
    """None を返す関数の失敗理由は、ログを 1 行にしたものしか画面に出ない。"""

    def test_number_argument_is_kept(self, capture):
        """status=%d の数値は原因そのもの。書式が崩れて %d のまま出ると、理由が読めなくなる。"""
        capture.emit(_log_record("lab_lounge.x", logging.ERROR, "応答 status=%d", (503,)))
        assert "status=503" in capture.last_problem(0)

    def test_mapping_argument_is_not_expanded(self, capture):
        """引数が辞書 1 つだと record.args は tuple でなくなる。中身は発話かもしれないので定型文だけを出す。"""
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "応答: %s", ({"短い発話": "x"},)))
        reason = capture.last_problem(0)
        assert "応答" in reason
        assert "短い発話" not in reason

    @pytest.mark.parametrize("message,args", [
        ("失敗: %s", ("1 行目\n2 行目",)),  # 短い引数は長さへ置き換えないので、改行がそのまま届く
        ("a=%s b=%s c=%s d=%s e=%s f=%s", ("x" * 40,) * 6),  # 1 つずつは短くても、合わせると長い
    ], ids=["newline", "long"])
    def test_reason_fits_one_row(self, capture, message, args):
        """理由は FAIL の行の中に入る。行が割れると、配信中のコンソールで他の経路の行と混ざる。"""
        capture.emit(_log_record("lab_lounge.x", logging.ERROR, message, args))
        reason = capture.last_problem(0)
        assert "\n" not in reason
        assert len(reason) <= preflight._MAX_DETAIL_CHARS


class TestCapturedLogs:
    """点検が差し替えた root logger を、終わったら元へ戻すこと。"""

    def test_root_level_returns_to_what_it_was(self):
        """main() を呼んだ側のログの量を、点検の前後で変えないため。

        前後の level を比べるだけのテストは、先に流れた main() が INFO のまま戻さないと前後とも
        INFO になって差が出ない。そのため INFO 以外から始めて確かめる。
        """
        root = logging.getLogger()
        original = root.level
        root.setLevel(logging.CRITICAL)
        try:
            with preflight._captured_logs():
                pass
            assert root.level == logging.CRITICAL
        finally:
            root.setLevel(original)


# ─── 環境の準備と uv.lock の突き合わせ（変異テストですり抜けた故障を塞ぐ） ───────────


_LOCK_WITH_CYCLE = """
[[package]]
name = "lab-lounge"
version = "0.1.0"
source = { editable = "." }
dependencies = [{ name = "first" }]

[[package]]
name = "first"
version = "1.0"
dependencies = [{ name = "second" }]

[[package]]
name = "second"
version = "2.0"
dependencies = [{ name = "first" }]
"""


def _lock_with_marker(marker):
    """marker の付いた依存 conditional と、付いていない依存 needed を持つ lock を返す。"""
    return f"""
[[package]]
name = "lab-lounge"
version = "0.1.0"
source = {{ editable = "." }}
dependencies = [{{ name = "needed" }}, {{ name = "conditional", marker = "{marker}" }}]

[[package]]
name = "needed"
version = "1.0"

[[package]]
name = "conditional"
version = "2.0"
"""


def _check_lock(root, lock_text, installed):
    """lock と入っている版を与えて _check_versions を流し、deps/uv.lock の結果を返す。

    Args:
        root:      uv.lock を置くディレクトリ
        lock_text: uv.lock の中身
        installed: {パッケージ名: 版}。ここにない名前は未インストールとして扱う
    """

    def fake_version(name):
        if name in installed:
            return installed[name]
        raise preflight.metadata.PackageNotFoundError(name)

    (root / "uv.lock").write_text(lock_text, encoding="utf-8")
    report = preflight.Report()
    with patch.object(preflight.metadata, "version", fake_version):
        preflight._check_versions(report, root)
    return _status(report, "deps/uv.lock")


def _with_read_budget(lock_data, budget=200):
    """lock の各 package を、読める回数に上限のある dict へ置き換えて返す。

    WHY: locked_closure が循環で止まらなくなると、テストは失敗せずに固まる。打ち切りの実時間を
         待たずに失敗として報告するため、読みすぎた時点で AssertionError にする。
         上限は、循環を一巡して止まるときの読み取り回数（十数回）より十分に大きくしてある。
    """
    reads = [0]

    class Package(dict):
        def _spend(self):
            reads[0] += 1
            assert reads[0] <= budget, "同じ依存を辿り続けています（循環で止まっていません）"

        def get(self, key, default=None):
            self._spend()
            return super().get(key, default)

        def __getitem__(self, key):
            self._spend()
            return super().__getitem__(key)

    return {**lock_data, "package": [Package(package) for package in lock_data["package"]]}


class TestEnvFiles:
    """env ファイルの読み方と、止めるときの案内を固定する。"""

    def test_inherited_endpoint_does_not_survive_through_a_reference(self, isolated):
        """掃除を読み込みより先に済ませることを確かめる。

        .env の ${NAME} は読み込む時点の環境から展開される。順序が逆だと、外したはずの接続先が別の
        変数へ写って残る。その変数を接続先に使っていれば、「外しました」と表示されたまま本番と別の
        相手を測ることになる。
        """
        os.environ["OPENAI_BASE_URL"] = "http://proxy.invalid/inherited"
        (isolated / ".env").write_text("L2_TEST_ENDPOINT=${OPENAI_BASE_URL}\n", encoding="utf-8")
        preflight.prepare_env("prod", isolated)
        assert "proxy.invalid" not in os.environ["L2_TEST_ENDPOINT"]

    def test_line_without_equals_is_skipped(self, isolated):
        """`KEY` だけの行は値を持たないので読み飛ばす。

        値の None をそのまま os.environ へ入れると TypeError になり、.env の 1 行で点検全体が落ちる。
        run スクリプトも load_dotenv もこの行を読み飛ばすので、本番と同じ扱いになる。
        """
        (isolated / ".env").write_text("L2_TEST_BARE\nL2_LLM_MODEL=from-file\n", encoding="utf-8")
        preflight.prepare_env("prod", isolated)
        assert "L2_TEST_BARE" not in os.environ
        assert os.environ["L2_LLM_MODEL"] == "from-file"

    def test_endpoint_name_without_a_value_is_not_refused(self, isolated):
        """`OPENAI_BASE_URL` とだけ書かれた行は接続先を指定していない。

        本番の読み込みでも環境に入らないので、ここで止めると、正常に動く環境で点検だけが通らなくなる。
        """
        (isolated / ".env").write_text("OPENAI_BASE_URL\n", encoding="utf-8")
        assert preflight.prepare_env("prod", isolated) == [isolated / ".env"]
        assert "OPENAI_BASE_URL" not in os.environ

    @pytest.mark.parametrize("file_name", [".env", ".env.dev"])
    def test_refusal_names_the_file_and_the_variable(self, isolated, capsys, file_name):
        """終了コード 2 だけでは、どのファイルのどの行を外せばよいか分からない。"""
        (isolated / file_name).write_text("OPENAI_BASE_URL=http://elsewhere.invalid\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            preflight.prepare_env("dev", isolated)
        out = capsys.readouterr().out
        assert file_name in out
        assert "OPENAI_BASE_URL" in out

    def test_nothing_inherited_prints_nothing(self, isolated, capsys):
        """外していないのに毎回「外しました」と出ると、本当に外したときの 1 行が読み飛ばされる。"""
        assert preflight.clean_llm_env() == []
        assert capsys.readouterr().out == ""


class TestEnvFileQuoting:
    r"""run_dev.ps1 は書いたとおりに読み、python-dotenv はダブルクォートの中の \r や \v を制御文字にする。"""

    def _quoting_rows(self, isolated: Path, capsys, env_text: bytes) -> tuple[list[preflight.Check], str]:
        (isolated / ".env").write_bytes(env_text)
        files = preflight.prepare_env("prod", isolated)
        names = {name for path in files for name in preflight.dotenv_values(path)}
        report = preflight.Report()
        preflight._check_env_quoting(report, names)
        return report.checks, capsys.readouterr().out

    @pytest.mark.parametrize("raw", [
        rb'L2_X="D:\zqx\reference\voices.json"',
        rb'L2_X="C:\zqx\voicepeak.exe"',  # \v だけ
        rb'L2_X="C:\zqx\new\y.exe"',  # \n だけ
        rb'L2_X="C:\zqx\tools\y.exe"',  # \t だけ
    ])
    def test_quoted_windows_path_warns_without_the_value(self, isolated, capsys, raw):
        checks, out = self._quoting_rows(isolated, capsys, raw + b"\n")
        assert [(c.name, c.status) for c in checks] == [("env/quoting", "WARN")]
        assert "L2_X" in checks[0].detail and "引用符" in checks[0].detail
        assert _MARK not in out

    @pytest.mark.parametrize("raw", [
        rb"L2_X=D:\zqx\reference\voices.json",  # 引用符なしは両方とも書いたとおり
        rb"L2_X='D:\zqx\reference\voices.json'",  # シングルクォートは制御文字にならない
        rb'L2_X="C:\\tools\\voicepeak.exe"',  # 二重の円記号はタブにならない
        rb'L2_X="plain value"',
        b"L2_X=value\r\nL2_Y=value2\r\n",  # CRLF の .env
    ])
    def test_values_both_readers_agree_on_stay_quiet(self, isolated, capsys, raw):
        checks, _ = self._quoting_rows(isolated, capsys, raw + b"\n")
        assert checks == []

    def test_only_names_from_the_env_files_are_looked_at(self, isolated):
        os.environ["L2_INHERITED"] = "a\tb"
        report = preflight.Report()
        preflight._check_env_quoting(report, {"REDIS_URL"})
        assert report.checks == []

    def test_main_reports_it_next_to_the_failure_it_explains(self, isolated, capsys):
        """引用符のせいで tts-config が赤いとき、直し方が同じ出力に並ぶこと。"""
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=r'"C:\reference\voices.json"')
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert "[WARN ] env/quoting: L2_IRODORI_VOICES_JSON" in out
        assert "[FAIL ] tts-config/mimi:" in out


class TestLoopbackProxy:
    """C2 と TTS への接続が環境のプロキシを通る設定を知らせる。run スクリプトの /health は -NoProxy なので、そちらでは見えない。"""

    # 出力に現れないことを確かめる印。
    _PROXY = "http://zqx-proxy.invalid:8080"

    @pytest.fixture(autouse=True)
    def _no_inherited_proxy(self, monkeypatch):
        """流すシェルのプロキシと NO_PROXY で結果が変わらないようにする。"""
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize("names,reach", [
        (["HTTP_PROXY"], "C2 と TTS への接続が"),
        # httpx (C2Retriever) は ALL_PROXY を読むが、urllib (TTS) は読まない。TTS まで疑わせない。
        (["ALL_PROXY"], "C2 への接続だけが"),
        (["all_proxy"], "C2 への接続だけが"),
        (["HTTP_PROXY", "ALL_PROXY"], "C2 と TTS への接続が"),
    ])
    def test_warns_without_showing_the_proxy(self, monkeypatch, names, reach):
        for name in names:
            monkeypatch.setenv(name, self._PROXY)
        report = preflight.Report()
        preflight._check_loopback_proxy(report)
        check = _status(report, "env/proxy")
        assert check.status == "WARN"
        assert reach in check.detail
        assert "zqx" not in check.detail

    @pytest.mark.parametrize("proxy_name", ["HTTP_PROXY", "ALL_PROXY"])
    @pytest.mark.parametrize("no_proxy", ["127.0.0.1", "localhost,127.0.0.1", "*"])
    def test_no_proxy_quiets_it(self, monkeypatch, no_proxy, proxy_name):
        monkeypatch.setenv(proxy_name, self._PROXY)
        monkeypatch.setenv("NO_PROXY", no_proxy)
        report = preflight.Report()
        preflight._check_loopback_proxy(report)
        assert report.checks == []

    def test_https_proxy_alone_stays_quiet(self, monkeypatch):
        """C2 と TTS は手元の http なので、https だけのプロキシは通らない。"""
        monkeypatch.setenv("HTTPS_PROXY", self._PROXY)
        report = preflight.Report()
        preflight._check_loopback_proxy(report)
        assert report.checks == []

    def test_no_row_without_a_proxy(self):
        # Windows は環境変数がないとレジストリの設定を読むので、手元の設定で結果が変わらないように差し替える。
        with patch("urllib.request.getproxies", return_value={}):
            report = preflight.Report()
            preflight._check_loopback_proxy(report)
        assert report.checks == []

    def test_static_shows_the_row(self, isolated, monkeypatch, capsys):
        """--static は run スクリプトが毎回流すので、ここに出れば起動のたびに見える。"""
        monkeypatch.setenv("HTTP_PROXY", self._PROXY)
        (isolated / ".env").write_text("REDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert "[WARN ] env/proxy:" in out
        assert "zqx" not in out

    def test_readme_tells_what_to_do(self):
        row = next(line for line in _readme_section().splitlines() if "`env/proxy`" in line)
        assert "NO_PROXY" in row
        assert "ALL_PROXY" in row


class TestRunScriptRefusals:
    """点検が緑でも、run_dev.ps1 / run_prod.ps1 は exit 1 で止まる組がある。python を直接起動する構成もあるので WARN まで。"""

    @pytest.mark.parametrize("profile,url,script", [
        ("dev", "http://127.0.0.1:8100", "run_dev.ps1"),
        ("dev", "http://127.0.0.1:8100/", "run_dev.ps1"),
        ("prod", "http://127.0.0.1:8101", "run_prod.ps1"),
        ("prod", "http://127.0.0.1:8101/", "run_prod.ps1"),
    ])
    def test_refused_pair_warns(self, isolated, profile, url, script):
        (isolated / ".env.dev").write_text("", encoding="utf-8")
        os.environ["L2_C2_URL"] = url
        report = preflight.Report()
        preflight._check_run_script_refusals(report, profile, isolated)
        check = _status(report, "run-script/L2_C2_URL")
        assert check.status == "WARN"
        assert script in check.detail

    @pytest.mark.parametrize("profile,url", [
        ("dev", "http://127.0.0.1:8101"),
        ("prod", "http://127.0.0.1:8100"),
        ("dev", "http://127.0.0.1:81000"),
        ("dev", None),  # スクリプトは未設定を拒否しない。既定の 8100 を補うと誤った警告になる
        ("prod", None),
    ])
    def test_pairs_the_scripts_accept_stay_quiet(self, isolated, profile, url):
        (isolated / ".env.dev").write_text("", encoding="utf-8")
        if url is not None:
            os.environ["L2_C2_URL"] = url
        report = preflight.Report()
        preflight._check_run_script_refusals(report, profile, isolated)
        assert report.checks == []

    def test_dev_without_env_dev_warns(self, isolated):
        report = preflight.Report()
        preflight._check_run_script_refusals(report, "dev", isolated)
        assert _status(report, "run-script/.env.dev").status == "WARN"
        prod = preflight.Report()
        preflight._check_run_script_refusals(prod, "prod", isolated)
        assert prod.checks == []

    @pytest.mark.parametrize("script,pattern", [("run_dev.ps1", ":8100/?$"), ("run_prod.ps1", ":8101/?$")])
    def test_conditions_match_the_scripts(self, script, pattern):
        """スクリプトの条件が変わったら、ここも変える。ずれると起動できるのに警告し続ける。"""
        assert f"-match '{pattern}'" in (_REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")

    def test_run_static_calls_it(self, isolated):
        with patch.object(preflight, "_check_run_script_refusals") as refusals:
            preflight.run_static(preflight.Report(), isolated, "prod", [], set())
        assert refusals.call_args.args[1:] == ("prod", isolated)


class TestLoopbackAllowance:
    """allow_loopback の判定を固定する。

    main() は allow_loopback を渡さないので、配信前の点検はこの経路を通らない。pytest の中で SDK を
    偽サーバへ向けるための口で、判定を誤ると実害はテストの側に出る。
    """

    def test_localhost_counts_as_loopback(self, isolated):
        """偽サーバを localhost で指したときに値を消すと、SDK は既定の実 API へ向かう。

        偽サーバで確かめるはずのテストが、実サービスを呼んでしまう。
        """
        os.environ["OPENAI_BASE_URL"] = "http://localhost:9999"
        assert preflight.clean_llm_env(allow_loopback=True) == []
        assert os.environ["OPENAI_BASE_URL"] == "http://localhost:9999"

    @pytest.mark.parametrize("name,value", [
        ("ANTHROPIC_AUTH_TOKEN", "a-token-is-not-a-url"),
        ("OPENAI_BASE_URL", "http://host:not-a-port/"),
    ])
    def test_value_that_is_not_a_url_is_removed(self, isolated, name, value):
        """外す対象には URL でないトークンも入っている。形を読めない値で落ちず、loopback でないものとして外す。"""
        os.environ[name] = value
        assert preflight.clean_llm_env(allow_loopback=True) == [name]
        assert name not in os.environ


class TestLockedClosure:
    """uv.lock から「入っているべきパッケージ」を求める処理を固定する。"""

    def test_dependency_whose_marker_holds_is_followed(self):
        """条件を満たす marker の依存も点検の対象にする。

        marker 付きをまとめて飛ばすと、Windows 専用の依存などが消えても緑のままになる。
        """
        wanted = preflight.locked_closure(tomllib.loads(_lock_with_marker("python_version >= '3'")))
        assert wanted == {"needed": {"1.0"}, "conditional": {"2.0"}}

    def test_marker_that_cannot_be_parsed_is_not_reported_missing(self, isolated):
        """読めない marker の依存を不足と報告すると、正常な環境で配信の起動を止めてしまう。"""
        check = _check_lock(isolated, _lock_with_marker("this is not a marker"), {"needed": "1.0"})
        assert check.status == "PASS"

    def test_marked_dependency_is_not_reported_missing_without_packaging(self, isolated, monkeypatch):
        """packaging が無いと marker を判定できないので、判定できない依存を不足に数えない。

        素の uv sync は packaging も消すので、この経路は実際に通る。不足に数えると、別の OS 向けの
        パッケージまで「入れるべきもの」として件数と一覧に混ざる。
        """
        monkeypatch.setitem(sys.modules, "packaging.markers", None)
        check = _check_lock(isolated, _lock_with_marker("python_version >= '3'"), {"needed": "1.0"})
        assert check.status == "PASS"

    def test_dependency_cycle_ends(self):
        """依存が循環していても止まる。止まらないと --static が固まり、配信前に何も分からない。"""
        lock = _with_read_budget(tomllib.loads(_LOCK_WITH_CYCLE))
        assert preflight.locked_closure(lock) == {"first": {"1.0"}, "second": {"2.0"}}

    @pytest.mark.parametrize("written", ["Some_Pkg", "some.pkg", "SOME--PKG"])
    def test_names_are_compared_after_normalizing(self, written):
        """依存側の表記がパッケージ側と違っても同じものとして辿り、正規化した名前で返す。

        uv は正規化した名前を書くが、素の文字列比較に頼ると、表記が違った依存は先の推移的な依存ごと
        点検から外れ、消えても緑のままになる。
        """
        lock = f"""
[[package]]
name = "lab-lounge"
version = "0.1.0"
source = {{ editable = "." }}
dependencies = [{{ name = "{written}" }}]

[[package]]
name = "some-pkg"
version = "1.0"
"""
        assert preflight.locked_closure(tomllib.loads(lock)) == {"some-pkg": {"1.0"}}


class TestVersionReport:
    """deps/uv.lock の判定と、FAIL のときに読む内訳を固定する。"""

    def test_version_mismatch_alone_is_fail(self, isolated):
        """不足が 1 件もなく版だけが違う環境も FAIL にする。

        これが PASS になると、手元の更新で lock と版がずれても緑のまま配信へ進み、依存の更新で壊れた
        経路を L2 が無音で握りつぶす。
        """
        installed = {"needed": "1.0", "from-extra": "2.0", "transitive": "9.9"}
        check = _check_lock(isolated, _LOCK, installed)
        assert check.status == "FAIL"
        assert "transitive" in check.detail

    @pytest.mark.parametrize("installed,named", [
        ({"needed": "1.0", "transitive": "3.0"}, "from-extra"),
        ({"needed": "1.0", "from-extra": "2.0", "transitive": "9.9"}, "transitive"),
    ], ids=["missing-only", "mismatch-only"])
    def test_detail_lists_only_the_kind_that_happened(self, isolated, installed, named):
        """FAIL の内訳に「0 件」の項目が混ざると、読む人が直す相手を探して迷う。"""
        check = _check_lock(isolated, _LOCK, installed)
        assert named in check.detail
        assert "0 件" not in check.detail


# ─── 静的な点検の取りこぼし（_check_imports 〜 _check_obs） ──────────────
#
# 変異テストで、故障を入れても緑のままだった箇所を固定する。既存のテストは主に「壊れた設定が赤くなる
# こと」を見ているので、ここでは次の 2 つを足す。
#   - 正常な設定を赤くしないこと。誤った警告が続くと、利用者は表を読まなくなり、本物の警告を見逃す。
#   - 結論の行が出ること。行が消えると、確かめて問題なかったのか、見ていないのかを読み分けられない。

# 定数でパラメータ化すると、一覧から消した名前はテストごと消える。名前は直書きする。
_BOOL_FLAG_NAMES = (
    "L2_USE_REAL_LLM", "L2_USE_REAL_TTS", "L2_USE_REAL_STT", "L2_ENABLE_TOOLS", "L2_ENABLE_RAG", "L2_USE_C2_RETRIEVER",
    "L2_USE_C2_RECENT", "L2_USE_INTENT_GATE", "L2_USE_LLM_ROUTER", "L2_USE_HANDRAISE", "L2_FILLER_ENABLED",
)
_OTHER_FLAG_NAMES = (
    "L2_STT_PROVIDER", "L2_WAKE_BACKEND", "L2_VAD_BACKEND", "L2_DEFAULT_SPEAKER",
    "L2_LLM_REASONING_EFFORT", "L2_LLM_MAX_OUTPUT_TOKENS",
)


def _warned(report: preflight.Report, name: str) -> bool:
    """name の行に WARN があるかを返す。

    _status は最初の行しか返さない。フラグの点検は INFO の行の後ろに同じ名前で WARN を足すので、全行を見る。
    """
    return any(c.name == name and c.status == "WARN" for c in report.checks)


class TestImportsAndModelsScope:
    """点検する SDK とプロバイダの範囲。本番が使う範囲とずれると、判定もずれる。"""

    @pytest.mark.parametrize("value", [None, "false"])
    def test_tool_sdks_are_not_required_when_tools_are_off(self, isolated, value):
        """ツールを使わない構成なら、langchain_tavily がなくても本番は動く。赤くすると配信の起動を止めてしまう。"""
        if value is not None:
            os.environ["L2_ENABLE_TOOLS"] = value

        def fake_import(name):
            if name in ("langchain.agents", "langchain_tavily", "langgraph.prebuilt"):
                raise ModuleNotFoundError(name)
            return MagicMock()

        report = preflight.Report()
        with patch("importlib.import_module", fake_import):
            preflight._check_imports(report, {"openai"})
        assert report.exit_code() == preflight.EXIT_OK

    def test_provider_used_only_by_the_filler_is_collected(self, isolated):
        """filler だけが使うプロバイダの SDK やキーが壊れると、つなぎの発話だけが無音で消える。点検から漏らさない。"""
        os.environ.update({"L2_USE_LLM_ROUTER": "false", "L2_USE_HANDRAISE": "false"})
        character = SimpleNamespace(slug="x", filler_model="gemini-filler")
        with patch("lab_lounge.pipeline._get_llm_mode", return_value=(True, "openai", "gpt-main")):
            providers = preflight._check_models(preflight.Report(), [character])
        assert providers == {"openai", "google"}

    def test_intent_gate_alone_keeps_the_intent_provider(self, isolated):
        """挙手を切っていても、意図ゲートは同じモデルを呼ぶ。漏らすと、ゲートだけが使うプロバイダのキーを見なくなる。"""
        os.environ.update({"L2_USE_INTENT_GATE": "true", "L2_USE_HANDRAISE": "false", "L2_INTENT_GATE_MODEL": "claude-intent"})
        assert preflight._check_models(preflight.Report(), []) == {"anthropic"}

    def test_character_row_shows_the_effective_models(self, isolated):
        """どのモデルを点検したのかを読む行（README の出力例にもある）。消えると、設定の取り違えに目で気づけない。"""
        character = SimpleNamespace(slug="x", filler_model="claude-filler")
        report = preflight.Report()
        with patch("lab_lounge.pipeline._get_llm_mode", return_value=(True, "openai", "gpt-main")):
            preflight._check_models(report, [character])
        detail = _status(report, "model/x").detail
        assert all(word in detail for word in ("openai", "gpt-main", "claude-filler"))


class TestKeyRows:
    """キーの行は set / unset で読ませる（README の約束）。行と判定が食い違うと、直す先を誤る。"""

    def test_missing_key_is_a_warning_in_dummy_mode(self, isolated):
        """ダミー起動は止めない。ただし PASS にすると、実 LLM へ切り替えてから初めてキーがないと分かる。"""
        report = preflight.Report()
        preflight._check_keys(report, {"openai"}, set())
        assert _status(report, "key/openai").status == "WARN"

    @pytest.mark.parametrize("value,status,word", [("dummy-tavily", "PASS", "set"), (None, "FAIL", "unset")])
    def test_tavily_row_agrees_with_its_status(self, isolated, value, status, word):
        """[FAIL] key/tavily: set と読めると、キーを入れ直せばよいことに気づけない。"""
        # 実 LLM の構成に揃える。ダミー起動での扱いと切り離して、行と判定の一致だけを見るため。
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_ENABLE_TOOLS"] = "true"
        if value is not None:
            os.environ["TAVILY_API_KEY"] = value
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        check = _status(report, "key/tavily")
        assert (check.status, check.detail.split()[0]) == (status, word)

    @pytest.mark.parametrize("name", ["LANGSMITH_API_KEY", "L2_PORCUPINE_ACCESS_KEY", "L2_OBS_WS_PASSWORD"])
    def test_absent_optional_secret_reads_unset(self, isolated, name):
        """OBS の認証やウェイクワードの失敗を調べるとき、この行で有無を読む。無いのに set と出ると調査を誤らせる。"""
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        assert _status(report, f"key/{name}").detail == "unset"


class TestGoogleKeyIsReadLikeProduction:
    """LangChain は最初に定義された名前だけを読む。空の GOOGLE_API_KEY は GEMINI_API_KEY を隠して Agent の構築で落ちる。"""

    def test_blank_google_key_hides_the_gemini_key(self, isolated):
        os.environ.update({"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "dummy-gemini"})
        real = preflight.Report()
        os.environ["L2_USE_REAL_LLM"] = "true"
        preflight._check_keys(real, {"google"}, set())
        check = _status(real, "key/google")
        assert check.status == "FAIL"
        # キーを足しても直らないので、消す行を名指しする。
        assert "GOOGLE_API_KEY" in check.detail and "消" in check.detail
        os.environ["L2_USE_REAL_LLM"] = "false"
        dummy = preflight.Report()
        preflight._check_keys(dummy, {"google"}, set())
        assert _status(dummy, "key/google").status == "WARN"

    @pytest.mark.parametrize("env", [
        {"GEMINI_API_KEY": "dummy-gemini"},
        {"GOOGLE_API_KEY": "dummy-google"},
        {"GOOGLE_API_KEY": "dummy-google", "GEMINI_API_KEY": ""},
    ])
    def test_keys_production_can_read_pass(self, isolated, env):
        os.environ.update({"L2_USE_REAL_LLM": "true", **env})
        report = preflight.Report()
        preflight._check_keys(report, {"google"}, set())
        assert _status(report, "key/google").status == "PASS"

    def test_production_really_stops_on_the_blank_key(self, isolated):
        """FAIL の根拠。LangChain の読み方が変わったら、このテストが落ちるので判定を見直す。"""
        pytest.importorskip("langchain_google_genai")
        from pydantic import ValidationError

        from lab_lounge.graph import _get_llm_for_agent

        os.environ.update({"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "dummy-gemini"})
        with pytest.raises(ValidationError):
            _get_llm_for_agent("google", "gemini-dummy")


class TestFlagRows:
    """フラグの表は配信前に目で読む。未設定と設定値を読み分けられることと、余計な警告が出ないことを固定する。"""

    @pytest.mark.parametrize("name", _BOOL_FLAG_NAMES)
    def test_unset_bool_flag_still_has_a_row(self, isolated, name):
        """行がないと、未設定なのか、そのフラグを見ていないのかを区別できない。"""
        report = preflight.Report()
        preflight._check_flags(report)
        check = _status(report, f"flag/{name}")
        assert check.status == "INFO" and check.detail

    @pytest.mark.parametrize("name", _OTHER_FLAG_NAMES)
    def test_other_flag_row_shows_the_value(self, isolated, name):
        """音声やモデルの設定の取り違えは、この値を読んで見つける。設定しても未設定と出ると、誤りに気づけない。"""
        os.environ[name] = "value-to-show"
        report = preflight.Report()
        preflight._check_flags(report)
        assert _status(report, f"flag/{name}").detail == "value-to-show"

    @pytest.mark.parametrize("name", _OTHER_FLAG_NAMES)
    def test_other_flag_row_says_when_unset(self, isolated, name):
        """未設定を空欄で出すと、空文字を設定したのか、未設定で既定が使われているのかを読み分けられない。"""
        report = preflight.Report()
        preflight._check_flags(report)
        check = _status(report, f"flag/{name}")
        assert check.status == "INFO" and check.detail

    @pytest.mark.parametrize("value", [None, "false", "true", "TRUE"])
    def test_c2_retriever_is_quiet_when_graph_reads_it_the_same(self, isolated, value):
        """graph.py と読み方が揃う値では警告しない。"1" のような本物の不一致の警告を埋もれさせないため。"""
        if value is not None:
            os.environ["L2_USE_C2_RETRIEVER"] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert not _warned(report, "flag/L2_USE_C2_RETRIEVER")

    @pytest.mark.parametrize("value", [None, "false"])
    def test_debug_artifacts_are_quiet_when_off(self, isolated, value):
        """無効なのに「実発話が書かれます」と出ると、本当に有効なときの警告が埋もれ、実発話の公開を見逃す。"""
        if value is not None:
            os.environ["L2_DEBUG_ARTIFACTS"] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert not _warned(report, "flag/L2_DEBUG_ARTIFACTS")

    @pytest.mark.parametrize("backend,key", [("porcupine", "dummy-key"), ("bg-continuous", None), (None, None)])
    def test_porcupine_warning_needs_both_conditions(self, isolated, backend, key):
        """キーがある porcupine と、porcupine を指定していない構成は正常なので、警告しない。"""
        if backend is not None:
            os.environ["L2_WAKE_BACKEND"] = backend
        if key is not None:
            os.environ["L2_PORCUPINE_ACCESS_KEY"] = key
        report = preflight.Report()
        preflight._check_flags(report)
        assert not _warned(report, "flag/L2_WAKE_BACKEND")


class TestSilentlyReplacedValues:
    """本番が例外にせず別の値で動く設定。止めずに、気づけるように知らせる。"""

    @pytest.mark.parametrize("name,value", [
        ("L2_DEFAULT_SPEAKER", "mimii"),
        ("L2_LLM_MAX_OUTPUT_TOKENS", "4k"),
        ("L2_LLM_REASONING_EFFORT", "ultra"),
        ("L2_LLM_REASONING_EFFORT", "LOW"),  # 本番はそのまま API へ渡す
        ("L2_USE_HANDRAISE", ""),
        ("L2_FILLER_ENABLED", ""),
        ("L2_USE_C2_RECENT", ""),
        ("L2_STT_HALLUCINATION_FILTER", ""),
        ("L2_LOG_TO_FILE", ""),
        ("L2_BUS_FAIL_OPEN", ""),  # 空は偽で、Redis の障害でターンが止まる
    ])
    def test_replaced_value_warns_and_does_not_stop(self, isolated, name, value):
        os.environ[name] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert _warned(report, f"flag/{name}")
        assert report.exit_code() == preflight.EXIT_OK

    @pytest.mark.parametrize("name,value", [
        ("L2_DEFAULT_SPEAKER", "mimi"),
        ("L2_DEFAULT_SPEAKER", ""),  # 既定と同じ octamaid になる
        ("L2_LLM_MAX_OUTPUT_TOKENS", "4096"),
        ("L2_LLM_MAX_OUTPUT_TOKENS", "0"),  # 説明どおりの「上限なし」
        ("L2_LLM_MAX_OUTPUT_TOKENS", "-1"),
        ("L2_LLM_MAX_OUTPUT_TOKENS", ""),
        ("L2_LLM_MAX_OUTPUT_TOKENS", " 4096 "),  # 本番は strip する
        ("L2_LLM_REASONING_EFFORT", "low"),
        ("L2_LLM_REASONING_EFFORT", "none"),
        ("L2_LLM_REASONING_EFFORT", ""),  # 説明どおりの「設定しない」
        ("L2_USE_HANDRAISE", "true"),
        ("L2_USE_HANDRAISE", "false"),
        ("L2_FILLER_ENABLED", "1"),
        ("L2_ENABLE_TOOLS", ""),  # 既定が偽なので、空でも読み方は変わらない
        ("L2_BUS_FAIL_OPEN", "false"),  # 書いたとおりに以前の挙動へ戻す
    ])
    def test_values_production_reads_as_written_stay_quiet(self, isolated, name, value):
        os.environ[name] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert not _warned(report, f"flag/{name}")

    def test_production_really_replaces_them(self, isolated):
        """WARN の根拠。本番が読み方を変えたら、このテストが落ちるので判定を見直す。"""
        from lab_lounge.characters import get_default_character
        from lab_lounge.dispatcher import _get_handraise_config
        from lab_lounge.filler import is_filler_enabled
        from lab_lounge.llm import get_llm_max_output_tokens, get_llm_reasoning_effort

        os.environ["L2_LLM_MAX_OUTPUT_TOKENS"] = "4k"
        assert get_llm_max_output_tokens() is None
        os.environ["L2_LLM_REASONING_EFFORT"] = "ultra"
        assert get_llm_reasoning_effort() == "ultra"
        os.environ["L2_DEFAULT_SPEAKER"] = "mimii"
        assert get_default_character().slug == "octamaid"
        os.environ.update({"L2_USE_HANDRAISE": "", "L2_FILLER_ENABLED": ""})
        assert _get_handraise_config()["use_handraise"] is False
        assert is_filler_enabled() is False
        from lab_lounge.bus import _fail_open

        os.environ["L2_BUS_FAIL_OPEN"] = ""
        assert _fail_open() is False

    def test_default_true_list_matches_production(self):
        found = set()
        for text in _production_sources().values():
            found |= set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)",\s*"true"\s*\)', text))
        assert found == set(preflight._DEFAULT_TRUE_FLAGS)

    @pytest.mark.parametrize("value", ["", "  "])
    @pytest.mark.parametrize("profile", ["dev", "prod"])
    def test_blank_stream_name_warns(self, isolated, profile, value):
        """本番は既定へ戻さず空の名前で publish するので、C2 と HUD に届かない。"""
        os.environ["REDIS_STREAM_KEY"] = value
        report = preflight.Report()
        preflight._check_stream(report, profile)
        check = _status(report, "stream")
        assert check.status == "WARN"
        assert "REDIS_STREAM_KEY" in check.detail


class TestInheritedFlags:
    """端末に残った値で測って緑でも、Explorer などから起動する本番ではフラグが偽になり、機能が無音で消える。"""

    def test_true_flag_only_in_the_parent_process_warns(self, isolated):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "1", "L2_USE_LLM_ROUTER": "true"})
        report = preflight.Report()
        preflight._check_flags(report, {"REDIS_STREAM_KEY", "L2_USE_LLM_ROUTER"})
        check = _status(report, "flag/inherited")
        assert check.status == "WARN"
        assert "L2_ENABLE_TOOLS" in check.detail and "L2_ENABLE_RAG" in check.detail
        assert "L2_USE_LLM_ROUTER" not in check.detail

    @pytest.mark.parametrize("value", ["false", "0", ""])
    def test_false_flag_is_quiet(self, isolated, value):
        os.environ["L2_ENABLE_TOOLS"] = value
        report = preflight.Report()
        preflight._check_flags(report, {"REDIS_STREAM_KEY"})
        assert not any(c.name == "flag/inherited" for c in report.checks)

    def test_without_env_files_the_source_is_not_judged(self, isolated):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        preflight._check_flags(report, set())
        assert not any(c.name == "flag/inherited" for c in report.checks)

    def test_main_passes_the_file_names_on(self, isolated, capsys):
        (isolated / ".env").write_text("REDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        os.environ["L2_ENABLE_TOOLS"] = "true"
        with patch.object(preflight, "_check_imports"):
            preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        assert "[WARN ] flag/inherited:" in capsys.readouterr().out


class TestRealModeSource:
    def test_no_env_file_means_the_source_is_not_judged(self, isolated):
        """env ファイルを読まない実行では、すべての値が継承に見える。出どころを判定できないので、誤って警告しない。"""
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_USE_REAL_TTS"] = "true"
        report = preflight.Report()
        preflight._check_real_modes(report, "prod", set(), paid=False)
        assert report.checks == []


class TestNoPlayRow:
    """L2_NO_PLAY が真だと run_loop は再生スレッドもフィラーも作らず、音を一切出さない。"""

    @pytest.mark.parametrize("profile,status,code", [("prod", "FAIL", 1), ("dev", "WARN", 0)])
    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_true_is_reported(self, isolated, profile, status, code, value):
        os.environ["L2_NO_PLAY"] = value
        os.environ["L2_USE_REAL_LLM"] = "true"
        os.environ["L2_USE_REAL_TTS"] = "true"
        report = preflight.Report()
        preflight._check_real_modes(report, profile, set(), paid=False)
        assert _status(report, "mode/L2_NO_PLAY").status == status
        assert report.exit_code() == code

    @pytest.mark.parametrize("value", [None, "", "false", "0", "no", " true"])
    def test_other_values_have_no_row(self, isolated, value):
        """空白つきの値は本番でも偽になる。"""
        if value is not None:
            os.environ["L2_NO_PLAY"] = value
        report = preflight.Report()
        preflight._check_real_modes(report, "prod", set(), paid=False)
        assert not [c for c in report.checks if c.name == "mode/L2_NO_PLAY"]

    def test_paid_on_the_dev_profile_warns(self, isolated):
        os.environ["L2_NO_PLAY"] = "true"
        report = preflight.Report()
        preflight._check_real_modes(report, "dev", set(), paid=True)
        assert _status(report, "mode/L2_NO_PLAY").status == "WARN"

    def test_prod_exit_code_comes_from_this_row(self, isolated, voices, capsys):
        """同じ設定で、この値だけで合否が変わること。別の FAIL で緑のテストにならないように並べる。"""
        env = {"L2_IRODORI_VOICES_JSON": str(voices), "REDIS_STREAM_KEY": "aibyss:events"}
        _write_real_env(isolated, **env, L2_NO_PLAY="false")
        assert preflight.main(["--static", "--profile", "prod"], repo_root=isolated) == preflight.EXIT_OK
        _write_real_env(isolated, **env, L2_NO_PLAY="true")
        assert preflight.main(["--static", "--profile", "prod"], repo_root=isolated) == preflight.EXIT_FAILED
        assert "[FAIL ] mode/L2_NO_PLAY" in capsys.readouterr().out

    def test_dev_reads_it_from_the_env_file_and_only_warns(self, isolated, voices, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), L2_NO_PLAY="true")
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_OK
        assert "[WARN ] mode/L2_NO_PLAY" in capsys.readouterr().out

    @pytest.mark.parametrize("name", ["run_loop.py", "run_once.py"])
    def test_production_reads_the_same_name_and_values(self, name):
        """本番が読み方を変えたら、この行も揃える。import すると load_dotenv が走るので文字列で見る。"""
        import lab_lounge

        source = next(Path(base) / name for base in lab_lounge.__path__ if (Path(base) / name).is_file())
        assert re.search(
            r'os\.environ\.get\("L2_NO_PLAY", "false"\)\.lower\(\) in \(\s*"true", "1", "yes"\s*\)',
            source.read_text(encoding="utf-8"),
        )

    def test_the_example_env_does_not_trigger_it(self):
        assert re.search(r"^L2_NO_PLAY=false$", (_REPO_ROOT / ".env.example").read_text(encoding="utf-8"), re.M)


class TestPersonaContent:
    """空の人格は人格なしの応答になり、UTF-8 でない人格は全ターンを例外で落とす。"""

    @pytest.mark.parametrize("content", [b"", b"   \n\t", "人格".encode("cp932")])
    def test_blank_or_unreadable_persona_is_fail(self, isolated, monkeypatch, content):
        monkeypatch.setattr("lab_lounge.characters._PROMPTS_DIR", isolated)
        (isolated / "system_mimi.txt").write_bytes(content)
        report = preflight.Report()
        preflight._check_personas(report, [get_character("mimi")])
        assert _status(report, "persona/mimi").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_failure_of_a_non_required_character_does_not_change_the_exit_code(self, isolated, monkeypatch):
        monkeypatch.setattr("lab_lounge.characters._PROMPTS_DIR", isolated)
        (isolated / "system_aruka.txt").write_bytes(b"")
        report = preflight.Report()
        preflight._check_personas(report, [get_character("aruka")])
        assert _status(report, "persona/aruka").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK

    def test_unreadable_sample_is_a_row_instead_of_a_crash(self, isolated, monkeypatch):
        """代替のサンプルが読めないと、例外で残りの点検が 1 つも流れなかった。"""
        monkeypatch.setattr("lab_lounge.characters._PROMPTS_DIR", isolated)
        (isolated / "samples").mkdir()
        (isolated / "samples" / "sample_empath.md").write_bytes("人格".encode("cp932"))
        report = preflight.Report()
        preflight._check_personas(report, [get_character("mimi")])
        assert _status(report, "persona/mimi").status == "FAIL"

    def _production_turn(self):
        """本番の 1 ターンをダミーの LLM と TTS で流す。publish を止め、実リポの data/ も読まない。

        本流の 3 件は bus.publish、吹き出しは pipeline.publish を通るので、両方を止める。
        """
        from lab_lounge import bus, pipeline

        with patch.object(pipeline, "publish"), patch.object(bus, "publish"), \
                patch.object(pipeline, "_BUBBLE_MESSAGES", {}):
            return pipeline.run_pipeline(
                "ミミ様", stream_id="s", session_id="s", trace_id="t", speaker_hint="mimi",
                on_pose_ready=lambda *_args, **_kwargs: None,
            )

    def test_production_fails_the_turn_on_a_non_utf8_persona(self, isolated, monkeypatch):
        """FAIL の根拠。_routing_node が UnicodeDecodeError も受けるように直ったら、見直す。"""
        monkeypatch.setattr("lab_lounge.characters._PROMPTS_DIR", isolated)
        (isolated / "system_mimi.txt").write_bytes("人格".encode("cp932"))
        with pytest.raises(UnicodeDecodeError):
            self._production_turn()

    def test_production_runs_without_a_persona_on_an_empty_file(self, isolated, monkeypatch):
        """FAIL の根拠。空のファイルは例外にならず、人格なしのまま応答まで進むので、--paid でも気づけない。"""
        from lab_lounge.characters import load_system_prompt

        monkeypatch.setattr("lab_lounge.characters._PROMPTS_DIR", isolated)
        (isolated / "system_mimi.txt").write_bytes(b"")
        assert load_system_prompt(get_character("mimi")) == ""
        assert self._production_turn().speaker == "mimi"

    def test_personas_in_the_repository_are_not_a_failure(self, isolated):
        """同梱の人格で初回の点検が赤くならないこと。"""
        report = preflight.Report()
        preflight._check_personas(report, get_all_characters())
        assert not [c for c in report.checks if c.status == "FAIL"]


class TestTtsConfigRows:
    """設定が正しいときに赤くしない。必須キャラの誤った FAIL は、--paid を課金の前に止めて点検を通せなくする。"""

    def test_existing_reference_wav_has_no_row(self, isolated):
        os.environ["L2_USE_REAL_TTS"] = "true"
        (isolated / "ok.wav").write_bytes(b"RIFF")
        path = isolated / "voices-with-ref.json"
        path.write_text(json.dumps({"voices": {"mimi": {"ref_wav": "ok.wav"}}}), encoding="utf-8")
        os.environ["L2_IRODORI_VOICES_JSON"] = str(path)
        report = preflight.Report()
        preflight._check_tts_config(report, [SimpleNamespace(slug="mimi", tts_provider="irodori_vd", tts_voice="mimi")])
        assert report.checks == []

    def test_found_voicepeak_has_no_row(self, isolated):
        # sys.executable は必ず実在する実行ファイルなので、PATH や OS の違いに左右されない。
        os.environ["L2_USE_REAL_TTS"] = "true"
        os.environ["L2_TTS_VOICEPEAK_PATH"] = sys.executable
        report = preflight.Report()
        preflight._check_tts_config(report, [SimpleNamespace(slug="mimi", tts_provider="voicepeak", tts_voice="v")])
        assert report.checks == []


class TestTtsConfigReasons:
    """型の名前だけの行では、どのファイルを直せばよいか分からない。全文は画面に出さず問題ログへ残す。"""

    def test_missing_voices_json_says_what_to_fix(self, isolated):
        missing = isolated / f"{_MARK}.json"
        log = isolated / "problem.log"
        os.environ.update({"L2_USE_REAL_TTS": "true", "L2_IRODORI_VOICES_JSON": str(missing)})
        report = preflight.Report()
        preflight._check_tts_config(report, [get_character("mimi")], preflight._LogCapture(log))
        check = _status(report, "tts-config/mimi")
        assert check.status == "FAIL"
        assert "L2_IRODORI_VOICES_JSON" in check.detail
        assert _MARK not in check.detail
        assert _MARK in log.read_text(encoding="utf-8")

    def test_unregistered_voice_names_the_voice_and_keeps_the_rest_in_the_log(self, isolated, voices):
        log = isolated / "problem.log"
        os.environ["L2_USE_REAL_TTS"] = "true"
        character = SimpleNamespace(slug="mimi", tts_provider="irodori_vd", tts_voice=_MARK)
        report = preflight.Report()
        preflight._check_tts_config(report, [character], preflight._LogCapture(log))
        check = _status(report, "tts-config/mimi")
        assert check.status == "FAIL"
        assert f"voice={_MARK}" in check.detail
        assert "ValueError" in check.detail
        # 本文の「登録済み」の一覧は問題ログにだけ残る。
        assert "登録済み" not in check.detail
        assert "登録済み" in log.read_text(encoding="utf-8")

    def test_main_passes_the_capture_on(self, isolated, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(isolated / "none.json"))
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        assert "詳細: logs/preflight/" in capsys.readouterr().out
        logs = list((isolated / "logs" / "preflight").glob("*.log"))
        assert any("tts-config/mimi" in path.read_text(encoding="utf-8") for path in logs)


class TestReadingsRow:
    """壊れた読み辞書は irodori の発話を全部落とし、ない辞書は誤読を無音で始める。"""

    def _report(self, *slugs):
        os.environ["L2_USE_REAL_TTS"] = "true"
        report = preflight.Report()
        preflight._check_readings(report, [get_character(slug) for slug in slugs or ("mimi",)])
        return report

    @pytest.mark.parametrize("body", [b"{broken", '{"global": {"波心": "ハシン"}}'.encode("cp932"), b'{"characters": []}'])
    def test_broken_dictionary_is_fail_like_production(self, isolated, body):
        from lab_lounge import tts

        path = isolated / "readings.json"
        path.write_bytes(body)
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        check = _status(self._report(), "tts-config/readings")
        assert check.status == "FAIL"
        with pytest.raises(Exception):  # noqa: B017 - 型は入力ごとに違う。本番も落ちることだけを確かめる
            tts._apply_readings("x", "mimi")

    def test_failure_text_does_not_show_the_path_or_the_value(self, isolated, capsys):
        path = isolated / "PATHMARK" / "readings.json"
        path.parent.mkdir()
        path.write_text("{VALUEMARK", encoding="utf-8")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        check = _status(self._report(), "tts-config/readings")
        out = capsys.readouterr().out
        assert "PATHMARK" not in out and "VALUEMARK" not in out
        # 例外文の本文が切り詰められて印が消えても、型だけの形であることで気づけるようにする。
        assert re.fullmatch(r"[^\\/:]*\(JSONDecodeError\)", check.detail)

    def test_missing_dictionary_only_warns(self, isolated):
        """辞書は任意。声は出るので終了コードを変えない（OSS の利用者には普通にない）。"""
        from lab_lounge import tts

        os.environ["L2_IRODORI_READINGS_JSON"] = str(isolated / "no-such.json")
        report = self._report()
        assert _status(report, "tts-config/readings").status == "WARN"
        assert report.exit_code() == preflight.EXIT_OK
        assert tts._apply_readings("波心", "chisame") == "波心"

    def test_empty_dictionary_warns_like_a_missing_one(self, isolated):
        from lab_lounge import tts

        path = isolated / "readings.json"
        path.write_text("{}", encoding="utf-8")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        assert _status(self._report(), "tts-config/readings").status == "WARN"
        assert tts._apply_readings("波心", "chisame") == "波心"

    def test_valid_dictionary_has_no_row_and_production_applies_it(self, isolated):
        from lab_lounge import tts

        path = isolated / "readings.json"
        path.write_text(json.dumps({"global": {"波心": "ハシン"}}, ensure_ascii=False), encoding="utf-8")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        assert self._report().checks == []
        assert tts._apply_readings("波心", "chisame") == "ハシン"

    def test_not_read_without_real_tts(self, isolated):
        os.environ["L2_IRODORI_READINGS_JSON"] = str(isolated / "no-such.json")
        report = preflight.Report()
        preflight._check_readings(report, [get_character("mimi")])
        assert report.checks == []

    def test_not_read_when_no_selected_character_uses_irodori(self, isolated):
        path = isolated / "readings.json"
        path.write_text("{broken", encoding="utf-8")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        assert self._report("octamaid").checks == []

    def test_broken_dictionary_for_a_non_required_character_does_not_change_the_exit_code(self, isolated):
        path = isolated / "readings.json"
        path.write_text("{broken", encoding="utf-8")
        os.environ["L2_IRODORI_READINGS_JSON"] = str(path)
        report = self._report("aruka")
        assert _status(report, "tts-config/readings").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK


class TestEndpointRows:
    def test_recommended_voicevox_port_is_not_a_warning(self, isolated):
        """WARN の案内どおり 15021 にしたら警告が消えること。消えないと、案内に従っても赤いままになる。"""
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:15021"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_TTS_VOICEVOX_URL").status == "INFO"


class TestEndpointsProductionReads:
    """形の誤りは、今の設定で本番が読む接続先だけ FAIL にする。--static は配信の起動前ゲートになるので。"""

    @pytest.mark.parametrize("name,env,slug,status", [
        ("L2_C2_URL", {"L2_USE_C2_RETRIEVER": "false"}, "mimi", "WARN"),
        ("L2_C2_URL_READONLY", {"L2_USE_C2_RETRIEVER": "false"}, "mimi", "WARN"),
        ("L2_C2_URL", {**_C2_ON, "L2_ENABLE_RAG": "false"}, "mimi", "WARN"),
        ("L2_C2_URL", _C2_ON, "mimi", "FAIL"),
        ("L2_C2_URL_READONLY", _C2_ON, "mimi", "FAIL"),
        ("L2_TTS_VOICEVOX_URL", {}, "octamaid", "WARN"),  # ダミー TTS
        ("L2_TTS_IRODORI_URL", {}, "mimi", "WARN"),  # ダミー TTS
        ("L2_TTS_VOICEVOX_URL", _REAL_TTS, "mimi", "WARN"),  # mimi は irodori
        ("L2_TTS_IRODORI_URL", _REAL_TTS, "octamaid", "WARN"),  # octamaid は VOICEVOX
        ("L2_TTS_VOICEVOX_URL", _REAL_TTS, "octamaid", "FAIL"),
        ("L2_TTS_IRODORI_URL", _REAL_TTS, "mimi", "FAIL"),
    ])
    @pytest.mark.parametrize("value", [f"{_MARK}-not-a-url", f"127.0.0.1:18080/{_MARK}"])
    def test_only_what_production_reads_can_fail(self, isolated, capsys, name, env, slug, status, value):
        os.environ.update({name: value, **env})
        report = preflight.Report()
        preflight._check_endpoints(report, [get_character(slug)])
        assert _status(report, f"url/{name}").status == status
        assert _MARK not in capsys.readouterr().out

    @pytest.mark.parametrize("name", ["L2_C2_URL", "L2_TTS_VOICEVOX_URL", "L2_TTS_IRODORI_URL"])
    def test_blank_value_of_an_unused_endpoint_does_not_stop_the_run(self, isolated, name):
        os.environ.update({name: "", "L2_USE_C2_RETRIEVER": "false"})
        report = preflight.Report()
        preflight._check_endpoints(report, [get_character("mimi")])
        assert _status(report, f"url/{name}").status == "WARN"
        assert report.exit_code() == preflight.EXIT_OK

    def test_every_engine_counts_when_no_characters_are_given(self, isolated):
        os.environ.update({"L2_TTS_VOICEVOX_URL": "x", "L2_TTS_IRODORI_URL": "x", **_REAL_TTS})
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_TTS_VOICEVOX_URL").status == "FAIL"
        assert _status(report, "url/L2_TTS_IRODORI_URL").status == "FAIL"

    def test_engine_used_only_off_the_air_does_not_fail_the_run(self, isolated):
        os.environ.update({"L2_TTS_IRODORI_URL": "x", **_REAL_TTS})
        report = preflight.Report()
        preflight._check_endpoints(report, [get_character("aruka")])
        check = _status(report, "url/L2_TTS_IRODORI_URL")
        assert (check.status, check.required) == ("FAIL", False)
        assert report.exit_code() == preflight.EXIT_OK

    def test_run_static_passes_the_characters_on(self, isolated):
        with patch.object(preflight, "_check_endpoints") as endpoints, patch.object(preflight, "_check_numeric_env") as numeric:
            preflight.run_static(preflight.Report(), isolated, "dev", [get_character("mimi")], set())
        assert endpoints.call_args.args[1] == [get_character("mimi")]
        assert numeric.call_args.args[1] == [get_character("mimi")]

    def test_blank_c2_url_with_c2_off_exits_zero(self, isolated):
        (isolated / ".env").write_text(
            "REDIS_STREAM_KEY=aibyss:events-dev\nL2_USE_C2_RETRIEVER=false\nL2_C2_URL=\nL2_TTS_VOICEVOX_URL=\n", encoding="utf-8",
        )
        with patch.object(preflight, "_check_imports"):
            assert preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated) == preflight.EXIT_OK


class TestRedisUrlIsReadLikeRedisPy:
    """本番は redis.from_url で読む。urlsplit で host が取れるかどうかでは本番とずれる。"""

    @pytest.mark.parametrize("url", [f"unix:///tmp/{_MARK}.sock", f"unix:///tmp/{_MARK}.sock?db=1"])
    def test_unix_socket_passes_where_python_supports_it(self, isolated, monkeypatch, capsys, url):
        import redis

        redis.from_url(url)
        monkeypatch.setattr(socket, "AF_UNIX", 1, raising=False)
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "INFO"
        assert _MARK not in capsys.readouterr().out

    def test_unix_socket_fails_where_python_has_no_af_unix(self, isolated, monkeypatch):
        """Windows の Python には AF_UNIX がなく、本番の publish が毎回 AttributeError になる。"""
        monkeypatch.delattr(socket, "AF_UNIX", raising=False)
        os.environ["REDIS_URL"] = "unix:///tmp/redis.sock"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "FAIL"

    @pytest.mark.parametrize("url", ["redis://", "redis:///0", f"redis://:{_MARK}@/0"])
    def test_omitted_host_is_not_red(self, isolated, capsys, url):
        """redis-py は受けて localhost と読む。localhost と同じ遅れがあるので WARN までにする。"""
        import redis

        redis.from_url(url)
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "WARN"
        assert _MARK not in capsys.readouterr().out

    @pytest.mark.parametrize("url", [
        f"http://127.0.0.1:6379/{_MARK}",  # host は取れるが本番は拒否する
        f"redis://127.0.0.1:6379/?socket_timeout={_MARK}",
        "127.0.0.1:6379",
        "redis://127.0.0.1:abc",
        f"redis://:{_MARK}/x@127.0.0.1:6379",
    ])
    def test_what_redis_py_refuses_is_fail(self, isolated, capsys, url):
        import redis

        with pytest.raises(ValueError):
            redis.from_url(url)
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").status == "FAIL"
        assert _MARK not in capsys.readouterr().out


class TestVoicevoxUrl:
    """tts.py は VOICEVOX の URL だけ rstrip しない。50021 が使えないのは Windows だけ。"""

    @pytest.mark.parametrize("value", ["http://127.0.0.1:15021/", "http://127.0.0.1:15021//", "http://127.0.0.1:15021/base/"])
    def test_trailing_slash_warns(self, isolated, value):
        os.environ["L2_TTS_VOICEVOX_URL"] = value
        report = preflight.Report()
        preflight._check_endpoints(report)
        check = _status(report, "url/L2_TTS_VOICEVOX_URL")
        assert check.status == "WARN"
        assert "//audio_query" in check.detail
        assert report.exit_code() == preflight.EXIT_OK

    def test_production_sends_the_double_slash(self, isolated):
        """WARN の根拠。tts.py が rstrip するように直ったら、この行を消す。"""
        from lab_lounge import tts

        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:15021/"
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("stop")) as urlopen, \
                pytest.raises(ConnectionRefusedError):
            tts.synthesize("テスト", provider="voicevox", voice="89", speaker="octamaid", output_dir=str(isolated))
        assert "//audio_query" in urlopen.call_args.args[0].full_url

    def test_url_without_a_trailing_slash_has_no_such_warning(self, isolated):
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:15021"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_TTS_VOICEVOX_URL").status == "INFO"

    def test_the_slash_does_not_hide_the_localhost_warning(self, isolated):
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://localhost:15021/"
        report = preflight.Report()
        preflight._check_endpoints(report)
        details = [c.detail for c in report.checks if c.name == "url/L2_TTS_VOICEVOX_URL"]
        assert any("//audio_query" in d for d in details)
        assert any("localhost" in d for d in details)

    def test_other_urls_are_stripped_by_production(self, isolated):
        """irodori と C2 は本番が末尾の / を落とすので、同じ値でも警告しない。"""
        from lab_lounge import retriever, tts

        os.environ["L2_TTS_IRODORI_URL"] = "http://127.0.0.1:18080/"
        os.environ["L2_C2_URL"] = "http://127.0.0.1:8100/"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/L2_TTS_IRODORI_URL").status == "INFO"
        assert _status(report, "url/L2_C2_URL").status == "INFO"
        assert tts._irodori_url() == "http://127.0.0.1:18080/synthesize"
        assert retriever.C2Retriever(os.environ["L2_C2_URL"])._base_url == "http://127.0.0.1:8100"

    @pytest.mark.parametrize("platform,status,shown", [("win32", "WARN", "15021"), ("darwin", "INFO", "loopback:50021"), ("linux", "INFO", "loopback:50021")])
    def test_50021_advice_is_only_for_windows(self, isolated, monkeypatch, platform, status, shown):
        monkeypatch.setattr(preflight.sys, "platform", platform)
        os.environ["L2_TTS_VOICEVOX_URL"] = "http://127.0.0.1:50021"
        report = preflight.Report()
        preflight._check_endpoints(report)
        check = _status(report, "url/L2_TTS_VOICEVOX_URL")
        assert check.status == status
        assert shown in check.detail


class TestStreamRow:
    @pytest.mark.parametrize("profile,stream", [("dev", "aibyss:events-dev"), ("prod", "aibyss:events")])
    def test_stream_matching_the_profile_is_info_and_named(self, isolated, profile, stream):
        """想定どおりの組で WARN が出ると、毎回の警告に慣れて、dev の発話が本番の正史に入る事故を見逃す。"""
        os.environ["REDIS_STREAM_KEY"] = stream
        report = preflight.Report()
        preflight._check_stream(report, profile)
        check = _status(report, "stream")
        assert (check.status, check.detail) == ("INFO", stream)


class TestEventSchemaRow:
    """本番は publish の前に必ず検証するので、スキーマがないとダミー起動でも全ターンが落ちる。"""

    def _report(self):
        from lab_lounge import events

        events._reset_validator_cache()
        report = preflight.Report()
        preflight._check_event_schema(report)
        return report

    def _production(self):
        """本番が publish の前に通す関数を、同じ環境で呼ぶ。"""
        from lab_lounge import events

        events._reset_validator_cache()
        return events.build_utterance_final(text="x", stream_id="s", session_id="s", trace_id="t")

    def test_missing_schema_is_fail_and_does_not_show_the_path(self, isolated, capsys):
        os.environ["AIBYSS_SCHEMA_PATH"] = str(isolated / "PATHMARK" / "no-such.json")
        check = _status(self._report(), "schema")
        assert check.status == "FAIL"
        assert "AIBYSS_SCHEMA_PATH" in check.detail
        assert "PATHMARK" not in capsys.readouterr().out
        # 例外文のパスが切り詰められて印が消えても、区切り文字で気づけるようにする。
        assert not re.search(r"[\\/]", check.detail)

    @pytest.mark.parametrize("body", [b"{broken", '{"type": "object"}'.encode("utf-16"), b'{"type": 1}'])
    def test_unreadable_schema_is_fail_like_production(self, isolated, body):
        """壊れた JSON、UTF-8 でないファイル、スキーマとして不正なものも、本番は同じく全ターン落ちる。"""
        path = isolated / "broken.json"
        path.write_bytes(body)
        os.environ["AIBYSS_SCHEMA_PATH"] = str(path)
        assert _status(self._report(), "schema").status == "FAIL"
        with pytest.raises(Exception):  # noqa: B017 - 型は入力ごとに違う。本番も落ちることだけを確かめる
            self._production()

    def test_readable_schema_has_no_row_and_production_passes(self, isolated):
        assert not [c for c in self._report().checks if c.name == "schema"]
        assert self._production()["type"] == "utterance.final"

    def test_clone_without_the_sibling_workspace_is_fail_like_production(self, isolated, monkeypatch):
        """AIBYSS_SCHEMA_PATH がなく、aibyss-workspace も併置していない clone で最初に踏む。"""
        from lab_lounge import events

        os.environ.pop("AIBYSS_SCHEMA_PATH", None)
        monkeypatch.setattr(events, "__file__", str(isolated / "lone" / "src" / "lab_lounge" / "events.py"))
        assert _status(self._report(), "schema").status == "FAIL"
        with pytest.raises(RuntimeError):
            self._production()

    def test_production_really_fails_without_the_schema(self, isolated):
        """FAIL の根拠。本番が既定のスキーマへ落ちるように直ったら、FAIL を見直す。"""
        os.environ["AIBYSS_SCHEMA_PATH"] = str(isolated / "no-such.json")
        with pytest.raises(RuntimeError):
            self._production()

    def test_paid_stops_before_billing(self, isolated, voices, capsys):
        from lab_lounge import events

        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices), AIBYSS_SCHEMA_PATH=str(isolated / "no-such.json"))
        events._reset_validator_cache()
        with patch.object(preflight, "run_paid") as run:
            code = preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        run.assert_not_called()
        assert "[FAIL ] schema" in capsys.readouterr().out


class TestStreamContextRow:
    def test_missing_file_is_reported_by_name(self, isolated):
        """ファイルがなくても配信は動くが、行がないと、今日の文脈が prompt に入らないことに気づけない。"""
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        check = _status(report, "stream_context")
        assert check.status == "INFO" and "no-such-context.md" in check.detail

    def test_unset_env_looks_at_the_default_path(self, isolated):
        """本番は未設定なら既定の current.md を読む。点検が見ないと、古い文脈が全キャラの prompt に入るのを見逃す。"""
        del os.environ["L2_STREAM_CONTEXT_FILE"]
        default = isolated / "current.md"
        default.write_text("古い配信の文脈", encoding="utf-8")
        old = 1_000_000_000
        os.utime(default, (old, old))
        report = preflight.Report()
        with patch("lab_lounge.stream_context._DEFAULT_PATH", default):
            preflight._check_stream_context(report, isolated)
        assert _status(report, "stream_context").status == "WARN"

    def test_fresh_stream_context_row_hides_the_body(self, isolated, capsys):
        """古いときの WARN の行は別のテストが見ている。普段の配信で出るのはこちらの行で、配信外のメモが本文に入りうる。"""
        path = isolated / "current.md"
        path.write_text(BODY, encoding="utf-8")
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        assert "更新" in _status(report, "stream_context").detail  # 本文を読む経路を通ったことの確認
        assert not _shows_answer(capsys.readouterr().out)


class TestObsRow:
    """OBS がなくても配信は続くので FAIL にはしない。ただし立ち絵が動かないことは、必ず行で知らせる。"""

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_url_warns_by_name_without_connecting(self, isolated, monkeypatch, value):
        """URL を入れ忘れただけなのに「接続できません」と出ると、OBS 側を調べに行ってしまう。"""
        if value is not None:
            os.environ["L2_OBS_WS_URL"] = value
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        check = _status(report, "obs")
        assert check.status == "WARN"
        assert "L2_OBS_WS_URL" in check.detail
        fake.ReqClient.assert_not_called()

    def test_missing_package_is_a_warning_that_names_it(self, isolated, monkeypatch):
        """素の uv sync で extras が消えると、立ち絵だけが動かないまま配信に入る。入れ直すものの名前を出す。"""
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        # sys.modules に None を入れると、その import は ImportError になる。
        monkeypatch.setitem(sys.modules, "obsws_python", None)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        check = _status(report, "obs")
        assert check.status == "WARN"
        assert "obsws-python" in check.detail

    def test_every_group_present_is_pass(self, isolated, monkeypatch):
        """立ち絵が揃った正常な状態。WARN のままだと、OBS のシーンを直しても直ったことが分からない。"""
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        fake = MagicMock()
        fake.ReqClient.return_value.get_group_scene_item_list.side_effect = (
            lambda slug: SimpleNamespace(scene_items=[{"sourceName": f"{slug}_neutral", "sceneItemId": 1}])
        )
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi"), get_character("sakura")])
        assert _status(report, "obs").status == "PASS"


class TestPoseSources:
    """グループがあっても中のソース名が規約と違うと、set_pose は何もしない。"""

    def _fake_obs(self, monkeypatch, items):
        fake = MagicMock()
        fake.ReqClient.return_value.get_group_scene_item_list.return_value = SimpleNamespace(scene_items=items)
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:4455"
        return fake

    @pytest.mark.parametrize("items,status", [
        ([{"sourceName": "mimi_neutral", "sceneItemId": 7}], "PASS"),
        ([{"sourceName": "mimi-立ち絵", "sceneItemId": 7}], "WARN"),
        ([{"sourceName": "mimi_happy", "sceneItemId": 7}], "WARN"),
        ([{"sourceName": "mimi_neutral"}], "WARN"),
        ([], "WARN"),
    ])
    def test_row_agrees_with_what_set_pose_does(self, isolated, monkeypatch, items, status):
        """同じ OBS に対する本番の set_pose と判定が一致すること。規約が変わったらここで分かる。"""
        from lab_lounge import obs

        fake = self._fake_obs(monkeypatch, items)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        assert _status(report, "obs").status == status
        assert report.exit_code() == preflight.EXIT_OK
        obs.set_pose("mimi", "neutral")
        assert fake.ReqClient.return_value.set_scene_item_enabled.called is (status == "PASS")

    def test_does_not_switch_the_pose_and_leaves_no_state(self, isolated, monkeypatch):
        """点検で立ち絵を動かさない。接続とキャッシュも残さない。"""
        from lab_lounge import obs

        fake = self._fake_obs(monkeypatch, [{"sourceName": "mimi_neutral", "sceneItemId": 7}])
        preflight._check_obs(preflight.Report(), [get_character("mimi")])
        client = fake.ReqClient.return_value
        client.set_scene_item_enabled.assert_not_called()
        client.disconnect.assert_called_once()
        assert obs._client is None
        assert obs._init_attempted is False
        assert obs._item_id_cache == {}

    def test_connection_failure_goes_to_the_problem_log_without_the_host(self, isolated, monkeypatch, capsys):
        from lab_lounge import obs

        fake = self._fake_obs(monkeypatch, [])
        fake.ReqClient.side_effect = ConnectionRefusedError("HOSTMARK:4455")
        log = isolated / "problems.log"
        report = preflight.Report()
        with preflight._captured_logs(log):
            preflight._check_obs(report, [get_character("mimi")])
        check = _status(report, "obs")
        assert check.status == "WARN"
        assert "logs/preflight/" in check.detail
        assert obs._init_attempted is False
        assert "ConnectionRefusedError" in log.read_text(encoding="utf-8")
        assert "HOSTMARK" not in log.read_text(encoding="utf-8")
        assert "HOSTMARK" not in capsys.readouterr().out

    def test_main_points_at_the_log_that_the_row_mentions(self, isolated, monkeypatch, capsys):
        """行の案内が嘘にならないこと。実際に logs/preflight/ に型が残り、末尾の「詳細:」で示される。"""
        fake = self._fake_obs(monkeypatch, [])
        fake.ReqClient.side_effect = ConnectionRefusedError("HOSTMARK:4455")
        (isolated / ".env").write_text("L2_OBS_WS_URL=ws://127.0.0.1:4455\nREDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        assert preflight.main(["--static", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated) == preflight.EXIT_OK
        assert "詳細: logs/preflight/" in capsys.readouterr().out
        logs = list((isolated / "logs" / "preflight").glob("*.log"))
        assert "ConnectionRefusedError" in logs[0].read_text(encoding="utf-8")


# ─── raw 経路と本応答の点検: 変異テストで見つかった検知の穴 ──────────────────

def _run_main_check(capture, *, result=None, build=None, invoke_error=None):
    """check_agent を Agent の経路で 1 回流し、(main/mimi の行, report) を返す。LLM は呼ばない。

    Args:
        capture:      ログの捕捉先
        result:       agent.invoke の戻り値
        build:        _build_agent_graph の代役。省略すると Agent を作れたことにする
        invoke_error: agent.invoke で起こす例外
    """
    os.environ["L2_ENABLE_TOOLS"] = "true"
    invoke = MagicMock(return_value=result, side_effect=invoke_error)
    report = preflight.Report()
    with patch("lab_lounge.graph._build_agent_graph", build or MagicMock(return_value=object())), \
            patch("lab_lounge.graph._invoke_agent_with_timeout", invoke):
        preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
    return _status(report, "main/mimi"), report


class TestRawAndAgentProblemLog:
    """画面には要約しか出さない例外の全文を、問題ログへ残すこと。

    WHY: 画面の行は型と status だけなので、ログに残らないと FAIL の原因を調べる手がかりがなくなる。
         「詳細:」の案内も wrote_problems を見ているので、一緒に消える。
    """

    def test_raw_path(self, isolated):
        path = isolated / "logs" / "p.log"
        capture = preflight._LogCapture(path)

        def boom():
            raise _FakeApiError(BODY)

        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", boom)
        assert capture.wrote_problems is True
        assert BODY in path.read_text(encoding="utf-8")
        assert "BODYSENTINEL" not in _status(report, "router").detail

    def test_agent_path(self, isolated):
        path = isolated / "logs" / "p.log"
        capture = preflight._LogCapture(path)
        check, _ = _run_main_check(capture, invoke_error=_FakeApiError(BODY))
        assert capture.wrote_problems is True
        assert BODY in path.read_text(encoding="utf-8")
        assert "BODYSENTINEL" not in check.detail


class TestFailReasons:
    """FAIL の行だけを読んで、止まった理由が分かること。"""

    def test_blank_raw_answer_without_logs_still_gives_a_reason(self, isolated, capture):
        """本番の関数が空文字を返すと、ログには何も残らない。それでも理由の欄を空にしない。"""
        report = preflight.Report()
        preflight._check_raw_llm(report, capture, "router", "m", lambda: "")
        assert "()" not in _status(report, "router").detail

    def test_no_tools_without_logs_still_gives_a_reason(self, isolated, capture):
        """ツールが 0 件のときの本番のログは INFO なので、拾える問題ログがない。理由の欄を空にしない。"""
        check, _ = _run_main_check(capture, build=MagicMock(return_value=None))
        assert check.status == "FAIL"
        assert "()" not in check.detail

    def test_no_tools_reports_the_logged_reason(self, isolated, capture):
        """ツールの読み込みに失敗した理由はログにしかないので、FAIL の説明に拾う。"""
        def build(*args, **kwargs):
            capture.emit(_log_record("lab_lounge.graph", logging.WARNING, "ツール読み込み失敗: %s", ("mcp-boom",)))
            return None

        check, _ = _run_main_check(capture, build=build)
        assert check.status == "FAIL"
        assert "mcp-boom" in check.detail

    def test_missing_structured_output_is_told_from_a_blank_answer(self):
        """どちらも FAIL だが、調べる場所が違う。同じ説明だと、SDK の破損をモデルの空応答と読み違える。"""
        missing = preflight._summarize_agent_result(_agent_result(None))
        blank = preflight._summarize_agent_result(_agent_result(SimpleNamespace(response="   ")))
        assert preflight._agent_problems(missing, [])
        assert preflight._agent_problems(blank, [])
        assert preflight._agent_problems(missing, []) != preflight._agent_problems(blank, [])

    @pytest.mark.parametrize("tool_calls,shown", [
        ([], "called=-"),  # README の出力例と同じ形
        ([{"name": "web_search_tool"}, {"name": "retrieve_memory_tool"}], "called=web_search_tool,retrieve_memory_tool"),
    ])
    def test_row_shows_which_tools_ran(self, isolated, capture, tool_calls, shown):
        """ツールの列が読めないと、プローブがツールを呼んでしまったのかを行から判断できない。"""
        check, _ = _run_main_check(capture, result=_agent_result(SimpleNamespace(response="x"), tool_calls=tool_calls))
        assert shown in check.detail


class TestAgentSummary:
    """agent.invoke の結果の読み方。SDK の更新で形が変わる場所なので、本番と同じ読み方を固定する。"""

    def test_output_tokens_reach_the_report(self, isolated, capture):
        """出力の tokens が集計から漏れると、点検 1 回の課金を少なく見積もってしまう。"""
        _, report = _run_main_check(capture, result=_agent_result(SimpleNamespace(response="x"), steps=2))
        assert (report.input_tokens, report.output_tokens) == (20, 10)

    @pytest.mark.parametrize("structured,status", [
        ({"response": "こんにちは"}, "PASS"),
        ({"response": "   "}, "FAIL"),
        ({"speed": 100}, "FAIL"),
    ])
    def test_dict_structured_response_is_read(self, isolated, capture, structured, status):
        """本番の _run_agent は dict の構造化出力も受け取る。

        WHY: 点検が dict の中身を読まないと、SDK の更新で dict が返るようになったときに、本番では
             声が出るのに点検だけが「応答が空」と FAIL を出して、配信前に止めてしまう。
        """
        check, _ = _run_main_check(capture, result=_agent_result(structured))
        assert check.status == status

    def test_real_langchain_messages_are_counted_by_ai_turns_only(self):
        """本物の LangChain のメッセージで、AI の段だけを読むこと。

        WHY: 構造化出力を tool で受ける戦略では、最後のメッセージが ToolMessage になる。AI の段に絞らないと
             finish_reason が空に見えて、切れた応答を緑にしてしまう。段数の列も人の発言や tool の結果の分だけ増える。
             本物のクラスを使うのは、SDK の更新で type の値が変わったときにここで気づくため。
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result = {
            "structured_response": SimpleNamespace(response="x"),
            "messages": [
                HumanMessage(content="質問"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "web_search_tool", "args": {}, "id": "1"}],
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    response_metadata={"finish_reason": "tool_calls"},
                ),
                ToolMessage(content="検索結果", tool_call_id="1"),
                # usage_metadata の既定は None。報告しないプロバイダの段でも読めること。
                AIMessage(content="切れた応答", response_metadata={"finish_reason": "length"}),
                ToolMessage(content="構造化出力を受け取りました", tool_call_id="2"),
            ],
        }
        summary = preflight._summarize_agent_result(result)
        assert summary["steps"] == 2
        assert summary["finish"] == "length"
        assert (summary["input_tokens"], summary["output_tokens"]) == (10, 5)
        assert summary["tools"] == ["web_search_tool"]

    @pytest.mark.parametrize("message", [
        SimpleNamespace(type="ai"),
        SimpleNamespace(type="ai", usage_metadata=None, response_metadata=None, tool_calls=None),
    ])
    def test_ai_message_without_optional_fields_can_be_summarized(self, message):
        """SDK の更新で省略できる属性が消えたり None になったりしても、要約を作れること。

        WHY: 読めないだけで FAIL にすると、本番では声が出るのに、点検が配信の前に止めてしまう。
        """
        summary = preflight._summarize_agent_result({"structured_response": SimpleNamespace(response="x"), "messages": [message]})
        assert (summary["input_tokens"], summary["output_tokens"], summary["tools"], summary["finish"]) == (0, 0, [], "")

    def test_result_without_messages_is_a_fail_row_not_a_crash(self, isolated, capture):
        """messages が空でも要約を作り、FAIL の 1 行にする。例外で点検全体を止めない。"""
        summary = preflight._summarize_agent_result({"structured_response": None, "messages": []})
        assert (summary["steps"], summary["finish"]) == (0, "")
        check, _ = _run_main_check(capture, result={})
        assert check.status == "FAIL"


class TestApiKind:
    """「gpt-5.5 の Chat + tools が通るか」を行で答える。応答の response_metadata から読む。"""

    @pytest.mark.parametrize("metadata,kind", [
        ({"model_provider": "openai", "finish_reason": "tool_calls"}, "chat"),
        ({"model_provider": "openai", "object": "response", "status": "completed"}, "responses"),
        ({"object": "response"}, "responses"),
        ({"model_provider": "anthropic", "stop_reason": "tool_use"}, "anthropic"),
        ({"model_provider": "google_genai", "finish_reason": "STOP"}, "google_genai"),
        ({}, "?"),
    ], ids=["chat", "responses", "responses-without-provider", "anthropic", "google", "unknown"])
    def test_reads_the_api_from_the_metadata(self, metadata, kind):
        assert preflight._api_kind(metadata) == kind

    @pytest.mark.parametrize("use_responses,kind", [(False, "chat"), (True, "responses")], ids=["chat", "responses"])
    def test_real_sdk_answers_tell_the_two_apart(self, isolated, use_responses, kind):
        """SDK の更新で印が変わると、行が別の API を名乗る。実物の SDK に応答を解釈させて確かめる。"""
        pytest.importorskip("langchain_openai")
        import httpx
        from langchain_openai import ChatOpenAI

        chat = {
            "id": "c", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        responses = {
            "id": "r", "object": "response", "created_at": 1, "model": "m", "status": "completed",
            "output": [{
                "type": "message", "id": "m1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "x", "annotations": []}],
            }],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

        def answer(request):
            return httpx.Response(200, json=responses if request.url.path.endswith("/responses") else chat)

        model = ChatOpenAI(
            model="m", api_key="dummy", base_url="http://fake.invalid/v1", max_retries=0, use_responses_api=use_responses,
            http_client=httpx.Client(transport=httpx.MockTransport(answer)),
        )
        assert preflight._api_kind(model.invoke("x").response_metadata) == kind

    @pytest.mark.parametrize("metadata,shown", [
        ({"model_provider": "openai", "finish_reason": "stop"}, "api=chat"),
        ({"model_provider": "openai", "object": "response"}, "api=responses"),
    ], ids=["chat", "responses"])
    def test_main_row_shows_the_api_and_the_called_tools(self, isolated, capture, metadata, shown):
        """tools=- は「ツールなしで流した」と読める。呼んだツールの列だと分かる名前にする。"""
        message = SimpleNamespace(type="ai", usage_metadata={}, response_metadata=metadata, tool_calls=[])
        result = {"structured_response": SimpleNamespace(response="x"), "messages": [message]}
        check, _ = _run_main_check(capture, result=result)
        assert check.status == "PASS"
        assert shown in check.detail
        assert "called=-" in check.detail
        assert "tools=" not in check.detail

    def test_api_comes_from_the_last_answer(self):
        """複数段のとき、最後に受けた応答で読む（finish と同じ）。"""
        messages = [
            SimpleNamespace(type="ai", usage_metadata={}, response_metadata=metadata, tool_calls=[])
            for metadata in ({"model_provider": "anthropic"}, {"model_provider": "openai"})
        ]
        summary = preflight._summarize_agent_result({"structured_response": None, "messages": messages})
        assert summary["api"] == "chat"

    def test_no_answer_is_unknown(self):
        assert preflight._summarize_agent_result({"structured_response": None, "messages": []})["api"] == "?"


class TestAgentAudit:
    """GraphRecursionError で終わると戻り値がない。課金した回数と、検証に落ちた応答の中身を失わないこと。"""

    @staticmethod
    def _answer():
        """検証に落ち続ける応答の形。usage は 1 回あたり in=100 / out=7。"""
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="",
            tool_calls=[{"name": "SakuraResponse", "args": {"say": "x"}, "id": "1"}],
            usage_metadata={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
        )

    def test_counts_through_the_real_callback_manager(self):
        """LangChain が handler に求める属性が足りないと、例外が握られて無音で数えなくなる。実物の経路で確かめる。"""
        import itertools

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        model = GenericFakeChatModel(messages=itertools.repeat(self._answer()))
        audit = preflight._AgentAudit()
        config = {"callbacks": [audit.callback()]}
        model.invoke("x", config=config)
        model.invoke("x", config=config)
        assert audit.calls == 2
        assert (audit.input_tokens, audit.output_tokens) == (200, 14)
        assert "SakuraResponse" in audit.last_output

    def test_recursion_failure_keeps_the_count_and_the_last_answer(self, isolated, tmp_path):
        """実物の LangGraph を上限まで回す。行に回数が出て、集計へ足され、応答の全文は問題ログにだけ残る。"""
        import itertools

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langgraph.graph import START, MessagesState, StateGraph

        model = GenericFakeChatModel(messages=itertools.repeat(self._answer()))

        def call_model(state, config):
            return {"messages": [model.invoke(state["messages"], config)]}

        builder = StateGraph(MessagesState)
        builder.add_node("model", call_model)
        builder.add_edge(START, "model")
        builder.add_edge("model", "model")

        os.environ["L2_ENABLE_TOOLS"] = "true"
        log = tmp_path / "problem.log"
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=builder.compile()):
            ok = preflight.check_agent(report, preflight._LogCapture(log), get_character("sakura"), None, required=True)
        limit = preflight._AGENT_RECURSION_LIMIT
        check = _status(report, "main/sakura")
        assert ok is False
        assert check.status == "FAIL"
        assert "GraphRecursionError" in check.detail and f"(LLM {limit} 回)" in check.detail
        assert "SakuraResponse" not in check.detail
        assert (report.input_tokens, report.output_tokens) == (100 * limit, 7 * limit)
        assert f"最後の応答 ({limit} 回目)" in log.read_text(encoding="utf-8")
        assert "SakuraResponse" in log.read_text(encoding="utf-8")

    def test_collab_row_shows_the_count_too(self, isolated, tmp_path):
        """協働の呼び出し側も同じ Agent なので、検証に落ち続ければ同じだけ課金する。"""
        from langchain_core.outputs import ChatGeneration, LLMResult

        def fake_invoke(_agent, _input, config, **_kwargs):
            callback = config["callbacks"][0]
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=self._answer())]]))
            raise RuntimeError("stop")

        report = preflight.Report()
        capture = preflight._LogCapture(tmp_path / "problem.log")
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke):
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        assert "(LLM 1 回)" in _status(report, "collab/sakura->chisame").detail
        assert report.input_tokens == 100

    def test_tokens_are_not_counted_twice_when_the_failure_comes_after_the_agent(self, isolated, capture):
        """Agent が返った後の TTS の待ちで落ちても、集計はすでに戻り値から足してある。"""
        from langchain_core.outputs import ChatGeneration, LLMResult

        def fake_invoke(_agent, _input, config, **_kwargs):
            callback = config["callbacks"][0]
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=self._answer())]]))
            return _agent_result(SimpleNamespace(response="x"), tool_calls=[{"name": "ask_character_tool"}])

        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke), \
                patch("lab_lounge.mcp_servers.ask_character.wait_bg_tts_complete", side_effect=RuntimeError("stop")):
            preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        check = _status(report, "collab/sakura->chisame")
        assert check.status == "FAIL"
        assert "LLM" not in check.detail
        # 戻り値の分（_agent_result は 1 段あたり in=10）だけが足され、コールバックの分（in=100）は足されない。
        assert report.input_tokens == 10

    def test_no_call_adds_nothing_to_the_row(self, isolated, capture):
        """呼ぶ前に落ちたときに「LLM 0 回」と出すと、課金があったように読める。"""
        audit = preflight._AgentAudit()
        assert audit.record_failure(preflight.Report(), capture, "main/x") == ""

    def test_broken_langchain_core_stays_on_the_row(self, isolated, capture):
        """langchain_core の import が壊れても、点検の行に留めて残りを流す。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch.dict(sys.modules, {"langchain_core.callbacks": None}):
            ok = preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert ok is False
        assert "ImportError" in _status(report, "main/mimi").detail or "ModuleNotFoundError" in _status(report, "main/mimi").detail


class TestProbeTextAndListRepr:
    """モデルへ送る点検の発話と、応答が list の repr かどうかの見分け。"""

    @pytest.mark.parametrize("wake_word,shown,hidden", [
        ("呼び名", "呼び名", "表示名"),  # 本番でルカが呼びかける形。表示名だと本番と違う入力を測ることになる
        (None, "表示名", "None"),  # ruka は wake word がない。そのまま埋めると「None、」とモデルへ送ってしまう
    ])
    def test_probe_calls_the_character_as_on_stream(self, wake_word, shown, hidden):
        text = preflight._probe_text(SimpleNamespace(wake_word=wake_word, display_name="表示名"))
        assert shown in text
        assert hidden not in text

    @pytest.mark.parametrize("text,status", [
        ("\n ['y']\n", "FAIL"),  # 前後に空白が付いても list の repr で、本番はそのまま読み上げてしまう
        ("[1], [2]", "PASS"),  # literal_eval では tuple。引用番号のような文で、list の repr ではない
        ("(['y'])", "PASS"),  # str(list) は必ず [ で始まるので、これも list の repr ではない
    ])
    def test_single_node_flags_only_list_reprs(self, isolated, capture, text, status):
        """list の repr だけを赤くする。見逃すと読み上げ事故になり、誤って赤くすると点検を信じられなくなる。"""
        fake = SimpleNamespace(text=text, input_tokens=1, output_tokens=1, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _status(report, "main/mimi").status == status


# ─── 前提サービスと外部サービスの点検 ──────────────────────────────

def _run_tavily(capture, *, returns=None, raises=None) -> preflight.Report:
    """TavilySearch を差し替えて check_tavily を 1 回流す。

    Args:
        capture: 問題ログの受け手
        returns: invoke の戻り値
        raises:  コンストラクタで投げる例外

    Returns:
        結果の入った Report
    """
    os.environ["L2_ENABLE_TOOLS"] = "true"
    search = MagicMock(side_effect=raises)
    search.return_value.invoke.return_value = returns
    report = preflight.Report()
    with patch("langchain_tavily.TavilySearch", search):
        preflight.check_tavily(report, capture)
    return report


def _run_embeddings_raising(capture, exc: Exception) -> None:
    """埋め込みの呼び出しが例外になる状況で check_embeddings を 1 回流す。"""
    os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
    with patch("lab_lounge.retriever._embed", side_effect=exc):
        preflight.check_embeddings(preflight.Report(), capture)


class TestTavilyRow:
    """Tavily の戻り値の形ごとの判定と、利用者が読む理由を確かめる。"""

    @pytest.mark.parametrize("value", [
        None,
        "error: quota exceeded",  # 文字列の中に error の字があっても、dict ではない
        ["error"],
        {},
        {"results": []},
        {"results": None},
        {"error": "quota", "results": [{"title": "t", "url": "u", "content": "c"}]},
    ])
    def test_unusable_shapes_are_one_fail_row(self, isolated, capture, value):
        """使えない形は、点検を止めずに 1 行の FAIL にする。

        WHY: この判定は try の外にある。形を読み違えて例外になると点検全体が止まって要約も出ず、
             空の結果を PASS にすると、検索が死んだまま配信に入る。
             本物のライブラリは今のところ空の結果を例外にするので、ライブラリを通すテストではここへ
             届かない。更新で戻り値が変わっても赤くなるように、形を直接渡して確かめる。
        """
        report = _run_tavily(capture, returns=value)
        assert _status(report, "tavily").status == "FAIL"

    def test_exception_in_the_error_key_tells_the_reason_on_one_row(self, isolated, capture, capsys):
        """キー切れやクレジット切れは {"error": ValueError("Error 401: ...")} の形で届く。

        WHY: 「結果が空です」では、キーを直すべきなのか検索の問題なのかを読み取れない。
             本文の改行や長さをそのまま出すと、1 行 1 経路の表が崩れて他の行が読みにくくなる。
        """
        _run_tavily(capture, returns={"error": ValueError("Error 401: Unauthorized\n" + "x" * 1000)})
        rows = capsys.readouterr().out.splitlines()
        assert len(rows) == 1
        assert "FAIL" in rows[0]
        assert "ValueError" in rows[0] and "401" in rows[0]
        assert len(rows[0]) < 1000

    def test_string_error_body_stays_off_the_screen(self, isolated, capture, capsys):
        """文字列の error は API の応答の本文で、入力のエコーが入りうる。

        WHY: 点検の出力は配信中のコンソールや記録に貼られることがあるので、本文を画面へ出さない約束がある。
        """
        report = _run_tavily(capture, returns={"error": f"Invalid query: {BODY}"})
        assert _status(report, "tavily").status == "FAIL"
        assert "BODYSENTINEL" not in capsys.readouterr().out

    def test_tavily_error_detail_is_not_shown(self, isolated, capture):
        """HTTP エラーの本文は API の応答なので、status だけを出す。"""
        pytest.importorskip("langchain_tavily")
        os.environ.update({"L2_ENABLE_TOOLS": "true", "TAVILY_API_KEY": "dummy"})
        response = MagicMock(status_code=401, reason="reason")
        response.json.return_value = {"detail": {"error": BODY}}
        report = preflight.Report()
        with patch("requests.post", return_value=response):
            preflight.check_tavily(report, capture)
        detail = _status(report, "tavily").detail
        assert not _shows_answer(detail)
        assert "401" in detail

    def test_other_value_error_is_described_as_before(self, isolated, capture):
        """status の形でない ValueError を status の行にしない。"""
        report = _run_tavily(capture, returns={"error": ValueError("bad parameter")})
        assert "bad parameter" in _status(report, "tavily").detail


class TestServiceProblemLog:
    """失敗の全文が問題ログに残ること。

    壊れると FAIL の原因を調べる手がかりが消え、「詳細:」の案内も出なくなる。API の応答に由来する例外は
    画面に本文を出さないので、全文を読める場所はここしかない。
    """

    @pytest.mark.parametrize("run", [
        lambda capture: _run_tavily(capture, raises=_FakeApiError(BODY)),
        lambda capture: _run_tavily(capture, returns={"error": ValueError(BODY)}),
        lambda capture: _run_embeddings_raising(capture, _FakeApiError(BODY)),
    ], ids=["tavily-raises", "tavily-error-key", "embeddings-raises"])
    def test_failure_text_reaches_the_problem_log(self, isolated, run):
        path = isolated / "logs" / "preflight" / "p.log"
        capture = preflight._LogCapture(path)
        run(capture)
        assert capture.wrote_problems is True
        assert "BODYSENTINEL" in path.read_text(encoding="utf-8")


class TestServiceLiveness:
    """前提サービスの生死の見方。読み違えると、止まるべきところで課金が始まる。"""

    def test_http_ok_gives_up_on_a_scheme_it_cannot_speak(self):
        """繋ぐ先が決まらない URL を「届く」にすると、C2 なしで課金が始まる。"""
        assert preflight._http_ok("c2://c2.invalid/health") is False

    def test_provider_without_a_local_engine_counts_as_usable(self):
        """edge_tts は README が載せるプロバイダで、キャラ単位で選べる。ローカルのエンジンを持たない。

        WHY: 確かめる相手がないので止めない。False にすると、edge_tts へ切り替えた利用者の --paid は
             「前提サービスがない」で必ず止まる。別のエンジン名を返すと、止まった理由も誤る。
        """
        assert preflight.tts_engine_status(SimpleNamespace(tts_provider="edge_tts")) == (True, "edge_tts")

    def test_dead_engine_of_a_non_required_character_does_not_fail_the_run(self, isolated):
        """別枠のキャラの TTS が落ちていても、必須の経路が緑なら配信できる。行は FAIL のまま見せる。"""
        guest = SimpleNamespace(slug="guest", tts_provider="voicevox")
        report = preflight.Report()
        with patch.object(preflight, "_http_ok", return_value=False):
            assert preflight.check_tts_engine(report, guest, required=False) is False
        assert _status(report, "tts/guest").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK

    @pytest.mark.parametrize("readonly", [None, "", "   "])
    def test_no_readonly_row_unless_it_is_configured(self, isolated, readonly):
        """副 C2 は任意。未設定なのに行が出ると、毎回 WARN が混ざって本物の警告が埋もれる。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"})
        if readonly is not None:
            os.environ["L2_C2_URL_READONLY"] = readonly
        response = MagicMock()
        response.read.return_value = b'{"search_mode": "semantic"}'
        response.__enter__.return_value = response
        report = preflight.Report()
        with patch("urllib.request.urlopen", return_value=response):
            preflight.check_c2(report)
        assert [check.name for check in report.checks] == ["c2"]


class TestMissingServicesList:
    """課金の前に止めるときに挙げる、足りないサービスの一覧。"""

    def test_engines_that_are_up_are_not_listed(self, isolated):
        """動いているエンジンを足りない扱いにすると、環境が健全でも --paid が一度も流せない。"""
        # 必須のキャラには今 voicepeak がいないので、足さないと shutil.which の経路を通らない。
        characters = [*get_all_characters(), SimpleNamespace(slug="mimi", tts_provider="voicepeak")]
        with patch.object(preflight, "_http_ok", return_value=True), \
                patch("shutil.which", return_value="voicepeak.exe") as which, patch("redis.Redis.from_url"):
            assert preflight.missing_services(characters, with_pipeline=True) == []
        which.assert_called()

    def test_engine_shared_by_two_characters_is_named_once(self, isolated):
        """同じ名前が並ぶと、課金の前に止めた理由の行で何を起動すればよいのかが読みにくい。"""
        shared = [SimpleNamespace(slug="mimi", tts_provider="voicevox"), SimpleNamespace(slug="sakura", tts_provider="voicevox")]
        with patch.object(preflight, "_http_ok", return_value=False), patch("redis.Redis.from_url"):
            assert preflight.missing_services(shared, with_pipeline=True) == ["voicevox"]


# ─── 記憶の経路 ───────────────────────────────────────────────────
#
# 本番は記憶の検索の失敗を文字列や空リストに変えて LLM へ渡すので、声は止まらず記憶だけが無音で消える。
# ここでは本番の _retrieve_memory_impl、_build_retriever_from_env、C2Retriever をそのまま通し、
# 通信の手前（httpx の送信と埋め込み）だけを差し替える。

def _write_seed_index(root):
    """読める最小の seed の索引を作り、その場所を返す。本文にセンチネルを入れて、表示の検査に使う。"""
    import numpy as np

    index = root / "index"
    index.mkdir(exist_ok=True)
    (index / "chunks.json").write_text(
        json.dumps([{"doc_id": "seed-1", "text": BODY, "source": "handbook.md"}], ensure_ascii=False), encoding="utf-8",
    )
    np.save(index / "embeddings.npy", np.ones((1, 3), dtype=np.float32))
    return index


def _memory_env(root, *, c2="true", **extra):
    """ツール + RAG + C2 が有効な環境を作る。seed の索引は 3 次元のダミーで、本文にセンチネルを入れる。"""
    index = _write_seed_index(root)
    os.environ.update({
        "L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": c2,
        "L2_C2_URL": "http://c2.invalid:8100", "L2_KB_PATH": str(index), **extra,
    })
    return index


def _fake_c2(routes):
    """本番の C2Retriever をそのまま通すために、httpx の送信だけを差し替える。"""
    import httpx

    real_client = httpx.Client
    return patch("httpx.Client", lambda timeout: real_client(transport=httpx.MockTransport(routes), timeout=timeout))


def _c2_ok(request):
    """本物の C2 と同じ形と k の上限で答える（coral_chronicle/api.py の Query(le=...)）。"""
    import httpx

    path = request.url.path
    k = int(request.url.params["k"])
    if k > (20 if path == "/retrieve" else 50):
        return httpx.Response(422, json={"detail": "k out of range"})
    results = [{"event_id": f"{path}-1", "type": "llm.final", "ts": "t", "stream_id": "s", "score": 0.9, "excerpt": BODY}]
    mode = "semantic" if path == "/retrieve" else "recent"
    return httpx.Response(200, json={"results": results, "timed_out": False, "elapsed_ms": 1, "search_mode": mode})


def _run_memory(routes=_c2_ok, *, embed=(1.0, 0.0, 0.0), stream_id=None):
    """check_memory を 1 回流して (行, report) を返す。"""
    report = preflight.Report()
    with _fake_c2(routes), patch("lab_lounge.retriever._embed", return_value=list(embed)), \
            preflight._captured_logs() as logs:
        preflight.check_memory(report, logs, stream_id)
    return _status(report, "memory"), report


class TestMemoryRow:
    """本番の入口で流した結果を、行の色と理由で確かめる。"""

    def test_healthy_path_is_pass_and_shows_only_ms_count_and_origins(self, isolated):
        _memory_env(isolated)
        check, report = _run_memory()
        assert check.status == "PASS"
        assert re.fullmatch(r"\d+ms docs=3 c2=1 c2-recent=1 seed=1", check.detail)
        assert report.exit_code() == preflight.EXIT_OK

    def test_seed_only_path_is_pass(self, isolated):
        _memory_env(isolated, c2="false")
        check, _ = _run_memory(lambda request: pytest.fail("C2 が無効なのに呼ばれました"))
        assert check.status == "PASS"
        assert re.fullmatch(r"\d+ms docs=1 seed=1", check.detail)

    @pytest.mark.parametrize("kind", ["no-such", "empty"])
    def test_missing_index_is_fail_though_embeddings_and_c2_are_green(self, isolated, kind):
        """LocalRetriever を最初に作るので、索引がないと C2 まで届かない。部品ごとの行では見えない。"""
        target = isolated / kind
        if kind == "empty":
            target.mkdir()
        _memory_env(isolated, L2_KB_PATH=str(target))
        reached = []

        def routes(request):
            reached.append(request.url.path)
            return _c2_ok(request)

        response = MagicMock()
        response.read.return_value = b'{"search_mode": "semantic"}'
        response.__enter__.return_value = response
        report = preflight.Report()
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0, 0.0]), \
                patch("urllib.request.urlopen", return_value=response):
            preflight.check_embeddings(report, preflight._LogCapture())
            preflight.check_c2(report)
        assert _status(report, "embeddings").status == "PASS"
        assert _status(report, "c2").status == "PASS"

        check, memory_report = _run_memory(routes)
        assert check.status == "FAIL"
        assert "chunks.json" in check.detail
        assert reached == []
        assert memory_report.exit_code() == preflight.EXIT_FAILED

    def test_over_budget_is_a_warning_with_the_reason_and_ms(self, isolated):
        """本番の C2Retriever は予算切れを空リストへ変える。1 回の遅さで配信を止めないので WARN に留める。"""
        import httpx

        _memory_env(isolated)

        def routes(request):
            if request.url.path == "/retrieve":
                raise httpx.ReadTimeout("slow", request=request)
            return _c2_ok(request)

        check, report = _run_memory(routes)
        assert check.status == "WARN"
        assert "タイムアウト" in check.detail
        assert re.match(r"\d+ms docs=2 c2-recent=1 seed=1 / ", check.detail)
        assert report.exit_code() == preflight.EXIT_OK

    def test_recent_endpoint_failure_is_a_warning_with_the_status(self, isolated):
        """c2 の行は /retrieve しか見ないので、/retrieve/recent だけの故障はここでしか見えない。"""
        import httpx

        _memory_env(isolated)

        def routes(request):
            if request.url.path == "/retrieve/recent":
                return httpx.Response(500, json={"detail": BODY})
            return _c2_ok(request)

        check, _ = _run_memory(routes)
        assert check.status == "WARN"
        assert "RecentC2Retriever" in check.detail
        assert "status=500" in check.detail
        assert not _shows_answer(check.detail)

    def test_top_k_over_the_c2_limit_is_a_warning(self, isolated):
        """CompositeRetriever は top_k*3 を送るので、L2_RAG_TOP_K=7 で /retrieve の上限 20 を超える。"""
        _memory_env(isolated, L2_RAG_TOP_K="7")
        check, _ = _run_memory()
        assert check.status == "WARN"
        assert "status=422" in check.detail

    def test_top_k_within_the_c2_limit_is_pass(self, isolated):
        _memory_env(isolated, L2_RAG_TOP_K="6")
        check, _ = _run_memory()
        assert check.status == "PASS"

    def test_stopped_readonly_c2_is_only_a_warning(self, isolated):
        """副 C2 は任意。止まっていても主の記憶は返るので、合否に数えない。"""
        import httpx

        _memory_env(isolated, L2_C2_URL_READONLY="http://readonly.invalid:8100")

        def routes(request):
            if request.url.host == "readonly.invalid":
                raise httpx.ConnectError("refused", request=request)
            return _c2_ok(request)

        check, report = _run_memory(routes)
        assert check.status == "WARN"
        assert "ConnectError" in check.detail
        assert " c2=" in check.detail  # 主の記憶は返っている
        assert report.exit_code() == preflight.EXIT_OK

    def test_same_reason_from_primary_and_readonly_is_shown_once(self, isolated):
        """主と副が同じ文言で落ちても 1 回だけ出す。主が落ちたかどうかは内訳の c2= で読める。"""
        import httpx

        _memory_env(isolated, L2_C2_URL_READONLY="http://readonly.invalid:8100", L2_USE_C2_RECENT="false")

        def routes(request):
            raise httpx.ReadTimeout("slow", request=request)

        check, _ = _run_memory(routes)
        assert check.status == "WARN"
        assert check.detail.count("タイムアウト") == 1
        assert re.match(r"\d+ms docs=1 seed=1 / ", check.detail)

    def test_warnings_from_earlier_checks_are_not_counted(self, isolated):
        """ログの捕捉は前の点検と共有する。前の行の警告で、記憶の行を赤くしない。"""
        _memory_env(isolated)
        report = preflight.Report()
        with _fake_c2(_c2_ok), patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0, 0.0]), \
                preflight._captured_logs() as logs:
            logging.getLogger("lab_lounge.router").warning("CompositeRetriever: %s が例外発生 → 空リスト扱い: %s", "X", "前の点検")
            preflight.check_memory(report, logs)
        assert _status(report, "memory").status == "PASS"

    def test_error_text_without_a_log_is_still_fail(self, isolated):
        """ログの文言に頼らず戻り値で判定する。エラー文には例外の本文が入るので、定型の理由だけを出す。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        answer = f"記憶検索でエラーが発生しました: {BODY}"
        with patch("lab_lounge.mcp_servers.retrieve_memory._retrieve_memory_impl", return_value=answer), \
                preflight._captured_logs() as logs:
            preflight.check_memory(report, logs)
        check = _status(report, "memory")
        assert check.status == "FAIL"
        assert "エラー文が返りました" in check.detail
        assert not _shows_answer(check.detail)

    def test_answer_of_an_unexpected_shape_is_a_fail_row(self, isolated):
        """戻り値の形が本番の変更で変わっても、点検全体を止めない。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        with patch("lab_lounge.mcp_servers.retrieve_memory._retrieve_memory_impl", return_value=None), \
                preflight._captured_logs() as logs:
            preflight.check_memory(report, logs)
        check = _status(report, "memory")
        assert check.status == "FAIL"
        assert "AttributeError" in check.detail

    @pytest.mark.parametrize("c2", ["true", "false"])
    def test_seed_search_failure_is_fail_with_or_without_c2(self, isolated, c2):
        """C2 が有効だと CompositeRetriever が例外を飲み込むが、同じ壊れ方なので判定を変えない。"""
        _memory_env(isolated, c2=c2)
        check, report = _run_memory(embed=(1.0, 0.0))  # 索引の次元と合わない
        assert check.status == "FAIL"
        assert "ValueError" in check.detail
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_zero_documents_is_a_warning(self, isolated):
        """検索は通っても 1 件も返らないなら、記憶は効いていない。緑にしない。"""
        import httpx
        import numpy as np

        index = _memory_env(isolated)
        (index / "chunks.json").write_text("[]", encoding="utf-8")
        np.save(index / "embeddings.npy", np.zeros((0, 3), dtype=np.float32))
        empty = lambda request: httpx.Response(200, json={"results": [], "search_mode": "like"})  # noqa: E731
        check, _ = _run_memory(empty)
        assert check.status == "WARN"
        assert "docs=0" in check.detail

    def test_setting_that_raises_outside_the_try_is_a_fail_row(self, isolated):
        """L2_RAG_TOP_K は本番の try の外で int() される。点検全体を止めない。"""
        _memory_env(isolated, L2_RAG_TOP_K="abc")
        check, _ = _run_memory()
        assert check.status == "FAIL"
        assert "ValueError" in check.detail

    @pytest.mark.parametrize("tools,rag", [(None, "true"), ("false", "true"), ("true", None), ("true", "false")])
    def test_skipped_without_billing_when_the_tool_cannot_run(self, isolated, capture, tools, rag):
        for name, value in (("L2_ENABLE_TOOLS", tools), ("L2_ENABLE_RAG", rag)):
            if value is not None:
                os.environ[name] = value
        report = preflight.Report()
        with patch("lab_lounge.mcp_servers.retrieve_memory._retrieve_memory_impl") as impl, \
                patch("lab_lounge.retriever._embed") as embed:
            preflight.check_memory(report, capture)
        assert _status(report, "memory").status == "SKIP"
        impl.assert_not_called()
        embed.assert_not_called()

    def test_stalled_embedding_is_cut_by_the_deadline(self, isolated, monkeypatch):
        """埋め込みは SDK の既定で最長 30 分待つ。相手が止まっても残りの行と要約を出す。"""
        _memory_env(isolated, c2="false")
        monkeypatch.setattr(preflight, "_EMBED_DEADLINE_SEC", 0.2)
        release = threading.Event()
        report = preflight.Report()
        try:
            with patch("lab_lounge.retriever._embed", side_effect=lambda _text: release.wait(10) or [1.0, 0.0, 0.0]), \
                    preflight._captured_logs() as logs:
                preflight.check_memory(report, logs)
        finally:
            release.set()
        check = _status(report, "memory")
        assert check.status == "FAIL"
        assert "TimeoutError" in check.detail


class TestMemoryParity:
    """本番と同じ予算、k、接続先、文脈で流していること。ずれると予算切れを緑で見逃す。"""

    def test_production_budget_k_and_both_endpoints_are_used(self, isolated):
        _memory_env(isolated, L2_C2_RETRIEVER_BUDGET_MS="123")
        seen = []

        def routes(request):
            seen.append((request.url.path, int(request.url.params["k"]), request.extensions["timeout"]["read"],
                         "exclude_event_ids" in request.url.params))
            return _c2_ok(request)

        check, _ = _run_memory(routes)
        assert check.status == "PASS"
        # top_k=5 を CompositeRetriever が max(5*3, 10) へ広げる。c2 の行の k=1 と違うところ。
        assert seen == [("/retrieve", 15, pytest.approx(0.123), True), ("/retrieve/recent", 15, pytest.approx(0.123), True)]

    def test_stream_id_reaches_recent_in_session_scope_and_the_context_is_cleared(self, isolated):
        """session の scope は stream_id がないと /retrieve/recent を呼ばない。後の点検へ文脈を持ち越さない。"""
        from lab_lounge.mcp_servers import retrieve_memory as rm

        _memory_env(isolated, L2_C2_RECENT_SCOPE="session")
        seen = {}

        def routes(request):
            seen[request.url.path] = request.url.params.get("stream_id")
            return _c2_ok(request)

        check, _ = _run_memory(routes, stream_id="preflight-x")
        assert check.status == "PASS"
        assert seen["/retrieve/recent"] == "preflight-x"
        assert rm._stream_id_var.get() is None
        assert rm._exclude_event_ids_var.get() == []

    def test_paid_runs_it_right_after_c2_with_the_stream_id(self, isolated, capture):
        """安い経路から順に流す約束。stream_id が届かないと session の scope を確かめられない。"""
        names = (
            "check_router", "check_filler", "check_tavily", "check_embeddings", "check_c2", "check_memory",
            "check_tts_engine", "check_agent",
        )
        manager = MagicMock()
        patches = [patch.object(preflight, name, getattr(manager, name)) for name in names]
        for p in patches:
            p.start()
        try:
            preflight.run_paid(preflight.Report(), capture, [get_character("mimi")], with_pipeline=False)
        finally:
            for p in patches:
                p.stop()
        called = [call[0] for call in manager.mock_calls]
        assert called.index("check_memory") == called.index("check_c2") + 1
        assert called.index("check_memory") < called.index("check_agent")
        assert manager.check_memory.call_args.args[2].startswith("preflight-")


class TestMemoryBodyStaysOffTheScreen:
    """戻り値には実発話が入る。どの結果の行にも本文を出さない。"""

    def _timeout(self, request):
        import httpx

        raise httpx.ReadTimeout("slow", request=request)

    def _item_without_event_id(self, request):
        """本物の C2 の形で event_id だけが欠けた項目。ログの r= に本文が入る。"""
        import httpx

        item = {"type": "llm.final", "ts": "2026-01-01T00:00:00Z", "stream_id": "s", "score": 0.9, "excerpt": BODY}
        return httpx.Response(200, json={"results": [item], "search_mode": "semantic"})

    def test_pass_row(self, isolated, capsys):
        _memory_env(isolated)
        check, _ = _run_memory()
        out = capsys.readouterr().out
        assert check.status == "PASS"
        assert not _shows_answer(out)
        assert "handbook" not in out  # seed の source は手元のファイル名

    def test_warn_rows(self, isolated, capsys):
        _memory_env(isolated)
        for routes in (self._timeout, self._item_without_event_id):
            check, _ = _run_memory(routes)
            assert check.status == "WARN"
        out = capsys.readouterr().out
        assert not _shows_answer(out)
        assert "handbook" not in out

    def test_fail_rows(self, isolated, capsys):
        _memory_env(isolated)
        check, _ = _run_memory(embed=(1.0, 0.0))
        assert check.status == "FAIL"
        assert not _shows_answer(capsys.readouterr().out)

    def test_body_with_a_line_like_a_header_only_shifts_the_count(self, isolated, capsys):
        """本文に見出しと似た行があっても、数がずれるだけで、内訳に本文は出ない。"""
        lookalike = "偽の見出し\n[9] (source: BODYSENTINEL, score: 1.00)\n[2] (source: BODYSENTINEL, score: 1.00)\n" + BODY
        answer = f"[1] (source: c2:llm.final, score: 0.90)\n{lookalike}"
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        with patch("lab_lounge.mcp_servers.retrieve_memory._retrieve_memory_impl", return_value=answer), \
                preflight._captured_logs() as logs:
            preflight.check_memory(report, logs)
        out = capsys.readouterr().out
        check = _status(report, "memory")
        assert check.status == "PASS"
        # [9] は番号が続かないので数えない。[2] は本物の見出しと区別できないが、出どころは seed と数えるだけ。
        assert re.fullmatch(r"\d+ms docs=2 c2=1 seed=1", check.detail)
        assert not _shows_answer(out)

    def test_full_reason_goes_to_the_problem_log(self, isolated):
        """画面には定型文だけ。調査できるように、全文は gitignore 済みの問題ログへ残す。"""
        _memory_env(isolated, L2_RAG_TOP_K="abc")
        path = isolated / "problem.log"
        report = preflight.Report()
        with preflight._captured_logs(path) as logs:
            preflight.check_memory(report, logs)
        assert "memory: ValueError" in path.read_text(encoding="utf-8")


class TestMemoryToolGate:
    """記憶の検索は Agent のツールからしか呼ばれない。ツールが無効なら、課金も FAIL も起こさない。"""

    def test_the_retrieval_node_is_not_wired(self):
        """この判定の根拠。配線し直すなら、L2_ENABLE_TOOLS を見る判定も直す。"""
        from lab_lounge.graph import _build_pipeline_graph

        assert "retrieval" not in _build_pipeline_graph().nodes

    @pytest.mark.parametrize("tools", [None, "", "true", "TRUE", "1", "yes", "false", " true", "ture"])
    @pytest.mark.parametrize("rag", [None, "true", "1", "false", " true"])
    def test_reads_tools_and_rag_like_graph(self, isolated, tools, rag):
        from lab_lounge.graph import _is_rag_enabled, _is_tools_enabled

        for name, value in (("L2_ENABLE_TOOLS", tools), ("L2_ENABLE_RAG", rag)):
            if value is not None:
                os.environ[name] = value
        assert preflight._memory_tool_enabled() is (_is_tools_enabled() and _is_rag_enabled())

    @pytest.mark.parametrize("tools,rag,c2,expected", [
        ("true", "true", "true", True),
        ("false", "true", "true", False),  # ツールが無効なら Agent を通らず、retrieve_memory が動かない
        ("true", "false", "true", False),
        ("true", "true", "false", False),
        ("true", "true", "1", False),  # graph.py は "true" だけを真と読む
    ])
    def test_c2_counts_as_used_only_when_retrieve_memory_can_run(self, isolated, tools, rag, c2, expected):
        os.environ.update({"L2_ENABLE_TOOLS": tools, "L2_ENABLE_RAG": rag, "L2_USE_C2_RETRIEVER": c2})
        assert preflight._c2_retriever_enabled() is expected

    def test_tools_off_needs_no_c2_and_no_embedding(self, isolated, capture):
        """.env.dev.example の既定の形（RAG と C2 だけ真）。本番は C2 も埋め込みも呼ばない。"""
        os.environ.update({"L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true"})
        with patch.object(preflight, "_http_ok", return_value=False) as ok:
            assert preflight.missing_services([], with_pipeline=False) == []
        ok.assert_not_called()
        report = preflight.Report()
        with patch("lab_lounge.retriever._embed") as embed, patch("urllib.request.urlopen") as urlopen:
            preflight.check_embeddings(report, capture)
            preflight.check_c2(report)
        embed.assert_not_called()
        urlopen.assert_not_called()
        assert _status(report, "embeddings").status == "SKIP"
        assert _status(report, "c2").status == "SKIP"

    def test_paid_is_not_stopped_by_a_c2_production_never_calls(self, isolated):
        _write_real_env(isolated, L2_ENABLE_TOOLS="false", L2_ENABLE_RAG="true", L2_USE_C2_RETRIEVER="true")
        with patch.object(preflight, "run_paid") as paid, patch.object(preflight, "_http_ok", return_value=False):
            code = preflight.main(["--paid", "--profile", "dev", "--characters", "octamaid", "--no-pipeline"], repo_root=isolated)
        assert code == preflight.EXIT_OK
        paid.assert_called_once()

    def test_rag_without_tools_is_warned(self, isolated):
        os.environ["L2_ENABLE_RAG"] = "true"
        report = preflight.Report()
        preflight._check_flags(report)
        warned = [c for c in report.checks if c.name == "flag/L2_ENABLE_RAG" and c.status == "WARN"]
        assert len(warned) == 1
        assert "RAG は効きません" in warned[0].detail

    @pytest.mark.parametrize("tools,rag", [("true", "true"), (None, None), ("true", None), (None, "false")])
    def test_no_warning_when_rag_is_not_misleading(self, isolated, tools, rag):
        for name, value in (("L2_ENABLE_TOOLS", tools), ("L2_ENABLE_RAG", rag)):
            if value is not None:
                os.environ[name] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert not [c for c in report.checks if c.name == "flag/L2_ENABLE_RAG" and c.status == "WARN"]


class TestEmbeddingKeyIsChecked:
    """埋め込みは OpenAI 固定。--characters で OpenAI のキャラを外しても、課金の前に止める。"""

    def _sakura_only_providers(self):
        with patch("lab_lounge.filler.is_filler_enabled", return_value=False):
            return preflight._check_models(preflight.Report(), [get_character("sakura")])

    def test_openai_is_required_when_the_tool_can_run(self, isolated):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        assert "openai" in self._sakura_only_providers()

    @pytest.mark.parametrize("tools,rag", [("true", None), (None, "true")])
    def test_openai_is_not_required_when_the_tool_cannot_run(self, isolated, tools, rag):
        for name, value in (("L2_ENABLE_TOOLS", tools), ("L2_ENABLE_RAG", rag)):
            if value is not None:
                os.environ[name] = value
        assert "openai" not in self._sakura_only_providers()

    def test_missing_key_stops_before_billing(self, isolated, voices):
        index = _memory_env(isolated)
        for name in ("L2_ENABLE_TOOLS", "L2_ENABLE_RAG", "L2_USE_C2_RETRIEVER", "L2_C2_URL", "L2_KB_PATH"):
            del os.environ[name]
        _write_real_env(
            isolated, OPENAI_API_KEY="", L2_ENABLE_RAG="true", L2_KB_PATH=str(index), L2_IRODORI_VOICES_JSON=str(voices),
        )
        with patch.object(preflight, "run_paid") as paid:
            code = preflight.main(["--paid", "--profile", "dev", "--characters", "sakura", "--no-pipeline"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        paid.assert_not_called()


class TestKbIndexRow:
    """--static で索引の不在を見つける。課金の前に、C2 の分も含めて記憶が全滅することが分かる。"""

    @pytest.mark.parametrize("real_llm,status", [("true", "FAIL"), ("false", "WARN"), (None, "WARN")])
    def test_missing_index(self, isolated, real_llm, status):
        """実 LLM でなければ Agent を通らないので、ダミー起動は止めない。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_KB_PATH": str(isolated / "no-such")})
        if real_llm is not None:
            os.environ["L2_USE_REAL_LLM"] = real_llm
        report = preflight.Report()
        preflight._check_kb_index(report)
        check = _status(report, "rag/index")
        assert check.status == status
        assert "FileNotFoundError" in check.detail
        assert str(isolated) not in check.detail

    def test_default_path_is_read_from_the_working_directory(self, isolated):
        """本番と同じく作業ディレクトリ相対で読む。main は repo_root へ chdir する。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        preflight._check_kb_index(report)
        assert _status(report, "rag/index").status == "WARN"
        _memory_env(isolated)
        del os.environ["L2_KB_PATH"]
        (isolated / "data").mkdir()
        (isolated / "index").rename(isolated / "data" / "index")
        report = preflight.Report()
        preflight._check_kb_index(report)
        assert report.checks == []

    def test_readable_index_adds_no_row(self, isolated):
        _memory_env(isolated)
        report = preflight.Report()
        preflight._check_kb_index(report)
        assert report.checks == []

    @pytest.mark.parametrize("tools,rag", [("true", None), (None, "true"), ("false", "false")])
    def test_not_read_when_the_tool_cannot_run(self, isolated, tools, rag):
        """本番が読まないもので --static を止めない。"""
        for name, value in (("L2_ENABLE_TOOLS", tools), ("L2_ENABLE_RAG", rag)):
            if value is not None:
                os.environ[name] = value
        os.environ.update({"L2_USE_REAL_LLM": "true", "L2_KB_PATH": str(isolated / "no-such")})
        report = preflight.Report()
        preflight._check_kb_index(report)
        assert report.checks == []

    def test_static_fails_without_billing(self, isolated, voices, monkeypatch):
        _write_real_env(
            isolated, L2_ENABLE_RAG="true", L2_KB_PATH=str(isolated / "no-such"), L2_IRODORI_VOICES_JSON=str(voices),
        )
        patches = _guard_entry_points()
        mocks = [p.start() for p in patches]
        connect = MagicMock(side_effect=OSError("blocked"))
        monkeypatch.setattr(socket.socket, "connect", connect)
        try:
            code = preflight.main(["--static", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated)
        finally:
            for p in patches:
                p.stop()
        assert code == preflight.EXIT_FAILED
        for mock in mocks:
            mock.assert_not_called()
        connect.assert_not_called()

    def test_paid_stops_before_billing(self, isolated):
        _write_real_env(isolated, L2_ENABLE_RAG="true", L2_KB_PATH=str(isolated / "no-such"))
        with patch.object(preflight, "run_paid") as paid, patch.object(preflight, "missing_services") as missing:
            code = preflight.main(["--paid", "--profile", "dev", "--characters", "octamaid", "--no-pipeline"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        paid.assert_not_called()
        missing.assert_not_called()

    def test_default_path_matches_production(self, isolated):
        """既定のパスは点検と本番に別々に書いてある。本番だけ変えると、索引があるのに赤くなる。"""
        from lab_lounge.pipeline import _get_rag_mode

        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        with patch("lab_lounge.retriever.LocalRetriever") as local:
            preflight._check_kb_index(preflight.Report())
        assert local.call_args.args[0] == _get_rag_mode()[2]


# ─── パイプラインと協働の点検で、判定の色に出ない穴 ───────────────────
#
# ここで固定するのは、外れても PASS / FAIL の色が変わらないので気づきにくい挙動:
# 問題ログ、トークンの集計、想定外のイベントで表が途切れないこと、表に出す列、--no-pipeline の省略。


def _run_pipeline_check(capture, *, events=None, side_effect=None):
    """check_pipeline を mimi で 1 回流し、(行, report, run_pipeline の偽物) を返す。"""
    report = preflight.Report()
    with patch("lab_lounge.pipeline.run_pipeline", return_value=_pipeline_result(events=events), side_effect=side_effect) as run:
        preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
    return _status(report, "pipeline/mimi"), report, run


def _run_collab_check(capture, *, tools=("ask_character_tool",), invoke_error=None):
    """check_collab を sakura -> chisame で 1 往復流し、(行, report, Agent へ渡した本文) を返す。

    Agent は 2 段で、1 段あたり in=10 / out=5 のトークンを使う（_agent_result の値）。
    """
    from lab_lounge.mcp_servers import ask_character as ac

    seen = {}

    def fake_invoke(*args, **kwargs):
        seen["content"] = args[1]["messages"][0]["content"]
        if invoke_error:
            raise invoke_error
        ac._on_tts_chunk_var.get()("file:///a.wav", "こんにちは", True, "chisame")
        for message in _COLLAB_OK_LOGS:
            capture.emit(_log_record("lab_lounge.x", logging.INFO, message))
        return _agent_result(SimpleNamespace(response="まとめ"), tool_calls=[{"name": name} for name in tools], steps=2)

    report = preflight.Report()
    with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
            patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke), \
            patch("lab_lounge.mcp_servers.ask_character._ask_character_impl", return_value="応答"):
        preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "preflight-x", None)
    return _status(report, "collab/sakura->chisame"), report, seen


def _run_paid_with_fakes(capture, *, characters=None, **kwargs):
    """run_paid の配下の点検をすべて偽物にして流し、(偽物の辞書, report) を返す。課金も接続も起きない。

    Args:
        capture:    ログの捕捉先
        characters: 点検するキャラ。省略すると全キャラ
        kwargs:     run_paid へそのまま渡す引数
    """
    names = (
        "check_router", "check_filler", "check_tavily", "check_embeddings", "check_c2", "check_memory",
        "check_tts_engine", "check_agent", "check_pipeline", "check_collab",
    )
    patches = {name: patch.object(preflight, name) for name in names}
    mocks = {name: p.start() for name, p in patches.items()}
    report = preflight.Report()
    try:
        preflight.run_paid(report, capture, characters or get_all_characters(), **kwargs)
    finally:
        for p in patches.values():
            p.stop()
    return mocks, report


class TestPipelineDetails:
    def test_exception_text_goes_to_the_problem_log(self, isolated):
        """画面には型と status しか出さない。全文がログに残らないと、FAIL の原因を後から調べられない。"""
        path = isolated / "problem.log"
        check, _, _ = _run_pipeline_check(preflight._LogCapture(path), side_effect=_FakeApiError(BODY))
        assert check.status == "FAIL"
        assert "BODYSENTINEL" not in check.detail
        assert BODY in path.read_text(encoding="utf-8")

    def test_tokens_come_only_from_llm_final(self, isolated, capture):
        """要約の tokens は課金の目安。同じ名前の値を持つ別のイベントまで足すと、実際より多く見える。"""
        events = [
            {"type": "utterance.final", "payload": {"input_tokens": 100, "output_tokens": 100}},
            {"type": "llm.final", "payload": {"input_tokens": 7, "output_tokens": 3}},
            *_pipeline_result().events,
        ]
        _, report, _ = _run_pipeline_check(capture, events=events)
        assert (report.input_tokens, report.output_tokens) == (7, 3)

    @pytest.mark.parametrize("event", [{"type": "llm.final"}, {"type": "llm.final", "payload": None}])
    def test_llm_final_without_payload_does_not_stop_the_run(self, isolated, capture, event):
        """トークンを数えるためだけの読み取りで例外になると、残りのキャラの行と要約が出なくなる。"""
        check, report, _ = _run_pipeline_check(capture, events=[event, *_pipeline_result().events])
        assert check.status == "PASS"
        assert (report.input_tokens, report.output_tokens) == (0, 0)

    @pytest.mark.parametrize("event", [{"type": "tts.done"}, {"type": "tts.done", "payload": None}])
    def test_tts_done_without_payload_is_a_fail_row(self, isolated, capture, event):
        """中身の読めない tts.done は本文なしの FAIL にする。例外で止まると、残りの行と要約が出なくなる。"""
        check, _, _ = _run_pipeline_check(capture, events=[event])
        assert check.status == "FAIL"

    def test_probe_calls_the_character_by_its_wake_word(self, isolated, capture):
        """本番の発話は呼びかけの形で届くので、点検も同じ呼び名で流す。"""
        _, _, run = _run_pipeline_check(capture)
        assert get_character("mimi").wake_word in run.call_args.args[0]


def _run_pipeline_with_logs(capture, records, *, events=None, required=True):
    """run_pipeline の中で records のログが出た形で check_pipeline を流し、行を返す。"""
    def fake_pipeline(*_args, **_kwargs):
        for record in records:
            capture.emit(record)
        return _pipeline_result(events=events)

    report = preflight.Report()
    with patch("lab_lounge.pipeline.run_pipeline", fake_pipeline):
        preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=required)
    return _status(report, "pipeline/mimi")


def _tts_done(**payload):
    """本番の Agent の形をした本文を持つ tts.done を作る。"""
    text = '{"response": "こんにちは", "speed": 100, "pose": "neutral"}'
    return {"type": "tts.done", "payload": {"text": text, **payload}}


class TestSpokenProblems:
    """読み上げる本文の判定。本番はこの文字列をそのまま TTS へ渡すので、pipeline、協働、単一ノードで同じ判定を使う。"""

    @pytest.mark.parametrize("text", ["None", " None\n"])
    def test_collab_null_response_is_empty(self, text):
        """{"response": null} は本番で str() され、協働でも「None」と読み上げる。"""
        problems = preflight._chunk_problems([("chisame", text)], "chisame")
        assert any("空" in p for p in problems)

    def test_the_production_parser_turns_null_into_the_word_none(self):
        """_chunk_problems へ届く文字列を、本番の抽出器で作って確かめる。"""
        from lab_lounge.tts import _parse_voicepeak_json

        spoken = _parse_voicepeak_json(json.dumps({"response": None, "speed": 100, "pose": "neutral"}))[0]
        problems = preflight._chunk_problems([("chisame", spoken)], "chisame")
        assert any("空" in p for p in problems)

    def test_a_sentence_containing_the_word_stays_clean(self):
        assert preflight._chunk_problems([("chisame", "None という名前の魚です")], "chisame") == []

    @pytest.mark.parametrize("text", [
        '{"response": "切れた',  # max_tokens で切れた JSON（sakura は finish が見えない）
        '{"say": "response がない"}',  # 本番は JSON のまま読み上げる
        '```json\n{"response": "切れた',
        "[]",
    ])
    def test_json_fragment_reaching_tts_is_a_fail_row(self, isolated, capture, text):
        """単一ノードの経路は構造化出力を通らないので、断片がそのまま読み上げに入る。"""
        events = [{"type": "tts.done", "payload": {"text": text, "duration_ms": 1000}}]
        check, _, _ = _run_pipeline_check(capture, events=events)
        assert check.status == "FAIL"
        assert "JSON の断片" in check.detail

    @pytest.mark.parametrize("text", [
        '{"response": "こんにちは", "speed": 100, "pose": "happy"}',
        '```json\n{"response": "こんにちは"}\n```',
        "こんにちは",  # プレーンな文は本番でも読める
        "[笑] こんにちは",  # ふつうの文を赤くしない
    ])
    def test_readable_text_stays_green(self, isolated, capture, text):
        events = [{"type": "tts.done", "payload": {"text": text, "duration_ms": 1000}}]
        check, _, _ = _run_pipeline_check(capture, events=events)
        assert check.status == "PASS"

    @pytest.mark.parametrize("text", ['{"response": "切れた', '{"say": "x"}', '{"response": null}'])
    def test_main_row_of_the_single_node_route_reads_the_spoken_text(self, isolated, capture, text):
        """pipeline の前に止まれば、同じ応答へ 2 回課金しなくて済む。"""
        os.environ["L2_ENABLE_TOOLS"] = "false"
        fake = SimpleNamespace(text=text, input_tokens=1, output_tokens=1, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            ok = preflight.check_agent(report, capture, get_character("sakura"), None, required=True)
        assert ok is False
        assert _status(report, "main/sakura").status == "FAIL"

    def test_main_row_of_the_single_node_route_stays_green_on_valid_json(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "false"
        fake = SimpleNamespace(text='{"response": "こんにちは", "speed": 100, "pose": "happy"}',
                               input_tokens=1, output_tokens=1, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            assert preflight.check_agent(report, capture, get_character("sakura"), None, required=True) is True


class TestSilentAudio:
    """0 フレームの音声でも本番は例外にしないので、tts.done の長さで見る。"""

    def test_zero_duration_is_a_fail_row(self, isolated, capture):
        """本番の合成は例外にならず声だけが消えるので、ここで見ないと配信で初めて気づく。"""
        check, _, _ = _run_pipeline_check(capture, events=[_tts_done(duration_ms=0)])
        assert check.status == "FAIL"
        assert "音声の長さが 0" in check.detail

    def test_real_duration_stays_green(self, isolated, capture):
        check, _, _ = _run_pipeline_check(capture, events=[_tts_done(duration_ms=1200)])
        assert check.status == "PASS"

    @pytest.mark.parametrize("payload", [{}, {"duration_ms": None}], ids=["absent", "null"])
    def test_missing_duration_is_not_a_failure(self, isolated, capture, payload):
        """本番の tts.done は必ず持つ値なので、形が変わっただけで配信を止めない。"""
        check, _, _ = _run_pipeline_check(capture, events=[_tts_done(**payload)])
        assert check.status == "PASS"

    def test_production_reads_a_zero_frame_wav_as_zero(self, isolated, tmp_path):
        """本番の合成は 0 フレームの WAV を例外にせず 0 を返す。この判定はその前提に乗っている。"""
        import wave

        from lab_lounge.tts import _generate_irodori_single_file

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"")
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = buffer.getvalue()
        with patch("urllib.request.urlopen", return_value=response):
            duration, _ = _generate_irodori_single_file(
                "x", caption="c", ref_wav=None, duration_scale=1.0, num_steps=1, t_schedule_mode="sway",
                cfg_scale_speaker=1.0, seed=None, filepath=tmp_path / "a.wav", url="http://127.0.0.1:1/x",
            )
        assert duration == 0


class TestPipelineLogCounts:
    """1 ターンの中で本番が握りつぶした失敗を、文言を挙げずにログのレベルで拾う。"""

    def test_clean_turn_stays_pass(self, isolated, capture):
        record = _log_record("lab_lounge.graph", logging.INFO, "Agent 実行完了")
        assert _run_pipeline_with_logs(capture, [record]).status == "PASS"

    def test_warning_makes_the_row_warn_with_a_count(self, isolated, capture):
        """記憶の検索や bubble の発行の失敗は声が出るので、ここで見ないと緑のまま機能が消える。"""
        records = [_log_record("lab_lounge.mcp_servers.retrieve_memory", logging.WARNING, "記憶検索失敗: x") for _ in range(2)]
        check = _run_pipeline_with_logs(capture, records)
        assert check.status == "WARN"
        assert "警告のログ 2 件" in check.detail

    def test_warning_does_not_fail_the_run(self, isolated, capture):
        """1 回の揺れで配信を止めない。"""
        report = preflight.Report()

        def fake_pipeline(*_args, **_kwargs):
            capture.emit(_log_record("lab_lounge.retriever", logging.WARNING, "C2Retriever タイムアウト"))
            return _pipeline_result()

        with patch("lab_lounge.pipeline.run_pipeline", fake_pipeline):
            preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
        assert report.exit_code() == preflight.EXIT_OK

    @pytest.mark.parametrize("level", [logging.ERROR, logging.CRITICAL], ids=["error", "critical"])
    def test_error_makes_the_row_fail_with_a_count(self, isolated, capture, level):
        """publish の失敗が例外でなく ERROR のログだけになっても、緑にしない。文言は挙げていない。"""
        check = _run_pipeline_with_logs(capture, [_log_record("lab_lounge.bus", level, "知らない文言")])
        assert check.status == "FAIL"
        assert "ERROR のログ 1 件" in check.detail

    def test_error_beats_warning(self, isolated, capture):
        records = [
            _log_record("lab_lounge.bus", logging.WARNING, "w"),
            _log_record("lab_lounge.bus", logging.ERROR, "e"),
        ]
        assert _run_pipeline_with_logs(capture, records).status == "FAIL"

    def test_known_failure_keeps_its_reason_next_to_the_count(self, isolated, capture):
        """理由の文言は残す。件数だけだと、調べる場所が分からない。"""
        record = _log_record("lab_lounge.graph", logging.ERROR, "Agent 実行失敗 [character=mimi]: x")
        check = _run_pipeline_with_logs(capture, [record])
        assert "Agent が失敗" in check.detail
        assert "ERROR のログ 1 件" in check.detail

    def test_log_body_stays_in_the_problem_log(self, isolated):
        """ログの引数には発話が入る。行には件数だけを出し、全文は問題ログに残す。"""
        path = isolated / "problem.log"
        capture = preflight._LogCapture(path)
        records = [
            _log_record("lab_lounge.bus", logging.ERROR, "e: %s", (BODY,)),
            _log_record("lab_lounge.bus", logging.WARNING, "w: %s", (BODY,)),
        ]
        check = _run_pipeline_with_logs(capture, records)
        assert "BODYS" not in check.detail
        assert path.read_text(encoding="utf-8").count("BODYS") == 2

    def test_warn_row_keeps_the_required_flag(self, isolated, capture):
        record = _log_record("lab_lounge.bus", logging.WARNING, "w")
        check = _run_pipeline_with_logs(capture, [record], required=False)
        assert (check.status, check.required) == ("WARN", False)

    def test_logs_before_the_turn_are_not_counted(self, isolated, capture):
        """前の行の警告で、このキャラの行を濁らせない。"""
        capture.emit(_log_record("lab_lounge.bus", logging.ERROR, "前の行の失敗"))
        assert _run_pipeline_with_logs(capture, []).status == "PASS"

    def test_other_libraries_are_not_counted(self, isolated, capture):
        """SDK の警告は本番の失敗ではないので、毎回 WARN になる行を作らない。"""
        records = [
            _log_record("httpx", logging.ERROR, "e"),
            _log_record("langchain_core", logging.WARNING, "w"),
            # 点検中は warnings もログへ流すので、pydantic などの警告はここへ届く。
            _log_record("py.warnings", logging.WARNING, "w"),
        ]
        assert _run_pipeline_with_logs(capture, records).status == "PASS"

    def test_healthy_turn_through_the_real_graph_stays_pass(self, isolated, capture):
        """正常な 1 ターンで本番がログを出すようになると、全キャラの行が毎回 WARN になって読まれなくなる。"""
        # 手元のファイルに左右されないよう、人格と吹き出しの文言は固定する。publish も Redis へ出さない
        # (本流の 3 件は bus.publish、吹き出しは pipeline.publish を通る)。
        with patch("lab_lounge.pipeline.publish"), patch("lab_lounge.bus.publish"), \
                patch("lab_lounge.pipeline._BUBBLE_MESSAGES", {}), \
                patch("lab_lounge.characters.load_system_prompt", return_value="x"):
            report = preflight.Report()
            preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
        assert _status(report, "pipeline/mimi").status == "PASS"
        assert capture.problem_counts(0) == (0, 0)

    def test_counts(self, capture):
        for level in (logging.INFO, logging.WARNING, logging.WARNING, logging.ERROR):
            capture.emit(_log_record("lab_lounge.x", level, "m"))
        assert capture.problem_counts(0) == (1, 2)
        assert capture.problem_counts(capture.mark()) == (0, 0)


class TestCollabDetails:
    def test_tokens_of_the_caller_agent_are_counted(self, isolated, capture):
        """協働の呼び出し側は本応答と同じ課金なので、要約の tokens から漏れると少なく見える。"""
        _, report, _ = _run_collab_check(capture)
        assert (report.input_tokens, report.output_tokens) == (20, 10)

    def test_exception_text_goes_to_the_problem_log(self, isolated):
        """協働の失敗も画面は要約だけ。全文がログに残らないと、FAIL の原因を後から調べられない。"""
        path = isolated / "problem.log"
        check, _, _ = _run_collab_check(preflight._LogCapture(path), invoke_error=_FakeApiError(BODY))
        assert check.status == "FAIL"
        assert BODY in path.read_text(encoding="utf-8")

    def test_row_names_the_tool_the_model_called(self, isolated, capture):
        """モデルが ask_character を呼んだかは、この列でしか読めない（README の出力例にある列）。"""
        check, _, _ = _run_collab_check(capture)
        assert check.status == "PASS"
        assert "called=ask_character_tool" in check.detail

    def test_tools_column_has_a_value_when_no_tool_was_called(self, isolated, capture):
        """空のままだと次の列と続いて読めず、呼ばなかったのか列が壊れたのか分からない。記号は問わない。"""
        check, _, _ = _run_collab_check(capture, tools=())
        assert check.status == "WARN"
        assert re.search(r"called=\S", check.detail)

    def test_probe_calls_the_caller_by_its_wake_word(self, isolated, capture):
        """本番の発話は呼びかけの形で届くので、点検も同じ呼び名で流す。"""
        _, _, seen = _run_collab_check(capture)
        assert get_character("sakura").wake_word in seen["content"]


class TestCollabOverrideRow:
    """L2_ASK_CHARACTER_MODEL_OVERRIDE は全員の協働先を置き換えるのに、他の行には現れない。"""

    def test_override_is_a_warning_row(self, isolated):
        os.environ.update(L2_ENABLE_TOOLS="true", L2_ASK_CHARACTER_MODEL_OVERRIDE="gpt-unknown", L2_LLM_PROVIDER="anthropic")
        report = preflight.Report()
        preflight._check_models(report, [get_character("mimi")])
        check = _status(report, "model/collab-override")
        assert check.status == "WARN"
        assert "anthropic:gpt-unknown" in check.detail

    def test_provider_defaults_like_production(self, isolated):
        os.environ.update(L2_ENABLE_TOOLS="true", L2_ASK_CHARACTER_MODEL_OVERRIDE="gpt-unknown")
        report = preflight.Report()
        preflight._check_models(report, [get_character("sakura")])
        assert "openai:gpt-unknown" in _status(report, "model/collab-override").detail

    def test_key_of_the_override_provider_is_checked(self, isolated):
        """キーがないと協働だけが課金の後で落ちる。--paid の前に止める。"""
        os.environ.update(L2_USE_REAL_LLM="true", L2_ENABLE_TOOLS="true", L2_ASK_CHARACTER_MODEL_OVERRIDE="gpt-unknown")
        report = preflight.Report()
        providers = preflight._check_models(report, [get_character("sakura")])
        preflight._check_keys(report, providers, set())
        assert _status(report, "key/openai").status == "FAIL"

    def test_tools_off_neither_shows_nor_asks_for_a_key(self, isolated):
        """ツールが無効なら本番は読まない。本番が読まないもので止めない。"""
        os.environ.update(L2_USE_REAL_LLM="true", L2_ASK_CHARACTER_MODEL_OVERRIDE="gpt-unknown")
        report = preflight.Report()
        providers = preflight._check_models(report, [get_character("sakura")])
        assert "openai" not in providers
        assert not [check for check in report.checks if check.name == "model/collab-override"]

    @pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
    def test_no_row_without_override(self, isolated, value):
        """本番は空文字を「指定なし」と読む。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        if value is not None:
            os.environ["L2_ASK_CHARACTER_MODEL_OVERRIDE"] = value
        report = preflight.Report()
        preflight._check_models(report, [get_character("mimi")])
        assert not [check for check in report.checks if check.name == "model/collab-override"]

    @pytest.mark.parametrize("env,shown", [
        ({"L2_LLM_PROVIDER": "anthropic"}, ("anthropic", "gpt-unknown")),
        ({}, ("openai", "gpt-unknown")),
    ], ids=["provider-set", "provider-default"])
    def test_production_calls_what_the_row_shows(self, isolated, env, shown):
        """本番側で変数の名前や既定が変わると、この行が無言で嘘になる。実物の本体を通して確かめる。"""
        from lab_lounge.mcp_servers import ask_character as module

        os.environ.update(L2_ASK_CHARACTER_MODEL_OVERRIDE="gpt-unknown", **env)
        module.set_ask_character_context(caller_slug="mimi", common={"stream_id": "s", "session_id": "s", "trace_id": "t"})
        try:
            with patch.object(module, "_run_collaboration_agent", return_value="x") as collab, \
                    patch.object(module, "_generate_intro", return_value=""):
                module._ask_character_impl("chisame", "q")
        finally:
            module.reset_ask_character_context()
        kwargs = collab.call_args.kwargs
        assert (kwargs["provider"], kwargs["model"]) == shown


class TestRunPaidDetails:
    def test_no_pipeline_skips_collab_even_with_tools_on(self, isolated, capture):
        """--no-pipeline は Redis と TTS の前提を確かめない（missing_services）ので、協働を流すと課金だけして壊れる。

        ツールが無効だと別の分岐でも SKIP になるので、有効にして確かめる。
        """
        os.environ["L2_ENABLE_TOOLS"] = "true"
        mocks, report = _run_paid_with_fakes(capture, with_pipeline=False)
        mocks["check_collab"].assert_not_called()
        assert _status(report, "collab").status == "SKIP"

    def test_no_skip_row_when_a_collab_pair_runs(self, isolated, capture):
        """協働を流したのに SKIP の行が残ると、未確認の経路があるように読めて、要約の SKIP の数もずれる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        mocks, report = _run_paid_with_fakes(capture, with_pipeline=True)
        mocks["check_collab"].assert_called_once()
        assert not any(check.name == "collab" for check in report.checks)

    def test_stream_id_row_shows_the_id_the_checks_use(self, isolated, capture):
        """--paid は dev の C2 と HUD に点検の発話を残す。後から見分けるには、実際に使った ID が表に要る。"""
        mocks, report = _run_paid_with_fakes(capture, with_pipeline=True)
        row = _status(report, "stream_id")
        assert row.status == "INFO"
        assert row.detail == mocks["check_pipeline"].call_args_list[0].args[3]


class TestCollabAfterMainFailure:
    """呼び出し側の本応答が FAIL なら協働を流さない。pipeline と揃え、同じ Agent へ課金を重ねない。"""

    @staticmethod
    def _run(monkeypatch, capture, failed_slug):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        for name in ("check_router", "check_filler", "check_tavily", "check_embeddings", "check_c2", "check_memory", "check_pipeline"):
            monkeypatch.setattr(preflight, name, lambda *a, **k: None)
        monkeypatch.setattr(preflight, "check_tts_engine", lambda *a, **k: True)
        monkeypatch.setattr(preflight, "check_agent",
                            lambda _r, _c, character, _s, *, required: character.slug != failed_slug)
        collab = MagicMock()
        monkeypatch.setattr(preflight, "check_collab", collab)
        report = preflight.Report()
        preflight.run_paid(report, capture, [get_character("sakura"), get_character("chisame")], with_pipeline=True)
        return report, collab

    def test_collab_is_skipped_when_the_caller_main_failed(self, isolated, capture, monkeypatch):
        """同じ Agent を通すので、検証に落ち続けると上限の回数ぶん課金が増える。"""
        report, collab = self._run(monkeypatch, capture, "sakura")
        collab.assert_not_called()
        assert _status(report, "collab/sakura->chisame").status == "SKIP"

    def test_collab_still_runs_when_only_the_target_main_failed(self, isolated, capture, monkeypatch):
        """協働先は別の Agent（create_react_agent）を通るので、飛ばすと情報が減る。"""
        _, collab = self._run(monkeypatch, capture, "chisame")
        collab.assert_called_once()


class TestGateGaps:
    """流していない必須の経路がある実行を、全経路の合格と読み違えないこと。"""

    def test_partial_character_list_warns(self, isolated, capsys):
        """--characters mimi は残り 3 キャラが壊れていても exit 0 になる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        code, _ = _run_main(isolated, "--paid", "--profile", "dev", "--characters", "mimi")
        assert code == preflight.EXIT_OK
        row = _first_line_starting_with(capsys.readouterr().out, "[WARN ] characters")
        assert row is not None
        assert "chisame, sakura, octamaid" in row
        assert "配信前の合格には使えません" in row

    def test_no_required_character_at_all_names_all_four(self, isolated, capsys):
        """必須を 1 人も測らないと「必須がすべて PASS」は空の真になる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _run_main(isolated, "--paid", "--profile", "dev", "--characters", "ruka,aruka")
        row = _first_line_starting_with(capsys.readouterr().out, "[WARN ] characters")
        assert row is not None
        assert all(slug in row for slug in preflight.REQUIRED_CHARACTERS)

    @pytest.mark.parametrize("extra", [(), ("--characters", "mimi,chisame,sakura,octamaid")], ids=["default", "required-only"])
    def test_every_required_character_has_no_row(self, isolated, capsys, extra):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _run_main(isolated, "--paid", "--profile", "dev", *extra)
        assert "[WARN ] characters" not in capsys.readouterr().out

    def test_tools_off_warns(self, isolated, capsys):
        """単一ノードの本番は測れているが、構造化出力と協働は通っていない。"""
        code, _ = _run_main(isolated, "--paid", "--profile", "dev")
        assert code == preflight.EXIT_OK
        row = _first_line_starting_with(capsys.readouterr().out, "[WARN ] tools")
        assert row is not None
        assert "配信前の合格には使えません" in row

    def test_tools_on_has_no_row(self, isolated, capsys):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _run_main(isolated, "--paid", "--profile", "dev")
        assert "[WARN ] tools" not in capsys.readouterr().out

    def test_static_run_has_no_gap_rows(self, isolated, capsys):
        """--static は課金なしの設定の点検で、ゲートの前段。経路を流さないのは当たり前なので知らせない。"""
        _run_main(isolated, "--static", "--profile", "dev", "--characters", "mimi")
        out = capsys.readouterr().out
        assert "[WARN ] characters" not in out
        assert "[WARN ] tools" not in out

    def test_stop_before_billing_has_no_gap_rows(self, isolated, capsys):
        """止めた実行には合格の読み違いがないので、止めた理由の前に余計な行を足さない。"""
        _run_main(isolated, "--paid", "--profile", "dev", "--characters", "mimi", rows=[("key/openai", "FAIL")])
        assert "[WARN ] characters" not in capsys.readouterr().out

    def test_collab_none_row_says_it_cannot_pass_the_gate(self, isolated, capture):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _, report = _run_paid_with_fakes(capture, with_pipeline=True, collab="none")
        check = _status(report, "collab")
        assert check.status == "WARN"
        assert "配信前の合格には使えません" in check.detail


def _stall(release):
    """release が立つまで止まる偽物を作る。打ち切りが効いていれば、放す前に戻ってくる。"""
    def stalled(*_args, **_kwargs):
        release.wait(30)
        raise AssertionError("打ち切られていない")

    return stalled


class TestDeadlines:
    """timeout のない経路で相手が応答を止めても、残りの行と要約を出す。"""

    def test_stalled_tavily_is_one_fail_row(self, isolated, capture, monkeypatch):
        """langchain_tavily は timeout なしで通信する。"""
        pytest.importorskip("langchain_tavily")
        os.environ.update({"L2_ENABLE_TOOLS": "true", "TAVILY_API_KEY": "dummy"})
        monkeypatch.setattr(preflight, "_TAVILY_DEADLINE_SEC", 0.2)
        release = threading.Event()
        report = preflight.Report()
        try:
            with patch("requests.post", side_effect=_stall(release)):
                preflight.check_tavily(report, capture)
        finally:
            release.set()
        check = _status(report, "tavily")
        assert check.status == "FAIL"
        assert "TimeoutError" in check.detail

    def test_stalled_embeddings_is_one_fail_row(self, isolated, capture, monkeypatch):
        """_embed は SDK の既定（600 秒 x 3 回）で待つ。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        monkeypatch.setattr(preflight, "_EMBED_DEADLINE_SEC", 0.2)
        release = threading.Event()
        report = preflight.Report()
        try:
            with patch("lab_lounge.retriever._embed", side_effect=_stall(release)):
                preflight.check_embeddings(report, capture)
        finally:
            release.set()
        assert _status(report, "embeddings").status == "FAIL"
        assert "TimeoutError" in _status(report, "embeddings").detail

    def test_stalled_pipeline_is_one_fail_row(self, isolated, capture, monkeypatch):
        """VOICEVOX は timeout なしで通信する。"""
        monkeypatch.setattr(preflight, "_PIPELINE_DEADLINE_SEC", 0.2)
        release = threading.Event()
        report = preflight.Report()
        try:
            with patch("lab_lounge.pipeline.run_pipeline", side_effect=_stall(release)):
                preflight.check_pipeline(report, capture, get_character("mimi"), "preflight-x", None, required=True)
        finally:
            release.set()
        assert _status(report, "pipeline/mimi").status == "FAIL"
        assert "TimeoutError" in _status(report, "pipeline/mimi").detail

    def test_stalled_direct_collab_is_one_fail_row(self, isolated, capture, monkeypatch):
        """モデルがツールを呼ばないときの直接呼びにも打ち切りがある。文脈は _with_deadline が引き継ぐ。"""
        monkeypatch.setattr(preflight, "_PIPELINE_DEADLINE_SEC", 0.2)
        release = threading.Event()
        report = preflight.Report()
        try:
            with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                    patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=_agent_result(SimpleNamespace(response="x"))), \
                    patch("lab_lounge.mcp_servers.ask_character._ask_character_impl", side_effect=_stall(release)):
                preflight.check_collab(report, capture, get_character("sakura"), get_character("chisame"), "x", None)
        finally:
            release.set()
        check = _status(report, "collab/sakura->chisame")
        assert check.status == "FAIL"
        assert "TimeoutError" in check.detail

    def test_deadline_keeps_the_timeout_raised_by_the_call_itself(self):
        """本番の irodori の timeout を「壁時計を超過」と読み替えない（3.11 以降は同じ型）。"""
        def own_timeout():
            raise TimeoutError("timed out")

        with pytest.raises(TimeoutError, match="timed out"):
            preflight._with_deadline(own_timeout, 5)

    def test_deadline_returns_the_value_and_keeps_the_context(self):
        import contextvars

        var = contextvars.ContextVar("v", default="unset")
        var.set("set")
        assert preflight._with_deadline(var.get, 5) == "set"


# ─── main の表示と _cli の入口 ────────────────────────────────────


def _run_main(root, *argv, rows=(), missing=(), stream="aibyss:events-dev"):
    """main を流して (終了コード, run_paid のモック) を返す。

    表示を確かめるテストが実環境の点検結果に左右されないよう、run_static は rows の行だけを
    報告する形に差し替える。--paid で課金につながる run_paid と missing_services も必ず差し替える。

    Args:
        root:    repo_root として渡す一時ディレクトリ
        argv:    コマンドライン引数
        rows:    run_static の代わりに報告する行（Report.add の位置引数のタプル）
        missing: missing_services が返す、足りない前提サービスの名前
        stream:  REDIS_STREAM_KEY の値。既定は dev（消すと本番ストリームとして拒否される）
    """
    os.environ["REDIS_STREAM_KEY"] = stream

    def fake_static(report, *_args, **_kwargs):
        for row in rows:
            report.add(*row)

    with patch.object(preflight, "run_static", fake_static), \
            patch.object(preflight, "missing_services", return_value=list(missing)), \
            patch.object(preflight, "run_paid") as run:
        code = preflight.main(list(argv), repo_root=root)
    return code, run


def _first_line_starting_with(out, prefix):
    """出力から prefix で始まる最初の行を返す。無ければ None。"""
    return next((line for line in out.splitlines() if line.startswith(prefix)), None)


def _run_entry(argv=("preflight",), main=None, on_exit=None):
    """_cli を同じプロセスで流し、os._exit が受けた値の一覧を返す。

    本物の os._exit はテストごと終わらせるので差し替える（別プロセスだと 1 件 3 秒かかる）。
    _cli が書き換える logging.lastResort は元へ戻す。

    Args:
        argv:    sys.argv として渡す値
        main:    preflight.main の代わり。None なら本物を使う
        on_exit: os._exit の時点で呼ぶ関数。そこまでに何が書き出されたかを調べるために使う
    """
    exits = []

    def fake_exit(code):
        exits.append(code)
        if on_exit is not None:
            on_exit()

    with patch.object(sys, "argv", list(argv)), patch("os._exit", fake_exit), \
            patch.object(logging, "lastResort", logging.lastResort), \
            patch.object(preflight, "main", main or preflight.main):
        preflight._cli()
    return exits


class TestRunHeader:
    """冒頭の 2 行。貼られた出力を後から読む人は、ここで「何を測った結果か」を判断する。"""

    @pytest.mark.parametrize("mode,other", [("--static", "--paid"), ("--paid", "--static")])
    def test_header_names_the_mode_profile_and_stream(self, isolated, capsys, mode, other):
        """--static の緑を --paid の緑と読み違えると、API を確かめないまま配信に入ってしまう。"""
        _run_main(isolated, mode, "--profile", "dev")
        header = _first_line_starting_with(capsys.readouterr().out, "==")
        assert header is not None
        assert mode in header and other not in header
        assert "profile=dev" in header and "aibyss:events-dev" in header

    @pytest.mark.parametrize("profile,shown,hidden", [("dev", ".env.dev", None), ("prod", ".env", ".env.dev")])
    def test_env_line_names_the_files_it_read(self, isolated, capsys, profile, shown, hidden):
        """読んだファイルが見えないと、本番の前に dev の設定を測ったことに気づけない。"""
        (isolated / ".env").write_text("", encoding="utf-8")
        (isolated / ".env.dev").write_text("", encoding="utf-8")
        _run_main(isolated, "--static", "--profile", profile)
        line = _first_line_starting_with(capsys.readouterr().out, "env:")
        assert line is not None
        assert shown in line
        assert hidden is None or hidden not in line

    def test_env_line_is_not_blank_without_files(self, isolated, capsys):
        """1 つも読めなかった実行（ルートの取り違えなど）が、空欄のまま読み流されないこと。"""
        _run_main(isolated, "--static", "--profile", "dev")
        line = _first_line_starting_with(capsys.readouterr().out, "env:")
        assert line is not None
        shown = line.split(":", 1)[1].strip()
        assert shown and ".env" not in shown


class TestStopReasons:
    """止まったとき、理由と次の一手が画面に残ること。終了コードだけでは何を直すか分からない。"""

    def test_prod_refusal_names_the_way_out(self, isolated, capsys):
        """拒否の理由と --allow-prod が読めないと、利用者はなぜ 2 で終わったのかを調べ直すことになる。"""
        code, run = _run_main(isolated, "--paid", "--profile", "dev", stream=preflight.PROD_STREAM)
        assert code == preflight.EXIT_REFUSED
        run.assert_not_called()
        assert "--allow-prod" in capsys.readouterr().out

    def test_static_failure_stop_lists_what_failed(self, isolated, capsys):
        """課金の前に止めたときも、集計と FAIL の一覧を出す。流れた行を遡らずに直す先が分かるように。"""
        code, run = _run_main(isolated, "--paid", "--profile", "dev", rows=[("key/openai", "FAIL")])
        assert code == preflight.EXIT_FAILED
        run.assert_not_called()
        out = capsys.readouterr().out
        assert re.search(r"FAIL 1\b", out)
        assert re.search(r"^\s+FAIL.*key/openai", out, re.M)

    def test_static_failure_stop_says_nothing_was_billed(self, isolated, capsys):
        """--paid の実行が FAIL で終わったとき、課金が起きたのかどうかを判定の行のすぐ上で判断できること。"""
        _run_main(isolated, "--paid", "--profile", "dev", rows=[("key/openai", "FAIL")])
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[-1].startswith("判定:")
        assert "課金" in lines[-2]

    def test_missing_service_stop_shows_the_summary(self, isolated, capsys):
        """前提サービスで止めたときも、それまでの点検の集計を出す。WARN を読み落とさないように。"""
        code, run = _run_main(isolated, "--paid", "--profile", "dev", rows=[("flag/x", "WARN")], missing=["Redis"])
        assert code == preflight.EXIT_MISSING_SERVICE
        run.assert_not_called()
        assert re.search(r"WARN 1\b", capsys.readouterr().out)

    def test_stop_for_a_missing_service_points_at_the_problem_log(self, isolated, capsys):
        """課金の前に止まる経路でも、全文の場所を示す。別枠の行の理由はログにしか残らないことがある。"""
        def fake_static(_report, *_args, **_kwargs):
            logging.getLogger("lab_lounge.x").warning("全文")

        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "run_static", fake_static), \
                patch.object(preflight, "missing_services", return_value=["Redis"]), \
                patch.object(preflight, "run_paid") as run:
            code = preflight.main(["--paid", "--profile", "dev"], repo_root=isolated)
        assert code == preflight.EXIT_MISSING_SERVICE
        run.assert_not_called()
        assert "詳細: logs/preflight/" in capsys.readouterr().out


class TestEndOfRun:
    """最後まで流した実行の締めの表示。"""

    def test_final_summary_counts_and_lists_failures(self, isolated, capsys):
        """流れた行を遡らなくても、何件落ちて、どれが落ちたかが最後に分かること。"""
        code, _ = _run_main(
            isolated, "--static", "--profile", "dev", rows=[("ok/row", "PASS"), ("broken/row", "FAIL")],
        )
        assert code == preflight.EXIT_FAILED
        out = capsys.readouterr().out
        assert re.search(r"PASS 1\D+FAIL 1\b", out)
        assert re.search(r"^\s+FAIL.*broken/row", out, re.M)

    def test_no_details_line_when_nothing_was_logged(self, isolated, capsys):
        """問題ログを書いていないのに「詳細:」と案内すると、存在しないファイルを探すことになる。"""
        _run_main(isolated, "--static", "--profile", "dev")
        assert "詳細:" not in capsys.readouterr().out

    def test_no_pipeline_row_is_a_warning(self, isolated, capsys):
        """PASS で出ると、publish と TTS を省いた実行が表の上では合格に見える。"""
        code, _ = _run_main(isolated, "--paid", "--profile", "dev", "--no-pipeline")
        assert code == preflight.EXIT_OK
        assert _first_line_starting_with(capsys.readouterr().out, "[WARN ] no-pipeline") is not None

    def test_full_paid_run_has_no_no_pipeline_row(self, isolated, capsys):
        """省いていない実行にも出ると、配信前の合格がいつも「使えません」と読めてしまう。"""
        _run_main(isolated, "--paid", "--profile", "dev")
        assert "no-pipeline" not in capsys.readouterr().out


class TestInheritedEndpointRow:
    """点検の中では外すが、同じシェルから起動する本番には残る。緑のまま別の接続先へ行くのを防ぐ。"""

    def test_removed_endpoint_is_a_counted_warning(self, isolated, capsys):
        os.environ["ANTHROPIC_BASE_URL"] = f"http://{SECRET}.invalid"
        code, _ = _run_main(isolated, "--static", "--profile", "dev")
        out = capsys.readouterr().out
        assert code == preflight.EXIT_OK
        row = _first_line_starting_with(out, "[WARN ] env/inherited")
        assert row is not None
        assert "ANTHROPIC_BASE_URL" in row
        assert "同じシェル" in row
        assert re.search(r"WARN 1\b", out)
        assert "SENT" not in out

    def test_every_removed_variable_is_named(self, isolated, capsys):
        os.environ["OPENAI_BASE_URL"] = "http://a.invalid"
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "x"
        _run_main(isolated, "--static", "--profile", "dev")
        row = _first_line_starting_with(capsys.readouterr().out, "[WARN ] env/inherited")
        assert row is not None
        assert "OPENAI_BASE_URL" in row and "ANTHROPIC_AUTH_TOKEN" in row

    def test_is_the_first_row(self, isolated, capsys):
        """ゲートの出力は最初に目に入る。本番と違う相手を測った印は先頭に置く。"""
        os.environ["ANTHROPIC_BASE_URL"] = "http://a.invalid"
        _run_main(isolated, "--static", "--profile", "dev", rows=[("x", "PASS")])
        rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[") and not line.startswith("[preflight]")]
        assert rows[0].startswith("[WARN ] env/inherited")

    def test_no_row_when_nothing_was_removed(self, isolated, capsys):
        _run_main(isolated, "--static", "--profile", "dev")
        assert "env/inherited" not in capsys.readouterr().out

    def test_variable_left_in_place_is_not_reported(self, isolated, capsys):
        """外せなかった変数を「外した」と書かない。"""
        os.environ["ANTHROPIC_BASE_URL"] = "http://a.invalid"
        with patch.object(preflight, "clean_llm_env", return_value=[]):
            _run_main(isolated, "--static", "--profile", "dev")
        assert "env/inherited" not in capsys.readouterr().out


def _last_line(out):
    """出力の最後の行を返す。判定の行は必ず最後に出る約束なので、ここを見る。"""
    return out.strip().splitlines()[-1]


class TestVerdictLine:
    """終了コードは画面に出ないので、どの経路で終わっても最後の行で合否を読めること。"""

    @pytest.mark.parametrize("argv,options,expected", [
        (("--static", "--profile", "dev"), {}, "判定: 合格 (exit 0)"),
        (("--static", "--profile", "dev"), {"rows": [("broken", "FAIL")]}, "判定: 不合格 (exit 1)"),
        (("--paid", "--profile", "dev"), {"rows": [("broken", "FAIL")]}, "判定: 不合格 (exit 1)"),
        (("--paid", "--profile", "dev"), {"missing": ["Redis"]}, "判定: 前提サービスなし (exit 3)"),
        (("--paid", "--profile", "dev"), {"stream": preflight.PROD_STREAM}, "判定: 拒否 (exit 2)"),
    ], ids=["static-green", "static-fail", "stop-before-billing", "missing-service", "prod-refused"])
    def test_returned_paths(self, isolated, capsys, argv, options, expected):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        code, _ = _run_main(isolated, *argv, **options)
        out = capsys.readouterr().out
        assert _last_line(out) == expected
        assert f"(exit {code})" in expected
        assert out.count("判定:") == 1

    def test_full_paid_run_is_a_plain_pass(self, isolated, capsys):
        """全部流した実行に「ただし」が付くと、配信前の合格がいつも条件付きに読める。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        _run_main(isolated, "--paid", "--profile", "dev")
        assert _last_line(capsys.readouterr().out) == "判定: 合格 (exit 0)"

    def test_failure_outside_the_required_routes_is_a_pass(self, isolated, capsys):
        """別枠の FAIL は終了コードに響かない。判定の言葉も揃える。"""
        def fake_static(report, *_args, **_kwargs):
            report.add("tts/ruka", "FAIL", "voicepeak", required=False)

        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "run_static", fake_static):
            code = preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_OK
        assert "FAIL 1" in out
        assert _last_line(out) == "判定: 合格 (exit 0)"

    def test_stop_before_billing_still_says_why_above_the_verdict(self, isolated, capsys):
        _run_main(isolated, "--paid", "--profile", "dev", missing=["Redis"])
        lines = capsys.readouterr().out.strip().splitlines()
        assert "課金" in lines[-2] and "Redis" in lines[-2]

    def test_refused_env_file_ends_with_a_verdict(self, isolated, capsys):
        """.env が接続先を上書きすると prepare_env が SystemExit で止める。点検の外なので出し忘れやすい。"""
        (isolated / ".env").write_text("ANTHROPIC_BASE_URL=http://a.invalid\n", encoding="utf-8")
        with pytest.raises(SystemExit) as stopped:
            preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        assert stopped.value.code == preflight.EXIT_REFUSED
        assert _last_line(capsys.readouterr().out) == "判定: 拒否 (exit 2)"

    @pytest.mark.parametrize("argv", [
        ("--static",),
        ("--static", "--profile", "dev", "--characters", "no-such-character"),
        ("--static", "--profile", "dev", "--characters", ","),
    ], ids=["parse-args", "unknown-character", "empty-characters"])
    def test_usage_errors_end_with_a_verdict(self, isolated, capsys, argv):
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "run_static"), pytest.raises(SystemExit) as stopped:
            preflight.main(list(argv), repo_root=isolated)
        assert stopped.value.code == preflight.EXIT_USAGE
        captured = capsys.readouterr()
        assert _last_line(captured.out) == "判定: 引数の誤り (exit 64)"
        assert "usage:" in captured.err

    def test_help_is_not_judged(self, isolated, capsys):
        """--help の exit 0 に「合格」と出すと、何も測っていないのに緑に読める。"""
        with pytest.raises(SystemExit) as stopped:
            preflight.main(["--help"], repo_root=isolated)
        assert stopped.value.code == preflight.EXIT_OK
        assert "判定:" not in capsys.readouterr().out

    @pytest.mark.parametrize("outcome,expected,code", [
        (RuntimeError("x"), "判定: 不合格 (exit 1)", preflight.EXIT_FAILED),
        (KeyboardInterrupt(), "判定: 中断 (exit 130)", 130),
        (SystemExit("文言"), "判定: 不合格 (exit 1)", preflight.EXIT_FAILED),
        (SystemExit(), "判定: 不合格 (exit 1)", preflight.EXIT_FAILED),
    ], ids=["exception", "interrupt", "text-exit", "bare-exit"])
    def test_entry_point_judges_what_main_did_not(self, isolated, capsys, outcome, expected, code):
        """想定外の例外や中断は main の外で受ける。ここでも最後の行で読めること。"""
        exits = _run_entry(main=MagicMock(side_effect=outcome))
        assert exits == [code]
        assert _last_line(capsys.readouterr().out) == expected

    def test_entry_point_does_not_repeat_the_verdict_of_main(self, isolated, capsys):
        real_main = preflight.main
        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "run_static"):
            exits = _run_entry(main=lambda: real_main(["--static", "--profile", "dev"], repo_root=isolated))
        assert exits == [preflight.EXIT_OK]
        assert capsys.readouterr().out.count("判定:") == 1

    def test_entry_point_does_not_repeat_the_verdict_of_a_usage_error(self, isolated, capsys):
        exits = _run_entry(argv=("preflight", "--static"))
        assert exits == [preflight.EXIT_USAGE]
        assert capsys.readouterr().out.count("判定:") == 1

    @pytest.mark.parametrize("argv,gap", [
        (("--no-pipeline",), "no-pipeline"),
        (("--characters", "mimi"), "characters"),
    ], ids=["no-pipeline", "characters"])
    def test_pass_with_a_gap_says_it_cannot_be_used(self, isolated, capsys, argv, gap):
        """最後の行だけを読む人に、流していない実行を配信前の合格と読ませない。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        code, _ = _run_main(isolated, "--paid", "--profile", "dev", *argv)
        line = _last_line(capsys.readouterr().out)
        assert code == preflight.EXIT_OK
        assert line.startswith("判定: 合格 (exit 0)")
        assert "配信前の合格には使えません" in line
        assert gap in line

    def test_tools_off_gap_reaches_the_verdict(self, isolated, capsys):
        _run_main(isolated, "--paid", "--profile", "dev")
        assert "tools" in _last_line(capsys.readouterr().out)

    def test_failure_needs_no_gap_note(self, isolated, capsys):
        """不合格に「ただし」を添えると、読み間違いのもとになる。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"

        def fail_in_paid(report, *_args, **_kwargs):
            report.add("main/mimi", "FAIL")

        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight, "run_static"), patch.object(preflight, "missing_services", return_value=[]), \
                patch.object(preflight, "run_paid", fail_in_paid):
            code = preflight.main(["--paid", "--profile", "dev", "--no-pipeline"], repo_root=isolated)
        assert code == preflight.EXIT_FAILED
        assert _last_line(capsys.readouterr().out) == "判定: 不合格 (exit 1)"

    def test_every_exit_code_has_a_word(self):
        for code in (preflight.EXIT_OK, preflight.EXIT_FAILED, preflight.EXIT_REFUSED, preflight.EXIT_MISSING_SERVICE, preflight.EXIT_USAGE):
            assert preflight._VERDICT_WORDS[code]

    def test_module_docstring_does_not_promise_a_single_table(self):
        """行を組み直す表は出さない。説明と出力を揃える。"""
        assert "1 枚の表" not in preflight.__doc__


class TestStdoutEncoding:
    """stdout の文字コードで点検が落ちないこと。"""

    def test_text_cp932_cannot_encode_does_not_stop_the_run(self, isolated):
        """日本語環境の Windows でリダイレクトすると、stdout は cp932 の strict になる。

        行の詳細には例外文やモデル名が入るので、cp932 にない文字（ここでは em dash）も来うる。
        落ちると判定の代わりに「想定外のエラー」だけが残る。既存の別プロセスのテストは
        cp932 で書ける文字しか出さないので、この失敗を見ていない。
        """
        raw = io.BytesIO()
        stdout = io.TextIOWrapper(raw, encoding="cp932", errors="strict")
        with patch.object(sys, "stdout", stdout):
            code, _ = _run_main(isolated, "--static", "--profile", "dev", rows=[("x", "PASS", "深海—灯り")])
        assert code == preflight.EXIT_OK
        assert "深海—灯り" in raw.getvalue().decode("utf-8")

    def test_stdout_without_reconfigure_still_runs(self, isolated):
        """差し替えられた stdout（redirect_stdout の StringIO など）には reconfigure がない。"""
        stdout = io.StringIO()
        with patch.object(sys, "stdout", stdout):
            code, _ = _run_main(isolated, "--static", "--profile", "dev")
        assert code == preflight.EXIT_OK
        assert "profile=dev" in stdout.getvalue()

    def test_lone_surrogate_in_a_row_does_not_stop_the_run(self, isolated):
        """応答や例外文には孤立サロゲートが混ざりうる。strict の utf-8 だと print が例外になり、表が途切れる。"""
        raw = io.BytesIO()
        stdout = io.TextIOWrapper(raw, encoding="cp932", errors="strict")
        with patch.object(sys, "stdout", stdout):
            code, _ = _run_main(isolated, "--static", "--profile", "dev", rows=[("x", "PASS", "text=\ud83d")])
        assert code == preflight.EXIT_OK
        assert b"text=" in raw.getvalue()


class TestEntryPoint:
    """コマンドラインの入口。使い方の案内、即座に終える約束、値を出さない約束を確かめる。"""

    def test_usage_error_prints_the_usage(self, isolated, capsys):
        """--profile の付け忘れなどで、正しい書き方がその場で読めること。"""
        with pytest.raises(SystemExit):
            preflight.main(["--static"], repo_root=isolated)
        assert "usage:" in capsys.readouterr().err

    def test_system_exit_ends_through_os_exit(self, isolated):
        """SystemExit も os._exit で終える。素通しすると普通の終了になり、worker の join で止まりうる。

        main 自身が SystemExit を出すのは worker を作る前だけだが、--paid の途中でライブラリが
        sys.exit を呼ぶと、打ち切った Agent の worker を待って終わらず、その間も API を呼び続けて
        課金が増える。既存の別プロセスのテストは終了コードしか見ていないので、素通しでも緑になる。
        """
        exits = _run_entry(main=MagicMock(side_effect=SystemExit(preflight.EXIT_USAGE)))
        assert exits == [preflight.EXIT_USAGE]

    @pytest.mark.parametrize("value", [_TEXT_REDACT_CANNOT_HIDE, None])
    def test_non_numeric_system_exit_is_a_failure_and_hides_its_text(self, isolated, capsys, value):
        """ライブラリの sys.exit("文言") や sys.exit() が途中で来ても、緑にせず、文言も出さない。

        os._exit は数値しか受けないので、そのまま渡すと TypeError で os._exit に届かなくなる。
        文言には接続先の URL などが入りうる。
        """
        exits = _run_entry(main=MagicMock(side_effect=SystemExit(value)))
        assert exits == [preflight.EXIT_FAILED]
        captured = capsys.readouterr()
        assert "SENTINELpw" not in captured.out + captured.err

    def test_help_text_reaches_a_pipe(self, isolated):
        """パイプ先の stdout はブロックバッファで、print_help は flush しない。

        os._exit はバッファを捨てるので、flush がないと `--help | more` が空になる。
        """
        raw = io.BytesIO()
        at_exit = []
        with patch.object(sys, "stdout", io.TextIOWrapper(raw, encoding="utf-8")):
            exits = _run_entry(argv=("preflight", "--help"), on_exit=lambda: at_exit.append(raw.getvalue()))
        assert exits == [preflight.EXIT_OK]
        assert b"--profile" in at_exit[0]

    def test_unterminated_stderr_line_is_not_lost(self, isolated):
        """本物の stderr は行バッファなので、改行で終わらない書き込みはバッファに残り、os._exit で消える。

        argparse の文言は改行で終わるので flush がなくても出る。ここで守るのは、SDK の進み具合の
        表示のように途中で切れた行だけ。
        """
        raw = io.BytesIO()
        at_exit = []

        def write_partial_line():
            sys.stderr.write("途中の行")
            return preflight.EXIT_OK

        with patch.object(sys, "stderr", io.TextIOWrapper(raw, encoding="utf-8", line_buffering=True)):
            _run_entry(main=write_partial_line, on_exit=lambda: at_exit.append(raw.getvalue()))
        assert "途中の行".encode() in at_exit[0]

    def test_module_entry_hides_an_unexpected_exception(self, isolated, capsys):
        """`python -m` の入口が _cli を通らないと、想定外の例外が traceback で値ごと stderr に出る。

        既存の別プロセスのテストは使い方の誤りだけを流すので、main を直接呼ぶ入口でも 64 で緑になる。
        """
        import runpy

        def boom():
            raise RuntimeError(_TEXT_REDACT_CANNOT_HIDE)

        # run_module は新しい名前空間で動くので repo_root を渡せない。実 .env を読まないように止める。
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        exits = []
        with patch.object(sys, "argv", ["preflight", "--static", "--profile", "dev"]), \
                patch("lab_lounge.bus._stream_key", boom), patch("os._exit", exits.append), \
                patch.object(logging, "lastResort", logging.lastResort), warnings.catch_warnings():
            # 読み込み済みのモジュールを実行し直すという警告。ここでは意図どおりなので黙らせる。
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("lab_lounge.preflight", run_name="__main__")
        assert exits == [preflight.EXIT_FAILED]
        captured = capsys.readouterr()
        assert "SENTINELpw" not in captured.out + captured.err
        assert "RuntimeError" in captured.out


# ─── 点検の抜けを直した箇所の固定 ──────────────────────────────────
#
# 変異テストの仕分けで見つかった、「緑のまま本番が壊れる」「例外で表が途切れる」「発話が画面へ出る」穴。


def _pydantic_error():
    """本物の pydantic の検証エラーを作る。input_value= に、検証に落ちた応答がそのまま入る。"""
    from pydantic import BaseModel, ValidationError

    class Response(BaseModel):
        speed: int

    try:
        # 本物の構造化出力は複数のキーを持つ。カンマが本文より前に来ないと、input_value を最初のカンマまでしか
        # 消さない誤りを見逃す。
        Response.model_validate({"pose": "x", "response": BODY})
    except ValidationError as exc:
        return exc
    raise AssertionError("検証に落ちるはずの入力です")


def _shows_answer(text):
    """応答の一部でも表示に残っていれば True を返す。

    pydantic は長い値の中ほどを ... に縮めるので、BODY の全文を探すと、出ていても見逃す。
    """
    return "BODY" in text or "実発話" in text


class TestModelOutputStaysOffTheScreen:
    """構造化出力の検証に落ちた応答は発話そのもの。例外の本文に載っても、FAIL の行へ出さない。

    SDK の更新で最初に壊れるのがこの経路なので、配信前の点検で実際に通る。
    """

    def test_structured_output_error_hides_the_answer(self):
        """langchain が再試行なしに投げる例外。前置きが短いので、応答の先頭が 200 字の中に入る。"""
        from langchain.agents.structured_output import StructuredOutputValidationError
        from langchain_core.messages import AIMessage

        error = StructuredOutputValidationError("Response", _pydantic_error(), AIMessage(content=""))
        described = preflight._describe_exc(error)
        assert not _shows_answer(described)
        assert "StructuredOutputValidationError" in described

    @pytest.mark.parametrize("make", [
        lambda: _pydantic_error(),
        lambda: __import__("langchain_core.exceptions", fromlist=["x"]).OutputParserException(f"Invalid json output: {BODY}"),
    ], ids=["pydantic", "output-parser"])
    def test_other_errors_that_carry_the_answer_show_only_their_type(self, make):
        """OutputParserException は input_value= の形を取らず、本文をそのまま持つ。"""
        error = make()
        described = preflight._describe_exc(error)
        assert not _shows_answer(described)
        assert type(error).__name__ in described

    @pytest.mark.parametrize("module", [
        "langchain.agents.structured_output",
        "langchain_core.exceptions",
        "pydantic.errors",
        "pydantic_core._pydantic_core",
    ])
    def test_errors_from_libraries_that_echo_the_answer_show_no_text(self, module):
        """input_value= の形だけを当てにすると、ライブラリの更新で文言の形が変わったときに本文が出る。"""
        error = type("Err", (Exception,), {"__module__": module})(BODY)
        assert not _shows_answer(preflight._describe_exc(error))

    def test_validation_text_wrapped_in_a_plain_value_error_is_hidden(self):
        """langchain の _parse_with_schema は検証エラーの本文を手元の ValueError に包むので、型では見分けられない。"""
        described = preflight._describe_exc(ValueError(f"Failed to parse data to Response: {_pydantic_error()}"))
        assert not _shows_answer(described)
        # 手元の例外の文言は調査に要るので残す。
        assert "Failed to parse data" in described

    def test_recursion_limit_keeps_its_text(self):
        """本文に応答を含まない例外まで隠すと、壊れた応答の再試行で止まったことが読めなくなる。"""
        from langgraph.errors import GraphRecursionError

        assert "Recursion limit" in preflight._describe_exc(GraphRecursionError("Recursion limit of 16 reached"))

    def test_long_container_argument_is_replaced_by_its_length(self, capture):
        """{'text': 発話} をそのまま出すログがある。引数が 2 つ以上だと args は tuple のまま届く。"""
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "r=%s n=%d", ({"text": BODY * 3}, 7)))
        reason = capture.last_problem(0)
        assert "BODYSENTINEL" not in reason
        assert "n=7" in reason

    @pytest.mark.parametrize("arg,shown", [(None, "None"), (1.5, "1.5")])
    def test_values_without_text_are_kept(self, capture, arg, shown):
        """本文を含まない値まで長さへ置き換えると、理由が読めなくなる。"""
        capture.emit(_log_record("lab_lounge.x", logging.WARNING, "値=%s", (arg, )))
        assert shown in capture.last_problem(0)


# ─── 画面へ出る host、相手の応答、ホームのパス ────────────────────────
#
# 例外の文言は壊れたときほど詳しくなり、配信中のコンソールや記録に貼られる。ここでは実物の例外を作って、
# host、発話、ユーザー名が行に残らないことを固定する。本文は pydantic が中ほどを ... に縮めたり、行の上限で
# 切られたりするので、全文ではなく短い印（_shows_answer）で探す。

_LEAK_LONG_BODY = f"{BODY} {'y' * 200} {BODY}"


def _leak_irodori_failure(raised, url="http://gpu-box.invalid:18080/synthesize"):
    """本番の _generate_irodori_single_file が実際に作る例外を受け取る。"""
    from lab_lounge import tts

    with patch("urllib.request.urlopen", side_effect=raised), pytest.raises(RuntimeError) as caught:
        tts._generate_irodori_single_file(
            _LEAK_LONG_BODY, caption="c", ref_wav=None, duration_scale=1.0, num_steps=1, t_schedule_mode="sway",
            cfg_scale_speaker=1.0, seed=None, filepath=Path("x.wav"), url=url,
        )
    return caught.value


def _leak_sidecar_answers(code, body):
    """サイドカーが code と body を返したときの例外を作る。"""
    import urllib.error

    return _leak_irodori_failure(urllib.error.HTTPError("http://x/", code, "reason", {}, io.BytesIO(body.encode("utf-8"))))


def _leak_sidecar_refuses():
    """サイドカーに繋がらないときの例外を作る。"""
    import urllib.error

    return _leak_irodori_failure(urllib.error.URLError(ConnectionRefusedError(10061, "refused")))


def _leak_redis_failure(url):
    """名前の解決に落ちたときの、実物の redis-py の例外を作る。文言に host:port が入る。"""
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    client = redis.Redis.from_url(url, retry=Retry(NoBackoff(), 0))
    with patch("socket.getaddrinfo", side_effect=socket.gaierror(11001, "getaddrinfo failed")), \
            pytest.raises(redis.exceptions.ConnectionError) as caught:
        client.ping()
    return caught.value


def _leak_set_home(monkeypatch, value):
    """Path.home の結果を固定する。実物のホームに頼ると、HOME=/ の環境でテストの意味が変わる。"""
    def home(_cls):
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(Path, "home", classmethod(home))


class TestLeakConfiguredHost:
    """host はライブラリや本番の文言に入るので、型ではなく設定の値で伏せる。"""

    def test_sidecar_connection_failure_hides_the_url(self, isolated):
        os.environ["L2_TTS_IRODORI_URL"] = "http://gpu-box.invalid:18080"
        described = preflight._describe_exc(_leak_sidecar_refuses())
        assert "gpu-box" not in described and "18080" not in described
        # 起動の案内と errno は調査に要るので残す。
        assert "接続できません" in described and "10061" in described

    def test_redis_failure_hides_the_host_and_keeps_the_errno(self, isolated):
        """型だけにすると、Redis の障害だと行から読めなくなる。"""
        os.environ["REDIS_URL"] = "redis://:pw-value@redis-box.invalid:6379/0"
        described = preflight._describe_exc(_leak_redis_failure(os.environ["REDIS_URL"]))
        assert "redis-box" not in described and "pw-value" not in described
        assert "ConnectionError" in described and "11001" in described

    @pytest.mark.parametrize("name", [
        "REDIS_URL", "L2_C2_URL", "L2_C2_URL_READONLY", "L2_TTS_VOICEVOX_URL", "L2_TTS_IRODORI_URL", "L2_OBS_WS_URL",
    ])
    def test_each_endpoint_is_hidden_by_value(self, isolated, name):
        """定数をループすると、一覧から消えた名前はテストからも消える。名前は直書きする。"""
        os.environ[name] = "http://Some-Host.invalid:1"
        assert "some-host" not in preflight._one_line("to some-host.invalid:1 and SOME-HOST.invalid").lower()

    @pytest.mark.parametrize("value,text", [
        ("redis://198.51.100.7:6379", "Error 10060 connecting to 198.51.100.7:6379."),
        ("redis://[2001:db8::7]:6379", "Error 10060 connecting to 2001:db8::7:6379."),
        ("redis://gpu-box:6379", "gpu-boxへ接続できません"),
    ], ids=["public-ip", "ipv6", "japanese-follows"])
    def test_other_spellings_of_the_host_are_hidden(self, isolated, value, text):
        """日本語の文字は \\w に一致するので、直後に続くと host を見逃しやすい。"""
        os.environ["REDIS_URL"] = value
        shown = preflight._one_line(text)
        assert "[HOST]" in shown
        assert not any(part in shown for part in ("198.51", "db8", "gpu-box"))

    def test_every_endpoint_setting_in_the_source_is_listed(self):
        """値で伏せる方式の弱点は、一覧へ足し忘れた接続先がそのまま出ること。本番のソースと突き合わせる。"""
        from lab_lounge import tts

        read = set()
        for path in Path(tts.__file__).parent.rglob("*.py"):
            if path.name != "preflight.py":
                read.update(re.findall(r"""environ\.get\(\s*["']([A-Z0-9_]*_URL[A-Z0-9_]*)["']""", path.read_text(encoding="utf-8")))
        assert read == set(preflight._ENDPOINT_ENV_NAMES)

    def test_dot_in_a_host_is_not_a_wildcard(self, isolated):
        os.environ["REDIS_URL"] = "redis://gpu.box:6379"
        assert preflight._one_line("gpuxbox gpu.box") == "gpuxbox [HOST]"

    def test_longer_host_is_replaced_first(self, isolated):
        """短い方を先に置き換えると [HOST].lan が残る。集合の順序は実行ごとに変わるので、順序そのものを見る。"""
        os.environ["L2_C2_URL"] = "http://box:8100"
        os.environ["REDIS_URL"] = "redis://box.lan:6379"
        assert preflight._endpoint_hosts() == ["box.lan", "box"]
        assert preflight._one_line("to box.lan:6379 and box:8100") == "to [HOST]:6379 and [HOST]:8100"

    def test_host_outside_the_settings_is_hidden_by_the_url_form(self, isolated):
        described = preflight._describe_exc(RuntimeError("cannot reach http://other-box.invalid:9/x (down)"))
        assert described == "RuntimeError: cannot reach <URL> (down)"

    def test_loopback_and_lookalike_words_are_left_alone(self, isolated):
        """loopback は伏せても情報が増えず、調査の手がかりを減らすだけ。別の語まで置き換えると読めなくなる。"""
        os.environ["REDIS_URL"] = "redis://127.0.0.1:6379"
        os.environ["L2_C2_URL"] = "http://c2:8100"
        assert preflight._one_line("127.0.0.1:6379") == "127.0.0.1:6379"
        assert preflight._one_line("L2_C2_URL abc2 c2-backup x.c2") == "L2_C2_URL abc2 c2-backup x.c2"
        assert preflight._one_line("to c2:8100") == "to [HOST]:8100"

    def test_row_names_and_the_problem_log_keep_their_text(self, isolated, capsys):
        """host が c2 のとき、行の名前まで置き換えると表が読めなくなる。問題ログは全文を残す場所。"""
        os.environ["L2_C2_URL"] = "http://c2:8100"
        preflight.Report().add("c2", "PASS", "search_mode=semantic")
        assert "c2: search_mode=semantic" in capsys.readouterr().out
        log = isolated / "p.log"
        preflight._LogCapture(log).note_exception("c2", OSError("to c2:8100"))
        assert "to c2:8100" in log.read_text(encoding="utf-8")

    def test_failed_pipeline_row_hides_the_host(self, isolated):
        os.environ["L2_TTS_IRODORI_URL"] = "http://gpu-box.invalid:18080"
        log = isolated / "problem.log"
        check, _, _ = _run_pipeline_check(preflight._LogCapture(log), side_effect=_leak_sidecar_refuses())
        assert check.status == "FAIL"
        assert "gpu-box" not in check.detail
        assert "gpu-box" in log.read_text(encoding="utf-8")

    def test_logged_exception_in_the_reason_hides_the_host(self, isolated, capture):
        """None を返す関数の理由はログから拾う。ここも _describe_exc を通る。"""
        os.environ["REDIS_URL"] = "redis://redis-box.invalid:6379"
        error = _leak_redis_failure(os.environ["REDIS_URL"])
        capture.emit(_log_record("lab_lounge.router", logging.ERROR, "LLM ルーター呼び出し失敗: %s", (error,)))
        assert "redis-box" not in capture.last_problem(0)


class TestLeakPeerText:
    """型は手元の例外でも、文言に相手の応答が載っていれば画面へ出さない。全文は問題ログへ残す。"""

    @pytest.mark.parametrize("code,body", [
        (500, json.dumps({"error": f"ValueError: too long: {_LEAK_LONG_BODY}"}, ensure_ascii=False)),
        (422, json.dumps({"detail": [{"msg": "Field required", "input": {"text": _LEAK_LONG_BODY}}]}, ensure_ascii=False)),
        (400, f"line1\n{BODY}\nline3"),
    ], ids=["sidecar-500", "validation-echo", "multiline"])
    def test_sidecar_response_shows_only_the_status(self, isolated, code, body):
        described = preflight._describe_exc(_leak_sidecar_answers(code, body))
        assert not _shows_answer(described)
        assert f"irodori サイドカー HTTP {code}" in described

    def test_voicepeak_output_is_hidden(self):
        """本番の _submit_voicepeak の例外。CLI の stderr / stdout は引数をそのまま含みうる。"""
        from lab_lounge import tts

        class FinishedWorker:
            """キューの代わり。受け取ったジョブを、CLI が 3 で終わった結果にする。"""

            def put(self, item):
                output = _LEAK_LONG_BODY.encode("utf-8")
                item[1].set_result(subprocess.CompletedProcess("x", 3, stdout=output, stderr=output))

        with patch.object(tts, "_ensure_voicepeak_worker", return_value=FinishedWorker()), \
                pytest.raises(RuntimeError) as caught:
            tts._submit_voicepeak("voicepeak --say x")
        assert "stderr=" in str(caught.value)  # 前提: 本番の文言が同じ形であること
        described = preflight._describe_exc(caught.value)
        assert not _shows_answer(described)
        assert "returncode=3" in described

    def test_cli_output_is_hidden_whichever_comes_first(self):
        """本番の文言の順序が変わっても、CLI の出力を行へ出さない。"""
        assert not _shows_answer(preflight._describe_exc(RuntimeError(f"failed: stdout={BODY!r}")))

    def test_error_dict_in_a_value_error_is_hidden(self):
        """langchain_openai は 200 の応答の error を dict のまま ValueError へ渡す。"""
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(api_key="dummy", model="m")
        with pytest.raises(ValueError) as caught:
            model._create_chat_result({"error": {"message": f"refused: {_LEAK_LONG_BODY}", "code": 400}})
        described = preflight._describe_exc(caught.value)
        assert not _shows_answer(described)
        assert "ValueError" in described

    def test_error_object_in_a_value_error_is_hidden(self):
        """Responses API の経路は pydantic のまま ValueError へ渡す。"""
        from langchain_openai.chat_models.base import _construct_lc_result_from_responses_api
        from openai.types.responses import ResponseError

        response = SimpleNamespace(error=ResponseError(code="server_error", message=f"failed: {_LEAK_LONG_BODY}"))
        with pytest.raises(ValueError) as caught:
            _construct_lc_result_from_responses_api(response)
        assert not _shows_answer(preflight._describe_exc(caught.value))

    @pytest.mark.parametrize("raw", [{"response": "x", "speed": _LEAK_LONG_BODY}, {"response": "x", "emotion": {"happy": _LEAK_LONG_BODY}}], ids=["speed", "emotion"])
    def test_model_value_in_a_failed_conversion_is_hidden(self, raw):
        """本番の _parse_voicepeak_json は int() で読む。組み込みの例外の文言に、モデルの返した値がそのまま載る。"""
        from lab_lounge.tts import _parse_voicepeak_json

        with pytest.raises(ValueError) as caught:
            _parse_voicepeak_json(json.dumps(raw, ensure_ascii=False))
        described = preflight._describe_exc(caught.value)
        assert not _shows_answer(described)
        assert "invalid literal for int()" in described

    def test_shortened_pydantic_value_leaves_no_fragment(self):
        """pydantic は長い値の中ほどを ... に縮める。全文を探す形だと、前後の断片が残っていても見逃す。"""
        from pydantic import BaseModel, ValidationError

        class Response(BaseModel):
            speed: int

        with pytest.raises(ValidationError) as caught:
            Response.model_validate({"speed": _LEAK_LONG_BODY})
        assert "..." in str(caught.value) and "BODY" in str(caught.value)  # 前提: 縮めた形が本当に出ていること
        described = preflight._describe_exc(ValueError(f"Failed to parse data to Response: {caught.value}"))
        assert not _shows_answer(described)
        assert "Failed to parse data" in described and "valid integer" in described

    def test_production_schema_leaves_no_fragment(self):
        """本番のスキーマを langchain の検証に通す。loc の行はスキーマの名前だけで、値は input_value にしか載らない。"""
        from langchain.agents.structured_output import _parse_with_schema

        from lab_lounge.graph import _get_character_response_schema

        raw = {"response": 3, "emotion": {"happy": _LEAK_LONG_BODY, _LEAK_LONG_BODY: 1}, "speed": _LEAK_LONG_BODY, _LEAK_LONG_BODY: 1}
        with pytest.raises(ValueError, match="Failed to parse data") as caught:
            _parse_with_schema(_get_character_response_schema("mimi"), "pydantic", raw)
        assert "BODY" in str(caught.value)  # 前提: 検証の本文に応答が載っていること
        described = preflight._describe_exc(caught.value)
        assert not _shows_answer(described)
        assert "Failed to parse data" in described

    def test_tool_invocation_error_is_hidden_through_its_parent_class(self):
        """langgraph の例外だが親は langchain_core の ToolException。kwargs の本文を文言に含む。"""
        from langgraph.prebuilt.tool_node import ToolInvocationError
        from pydantic import BaseModel, ValidationError

        class Args(BaseModel):
            question: int

        with pytest.raises(ValidationError) as caught:
            Args.model_validate({"question": BODY})
        error = ToolInvocationError("ask_character_tool", caught.value, {"question": BODY})
        assert "BODY" in str(error)  # 前提: 本文が文言に入ること
        assert preflight._describe_exc(error) == "ToolInvocationError"

    @pytest.mark.parametrize("error,kept", [
        (ValueError("Failed to parse data"), "Failed to parse data"),
        (TypeError("unexpected keyword argument 'temperature'"), "temperature"),
        (OSError(10061, "refused"), "10061"),
    ], ids=["plain-value-error", "type-error", "os-error"])
    def test_local_text_is_kept(self, error, kept):
        """手元の例外の文言まで隠すと、SDK の使い方が壊れたときの原因が読めなくなる。"""
        assert kept in preflight._describe_exc(error)

    def test_failed_pipeline_row_hides_the_body_and_the_log_keeps_it(self, isolated):
        log = isolated / "problem.log"
        error = _leak_sidecar_answers(500, json.dumps({"error": _LEAK_LONG_BODY}, ensure_ascii=False))
        check, _, _ = _run_pipeline_check(preflight._LogCapture(log), side_effect=error)
        assert check.status == "FAIL"
        assert not _shows_answer(check.detail)
        assert "BODY" in log.read_text(encoding="utf-8")

    def test_logged_exception_in_the_reason_hides_the_body(self, capture):
        error = ValueError({"message": f"refused: {_LEAK_LONG_BODY}"})
        capture.emit(_log_record("lab_lounge.router", logging.ERROR, "LLM ルーター呼び出し失敗: %s", (error,)))
        assert not _shows_answer(capture.last_problem(0))


class TestLeakHomePath:
    """OSError の本文はパスを repr で持つので、Windows では区切りが 2 文字になり、全文一致に当たらない。"""

    @pytest.mark.parametrize("home", ["C:\\Users\\dummyuser", "/home/dummyuser"], ids=["windows", "posix"])
    @pytest.mark.parametrize("render", [
        lambda home: str(FileNotFoundError(2, "No such file", f"{home}/x.wav")),
        lambda home: str(PermissionError(13, "denied", home.replace("/", "\\") + "\\x.wav")),
        lambda home: str(FileExistsError(17, "exists", f"{home}/a", f"{home}/b")),
        lambda home: f"no file {home}/x",
        lambda home: f"no file {home}\\x",
        lambda home: f"no file {home.replace(chr(92), '/')}/x",
        lambda home: f"no file {home.lower()}/x",
        lambda home: f"no file {home.upper()}/x",
        lambda home: repr(repr(f"{home}/x")),
    ], ids=["repr", "repr-backslash", "two-paths", "plain-slash", "plain-backslash", "forward", "lower", "upper", "nested-repr"])
    def test_home_never_reaches_the_screen(self, monkeypatch, home, render):
        _leak_set_home(monkeypatch, home)
        described = preflight._describe_exc(OSError(render(home)))
        assert "dummyuser" not in described.lower()
        assert "~" in described

    def test_real_oserror_is_hidden(self, monkeypatch, tmp_path):
        """実物の OSError で確かめる。文言の形は Python が決めるので、作った文字列だけでは足りない。"""
        home = tmp_path / "dummyuser"
        _leak_set_home(monkeypatch, home)
        with pytest.raises(OSError) as caught:
            (home / "no-such-dir" / "x.wav").read_text(encoding="utf-8")
        described = preflight._describe_exc(caught.value)
        assert "dummyuser" not in described.lower()
        assert "no-such-dir" in described

    def test_home_with_regex_characters_is_taken_literally(self, monkeypatch):
        """ユーザー名の ( や + を正規表現として読むと、例外になるか見逃す。"""
        _leak_set_home(monkeypatch, "C:\\Users\\du(mmy+user")
        assert preflight.redact("C:\\Users\\du(mmy+user\\x") == "~\\x"

    def test_japanese_user_name_is_hidden(self, monkeypatch):
        _leak_set_home(monkeypatch, "C:\\Users\\ダミー")
        assert "ダミー" not in preflight.redact(str(FileNotFoundError(2, "No such file", "C:\\Users\\ダミー\\x")))

    def test_other_directories_are_left_alone(self, monkeypatch):
        _leak_set_home(monkeypatch, "C:\\Users\\dummyuser")
        text = "D:\\work\\x C:\\Users\\Public\\y deps/uv.lock"
        assert preflight.redact(text) == text

    def test_problem_log_also_hides_the_home(self, monkeypatch, tmp_path):
        """問題ログは host を残すが、ユーザー名は残さない（既存の redact の範囲）。"""
        _leak_set_home(monkeypatch, "C:\\Users\\dummyuser")
        log = tmp_path / "p.log"
        preflight._LogCapture(log).note_exception("x", FileNotFoundError(2, "No such file", "C:\\Users\\dummyuser\\x"))
        assert "dummyuser" not in log.read_text(encoding="utf-8")


class TestLeakUnusualHome:
    """UID だけを指定したコンテナは HOME=/ になり、ホームが決まらない環境もある。どちらでも表を読めるようにする。"""

    @pytest.mark.parametrize("home", ["/", "\\", "C:\\", "C:/", "", "."])
    def test_home_at_the_root_does_not_rewrite_the_table(self, monkeypatch, home):
        """置き換えると deps/uv.lock が deps~uv.lock になり、key/openai の行も崩れる。"""
        _leak_set_home(monkeypatch, home)
        text = "deps/uv.lock key/openai C:\\x data.json"
        assert preflight.redact(text) == text

    def test_undeterminable_home_does_not_stop_the_output(self, monkeypatch):
        _leak_set_home(monkeypatch, RuntimeError("Could not determine home directory."))
        assert preflight.redact("key/openai") == "key/openai"

    def test_static_run_reaches_the_summary_without_a_home(self, isolated, monkeypatch, capsys):
        """redact が例外を出すと _emit がすべて失敗し、1 行も出ないまま終わる。"""
        _leak_set_home(monkeypatch, RuntimeError("Could not determine home directory."))
        preflight.main(["--static", "--profile", "dev", "--characters", "mimi"], repo_root=isolated)
        out = capsys.readouterr().out
        assert "== preflight --static" in out
        assert "PASS " in out and "FAIL " in out

    def test_problem_log_does_not_reach_the_caller_without_a_home(self, isolated, monkeypatch):
        """ログの取り込みで例外を出すと、ログを出した本番の呼び出し側まで巻き込む。"""
        _leak_set_home(monkeypatch, RuntimeError("Could not determine home directory."))
        path = isolated / "p.log"
        with preflight._captured_logs(path):
            logging.getLogger("lab_lounge.x").error("失敗")
        assert "失敗" in path.read_text(encoding="utf-8")


class TestLeakRedactFailure:
    def test_surrogate_in_a_secret(self, isolated):
        """POSIX の非 UTF-8 の環境変数は孤立サロゲートになり、既定の quote は例外を出す。"""
        os.environ["L2_OBS_WS_PASSWORD"] = "abcd\udc80efgh"
        assert preflight.redact("x abcd\udc80efgh y") == "x [REDACTED] y"

    def test_emit_prints_one_line_without_the_value(self, capsys):
        """伏せられなかった行は値を含みうるので、そのまま出さない。"""
        with patch.object(preflight, "redact", side_effect=RuntimeError("redact")):
            preflight._emit(f"line {SECRET}")
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert SECRET not in lines[0] and "RuntimeError" in lines[0]

    def test_last_resort_leaves_a_line_when_redact_fails(self, isolated, capsys):
        """最後の砦も _emit を通る。ここで何も出ないと、exit 1 の理由を読めない。"""
        with patch.object(preflight, "redact", side_effect=RuntimeError("redact")):
            exits = _run_entry(main=MagicMock(side_effect=ValueError("x")))
        assert exits == [preflight.EXIT_FAILED]
        assert capsys.readouterr().out.strip() != ""


class TestLeakToolName:
    """ツール名はモデルの出力。本文を一切出さない行のうち、ここだけ素通しになると発話が PASS の行に載る。"""

    @staticmethod
    def _summary(*names):
        from langchain_core.messages import AIMessage

        calls = [{"name": name, "args": {}, "id": str(i)} for i, name in enumerate(names)]
        return preflight._summarize_agent_result({"messages": [AIMessage(content="", tool_calls=calls)]})

    @pytest.mark.parametrize("name", [BODY, "a" * 65, "with.dot", "two words", "name\n", ""], ids=["body", "long", "dot", "space", "newline", "empty"])
    def test_odd_name_is_not_shown(self, name):
        assert self._summary(name, "web_search_tool")["tools"] == ["<不正な名前>", "web_search_tool"]

    @pytest.mark.parametrize("name", ["ask_character_tool", "web-search_2", "a" * 64])
    def test_ordinary_name_is_kept(self, name):
        """協働の点検は ask_character_tool の有無で判定する。ここを崩すと 2 段目の確認が消える。"""
        assert self._summary(name)["tools"] == [name]

    def test_structured_output_tool_is_still_left_out(self):
        assert self._summary("MimiResponse")["tools"] == []

    def test_pass_row_of_the_main_route_hides_the_name(self, isolated, capture, capsys):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        result = _agent_result(SimpleNamespace(response="こんにちは"), tool_calls=[{"name": BODY}])
        report = preflight.Report()
        with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
                patch("lab_lounge.graph._invoke_agent_with_timeout", return_value=result):
            assert preflight.check_agent(report, capture, get_character("mimi"), None, required=True) is True
        assert not _shows_answer(capsys.readouterr().out)
        assert "<不正な名前>" in _status(report, "main/mimi").detail

    def test_collab_row_hides_the_name(self, isolated, capture):
        check, _, _ = _run_collab_check(capture, tools=(BODY,))
        assert not _shows_answer(check.detail)


class TestLeakImportFailure:
    """pydantic の例外は画面が型の名前だけになる。ログにもなければ、本文がどこにも残らない。"""

    def test_full_text_reaches_the_problem_log(self, isolated):
        from pydantic.errors import PydanticUserError

        log = isolated / "logs" / "preflight" / "p.log"
        report = preflight.Report()
        with patch.object(preflight.importlib, "import_module", side_effect=PydanticUserError("KEEPTHIS", code=None)):
            preflight._check_imports(report, {"openai"}, preflight._LogCapture(log))
        assert "KEEPTHIS" in log.read_text(encoding="utf-8")
        assert "KEEPTHIS" not in report.checks[0].detail

    @staticmethod
    def _run_main_with_a_broken_sdk(isolated, mode):
        """langchain_openai の import だけを落として main を流す。--paid は設定の FAIL で止まるので課金しない。"""
        from pydantic.errors import PydanticUserError

        real = preflight.importlib.import_module

        def boom(name, *args, **kwargs):
            if name == "langchain_openai":
                raise PydanticUserError("KEEPTHIS", code=None)
            return real(name, *args, **kwargs)

        os.environ["REDIS_STREAM_KEY"] = "aibyss:events-dev"
        with patch.object(preflight.importlib, "import_module", boom):
            return preflight.main([mode, "--profile", "dev", "--characters", "mimi"], repo_root=isolated)

    @pytest.mark.parametrize("mode", ["--static", "--paid"])
    def test_main_passes_the_capture_on(self, isolated, capsys, mode):
        """入口で渡し忘れると、上のテストは緑のまま本番ではログへ届かない。"""
        self._run_main_with_a_broken_sdk(isolated, mode)
        logs = list((isolated / "logs" / "preflight").glob("*.log"))
        assert any("KEEPTHIS" in path.read_text(encoding="utf-8") for path in logs)
        # 画面は型の名前だけなので、全文の場所を示さないと原因を読みに行けない。--paid は途中で止まる経路を通る。
        assert "詳細:" in capsys.readouterr().out


def _leak_embeddings_error(capture):
    os.environ["L2_ENABLE_TOOLS"] = "true"
    _run_embeddings_raising(capture, _FakeApiError(BODY))


def _leak_router_answer(capture):
    os.environ.update({"L2_USE_LLM_ROUTER": "true", "L2_USE_HANDRAISE": "false"})
    with patch("lab_lounge.router._call_router_llm", return_value=BODY):
        preflight.check_router(preflight.Report(), capture)


def _leak_filler_answer(capture):
    character = SimpleNamespace(slug="x", filler_model="m")
    with patch("lab_lounge.filler._call_filler_llm", return_value='{"response": "' + BODY), \
            patch("lab_lounge.filler._build_filler_prompt", return_value="P"):
        preflight.check_filler(preflight.Report(), capture, character, required=True)


def _leak_collab_chunk(capture):
    from lab_lounge.mcp_servers import ask_character as ac

    def fake_invoke(*_args, **_kwargs):
        ac._on_tts_chunk_var.get()("u", f"エラー: {BODY}", True, "chisame")
        return _agent_result(SimpleNamespace(response="x"), tool_calls=[{"name": "ask_character_tool"}])

    with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
            patch("lab_lounge.graph._invoke_agent_with_timeout", fake_invoke):
        preflight.check_collab(preflight.Report(), capture, get_character("sakura"), get_character("chisame"), "x", None)


def _leak_c2_exception(_capture):
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(f"host.invalid {BODY}")):
        preflight._check_c2_endpoint(preflight.Report(), "c2", "http://127.0.0.1:8100", required=True)


def _leak_c2_wrong_shape(_capture):
    response = MagicMock()
    response.read.return_value = json.dumps([BODY]).encode("utf-8")
    response.__enter__.return_value = response
    with patch("urllib.request.urlopen", return_value=response):
        preflight._check_c2_endpoint(preflight.Report(), "c2", "http://127.0.0.1:8100", required=True)


@pytest.mark.parametrize("run", [
    lambda capture: _run_tavily(capture, raises=_FakeApiError(BODY)),
    _leak_embeddings_error,
    _leak_router_answer,
    _leak_filler_answer,
    lambda capture: _run_pipeline_check(capture, events=[{"type": "tts.done", "payload": {"text": f"エラー: {BODY}"}}]),
    _leak_collab_chunk,
    _leak_c2_exception,
    _leak_c2_wrong_shape,
], ids=["tavily-raises", "embeddings-raises", "router-answer", "filler-answer", "tts-done-text", "collab-chunk", "c2-exception", "c2-body"])
def test_no_failure_row_shows_the_body(isolated, capture, capsys, run):
    """README の約束: 画面には型と status だけで、発話本文を出さない。全文は問題ログへ残す。

    点検の出力は配信中のコンソールに貼られることがあるので、緑のまま本文が出るようになると気づけない。
    """
    run(capture)
    out = capsys.readouterr().out
    assert "FAIL" in out  # 失敗の経路を通ったことの確認。通っていなければ下の assert は何も見ていない
    assert not _shows_answer(out)


class TestRedactionLengthRule:
    @pytest.mark.parametrize("value", ["    ", "   a"])
    def test_padded_short_value_does_not_wipe_the_table(self, isolated, value):
        """空白だけの値を strip して置換すると全文字の間に印が入り、1 文字の値は無関係な文字まで消す。"""
        os.environ["OPENAI_API_KEY"] = value
        assert preflight.redact("banana") == "banana"

    def test_padded_real_value_is_still_hidden(self, isolated):
        """.env の引用符に空白が入っても、中身が十分長ければ伏せる。"""
        os.environ["OPENAI_API_KEY"] = "  abcd  "
        assert "abcd" not in preflight.redact("key abcd")


class TestLoggingNeverReachesTheCaller:
    """取り込みの例外はログを出した本番の呼び出し側へ届く。本番の except に捕まると別の経路へ落ち、点検の結果が変わる。"""

    def test_format_that_does_not_match_its_mapping(self, isolated):
        with preflight._captured_logs() as capture:
            logging.getLogger("lab_lounge.x").warning("x %(a)s", {"b": 1})
        assert capture.messages_since(0) == ["x %(a)s"]
        assert capture.last_problem(0) == "x %(a)s"

    def test_argument_whose_str_raises(self, isolated):
        class Broken:
            def __str__(self):
                raise RuntimeError("broken")

        with preflight._captured_logs() as capture:
            logging.getLogger("lab_lounge.x").warning("x %s", Broken())
        assert capture.messages_since(0) == ["x %s"]
        assert capture.last_problem(0) == "x %s"

    def test_lone_surrogate_reaches_the_problem_log(self, isolated):
        """応答の JSON の断片には孤立サロゲートが混ざりうる。utf-8 では書けず、ValueError の一種になる。"""
        path = isolated / "p.log"
        with preflight._captured_logs(path) as capture:
            logging.getLogger("lab_lounge.x").warning("text=%s", "\ud83d")
        assert capture.wrote_problems is True
        assert "text=" in path.read_text(encoding="utf-8")


class TestEndpointOverrideRefusal:
    @pytest.mark.parametrize("line", [
        "OPENAI_BASE_URL=",  # SDK は "" を既定へ戻さず、呼び出しごとに接続エラーになる
        "ANTHROPIC_AUTH_TOKEN=",
        "openai_base_url=http://elsewhere.invalid",  # Windows では同じ変数になる
    ])
    def test_empty_or_lowercase_override_is_refused(self, isolated, line):
        """止めないと --static は緑のまま、本番の LLM の経路だけが無音で壊れる。"""
        (isolated / ".env").write_text(line + "\n", encoding="utf-8")
        with pytest.raises(SystemExit) as stopped:
            preflight.prepare_env("prod", isolated)
        assert stopped.value.code == preflight.EXIT_REFUSED


class TestUnreadableInputsAreRows:
    """壊れた入力でも FAIL の 1 行にする。例外で抜けると、原因のファイルが分からないまま残りの行も要約も失う。"""

    @pytest.mark.parametrize("marker", ["python_version ~= 'abc'", "'x' in extras"])
    def test_marker_that_cannot_be_evaluated_is_out_of_scope(self, marker):
        """packaging は形の誤りと評価の失敗で別々の例外を出す。どちらも対象外として扱う。"""
        assert preflight._marker_holds(marker) is False

    @pytest.mark.parametrize("raw", [
        b"not [ toml",
        b'[[package]]\nversion = "1"\n',  # name がない
        b"package = 1\n",  # 形の変更
        b"\xff\xfe",  # 文字コードの誤り
    ], ids=["broken-toml", "no-name", "not-a-list", "not-utf8"])
    def test_broken_lock_is_one_fail_row(self, isolated, raw):
        (isolated / "uv.lock").write_bytes(raw)
        report = preflight.Report()
        preflight._check_versions(report, isolated)
        assert _status(report, "deps/uv.lock").status == "FAIL"

    @pytest.mark.parametrize("raw", ["深海".encode("utf-16"), "深海".encode("cp932")], ids=["utf-16", "cp932"])
    def test_stream_context_not_in_utf8_is_a_fail_row(self, isolated, raw):
        """PowerShell 5 の > で書くと UTF-16 になる。本番の load_stream_context も同じ読み方で起動時に落ちる。"""
        path = isolated / "current.md"
        path.write_bytes(raw)
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        preflight._check_stream_context(report, isolated)
        check = _status(report, "stream_context")
        assert check.status == "FAIL"
        assert "current.md" in check.detail

    def test_stream_context_that_cannot_be_opened_is_a_fail_row(self, isolated):
        path = isolated / "current.md"
        path.write_text("x", encoding="utf-8")
        os.environ["L2_STREAM_CONTEXT_FILE"] = str(path)
        report = preflight.Report()
        with patch.object(Path, "read_text", side_effect=PermissionError("locked")):
            preflight._check_stream_context(report, isolated)
        assert _status(report, "stream_context").status == "FAIL"

    @pytest.mark.parametrize("error", [ValueError, SyntaxError, TypeError, MemoryError, RecursionError])
    def test_any_failure_of_literal_eval_means_not_a_list(self, error):
        """literal_eval の説明に挙がっている例外。読み上げのチャンク 1 つで点検全体を止めない。"""
        with patch("ast.literal_eval", side_effect=error("x")):
            assert preflight._is_list_repr("[1]") is False

    def test_chunk_that_literal_eval_chokes_on_is_read_as_plain_text(self):
        assert preflight._chunk_problems([("chisame", "[{[1]: 2}]")], "chisame") == []

    @pytest.mark.parametrize("result", [
        [],  # 戻り値の形そのものの変更
        {"messages": [SimpleNamespace(type="ai", tool_calls=[{"args": {}}])]},  # name のない tool_call
    ], ids=["not-a-dict", "tool-call-without-name"])
    def test_result_that_cannot_be_summarized_is_a_fail_row(self, isolated, capture, result):
        """SDK の更新で戻り値の形が変わっても、残りのキャラの行と要約を失わない。"""
        check, _ = _run_main_check(capture, result=result)
        assert check.status == "FAIL"

    def test_single_node_failure_of_a_non_required_character_does_not_fail_the_run(self, isolated, capture):
        """ツールが無効な構成でも、別枠のキャラの FAIL で配信を止めない。"""
        empty = SimpleNamespace(text="", input_tokens=0, output_tokens=0, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=empty):
            assert preflight.check_agent(report, capture, get_character("ruka"), None, required=False) is False
        assert _status(report, "main/ruka").status == "FAIL"
        assert report.exit_code() == preflight.EXIT_OK

    def test_lock_without_the_project_has_no_closure(self):
        """プロジェクト自身が見つからない lock は空で返し、呼び出し側が「読み取れません」の FAIL にする。"""
        assert preflight.locked_closure({"package": [{"name": "x", "version": "1.0"}]}) == {}

    def test_null_token_counts_read_as_zero(self):
        """数えられないだけで本応答を FAIL にしない。tokens は課金の目安で、合否の材料ではない。"""
        message = SimpleNamespace(type="ai", usage_metadata={"input_tokens": None, "output_tokens": None})
        summary = preflight._summarize_agent_result({"messages": [message]})
        assert (summary["input_tokens"], summary["output_tokens"]) == (0, 0)

    def test_embeddings_returning_none_is_a_fail_row(self, isolated, capture):
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        with patch("lab_lounge.retriever._embed", return_value=None):
            preflight.check_embeddings(report, capture)
        check = _status(report, "embeddings")
        assert check.status == "FAIL"
        assert "dim=0" in check.detail

    @pytest.mark.parametrize("error", [OSError("refused"), UnicodeError("label empty")], ids=["refused", "bad-host"])
    def test_http_failure_of_any_kind_means_not_reachable(self, error):
        """host の形が不正だと IDNA 変換で UnicodeError になる。抜けると前提サービスの一覧も出ない。"""
        with patch("urllib.request.urlopen", side_effect=error):
            assert preflight._http_ok("http://a..b:8100/health") is False


class TestReadsFlagsLikeProduction:
    """点検が本番より甘いと、本番がダミーで動く設定を緑にする。本番と同じ読み方であることを固定する。"""

    @pytest.mark.parametrize("value", [" true", "true ", " 1 "])
    def test_padded_value_is_false_on_both_sides(self, isolated, value):
        from lab_lounge.pipeline import _get_tts_mode

        os.environ["L2_USE_REAL_TTS"] = value
        assert preflight._is_truthy(value) is False
        assert _get_tts_mode()[0] is False

    def test_padded_value_warns_and_stops_the_real_mode_check(self, isolated):
        """.env の引用符の中や親プロセスの環境変数で届く。緑のままだと本番はダミー応答になる。"""
        os.environ.update({"L2_USE_REAL_LLM": " true", "L2_USE_REAL_TTS": "true"})
        flags = preflight.Report()
        preflight._check_flags(flags)
        assert _status(flags, "flag/L2_USE_REAL_LLM").status == "WARN"
        modes = preflight.Report()
        preflight._check_real_modes(modes, "prod", set(), paid=False)
        assert _status(modes, "mode/L2_USE_REAL_LLM").status == "FAIL"

    @pytest.mark.parametrize("name", ["L2_USE_C2_RETRIEVER", "L2_USE_C2_RECENT"])
    @pytest.mark.parametrize("value,warned", [("1", True), ("yes", True), ("true", False), ("TRUE", False), ("false", False)])
    def test_c2_flags_warn_on_spellings_graph_reads_as_false(self, isolated, name, value, warned):
        """graph.py はこの 2 つを "true" だけで読む。1 や yes は偽になり、記憶の経路だけが無音で消える。"""
        os.environ[name] = value
        report = preflight.Report()
        preflight._check_flags(report)
        assert _warned(report, f"flag/{name}") is warned

    @pytest.mark.parametrize("rag,use_c2,expected", [
        ("true", "true", True),
        ("false", "true", False),  # retrieve_memory が登録されないので、本番は C2 を呼ばない
        (None, "true", False),
        ("true", "false", False),
    ])
    def test_c2_counts_as_used_only_when_production_calls_it(self, isolated, rag, use_c2, expected):
        os.environ["L2_ENABLE_TOOLS"] = "true"
        if rag is not None:
            os.environ["L2_ENABLE_RAG"] = rag
        os.environ["L2_USE_C2_RETRIEVER"] = use_c2
        assert preflight._c2_retriever_enabled() is expected

    def test_c2_that_production_does_not_call_is_not_required(self, isolated):
        """止まっている C2 で --paid を止めたり FAIL にしたりすると、正常な構成で点検を通せない。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_USE_C2_RETRIEVER": "true"})
        with patch.object(preflight, "_http_ok", return_value=False) as ok:
            assert preflight.missing_services([], with_pipeline=False) == []
        ok.assert_not_called()
        report = preflight.Report()
        preflight.check_c2(report)
        assert _status(report, "c2").status == "SKIP"

    def test_disabled_filler_does_not_require_its_provider(self, isolated):
        """使わない経路のキーがないだけで --paid を止めない。--characters で絞ったときに起きる。"""
        os.environ.update({"L2_USE_LLM_ROUTER": "false", "L2_USE_HANDRAISE": "false", "L2_FILLER_ENABLED": "false"})
        character = SimpleNamespace(slug="x", filler_model="gemini-filler")
        with patch("lab_lounge.pipeline._get_llm_mode", return_value=(True, "openai", "gpt-main")):
            assert preflight._check_models(preflight.Report(), [character]) == {"openai"}

    def test_padded_router_answer_is_rejected(self):
        """本番の _route_by_llm は == で比べる。点検だけが strip すると、本番が捨てる答えを緑にする。"""
        assert preflight._validate_router_answer("mimi") is None
        assert preflight._validate_router_answer(" mimi\n") is not None

    @pytest.mark.parametrize("text,ok", [
        ('{"response": "こんにちは"}', True),
        ("プレーンな文", True),
        ('{"response": null}', False),  # 本番は str() して「None」と読み上げる
        ('{"response": ["a"]}', False),  # 本番は "['a']" と読み上げる
        ('{"response": 1}', False),
    ])
    def test_filler_response_must_be_text(self, text, ok):
        assert (preflight._validate_filler_answer(text) is None) is ok

    def test_tts_done_whose_text_is_not_a_string_is_a_fail_row(self, isolated, capture):
        """読み上げる本文を読めていない。str() で読むと、123 を本文として緑にしてしまう。"""
        check, _, _ = _run_pipeline_check(capture, events=[{"type": "tts.done", "payload": {"text": 123}}])
        assert check.status == "FAIL"

    def test_null_response_reaching_tts_is_empty(self, isolated, capture):
        """単一ノードの経路で {"response": null} が届くと、本番は「None」と読み上げる。"""
        events = [{"type": "tts.done", "payload": {"text": '{"response": null}'}}]
        check, _, _ = _run_pipeline_check(capture, events=events)
        assert check.status == "FAIL"

    def test_malformed_obs_url_is_a_fail_row(self, isolated, monkeypatch):
        """接続の失敗と違い、本番は起動できない。WARN のままだと緑で配信が始まらない。"""
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:abc"
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        check = _status(report, "obs")
        assert check.status == "FAIL"
        assert "abc" not in check.detail
        fake.ReqClient.assert_not_called()

    def test_production_really_stops_on_a_malformed_obs_url(self, isolated, monkeypatch):
        """上の FAIL の根拠。init_obs が no-op に直ったら、このテストが落ちるので FAIL を WARN へ下げる。"""
        from lab_lounge import obs

        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:abc"
        monkeypatch.setitem(sys.modules, "obsws_python", MagicMock())
        monkeypatch.setattr(obs, "_init_attempted", False)
        with pytest.raises(ValueError):
            obs.init_obs()


class TestEndpointValues:
    @pytest.mark.parametrize("name", ["REDIS_URL", "L2_C2_URL", "L2_TTS_VOICEVOX_URL", "L2_TTS_IRODORI_URL"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_value_does_not_fall_back_to_the_default(self, isolated, name, value):
        """本番も os.environ.get で読むので既定に戻らない。redis.from_url("") は ValueError になる。"""
        os.environ.update({name: value, "L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true", "L2_USE_C2_RETRIEVER": "true", "L2_USE_REAL_TTS": "true"})
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, f"url/{name}").status == "FAIL"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_readonly_url_means_unused(self, isolated, value):
        """既定が空で、本番も strip して無効と読むので、行を出さない。"""
        os.environ["L2_C2_URL_READONLY"] = value
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert not any(check.name == "url/L2_C2_URL_READONLY" for check in report.checks)

    def test_omitted_port_shows_the_scheme_default(self, isolated):
        """loopback:None と出ると、壊れた設定に見える。"""
        os.environ["REDIS_URL"] = "redis://127.0.0.1"
        report = preflight.Report()
        preflight._check_endpoints(report)
        assert _status(report, "url/REDIS_URL").detail == "loopback:6379"


class TestKeysUseOneRule:
    def test_missing_tavily_key_is_a_warning_in_dummy_mode(self, isolated):
        """本番はダミー起動なら Agent を通らず Tavily を呼ばない。赤くすると OSS 利用者のダミー起動を止める。"""
        os.environ["L2_ENABLE_TOOLS"] = "true"
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        assert _status(report, "key/tavily").status == "WARN"

    def test_inherited_tavily_key_is_a_warning(self, isolated):
        """Explorer から起動する本番ではキーがなくなる。他のキーと同じく WARN で知らせる。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "TAVILY_API_KEY": "dummy"})
        report = preflight.Report()
        preflight._check_keys(report, set(), {"OPENAI_API_KEY"})
        assert _status(report, "key/tavily").status == "WARN"


class TestServiceRowsDetails:
    @pytest.mark.parametrize("value", [None, "false"])
    def test_tavily_key_is_not_checked_when_tools_are_off(self, isolated, value):
        """使わないキーの行が FAIL で並ぶと配信前に止まり、WARN でも本物の警告が埋もれる。"""
        os.environ["L2_USE_REAL_LLM"] = "true"
        if value is not None:
            os.environ["L2_ENABLE_TOOLS"] = value
        report = preflight.Report()
        preflight._check_keys(report, set(), set())
        assert not any(check.name == "key/tavily" for check in report.checks)

    @pytest.mark.parametrize("found,usable", [("voicepeak.exe", True), (None, False)])
    def test_voicepeak_is_judged_by_its_executable(self, found, usable):
        """未知の provider と同じ扱いに落ちると、VOICEPEAK が入っていても前提サービスなしで止まる。"""
        with patch("shutil.which", return_value=found):
            assert preflight.tts_engine_status(SimpleNamespace(tts_provider="voicepeak")) == (usable, "voicepeak")

    def test_embeddings_row_shows_the_dimension(self, isolated, capture):
        """次元数はモデルの取り違えの手がかり。C2 の索引と食い違うと検索が壊れる。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})
        report = preflight.Report()
        with patch("lab_lounge.retriever._embed", return_value=[0.1, 0.2, 0.3]):
            preflight.check_embeddings(report, capture)
        check = _status(report, "embeddings")
        assert check.status == "PASS"
        assert "dim=3" in check.detail

    def test_failure_to_disconnect_from_obs_does_not_change_the_result(self, isolated, monkeypatch):
        """確かめ終わった後の切断の失敗は、立ち絵の状態と関係ない。例外で抜けると要約まで失う。"""
        os.environ["L2_OBS_WS_URL"] = "ws://127.0.0.1:1"
        fake = MagicMock()
        fake.ReqClient.return_value.get_group_scene_item_list.return_value = SimpleNamespace(
            scene_items=[{"sourceName": "mimi_neutral", "sceneItemId": 1}],
        )
        fake.ReqClient.return_value.disconnect.side_effect = OSError("closed")
        monkeypatch.setitem(sys.modules, "obsws_python", fake)
        report = preflight.Report()
        preflight._check_obs(report, [get_character("mimi")])
        assert _status(report, "obs").status == "PASS"

    def test_unknown_tts_provider_is_not_usable(self):
        """本番の synthesize は ValueError にする。誤記を PASS にすると、配信で初めて声が出ないと分かる。"""
        assert preflight.tts_engine_status(SimpleNamespace(tts_provider="voicvx")) == (False, "voicvx")

    def test_connection_error_in_the_tavily_error_key_hides_its_text(self, isolated, capture, capsys):
        """TavilySearch はどの例外も {"error": e} へ包む。接続の失敗の本文には host が入る。"""
        import requests

        _run_tavily(capture, returns={"error": requests.ConnectionError(f"host.invalid {BODY}")})
        out = capsys.readouterr().out
        assert "ConnectionError" in out
        assert "host.invalid" not in out
        assert "BODYSENTINEL" not in out

    def test_no_pipeline_does_not_count_the_tts_rows(self, isolated, capture):
        """--no-pipeline は TTS を前提にしないので、止まっていても終了コードに響かせない。"""
        mocks, _ = _run_paid_with_fakes(capture, with_pipeline=False)
        assert {call.kwargs["required"] for call in mocks["check_tts_engine"].call_args_list} == {False}

    def test_full_run_counts_the_tts_rows_of_required_characters(self, isolated, capture):
        mocks, _ = _run_paid_with_fakes(capture, with_pipeline=True)
        required = {call.args[1].slug: call.kwargs["required"] for call in mocks["check_tts_engine"].call_args_list}
        assert required["mimi"] is True
        assert required["ruka"] is False


class _Exited(BaseException):
    """os._exit の代役が投げる。本物は戻らないので、戻ると finally の後ろの動きを確かめられない。"""


class _BrokenPipe(io.StringIO):
    """閉じたパイプ（`| Select-Object -First 5` など）への書き込みと同じ失敗をする stdout。"""

    def write(self, _text):
        raise BrokenPipeError

    def flush(self):
        raise BrokenPipeError


class TestCliStrictness:
    @pytest.mark.parametrize("argv", [
        ("--paid", "--profile", "dev", "--all"),  # --allow-prod の省略として通ると、本番の拒否が外れる
        ("--pa", "--profile", "dev"),
        ("--static", "--prof", "dev"),
    ], ids=["all", "pa", "prof"])
    def test_abbreviated_options_are_usage_errors(self, isolated, argv):
        with pytest.raises(SystemExit) as stopped:
            _run_main(isolated, *argv)
        assert stopped.value.code == preflight.EXIT_USAGE

    def test_unknown_character_is_named_in_the_error(self, isolated, capsys):
        """「1 つもありません」と読めると、綴りの誤りに気づけない。"""
        with pytest.raises(SystemExit) as stopped:
            _run_main(isolated, "--static", "--profile", "dev", "--characters", "no-such-character")
        assert stopped.value.code == preflight.EXIT_USAGE
        assert "no-such-character" in capsys.readouterr().err

    def test_equals_form_is_not_an_abbreviation(self, isolated):
        code, _ = _run_main(isolated, "--static", "--profile=dev")
        assert code == preflight.EXIT_OK

    @pytest.mark.parametrize("value", [",", "", " ", " , "])
    def test_empty_character_list_is_a_usage_error(self, isolated, value):
        """測っていないのに緑を返さない。空の変数を渡した誤りを、全キャラへの課金へ広げもしない。"""
        with pytest.raises(SystemExit) as stopped:
            _run_main(isolated, "--static", "--profile", "dev", "--characters", value)
        assert stopped.value.code == preflight.EXIT_USAGE

    def test_help_on_a_cp932_stdout(self, isolated):
        """日本語環境のリダイレクト先は cp932 の strict で、ヘルプの em dash を書けない。"""
        raw = io.BytesIO()
        with patch.object(sys, "stdout", io.TextIOWrapper(raw, encoding="cp932", errors="strict")):
            exits = _run_entry(argv=("preflight", "--help"))
        assert exits == [preflight.EXIT_OK]

    def test_non_numeric_system_exit_says_it_stopped(self, isolated, capsys):
        """文言は値を含みうるので出さないが、何も出さずに 1 で終わると理由を読めない。"""
        _run_entry(main=MagicMock(side_effect=SystemExit("文言")))
        out = capsys.readouterr().out
        assert "SystemExit" in out
        assert "文言" not in out

    @pytest.mark.parametrize("stdout,outcome,code", [
        (_BrokenPipe(), {"return_value": 0}, 0),
        (_BrokenPipe(), {"side_effect": RuntimeError("x")}, preflight.EXIT_FAILED),
        (_BrokenPipe(), {"side_effect": KeyboardInterrupt()}, 130),
        (None, {"return_value": 0}, 0),  # pythonw には stdout がない
    ], ids=["flush-fails", "emit-fails", "interrupted", "no-stdout"])
    def test_output_failure_still_ends_through_os_exit(self, isolated, stdout, outcome, code):
        """普通の終了へ落ちると worker の join で止まり、その間も API を呼び続けて課金が増える。"""
        exits = []

        def fake_exit(value):
            exits.append(value)
            raise _Exited

        with patch.object(sys, "stdout", stdout), patch("os._exit", fake_exit), \
                patch.object(logging, "lastResort", logging.lastResort), \
                patch.object(preflight, "main", MagicMock(**outcome)), pytest.raises(_Exited):
            preflight._cli()
        assert exits == [code]


def _call_name_of(wake_word):
    return preflight._call_name(SimpleNamespace(wake_word=wake_word, display_name="表示名"))


def test_call_name_falls_back_to_the_display_name():
    """ruka には wake word がない。そのまま埋めると「None、」とモデルへ送ってしまう。"""
    assert _call_name_of("呼び名") == "呼び名"
    assert _call_name_of(None) == "表示名"


# ─── 初回の点検 ───────────────────────────────────────────────────


class TestFirstRun:
    """テンプレートを写しただけの環境と、本番相当に値を入れた環境を、点検の判定で不当に止めない。"""

    def _copy_examples(self, root: Path, *extra: str) -> None:
        env = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        (root / ".env").write_text("\n".join((env.rstrip("\n"), *extra)) + "\n", encoding="utf-8")
        (root / ".env.dev").write_text((_REPO_ROOT / ".env.dev.example").read_text(encoding="utf-8"), encoding="utf-8")

    def _rows(self, out: str, mark: str) -> list[str]:
        return [line for line in out.splitlines() if line.startswith(mark)]

    def test_dev_copy_of_the_templates_passes(self, isolated, capsys):
        self._copy_examples(isolated)
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_OK
        assert self._rows(out, "[FAIL") == []
        # テンプレートの値だけで出てよい WARN を固定する。増えると、毎回の警告に慣れて本物を見逃す。
        allowed = ("[WARN ] key/", "[WARN ] obs:", "[WARN ] deps/uv.lock:", "[WARN ] persona/", "[WARN ] flag/L2_ENABLE_RAG:")
        assert [line for line in self._rows(out, "[WARN") if not line.startswith(allowed)] == []

    def test_prod_copy_of_the_templates_stops_only_for_the_dummy_modes(self, isolated, capsys):
        self._copy_examples(isolated)
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "prod"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_FAILED
        assert {line.split(":")[0] for line in self._rows(out, "[FAIL")} == {
            "[FAIL ] mode/L2_USE_REAL_LLM", "[FAIL ] mode/L2_USE_REAL_TTS",
        }

    @pytest.mark.parametrize("files", [{".env": ".env.example"}, {}], ids=["env-only", "nothing-copied"])
    def test_partial_copy_passes_the_static_check(self, isolated, capsys, files):
        """README のとおりに置く前の状態でも、足した行で赤くならないこと。"""
        for name, source in files.items():
            (isolated / name).write_bytes((_REPO_ROOT / source).read_bytes())
        code = preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_OK
        for row in ("schema", "tts-config/readings", "filler-cache", "mode/L2_NO_PLAY"):
            assert f"] {row}" not in out
        # aruka の人格を含まないブランチでは WARN の行が出るので、FAIL だけを見る。
        assert "] persona/" not in out.replace("[WARN ] persona/", "")

    @pytest.mark.parametrize("profile", ["dev", "prod"])
    def test_templates_with_real_values_pass(self, isolated, voices, capsys, profile):
        """本番相当に有効化しても、数値や接続先の判定で赤くならない。"""
        self._copy_examples(
            isolated,
            "L2_USE_REAL_LLM=true", "L2_USE_REAL_TTS=true", "L2_ENABLE_TOOLS=true", "L2_ENABLE_RAG=true", "L2_USE_C2_RETRIEVER=true",
            "OPENAI_API_KEY=dummy-a", "ANTHROPIC_API_KEY=dummy-b", "GOOGLE_API_KEY=dummy-c", "TAVILY_API_KEY=dummy-d",
            f"L2_IRODORI_VOICES_JSON={voices}", f"L2_IRODORI_READINGS_JSON={isolated / 'no-readings.json'}",
            f"L2_TTS_VOICEPEAK_PATH={sys.executable}", f"L2_KB_PATH={_write_seed_index(isolated)}",
        )
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", profile], repo_root=isolated)
        out = capsys.readouterr().out
        assert self._rows(out, "[FAIL") == []
        assert code == preflight.EXIT_OK

    def test_third_party_can_start_from_octamaid(self, isolated, capsys):
        """irodori のアンカーを持たない人の道。voices.json がなくても OpenAI と VOICEVOX だけで緑になる。"""
        self._copy_examples(isolated, "L2_USE_REAL_LLM=true", "L2_USE_REAL_TTS=true", "OPENAI_API_KEY=dummy-a", "ANTHROPIC_API_KEY=dummy-b")
        with patch.object(preflight, "_check_imports"):
            code = preflight.main(["--static", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated)
        out = capsys.readouterr().out
        assert self._rows(out, "[FAIL") == []
        assert code == preflight.EXIT_OK

    def test_real_voicevox_on_a_new_machine_only_warns(self, isolated):
        """WAV も読み辞書もない新しい環境。VOICEVOX だけで実 TTS を始めても赤くしない。"""
        _write_real_env(isolated)
        code = preflight.main(["--static", "--profile", "dev", "--characters", "octamaid"], repo_root=isolated)
        assert code == preflight.EXIT_OK

    def test_real_irodori_without_a_dictionary_only_warns(self, isolated, voices, capsys):
        _write_real_env(isolated, L2_IRODORI_VOICES_JSON=str(voices))
        code = preflight.main(["--static", "--profile", "dev", "--characters", "mimi,chisame,sakura,octamaid"], repo_root=isolated)
        out = capsys.readouterr().out
        assert code == preflight.EXIT_OK
        assert "[WARN ] tts-config/readings" in out
        assert "[WARN ] filler-cache" in out

    def test_lone_clone_fails_and_the_readme_remedy_clears_it(self, isolated, monkeypatch, capsys):
        """README の前提を外して単独で clone した場合だけ FAIL になり、案内どおりに設定すれば通る。"""
        from lab_lounge import events

        schema = os.environ.pop("AIBYSS_SCHEMA_PATH")
        monkeypatch.setattr(events, "__file__", str(isolated / "lone" / "src" / "lab_lounge" / "events.py"))
        events._reset_validator_cache()
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_FAILED
        out = capsys.readouterr().out
        assert "[FAIL ] schema" in out and "AIBYSS_SCHEMA_PATH" in out and "aibyss-workspace" in out
        (isolated / ".env").write_text(f"AIBYSS_SCHEMA_PATH={schema}\nREDIS_STREAM_KEY=aibyss:events-dev\n", encoding="utf-8")
        assert preflight.main(["--static", "--profile", "dev"], repo_root=isolated) == preflight.EXIT_OK


# ─── 文書と公開物 ─────────────────────────────────────────────────
# 第三者は README だけを読んで動かす。表や出力例がコードから離れると、書いてあるとおりに動かして詰まる。
# コードを直したときに文書を直し忘れた箇所を、ここで捕まえる。

# 終了コードの一覧。130 は _cli が KeyboardInterrupt で返す値で、定数を持たない。
_ALL_EXIT_CODES = {
    preflight.EXIT_OK, preflight.EXIT_FAILED, preflight.EXIT_REFUSED,
    preflight.EXIT_MISSING_SERVICE, preflight.EXIT_USAGE, 130,
}

# 表の 1 行の形。[状態] 名前: 補足
_TABLE_ROW = re.compile(r"\[(\w+)(\*?) *\] (\S+): (.+)")


def _read_public(rel: str) -> str:
    """リポジトリの中のファイルを読む。read_text が改行をそろえるので、CRLF のファイルも行で比べられる。"""
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


# 公開する本文の置き場所と拡張子。data/ と logs/ は本文ではなく、キャラの資料 (yml) は本文の言葉として
# 内部の呼び名と同じ語を使うので見ない。
_PUBLIC_TEXT_ROOTS = ("src", "tests", "scripts", "sidecar")
_PUBLIC_TEXT_SUFFIXES = (".py", ".md", ".ps1", ".bat")
_PUBLIC_TEXT_TOP_FILES = ("README.md", "CHARACTERS.md", ".env.example", ".env.dev.example")


def _public_text_files() -> list[str]:
    """公開する本文 (コード、テスト、スクリプト、文書、env の雛形) の相対パスを返す。

    git の作業ツリーなら追跡中のファイルから、そうでなければ同じ置き場所の glob から選ぶ。

    WHY: 一覧を手で並べると、新しいファイルや、一覧にないファイルへの追記が漏れる (2026-11-15 に bus.py などへ
         計画の番号を書き、一覧にあった 1 ファイルの分しか赤くならなかった)。追跡外の手元のメモは公開されないので、
         git があるときは見ない。
    """
    if shutil.which("git") is not None and (_REPO_ROOT / ".git").exists():
        done = subprocess.run(
            ["git", "--no-optional-locks", "ls-files", "--", *_PUBLIC_TEXT_ROOTS, *_PUBLIC_TEXT_TOP_FILES],
            cwd=_REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=True, timeout=60,
        )
        tracked = done.stdout.splitlines()
    else:
        tracked = [
            path.relative_to(_REPO_ROOT).as_posix()
            for root in _PUBLIC_TEXT_ROOTS
            for path in (_REPO_ROOT / root).rglob("*")
            if path.is_file()
        ] + [name for name in _PUBLIC_TEXT_TOP_FILES if (_REPO_ROOT / name).is_file()]
    return sorted(
        rel for rel in tracked
        if rel in _PUBLIC_TEXT_TOP_FILES or (rel.endswith(_PUBLIC_TEXT_SUFFIXES) and "__pycache__" not in rel)
    )


def _readme_section() -> str:
    """README の preflight の節を返す。"""
    return _read_public("README.md").split("### 4. `preflight`")[1].split("\n---\n")[0]


def _readme_sample() -> list[str]:
    """README の「期待される出力」を行の一覧で返す。"""
    return _readme_section().split("```text")[1].split("```")[0].strip().splitlines()


def _sample_rows() -> list[re.Match[str]]:
    """出力例のうち表の行を、状態と名前に分けて返す。"""
    rows = [_TABLE_ROW.fullmatch(line) for line in _readme_sample() if line.startswith("[")]
    assert rows and all(rows), _readme_sample()
    return rows


def _run_paid_scene(root: Path, voices: Path, capsys) -> list[re.Match[str]]:
    """課金の入口をすべて偽物に替えて --paid を最後まで流し、実際に出た表の行を返す。"""
    from lab_lounge.pipeline import _get_llm_mode

    _write_real_env(root, L2_IRODORI_VOICES_JSON=str(voices))
    agent = MagicMock()
    agent.invoke.return_value = {
        "structured_response": SimpleNamespace(response="ok"),
        "messages": [SimpleNamespace(type="ai", usage_metadata={"input_tokens": 1, "output_tokens": 1},
                                     tool_calls=[], response_metadata={"finish_reason": "stop"})],
    }
    search = MagicMock()
    search.return_value.invoke.return_value = {"results": [{"title": "t"}]}

    def fake_pipeline(_text, **kwargs):
        # 本番の Agent は完了時にこのログを出し、点検はそれで構造化出力を確かめる。
        logging.getLogger("lab_lounge.graph").info("Agent 実行完了 [character=x]: source=structured_response")
        return SimpleNamespace(speaker=kwargs["speaker_hint"],
                               events=[{"type": "tts.done", "payload": {"text": '{"response": "ok"}'}}])

    def fake_collab(report, _capture, caller, target, _stream_id, _stream_context):
        report.add(f"collab/{caller.slug}->{target.slug}", "PASS", f"{_get_llm_mode(caller)[2]} 1.0s")

    with patch.object(preflight, "missing_services", return_value=[]), \
            patch.object(preflight, "tts_engine_status", return_value=(True, "fake")), \
            patch.object(preflight, "check_collab", autospec=True, side_effect=fake_collab), \
            patch("lab_lounge.router._call_router_llm", return_value="mimi"), \
            patch("lab_lounge.filler._call_filler_llm", return_value="こんにちは"), \
            patch("langchain_tavily.TavilySearch", search), \
            patch("lab_lounge.graph._build_agent_graph", return_value=agent), \
            patch("lab_lounge.pipeline.run_pipeline", side_effect=fake_pipeline):
        preflight.main(["--paid", "--profile", "dev"], repo_root=root)
    rows = [_TABLE_ROW.fullmatch(line) for line in capsys.readouterr().out.splitlines()]
    return [row for row in rows if row]


class TestReadmeMatchesTheCode:
    """引数、終了コード、コマンドの例をコードと突き合わせる。"""

    def test_every_option_is_documented(self):
        parser = preflight._build_parser()
        options = {s for a in parser._actions for s in a.option_strings if s.startswith("--")}
        documented = set(re.findall(r"^\| `(--[a-z-]+)", _readme_section(), flags=re.M))
        assert documented == options - {"--help", "--static", "--paid"}

    def test_every_exit_code_is_documented(self):
        documented = {int(code) for code in re.findall(r"^\| `(\d+)` \|", _readme_section(), flags=re.M)}
        assert documented == _ALL_EXIT_CODES

    def test_module_docstring_lists_the_same_exit_codes(self):
        block = preflight.__doc__.split("【終了コード】")[1].split("【")[0]
        assert {int(code) for code in re.findall(r"^\s*(\d+):", block, flags=re.M)} == _ALL_EXIT_CODES

    def test_unexpected_error_is_documented_with_exit_1(self):
        """想定外のエラーも 1 で返るので、要約なしの 1 が何かを表から読めること。"""
        row = next(line for line in _readme_section().splitlines() if line.startswith("| `1` |"))
        assert "想定外のエラー" in row

    def test_every_command_in_the_section_parses(self):
        """書いてあるコマンドをそのまま写して流す人がいる。引数の綴りの誤りは終了コード 64 で止まる。"""
        commands = re.findall(r"python -m lab_lounge\.preflight (.+)", _readme_section())
        assert len(commands) >= 3
        for command in commands:
            preflight._build_parser().parse_args(command.split())

    def test_required_characters_and_default_collab_are_the_real_ones(self):
        section = _readme_section()
        caller, target = preflight._COLLAB_PAIRS[0]
        assert " / ".join(preflight.REQUIRED_CHARACTERS) in section
        assert f"{caller} から {target}" in section

    def test_no_pipeline_help_mentions_collab(self):
        """--help と README で省くものが食い違わないこと。協働も省かれる。"""
        text = preflight._build_parser().format_help()
        assert "協働" in text.split("--no-pipeline")[-1].split("--collab")[0]


class TestReadmeSample:
    """「期待される出力」を、本物の表示の関数と実際の流れに通して比べる。"""

    def test_rows_are_what_report_prints(self, isolated, capsys):
        for row in _sample_rows():
            status, star, name, detail = row.groups()
            preflight.Report().add(name, status, detail, required=not star)
            assert capsys.readouterr().out.rstrip("\n") == row.group(0)

    def test_summary_is_what_report_prints(self, isolated, capsys):
        sample = _readme_sample()
        summary = next(line for line in sample if line.startswith("PASS "))
        passed, failed, warned, skipped, tokens_in, tokens_out = map(int, re.findall(r"\d+", summary))
        failures = [row for row in _sample_rows() if row.group(1) == "FAIL"]
        assert len(failures) == failed
        report = preflight.Report()
        report.input_tokens, report.output_tokens = tokens_in, tokens_out
        report.checks.extend(preflight.Check(row.group(3), "FAIL", "", not row.group(2)) for row in failures)
        for status, count in (("PASS", passed), ("WARN", warned), ("SKIP", skipped)):
            report.checks.extend(preflight.Check("x", status, "", True) for _ in range(count))
        report.print_summary()
        printed = capsys.readouterr().out.splitlines()
        assert summary in printed
        for line in sample:
            if line.startswith("  FAIL"):
                assert line in printed

    def test_counts_fit_the_rows_that_always_exist(self):
        """ruka が流れれば pipeline/ruka は必ず SKIP になる。SKIP 0 の例は本物の出力と食い違う。"""
        sample = "\n".join(_readme_sample())
        if "filler/ruka" in sample:
            assert not re.search(r"SKIP 0\b", sample)

    def test_rows_appear_in_a_real_run_in_the_same_order(self, isolated, voices, capsys):
        """抜粋の名前と順序が、実際の --paid の流れと同じであること。"""
        real = [row.group(3) for row in _run_paid_scene(isolated, voices, capsys)]
        wanted = [row.group(3) for row in _sample_rows()]
        remaining = iter(real)
        assert all(name in remaining for name in wanted), (wanted, real)

    def test_values_that_do_not_depend_on_the_machine_match_a_real_run(self, isolated, voices, capsys):
        """モデル名は env の既定から来ることがあり、思い込みで書くと実際の行と違う。"""
        real = _run_paid_scene(isolated, voices, capsys)
        by_name = {row.group(3): row for row in real}
        for row in _sample_rows():
            name = row.group(3)
            if name in ("key/openai", "model/mimi", "pipeline/ruka"):
                assert by_name[name].group(0) == row.group(0)
            elif name.startswith(("filler/", "main/", "collab/")):
                # 先頭の語がモデル名。FAIL の行はモデル名の直後に : が付く。
                assert by_name[name].group(4).split()[0].rstrip(":") == row.group(4).split()[0].rstrip(":"), name


class TestReadmeStatements:
    """手順と範囲の説明。読んだ人が詰まる箇所と、緑を過信する箇所。"""

    def test_prerequisites_come_before_the_tables(self):
        head = _readme_section().split("| 引数 |")[0]
        assert "uv sync --frozen --all-extras" in head
        assert ".env.dev" in head

    def test_version_check_really_looks_at_every_extra(self):
        """「全 extra を見る」の説明を判定から離さない。dev の pytest は配信に要らないが、閉包に入る。"""
        wanted = preflight.locked_closure(tomllib.loads(_read_public("uv.lock")))
        assert "pytest" in wanted

    def test_extras_section_does_not_contradict_the_check(self):
        """extra を選ぶ手順のままだと版の検査が FAIL になる。選ぶ手順の側からも案内する。"""
        section = _read_public("README.md").split("### optional extras")[1].split("### 環境変数")[0]
        assert "uv sync --frozen --all-extras" in section

    def test_env_dev_example_turns_the_stream_away_from_prod(self):
        """拒否の案内が指す雛形で、本当に本番から外れること。"""
        values = dotenv_values(_REPO_ROOT / ".env.dev.example")
        assert values["REDIS_STREAM_KEY"] != preflight.PROD_STREAM

    def test_first_small_run_is_documented(self):
        assert "--paid --profile dev --characters octamaid --no-pipeline --collab none" in _readme_section()

    def test_first_small_run_reaches_the_llm_without_tts(self, isolated):
        """書いてあるコマンドのとおりに流すと、TTS がなくても課金の手前で止まらず、octamaid だけが流れること。"""
        line = next(row for row in _readme_section().splitlines() if "python -m" in row and "--no-pipeline" in row)
        argv = line.split("lab_lounge.preflight ")[1].split()
        _write_real_env(isolated, L2_USE_REAL_TTS="false")
        with patch.object(preflight, "missing_services", return_value=[]), \
                patch.object(preflight, "run_paid") as run:
            code = preflight.main(argv, repo_root=isolated)
        assert code == preflight.EXIT_OK
        assert [character.slug for character in run.call_args.args[2]] == ["octamaid"]
        assert run.call_args.kwargs["with_pipeline"] is False

    def test_lists_what_is_not_verified(self):
        tail = _readme_section().split("**確かめないもの**")[1]
        for word in ("STT", "ウェイクワード", "VAD", "挙手", "C2", "V2"):
            assert word in tail

    def test_lists_the_paths_paid_does_not_run(self):
        from lab_lounge.router import _ROUTING_EXCLUDED_SLUGS

        text = _readme_section().split("**`--paid` が流さない経路**")[1].split("**`--paid` の副作用**")[0]
        for slug in _ROUTING_EXCLUDED_SLUGS:
            assert f"pipeline/{slug}" in text

    def test_entry_point_sentence_counts_the_check_tool(self):
        """直後に 4 つ目の節があるので、「3 つ」だけだと数が合わない。"""
        head = _read_public("README.md").split("## 実行方法")[1].split("### 1. ")[0]
        assert "点検" in head

    def test_directory_listing_names_every_module(self):
        listing = _read_public("README.md").split("## ディレクトリ構成")[1].split("```")[1]
        modules = {path.name for path in (_REPO_ROOT / "src" / "lab_lounge").glob("*.py")} - {"__init__.py"}
        assert "preflight.py" in modules
        assert not {name for name in modules if name not in listing}

    def test_no_test_counts_that_go_stale(self):
        """件数は増減のたびに古くなり、数字の嘘になる。"""
        text = _read_public("README.md")
        assert not re.search(r"\d+ passed", text)
        assert not re.search(r"\d+ テストファイル", text)


class TestSetupOfTheScripts:
    """PowerShell の版と irodori の初回の準備。"""

    @pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh がない環境")
    def test_sidecar_script_parses_in_powershell_7(self):
        """構文だけを確かめる。実行はしない。"""
        command = (
            "$e=$null;$t=$null;"
            "[void][System.Management.Automation.Language.Parser]::ParseFile($env:PS1_TO_PARSE,[ref]$t,[ref]$e);"
            "exit $e.Count"
        )
        env = {**os.environ, "PS1_TO_PARSE": str(_REPO_ROOT / "scripts" / "run_irodori_sidecar.ps1")}
        done = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
                              env=env, check=False, timeout=60)
        assert done.returncode == 0

    def test_readme_states_the_shell_in_the_prerequisites(self):
        assert "PowerShell 7" in _read_public("README.md").split("### インストール")[0]
        assert "PowerShell 7" in _read_public("sidecar/README.md").split("## 前提")[1].split("## 起動")[0]

    def test_first_download_is_mentioned_where_irodori_is_started(self):
        """既定は取得済みの weights だけを使うので、初回は必ず失敗する。"""
        assert "[switch]$AllowDownload" in _read_public("scripts/run_irodori_sidecar.ps1")
        readme = _read_public("README.md")
        assert "-AllowDownload" in readme.split("irodori-TTS（ニューラル音声クローン")[1].split("### 環境変数")[0]
        row = next(line for line in _readme_section().splitlines() if "終了コード 3 で `irodori`" in line)
        assert "-AllowDownload" in row

    def test_sidecar_readme_names_the_override_variables(self):
        """既定の場所は作者の環境のものなので、上書きの変数が読めないと第三者は動かせない。"""
        head = _read_public("sidecar/README.md").split("## 前提")[1].split("## 起動")[0]
        script = _read_public("scripts/run_irodori_sidecar.ps1")
        for name in ("L2_IRODORI_REPO", "L2_IRODORI_HF_HOME"):
            assert name in head
            assert f"$env:{name}" in script

    def test_sidecar_readme_has_no_stale_download_row(self):
        """offline が既定なので、毎回取り直すことはもう起きない。"""
        assert "毎回 DL" not in _read_public("sidecar/README.md")

    def test_sidecar_request_example_has_no_local_path(self):
        assert not re.search(r'"ref_wav": "[A-Z]:', _read_public("sidecar/README.md"))


class TestPublicFiles:
    """公開リポジトリへ手元の情報と内部の呼び名を混ぜない。"""

    @pytest.mark.parametrize("rel", [
        "README.md", "sidecar/README.md", "scripts/run_irodori_sidecar.ps1", "src/lab_lounge/preflight.py",
        ".env.example", ".env.dev.example",
        "scripts/run_dev.ps1", "scripts/run_prod.ps1", "scripts/run_common.ps1", "scripts/l2-start-prod.bat",
        "tests/conftest.py", "tests/test_isolation.py", "tests/test_run_scripts.py",
    ])
    def test_no_local_details(self, rel):
        text = _read_public(rel)
        assert not re.search(r"\b[A-Z]:[\\/]+(?:Users|repos)\b", text)
        assert not preflight._LAN_ADDRESS.search(text)
        assert not re.search(r"notion\.(?:so|site)|[\w.+-]+@(?:gmail|outlook|yahoo)\.", text)

    def test_no_local_details_in_the_tests(self):
        """テストはダミーのパスと LAN アドレスを使うので、実物のリポジトリのパス、Notion の URL、個人のメールだけを見る。"""
        text = _read_public("tests/test_preflight.py")
        assert not re.search(r"\b[A-Z]:[\\/]+repos[\\/]+aibyss|notion\.(?:so|site)|[\w.+-]+@(?:gmail|outlook|yahoo)\.", text)

    @pytest.mark.parametrize("rel", _public_text_files())
    def test_no_private_plan_names(self, rel):
        """非公開の計画の番号と内部の呼び名は、第三者が辿れない。"""
        text = _read_public(rel)
        # \b は日本語の直後で切れないので、前後を ASCII で見る。
        assert not re.search(r"(?<![A-Za-z0-9_])P\d{1,2}(?![A-Za-z0-9_])", text)
        assert "台" + "帳" not in text  # 内部の呼び名。続けて書くと、このテスト自身に当たる

    def test_gitignore_names_the_files_with_real_utterances(self):
        """問題ログを、別の節の *.log の 1 行だけに頼らない。消されても気づけないので。"""
        rules = set(_read_public(".gitignore").splitlines())
        assert {"logs/preflight/", "data/audio/*.wav", "data/audio/*.mp3"} <= rules

    @pytest.mark.skipif(
        shutil.which("git") is None or not (_REPO_ROOT / ".git").exists(), reason="git の作業ツリーでない環境",
    )
    @pytest.mark.parametrize("path", ["logs/preflight/preflight_x.log", "data/audio/x.wav", "data/audio/x.mp3"])
    def test_files_with_real_utterances_are_ignored_by_git(self, path, tmp_path):
        """規則の字面だけでなく、git が実際に無視することを確かめる。

        WHY: 実リポの作業ツリーへファイルを作らないよう、.gitignore の写しを置いた一時ディレクトリを作業ツリーに
             見立てて聞く。--no-optional-locks は index を書き換えないため。
        """
        shutil.copy(_REPO_ROOT / ".gitignore", tmp_path / ".gitignore")
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).touch()
        done = subprocess.run(
            ["git", "--no-optional-locks", f"--work-tree={tmp_path}", "status", "--porcelain", "--ignored", "--", path],
            cwd=_REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=False, timeout=60,
        )
        if done.returncode != 0:
            # 所有者の違う checkout（safe.directory）などで git が答えないときは、ここでは判定できない。
            pytest.skip("git が答えない環境")
        assert done.stdout.startswith("!! "), done.stdout


def _boom(*_args, **_kwargs):
    raise RuntimeError("boom")


def _raw_exception(report, capture, required):
    preflight._check_raw_llm(report, capture, "x", "m", _boom, required=required)


def _raw_validate(report, capture, required):
    preflight._check_raw_llm(report, capture, "x", "m", lambda: "a", validate=lambda _text: "bad", required=required)


def _filler_none(report, capture, required):
    with patch("lab_lounge.filler._call_filler_llm", return_value=None):
        preflight.check_filler(report, capture, get_character("mimi"), required=required)


def _main_exception(report, capture, required):
    os.environ["L2_ENABLE_TOOLS"] = "true"
    with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
            patch("lab_lounge.graph._invoke_agent_with_timeout", side_effect=_boom):
        preflight.check_agent(report, capture, get_character("mimi"), None, required=required)


def _main_problems(report, capture, required):
    os.environ["L2_ENABLE_TOOLS"] = "true"
    with patch("lab_lounge.graph._build_agent_graph", return_value=object()), \
            patch("lab_lounge.graph._invoke_agent_with_timeout", return_value={"messages": []}):
        preflight.check_agent(report, capture, get_character("mimi"), None, required=required)


def _main_no_agent(report, capture, required):
    os.environ["L2_ENABLE_TOOLS"] = "true"
    with patch("lab_lounge.graph._build_agent_graph", return_value=None):
        preflight.check_agent(report, capture, get_character("mimi"), None, required=required)


def _main_tools_off(report, capture, required):
    fake = SimpleNamespace(text='{"response": "切', input_tokens=1, output_tokens=1, finish_reason="stop")
    with patch("lab_lounge.llm.call_llm", return_value=fake):
        preflight.check_agent(report, capture, get_character("mimi"), None, required=required)


def _tts_row(report, _capture, required):
    with patch.object(preflight, "tts_engine_status", return_value=(False, "x")):
        preflight.check_tts_engine(report, get_character("mimi"), required=required)


def _pipeline_exception(report, capture, required):
    with patch("lab_lounge.pipeline.run_pipeline", side_effect=_boom):
        preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=required)


def _pipeline_problems(report, capture, required):
    with patch("lab_lounge.pipeline.run_pipeline", return_value=SimpleNamespace(events=[], speaker="mimi")):
        preflight.check_pipeline(report, capture, get_character("mimi"), "s", None, required=required)


def _c2_unreachable(report, _capture, required):
    with patch("urllib.request.urlopen", side_effect=OSError("x")):
        preflight._check_c2_endpoint(report, "c2", "http://127.0.0.1:1", required=required)


def _c2_shape(report, _capture, required):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"[]"
    with patch("urllib.request.urlopen", return_value=response):
        preflight._check_c2_endpoint(report, "c2", "http://127.0.0.1:1", required=required)


_REQUIRED_ROW_CASES = (
    _raw_exception, _raw_validate, _filler_none, _main_exception, _main_problems, _main_no_agent, _main_tools_off,
    _tts_row, _pipeline_exception, _pipeline_problems, _c2_unreachable, _c2_shape,
)


class TestRequiredRowsDecideTheExitCode:
    """必須の行の FAIL だけが終了コードを 1 にする。

    WHY: 行の状態だけを見ていると、required の渡し忘れで `[FAIL*]` の「判定: 合格」が通っても緑のまま。
    """

    @pytest.mark.parametrize("case", _REQUIRED_ROW_CASES, ids=lambda case: case.__name__)
    def test_a_required_row_stops_the_run(self, isolated, capture, case):
        report = preflight.Report()
        case(report, capture, True)
        assert report.exit_code() == preflight.EXIT_FAILED

    @pytest.mark.parametrize("case", _REQUIRED_ROW_CASES, ids=lambda case: case.__name__)
    def test_a_non_required_row_does_not(self, isolated, capture, case):
        report = preflight.Report()
        case(report, capture, False)
        assert report.exit_code() == preflight.EXIT_OK

    @pytest.mark.parametrize("name,env,slug", [
        ("L2_C2_URL", _C2_ON, "mimi"),
        ("L2_C2_URL_READONLY", _C2_ON, "mimi"),
        ("L2_TTS_VOICEVOX_URL", _REAL_TTS, "octamaid"),
        ("L2_TTS_IRODORI_URL", _REAL_TTS, "mimi"),
    ])
    def test_a_broken_url_on_a_live_path_stops_the_run(self, isolated, name, env, slug):
        os.environ.update({name: "not-a-url", **env})
        report = preflight.Report()
        preflight._check_endpoints(report, [get_character(slug)])
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_a_missing_reference_wav_stops_the_run(self, isolated):
        path = isolated / "voices.json"
        path.write_text(json.dumps({"voices": {"mimi": {"caption": "c", "ref_wav": "none.wav"}}}), encoding="utf-8")
        os.environ.update({"L2_IRODORI_VOICES_JSON": str(path), **_REAL_TTS})
        report = preflight.Report()
        preflight._check_tts_config(report, [get_character("mimi")])
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_a_missing_voicepeak_stops_the_run_for_a_required_character(self, isolated):
        os.environ.update({"L2_TTS_VOICEPEAK_PATH": str(isolated / "no-such"), **_REAL_TTS})
        character = dataclasses.replace(get_character("mimi"), tts_provider="voicepeak")
        report = preflight.Report()
        preflight._check_tts_config(report, [character])
        assert report.exit_code() == preflight.EXIT_FAILED

    @pytest.mark.parametrize("url,status,code", [
        ("redis://", "WARN", 0),
        ("redis:///0", "WARN", 0),
        ("unix:///tmp/x.sock", "INFO", 0),
        ("http://127.0.0.1:1", "FAIL", 1),
    ])
    def test_redis_url_has_one_row_and_the_right_exit_code(self, isolated, monkeypatch, url, status, code):
        """_status は同じ名前の最初の行しか見ないので、2 行目の FAIL を見逃さないよう全部の行を比べる。"""
        monkeypatch.setattr(socket, "AF_UNIX", 1, raising=False)
        os.environ["REDIS_URL"] = url
        report = preflight.Report()
        preflight._check_endpoints(report, [])
        assert [check.status for check in report.checks if check.name == "url/REDIS_URL"] == [status]
        assert report.exit_code() == code


class TestDeadlineDoesNotWait:
    def test_deadline_returns_without_waiting_for_the_worker(self):
        """相手が応答を止めたままでも戻る。worker を待つ実装だと、打ち切りが丸ごと無効になる。"""
        release = threading.Event()
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="超過"):
                preflight._with_deadline(lambda: release.wait(30), 0.2)
            assert time.monotonic() - started < 5
        finally:
            release.set()


class TestValueRows:
    """読めない値と、読める値の境目を両側から固定する。"""

    @pytest.mark.parametrize("name", ["L2_LLM_MAX_RETRIES", "L2_RAG_TOP_K", "L2_RECORD_SECONDS"])
    def test_blank_number_is_fail(self, isolated, name):
        """本番は int("") / float("") で落ちる。空を許すのは L2_TTS_IRODORI_SEED だけ。"""
        os.environ[name] = ""
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert report.exit_code() == preflight.EXIT_FAILED

    @pytest.mark.parametrize("on", ["L2_USE_LLM_ROUTER", "L2_USE_INTENT_GATE", "L2_USE_HANDRAISE", "L2_FILLER_ENABLED"])
    def test_router_timeout_is_live_when_any_raw_route_is_on(self, isolated, on):
        os.environ.update({**_RAW_LLM_OFF, on: "true", "L2_ROUTER_LLM_TIMEOUT_SEC": _MARK})
        report = preflight.Report()
        preflight._check_numeric_env(report)
        assert report.exit_code() == preflight.EXIT_FAILED

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "none"])
    def test_documented_reasoning_efforts_have_no_row(self, isolated, effort):
        os.environ["L2_LLM_REASONING_EFFORT"] = effort
        report = preflight.Report()
        preflight._check_silently_replaced_values(report)
        assert report.checks == []

    def test_blank_tokens_mean_no_limit_and_have_no_row(self, isolated):
        os.environ["L2_LLM_MAX_OUTPUT_TOKENS"] = "   "
        report = preflight.Report()
        preflight._check_silently_replaced_values(report)
        assert report.checks == []

    def test_debug_artifacts_false_has_no_row(self, isolated):
        report = preflight.Report()
        preflight._report_silenced_outputs(report, {"L2_DEBUG_ARTIFACTS": "false"})
        assert report.checks == []

    def test_value_less_line_in_the_env_file_is_not_a_failure(self, isolated):
        report = preflight.Report()
        preflight._check_env_quoting(report, {"NO_SUCH_VALUE_LESS_NAME"})
        assert report.checks == []

    @pytest.mark.parametrize("value", ["a\ab", "a\bb", "a\fb", "a\nb", "a\rb", "a\tb", "a\vb"])
    def test_every_control_char_python_dotenv_makes_is_reported(self, isolated, value):
        os.environ["L2_X"] = value
        report = preflight.Report()
        preflight._check_env_quoting(report, {"L2_X"})
        assert report.checks[0].status == "WARN"

    def test_empty_exception_is_described(self, isolated):
        assert preflight._describe_exc(ValueError()) == "ValueError"

    def test_audit_keeps_the_last_output_and_plain_text(self, isolated):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        audit = preflight._AgentAudit()
        for text in ("FIRST", "LAST"):
            audit.count(LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]]))
        assert "LAST" in audit.last_output

    def test_failure_before_any_llm_call_writes_no_line(self, tmp_path):
        """呼ぶ前に落ちたときは、問題ログへ空の「最後の応答」を残さない。"""
        log = tmp_path / "problems.log"
        report = preflight.Report()
        assert preflight._AgentAudit().record_failure(report, preflight._LogCapture(log), "main/x") == ""
        assert not log.exists()

    def test_verdict_without_a_report_is_one_line(self, capsys):
        """点検を流す前に止まったときは、流していない経路を調べる相手がない。"""
        preflight._emit_verdict(preflight.EXIT_OK)
        assert capsys.readouterr().out.strip() == "判定: 合格 (exit 0)"

    def test_list_repr_with_surrounding_space_is_found(self):
        """literal_eval は先頭の空白を IndentationError にするので、落としてから読む。"""
        assert preflight._is_list_repr("  [1]\n") is True

    def test_missing_key_row_names_every_accepted_name(self, isolated):
        """どちらの名前でもよいと分かるように、受け付ける名前を全部出す。"""
        report = preflight.Report()
        preflight._check_keys(report, {"google"}, set())
        assert _status(report, "key/google").detail == "unset (GOOGLE_API_KEY / GEMINI_API_KEY)"

    def test_redis_error_from_the_url_parser_is_one_fail_row(self, isolated):
        """redis-py はクエリの組み合わせで RedisError も出す。例外のままだと残りの接続先の行を失う。"""
        import redis

        report = preflight.Report()
        with patch.object(redis.ConnectionPool, "from_url", side_effect=redis.RedisError(_MARK)):
            preflight._check_endpoints(report, [])
        row = _status(report, "url/REDIS_URL")
        assert row.status == "FAIL"
        assert _MARK not in row.detail

    def test_connection_error_of_tavily_is_not_read_as_a_status(self, isolated, capture):
        """接続の失敗の "Error 111 ..." を HTTP の status と読まない。"""
        pytest.importorskip("langchain_tavily")
        os.environ.update({"L2_ENABLE_TOOLS": "true", "TAVILY_API_KEY": "dummy"})
        report = preflight.Report()
        with patch("langchain_tavily.TavilySearch") as search:
            search.return_value.invoke.return_value = {"error": ConnectionError("Error 111 connecting")}
            preflight.check_tavily(report, capture)
        row = _status(report, "tavily")
        assert row.status == "FAIL"
        assert "status=" not in row.detail
        assert row.detail.startswith("ConnectionError")


class TestNewRowsHideValues:
    """新しい経路の行にも、本文や値を出さない。"""

    def test_system_exit_text_is_not_in_the_verdict(self, isolated, capsys):
        with patch.object(preflight, "_run_checks", side_effect=SystemExit(f"{_MARK} secret")), pytest.raises(SystemExit):
            preflight.main(["--static", "--profile", "dev"], repo_root=isolated)
        assert _MARK not in capsys.readouterr().out

    def test_single_node_row_hides_the_spoken_text(self, isolated, capture):
        fake = SimpleNamespace(text='{"response": "' + _MARK + '切', input_tokens=1, output_tokens=1, finish_reason="stop")
        report = preflight.Report()
        with patch("lab_lounge.llm.call_llm", return_value=fake):
            preflight.check_agent(report, capture, get_character("mimi"), None, required=True)
        assert _MARK not in report.checks[0].detail

    def test_persona_row_hides_the_path_of_an_oserror(self, isolated):
        from lab_lounge import characters

        report = preflight.Report()
        with patch.object(characters, "load_system_prompt", side_effect=PermissionError(13, "denied", f"C:/{_MARK}/system.txt")):
            preflight._check_personas(report, [get_character("mimi")])
        assert report.checks[0].status == "FAIL"
        assert _MARK not in report.checks[0].detail

    @pytest.mark.parametrize("arg", [{"excerpt": f"{_MARK}短い発話"}, f"{_MARK}短い発話"], ids=["dict", "str"])
    def test_memory_row_hides_a_short_excerpt_in_the_parse_failure_log(self, isolated, capture, capsys, arg):
        """C2 の応答の形が変わると、短い本文がログの引数のまま memory の行へ出る。長さでは見分けられない。"""
        os.environ.update({"L2_ENABLE_TOOLS": "true", "L2_ENABLE_RAG": "true"})

        def impl(_query):
            logging.getLogger("lab_lounge.retriever").warning(
                "C2Retriever: レスポンス項目のパース失敗: %s (r=%s)", KeyError("event_id"), arg,
            )
            return "[1] (source: x, score: 0.1)\ny"

        report = preflight.Report()
        with patch("lab_lounge.mcp_servers.retrieve_memory._retrieve_memory_impl", side_effect=impl), \
                patch.object(logging.getLogger(), "handlers", [capture]), \
                patch.object(logging.getLogger("lab_lounge.retriever"), "level", logging.INFO):
            preflight.check_memory(report, capture, "s")
        row = _status(report, "memory")
        assert row.status == "WARN"
        assert _MARK not in row.detail
        assert _MARK not in capsys.readouterr().out

    @pytest.mark.parametrize("arg,shown", [
        ("gpt-5.5", "gpt-5.5"),
        (404, "404"),
        ("a" * 40, "a" * 40),
        ("a" * 41, "<41 文字>"),
        ("a b", "<3 文字>"),
    ])
    def test_short_ascii_and_numbers_stay_readable(self, arg, shown):
        """モデル名や status は理由として読める形で残す。長い値と空白のある文字列は発話とみなして長さだけにする。"""
        record = logging.LogRecord("lab_lounge.x", logging.WARNING, "f", 1, "理由 %s", (arg,), None)
        assert preflight._safe_log_message(record) == f"理由 {shown}"

    @pytest.mark.parametrize("text,shown", [
        ("接続できません (http://box:1/x): refused", "接続できません (<URL>): refused"),
        ("rediss+sentinel://box/0 に届きません", "<URL> に届きません"),
    ])
    def test_url_in_a_local_error_is_cut_at_its_end(self, isolated, text, shown):
        """URL の後ろの理由は読めるまま残す。長い scheme の URL も host ごと伏せる。"""
        assert preflight._describe_exc(RuntimeError(text)) == f"RuntimeError: {shown}"


class TestBrokenLookingInputs:
    """見た目は読めるが、本番では壊れる入力。"""

    @pytest.mark.parametrize("body", [
        {"global": {"JSON": 123}},
        {"characters": {"mimi": {"JSON": 1}}},
        {"global": {"JSON": None}},
    ])
    def test_reading_that_is_not_text_is_fail(self, isolated, body):
        """本番は置換の時点で TypeError（null は黙って語を消す）。空文字で流すと置換の関数を通らない。"""
        path = isolated / "readings.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        os.environ.update({"L2_IRODORI_READINGS_JSON": str(path), **_REAL_TTS})
        report = preflight.Report()
        preflight._check_readings(report, [get_character("mimi")])
        assert report.exit_code() == preflight.EXIT_FAILED

    @pytest.mark.parametrize("data", [b"\xef\xbb\xbf", b"\xef\xbb\xbf \r\n"], ids=["bom", "bom-crlf"])
    def test_persona_with_only_a_bom_is_empty(self, isolated, data):
        """PowerShell 5 の Out-File -Encoding utf8 は空でも BOM を書く。strip は BOM を落とさない。"""
        from lab_lounge import characters

        prompts = isolated / "prompts"
        prompts.mkdir()
        character = get_character("mimi")
        (prompts / character.system_prompt_file).write_bytes(data)
        report = preflight.Report()
        with patch.object(characters, "_PROMPTS_DIR", prompts):
            preflight._check_personas(report, [character])
        assert report.exit_code() == preflight.EXIT_FAILED

    def test_missing_persona_and_sample_row_says_so(self, isolated):
        """読めないのではなく、ないと分かれば置くだけで済む。"""
        from lab_lounge import characters

        report = preflight.Report()
        with patch.object(characters, "load_system_prompt", side_effect=FileNotFoundError(2, "x")):
            preflight._check_personas(report, [get_character("mimi")])
        assert report.checks[0].status == "FAIL"
        assert "もサンプルもありません" in report.checks[0].detail

    @pytest.mark.parametrize("body", [[], None, 0, ""], ids=["list", "null", "zero", "blank"])
    def test_empty_readings_of_any_type_are_the_same_as_no_readings(self, isolated, body):
        """本番は偽の値を「辞書なし」と読むので、形の誤りとして FAIL にしない。"""
        path = isolated / "readings.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        os.environ.update({"L2_IRODORI_READINGS_JSON": str(path), **_REAL_TTS})
        report = preflight.Report()
        preflight._check_readings(report, [get_character("mimi")])
        assert [(check.name, check.status) for check in report.checks] == [("tts-config/readings", "WARN")]
