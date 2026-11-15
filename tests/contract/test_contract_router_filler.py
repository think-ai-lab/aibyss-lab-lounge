"""
router と filler の raw SDK 呼び出しの契約テスト

``_call_router_llm``（発話の宛先の判定）と ``_call_filler_llm``（待ち時間の独り言）は、LangChain を通さず
各社の SDK を直接呼ぶ。どちらも例外をすべて None に変えるので、SDK の更新で壊れても配信は止まらず、
判定がパターンマッチへ、フィラーが定型文へ黙って落ちる。ここでは本物の SDK を偽の LLM サーバへ向けて、
次の 2 種類を確かめる。

- 挙動のテスト: 今の正常な結果が出る。SDK の更新で壊れたら赤くなる
- 方針のテスト: 送ってはいけないものを送らない、壊れた応答で黙らない。直すまで xfail(strict) にする。
  reason には finding ID と直す中身を書く。直した commit で XPASS になって落ちるので、印を外し忘れない

1 つのテストで router と filler の両方を呼ぶ。WHY: 2 つは同じ形の分岐（モデル名でプロバイダを選び、
同じ読み方で本文を取り出す）で、SDK の更新は両方に同時に効く。どちらが外れたかは assert の文面で分かる。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pytest

from lab_lounge.filler import _call_filler_llm
from lab_lounge.router import _call_router_llm

# プロバイダごとの、呼ぶモデルと、要求から出力の上限を読む方法。
# モデルは今のキャラの filler_model と router の既定（CHARACTERS.md / router.py）に合わせる。
_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "model": "claude-haiku-4-5",
        "max_out": lambda body: body["max_tokens"],
    },
    "openai": {
        "model": "gpt-5.4-mini",
        "max_out": lambda body: body["max_completion_tokens"],
    },
    "google": {
        "model": "gemini-3.1-flash-lite",
        # google-genai は設定を generationConfig にまとめ、名前を camelCase にして送る。
        "max_out": lambda body: body["generationConfig"]["maxOutputTokens"],
    },
}

_LOGGERS = {"router": "lab_lounge.router", "filler": "lab_lounge.filler"}


def _call_both(model: str) -> dict[str, str | None]:
    """router と filler を、同じモデルで 1 回ずつ呼ぶ。"""
    return {
        "router": _call_router_llm(model, "宛先を判定してください。", "ミミちゃん、こんにちは"),
        "filler": _call_filler_llm(model, "独り言を 1 文で。", "ミミちゃん、こんにちは"),
    }


def _warnings_of(caplog: pytest.LogCaptureFixture, which: str) -> list[str]:
    """router か filler のロガーが出した、WARNING 以上のログの文面を返す。"""
    return [r.getMessage() for r in caplog.records if r.name == _LOGGERS[which] and r.levelno >= logging.WARNING]


def _assert_silent_none_is_explained(results: dict, caplog: pytest.LogCaptureFixture, reason: str) -> None:
    """両方が None を返し、それぞれが理由（stop_reason / finish_reason）をログに出したことを確かめる。

    WHY: None 自体は今の契約（呼び出し元が定型文へ落とす）。黙っていると、配信の後でログを読んでも、
         上限切れか拒否か SDK の故障かを分けられない。
    """
    for which, value in results.items():
        assert value is None, f"{which}: 本文の無い応答で None を返さなかった: {value!r}"
        logs = _warnings_of(caplog, which)
        assert any(reason in line.lower() for line in logs), f"{which}: ログに {reason} が無い: {logs}"


# ─── 挙動のテスト ─────────────────────────────────────────────────


@pytest.mark.parametrize("provider", list(_PROVIDERS))
def test_normal_reply_is_returned(fake_llm, provider):
    """正常な応答で、router は小文字の答えを、filler は前後の空白を除いた本文を返す。

    WHY 大文字と空白を混ぜる: router の後段は slug と完全一致で比べるので、strip と lower を SDK の
    応答のどこに掛けるか（本文の取り出し方）まで、この結果で確かめられる。
    Anthropic のこのテストは、SDK が temperature を受け付けなくなったとき（CL-01）に赤くなる要でもある。
    """
    spec = _PROVIDERS[provider]
    fake_llm.scene(provider, "ok", text="  Mimi\n")

    results = _call_both(spec["model"])

    assert results["router"] == "mimi", f"router: {results['router']!r}"
    assert results["filler"] == "Mimi", f"filler: {results['filler']!r}"
    sent = fake_llm.requests_for(provider)
    assert [r.model for r in sent] == [spec["model"]] * 2, [(r.provider, r.path) for r in fake_llm.requests]
    # 上限は L2 が決めている契約（router は答えの 1 語、filler は 1 文）。SDK が名前を変えても届くこと。
    max_out: Callable[[dict], int] = spec["max_out"]
    assert [max_out(r.body) for r in sent] == [20, 150], "router, filler の順の出力の上限"


def test_anthropic_http_400_returns_none_with_log(fake_llm, caplog):
    """400 の応答で、両方とも例外を外へ出さず None を返し、失敗をログに残す。"""
    fake_llm.scene("anthropic", "http_error", status=400, message="fake bad request")

    results = _call_both(_PROVIDERS["anthropic"]["model"])

    for which, value in results.items():
        assert value is None, f"{which}: {value!r}"
        logs = _warnings_of(caplog, which)
        assert any("fake bad request" in line for line in logs), f"{which}: 失敗のログが無い: {logs}"


# ─── 方針のテスト ─────────────────────────────────────────────────


@pytest.mark.xfail(reason="CL-04: content[0] だけを読むので、先頭が thinking だと本文を取れない。type が text のブロックを探して読む")
def test_anthropic_thinking_first_reads_the_text_block(fake_llm):
    """先頭が thinking でも、本文のブロックを読んで返す。"""
    fake_llm.scene("anthropic", "thinking_first", text="  Mimi\n")

    results = _call_both(_PROVIDERS["anthropic"]["model"])

    assert results["router"] == "mimi", f"router: {results['router']!r}"
    assert results["filler"] == "Mimi", f"filler: {results['filler']!r}"


@pytest.mark.xfail(reason="CL-04: 本文の無い応答を AttributeError で None に変え、stop_reason を捨てる。None と stop_reason をログに出す")
def test_anthropic_thinking_only_max_tokens_logs_stop_reason(fake_llm, caplog):
    """thinking だけで出力の上限に達したら、None を返し、stop_reason（max_tokens）をログに出す。"""
    fake_llm.scene("anthropic", "thinking_only_max_tokens")

    _assert_silent_none_is_explained(_call_both(_PROVIDERS["anthropic"]["model"]), caplog, "max_tokens")


@pytest.mark.xfail(reason="CL-06: content が空の拒否を IndexError で None に変え、refusal を捨てる。None と stop_reason をログに出す")
def test_anthropic_refusal_logs_stop_reason(fake_llm, caplog):
    """拒否（content が空）なら、None を返し、stop_reason（refusal）をログに出す。"""
    fake_llm.scene("anthropic", "refusal")

    _assert_silent_none_is_explained(_call_both(_PROVIDERS["anthropic"]["model"]), caplog, "refusal")


@pytest.mark.xfail(reason="CL-01: Anthropic へ temperature を送っている。新しい API と SDK は受け付けないので、送らない")
def test_anthropic_does_not_send_temperature(fake_llm):
    """Anthropic への要求に temperature を入れない。"""
    results = _call_both(_PROVIDERS["anthropic"]["model"])

    sent = fake_llm.requests_for("anthropic")
    assert len(sent) == 2, f"要求の数: {[r.path for r in fake_llm.requests]} / 結果: {results}"
    for which, req in zip(_LOGGERS, sent):
        assert "temperature" not in req.body, f"{which}: temperature={req.body['temperature']!r}"


@pytest.mark.xfail(reason="OAI-05: 本文が空で length の応答を、router は空文字で返し、filler は黙って None にする。None と finish_reason をログに出す")
def test_openai_empty_length_logs_finish_reason(fake_llm, caplog):
    """推論で上限を使い切って本文が空なら、None を返し、finish_reason（length）をログに出す。"""
    fake_llm.scene("openai", "empty_length")

    _assert_silent_none_is_explained(_call_both(_PROVIDERS["openai"]["model"]), caplog, "length")


@pytest.mark.xfail(reason="GEM-01, GEM-13: resp.text が None の応答を AttributeError で None に変え、finishReason を捨てる。None と finishReason をログに出す")
def test_google_max_tokens_without_text_logs_finish_reason(fake_llm, caplog):
    """上限に達して本文が無ければ、None を返し、finishReason（MAX_TOKENS）をログに出す。"""
    fake_llm.scene("google", "max_tokens_no_text")

    _assert_silent_none_is_explained(_call_both(_PROVIDERS["google"]["model"]), caplog, "max_tokens")
