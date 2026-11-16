"""
支出の記録（spend.UsageMeter）の契約テスト

UsageMeter は、有料の計測の間だけ httpx の Client.send を包み、3 社の API の応答から usage を読んで
支出の記録へ足す。ここでは本物の SDK（raw の 3 社と LangChain の 3 社の adapter）を偽の LLM サーバへ向け、
次のことを確かめる。

- 6 通りの呼び出しのどれでも、usage が拾えて金額になる（SDK を上げて黙って拾えなくなったら赤くなる）
- 上限と価格表にないモデルは、送る前に止まる。止める例外は、L2 の ``except Exception`` に飲まれない
- 支出の記録に、要求と応答の本文が残らない

本文には目印の文字列（_SENTINEL）だけを使う。実発話は使わない。
"""

from __future__ import annotations

import json

import pytest

from lab_lounge import spend
from lab_lounge.filler import _call_filler_llm
from lab_lounge.llm import call_llm
from lab_lounge.router import _call_router_llm

# 本文に流す目印。記録と画面にこの文字列が出たら、本文が漏れている。
_SENTINEL = "SENTINEL-spend-7f3a"

_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "google": "gemini-3.1-flash-lite",
}

# 偽サーバの usage（fake_llm._USAGE_IN / _USAGE_OUT と同じ値）。
_IN, _OUT = 12, 3


def _expected_usd(model: str, calls: int) -> float:
    return calls * spend.cost_usd(spend.PRICES[model], spend.Usage(input=_IN, output=_OUT))


def _assert_no_text(directory) -> None:
    """支出の記録のファイルに、目印の文字列が無い。"""
    for path in directory.iterdir():
        assert _SENTINEL not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize("provider", list(_MODELS))
def test_raw_sdk_usage_is_counted(fake_llm, tmp_path, provider):
    """router と filler（raw の SDK）の 1 回ずつが、usage の金額で記録される。

    WHY router と filler: どちらも本文しか返さないので、tokens は HTTP の層でしか拾えない。
    """
    model = _MODELS[provider]
    fake_llm.scene(provider, "ok", text=_SENTINEL)
    ledger = spend.SpendLedger(tmp_path)

    with spend.UsageMeter(ledger, "contract", max_usd=1.0) as meter:
        assert _call_router_llm(model, f"判定 {_SENTINEL}", _SENTINEL) == _SENTINEL.lower()
        assert _call_filler_llm(model, f"独り言 {_SENTINEL}", _SENTINEL) == _SENTINEL
        meter.check_capture(2)

    assert (meter.stats.sent, meter.stats.counted, meter.stats.uncounted) == (2, 2, 0)
    row = ledger.data["by_model"][model]
    assert (row["calls"], row["input"], row["output"]) == (2, 2 * _IN, 2 * _OUT)
    assert ledger.total_usd == pytest.approx(_expected_usd(model, 2))
    _assert_no_text(tmp_path)


@pytest.mark.parametrize("provider", list(_MODELS))
def test_langchain_adapter_usage_is_counted(fake_llm, tmp_path, provider):
    """LangChain の 3 社の adapter（Agent と同じ部品）を通した呼び出しも記録される。"""
    model = _MODELS[provider]
    fake_llm.scene(provider, "ok", text=_SENTINEL)
    ledger = spend.SpendLedger(tmp_path)

    with spend.UsageMeter(ledger, "contract", max_usd=1.0) as meter:
        result = call_llm(_SENTINEL, model=model, provider=provider, system_prompt=_SENTINEL)

    assert result.text == _SENTINEL
    assert (meter.stats.sent, meter.stats.counted) == (1, 1)
    assert ledger.total_usd == pytest.approx(_expected_usd(model, 1))
    _assert_no_text(tmp_path)


def test_budget_stop_is_not_swallowed_by_the_router(fake_llm, tmp_path):
    """上限で止める例外は、router の ``except Exception`` を抜けて呼び出し側へ届き、要求は送られない。

    WHY: 飲まれると None（判定の失敗）になり、リプレイの次の行がそのまま課金を続ける。
    """
    ledger = spend.SpendLedger(tmp_path)

    with spend.UsageMeter(ledger, "contract", max_usd=1e-9), pytest.raises(spend.BudgetExceeded):
        _call_router_llm(_MODELS["anthropic"], "判定", _SENTINEL)

    assert fake_llm.requests == []
    assert ledger.total_usd == 0


def test_unpriced_model_is_not_sent(fake_llm, tmp_path):
    """価格表にないモデルは、送る前に止まる。"""
    ledger = spend.SpendLedger(tmp_path)

    with spend.UsageMeter(ledger, "contract", max_usd=1.0), pytest.raises(spend.UnpricedModel):
        _call_router_llm("claude-not-in-the-table-9", "判定", _SENTINEL)

    assert fake_llm.requests == []


def test_http_error_costs_nothing_and_keeps_no_text(fake_llm, tmp_path, capsys):
    """400 の応答は 0 円で記録され、エラー本文（入力のエコーを含みうる）は記録にも画面にも出ない。"""
    fake_llm.scene("anthropic", "http_error", status=400, message=_SENTINEL)
    ledger = spend.SpendLedger(tmp_path)

    with spend.UsageMeter(ledger, "contract", max_usd=1.0) as meter:
        assert _call_router_llm(_MODELS["anthropic"], "判定", _SENTINEL) is None

    assert meter.stats.http_errors >= 1
    assert ledger.total_usd == 0
    statuses = {json.loads(line)["status"] for line in (tmp_path / "calls.jsonl").read_text("utf-8").splitlines()}
    assert statuses == {"http_error"}
    _assert_no_text(tmp_path)
    captured = capsys.readouterr()
    assert _SENTINEL not in captured.out + captured.err
