"""
router と filler の raw SDK 呼び出しの契約テスト

``_call_router_llm``（発話の宛先の判定）と ``_call_filler_llm``（待ち時間の独り言）は、LangChain を通さず
各社の SDK を直接呼ぶ。どちらも例外をすべて None に変えるので、SDK の更新で壊れても配信は止まらず、
判定がパターンマッチへ、フィラーが定型文へ黙って落ちる。ここでは本物の SDK を偽の LLM サーバへ向けて、
今の正常な結果が出ることを固定する。

1 つのテストで router と filler の両方を呼ぶ。WHY: 2 つは同じ形の分岐（モデル名でプロバイダを選び、
同じ読み方で本文を取り出す）で、SDK の更新は両方に同時に効く。分けると件数だけが倍になる。
"""

from __future__ import annotations

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


def _call_both(model: str) -> tuple[str | None, str | None]:
    """router と filler を、同じモデルで 1 回ずつ呼ぶ。"""
    routed = _call_router_llm(model, "宛先を判定してください。", "ミミちゃん、こんにちは")
    filled = _call_filler_llm(model, "独り言を 1 文で。", "ミミちゃん、こんにちは")
    return routed, filled


@pytest.mark.parametrize("provider", list(_PROVIDERS))
def test_normal_reply_is_returned(fake_llm, provider):
    """正常な応答で、router は小文字の答えを、filler は前後の空白を除いた本文を返す。

    WHY 大文字と空白を混ぜる: router の後段は slug と完全一致で比べるので、strip と lower を SDK の
    応答のどこに掛けるか（本文の取り出し方）まで、この結果で確かめられる。
    """
    spec = _PROVIDERS[provider]
    fake_llm.scene(provider, "ok", text="  Mimi\n")

    routed, filled = _call_both(spec["model"])

    assert routed == "mimi"
    assert filled == "Mimi"
    sent = fake_llm.requests_for(provider)
    assert [r.model for r in sent] == [spec["model"]] * 2, [(r.provider, r.path) for r in fake_llm.requests]
    # 上限は L2 が決めている契約（router は答えの 1 語、filler は 1 文）。SDK が名前を変えても届くこと。
    max_out: Callable[[dict], int] = spec["max_out"]
    assert [max_out(r.body) for r in sent] == [20, 150]
