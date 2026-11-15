"""
call_llm（LangChain の 3 社の adapter）の契約テスト

``call_llm`` は、Agent が失敗したときのフォールバックと、ask_character の導入セリフが使う単発の呼び出し。
ChatOpenAI、ChatAnthropic、ChatGoogleGenerativeAI を組み立てて 1 回 invoke し、本文とトークン数を
LLMResult に詰める。例外は呼び出し元へ抜ける（router / filler と違って None に変えない）。

ここでは本物の LangChain と SDK を偽の LLM サーバへ向け、応答の本文とトークン数が LLMResult に
届くことを固定する。adapter の更新で content の形（文字列 / ブロックのリスト）や usage_metadata の
読み方が変わると赤くなる。
"""

from __future__ import annotations

import pytest

from lab_lounge.llm import call_llm

# 今のキャラの filler_model（導入セリフで使う）に合わせる。
_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-3.1-flash-lite",
}


@pytest.mark.parametrize("provider", list(_MODELS))
def test_normal_reply_reaches_llm_result(fake_llm, provider):
    """正常な応答の本文とトークン数が、LLMResult に届く。

    WHY トークン数も見る: 導入セリフとフォールバックの課金は、ここの数字でしかログに出ない。
    0 に落ちても動作は変わらないので、テストで見ないと気づけない。
    """
    fake_llm.scene(provider, "ok", text="こんにちは")

    result = call_llm("やあ", model=_MODELS[provider], provider=provider, system_prompt="あなたはミミです。")

    assert result.text == "こんにちは"
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    (sent,) = fake_llm.requests_for(provider)
    assert sent.model == _MODELS[provider]


@pytest.mark.xfail(reason="CL-03: content がブロックのリストのとき str(list) をそのまま本文にする。type が text のブロックだけをつなぐ")
def test_anthropic_thinking_and_text_returns_only_the_text(fake_llm):
    """thinking と本文のブロックが並ぶ応答でも、本文だけを返す。

    WHY: ChatAnthropic はブロックが 2 つ以上あると content をリストのまま返す。今の実装は str() で
         リストの表記（[{'type': 'thinking', ...}]）を本文にするので、TTS が考えの中身ごと読み上げる。
    """
    fake_llm.scene("anthropic", "thinking_first", text="こんにちは")

    result = call_llm("やあ", model=_MODELS["anthropic"], provider="anthropic", system_prompt="あなたはさくらです。")

    assert result.text == "こんにちは"
