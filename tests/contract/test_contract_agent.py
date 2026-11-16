"""
Agent（graph.run_graph の道具つきの道）の契約テスト

本応答は、langchain の create_agent で組み立てた Agent が作る。応答の型（<Slug>Response）は、プロバイダに
よって 2 通りの戦略で強制される。

- ToolStrategy: 型を道具として tools に足し、tool_choice で必ず呼ばせる（OpenAI は "required"、Anthropic は any）。
  L2 が bind_tools 済みのモデルを渡すので、create_agent はモデルの対応を確かめられず、こちらを選ぶ
- ProviderStrategy: 型の JSON Schema を API の構造化出力に渡す（Gemini の responseJsonSchema）。
  Gemini は bind_tools せずに渡すので、create_agent がこちらを選ぶ

ここでは今の戦略と道具の一覧を、記録した要求から固定する（AGENT-05）。langchain を上げて戦略が変わったら
赤くなり、そのときにこの表を読み直す。あわせて、Agent が壊れたときにフォールバックへ流れるかを確かめる。
"""

from __future__ import annotations

import json
import logging

import pytest

from fake_llm import tool_schemas
from lab_lounge import graph
from lab_lounge.characters import get_character

# 今の 4 人の、構造化の戦略。tool_choice は記録した要求に載った値。
_STRATEGY = {
    "mimi": ("tool", "required"),
    "chisame": ("provider", None),
    "sakura": ("tool", {"type": "any"}),
    "octamaid": ("tool", "required"),
}
# Agent の道具の一覧（L2_ENABLE_RAG=false のとき）。retrieve_memory_tool は付かない。
_AGENT_TOOLS = {"web_search_tool", "ask_character_tool"}


def _model_of(slug: str) -> tuple[str, str]:
    """キャラの本応答の (provider, model) を返す。model が空のキャラは L2_LLM_MODEL の既定を使う。"""
    char = get_character(slug)
    return char.llm_provider, char.llm_model or "gpt-5.4-mini"


def _run(env, slug: str):
    """キャラとして run_graph を 1 回呼ぶ（_generation_node と同じ引数の形）。"""
    provider, model = _model_of(slug)
    return graph.run_graph(
        "やあ", model=model, provider=provider, system_prompt="あなたはテストのキャラです。",
        character_slug=slug, common=env.common,
    )


# ─── 挙動のテスト ─────────────────────────────────────────────────


@pytest.mark.parametrize("slug", list(_STRATEGY))
def test_agent_returns_structured_response_with_current_strategy(agent_env, caplog, slug):
    """構造化の応答が 1 往復で返り、戦略と道具の一覧が今の形のままである。"""
    caplog.set_level(logging.INFO, logger="lab_lounge.graph")
    provider, model = _model_of(slug)
    agent_env.llm.scene(provider, "ok", text="こんにちは")

    result = _run(agent_env, slug)

    assert json.loads(result.text) == {"response": "こんにちは", "speed": 100, "pose": "neutral"}
    # フォールバックの call_llm を通らず、structured_response から読んだこと。
    assert any("source=structured_response" in r.getMessage() for r in caplog.records), "structured_response を読んでいない"
    (sent,) = agent_env.llm.requests_for(provider, model)
    tools = set(tool_schemas(sent))
    strategy, tool_choice = _STRATEGY[slug]
    type_name = f"{slug.capitalize()}Response"
    if strategy == "tool":
        assert tools == _AGENT_TOOLS | {type_name}
        assert sent.body.get("tool_choice") == tool_choice
    else:
        assert tools == _AGENT_TOOLS
        config = sent.body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["title"] == type_name


def test_google_empty_reply_falls_back(agent_env, caplog):
    """Gemini が上限切れで本文なしなら、構造化の読み取りに失敗し、単一ノードのフォールバックで答える。

    WHY 挙動のテスト: ProviderStrategy は空の本文を JSON として読めず例外にするので、今もフォールバックへ流れる。
    OpenAI（ToolStrategy）では同じ場面で黙って空の応答になる（下の方針のテスト）。
    """
    provider, model = _model_of("chisame")
    agent_env.llm.script(model, ["max_tokens_no_text"])
    agent_env.llm.scene(provider, "ok", text="フォールバックです")

    result = _run(agent_env, "chisame")

    assert result.text == "フォールバックです"
    fallback = agent_env.llm.requests_for(provider, model)[-1]
    assert "tools" not in fallback.body, "フォールバックの要求に道具が載っている"


# ─── 方針のテスト ─────────────────────────────────────────────────


@pytest.mark.parametrize("slug, banned", [
    pytest.param(
        "mimi", lambda body: body.get("parallel_tool_calls") is False,
        marks=pytest.mark.xfail(raises=AssertionError, reason="AGENT-01: bind_tools の parallel_tool_calls=False を create_agent が bind し直して消す。create_agent に渡した後も残る形で付ける"),
        id="openai",
    ),
    pytest.param(
        "sakura", lambda body: (body.get("tool_choice") or {}).get("disable_parallel_tool_use") is True,
        marks=pytest.mark.xfail(raises=AssertionError, reason="AGENT-01, CL-08: tool_choice の disable_parallel_tool_use が create_agent の bind し直しで消える。消えない形で付ける"),
        id="anthropic",
    ),
])
def test_parallel_tool_calls_are_disabled_on_the_wire(agent_env, slug, banned):
    """並列の道具の呼び出しの禁止が、実際に送る要求に載っている。

    WHY: 同じ応答で ask_character を 2 回並べて呼ぶと、2 人の協働先が同時に話しだす（以前に配信で起きた）。
         コードは bind_tools で禁止しているが、要求に載らなければ効かない。
    """
    provider, model = _model_of(slug)

    _run(agent_env, slug)

    sent = agent_env.llm.requests_for(provider, model)[0]
    assert banned(sent.body), {k: sent.body.get(k) for k in ("parallel_tool_calls", "tool_choice")}


@pytest.mark.xfail(raises=AssertionError, reason="AGENT-04, OAI-09: ToolStrategy で本文が空の length が返ると、空の応答をそのまま返す。finish_reason を見てフォールバックへ流す")
def test_openai_empty_length_falls_back(agent_env):
    """OpenAI が推論で上限を使い切って本文が空なら、単一ノードのフォールバックで答える。"""
    provider, model = _model_of("mimi")
    agent_env.llm.script(model, ["empty_length"])
    agent_env.llm.scene(provider, "ok", text="フォールバックです")

    result = _run(agent_env, "mimi")

    assert result.text == "フォールバックです"


@pytest.mark.xfail(raises=ValueError, reason="AGENT-03: Agent の組み立て（モデルの構築や create_agent）の例外が run_graph の外へ抜ける。組み立ても try に入れてフォールバックへ流す")
def test_agent_build_failure_falls_back(agent_env, monkeypatch):
    """Agent の組み立てで例外が出ても、単一ノードのフォールバックで答える。

    WHY 組み立てを壊す: SDK の更新で、コンストラクタの引数の検証が厳しくなると、ここで例外になる。
         その経路は今、例外がターンの外へ抜け、キャラが黙る。
    """
    provider, model = _model_of("mimi")
    agent_env.llm.scene(provider, "ok", text="フォールバックです")

    def broken(*_args, **_kwargs):
        raise ValueError("契約テスト: 組み立ての失敗")

    monkeypatch.setattr(graph, "_get_llm_for_agent", broken)

    result = _run(agent_env, "mimi")

    assert result.text == "フォールバックです"
