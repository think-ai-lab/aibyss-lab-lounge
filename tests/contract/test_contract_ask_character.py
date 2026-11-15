"""
ask_character（キャラ間の協働）の往復の契約テスト

caller（ここではミミ）の Agent が ask_character_tool を呼ぶと、次の順に進む。

1. 道具の中で、caller の filler_model で導入セリフを作る（call_llm。同期）
2. 協働先（ここではさくら）の Agent を別のスレッドで走らせる（create_react_agent と _run_agent）
3. 協働先の答えを整えて、道具の結果として caller へ返す
4. caller の Agent が、それを受けて最後の応答を作る

ここでは本物の LangChain と SDK を偽の LLM サーバへ向けて、この往復が今の形で通ることを固定する（TEST-01）。
caller の情報（caller_slug など）は contextvars で道具まで届く必要があり、届かないと導入セリフが丸ごと
抜ける。協働先に OpenAI 以外（Anthropic）を選んだのは、協働先の並列の禁止と、失敗時のフォールバックの
provider を確かめるため。
"""

from __future__ import annotations

import logging
import re

import pytest

from lab_lounge import graph
from lab_lounge.characters import get_character
from lab_lounge.mcp_servers import ask_character
from lab_lounge.mcp_servers.ask_character import set_ask_character_context

_CALLER = "mimi"
_TARGET = "sakura"
_QUESTION = "どう思う？"


def _model_of(slug: str) -> tuple[str, str]:
    char = get_character(slug)
    return char.llm_provider, char.llm_model


def _start_round_trip(env, *, target_scene: str = "ok", **target_params) -> object:
    """caller が協働先へ 1 回振り、受けて締める往復を台本にして、run_graph を呼ぶ。

    caller の本応答のモデルには「ask_character_tool を呼ぶ」を 1 回だけ並べ、2 回目からは既定の
    構造化の応答（"まとめ"）を返す。導入セリフ（caller の filler_model）は本文だけを返す。
    """
    caller = get_character(_CALLER)
    caller_provider, caller_model = _model_of(_CALLER)
    target_provider, _ = _model_of(_TARGET)
    set_ask_character_context(caller_slug=_CALLER, common=env.common, **env_tts(env))
    env.llm.script(caller_model, [
        ("tool_call", {"name": "ask_character_tool", "args": {"character_slug": _TARGET, "question": _QUESTION}}),
    ])
    env.llm.script(caller.filler_model, [("text", {"text": "ミミの導入"})])
    env.llm.scene(caller_provider, "ok", text="まとめ")
    env.llm.scene(target_provider, target_scene, **(target_params or {"text": "さくらの答え"}))
    return graph.run_graph(
        "さくらにも聞いてみて", model=caller_model, provider=caller_provider,
        system_prompt="あなたはミミです。", character_slug=_CALLER, common=env.common,
    )


def env_tts(env) -> dict:
    """set_ask_character_context に渡す TTS の口。テストが env.tts_chunks を用意したときだけ渡す。"""
    chunks = getattr(env, "tts_chunks", None)
    return {} if chunks is None else {"on_tts_chunk": lambda *args: chunks.append(args)}


def _last_user_text(req) -> str:
    """OpenAI の要求の、最後のメッセージの本文を返す（道具の結果もここに入る）。"""
    content = req.body["messages"][-1].get("content")
    return content if isinstance(content, str) else str(content)


def _caller_steps(caplog) -> list[int]:
    """caller の「Agent LLM step N 開始」の N を、ログの順に返す。"""
    pattern = re.compile(rf"Agent LLM step (\d+) 開始 \[character={_CALLER}\]")
    return [int(m[1]) for r in caplog.records if (m := pattern.search(r.getMessage()))]


# ─── 挙動のテスト ─────────────────────────────────────────────────


def test_round_trip_reaches_target_and_back(agent_env):
    """caller → 導入セリフ → 協働先 → caller の往復が通り、途中の各段が今の形で送られる。"""
    caller = get_character(_CALLER)
    target = get_character(_TARGET)
    _, caller_model = _model_of(_CALLER)
    target_provider, target_model = _model_of(_TARGET)

    result = _start_round_trip(agent_env)

    assert '"response":"まとめ"' in result.text
    # on_tool_start の吹き出しは caller に出る（HUD の「協働中」）。
    assert agent_env.bubbles(_CALLER) == [
        {"character": _CALLER, "step": "searching", "text": agent_env.bubble_text, "category": "speech_status"},
    ]
    # 導入セリフは caller の filler_model で作られた。contextvars の caller_slug が道具のスレッドまで届いた印。
    (intro,) = agent_env.llm.requests_for("openai", caller.filler_model)
    assert target.nickname in _last_user_text(intro)
    # 協働先は自分の provider で呼ばれ、導入セリフを含む質問を受け、道具は ask_character を除いた一覧。
    (collab,) = agent_env.llm.requests_for(target_provider, target_model)
    assert "ミミの導入" in str(collab.body["messages"][-1]["content"])
    assert [t["name"] for t in collab.body["tools"]] == ["web_search_tool"]
    # 協働先の並列の禁止は、create_react_agent が bind し直さないので要求に残っている。
    assert collab.body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    # caller の 2 回目の要求に、整えた協働先の答えが道具の結果として入る。
    back = agent_env.llm.requests_for("openai", caller_model)[-1]
    assert f"【{target.display_name}からの応答】\nさくらの答え" in _last_user_text(back)


def test_target_failure_returns_no_reply_without_target_voice(agent_env, monkeypatch):
    """協働先が失敗したら、caller には「応答を得られませんでした」だけが返り、協働先の声と吹き出しは出ない。

    WHY（AGENT-02 の網）: 以前は失敗の文（「エラー: ...」）を協働先の声で読み上げ、Agent の失敗時の
    フォールバックが provider を渡し忘れて、協働先のモデル名を OpenAI へ送っていた。
    """
    target = get_character(_TARGET)
    _, caller_model = _model_of(_CALLER)
    target_provider, target_model = _model_of(_TARGET)
    agent_env.tts_chunks = []
    synthesized: list[str] = []
    monkeypatch.setenv("L2_USE_REAL_TTS", "true")
    monkeypatch.setattr("lab_lounge.tts.synthesize", lambda text, **kw: synthesized.append(kw.get("speaker")))
    monkeypatch.setattr(ask_character, "_COLLAB_JOIN_TIMEOUT_SEC", 30.0)

    _start_round_trip(agent_env, target_scene="http_error", status=400, message="fake target failure")

    back = agent_env.llm.requests_for("openai", caller_model)[-1]
    assert f"{target.display_name}から応答を得られませんでした" in _last_user_text(back)
    # 声は導入セリフ（caller）だけ。協働先の声は合成しない。
    assert synthesized == [_CALLER]
    assert agent_env.bubbles(_TARGET) == []
    # 協働先の Agent と、そのフォールバックは、どちらも協働先の provider へ行った（OpenAI へ化けない）。
    target_calls = [r for r in agent_env.llm.requests if r.model == target_model]
    assert len(target_calls) == 2, [(r.provider, r.path) for r in target_calls]
    assert {r.provider for r in target_calls} == {target_provider}


# ─── 方針のテスト ─────────────────────────────────────────────────


@pytest.mark.xfail(reason="導入セリフの call_llm が、親のコールバックを contextvars で受け継ぎ、caller の Agent LLM step に数えられる。道具の実行の子の呼び出しを step から外し、補助として別の行にする")
def test_intro_is_not_counted_as_caller_step(agent_env, caplog):
    """caller の step は、本応答のモデルの 2 回（道具を呼ぶ、締める）だけで、導入セリフは数えない。

    WHY: step の数とトークンは、どこが何回モデルを呼んだかを測る数字。導入セリフ（filler_model）が
         混ざると、本応答のモデルの回数と課金を読み違える。
    """
    caplog.set_level(logging.INFO, logger="lab_lounge.graph")

    _start_round_trip(agent_env)

    assert _caller_steps(caplog) == [1, 2]
