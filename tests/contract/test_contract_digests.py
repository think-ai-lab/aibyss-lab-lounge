"""
送信内容のダイジェスト（golden）

SDK が偽の LLM サーバへ送った要求を、文字列の中身を伏せた形（ダイジェスト）にして、
tests/contract/goldens/ の golden と比べる。SDK や LangChain を上げて、送る引数の名前、構造、
数値、ヘッダが変わると、ここが赤くなる。ふだんは書き換えない。

更新のしかた:
    uv run --frozen --all-extras pytest tests/contract --update-goldens
    git diff tests/contract/goldens/     # 差分を 1 行ずつ読んでから commit する

ダイジェストに残すもの:
    - 本文の鍵の構造、数値、真偽値、null
    - 許可した鍵（_KEPT_STRING_KEYS）の文字列。モデル名、役割、道具の名前、tool_choice、必須の引数の名前など
    - パスと、送られたヘッダの名前、許可したヘッダの値（fake_llm.RECORDED_HEADER_VALUES）
    ほかの文字列は "<str>" にする。

WHY 文字列の中身を残さない: system prompt に今日の日付やキャラの設定が入るので、中身や長さを残すと
    golden が日ごとに変わる。ここで見たいのは SDK が送る形と引数の変化で、文面ではない。
WHY 9 本: raw の SDK（router）、Agent（create_agent）、call_llm の 3 つの道 × 3 社。同じ社でも道ごとに
    SDK の層が違う（raw / LangChain の adapter / create_agent の束ね方）ので、それぞれ 1 本ずつ残す。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lab_lounge import graph
from lab_lounge.characters import get_character
from lab_lounge.llm import call_llm
from lab_lounge.router import _call_router_llm

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"

# 中身を残す文字列の鍵。値が SDK の挙動の印になるもの（モデル名、役割、道具の名前、強制の仕方など）。
_KEPT_STRING_KEYS = frozenset({
    "model",
    "type",
    "role",
    "name",
    "tool_choice",
    "reasoning_effort",
    "responseMimeType",
    "mode",
    "title",
    "format",
    # 必須の引数の名前と列挙の値は、型の中身そのもの。SDK が落とすと構造化が崩れる。
    "required",
    "enum",
})

# golden に入ってはいけない、鍵の形の文字列（OpenAI / Anthropic の sk-、Google の AIza）。
_KEY_SHAPES = ("sk-", "AIza")

_RAW_MODELS = {"anthropic": "claude-haiku-4-5", "openai": "gpt-5.4-mini", "google": "gemini-3.1-flash-lite"}
_AGENT_SLUGS = {"anthropic": "sakura", "openai": "mimi", "google": "chisame"}


def digest(value: Any, key: str | None = None) -> Any:
    """要求の本文を、文字列の中身を伏せた形にする。

    Args:
        value: 本文（またはその一部）
        key:   value が入っていた鍵の名前（文字列を残すかどうかの判断に使う）
    """
    if isinstance(value, dict):
        return {k: digest(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [digest(v, key) for v in value]
    if isinstance(value, str):
        return value if key in _KEPT_STRING_KEYS else "<str>"
    return value


def _request_digest(req) -> dict[str, Any]:
    """記録した要求 1 件を、golden に書く形にする。"""
    return {
        "path": req.path,
        "header_names": req.header_names,
        "headers": dict(sorted(req.headers.items())),
        "body": digest(req.body),
    }


def _check_golden(name: str, requests: list, update: bool) -> None:
    """要求の列のダイジェストを golden と比べる。update なら書き直す。"""
    text = json.dumps([_request_digest(r) for r in requests], ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    for shape in _KEY_SHAPES:
        assert shape not in text, f"golden に鍵の形の文字列 {shape!r} が入っている"
    path = _GOLDEN_DIR / f"{name}.json"
    if update:
        _GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.is_file(), f"golden がありません: {path.name}。--update-goldens で作り、差分を読んでから commit してください"
    expected = path.read_text(encoding="utf-8")
    assert text == expected, (
        f"{path.name} と送信内容が違います。SDK の更新なら --update-goldens で書き直し、git diff を読んでください"
    )


@pytest.mark.parametrize("provider", list(_RAW_MODELS))
def test_router_request_digest(fake_llm, update_goldens, provider):
    """raw の SDK（router）が送る要求の形。"""
    _call_router_llm(_RAW_MODELS[provider], "宛先を判定してください。", "ミミちゃん、こんにちは")

    _check_golden(f"router_{provider}", fake_llm.requests_for(provider), update_goldens)


@pytest.mark.parametrize("provider", list(_RAW_MODELS))
def test_call_llm_request_digest(fake_llm, update_goldens, provider):
    """call_llm（LangChain の adapter）が送る要求の形。"""
    call_llm("やあ", model=_RAW_MODELS[provider], provider=provider, system_prompt="あなたはテストのキャラです。")

    _check_golden(f"call_llm_{provider}", fake_llm.requests_for(provider), update_goldens)


@pytest.mark.parametrize("provider", list(_AGENT_SLUGS))
def test_agent_request_digest(agent_env, update_goldens, provider):
    """Agent（create_agent）が送る要求の形。構造化の応答で 1 往復に終わる。"""
    slug = _AGENT_SLUGS[provider]
    char = get_character(slug)

    graph.run_graph(
        "やあ", model=char.llm_model, provider=char.llm_provider, system_prompt="あなたはテストのキャラです。",
        character_slug=slug, common=agent_env.common,
    )

    _check_golden(f"agent_{slug}", agent_env.llm.requests_for(provider), update_goldens)
