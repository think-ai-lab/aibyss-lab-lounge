"""
偽の LLM サーバ（fake_llm.py）そのものの確かめ

公式の拒否の規則は、今のキャラのモデルと SDK では発火しない。発火しない規則は、壊れていても気づけないので、
ここで規則に当たる要求を直接送って確かめる。

http.client で送る理由: urllib.request は Windows のレジストリのプロキシ設定を読むので、手元の設定で
結果が変わりうる。http.client はプロキシを見ない。
"""

from __future__ import annotations

import http.client
import io
import json
from urllib.parse import urlsplit

import pytest

from fake_llm import _read_body


def _post(url: str, path: str, body: dict, headers: dict[str, str]) -> int:
    """偽のサーバへ POST し、状態コードを返す。"""
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **headers})
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def _post_json(url: str, path: str, body: dict) -> tuple[int, dict]:
    """偽のサーバへ POST し、(状態コード, 応答の JSON) を返す。"""
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def _anthropic_text_of(fake_llm, model: str) -> str:
    status, reply = _post_json(
        fake_llm.url, "/v1/messages",
        {"model": model, "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200
    return reply["content"][0]["text"]


def test_official_rejection_rules_fire_only_where_documented(fake_llm):
    """規則に当たる組み合わせだけ 400 になり、今のモデルと版の組み合わせは通る。"""
    old, new = {"anthropic-version": "2023-06-01"}, {"anthropic-version": "2026-10-21"}
    base = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    cases = [
        # (説明, モデル, 本文に足すもの, ヘッダ, 期待する状態コード)
        ("新しい版の temperature", "claude-haiku-4-5", {"temperature": 0}, new, 400),
        ("古い版の temperature", "claude-haiku-4-5", {"temperature": 0}, old, 200),
        ("Sonnet 5.5 の強制の tool_choice", "claude-sonnet-5-5", {"tool_choice": {"type": "any"}}, old, 400),
        ("Opus 5.5 の thinking disabled", "claude-opus-5-5", {"thinking": {"type": "disabled"}}, old, 400),
        ("Sonnet 4.6 の強制の tool_choice", "claude-sonnet-4-6", {"tool_choice": {"type": "any"}}, old, 200),
        ("Sonnet 5.5 の auto の tool_choice", "claude-sonnet-5-5", {"tool_choice": {"type": "auto"}}, old, 200),
    ]

    got = [
        (label, _post(fake_llm.url, "/v1/messages", {**base, "model": model, **extra}, headers))
        for label, model, extra, headers, _ in cases
    ]

    assert got == [(label, expected) for label, *_, expected in cases]


def test_script_replies_in_order_then_falls_back_to_scene(fake_llm):
    """台本は呼ばれた順に使われ、使い切ったら scene（既定の "ok"）へ戻る。モデルごとに別の列。"""
    fake_llm.script("claude-a", [("text", {"text": "一つ目"}), ("text", {"text": "二つ目"})])
    fake_llm.script("claude-b", [("text", {"text": "別のモデル"})])

    got = [
        _anthropic_text_of(fake_llm, "claude-a"),
        _anthropic_text_of(fake_llm, "claude-b"),
        _anthropic_text_of(fake_llm, "claude-a"),
        _anthropic_text_of(fake_llm, "claude-a"),
        _anthropic_text_of(fake_llm, "claude-b"),
    ]

    assert got == ["一つ目", "別のモデル", "二つ目", "ok", "ok"]


def test_unknown_route_is_404_and_recorded_as_unknown(fake_llm):
    """SDK が道を変えたことに気づけるよう、知らない道は 404 で、provider="unknown" として記録する。"""
    status, reply = _post_json(fake_llm.url, "/v2/something", {"model": "m"})

    assert status == 404 and "知らない道" in reply["error"]["message"]
    (recorded,) = fake_llm.requests
    assert (recorded.provider, recorded.route, recorded.model) == ("unknown", "/v2/something", "m")


def test_internal_failure_is_500_and_recorded_in_errors(fake_llm, monkeypatch):
    """偽サーバ自身の例外は 500 と文面で返し、errors に残す（後始末の assert が赤くする元になる）。"""

    def boom(_req):
        raise RuntimeError("scene exploded")

    monkeypatch.setattr(fake_llm, "_next_reply", boom)

    status, reply = _post_json(fake_llm.url, "/v1/messages", {"model": "claude-a"})

    assert status == 500 and "scene exploded" in reply["error"]["message"]
    assert len(fake_llm.errors) == 1 and fake_llm.errors[0].startswith("RuntimeError: scene exploded")
    fake_llm.errors.clear()  # このテストは失敗を起こす側。後始末の「errors が空」の確かめに引っかからないようにする


class _FakeHandler:
    """_read_body が読む 2 つの口（headers と rfile）だけを持つ、HTTP ハンドラの代役。"""

    def __init__(self, raw: bytes, headers: dict[str, str]):
        self.rfile = io.BytesIO(raw)
        self.headers = headers


@pytest.mark.parametrize(
    ("raw", "headers", "expected"),
    [
        (b'{"a":1}', {"Content-Length": "7"}, b'{"a":1}'),
        # chunked: 16 進のサイズ行（拡張つきも）、本文、最後の 0 の行。
        (b"3\r\nabc\r\n4;ext=1\r\ndefg\r\n0\r\n\r\n", {"Transfer-Encoding": "chunked"}, b"abcdefg"),
        (b"a\r\n0123456789\r\n0\r\n\r\n", {"Transfer-Encoding": "Chunked"}, b"0123456789"),
        (b"", {}, b""),
    ],
    ids=["content-length", "chunked-with-extension", "chunked-hex-size-and-case", "no-body"],
)
def test_read_body_handles_content_length_and_chunked(raw, headers, expected):
    assert _read_body(_FakeHandler(raw, headers)) == expected
