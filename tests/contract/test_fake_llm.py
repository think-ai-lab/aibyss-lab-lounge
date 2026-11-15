"""
偽の LLM サーバ（fake_llm.py）そのものの確かめ

公式の拒否の規則は、今のキャラのモデルと SDK では発火しない。発火しない規則は、壊れていても気づけないので、
ここで規則に当たる要求を直接送って確かめる。

http.client で送る理由: urllib.request は Windows のレジストリのプロキシ設定を読むので、手元の設定で
結果が変わりうる。http.client はプロキシを見ない。
"""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit


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
