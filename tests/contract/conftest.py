"""
tests/contract/conftest.py — 契約テストの共通の fixture

契約テストは、本物の SDK（anthropic、openai、google-genai と LangChain の各 adapter）を通して、
L2 の LLM 呼び出しが偽の LLM サーバ（fake_llm.py）へ何を送り、応答をどう読むかを確かめる。
API キーなしで流れ、SDK を上げて黙って壊れたら赤くなるのが役目。

ここで用意するもの:
    - ``fake_llm``: 偽のサーバを立て、3 社の SDK の向け先とダミーの鍵を環境変数に入れる

上の tests/conftest.py が毎テストの前に OPENAI*、ANTHROPIC*、GOOGLE*、L2_* などの環境変数を消すので、
向け先と鍵はここで、テストごとに入れ直す（autouse の fixture は、ここの fixture より先に走る）。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from fake_llm import DUMMY_API_KEY, FakeLLMServer

# 失敗の場面（400 や時間切れ）を速くする。既定は 20〜40 秒の時間切れと 1 回の再試行。
_FAST_TIMEOUT_SEC = "5"


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLLMServer]:
    """偽の LLM サーバを立て、L2 の 6 通りの LLM クライアントをそこへ向ける。

    向け先の環境変数（SDK はクライアントを組み立てるたびに読み、L2 はクライアントをキャッシュしない）:
        - ANTHROPIC_BASE_URL:     anthropic.Anthropic と ChatAnthropic
        - OPENAI_BASE_URL:        openai.OpenAI と ChatOpenAI（値に /v1 を含める）
        - GOOGLE_GEMINI_BASE_URL: google.genai.Client と ChatGoogleGenerativeAI

    ANTHROPIC_API_URL と OPENAI_API_BASE は入れない。LangChain はこちらを先に読むので、入っていると
    raw の SDK と LangChain で向け先が分かれる。上の conftest が消しているので、ここでは入れないだけでよい。

    鍵は 3 社ともダミー（DUMMY_API_KEY）。鍵の形にしないので、万一外へ出ても使えない。
    外への接続そのものは、上の conftest の socket の網が止めて、テストを失敗にする。

    使い方（ログを確かめるテストでも同じ）::

        def test_x(fake_llm, caplog):
            fake_llm.scene("google", "ok", text="こんにちは")
            ...                                  # L2 の関数を呼ぶ
            req = fake_llm.requests_for("google")[-1]

    Yields:
        FakeLLMServer: 立てたサーバ。テストの終わりに止める
    """
    server = FakeLLMServer()
    env = {
        "ANTHROPIC_BASE_URL": server.url,
        "OPENAI_BASE_URL": server.openai_base_url,
        "GOOGLE_GEMINI_BASE_URL": server.url,
        "ANTHROPIC_API_KEY": DUMMY_API_KEY,
        "OPENAI_API_KEY": DUMMY_API_KEY,
        "GOOGLE_API_KEY": DUMMY_API_KEY,
        # WHY 0: 400 や 5xx の場面で、SDK が待って再試行するぶんテストが遅くなる。
        #     google の adapter は 0 を「既定の回数」と読むので、L2 の側が 1 に直す（そのまま 1 回は再試行する）。
        "L2_LLM_MAX_RETRIES": "0",
        "L2_LLM_TIMEOUT_SEC": _FAST_TIMEOUT_SEC,
        "L2_ROUTER_LLM_TIMEOUT_SEC": _FAST_TIMEOUT_SEC,
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        yield server
    finally:
        server.close()
