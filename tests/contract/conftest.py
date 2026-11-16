"""
tests/contract/conftest.py — 契約テストの共通の fixture

契約テストは、本物の SDK（anthropic、openai、google-genai と LangChain の各 adapter）を通して、
L2 の LLM 呼び出しが偽の LLM サーバ（fake_llm.py）へ何を送り、応答をどう読むかを確かめる。
API キーなしで流れ、SDK を上げて黙って壊れたら赤くなるのが役目。

ここで用意するもの:
    - ``fake_llm``: 偽のサーバを立て、3 社の SDK の向け先とダミーの鍵を環境変数に入れる
    - ``agent_env``: Agent（道具つき）を、外へ出ない形で組み立てられるようにする。``fake_llm`` も立てる

上の tests/conftest.py が毎テストの前に OPENAI*、ANTHROPIC*、GOOGLE*、L2_* などの環境変数を消すので、
向け先と鍵はここで、テストごとに入れ直す（autouse の fixture は、ここの fixture より先に走る）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from fake_llm import DUMMY_API_KEY, FakeLLMServer

# 失敗の場面（400 や時間切れ）を速くする。既定は 20〜40 秒の時間切れと 1 回の再試行。
_FAST_TIMEOUT_SEC = "5"

# 上の tests/conftest.py の socket の網が loopback で通す、ポートの下限（_EPHEMERAL_PORT_MIN と同じ値）。
_SOCKET_GUARD_MIN_PORT = 32768


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
    port = int(server.url.rsplit(":", 1)[1])
    if port < _SOCKET_GUARD_MIN_PORT:
        server.close()
        pytest.fail(
            f"偽サーバのポートが {port} で、socket の網が通す {_SOCKET_GUARD_MIN_PORT} 以上ではありません。"
            "OS が選ぶ範囲が違う環境です（Linux の既定は 32768 から）。tests/conftest.py の _EPHEMERAL_PORT_MIN を確かめてください",
            pytrace=False,
        )
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
        # WHY: Windows のレジストリや HTTP_PROXY の設定が、loopback への要求をプロキシへ回すことがある。
        #     偽サーバへ届かず、テストが理由の分からない接続エラーになる。両方の綴りで loopback を除外する。
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        yield server
    finally:
        server.close()
    # 偽サーバ自身の失敗は、L2 の None や例外と見分けがつかないので、ここで赤くして原因を示す。
    assert not server.errors, "偽サーバ自身が失敗しました:\n" + "\n".join(server.errors)


# Agent の呼び出しに渡す、ストリームの識別子。BubbleToolCallbackHandler と ask_character は、
# これがあるときだけ吹き出しを出し、step のログを付ける。
AGENT_COMMON = {"stream_id": "contract-stream", "session_id": "contract-session", "trace_id": "contract-trace"}

# 吹き出しの文面。手元の data/bubble_messages.json に頼らず、テストの中で決める。
BUBBLE_TEXT = "（契約テストの吹き出し）"


@dataclass
class AgentEnv:
    """``agent_env`` が返すもの。

    Attributes:
        llm:         偽の LLM サーバ（``fake_llm`` と同じもの）
        published:   bus へ publish された event（publish の順）。Redis へは送らない
        common:      Agent に渡す stream_id / session_id / trace_id
        bubble_text: 吹き出しの文面（全キャラ、全場面で同じ）
    """

    llm: FakeLLMServer
    published: list[dict] = field(default_factory=list)
    common: dict = field(default_factory=lambda: dict(AGENT_COMMON))
    bubble_text: str = BUBBLE_TEXT

    def bubbles(self, character: str | None = None) -> list[dict]:
        """publish された bubble.update の payload を、キャラで絞って返す。"""
        return [
            e["payload"] for e in self.published
            if e.get("type") == "bubble.update" and (character is None or e["payload"].get("character") == character)
        ]


@pytest.fixture
def agent_env(fake_llm: FakeLLMServer, monkeypatch: pytest.MonkeyPatch) -> Iterator[AgentEnv]:
    """Agent の経路（graph.run_graph、ask_character）を、偽の LLM サーバだけで流せるようにする。

    やること:
        - L2_ENABLE_TOOLS=true（Agent を組み立てる）、L2_ENABLE_RAG=false（retrieve_memory を付けない。C2 へ行くため）
        - TAVILY_API_KEY にダミーを入れる。web_search_tool は一覧に載るが、テストは呼ばせない
        - bus の publish を記録係に差し替える。上の conftest が L2_BUS_FAIL_OPEN=false にしているので、
          差し替え漏れは Redis への接続の失敗として赤くなる
        - 吹き出しの文面を BUBBLE_TEXT に固定する
        - 終わりに ask_character の contextvars とセッションの状態を戻す（次のテストへ持ち越さない）

    WHY 道具の一覧を確かめられる形で残す: graph._load_mcp_tools は失敗を握って空の一覧を返し、そのとき
    Agent ではない単一ノードの道へ黙って落ちる。テストは記録した要求の tools を見て、Agent を通ったことを確かめる。

    使い方::

        def test_x(agent_env):
            agent_env.llm.scene("openai", "ok", text="こんにちは")
            result = graph.run_graph("やあ", model="gpt-5.5", provider="openai",
                                     character_slug="mimi", common=agent_env.common)
            assert agent_env.bubbles("mimi") == []

    Yields:
        AgentEnv
    """
    from lab_lounge import pipeline
    from lab_lounge.mcp_servers.ask_character import reset_ask_character_context

    monkeypatch.setenv("L2_ENABLE_TOOLS", "true")
    monkeypatch.setenv("L2_ENABLE_RAG", "false")
    monkeypatch.setenv("TAVILY_API_KEY", DUMMY_API_KEY)
    env = AgentEnv(llm=fake_llm)

    def record(event: dict) -> str:
        env.published.append(event)
        return f"0-{len(env.published)}"

    # graph の handler は関数の中で bus から import し、pipeline はモジュールの先頭で import している。両方を差し替える。
    monkeypatch.setattr("lab_lounge.bus.publish", record)
    monkeypatch.setattr(pipeline, "publish", record)
    bubble_keys = ("searching", "web_search", "ask_character", "thinking", "answering")
    monkeypatch.setattr(
        pipeline, "_BUBBLE_MESSAGES",
        {slug: dict.fromkeys(bubble_keys, BUBBLE_TEXT) for slug in ("mimi", "chisame", "sakura", "octamaid", "ruka")},
    )
    reset_ask_character_context()
    try:
        yield env
    finally:
        reset_ask_character_context()
