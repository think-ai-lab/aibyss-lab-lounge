"""
fake_llm.py — Anthropic / OpenAI / Gemini の HTTP API を真似る、テスト用の偽サーバ

契約テスト（tests/contract/）が、本物の SDK を通して L2 の LLM 呼び出しを確かめるための部品。
SDK は環境変数の向け先（ANTHROPIC_BASE_URL など）に従ってこのサーバへ HTTP を送り、
サーバは送られた要求を記録して、テストが選んだ応答を返す。

WHY 偽の HTTP サーバか（ほかの手では SDK の中身を通らない）:
    - unittest.mock の create_autospec は、関数の署名しか見えない。SDK が引数をどう HTTP に変えるかは見えない
    - LangChain の偽モデル（GenericFakeChatModel など）は、プロバイダごとの変換を通らない
    - httpx.MockTransport は、httpx 以外の HTTP クライアントへ移った SDK には差し込めない
    - HTTP の形は API 側に固定されているので、SDK を替えても網がそのまま生き残る
    嬉しくない点: 応答の形と拒否の規則は、自分たちの理解を固定したものにすぎない。実 API が受け付けるかは、
    API キーを使う確認（preflight --paid）に任せる。

使い方（ふつうは tests/contract/conftest.py の ``fake_llm`` fixture から使う）::

    def test_router(fake_llm):
        fake_llm.scene("anthropic", "ok", text=" Mimi\\n")      # Anthropic への応答を決める
        assert _call_router_llm("claude-haiku-4-5", "sys", "hi") == "mimi"
        (req,) = fake_llm.requests_for("anthropic")             # 送られた要求を見る
        assert req.body["max_tokens"] == 20

    - ``scene(provider, name, **params)``: そのプロバイダへの既定の応答を、名前で選ぶ。選べる名前は
      ``SCENES[provider]`` の鍵。何も選ばなければ "ok"
    - "ok" は Agent の構造化に合わせて答える。要求の tools に応答の型（MimiResponse など）があればそれを呼び、
      Gemini の構造化出力（responseJsonSchema）なら JSON の本文を返す。どちらでもなければ本文だけ（"text" と同じ）
    - ``script(model, [reply, ...])``: そのモデルへの応答を、呼ばれた順に並べる。使い切ったら scene に戻る。
      reply は scene の名前（文字列）、``(名前, {params})`` の組、または ``Reply`` そのもの。
      道具を呼ばせるときは ``("tool_call", {"name": "ask_character_tool", "args": {...}})``
    - ``requests`` / ``requests_for(provider, model)``: 記録した要求。鍵の値は残さない（ヘッダの名前だけ残す）
    - 公式の拒否の規則（``official_rejection``）は、場面と台本より先に見る。当たれば、場面に関わらず 400

受ける道:
    - Anthropic: POST /v1/messages
    - OpenAI:    POST /v1/chat/completions, POST /v1/responses
    - Gemini:    POST /v1*/models/{model}:generateContent
    それ以外の道は 404 を返し、provider="unknown" として記録する（SDK が道を変えたことに気づけるように）。

conftest の socket の網は、loopback の 32768 以上のポートだけを通す。サーバは 127.0.0.1 のポート 0 に立てて
OS にポートを選ばせるので、網の内側で動く。
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

# 値を記録してよいヘッダ（小文字）。SDK が送る API の版と機能の印で、ダイジェストで変化を見たいもの。
# WHY 許可制: 鍵（x-api-key、authorization、x-goog-api-key）や、SDK の版で毎回変わる user-agent を
#     記録に残さない。新しいヘッダは、名前だけが header_names に出る。
RECORDED_HEADER_VALUES = frozenset({
    "anthropic-version",
    "anthropic-beta",
    "content-type",
    "openai-beta",
})

# 契約テストが入れるダミーの鍵。鍵の形（sk-、AIza で始まるもの）にしない。
DUMMY_API_KEY = "dummy-not-a-real-key"

_GEMINI_PATH = re.compile(r"^/v1[^/]*/models/(?P<model>[^/:]+):(?P<method>[A-Za-z]+)$")


@dataclass
class RecordedRequest:
    """サーバが受けた 1 回の要求。

    Attributes:
        provider:     "anthropic" | "openai" | "google" | "unknown"
        path:         送られたパス（クエリを含む、そのままの形）
        route:        クエリを除いたパス
        model:        要求のモデル名（Gemini はパスから、ほかは本文から読む）。読めなければ None
        body:         本文の JSON。JSON でなければ {"_raw": 文字列}
        header_names: 送られたヘッダの名前（小文字、並べ替え済み）。値は残さない
        headers:      RECORDED_HEADER_VALUES にあるヘッダだけの、名前と値
    """

    provider: str
    path: str
    route: str
    model: str | None
    body: dict[str, Any]
    header_names: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Reply:
    """サーバが返す 1 回の応答。"""

    status: int
    body: dict[str, Any]


# ─── 応答の場面 ───────────────────────────────────────────────────
#
# 場面は (要求, **params) -> Reply の関数。プロバイダごとに名前で選ぶ。
# 応答は、各社の API リファレンスにある最小の正しい形に、usage（トークン数）を足したもの。
# usage を必ず入れる理由: ログが usage を読むかどうかを、この網で確かめられるようにするため。

_USAGE_IN = 12
_USAGE_OUT = 3


def _error_body(provider: str, status: int, message: str) -> dict[str, Any]:
    """各社のエラー本文の形を返す。SDK はこの形から例外の文面を作る。"""
    if provider == "anthropic":
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    if provider == "openai":
        return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}
    return {"error": {"code": status, "message": message, "status": "INVALID_ARGUMENT"}}


def _anthropic_text(req: RecordedRequest, *, text: str = "ok") -> Reply:
    """Anthropic Messages API の、本文 1 つの正常な応答。"""
    return Reply(200, {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": req.model or "claude-fake",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": _USAGE_IN, "output_tokens": _USAGE_OUT},
    })


def _openai_text(req: RecordedRequest, *, text: str = "ok") -> Reply:
    """OpenAI の、本文だけの正常な応答。Chat Completions と Responses の両方の道に答える。"""
    model = req.model or "gpt-fake"
    if req.route.endswith("/responses"):
        return Reply(200, {
            "id": "resp_fake",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_fake",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": _USAGE_IN,
                "output_tokens": _USAGE_OUT,
                "total_tokens": _USAGE_IN + _USAGE_OUT,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        })
    return Reply(200, {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "refusal": None},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": _USAGE_IN,
            "completion_tokens": _USAGE_OUT,
            "total_tokens": _USAGE_IN + _USAGE_OUT,
        },
    })


def _google_text(req: RecordedRequest, *, text: str = "ok") -> Reply:
    """Gemini generateContent の、本文 1 つの正常な応答。"""
    return Reply(200, {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": _USAGE_IN,
            "candidatesTokenCount": _USAGE_OUT,
            "totalTokenCount": _USAGE_IN + _USAGE_OUT,
        },
        "modelVersion": req.model or "gemini-fake",
        "responseId": "fake",
    })


def _anthropic_thinking_first(req: RecordedRequest, *, text: str = "ok") -> Reply:
    """先頭に thinking のブロック、続けて本文のブロックを返す。

    extended thinking（や、既定で考えるモデル）の応答は、content の先頭が thinking になる。
    """
    reply = _anthropic_text(req, text=text)
    reply.body["content"] = [
        {"type": "thinking", "thinking": "考えています", "signature": "fake-signature"},
        {"type": "text", "text": text},
    ]
    return reply


def _anthropic_thinking_only_max_tokens(req: RecordedRequest) -> Reply:
    """考えているうちに出力の上限に達し、本文のブロックが 1 つも無い応答を返す。"""
    reply = _anthropic_text(req)
    reply.body["content"] = [{"type": "thinking", "thinking": "考えて", "signature": "fake-signature"}]
    reply.body["stop_reason"] = "max_tokens"
    return reply


def _anthropic_refusal(req: RecordedRequest) -> Reply:
    """安全上の理由で応答を拒み、content が空の応答を返す（stop_reason が refusal）。"""
    reply = _anthropic_text(req)
    reply.body["content"] = []
    reply.body["stop_reason"] = "refusal"
    return reply


def _openai_empty_length(req: RecordedRequest) -> Reply:
    """推論で出力の上限を使い切り、本文が空のまま finish_reason が length の応答を返す。"""
    reply = _openai_text(req, text="")
    if "choices" in reply.body:
        reply.body["choices"][0]["finish_reason"] = "length"
        reply.body["usage"]["completion_tokens_details"] = {"reasoning_tokens": _USAGE_OUT}
    else:
        reply.body["status"] = "incomplete"
        reply.body["incomplete_details"] = {"reason": "max_output_tokens"}
        reply.body["output"] = []
    return reply


def _google_max_tokens_no_text(req: RecordedRequest) -> Reply:
    """考えているうちに出力の上限に達し、parts の無い候補を返す（finishReason が MAX_TOKENS）。"""
    reply = _google_text(req)
    reply.body["candidates"] = [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS", "index": 0}]
    reply.body["usageMetadata"]["thoughtsTokenCount"] = _USAGE_OUT
    return reply


def _http_error(provider: str) -> Callable[..., Reply]:
    """HTTP のエラーを返す場面を作る。既定は 400。"""

    def scene(req: RecordedRequest, *, status: int = 400, message: str = "fake error") -> Reply:
        return Reply(status, _error_body(provider, status, message))

    return scene


# ─── 道具の呼び出しと構造化の応答 ───────────────────────────────
#
# Agent（langchain の create_agent）は、応答の型（MimiResponse など）を 2 通りで強制する。
#   - ToolStrategy: 型を「道具」として tools に足し、tool_choice で必ずどれかを呼ばせる（OpenAI、Anthropic）
#   - ProviderStrategy: 型の JSON Schema を API の構造化出力に渡す（Gemini の responseJsonSchema）
# 既定の "ok" は、要求にどちらかがあれば、その JSON Schema から妥当な値を作って返す。無ければ本文だけを返す。
# WHY: テストが台本を書かなくても、Agent が 1 往復で構造化の応答に着く。

# 構造化の型の道具の名前（graph._get_character_response_schema が作る <Slug>Response）。
_STRUCTURED_TOOL = re.compile(r"^[A-Z][A-Za-z0-9]*Response$")


def example_from_schema(schema: dict[str, Any], *, text: str) -> Any:
    """JSON Schema から、妥当な値を 1 つ作る。

    既定値があればそれを使い、無ければ型から作る。文字列は text にする（"response" を確かめられるように）。
    $ref と anyOf は扱わない（今の応答の型には出てこない）。
    """
    if "default" in schema:
        return schema["default"]
    kind = str(schema.get("type", "string")).lower()
    if kind == "object":
        return {name: example_from_schema(sub, text=text) for name, sub in schema.get("properties", {}).items()}
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0))
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    return text


def tool_schemas(req: RecordedRequest) -> dict[str, dict[str, Any]]:
    """要求の tools を、名前 → 引数の JSON Schema の辞書にして返す（3 社の形を読む）。"""
    found: dict[str, dict[str, Any]] = {}
    for tool in req.body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:  # OpenAI Chat Completions
            found[tool["function"].get("name", "")] = tool["function"].get("parameters") or {}
        elif "input_schema" in tool:  # Anthropic
            found[tool.get("name", "")] = tool["input_schema"]
        elif tool.get("type") == "function" and "name" in tool:  # OpenAI Responses
            found[tool["name"]] = tool.get("parameters") or {}
        for decl in tool.get("functionDeclarations") or []:  # Gemini
            found[decl.get("name", "")] = decl.get("parameters") or decl.get("parametersJsonSchema") or {}
    return found


def structured_tool_name(req: RecordedRequest) -> str | None:
    """要求の tools に構造化の型の道具（ToolStrategy）があれば、その名前を返す。"""
    return next((name for name in tool_schemas(req) if _STRUCTURED_TOOL.match(name)), None)


def google_response_schema(req: RecordedRequest) -> dict[str, Any] | None:
    """Gemini の要求が構造化出力（ProviderStrategy）なら、その JSON Schema を返す。"""
    config = req.body.get("generationConfig") or {}
    if config.get("responseMimeType") != "application/json":
        return None
    return config.get("responseJsonSchema") or config.get("responseSchema") or {}


# 道具の呼び出しの ID。会話の中で重ならないように、プロセスで 1 つずつ増やす。
_call_ids = itertools.count(1)


def _tool_call(provider: str, req: RecordedRequest, name: str, args: dict[str, Any]) -> Reply:
    """道具 name を引数 args で呼ぶ応答を、そのプロバイダの形で返す。"""
    call_id = next(_call_ids)
    arguments = json.dumps(args, ensure_ascii=False)
    if provider == "anthropic":
        reply = _anthropic_text(req)
        reply.body["content"] = [{"type": "tool_use", "id": f"toolu_fake_{call_id}", "name": name, "input": args}]
        reply.body["stop_reason"] = "tool_use"
        return reply
    if provider == "openai":
        reply = _openai_text(req)
        if req.route.endswith("/responses"):
            reply.body["output"] = [{
                "type": "function_call", "id": f"fc_fake_{call_id}", "call_id": f"call_fake_{call_id}",
                "name": name, "arguments": arguments, "status": "completed",
            }]
            return reply
        choice = reply.body["choices"][0]
        choice["message"] = {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [{
                "id": f"call_fake_{call_id}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        }
        choice["finish_reason"] = "tool_calls"
        return reply
    reply = _google_text(req)
    reply.body["candidates"][0]["content"]["parts"] = [{"functionCall": {"name": name, "args": args}}]
    return reply


def _scene_tool_call(provider: str) -> Callable[..., Reply]:
    """道具を呼ぶ場面を作る。台本では ("tool_call", {"name": ..., "args": {...}}) の形で使う。"""

    def scene(req: RecordedRequest, *, name: str, args: dict[str, Any]) -> Reply:
        return _tool_call(provider, req, name, args)

    return scene


def _scene_ok(provider: str) -> Callable[..., Reply]:
    """既定の正常な応答の場面を作る。構造化の要求には構造化で、ほかは本文で答える。"""
    text_reply = {"anthropic": _anthropic_text, "openai": _openai_text, "google": _google_text}[provider]

    def scene(req: RecordedRequest, *, text: str = "ok") -> Reply:
        name = structured_tool_name(req)
        if name is not None:
            return _tool_call(provider, req, name, example_from_schema(tool_schemas(req)[name], text=text))
        schema = google_response_schema(req) if provider == "google" else None
        if schema is not None:
            return text_reply(req, text=json.dumps(example_from_schema(schema, text=text), ensure_ascii=False))
        return text_reply(req, text=text)

    return scene


SCENES: dict[str, dict[str, Callable[..., Reply]]] = {
    "anthropic": {
        "ok": _scene_ok("anthropic"),
        "text": _anthropic_text,
        "tool_call": _scene_tool_call("anthropic"),
        "thinking_first": _anthropic_thinking_first,
        "thinking_only_max_tokens": _anthropic_thinking_only_max_tokens,
        "refusal": _anthropic_refusal,
        "http_error": _http_error("anthropic"),
    },
    "openai": {
        "ok": _scene_ok("openai"),
        "text": _openai_text,
        "tool_call": _scene_tool_call("openai"),
        "empty_length": _openai_empty_length,
        "http_error": _http_error("openai"),
    },
    "google": {
        "ok": _scene_ok("google"),
        "text": _google_text,
        "tool_call": _scene_tool_call("google"),
        "max_tokens_no_text": _google_max_tokens_no_text,
        "http_error": _http_error("google"),
    },
}


# ─── 公式の拒否の規則 ─────────────────────────────────────────────
#
# 実 API が 400 で拒むと公式ドキュメントに書かれている組み合わせ（2026-11 時点）だけを真似る。
# 場面や台本より先に見るので、テストが "ok" を選んでいても 400 になる（実 API と同じ）。
# WHY: 今のキャラのモデルと SDK では発火しない。モデルや SDK を替えたとき、黙って 400 → None に落ちる
#     経路を、この網で赤くするための備え。
# 嬉しくない点: 規則を足し忘れると、実 API が拒むものを網が通してしまう。実 API の受理は --paid の点検で見る。

# この版以降の anthropic-version では、temperature を受け付けない。
_ANTHROPIC_NO_TEMPERATURE_SINCE = "2026-10-21"
# 強制の tool_choice（any / tool）と、thinking の disabled を拒むモデル（名前の先頭で見る）。
_ANTHROPIC_NO_FORCED_TOOL_CHOICE = ("claude-sonnet-5-5", "claude-opus-5-5")


def official_rejection(req: RecordedRequest) -> Reply | None:
    """公式の拒否の規則に当たれば 400 の応答を、当たらなければ None を返す。"""
    if req.provider != "anthropic":
        return None
    body = req.body
    version = req.headers.get("anthropic-version", "")
    if "temperature" in body and version >= _ANTHROPIC_NO_TEMPERATURE_SINCE:
        return Reply(400, _error_body("anthropic", 400, "temperature: not supported in this API version"))
    if (req.model or "").startswith(_ANTHROPIC_NO_FORCED_TOOL_CHOICE):
        tool_choice = body.get("tool_choice") or {}
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("any", "tool"):
            return Reply(400, _error_body("anthropic", 400, "tool_choice: forced tool use is not supported"))
        thinking = body.get("thinking") or {}
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            return Reply(400, _error_body("anthropic", 400, "thinking: disabled is not supported"))
    return None


# ─── サーバ ───────────────────────────────────────────────────────


def _classify(route: str, body: dict[str, Any]) -> tuple[str, str | None]:
    """パスからプロバイダを、パスか本文からモデル名を読む。"""
    if route == "/v1/messages":
        return "anthropic", body.get("model")
    if route in ("/v1/chat/completions", "/v1/responses"):
        return "openai", body.get("model")
    match = _GEMINI_PATH.match(route)
    if match and match["method"] == "generateContent":
        return "google", match["model"]
    return "unknown", body.get("model") if isinstance(body.get("model"), str) else None


class FakeLLMServer:
    """3 社の LLM API を真似る偽サーバ。127.0.0.1 の空いたポートで、デーモンのスレッドに立てる。

    Attributes:
        url:             サーバの根の URL（例: http://127.0.0.1:50123）。Anthropic と Gemini の向け先
        openai_base_url: OpenAI の向け先（url + "/v1"）。OpenAI の SDK は base_url に /v1 を含める
        requests:        受けた要求の記録（受けた順）
        errors:          偽サーバ自身の失敗（応答を作る途中の例外）の記録。空であるべきもの。
                         ``fake_llm`` fixture が、テストの後に空であることを確かめる
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.errors: list[str] = []
        self._scenes: dict[str, tuple[str, dict[str, Any]]] = {p: ("ok", {}) for p in SCENES}
        self._scripts: dict[str, deque[Any]] = defaultdict(deque)
        # WHY ロック: ThreadingHTTPServer は要求ごとにスレッドを立てる。協働（ask_character）では
        #     2 つの Agent が並んで呼ぶので、記録と台本の取り出しを同時に触らせない。
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server の規約
                server._handle(self)

            def do_GET(self) -> None:  # noqa: N802
                server._handle(self)

            def log_message(self, *_args: object) -> None:
                """テストの出力へアクセスログを混ぜない。"""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.openai_base_url = f"{self.url}/v1"
        # WHY poll_interval: shutdown() は次の見回りまで待つ。既定の 0.5 秒だと、テスト 1 件ごとに 0.5 秒の待ちが積もる。
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, name="fake-llm", daemon=True,
        )
        self._thread.start()

    # ── テストが使う口 ──

    def scene(self, provider: str, name: str, **params: Any) -> None:
        """そのプロバイダへの既定の応答を選ぶ。

        Args:
            provider: "anthropic" | "openai" | "google"
            name:     SCENES[provider] の鍵（"ok"、"http_error" など）
            **params: 場面の関数へ渡す引数（"ok" の text、"http_error" の status など）

        Raises:
            KeyError: 知らないプロバイダか場面の名前
        """
        SCENES[provider][name]  # 打ち間違いをテストの側で早く落とす
        self._scenes[provider] = (name, params)

    def script(self, model: str, replies: list[Any]) -> None:
        """そのモデルへの応答を、呼ばれた順に並べる。使い切ったら scene の応答に戻る。

        Args:
            model:   要求のモデル名（Gemini はパスのモデル名）
            replies: 各要素は、場面の名前（str）、(場面の名前, params) の組、または Reply
        """
        with self._lock:
            self._scripts[model].extend(replies)

    def requests_for(self, provider: str | None = None, model: str | None = None) -> list[RecordedRequest]:
        """記録した要求を、プロバイダとモデルで絞って返す。"""
        with self._lock:
            return [
                r for r in self.requests
                if (provider is None or r.provider == provider) and (model is None or r.model == model)
            ]

    def close(self) -> None:
        """サーバを止める。"""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    # ── 中身 ──

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        """1 回の要求を読み、記録し、応答を返す。偽サーバ自身の例外は 500 にして、errors へ残す。

        WHY: 例外をそのままにすると、接続が切れて SDK の接続エラーになり、L2 は None を返す。
        それが「偽サーバが壊れた」のか「L2 が本当に None を返した」のか、テストから見分けられない。
        500 と文面で返し、errors に残せば、fixture の後始末が赤くして、原因を示せる。
        """
        try:
            reply = self._serve(handler)
        except Exception as exc:  # noqa: BLE001 - 何が起きても 500 で返す
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            reply = Reply(500, {"error": {"message": f"fake_llm 自身の失敗: {type(exc).__name__}: {exc}"}})
        self._send(handler, reply)

    def _serve(self, handler: BaseHTTPRequestHandler) -> Reply:
        """要求を読んで記録し、返す応答を決める。"""
        raw = _read_body(handler)
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        if not isinstance(body, dict):
            body = {"_raw": body}
        path = handler.path
        route = path.split("?", 1)[0]
        provider, model = _classify(route, body)
        names = sorted({k.lower() for k in handler.headers.keys()})
        recorded = RecordedRequest(
            provider=provider,
            path=path,
            route=route,
            model=model,
            body=body,
            header_names=names,
            headers={k.lower(): v for k, v in handler.headers.items() if k.lower() in RECORDED_HEADER_VALUES},
        )
        with self._lock:
            self.requests.append(recorded)
            reply = self._next_reply(recorded)
        return reply

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, reply: Reply) -> None:
        """応答を書く。"""
        payload = json.dumps(reply.body, ensure_ascii=False).encode("utf-8")
        try:
            handler.send_response(reply.status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        except OSError:
            pass  # 時間切れの場面では、相手が先に切っている

    def _next_reply(self, req: RecordedRequest) -> Reply:
        """台本があれば台本から、無ければ scene から応答を作る。ロックの中で呼ぶ。"""
        if req.provider == "unknown":
            return Reply(404, {"error": {"message": f"fake_llm: 知らない道です: {req.route}"}})
        rejected = official_rejection(req)
        if rejected is not None:
            return rejected
        queue = self._scripts.get(req.model or "")
        item: Any = queue.popleft() if queue else self._scenes[req.provider]
        if isinstance(item, Reply):
            return item
        name, params = (item, {}) if isinstance(item, str) else item
        return SCENES[req.provider][name](req, **params)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    """要求の本文を読む。Content-Length と chunked の両方に対応する。

    WHY chunked: SDK の HTTP クライアントによっては、本文の長さを先に送らず chunked で送る。
    """
    length = handler.headers.get("Content-Length")
    if length is not None:
        return handler.rfile.read(int(length))
    if "chunked" in (handler.headers.get("Transfer-Encoding") or "").lower():
        chunks = []
        while True:
            size = int(handler.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                handler.rfile.readline()
                break
            chunks.append(handler.rfile.read(size))
            handler.rfile.readline()
        return b"".join(chunks)
    return b""
