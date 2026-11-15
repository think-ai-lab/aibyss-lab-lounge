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
    - ``script(model, [reply, ...])``: そのモデルへの応答を、呼ばれた順に並べる。使い切ったら scene に戻る。
      reply は scene の名前（文字列）、``(名前, {params})`` の組、または ``Reply`` そのもの
    - ``requests`` / ``requests_for(provider, model)``: 記録した要求。鍵の値は残さない（ヘッダの名前だけ残す）

受ける道:
    - Anthropic: POST /v1/messages
    - OpenAI:    POST /v1/chat/completions, POST /v1/responses
    - Gemini:    POST /v1*/models/{model}:generateContent
    それ以外の道は 404 を返し、provider="unknown" として記録する（SDK が道を変えたことに気づけるように）。

conftest の socket の網は、loopback の 32768 以上のポートだけを通す。サーバは 127.0.0.1 のポート 0 に立てて
OS にポートを選ばせるので、網の内側で動く。
"""

from __future__ import annotations

import json
import re
import threading
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


def _anthropic_ok(req: RecordedRequest, *, text: str = "ok") -> Reply:
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


def _openai_ok(req: RecordedRequest, *, text: str = "ok") -> Reply:
    """OpenAI の正常な応答。Chat Completions と Responses の両方の道に答える。"""
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


def _google_ok(req: RecordedRequest, *, text: str = "ok") -> Reply:
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


def _http_error(provider: str) -> Callable[..., Reply]:
    """HTTP のエラーを返す場面を作る。既定は 400。"""

    def scene(req: RecordedRequest, *, status: int = 400, message: str = "fake error") -> Reply:
        return Reply(status, _error_body(provider, status, message))

    return scene


SCENES: dict[str, dict[str, Callable[..., Reply]]] = {
    "anthropic": {"ok": _anthropic_ok, "http_error": _http_error("anthropic")},
    "openai": {"ok": _openai_ok, "http_error": _http_error("openai")},
    "google": {"ok": _google_ok, "http_error": _http_error("google")},
}


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
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
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
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-llm", daemon=True)
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
        """1 回の要求を読み、記録し、応答を返す。"""
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
