"""
llm.py — LLM アダプタ

責務:
  - LLM API 呼び出しをこのファイルに閉じ込める
  - provider 切替しやすい構造にする（現在: "openai" のみ実装）
  - call_llm() は LLMResult を返す
  - graph.py から呼ばれる。直接呼ぶことも想定する

【拡張方法】
  provider を追加する場合は _PROVIDERS ディクショナリに関数を登録する。
  例: _PROVIDERS["ollama"] = _call_ollama

【前提パッケージ (real mode)】
  langchain-openai が必要。
  uv add --optional llm langchain-openai でインストールしてください。
  OPENAI_API_KEY 環境変数が必要。
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─── LLM クライアントの timeout / retry 設定 ───────────────────────

def get_llm_timeout_config(scope: str = "agent") -> tuple[float, int]:
    """LLM クライアントの per-request timeout(秒) と max_retries を env から取得する。

    WHY: timeout 無しの LLM クライアントは応答 stall 時に例外を投げず無限ハングし、
    配信フリーズの原因になる (実走 run_loop_20260531_164001 で観測)。全 LLM 呼出経路
    (agent / fallback / router / filler) で生成時に per-request timeout + 自動 retry を
    一元的な既定値で適用するためのヘルパー。値は env で配信中に調整できる。

    Args:
        scope: "agent"  = 本応答 Agent / フォールバック (大きい応答。既定 timeout 40s)
               "router" = ルーター / 意図判定 / filler (max_tokens 小。既定 timeout 20s。
                          dispatcher 判定経路なので短めにして早期復帰させる)

    Returns:
        (timeout_sec, max_retries) のタプル。

    注意 (google genai): google の SDK は max_retries=0 を「デフォルト(5 回)」と解釈する
    ため、google クライアントへ渡す際は 0 を避ける (呼出側で +1 等の補正を行う)。
    """
    if scope == "router":
        timeout = float(os.environ.get("L2_ROUTER_LLM_TIMEOUT_SEC", "20"))
    else:
        timeout = float(os.environ.get("L2_LLM_TIMEOUT_SEC", "40"))
    max_retries = int(os.environ.get("L2_LLM_MAX_RETRIES", "1"))
    return timeout, max_retries


def get_llm_reasoning_effort() -> str | None:
    """OpenAI reasoning 系モデル (gpt-5 系) の reasoning_effort を env から取得する。

    WHY: gpt-5 系は既定で推論を厚く行い、Agent モード (tools + 構造化出力) の継続ステップで
    推論が過大化する。これが出力暴走 (実走 20260531 16:40、~128k トークン) や応答ストール
    (実走 19:08、60s × 3) を引き起こした。キャラ対話は persona/Skills が応答を駆動するため
    深い推論は不要なので、reasoning_effort を下げて推論時間そのものを抑える
    (max_tokens は出力量しか抑えないため、推論時間ストールには効かない。本設定が補完する)。

    対象は OpenAI provider のみ (anthropic/google は別機構)。現状 OpenAI 側は gpt-5 系
    (gpt-5.5 / gpt-5.4-mini / gpt-5.4-nano) のみで、いずれも reasoning_effort に対応する。

    Returns:
        L2_LLM_REASONING_EFFORT の値 (既定 "low")。空文字 = 設定しない (None、モデル既定に従う)。
        有効値: "minimal" / "low" / "medium" / "high" / "none" (モデル依存)。万一モデルが
        本パラメータを拒否する場合は env に空文字を設定して無効化できる。
    """
    value = os.environ.get("L2_LLM_REASONING_EFFORT", "low").strip()
    return value or None


def get_llm_max_output_tokens() -> int | None:
    """LLM の 1 リクエストあたり出力トークン上限を env から取得する。

    WHY: 出力上限が無いと、reasoning 系モデル (gpt-5.5 等) が degenerate loop に入った際に
    1 リクエストで 10 万トークン超を生成し、(1) 数分間フリーズに見え、(2) 巨額課金を招く
    (実走 run_loop_20260531_164001 で mimi step3 が ~128k トークン出力を観測)。出力に
    天井を設けることで、どんな原因のループでも 1 ステップの被害を有界化する (安全弁)。

    Returns:
        L2_LLM_MAX_OUTPUT_TOKENS の整数値 (既定 4096)。"0" / 空文字 / 負値は None
        (= 上限なし、opt-out) を返す。

    注意: 正常な可視出力は ~250-400 トークン程度 (= 既定 4096 は十分な余裕)。上限超過時は
    finish_reason=length で打ち切られ、構造化出力の JSON が壊れて呼出元のフォールバックに
    流れる。極端に長い応答を出したい用途では env で引き上げる。
    """
    raw = os.environ.get("L2_LLM_MAX_OUTPUT_TOKENS", "4096").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


# ─── 戻り値型 ─────────────────────────────────────────────────────

@dataclass
class LLMResult:
    """LLM 呼び出し結果。pipeline.py / graph.py から参照する。"""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str


# ─── OpenAI adapter ───────────────────────────────────────────────

def _call_openai(
    text: str,
    *,
    model: str,
    context: str | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> LLMResult:
    """
    ChatOpenAI (langchain-openai) を使って LLM を呼び出す。

    langchain-openai が未インストールの場合は ImportError を送出する。
    OPENAI_API_KEY 環境変数が必要。

    Args:
        text:          ユーザー発話テキスト
        model:         使用するモデル名
        context:       RAG で取得した参照テキスト（省略時は non-RAG 動作）
        system_prompt: キャラクター別システムプロンプト（省略時は従来動作）
        **kwargs:      ChatOpenAI に渡す追加オプション
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai が必要です。"
            " uv add --optional llm langchain-openai でインストールしてください。"
        ) from exc

    t0 = time.monotonic()
    # per-request timeout + 自動 retry を既定適用 (caller が明示指定した場合はそれを優先)。
    # WHY: timeout 無しだと応答 stall 時に無限ハングし配信が固まる (graph.py の
    # フォールバック先でもあるため、ここが固まると復帰経路ごと止まる)。
    _timeout, _max_retries = get_llm_timeout_config("agent")
    kwargs.setdefault("timeout", _timeout)
    kwargs.setdefault("max_retries", _max_retries)
    # 出力トークン上限 (= 暴走生成の安全弁)。None なら付与しない (上限なし)。
    _max_out = get_llm_max_output_tokens()
    if _max_out is not None:
        kwargs.setdefault("max_tokens", _max_out)
    # reasoning_effort (= 推論時間の抑制、gpt-5 系のみ)。フォールバックも遅延/ストールを避ける。
    _effort = get_llm_reasoning_effort()
    if _effort is not None:
        kwargs.setdefault("reasoning_effort", _effort)
    llm = ChatOpenAI(model=model, **kwargs)

    # SystemMessage の組み立て: system_prompt + RAG context を結合
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if context:
        parts.append(f"## 参照情報\n\n{context}")

    if parts:
        messages = [SystemMessage(content="\n\n---\n\n".join(parts)), HumanMessage(content=text)]
    else:
        messages = [HumanMessage(content=text)]

    response = llm.invoke(messages)
    latency_ms = int((time.monotonic() - t0) * 1000)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    finish_reason = (
        (getattr(response, "response_metadata", None) or {}).get("finish_reason", "stop")
    )

    return LLMResult(
        text=str(response.content),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


# ─── Google Gemini adapter ────────────────────────────────────────

def _call_google(
    text: str,
    *,
    model: str,
    context: str | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> LLMResult:
    """
    ChatGoogleGenerativeAI (langchain-google-genai) を使って LLM を呼び出す。

    GOOGLE_API_KEY 環境変数が必要。
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain-google-genai が必要です。"
            " uv sync --extra llm-google でインストールしてください。"
        ) from exc

    t0 = time.monotonic()
    # per-request timeout + 自動 retry を既定適用 (caller の明示指定を優先)。
    # google genai は max_retries=0 を「デフォルト(5 回)」と解釈するため 0 を避ける。
    _timeout, _max_retries = get_llm_timeout_config("agent")
    kwargs.setdefault("timeout", _timeout)
    kwargs.setdefault("max_retries", max(1, _max_retries))
    # 出力トークン上限。google は param 名が max_output_tokens (openai/anthropic と異なる)。
    _max_out = get_llm_max_output_tokens()
    if _max_out is not None:
        kwargs.setdefault("max_output_tokens", _max_out)
    llm = ChatGoogleGenerativeAI(model=model, **kwargs)

    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if context:
        parts.append(f"## 参照情報\n\n{context}")

    if parts:
        messages = [SystemMessage(content="\n\n---\n\n".join(parts)), HumanMessage(content=text)]
    else:
        messages = [HumanMessage(content=text)]

    response = llm.invoke(messages)
    latency_ms = int((time.monotonic() - t0) * 1000)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    finish_reason = (
        (getattr(response, "response_metadata", None) or {}).get("finish_reason", "stop")
    )

    # Gemini は content をリスト形式で返すことがある
    # [{'type': 'text', 'text': '...', 'extras': {...}}]
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        response_text = "".join(text_parts)
    else:
        response_text = str(content)

    return LLMResult(
        text=response_text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


# ─── Anthropic Claude adapter ────────────────────────────────────

def _call_anthropic(
    text: str,
    *,
    model: str,
    context: str | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> LLMResult:
    """
    ChatAnthropic (langchain-anthropic) を使って LLM を呼び出す。

    ANTHROPIC_API_KEY 環境変数が必要。
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic が必要です。"
            " uv sync --extra llm-anthropic でインストールしてください。"
        ) from exc

    t0 = time.monotonic()
    # per-request timeout + 自動 retry を既定適用 (caller が明示指定した場合はそれを優先)。
    _timeout, _max_retries = get_llm_timeout_config("agent")
    kwargs.setdefault("timeout", _timeout)
    kwargs.setdefault("max_retries", _max_retries)
    # 出力トークン上限 (= 暴走生成の安全弁)。None なら付与しない。
    _max_out = get_llm_max_output_tokens()
    if _max_out is not None:
        kwargs.setdefault("max_tokens", _max_out)
    llm = ChatAnthropic(model=model, **kwargs)

    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    if context:
        parts.append(f"## 参照情報\n\n{context}")

    if parts:
        messages = [SystemMessage(content="\n\n---\n\n".join(parts)), HumanMessage(content=text)]
    else:
        messages = [HumanMessage(content=text)]

    response = llm.invoke(messages)
    latency_ms = int((time.monotonic() - t0) * 1000)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    finish_reason = (
        (getattr(response, "response_metadata", None) or {}).get("finish_reason", "stop")
    )

    return LLMResult(
        text=str(response.content),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "openai": _call_openai,
    "google": _call_google,
    "anthropic": _call_anthropic,
}


# ─── 公開 API ────────────────────────────────────────────────────

def call_llm(
    text: str,
    *,
    model: str,
    provider: str = "openai",
    context: str | None = None,
    system_prompt: str | None = None,
    caller_slug: str | None = None,
    **kwargs,
) -> LLMResult:
    """
    LLM を呼び出して LLMResult を返す。

    Args:
        text:          入力テキスト（utterance.final の payload.text）
        model:         使用するモデル名
        provider:      LLM プロバイダ（現在 "openai" のみ対応）
        context:       RAG で取得した参照テキスト（省略時は non-RAG 動作）
        system_prompt: キャラクター別システムプロンプト（省略時は従来動作）
        caller_slug:   応答を生成するキャラ slug。ログ強化 L-3 (Phase 0.5-A 後) で
                       追加。本セッション/ターン内で複数キャラの LLM 呼出が並列に
                       走るときに「どのキャラの呼出か」をログで識別できるよう
                       にする (e.g., 通常応答中の ask_character や挙手 BG LLM)。
                       None なら "?" 表示 (= 旧経路 / フォールバック)。
        **kwargs:      プロバイダ固有のオプション（temperature 等）

    Returns:
        LLMResult

    Raises:
        ValueError:  未対応 provider
        ImportError: provider のパッケージが未インストール
    """
    fn = _PROVIDERS.get(provider)
    if fn is None:
        supported = ", ".join(f'"{p}"' for p in _PROVIDERS)
        raise ValueError(
            f"未対応の provider: {provider!r}。対応プロバイダ: {supported}"
        )

    char_tag = f"[character={caller_slug or '?'}]"
    logger.info(
        "LLM 呼び出し開始 %s: provider=%s model=%s rag=%s",
        char_tag, provider, model, context is not None,
    )
    if context is not None:
        kwargs["context"] = context
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    result: LLMResult = fn(text, model=model, **kwargs)
    # ログ強化 L-3: text 全文出力をやめ、長さ + 冒頭 60 文字 preview に変更。
    # 全文は llm.final payload に残るため、ログ調査時の手掛かりは preview で十分。
    text_preview = result.text[:60].replace("\n", " ")
    text_suffix = "..." if len(result.text) > 60 else ""
    logger.info(
        "LLM 呼び出し完了 %s: latency_ms=%d input_tokens=%d output_tokens=%d "
        "finish_reason=%s text_len=%d text=%r%s",
        char_tag, result.latency_ms, result.input_tokens, result.output_tokens,
        result.finish_reason, len(result.text), text_preview, text_suffix,
    )
    return result
