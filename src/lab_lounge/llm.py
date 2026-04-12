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
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    logger.info(
        "LLM 呼び出し開始: provider=%s model=%s rag=%s",
        provider, model, context is not None,
    )
    if context is not None:
        kwargs["context"] = context
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    result: LLMResult = fn(text, model=model, **kwargs)
    logger.info(
        "LLM 呼び出し完了: latency_ms=%d input_tokens=%d output_tokens=%d finish_reason=%s text=%s",
        result.latency_ms,
        result.input_tokens,
        result.output_tokens,
        result.finish_reason,
        result.text,
    )
    return result
