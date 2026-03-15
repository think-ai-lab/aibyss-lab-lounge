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

def _call_openai(text: str, *, model: str, **kwargs) -> LLMResult:
    """
    ChatOpenAI (langchain-openai) を使って LLM を呼び出す。

    langchain-openai が未インストールの場合は ImportError を送出する。
    OPENAI_API_KEY 環境変数が必要。
    """
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai が必要です。"
            " uv add --optional llm langchain-openai でインストールしてください。"
        ) from exc

    t0 = time.monotonic()
    llm = ChatOpenAI(model=model, **kwargs)
    response = llm.invoke([HumanMessage(content=text)])
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
}


# ─── 公開 API ────────────────────────────────────────────────────

def call_llm(text: str, *, model: str, provider: str = "openai", **kwargs) -> LLMResult:
    """
    LLM を呼び出して LLMResult を返す。

    Args:
        text:     入力テキスト（utterance.final の payload.text）
        model:    使用するモデル名
        provider: LLM プロバイダ（現在 "openai" のみ対応）
        **kwargs: プロバイダ固有のオプション（temperature 等）

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

    logger.info("LLM 呼び出し開始: provider=%s model=%s", provider, model)
    result: LLMResult = fn(text, model=model, **kwargs)
    logger.info(
        "LLM 呼び出し完了: latency_ms=%d input_tokens=%d output_tokens=%d finish_reason=%s",
        result.latency_ms,
        result.input_tokens,
        result.output_tokens,
        result.finish_reason,
    )
    return result
