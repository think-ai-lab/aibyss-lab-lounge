"""
test_llm_timeout_retry.py — LLM 呼び出しの timeout + retry + ログ強化のテスト

背景: 実走 run_loop_20260531_164001 で、ask_character 後の Agent LLM step が timeout 無しで
無限ハングし配信がフリーズした。本テストは以下の防御を検証する:
  - 全 LLM クライアントが per-request timeout + max_retries 付きで生成される
  - agent.invoke が外側壁時計ガードで打ち切られ、フォールバックで自力復帰する
  - フォールバックが元の provider を維持する (旧実装の "openai" 固定バグの回帰防止)
  - SDK のリトライログが安全な範囲で可視化される

全テストは LLM を実呼出しせず mock する (= 決定性確保)。
"""

import contextvars
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.graph import (
    _get_agent_invoke_timeout,
    _get_llm_for_agent,
    _invoke_agent_with_timeout,
    _run_agent,
)
from lab_lounge.llm import (
    LLMResult,
    get_llm_max_output_tokens,
    get_llm_reasoning_effort,
    get_llm_timeout_config,
)
from lab_lounge.log_setup import _configure_llm_retry_loggers


def _sentinel_result(text: str = "fallback") -> LLMResult:
    """フォールバック検証用のセンチネル LLMResult。"""
    return LLMResult(
        text=text, model="m", input_tokens=0, output_tokens=0,
        latency_ms=0, finish_reason="stop",
    )


# ═══════════════════════════════════════════════════════════════════
# get_llm_timeout_config — env からの設定取得
# ═══════════════════════════════════════════════════════════════════


class TestGetLLMTimeoutConfig:
    def test_agent_scope_defaults(self, monkeypatch):
        """agent scope の既定は (40.0, 1)。"""
        monkeypatch.delenv("L2_LLM_TIMEOUT_SEC", raising=False)
        monkeypatch.delenv("L2_LLM_MAX_RETRIES", raising=False)
        assert get_llm_timeout_config("agent") == (40.0, 1)

    def test_router_scope_shorter_default(self, monkeypatch):
        """router scope の既定 timeout は agent より短い (20.0)。"""
        monkeypatch.delenv("L2_ROUTER_LLM_TIMEOUT_SEC", raising=False)
        monkeypatch.delenv("L2_LLM_MAX_RETRIES", raising=False)
        timeout, retries = get_llm_timeout_config("router")
        assert timeout == 20.0
        assert retries == 1

    def test_env_overrides(self, monkeypatch):
        """env で timeout / retries を上書きできる。"""
        monkeypatch.setenv("L2_LLM_TIMEOUT_SEC", "45")
        monkeypatch.setenv("L2_ROUTER_LLM_TIMEOUT_SEC", "10")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "3")
        assert get_llm_timeout_config("agent") == (45.0, 3)
        assert get_llm_timeout_config("router") == (10.0, 3)


# ═══════════════════════════════════════════════════════════════════
# get_llm_max_output_tokens — 出力トークン上限 (暴走生成の安全弁)
# ═══════════════════════════════════════════════════════════════════


class TestGetLLMMaxOutputTokens:
    def test_default_is_4096(self, monkeypatch):
        monkeypatch.delenv("L2_LLM_MAX_OUTPUT_TOKENS", raising=False)
        assert get_llm_max_output_tokens() == 4096

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "2000")
        assert get_llm_max_output_tokens() == 2000

    def test_zero_means_uncapped(self, monkeypatch):
        """"0" は opt-out (上限なし) = None。"""
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "0")
        assert get_llm_max_output_tokens() is None

    def test_empty_means_uncapped(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "")
        assert get_llm_max_output_tokens() is None

    def test_invalid_means_uncapped(self, monkeypatch):
        """不正値 (非整数) は None にフォールバック (例外を投げない)。"""
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "abc")
        assert get_llm_max_output_tokens() is None


# ═══════════════════════════════════════════════════════════════════
# get_llm_reasoning_effort — gpt-5 系の推論強度 (暴走/ストールの根治)
# ═══════════════════════════════════════════════════════════════════


class TestGetLLMReasoningEffort:
    def test_default_is_low(self, monkeypatch):
        monkeypatch.delenv("L2_LLM_REASONING_EFFORT", raising=False)
        assert get_llm_reasoning_effort() == "low"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "minimal")
        assert get_llm_reasoning_effort() == "minimal"

    def test_empty_means_unset(self, monkeypatch):
        """空文字 = 設定しない (None、モデル既定に従う / 拒否モデルの opt-out)。"""
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "")
        assert get_llm_reasoning_effort() is None


# ═══════════════════════════════════════════════════════════════════
# _get_llm_for_agent — クライアント生成時の timeout / max_retries 付与
# ═══════════════════════════════════════════════════════════════════


class TestGetLLMForAgentKwargs:
    def test_openai_client_gets_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_TIMEOUT_SEC", "60")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "2")
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "4096")
        with patch("langchain_openai.ChatOpenAI") as MockChat:
            _get_llm_for_agent("openai", "gpt-x")
        MockChat.assert_called_once_with(
            model="gpt-x", timeout=60.0, max_retries=2, max_tokens=4096,
        )

    def test_openai_agent_omits_reasoning_effort(self, monkeypatch):
        """Agent (tools 束縛) 経路には reasoning_effort を付けない (回帰防止)。

        gpt-5.5 は「function tools + reasoning_effort」を /v1/chat/completions で 400 拒否する
        (実走 run_loop_20260531_193332)。付与すると毎回フォールバックに落ち、ask_character が
        一切呼ばれなくなる。reasoning_effort はツールなし経路 (llm._call_openai) のみで使う。
        """
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "low")
        with patch("langchain_openai.ChatOpenAI") as MockChat:
            _get_llm_for_agent("openai", "gpt-x")
        assert "reasoning_effort" not in MockChat.call_args.kwargs

    def test_anthropic_client_gets_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_TIMEOUT_SEC", "60")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "2")
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "low")
        with patch("langchain_anthropic.ChatAnthropic") as MockChat:
            _get_llm_for_agent("anthropic", "claude-x")
        # anthropic には reasoning_effort を渡さない (OpenAI 専用パラメータ)
        MockChat.assert_called_once_with(
            model="claude-x", timeout=60.0, max_retries=2, max_tokens=4096,
        )
        assert "reasoning_effort" not in MockChat.call_args.kwargs

    def test_google_client_gets_timeout_and_retries(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_TIMEOUT_SEC", "60")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "2")
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "low")
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockChat:
            _get_llm_for_agent("google", "gemini-x")
        # google は param 名が max_output_tokens (max_tokens ではない)。
        # reasoning_effort は渡さない (OpenAI 専用パラメータ)。
        MockChat.assert_called_once_with(
            model="gemini-x", timeout=60.0, max_retries=2, max_output_tokens=4096,
        )
        assert "reasoning_effort" not in MockChat.call_args.kwargs

    def test_max_output_tokens_omitted_when_opted_out(self, monkeypatch):
        """L2_LLM_MAX_OUTPUT_TOKENS=0 (opt-out) 時は max_tokens を付与しない。"""
        monkeypatch.setenv("L2_LLM_MAX_OUTPUT_TOKENS", "0")
        with patch("langchain_openai.ChatOpenAI") as MockChat:
            _get_llm_for_agent("openai", "gpt-x")
        assert "max_tokens" not in MockChat.call_args.kwargs

    def test_google_max_retries_floor_avoids_zero(self, monkeypatch):
        """google は max_retries=0 を「デフォルト(5)」と解釈するため最低 1 に床上げする。"""
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "0")
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockChat:
            _get_llm_for_agent("google", "gemini-x")
        assert MockChat.call_args.kwargs.get("max_retries") == 1


# ═══════════════════════════════════════════════════════════════════
# _call_openai (ツールなしフォールバック) — reasoning_effort を渡す
# ═══════════════════════════════════════════════════════════════════


class TestCallOpenAIReasoningEffort:
    def test_call_openai_passes_reasoning_effort(self, monkeypatch):
        """ツールなしフォールバック (_call_openai) は reasoning_effort を ChatOpenAI へ渡す。

        gpt-5.5 はツールなしなら reasoning_effort を chat/completions で受け付ける
        (実走 run_loop_20260531_193332 のフォールバックで実証)。Agent (tools) 経路とは異なり
        ここでは安全に推論抑制できる。
        """
        monkeypatch.setenv("L2_LLM_REASONING_EFFORT", "low")
        from lab_lounge import llm as llm_mod

        mock_resp = MagicMock()
        mock_resp.content = "応答"
        mock_resp.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
        mock_resp.response_metadata = {"finish_reason": "stop"}

        with patch("langchain_openai.ChatOpenAI") as MockChat:
            MockChat.return_value.invoke.return_value = mock_resp
            llm_mod._call_openai("text", model="gpt-x")

        assert MockChat.call_args.kwargs.get("reasoning_effort") == "low"


# ═══════════════════════════════════════════════════════════════════
# 外側壁時計ガード — agent.invoke のハング打ち切り + フォールバック
# ═══════════════════════════════════════════════════════════════════


class TestOuterWallClockGuard:
    def test_invoke_hang_times_out_and_falls_back(self, monkeypatch):
        """agent.invoke が壁時計上限を超えたら中断し、単一ノードへフォールバックする。"""
        monkeypatch.setenv("L2_AGENT_INVOKE_TIMEOUT_SEC", "0.1")
        sentinel = _sentinel_result("guard-fallback")

        mock_agent = MagicMock()

        def slow_invoke(*args, **kwargs):
            time.sleep(0.5)  # ガード(0.1s)より長くブロック
            return {"messages": []}

        mock_agent.invoke.side_effect = slow_invoke

        with patch("lab_lounge.graph.call_llm", return_value=sentinel) as mock_llm:
            result = _run_agent(
                mock_agent, "テスト", "gpt-x",
                provider="openai",
                character_slug="mimi",
                common={"session_id": "test-session"},
            )

        assert result.text == "guard-fallback"
        assert mock_llm.called

    def test_get_agent_invoke_timeout_default(self, monkeypatch):
        monkeypatch.delenv("L2_AGENT_INVOKE_TIMEOUT_SEC", raising=False)
        assert _get_agent_invoke_timeout() == 90.0

    def test_guard_propagates_contextvars_to_worker(self):
        """外側ガードの worker thread に contextvar が引き継がれる。

        ThreadPoolExecutor は既定で contextvars を複製しないため、copy_context() が無いと
        agent 内部の ask_character が参照する contextvar (caller_slug / on_tts_chunk 等) が
        worker thread で既定値に戻り、導入セリフ・協働先 TTS が消える
        (run_loop_20260531_185642 で観測した regression)。本テストはその伝播を保証する。
        """
        test_var = contextvars.ContextVar("test_guard_var", default="UNSET")
        test_var.set("SET_IN_PARENT")
        seen = {}

        mock_agent = MagicMock()

        def capture(*args, **kwargs):
            seen["value"] = test_var.get()
            return {"messages": []}

        mock_agent.invoke.side_effect = capture

        _invoke_agent_with_timeout(
            mock_agent, {"messages": []}, None,
            timeout_sec=10,
            character_slug="mimi",
            provider="openai",
            model="gpt-x",
            handler=None,
            session_id="s",
        )

        assert seen["value"] == "SET_IN_PARENT"

    def test_invoke_success_passes_through_guard(self, monkeypatch):
        """正常時: ガードは agent.invoke の結果を素通しし LLMResult に変換する。"""
        monkeypatch.delenv("L2_AGENT_INVOKE_TIMEOUT_SEC", raising=False)
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "正常応答"
        mock_msg.usage_metadata = {"input_tokens": 5, "output_tokens": 3}
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}

        result = _run_agent(
            mock_agent, "テスト", "gpt-x",
            provider="openai",
            character_slug="mimi",
            common={"session_id": "s"},
        )

        assert result.text == "正常応答"
        assert result.model == "gpt-x"
        # フォールバックではなく agent.invoke が 1 回呼ばれた (素通し) ことを確認
        assert mock_agent.invoke.call_count == 1


# ═══════════════════════════════════════════════════════════════════
# フォールバックの provider 維持 (旧 "openai" 固定バグの回帰防止)
# ═══════════════════════════════════════════════════════════════════


class TestFallbackPreservesProvider:
    def test_anthropic_agent_falls_back_to_anthropic(self, monkeypatch):
        """anthropic キャラが invoke 失敗時、フォールバックも provider=anthropic を使う。"""
        sentinel = _sentinel_result()
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("boom")

        with patch("lab_lounge.graph.call_llm", return_value=sentinel) as mock_llm:
            _run_agent(
                mock_agent, "テスト", "claude-x",
                provider="anthropic",
                character_slug="sakura",
                common={"session_id": "s"},
            )

        assert mock_llm.call_args.kwargs.get("provider") == "anthropic"

    def test_google_agent_falls_back_to_google(self, monkeypatch):
        sentinel = _sentinel_result()
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("boom")

        with patch("lab_lounge.graph.call_llm", return_value=sentinel) as mock_llm:
            _run_agent(
                mock_agent, "テスト", "gemini-x",
                provider="google",
                character_slug="chisame",
                common={"session_id": "s"},
            )

        assert mock_llm.call_args.kwargs.get("provider") == "google"


# ═══════════════════════════════════════════════════════════════════
# on_llm_error の原因究明ログ強化
# ═══════════════════════════════════════════════════════════════════


class TestOnLLMErrorLogging:
    def test_error_log_includes_provider_model(self, caplog):
        """LLM step エラー時に provider / model / step が併記される。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        handler = BubbleToolCallbackHandler(
            "mimi", {"session_id": "s"}, model="gpt-5.5", provider="openai",
        )
        handler._llm_step_count = 3
        handler._llm_step_t0 = time.monotonic()

        with caplog.at_level(logging.WARNING, logger="lab_lounge.graph"):
            handler.on_llm_error(TimeoutError("simulated stall"))

        msg = caplog.text
        assert "step 3" in msg
        assert "openai" in msg
        assert "gpt-5.5" in msg
        assert "TimeoutError" in msg


# ═══════════════════════════════════════════════════════════════════
# log_setup — リトライログの安全な解放
# ═══════════════════════════════════════════════════════════════════


class TestRetryLoggerConfiguration:
    def test_retry_loggers_info_by_default(self, monkeypatch):
        """既定では openai/anthropic の _base_client を INFO に解放する (retry 行可視化)。"""
        monkeypatch.delenv("L2_LLM_DEBUG_HTTP", raising=False)
        _configure_llm_retry_loggers()
        assert logging.getLogger("openai._base_client").level == logging.INFO
        assert logging.getLogger("anthropic._base_client").level == logging.INFO

    def test_retry_loggers_debug_under_flag(self, monkeypatch):
        """L2_LLM_DEBUG_HTTP=true で DEBUG に下げ、HTTP 詳細も出す。"""
        monkeypatch.setenv("L2_LLM_DEBUG_HTTP", "true")
        _configure_llm_retry_loggers()
        assert logging.getLogger("openai._base_client").level == logging.DEBUG
        assert logging.getLogger("anthropic._base_client").level == logging.DEBUG

    def test_httpx_stays_suppressed(self, monkeypatch):
        """httpx は URL 全文 (キー含みうる) を出すため WARNING のまま据え置く。"""
        from lab_lounge.log_setup import _suppress_sensitive_loggers
        _suppress_sensitive_loggers()
        assert logging.getLogger("httpx").level == logging.WARNING


# ═══════════════════════════════════════════════════════════════════
# router / filler — raw SDK クライアントへの timeout 付与
# ═══════════════════════════════════════════════════════════════════


class TestRouterFillerRawSDKTimeout:
    def test_router_openai_client_gets_timeout(self, monkeypatch):
        """_call_router_llm の openai クライアントが router scope の timeout で生成される。"""
        monkeypatch.setenv("L2_ROUTER_LLM_TIMEOUT_SEC", "20")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "2")
        from lab_lounge import router

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="mimi"))]
        )
        with patch("openai.OpenAI", return_value=mock_client) as MockOpenAI:
            router._call_router_llm("gpt-5.4-mini", "sys", "user")
        MockOpenAI.assert_called_once_with(timeout=20.0, max_retries=2)

    def test_filler_openai_client_gets_timeout(self, monkeypatch):
        """_call_filler_llm の openai クライアントが router scope の timeout で生成される。"""
        monkeypatch.setenv("L2_ROUTER_LLM_TIMEOUT_SEC", "20")
        monkeypatch.setenv("L2_LLM_MAX_RETRIES", "2")
        from lab_lounge import filler

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ふむ…"))]
        )
        with patch("openai.OpenAI", return_value=mock_client) as MockOpenAI:
            filler._call_filler_llm("gpt-5.4-nano", "sys", "user")
        MockOpenAI.assert_called_once_with(timeout=20.0, max_retries=2)
