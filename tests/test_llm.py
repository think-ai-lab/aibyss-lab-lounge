"""
test_llm.py — llm.call_llm テスト

外部 LLM API は呼ばない（_PROVIDERS dict をモックして検証する）。
langchain-openai がインストールされていなくても動作する。
"""

import pytest
from unittest.mock import MagicMock, patch

import lab_lounge.llm as llm_mod
from lab_lounge.llm import LLMResult, call_llm


FAKE_RESULT = LLMResult(
    text="テスト応答",
    model="gpt-4o-mini",
    input_tokens=10,
    output_tokens=5,
    latency_ms=123,
    finish_reason="stop",
)


def _mock_openai(fake: LLMResult):
    """_PROVIDERS["openai"] に差し込む MagicMock を返す。"""
    m = MagicMock(return_value=fake)
    return m


class TestCallLlm:
    def test_unsupported_provider_raises_value_error(self):
        """未対応 provider は ValueError"""
        with pytest.raises(ValueError, match="未対応"):
            call_llm("hello", model="gpt-4o-mini", provider="unsupported_xyz")

    def test_openai_dispatches_to_openai_adapter(self):
        """provider='openai' のとき _PROVIDERS["openai"] が呼ばれる"""
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        mock_fn.assert_called_once_with("テスト", model="gpt-4o-mini")
        assert result == FAKE_RESULT

    def test_returns_llm_result_instance(self):
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        assert isinstance(result, LLMResult)

    def test_result_text_field(self):
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        assert result.text == "テスト応答"

    def test_result_token_fields(self):
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_result_latency_ms(self):
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        assert result.latency_ms == 123

    def test_result_finish_reason(self):
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            result = call_llm("テスト", model="gpt-4o-mini", provider="openai")
        assert result.finish_reason == "stop"

    def test_kwargs_forwarded_to_adapter(self):
        """**kwargs が adapter に転送される（temperature 等）"""
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            call_llm("テスト", model="gpt-4o-mini", provider="openai", temperature=0.7)
        mock_fn.assert_called_once_with("テスト", model="gpt-4o-mini", temperature=0.7)

    def test_model_passed_to_adapter(self):
        """model 引数が adapter に正しく渡る"""
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            call_llm("テスト", model="gpt-4o", provider="openai")
        mock_fn.assert_called_once_with("テスト", model="gpt-4o")

    def test_caller_slug_logged(self, caplog):
        """ログ強化 L-3: caller_slug が指定されるとログに [character=...] が含まれる。"""
        import logging
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            with caplog.at_level(logging.INFO, logger="lab_lounge.llm"):
                call_llm(
                    "テスト", model="gpt-4o-mini", provider="openai",
                    caller_slug="mimi",
                )
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "[character=mimi]" in joined

    def test_caller_slug_none_logs_question_mark(self, caplog):
        """caller_slug 未指定時はログに [character=?] と表示される (= 旧経路 / 不明)。"""
        import logging
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            with caplog.at_level(logging.INFO, logger="lab_lounge.llm"):
                call_llm("テスト", model="gpt-4o-mini", provider="openai")
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "[character=?]" in joined

    def test_caller_slug_not_forwarded_to_adapter(self):
        """caller_slug は llm.py 内のログ用なので adapter には渡らない (= 副作用なし)。"""
        mock_fn = _mock_openai(FAKE_RESULT)
        with patch.dict(llm_mod._PROVIDERS, {"openai": mock_fn}):
            call_llm(
                "テスト", model="gpt-4o", provider="openai", caller_slug="sakura",
            )
        # adapter には caller_slug は渡らない (= 既存 adapter の互換性維持)
        mock_fn.assert_called_once_with("テスト", model="gpt-4o")
