"""
test_router.py — 発話ルーティングのテスト
"""

import pytest
from unittest.mock import patch, MagicMock

from lab_lounge.router import (
    RoutingDecision,
    route,
    _find_all_matches,
    _route_by_llm,
)
from lab_lounge.characters import get_all_characters


class TestRouteByNameHint:
    def test_slug_hint(self):
        result = route("こんにちは", name_hint="mimi")
        assert result.speaker == "mimi"
        assert result.reason == "name_hint"

    def test_wake_word_hint(self):
        result = route("何か教えて", name_hint="ミミ様")
        assert result.speaker == "mimi"
        assert result.reason == "name_hint"

    def test_display_name_hint(self):
        result = route("テスト", name_hint="波心ちさめ")
        assert result.speaker == "chisame"
        assert result.reason == "name_hint"

    def test_unknown_hint_falls_through(self):
        result = route("こんにちは", name_hint="unknown_character")
        # name_hint が不明 → テキストマッチ → デフォルト
        assert result.reason in ("text_match", "default")


class TestRouteByTextMatch:
    def test_wake_word_in_text(self):
        result = route("ミミ様、今日の天気は？")
        assert result.speaker == "mimi"
        assert result.reason == "text_match"

    def test_chisame_wake_word(self):
        result = route("ちさめさん、AIについて教えて")
        assert result.speaker == "chisame"
        assert result.reason == "text_match"

    def test_sakura_wake_word(self):
        result = route("さくらさん、大丈夫？")
        assert result.speaker == "sakura"
        assert result.reason == "text_match"

    def test_octamaid_in_text(self):
        result = route("オクタメイド、次の手順は？")
        assert result.speaker == "octamaid"
        assert result.reason == "text_match"

    def test_alias_match(self):
        result = route("お嬢様、お茶をどうぞ")
        assert result.speaker == "mimi"
        assert result.reason == "text_match"


class TestRouteDefault:
    def test_no_match_uses_default(self):
        result = route("こんにちは")
        assert result.speaker == "octamaid"  # デフォルト
        assert result.reason == "default"

    def test_custom_default(self, monkeypatch):
        monkeypatch.setenv("L2_DEFAULT_SPEAKER", "chisame")
        result = route("こんにちは")
        assert result.speaker == "chisame"
        assert result.reason == "default"

    def test_name_hint_overrides_text_match(self):
        # テキストには "ミミ様" だが hint は "chisame"
        result = route("ミミ様、元気？", name_hint="chisame")
        assert result.speaker == "chisame"
        assert result.reason == "name_hint"


class TestFindAllMatches:
    """_find_all_matches のユニットテスト。"""

    def test_single_name(self):
        chars = get_all_characters()
        matches = _find_all_matches("ちさめさん、AIについて教えて", chars)
        assert len(matches) == 1
        assert matches[0].slug == "chisame"

    def test_multiple_names(self):
        chars = get_all_characters()
        matches = _find_all_matches(
            "ミミ様が言ったことについて、ちさめさんはどう思いますか？", chars
        )
        slugs = [c.slug for c in matches]
        assert "mimi" in slugs
        assert "chisame" in slugs
        assert len(matches) == 2

    def test_no_name(self):
        chars = get_all_characters()
        matches = _find_all_matches("こんにちは", chars)
        assert len(matches) == 0

    def test_alias_detected(self):
        chars = get_all_characters()
        matches = _find_all_matches("お嬢様が仰ったことについて", chars)
        assert len(matches) == 1
        assert matches[0].slug == "mimi"


class TestLLMRouter:
    """LLM ルーターの単体テスト (OpenAI API モック)。"""

    def _mock_completion(self, content: str):
        """OpenAI Chat Completion のモックレスポンスを作る。"""
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_llm_returns_correct_slug(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_ROUTER_MODEL", "gpt-5.4-nano")

        chars = get_all_characters()
        candidates = [c for c in chars if c.slug in ("mimi", "chisame")]

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion("chisame")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _route_by_llm(
                "ミミ様が言ったことについて、ちさめさんはどう思いますか？",
                candidates,
            )

        assert result is not None
        assert result.slug == "chisame"

    def test_llm_returns_invalid_slug(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_ROUTER_MODEL", "gpt-5.4-nano")

        chars = get_all_characters()
        candidates = [c for c in chars if c.slug in ("mimi", "chisame")]

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion("unknown")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _route_by_llm("テスト", candidates)

        assert result is None

    def test_llm_api_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("L2_LLM_ROUTER_MODEL", "gpt-5.4-nano")

        chars = get_all_characters()
        candidates = [c for c in chars if c.slug in ("mimi", "chisame")]

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _route_by_llm("テスト", candidates)

        assert result is None

    def test_llm_returns_none_for_no_callout(self, monkeypatch):
        """LLM が 'none' を返したら string 'none' を返す（呼びかけなし）。"""
        monkeypatch.setenv("L2_LLM_ROUTER_MODEL", "gpt-5.4-nano")

        chars = get_all_characters()
        candidates = [c for c in chars if c.slug in ("mimi", "chisame")]

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion("none")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = _route_by_llm("ミミ様の仕組みは、ちさめさんが解説してましたね", candidates)

        assert result == "none"

    def test_llm_prompt_mentions_none_option(self, monkeypatch):
        """LLM ルータープロンプトに 'none' オプションが含まれる。"""
        monkeypatch.setenv("L2_LLM_ROUTER_MODEL", "gpt-5.4-nano")

        chars = get_all_characters()
        candidates = [c for c in chars if c.slug in ("mimi", "chisame")]

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_completion("none")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            _route_by_llm("テスト", candidates)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "none" in system_msg["content"]
        assert "呼びかけが存在しない" in system_msg["content"]


class TestHybridRouting:
    """ハイブリッドルーティング統合テスト。"""

    def test_single_name_no_llm_call(self, monkeypatch):
        """単一キャラクター名はLLMを呼ばずに即確定。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")

        with patch("lab_lounge.router._route_by_llm") as mock_llm:
            result = route("ちさめさん、AIについて教えて")

        assert result.speaker == "chisame"
        assert result.reason == "text_match"
        mock_llm.assert_not_called()

    def test_multiple_names_calls_llm(self, monkeypatch):
        """複数キャラクター名が検出されたらLLMルーターが呼ばれる。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")

        chars = get_all_characters()
        chisame = next(c for c in chars if c.slug == "chisame")

        with patch("lab_lounge.router._route_by_llm", return_value=chisame) as mock_llm:
            result = route("ミミ様が言ったことについて、ちさめさんはどう思いますか？")

        assert result.speaker == "chisame"
        assert result.reason == "llm_router"
        mock_llm.assert_called_once()

    def test_multiple_names_llm_disabled_falls_back(self, monkeypatch):
        """LLMルーター無効時は従来のパターンマッチ。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "false")

        result = route("ミミ様が言ったことについて、ちさめさんはどう思いますか？")
        # 従来ロジック: wake_word 優先で最初にマッチしたもの
        assert result.speaker == "mimi"
        assert result.reason == "text_match"

    def test_multiple_names_llm_failure_falls_back(self, monkeypatch):
        """LLMルーター失敗時は従来パターンマッチにフォールバック。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")

        with patch("lab_lounge.router._route_by_llm", return_value=None):
            result = route("ミミ様が言ったことについて、ちさめさんはどう思いますか？")

        assert result.speaker == "mimi"
        assert result.reason == "text_match"

    def test_no_match_default(self, monkeypatch):
        """名前なしはデフォルト（LLMルーター有効でも呼ばない）。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")

        with patch("lab_lounge.router._route_by_llm") as mock_llm:
            result = route("こんにちは")

        assert result.reason == "default"
        mock_llm.assert_not_called()

    def test_multiple_names_llm_returns_none_becomes_default(self, monkeypatch):
        """
        複数キャラ検出 + LLM ルーターが 'none'（呼びかけなし）を返したら
        reason="default" になる。意図ゲートは呼ばれない。
        """
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")

        with patch("lab_lounge.router._route_by_llm", return_value="none") as mock_llm:
            result = route("ミミ様の仕組みは、ちさめさんが解説してましたね")

        assert result.reason == "default"
        mock_llm.assert_called_once()

    def test_multiple_names_llm_returns_none_uses_default_speaker(self, monkeypatch):
        """'none' 判定時のデフォルトキャラクターが L2_DEFAULT_SPEAKER を尊重する。"""
        monkeypatch.setenv("L2_USE_LLM_ROUTER", "true")
        monkeypatch.setenv("L2_DEFAULT_SPEAKER", "octamaid")

        with patch("lab_lounge.router._route_by_llm", return_value="none"):
            result = route("ミミ様とちさめさんについての話ですね")

        assert result.reason == "default"
        assert result.speaker == "octamaid"


# ═══════════════════════════════════════════════════════════════════
# LLM 意図ゲート (呼び出しゲート Phase 2)
# ═══════════════════════════════════════════════════════════════════


class TestIsIntentGateEnabled:
    """is_intent_gate_enabled() のテスト。"""

    def test_default_false(self, monkeypatch):
        monkeypatch.delenv("L2_USE_INTENT_GATE", raising=False)
        from lab_lounge.router import is_intent_gate_enabled
        assert is_intent_gate_enabled() is False

    def test_true_string(self, monkeypatch):
        monkeypatch.setenv("L2_USE_INTENT_GATE", "true")
        from lab_lounge.router import is_intent_gate_enabled
        assert is_intent_gate_enabled() is True

    def test_one_string(self, monkeypatch):
        monkeypatch.setenv("L2_USE_INTENT_GATE", "1")
        from lab_lounge.router import is_intent_gate_enabled
        assert is_intent_gate_enabled() is True

    def test_yes_string(self, monkeypatch):
        monkeypatch.setenv("L2_USE_INTENT_GATE", "yes")
        from lab_lounge.router import is_intent_gate_enabled
        assert is_intent_gate_enabled() is True

    def test_false_string(self, monkeypatch):
        monkeypatch.setenv("L2_USE_INTENT_GATE", "false")
        from lab_lounge.router import is_intent_gate_enabled
        assert is_intent_gate_enabled() is False


class TestCheckIntent:
    """check_intent() のテスト。_call_router_llm をモックして LLM 呼び出しを回避する。"""

    def test_callout_response(self):
        """LLM が 'callout' を返したら 'callout'。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="callout"):
            result = check_intent("ねぇ、ミミ様、どう思う？", "mimi")
        assert result == "callout"

    def test_mention_response(self):
        """LLM が 'mention' を返したら 'mention'。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="mention"):
            result = check_intent("ミミ様の仕組みは、すごいですね", "mimi")
        assert result == "mention"

    def test_unknown_on_llm_error(self):
        """LLM 呼び出し失敗 (None) → 'unknown' (fail-open)。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value=None):
            result = check_intent("テスト", "mimi")
        assert result == "unknown"

    def test_unknown_on_invalid_response(self):
        """LLM が予期しない応答を返したら 'unknown'。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="maybe"):
            result = check_intent("テスト", "mimi")
        assert result == "unknown"

    def test_uses_env_model(self, monkeypatch):
        """L2_INTENT_GATE_MODEL が _call_router_llm に渡される。"""
        from lab_lounge.router import check_intent
        monkeypatch.setenv("L2_INTENT_GATE_MODEL", "gpt-5.4-nano")
        with patch("lab_lounge.router._call_router_llm", return_value="callout") as mock_llm:
            check_intent("テスト", "mimi")
        # 第1引数が model
        assert mock_llm.call_args.args[0] == "gpt-5.4-nano"

    def test_default_model_when_env_not_set(self, monkeypatch):
        """L2_INTENT_GATE_MODEL 未設定時はデフォルトモデルが使われる。"""
        from lab_lounge.router import check_intent
        monkeypatch.delenv("L2_INTENT_GATE_MODEL", raising=False)
        with patch("lab_lounge.router._call_router_llm", return_value="callout") as mock_llm:
            check_intent("テスト", "mimi")
        # デフォルトモデル (claude-haiku-4-5-20251001) が使われる
        assert "claude" in mock_llm.call_args.args[0].lower()

    def test_prompt_includes_character_slug(self):
        """system_prompt にキャラクター slug が含まれる。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="callout") as mock_llm:
            check_intent("テスト", "chisame")
        system_prompt = mock_llm.call_args.args[1]
        assert "chisame" in system_prompt

    def test_context_passed_as_user_text(self):
        """context が _call_router_llm に user_text として渡される。"""
        from lab_lounge.router import check_intent
        context = "今日はいい天気ですね。ミミ様、お出かけされますか？"
        with patch("lab_lounge.router._call_router_llm", return_value="callout") as mock_llm:
            check_intent(context, "mimi")
        # 第3引数が user_text
        assert mock_llm.call_args.args[2] == context
