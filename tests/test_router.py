"""
test_router.py — 発話ルーティングのテスト
"""

import pytest
from unittest.mock import patch, MagicMock

from lab_lounge.router import (
    ApprovalResult,
    IntentResult,
    RoutingDecision,
    check_approval,
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
    """check_intent() の Phase 2 互換モード (character_slug 指定) のテスト。
    Phase 0.5-A で戻り値が str → IntentResult に変わったため、assertion を更新。
    _call_router_llm をモックして LLM 呼び出しを回避する。"""

    def test_callout_response(self):
        """LLM が 'callout' を返したら IntentResult(intent='callout')。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="callout"):
            result = check_intent("ねぇ、ミミ様、どう思う？", "mimi")
        assert result.intent == "callout"
        assert result.target_slug == "mimi"
        assert result.confidence == 1.0

    def test_mention_response(self):
        """LLM が 'mention' を返したら IntentResult(intent='mention')。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="mention"):
            result = check_intent("ミミ様の仕組みは、すごいですね", "mimi")
        assert result.intent == "mention"
        assert result.target_slug == "mimi"
        assert result.confidence == 1.0

    def test_unknown_on_llm_error(self):
        """LLM 呼び出し失敗 (None) → IntentResult(intent='unknown') (fail-open)。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value=None):
            result = check_intent("テスト", "mimi")
        assert result.intent == "unknown"
        assert result.target_slug == "mimi"
        assert result.confidence == 0.0

    def test_unknown_on_invalid_response(self):
        """LLM が予期しない応答を返したら IntentResult(intent='unknown')。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="maybe"):
            result = check_intent("テスト", "mimi")
        assert result.intent == "unknown"
        assert result.target_slug == "mimi"

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


# ─── Phase 0.5-A で追加されたテストクラス ──────────────────────


class TestIntentResult:
    """IntentResult dataclass の挙動 (Phase 0.5-A で導入)。"""

    def test_frozen(self):
        """frozen dataclass — フィールド書換は FrozenInstanceError。"""
        from dataclasses import FrozenInstanceError
        result = IntentResult(intent="callout", target_slug="mimi", confidence=1.0)
        with pytest.raises(FrozenInstanceError):
            result.intent = "mention"  # type: ignore[misc]

    def test_field_types(self):
        """フィールドの型と値の確認。"""
        result = IntentResult(
            intent="interjection_candidate",
            target_slug="mimi",
            confidence=0.85,
        )
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "mimi"
        assert result.confidence == 0.85

    def test_target_slug_can_be_none(self):
        """target_slug は None も許容 (interjection_candidate が見つからない時)。"""
        result = IntentResult(intent="unknown", target_slug=None, confidence=0.0)
        assert result.target_slug is None

    def test_supports_all_intent_values(self):
        """4 値の intent を全て格納できる (Literal 型)。"""
        for intent_val in ("callout", "mention", "unknown", "interjection_candidate"):
            r = IntentResult(intent=intent_val, target_slug="mimi", confidence=1.0)
            assert r.intent == intent_val


class TestApprovalResult:
    """ApprovalResult dataclass の挙動 (Phase 0.5-A で導入)。"""

    def test_frozen(self):
        """frozen dataclass — フィールド書換は FrozenInstanceError。"""
        from dataclasses import FrozenInstanceError
        result = ApprovalResult(granted=True, target_slug="mimi", confidence=1.0)
        with pytest.raises(FrozenInstanceError):
            result.granted = False  # type: ignore[misc]

    def test_granted_true(self):
        result = ApprovalResult(granted=True, target_slug="chisame", confidence=1.0)
        assert result.granted is True
        assert result.target_slug == "chisame"
        assert result.confidence == 1.0

    def test_granted_false(self):
        result = ApprovalResult(granted=False, target_slug="sakura", confidence=0.9)
        assert result.granted is False
        assert result.target_slug == "sakura"


class TestCheckIntentInterjectionCandidate:
    """check_intent() の Phase 0.5-A interjection_candidate モード (character_slug=None)。

    挙手システム用に、特定キャラへの呼びかけがない発話に対して「自発介入したそうな
    キャラ」を判定する。Notion §C1 確定: octamaid は候補から除外。
    """

    def test_returns_interjection_candidate_with_slug(self):
        """LLM が候補 slug (mimi) を返したら IntentResult(intent='interjection_candidate')。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="mimi"):
            result = check_intent("AI 倫理について興味がある")
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "mimi"
        assert result.confidence == 1.0

    def test_returns_unknown_when_none(self):
        """LLM が 'none' を返したら IntentResult(intent='unknown', target_slug=None)。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none"):
            result = check_intent("今日はいい天気だね")
        assert result.intent == "unknown"
        assert result.target_slug is None

    def test_returns_unknown_on_llm_error(self):
        """LLM 呼び出し失敗 (None) → IntentResult(intent='unknown') (fail-open)。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value=None):
            result = check_intent("テスト")
        assert result.intent == "unknown"
        assert result.target_slug is None
        assert result.confidence == 0.0

    def test_returns_unknown_on_invalid_slug(self):
        """LLM が候補外 slug (octamaid 含む) を返したら IntentResult(intent='unknown')。"""
        from lab_lounge.router import check_intent
        # octamaid は候補から除外されているので、LLM が octamaid を返しても候補外扱い
        with patch("lab_lounge.router._call_router_llm", return_value="octamaid"):
            result = check_intent("テスト")
        assert result.intent == "unknown"

    def test_returns_unknown_on_unknown_string(self):
        """LLM が候補にもないランダム文字列を返したら IntentResult(intent='unknown')。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="invalid_char"):
            result = check_intent("テスト")
        assert result.intent == "unknown"

    def test_octamaid_excluded_from_candidates(self):
        """Notion C1 確定: octamaid はプロンプト内の候補リストに含まれない。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")  # character_slug=None
        system_prompt = mock_llm.call_args.args[1]
        # 候補リストに octamaid が含まれない
        assert "octamaid" not in system_prompt
        # 他の主要キャラは含まれる
        assert "mimi" in system_prompt
        assert "chisame" in system_prompt
        assert "sakura" in system_prompt

    def test_prompt_includes_all_eligible_slugs(self):
        """プロンプトに octamaid 以外の候補 slug が全て含まれる。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")
        system_prompt = mock_llm.call_args.args[1]
        for slug in ("mimi", "chisame", "sakura"):
            assert slug in system_prompt

    def test_text_passed_as_user_text(self):
        """text が _call_router_llm の user_text として渡される。"""
        from lab_lounge.router import check_intent
        text = "AI の倫理問題について考えていた"
        with patch("lab_lounge.router._call_router_llm", return_value="mimi") as mock_llm:
            check_intent(text)
        assert mock_llm.call_args.args[2] == text

    def test_prompt_includes_interest_areas(self):
        """Phase 0.5-A フェーズ 8: プロンプトに各キャラの担当エリアが含まれる。

        skills/characters/<slug>.md の `**担当エリア**:` 行を読み取って LLM に
        提示することで、判定精度を改善する (実走で 'AI 倫理' が候補なし判定された問題対応)。
        """
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")
        system_prompt = mock_llm.call_args.args[1]
        # 各キャラの担当エリアキーワードがプロンプトに含まれる (md 内容に依存)
        # mimi: 抽象 × 感情（美学・価値観・ノブレスオブリージュ）
        # chisame: 具体 × 論理（データ・根拠・実行計画）
        # sakura: 具体 × 感情（感情フォロー・心理安全・倫理的配慮）
        assert "担当" in system_prompt
        assert "美学" in system_prompt or "ノブレスオブリージュ" in system_prompt  # mimi
        assert "データ" in system_prompt or "実行計画" in system_prompt  # chisame
        assert "感情フォロー" in system_prompt or "倫理的配慮" in system_prompt  # sakura

    def test_prompt_has_judgment_criteria(self):
        """プロンプトに判定基準 (関連性 / callout 経路への譲渡 / 無関係雑談除外) が含まれる。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")
        system_prompt = mock_llm.call_args.args[1]
        # Phase 0.5-A フェーズ 8: 「直接触れる」を「触れている / 関連している」に緩和
        assert "触れている" in system_prompt or "関連している" in system_prompt
        # 呼びかけ済み時の callout 経路への譲渡
        assert "呼びかけ" in system_prompt
        # 無関係な雑談は除外
        assert "雑談" in system_prompt or "無関係" in system_prompt

    def test_prompt_includes_few_shot_example(self):
        """Phase 0.5-A フェーズ 8: プロンプトに具体例 (AI 倫理 → sakura/mimi 等) が含まれる。

        LLM の保守的判定 (具体性が無いと none を返す傾向) を抑制するため、
        実走で問題になった「AI 倫理について気になっている」のような問題提起を
        few-shot 例として明示する。
        """
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")
        system_prompt = mock_llm.call_args.args[1]
        # AI 倫理の例 (sakura / mimi がマッチする想定)
        assert "AI倫理" in system_prompt or "倫理" in system_prompt
        # 例として「データ」または「論理」キャラの例も含まれる
        assert "データ" in system_prompt or "論理" in system_prompt

    def test_prompt_strict_output_format(self):
        """プロンプトに出力形式厳守 (装飾文字 / 改行 / 説明禁止) が明示される。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_intent("テスト")
        system_prompt = mock_llm.call_args.args[1]
        # 「厳守」「禁止」「一切付けない」等の強い表現で出力形式を縛る
        assert "厳守" in system_prompt or "一切" in system_prompt
        # 装飾文字の例示
        assert "**" in system_prompt or "装飾" in system_prompt

    def test_parses_first_token_when_llm_adds_explanation(self):
        """Phase 0.5-A フェーズ 8 実走対応: LLM が 'none\\n\\n**理由**: ...' で返しても
        最初のトークン 'none' で判定する (パース強化)。
        """
        from lab_lounge.router import check_intent
        # 実走で観測された LLM 応答パターン
        with patch(
            "lab_lounge.router._call_router_llm",
            return_value="none\n\n**理由**: 発話が問題提起の段階で、具",
        ):
            result = check_intent("最近のAI倫理について気になっている")
        assert result.intent == "unknown"
        assert result.target_slug is None

    def test_parses_slug_with_explanation_suffix(self):
        """LLM が 'sakura\\n\\n理由: 倫理的配慮を担当' で返したら sakura で判定する。"""
        from lab_lounge.router import check_intent
        with patch(
            "lab_lounge.router._call_router_llm",
            return_value="sakura\n\n理由: 倫理的配慮を担当",
        ):
            result = check_intent("最近のAI倫理について")
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "sakura"

    def test_parses_slug_with_punctuation_suffix(self):
        """LLM が 'mimi.' や 'mimi。' で返しても末尾句読点を除去して判定する。"""
        from lab_lounge.router import check_intent
        # 半角ピリオド
        with patch("lab_lounge.router._call_router_llm", return_value="mimi."):
            result = check_intent("テスト")
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "mimi"
        # 全角句点
        with patch("lab_lounge.router._call_router_llm", return_value="chisame。"):
            result = check_intent("テスト")
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "chisame"
        # markdown 装飾
        with patch("lab_lounge.router._call_router_llm", return_value="**sakura**"):
            result = check_intent("テスト")
        assert result.intent == "interjection_candidate"
        assert result.target_slug == "sakura"

    def test_parses_empty_response_as_unknown(self):
        """空文字 / 空白のみの応答は unknown (fail-open)。"""
        from lab_lounge.router import check_intent
        with patch("lab_lounge.router._call_router_llm", return_value=""):
            result = check_intent("テスト")
        assert result.intent == "unknown"
        with patch("lab_lounge.router._call_router_llm", return_value="   \n\n  "):
            result = check_intent("テスト")
        assert result.intent == "unknown"


class TestLoadCharacterInterestArea:
    """Phase 0.5-A フェーズ 8: skills/characters/<slug>.md からの担当エリア抽出。"""

    def test_returns_mimi_interest_area(self):
        """mimi.md から担当エリアを抽出。"""
        from lab_lounge.router import _load_character_interest_area
        result = _load_character_interest_area("mimi")
        assert result is not None
        # 担当エリア: 抽象 × 感情（美学・価値観・ノブレスオブリージュ）
        assert "美学" in result or "ノブレスオブリージュ" in result

    def test_returns_chisame_interest_area(self):
        """chisame.md から担当エリアを抽出。"""
        from lab_lounge.router import _load_character_interest_area
        result = _load_character_interest_area("chisame")
        assert result is not None
        assert "論理" in result or "データ" in result

    def test_returns_sakura_interest_area(self):
        """sakura.md から担当エリアを抽出。"""
        from lab_lounge.router import _load_character_interest_area
        result = _load_character_interest_area("sakura")
        assert result is not None
        assert "感情" in result or "倫理" in result

    def test_returns_none_for_unknown_slug(self):
        """skills/characters/<slug>.md が無い slug は None。"""
        from lab_lounge.router import _load_character_interest_area
        assert _load_character_interest_area("nonexistent_slug") is None


class TestCheckApproval:
    """check_approval() の挙動 (Phase 0.5-A で導入)。

    handraising 中のキャラに対して、ルカの発話を承認/却下/関係なしに分類する。
    """

    def test_returns_granted_for_approval(self):
        """LLM が 'granted:mimi' を返したら ApprovalResult(granted=True)。"""
        with patch("lab_lounge.router._call_router_llm", return_value="granted:mimi"):
            result = check_approval("ミミ、どうぞ", ["mimi"])
        assert result is not None
        assert result.granted is True
        assert result.target_slug == "mimi"
        assert result.confidence == 1.0

    def test_returns_denied_for_denial(self):
        """LLM が 'denied:chisame' を返したら ApprovalResult(granted=False)。"""
        with patch("lab_lounge.router._call_router_llm", return_value="denied:chisame"):
            result = check_approval("いや、いいわ", ["chisame"])
        assert result is not None
        assert result.granted is False
        assert result.target_slug == "chisame"

    def test_returns_none_for_unrelated(self):
        """LLM が 'none' を返したら None (関係ない発話 → 通常処理へ)。"""
        with patch("lab_lounge.router._call_router_llm", return_value="none"):
            result = check_approval("今日はいい天気だね", ["mimi"])
        assert result is None

    def test_returns_none_on_llm_error(self):
        """LLM 失敗時 None (fail-open: 通常処理へ流す)。"""
        with patch("lab_lounge.router._call_router_llm", return_value=None):
            result = check_approval("テスト", ["mimi"])
        assert result is None

    def test_returns_none_on_unknown_slug(self):
        """LLM が候補外 slug を返したら None。"""
        with patch("lab_lounge.router._call_router_llm", return_value="granted:unknown_char"):
            result = check_approval("テスト", ["mimi"])
        assert result is None

    def test_returns_none_on_invalid_format(self):
        """LLM が予期しない形式の応答を返したら None。"""
        with patch("lab_lounge.router._call_router_llm", return_value="maybe"):
            result = check_approval("テスト", ["mimi"])
        assert result is None

    def test_returns_none_when_candidates_empty(self):
        """candidate_slugs が空なら LLM 呼ばずに None。"""
        with patch("lab_lounge.router._call_router_llm") as mock_llm:
            result = check_approval("テスト", [])
        assert result is None
        mock_llm.assert_not_called()

    def test_prompt_includes_candidate_slugs(self):
        """system_prompt に candidate_slugs が全て含まれる (キャラ名と共に)。"""
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_approval("テスト", ["mimi", "chisame"])
        system_prompt = mock_llm.call_args.args[1]
        assert "mimi" in system_prompt
        assert "chisame" in system_prompt

    def test_handles_multiple_candidates(self):
        """複数挙手連鎖時、複数候補から正しい slug を選ぶ。"""
        with patch("lab_lounge.router._call_router_llm", return_value="granted:sakura"):
            result = check_approval(
                "さくら、どうぞ", ["mimi", "chisame", "sakura"]
            )
        assert result is not None
        assert result.granted is True
        assert result.target_slug == "sakura"

    def test_text_passed_as_user_text(self):
        """text が _call_router_llm の user_text として渡される。"""
        text = "ミミ、どうぞ"
        with patch("lab_lounge.router._call_router_llm", return_value="granted:mimi") as mock_llm:
            check_approval(text, ["mimi"])
        assert mock_llm.call_args.args[2] == text

    def test_uses_intent_gate_model_env(self, monkeypatch):
        """L2_INTENT_GATE_MODEL を流用する (新規環境変数を増やさない)。"""
        monkeypatch.setenv("L2_INTENT_GATE_MODEL", "gpt-5.4-nano")
        with patch("lab_lounge.router._call_router_llm", return_value="none") as mock_llm:
            check_approval("テスト", ["mimi"])
        assert mock_llm.call_args.args[0] == "gpt-5.4-nano"
