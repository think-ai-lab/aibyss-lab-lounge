"""
test_router.py — 発話ルーティングのテスト
"""

import pytest

from lab_lounge.router import RoutingDecision, route


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
