"""
test_characters.py — CharacterConfig レジストリのテスト
"""

import pytest

from lab_lounge.characters import (
    CHARACTER_REGISTRY,
    get_all_characters,
    get_character,
    get_default_character,
    load_system_prompt,
)


class TestGetCharacter:
    def test_known_slugs(self):
        for slug in ("mimi", "chisame", "sakura", "octamaid", "ruka"):
            config = get_character(slug)
            assert config.slug == slug

    def test_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="未知のキャラクター"):
            get_character("nonexistent")

    def test_character_has_display_name(self):
        config = get_character("mimi")
        assert config.display_name == "ミミ・オクタヴィア"

    def test_character_has_wake_word(self):
        config = get_character("mimi")
        assert config.wake_word == "ミミ様"

    def test_ruka_has_no_wake_word(self):
        config = get_character("ruka")
        assert config.wake_word is None

    def test_character_has_tts_config(self):
        config = get_character("octamaid")
        assert config.tts_provider == "voicevox"
        assert config.tts_voice == "89"


class TestGetDefaultCharacter:
    def test_default_is_octamaid(self):
        config = get_default_character()
        assert config.slug == "octamaid"

    def test_custom_default(self, monkeypatch):
        monkeypatch.setenv("L2_DEFAULT_SPEAKER", "mimi")
        config = get_default_character()
        assert config.slug == "mimi"

    def test_invalid_default_fallback(self, monkeypatch):
        monkeypatch.setenv("L2_DEFAULT_SPEAKER", "invalid_slug")
        config = get_default_character()
        assert config.slug == "octamaid"  # フォールバック


class TestGetAllCharacters:
    def test_returns_all(self):
        chars = get_all_characters()
        assert len(chars) == len(CHARACTER_REGISTRY)

    def test_slugs_match_registry(self):
        slugs = {c.slug for c in get_all_characters()}
        assert slugs == set(CHARACTER_REGISTRY.keys())


class TestLoadSystemPrompt:
    def test_load_existing_prompt(self):
        config = get_character("octamaid")
        prompt = load_system_prompt(config)
        assert "オクタメイド" in prompt
        assert len(prompt) > 100

    def test_load_mimi_prompt(self):
        config = get_character("mimi")
        prompt = load_system_prompt(config)
        assert "ミミ・オクタヴィア" in prompt

    def test_missing_prompt_raises(self):
        from lab_lounge.characters import CharacterConfig
        fake = CharacterConfig(
            slug="fake",
            display_name="Fake",
            wake_word=None,
            tts_provider="voicevox",
            tts_voice="0",
            system_prompt_file="nonexistent.txt",
        )
        with pytest.raises(FileNotFoundError):
            load_system_prompt(fake)
