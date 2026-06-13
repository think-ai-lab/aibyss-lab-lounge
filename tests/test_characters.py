"""
test_characters.py — CharacterConfig レジストリのテスト
"""

import pytest

from lab_lounge.characters import (
    CHARACTER_REGISTRY,
    _PROMPTS_DIR,
    get_all_characters,
    get_character,
    get_default_character,
    load_system_prompt,
)


def _real_prompt_exists(filename: str) -> bool:
    """実キャラ system prompt がローカルに在るか (公開リポでは untrack されるため
    fresh clone では不在。本文 assert はその場合スキップする)。"""
    return (_PROMPTS_DIR / filename).is_file()


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
    @pytest.mark.skipif(
        not _real_prompt_exists("system_octamaid.txt"),
        reason="実キャラ prompt はローカル限定 (公開リポでは untrack 済み)",
    )
    def test_load_existing_prompt(self):
        config = get_character("octamaid")
        prompt = load_system_prompt(config)
        assert "オクタメイド" in prompt
        assert len(prompt) > 100

    @pytest.mark.skipif(
        not _real_prompt_exists("system_mimi.txt"),
        reason="実キャラ prompt はローカル限定 (公開リポでは untrack 済み)",
    )
    def test_load_mimi_prompt(self):
        config = get_character("mimi")
        prompt = load_system_prompt(config)
        assert "ミミ・オクタヴィア" in prompt

    def test_missing_prompt_raises(self):
        # 実キャラもサンプルも無い場合のみ FileNotFoundError (サンプルも不在に設定)
        from lab_lounge.characters import CharacterConfig
        fake = CharacterConfig(
            slug="fake",
            display_name="Fake",
            wake_word=None,
            tts_provider="voicevox",
            tts_voice="0",
            system_prompt_file="nonexistent.txt",
            sample_fallback="nonexistent_sample.md",
        )
        with pytest.raises(FileNotFoundError):
            load_system_prompt(fake)

    def test_falls_back_to_sample_when_real_absent(self):
        """実キャラ prompt 不在 → samples/<sample_fallback> が返る (fresh clone での起動可能性)。"""
        from lab_lounge.characters import CharacterConfig
        cfg = CharacterConfig(
            slug="x",
            display_name="X",
            wake_word=None,
            tts_provider="voicevox",
            tts_voice="0",
            system_prompt_file="definitely_absent_real_prompt.txt",
            sample_fallback="sample_logic.md",
        )
        prompt = load_system_prompt(cfg)
        assert "サンプルキャラクター" in prompt
        assert len(prompt) > 100

    def test_bundled_samples_exist_with_json_contract(self):
        """同梱サンプル2体が存在し、応答 JSON 契約 (response/speed/pose) を持つ
        (公開リポでの起動を保証する不変条件)。"""
        for name in ("sample_logic.md", "sample_empath.md"):
            p = _PROMPTS_DIR / "samples" / name
            assert p.is_file(), f"サンプル {name} が無い"
            text = p.read_text(encoding="utf-8")
            assert '"response"' in text and '"speed"' in text and '"pose"' in text
            assert "ask_character" in text  # 委譲バイアスの実演
