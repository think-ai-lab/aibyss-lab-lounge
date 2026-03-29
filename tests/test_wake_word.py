"""
test_wake_word.py — ウェイクワード検知のテスト

pvporcupine をモックして、キャラクターマッピングとエラーハンドリングを検証する。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.wake_word import (
    WakeWordResult,
    _find_keyword_paths,
    _get_platform_suffix,
)


class TestFindKeywordPaths:
    def test_finds_existing_models(self):
        """porcupine ディレクトリに .ppn ファイルがあればリストに含まれる。"""
        from lab_lounge.characters import get_all_characters

        model_dir = Path(__file__).resolve().parents[1] / "porcupine"
        if not model_dir.is_dir():
            pytest.skip("porcupine ディレクトリが見つかりません")

        chars = get_all_characters()
        entries = _find_keyword_paths(chars, model_dir)
        slugs = [c.slug for c, _ in entries]

        # 少なくとも octamaid は存在するはず
        assert "octamaid" in slugs

    def test_skips_missing_models(self, tmp_path):
        """存在しないディレクトリではモデルが見つからない。"""
        from lab_lounge.characters import get_all_characters

        chars = get_all_characters()
        entries = _find_keyword_paths(chars, tmp_path)
        assert len(entries) == 0

    def test_ruka_has_no_model(self):
        """ruka は porcupine_model=None なので常にスキップ。"""
        from lab_lounge.characters import get_all_characters

        model_dir = Path(__file__).resolve().parents[1] / "porcupine"
        if not model_dir.is_dir():
            pytest.skip("porcupine ディレクトリが見つかりません")

        chars = get_all_characters()
        entries = _find_keyword_paths(chars, model_dir)
        slugs = [c.slug for c, _ in entries]
        assert "ruka" not in slugs


class TestGetPlatformSuffix:
    def test_returns_string(self):
        suffix = _get_platform_suffix()
        assert suffix in ("windows", "mac", "linux")


class TestPorcupineListenerInit:
    def test_missing_access_key_raises(self, monkeypatch):
        """アクセスキーなしで ValueError。"""
        monkeypatch.delenv("L2_PORCUPINE_ACCESS_KEY", raising=False)
        from lab_lounge.wake_word import PorcupineListener

        with pytest.raises((ValueError, ImportError)):
            PorcupineListener(access_key="")


class TestWakeWordResult:
    def test_dataclass(self):
        result = WakeWordResult(
            keyword="ミミ様",
            character_slug="mimi",
            keyword_index=0,
        )
        assert result.keyword == "ミミ様"
        assert result.character_slug == "mimi"
        assert result.keyword_index == 0


class TestWakeWordResultTranscript:
    def test_transcript_defaults_to_none(self):
        result = WakeWordResult(keyword="ミミ様", character_slug="mimi", keyword_index=0)
        assert result.transcript is None

    def test_transcript_can_be_set(self):
        result = WakeWordResult("ミミ様", "mimi", 0, transcript="ミミ様、こんにちは")
        assert result.transcript == "ミミ様、こんにちは"

    def test_frozen_dataclass_immutable(self):
        result = WakeWordResult("ミミ様", "mimi", 0)
        with pytest.raises(Exception):
            result.transcript = "x"  # type: ignore[misc]
