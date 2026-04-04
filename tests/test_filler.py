"""
test_filler.py — フィラー音声のテスト
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.filler import (
    FillerPhrase,
    FillerPhraseSet,
    _generate_filler_text,
    _parse_emotion,
    get_cached_filler_paths,
    get_filler_duration_ms,
    is_filler_enabled,
    load_filler_phrases,
    run_filler_loop,
    select_filler_path,
)


# ─── TestParseEmotion ─────────────────────────────────────────────

class TestParseEmotion:
    def test_single_value(self):
        assert _parse_emotion("happy=80") == {"happy": 80}

    def test_multiple_values(self):
        result = _parse_emotion("happy=80,sad=0,angry=20")
        assert result == {"happy": 80, "sad": 0, "angry": 20}

    def test_empty_string(self):
        assert _parse_emotion("") == {}

    def test_invalid_format_skipped(self):
        assert _parse_emotion("invalid") == {}


# ─── TestFillerPhraseSet ──────────────────────────────────────────

class TestFillerPhraseSet:
    def test_empty_set(self):
        ps = FillerPhraseSet()
        assert len(ps) == 0
        assert ps.all_phrases == []

    def test_len_counts_all_categories(self):
        ps = FillerPhraseSet(
            opener=[FillerPhrase("a"), FillerPhrase("b")],
            continue_=[FillerPhrase("c")],
            closer=[FillerPhrase("d")],
        )
        assert len(ps) == 4

    def test_all_phrases_preserves_category(self):
        ps = FillerPhraseSet(
            opener=[FillerPhrase("op")],
            continue_=[FillerPhrase("cont")],
            closer=[FillerPhrase("cl")],
        )
        items = ps.all_phrases
        assert items[0] == ("opener", FillerPhrase("op"))
        assert items[1] == ("continue", FillerPhrase("cont"))
        assert items[2] == ("closer", FillerPhrase("cl"))


# ─── TestLoadFillerPhrases ────────────────────────────────────────

class TestLoadFillerPhrases:
    def test_load_existing_phrases(self):
        result = load_filler_phrases("mimi")
        assert isinstance(result, FillerPhraseSet)
        assert len(result) >= 8  # opener 4+ continue 4+

    def test_load_nonexistent_slug_returns_empty(self):
        result = load_filler_phrases("nonexistent_character")
        assert isinstance(result, FillerPhraseSet)
        assert len(result) == 0

    def test_all_characters_have_phrases(self):
        for slug in ("mimi", "chisame", "sakura", "octamaid"):
            result = load_filler_phrases(slug)
            assert len(result.opener) >= 3, f"{slug} の opener が少なすぎます"
            assert len(result.continue_) >= 5, f"{slug} の continue が少なすぎます"

    def test_opener_and_continue_separated(self):
        result = load_filler_phrases("mimi")
        assert len(result.opener) >= 3
        assert len(result.continue_) >= 5

    def test_mimi_has_emotion(self):
        result = load_filler_phrases("mimi")
        all_phrases = [p for _, p in result.all_phrases]
        with_emotion = [p for p in all_phrases if p.emotion is not None]
        assert len(with_emotion) >= 3

    def test_octamaid_no_emotion(self):
        result = load_filler_phrases("octamaid")
        for _, p in result.all_phrases:
            assert p.emotion is None

    def test_legacy_file_without_sections(self, tmp_path):
        """セクションヘッダーなしのファイル → 全て opener に入る。"""
        legacy = tmp_path / "legacy.txt"
        legacy.write_text("フレーズA\nフレーズB\n", encoding="utf-8")

        with patch("lab_lounge.filler._FILLER_PHRASES_DIR", tmp_path):
            result = load_filler_phrases("legacy")

        assert len(result.opener) == 2
        assert len(result.continue_) == 0

    def test_emotion_parsed_correctly(self):
        result = load_filler_phrases("mimi")
        all_phrases = [p for _, p in result.all_phrases]
        first_with_emotion = next(p for p in all_phrases if p.emotion)
        assert "happy" in first_with_emotion.emotion
        assert isinstance(first_with_emotion.emotion["happy"], int)


# ─── TestIsFillerEnabled ──────────────────────────────────────────

class TestIsFillerEnabled:
    def test_default_is_true(self):
        assert is_filler_enabled() is True

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_FILLER_ENABLED", "false")
        assert is_filler_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_FILLER_ENABLED", "true")
        assert is_filler_enabled() is True


# ─── TestSelectFillerPath ─────────────────────────────────────────

class TestSelectFillerPath:
    def test_nonexistent_slug_returns_none(self):
        path, idx = select_filler_path("nonexistent", "opener")
        assert path is None
        assert idx == -1

    def test_returns_valid_path_from_cache(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        for i in range(3):
            (cache_dir / f"opener_{i:02d}.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            path, idx = select_filler_path("test_slug", "opener")
            assert path is not None
            assert path.name.startswith("opener_")
            assert 0 <= idx < 3

    def test_selects_correct_category(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            path, _ = select_filler_path("test_slug", "continue")
            assert path is not None
            assert path.name.startswith("continue_")

    def test_avoids_last_index(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        for i in range(3):
            (cache_dir / f"continue_{i:02d}.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            for _ in range(100):
                _, idx = select_filler_path("test_slug", "continue", last_index=1)
                assert idx != 1


# ─── TestGetFillerDurationMs ──────────────────────────────────────

class TestGetFillerDurationMs:
    def test_nonexistent_file_returns_zero(self, tmp_path):
        assert get_filler_duration_ms(tmp_path / "nope.wav") == 0

    def test_valid_wav_returns_duration(self, tmp_path):
        import wave
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)
        duration = get_filler_duration_ms(wav_path)
        assert duration == pytest.approx(1000, abs=10)


# ─── TestRunFillerLoop ────────────────────────────────────────────

class TestRunFillerLoop:
    def test_immediate_stop_exits(self, tmp_path):
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        stop.set()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            run_filler_loop("mimi", stop)

    def test_no_cache_exits(self):
        stop = threading.Event()
        with patch("lab_lounge.filler._FILLER_CACHE_DIR", Path("/nonexistent")):
            run_filler_loop("mimi", stop)

    def test_plays_opener_then_llm_continue(self, tmp_path):
        """opener → LLM 生成 continue の順序で再生される。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(str(path))
            return True

        mock_tts_result = MagicMock()
        mock_tts_result.audio_url = "file:///tmp/filler_runtime/test.wav"

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value="えーと、そうですわねぇ"), \
             patch("lab_lounge.tts.synthesize", return_value=mock_tts_result), \
             patch.object(Path, "unlink"), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        assert len(played) == 2
        assert "opener_00" in played[0]

    def test_llm_failure_falls_back_to_cached(self, tmp_path):
        """LLM 失敗 → 事前生成 continue にフォールバック。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(Path(path).name)
            return True

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value=None), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        assert len(played) == 2
        assert played[0].startswith("opener_")
        assert played[1].startswith("continue_")

    def test_opener_only_when_no_continue_and_no_llm(self, tmp_path):
        """continue なし + LLM 失敗 → opener のみ再生して終了。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(Path(path).name)
            return True

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value=None):
            run_filler_loop("mimi", stop)

        assert len(played) == 1
        assert played[0].startswith("opener_")

    def test_no_loop_after_llm_continue(self, tmp_path):
        """LLM continue は 1 回のみ。ループしない。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(str(path))
            return True

        mock_tts_result = MagicMock()
        mock_tts_result.audio_url = "file:///tmp/filler_runtime/test.wav"

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value="テスト"), \
             patch("lab_lounge.tts.synthesize", return_value=mock_tts_result), \
             patch.object(Path, "unlink"), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        # opener + LLM continue の計 2 回のみ（ループしない）
        assert len(played) == 2


# ─── TestGetCachedFillerPaths ─────────────────────────────────────

class TestGetCachedFillerPaths:
    def test_empty_dir_returns_empty_dict(self, tmp_path):
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert result == {"opener": [], "continue": [], "closer": []}

    def test_returns_categorized_paths(self, tmp_path):
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "opener_01.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert len(result["opener"]) == 2
            assert len(result["continue"]) == 1
            assert len(result["closer"]) == 0

    def test_ignores_old_naming(self, tmp_path):
        """旧形式の 00.wav は無視される。"""
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        (cache_dir / "00.wav").touch()
        (cache_dir / "opener_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert len(result["opener"]) == 1


# ─── TestGenerateFillerText ───────────────────────────────────────

def _make_openai_mock(response_text: str = "えーと、そうですわねぇ……"):
    """openai モジュールのモックを生成する。"""
    mock_openai = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = response_text
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_resp
    return mock_openai


class TestGenerateFillerText:
    def test_returns_text_on_success(self):
        mock_openai = _make_openai_mock("えーと、そうですわねぇ……")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result == "えーと、そうですわねぇ……"

    def test_returns_none_on_api_error(self):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result is None

    def test_returns_none_for_unknown_slug(self):
        # get_character() raises KeyError for unknown slugs
        result = _generate_filler_text("nonexistent_character")
        # _generate_filler_text catches the exception and returns None
        assert result is None

    def test_returns_none_on_empty_response(self):
        mock_openai = _make_openai_mock("")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result is None
