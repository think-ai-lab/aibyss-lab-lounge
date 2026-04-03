"""
test_speech_activated.py — SpeechActivatedListener のテスト

sounddevice / soundfile をモックし、VAD + STT ルーティングを検証する。
numpy は実際の RMS 計算に必要なため実物を使用する。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lab_lounge import stt as _stt
from lab_lounge.stt import STTResult
from lab_lounge.wake_word import (
    SpeechActivatedListener,
    _DEFAULT_SILENCE_FRAMES,
    _DEFAULT_VAD_HOLD_FRAMES,
)


# ─── テスト用フレームヘルパー ────────────────────────────────────────

def _loud_frame(n: int = 512) -> tuple:
    """RMS > 0.01 のフレーム (value=800 → RMS ≈ 0.024)"""
    arr = np.full((n, 1), 800, dtype=np.int16)
    return (arr, False)


def _silent_frame(n: int = 512) -> tuple:
    """RMS ≈ 0 のサイレントフレーム"""
    arr = np.zeros((n, 1), dtype=np.int16)
    return (arr, False)


def _detection_frames() -> list:
    """発話開始 → 録音 → 無音終了の標準フレームシーケンス。"""
    frames = []
    # Phase 1: onset (_DEFAULT_VAD_HOLD_FRAMES 個のラウドフレーム)
    for _ in range(_DEFAULT_VAD_HOLD_FRAMES):
        frames.append(_loud_frame())
    # Phase 2: 1 フレームの録音データ
    frames.append(_loud_frame())
    # Phase 2: _DEFAULT_SILENCE_FRAMES 個の無音 → 録音終了
    for _ in range(_DEFAULT_SILENCE_FRAMES):
        frames.append(_silent_frame())
    return frames


class _MockInputStream:
    """sounddevice.InputStream のシンプルなモック。"""

    def __init__(self, frames: list):
        self._frames = iter(frames)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n: int):
        try:
            return next(self._frames)
        except StopIteration:
            return (np.zeros((n, 1), dtype=np.int16), False)


def _make_sd_mock(frames: list) -> MagicMock:
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = _MockInputStream(frames)
    return mock_sd


def _stt_result(text: str) -> STTResult:
    return STTResult(text=text, confidence=None, lang="ja", duration_ms=1000)


# ─── TestSpeechActivatedListenerInit ────────────────────────────────

class TestSpeechActivatedListenerInit:
    def test_default_init(self, monkeypatch):
        monkeypatch.delenv("L2_SILENCE_THRESHOLD", raising=False)
        listener = SpeechActivatedListener()
        assert listener._vad_threshold == pytest.approx(0.01)
        assert listener._device is None
        assert listener._stt_provider == "openai"
        assert listener._stt_lang == "ja"

    def test_vad_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("L2_SILENCE_THRESHOLD", "0.05")
        listener = SpeechActivatedListener()
        assert listener._vad_threshold == pytest.approx(0.05)

    def test_explicit_threshold_overrides_env(self, monkeypatch):
        monkeypatch.setenv("L2_SILENCE_THRESHOLD", "0.05")
        listener = SpeechActivatedListener(vad_threshold=0.02)
        assert listener._vad_threshold == pytest.approx(0.02)

    def test_zero_threshold_not_overridden_by_env(self, monkeypatch):
        """vad_threshold=0.0 は明示指定なので env を上書きしない。"""
        monkeypatch.setenv("L2_SILENCE_THRESHOLD", "0.05")
        listener = SpeechActivatedListener(vad_threshold=0.0)
        assert listener._vad_threshold == pytest.approx(0.0)

    def test_cleanup_is_noop(self):
        listener = SpeechActivatedListener()
        listener.cleanup()  # 例外を発生させない


# ─── TestListenOnceCharacterDetection ───────────────────────────────

class TestListenOnceCharacterDetection:
    """キャラクター検出ロジックのテスト。"""

    def _run(self, transcript: str) -> "object | None":
        """共通ヘルパー: 指定した転写テキストで listen_once を実行する。"""
        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock(_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result(transcript)), \
             patch("time.monotonic", return_value=0.0):
            return listener.listen_once(timeout_seconds=30.0)

    def test_mimi_routing(self):
        result = self._run("ミミ様、今日の天気は？")
        assert result is not None
        assert result.character_slug == "mimi"

    def test_chisame_routing(self):
        result = self._run("ちさめさん、おはよう")
        assert result is not None
        assert result.character_slug == "chisame"

    def test_sakura_alias_routing(self):
        result = self._run("さくらさん、教えてください")
        assert result is not None
        assert result.character_slug == "sakura"

    def test_no_name_returns_none(self):
        """名前ゲート: キャラクター名なしは None を返す（応答しない）。"""
        result = self._run("今日の天気は？")
        assert result is None

    def test_transcript_field_populated(self):
        text = "ミミ様、こんにちは"
        result = self._run(text)
        assert result is not None
        assert result.transcript == text

    def test_keyword_index_is_zero(self):
        result = self._run("ミミ様、テスト")
        assert result is not None
        assert result.keyword_index == 0

    def test_result_is_wake_word_result(self):
        from lab_lounge.wake_word import WakeWordResult
        result = self._run("ちさめさん、こんにちは")
        assert isinstance(result, WakeWordResult)


# ─── TestListenOnceTimeout ──────────────────────────────────────────

class TestListenOnceTimeout:
    def test_all_silent_with_expired_deadline_returns_none(self):
        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock([_silent_frame() for _ in range(5)])
        mock_sf = MagicMock()

        # deadline = 0.0 + 30.0 = 30.0。次の monotonic() = 100.0 → ループ未実行 → None
        monotonic_values = [0.0] + [100.0] * 10
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("time.monotonic", side_effect=monotonic_values):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None

    def test_timeout_with_short_seconds(self):
        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock([_silent_frame() for _ in range(5)])
        mock_sf = MagicMock()

        monotonic_values = [0.0] + [5.0] * 10  # deadline = 0.0 + 1.0 = 1.0
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("time.monotonic", side_effect=monotonic_values):
            result = listener.listen_once(timeout_seconds=1.0)

        assert result is None


# ─── TestListenOnceEdgeCases ────────────────────────────────────────

class TestListenOnceEdgeCases:
    def test_empty_transcript_returns_none(self):
        """名前ゲート: 空の転写は名前なしとして None を返す。"""
        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock(_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("")), \
             patch("time.monotonic", return_value=0.0):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None

    def test_stt_exception_propagates(self):
        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock(_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", side_effect=RuntimeError("STT failed")), \
             patch("time.monotonic", return_value=0.0):
            with pytest.raises(RuntimeError, match="STT failed"):
                listener.listen_once(timeout_seconds=30.0)
