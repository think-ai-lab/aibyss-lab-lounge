"""
test_sherpa_streaming.py — SherpaStreamingListener のテスト

sherpa_onnx / sounddevice をモックし、VAD + Sherpa 認識 + ルーティングを検証する。
"""

import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from lab_lounge.wake_word import (
    _DEFAULT_SILENCE_FRAMES,
    _DEFAULT_VAD_HOLD_FRAMES,
)


# ─── テスト用フレームヘルパー (float32) ────────────────────────────

def _loud_frame(n: int = 512) -> tuple:
    """RMS > 0.01 のフレーム (float32, value≈0.025)"""
    arr = np.full((n, 1), 0.025, dtype=np.float32)
    return (arr, False)


def _silent_frame(n: int = 512) -> tuple:
    """RMS ≈ 0 のサイレントフレーム (float32)"""
    arr = np.zeros((n, 1), dtype=np.float32)
    return (arr, False)


def _detection_frames() -> list:
    """発話開始 → 録音 → 無音終了の標準フレームシーケンス。"""
    frames = []
    for _ in range(_DEFAULT_VAD_HOLD_FRAMES):
        frames.append(_loud_frame())
    frames.append(_loud_frame())
    for _ in range(_DEFAULT_SILENCE_FRAMES):
        frames.append(_silent_frame())
    return frames


class _MockInputStream:
    """sounddevice.InputStream のモック。"""

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
            return (np.zeros((n, 1), dtype=np.float32), False)


def _make_sd_mock(frames: list) -> MagicMock:
    mock_sd = MagicMock()
    mock_sd.InputStream.return_value = _MockInputStream(frames)
    return mock_sd


def _make_sherpa_mock(transcript: str = "") -> MagicMock:
    """sherpa_onnx モジュールのモックを生成する。"""
    mock_sherpa = MagicMock()

    # stream.result.text を返すモック
    mock_stream = MagicMock()
    mock_result = MagicMock()
    mock_result.text = transcript
    mock_stream.result = mock_result

    # recognizer のモック
    mock_recognizer = MagicMock()
    mock_recognizer.create_stream.return_value = mock_stream
    mock_sherpa.OfflineRecognizer.from_transducer.return_value = mock_recognizer

    return mock_sherpa, mock_recognizer


def _make_model_files(tmp_path: Path) -> Path:
    """ダミーのモデルファイルを作成し、model_dir を返す。"""
    for name in (
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "tokens.txt",
    ):
        (tmp_path / name).touch()
    return tmp_path


# ─── TestSherpaStreamingListenerInit ──────────────────────────────

class TestSherpaStreamingListenerInit:
    def test_default_init(self, monkeypatch, tmp_path):
        monkeypatch.delenv("L2_SILENCE_THRESHOLD", raising=False)
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, _ = _make_sherpa_mock()

        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(model_dir=str(model_dir))

        assert listener._vad_threshold == pytest.approx(0.01)
        assert listener._device is None
        mock_sherpa.OfflineRecognizer.from_transducer.assert_called_once()

    def test_provider_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("L2_SHERPA_PROVIDER", "cpu")
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, _ = _make_sherpa_mock()

        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(model_dir=str(model_dir))

        call_kwargs = mock_sherpa.OfflineRecognizer.from_transducer.call_args
        assert call_kwargs.kwargs["provider"] == "cpu"

    def test_missing_model_raises(self, tmp_path):
        mock_sherpa, _ = _make_sherpa_mock()

        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from lab_lounge.wake_word import SherpaStreamingListener
            with pytest.raises(FileNotFoundError, match="Sherpa モデル"):
                SherpaStreamingListener(model_dir=str(tmp_path))

    def test_cleanup(self, tmp_path):
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, _ = _make_sherpa_mock()

        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(model_dir=str(model_dir))
            listener.cleanup()
            assert listener._recognizer is None


# ─── TestSherpaListenOnce ─────────────────────────────────────────

class TestSherpaListenOnce:
    def _run(self, transcript: str, tmp_path: Path) -> "object | None":
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, mock_recognizer = _make_sherpa_mock(transcript)
        mock_sd = _make_sd_mock(_detection_frames())

        with patch.dict(sys.modules, {
            "sherpa_onnx": mock_sherpa,
            "sounddevice": mock_sd,
        }):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(
                model_dir=str(model_dir),
                vad_threshold=0.005,
            )
            # recognizer を差し替え (init で作成済みを上書き)
            listener._recognizer = mock_recognizer

            with patch("time.monotonic", return_value=0.0):
                return listener.listen_once(timeout_seconds=30.0)

    def test_mimi_routing(self, tmp_path):
        result = self._run("ミミ様、今日の天気は？", tmp_path)
        assert result is not None
        assert result.character_slug == "mimi"
        assert result.transcript == "ミミ様、今日の天気は？"

    def test_chisame_routing(self, tmp_path):
        result = self._run("ちさめさん、おはよう", tmp_path)
        assert result is not None
        assert result.character_slug == "chisame"

    def test_sakura_routing(self, tmp_path):
        result = self._run("さくらさん、教えてください", tmp_path)
        assert result is not None
        assert result.character_slug == "sakura"

    def test_no_name_returns_none(self, tmp_path):
        """名前ゲート: キャラクター名なしは None を返す。"""
        result = self._run("今日の天気は？", tmp_path)
        assert result is None

    def test_empty_transcript_returns_none(self, tmp_path):
        """名前ゲート: 空の転写は None を返す。"""
        result = self._run("", tmp_path)
        assert result is None

    def test_result_has_transcript(self, tmp_path):
        result = self._run("ミミ様、テスト", tmp_path)
        assert result is not None
        assert result.transcript == "ミミ様、テスト"
        assert result.keyword_index == 0


# ─── TestSherpaListenOnceTimeout ──────────────────────────────────

class TestSherpaListenOnceTimeout:
    def test_timeout_returns_none(self, tmp_path):
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, _ = _make_sherpa_mock()
        mock_sd = _make_sd_mock([_silent_frame() for _ in range(5)])

        with patch.dict(sys.modules, {
            "sherpa_onnx": mock_sherpa,
            "sounddevice": mock_sd,
        }):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(
                model_dir=str(model_dir),
                vad_threshold=0.005,
            )

            monotonic_values = [0.0] + [100.0] * 10
            with patch("time.monotonic", side_effect=monotonic_values):
                result = listener.listen_once(timeout_seconds=30.0)

        assert result is None


# ─── TestSherpaRecognizerInteraction ──────────────────────────────

class TestSherpaRecognizerInteraction:
    def test_audio_fed_to_recognizer(self, tmp_path):
        """フレームデータが recognizer に渡されることを確認。"""
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, mock_recognizer = _make_sherpa_mock("ミミ様、テスト")
        mock_sd = _make_sd_mock(_detection_frames())
        mock_stream = mock_recognizer.create_stream.return_value

        with patch.dict(sys.modules, {
            "sherpa_onnx": mock_sherpa,
            "sounddevice": mock_sd,
        }):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(
                model_dir=str(model_dir),
                vad_threshold=0.005,
            )
            listener._recognizer = mock_recognizer

            with patch("time.monotonic", return_value=0.0):
                listener.listen_once(timeout_seconds=30.0)

        # accept_waveform が呼ばれたこと
        mock_stream.accept_waveform.assert_called_once()
        call_args = mock_stream.accept_waveform.call_args
        assert call_args[0][0] == 16000  # sample_rate

        # decode_stream が呼ばれたこと
        mock_recognizer.decode_stream.assert_called_once_with(mock_stream)

    def test_no_wav_file_created(self, tmp_path):
        """WAV ファイルが作成されないことを確認。"""
        model_dir = _make_model_files(tmp_path)
        mock_sherpa, mock_recognizer = _make_sherpa_mock("ミミ様、テスト")
        mock_sd = _make_sd_mock(_detection_frames())

        with patch.dict(sys.modules, {
            "sherpa_onnx": mock_sherpa,
            "sounddevice": mock_sd,
        }):
            from lab_lounge.wake_word import SherpaStreamingListener
            listener = SherpaStreamingListener(
                model_dir=str(model_dir),
                vad_threshold=0.005,
            )
            listener._recognizer = mock_recognizer

            with patch("time.monotonic", return_value=0.0):
                listener.listen_once(timeout_seconds=30.0)

        # tmp_path にモデルファイル以外のファイルがないことを確認
        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 0
