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


# ═══════════════════════════════════════════════════════════════════
# WebRTC VAD 統合テスト
# ═══════════════════════════════════════════════════════════════════


def _webrtc_loud_frame(n: int = 320) -> tuple:
    """WebRTC 用 320 サンプルの「発話あり」フレーム。"""
    arr = np.full((n, 1), 800, dtype=np.int16)
    return (arr, False)


def _webrtc_silent_frame(n: int = 320) -> tuple:
    """WebRTC 用 320 サンプルの「無音」フレーム。"""
    arr = np.zeros((n, 1), dtype=np.int16)
    return (arr, False)


def _webrtc_detection_frames() -> list:
    """WebRTC 経路用の発話 → 無音シーケンス。

    is_speech=True を返すモック VAD を使う前提なので、各フレームの実値は
    意味を持たない (モックが固定値を返す)。フレーム数だけが重要。
    """
    from lab_lounge.wake_word import (
        _DEFAULT_VAD_HOLD_FRAMES,
        _WEBRTC_SILENCE_FRAMES,
    )
    frames = []
    # onset
    for _ in range(_DEFAULT_VAD_HOLD_FRAMES):
        frames.append(_webrtc_loud_frame())
    # 1 録音フレーム
    frames.append(_webrtc_loud_frame())
    # 無音終了 (WebRTC 経路は 75 silence frames)
    for _ in range(_WEBRTC_SILENCE_FRAMES):
        frames.append(_webrtc_silent_frame())
    return frames


class TestSpeechActivatedListenerWebRTC:
    """L2_VAD_BACKEND=webrtc 設定下での SpeechActivatedListener 検証。"""

    def _make_mock_webrtcvad(self, *, is_speech_pattern: list[bool] | None = None):
        """webrtcvad モジュールのモックを返す。

        is_speech_pattern を指定すると、_check_webrtc 呼び出しごとに順番に True/False を返す。
        最後まで使い切ったら最後の値を繰り返す。
        """
        mock_vad_instance = MagicMock()
        if is_speech_pattern:
            iterator = iter(is_speech_pattern)
            last = [is_speech_pattern[-1]]

            def side_effect(*args, **kwargs):
                try:
                    val = next(iterator)
                    last[0] = val
                    return val
                except StopIteration:
                    return last[0]

            mock_vad_instance.is_speech.side_effect = side_effect
        else:
            mock_vad_instance.is_speech.return_value = True

        mock_webrtcvad = MagicMock()
        mock_webrtcvad.Vad = MagicMock(return_value=mock_vad_instance)
        return mock_webrtcvad

    def test_listen_once_uses_320_blocksize(self, monkeypatch):
        """L2_VAD_BACKEND=webrtc で sd.InputStream に blocksize=320 が渡される。"""
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")

        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock(_webrtc_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"
        mock_webrtcvad = self._make_mock_webrtcvad(
            # onset 8 + 1 録音 で True、その後 False (無音)
            is_speech_pattern=[True] * 9 + [False],
        )

        with patch.dict(sys.modules, {
            "sounddevice": mock_sd,
            "soundfile": mock_sf,
            "webrtcvad": mock_webrtcvad,
        }), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("ミミ様、テスト")), \
             patch("time.monotonic", return_value=0.0):
            listener.listen_once(timeout_seconds=30.0)

        # sd.InputStream が blocksize=320 で呼ばれた
        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args.kwargs
        assert call_kwargs["blocksize"] == 320

    def test_listen_once_uses_webrtc_silence_frames(self, monkeypatch):
        """75 silence frames で停止判定 (WebRTC モード)。

        77 frames 用意 (8 onset + 1 録音 + 75 silence + 余白) して止まることを確認。
        """
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")

        listener = SpeechActivatedListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock(_webrtc_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"
        mock_webrtcvad = self._make_mock_webrtcvad(
            # 8 onset + 1 録音 = 9 True, 残りすべて False (無音)
            is_speech_pattern=[True] * 9 + [False] * 100,
        )

        with patch.dict(sys.modules, {
            "sounddevice": mock_sd,
            "soundfile": mock_sf,
            "webrtcvad": mock_webrtcvad,
        }), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("ミミ様、テスト")), \
             patch("time.monotonic", return_value=0.0):
            result = listener.listen_once(timeout_seconds=30.0)

        # is_speech() が呼ばれた回数: onset(8) + 録音(1) + 75 silence + 録音終了処理 = 84+
        mock_vad = mock_webrtcvad.Vad.return_value
        # 少なくとも 75 + 9 = 84 回呼ばれているはず
        assert mock_vad.is_speech.call_count >= 84
        # 結果が成立 (mimi 検出)
        assert result is not None
        assert result.character_slug == "mimi"

    def test_listen_once_falls_back_to_rms_when_webrtcvad_missing(self, monkeypatch):
        """L2_VAD_BACKEND=webrtc でも webrtcvad import 失敗時は RMS にフォールバックする。"""
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")

        listener = SpeechActivatedListener(vad_threshold=0.005)
        # フォールバック後は 512 サンプルの既存フレームを使う
        mock_sd = _make_sd_mock(_detection_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        # webrtcvad import を失敗させる
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "webrtcvad":
                raise ImportError("mocked: webrtcvad not installed")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch.object(builtins, "__import__", side_effect=mock_import), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("ミミ様、テスト")), \
             patch("time.monotonic", return_value=0.0):
            result = listener.listen_once(timeout_seconds=30.0)

        # フォールバック後は RMS の 512 blocksize
        mock_sd.InputStream.assert_called_once()
        assert mock_sd.InputStream.call_args.kwargs["blocksize"] == 512
        # 動作は正常 (mimi 検出)
        assert result is not None
        assert result.character_slug == "mimi"
