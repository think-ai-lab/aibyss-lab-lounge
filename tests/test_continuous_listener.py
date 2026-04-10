"""
test_continuous_listener.py — ContinuousListener のテスト

sounddevice / soundfile をモックし、常時文字起こし + バッファ蓄積 +
キャラクター名検出によるルーティングを検証する。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lab_lounge import stt as _stt
from lab_lounge.stt import STTResult
from lab_lounge.wake_word import (
    _DEFAULT_SILENCE_FRAMES,
    _DEFAULT_VAD_HOLD_FRAMES,
)


# ─── テスト用フレームヘルパー ────────────────────────────────────────

def _loud_frame(n: int = 512) -> tuple:
    arr = np.full((n, 1), 800, dtype=np.int16)
    return (arr, False)


def _silent_frame(n: int = 512) -> tuple:
    arr = np.zeros((n, 1), dtype=np.int16)
    return (arr, False)


def _one_utterance_frames() -> list:
    """1 発話分: onset + 1 loud + silence。"""
    frames = []
    for _ in range(_DEFAULT_VAD_HOLD_FRAMES):
        frames.append(_loud_frame())
    frames.append(_loud_frame())
    for _ in range(_DEFAULT_SILENCE_FRAMES):
        frames.append(_silent_frame())
    return frames


def _two_utterance_frames() -> list:
    """2 発話分: 第1発話 + 無音ギャップ + 第2発話。"""
    frames = _one_utterance_frames()
    # 無音ギャップ（onset リセット用）
    for _ in range(5):
        frames.append(_silent_frame())
    # 第2発話
    frames.extend(_one_utterance_frames())
    return frames


class _MockInputStream:
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


# ─── TestContinuousListenerInit ───────────────────────────────────

class TestContinuousListenerInit:
    def test_default_init(self, monkeypatch):
        monkeypatch.delenv("L2_SILENCE_THRESHOLD", raising=False)
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener()
        assert listener._vad_threshold == pytest.approx(0.01)
        assert listener._stt_provider == "faster-whisper"
        assert listener._device is None

    def test_stt_provider_default_is_faster_whisper(self):
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener()
        assert listener._stt_provider == "faster-whisper"

    def test_custom_context_window(self):
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(context_window_sec=60.0, context_max_chars=500)
        assert listener._buffer._window_sec == 60.0
        assert listener._buffer._max_chars == 500

    def test_cleanup_clears_buffer(self):
        from lab_lounge.wake_word import ContinuousListener
        from lab_lounge.transcript_buffer import TranscriptSegment
        listener = ContinuousListener()
        listener._buffer.add(TranscriptSegment(text="test", timestamp=0.0, duration_ms=100))
        assert len(listener._buffer) == 1
        listener.cleanup()
        assert len(listener._buffer) == 0


# ─── TestContinuousListenOnce ─────────────────────────────────────

class TestContinuousListenOnce:
    def _run_single(self, transcript: str):
        """1 セグメントで listen_once を実行する。"""
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result(transcript)), \
             patch("time.monotonic", return_value=0.0):
            return listener.listen_once(timeout_seconds=30.0)

    def test_name_in_first_segment_returns_result(self):
        result = self._run_single("ミミ様、今日の天気は？")
        assert result is not None
        assert result.character_slug == "mimi"

    def test_mimi_routing(self):
        result = self._run_single("ミミ様、テスト")
        assert result is not None
        assert result.character_slug == "mimi"

    def test_chisame_routing(self):
        result = self._run_single("ちさめさん、おはよう")
        assert result is not None
        assert result.character_slug == "chisame"

    def test_no_name_returns_none_on_timeout(self):
        """名前なしセグメントのみ → タイムアウト → None。"""
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        # 1回目の STT → 名前なし、2回目以降 → タイムアウト
        monotonic_calls = [0.0] * 50 + [100.0] * 10
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("今日はいい天気です")), \
             patch("time.monotonic", side_effect=monotonic_calls):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None

    def test_name_in_second_segment_returns_context(self):
        """核心テスト: 2セグメント目で名前検出 → 両方の文脈が含まれる。"""
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_two_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        # 1回目: 名前なし、2回目: 名前あり
        stt_results = [
            _stt_result("今日はいい天気ですね"),
            _stt_result("ミミ様はどう思いますか"),
        ]

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", side_effect=stt_results), \
             patch("time.monotonic", return_value=0.0):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is not None
        assert result.character_slug == "mimi"
        # 文脈に両方のセグメントが含まれる
        assert "今日はいい天気ですね" in result.transcript
        assert "ミミ様はどう思いますか" in result.transcript

    def test_buffer_cleared_after_successful_route(self):
        """名前検出後にバッファがクリアされる。"""
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("ミミ様、テスト")), \
             patch("time.monotonic", return_value=0.0):
            listener.listen_once(timeout_seconds=30.0)

        assert len(listener._buffer) == 0

    def test_transcript_field_contains_context(self):
        result = self._run_single("ちさめさん、AIについて教えて")
        assert result is not None
        assert "ちさめさん" in result.transcript


# ─── TestContinuousListenOnceTimeout ──────────────────────────────

class TestContinuousListenOnceTimeout:
    def test_timeout_returns_none(self):
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005)
        mock_sd = _make_sd_mock([_silent_frame() for _ in range(5)])
        mock_sf = MagicMock()

        monotonic_values = [0.0] + [100.0] * 10
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("time.monotonic", side_effect=monotonic_values):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None


# ─── TestContinuousListenOnceEdgeCases ────────────────────────────

class TestContinuousListenOnceEdgeCases:
    def test_empty_transcript_segment_skipped(self):
        """空の転写結果はバッファに追加されない。"""
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        monotonic_calls = [0.0] * 50 + [100.0] * 10
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("")), \
             patch("time.monotonic", side_effect=monotonic_calls):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None
        assert len(listener._buffer) == 0

    def test_stt_exception_propagates(self):
        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
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
# LLM 意図ゲート (呼び出しゲート Phase 2) 統合テスト
# ═══════════════════════════════════════════════════════════════════


class TestContinuousListenerIntentGate:
    """ContinuousListener と意図ゲートの統合テスト。"""

    def _run_single_with_env(self, transcript: str, env: dict | None = None, monkeypatch=None):
        """1 セグメントで listen_once を実行する（環境変数設定可能）。"""
        if env and monkeypatch:
            for k, v in env.items():
                monkeypatch.setenv(k, v)

        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result(transcript)), \
             patch("time.monotonic", return_value=0.0):
            return listener.listen_once(timeout_seconds=30.0)

    def test_intent_gate_disabled_skips_check(self, monkeypatch):
        """L2_USE_INTENT_GATE 未設定時は check_intent が呼ばれない。"""
        monkeypatch.delenv("L2_USE_INTENT_GATE", raising=False)
        with patch("lab_lounge.router.check_intent") as mock_check:
            result = self._run_single_with_env("ミミ様、テスト")
        assert result is not None
        assert result.character_slug == "mimi"
        mock_check.assert_not_called()

    def test_intent_gate_callout_returns_result(self, monkeypatch):
        """意図ゲートが 'callout' → 通常通り結果を返す。"""
        monkeypatch.setenv("L2_USE_INTENT_GATE", "true")
        with patch("lab_lounge.router.check_intent", return_value="callout") as mock_check:
            result = self._run_single_with_env("ねぇミミ様、これどう思う？")
        assert result is not None
        assert result.character_slug == "mimi"
        mock_check.assert_called_once()

    def test_intent_gate_unknown_proceeds_fail_open(self, monkeypatch):
        """意図ゲートが 'unknown' → fail-open で通常通り返す。"""
        monkeypatch.setenv("L2_USE_INTENT_GATE", "true")
        with patch("lab_lounge.router.check_intent", return_value="unknown") as mock_check:
            result = self._run_single_with_env("ミミ様、テスト")
        assert result is not None
        assert result.character_slug == "mimi"
        mock_check.assert_called_once()

    def test_intent_gate_mention_continues_loop(self, monkeypatch):
        """意図ゲートが 'mention' → バッファ保持で次のセグメント待ち → タイムアウトで None。"""
        monkeypatch.setenv("L2_USE_INTENT_GATE", "true")

        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_one_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        # 1回目: 言及と判定 → loop continue → タイムアウト
        monotonic_calls = [0.0] * 50 + [100.0] * 10
        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", return_value=_stt_result("ミミ様の仕組みはすごいですね")), \
             patch("lab_lounge.router.check_intent", return_value="mention"), \
             patch("time.monotonic", side_effect=monotonic_calls):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is None
        # バッファは保持されている（クリアされていない）
        assert len(listener._buffer) == 1

    def test_intent_gate_mention_then_callout(self, monkeypatch):
        """1 セグメント目 mention → 2 セグメント目 callout → バッファ累積で return。"""
        monkeypatch.setenv("L2_USE_INTENT_GATE", "true")

        from lab_lounge.wake_word import ContinuousListener
        listener = ContinuousListener(vad_threshold=0.005, stt_provider="openai")
        mock_sd = _make_sd_mock(_two_utterance_frames())
        mock_sf = MagicMock()
        mock_ntf = MagicMock()
        mock_ntf.return_value.name = "/tmp/fake_test.wav"

        stt_results = [
            _stt_result("ミミ様って可愛いよね"),       # 1 回目: 言及
            _stt_result("ねぇミミ様、聞いていい？"),  # 2 回目: 呼びかけ
        ]
        intent_results = ["mention", "callout"]

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "soundfile": mock_sf}), \
             patch("lab_lounge.wake_word.tempfile.NamedTemporaryFile", mock_ntf), \
             patch.object(Path, "unlink"), \
             patch.object(_stt, "transcribe_audio_file", side_effect=stt_results), \
             patch("lab_lounge.router.check_intent", side_effect=intent_results) as mock_check, \
             patch("time.monotonic", return_value=0.0):
            result = listener.listen_once(timeout_seconds=30.0)

        assert result is not None
        assert result.character_slug == "mimi"
        # 両方のセグメントが文脈に含まれる
        assert "ミミ様って可愛いよね" in result.transcript
        assert "ねぇミミ様、聞いていい？" in result.transcript
        # 意図ゲートは 2 回呼ばれた（1 回目 mention, 2 回目 callout）
        assert mock_check.call_count == 2


# ═══════════════════════════════════════════════════════════════════
# TestContinuousListenerWebRTC — WebRTC VAD バックエンド統合テスト
# ═══════════════════════════════════════════════════════════════════


def _make_mock_webrtcvad(*, is_speech_returns: bool = True) -> MagicMock:
    """webrtcvad モジュールのモックを作成する。"""
    mock_vad_instance = MagicMock()
    mock_vad_instance.is_speech.return_value = is_speech_returns

    mock_webrtcvad = MagicMock()
    mock_webrtcvad.Vad = MagicMock(return_value=mock_vad_instance)
    return mock_webrtcvad


class TestContinuousListenerWebRTC:
    """L2_VAD_BACKEND=webrtc 設定下で ContinuousListener が
    WebRTC 用のフレームパラメータで初期化されることを検証する。"""

    def test_init_sets_320_frame_samples(self, monkeypatch):
        """L2_VAD_BACKEND=webrtc で _vad_frame_samples == 320。"""
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")
        mock_webrtcvad = _make_mock_webrtcvad()
        from lab_lounge.wake_word import (
            _WEBRTC_FRAME_SAMPLES,
            ContinuousListener,
        )
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            listener = ContinuousListener()
        assert listener._vad_frame_samples == _WEBRTC_FRAME_SAMPLES
        assert listener._vad_frame_samples == 320

    def test_init_sets_75_silence_frames(self, monkeypatch):
        """L2_VAD_BACKEND=webrtc で _silence_frames == 75。"""
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")
        mock_webrtcvad = _make_mock_webrtcvad()
        from lab_lounge.wake_word import (
            _WEBRTC_SILENCE_FRAMES,
            ContinuousListener,
        )
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            listener = ContinuousListener()
        assert listener._silence_frames == _WEBRTC_SILENCE_FRAMES
        assert listener._silence_frames == 75

    def test_init_with_rms_default(self, monkeypatch):
        """env 未設定 (RMS) で 512 / 24 になる回帰テスト。"""
        monkeypatch.delenv("L2_VAD_BACKEND", raising=False)
        from lab_lounge.wake_word import (
            _DEFAULT_FRAME_SAMPLES,
            _DEFAULT_SILENCE_FRAMES,
            ContinuousListener,
        )
        listener = ContinuousListener()
        assert listener._vad_frame_samples == _DEFAULT_FRAME_SAMPLES
        assert listener._vad_frame_samples == 512
        assert listener._silence_frames == _DEFAULT_SILENCE_FRAMES
        assert listener._silence_frames == 24

    def test_init_falls_back_when_webrtcvad_missing(self, monkeypatch):
        """L2_VAD_BACKEND=webrtc でも webrtcvad import 失敗時に
        RMS パラメータ (512/24) にフォールバックする。"""
        import builtins
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")

        from lab_lounge.wake_word import (
            _DEFAULT_FRAME_SAMPLES,
            _DEFAULT_SILENCE_FRAMES,
            ContinuousListener,
        )

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "webrtcvad":
                raise ImportError("mocked: webrtcvad not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            listener = ContinuousListener()

        # RMS パラメータにフォールバック
        assert listener._vad_frame_samples == _DEFAULT_FRAME_SAMPLES
        assert listener._vad_frame_samples == 512
        assert listener._silence_frames == _DEFAULT_SILENCE_FRAMES
        assert listener._silence_frames == 24
