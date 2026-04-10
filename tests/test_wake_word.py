"""
test_wake_word.py — ウェイクワード検知のテスト

pvporcupine をモックして、キャラクターマッピングとエラーハンドリングを検証する。
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lab_lounge.wake_word import (
    WakeWordResult,
    _create_vad_checker,
    _find_keyword_paths,
    _get_platform_suffix,
    _get_vad_backend,
    _DEFAULT_FRAME_SAMPLES,
    _DEFAULT_SILENCE_FRAMES,
    _WEBRTC_FRAME_SAMPLES,
    _WEBRTC_SILENCE_FRAMES,
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


# ═══════════════════════════════════════════════════════════════════
# VAD バックエンド選択 + ファクトリ
# ═══════════════════════════════════════════════════════════════════


class TestGetVadBackend:
    """_get_vad_backend() の env 制御テスト。"""

    def test_default_returns_rms(self, monkeypatch):
        monkeypatch.delenv("L2_VAD_BACKEND", raising=False)
        assert _get_vad_backend() == "rms"

    def test_webrtc_when_env_set(self, monkeypatch):
        monkeypatch.setenv("L2_VAD_BACKEND", "webrtc")
        assert _get_vad_backend() == "webrtc"

    def test_arbitrary_value_returned_as_is(self, monkeypatch):
        """env はそのまま返す (バリデーションは下流の責任)。"""
        monkeypatch.setenv("L2_VAD_BACKEND", "unknown_backend")
        assert _get_vad_backend() == "unknown_backend"


def _make_mock_webrtcvad(*, is_speech_returns: bool = True) -> MagicMock:
    """webrtcvad モジュールのモックを作成する。"""
    mock_vad_instance = MagicMock()
    mock_vad_instance.is_speech.return_value = is_speech_returns

    mock_webrtcvad = MagicMock()
    mock_webrtcvad.Vad = MagicMock(return_value=mock_vad_instance)
    return mock_webrtcvad


class TestCreateVadCheckerRMS:
    """_create_vad_checker('rms', ...) の戻り値検証。"""

    def test_returns_512_frame_samples(self):
        is_speech, frame_samples, silence_frames = _create_vad_checker("rms", 0.01)
        assert frame_samples == _DEFAULT_FRAME_SAMPLES
        assert frame_samples == 512

    def test_returns_24_silence_frames(self):
        is_speech, frame_samples, silence_frames = _create_vad_checker("rms", 0.01)
        assert silence_frames == _DEFAULT_SILENCE_FRAMES
        assert silence_frames == 24

    def test_rms_checker_detects_loud_and_silent_frames(self):
        """実 numpy で RMS 計算。閾値 0.01 で 800 値は超え、0 値は超えない。"""
        is_speech, _, _ = _create_vad_checker("rms", 0.01)
        # 値 800 → RMS ≈ 800/32768 ≈ 0.0244 > 0.01 → True
        loud = np.full(512, 800, dtype=np.int16)
        assert is_speech(loud) is True
        # 値 0 → RMS = 0 < 0.01 → False
        silent = np.zeros(512, dtype=np.int16)
        assert is_speech(silent) is False


class TestCreateVadCheckerWebRTC:
    """_create_vad_checker('webrtc', ...) の戻り値検証 (webrtcvad モック使用)。"""

    def test_returns_320_frame_samples(self):
        mock_webrtcvad = _make_mock_webrtcvad()
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            _, frame_samples, _ = _create_vad_checker("webrtc", 0.01)
        assert frame_samples == _WEBRTC_FRAME_SAMPLES
        assert frame_samples == 320

    def test_returns_75_silence_frames(self):
        mock_webrtcvad = _make_mock_webrtcvad()
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            _, _, silence_frames = _create_vad_checker("webrtc", 0.01)
        assert silence_frames == _WEBRTC_SILENCE_FRAMES
        assert silence_frames == 75

    def test_aggressiveness_default_2(self, monkeypatch):
        """L2_VAD_AGGRESSIVENESS 未設定で webrtcvad.Vad(2)。"""
        monkeypatch.delenv("L2_VAD_AGGRESSIVENESS", raising=False)
        mock_webrtcvad = _make_mock_webrtcvad()
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            _create_vad_checker("webrtc", 0.01)
        mock_webrtcvad.Vad.assert_called_once_with(2)

    def test_aggressiveness_from_env(self, monkeypatch):
        """L2_VAD_AGGRESSIVENESS=3 で webrtcvad.Vad(3)。"""
        monkeypatch.setenv("L2_VAD_AGGRESSIVENESS", "3")
        mock_webrtcvad = _make_mock_webrtcvad()
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            _create_vad_checker("webrtc", 0.01)
        mock_webrtcvad.Vad.assert_called_once_with(3)

    def test_webrtc_checker_calls_vad_is_speech(self):
        """is_speech() がモック VAD オブジェクトの is_speech() に委譲される。"""
        mock_webrtcvad = _make_mock_webrtcvad(is_speech_returns=True)
        with patch.dict(sys.modules, {"webrtcvad": mock_webrtcvad}):
            is_speech, _, _ = _create_vad_checker("webrtc", 0.01)

        # is_speech 呼び出し
        pcm = np.zeros(320, dtype=np.int16)
        result = is_speech(pcm)

        assert result is True
        mock_vad = mock_webrtcvad.Vad.return_value
        mock_vad.is_speech.assert_called_once()
        # bytes と sample_rate=16000 が渡される
        call_args = mock_vad.is_speech.call_args
        assert isinstance(call_args.args[0], bytes)
        assert call_args.kwargs.get("sample_rate") == 16000


class TestVadFallback:
    """webrtcvad 未インストール時の RMS フォールバック検証。"""

    def test_webrtc_falls_back_to_rms_on_import_error(self):
        """webrtcvad import 失敗 → RMS パラメータ (512/24) を返す。"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "webrtcvad":
                raise ImportError("mocked: webrtcvad not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            _, frame_samples, silence_frames = _create_vad_checker("webrtc", 0.01)

        # RMS パラメータにフォールバック
        assert frame_samples == _DEFAULT_FRAME_SAMPLES
        assert silence_frames == _DEFAULT_SILENCE_FRAMES

    def test_fallback_logs_warning(self, caplog):
        """フォールバック時に warning ログが出る。"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "webrtcvad":
                raise ImportError("mocked: webrtcvad not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.wake_word"):
                _create_vad_checker("webrtc", 0.01)

        warning_msgs = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("webrtcvad" in m for m in warning_msgs)
        assert any("RMS" in m or "rms" in m.lower() for m in warning_msgs)
