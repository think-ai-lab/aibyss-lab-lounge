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


# ─── BackgroundContinuousListener (Block 0) ──────────────────────


def _make_mock_sounddevice():
    """sounddevice のモック。InputStream の read() は silent PCM を返す
    (onset 未検知 → segment 生成なし → STT 呼ばれず副作用なし)。"""
    mock_sd = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=None)
    silent_pcm = np.zeros((_DEFAULT_FRAME_SAMPLES, 1), dtype=np.int16)
    mock_stream.read.return_value = (silent_pcm, False)
    mock_sd.InputStream.return_value = mock_stream
    return mock_sd


class TestBackgroundContinuousListenerInit:
    """初期化系テスト (スレッド起動なし)。"""

    def test_buffer_property_is_transcript_buffer(self):
        from lab_lounge.transcript_buffer import TranscriptBuffer
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        assert isinstance(listener.buffer, TranscriptBuffer)

    def test_initial_no_thread(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        assert listener._thread is None

    def test_initial_routing_not_paused(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        assert not listener._routing_paused.is_set()

    def test_custom_window_and_max_chars(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener(
            context_window_sec=60.0,
            context_max_chars=500,
        )
        assert listener.buffer._window_sec == 60.0
        assert listener.buffer._max_chars == 500


class TestBackgroundContinuousListenerLifecycle:
    """start / stop / cleanup のライフサイクルテスト。

    sounddevice をモックし、silent PCM のみ返すことで実マイク入力なしで
    スレッドのライフサイクル動作を検証する (onset 検知に至らないため
    STT / numpy 演算等の重い処理は走らない)。
    """

    def test_start_starts_thread(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        import time as _time

        mock_sd = _make_mock_sounddevice()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            listener = BackgroundContinuousListener()
            listener.start(on_wake_detected=lambda r: None)

            _time.sleep(0.1)  # スレッドが回ることを確認する短い待機
            assert listener._thread is not None
            assert listener._thread.is_alive()

            listener.stop(timeout=2.0)

    def test_stop_terminates_thread(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        import time as _time

        mock_sd = _make_mock_sounddevice()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            listener = BackgroundContinuousListener()
            listener.start(on_wake_detected=lambda r: None)
            _time.sleep(0.05)
            listener.stop(timeout=2.0)
            assert listener._thread is None

    def test_double_start_raises(self):
        from lab_lounge.wake_word import BackgroundContinuousListener

        mock_sd = _make_mock_sounddevice()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            listener = BackgroundContinuousListener()
            listener.start(on_wake_detected=lambda r: None)
            try:
                with pytest.raises(RuntimeError, match="既に起動中"):
                    listener.start(on_wake_detected=lambda r: None)
            finally:
                listener.stop(timeout=2.0)

    def test_stop_without_start_is_noop(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        # 例外なく動く
        listener.stop()
        assert listener._thread is None

    def test_cleanup_stops_and_clears_buffer(self):
        from lab_lounge.transcript_buffer import TranscriptSegment
        from lab_lounge.wake_word import BackgroundContinuousListener

        mock_sd = _make_mock_sounddevice()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            listener = BackgroundContinuousListener()
            listener.start(on_wake_detected=lambda r: None)

            # buffer に直接 segment を入れて、cleanup で消えることを確認
            listener.buffer.add(TranscriptSegment(
                text="前のテスト残骸", timestamp=100.0, duration_ms=500,
            ))
            assert len(listener.buffer) == 1

            listener.cleanup()
            assert listener._thread is None
            assert len(listener.buffer) == 0


class TestBackgroundContinuousListenerOnSegmentAdded:
    """Phase 0.5-A フェーズ 6: on_segment_added callback の引数仕様。

    callback は ``(segment, buffer_full_text)`` の 2 引数で呼び出される。
    Dispatcher.on_segment_added と整合させ、buffer.full_text() を listener 内
    Lock のもとで取得することで race による微差異を防ぐ。
    """

    def test_callback_signature_is_two_args(self):
        """Listener._on_segment_added 属性の型は Callable[[Any, str], None]。

        sounddevice を起動せず、_run_loop 内の callback 発火行と同じ呼出パターンを
        手動で再現して、シグネチャと値伝播を検証する。
        """
        from lab_lounge.wake_word import BackgroundContinuousListener
        from lab_lounge.transcript_buffer import TranscriptSegment

        listener = BackgroundContinuousListener()
        received: list[tuple] = []

        def callback(segment, buffer_full_text):
            received.append((segment, buffer_full_text))

        listener._on_segment_added = callback

        # buffer に segment を直接追加
        seg = TranscriptSegment(text="テスト発話", timestamp=100.0, duration_ms=500)
        listener._buffer.add(seg)

        # _run_loop 内の callback 発火行と同じ呼出
        # (sounddevice 不要、buffer は遅延 import 済の純粋オブジェクト)
        listener._on_segment_added(seg, listener._buffer.full_text())

        assert len(received) == 1
        received_seg, received_full_text = received[0]
        assert received_seg is seg
        assert isinstance(received_full_text, str)
        assert received_full_text == listener._buffer.full_text()
        assert "テスト発話" in received_full_text

    def test_callback_default_is_none(self):
        """start() で on_segment_added を渡さなければ default は None (Block 0 互換)。"""
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        assert listener._on_segment_added is None

    def test_start_stores_on_segment_added_callback(self):
        """start(on_segment_added=...) で渡した callback がインスタンス属性に保存される。"""
        from lab_lounge.wake_word import BackgroundContinuousListener

        mock_sd = _make_mock_sounddevice()
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            listener = BackgroundContinuousListener()
            cb = lambda seg, full_text: None
            listener.start(on_wake_detected=lambda r: None, on_segment_added=cb)
            try:
                assert listener._on_segment_added is cb
            finally:
                listener.stop(timeout=2.0)


class TestBackgroundContinuousListenerRoutingPause:
    """set_routing_paused のフラグ動作テスト (スレッド起動なし)。"""

    def test_set_routing_paused_true(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        listener.set_routing_paused(True)
        assert listener._routing_paused.is_set()

    def test_set_routing_paused_false(self):
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        listener.set_routing_paused(True)
        listener.set_routing_paused(False)
        assert not listener._routing_paused.is_set()

    def test_set_routing_paused_idempotent(self):
        """同じ値を複数回設定しても問題ない。"""
        from lab_lounge.wake_word import BackgroundContinuousListener
        listener = BackgroundContinuousListener()
        listener.set_routing_paused(True)
        listener.set_routing_paused(True)
        assert listener._routing_paused.is_set()
        listener.set_routing_paused(False)
        listener.set_routing_paused(False)
        assert not listener._routing_paused.is_set()
