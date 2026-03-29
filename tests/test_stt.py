"""
test_stt.py — stt.transcribe_audio_file テスト

実 STT API / 実音声ファイルは使わない。
_PROVIDERS dict を pytest.MonkeyPatch で差し替えて検証する。
"""

import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.stt as stt_mod
from lab_lounge.stt import STTResult, transcribe_audio_file


FAKE_STT_RESULT = STTResult(
    text="こんにちは、今日の天気を教えてください",
    confidence=None,
    lang="ja",
    duration_ms=3500,
    words=None,
)


def _mock_provider(fake: STTResult) -> MagicMock:
    """_PROVIDERS に差し込む MagicMock を返す。(ファイル存在チェックはスキップする)"""
    return MagicMock(return_value=fake)


class TestTranscribeAudioFile:
    def test_unsupported_provider_raises_value_error(self, tmp_path):
        """未対応 provider は ValueError"""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        with pytest.raises(ValueError, match="未対応"):
            transcribe_audio_file(str(dummy), provider="unknown_xyz", lang="ja")

    def test_file_not_found_raises(self):
        """存在しないファイルを渡すと FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            transcribe_audio_file("/nonexistent/audio.wav", provider="openai", lang="ja")

    def test_openai_dispatches_to_provider(self, tmp_path):
        """provider='openai' のとき _PROVIDERS["openai"] が呼ばれる"""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), provider="openai", lang="ja")
        mock_fn.assert_called_once_with(str(dummy), lang="ja", prompt=None)
        assert result == FAKE_STT_RESULT

    def test_returns_stt_result_instance(self, tmp_path):
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert isinstance(result, STTResult)

    def test_result_text_field(self, tmp_path):
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert result.text == "こんにちは、今日の天気を教えてください"

    def test_result_lang_field(self, tmp_path):
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert result.lang == "ja"

    def test_result_duration_ms_field(self, tmp_path):
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert result.duration_ms == 3500

    def test_result_confidence_is_none(self, tmp_path):
        """Whisper API は confidence を返さないので None になる"""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert result.confidence is None

    def test_result_words_default_none(self, tmp_path):
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            result = transcribe_audio_file(str(dummy), lang="ja")
        assert result.words is None

    def test_kwargs_forwarded_to_provider(self, tmp_path):
        """**kwargs (model 等) が provider に転送される"""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        mock_fn = _mock_provider(FAKE_STT_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(stt_mod._PROVIDERS, "openai", mock_fn)
            transcribe_audio_file(str(dummy), lang="ja", model="whisper-1")
        mock_fn.assert_called_once_with(str(dummy), lang="ja", prompt=None, model="whisper-1")

    def test_openai_import_error_without_package(self, tmp_path):
        """openai パッケージが未インストールの場合 ImportError を送出する"""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="openai"):
                stt_mod._call_openai_whisper(str(dummy), lang="ja")


class TestFileDurationMs:
    def test_wav_file_duration(self, tmp_path):
        """WAV ファイルから duration_ms を取得できる"""
        import struct
        import wave

        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # 1 秒 = 16000 frames
            wf.writeframes(b"\x00\x00" * 16000)

        duration = stt_mod._file_duration_ms(str(wav_path))
        assert duration == 1000

    def test_nonexistent_file_returns_zero(self, tmp_path):
        """存在しないファイルは 0 を返す（例外を送出しない）"""
        duration = stt_mod._file_duration_ms(str(tmp_path / "nonexistent.wav"))
        assert duration == 0
