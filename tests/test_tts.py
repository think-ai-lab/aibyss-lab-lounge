"""
test_tts.py — tts.synthesize テスト

外部 TTS API は呼ばない（_PROVIDERS dict をモックして検証する）。
edge-tts / mutagen がインストールされていなくても動作する。
"""

import pytest
from unittest.mock import MagicMock

import lab_lounge.tts as tts_mod
from lab_lounge.tts import TTSResult, synthesize


FAKE_TTS_RESULT = TTSResult(
    audio_url="file:///tmp/test-audio.mp3",
    duration_ms=2500,
    voice="ja-JP-NanamiNeural",
    format="mp3",
    sample_rate=24000,
    speaker="Nanami",
)


def _mock_provider(fake: TTSResult) -> MagicMock:
    """_PROVIDERS に差し込む MagicMock を返す。"""
    return MagicMock(return_value=fake)


class TestSynthesize:
    def test_unsupported_provider_raises_value_error(self):
        """未対応 provider は ValueError"""
        with pytest.raises(ValueError, match="未対応"):
            synthesize("テスト", provider="unsupported_xyz", voice="ja-JP-NanamiNeural")

    def test_edge_tts_dispatches_to_provider(self):
        """provider='edge_tts' のとき _PROVIDERS["edge_tts"] が呼ばれる"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        mock_fn.assert_called_once_with(
            "テスト", voice="ja-JP-NanamiNeural", output_dir="./data/audio"
        )
        assert result == FAKE_TTS_RESULT

    def test_voicevox_dispatches_to_provider(self):
        """provider='voicevox' のとき _PROVIDERS["voicevox"] が呼ばれる"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "voicevox", mock_fn)
            result = synthesize("テスト", provider="voicevox", voice="3")
        mock_fn.assert_called_once_with("テスト", voice="3", output_dir="./data/audio")
        assert result == FAKE_TTS_RESULT

    def test_returns_tts_result_instance(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert isinstance(result, TTSResult)

    def test_result_audio_url_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.audio_url == "file:///tmp/test-audio.mp3"

    def test_result_duration_ms_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.duration_ms == 2500

    def test_result_voice_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.voice == "ja-JP-NanamiNeural"

    def test_result_format_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.format == "mp3"

    def test_result_sample_rate_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.sample_rate == 24000

    def test_result_speaker_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", voice="ja-JP-NanamiNeural")
        assert result.speaker == "Nanami"

    def test_output_dir_forwarded(self):
        """output_dir 引数が provider に転送される"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            synthesize("テスト", voice="ja-JP-NanamiNeural", output_dir="/custom/audio")
        mock_fn.assert_called_once_with(
            "テスト", voice="ja-JP-NanamiNeural", output_dir="/custom/audio"
        )

    def test_kwargs_forwarded_to_provider(self):
        """**kwargs が provider に転送される（speaker 等）"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            synthesize(
                "テスト", voice="ja-JP-NanamiNeural", speaker="Nanami"
            )
        mock_fn.assert_called_once_with(
            "テスト",
            voice="ja-JP-NanamiNeural",
            output_dir="./data/audio",
            speaker="Nanami",
        )

    def test_edge_tts_import_error_without_package(self):
        """edge-tts が未インストールの場合、ImportError を送出する"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "edge_tts":
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        import unittest.mock as mock
        with mock.patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="edge-tts"):
                tts_mod._call_edge_tts("テスト", voice="ja-JP-NanamiNeural", output_dir="/tmp")

    def test_voicevox_invalid_voice_raises_value_error(self):
        """voicevox provider に非整数文字列の voice を渡すと ValueError"""
        with pytest.raises(ValueError, match="整数文字列"):
            tts_mod._call_voicevox(
                "テスト", voice="not-a-number", output_dir="/tmp"
            )
