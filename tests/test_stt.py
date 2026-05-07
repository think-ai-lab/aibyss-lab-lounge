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


class TestSttLogSanitization:
    """配信中の機密情報漏洩を防ぐ: STT ログがファイルパスを含まない。"""

    def test_stt_info_log_does_not_contain_full_path(self, tmp_path, caplog):
        """INFO レベルのログに完全パス (一時ディレクトリ) が含まれない。"""
        import logging
        # ダミー WAV ファイル
        dummy = tmp_path / "speech_input.wav"
        dummy.write_bytes(b"\x00" * 100)

        # _PROVIDERS をモック
        original_providers = stt_mod._PROVIDERS.copy()
        stt_mod._PROVIDERS["openai"] = _mock_provider(FAKE_STT_RESULT)
        try:
            with caplog.at_level(logging.INFO, logger="lab_lounge.stt"):
                transcribe_audio_file(str(dummy), provider="openai", lang="ja")
        finally:
            stt_mod._PROVIDERS.clear()
            stt_mod._PROVIDERS.update(original_providers)

        info_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        all_info = " ".join(info_messages)
        # 完全パス (tmp_path) は INFO に含まれない
        assert str(tmp_path) not in all_info
        # ファイル名は含まれてよい
        assert "speech_input.wav" in all_info


# ─── TestHallucinationFilter (Phase 0.5-A フェーズ 0) ──────────────


class TestHallucinationFilter:
    """Whisper の hallucination パターン検出ロジック。

    無音 / ノイズ入力で生成される YouTube 字幕由来の定型句を除外する。
    """

    def test_typical_youtube_phrase_detected(self):
        """典型的な YouTube 字幕由来フレーズを検出する。"""
        from lab_lounge.stt import _is_likely_hallucination

        assert _is_likely_hallucination("ご視聴ありがとうございました", 2000)
        assert _is_likely_hallucination("ご視聴ありがとうございます", 1500)
        assert _is_likely_hallucination("おやすみなさい", 1000)
        assert _is_likely_hallucination("ご覧いただきありがとうございました。", 2000)

    def test_pattern_with_trailing_punctuation(self):
        """末尾の句読点を除去して判定する。"""
        from lab_lounge.stt import _is_likely_hallucination

        assert _is_likely_hallucination("ご視聴ありがとうございました。", 2000)
        assert _is_likely_hallucination("おやすみなさい。", 1000)
        assert _is_likely_hallucination("また次回!", 1000)
        assert _is_likely_hallucination("ご視聴ありがとうございました…", 1500)

    def test_pattern_at_end(self):
        """末尾一致でも検出する (前置きあり)。"""
        from lab_lounge.stt import _is_likely_hallucination

        # 「(無音) → ご視聴ありがとうございました」のような hallucination
        assert _is_likely_hallucination("...ご視聴ありがとうございました", 2000)

    def test_short_audio_long_text_unmatched(self):
        """1 秒未満の録音で 10 文字以上の出力は不審。"""
        from lab_lounge.stt import _is_likely_hallucination

        # 短時間 + 長文 → 不審
        assert _is_likely_hallucination("これは長い不審なテキスト", 500)
        # 短時間 + 短文 → 通常 (フィルタしない)
        assert not _is_likely_hallucination("はい", 500)

    def test_complete_repetition_detected(self):
        """完全反復 (半分のフレーズが 2 回以上) を検出する。"""
        from lab_lounge.stt import _is_likely_hallucination

        # "ありがとう" を繰り返す典型的 hallucination
        assert _is_likely_hallucination(
            "ありがとうございますありがとうございます", 3000,
        )

    def test_normal_callout_not_detected(self):
        """通常の callout は検出されない。"""
        from lab_lounge.stt import _is_likely_hallucination

        assert not _is_likely_hallucination(
            "ミミ様、深海って怖い場所だと思いますか?", 4000,
        )
        assert not _is_likely_hallucination(
            "ちさめさん、データ的にはどう見えますか?", 5000,
        )

    def test_empty_text_not_detected(self):
        """空文字 / 空白のみは hallucination 扱いしない (フィルタ対象外)。"""
        from lab_lounge.stt import _is_likely_hallucination

        assert not _is_likely_hallucination("", 1000)
        assert not _is_likely_hallucination("   ", 1000)

    def test_extra_patterns_via_env(self, monkeypatch):
        """L2_STT_HALLUCINATION_EXTRA_PATTERNS で追加パターンを定義可能。"""
        from lab_lounge.stt import _is_likely_hallucination

        monkeypatch.setenv("L2_STT_HALLUCINATION_EXTRA_PATTERNS", "またね,お疲れ様でした")

        assert _is_likely_hallucination("またね", 500)
        assert _is_likely_hallucination("お疲れ様でした", 1500)
        # 既存パターンは引き続き機能
        assert _is_likely_hallucination("ご視聴ありがとうございました", 2000)

    def test_filter_disabled_via_env(self, monkeypatch):
        """L2_STT_HALLUCINATION_FILTER=false で apply_hallucination_filter は素通し。"""
        from lab_lounge.stt import _apply_hallucination_filter

        monkeypatch.setenv("L2_STT_HALLUCINATION_FILTER", "false")

        result = STTResult(
            text="ご視聴ありがとうございました",
            confidence=None,
            lang="ja",
            duration_ms=2000,
        )
        out = _apply_hallucination_filter(result)
        # フィルタ無効 → text は変わらない
        assert out.text == "ご視聴ありがとうございました"

    def test_filter_replaces_text_when_detected(self, monkeypatch):
        """フィルタ有効時、hallucination 検出で text が空文字化される。"""
        from lab_lounge.stt import _apply_hallucination_filter

        monkeypatch.setenv("L2_STT_HALLUCINATION_FILTER", "true")

        result = STTResult(
            text="ご視聴ありがとうございました",
            confidence=None,
            lang="ja",
            duration_ms=2000,
        )
        out = _apply_hallucination_filter(result)
        # フィルタ有効 + 検出 → text が空文字化
        assert out.text == ""
        # 他のフィールドは維持
        assert out.lang == "ja"
        assert out.duration_ms == 2000

    def test_filter_preserves_normal_text(self, monkeypatch):
        """フィルタ有効時、通常テキストは維持される。"""
        from lab_lounge.stt import _apply_hallucination_filter

        monkeypatch.setenv("L2_STT_HALLUCINATION_FILTER", "true")

        result = STTResult(
            text="ミミ様、深海について教えてください",
            confidence=None,
            lang="ja",
            duration_ms=4000,
        )
        out = _apply_hallucination_filter(result)
        assert out.text == "ミミ様、深海について教えてください"

    def test_filter_integrated_in_transcribe_audio_file(self, tmp_path, monkeypatch):
        """transcribe_audio_file 全体で hallucination フィルタが効く。"""
        monkeypatch.setenv("L2_STT_HALLUCINATION_FILTER", "true")

        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"fake")

        # provider が hallucination を返す
        fake = STTResult(
            text="ご視聴ありがとうございました",
            confidence=None,
            lang="ja",
            duration_ms=2000,
        )
        original_providers = dict(stt_mod._PROVIDERS)
        try:
            stt_mod._PROVIDERS["openai"] = _mock_provider(fake)
            result = transcribe_audio_file(str(dummy), provider="openai", lang="ja")
        finally:
            stt_mod._PROVIDERS.clear()
            stt_mod._PROVIDERS.update(original_providers)

        # transcribe_audio_file の戻り値で text が空文字化されている
        assert result.text == ""
        assert result.duration_ms == 2000
