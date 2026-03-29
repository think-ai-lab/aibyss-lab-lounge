"""
test_run_once.py — run_once のテスト

録音・再生は実デバイス不要のモック前提。
実 LLM / TTS / STT も呼ばない。CI でも安全に実行できる。

テスト対象:
  - run_once() の正常フロー: 録音 → STT → pipeline → 再生
  - 無音 / 録音失敗での中断
  - STT 失敗で後続に進まないこと
  - skip_playback=True のとき再生されないこと
  - audio_io._uri_to_path の URI 変換
"""

import pytest
from unittest.mock import patch

import lab_lounge.run_once as run_once_mod
from lab_lounge.audio_io import RecordError, SilenceError
from lab_lounge.pipeline import PipelineResult
from lab_lounge.run_once import run_once


# ─── テスト用定数 ─────────────────────────────────────────────────

_RECORDED_PATH = "/tmp/l2_rec_test.wav"

FAKE_EVENTS = [
    {
        "type": "utterance.final",
        "event_id": "utt-1",
        "seq": 0,
        "payload": {"text": "テスト発話"},
        "links": [],
    },
    {
        "type": "llm.final",
        "event_id": "llm-1",
        "seq": 1,
        "payload": {"text": "LLM 応答テスト", "latency_ms": 200},
        "links": ["utt-1"],
    },
    {
        "type": "tts.done",
        "event_id": "tts-1",
        "seq": 2,
        "payload": {"audio_url": "file:///tmp/test.wav", "duration_ms": 1000},
        "links": ["llm-1"],
    },
]

FAKE_RESULT = PipelineResult(
    stream_id="test-stream",
    session_id="test-session",
    trace_id="test-trace",
    speaker="octamaid",
    events=FAKE_EVENTS,
)


# ─── フィクスチャ ─────────────────────────────────────────────────

@pytest.fixture()
def mock_record():
    with patch.object(run_once_mod, "record_to_file", return_value=_RECORDED_PATH) as m:
        yield m


@pytest.fixture()
def mock_transcribe():
    with patch.object(run_once_mod, "_transcribe_audio", return_value=("テスト発話", {})) as m:
        yield m


@pytest.fixture()
def mock_pipeline():
    with patch.object(run_once_mod, "run_pipeline", return_value=FAKE_RESULT) as m:
        yield m


@pytest.fixture()
def mock_play():
    with patch.object(run_once_mod, "play_audio_file", return_value=True) as m:
        yield m


# ─── 正常フロー ───────────────────────────────────────────────────

class TestRunOnceHappyPath:
    def test_returns_pipeline_result(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """正常フローで PipelineResult が返ること"""
        result = run_once(record_seconds=3.0)
        assert isinstance(result, PipelineResult)

    def test_record_called_once(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """record_to_file が 1 回呼ばれること"""
        run_once(record_seconds=3.0)
        mock_record.assert_called_once()

    def test_transcribe_called_with_recorded_path(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """_transcribe_audio が録音ファイルパスで呼ばれること"""
        run_once(record_seconds=3.0)
        mock_transcribe.assert_called_once_with(_RECORDED_PATH)

    def test_pipeline_called_once(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """run_pipeline が 1 回呼ばれること"""
        run_once(record_seconds=3.0)
        mock_pipeline.assert_called_once()

    def test_pipeline_receives_transcribed_text(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """run_pipeline に STT の変換テキストが渡ること"""
        run_once(record_seconds=3.0)
        assert mock_pipeline.call_args.args[0] == "テスト発話"

    def test_play_called_with_audio_url(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """play_audio_file に tts.done の audio_url が渡ること"""
        run_once(record_seconds=3.0)
        mock_play.assert_called_once_with("file:///tmp/test.wav")

    def test_play_not_called_when_skip_playback(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """skip_playback=True のとき play_audio_file が呼ばれないこと"""
        run_once(record_seconds=3.0, skip_playback=True)
        mock_play.assert_not_called()

    def test_stream_id_passed_to_pipeline(self, mock_record, mock_transcribe, mock_pipeline, mock_play):
        """stream_id 指定時に run_pipeline に渡ること"""
        run_once(record_seconds=3.0, stream_id="custom-stream-001")
        assert mock_pipeline.call_args.kwargs["stream_id"] == "custom-stream-001"

    def test_utterance_meta_propagated(self, mock_record, mock_pipeline, mock_play):
        """STT が返した utterance_meta が run_pipeline に渡ること"""
        meta = {"lang": "ja", "confidence": 0.9, "duration_ms": 2000}
        with patch.object(run_once_mod, "_transcribe_audio", return_value=("発話", meta)):
            run_once(record_seconds=3.0)
        kw = mock_pipeline.call_args.kwargs
        assert kw["utterance_meta"]["lang"] == "ja"
        assert kw["utterance_meta"]["duration_ms"] == 2000


# ─── エラーフロー ─────────────────────────────────────────────────

class TestRunOnceErrorPaths:
    def test_silence_returns_none(self, mock_transcribe, mock_pipeline, mock_play):
        """無音検出時に None を返すこと"""
        with patch.object(run_once_mod, "record_to_file", side_effect=SilenceError("無音")):
            result = run_once(record_seconds=3.0)
        assert result is None

    def test_silence_does_not_call_transcribe(self, mock_transcribe, mock_pipeline, mock_play):
        """無音検出時に STT を呼ばないこと"""
        with patch.object(run_once_mod, "record_to_file", side_effect=SilenceError("無音")):
            run_once(record_seconds=3.0)
        mock_transcribe.assert_not_called()

    def test_silence_does_not_call_pipeline(self, mock_transcribe, mock_pipeline, mock_play):
        """無音検出時に pipeline を呼ばないこと"""
        with patch.object(run_once_mod, "record_to_file", side_effect=SilenceError("無音")):
            run_once(record_seconds=3.0)
        mock_pipeline.assert_not_called()

    def test_record_error_returns_none(self, mock_transcribe, mock_pipeline, mock_play):
        """録音デバイスエラー時に None を返すこと"""
        with patch.object(run_once_mod, "record_to_file", side_effect=RecordError("デバイス不可")):
            result = run_once(record_seconds=3.0)
        assert result is None

    def test_record_error_does_not_call_pipeline(self, mock_transcribe, mock_pipeline, mock_play):
        """録音デバイスエラー時に pipeline を呼ばないこと"""
        with patch.object(run_once_mod, "record_to_file", side_effect=RecordError("デバイス不可")):
            run_once(record_seconds=3.0)
        mock_pipeline.assert_not_called()

    def test_stt_failure_returns_none(self, mock_record, mock_pipeline, mock_play):
        """STT 失敗時に None を返すこと"""
        with patch.object(run_once_mod, "_transcribe_audio", side_effect=RuntimeError("STT API エラー")):
            result = run_once(record_seconds=3.0)
        assert result is None

    def test_stt_failure_does_not_call_pipeline(self, mock_record, mock_pipeline, mock_play):
        """STT 失敗時に pipeline を呼ばないこと"""
        with patch.object(run_once_mod, "_transcribe_audio", side_effect=RuntimeError("STT API エラー")):
            run_once(record_seconds=3.0)
        mock_pipeline.assert_not_called()

    def test_stt_failure_does_not_call_play(self, mock_record, mock_pipeline, mock_play):
        """STT 失敗時に再生を呼ばないこと"""
        with patch.object(run_once_mod, "_transcribe_audio", side_effect=RuntimeError("STT API エラー")):
            run_once(record_seconds=3.0)
        mock_play.assert_not_called()

    def test_play_failure_still_returns_result(self, mock_record, mock_transcribe, mock_pipeline):
        """再生失敗でも PipelineResult を返すこと (再生はベストエフォート)"""
        with patch.object(run_once_mod, "play_audio_file", return_value=False):
            result = run_once(record_seconds=3.0)
        assert result is not None


# ─── audio_io._uri_to_path テスト ────────────────────────────────

class TestUriToPath:
    """audio_io._uri_to_path の URI 変換テスト。実デバイス不要。"""

    def test_plain_unix_path_unchanged(self):
        from lab_lounge.audio_io import _uri_to_path
        assert _uri_to_path("/tmp/test.wav") == "/tmp/test.wav"

    def test_file_uri_unix(self):
        from lab_lounge.audio_io import _uri_to_path
        assert _uri_to_path("file:///tmp/test.wav") == "/tmp/test.wav"

    def test_file_uri_windows(self):
        from lab_lounge.audio_io import _uri_to_path
        result = _uri_to_path("file:///C:/data/test.wav")
        assert result == "C:/data/test.wav"

    def test_non_file_uri_unchanged(self):
        from lab_lounge.audio_io import _uri_to_path
        url = "https://example.com/audio.wav"
        assert _uri_to_path(url) == url

    def test_relative_path_unchanged(self):
        from lab_lounge.audio_io import _uri_to_path
        assert _uri_to_path("./data/audio/test.wav") == "./data/audio/test.wav"
