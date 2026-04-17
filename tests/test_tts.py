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
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert isinstance(result, TTSResult)

    def test_result_audio_url_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.audio_url == "file:///tmp/test-audio.mp3"

    def test_result_duration_ms_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.duration_ms == 2500

    def test_result_voice_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.voice == "ja-JP-NanamiNeural"

    def test_result_format_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.format == "mp3"

    def test_result_sample_rate_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.sample_rate == 24000

    def test_result_speaker_field(self):
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            result = synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural")
        assert result.speaker == "Nanami"

    def test_output_dir_forwarded(self):
        """output_dir 引数が provider に転送される"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            synthesize("テスト", provider="edge_tts", voice="ja-JP-NanamiNeural", output_dir="/custom/audio")
        mock_fn.assert_called_once_with(
            "テスト", voice="ja-JP-NanamiNeural", output_dir="/custom/audio"
        )

    def test_kwargs_forwarded_to_provider(self):
        """**kwargs が provider に転送される（speaker 等）"""
        mock_fn = _mock_provider(FAKE_TTS_RESULT)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(tts_mod._PROVIDERS, "edge_tts", mock_fn)
            synthesize(
                "テスト", provider="edge_tts", voice="ja-JP-NanamiNeural", speaker="Nanami"
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


class TestCallVoicevoxChunking:
    """Sprint Axis D (2026-04-14): _call_voicevox が VOICEPEAK と同じく
    140 字分割 + on_chunk_ready コールバックを行うことを検証。

    VOICEVOX API 自体は urllib で呼び出すため、`_generate_voicevox_single_file`
    をモックして API 呼び出しを回避する。
    """

    @pytest.fixture(autouse=True)
    def _mock_single_file(self, monkeypatch, tmp_path):
        """各チャンクのファイル生成をモックして API を叩かない。"""
        def fake_generate(text, *, speaker_id, voicevox_url, filepath):
            # 最小限の WAV ヘッダを書き込む (再生されることはないのでデータなしで可)
            # 呼び出し元は (duration_ms, sample_rate) を受け取れればよい
            filepath.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            return (1000, 24000)  # 1 秒, 24kHz

        monkeypatch.setattr(tts_mod, "_generate_voicevox_single_file", fake_generate)
        return fake_generate

    def test_short_text_single_chunk(self, tmp_path):
        """140 字以内の短文 → 1 チャンクのみ生成。"""
        result = tts_mod._call_voicevox(
            "短いテキストです。",
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
        )
        assert len(result.chunk_audio_urls) == 1
        assert result.audio_url == result.chunk_audio_urls[0]

    def test_long_text_split_into_multiple_chunks(self, tmp_path):
        """140 字超のテキスト → 複数チャンクに分割される (VOICEPEAK と同じロジック)。"""
        long_text = "これは長いテキストです。" * 20  # ~220 字
        result = tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
        )
        assert len(result.chunk_audio_urls) > 1

    def test_on_chunk_ready_called_per_chunk(self, tmp_path):
        """on_chunk_ready が各チャンク生成後に呼ばれる (url, text, is_last, speaker)。"""
        calls: list[tuple[str, str, bool, str]] = []
        long_text = "これは長いテキストです。" * 20

        result = tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
            on_chunk_ready=lambda url, text, is_last, speaker: calls.append(
                (url, text, is_last, speaker)
            ),
        )

        assert len(calls) == len(result.chunk_audio_urls)
        urls = [c[0] for c in calls]
        assert all(url.startswith("file:///") for url in urls)

    def test_on_chunk_ready_is_last_flag(self, tmp_path):
        """on_chunk_ready の is_last は最終チャンクのみ True (VOICEPEAK と同じ挙動)。"""
        is_last_flags: list[bool] = []
        long_text = "これは長いテキストです。" * 20

        tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
            on_chunk_ready=lambda url, text, is_last, speaker: is_last_flags.append(is_last),
        )

        assert len(is_last_flags) > 1
        assert is_last_flags[-1] is True
        assert all(f is False for f in is_last_flags[:-1])

    def test_on_chunk_ready_passes_speaker(self, tmp_path):
        """on_chunk_ready の第 4 引数に speaker (character slug) が渡る。"""
        speakers: list[str] = []
        long_text = "これは長いテキストです。" * 20

        tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
            on_chunk_ready=lambda url, text, is_last, speaker: speakers.append(speaker),
        )

        assert len(speakers) > 0
        assert all(s == "octamaid" for s in speakers)

    def test_on_chunk_ready_passes_chunk_text(self, tmp_path):
        """on_chunk_ready の第 2 引数に VOICEVOX に投入したチャンクテキストが渡る。"""
        texts: list[str] = []
        long_text = "これは長いテキストです。" * 20

        tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            speaker="octamaid",
            on_chunk_ready=lambda url, text, is_last, speaker: texts.append(text),
        )

        assert len(texts) > 1
        # 各チャンクは 140 字以内
        assert all(len(t) <= 140 for t in texts)

    def test_on_chunk_ready_not_called_when_none(self, tmp_path):
        """コールバック未指定時はエラーなく動作する。"""
        long_text = "これは長いテキストです。" * 20
        result = tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
            on_chunk_ready=None,
        )
        assert len(result.chunk_audio_urls) > 1

    def test_combined_duration_is_sum(self, tmp_path):
        """結合 duration_ms が各チャンクの合計になる。"""
        long_text = "これは長いテキストです。" * 20
        result = tts_mod._call_voicevox(
            long_text,
            voice="89",
            output_dir=str(tmp_path),
        )
        # fake_generate は各チャンク 1000ms を返すので合計 = 1000 × チャンク数
        expected = 1000 * len(result.chunk_audio_urls)
        assert result.duration_ms == expected

    def test_speaker_defaults_to_voicevox_prefix(self, tmp_path):
        """speaker 未指定時は TTSResult.speaker が voicevox-<id> 形式。"""
        result = tts_mod._call_voicevox(
            "短いテキスト",
            voice="89",
            output_dir=str(tmp_path),
        )
        assert result.speaker == "voicevox-89"


class TestParseVoicepeakJson:
    """_parse_voicepeak_json の pose 抽出テスト (4-tuple return)。"""

    def test_json_with_pose(self):
        """pose フィールド付き JSON → pose が抽出される。"""
        text = '{"emotion": {"happy": 50}, "speed": 100, "pose": "happy", "response": "テスト"}'
        say, emotion, speed, pose = tts_mod._parse_voicepeak_json(text)
        assert say == "テスト"
        assert emotion == {"happy": 50}
        assert speed == 100
        assert pose == "happy"

    def test_json_without_pose(self):
        """pose フィールドなし → pose が None。"""
        text = '{"emotion": {"happy": 50}, "speed": 100, "response": "テスト"}'
        _, _, _, pose = tts_mod._parse_voicepeak_json(text)
        assert pose is None

    def test_pose_normalized_to_lowercase(self):
        """pose 値は小文字化される。"""
        text = '{"pose": "HAPPY", "response": "テスト"}'
        _, _, _, pose = tts_mod._parse_voicepeak_json(text)
        assert pose == "happy"

    def test_pose_whitespace_stripped(self):
        """pose 値の前後空白は除去される。"""
        text = '{"pose": "  sad  ", "response": "テスト"}'
        _, _, _, pose = tts_mod._parse_voicepeak_json(text)
        assert pose == "sad"

    def test_pose_empty_string_becomes_none(self):
        """pose が空文字 → None。"""
        text = '{"pose": "", "response": "テスト"}'
        _, _, _, pose = tts_mod._parse_voicepeak_json(text)
        assert pose is None

    def test_non_json_returns_all_none(self):
        """非 JSON テキスト → 4-tuple (text, None, None, None)。"""
        text = "ただのテキストです"
        say, emotion, speed, pose = tts_mod._parse_voicepeak_json(text)
        assert say == text
        assert emotion is None
        assert speed is None
        assert pose is None

    def test_json_with_markdown_block(self):
        """マークダウンコードブロック付き JSON でも pose 抽出できる。"""
        text = '```json\n{"pose": "fun", "response": "テスト"}\n```'
        _, _, _, pose = tts_mod._parse_voicepeak_json(text)
        assert pose == "fun"

    def test_returns_4_tuple(self):
        """戻り値は 4-tuple である。"""
        result = tts_mod._parse_voicepeak_json('{"response": "x"}')
        assert len(result) == 4
