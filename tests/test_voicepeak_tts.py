"""
test_voicepeak_tts.py — VOICEPEAK TTS アダプタのテスト

subprocess をモックして voicepeak コマンド呼び出しを検証する。
"""

import shlex
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.tts import _call_voicepeak, _split_text_for_voicepeak


def _create_dummy_wav(filepath: Path, duration_ms: int = 1000, sample_rate: int = 24000) -> None:
    """テスト用のダミー WAV ファイルを作成する。"""
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _parse_cmd(cmd) -> list[str]:
    """mock_run に渡された cmd (文字列 or リスト) をトークンリストに変換する。"""
    if isinstance(cmd, str):
        # Windows 環境の subprocess.list2cmdline 出力を再パース
        # shlex.split(posix=False) で Windows 形式のクォートを扱う
        return shlex.split(cmd, posix=False)
    return list(cmd)


def _extract_arg(tokens: list[str], flag: str) -> str | None:
    """トークンリストから --flag VALUE のペアを取得する。"""
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            # shlex がダブルクォートを残す場合があるので strip
            return tokens[i + 1].strip('"')
    return None


def _make_mock_run(*, captured_cmd: list | None = None, call_count: list | None = None,
                   say_texts: list | None = None, duration_ms: int = 1000):
    """テスト用の subprocess.run モックファクトリ。"""
    def mock_run(cmd, **kwargs):
        tokens = _parse_cmd(cmd)
        if call_count is not None:
            call_count[0] += 1
        if captured_cmd is not None:
            captured_cmd.clear()
            captured_cmd.extend(tokens)
        if say_texts is not None:
            v = _extract_arg(tokens, "--say")
            if v is not None:
                say_texts.append(v)
        out_path = _extract_arg(tokens, "--out")
        if out_path:
            _create_dummy_wav(Path(out_path), duration_ms=duration_ms)
        return MagicMock(returncode=0)
    return mock_run


class TestCallVoicepeak:
    def test_calls_subprocess(self, tmp_path):
        """voicepeak コマンドが正しい引数で呼ばれる。"""
        with patch.object(subprocess, "run", side_effect=_make_mock_run()):
            result = _call_voicepeak(
                "テスト音声",
                voice="彩澄りりせ",
                output_dir=str(tmp_path),
                speaker="mimi",
            )

        assert result.voice == "彩澄りりせ"
        assert result.speaker == "mimi"
        assert result.format == "wav"
        assert result.duration_ms > 0
        assert result.audio_url.startswith("file:///")

    def test_custom_speed(self, tmp_path):
        """--speed パラメータが渡される。"""
        captured_cmd = []

        with patch.object(subprocess, "run", side_effect=_make_mock_run(captured_cmd=captured_cmd)):
            _call_voicepeak(
                "高速テスト",
                voice="宮舞モカ",
                output_dir=str(tmp_path),
                speed=150,
            )

        assert "--speed" in captured_cmd
        assert "150" in captured_cmd

    def test_command_not_found_raises(self, tmp_path):
        """voicepeak コマンドが見つからない場合 RuntimeError。"""
        from lab_lounge.tts import _submit_voicepeak

        def mock_submit(cmd_str):
            raise RuntimeError("VOICEPEAK 実行エラー: voicepeak not found")

        with patch("lab_lounge.tts._submit_voicepeak", side_effect=mock_submit):
            with pytest.raises(RuntimeError, match="VOICEPEAK"):
                _call_voicepeak(
                    "テスト",
                    voice="彩澄りりせ",
                    output_dir=str(tmp_path),
                )

    def test_provider_registered(self):
        """voicepeak が _PROVIDERS に登録されている。"""
        from lab_lounge.tts import _PROVIDERS
        assert "voicepeak" in _PROVIDERS

    def test_json_text_extracts_response_and_emotion(self, tmp_path):
        """JSON 形式の text から response / emotion / speed を分離する。"""
        import json

        json_text = json.dumps({
            "emotion": {"bosoboso": 10, "honwaka": 15},
            "speed": 100,
            "response": "了解です。",
        })
        captured_cmd = []

        with patch.object(subprocess, "run", side_effect=_make_mock_run(captured_cmd=captured_cmd)):
            result = _call_voicepeak(
                json_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        # --say には response テキストのみが渡される
        say_value = _extract_arg(captured_cmd, "--say")
        assert say_value is not None
        assert "了解です。" in say_value
        assert "emotion" not in say_value
        assert "speed" not in say_value

        # --emotion が渡される
        emo_value = _extract_arg(captured_cmd, "--emotion")
        assert emo_value is not None
        assert "bosoboso=10" in emo_value
        assert "honwaka=15" in emo_value

        # --speed が渡される
        speed_value = _extract_arg(captured_cmd, "--speed")
        assert speed_value == "100"

    def test_json_normalized_fullwidth_text(self, tmp_path):
        """全角に正規化済みの JSON も正しくパースされる。"""
        from lab_lounge.tts import _normalize_for_voicepeak
        import json

        original = json.dumps({
            "emotion": {"happy": 50},
            "speed": 120,
            "response": "こんにちは",
        })
        normalized = _normalize_for_voicepeak(original)
        captured_cmd = []

        with patch.object(subprocess, "run", side_effect=_make_mock_run(captured_cmd=captured_cmd)):
            _call_voicepeak(
                normalized,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        say_value = _extract_arg(captured_cmd, "--say")
        assert say_value is not None
        assert "こんにちは" in say_value
        assert _extract_arg(captured_cmd, "--emotion") is not None
        assert _extract_arg(captured_cmd, "--speed") is not None

    def test_plain_text_unchanged(self, tmp_path):
        """JSON でない通常テキストはそのまま --say に渡される。"""
        captured_cmd = []

        with patch.object(subprocess, "run", side_effect=_make_mock_run(captured_cmd=captured_cmd)):
            _call_voicepeak(
                "普通のテキスト",
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        say_value = _extract_arg(captured_cmd, "--say")
        assert say_value is not None
        assert "普通のテキスト" in say_value
        assert _extract_arg(captured_cmd, "--emotion") is None

    def test_json_speed_overrides_kwarg_speed(self, tmp_path):
        """JSON 内の speed は引数 speed より優先される。"""
        import json

        json_text = json.dumps({
            "speed": 80,
            "response": "ゆっくり",
        })
        captured_cmd = []

        with patch.object(subprocess, "run", side_effect=_make_mock_run(captured_cmd=captured_cmd)):
            _call_voicepeak(
                json_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
                speed=150,  # JSON の 80 が優先される
            )

        assert _extract_arg(captured_cmd, "--speed") == "80"


class TestSplitTextForVoicepeak:
    """_split_text_for_voicepeak の分割ロジックテスト。"""

    def test_short_text_single_chunk(self):
        """140字以内のテキストは分割しない。"""
        chunks = _split_text_for_voicepeak("短いテキストです。")
        assert chunks == ["短いテキストです。"]

    def test_empty_text(self):
        chunks = _split_text_for_voicepeak("")
        assert chunks == [""]

    def test_exactly_140_chars_no_split(self):
        text = "あ" * 140
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_period(self):
        """句点で分割される。"""
        text = "あ" * 100 + "。" + "い" * 100
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 100 + "。"
        assert chunks[1] == "い" * 100

    def test_splits_at_question_mark(self):
        """疑問符で分割される。"""
        text = "あ" * 100 + "？" + "い" * 100
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 100 + "？"

    def test_splits_at_exclamation(self):
        text = "あ" * 100 + "！" + "い" * 100
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 100 + "！"

    def test_prefers_last_primary_split(self):
        """複数の句点がある場合、最後の句点で分割 (140字最大活用)。"""
        text = "あ" * 50 + "。" + "い" * 50 + "。" + "う" * 50
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 50 + "。" + "い" * 50 + "。"

    def test_falls_back_to_comma(self):
        """句点がない場合は読点で分割。"""
        text = "あ" * 100 + "、" + "い" * 100
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 100 + "、"

    def test_hard_cut_when_no_punctuation(self):
        """句読点がない場合は 140 字で強制カット。"""
        text = "あ" * 200
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2
        assert len(chunks[0]) == 140
        assert len(chunks[1]) == 60

    def test_three_chunks(self):
        """複数回の分割が正しく動作する。"""
        text = "あ" * 130 + "。" + "い" * 130 + "。" + "う" * 50
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 3

    def test_custom_max_chars(self):
        text = "あ" * 10 + "。" + "い" * 10
        chunks = _split_text_for_voicepeak(text, max_chars=11)
        assert len(chunks) == 2
        assert chunks[0] == "あ" * 10 + "。"
        assert chunks[1] == "い" * 10

    def test_newline_split(self):
        """改行で分割される。"""
        text = "あ" * 100 + "\n" + "い" * 100
        chunks = _split_text_for_voicepeak(text)
        assert len(chunks) == 2


class TestVoicepeakChunking:
    """複数チャンク生成の統合テスト。"""

    def test_multi_chunk_calls_subprocess_per_chunk(self, tmp_path):
        """長いテキストが複数の subprocess 呼び出しに分割される。"""
        call_count = [0]
        long_text = "これは長いテキストです。" * 20

        with patch.object(subprocess, "run", side_effect=_make_mock_run(call_count=call_count)):
            result = _call_voicepeak(
                long_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        assert call_count[0] > 1
        assert len(result.chunk_audio_urls) == call_count[0]
        assert result.duration_ms > 0
        assert result.audio_url.startswith("file:///")

    def test_single_chunk_no_concatenation(self, tmp_path):
        """短いテキストは 1 チャンクで結合なし。"""
        with patch.object(subprocess, "run", side_effect=_make_mock_run()):
            result = _call_voicepeak(
                "短いテキスト",
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        assert len(result.chunk_audio_urls) == 1
        assert result.audio_url == result.chunk_audio_urls[0]

    def test_on_chunk_ready_called_per_chunk(self, tmp_path):
        """on_chunk_ready が各チャンク生成後に呼ばれる。"""
        chunk_urls = []
        long_text = "これは長いテキストです。" * 20

        with patch.object(subprocess, "run", side_effect=_make_mock_run()):
            result = _call_voicepeak(
                long_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
                on_chunk_ready=lambda url: chunk_urls.append(url),
            )

        assert len(chunk_urls) == len(result.chunk_audio_urls)
        assert all(url.startswith("file:///") for url in chunk_urls)

    def test_on_chunk_ready_not_called_when_none(self, tmp_path):
        """コールバック未指定時はエラーなく動作する。"""
        long_text = "これは長いテキストです。" * 20

        with patch.object(subprocess, "run", side_effect=_make_mock_run()):
            result = _call_voicepeak(
                long_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
                on_chunk_ready=None,
            )

        assert len(result.chunk_audio_urls) > 1

    def test_combined_duration_is_sum(self, tmp_path):
        """結合 duration_ms が各チャンクの合計になる。"""
        long_text = "これは長いテキストです。" * 20

        with patch.object(subprocess, "run", side_effect=_make_mock_run(duration_ms=500)):
            result = _call_voicepeak(
                long_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        expected_total = 500 * len(result.chunk_audio_urls)
        assert result.duration_ms == expected_total

    def test_each_chunk_under_140_chars(self, tmp_path):
        """各チャンクの --say テキストが 140 字以内。"""
        say_texts = []
        long_text = "これは長いテキストです。" * 20

        with patch.object(subprocess, "run", side_effect=_make_mock_run(say_texts=say_texts)):
            _call_voicepeak(
                long_text,
                voice="宮舞モカ",
                output_dir=str(tmp_path),
            )

        assert len(say_texts) > 1
        for t in say_texts:
            assert len(t) <= 140, f"チャンクが 140 字超: {len(t)} 字"
