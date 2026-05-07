"""
test_filler.py — フィラー音声のテスト
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.filler import (
    FillerPhrase,
    FillerPhraseSet,
    _generate_filler_text,
    _parse_emotion,
    get_cached_filler_paths,
    get_filler_duration_ms,
    is_filler_enabled,
    load_filler_phrases,
    run_filler_loop,
    select_filler_path,
    select_filler_phrase,
)


# ─── TestParseEmotion ─────────────────────────────────────────────

class TestParseEmotion:
    def test_single_value(self):
        assert _parse_emotion("happy=80") == {"happy": 80}

    def test_multiple_values(self):
        result = _parse_emotion("happy=80,sad=0,angry=20")
        assert result == {"happy": 80, "sad": 0, "angry": 20}

    def test_empty_string(self):
        assert _parse_emotion("") == {}

    def test_invalid_format_skipped(self):
        assert _parse_emotion("invalid") == {}


# ─── TestFillerPhraseSet ──────────────────────────────────────────

class TestFillerPhraseSet:
    def test_empty_set(self):
        ps = FillerPhraseSet()
        assert len(ps) == 0
        assert ps.all_phrases == []

    def test_len_counts_all_categories(self):
        ps = FillerPhraseSet(
            opener=[FillerPhrase("a"), FillerPhrase("b")],
            continue_=[FillerPhrase("c")],
            closer=[FillerPhrase("d")],
        )
        assert len(ps) == 4

    def test_all_phrases_preserves_category(self):
        ps = FillerPhraseSet(
            opener=[FillerPhrase("op")],
            continue_=[FillerPhrase("cont")],
            closer=[FillerPhrase("cl")],
        )
        items = ps.all_phrases
        assert items[0] == ("opener", FillerPhrase("op"))
        assert items[1] == ("continue", FillerPhrase("cont"))
        assert items[2] == ("closer", FillerPhrase("cl"))


# ─── TestLoadFillerPhrases ────────────────────────────────────────

class TestLoadFillerPhrases:
    def test_load_existing_phrases(self):
        result = load_filler_phrases("mimi")
        assert isinstance(result, FillerPhraseSet)
        assert len(result) >= 8  # opener 4+ continue 4+

    def test_load_nonexistent_slug_returns_empty(self):
        result = load_filler_phrases("nonexistent_character")
        assert isinstance(result, FillerPhraseSet)
        assert len(result) == 0

    def test_all_characters_have_phrases(self):
        for slug in ("mimi", "chisame", "sakura", "octamaid"):
            result = load_filler_phrases(slug)
            assert len(result.opener) >= 3, f"{slug} の opener が少なすぎます"
            assert len(result.continue_) >= 5, f"{slug} の continue が少なすぎます"

    def test_opener_and_continue_separated(self):
        result = load_filler_phrases("mimi")
        assert len(result.opener) >= 3
        assert len(result.continue_) >= 5

    def test_mimi_has_emotion(self):
        result = load_filler_phrases("mimi")
        all_phrases = [p for _, p in result.all_phrases]
        with_emotion = [p for p in all_phrases if p.emotion is not None]
        assert len(with_emotion) >= 3

    def test_octamaid_no_emotion(self):
        result = load_filler_phrases("octamaid")
        for _, p in result.all_phrases:
            assert p.emotion is None

    def test_legacy_file_without_sections(self, tmp_path):
        """セクションヘッダーなしのファイル → 全て opener に入る。"""
        legacy = tmp_path / "legacy.txt"
        legacy.write_text("フレーズA\nフレーズB\n", encoding="utf-8")

        with patch("lab_lounge.filler._FILLER_PHRASES_DIR", tmp_path):
            result = load_filler_phrases("legacy")

        assert len(result.opener) == 2
        assert len(result.continue_) == 0

    def test_emotion_parsed_correctly(self):
        result = load_filler_phrases("mimi")
        all_phrases = [p for _, p in result.all_phrases]
        first_with_emotion = next(p for p in all_phrases if p.emotion)
        assert "happy" in first_with_emotion.emotion
        assert isinstance(first_with_emotion.emotion["happy"], int)


# ─── TestIsFillerEnabled ──────────────────────────────────────────

class TestIsFillerEnabled:
    def test_default_is_true(self):
        assert is_filler_enabled() is True

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_FILLER_ENABLED", "false")
        assert is_filler_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_FILLER_ENABLED", "true")
        assert is_filler_enabled() is True


# ─── TestSelectFillerPath ─────────────────────────────────────────

class TestSelectFillerPath:
    def test_nonexistent_slug_returns_none(self):
        path, idx = select_filler_path("nonexistent", "opener")
        assert path is None
        assert idx == -1

    def test_returns_valid_path_from_cache(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        for i in range(3):
            (cache_dir / f"opener_{i:02d}.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            path, idx = select_filler_path("test_slug", "opener")
            assert path is not None
            assert path.name.startswith("opener_")
            assert 0 <= idx < 3

    def test_selects_correct_category(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            path, _ = select_filler_path("test_slug", "continue")
            assert path is not None
            assert path.name.startswith("continue_")

    def test_avoids_last_index(self, tmp_path):
        cache_dir = tmp_path / "test_slug"
        cache_dir.mkdir()
        for i in range(3):
            (cache_dir / f"continue_{i:02d}.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            for _ in range(100):
                _, idx = select_filler_path("test_slug", "continue", last_index=1)
                assert idx != 1


# ─── TestGetFillerDurationMs ──────────────────────────────────────

class TestGetFillerDurationMs:
    def test_nonexistent_file_returns_zero(self, tmp_path):
        assert get_filler_duration_ms(tmp_path / "nope.wav") == 0

    def test_valid_wav_returns_duration(self, tmp_path):
        import wave
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)
        duration = get_filler_duration_ms(wav_path)
        assert duration == pytest.approx(1000, abs=10)


# ─── TestRunFillerLoop ────────────────────────────────────────────

class TestRunFillerLoop:
    def test_immediate_stop_exits(self, tmp_path):
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        stop.set()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            run_filler_loop("mimi", stop)

    def test_no_cache_exits(self):
        stop = threading.Event()
        with patch("lab_lounge.filler._FILLER_CACHE_DIR", Path("/nonexistent")):
            run_filler_loop("mimi", stop)

    def test_plays_opener_then_llm_continue(self, tmp_path):
        """opener → LLM 生成 continue の順序で再生される。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(str(path))
            # 2 回再生後に stop（Phase 4 のブロック防止）
            if len(played) >= 2:
                stop.set()
            return True

        mock_tts_result = MagicMock()
        mock_tts_result.audio_url = "file:///tmp/filler_runtime/test.wav"

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value="えーと、そうですわねぇ"), \
             patch("lab_lounge.tts.synthesize", return_value=mock_tts_result), \
             patch.object(Path, "unlink"), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        assert len(played) == 2
        assert "opener_00" in played[0]

    def test_llm_failure_falls_back_to_cached(self, tmp_path):
        """LLM 失敗 → 事前生成 continue にフォールバック。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(Path(path).name)
            if len(played) >= 2:
                stop.set()
            return True

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value=None), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        assert len(played) == 2
        assert played[0].startswith("opener_")
        assert played[1].startswith("continue_")

    def test_opener_only_when_no_continue_and_no_llm(self, tmp_path):
        """continue なし + LLM 失敗 → opener のみ再生して終了。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(Path(path).name)
            stop.set()  # Phase 4 ブロック防止
            return True

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value=None):
            run_filler_loop("mimi", stop)

        assert len(played) == 1
        assert played[0].startswith("opener_")

    def test_no_loop_after_llm_continue(self, tmp_path):
        """LLM continue は 1 回のみ。ループしない。"""
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()

        stop = threading.Event()
        played = []

        def mock_play(path):
            played.append(str(path))
            if len(played) >= 2:
                stop.set()  # Phase 4 ブロック防止
            return True

        mock_tts_result = MagicMock()
        mock_tts_result.audio_url = "file:///tmp/filler_runtime/test.wav"

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path), \
             patch("lab_lounge.audio_io.play_audio_file", side_effect=mock_play), \
             patch("lab_lounge.filler._generate_filler_text", return_value="テスト"), \
             patch("lab_lounge.tts.synthesize", return_value=mock_tts_result), \
             patch.object(Path, "unlink"), \
             patch("time.sleep"):
            run_filler_loop("mimi", stop)

        # opener + LLM continue の計 2 回のみ
        assert len(played) == 2


# ─── TestGetCachedFillerPaths ─────────────────────────────────────

class TestGetCachedFillerPaths:
    def test_empty_dir_returns_empty_dict(self, tmp_path):
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            # Phase 0.5-A フェーズ 2: handraise カテゴリを含む 5 カテゴリすべてが空 list
            assert result == {
                "opener": [], "continue": [], "bridge": [],
                "closer": [], "handraise": [],
            }

    def test_returns_categorized_paths(self, tmp_path):
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").touch()
        (cache_dir / "opener_01.wav").touch()
        (cache_dir / "continue_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert len(result["opener"]) == 2
            assert len(result["continue"]) == 1
            assert len(result["closer"]) == 0
            # 5 カテゴリ全部が dict に存在 (Phase 0.5-A フェーズ 2 の対応漏れ修正で追加)
            assert "handraise" in result

    def test_ignores_old_naming(self, tmp_path):
        """旧形式の 00.wav は無視される。"""
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        (cache_dir / "00.wav").touch()
        (cache_dir / "opener_00.wav").touch()

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert len(result["opener"]) == 1

    def test_recognizes_handraise_paths(self, tmp_path):
        """Phase 0.5-A フェーズ 2: handraise_*.wav が dict に取り込まれる。"""
        cache_dir = tmp_path / "test"
        cache_dir.mkdir()
        (cache_dir / "handraise_00.wav").touch()
        (cache_dir / "handraise_01.wav").touch()
        (cache_dir / "handraise_02.wav").touch()
        (cache_dir / "opener_00.wav").touch()  # 別カテゴリの混入確認

        with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path):
            result = get_cached_filler_paths("test")
            assert len(result["handraise"]) == 3
            assert len(result["opener"]) == 1
            # ファイル名末尾が handraise_NN.wav パターン以外は混入しない
            for p in result["handraise"]:
                assert p.name.startswith("handraise_")
                assert p.name.endswith(".wav")


class TestEnsureFillerCacheHandraise:
    """Phase 0.5-A フェーズ 2 対応漏れ修正: ensure_filler_cache が handraise
    セクションの phrase で KeyError を起こさず、result_paths に格納される。

    実走 (2026-05-07) で ``KeyError: 'handraise'`` が判明したため回帰テストとして
    追加。原因は ``counters`` / ``result_paths`` dict に "handraise" キーが
    無かったこと (5 箇所の対応漏れ)。
    """

    def test_handraise_phrase_does_not_cause_key_error(self, tmp_path, monkeypatch):
        """handraise エントリがある phrase_set でも KeyError なく完了する。"""
        from lab_lounge.filler import (
            FillerPhrase,
            FillerPhraseSet,
            ensure_filler_cache,
        )

        # ダミー phrase_set: opener と handraise を 1 つずつ
        fake_phrase_set = FillerPhraseSet(
            opener=[FillerPhrase(text="oa", emotion=None)],
            handraise=[FillerPhrase(text="ha", emotion=None)],
        )
        monkeypatch.setattr(
            "lab_lounge.filler.load_filler_phrases", lambda slug: fake_phrase_set,
        )
        monkeypatch.setattr("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path)

        # 既存 wav を準備して TTS 呼出をスキップさせる (target.is_file() and not force)
        cache_dir = tmp_path / "mimi"
        cache_dir.mkdir()
        (cache_dir / "opener_00.wav").write_bytes(b"\x00\x00")
        (cache_dir / "handraise_00.wav").write_bytes(b"\x00\x00")

        # KeyError なく完了 (修正前は handraise エントリ反復時に KeyError)
        result = ensure_filler_cache("mimi")

        # 5 カテゴリ全部が dict に存在
        assert set(result.keys()) == {
            "opener", "continue", "bridge", "closer", "handraise",
        }
        # handraise の wav パスが正しく格納される
        assert len(result["handraise"]) == 1
        assert result["handraise"][0].name == "handraise_00.wav"
        # opener も同時に格納
        assert len(result["opener"]) == 1
        assert result["opener"][0].name == "opener_00.wav"

    def test_empty_phrase_set_returns_5_categories(self, tmp_path, monkeypatch):
        """phrase_set が空でも返り値 dict は 5 カテゴリすべて含む。"""
        from lab_lounge.filler import FillerPhraseSet, ensure_filler_cache

        empty_set = FillerPhraseSet()
        monkeypatch.setattr(
            "lab_lounge.filler.load_filler_phrases", lambda slug: empty_set,
        )
        monkeypatch.setattr("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path)

        result = ensure_filler_cache("mimi")
        assert set(result.keys()) == {
            "opener", "continue", "bridge", "closer", "handraise",
        }
        for cat_paths in result.values():
            assert cat_paths == []


# ─── TestGenerateFillerText ───────────────────────────────────────

def _make_openai_mock(response_text: str = "えーと、そうですわねぇ……"):
    """openai モジュールのモックを生成する。"""
    mock_openai = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = response_text
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_resp
    return mock_openai


class TestGenerateFillerText:
    def test_returns_text_on_success(self):
        mock_openai = _make_openai_mock("えーと、そうですわねぇ……")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result == "えーと、そうですわねぇ……"

    def test_returns_none_on_api_error(self):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result is None

    def test_returns_none_for_unknown_slug(self):
        # get_character() raises KeyError for unknown slugs
        result = _generate_filler_text("nonexistent_character")
        # _generate_filler_text catches the exception and returns None
        assert result is None

    def test_returns_none_on_empty_response(self):
        mock_openai = _make_openai_mock("")

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result is None

    def test_json_response_not_stripped(self):
        """LLM が JSON (emotion 付き) を返した場合、ストリップされずそのまま返る。"""
        import json as _json
        json_resp = _json.dumps({"response": "ふふ、お調べしますわ", "emotion": {"happy": 50}})
        mock_openai = _make_openai_mock(json_resp)

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result == json_resp
        # JSON が壊れていないことを確認
        parsed = _json.loads(result)
        assert parsed["response"] == "ふふ、お調べしますわ"
        assert parsed["emotion"]["happy"] == 50

    def test_plain_text_fallback_still_stripped(self):
        """LLM がプレーンテキストを返した場合、従来通りクォートが除去される。"""
        mock_openai = _make_openai_mock('"ふむ、考え中ですわ"')

        with patch.dict(sys.modules, {"openai": mock_openai}):
            result = _generate_filler_text("mimi")

        assert result == "ふむ、考え中ですわ"  # 外側の " が strip される


class TestBuildFillerPrompt:
    """_build_filler_prompt のテスト。"""

    def test_voicepeak_character_includes_emotion_keys(self):
        """mimi (voicepeak) のプロンプトに emotion キーと JSON 指示が含まれる。"""
        from lab_lounge.filler import _build_filler_prompt
        prompt = _build_filler_prompt("mimi")
        assert "JSON" in prompt
        assert '"happy"' in prompt
        assert '"fun"' in prompt
        assert '"sulky"' in prompt
        assert "テキストのみ出力" not in prompt  # 旧指示が除去されている

    def test_voicevox_character_skips_json(self):
        """octamaid (voicevox) は JSON 形式指示なし、プレーンテキストプロンプトのまま。"""
        from lab_lounge.filler import _build_filler_prompt
        prompt = _build_filler_prompt("octamaid")
        assert '"emotion"' not in prompt  # emotion JSON 指示が無い
        assert "テキストのみ出力" in prompt  # 旧指示が残っている

    def test_voicepeak_character_includes_pose(self):
        """mimi (voicepeak) のプロンプトに pose 指示が含まれる。"""
        from lab_lounge.filler import _build_filler_prompt
        prompt = _build_filler_prompt("mimi")
        assert '"pose"' in prompt
        assert "neutral" in prompt
        assert "happy" in prompt
        assert "fun" in prompt


# ─── TestHandraiseSection (Phase 0.5-A フェーズ 2) ─────────────────


class TestHandraiseSection:
    """data/filler_phrases/<slug>.txt の [handraise] セクションが読み込まれる。

    Phase 0.5-A で追加された挙手機能用のフレーズ。
    """

    def test_handraise_loaded_for_mimi(self):
        """mimi の [handraise] セクションが読み込まれる。"""
        phrase_set = load_filler_phrases("mimi")
        assert len(phrase_set.handraise) > 0
        texts = [p.text for p in phrase_set.handraise]
        # Phase 0.5-A で追加した代表フレーズが含まれる
        assert any("わたくし" in t for t in texts)

    def test_handraise_loaded_for_chisame(self):
        """chisame の [handraise] セクションが読み込まれる。"""
        phrase_set = load_filler_phrases("chisame")
        assert len(phrase_set.handraise) > 0

    def test_handraise_loaded_for_sakura(self):
        """sakura の [handraise] セクションが読み込まれる。"""
        phrase_set = load_filler_phrases("sakura")
        assert len(phrase_set.handraise) > 0

    def test_handraise_in_all_phrases(self):
        """all_phrases プロパティに handraise カテゴリが含まれる。"""
        phrase_set = load_filler_phrases("mimi")
        cats = {cat for cat, _ in phrase_set.all_phrases}
        assert "handraise" in cats

    def test_handraise_counted_in_len(self):
        """__len__ に handraise の数が含まれる。"""
        empty = FillerPhraseSet()
        assert len(empty) == 0

        # handraise だけのセット
        only_handraise = FillerPhraseSet(
            handraise=[FillerPhrase(text="testA"), FillerPhrase(text="testB")]
        )
        assert len(only_handraise) == 2


# ─── TestSelectFillerPhrase (Phase 0.5-A フェーズ 2) ─────────────────


class TestSelectFillerPhrase:
    """select_filler_phrase は Path と FillerPhrase の両方を返す。

    handraise wav 再生時に bubble.text として元フレーズを取得する用途。
    """

    def test_returns_none_when_no_cache(self):
        """wav cache が空のキャラなら (None, None, -1) を返す。"""
        result = select_filler_phrase("nonexistent_slug_xyz", "handraise")
        assert result == (None, None, -1)

    def test_returns_none_when_no_phrases(self):
        """phrase 定義もキャッシュも空なら (None, None, -1)。"""
        # nonexistent slug は filler_phrases ファイルもないので空
        result = select_filler_phrase("nonexistent_slug_xyz", "opener")
        assert result == (None, None, -1)
