"""
test_irodori_tts.py — irodori-TTS adapter (VoiceDesign / pose-only / voices.json データ駆動) テスト

外部 HTTP サイドカー / GPU / 実モデルは一切呼ばない。
  - 設定は自己完結の voices.json fixture (tmp に書き L2_IRODORI_VOICES_JSON で指す)
  - チャンク生成 (_generate_irodori_single_file) をモックして HTTP を回避
  - HTTP クライアント自体は urllib.request.urlopen をモックして検証
全て決定性 (記憶: pytest 全 mock 方針)。CI / 無 GPU でも通る。

確定設計 (pose-only):
  - 声は固定 (anchor + caption + seed)。表現は pose (→正規絵文字 + caption サフィックス) と
    speed (→ds=max(0.85,100/speed)) のみ。emotion は使わない。
  - キャラ別設定は voices.json をデータ駆動で実行時ロード (単一の真実源)。
"""

import io
import json
import wave

import pytest

import lab_lounge.tts as tts_mod
from lab_lounge.tts import TTSResult, synthesize

# irodori_tts/duration.py の ALLOWED_ANNOTATION_EMOJIS の写し (リスト外は「まん」異音化)。
_ALLOWED_EMOJIS = set(
    "⏩⏱️⏸️🌬️🍭🎛️🎭🎵🐢🐱👂👃👅👌👏💋💥💦💪📄📞📢📣"
    "😆😊😌😎😏😒😖😟😠😪😭😮😰😱😲😴🙄🙏🤐🤔🤢🤧🤭🥤🥱🥴🥵🥹🥺🫣🫶📖"
)
_ALLOWED_EMOJIS.add("😮‍💨")  # ZWJ シーケンス

# 自己完結のテスト用 voices.json (実 reference_voices/ 非依存・決定性)。
# 構造は本物と同じ: common + voices{slug:{ref_wav, caption, seed, pose_map}}。
# pose_map は確定整合後の形 (mimi happy/fun 分離、chisame honwaka なし)。
_FIXTURE = {
    "model": "test",
    "common": {"cfg_scale_speaker": 5.0, "realtime": {"t_schedule_mode": "sway", "num_steps": 24}},
    "voices": {
        "mimi": {
            "ref_wav": "mimi_ref.wav", "caption": "テスト用ミミの声。", "seed": 3,
            "pose_map": {
                "neutral": {"emoji": "", "caption_suffix": ""},
                "happy": {"emoji": "🤭", "caption_suffix": ""},
                "fun": {"emoji": "🤭", "caption_suffix": ""},
                "special_sulky": {"emoji": "😏", "caption_suffix": " 拗ねてみせる演技。"},
            },
        },
        "chisame": {
            "ref_wav": "chisame_ref.wav", "caption": "テスト用ちさめの声。", "seed": 2,
            "pose_map": {
                "neutral": {"emoji": "", "caption_suffix": ""},
                "special_doya": {"emoji": "", "caption_suffix": " 自信を持って簡潔に。"},
            },
        },
        "sakura": {
            "ref_wav": "sakura_ref.wav", "caption": "テスト用さくらの声。", "seed": 11,
            "pose_map": {
                "neutral": {"emoji": "", "caption_suffix": ""},
                "happy": {"emoji": "🫶", "caption_suffix": ""},
                "special_whisper": {"emoji": "👂😮‍💨", "caption_suffix": " 囁くように。"},
            },
        },
        "aruka": {
            "ref_wav": "aruka_ref.wav", "caption": "テスト用アルカの声。", "seed": 2,
            "pose_map": {
                "neutral": {"emoji": "", "caption_suffix": ""},
                "special_misty": {"emoji": "", "caption_suffix": " 輪郭が霞むように。"},
            },
        },
    },
}


@pytest.fixture(autouse=True)
def _voices_fixture(tmp_path, monkeypatch):
    """テスト用 voices.json を tmp に書き、L2_IRODORI_VOICES_JSON で指す (全テスト共通)。

    読み辞書 (readings.json) は既定で **存在しないパス** を指して passthrough にする
    (実 reference_voices/readings.json に依存させず、既存テキスト assert を決定的に保つ)。
    読み適用を検証するテストは個別に readings fixture を書いて env を上書きする。
    """
    vj = tmp_path / "voices.json"
    vj.write_text(json.dumps(_FIXTURE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("L2_IRODORI_VOICES_JSON", str(vj))
    monkeypatch.setenv("L2_IRODORI_READINGS_JSON", str(tmp_path / "no_readings.json"))
    tts_mod._voices_cache.clear()  # 前テストの実 voices.json 等が残らないように
    tts_mod._readings_cache.clear()
    yield
    tts_mod._voices_cache.clear()
    tts_mod._readings_cache.clear()


def _make_wav_bytes(sample_rate: int = 48000, n_frames: int = 48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


# ─── speed clamp / provider 集合 ─────────────────────────────────────


class TestIrodoriMisc:
    def test_speed_to_duration_scale_clamped(self):
        assert tts_mod._speed_to_duration_scale(100) == 1.0
        assert tts_mod._speed_to_duration_scale(50) == 2.0
        assert tts_mod._speed_to_duration_scale(200) == 0.85   # クランプ
        assert tts_mod._speed_to_duration_scale(118) == 0.85
        assert tts_mod._speed_to_duration_scale(110) == round(100 / 110, 3)
        assert tts_mod._speed_to_duration_scale(None) == 1.0
        assert tts_mod._speed_to_duration_scale(0) == 1.0

    def test_irodori_vd_not_emotion_json(self):
        # irodori は pose-only。emotion JSON 包装の対象は VOICEPEAK のみ。
        assert not tts_mod.provider_uses_emotion_json("irodori_vd")
        assert tts_mod.provider_uses_emotion_json("voicepeak")


# ─── voices.json データ駆動 resolve ─────────────────────────────────


class TestResolveIrodoriVoice:
    @pytest.mark.parametrize("voice,seed", [("mimi", 3), ("chisame", 2), ("sakura", 11), ("aruka", 2)])
    def test_resolve_from_voices_json(self, voice, seed):
        e = tts_mod._resolve_irodori_voice(voice)
        # ref_wav は voices.json のあるディレクトリ基準で解決
        assert e["ref_wav"].endswith(f"{voice}_ref.wav")
        assert e["seed"] == seed
        assert e["cfg_scale_speaker"] == 5.0
        assert e["num_steps"] == 24 and e["t_schedule_mode"] == "sway"
        assert e["caption"]

    def test_unknown_voice_raises(self):
        with pytest.raises(ValueError, match="voices.json"):
            tts_mod._resolve_irodori_voice("does-not-exist")

    def test_missing_voices_json_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("L2_IRODORI_VOICES_JSON", str(tmp_path / "nope.json"))
        tts_mod._voices_cache.clear()
        with pytest.raises(FileNotFoundError, match="voices.json"):
            tts_mod._resolve_irodori_voice("mimi")


# ─── pose-only 制御マップ ───────────────────────────────────────────


class TestIrodoriControl:
    def test_mimi_pose_mapping(self):
        assert tts_mod._irodori_control("mimi", "happy") == ("🤭", "")
        assert tts_mod._irodori_control("mimi", "fun") == ("🤭", "")           # happy_fun 分離後
        emoji, suffix = tts_mod._irodori_control("mimi", "special_sulky")
        assert emoji == "😏" and "拗ねて" in suffix
        # pose_map に無い pose は neutral 扱い (mimi angry/sad はエントリなし)
        assert tts_mod._irodori_control("mimi", "angry") == ("", "")
        assert tts_mod._irodori_control("mimi", "neutral") == ("", "")

    def test_chisame_caption_only_and_no_honwaka(self):
        emoji, suffix = tts_mod._irodori_control("chisame", "special_doya")
        assert emoji == "" and suffix.strip() != ""
        # honwaka は廃止 → 制御なし
        assert tts_mod._irodori_control("chisame", "honwaka") == ("", "")

    def test_sakura_pose_mapping(self):
        assert tts_mod._irodori_control("sakura", "happy") == ("🫶", "")
        emoji, suffix = tts_mod._irodori_control("sakura", "special_whisper")
        assert emoji == "👂😮‍💨" and "囁く" in suffix

    def test_aruka_pose_only(self):
        emoji, suffix = tts_mod._irodori_control("aruka", "special_misty")
        assert emoji == "" and "霞む" in suffix
        assert tts_mod._irodori_control("aruka", "neutral") == ("", "")

    def test_no_pose_returns_empty(self):
        assert tts_mod._irodori_control("mimi", None) == ("", "")

    def test_unknown_char_returns_empty(self):
        assert tts_mod._irodori_control("nobody", "happy") == ("", "")

    def test_fixture_emojis_are_allowed(self):
        """fixture pose_map の絵文字が正規 ALLOWED に収まる (異音防止)。"""
        bad = []
        for v in _FIXTURE["voices"].values():
            for ctrl in v["pose_map"].values():
                rest = ctrl["emoji"]
                for tok in sorted(_ALLOWED_EMOJIS, key=len, reverse=True):
                    rest = rest.replace(tok, "")
                if rest.strip():
                    bad.append(ctrl["emoji"])
        assert not bad, f"非正規絵文字: {bad!r}"


# ─── _call_irodori (pose-only) — 単一ファイル生成はモック ───────────


class _CapturingGen:
    """_generate_irodori_single_file の差し替え。呼び出し引数を記録する。"""

    def __init__(self, sample_rate: int = 48000, duration_ms: int = 1000):
        self.calls: list[dict] = []
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms

    def __call__(self, chunk_text, *, caption, ref_wav, duration_scale, num_steps,
                 t_schedule_mode, cfg_scale_speaker, seed, filepath, url, seconds=None):
        self.calls.append({
            "text": chunk_text, "caption": caption, "ref_wav": ref_wav,
            "duration_scale": duration_scale, "num_steps": num_steps,
            "t_schedule_mode": t_schedule_mode, "cfg_scale_speaker": cfg_scale_speaker,
            "seed": seed, "url": url, "seconds": seconds,
        })
        filepath.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        return (self.duration_ms, self.sample_rate)


@pytest.fixture
def cap_gen(monkeypatch):
    gen = _CapturingGen()
    monkeypatch.setattr(tts_mod, "_generate_irodori_single_file", gen)
    return gen


class TestCallIrodoriVd:
    def test_short_text_single_chunk(self, tmp_path, cap_gen):
        result = tts_mod._call_irodori("短い文です。", voice="mimi", output_dir=str(tmp_path), speaker="mimi")
        assert len(result.chunk_audio_urls) == 1
        assert result.format == "wav" and result.sample_rate == 48000 and result.speaker == "mimi"

    def test_speaker_defaults(self, tmp_path, cap_gen):
        result = tts_mod._call_irodori("短文", voice="mimi", output_dir=str(tmp_path))
        assert result.speaker == "irodori-mimi"

    def test_long_text_multiple_chunks(self, tmp_path, cap_gen):
        long_text = "これは長いテキストです。" * 20
        result = tts_mod._call_irodori(long_text, voice="mimi", output_dir=str(tmp_path))
        assert len(result.chunk_audio_urls) > 1
        assert result.duration_ms == 1000 * len(result.chunk_audio_urls)

    def test_voices_json_params_sent(self, tmp_path, cap_gen):
        """voices.json の seed/cfg/num_steps/schedule/ref が送られる。"""
        tts_mod._call_irodori("テスト", voice="mimi", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert sent["seed"] == 3 and sent["cfg_scale_speaker"] == 5.0
        assert sent["num_steps"] == 24 and sent["t_schedule_mode"] == "sway"
        assert sent["ref_wav"].endswith("mimi_ref.wav")
        assert sent["caption"] == "テスト用ミミの声。"

    def test_env_overrides(self, tmp_path, cap_gen, monkeypatch):
        monkeypatch.setenv("L2_TTS_IRODORI_NUM_STEPS", "32")
        monkeypatch.setenv("L2_TTS_IRODORI_SCHEDULE", "linear")
        monkeypatch.setenv("L2_TTS_IRODORI_SEED", "99")
        tts_mod._call_irodori("テスト", voice="mimi", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert sent["num_steps"] == 32 and sent["t_schedule_mode"] == "linear" and sent["seed"] == 99

    def test_pose_drives_emoji_and_suffix(self, tmp_path, cap_gen):
        """pose → 本文末 emoji + caption サフィックス (emotion は無視)。"""
        text = '{"response": "こんにちは", "pose": "special_sulky", "speed": 100}'
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert sent["text"].endswith("😏")
        assert "テスト用ミミの声。" in sent["caption"] and "拗ねて" in sent["caption"]

    def test_emotion_is_ignored(self, tmp_path, cap_gen):
        """emotion を含む JSON でも pose=neutral なら制御なし (emotion 不参照)。"""
        text = '{"response": "やあ", "emotion": {"happy": 90}, "pose": "neutral", "speed": 100}'
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert sent["text"] == "やあ"  # 絵文字なし
        assert sent["caption"] == "テスト用ミミの声。"  # サフィックスなし

    def test_happy_pose_appends_emoji(self, tmp_path, cap_gen):
        text = '{"response": "うふふ", "pose": "happy", "speed": 100}'
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        assert cap_gen.calls[0]["text"].endswith("🤭")

    def test_on_chunk_ready_text_has_no_emoji(self, tmp_path, cap_gen):
        calls = []
        text = '{"response": "' + ("これは長いテキストです。" * 20) + '", "pose": "happy"}'
        tts_mod._call_irodori(
            text, voice="mimi", output_dir=str(tmp_path), speaker="mimi",
            on_chunk_ready=lambda url, t, is_last, sp: calls.append((url, t, is_last, sp)),
        )
        assert len(calls) > 1
        assert all("🤭" not in t for _, t, _, _ in calls)
        assert calls[-1][2] is True and all(c[2] is False for c in calls[:-1])

    def test_speed_clamped_into_ds(self, tmp_path, cap_gen):
        text = '{"response": "テスト", "speed": 200, "pose": "neutral"}'
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        assert cap_gen.calls[0]["duration_scale"] == 0.85

    def test_aruka_pose_suffix(self, tmp_path, cap_gen):
        text = '{"response": "整えますね", "speed": 90, "pose": "special_misty"}'
        tts_mod._call_irodori(text, voice="aruka", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert "霞む" in sent["caption"] and sent["text"] == "整えますね" and sent["seed"] == 2


# ─── HTTP クライアント (_generate_irodori_single_file) — urllib をモック ──


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestGenerateIrodoriSingleFile:
    def test_posts_vd_payload_and_writes_wav(self, tmp_path, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = req.data
            captured["method"] = req.get_method()
            return _FakeResp(_make_wav_bytes())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        out = tmp_path / "c.wav"
        dur, sr = tts_mod._generate_irodori_single_file(
            "こんにちは", caption="やわらかい声。", ref_wav="R:/a.wav",
            duration_scale=0.9, num_steps=24, t_schedule_mode="sway",
            cfg_scale_speaker=5.0, seed=3, filepath=out, url="http://127.0.0.1:50080/synthesize",
        )
        assert sr == 48000 and dur == 1000 and out.is_file()
        body = json.loads(captured["body"].decode("utf-8"))
        assert captured["method"] == "POST"
        assert body == {
            "mode": "vd", "text": "こんにちは", "caption": "やわらかい声。",
            "duration_scale": 0.9, "num_steps": 24, "t_schedule_mode": "sway",
            "cfg_scale_speaker": 5.0, "ref_wav": "R:/a.wav", "seed": 3,
        }

    def test_http_error_becomes_runtime_error(self, tmp_path, monkeypatch):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b'{"error":"boom"}'))

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="boom"):
            tts_mod._generate_irodori_single_file(
                "x", caption="c", ref_wav="r.wav", duration_scale=1.0, num_steps=24,
                t_schedule_mode="sway", cfg_scale_speaker=5.0, seed=None,
                filepath=tmp_path / "e.wav", url="http://127.0.0.1:50080/synthesize",
            )

    def test_connection_error_becomes_runtime_error(self, tmp_path, monkeypatch):
        import urllib.error
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        with pytest.raises(RuntimeError, match="接続できません"):
            tts_mod._generate_irodori_single_file(
                "x", caption="c", ref_wav="r.wav", duration_scale=1.0, num_steps=24,
                t_schedule_mode="sway", cfg_scale_speaker=5.0, seed=None,
                filepath=tmp_path / "e.wav", url="http://127.0.0.1:50080/synthesize",
            )


# ─── provider 登録 + synthesize + characters ────────────────────────


class TestProviderAndCharacters:
    def test_only_vd_registered(self):
        assert "irodori_vd" in tts_mod._PROVIDERS
        assert "irodori_clone" not in tts_mod._PROVIDERS

    def test_synthesize_dispatches_irodori_vd(self, tmp_path, cap_gen):
        result = synthesize(
            '{"response":"テスト","pose":"happy"}', provider="irodori_vd", voice="mimi",
            output_dir=str(tmp_path), speaker="mimi",
        )
        assert isinstance(result, TTSResult) and result.sample_rate == 48000
        assert cap_gen.calls[0]["ref_wav"].endswith("mimi_ref.wav")
        assert cap_gen.calls[0]["text"].endswith("🤭")  # happy pose

    def test_irodori_characters_have_no_emotion_axis(self):
        from lab_lounge.characters import get_character
        for slug in ("mimi", "chisame", "sakura", "aruka"):
            c = get_character(slug)
            assert c.tts_provider == "irodori_vd"
            assert c.voicepeak_emotion_keys == ()


# ─── 短文 manual duration (末尾幻聴抑制) ─────────────────────────────


class TestShortTextSeconds:
    @pytest.fixture(autouse=True)
    def _fixed_constants(self, monkeypatch):
        # env 汚染に左右されないよう定数を既定値に固定 (12 字 / 0.26 / 下限 0.6)。
        monkeypatch.setattr(tts_mod, "_IRODORI_SHORT_CHARS", 12)
        monkeypatch.setattr(tts_mod, "_IRODORI_SEC_PER_CHAR", 0.26)
        monkeypatch.setattr(tts_mod, "_IRODORI_MIN_SEC", 0.6)

    def test_short_text_gets_manual_seconds(self):
        assert tts_mod._short_text_seconds("橋を渡りますわ。") == 2.08  # 8 字 × 0.26

    def test_very_short_hits_floor(self):
        assert tts_mod._short_text_seconds("はい") == 0.6  # 2 × 0.26 = 0.52 → 下限 0.6

    def test_long_text_returns_none(self):
        # 12 字超は predictor 任せ (None)
        assert tts_mod._short_text_seconds("これは十二文字を超える長い文章ですわよ") is None

    def test_threshold_boundary(self):
        assert tts_mod._short_text_seconds("あ" * 12) == round(12 * 0.26, 2)  # ちょうど閾値 → manual
        assert tts_mod._short_text_seconds("あ" * 13) is None                 # 1 字超 → None

    def test_empty_or_blank_returns_none(self):
        assert tts_mod._short_text_seconds("") is None
        assert tts_mod._short_text_seconds("   ") is None

    def test_strips_whitespace_before_count(self):
        assert tts_mod._short_text_seconds("  橋を渡りますわ。  ") == 2.08  # 前後空白は数えない


# ─── 読み辞書 (readings) ────────────────────────────────────────────


_READINGS_FIXTURE = {
    "global": {
        "RAG": "ラグ", "Think-AI Lab": "シンカイラボ", "Lab": "ラボ",
        "波心": "ハゴコロ", "A.I.byss": "アイビス",
    },
    "characters": {"mimi": {"RAG": "ミミ用ラグ"}},   # キャラ別 override
    "_excluded_review": {"方": "カタ"},               # 適用されない多音字
    "_accent_meta": {"RAG": {"accentType": 0}},       # inert
}


@pytest.fixture
def readings(tmp_path, monkeypatch):
    """読み辞書 fixture を書いて L2_IRODORI_READINGS_JSON で指す (キャッシュもクリア)。"""
    rj = tmp_path / "readings.json"
    rj.write_text(json.dumps(_READINGS_FIXTURE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("L2_IRODORI_READINGS_JSON", str(rj))
    tts_mod._readings_cache.clear()
    yield
    tts_mod._readings_cache.clear()


class TestReadings:
    def test_missing_file_passthrough(self):
        # autouse fixture が存在しないパスを指す → 置換せずそのまま返る (辞書なしでも動く)
        assert tts_mod._apply_readings("RAGで検索", "mimi") == "RAGで検索"

    def test_load_missing_returns_empty(self):
        assert tts_mod._load_readings() == {}

    def test_global_applied(self, readings):
        assert tts_mod._apply_readings("波心さん", "sakura") == "ハゴコロさん"

    def test_longest_match_first(self, readings):
        # "Think-AI Lab" を "Lab" より優先
        assert tts_mod._apply_readings("Think-AI Labへようこそ", "sakura") == "シンカイラボへようこそ"
        # 単独の "Lab" は "ラボ"
        assert tts_mod._apply_readings("Lab見学", "sakura") == "ラボ見学"

    def test_character_override_beats_global(self, readings):
        assert tts_mod._apply_readings("RAG", "mimi") == "ミミ用ラグ"   # mimi 別定義
        assert tts_mod._apply_readings("RAG", "sakura") == "ラグ"        # global

    def test_excluded_review_not_applied(self, readings):
        # _excluded_review の "方" は loader が無視 → 置換されない (irodori に任せる)
        assert tts_mod._apply_readings("あちらの方へ", "sakura") == "あちらの方へ"

    def test_unregistered_passthrough(self, readings):
        assert tts_mod._apply_readings("ふつうの文章です", "sakura") == "ふつうの文章です"

    def test_symbol_surface_escaped(self, readings):
        # "A.I.byss" のドットは literal 化されてマッチ
        assert tts_mod._apply_readings("A.I.byssツール", "sakura") == "アイビスツール"
        # ドットが正規表現の任意一致にならない (AXIXbyss は無変換)
        assert tts_mod._apply_readings("AXIXbyss", "sakura") == "AXIXbyss"

    def test_single_pass_no_double_substitution(self, readings):
        # 1 パス置換なので、置換後カタカナが別 surface に再マッチしない
        assert tts_mod._apply_readings("Lab", "sakura") == "ラボ"


class TestCallIrodoriReadingsAndSeconds:
    def test_readings_applied_to_request_not_hud(self, tmp_path, cap_gen, readings):
        captured_hud = []
        text = '{"response": "波心さんとRAGの話", "pose": "neutral"}'
        tts_mod._call_irodori(
            text, voice="sakura", output_dir=str(tmp_path), speaker="sakura",
            on_chunk_ready=lambda url, t, is_last, sp: captured_hud.append(t),
        )
        # サイドカーへは読み補正済みテキスト (sakura は global)
        assert cap_gen.calls[0]["text"] == "ハゴコロさんとラグの話"
        # HUD (on_chunk_ready) には元テキストを渡す (画面は「波心」「RAG」のまま)
        assert captured_hud == ["波心さんとRAGの話"]

    def test_character_readings_in_call(self, tmp_path, cap_gen, readings):
        # mimi 経由なら RAG→ミミ用ラグ (キャラ別 override が効く)
        text = '{"response": "RAG", "pose": "neutral"}'
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        assert cap_gen.calls[0]["text"] == "ミミ用ラグ"

    def test_short_text_passes_seconds(self, tmp_path, cap_gen, monkeypatch):
        monkeypatch.setattr(tts_mod, "_IRODORI_SHORT_CHARS", 12)
        monkeypatch.setattr(tts_mod, "_IRODORI_SEC_PER_CHAR", 0.26)
        monkeypatch.setattr(tts_mod, "_IRODORI_MIN_SEC", 0.6)
        tts_mod._call_irodori("短い文です。", voice="mimi", output_dir=str(tmp_path))
        assert cap_gen.calls[0]["seconds"] == 1.56  # 6 字 × 0.26

    def test_long_text_seconds_none(self, tmp_path, cap_gen, monkeypatch):
        monkeypatch.setattr(tts_mod, "_IRODORI_SHORT_CHARS", 12)
        tts_mod._call_irodori(
            "これは十二文字を超える長い本文なので予測器に任せます。",
            voice="mimi", output_dir=str(tmp_path),
        )
        assert cap_gen.calls[0]["seconds"] is None

    def test_seconds_measured_on_spoken_text_not_emoji(self, tmp_path, cap_gen, monkeypatch):
        monkeypatch.setattr(tts_mod, "_IRODORI_SHORT_CHARS", 12)
        monkeypatch.setattr(tts_mod, "_IRODORI_SEC_PER_CHAR", 0.26)
        monkeypatch.setattr(tts_mod, "_IRODORI_MIN_SEC", 0.6)
        text = '{"response": "うふふ", "pose": "happy"}'  # 末尾に 🤭 が付く
        tts_mod._call_irodori(text, voice="mimi", output_dir=str(tmp_path))
        sent = cap_gen.calls[0]
        assert sent["text"].endswith("🤭")
        assert sent["seconds"] == 0.78  # "うふふ" = 3 字 × 0.26 (絵文字は尺基準に含めない)

    def test_decimal_normalized_in_request_not_hud(self, tmp_path, cap_gen):
        captured_hud = []
        text = '{"response": "GPT-5.5は賢いですわ", "pose": "neutral"}'
        tts_mod._call_irodori(
            text, voice="mimi", output_dir=str(tmp_path), speaker="mimi",
            on_chunk_ready=lambda url, t, is_last, sp: captured_hud.append(t),
        )
        # サイドカーへは小数正規化済み ("5.5"→"5てんご")
        assert cap_gen.calls[0]["text"] == "GPT-5てんごは賢いですわ"
        # HUD は元テキスト ("5.5" のまま)
        assert captured_hud == ["GPT-5.5は賢いですわ"]

    def test_readings_then_decimals_both_applied(self, tmp_path, cap_gen, readings):
        # 読み辞書 (A.I.byss→アイビス) → 小数正規化 (1.5→1てんご) の順で両方効く
        text = '{"response": "A.I.byss 1.5版", "pose": "neutral"}'
        tts_mod._call_irodori(text, voice="sakura", output_dir=str(tmp_path))
        assert cap_gen.calls[0]["text"] == "アイビス 1てんご版"


class TestNormalizeDecimals:
    def test_basic_single_digit(self):
        assert tts_mod._normalize_decimals("5.5") == "5てんご"
        assert tts_mod._normalize_decimals("3.1") == "3てんいち"
        assert tts_mod._normalize_decimals("4.6") == "4てんろく"

    def test_in_sentence(self):
        assert tts_mod._normalize_decimals("GPT-5.5は賢い") == "GPT-5てんごは賢い"

    def test_multi_digit_fraction_is_per_digit(self):
        # 小数部は桁読み: 14 を「じゅうよん」でなく「いちよん」
        assert tts_mod._normalize_decimals("3.14") == "3てんいちよん"

    def test_multi_digit_integer_kept_as_digits(self):
        assert tts_mod._normalize_decimals("12.5") == "12てんご"

    def test_zero_in_fraction(self):
        assert tts_mod._normalize_decimals("0.5") == "0てんご"
        assert tts_mod._normalize_decimals("2.05") == "2てんゼロご"

    def test_integer_untouched(self):
        assert tts_mod._normalize_decimals("2026年") == "2026年"
        assert tts_mod._normalize_decimals("5個ありますわ") == "5個ありますわ"

    def test_non_digit_dot_untouched(self):
        # 数字に挟まれない "." は対象外 (A.I.byss / 文末の句点)
        assert tts_mod._normalize_decimals("A.I.byss") == "A.I.byss"
        assert tts_mod._normalize_decimals("文の終わり。") == "文の終わり。"

    def test_multiple_decimals_in_text(self):
        assert tts_mod._normalize_decimals("5.5と3.1") == "5てんごと3てんいち"


class TestGenerateSecondsPayload:
    def test_posts_seconds_when_given(self, tmp_path, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = req.data
            return _FakeResp(_make_wav_bytes())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        tts_mod._generate_irodori_single_file(
            "やあ", caption="c", ref_wav="r.wav", duration_scale=1.0, num_steps=24,
            t_schedule_mode="sway", cfg_scale_speaker=5.0, seed=3,
            filepath=tmp_path / "s.wav", url="http://x/synthesize", seconds=2.08,
        )
        body = json.loads(captured["body"].decode("utf-8"))
        assert body["seconds"] == 2.08

    def test_no_seconds_key_when_none(self, tmp_path, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = req.data
            return _FakeResp(_make_wav_bytes())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        tts_mod._generate_irodori_single_file(
            "やあ", caption="c", ref_wav="r.wav", duration_scale=1.0, num_steps=24,
            t_schedule_mode="sway", cfg_scale_speaker=5.0, seed=3,
            filepath=tmp_path / "s.wav", url="http://x/synthesize",
        )
        body = json.loads(captured["body"].decode("utf-8"))
        assert "seconds" not in body
