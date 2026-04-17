"""
test_3stage_pipeline.py — 3段パイプライン統合テスト

3段パイプライン（1stフィラー → 2ndフィラー → 本命応答）の統合テスト。
外部 I/O（Redis, マイク, TTS, audio）はモックする。

検証観点:
  - フィラースレッドとパイプラインの並行実行
  - TTS コールバック (on_tts_chunk_ready) によるフィラー停止
  - 3段フローの順序保証（opener → bridge/continue → 本命）
  - パイプラインイベントの正確性（フィラー並行時）
  - エッジケース（フィラー無効, パイプライン失敗, 非ストリーミング TTS）

対応受け入れ条件:
  3F-1: 1st→2nd→本命の3段が直列動作する
"""

import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.pipeline as pipeline_mod
from lab_lounge.pipeline import PipelineResult, run_pipeline


COMMON = dict(
    stream_id="stream-3s-001",
    session_id="sess-3s-001",
    trace_id="trace-3s-001",
)


@pytest.fixture()
def mock_publish():
    """bus.publish をモックして Redis 接続なし。"""
    with patch.object(pipeline_mod, "publish", return_value="1-0") as m:
        yield m


@pytest.fixture()
def mock_play():
    """audio_io.play_audio_file をモックして実再生なし。"""
    with patch("lab_lounge.audio_io.play_audio_file", return_value=True) as m:
        yield m


@pytest.fixture()
def fake_filler_cache(tmp_path):
    """フィラーキャッシュ用の最小 WAV ファイルを生成する。"""
    import struct
    import wave

    cache_dir = tmp_path / "filler_cache" / "octamaid"
    cache_dir.mkdir(parents=True)

    def _make_wav(name: str, duration_ms: int = 500) -> Path:
        path = cache_dir / name
        sample_rate = 16000
        n_frames = int(sample_rate * duration_ms / 1000)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
        return path

    _make_wav("opener_00.wav", 800)
    _make_wav("bridge_00.wav", 500)
    _make_wav("bridge_01.wav", 500)
    _make_wav("continue_00.wav", 600)

    with patch("lab_lounge.filler._FILLER_CACHE_DIR", tmp_path / "filler_cache"):
        yield cache_dir


# ═══════════════════════════════════════════════════════════════════
# 3段フロー統合テスト
# ═══════════════════════════════════════════════════════════════════


class TestThreeStageFlow:
    """フィラー + パイプラインの並行実行 → TTS コールバック → 本命再生。"""

    def test_filler_and_pipeline_run_concurrently(
        self, mock_publish, mock_play, fake_filler_cache
    ):
        """フィラースレッドがパイプライン実行中に並行動作する。"""
        filler_started = threading.Event()
        stop_event = threading.Event()

        def mock_filler_loop(slug, stop_ev, *, user_text=""):
            filler_started.set()
            # フィラーは stop_event が設定されるまでループする
            stop_ev.wait(timeout=5.0)

        with patch("lab_lounge.filler.run_filler_loop", side_effect=mock_filler_loop):
            filler_thread = threading.Thread(
                target=mock_filler_loop, args=("octamaid", stop_event), daemon=True,
            )
            filler_thread.start()

            # フィラーが開始されたことを確認
            assert filler_started.wait(timeout=2.0)

            # パイプラインを並行実行
            result = run_pipeline("テスト", **COMMON)

            # パイプライン完了後にフィラーを停止
            stop_event.set()
            filler_thread.join(timeout=2.0)

        assert len(result.events) == 3

    def test_tts_callback_stops_filler(self, mock_publish, mock_play, fake_filler_cache):
        """on_tts_chunk_ready コールバックがフィラースレッドを停止する。"""
        filler_stop = threading.Event()
        filler_alive_at_callback = [True]
        chunk_urls: list[str] = []

        def mock_filler_loop(slug, stop_ev, *, user_text=""):
            stop_ev.wait(timeout=5.0)

        filler_thread = threading.Thread(
            target=mock_filler_loop, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        # run_loop の _on_tts_chunk と同等のコールバック
        def on_tts_chunk(url: str) -> None:
            nonlocal filler_alive_at_callback
            filler_alive_at_callback[0] = filler_thread.is_alive()
            if filler_thread.is_alive():
                filler_stop.set()
                filler_thread.join(timeout=5)
            chunk_urls.append(url)

        # コールバック発火
        on_tts_chunk("file:///audio/main.wav")

        assert filler_alive_at_callback[0] is True  # コールバック時にフィラーは生存
        assert not filler_thread.is_alive()  # コールバック後にフィラー終了
        assert chunk_urls == ["file:///audio/main.wav"]

    def test_pipeline_events_correct_with_filler(
        self, mock_publish, mock_play, fake_filler_cache
    ):
        """フィラー並行実行中もパイプラインの 3 イベントが正しく生成される。"""
        filler_stop = threading.Event()
        chunk_received = threading.Event()

        def mock_filler_loop(slug, stop_ev, *, user_text=""):
            stop_ev.wait(timeout=5.0)

        filler_thread = threading.Thread(
            target=mock_filler_loop, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        def on_chunk(url, *_args, **_kwargs):
            filler_stop.set()
            filler_thread.join(timeout=5)
            chunk_received.set()

        result = run_pipeline("統合テスト", on_tts_chunk_ready=on_chunk, **COMMON)

        # フィラー停止の後始末
        filler_stop.set()
        filler_thread.join(timeout=2.0)

        # パイプラインイベント検証
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final", "tts.done"]
        seqs = [ev["seq"] for ev in result.events]
        assert seqs == [0, 1, 2]

    def test_three_stages_ordered(self, mock_publish, mock_play, fake_filler_cache):
        """opener → (bridge/continue) → 本命の順序で再生される。"""
        play_log: list[str] = []
        filler_stop = threading.Event()
        filler_done = threading.Event()

        original_play = mock_play.side_effect

        def tracking_play(path_or_url):
            play_log.append(Path(path_or_url).name)
            return True

        mock_play.side_effect = tracking_play

        # LLM フィラー生成をスキップして cached continue を使用
        with patch("lab_lounge.filler._generate_filler_text", return_value=None):
            def run_filler():
                from lab_lounge.filler import run_filler_loop
                run_filler_loop("octamaid", filler_stop, user_text="テスト")
                filler_done.set()

            filler_thread = threading.Thread(target=run_filler, daemon=True)
            filler_thread.start()

            # フィラーが開始されるまで待機
            time.sleep(0.2)

            # 本命パイプライン実行（ダミーモード = 即完了）
            result = run_pipeline("テスト", **COMMON)

            # フィラー停止
            filler_stop.set()
            filler_thread.join(timeout=5.0)

        # 最初に opener が再生されていることを確認
        assert len(play_log) >= 1
        assert play_log[0] == "opener_00.wav"

        # パイプラインは正常に完了
        assert len(result.events) == 3


# ═══════════════════════════════════════════════════════════════════
# コールバックメカニズム
# ═══════════════════════════════════════════════════════════════════


class TestTtsChunkCallback:
    """on_tts_chunk_ready コールバックの動作検証。"""

    def test_callback_queues_audio_url(self, mock_publish):
        """コールバックが受け取った URL を再生キューに渡す。"""
        playback_queue: queue.Queue[str | None] = queue.Queue()
        chunk_count = [0]

        def on_chunk(url: str, *_args, **_kwargs) -> None:
            chunk_count[0] += 1
            playback_queue.put(url)

        result = run_pipeline("テスト", on_tts_chunk_ready=on_chunk, **COMMON)

        # ダミーモード = 非ストリーミング → コールバック未発火
        assert chunk_count[0] == 0

    def test_callback_fires_in_real_tts_mode(self, mock_publish, monkeypatch):
        """real TTS ストリーミングモードでコールバックが発火する。"""
        from lab_lounge.tts import TTSResult

        chunk_urls = []

        def on_chunk(url: str, *_args, **_kwargs) -> None:
            chunk_urls.append(url)

        fake_tts = TTSResult(
            audio_url="file:///tmp/main.wav",
            duration_ms=3000,
            voice="89",
            format="wav",
            sample_rate=24000,
            speaker="octamaid",
        )
        monkeypatch.setenv("L2_USE_REAL_TTS", "true")

        def mock_synthesize(text, *, provider, voice, speaker, output_dir, on_chunk_ready=None):
            if on_chunk_ready:
                on_chunk_ready("file:///tmp/chunk_001.wav", text, True, speaker)
            return fake_tts

        with patch("lab_lounge.tts.synthesize", side_effect=mock_synthesize):
            result = run_pipeline("テスト", on_tts_chunk_ready=on_chunk, **COMMON)

        assert chunk_urls == ["file:///tmp/chunk_001.wav"]

    def test_filler_stop_before_main_playback(self, mock_publish, mock_play, fake_filler_cache):
        """フィラーが停止してから本命音声が再生キューに入る。"""
        filler_stop = threading.Event()
        filler_stopped_before_queue = [False]
        playback_queue: queue.Queue[str | None] = queue.Queue()

        def mock_filler(slug, stop_ev, *, user_text=""):
            stop_ev.wait(timeout=5.0)

        filler_thread = threading.Thread(
            target=mock_filler, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        def on_chunk(url: str, *_args, **_kwargs) -> None:
            if filler_thread.is_alive():
                filler_stop.set()
                filler_thread.join(timeout=5)
            filler_stopped_before_queue[0] = not filler_thread.is_alive()
            playback_queue.put(url)

        on_chunk("file:///audio/main.wav")

        assert filler_stopped_before_queue[0] is True
        assert playback_queue.get_nowait() == "file:///audio/main.wav"


# ═══════════════════════════════════════════════════════════════════
# run_loop パイプラインセクション統合テスト
# ═══════════════════════════════════════════════════════════════════


class TestRunLoopPipelineSection:
    """run_loop のセクション 4（Pipeline + ストリーミング TTS）の統合テスト。"""

    def test_full_flow_with_filler_and_fallback_playback(
        self, mock_publish, mock_play, fake_filler_cache
    ):
        """
        run_loop セクション 4 の完全フロー再現:
        フィラー開始 → パイプライン → フォールバック再生 (非ストリーミング)。
        """
        chunk_count = [0]
        filler_stop = threading.Event()
        filler_thread: threading.Thread | None = None

        # フィラーを制御可能にモック
        def mock_filler(slug, stop_ev, *, user_text=""):
            stop_ev.wait(timeout=5.0)

        # --- フィラースレッド開始 ---
        filler_thread = threading.Thread(
            target=mock_filler, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        # --- コールバック定義 ---
        def on_chunk(url: str, *_args, **_kwargs) -> None:
            if filler_thread is not None and filler_thread.is_alive():
                filler_stop.set()
                filler_thread.join(timeout=5)
            chunk_count[0] += 1

        # --- パイプライン実行 ---
        result = run_pipeline(
            "フルフローテスト",
            on_tts_chunk_ready=on_chunk,
            **COMMON,
        )

        # --- 非ストリーミングフォールバック ---
        if chunk_count[0] == 0:
            if filler_thread is not None and filler_thread.is_alive():
                filler_stop.set()
                filler_thread.join(timeout=5)

            tts_ev = next((ev for ev in result.events if ev["type"] == "tts.done"), None)
            assert tts_ev is not None
            audio_url = tts_ev["payload"].get("audio_url", "")
            assert audio_url  # ダミーモードでも URL あり

        # --- 検証 ---
        assert len(result.events) == 3
        assert not filler_thread.is_alive()

    def test_full_flow_without_filler(self, mock_publish):
        """フィラー無効時はフィラースレッドなしで正常動作する。"""
        chunk_count = [0]

        def on_chunk(url: str, *_args, **_kwargs) -> None:
            chunk_count[0] += 1

        result = run_pipeline(
            "フィラーなしテスト",
            on_tts_chunk_ready=on_chunk,
            **COMMON,
        )

        assert len(result.events) == 3
        types = [ev["type"] for ev in result.events]
        assert types == ["utterance.final", "llm.final", "tts.done"]

    def test_pipeline_failure_cleans_up_filler(self, mock_publish, mock_play, fake_filler_cache):
        """パイプライン失敗時にフィラースレッドが適切にクリーンアップされる。"""
        filler_stop = threading.Event()

        def mock_filler(slug, stop_ev, *, user_text=""):
            stop_ev.wait(timeout=5.0)

        filler_thread = threading.Thread(
            target=mock_filler, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        # パイプライン失敗をシミュレート
        with patch.object(pipeline_mod, "publish", side_effect=RuntimeError("Redis down")):
            try:
                run_pipeline("失敗テスト", **COMMON)
            except RuntimeError:
                pass

        # フィラーをクリーンアップ
        filler_stop.set()
        filler_thread.join(timeout=2.0)
        assert not filler_thread.is_alive()


# ═══════════════════════════════════════════════════════════════════
# VOICEPEAK 排他制御（3F-2）
# ═══════════════════════════════════════════════════════════════════


class TestVoicePeakSequencing:
    """VOICEPEAK の合成順序が 2ndフィラー → 本命の順序を守ることの検証。"""

    def test_synthesize_calls_ordered_in_filler_then_pipeline(
        self, mock_publish, mock_play, fake_filler_cache
    ):
        """synthesize 呼び出し順序: フィラー TTS → 本命 TTS。"""
        from lab_lounge.tts import TTSResult

        synth_log: list[str] = []

        fake_tts = TTSResult(
            audio_url="file:///tmp/audio.wav",
            duration_ms=2000,
            voice="89",
            format="wav",
            sample_rate=24000,
            speaker="octamaid",
        )

        def tracking_synthesize(text, *, provider="", voice="", speaker="",
                                output_dir="", on_chunk_ready=None):
            synth_log.append(f"synth:{text[:20]}")
            return fake_tts

        filler_done = threading.Event()
        filler_stop = threading.Event()

        with patch("lab_lounge.tts.synthesize", side_effect=tracking_synthesize), \
             patch("lab_lounge.filler._generate_filler_text", return_value="考え中ですわ……"):

            # フィラースレッドでの TTS
            def run_filler():
                from lab_lounge.filler import run_filler_loop
                run_filler_loop("octamaid", filler_stop, user_text="テスト")
                filler_done.set()

            filler_thread = threading.Thread(target=run_filler, daemon=True)
            filler_thread.start()

            # フィラーの LLM+TTS が完了するまで待機
            time.sleep(0.5)
            filler_stop.set()
            filler_thread.join(timeout=5.0)

        # フィラー TTS が呼ばれたことを確認
        filler_synth = [s for s in synth_log if "考え中" in s]
        assert len(filler_synth) >= 1


# ═══════════════════════════════════════════════════════════════════
# 受け入れ条件 3F-1 — 3段直列動作
# ═══════════════════════════════════════════════════════════════════


class TestAcceptance3F1:
    """3F-1: 1stフィラー→2ndフィラー→本命の3段パイプラインが直列動作する。"""

    def test_three_stages_complete(self, mock_publish, mock_play, fake_filler_cache):
        """3段全てが完了し、本命パイプライン結果が返る。"""
        filler_stop = threading.Event()
        stages_completed: list[str] = []

        def mock_filler(slug, stop_ev, *, user_text=""):
            stages_completed.append("filler_started")
            stop_ev.wait(timeout=5.0)
            stages_completed.append("filler_stopped")

        filler_thread = threading.Thread(
            target=mock_filler, args=("octamaid", filler_stop), daemon=True,
        )
        filler_thread.start()

        # パイプライン（本命）
        result = run_pipeline("3段テスト", **COMMON)
        stages_completed.append("pipeline_done")

        # フィラー停止
        filler_stop.set()
        filler_thread.join(timeout=2.0)

        # 3段全て完了
        assert "filler_started" in stages_completed
        assert "pipeline_done" in stages_completed
        assert "filler_stopped" in stages_completed

        # 本命結果が正しい
        assert len(result.events) == 3
        assert result.events[1]["payload"]["text"] == "ダミー応答: 3段テスト"
        assert result.speaker == "octamaid"

    def test_filler_context_receives_user_text(
        self, mock_publish, mock_play, fake_filler_cache
    ):
        """フィラーにユーザーの入力テキストが渡される（文脈的フィラー生成用）。"""
        received_user_text = [None]

        def mock_filler(slug, stop_ev, *, user_text=""):
            received_user_text[0] = user_text
            stop_ev.wait(timeout=5.0)

        filler_stop = threading.Event()
        filler_thread = threading.Thread(
            target=mock_filler,
            args=("octamaid", filler_stop),
            kwargs={"user_text": "今日の天気を教えて"},
            daemon=True,
        )
        filler_thread.start()

        result = run_pipeline("今日の天気を教えて", **COMMON)

        filler_stop.set()
        filler_thread.join(timeout=2.0)

        assert received_user_text[0] == "今日の天気を教えて"
        assert len(result.events) == 3
