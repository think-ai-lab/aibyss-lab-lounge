"""
test_run_loop.py — run_loop.py の _run_playback_worker の単体テスト

Sprint Axis D Block 1: OBS セリフテロップ表示
worker が speaking / done の bubble.update を正しいタイミングで publish することを検証。
"""

import queue
import threading
import time
from unittest.mock import MagicMock

from lab_lounge.run_loop import _run_playback_worker


class TestRunPlaybackWorker:
    """_run_playback_worker 単体テスト。"""

    def _start_worker(self, q, *, publish_fn=None, play_fn=None, cleanup_fn=None,
                      done_delay=0.01):
        """worker を別スレッドで起動する。"""
        publish_fn = publish_fn or MagicMock()
        play_fn = play_fn or MagicMock()
        cleanup_fn = cleanup_fn or MagicMock()
        thread = threading.Thread(
            target=_run_playback_worker,
            args=(q,),
            kwargs={
                "publish_bubble_fn": publish_fn,
                "play_audio_fn": play_fn,
                "cleanup_audio_fn": cleanup_fn,
                "done_delay_seconds": done_delay,
            },
            daemon=True,
        )
        thread.start()
        return thread, publish_fn, play_fn, cleanup_fn

    def test_publishes_speaking_before_playing_audio(self):
        """task dict 受信時、play_audio_file より前に speaking を publish する。"""
        call_order: list[str] = []
        q: queue.Queue = queue.Queue()

        def publish_fn(character, step, text):
            call_order.append(f"publish:{step}")

        def play_fn(url):
            call_order.append(f"play:{url}")

        def cleanup_fn(url):
            call_order.append(f"cleanup:{url}")

        thread, _, _, _ = self._start_worker(
            q, publish_fn=publish_fn, play_fn=play_fn, cleanup_fn=cleanup_fn,
        )

        q.put({"url": "file:///a.wav", "text": "こんにちは", "is_last": False, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        # 順序: publish(speaking) → play → cleanup → publish(done)
        assert call_order[0] == "publish:speaking"
        assert call_order[1] == "play:file:///a.wav"
        assert call_order[2] == "cleanup:file:///a.wav"
        assert call_order[3] == "publish:done"

    def test_speaking_payload_contains_chunk_text_and_character(self):
        """speaking event の text と character が task dict の内容と一致する。"""
        q: queue.Queue = queue.Queue()
        thread, publish_fn, _, _ = self._start_worker(q)

        q.put({
            "url": "file:///a.wav",
            "text": "今日はいい天気ですね",
            "is_last": True,
            "character": "chisame",
        })
        q.put(None)
        thread.join(timeout=2.0)

        # 最初の呼び出しは speaking
        first_call = publish_fn.call_args_list[0]
        assert first_call.args == ("chisame", "speaking", "今日はいい天気ですね")

    def test_publishes_done_after_done_delay_when_speaking_was_published(self):
        """speaking を発行した後に None 受信 → done_delay 秒待って done publish。"""
        q: queue.Queue = queue.Queue()
        publish_times: list[float] = []
        publish_steps: list[str] = []

        def publish_fn(character, step, text):
            publish_times.append(time.monotonic())
            publish_steps.append(step)

        thread, _, _, _ = self._start_worker(
            q, publish_fn=publish_fn, done_delay=0.2,
        )

        q.put({"url": "file:///a.wav", "text": "x", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        assert publish_steps == ["speaking", "done"]
        # done は speaking の ≥0.2 秒後
        delta = publish_times[1] - publish_times[0]
        assert delta >= 0.18, f"done_delay が機能していない (delta={delta:.3f}s)"

    def test_skips_done_delay_when_no_speaking_published(self):
        """speaking 未発行で None 受信 → 待たずに break、done も publish しない。"""
        q: queue.Queue = queue.Queue()
        thread, publish_fn, play_fn, cleanup_fn = self._start_worker(
            q, done_delay=5.0,  # 長い delay でも待たないことを確認
        )

        start = time.monotonic()
        q.put(None)
        thread.join(timeout=1.0)
        elapsed = time.monotonic() - start

        assert not thread.is_alive(), "worker がタイムアウトせず終了するはず"
        assert elapsed < 1.0, f"待機が発生している (elapsed={elapsed:.3f}s)"
        publish_fn.assert_not_called()
        play_fn.assert_not_called()
        cleanup_fn.assert_not_called()

    def test_multiple_chunks_publish_speaking_per_task(self):
        """複数チャンクの場合、各 task 受信時に speaking を publish する。"""
        q: queue.Queue = queue.Queue()
        thread, publish_fn, play_fn, _ = self._start_worker(q)

        q.put({"url": "file:///a.wav", "text": "A", "is_last": False, "character": "mimi"})
        q.put({"url": "file:///b.wav", "text": "B", "is_last": False, "character": "mimi"})
        q.put({"url": "file:///c.wav", "text": "C", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        # speaking 3 回 + done 1 回
        assert publish_fn.call_count == 4
        steps = [c.args[1] for c in publish_fn.call_args_list]
        texts = [c.args[2] for c in publish_fn.call_args_list]
        assert steps == ["speaking", "speaking", "speaking", "done"]
        assert texts == ["A", "B", "C", ""]
        # 3 回再生された
        assert play_fn.call_count == 3

    def test_done_uses_last_character_slug(self):
        """done publish には最後の task の character が使われる。"""
        q: queue.Queue = queue.Queue()
        thread, publish_fn, _, _ = self._start_worker(q)

        q.put({"url": "file:///a.wav", "text": "A", "is_last": False, "character": "mimi"})
        q.put({"url": "file:///b.wav", "text": "B", "is_last": True, "character": "sakura"})
        q.put(None)
        thread.join(timeout=2.0)

        # done publish の character は最後の task の "sakura"
        done_call = publish_fn.call_args_list[-1]
        assert done_call.args == ("sakura", "done", "")

    def test_resilient_publish_fn_allows_worker_to_continue(self):
        """publish_fn 内で例外処理する実装 (= _publish_bubble_safe) では
        worker が次の task を処理し続けることを検証する。

        注: _run_playback_worker は publish_fn の例外を握りつぶさない設計。
        run_loop.py の _publish_bubble_safe ラッパーが try/except を担うため、
        production でもこのテストのように worker は継続する。
        """
        q: queue.Queue = queue.Queue()
        publish_calls: list[tuple] = []

        def safe_publish(character, step, text):
            # production と同じく内部で try/except
            try:
                publish_calls.append((character, step, text))
                if step == "speaking" and text == "FAIL":
                    raise RuntimeError("Redis 接続失敗")
            except RuntimeError:
                pass  # warning log のみを想定

        play_fn = MagicMock()
        thread, _, _, _ = self._start_worker(
            q, publish_fn=safe_publish, play_fn=play_fn,
        )

        q.put({"url": "file:///fail.wav", "text": "FAIL", "is_last": False, "character": "mimi"})
        q.put({"url": "file:///ok.wav", "text": "OK", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        # 両方の task が処理され、done も publish される
        assert play_fn.call_count == 2
        steps = [c[1] for c in publish_calls]
        assert steps == ["speaking", "speaking", "done"]


class TestInitListenerBgContinuous:
    """Block 0: _init_listener の bg-continuous 分岐テスト。

    BackgroundContinuousListener.__init__ は sounddevice を遅延 import するため、
    モックなしで初期化できる (録音スレッドは start() で起動)。
    """

    def test_returns_background_continuous_listener(self):
        from lab_lounge.run_loop import _init_listener
        from lab_lounge.wake_word import BackgroundContinuousListener

        listener, effective = _init_listener(
            backend="bg-continuous",
            audio_device=None,
            tmp_dir=None,
        )
        try:
            assert isinstance(listener, BackgroundContinuousListener)
            assert effective == "bg-continuous"
        finally:
            listener.cleanup()

    def test_passes_stt_provider(self):
        """stt_provider 引数が listener に伝わる。"""
        from lab_lounge.run_loop import _init_listener

        listener, _ = _init_listener(
            backend="bg-continuous",
            audio_device=None,
            tmp_dir=None,
            stt_provider="faster-whisper",
        )
        try:
            assert listener._stt_provider == "faster-whisper"
        finally:
            listener.cleanup()


class TestRunLoopBgContinuousWiring:
    """Phase 0.5-A フェーズ 6: run_loop の bg-continuous モードでの

    Dispatcher / Listener wiring 検証。

    run_loop() を ``max_turns=0`` で起動して while ループに入らず即座 finally に
    到達させる。Listener.start と Dispatcher の生成 kwargs を spy して、
    フェーズ 6 で接続した callback が正しく渡されることを検証する。
    sounddevice 等の重い依存はモック化する (実際のスレッド起動は省略)。
    """

    def _setup_spies(self, monkeypatch):
        """spy オブジェクト + monkeypatch のセットアップを共通化。

        Returns:
            (listener_start_kwargs, dispatcher_kwargs)
        """
        listener_start_kwargs: dict = {}
        dispatcher_kwargs: dict = {}

        # listener.start を spy (実際のスレッド起動はスキップ)
        from lab_lounge.wake_word import BackgroundContinuousListener

        def spy_start(self, **kwargs):
            listener_start_kwargs.update(kwargs)
            self._on_wake_detected = kwargs["on_wake_detected"]
            self._on_segment_added = kwargs.get("on_segment_added")

        monkeypatch.setattr(BackgroundContinuousListener, "start", spy_start)

        # Dispatcher を spy (kwargs を保存して、wait_for_next_event は即 None)
        from lab_lounge import dispatcher as dispatcher_mod

        OriginalDispatcher = dispatcher_mod.Dispatcher

        class SpyDispatcher(OriginalDispatcher):
            def __init__(self, **kwargs):
                dispatcher_kwargs.update(kwargs)
                super().__init__(**kwargs)

            def wait_for_next_event(self, timeout):
                return None  # 即座 timeout (max_turns=0 でループに入らないので呼ばれない)

        # run_loop は from .dispatcher import Dispatcher で関数内 import するため、
        # dispatcher_mod の属性を差し替える
        monkeypatch.setattr(dispatcher_mod, "Dispatcher", SpyDispatcher)

        # publish を no-op に (bus が Redis に繋がない)
        from lab_lounge import run_loop as run_loop_mod
        monkeypatch.setattr(run_loop_mod, "publish", lambda ev: None)

        return listener_start_kwargs, dispatcher_kwargs

    def test_listener_start_receives_on_segment_added(self, monkeypatch):
        """listener.start kwargs に on_segment_added が渡される。"""
        from lab_lounge.run_loop import run_loop

        listener_start_kwargs, _ = self._setup_spies(monkeypatch)

        # max_turns=0 で while ループに入らず即座 finally
        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        assert "on_segment_added" in listener_start_kwargs
        assert listener_start_kwargs["on_segment_added"] is not None
        assert callable(listener_start_kwargs["on_segment_added"])
        # on_wake_detected も合わせて確認 (Block 0 既存)
        assert "on_wake_detected" in listener_start_kwargs
        assert callable(listener_start_kwargs["on_wake_detected"])

    def test_dispatcher_receives_three_callbacks(self, monkeypatch):
        """Dispatcher 生成時に on_queue_update / on_handraise_update / on_bubble_update が渡される。"""
        from lab_lounge.run_loop import run_loop

        _, dispatcher_kwargs = self._setup_spies(monkeypatch)

        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        assert "on_queue_update" in dispatcher_kwargs
        assert "on_handraise_update" in dispatcher_kwargs
        assert "on_bubble_update" in dispatcher_kwargs
        assert callable(dispatcher_kwargs["on_queue_update"])
        assert callable(dispatcher_kwargs["on_handraise_update"])
        assert callable(dispatcher_kwargs["on_bubble_update"])

    def test_dispatcher_receives_phase7_callbacks(self, monkeypatch):
        """Phase 0.5-A フェーズ 7: bg_runner / on_handraise_started /
        on_handraise_phrase_pending_release / on_handraise_approved も渡される。
        """
        from lab_lounge.run_loop import run_loop

        _, dispatcher_kwargs = self._setup_spies(monkeypatch)

        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        assert "bg_runner" in dispatcher_kwargs
        assert "on_handraise_started" in dispatcher_kwargs
        assert "on_handraise_phrase_pending_release" in dispatcher_kwargs
        assert "on_handraise_approved" in dispatcher_kwargs
        assert callable(dispatcher_kwargs["bg_runner"])
        assert callable(dispatcher_kwargs["on_handraise_started"])
        assert callable(dispatcher_kwargs["on_handraise_phrase_pending_release"])
        assert callable(dispatcher_kwargs["on_handraise_approved"])


# ─── Phase 0.5-A フェーズ 7 (BG LLM + handraise 再生統合) ──────────


class TestSpawnHandraisePhrasePlayback:
    """Phase 0.5-A フェーズ 7: _spawn_handraise_phrase_playback の単体テスト。"""

    def test_returns_none_when_phrase_path_is_none(self):
        from lab_lounge.run_loop import _spawn_handraise_phrase_playback
        result = _spawn_handraise_phrase_playback("mimi", None)
        assert result is None

    def test_calls_play_audio_fn_with_path(self):
        from pathlib import Path
        from lab_lounge.run_loop import _spawn_handraise_phrase_playback
        called: list[str] = []

        def fake_play(path):
            called.append(path)
            return True

        thread = _spawn_handraise_phrase_playback(
            "mimi",
            Path("/tmp/x.wav"),
            padding_sec=0.0,
            play_audio_fn=fake_play,
        )
        thread.join(timeout=2.0)
        assert called == [str(Path("/tmp/x.wav"))]

    def test_padding_sec_delays_playback(self):
        """padding_sec > 0 で再生開始が遅延する (RESPONDING → IDLE 直後の連続再生回避)。"""
        from pathlib import Path
        from lab_lounge.run_loop import _spawn_handraise_phrase_playback
        called_at: list[float] = []

        def fake_play(path):
            called_at.append(time.monotonic())

        start = time.monotonic()
        thread = _spawn_handraise_phrase_playback(
            "mimi",
            Path("/tmp/x.wav"),
            padding_sec=0.2,
            play_audio_fn=fake_play,
        )
        thread.join(timeout=2.0)
        assert len(called_at) == 1
        delta = called_at[0] - start
        assert delta >= 0.18, f"padding が機能していない (delta={delta:.3f}s)"

    def test_play_exception_is_swallowed(self):
        """play_audio_fn 例外は warning ログのみで daemon thread が落ちない。"""
        from pathlib import Path
        from lab_lounge.run_loop import _spawn_handraise_phrase_playback

        def failing_play(path):
            raise RuntimeError("boom")

        thread = _spawn_handraise_phrase_playback(
            "mimi",
            Path("/tmp/x.wav"),
            play_audio_fn=failing_play,
        )
        thread.join(timeout=2.0)
        assert not thread.is_alive()


class TestCreateHandraiseRunnerAndCallbacks:
    """Phase 0.5-A フェーズ 7: factory が 4 つの callable を返す + 挙動検証。"""

    def _factory(self, **overrides):
        from lab_lounge.run_loop import _create_handraise_runner_and_callbacks
        defaults = dict(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
        )
        defaults.update(overrides)
        return _create_handraise_runner_and_callbacks(**defaults)

    def test_returns_four_callables(self):
        bg_runner, on_started, on_release, on_approved = self._factory()
        assert callable(bg_runner)
        assert callable(on_started)
        assert callable(on_release)
        assert callable(on_approved)

    def test_on_handraise_started_idle_calls_spawn(self, monkeypatch):
        """se_pending=False で _spawn_handraise_phrase_playback を呼ぶ (padding=0)。"""
        from pathlib import Path
        spawn_calls: list[tuple] = []

        def fake_spawn(slug, phrase_path, padding_sec=0.0, **kw):
            spawn_calls.append((slug, phrase_path, padding_sec))
            return None

        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_phrase_playback", fake_spawn,
        )
        _, on_started, _, _ = self._factory()
        on_started("mimi", Path("/tmp/x.wav"), False)
        assert spawn_calls == [("mimi", Path("/tmp/x.wav"), 0.0)]

    def test_on_handraise_started_responding_does_not_spawn(self, monkeypatch):
        """se_pending=True ではスキップ (release callback 経由で後ほど再生)。"""
        from pathlib import Path
        spawn_calls: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_phrase_playback",
            lambda *a, **kw: spawn_calls.append(a),
        )
        _, on_started, _, _ = self._factory()
        on_started("mimi", Path("/tmp/x.wav"), True)
        assert spawn_calls == []

    def test_on_handraise_phrase_pending_release_uses_padding_env(self, monkeypatch):
        """release callback は L2_HANDRAISE_RESPONDING_PADDING_SEC env を参照。"""
        from pathlib import Path
        monkeypatch.setenv("L2_HANDRAISE_RESPONDING_PADDING_SEC", "0.7")
        spawn_calls: list[tuple] = []

        def fake_spawn(slug, phrase_path, padding_sec=0.0, **kw):
            spawn_calls.append((slug, phrase_path, padding_sec))
            return None

        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_phrase_playback", fake_spawn,
        )
        _, _, on_release, _ = self._factory()
        on_release("sakura", Path("/tmp/x.wav"))
        assert spawn_calls == [("sakura", Path("/tmp/x.wav"), 0.7)]

    def test_on_handraise_approved_with_chunks_publishes_and_plays(self, monkeypatch):
        """bg_result 有 + chunks 有で answering bubble 発行 + chunks 再生。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        published: list = []
        spawn_calls: list[tuple] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish",
            lambda ev: published.append(ev),
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda slug, chunks, trace_id, **kw: spawn_calls.append(
                (slug, list(chunks), trace_id)
            ),
        )
        _, _, _, on_approved = self._factory()

        # bg_result.result.events 内の llm.final から text を取る
        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": "わたくしの見解は…"}},
            {"type": "tts.done"},
        ]
        bg = HandraiseBgResult(
            chunks=[{"url": "u1", "text": "x", "is_last": True, "character": "mimi"}],
            result=fake_result,
            trace_id="bg-abc",
        )
        on_approved("mimi", bg, "snap", "trace-orig")

        # bubble.update("answering") が 1 回発行
        bubble_calls = [
            e for e in published
            if e.get("type") == "bubble.update"
            and e.get("payload", {}).get("step") == "answering"
        ]
        assert len(bubble_calls) == 1
        assert bubble_calls[0]["payload"]["character"] == "mimi"
        assert bubble_calls[0]["payload"]["text"] == "わたくしの見解は…"
        # Phase 0.5-A 8-10 (A2 確定): 承認後応答は category="speech"
        # WHY: 挙手バブルは消費され、応答は通常応答エリアで表示する設計
        assert bubble_calls[0]["payload"]["category"] == "speech"
        # chunks 再生も呼ばれる
        assert len(spawn_calls) == 1
        assert spawn_calls[0][0] == "mimi"

    def test_on_handraise_approved_with_none_result_uses_fallback(self, monkeypatch):
        """bg_result=None でフォールバックスレッドが起動する。"""
        called: list[tuple] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._approved_synthesize_fallback",
            lambda slug, snap, trace, **kw: called.append((slug, snap, trace)),
        )
        _, _, _, on_approved = self._factory()
        on_approved("mimi", None, "snap", "trace-x")
        # daemon thread 内 fallback なのでポーリング待機
        for _ in range(40):
            if called:
                break
            time.sleep(0.05)
        assert len(called) == 1
        assert called[0] == ("mimi", "snap", "trace-x")

    def test_on_handraise_approved_with_empty_chunks_uses_fallback(self, monkeypatch):
        """bg_result 有だが chunks 空でも fallback に流れる。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        called: list[str] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._approved_synthesize_fallback",
            lambda slug, snap, trace, **kw: called.append(slug),
        )
        bg = HandraiseBgResult(chunks=[], result=None, trace_id="x")
        _, _, _, on_approved = self._factory()
        on_approved("mimi", bg, "snap", "trace-x")
        for _ in range(40):
            if called:
                break
            time.sleep(0.05)
        assert called == ["mimi"]


class TestApprovedSynthesizeFallback:
    """Phase 0.5-A フェーズ 7: _approved_synthesize_fallback の単体テスト。"""

    def test_calls_run_pipeline_with_speaker_hint(self, monkeypatch):
        from lab_lounge.run_loop import _approved_synthesize_fallback
        called: list[tuple] = []

        def fake_run_pipeline(text, **kw):
            called.append((text, kw))
            return MagicMock()

        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline", fake_run_pipeline,
        )
        _approved_synthesize_fallback(
            "mimi", "テスト発話", "trace-x",
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context="配信文脈",
        )
        assert len(called) == 1
        text, kw = called[0]
        assert text == "テスト発話"
        assert kw["speaker_hint"] == "mimi"
        assert kw["stream_id"] == "s1"
        assert kw["session_id"] == "ses1"
        assert kw["stream_context"] == "配信文脈"
        assert kw["suppress_bubble_answering"] is False  # graph 側で answering 発行
        # Phase 0.5-A フェーズ 8 修正: chunks 蓄積用 callback が渡される (再生のために必須)
        assert "on_tts_chunk_ready" in kw
        assert callable(kw["on_tts_chunk_ready"])

    def test_run_pipeline_exception_is_swallowed(self, monkeypatch):
        """run_pipeline 例外は warning ログのみで例外は伝播しない。"""
        from lab_lounge.run_loop import _approved_synthesize_fallback

        def failing_pipeline(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("lab_lounge.run_loop.run_pipeline", failing_pipeline)
        spawn_calls: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: spawn_calls.append(a),
        )
        # 例外なく完了
        _approved_synthesize_fallback(
            "mimi", "snap", "trace",
            session_stream_id="s1", session_id_root="ses1", stream_context=None,
        )
        # 例外時は playback も起動されない
        assert spawn_calls == []

    def test_collects_chunks_and_spawns_playback(self, monkeypatch):
        """Phase 0.5-A フェーズ 8 修正: TTS chunks が蓄積され、playback worker に渡される。

        run_pipeline 内で on_tts_chunk_ready callback が呼ばれると chunks list に
        蓄積され、run_pipeline 完了後に _spawn_handraise_response_playback で再生される
        ことを検証。これが無いと bg_result=None 時の fallback パスで音声が再生されない。
        """
        from lab_lounge.run_loop import _approved_synthesize_fallback

        # run_pipeline mock: on_tts_chunk_ready を 2 回呼んで chunks を蓄積させる
        def fake_run_pipeline(text, **kw):
            on_chunk = kw["on_tts_chunk_ready"]
            on_chunk(
                "file:///c1.wav", "こんにちは", False, "mimi",
            )
            on_chunk(
                "file:///c2.wav", "ですわ", True, "mimi",
            )
            return MagicMock()

        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline", fake_run_pipeline,
        )

        spawn_calls: list[tuple] = []

        def fake_spawn(slug, chunks, trace_id, **kw):
            spawn_calls.append((slug, list(chunks), trace_id, kw))

        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback", fake_spawn,
        )

        _approved_synthesize_fallback(
            "mimi", "snap", "trace-x",
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
        )

        # playback worker が起動された
        assert len(spawn_calls) == 1
        slug, chunks, trace_id, kw = spawn_calls[0]
        assert slug == "mimi"
        assert trace_id == "trace-x"
        # chunks が 2 件蓄積されている
        assert len(chunks) == 2
        assert chunks[0]["url"] == "file:///c1.wav"
        assert chunks[0]["text"] == "こんにちは"
        assert chunks[0]["is_last"] is False
        assert chunks[1]["url"] == "file:///c2.wav"
        assert chunks[1]["is_last"] is True
        # session_stream_id / session_id_root が渡される
        assert kw["session_stream_id"] == "s1"
        assert kw["session_id_root"] == "ses1"

    def test_empty_chunks_skips_playback(self, monkeypatch):
        """run_pipeline が on_tts_chunk_ready を呼ばない (= ダミー TTS or TTS 失敗) 場合、
        playback worker を起動せず warning ログだけ出す。
        """
        from lab_lounge.run_loop import _approved_synthesize_fallback

        # on_tts_chunk_ready を呼ばない (chunks 空)
        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline",
            lambda text, **kw: MagicMock(),
        )

        spawn_calls: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: spawn_calls.append(a),
        )

        _approved_synthesize_fallback(
            "mimi", "snap", "trace-x",
            session_stream_id="s1", session_id_root="ses1", stream_context=None,
        )

        # playback worker は起動されない
        assert spawn_calls == []

    def test_chunk_with_pose_passed_through(self, monkeypatch):
        """on_tts_chunk_ready の pose 引数が chunk dict に正しく格納される。"""
        from lab_lounge.run_loop import _approved_synthesize_fallback

        def fake_run_pipeline(text, **kw):
            kw["on_tts_chunk_ready"](
                "file:///c.wav", "x", True, "mimi", pose="happy",
            )
            return MagicMock()

        monkeypatch.setattr("lab_lounge.run_loop.run_pipeline", fake_run_pipeline)
        captured: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda slug, chunks, trace_id, **kw: captured.append(list(chunks)),
        )

        _approved_synthesize_fallback(
            "mimi", "snap", "trace-x",
            session_stream_id="s1", session_id_root="ses1", stream_context=None,
        )

        assert len(captured) == 1
        assert captured[0][0]["pose"] == "happy"

    def test_uses_new_uuid_when_trace_id_empty(self, monkeypatch):
        from lab_lounge.run_loop import _approved_synthesize_fallback
        called: list[dict] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline",
            lambda text, **kw: called.append(kw) or MagicMock(),
        )
        _approved_synthesize_fallback(
            "mimi", "snap", "",  # trace_id 空
            session_stream_id="s1", session_id_root="ses1", stream_context=None,
        )
        assert len(called) == 1
        # 新 UUID が割り当てられている (空文字ではない)
        assert called[0]["trace_id"]
        assert len(called[0]["trace_id"]) == 36  # UUID4 形式
