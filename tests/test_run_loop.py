"""
test_run_loop.py — run_loop.py の _run_playback_worker の単体テスト

Sprint Axis D Block 1: OBS セリフテロップ表示
worker が speaking / done の bubble.update を正しいタイミングで publish することを検証。
"""

import queue
import threading
import time
from unittest.mock import MagicMock

from lab_lounge.character_status import CharacterStatus, CharacterStatusManager
from lab_lounge.run_loop import (
    _approved_synthesize_fallback,
    _create_handraise_runner_and_callbacks,
    _run_playback_worker,
    _spawn_handraise_response_playback,
)


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


class TestRunPlaybackWorkerOnLastChunkPlayed:
    """_run_playback_worker の on_last_chunk_played callback の挙動 (Phase 0.5-A 8-11)。

    最終 chunk 物理再生完了直後 (= None sentinel 受信時、time.sleep(done_delay) の前) に
    callback を発火することで、handraise wav の遅延再生を 5 秒早めることが目的。
    本クラスはタイミング順序と冪等性を検証する。
    """

    def _start_worker(
        self,
        q,
        *,
        on_last_chunk_played=None,
        publish_fn=None,
        play_fn=None,
        cleanup_fn=None,
        done_delay=0.05,
    ):
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
                "on_last_chunk_played": on_last_chunk_played,
            },
            daemon=True,
        )
        thread.start()
        return thread, publish_fn, play_fn, cleanup_fn

    def test_callback_fires_before_done_delay_sleep(self):
        """sentinel 受信時、callback 発火 → time.sleep(done_delay) → done publish の順。

        handraise wav の遅延再生を 5 秒早めるためには、callback が done_delay 待機の
        "前" に発火することが必須 (= 5 秒の sleep 中に handraise wav が並列再生される)。
        """
        q: queue.Queue = queue.Queue()
        events: list[tuple[str, float]] = []

        def callback():
            events.append(("callback", time.monotonic()))

        def publish_fn(character, step, text):
            events.append((f"publish:{step}", time.monotonic()))

        thread, _, _, _ = self._start_worker(
            q, on_last_chunk_played=callback, publish_fn=publish_fn,
            done_delay=0.2,
        )

        q.put({"url": "file:///a.wav", "text": "x", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        # 順序: speaking → callback → done (callback と done の間に done_delay 0.2s)
        names = [e[0] for e in events]
        assert names == ["publish:speaking", "callback", "publish:done"]
        # callback と done の間隔が done_delay 以上
        callback_t = events[1][1]
        done_t = events[2][1]
        assert done_t - callback_t >= 0.18  # done_delay=0.2 で margin 0.02

    def test_callback_not_fired_when_no_speaking_published(self):
        """speaking 未発行 (空 task のみ) なら callback も done publish も発火しない。

        既存の break 経路 (空 sentinel で即終了) を維持する (回帰)。
        """
        q: queue.Queue = queue.Queue()
        callback_calls: list[None] = []

        def callback():
            callback_calls.append(None)

        thread, publish_fn, _, _ = self._start_worker(
            q, on_last_chunk_played=callback,
        )

        # speaking なしで sentinel のみ
        q.put(None)
        thread.join(timeout=2.0)

        assert callback_calls == []  # callback も発火しない
        assert publish_fn.call_count == 0  # done bubble も発行されない

    def test_callback_exception_does_not_propagate(self):
        """callback 内例外で worker が止まらない。done bubble も予定通り発行される。"""
        q: queue.Queue = queue.Queue()

        def failing_callback():
            raise RuntimeError("flush failed")

        publish_fn = MagicMock()
        thread, _, _, _ = self._start_worker(
            q, on_last_chunk_played=failing_callback, publish_fn=publish_fn,
        )

        q.put({"url": "file:///a.wav", "text": "x", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        # done publish は callback 例外を吸収して通常通り発行される
        steps = [c.args[1] for c in publish_fn.call_args_list]
        assert steps == ["speaking", "done"]

    def test_default_none_preserves_existing_behavior(self):
        """on_last_chunk_played=None (default) で既存挙動が壊れない (回帰)。"""
        q: queue.Queue = queue.Queue()
        publish_fn = MagicMock()

        thread, _, _, _ = self._start_worker(
            q, on_last_chunk_played=None, publish_fn=publish_fn,
        )

        q.put({"url": "file:///a.wav", "text": "x", "is_last": True, "character": "mimi"})
        q.put(None)
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        # speaking → done のみ (callback 経路は無し)
        steps = [c.args[1] for c in publish_fn.call_args_list]
        assert steps == ["speaking", "done"]


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

    def test_dispatcher_receives_status_manager(self, monkeypatch):
        """Phase 0.5-B-α: Dispatcher 生成時に status_manager (CharacterStatusManager)
        が kwargs として渡される。"""
        from lab_lounge.run_loop import run_loop

        _, dispatcher_kwargs = self._setup_spies(monkeypatch)

        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        assert "status_manager" in dispatcher_kwargs
        manager = dispatcher_kwargs["status_manager"]
        assert manager is not None
        assert isinstance(manager, CharacterStatusManager)


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

    def test_on_handraise_approved_with_llm_result_runs_tts_only_and_plays(self, monkeypatch):
        """W'-1: bg_result.result が ready なら TTS-only graph で TTS 実行 + chunks 再生。

        新設計 (W'-1): bg_runner は LLM のみ先行 → bg_result.chunks は常に空。
        承認時 on_handraise_approved は run_pipeline_tts_only を呼んで chunks を
        生成し、_spawn_handraise_response_playback で再生する。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        published: list = []
        spawn_calls: list[tuple] = []
        tts_only_calls: list[tuple] = []

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

        def fake_tts_only(llm_result, *, on_tts_chunk_ready=None, on_pose_ready=None):
            """run_pipeline_tts_only の差し替え。on_chunk で chunks を流す。"""
            tts_only_calls.append((llm_result, on_tts_chunk_ready, on_pose_ready))
            if on_tts_chunk_ready is not None:
                on_tts_chunk_ready("u1", "わたくしの見解は", True, "mimi", None)
            return llm_result

        # WHY: import 経路を on_handraise_approved 内の局所 import に合わせる必要が
        # ある。`from .pipeline import run_pipeline_tts_only` 後の局所名を差し替えるため、
        # lab_lounge.pipeline.run_pipeline_tts_only と lab_lounge.run_loop の両方を
        # patch する必要はなく、lab_lounge.pipeline をパッチすれば局所 import で取得される。
        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only", fake_tts_only,
        )

        _, _, _, on_approved = self._factory()

        # bg_result.result.events に llm.final がある (= 案 W'-1 の実態に合わせて
        # Pydantic JSON 文字列形式)
        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {
                "type": "llm.final",
                "payload": {
                    "text": (
                        '{"response": "わたくしの見解は…",'
                        ' "emotion": {"happy": 50}, "speed": 100, "pose": "neutral"}'
                    ),
                },
            },
        ]
        bg = HandraiseBgResult(
            chunks=[],  # LLM-only なので空
            result=fake_result,
            trace_id="bg-abc",
        )
        on_approved("mimi", bg, "snap", "trace-orig")

        # daemon thread 内で _tts_and_play が走るので待機
        for _ in range(40):
            if spawn_calls:
                break
            time.sleep(0.05)

        # bubble.update("answering") が 1 回発行
        bubble_calls = [
            e for e in published
            if e.get("type") == "bubble.update"
            and e.get("payload", {}).get("step") == "answering"
        ]
        assert len(bubble_calls) == 1
        assert bubble_calls[0]["payload"]["character"] == "mimi"
        # バグ 4 修正: bubble.text は JSON の response 部分のみ (= 生 JSON 文字列ではない)
        assert bubble_calls[0]["payload"]["text"] == "わたくしの見解は…"
        # Phase 0.5-A 8-10 (A2 確定): 承認後応答は category="speech"
        assert bubble_calls[0]["payload"]["category"] == "speech"
        # run_pipeline_tts_only が呼ばれた (= TTS-only graph 経由)
        assert len(tts_only_calls) == 1
        # _spawn_handraise_response_playback が呼ばれた
        assert len(spawn_calls) == 1
        assert spawn_calls[0][0] == "mimi"
        assert spawn_calls[0][2] == "bg-abc"  # bg_trace_id 引継ぎ

    def test_on_handraise_approved_bubble_text_is_response_only_not_raw_json(
        self, monkeypatch,
    ):
        """W'-1 バグ 4 修正: bubble.text は LLM 応答 JSON の response フィールドのみ。

        WHY: 実走 logs/runs/run_loop_20260508_184930.log で
        bubble.update [character=sakura step=answering] の payload.text に
        生 JSON 全体 ({"response":"...","emotion":{...},...}) が入っていた。
        通常応答パスでは _tts_node 内で _parse_voicepeak_json で処理されるが、
        W'-1 で run_loop が bubble を発行する経路では明示的に parse する必要が
        ある (= バグ 4 の核心テスト)。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        published: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish",
            lambda ev: published.append(ev),
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only",
            lambda llm_result, **kw: llm_result,
        )

        _, _, _, on_approved = self._factory()

        # 実走で観察された JSON 文字列を再現 (= sakura の応答)
        raw_json = (
            '{"response":"ん〜……AI倫理って、深く考えれば考えるほど、答えが'
            '一つじゃないって気づきますよねぇ。",'
            '"emotion":{"happy":30,"sad":10,"angry":0,"whisper":40,"cool":20},'
            '"speed":90,"pose":"special_whisper"}'
        )
        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": raw_json}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="bg-x")
        on_approved("sakura", bg, "snap", "trace-x")

        # daemon thread が走るので bubble 発行を待つ
        for _ in range(40):
            if any(
                e.get("payload", {}).get("step") == "answering"
                for e in published
                if e.get("type") == "bubble.update"
            ):
                break
            time.sleep(0.05)

        bubble_calls = [
            e for e in published
            if e.get("type") == "bubble.update"
            and e.get("payload", {}).get("step") == "answering"
        ]
        assert len(bubble_calls) == 1
        bubble_text = bubble_calls[0]["payload"]["text"]
        # 期待: response 部分のみ (= response key の値)
        assert bubble_text == (
            "ん〜……AI倫理って、深く考えれば考えるほど、答えが一つじゃないって気づきますよねぇ。"
        )
        # 確認: 生 JSON のキー文字列 (= "emotion" / "pose" 等) が含まれていない
        assert '"emotion"' not in bubble_text
        assert '"pose"' not in bubble_text
        assert '"response"' not in bubble_text

    def test_on_handraise_approved_bubble_text_fallback_when_not_json(
        self, monkeypatch,
    ):
        """parse 失敗時 (= 生テキスト等) は元テキストをそのまま使う (fallback)。

        WHY: _parse_voicepeak_json は JSON でない / response key 無しの場合、
        (元テキスト, None, None, None) を返す → bubble.text には元テキストが
        そのまま入る (= 過去動作との後方互換)。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        published: list = []
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish",
            lambda ev: published.append(ev),
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only",
            lambda llm_result, **kw: llm_result,
        )

        _, _, _, on_approved = self._factory()

        # JSON ではない生テキスト (= ダミー LLM モード等)
        plain_text = "ダミー応答: 最近のAI倫理について深く考えています。"
        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": plain_text}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="bg-x")
        on_approved("sakura", bg, "snap", "trace-x")

        for _ in range(40):
            if any(
                e.get("payload", {}).get("step") == "answering"
                for e in published
                if e.get("type") == "bubble.update"
            ):
                break
            time.sleep(0.05)

        bubble_calls = [
            e for e in published
            if e.get("type") == "bubble.update"
            and e.get("payload", {}).get("step") == "answering"
        ]
        assert len(bubble_calls) == 1
        # parse 失敗時は元テキストをそのまま使う
        assert bubble_calls[0]["payload"]["text"] == plain_text

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

    def test_on_handraise_approved_with_empty_result_uses_fallback(self, monkeypatch):
        """W'-1: bg_result はあるが result=None なら fallback (= LLM 失敗時の救済)。"""
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

    def test_on_handraise_approved_with_empty_llm_text_falls_back(self, monkeypatch):
        """W'-1: bg_result.result はあるが llm.final.text が空なら fallback。

        WHY: BG LLM が成功してもテキストが空文字列なケース (= API 異常応答 etc.)
        に対する救済。TTS-only graph に空 text を渡すと _tts_node 内で例外になる
        ので、事前に判定して fallback パスに流す。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        called: list[str] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._approved_synthesize_fallback",
            lambda slug, snap, trace, **kw: called.append(slug),
        )
        # llm.final.text が空文字列
        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": ""}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="x")
        _, _, _, on_approved = self._factory()
        on_approved("mimi", bg, "snap", "trace-x")
        for _ in range(40):
            if called:
                break
            time.sleep(0.05)
        assert called == ["mimi"]

    def test_on_handraise_approved_tts_exception_falls_back(self, monkeypatch):
        """W'-1: TTS-only graph 実行が例外で失敗したら fallback パスに流れる。

        WHY: VOICEPEAK 起動失敗 / Gemini API 一時障害などの稀ケース。daemon thread
        内で例外を catch して fallback パスで救済。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        called: list[str] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._approved_synthesize_fallback",
            lambda slug, snap, trace, **kw: called.append(slug),
        )
        # publish は副作用ありなので noop に
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish", lambda ev: None,
        )

        def raising_tts_only(*args, **kwargs):
            raise RuntimeError("simulated TTS failure")

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only", raising_tts_only,
        )

        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": "test"}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="x")
        _, _, _, on_approved = self._factory()
        on_approved("mimi", bg, "snap", "trace-x")
        for _ in range(40):
            if called:
                break
            time.sleep(0.05)
        assert called == ["mimi"]

    def test_on_handraise_approved_chunks_empty_after_tts_falls_back(self, monkeypatch):
        """W'-1: TTS-only graph 完了したが chunks が 0 件なら fallback パスに流れる。

        WHY: ダミー TTS モード (L2_USE_REAL_TTS=false) または TTS 出力が無いケース
        の救済。on_tts_chunk_ready が呼ばれず chunks=[] のまま完了したら fallback
        で実音声を生成し直す。
        """
        from lab_lounge.dispatcher import HandraiseBgResult
        called: list[str] = []
        monkeypatch.setattr(
            "lab_lounge.run_loop._approved_synthesize_fallback",
            lambda slug, snap, trace, **kw: called.append(slug),
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish", lambda ev: None,
        )

        # chunks を流さずに完了 (= chunks=[] のまま)
        def empty_tts_only(llm_result, *, on_tts_chunk_ready=None, on_pose_ready=None):
            return llm_result

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only", empty_tts_only,
        )

        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": "test"}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="x")
        _, _, _, on_approved = self._factory()
        on_approved("mimi", bg, "snap", "trace-x")
        for _ in range(40):
            if called:
                break
            time.sleep(0.05)
        assert called == ["mimi"]

    def test_bg_runner_uses_run_pipeline_llm_only(self, monkeypatch):
        """W'-1: bg_runner が run_pipeline_llm_only を呼ぶ (run_pipeline は呼ばない)。"""
        import threading
        llm_only_calls: list = []
        run_pipeline_calls: list = []
        completed: list = []

        # run_pipeline_llm_only を spy 化、ダミー PipelineResult を返す
        def fake_llm_only(text, **kwargs):
            llm_only_calls.append((text, kwargs))
            return MagicMock(events=[
                {"type": "utterance.final"},
                {"type": "llm.final", "payload": {"text": "llm 結果"}},
            ])

        def fake_run_pipeline(*args, **kwargs):
            run_pipeline_calls.append((args, kwargs))
            raise RuntimeError("run_pipeline should not be called from bg_runner")

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_llm_only", fake_llm_only,
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline", fake_run_pipeline,
        )

        bg_runner, _, _, _ = self._factory()
        cancel_event = threading.Event()
        thread = bg_runner(
            target_slug="mimi",
            transcript_snapshot="テスト発話",
            cancel_event=cancel_event,
            on_complete=lambda r: completed.append(r),
        )
        thread.join(timeout=5.0)

        # run_pipeline_llm_only が呼ばれ、run_pipeline (旧) は呼ばれていない
        assert len(llm_only_calls) == 1
        assert llm_only_calls[0][0] == "テスト発話"
        assert llm_only_calls[0][1]["speaker_hint"] == "mimi"
        assert run_pipeline_calls == []
        # on_complete が呼ばれた
        assert len(completed) == 1
        # chunks は LLM-only なので常に空
        assert completed[0].chunks == []
        # result は llm_only_calls の戻り値そのまま
        assert completed[0].result is not None

    # ─── Phase 0.5-B-β-1 commit 3: ask_character 対話 TTS の BG LLM 経路配線 ──
    # WHY: factory に on_tts_chunk_ready_ref / on_pose_ready_ref (= mutable list)
    # を渡すと、bg_runner._body が ref[0] 経由で最新ターンの _on_tts_chunk /
    # _on_pose_ready closure を取得して run_pipeline_llm_only に渡す。これが A1
    # (ask_character TTS 投入欠落) の最終配線。

    def test_factory_accepts_on_tts_chunk_ready_ref_kwarg(self):
        """factory が on_tts_chunk_ready_ref キーワード引数を受け付ける (Phase 0.5-B-β-1 commit 3)。

        WHY: ターン毎に再構築される _on_tts_chunk closure を mutable list 経由で
        参照するため、factory のシグネチャ拡張が必要。本テストは API 受付のみ確認。
        """
        cb_ref: list = [None]
        bg_runner, *_ = self._factory(on_tts_chunk_ready_ref=cb_ref)
        assert callable(bg_runner)

    def test_factory_accepts_on_pose_ready_ref_kwarg(self):
        """factory が on_pose_ready_ref キーワード引数を受け付ける (Phase 0.5-B-β-1 commit 3)。"""
        pose_ref: list = [None]
        bg_runner, *_ = self._factory(on_pose_ready_ref=pose_ref)
        assert callable(bg_runner)

    def test_bg_runner_passes_callbacks_to_pipeline_when_ref_set(self, monkeypatch):
        """bg_runner._body が ref[0] の最新値を run_pipeline_llm_only に渡す
        (Phase 0.5-B-β-1 commit 3、A1 修正の最終配線)。

        WHY: ターン中 (= _on_tts_chunk / _on_pose_ready 構築済み) に挙手 BG LLM が
        起動した場合、ref[0] には最新 closure が入っている。bg_runner はこれを
        run_pipeline_llm_only に渡し、ask_character ツール内で対話 TTS が起動
        する経路が完成する。
        """
        import threading
        llm_only_calls: list[dict] = []

        def fake_llm_only(text, **kwargs):
            llm_only_calls.append(dict(kwargs))
            return MagicMock(events=[
                {"type": "utterance.final"},
                {"type": "llm.final", "payload": {"text": "llm 結果"}},
            ])

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_llm_only", fake_llm_only,
        )

        def my_tts_cb(url, chunk_text, is_last, character):
            pass

        def my_pose_cb(slug, pose):
            pass

        # ref[0] に最新 closure をセットした状態で factory 経由 bg_runner 起動
        tts_ref: list = [my_tts_cb]
        pose_ref: list = [my_pose_cb]
        bg_runner, *_ = self._factory(
            on_tts_chunk_ready_ref=tts_ref,
            on_pose_ready_ref=pose_ref,
        )

        cancel_event = threading.Event()
        thread = bg_runner(
            target_slug="mimi",
            transcript_snapshot="hello",
            cancel_event=cancel_event,
            on_complete=lambda r: None,
        )
        thread.join(timeout=5.0)

        # 1 回呼出 + 渡された callback が run_pipeline_llm_only に届いている
        assert len(llm_only_calls) == 1
        assert llm_only_calls[0].get("on_tts_chunk_ready") is my_tts_cb
        assert llm_only_calls[0].get("on_pose_ready") is my_pose_cb

    def test_bg_runner_passes_none_to_pipeline_when_ref_unset(self, monkeypatch):
        """bg_runner._body が ref 未指定 / ref[0]=None の場合、pipeline に None を渡す
        (Phase 0.5-B-β-1 commit 3 後方互換)。

        WHY: ref を渡さない既存呼出元 (= テスト等) では従来通り on_tts_chunk_ready=None
        で run_pipeline_llm_only が呼ばれる。挙手中 IDLE ターン外も ref[0]=None
        になる想定で、安全に no-op で通る。
        """
        import threading
        llm_only_calls: list[dict] = []

        def fake_llm_only(text, **kwargs):
            llm_only_calls.append(dict(kwargs))
            return MagicMock(events=[
                {"type": "utterance.final"},
                {"type": "llm.final", "payload": {"text": "x"}},
            ])

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_llm_only", fake_llm_only,
        )

        # ref 未指定 (= 既存呼出パターン) と ref[0]=None の両ケース
        for ref_kwargs in [
            {},  # ref 未指定
            {"on_tts_chunk_ready_ref": [None], "on_pose_ready_ref": [None]},
        ]:
            llm_only_calls.clear()
            bg_runner, *_ = self._factory(**ref_kwargs)
            cancel_event = threading.Event()
            thread = bg_runner(
                target_slug="mimi",
                transcript_snapshot="x",
                cancel_event=cancel_event,
                on_complete=lambda r: None,
            )
            thread.join(timeout=5.0)
            assert len(llm_only_calls) == 1
            assert llm_only_calls[0].get("on_tts_chunk_ready") is None
            assert llm_only_calls[0].get("on_pose_ready") is None


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
        # Phase 0.5-A バグ 3 修正 (案 A): fallback パスでは ask_character ツールを
        # 無効化して並行 TTS との deadlock を回避する
        assert kw.get("disable_tools") == ["ask_character"]

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


# ─── Phase 0.5-A 案 W'-3: ログ強化境界テスト ──────────────────────


class TestApprovedFlowLoggingProgression:
    """Phase 0.5-A 案 W'-3: 挙手承認パス各経路のログ進行を caplog で検証。

    実走時のシナリオ B/C 再走で「TTS 再生キュー投入」が出ない問題 (= 問題 5) を
    確実に検出できるよう、各ステージにログを仕込んだことの単体保証。grep パターン
    で「どの経路を通ったか」が 1 行で追跡可能になることを保証する。
    """

    def test_fallback_logs_chunks_count(self, monkeypatch, caplog):
        """fallback パスで chunks 蓄積完了ログが出る (W'-3 ログ強化)。"""
        import logging
        from lab_lounge.run_loop import _approved_synthesize_fallback

        # run_pipeline を spy 化、on_chunk callback で 2 chunks 流す
        def fake_run_pipeline(text, **kw):
            cb = kw.get("on_tts_chunk_ready")
            if cb is not None:
                cb("u1", "first chunk", False, "mimi", None)
                cb("u2", "second chunk text", True, "mimi", None)
            return MagicMock()

        monkeypatch.setattr(
            "lab_lounge.run_loop.run_pipeline", fake_run_pipeline,
        )
        # _spawn_handraise_response_playback を no-op に (= playback 起動を抑止)
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: None,
        )

        with caplog.at_level(logging.INFO, logger="lab_lounge.run_loop"):
            _approved_synthesize_fallback(
                "mimi", "テスト発話", "trace-fb",
                session_stream_id="s1", session_id_root="ses1", stream_context=None,
            )

        # 開始ログ + 蓄積完了ログの両方が出ること
        starts = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "fallback 同期再生成 開始" in r.getMessage()
            and "[character=mimi]" in r.getMessage()
        ]
        completions = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "fallback chunks 蓄積完了" in r.getMessage()
            and "[character=mimi]" in r.getMessage()
            and "count=2" in r.getMessage()
        ]
        assert len(starts) == 1
        assert len(completions) == 1

    def test_approved_tts_logs_progression(self, monkeypatch, caplog):
        """承認 TTS 同期実行パスで「開始 → 完了 → playback 起動」順にログが出る。

        WHY: シナリオ B/C 再走で「TTS まで完了したが playback まで到達したか」を
        1 行 grep で追跡可能にするための保証。
        """
        import logging
        import time
        from lab_lounge.dispatcher import HandraiseBgResult
        from lab_lounge.run_loop import _create_handraise_runner_and_callbacks

        # run_pipeline_tts_only spy: chunks を on_chunk で 1 件流す
        def fake_tts_only(llm_result, *, on_tts_chunk_ready=None, on_pose_ready=None):
            if on_tts_chunk_ready is not None:
                on_tts_chunk_ready("u1", "test", True, "mimi", None)
            return llm_result

        monkeypatch.setattr(
            "lab_lounge.pipeline.run_pipeline_tts_only", fake_tts_only,
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop.publish", lambda ev: None,
        )

        _, _, _, on_approved = _create_handraise_runner_and_callbacks(
            session_stream_id="s1", session_id_root="ses1", stream_context=None,
        )

        fake_result = MagicMock()
        fake_result.events = [
            {"type": "utterance.final"},
            {"type": "llm.final", "payload": {"text": "わたくしの見解は…"}},
        ]
        bg = HandraiseBgResult(chunks=[], result=fake_result, trace_id="bg-tr")

        with caplog.at_level(logging.INFO, logger="lab_lounge.run_loop"):
            on_approved("mimi", bg, "snap", "trace-orig")
            # daemon thread 内で _tts_and_play が走る
            for _ in range(40):
                if any(
                    "挙手承認 playback worker 起動" in r.getMessage()
                    for r in caplog.records
                ):
                    break
                time.sleep(0.05)

        # 期待: 開始 → 完了 → playback 起動 の 3 ログが順序で出ている
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        starts = [m for m in msgs if "挙手承認 TTS 同期実行 開始" in m and "[character=mimi]" in m]
        completes = [m for m in msgs if "挙手承認 TTS 同期実行 完了" in m and "[character=mimi]" in m]
        playbacks = [m for m in msgs if "挙手承認 playback worker 起動" in m and "[character=mimi]" in m]
        assert len(starts) == 1
        assert len(completes) == 1
        assert len(playbacks) == 1
        # 順序確認: starts → completes → playbacks の index 関係
        start_idx = msgs.index(starts[0])
        complete_idx = msgs.index(completes[0])
        playback_idx = msgs.index(playbacks[0])
        assert start_idx < complete_idx < playback_idx


# ─── TestStatusManagerWiring (Phase 0.5-B-α) ─────────────────────


class TestStatusManagerWiring:
    """Phase 0.5-B-α: run_loop 経由の CharacterStatusManager 配線検証。

    _spawn_handraise_response_playback / _approved_synthesize_fallback /
    _create_handraise_runner_and_callbacks に optional な status_manager 引数が
    伝播し、Talking + Ready 反映 / metadata 流れが正しいことを検証する。
    """

    def test_spawn_handraise_response_playback_sets_talking_then_ready(self):
        """_spawn_handraise_response_playback で Talking + metadata 反映、worker 完了で Ready。

        chunks=[] なら worker は即 sentinel を受信して終了するため、外部依存
        (audio_io 等) の mock 不要で完結する。
        """
        callback = MagicMock()
        manager = CharacterStatusManager(on_status_changed=callback)
        metadata = {"pose": "smile", "text": "こんにちは"}

        t = _spawn_handraise_response_playback(
            slug="mimi",
            chunks=[],
            trace_id="t1",
            session_stream_id="s1",
            session_id_root="ss1",
            status_manager=manager,
            talking_metadata=metadata,
        )
        t.join(timeout=2.0)

        # callback は 2 回発火: Ready→Talking, Talking→Ready
        assert callback.call_count == 2
        # 1 回目: Talking 反映 + metadata 付き
        first = callback.call_args_list[0]
        assert first.args == (
            "mimi", CharacterStatus.TALKING, CharacterStatus.READY, metadata,
        )
        # 2 回目: Ready 反映 (metadata=None)
        second = callback.call_args_list[1]
        assert second.args == (
            "mimi", CharacterStatus.READY, CharacterStatus.TALKING, None,
        )
        # 最終状態は Ready
        assert manager.get_status("mimi") == CharacterStatus.READY

    def test_spawn_handraise_response_playback_no_status_manager_no_op(self):
        """status_manager=None で例外なく動く (= 後方互換、既存呼出経路の互換性確認)。"""
        t = _spawn_handraise_response_playback(
            slug="mimi",
            chunks=[],
            trace_id="t1",
            session_stream_id="s1",
            session_id_root="ss1",
            # status_manager / talking_metadata を渡さない (default None)
        )
        t.join(timeout=2.0)
        # 例外なく完了

    def test_spawn_handraise_response_playback_ready_on_worker_exception(self, monkeypatch):
        """worker 内例外時も finally 経路で Ready 反映 (= 状態が Talking のまま残らない)。"""
        callback = MagicMock()
        manager = CharacterStatusManager(on_status_changed=callback)

        # _run_playback_worker を例外を上げる関数に差し替え
        def boom(*args, **kwargs):
            raise RuntimeError("worker 例外テスト")

        monkeypatch.setattr("lab_lounge.run_loop._run_playback_worker", boom)

        t = _spawn_handraise_response_playback(
            slug="mimi",
            chunks=[],
            trace_id="t1",
            session_stream_id="s1",
            session_id_root="ss1",
            status_manager=manager,
            talking_metadata={"pose": "smile"},
        )
        t.join(timeout=2.0)

        # 例外発生でも finally で Ready 反映される
        assert manager.get_status("mimi") == CharacterStatus.READY
        # callback: TALKING + READY の 2 回 (= finally 経由)
        assert callback.call_count == 2

    def test_create_factory_accepts_status_manager(self):
        """_create_handraise_runner_and_callbacks が status_manager 引数を受け取れる。"""
        manager = CharacterStatusManager()
        bg_runner, on_started, on_phrase_pending, on_approved = (
            _create_handraise_runner_and_callbacks(
                session_stream_id="s1",
                session_id_root="ss1",
                stream_context=None,
                status_manager=manager,
            )
        )
        assert callable(bg_runner)
        assert callable(on_started)
        assert callable(on_phrase_pending)
        assert callable(on_approved)

    def test_create_factory_status_manager_optional(self):
        """status_manager 引数なしでも factory が動く (= 既存テスト互換)。"""
        bg_runner, on_started, on_phrase_pending, on_approved = (
            _create_handraise_runner_and_callbacks(
                session_stream_id="s1",
                session_id_root="ss1",
                stream_context=None,
            )
        )
        assert callable(bg_runner)

    def test_approved_synthesize_fallback_passes_status_manager_to_spawn(self, monkeypatch):
        """_approved_synthesize_fallback が status_manager を _spawn_handraise_response_playback に伝播する。"""
        manager = CharacterStatusManager()
        captured_kwargs: dict = {}

        # _spawn_handraise_response_playback を spy 化
        def fake_spawn(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_response_playback", fake_spawn,
        )

        # run_pipeline は chunks を 1 つ蓄積する mock
        def fake_run_pipeline(text, *, on_tts_chunk_ready, **kwargs):
            on_tts_chunk_ready("url1", "chunk text", True, "mimi", pose="smile")

        monkeypatch.setattr("lab_lounge.run_loop.run_pipeline", fake_run_pipeline)

        _approved_synthesize_fallback(
            "mimi", "transcript", "trace1",
            session_stream_id="s1",
            session_id_root="ss1",
            stream_context=None,
            status_manager=manager,
        )

        # _spawn_handraise_response_playback に status_manager + talking_metadata が渡されている
        assert captured_kwargs.get("status_manager") is manager
        assert captured_kwargs.get("talking_metadata") is not None
        meta = captured_kwargs["talking_metadata"]
        assert meta.get("pose") == "smile"
        # text は chunks の text を結合 (= "chunk text")
        assert meta.get("text") == "chunk text"
