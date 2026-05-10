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
    _create_handraise_runner_and_callbacks,
    _run_playback_worker,
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


# ─── Phase 0.5-B-β-2 commit 3: playback queue drain (却下/lapse cleanup) ──────


class TestPlaybackWorkerDrain:
    """_run_playback_worker の drain task 対応 (Phase 0.5-B-β-2 commit 3)。

    on_handraise_close callback が q.put({"_drain": True}) で投入する drain task
    を受信した worker は、queue 内の通常 task を全破棄 (= 案 A の音声漏れ最小化)。
    None sentinel と他の drain task は維持して、通常の cleanup 経路 (= None 受信
    時の done bubble publish) は機能し続ける。
    """

    def _start_worker(self, q, *, publish_fn=None, play_fn=None, cleanup_fn=None,
                      done_delay=0.01):
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

    def test_drain_task_discards_pending_chunks(self):
        """drain task 受信時、queue 内の通常 task を全破棄して再生されない。

        WHY: 却下/lapse 時の音声漏れ最小化の核心。drain task を最初に置くこと
        で、続けて投入される 3 件の通常 task は受信時点で q.queue 内に残って
        おり、drain で破棄される。play_fn は一度も呼ばれない (= 残音声ゼロ)。
        """
        import queue as queue_mod
        q: queue_mod.Queue = queue_mod.Queue()

        # 順序が重要: 通常 task 3 件 + drain task + None を put
        # worker が起動する前に全部 enqueue することで、worker は drain task を
        # 取り出した時点で q.queue 内に通常 task 2 件 + None が残っている状態
        # になる (= 1 件目は worker が get で取り出して再生中の可能性、テストで
        # は MagicMock 即時 return なので即時完了 → 2 件目を drain 前に処理)
        # → タイミング依存を避けるため、drain task を **先に** 置く。
        q.put({"_drain": True})
        for i in range(3):
            q.put({
                "url": f"file:///{i}.wav", "text": f"t{i}",
                "is_last": False, "character": "mimi",
            })
        q.put(None)

        thread, publish_fn, play_fn, _ = self._start_worker(q)
        thread.join(timeout=2.0)

        # drain で 3 件全破棄、None で終了。play_fn は一度も呼ばれない
        assert play_fn.call_count == 0
        # speaking publish も 0 回 (= speaking_published=False で done もスキップ)
        speaking_calls = [
            c for c in publish_fn.call_args_list if c.args[1] == "speaking"
        ]
        assert speaking_calls == []

    def test_drain_task_keeps_none_sentinel(self):
        """drain task 受信後、None sentinel は維持されて通常終了経路を通る。

        WHY: drain は queue cleanup 専用で、worker 自体の終了は None sentinel が
        担う設計。drain で None まで破棄してしまうと worker が無限待ちになる。
        """
        import queue as queue_mod
        q: queue_mod.Queue = queue_mod.Queue()

        # drain task → None の順 (= 残 task ゼロで drain、None で正常終了)
        q.put({"_drain": True})
        q.put(None)

        thread, _, _, _ = self._start_worker(q)
        thread.join(timeout=2.0)

        # worker が無限待ちせず終了する (= None sentinel が drain で破棄されていない)
        assert not thread.is_alive()

    def test_drain_task_no_op_when_empty(self):
        """queue が空状態で drain task を受信しても crash せず continue する。

        WHY: 却下時に既に playback queue が空 (= まだ ask_character TTS が投入
        されていない、or 既に再生完了) のシナリオで、drain は no-op で安全に通る。
        その後 None sentinel で worker は正常終了する。
        """
        import queue as queue_mod
        q: queue_mod.Queue = queue_mod.Queue()

        # 空 queue → drain → None
        q.put({"_drain": True})
        q.put(None)

        thread, _, _, _ = self._start_worker(q)
        thread.join(timeout=2.0)

        # 例外なく完了 (= no-op drain + 正常終了)
        assert not thread.is_alive()


# ─── Phase 0.5-B-β-3 commit 2: is_last chunk 再生完了時の READY 反映 ─────────


class TestRunPlaybackWorkerStatusReady:
    """_run_playback_worker の status_manager 経由 READY 反映 (Phase 0.5-B-β-3 commit 2)。

    is_last=True chunk の **物理再生完了時** に status_manager.set_status(character, READY)
    を呼ぶ機能。bg_tts 合成完了 != 物理再生完了の不整合を解消する (= シナリオ 2 で
    観察した「HUD で発話途中に灰色化する」不具合の修正、ask_character target キャラ用)。
    通常応答 caller の最終応答経路では _bg_cleanup_pipeline と二重発火するが
    冪等性で害なし。
    """

    def _start_worker(self, q, *, status_manager=None, publish_fn=None,
                      play_fn=None, cleanup_fn=None, done_delay=0.01):
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
                "status_manager": status_manager,
            },
            daemon=True,
        )
        thread.start()
        return thread

    def test_is_last_true_invokes_set_status_ready(self):
        """is_last=True chunk 再生完了時に status_manager.set_status(character, READY) 呼出。

        WHY: シナリオ 2 観察「HUD で発話途中に灰色化」の修正の核心経路。物理再生
        完了 (= cleanup_audio_fn 後) のタイミングで READY 反映することで、視聴者
        体験と HUD 表示が同期する。
        """
        from lab_lounge.character_status import CharacterStatus
        q: queue.Queue = queue.Queue()
        mock_status = MagicMock()

        thread = self._start_worker(q, status_manager=mock_status)
        q.put({
            "url": "file:///a.wav", "text": "x", "is_last": True, "character": "chisame",
        })
        q.put(None)
        thread.join(timeout=2.0)

        # is_last 再生完了で chisame -> READY が 1 回呼ばれる
        ready_calls = [
            c for c in mock_status.set_status.call_args_list
            if c.args[:2] == ("chisame", CharacterStatus.READY)
        ]
        assert len(ready_calls) == 1, (
            f"is_last=True chunk 再生完了時に READY 反映されること: {ready_calls}"
        )

    def test_status_manager_none_no_op(self):
        """status_manager=None なら set_status は呼ばれない (= 後方互換)。

        WHY: Phase 0.5-A 以前の呼出元 (= status_manager 未注入) で動作することを
        保証。引数追加で既存呼出が壊れないこと (= 後方互換) の確認。
        """
        q: queue.Queue = queue.Queue()
        thread = self._start_worker(q, status_manager=None)
        q.put({
            "url": "file:///a.wav", "text": "x", "is_last": True, "character": "mimi",
        })
        q.put(None)
        thread.join(timeout=2.0)
        # status_manager=None なので set_status は一切呼ばれない (= 例外も出ない)
        assert not thread.is_alive()

    def test_non_is_last_does_not_invoke_set_status(self):
        """is_last=False chunk では status_manager.set_status は呼ばれない。

        WHY: 中間 chunks は通常再生のみ (= TALKING 状態の維持)。READY 反映は最後の
        chunk (= is_last=True) のときだけ。途中で READY にすると HUD が灰色化する。
        """
        from lab_lounge.character_status import CharacterStatus
        q: queue.Queue = queue.Queue()
        mock_status = MagicMock()

        thread = self._start_worker(q, status_manager=mock_status)
        # is_last=False の中間 chunks を 2 件 + None
        q.put({
            "url": "file:///a.wav", "text": "x", "is_last": False, "character": "mimi",
        })
        q.put({
            "url": "file:///b.wav", "text": "y", "is_last": False, "character": "mimi",
        })
        q.put(None)
        thread.join(timeout=2.0)

        # is_last=False では READY 反映なし
        ready_calls = [
            c for c in mock_status.set_status.call_args_list
            if c.args[:2] == ("mimi", CharacterStatus.READY)
        ]
        assert ready_calls == [], (
            f"is_last=False chunks では READY 反映されないこと: {ready_calls}"
        )


class TestRunPlaybackWorkerPrePlay:
    """_run_playback_worker の _pre_play_status / _pre_play_bubble dispatch (Phase 0.5-D-2)。

    案 C 設計の核心: defer モード (= BG LLM 経路) で chunk dict に埋め込まれた
    inline metadata を物理再生開始時に発火することで、TALKING / answering bubble
    の「buffer 投入時 → 物理再生時」タイミング移動を実現する。承認前に target が
    TALKING 表示される UX 不具合を構造的に防ぐ。

    通常応答経路 (= defer=False) の chunks にはこのキーが含まれないため (=
    ask_character.py の defer 分岐でのみ埋め込み)、本 dispatch は既存 chunks には
    影響しない (= 後方互換)。
    """

    def _start_worker(self, q, *, status_manager=None, publish_fn=None,
                      play_fn=None, cleanup_fn=None, done_delay=0.01):
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
                "status_manager": status_manager,
            },
            daemon=True,
        )
        thread.start()
        return thread, publish_fn, play_fn

    def test_pre_play_status_dispatched_at_chunk_pop(self):
        """chunk dict の _pre_play_status を pop 時に status_manager.set_status で発火。"""
        from lab_lounge.character_status import CharacterStatus
        q: queue.Queue = queue.Queue()
        mock_status = MagicMock()

        thread, _, _ = self._start_worker(q, status_manager=mock_status)
        q.put({
            "url": "file:///a.wav", "text": "x", "is_last": True, "character": "chisame",
            "_pre_play_status": {
                "slug": "chisame",
                "status": "TALKING",
                "metadata": {"pose": "special_doya", "text": "テスト応答"},
            },
        })
        q.put(None)
        thread.join(timeout=2.0)

        # chisame の TALKING が呼ばれたことを確認 (= chunk pop 時に inline metadata から発火)
        talking_calls = [
            c for c in mock_status.set_status.call_args_list
            if c.args[:2] == ("chisame", CharacterStatus.TALKING)
        ]
        assert len(talking_calls) == 1, (
            f"_pre_play_status から TALKING 反映 1 回 (実際: {talking_calls})"
        )
        # metadata も伝播
        metadata = talking_calls[0].kwargs.get("metadata", {})
        assert metadata.get("pose") == "special_doya"
        assert metadata.get("text") == "テスト応答"

    def test_pre_play_bubble_dispatched_at_chunk_pop(self):
        """chunk dict の _pre_play_bubble を pop 時に publish_bubble_fn で発火。"""
        q: queue.Queue = queue.Queue()
        mock_publish = MagicMock()

        thread, _, _ = self._start_worker(q, publish_fn=mock_publish)
        q.put({
            "url": "file:///a.wav", "text": "x", "is_last": True, "character": "chisame",
            "_pre_play_bubble": {
                "slug": "chisame",
                "step": "answering",
                "text": "ええ、答えますわね",
            },
        })
        q.put(None)
        thread.join(timeout=2.0)

        # chisame の answering bubble が発行されたことを確認
        answering_calls = [
            c for c in mock_publish.call_args_list
            if c.args[:2] == ("chisame", "answering")
        ]
        assert len(answering_calls) == 1, (
            f"_pre_play_bubble から answering bubble 発行 1 回 (実際: {answering_calls})"
        )
        # text も伝播
        assert answering_calls[0].args[2] == "ええ、答えますわね"

    def test_pre_play_dispatched_before_audio_play(self):
        """_pre_play_status / _pre_play_bubble の dispatch は play_audio_fn より先に呼ばれる。

        WHY: 「物理再生開始の直前」に metadata 発火するのが案 C 設計の核 (= HUD
        表示と音声再生の同期)。順序逆転すると「音だけ流れて HUD 反応なし」の
        UX 違和感が出るため、play_audio_fn の前に dispatch されることを保証する。
        """
        from lab_lounge.character_status import CharacterStatus
        q: queue.Queue = queue.Queue()
        mock_status = MagicMock()
        mock_publish = MagicMock()
        mock_play = MagicMock()

        # call 順を記録するため shared list を使う
        call_log: list[str] = []
        mock_status.set_status.side_effect = lambda *a, **kw: call_log.append("set_status")
        mock_publish.side_effect = lambda *a, **kw: call_log.append(
            f"publish:{a[1] if len(a) > 1 else '?'}"
        )
        mock_play.side_effect = lambda *a, **kw: call_log.append("play_audio")

        thread, _, _ = self._start_worker(
            q, status_manager=mock_status,
            publish_fn=mock_publish, play_fn=mock_play,
        )
        q.put({
            "url": "file:///a.wav", "text": "テスト", "is_last": True, "character": "chisame",
            "_pre_play_status": {
                "slug": "chisame", "status": "TALKING",
                "metadata": {"pose": "neutral"},
            },
            "_pre_play_bubble": {
                "slug": "chisame", "step": "answering", "text": "返答",
            },
        })
        q.put(None)
        thread.join(timeout=2.0)

        # play_audio の前に set_status と publish:answering が呼ばれている
        play_idx = call_log.index("play_audio")
        set_status_idx = call_log.index("set_status")
        answering_idx = next(
            (i for i, c in enumerate(call_log) if c == "publish:answering"),
            -1,
        )
        assert set_status_idx < play_idx, (
            f"set_status は play_audio より先 (call_log={call_log})"
        )
        assert answering_idx < play_idx, (
            f"publish(answering) は play_audio より先 (call_log={call_log})"
        )

    def test_no_pre_play_keys_existing_behavior_preserved(self):
        """通常応答 chunks (= _pre_play_* キーなし) は dispatch なし、既存挙動維持。"""
        from lab_lounge.character_status import CharacterStatus
        q: queue.Queue = queue.Queue()
        mock_status = MagicMock()
        mock_publish = MagicMock()

        thread, _, _ = self._start_worker(
            q, status_manager=mock_status, publish_fn=mock_publish,
        )
        # _pre_play_* 無しの chunk (= 通常応答経路)
        q.put({
            "url": "file:///a.wav", "text": "通常応答", "is_last": True, "character": "mimi",
        })
        q.put(None)
        thread.join(timeout=2.0)

        # set_status は is_last=True の READY 反映のみ (= 既存挙動)
        # _pre_play_status からの TALKING 発火はない
        talking_calls = [
            c for c in mock_status.set_status.call_args_list
            if len(c.args) >= 2 and c.args[1] == CharacterStatus.TALKING
        ]
        assert talking_calls == [], (
            f"_pre_play_status なし時は TALKING 反映なし (実際: {talking_calls})"
        )
        # publish_bubble は speaking + done のみ (= answering 発火なし)
        answering_calls = [
            c for c in mock_publish.call_args_list
            if len(c.args) >= 2 and c.args[1] == "answering"
        ]
        assert answering_calls == [], (
            f"_pre_play_bubble なし時は answering 発行なし (実際: {answering_calls})"
        )


class TestRunPlaybackWorkerSpeakerSwitchSleep:
    """Phase 0.5-D-3 follow-up 3: 話者切替時の 0.5 秒間挿入テスト。

    中間実走 5 回目 (logs/runs/run_loop_20260509_194304.log) で観察された
    「フローが滑らかすぎて畳み掛けられている感じ」への対処。話者切替境界で
    0.5 秒の間を取ることで、視聴者の認知的に「会話のターン交代」が知覚
    しやすくなる。streaming spawn (= follow-up 2) の連続再生の自然さを
    補完する設計。
    """

    def _start_worker(self, q, *, status_manager=None, publish_fn=None,
                      play_fn=None, cleanup_fn=None, done_delay=0.01):
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
                "status_manager": status_manager,
            },
            daemon=True,
        )
        thread.start()
        return thread

    def test_no_sleep_on_first_chunk(self, monkeypatch):
        """初回 chunk (= prev_played_character=None) では話者切替 sleep が呼ばれない。

        WHY: approval → 即時音声再生 (= follow-up 2 streaming spawn) の効果を維持
        するため、最初の chunk の物理再生開始は遅延させない。
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("lab_lounge.run_loop.time.sleep", sleep_calls.append)

        q: queue.Queue = queue.Queue()
        thread = self._start_worker(q)
        q.put({
            "url": "file:///a.wav", "text": "first", "is_last": True, "character": "mimi",
        })
        q.put(None)
        thread.join(timeout=2.0)

        # 話者切替 sleep (= 0.5 秒) のみフィルタ (done_delay=0.01 とは別物)
        switch_sleeps = [s for s in sleep_calls if 0.49 <= s <= 0.51]
        assert switch_sleeps == [], (
            f"初回 chunk では話者切替 sleep 呼ばれない (実際: {switch_sleeps})"
        )

    def test_no_sleep_on_same_character_consecutive(self, monkeypatch):
        """同一キャラ連続 chunks では話者切替 sleep が呼ばれない。"""
        sleep_calls: list[float] = []
        monkeypatch.setattr("lab_lounge.run_loop.time.sleep", sleep_calls.append)

        q: queue.Queue = queue.Queue()
        thread = self._start_worker(q)
        for i in range(3):
            q.put({
                "url": f"file:///{i}.wav", "text": str(i),
                "is_last": (i == 2), "character": "mimi",
            })
        q.put(None)
        thread.join(timeout=2.0)

        switch_sleeps = [s for s in sleep_calls if 0.49 <= s <= 0.51]
        assert switch_sleeps == [], (
            f"同一キャラ連続では話者切替 sleep 呼ばれない (実際: {switch_sleeps})"
        )

    def test_sleep_inserted_on_character_switch(self, monkeypatch):
        """話者切替時に time.sleep(0.5) が呼ばれる。

        WHY: 「caller 導入セリフ → bridge filler (target) → target 応答 → caller 〆」
        のような切替境界で短い間を入れる。chunks: mimi → chisame → mimi の場合、
        切替 2 箇所で sleep(0.5) が 2 回呼ばれる。
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("lab_lounge.run_loop.time.sleep", sleep_calls.append)

        q: queue.Queue = queue.Queue()
        thread = self._start_worker(q)
        chunks_data = [
            ("mimi", "intro"),
            ("chisame", "response"),
            ("mimi", "outro"),
        ]
        for i, (char, text) in enumerate(chunks_data):
            q.put({
                "url": f"file:///{i}.wav", "text": text,
                "is_last": (i == len(chunks_data) - 1), "character": char,
            })
        q.put(None)
        thread.join(timeout=2.0)

        switch_sleeps = [s for s in sleep_calls if 0.49 <= s <= 0.51]
        assert len(switch_sleeps) == 2, (
            f"mimi → chisame → mimi の 2 切替で sleep 2 回 "
            f"(実際: {switch_sleeps}、全 sleep: {sleep_calls})"
        )

    def test_sleep_inserted_even_for_text_empty_bridge_filler(self, monkeypatch):
        """text="" (= bridge filler) でも話者切替判定が機能する。

        WHY: bridge filler (= caller=mimi 後の target=sakura、text="") を挟んで
        「mimi → sakura(bridge) → sakura(本応答)」となるシナリオで、
        mimi → sakura で 1 回 sleep、sakura(bridge) → sakura(本応答) で 0 回 sleep
        になることを保証。prev_played_character は text 有無に関わらず更新される。
        """
        sleep_calls: list[float] = []
        monkeypatch.setattr("lab_lounge.run_loop.time.sleep", sleep_calls.append)

        q: queue.Queue = queue.Queue()
        thread = self._start_worker(q)
        # mimi 導入 → sakura bridge filler (text="") → sakura 本応答
        q.put({
            "url": "file:///mimi.wav", "text": "ルカ、その問いは",
            "is_last": False, "character": "mimi",
        })
        q.put({
            "url": "file:///sakura_bridge.wav", "text": "",
            "is_last": False, "character": "sakura",
        })
        q.put({
            "url": "file:///sakura_resp.wav", "text": "ん〜……",
            "is_last": True, "character": "sakura",
        })
        q.put(None)
        thread.join(timeout=2.0)

        switch_sleeps = [s for s in sleep_calls if 0.49 <= s <= 0.51]
        # mimi → sakura で 1 回 (= bridge filler 投入時点)、sakura → sakura で 0 回
        assert len(switch_sleeps) == 1, (
            f"mimi → sakura(bridge) で 1 切替、sakura(bridge) → sakura(本応答) で 0 切替 "
            f"(実際: {switch_sleeps}、全 sleep: {sleep_calls})"
        )


class TestHandraiseCloseFlow:
    """factory の on_handraise_close callback (Phase 0.5-B-β-2 commit 3)。"""

    def _factory(self, **overrides):
        from lab_lounge.run_loop import _create_handraise_runner_and_callbacks
        defaults = dict(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
        )
        defaults.update(overrides)
        return _create_handraise_runner_and_callbacks(**defaults)

    def test_factory_returns_six_callables_including_progressing(self):
        """factory 戻り値 tuple が 6 要素 (Phase 0.5-D-d-2 で +1、旧 5 要素から拡張)。

        WHY: Phase 0.5-D-d-2 で戻り値を 5 → 6 要素に拡張した (= bridge filler 即時
        再生 callback ``on_approval_progressing`` の追加)。run_loop と既存テスト
        が unpack する側で同期更新されているか保証する。Phase 0.5-B-β-2 commit 3
        で 4 → 5 要素にした拡張パターンを踏襲。
        """
        bg_runner, on_started, on_release, on_approved, on_close, on_progressing = self._factory()
        assert callable(bg_runner)
        assert callable(on_started)
        assert callable(on_release)
        # on_approved は F-4-c で None 化 (= factory 全体は F-4-d で削除予定)
        assert callable(on_close)
        assert callable(on_progressing)

    def test_on_handraise_close_invokes_cancel_bg_tts_and_drain(self, monkeypatch):
        """on_handraise_close 起動で cancel_bg_tts(session_id) + drain task 投入。

        WHY: 案 A の音声漏れ最小化の核心経路。dispatcher 経由で本 callback が
        呼ばれた時、ask_character の bg_tts キャンセル (= 未起動 thread 阻止) +
        playback queue drain (= 投入済 chunks 破棄) の両方が実行されることを保証。
        """
        import queue as queue_mod

        # cancel_bg_tts を spy
        cancel_calls: list[str] = []
        monkeypatch.setattr(
            "lab_lounge.mcp_servers.ask_character.cancel_bg_tts",
            lambda session_id: cancel_calls.append(session_id) or 0,
        )

        # 実 queue.Queue を渡して drain task 投入を観測
        playback_q: queue_mod.Queue = queue_mod.Queue()
        playback_ref: list = [playback_q]

        _, _, _, _, on_close, _ = self._factory(
            session_id_root="ses1-test",
            playback_queue_ref=playback_ref,
        )
        on_close("mimi", "denied")

        # cancel_bg_tts(session_id="ses1-test") で呼ばれた
        assert cancel_calls == ["ses1-test"]
        # playback queue に drain task が投入された
        assert not playback_q.empty()
        item = playback_q.get_nowait()
        assert isinstance(item, dict)
        assert item.get("_drain") is True


# ─── Phase 0.5-B-β-2 commit 4: 却下/lapse drain 貫通テスト ───────────────


class TestLlmOnlyAskCharacterDenialDrain:
    """却下/lapse 経由の cancel_bg_tts + drain 統合テスト (Phase 0.5-B-β-2 commit 4)。

    β-2-1 (= dispatcher の on_handraise_close 発火) → β-2-3 (= factory の
    on_handraise_close 実装で cancel_bg_tts + drain) → β-2-2 (= ask_character の
    bg_tts キャンセル) の貫通動作を保証する。run_loop 全体を起動せず、
    dispatcher と factory を直接結合して却下/lapse 経路を観測する。
    """

    class _FakeTimer:
        """threading.Timer 差替用 (= test_dispatcher._FakeTimer 相当のローカル版)。"""

        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or []
            self.kwargs = kwargs or {}

        def start(self) -> None:
            pass

        def cancel(self) -> None:
            pass

        def fire(self) -> None:
            """手動発火 = lapse_timer のタイムアウトをシミュレート。"""
            self.function(*self.args, **self.kwargs)

    def _build_dispatcher_with_close(self, monkeypatch, *, slug: str = "mimi",
                                     session_id_root: str = "ses1"):
        """factory + Dispatcher を結合し、cancel_bg_tts を spy 化したセットを返す。

        test_dispatcher._patch_filler / _patch_lapse_timer 相当の mock を class 内
        helper としてインラインに書く (= test ファイル間の dependency を避ける)。
        """
        import queue as queue_mod
        from pathlib import Path

        from lab_lounge.dispatcher import Dispatcher
        from lab_lounge.run_loop import _create_handraise_runner_and_callbacks

        # cancel_bg_tts を spy (= ask_character module の関数を差し替え、
        # factory の on_handraise_close 内 import で取得される側を patch)
        cancel_calls: list[str] = []

        def _spy_cancel(s_id):
            cancel_calls.append(s_id)
            return 0

        monkeypatch.setattr(
            "lab_lounge.mcp_servers.ask_character.cancel_bg_tts", _spy_cancel,
        )

        # filler を mock (= test_dispatcher._patch_filler 相当、handraise wav の
        # 物理ファイル参照を回避するため select_filler_phrase をスタブ化)
        fake_path = Path(f"/tmp/{slug}_handraise.wav")
        fake_phrase = MagicMock()
        fake_phrase.text = "挙手します"
        monkeypatch.setattr(
            "lab_lounge.filler.select_filler_phrase",
            lambda s, category="opener", **kw: (fake_path, fake_phrase, 0),
        )

        # bubble_messages を mock
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {slug: {"denied": "また今度", "lapsed": "静かに"}},
        )

        # factory で 5 callable 取得 (= playback_queue_ref を渡して drain 観測可能に)
        playback_q: queue_mod.Queue = queue_mod.Queue()
        playback_ref: list = [playback_q]

        _, _, _, _, on_close, _ = _create_handraise_runner_and_callbacks(
            session_stream_id="s1",
            session_id_root=session_id_root,
            stream_context=None,
            playback_queue_ref=playback_ref,
        )

        d = Dispatcher(on_handraise_close=on_close)

        # lapse_timer を _FakeTimer で差し替え (= test_dispatcher._patch_lapse_timer 相当)
        fake_timers: list = []

        def fake_create(target_slug: str, delay_sec: float):
            t = self._FakeTimer(
                delay_sec,
                d.on_lapse_timeout,
                args=[target_slug],
            )
            fake_timers.append(t)
            return t

        monkeypatch.setattr(d, "_create_lapse_timer", fake_create)
        return d, cancel_calls, playback_q, fake_timers

    def test_denial_triggers_cancel_and_drain_via_dispatcher(self, monkeypatch):
        """dispatcher.on_approval_denied → on_handraise_close → cancel_bg_tts + drain。

        WHY: β-2 全 commit の最終的な統合動作を保証する。dispatcher 経由で
        却下を起こすと、ask_character の bg_tts キャンセル + playback queue の
        drain task 投入が両方走る (= 案 A の音声漏れ最小化が end-to-end で機能)。
        """
        d, cancel_calls, playback_q, _ = self._build_dispatcher_with_close(
            monkeypatch, session_id_root="ses1-denial",
        )

        # handraise 状態を作る (= on_interjection_candidate 経由)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # 却下発火
        d.on_approval_denied("mimi")

        # cancel_bg_tts(session_id="ses1-denial") が呼ばれた
        assert cancel_calls == ["ses1-denial"]
        # playback queue に drain task が投入された
        assert not playback_q.empty()
        item = playback_q.get_nowait()
        assert isinstance(item, dict)
        assert item.get("_drain") is True

    def test_lapse_triggers_cancel_and_drain_via_dispatcher(self, monkeypatch):
        """dispatcher.on_lapse_timeout → on_handraise_close → cancel_bg_tts + drain。

        WHY: lapse (= タイムアウト) も denial と同じ cleanup 経路を通ること。
        reason="lapsed" で run_loop は識別できるが、現状の cleanup ロジックは
        denial と同じ (= 将来の metric 分離の基盤は β-2-1 で確保済み)。
        """
        d, cancel_calls, playback_q, timers = self._build_dispatcher_with_close(
            monkeypatch, session_id_root="ses1-lapse",
        )

        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # lapse 発火 (= FakeTimer の手動発火 = on_lapse_timeout 呼出)
        timers[0].fire()

        assert cancel_calls == ["ses1-lapse"]
        assert not playback_q.empty()
        item = playback_q.get_nowait()
        assert isinstance(item, dict)
        assert item.get("_drain") is True


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
        """Dispatcher kwargs の signature 検証 (Phase 0.5-F-3 で案 R 用に改変)。

        【改変履歴】
        - Phase 0.5-A: bg_runner / on_handraise_started / on_handraise_phrase_pending_release
          / on_handraise_approved の 4 callable
        - Phase 0.5-B-β-2 commit 3: on_handraise_close 追加 (= 5 callable)
        - Phase 0.5-D-d-2: on_approval_progressing 追加 (= 6 callable)
        - **Phase 0.5-F-3 (本テスト改変)**: 案 R 経路に統合
          - bg_runner=None (= LLM 先行計算なし、Phase 0.5-A 案 W'-1 破棄)
          - on_handraise_approved=None (= 承認時 callout 経路統合)
          - on_approval_progressing=None (= bridge filler 専用経路廃止、R-1-b)
          - on_approval_replay=callable 新規追加 (= 案 R の中核)

        F-3 で本テストは案 R 用に改変済。配信事故レベル沈黙問題 (= 中間実走 13
        シナリオ γ) への構造的対処。詳細は plan ファイル参照。
        """
        from lab_lounge.run_loop import run_loop

        _, dispatcher_kwargs = self._setup_spies(monkeypatch)

        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        # 必須 kwargs が dispatcher_kwargs に含まれていること
        assert "bg_runner" in dispatcher_kwargs
        assert "on_handraise_started" in dispatcher_kwargs
        assert "on_handraise_phrase_pending_release" in dispatcher_kwargs
        assert "on_handraise_approved" in dispatcher_kwargs
        assert "on_handraise_close" in dispatcher_kwargs
        assert "on_approval_replay" in dispatcher_kwargs

        # 案 R: 廃止経路は None 固定
        assert dispatcher_kwargs["bg_runner"] is None, (
            "案 R では bg_runner=None (= LLM 先行計算なし)"
        )
        assert dispatcher_kwargs["on_handraise_approved"] is None, (
            "案 R では on_handraise_approved=None (= callout 経路統合)"
        )

        # 残る経路は callable
        assert callable(dispatcher_kwargs["on_handraise_started"])
        assert callable(dispatcher_kwargs["on_handraise_phrase_pending_release"])
        assert callable(dispatcher_kwargs["on_handraise_close"])

        # 案 R 中核 callback は callable
        assert callable(dispatcher_kwargs["on_approval_replay"]), (
            "案 R の中核 callback `on_approval_replay` が wired されていない"
        )

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

    def test_returns_six_callables(self):
        """factory 戻り値 tuple が 6 要素 (Phase 0.5-D-d-2 で +1、Phase 0.5-B-β-2
        commit 3 の 5 要素から拡張)。"""
        bg_runner, on_started, on_release, on_approved, on_close, on_progressing = self._factory()
        assert callable(bg_runner)
        assert callable(on_started)
        assert callable(on_release)
        # on_approved は F-4-c で None 化 (= factory 全体は F-4-d で削除予定)
        assert callable(on_close)
        assert callable(on_progressing)

    # ─── Phase 0.5-D-d-2: on_approval_progressing テスト ────────────

    def test_on_approval_progressing_calls_bridge_filler(self, monkeypatch):
        """on_approval_progressing が target キャラの bridge filler を即時再生する
        (Phase 0.5-D-d-2)。

        ルカが「ミミ様、どうぞ」承認時に BG LLM 未完了の場合、dispatcher が本 callback
        を発火 → run_loop が target キャラの bridge filler (4-8 秒程度の繋ぎセリフ) を
        専用 mini playback worker で即時再生する経路。30 秒沈黙の配信事故レベル対処。
        """
        from pathlib import Path
        spawn_calls: list[tuple] = []
        select_calls: list[tuple] = []

        def fake_select_filler(slug, category, *, last_index=-1):
            select_calls.append((slug, category))
            return Path("/tmp/bridge.wav"), 0

        def fake_spawn(slug, phrase_path, padding_sec=0.0, **kw):
            spawn_calls.append((slug, phrase_path, padding_sec))
            return None

        monkeypatch.setattr(
            "lab_lounge.filler.select_filler_path", fake_select_filler,
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_phrase_playback", fake_spawn,
        )

        _, _, _, _, _, on_progressing = self._factory()
        on_progressing("chisame")

        # bridge category で select_filler_path が呼ばれる
        assert select_calls == [("chisame", "bridge")]
        # _spawn_handraise_phrase_playback が padding=0 で呼ばれる
        assert spawn_calls == [("chisame", Path("/tmp/bridge.wav"), 0.0)]

    def test_on_approval_progressing_no_bridge_filler_no_op(self, monkeypatch):
        """bridge filler が存在しない (= select_filler_path が (None, -1) 返す)
        場合、on_approval_progressing は no-op + 例外なし (Phase 0.5-D-d-2)。

        WHY: filler ディレクトリが空 / カテゴリ未定義のキャラでも例外発生せず、
        他経路 (= bg_completed.wait + 通常 fallback) は通常通り動く設計。
        """
        spawn_calls: list = []

        def fake_select_filler(slug, category, *, last_index=-1):
            return None, -1

        def fake_spawn(slug, phrase_path, padding_sec=0.0, **kw):
            spawn_calls.append(slug)
            return None

        monkeypatch.setattr(
            "lab_lounge.filler.select_filler_path", fake_select_filler,
        )
        monkeypatch.setattr(
            "lab_lounge.run_loop._spawn_handraise_phrase_playback", fake_spawn,
        )

        _, _, _, _, _, on_progressing = self._factory()
        on_progressing("mimi")  # 例外発生せず

        # bridge filler なし → spawn 呼ばれない
        assert spawn_calls == []

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
        _, on_started, _, _, _, _ = self._factory()
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
        _, on_started, _, _, _, _ = self._factory()
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
        _, _, on_release, _, _, _ = self._factory()
        on_release("sakura", Path("/tmp/x.wav"))
        assert spawn_calls == [("sakura", Path("/tmp/x.wav"), 0.7)]

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

        bg_runner, _, _, _, _, _ = self._factory()
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


# ─── TestStatusManagerWiring (Phase 0.5-B-α) ─────────────────────


class TestStatusManagerWiring:
    """Phase 0.5-B-α: run_loop 経由の CharacterStatusManager 配線検証。

    _create_handraise_runner_and_callbacks に optional な status_manager 引数が
    伝播し、Talking + Ready 反映 / metadata 流れが正しいことを検証する。
    """

    def test_create_factory_accepts_status_manager(self):
        """_create_handraise_runner_and_callbacks が status_manager 引数を受け取れる。"""
        manager = CharacterStatusManager()
        bg_runner, on_started, on_phrase_pending, on_approved, on_close, on_progressing = (
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
        # on_approved は F-4-c で None 化 (= factory 全体は F-4-d で削除予定)
        assert callable(on_close)

    def test_create_factory_status_manager_optional(self):
        """status_manager 引数なしでも factory が動く (= 既存テスト互換)。"""
        bg_runner, on_started, on_phrase_pending, on_approved, on_close, on_progressing = (
            _create_handraise_runner_and_callbacks(
                session_stream_id="s1",
                session_id_root="ss1",
                stream_context=None,
            )
        )
        assert callable(bg_runner)


# ─── Phase 0.5-F-2: 案 R 用 replay factory のテスト ─────────────────


class TestCreateReplayRunnerAndCallbacks:
    """Phase 0.5-F-2: ``_create_replay_runner_and_callbacks`` の単体テスト。

    案 R (= raisehand を callout 経路に統合) のための新 factory。F-2 では wiring
    していないので、単体テストで return tuple の構造 + 各 callable の動作のみを
    検証する。F-3 commit で run_loop が本 factory に切替えると初めて動作する。
    """

    def test_factory_returns_4_callables(self):
        """factory は 4 つの callable を tuple で返す。

        既存 ``_create_handraise_runner_and_callbacks`` の 6-tuple に対し、
        案 R 用は 4-tuple (= bg_runner / on_handraise_approved / on_approval_progressing 削除)。
        """
        from lab_lounge.run_loop import _create_replay_runner_and_callbacks

        result = _create_replay_runner_and_callbacks(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
        )
        assert len(result) == 4
        for cb in result:
            assert callable(cb), f"factory の return 要素が callable ではない: {cb}"

    def test_on_handraise_close_is_no_op(self, caplog):
        """on_handraise_close は no-op (= 案 R では cancel 不要)。

        WHY: 案 R では bg_runner._body が起動しないので、cancel 対象の bg_tts thread
        は存在しない。playback queue drain は通常応答 TTS を誤 drain する害があるため
        呼ばない。ログのみ残す。
        """
        import logging
        from lab_lounge.run_loop import _create_replay_runner_and_callbacks

        _, _, on_handraise_close, _ = _create_replay_runner_and_callbacks(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
        )

        caplog.set_level(logging.INFO, logger="lab_lounge.run_loop")
        # 例外 raise なしで完走すること
        on_handraise_close("mimi", "denied")
        on_handraise_close("chisame", "lapsed")

        # ログに reason が記録されている
        denied_logs = [r for r in caplog.records if "reason=denied" in r.message]
        lapsed_logs = [r for r in caplog.records if "reason=lapsed" in r.message]
        assert denied_logs, "denied reason のログが出ていない"
        assert lapsed_logs, "lapsed reason のログが出ていない"

    def test_on_approval_replay_injects_wake_event(self):
        """on_approval_replay は ``dispatcher.on_wake_detected(WakeWordResult)`` を呼ぶ。

        案 R の中核経路: 承認時に wake_event queue に inject することで、callout 経路と
        完全に同じコードパスを通る (= run_pipeline 起動)。
        """
        from lab_lounge.run_loop import _create_replay_runner_and_callbacks

        # dispatcher_ref に MagicMock を late binding で設定
        mock_dispatcher = MagicMock()
        dispatcher_ref = [mock_dispatcher]

        _, _, _, on_approval_replay = _create_replay_runner_and_callbacks(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
            dispatcher_ref=dispatcher_ref,
        )

        on_approval_replay("mimi", "ルカ発話の transcript")

        # on_wake_detected が呼ばれていること
        assert mock_dispatcher.on_wake_detected.called, "on_wake_detected が呼ばれていない"
        wake_result = mock_dispatcher.on_wake_detected.call_args[0][0]
        assert wake_result.character_slug == "mimi"
        assert wake_result.transcript == "ルカ発話の transcript"
        assert wake_result.keyword == "<approval>"  # 案 R 専用識別子
        assert wake_result.keyword_index == -1

    def test_on_approval_replay_handles_none_transcript(self):
        """transcript_snapshot が None / 空文字でも空文字で inject する。

        WHY: run_pipeline は空文字でも処理可能 (= 通常応答経路で transcript=None の
        ケースもある)、no-op で skip するより inject する方が一貫性高い。
        """
        from lab_lounge.run_loop import _create_replay_runner_and_callbacks

        mock_dispatcher = MagicMock()
        dispatcher_ref = [mock_dispatcher]
        _, _, _, on_approval_replay = _create_replay_runner_and_callbacks(
            session_stream_id="s1",
            session_id_root="ses1",
            stream_context=None,
            dispatcher_ref=dispatcher_ref,
        )

        on_approval_replay("chisame", None)

        wake_result = mock_dispatcher.on_wake_detected.call_args[0][0]
        assert wake_result.transcript == ""

    def test_on_approval_replay_no_op_when_dispatcher_ref_none(self, caplog):
        """dispatcher_ref が None / [None] のとき、no-op + warning ログ。

        WHY: F-3 wiring 完了前に F-2 factory が単独テストされるとき、dispatcher が
        構築されていない。warning を出すが例外は raise しない (= 防衛的、F-3 wiring
        前後の挙動を区別可能)。
        """
        import logging
        from lab_lounge.run_loop import _create_replay_runner_and_callbacks

        # ケース 1: dispatcher_ref=None
        _, _, _, replay_a = _create_replay_runner_and_callbacks(
            session_stream_id="s1", session_id_root="ses1",
            stream_context=None, dispatcher_ref=None,
        )
        # ケース 2: dispatcher_ref=[None]
        _, _, _, replay_b = _create_replay_runner_and_callbacks(
            session_stream_id="s1", session_id_root="ses1",
            stream_context=None, dispatcher_ref=[None],
        )

        caplog.set_level(logging.WARNING, logger="lab_lounge.run_loop")
        replay_a("mimi", "snap")  # 例外なし
        replay_b("chisame", "snap")  # 例外なし

        warning_records = [
            r for r in caplog.records if "dispatcher_ref" in r.message
        ]
        assert len(warning_records) >= 2, (
            f"dispatcher_ref 未設定の warning が 2 件出ていない: "
            f"{[r.message for r in caplog.records]}"
        )
