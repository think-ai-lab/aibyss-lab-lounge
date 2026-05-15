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

    def test_is_last_resets_pose_to_neutral(self):
        """Phase 0.5-K-4: is_last=True chunk の物理再生完了時に set_pose_fn(character, "neutral")
        が呼ばれる。

        発話終了後に立ち絵が special_overdrive / special_pondering 等の特殊ポーズの
        ままになる事象 (= run_loop_20260516_045722.log で観察) を構造的に解消する。
        ask_character の caller / target それぞれ独立に is_last chunk を持つため、
        本テストは複数キャラ混在シナリオで個別に neutral 戻しが呼ばれることを検証する。
        """
        q: queue.Queue = queue.Queue()
        pose_calls: list[tuple[str, str]] = []
        publish_fn = MagicMock()
        play_fn = MagicMock()
        cleanup_fn = MagicMock()

        def set_pose_fn(character: str, pose: str) -> None:
            pose_calls.append((character, pose))

        thread = threading.Thread(
            target=_run_playback_worker,
            args=(q,),
            kwargs={
                "publish_bubble_fn": publish_fn,
                "play_audio_fn": play_fn,
                "cleanup_audio_fn": cleanup_fn,
                "set_pose_fn": set_pose_fn,
                "done_delay_seconds": 0.01,
            },
            daemon=True,
        )
        thread.start()

        # mimi → sakura → mimi、最後 is_last はキャラごとに発火
        # 注: pose 変更は set_pose_fn 経由で起きるため、character 切替ごとに先頭の pose 切替も
        #     pose_calls に乗る。 本 test では neutral 戻しが is_last chunk 後に発火することのみ確認。
        q.put({"url": "file:///a.wav", "text": "A", "is_last": False, "character": "mimi", "pose": "happy"})
        q.put({"url": "file:///b.wav", "text": "B", "is_last": True, "character": "sakura", "pose": "fun"})
        q.put({"url": "file:///c.wav", "text": "C", "is_last": True, "character": "mimi", "pose": "happy"})
        q.put(None)
        thread.join(timeout=2.0)

        # is_last=True chunk ごとに neutral 戻しが呼ばれていること
        neutral_calls = [c for c in pose_calls if c[1] == "neutral"]
        assert ("sakura", "neutral") in neutral_calls
        assert ("mimi", "neutral") in neutral_calls

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


# ─── Phase 0.5-B-β-2 commit 4: 却下/lapse drain 貫通テスト ───────────────


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
        """Dispatcher kwargs の signature 検証 (Phase 0.5-F-3 → F-4-f-2 で簡素化)。

        【改変履歴】
        - Phase 0.5-A: bg_runner / on_handraise_started / on_handraise_phrase_pending_release
          / on_handraise_approved の 4 callable
        - Phase 0.5-B-β-2 commit 3: on_handraise_close 追加 (= 5 callable)
        - Phase 0.5-D-d-2: on_approval_progressing 追加 (= 6 callable)
        - Phase 0.5-F-3: 案 R 経路に統合 (= bg_runner / on_handraise_approved /
          on_approval_progressing を None で wiring、on_approval_replay 新規)
        - **Phase 0.5-F-4-f-2 (本テスト改変)**: 旧引数完全削除
          - bg_runner / on_handraise_approved / on_approval_progressing 引数自体を
            Dispatcher.__init__ から削除 (= None 固定すら不要、引数ごと消滅)
          - 案 R 経路 (= on_approval_replay) のみが残る
        """
        from lab_lounge.run_loop import run_loop

        _, dispatcher_kwargs = self._setup_spies(monkeypatch)

        run_loop(
            max_turns=0,
            wake_backend="bg-continuous",
            wake_timeout=0.1,
        )

        # 必須 kwargs が dispatcher_kwargs に含まれていること
        assert "on_handraise_started" in dispatcher_kwargs
        assert "on_handraise_phrase_pending_release" in dispatcher_kwargs
        assert "on_handraise_close" in dispatcher_kwargs
        assert "on_approval_replay" in dispatcher_kwargs

        # F-4-f-2: 旧 kwargs (bg_runner / on_handraise_approved / on_approval_progressing)
        # は引数自体が削除されているので存在しないことを確認
        assert "bg_runner" not in dispatcher_kwargs
        assert "on_handraise_approved" not in dispatcher_kwargs
        assert "on_approval_progressing" not in dispatcher_kwargs

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
