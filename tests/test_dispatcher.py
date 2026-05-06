"""
test_dispatcher.py — Dispatcher のテスト (Block 0)

責務:
  - 状態遷移 (IDLE / RESPONDING / HANDRAISING) の検証
  - wake_event_queue の add / dequeue / drain ルール検証
  - 60 秒期限切れ + 上限 3 件のドレイン戦略検証
  - publish callback の発火タイミング検証
  - Phase 0.5 用 API が NotImplementedError を raise すること

外部依存 (sounddevice / STT 等) はなく、純粋な状態機械のテスト。
"""

import threading
import time

import pytest

from lab_lounge.dispatcher import (
    DRAIN_MAX_AGE_SEC,
    DRAIN_MAX_EVENTS,
    Dispatcher,
    DispatcherState,
    QueuedWakeEvent,
)
from lab_lounge.wake_word import WakeWordResult


# ─── ヘルパー ─────────────────────────────────────────────────────


def _make_wake(slug: str, transcript: str | None = None) -> WakeWordResult:
    """テスト用の WakeWordResult を生成する。"""
    return WakeWordResult(
        keyword=f"{slug}-wake",
        character_slug=slug,
        keyword_index=0,
        transcript=transcript,
    )


# ─── TestDispatcherInit ───────────────────────────────────────────


class TestDispatcherInit:
    def test_initial_state_is_idle(self):
        d = Dispatcher()
        assert d.get_state() == DispatcherState.IDLE

    def test_initial_queue_is_empty(self):
        d = Dispatcher()
        assert d.get_queue_snapshot() == []

    def test_default_drain_constants(self):
        """DRAIN_MAX_EVENTS=3 / DRAIN_MAX_AGE_SEC=60.0 が仮置きで設定されている。"""
        assert DRAIN_MAX_EVENTS == 3
        assert DRAIN_MAX_AGE_SEC == 60.0

    def test_handraise_dicts_are_empty(self):
        """Phase 0.5 用 dict は空で予約されている (Block 0 では使わない)。"""
        d = Dispatcher()
        assert d._handraise_states == {}
        assert d._cooldowns == {}


# ─── TestDispatcherStateTransition ────────────────────────────────


class TestDispatcherStateTransition:
    def test_transition_to_responding(self):
        d = Dispatcher()
        d.transition_to(DispatcherState.RESPONDING)
        assert d.get_state() == DispatcherState.RESPONDING

    def test_transition_back_to_idle(self):
        d = Dispatcher()
        d.transition_to(DispatcherState.RESPONDING)
        d.transition_to(DispatcherState.IDLE)
        assert d.get_state() == DispatcherState.IDLE

    def test_on_pipeline_complete_returns_to_idle(self):
        d = Dispatcher()
        d.transition_to(DispatcherState.RESPONDING)
        d.on_pipeline_complete()
        assert d.get_state() == DispatcherState.IDLE


# ─── TestDispatcherOnWakeDetected ─────────────────────────────────


class TestDispatcherOnWakeDetected:
    def test_event_is_enqueued(self):
        d = Dispatcher()
        d.on_wake_detected(_make_wake("mimi"))
        snap = d.get_queue_snapshot()
        assert len(snap) == 1
        assert snap[0].event.character_slug == "mimi"

    def test_multiple_events_preserve_order(self):
        d = Dispatcher()
        d.on_wake_detected(_make_wake("mimi"))
        d.on_wake_detected(_make_wake("chisame"))
        d.on_wake_detected(_make_wake("sakura"))
        snap = d.get_queue_snapshot()
        assert [q.event.character_slug for q in snap] == ["mimi", "chisame", "sakura"]

    def test_enqueue_during_responding_is_allowed(self):
        """RESPONDING 中でも queue には積まれる。"""
        d = Dispatcher()
        d.transition_to(DispatcherState.RESPONDING)
        d.on_wake_detected(_make_wake("chisame"))
        snap = d.get_queue_snapshot()
        assert len(snap) == 1
        # 状態は RESPONDING のまま (on_wake_detected で勝手に変わらない)
        assert d.get_state() == DispatcherState.RESPONDING

    def test_enqueued_at_is_recorded(self):
        d = Dispatcher()
        before = time.monotonic()
        d.on_wake_detected(_make_wake("mimi"))
        after = time.monotonic()
        snap = d.get_queue_snapshot()
        assert before <= snap[0].enqueued_at <= after


# ─── TestDispatcherWaitForNextEvent ───────────────────────────────


class TestDispatcherWaitForNextEvent:
    def test_returns_none_on_timeout(self):
        d = Dispatcher()
        result = d.wait_for_next_event(timeout=0.05)
        assert result is None

    def test_returns_event_immediately_when_available(self):
        d = Dispatcher()
        d.on_wake_detected(_make_wake("mimi"))
        result = d.wait_for_next_event(timeout=1.0)
        assert result is not None
        assert result.character_slug == "mimi"

    def test_dequeue_transitions_to_responding(self):
        d = Dispatcher()
        d.on_wake_detected(_make_wake("mimi"))
        d.wait_for_next_event(timeout=1.0)
        assert d.get_state() == DispatcherState.RESPONDING

    def test_dequeue_removes_from_queue(self):
        d = Dispatcher()
        d.on_wake_detected(_make_wake("mimi"))
        d.on_wake_detected(_make_wake("chisame"))
        first = d.wait_for_next_event(timeout=1.0)
        assert first is not None and first.character_slug == "mimi"
        snap = d.get_queue_snapshot()
        assert len(snap) == 1
        assert snap[0].event.character_slug == "chisame"

    def test_blocking_wait_unblocked_by_on_wake_detected(self):
        """別スレッドから on_wake_detected が呼ばれたら wait_for_next_event が起きる。"""
        d = Dispatcher()
        result_holder: list[WakeWordResult | None] = [None]

        def waiter():
            result_holder[0] = d.wait_for_next_event(timeout=2.0)

        t = threading.Thread(target=waiter)
        t.start()
        # waiter が wait に入るのを少し待つ
        time.sleep(0.1)
        d.on_wake_detected(_make_wake("sakura"))
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert result_holder[0] is not None
        assert result_holder[0].character_slug == "sakura"


# ─── TestDispatcherDrainRules ─────────────────────────────────────


class TestDispatcherDrainRules:
    def test_overflow_evicts_oldest(self):
        """4 件目を入れると古い 1 件が破棄される (上限 3 件)。"""
        d = Dispatcher(max_events=3)
        d.on_wake_detected(_make_wake("a"))
        d.on_wake_detected(_make_wake("b"))
        d.on_wake_detected(_make_wake("c"))
        d.on_wake_detected(_make_wake("d"))  # ここで a が evict
        snap = d.get_queue_snapshot()
        assert [q.event.character_slug for q in snap] == ["b", "c", "d"]

    def test_expired_events_evicted_on_pipeline_complete(self):
        """on_pipeline_complete で max_age_sec 超の event が破棄される。"""
        d = Dispatcher(max_events=10, max_age_sec=0.05)
        d.on_wake_detected(_make_wake("old"))
        time.sleep(0.1)  # max_age_sec 超
        d.on_wake_detected(_make_wake("fresh"))
        # Note: 上の on_wake_detected 内でも _evict_expired_unlocked が呼ばれるため
        # この時点で old は既に evict されているはず
        snap = d.get_queue_snapshot()
        slugs = [q.event.character_slug for q in snap]
        assert "old" not in slugs
        assert "fresh" in slugs

    def test_expired_events_evicted_on_wait(self):
        """wait_for_next_event 内でも期限切れが除去される。"""
        d = Dispatcher(max_events=10, max_age_sec=0.05)
        d.on_wake_detected(_make_wake("old"))
        time.sleep(0.1)
        # queue に old があるが期限切れ → wait_for_next_event は timeout で None
        result = d.wait_for_next_event(timeout=0.2)
        assert result is None
        snap = d.get_queue_snapshot()
        assert snap == []

    def test_fresh_events_preserved_after_pipeline_complete(self):
        """期限切れていない event は on_pipeline_complete 後も残る。"""
        d = Dispatcher(max_events=10, max_age_sec=10.0)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_wake_detected(_make_wake("fresh"))
        d.on_pipeline_complete()
        snap = d.get_queue_snapshot()
        assert len(snap) == 1
        assert snap[0].event.character_slug == "fresh"


# ─── TestDispatcherPublishCallback ────────────────────────────────


class TestDispatcherPublishCallback:
    def test_callback_fires_on_enqueue(self):
        calls: list[list[QueuedWakeEvent]] = []
        d = Dispatcher(on_queue_update=lambda q: calls.append(q))
        d.on_wake_detected(_make_wake("mimi"))
        assert len(calls) == 1
        assert len(calls[0]) == 1
        assert calls[0][0].event.character_slug == "mimi"

    def test_callback_fires_on_dequeue(self):
        calls: list[list[QueuedWakeEvent]] = []
        d = Dispatcher(on_queue_update=lambda q: calls.append(q))
        d.on_wake_detected(_make_wake("mimi"))  # 1 回目: enqueue
        calls.clear()
        d.wait_for_next_event(timeout=1.0)  # 2 回目: dequeue
        assert len(calls) == 1
        assert len(calls[0]) == 0  # dequeue 後は queue 空

    def test_callback_fires_on_overflow_eviction(self):
        calls: list[list[QueuedWakeEvent]] = []
        d = Dispatcher(max_events=2, on_queue_update=lambda q: calls.append(q))
        d.on_wake_detected(_make_wake("a"))
        d.on_wake_detected(_make_wake("b"))
        d.on_wake_detected(_make_wake("c"))  # a が evict されつつ c が enqueue
        # 各 add 後に publish される
        assert len(calls) == 3
        # 最後の publish 時点では a が消えて [b, c]
        assert [q.event.character_slug for q in calls[-1]] == ["b", "c"]

    def test_callback_exception_does_not_break_dispatcher(self):
        """callback で例外が出ても dispatcher は動き続ける。"""
        def bad_callback(q):
            raise RuntimeError("boom")

        d = Dispatcher(on_queue_update=bad_callback)
        # 例外 raise されない
        d.on_wake_detected(_make_wake("mimi"))
        # queue には正常に積まれている
        snap = d.get_queue_snapshot()
        assert len(snap) == 1

    def test_no_callback_when_not_set(self):
        """on_queue_update=None なら callback は呼ばれない (例外なく動く)。"""
        d = Dispatcher(on_queue_update=None)
        d.on_wake_detected(_make_wake("mimi"))
        # 例外なく動作すれば OK


# ─── TestDispatcherEndToEnd ───────────────────────────────────────


class TestDispatcherEndToEnd:
    """Block 0 のフロー全体: IDLE → RESPONDING → IDLE → RESPONDING ..."""

    def test_full_cycle(self):
        d = Dispatcher()

        # 1. 最初の event
        d.on_wake_detected(_make_wake("mimi"))
        assert d.get_state() == DispatcherState.IDLE

        # 2. メインスレッドが取り出して RESPONDING へ
        e1 = d.wait_for_next_event(timeout=1.0)
        assert e1 is not None and e1.character_slug == "mimi"
        assert d.get_state() == DispatcherState.RESPONDING

        # 3. RESPONDING 中に別 event が来て queue に積まれる
        d.on_wake_detected(_make_wake("chisame"))
        assert len(d.get_queue_snapshot()) == 1
        assert d.get_state() == DispatcherState.RESPONDING  # まだ応答中

        # 4. pipeline 完了 → IDLE
        d.on_pipeline_complete()
        assert d.get_state() == DispatcherState.IDLE

        # 5. queue から chisame を取り出す
        e2 = d.wait_for_next_event(timeout=1.0)
        assert e2 is not None and e2.character_slug == "chisame"
        assert d.get_state() == DispatcherState.RESPONDING

        # 6. queue 空状態で pipeline 完了
        d.on_pipeline_complete()
        assert d.get_state() == DispatcherState.IDLE
        assert d.get_queue_snapshot() == []


# ─── TestDispatcherPhase05APIReserved ─────────────────────────────


class TestDispatcherPhase05APIReserved:
    """Phase 0.5 用 API が NotImplementedError を raise することを確認する。

    Block 0 ではこれらのメソッドは「予約のみ」で、Phase 0.5 着手時に中身を実装する。
    シグネチャを変えずに raise を解除するだけで完成する状態を維持する。
    """

    def test_on_interjection_candidate_raises(self):
        d = Dispatcher()
        with pytest.raises(NotImplementedError):
            d.on_interjection_candidate("mimi", transcript_snapshot=None)

    def test_on_approval_granted_raises(self):
        d = Dispatcher()
        with pytest.raises(NotImplementedError):
            d.on_approval_granted("mimi")

    def test_on_approval_denied_raises(self):
        d = Dispatcher()
        with pytest.raises(NotImplementedError):
            d.on_approval_denied("mimi")

    def test_on_lapse_timeout_raises(self):
        d = Dispatcher()
        with pytest.raises(NotImplementedError):
            d.on_lapse_timeout("mimi")

    def test_handraising_state_value_exists(self):
        """HANDRAISING 状態値が enum に存在する (Block 0 では遷移しない)。"""
        assert DispatcherState.HANDRAISING.value == "handraising"
