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
    CooldownState,
    Dispatcher,
    DispatcherState,
    HandraiseState,
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
        """Phase 0.5 用 dict は空で予約されている。"""
        d = Dispatcher()
        assert d._handraise_states == {}
        assert d._cooldowns == {}

    def test_default_callbacks_are_none(self):
        """Phase 0.5-A 新規 callback (on_handraise_update / on_bubble_update) は default None。"""
        d = Dispatcher()
        assert d._on_handraise_update is None
        assert d._on_bubble_update is None

    def test_default_use_handraise_is_true(self, monkeypatch):
        """L2_USE_HANDRAISE 未設定時は True (default)。"""
        monkeypatch.delenv("L2_USE_HANDRAISE", raising=False)
        d = Dispatcher()
        assert d._use_handraise is True

    def test_default_lapse_sec_is_300(self, monkeypatch):
        """L2_HANDRAISE_LAPSE_SEC 未設定時は 300 (= 5 分)。"""
        monkeypatch.delenv("L2_HANDRAISE_LAPSE_SEC", raising=False)
        d = Dispatcher()
        assert d._lapse_sec == 300.0

    def test_default_lapse_utterance_count_is_8(self, monkeypatch):
        """L2_HANDRAISE_LAPSE_UTTERANCE_COUNT 未設定時は 8。"""
        monkeypatch.delenv("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", raising=False)
        d = Dispatcher()
        assert d._lapse_utterance_count == 8


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


# ─── TestHandraiseStateInit (Phase 0.5-A で拡張) ──────────────────


class TestHandraiseStateInit:
    """HandraiseState dataclass 単独の挙動 (Phase 0.5-A で拡張)。

    フェーズ 5a で target_slug + started_at から実用フィールド一式に拡張された。
    フェーズ 7 で bg_thread / phrase_path 再生の実体が組み込まれる。
    """

    def test_can_create_minimal(self):
        """target_slug + started_at だけで生成できる (default 値が完全)。"""
        state = HandraiseState(target_slug="mimi", started_at=0.0)
        assert state.target_slug == "mimi"
        assert state.started_at == 0.0

    def test_default_values(self):
        """default フィールドの値を確認 (フェーズ 5b の placeholder と整合)。"""
        state = HandraiseState(target_slug="mimi", started_at=0.0)
        assert state.transcript_snapshot is None
        assert state.bg_thread is None
        assert state.bg_result is None
        assert state.phrase == ""
        assert state.phrase_path is None
        assert state.se_pending is False
        assert state.lapse_timer is None
        assert state.utterance_count_since == 0
        assert state.trace_id == ""

    def test_cancel_event_independent_per_instance(self):
        """cancel_event は default_factory で生成されるので、インスタンスごとに独立。"""
        s1 = HandraiseState(target_slug="mimi", started_at=0.0)
        s2 = HandraiseState(target_slug="chisame", started_at=0.0)
        s1.cancel_event.set()
        # s1 を set しても s2 は影響を受けない
        assert s1.cancel_event.is_set()
        assert not s2.cancel_event.is_set()

    def test_bg_completed_independent_per_instance(self):
        """bg_completed も独立 (default_factory パターン)。"""
        s1 = HandraiseState(target_slug="mimi", started_at=0.0)
        s2 = HandraiseState(target_slug="chisame", started_at=0.0)
        s1.bg_completed.set()
        assert s1.bg_completed.is_set()
        assert not s2.bg_completed.is_set()


# ─── TestCooldownStateInit (Phase 0.5-A で導入) ───────────────────


class TestCooldownStateInit:
    """CooldownState dataclass 単独の挙動 (Phase 0.5-A で導入)。

    Phase 0.5-A は consecutive_denials のみ増減する。
    threshold_multiplier の適用は Phase 0.5-B 以降。
    """

    def test_default_values(self):
        """default 値: cooldown_until=0.0 / consecutive_denials=0 / threshold_multiplier=1.0。"""
        cd = CooldownState()
        assert cd.cooldown_until == 0.0
        assert cd.consecutive_denials == 0
        assert cd.threshold_multiplier == 1.0

    def test_consecutive_denials_increment(self):
        """consecutive_denials を直接インクリメントできる (Phase 0.5-A は dataclass 直接更新)。"""
        cd = CooldownState()
        cd.consecutive_denials += 1
        assert cd.consecutive_denials == 1
        cd.consecutive_denials += 1
        assert cd.consecutive_denials == 2

    def test_threshold_multiplier_phase_a_is_one(self):
        """Phase 0.5-A は threshold_multiplier が常に 1.0 固定で発行される設計。

        この値は CooldownState 単体ではユーザーが書き換えられるが、
        Dispatcher の API (on_approval_denied 等) が 1.0 で初期化する。
        """
        cd = CooldownState()
        assert cd.threshold_multiplier == 1.0

    def test_can_initialize_with_values(self):
        """initkw で全フィールドを指定して生成できる (将来の Phase 0.5-B で利用)。"""
        cd = CooldownState(
            cooldown_until=100.0,
            consecutive_denials=3,
            threshold_multiplier=1.5,
        )
        assert cd.cooldown_until == 100.0
        assert cd.consecutive_denials == 3
        assert cd.threshold_multiplier == 1.5


# ─── TestDispatcherEnvVarOverride (Phase 0.5-A で導入) ────────────


class TestDispatcherEnvVarOverride:
    """環境変数で挙手機能設定が上書きできることを確認 (Phase 0.5-A で導入)。

    L2_USE_HANDRAISE / L2_HANDRAISE_LAPSE_SEC / L2_HANDRAISE_LAPSE_UTTERANCE_COUNT。
    各 Dispatcher インスタンスは生成時に環境変数を 1 回読み込んで保持する。
    """

    def test_use_handraise_false_via_env(self, monkeypatch):
        """L2_USE_HANDRAISE=false で機能 off。"""
        monkeypatch.setenv("L2_USE_HANDRAISE", "false")
        d = Dispatcher()
        assert d._use_handraise is False

    def test_use_handraise_accepts_truthy_values(self, monkeypatch):
        """L2_USE_HANDRAISE は 1 / true / yes を真として受け入れる。"""
        for val in ("1", "true", "TRUE", "yes", "YES"):
            monkeypatch.setenv("L2_USE_HANDRAISE", val)
            d = Dispatcher()
            assert d._use_handraise is True

    def test_lapse_sec_override_via_env(self, monkeypatch):
        """L2_HANDRAISE_LAPSE_SEC で時間 lapse 閾値を上書き。"""
        monkeypatch.setenv("L2_HANDRAISE_LAPSE_SEC", "60")
        d = Dispatcher()
        assert d._lapse_sec == 60.0

    def test_lapse_utterance_count_override_via_env(self, monkeypatch):
        """L2_HANDRAISE_LAPSE_UTTERANCE_COUNT で utterance lapse 閾値を上書き。"""
        monkeypatch.setenv("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", "3")
        d = Dispatcher()
        assert d._lapse_utterance_count == 3


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
