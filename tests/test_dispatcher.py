"""
test_dispatcher.py — Dispatcher のテスト (Block 0 + Phase 0.5-A)

責務:
  - 状態遷移 (IDLE / RESPONDING / HANDRAISING) の検証
  - wake_event_queue の add / dequeue / drain ルール検証
  - 60 秒期限切れ + 上限 3 件のドレイン戦略検証
  - publish callback の発火タイミング検証 (queue / handraise / bubble)
  - Phase 0.5-A 挙手フロー API (on_segment_added / on_interjection_candidate /
    on_approval_granted / on_approval_denied / on_lapse_timeout) の検証

外部依存 (sounddevice / STT / LLM) はなく、純粋な状態機械のテスト。
threading.Timer は _FakeTimer に差し替え、決定論的に発火させる。
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

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
from lab_lounge.router import ApprovalResult, IntentResult
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

    def test_handraising_state_value_exists(self):
        """HANDRAISING 状態値が enum に存在する (Phase 0.5-A 用に予約済)。"""
        assert DispatcherState.HANDRAISING.value == "handraising"


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


# ─── Phase 0.5-A 用テストヘルパー ──────────────────────────────────


class _FakeTimer:
    """threading.Timer の差し替え用 (Phase 0.5-A)。

    実時間で待たずに ``.fire()`` で手動発火することで、テストを決定論的にする。
    ``.start()`` / ``.cancel()`` は no-op (フラグのみ)。
    """

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        """テストから手動発火 (Timer タイムアウトと同等)。"""
        self.function(*self.args, **self.kwargs)


def _patch_filler(monkeypatch, *, slug="mimi", text="挙手します") -> None:
    """``filler.select_filler_phrase`` を固定値返却に差し替える。

    dispatcher._start_handraise の関数内 import 経由でも本物が差し替わるよう、
    ``lab_lounge.filler.select_filler_phrase`` 自体を mock する。
    """
    fake_path = Path(f"/tmp/{slug}_handraise.wav")
    fake_phrase = MagicMock()
    fake_phrase.text = text
    monkeypatch.setattr(
        "lab_lounge.filler.select_filler_phrase",
        lambda slug, category="opener", **kw: (fake_path, fake_phrase, 0),
    )


def _patch_lapse_timer(monkeypatch, dispatcher: Dispatcher) -> list[_FakeTimer]:
    """Dispatcher._create_lapse_timer を _FakeTimer 返却に差し替える。

    返り値の list には生成された _FakeTimer が順に追加される。
    テストは ``timers[0].fire()`` で lapse を発火できる。
    """
    timers: list[_FakeTimer] = []

    def fake_create(target_slug: str, delay_sec: float) -> _FakeTimer:
        t = _FakeTimer(
            delay_sec,
            dispatcher.on_lapse_timeout,
            args=[target_slug],
        )
        timers.append(t)
        return t

    monkeypatch.setattr(dispatcher, "_create_lapse_timer", fake_create)
    return timers


# ─── TestDispatcherInterjectionFlow (Phase 0.5-A) ─────────────────


class TestDispatcherInterjectionFlow:
    """on_interjection_candidate → _start_handraise の挙動 (Phase 0.5-A)。"""

    def test_creates_handraise_state(self, monkeypatch):
        """on_interjection_candidate で _handraise_states に slug が登録される。"""
        _patch_filler(monkeypatch, slug="mimi", text="ねぇ")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="test text")
        assert "mimi" in d._handraise_states
        assert d._handraise_states["mimi"].target_slug == "mimi"
        assert d._handraise_states["mimi"].phrase == "ねぇ"
        assert d._handraise_states["mimi"].transcript_snapshot == "test text"

    def test_idempotent_when_slug_already_handraising(self, monkeypatch):
        """既に handraising 中の slug は冪等 (二重登録しない)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t1")
        first_state = d._handraise_states["mimi"]
        d.on_interjection_candidate("mimi", transcript_snapshot="t2")  # 2 回目
        # 同じインスタンスが残っている (transcript_snapshot は更新されない)
        assert d._handraise_states["mimi"] is first_state

    def test_no_op_when_use_handraise_false(self, monkeypatch):
        """L2_USE_HANDRAISE=false 時は no-op。"""
        monkeypatch.setenv("L2_USE_HANDRAISE", "false")
        d = Dispatcher()
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states == {}

    def test_lapse_timer_started(self, monkeypatch):
        """lapse_timer が start() 済みになる。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert len(timers) == 1
        assert timers[0].started is True
        assert timers[0].interval == d._lapse_sec  # default 300

    def test_se_pending_in_responding(self, monkeypatch):
        """RESPONDING 中の挙手では se_pending=True (フェーズ 7 で SE 遅延発火)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].se_pending is True

    def test_se_pending_in_idle(self, monkeypatch):
        """IDLE 中の挙手では se_pending=False (フェーズ 7 で SE 即時発火)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].se_pending is False

    def test_trace_id_is_uuid(self, monkeypatch):
        """trace_id が UUID4 形式で割当される。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        trace_id = d._handraise_states["mimi"].trace_id
        assert trace_id != ""
        assert len(trace_id) == 36  # UUID4 標準形式

    def test_bg_completed_set_in_phase_5(self, monkeypatch):
        """フェーズ 5 では bg_completed が即座に set されている (no-op placeholder)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].bg_completed.is_set()
        assert d._handraise_states["mimi"].bg_thread is None  # フェーズ 7 で実体


# ─── TestDispatcherApprovalFlow (Phase 0.5-A) ─────────────────────


class TestDispatcherApprovalFlow:
    """on_approval_granted / on_approval_denied の挙動 (Phase 0.5-A)。"""

    def _setup_handraise(
        self, monkeypatch, d: Dispatcher, *, slug: str = "mimi",
    ) -> list[_FakeTimer]:
        """ヘルパー: handraising 状態を作る。bubble_messages も mock。"""
        _patch_filler(monkeypatch, slug=slug)
        timers = _patch_lapse_timer(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {slug: {"denied": "また今度", "lapsed": "静かに"}},
        )
        d.on_interjection_candidate(slug, transcript_snapshot="t")
        return timers

    def test_granted_removes_state(self, monkeypatch):
        d = Dispatcher()
        self._setup_handraise(monkeypatch, d)
        assert "mimi" in d._handraise_states
        d.on_approval_granted("mimi")
        assert d._handraise_states == {}

    def test_granted_resets_consecutive_denials(self, monkeypatch):
        """承認は consecutive_denials を 0 にリセット。"""
        d = Dispatcher()
        # 既に denial カウントあり
        d._cooldowns["mimi"] = CooldownState(consecutive_denials=2)
        self._setup_handraise(monkeypatch, d)
        d.on_approval_granted("mimi")
        assert d._cooldowns["mimi"].consecutive_denials == 0

    def test_granted_cancels_lapse_timer(self, monkeypatch):
        d = Dispatcher()
        timers = self._setup_handraise(monkeypatch, d)
        d.on_approval_granted("mimi")
        assert timers[0].cancelled is True

    def test_granted_sets_cancel_event(self, monkeypatch):
        """承認時に cancel_event を set (BG LLM ベストエフォート cleanup)。"""
        d = Dispatcher()
        self._setup_handraise(monkeypatch, d)
        # 状態取得 (削除前にリファレンスを保持)
        state = d._handraise_states["mimi"]
        d.on_approval_granted("mimi")
        assert state.cancel_event.is_set()

    def test_granted_idempotent_on_unknown_slug(self, monkeypatch):
        d = Dispatcher()
        d.on_approval_granted("nonexistent")  # 例外なく動く
        assert d._handraise_states == {}

    def test_denied_removes_state(self, monkeypatch):
        d = Dispatcher()
        self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert d._handraise_states == {}

    def test_denied_increments_consecutive_denials(self, monkeypatch):
        d = Dispatcher()
        self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert "mimi" in d._cooldowns
        assert d._cooldowns["mimi"].consecutive_denials == 1
        # 2 回目の denial で +1 (再 handraise → denied のサイクル)
        self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert d._cooldowns["mimi"].consecutive_denials == 2

    def test_denied_threshold_multiplier_remains_1(self, monkeypatch):
        """Phase 0.5-A は threshold_multiplier=1.0 固定。Phase 0.5-B で変動。"""
        d = Dispatcher()
        self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert d._cooldowns["mimi"].threshold_multiplier == 1.0

    def test_denied_publishes_denied_bubble(self, monkeypatch):
        bubble_calls = []
        d = Dispatcher(
            on_bubble_update=lambda *a: bubble_calls.append(a),
        )
        self._setup_handraise(monkeypatch, d)
        bubble_calls.clear()  # handraise bubble を捨てて denied だけ確認
        d.on_approval_denied("mimi")
        assert len(bubble_calls) == 1
        # Phase 0.5-A 8-10: callback シグネチャは (character, step, text, ttl_ms, category)
        char, step, text, ttl_ms, category = bubble_calls[0]
        assert char == "mimi"
        assert step == "denied"
        assert text == "また今度"  # mock した bubble_messages から
        assert ttl_ms == 2000
        assert category == "handraise"

    def test_denied_cancels_lapse_timer(self, monkeypatch):
        d = Dispatcher()
        timers = self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert timers[0].cancelled is True

    def test_denied_idempotent_on_unknown_slug(self, monkeypatch):
        d = Dispatcher()
        d.on_approval_denied("nonexistent")
        assert d._cooldowns == {}  # 冪等: cooldown も加算されない

    def test_no_op_when_use_handraise_false(self, monkeypatch):
        """L2_USE_HANDRAISE=false 時は state を直接 set しても API は no-op。"""
        monkeypatch.setenv("L2_USE_HANDRAISE", "false")
        d = Dispatcher()
        d._handraise_states["mimi"] = HandraiseState(
            target_slug="mimi", started_at=0.0,
        )
        d.on_approval_denied("mimi")
        assert "mimi" in d._handraise_states  # 機能 off なら何もしない


# ─── TestDispatcherLapse (Phase 0.5-A) ────────────────────────────


class TestDispatcherLapse:
    """lapse_timer 発火 → on_lapse_timeout の挙動 (Phase 0.5-A)。"""

    def test_lapse_removes_state(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"lapsed": "静かに"}},
        )
        d = Dispatcher()
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        timers[0].fire()  # FakeTimer を手動発火
        assert d._handraise_states == {}

    def test_lapse_does_not_increment_denials(self, monkeypatch):
        """lapse は cooldown.consecutive_denials を変えない (denial と異なる扱い)。"""
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        d = Dispatcher()
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        timers[0].fire()
        assert d._cooldowns == {}

    def test_lapse_publishes_lapsed_bubble(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"lapsed": "静かに"}},
        )
        bubble_calls = []
        d = Dispatcher(on_bubble_update=lambda *a: bubble_calls.append(a))
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        bubble_calls.clear()
        timers[0].fire()
        assert len(bubble_calls) == 1
        # Phase 0.5-A 8-10: callback シグネチャは (character, step, text, ttl_ms, category)
        char, step, text, ttl_ms, category = bubble_calls[0]
        assert char == "mimi"
        assert step == "lapsed"
        assert text == "静かに"
        assert ttl_ms == 2000
        assert category == "handraise"

    def test_lapse_idempotent_on_unknown_slug(self, monkeypatch):
        d = Dispatcher()
        d.on_lapse_timeout("nonexistent")  # 例外なく動く

    def test_lapse_uses_default_text_for_octamaid(self, monkeypatch):
        """bubble_messages に該当キャラがない場合は default フォールバック。"""
        _patch_filler(monkeypatch, slug="mimi")
        # 空の messages で default にフォールバック
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        bubble_calls = []
        d = Dispatcher(on_bubble_update=lambda *a: bubble_calls.append(a))
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        bubble_calls.clear()
        timers[0].fire()
        # default フォールバック text が使われる
        assert "(静まりました)" == bubble_calls[0][2]

    def test_no_op_when_use_handraise_false(self, monkeypatch):
        monkeypatch.setenv("L2_USE_HANDRAISE", "false")
        d = Dispatcher()
        d._handraise_states["mimi"] = HandraiseState(
            target_slug="mimi", started_at=0.0,
        )
        d.on_lapse_timeout("mimi")
        assert "mimi" in d._handraise_states


# ─── TestDispatcherSegmentDispatch (Phase 0.5-A) ──────────────────


class TestDispatcherSegmentDispatch:
    """on_segment_added の状態分岐 (Phase 0.5-A)。

    handraising キャラの有無で check_approval / check_intent を呼び分ける。
    """

    def test_no_op_when_use_handraise_false(self, monkeypatch):
        monkeypatch.setenv("L2_USE_HANDRAISE", "false")
        d = Dispatcher()
        check_intent_mock = MagicMock()
        check_approval_mock = MagicMock()
        monkeypatch.setattr("lab_lounge.router.check_intent", check_intent_mock)
        monkeypatch.setattr("lab_lounge.router.check_approval", check_approval_mock)
        d.on_segment_added(MagicMock(), "test text")
        # 機能 off なら LLM 呼び出しすらしない
        check_intent_mock.assert_not_called()
        check_approval_mock.assert_not_called()

    def test_idle_calls_check_intent_for_interjection(self, monkeypatch):
        """handraising キャラ無し時、check_intent (interjection_candidate モード) を呼ぶ。"""
        d = Dispatcher()
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="unknown", target_slug=None, confidence=0.0,
            ),
        )
        check_approval_mock = MagicMock()
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", check_approval_mock,
        )
        d.on_segment_added(MagicMock(), "test text")
        # check_approval は handraising キャラ無しで呼ばない
        check_approval_mock.assert_not_called()

    def test_handraising_calls_check_approval_first(self, monkeypatch):
        """handraising キャラあり時、check_approval を先に呼ぶ。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # check_approval が None を返す (関係ない発話)
        check_approval_mock = MagicMock(return_value=None)
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", check_approval_mock,
        )
        check_intent_mock = MagicMock(
            return_value=IntentResult(
                intent="unknown", target_slug=None, confidence=0.0,
            ),
        )
        monkeypatch.setattr(
            "lab_lounge.router.check_intent", check_intent_mock,
        )
        d.on_segment_added(MagicMock(), "テスト")
        # check_approval が先に呼ばれる
        check_approval_mock.assert_called_once_with("テスト", ["mimi"])

    def test_segment_triggers_interjection_when_candidate(self, monkeypatch):
        """interjection_candidate → on_interjection_candidate が呼ばれる。"""
        _patch_filler(monkeypatch, slug="mimi", text="挙手")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="interjection_candidate",
                target_slug="mimi",
                confidence=1.0,
            ),
        )
        d.on_segment_added(MagicMock(), "AI 倫理について興味がある")
        assert "mimi" in d._handraise_states

    def test_approval_granted_via_segment(self, monkeypatch):
        """check_approval が granted を返したら on_approval_granted が呼ばれる。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        monkeypatch.setattr(
            "lab_lounge.router.check_approval",
            lambda text, slugs: ApprovalResult(
                granted=True, target_slug="mimi", confidence=1.0,
            ),
        )
        d.on_segment_added(MagicMock(), "ミミ、どうぞ")
        assert d._handraise_states == {}

    def test_approval_denied_via_segment(self, monkeypatch):
        """check_approval が denied を返したら on_approval_denied が呼ばれる。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        monkeypatch.setattr(
            "lab_lounge.router.check_approval",
            lambda text, slugs: ApprovalResult(
                granted=False, target_slug="mimi", confidence=1.0,
            ),
        )
        d.on_segment_added(MagicMock(), "いや、いいわ")
        assert d._handraise_states == {}
        assert d._cooldowns["mimi"].consecutive_denials == 1

    def test_utterance_count_increments_on_unrelated(self, monkeypatch):
        """check_approval が None で utterance_count_since が +1 (lapse 判定用)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", lambda text, slugs: None,
        )
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="unknown", target_slug=None, confidence=0.0,
            ),
        )
        d.on_segment_added(MagicMock(), "今日はいい天気だね")
        assert d._handraise_states["mimi"].utterance_count_since == 1
        d.on_segment_added(MagicMock(), "明日も晴れるかな")
        assert d._handraise_states["mimi"].utterance_count_since == 2

    def test_utterance_count_threshold_triggers_lapse(self, monkeypatch):
        """utterance_count_since が閾値超えると自動 lapse。"""
        monkeypatch.setenv("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", "2")
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", lambda text, slugs: None,
        )
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="unknown", target_slug=None, confidence=0.0,
            ),
        )
        d.on_segment_added(MagicMock(), "発話 1")
        d.on_segment_added(MagicMock(), "発話 2")  # 閾値到達 → 自動 lapse
        assert d._handraise_states == {}


# ─── TestDispatcherHandraisePublish (Phase 0.5-A) ─────────────────


class TestDispatcherHandraisePublish:
    """on_handraise_update / on_bubble_update callback の発火タイミング (Phase 0.5-A)。

    既存 TestDispatcherPublishCallback パターンを踏襲。callback は Lock 外で呼ばれ、
    例外を投げても dispatcher 本体は止まらない。
    """

    def test_handraise_callback_fires_on_start(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(
            on_handraise_update=lambda states, cd: calls.append(
                (dict(states), dict(cd)),
            ),
        )
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert len(calls) == 1
        states, cooldowns = calls[0]
        assert "mimi" in states
        assert cooldowns == {}

    def test_handraise_callback_fires_on_grant(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        calls = []
        d = Dispatcher(
            on_handraise_update=lambda states, cd: calls.append(
                (dict(states), dict(cd)),
            ),
        )
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_approval_granted("mimi")
        # start + grant で 2 回呼ばれる
        assert len(calls) == 2
        # grant 後は state 空
        assert calls[1][0] == {}

    def test_handraise_callback_fires_on_deny(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        calls = []
        d = Dispatcher(
            on_handraise_update=lambda states, cd: calls.append(
                (dict(states), dict(cd)),
            ),
        )
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_approval_denied("mimi")
        assert len(calls) == 2
        # deny 後は state 空、cooldown に mimi
        assert calls[1][0] == {}
        assert "mimi" in calls[1][1]
        assert calls[1][1]["mimi"].consecutive_denials == 1

    def test_handraise_callback_fires_on_lapse(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        calls = []
        d = Dispatcher(
            on_handraise_update=lambda states, cd: calls.append(
                (dict(states), dict(cd)),
            ),
        )
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        timers[0].fire()
        assert len(calls) == 2
        assert calls[1][0] == {}

    def test_bubble_callback_fires_on_handraise_step(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi", text="挙手")
        bubble_calls = []
        d = Dispatcher(on_bubble_update=lambda *a: bubble_calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert len(bubble_calls) == 1
        # Phase 0.5-A 8-10: callback シグネチャは (character, step, text, ttl_ms, category)
        char, step, text, ttl_ms, category = bubble_calls[0]
        assert char == "mimi"
        assert step == "handraise"
        assert text == "挙手"
        assert ttl_ms is None  # handraise は ttl_ms=None で永続表示
        assert category == "handraise"

    def test_bubble_callback_fires_on_denied_step(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"denied": "また今度"}},
        )
        bubble_calls = []
        d = Dispatcher(on_bubble_update=lambda *a: bubble_calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        bubble_calls.clear()
        d.on_approval_denied("mimi")
        assert len(bubble_calls) == 1
        # Phase 0.5-A 8-10: tuple は (character, step, text, ttl_ms, category)
        assert bubble_calls[0] == ("mimi", "denied", "また今度", 2000, "handraise")

    def test_bubble_callback_fires_on_lapsed_step(self, monkeypatch):
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"lapsed": "静かに"}},
        )
        bubble_calls = []
        d = Dispatcher(on_bubble_update=lambda *a: bubble_calls.append(a))
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        bubble_calls.clear()
        timers[0].fire()
        assert len(bubble_calls) == 1
        # Phase 0.5-A 8-10: tuple は (character, step, text, ttl_ms, category)
        assert bubble_calls[0] == ("mimi", "lapsed", "静かに", 2000, "handraise")

    def test_no_callback_when_not_set(self, monkeypatch):
        """callback 未設定 (None) でも例外なく動く。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()  # callback いずれも None
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # 例外なく完了

    def test_callback_exception_does_not_propagate(self, monkeypatch):
        """callback で例外が出ても dispatcher は止まらない。"""
        _patch_filler(monkeypatch, slug="mimi")

        def raise_on_call(*args, **kwargs):
            raise RuntimeError("test exception")

        d = Dispatcher(
            on_handraise_update=raise_on_call,
            on_bubble_update=raise_on_call,
        )
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # 例外は warning ログに留まり、state は登録される
        assert "mimi" in d._handraise_states


# ─── Phase 0.5-A フェーズ 7 (BG LLM 実体組込) ──────────────────────


class TestHandraiseBgResult:
    """Phase 0.5-A フェーズ 7: HandraiseBgResult dataclass の挙動。"""

    def test_default_values(self):
        from lab_lounge.dispatcher import HandraiseBgResult
        r = HandraiseBgResult()
        assert r.chunks == []
        assert r.result is None
        assert r.trace_id == ""

    def test_with_explicit_values(self):
        from lab_lounge.dispatcher import HandraiseBgResult
        sentinel_result = MagicMock()
        r = HandraiseBgResult(
            chunks=[{"url": "x", "text": "a", "is_last": True, "character": "mimi"}],
            result=sentinel_result,
            trace_id="abc",
        )
        assert len(r.chunks) == 1
        assert r.chunks[0]["text"] == "a"
        assert r.result is sentinel_result
        assert r.trace_id == "abc"

    def test_chunks_default_factory_independent(self):
        """default_factory なので mutable default の罠を踏まない。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        r1 = HandraiseBgResult()
        r2 = HandraiseBgResult()
        r1.chunks.append({"url": "x"})
        assert r2.chunks == []  # 独立


class TestDispatcherBgRunner:
    """Phase 0.5-A フェーズ 7: bg_runner 起動と _bg_set_result の挙動。"""

    def _setup(self, monkeypatch, d):
        _patch_filler(monkeypatch, slug="mimi")
        _patch_lapse_timer(monkeypatch, d)

    def test_bg_runner_called_with_kwargs(self, monkeypatch):
        """_start_handraise が bg_runner を必要 kwargs で呼ぶ。"""
        bg_runner_calls = []
        fake_thread = MagicMock(spec=threading.Thread)

        def fake_runner(**kwargs):
            bg_runner_calls.append(kwargs)
            return fake_thread

        d = Dispatcher(bg_runner=fake_runner)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="snap")

        assert len(bg_runner_calls) == 1
        kwargs = bg_runner_calls[0]
        assert kwargs["target_slug"] == "mimi"
        assert kwargs["transcript_snapshot"] == "snap"
        assert isinstance(kwargs["cancel_event"], threading.Event)
        assert callable(kwargs["on_complete"])

    def test_bg_runner_thread_attached_to_state(self, monkeypatch):
        """bg_runner が返した thread が state.bg_thread に格納される。"""
        fake_thread = MagicMock(spec=threading.Thread)
        d = Dispatcher(bg_runner=lambda **kw: fake_thread)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].bg_thread is fake_thread

    def test_bg_completed_not_set_when_bg_runner_provided(self, monkeypatch):
        """bg_runner 注入時、bg_completed はまだ set されていない (BG が完了通知まで)。"""
        fake_thread = MagicMock(spec=threading.Thread)
        d = Dispatcher(bg_runner=lambda **kw: fake_thread)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert not d._handraise_states["mimi"].bg_completed.is_set()

    def test_bg_completed_set_when_bg_runner_none(self, monkeypatch):
        """bg_runner=None のデフォルトでは bg_completed が即時 set (フォールバック)。"""
        d = Dispatcher()  # bg_runner=None
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].bg_completed.is_set()
        assert d._handraise_states["mimi"].bg_thread is None

    def test_bg_runner_exception_falls_back(self, monkeypatch):
        """bg_runner が例外時は bg_completed.set() でフォールバック。state は残る。"""
        def failing_runner(**kw):
            raise RuntimeError("boom")
        d = Dispatcher(bg_runner=failing_runner)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert "mimi" in d._handraise_states
        assert d._handraise_states["mimi"].bg_completed.is_set()
        assert d._handraise_states["mimi"].bg_thread is None

    def test_bg_set_result_stores_result(self, monkeypatch):
        """_bg_set_result が bg_result + bg_completed をセットする。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        fake_thread = MagicMock(spec=threading.Thread)
        d = Dispatcher(bg_runner=lambda **kw: fake_thread)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert not d._handraise_states["mimi"].bg_completed.is_set()
        result = HandraiseBgResult(chunks=[{"x": 1}], trace_id="abc")
        d._bg_set_result("mimi", result)
        assert d._handraise_states["mimi"].bg_result is result
        assert d._handraise_states["mimi"].bg_completed.is_set()

    def test_bg_set_result_idempotent_after_state_removed(self, monkeypatch):
        """state 削除後の _bg_set_result は冪等で no-op (例外なし)。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        fake_thread = MagicMock(spec=threading.Thread)
        d = Dispatcher(bg_runner=lambda **kw: fake_thread)
        self._setup(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"denied": "x", "lapsed": "y"}},
        )
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_approval_denied("mimi")  # state 削除
        # 削除後の _bg_set_result は no-op (例外なし)
        d._bg_set_result("mimi", HandraiseBgResult())


class TestDispatcherHandraiseStartedCallback:
    """Phase 0.5-A フェーズ 7: on_handraise_started callback の発火検証。"""

    def test_callback_fires_on_idle_with_se_pending_false(self, monkeypatch):
        """IDLE 中の挙手で se_pending=False、callback の引数も False。"""
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(on_handraise_started=lambda *a: calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert len(calls) == 1
        slug, path, se_pending = calls[0]
        assert slug == "mimi"
        assert path is not None
        assert se_pending is False

    def test_callback_fires_on_responding_with_se_pending_true(self, monkeypatch):
        """RESPONDING 中の挙手で se_pending=True、callback の引数も True。"""
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(on_handraise_started=lambda *a: calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert len(calls) == 1
        slug, path, se_pending = calls[0]
        assert slug == "mimi"
        assert se_pending is True

    def test_no_callback_when_not_set(self, monkeypatch):
        """on_handraise_started 未設定でも例外なく動く。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher()  # callback 未注入
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert "mimi" in d._handraise_states

    def test_callback_exception_does_not_propagate(self, monkeypatch):
        """callback 内例外が dispatcher 外に伝播しない (warning ログのみ)。"""
        _patch_filler(monkeypatch, slug="mimi")

        def failing_callback(*args):
            raise RuntimeError("boom")

        d = Dispatcher(on_handraise_started=failing_callback)
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert "mimi" in d._handraise_states


class TestDispatcherPhrasePendingRelease:
    """Phase 0.5-A フェーズ 7: on_pipeline_complete 内の保留 wav リリース検証。"""

    def test_release_fires_for_se_pending_true(self, monkeypatch):
        """se_pending=True の挙手は on_pipeline_complete で release callback 発火。"""
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(on_handraise_phrase_pending_release=lambda *a: calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert calls == []  # まだ pipeline_complete 前
        d.on_pipeline_complete()
        assert len(calls) == 1
        slug, path = calls[0]
        assert slug == "mimi"
        assert path is not None

    def test_release_does_not_fire_for_se_pending_false(self, monkeypatch):
        """se_pending=False (IDLE 挙手) は on_pipeline_complete で release されない。"""
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(on_handraise_phrase_pending_release=lambda *a: calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_pipeline_complete()
        assert calls == []

    def test_release_fires_only_once_across_multiple_pipeline_completes(self, monkeypatch):
        """se_pending を False に巻き戻すので 2 回目以降の pipeline_complete では発火しない。"""
        _patch_filler(monkeypatch, slug="mimi")
        calls = []
        d = Dispatcher(on_handraise_phrase_pending_release=lambda *a: calls.append(a))
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_pipeline_complete()  # 1 回目: 発火
        # state は依然存在するが se_pending は False に戻っている
        d.transition_to(DispatcherState.RESPONDING)
        d.on_pipeline_complete()  # 2 回目: 発火しない
        assert len(calls) == 1

    def test_release_resets_se_pending_to_false(self, monkeypatch):
        """release 後、state.se_pending は False に巻き戻される (多重再生防止)。"""
        _patch_filler(monkeypatch, slug="mimi")
        d = Dispatcher(on_handraise_phrase_pending_release=lambda *a: None)
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].se_pending is True
        d.on_pipeline_complete()
        assert d._handraise_states["mimi"].se_pending is False

    def test_release_callback_exception_does_not_propagate(self, monkeypatch):
        """release callback 例外が on_pipeline_complete を止めない。"""
        _patch_filler(monkeypatch, slug="mimi")

        def failing_release(*a):
            raise RuntimeError("boom")

        d = Dispatcher(on_handraise_phrase_pending_release=failing_release)
        _patch_lapse_timer(monkeypatch, d)
        d.transition_to(DispatcherState.RESPONDING)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # 例外なく完了
        d.on_pipeline_complete()
        # state 自体は残っている (release 失敗でも se_pending は False に巻き戻る)
        assert d._handraise_states["mimi"].se_pending is False


class TestDispatcherApprovedCallback:
    """Phase 0.5-A フェーズ 7: on_handraise_approved callback の発火検証。"""

    def _setup(self, monkeypatch, d, *, slug="mimi"):
        _patch_filler(monkeypatch, slug=slug)
        _patch_lapse_timer(monkeypatch, d)
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {slug: {"denied": "x", "lapsed": "y"}},
        )

    def test_callback_fires_with_bg_result(self, monkeypatch):
        """bg_result セット済の state を granted すると callback に bg_result が渡る。"""
        from lab_lounge.dispatcher import HandraiseBgResult
        fake_thread = MagicMock(spec=threading.Thread)
        calls = []
        d = Dispatcher(
            bg_runner=lambda **kw: fake_thread,
            on_handraise_approved=lambda *a: calls.append(a),
        )
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="snap")
        result = HandraiseBgResult(chunks=[{"x": 1}], trace_id="bg-abc")
        d._bg_set_result("mimi", result)
        d.on_approval_granted("mimi")
        assert len(calls) == 1
        slug, bg_result, transcript_snapshot, trace_id = calls[0]
        assert slug == "mimi"
        assert bg_result is result
        assert transcript_snapshot == "snap"
        assert trace_id != ""

    def test_callback_fires_with_none_bg_result(self, monkeypatch):
        """bg_result 未完成 (BG 失敗 or BG 起動前 grant) でも callback 発火、bg_result=None。"""
        calls = []
        d = Dispatcher(on_handraise_approved=lambda *a: calls.append(a))  # bg_runner=None
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="snap")
        # bg_runner=None なのでフォールバック (bg_result は None のまま)
        d.on_approval_granted("mimi")
        assert len(calls) == 1
        slug, bg_result, _, _ = calls[0]
        assert slug == "mimi"
        assert bg_result is None

    def test_no_callback_on_unknown_slug(self, monkeypatch):
        """granted on unknown slug は冪等で callback 発火しない。"""
        calls = []
        d = Dispatcher(on_handraise_approved=lambda *a: calls.append(a))
        d.on_approval_granted("unknown")  # state 無し
        assert calls == []

    def test_callback_exception_does_not_propagate(self, monkeypatch):
        """callback 例外が dispatcher 外に伝播しない。state 削除は完了する。"""

        def failing_callback(*args):
            raise RuntimeError("boom")

        d = Dispatcher(on_handraise_approved=failing_callback)
        self._setup(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        d.on_approval_granted("mimi")  # 例外なく完了
        assert d._handraise_states == {}
