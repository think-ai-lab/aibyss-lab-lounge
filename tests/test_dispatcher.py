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

from lab_lounge.character_status import (
    CharacterStatus,
    CharacterStatusManager,
)
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

    # ─── Phase 0.5-F-1: on_approval_replay callback ──────────────────

    def test_on_approval_replay_default_is_none(self):
        """on_approval_replay は default None (= F-1 で追加された opt-in callback)。

        Phase 0.5-F-1 (案 R 移行): 既存 on_handraise_approved 経路を破壊せず、
        opt-in で案 R 経路に切替できるよう default None。non-None なら
        approve 後に on_handraise_approved を skip して replay を呼ぶ仕様。
        """
        d = Dispatcher()
        assert d._on_approval_replay is None

    def test_on_approval_replay_stored_when_provided(self):
        """on_approval_replay=callable を渡すと self._on_approval_replay に格納される。"""
        replay_callback = lambda slug, snapshot: None  # noqa: E731
        d = Dispatcher(on_approval_replay=replay_callback)
        assert d._on_approval_replay is replay_callback

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

    # ─── Phase 0.5-D-e-1: bg_completed timeout 環境変数化 ──────────────

    def test_approval_bg_completed_timeout_default_60s(self, monkeypatch):
        """L2_APPROVAL_BG_TIMEOUT_SEC 未設定で default 60.0 秒。

        Phase 0.5-D-e-1: 中間実走 11 回目で 30s が構造的に不足することを発見。
        chisame Gemini (49s) / mimi 多段階 (50-60s) を救済するため 60s に延長。
        """
        monkeypatch.delenv("L2_APPROVAL_BG_TIMEOUT_SEC", raising=False)
        d = Dispatcher()
        assert d._approval_bg_completed_timeout == 60.0

    def test_approval_bg_completed_timeout_env_override(self, monkeypatch):
        """L2_APPROVAL_BG_TIMEOUT_SEC で任意秒数に上書きできる。

        配信運用で別 LLM provider 採用時にコード変更なしで調整できるよう、
        環境変数 override を可能にしている (= D-e-1 設計)。
        """
        monkeypatch.setenv("L2_APPROVAL_BG_TIMEOUT_SEC", "90.0")
        d = Dispatcher()
        assert d._approval_bg_completed_timeout == 90.0

    def test_approval_bg_completed_timeout_invalid_env_falls_back(self, monkeypatch):
        """不正値 (= 数値変換不能) では 60.0 にフォールバック。

        配信中断回避優先の設計。warning ログ + default 値で動作継続する。
        """
        monkeypatch.setenv("L2_APPROVAL_BG_TIMEOUT_SEC", "not-a-number")
        d = Dispatcher()
        assert d._approval_bg_completed_timeout == 60.0

    # ─── Phase 0.5-D-e-4-2: 起動時 timeout 値ログ ───────────────────────

    def test_init_logs_timeout_value(self, monkeypatch, caplog):
        """Dispatcher 起動時に bg_completed_timeout 値が INFO ログに出る。

        Phase 0.5-D-e-4-2 (= 中間実走 12 検分): 環境変数 typo / 不正値 fallback
        / 運用ミスでの 30s 戻り等を実走で早期発見できるよう、起動時 1 行だけ
        現在の解決値 + 環境変数の生値を出す。
        """
        import logging as _logging
        monkeypatch.delenv("L2_APPROVAL_BG_TIMEOUT_SEC", raising=False)
        caplog.set_level(_logging.INFO, logger="lab_lounge.dispatcher")
        Dispatcher()
        matched = [
            r for r in caplog.records
            if "bg_completed_timeout=60.0s" in r.message
            and "(未設定)" in r.message
        ]
        assert matched, (
            f"Dispatcher 設定ログが出ていない: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_init_logs_timeout_value_with_env_override(self, monkeypatch, caplog):
        """環境変数 override 時には override 値が起動時ログに出る。"""
        import logging as _logging
        monkeypatch.setenv("L2_APPROVAL_BG_TIMEOUT_SEC", "90")
        caplog.set_level(_logging.INFO, logger="lab_lounge.dispatcher")
        Dispatcher()
        matched = [
            r for r in caplog.records
            if "bg_completed_timeout=90.0s" in r.message
            and "L2_APPROVAL_BG_TIMEOUT_SEC=90" in r.message
        ]
        assert matched, (
            f"override timeout ログが出ていない: "
            f"{[r.message for r in caplog.records]}"
        )


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

    # ─── Phase 0.5-B-β-2 commit 1: on_handraise_close callback (denied) ──

    def test_denied_invokes_on_handraise_close_with_denied(self, monkeypatch):
        """on_approval_denied で on_handraise_close(slug, "denied") が呼ばれる
        (Phase 0.5-B-β-2 commit 1)。

        WHY: run_loop が本 callback を受けて ask_character の bg_tts キャンセル +
        playback queue drain を実行する経路を保証 (= β-2-2 / β-2-3 で実装する
        run_loop 側の機能が dispatcher 経由で起動できるようにする)。
        """
        close_calls: list[tuple[str, str]] = []
        d = Dispatcher(
            on_handraise_close=lambda slug, reason: close_calls.append((slug, reason)),
        )
        self._setup_handraise(monkeypatch, d)
        d.on_approval_denied("mimi")
        assert close_calls == [("mimi", "denied")], (
            f"close callback が (slug='mimi', reason='denied') で 1 回呼ばれること: "
            f"{close_calls}"
        )

    def test_denied_safe_when_callback_none(self, monkeypatch):
        """on_handraise_close=None でも on_approval_denied が例外なく完了 (後方互換)。

        WHY: Phase 0.5-A 以前の Dispatcher 構築 (= callback 未指定) で動作する
        ことを保証する後方互換テスト。本 callback 追加で既存呼出が壊れない
        ことを確認する。
        """
        d = Dispatcher()  # on_handraise_close 未指定
        self._setup_handraise(monkeypatch, d)
        # 例外なく完了する
        d.on_approval_denied("mimi")
        # 既存挙動 (= state pop + cooldown 加算) も維持されている
        assert d._handraise_states == {}
        assert d._cooldowns["mimi"].consecutive_denials == 1

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

    # ─── Phase 0.5-B-β-2 commit 1: on_handraise_close callback (lapsed) ──

    def test_lapse_invokes_on_handraise_close_with_lapsed(self, monkeypatch):
        """on_lapse_timeout で on_handraise_close(slug, "lapsed") が呼ばれる
        (Phase 0.5-B-β-2 commit 1)。

        WHY: lapse は denied と異なる扱い (= cooldown 加算なし) だが、close 通知
        の必要性は同じ (= ask_character bg_tts キャンセル + playback queue drain)。
        reason="lapsed" で run_loop に区別を伝え、将来的な metric 分離の基盤を作る。
        """
        _patch_filler(monkeypatch, slug="mimi")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages",
            lambda: {"mimi": {"lapsed": "静かに"}},
        )
        close_calls: list[tuple[str, str]] = []
        d = Dispatcher(
            on_handraise_close=lambda slug, reason: close_calls.append((slug, reason)),
        )
        timers = _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        timers[0].fire()  # FakeTimer の手動発火 = lapse 通知
        assert close_calls == [("mimi", "lapsed")], (
            f"close callback が (slug='mimi', reason='lapsed') で 1 回呼ばれること: "
            f"{close_calls}"
        )

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
        result = d.on_segment_added(MagicMock(), "test text")
        # 機能 off なら LLM 呼び出しすらしない
        check_intent_mock.assert_not_called()
        check_approval_mock.assert_not_called()
        # W'-2: 機能 off は wake 経路を通常通り走らせる (= False)
        assert result is False

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
        result = d.on_segment_added(MagicMock(), "test text")
        # check_approval は handraising キャラ無しで呼ばない
        check_approval_mock.assert_not_called()
        # W'-2: unknown 帰着 (= 通常発話) は wake 経路を走らせる (= False)
        assert result is False

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
        result = d.on_segment_added(MagicMock(), "テスト")
        # check_approval が先に呼ばれる
        check_approval_mock.assert_called_once_with("テスト", ["mimi"])
        # W'-2: approval=None + check_intent=unknown + lapse 未到達 (デフォルト 8)
        # → +1 加算のみ、wake 経路は走らせる (= False)
        assert result is False

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
        result = d.on_segment_added(MagicMock(), "AI 倫理について興味がある")
        assert "mimi" in d._handraise_states
        # W'-2: 新規挙手確定 → wake skip (= True)
        assert result is True

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
        result = d.on_segment_added(MagicMock(), "ミミ、どうぞ")
        assert d._handraise_states == {}
        # W'-2: granted (= 承認発話「ミミ、どうぞ」) → wake skip (= True)。
        # 旧シナリオで観察された 3 重発火の (3) wake_event 経路を停止する核心点。
        assert result is True

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
        result = d.on_segment_added(MagicMock(), "いや、いいわ")
        assert d._handraise_states == {}
        assert d._cooldowns["mimi"].consecutive_denials == 1
        # W'-2: denied → wake skip (= True)
        assert result is True

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
        result1 = d.on_segment_added(MagicMock(), "今日はいい天気だね")
        assert d._handraise_states["mimi"].utterance_count_since == 1
        # W'-2: +1 加算のみ (閾値 8 未到達) → wake 通す (= False)。
        # 「自然な雑談中の名前呼びで別ターンを発火させたい」設計意図。
        assert result1 is False
        result2 = d.on_segment_added(MagicMock(), "明日も晴れるかな")
        assert d._handraise_states["mimi"].utterance_count_since == 2
        assert result2 is False

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
        result1 = d.on_segment_added(MagicMock(), "発話 1")
        # 1 回目: utterance_count=1 (閾値 2 未到達) → False
        assert result1 is False
        result2 = d.on_segment_added(MagicMock(), "発話 2")  # 閾値到達 → 自動 lapse
        assert d._handraise_states == {}
        # W'-2: 閾値到達で自動 lapse 発火 → 状態遷移あり → wake skip (= True)
        assert result2 is True


# ─── TestDispatcherOnSegmentReturnValue (Phase 0.5-A 案 W'-2) ────


class TestDispatcherOnSegmentReturnValue:
    """on_segment_added の戻り値 bool 仕様 (Phase 0.5-A 案 W'-2)。

    実装側の各 return 経路を網羅し、戻り値設計の意図を保証する。
    既存テスト群 (TestDispatcherSegmentDispatch) でも assertion を追加済だが、
    本クラスでは「戻り値の境界条件」をピンポイントで検証する。
    """

    def test_returns_false_when_handraising_existing_candidate_idempotent(
        self, monkeypatch,
    ):
        """既に handraising 中の slug への candidate は冪等 no-op で False を返す。

        WHY: 同一 slug の重複 candidate は state 変化なし → wake 判定を阻害しない。
        """
        _patch_filler(monkeypatch, slug="mimi", text="挙手")
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        # 先に 1 回 handraise を立ち上げる
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert "mimi" in d._handraise_states
        # check_approval=None、check_intent は同じ slug の interjection_candidate を返す
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", lambda text, slugs: None,
        )
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="interjection_candidate",
                target_slug="mimi",
                confidence=1.0,
            ),
        )
        result = d.on_segment_added(MagicMock(), "AI 倫理について再度…")
        # 冪等 no-op → False (= state 変化なし、wake 経路は通常通り)
        assert result is False
        # state は 1 件のまま
        assert list(d._handraise_states.keys()) == ["mimi"]

    def test_returns_true_only_for_lapse_triggering_utterance(self, monkeypatch):
        """lapse 発火を引き起こした segment のみ True、未到達の +1 加算は False。

        WHY: ルカの自然な雑談で、たまたま挙手中キャラの名前を含む発話があれば
        wake 経路で別キャラへの呼びかけとして処理させたい (= 「ミミ、後で考えよう」
        のような発話で挙手 lapse 中でも別キャラに呼びかけ可能にする)。
        """
        monkeypatch.setenv("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", "3")
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
        # +1 加算を 3 回繰り返し、3 回目で閾値到達
        r1 = d.on_segment_added(MagicMock(), "雑談 1")
        assert r1 is False  # count=1 (閾値 3 未到達)
        r2 = d.on_segment_added(MagicMock(), "雑談 2")
        assert r2 is False  # count=2 (閾値 3 未到達)
        r3 = d.on_segment_added(MagicMock(), "雑談 3")  # 閾値到達 → lapse
        assert r3 is True   # 状態遷移あり → wake skip
        assert d._handraise_states == {}

    def test_returns_true_when_lapse_with_simultaneous_candidate(self, monkeypatch):
        """lapse 発火と同時に別 slug の interjection_candidate も立つケースで True。

        WHY: 「mimi 挙手中、ルカの発話で mimi が lapse 閾値到達 + sakura が新規挙手」
        の同時並行ケース。両方の経路で True を返すべき (= or 結合)。
        """
        monkeypatch.setenv("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", "1")
        _patch_filler(monkeypatch, slug="mimi", text="挙手 m")
        monkeypatch.setattr(
            "lab_lounge.dispatcher._load_bubble_messages", lambda: {},
        )
        d = Dispatcher()
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        # 同一 segment で mimi lapse + sakura interjection_candidate が両方立つ
        monkeypatch.setattr(
            "lab_lounge.router.check_approval", lambda text, slugs: None,
        )
        # filler を sakura でも patch (= _start_handraise 内で wav 取得が走る)
        _patch_filler(monkeypatch, slug="sakura", text="挙手 s")
        monkeypatch.setattr(
            "lab_lounge.router.check_intent",
            lambda text, character_slug=None: IntentResult(
                intent="interjection_candidate",
                target_slug="sakura",
                confidence=1.0,
            ),
        )
        result = d.on_segment_added(MagicMock(), "sakura、何か思うことある?")
        # mimi は lapse、sakura は新規挙手 → どちらの経路でも True
        assert result is True
        # mimi lapse で消え、sakura が新規挙手で残る
        assert "mimi" not in d._handraise_states
        assert "sakura" in d._handraise_states

    def test_returns_false_for_unknown_intent_no_handraising(self, monkeypatch):
        """handraising なし + check_intent=unknown の通常発話で False。

        WHY: dispatcher が触らなかった発話は wake 経路を通常通り走らせる。
        この境界条件をピンポイントで保証 (= 関数末尾の暗黙経路をカバー)。
        """
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
        result = d.on_segment_added(MagicMock(), "ねぇ、さくら、おはよう")
        # handraising キャラ無し → check_approval は呼ばれない
        check_approval_mock.assert_not_called()
        # unknown → 通常発話扱い、wake 通す
        assert result is False


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
        # Phase 0.5-D-d-2: on_approval_granted が daemon thread 化されたため、
        # _approve_after_bg_complete 完了 (= 2 回目の publish) を polling で待つ。
        # bg_runner=None で bg_completed 即 set 済なので 10-20ms で抜ける。
        for _ in range(50):
            if len(calls) >= 2:
                break
            time.sleep(0.01)
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


# ─── TestDispatcherFlushPendingHandraiseReleases (Phase 0.5-A 8-11) ──────


class TestDispatcherFlushPendingHandraiseReleases:
    """flush_pending_handraise_releases メソッドの挙動 (Phase 0.5-A フェーズ 8-11)。

    通常応答の最終 chunk 物理再生完了直後に呼ばれて、RESPONDING 中保留された
    handraise wav を 5 秒早く release する経路。on_pipeline_complete との
    冪等性 / state 遷移分離が設計上の要点。
    """

    def _setup_responding_with_pending_handraise(
        self, monkeypatch, d: Dispatcher,
    ) -> None:
        """RESPONDING 中に挙手された state を準備する (se_pending=True)。"""
        _patch_filler(monkeypatch, slug="mimi", text="挙手")
        _patch_lapse_timer(monkeypatch, d)
        # RESPONDING 中に挙手 → se_pending=True で state がセットされる
        d._state = DispatcherState.RESPONDING
        d.on_interjection_candidate("mimi", transcript_snapshot="t")

    def test_flush_releases_pending_handraise(self, monkeypatch):
        """se_pending=True かつ phrase_path 有の handraise を release callback で発火する。"""
        release_calls: list[tuple] = []
        d = Dispatcher(
            on_handraise_phrase_pending_release=lambda slug, path: release_calls.append((slug, path)),
        )
        self._setup_responding_with_pending_handraise(monkeypatch, d)
        # 前提: se_pending=True
        assert d._handraise_states["mimi"].se_pending is True

        d.flush_pending_handraise_releases()

        assert len(release_calls) == 1
        assert release_calls[0][0] == "mimi"
        # 二重発火防止: se_pending が False に巻き戻されている
        assert d._handraise_states["mimi"].se_pending is False

    def test_flush_idempotent_with_on_pipeline_complete(self, monkeypatch):
        """flush 後に on_pipeline_complete を呼んでも release callback は再発火しない。

        flush 内で se_pending=False に巻き戻すため、後続の on_pipeline_complete は
        pending_releases が空となり no-op として安全に動く (二重発火防止)。
        """
        release_calls: list[tuple] = []
        d = Dispatcher(
            on_handraise_phrase_pending_release=lambda slug, path: release_calls.append((slug, path)),
        )
        self._setup_responding_with_pending_handraise(monkeypatch, d)

        d.flush_pending_handraise_releases()
        assert len(release_calls) == 1  # 1 回だけ発火

        # その後 on_pipeline_complete を呼んでも再発火しない
        d.on_pipeline_complete()
        assert len(release_calls) == 1  # 不変

    def test_flush_does_not_change_state(self, monkeypatch):
        """flush 単体では state を変えない (RESPONDING のまま)。

        WHY: 物理再生完了直後はまだ done bubble 発行前で、State 上は RESPONDING のまま
        が正しい (= 次の挙手判定で se_pending=True を維持できる)。
        state 遷移 (RESPONDING → IDLE) は on_pipeline_complete の責務として残す。
        """
        d = Dispatcher(on_handraise_phrase_pending_release=lambda *a: None)
        self._setup_responding_with_pending_handraise(monkeypatch, d)
        assert d.get_state() == DispatcherState.RESPONDING

        d.flush_pending_handraise_releases()

        # state は RESPONDING のまま (IDLE にはならない)
        assert d.get_state() == DispatcherState.RESPONDING

    def test_flush_no_pending_is_noop(self, monkeypatch):
        """se_pending=True の handraise が無ければ release callback は発火しない (no-op)。"""
        release_calls: list[tuple] = []
        d = Dispatcher(
            on_handraise_phrase_pending_release=lambda slug, path: release_calls.append((slug, path)),
        )
        # IDLE 中に挙手 → se_pending=False で state がセットされる (即再生経路)
        _patch_filler(monkeypatch, slug="mimi")
        _patch_lapse_timer(monkeypatch, d)
        d.on_interjection_candidate("mimi", transcript_snapshot="t")
        assert d._handraise_states["mimi"].se_pending is False

        d.flush_pending_handraise_releases()

        # IDLE 中の挙手は se_pending=False なので flush 対象外
        assert release_calls == []

    def test_flush_callback_exception_does_not_propagate(self, monkeypatch):
        """release callback の例外が flush 外に伝播しない。

        callback で例外が出ても dispatcher 本体は止めない (既存 callback パターン踏襲)。
        例外発生後の他キャラの release は引き続き処理される。
        """

        def failing_callback(slug, path):
            raise RuntimeError("boom")

        d = Dispatcher(on_handraise_phrase_pending_release=failing_callback)
        self._setup_responding_with_pending_handraise(monkeypatch, d)

        d.flush_pending_handraise_releases()  # 例外なく完了
        # se_pending は False に巻き戻されている (二重発火防止は保証される)
        assert d._handraise_states["mimi"].se_pending is False


# ─── ログ強化 L-4: on_pipeline_complete の completed_slug ──────────


class TestDispatcherOnPipelineCompleteLogging:
    """ログ強化 L-4: on_pipeline_complete に completed_slug を渡せて、
    ログに [character=...] が含まれることを検証する。
    """

    def test_completed_slug_appears_in_log(self, caplog):
        """completed_slug 指定でログに [character=mimi] が含まれる。"""
        import logging
        d = Dispatcher()
        with caplog.at_level(logging.INFO, logger="lab_lounge.dispatcher"):
            d.on_pipeline_complete(completed_slug="mimi")
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "[character=mimi]" in joined

    def test_no_completed_slug_logs_question_mark(self, caplog):
        """completed_slug 未指定でログに [character=?] と表示される (= 後方互換)。"""
        import logging
        d = Dispatcher()
        with caplog.at_level(logging.INFO, logger="lab_lounge.dispatcher"):
            d.on_pipeline_complete()
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "[character=?]" in joined

    def test_completed_slug_does_not_change_behavior(self, monkeypatch):
        """completed_slug 引数は logging のみへの影響、state 遷移挙動は不変。

        既存テストの後方互換維持を担保する。
        """
        d = Dispatcher()
        d._state = DispatcherState.RESPONDING
        d.on_pipeline_complete(completed_slug="sakura")
        assert d.get_state() == DispatcherState.IDLE


# ─── TestDispatcherStatusManager (Phase 0.5-B-α) ────────────────────


class TestDispatcherStatusManager:
    """Dispatcher の handraise 経路で CharacterStatusManager に状態を反映する検証
    (Phase 0.5-B-α)。

    Manager 注入は optional kwarg なので、既存テストの非破壊的拡張になっている
    ことも合わせて確認する。実 timer の非決定性を回避するため _FakeTimer に差し替え。
    """

    def _make_dispatcher(self, manager: CharacterStatusManager, monkeypatch) -> Dispatcher:
        """status_manager 注入 + lapse_timer を _FakeTimer に差し替えた Dispatcher。"""
        d = Dispatcher(status_manager=manager)
        monkeypatch.setattr(
            d, "_create_lapse_timer",
            lambda slug, sec: _FakeTimer(sec, lambda: None),
        )
        return d

    def test_init_status_manager_optional(self):
        """Dispatcher() で status_manager 引数なしでも動く (= 既存テスト回帰互換)。"""
        d = Dispatcher()
        assert d._status_manager is None

    def test_start_handraise_sets_raisehand_progressing(self, monkeypatch):
        """_start_handraise で RAISEHAND_PROGRESSING が反映される (Phase 0.5-D-d-1)。

        bg_runner=None の場合、bg_completed が即 set 済になり、_start_handraise の
        末尾で RAISEHAND_PROGRESSING → RAISEHAND_READY への即時遷移が走る (= 構造的に
        テスト経路 / 起動失敗時のフォールバック経路を維持)。本テストは bg_runner=None で
        最終状態が RAISEHAND_READY になることを確認する (= 旧テストの RAISEHAND 期待値
        の置換、5 → 7 値拡張)。

        実 BG LLM 経路 (= bg_runner 注入で bg_completed が _bg_set_result まで未 set)
        の挙動は test_bg_set_result_transitions_to_raisehand_ready で別途検証する。
        """
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        # bg_runner=None なので bg_completed 即 set 済 → 最終状態は RAISEHAND_READY
        assert manager.get_status("mimi") == CharacterStatus.RAISEHAND_READY

    def test_approval_granted_resets_to_ready(self, monkeypatch):
        """on_approval_granted で Manager に READY が反映される (Raisehand → Ready)。"""
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        d.on_approval_granted("mimi")
        # Phase 0.5-D-d-2: daemon thread 完了 (= READY 反映) を polling で待つ。
        # bg_runner=None で bg_completed 即 set 済なので 10-20ms で抜ける。
        for _ in range(50):
            if manager.get_status("mimi") == CharacterStatus.READY:
                break
            time.sleep(0.01)
        assert manager.get_status("mimi") == CharacterStatus.READY

    def test_approval_denied_resets_to_ready(self, monkeypatch):
        """on_approval_denied で Manager に READY が反映される (Raisehand → Ready)。"""
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        d.on_approval_denied("mimi")
        assert manager.get_status("mimi") == CharacterStatus.READY

    def test_lapse_timeout_resets_to_ready(self, monkeypatch):
        """on_lapse_timeout で Manager に READY が反映される (Raisehand → Ready)。"""
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        d.on_lapse_timeout("mimi")
        assert manager.get_status("mimi") == CharacterStatus.READY

    # ─── Phase 0.5-D-d-1: 新ステータス遷移テスト ────────────────────

    def test_callback_fires_on_raisehand_to_ready(self, monkeypatch):
        """Manager の on_status_changed callback が遷移で発火する (metadata=None)。

        Phase 0.5-D-d-1: bg_runner=None の場合、bg_completed 即 set 済で
        RAISEHAND_PROGRESSING → RAISEHAND_READY への即時遷移が走るため、callback は
        3 回発火する (ready → raisehand_progressing → raisehand_ready → ready)。
        """
        callback = MagicMock()
        manager = CharacterStatusManager(on_status_changed=callback)
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        d.on_approval_granted("mimi")
        # Phase 0.5-D-d-2: on_approval_granted が daemon thread 化されたため、
        # 3 回目の callback (= raisehand_ready → ready) を polling で待つ。
        # bg_runner=None で bg_completed 即 set 済なので 10-20ms で抜ける。
        for _ in range(50):
            if callback.call_count >= 3:
                break
            time.sleep(0.01)
        # 3 回 callback 発火 (Phase 0.5-D-d-1):
        #   ready → raisehand_progressing
        #   raisehand_progressing → raisehand_ready (bg_completed 即 set 済)
        #   raisehand_ready → ready (承認時)
        assert callback.call_count == 3
        first_call = callback.call_args_list[0]
        assert first_call.args == (
            "mimi",
            CharacterStatus.RAISEHAND_PROGRESSING,
            CharacterStatus.READY,
            None,
        )
        second_call = callback.call_args_list[1]
        assert second_call.args == (
            "mimi",
            CharacterStatus.RAISEHAND_READY,
            CharacterStatus.RAISEHAND_PROGRESSING,
            None,
        )
        third_call = callback.call_args_list[2]
        assert third_call.args == (
            "mimi",
            CharacterStatus.READY,
            CharacterStatus.RAISEHAND_READY,
            None,
        )

    # ─── Phase 0.5-B-β-3 commit 3: 同一キャラ挙手の防止 ────────────────

    def test_start_handraise_skipped_when_already_talking(self, monkeypatch):
        """既に TALKING 状態のキャラは _start_handraise が no-op (Phase 0.5-B-β-3 commit 3)。

        WHY: シナリオ 3 で観察した「自分が応答中なのに raisehand 遷移」のカオス
        フローを防ぐ。応答中のキャラ (= talking) は raisehand 対象から除外し、
        現在の発話を続行させる (= 既に話す権利を持っているので追加挙手は不自然)。
        """
        _patch_filler(monkeypatch, slug="mimi")
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)

        # 事前に mimi を TALKING に設定 (= 既に応答中の状況を再現)
        manager.set_status("mimi", CharacterStatus.TALKING)

        # interjection_candidate 経由で _start_handraise を呼ぶ
        d.on_interjection_candidate("mimi", transcript_snapshot=None)

        # 期待: status は TALKING のまま (= raisehand に遷移しない)
        assert manager.get_status("mimi") == CharacterStatus.TALKING
        # 期待: handraise_states に追加されない (= 挙手扱いされない)
        assert "mimi" not in d._handraise_states

    def test_start_handraise_proceeds_when_status_ready(self, monkeypatch):
        """status=READY のキャラは _start_handraise が通常進行 (Phase 0.5-B-β-3 commit 3)。

        WHY: 後方互換確認。READY (= 応答していない) のキャラは挙手対象として
        通常通り扱われ、RAISEHAND に遷移する。β-3-3 のガード追加で既存挙動が
        壊れないことを保証する (= 既存テスト test_start_handraise_sets_raisehand
        と同じシナリオを明示的に talking ガードと組み合わせて検証)。
        """
        _patch_filler(monkeypatch, slug="mimi")
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)

        # mimi は READY デフォルト
        assert manager.get_status("mimi") == CharacterStatus.READY

        d.on_interjection_candidate("mimi", transcript_snapshot=None)

        # 期待: RAISEHAND_PROGRESSING → RAISEHAND_READY に遷移 (Phase 0.5-D-d-1、
        # bg_runner=None なので bg_completed 即 set 済 → 即遷移、β-3-3 ガード未抵触)
        assert manager.get_status("mimi") == CharacterStatus.RAISEHAND_READY
        assert "mimi" in d._handraise_states

    # ─── Phase 0.5-D-d-2: daemon thread + on_approval_progressing テスト ────

    def test_approval_granted_bg_completed_already_set_runs_synchronously(self, monkeypatch):
        """bg_completed 即 set 済 (= bg_runner=None) で on_approval_granted を呼ぶと、
        daemon thread の wait は即 return → state pop 即実行 (Phase 0.5-D-d-2)。

        既存テスト (test_granted_removes_state 等) と同じシナリオだが、daemon thread
        化されたことを polling pattern で明示的に検証する。bg_completed.wait() は即
        return するため、polling は通常 10-20ms で抜ける。
        """
        manager = CharacterStatusManager()
        d = self._make_dispatcher(manager, monkeypatch)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        d.on_approval_granted("mimi")

        # daemon thread 完了を polling で待つ
        for _ in range(50):
            with d._lock:
                if "mimi" not in d._handraise_states:
                    break
            time.sleep(0.01)
        assert "mimi" not in d._handraise_states

