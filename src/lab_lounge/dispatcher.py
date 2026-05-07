"""
dispatcher.py — 状態管理 + wake_event queue (Block 0 録音常時化)

責務:
  - アプリケーション全体の状態 (IDLE / RESPONDING / HANDRAISING) を一元管理
  - BackgroundContinuousListener が検知した wake_event を受け取り、
    状態に応じてメインスレッドへ通知 / queue に積む
  - 応答完了後に queue を drain (FIFO + 上限 3 件 + 60 秒期限切れ破棄)
  - queue 変化時に callback を呼び出して bus.publish(dispatcher.queue.update) を
    可能にする (HUD のデバッグ dashboard 用、配信画面非表示)

Phase 0.5 (挙手システム) への接続点:
  - DispatcherState.HANDRAISING enum 値を予約
  - on_interjection_candidate / on_approval_granted / on_approval_denied /
    on_lapse_timeout の API シグネチャを予約 (NotImplementedError)
  - _handraise_states / _cooldowns dict を空で予約

【スレッドセーフティ】
  - 全 public メソッドは self._lock で保護される。
  - threading.Condition で「queue に event が積まれた」イベントをメインスレッドに
    通知する (wait_for_next_event の wait/notify_all)。
  - publish callback は Lock 外で呼ぶ (callback 内で長時間処理しても dispatcher
    が止まらないように、また callback が dispatcher を再呼び出しする deadlock を
    回避するため)。

【設計記録】
  Block 0 の設計議論は plans/sparkling-fluttering-penguin.md を参照。
  Phase 0.5 の設計記録は Notion 346e38612fe88190a79cd07c0d9c1484。
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .wake_word import WakeWordResult

logger = logging.getLogger(__name__)

# ─── ドレイン戦略の定数 (Block 0 仮置き、配信実走で調整) ──────────────
# 上限 3 件: 応答中の連続 wake_event を最大 3 件まで保持。それ以上は古いものから破棄
# 60 秒期限切れ: enqueue から 60 秒経過した event は drain 時に破棄。
#                TranscriptBuffer.window_sec のデフォルト 30 秒より長めに設定し、
#                長文応答中の有効発話を取りこぼさないようにする。
DRAIN_MAX_EVENTS = 3
DRAIN_MAX_AGE_SEC = 60.0


# ─── Phase 0.5-A 挙手システム設定 (環境変数読込) ──────────────────────
# 環境変数を Dispatcher.__init__ で 1 回読み込み、self に保持して以後参照する。
# (環境変数を直接散らさず一元管理し、テストで monkeypatch しやすくする)
def _get_handraise_config() -> dict[str, Any]:
    """Phase 0.5-A の挙手機能設定を環境変数から読み込む。

    返り値の dict は次のキーを含む:
      - "use_handraise":  挙手機能の on/off (default True)
      - "lapse_sec":      時間 lapse 閾値 (default 300 = 5 分)
      - "lapse_utterance_count":
                          utterance lapse 閾値 (default 8 utterance)

    環境変数:
      L2_USE_HANDRAISE                      "true" / "false" (default true)
      L2_HANDRAISE_LAPSE_SEC                秒数 (default 300)
      L2_HANDRAISE_LAPSE_UTTERANCE_COUNT    回数 (default 8)
    """
    return {
        "use_handraise": os.environ.get("L2_USE_HANDRAISE", "true").lower()
            in ("1", "true", "yes"),
        "lapse_sec": float(os.environ.get("L2_HANDRAISE_LAPSE_SEC", "300")),
        "lapse_utterance_count": int(
            os.environ.get("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", "8")
        ),
    }


class DispatcherState(Enum):
    """
    Dispatcher の状態。

    Block 0 で実際に遷移するのは IDLE / RESPONDING の 2 状態のみ。
    HANDRAISING は Phase 0.5 で実装される予定の状態として予約してある。
    """

    IDLE = "idle"
    RESPONDING = "responding"
    HANDRAISING = "handraising"  # Phase 0.5 で発火、Block 0 では遷移しない


@dataclass
class QueuedWakeEvent:
    """
    wake_event_queue に積まれるエントリ。

    WakeWordResult に「キューに積まれた時刻」を付加することで、TTL
    (DRAIN_MAX_AGE_SEC) 判定と HUD 表示用の age 算出を可能にする。
    """

    event: WakeWordResult
    enqueued_at: float  # time.monotonic() at enqueue


@dataclass
class HandraiseState:
    """
    Phase 0.5-A で使う挙手中キャラの状態。

    Block 0 で予約された target_slug + started_at に加え、Phase 0.5-A で
    BG LLM スレッド管理 / lapse タイマー / bubble 表示用フィールドを追加。
    フェーズ 5b で API メソッドが state を読み書きし、フェーズ 7 で
    bg_thread / phrase_path 再生の実体組込を行う。

    フィールド:
      target_slug:           挙手中キャラの slug (一意キー)
      started_at:            time.monotonic() 基準の開始時刻 (HUD 表示用 age 算出)
      transcript_snapshot:   挙手判定時の TranscriptBuffer snapshot (T7 で再生成)
      cancel_event:          却下/lapse 時に set。BG LLM はベストエフォートで観察
      bg_thread:             BG LLM 生成スレッド (フェーズ 7 で起動、5 では None)
      bg_result:             BG LLM 生成結果 (フェーズ 7 で実体投入)
      bg_completed:          BG LLM 完了フラグ (フェーズ 5 では即座 set で no-op)
      phrase:                bubble.text 用、handraise wav と同じテキスト
      phrase_path:           handraise wav パス (フェーズ 7 で再生)
      se_pending:            応答中なら True (フェーズ 7 で on_pipeline_complete 後発火)
      lapse_timer:           threading.Timer。utterance count 上限超過でも発火
      utterance_count_since: 挙手以降のルカ発話数 (8 で lapse)
      trace_id:              handraise 単位の trace_id (bubble.update / handraise.update に付与)
    """

    target_slug: str
    started_at: float
    transcript_snapshot: Any = None         # TranscriptBuffer (循環 import 回避で Any)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    bg_thread: threading.Thread | None = None
    bg_result: Any = None
    bg_completed: threading.Event = field(default_factory=threading.Event)
    phrase: str = ""
    phrase_path: Path | None = None
    se_pending: bool = False
    lapse_timer: threading.Timer | None = None
    utterance_count_since: int = 0
    trace_id: str = ""


@dataclass
class CooldownState:
    """
    Phase 0.5-A: 連続却下による cooldown 状態。

    consecutive_denials は却下のたびに +1、承認で 0 にリセット。
    cooldown_until / threshold_multiplier は Phase 0.5-A では値を持たないが
    (multiplier=1.0 固定)、Phase 0.5-B で「連続却下が増えるほど挙手しにくくする」
    閾値変動ロジックの土台として保持しておく。

    フィールド:
      cooldown_until:        cooldown 解除時刻 (time.monotonic() 基準)
                             Phase 0.5-A では 0.0 のまま (cooldown 強制発動なし)
      consecutive_denials:   連続却下回数 (承認 / lapse でリセットしない設計)
                             Phase 0.5-B で閾値変動の入力に使う
      threshold_multiplier:  挙手閾値の倍率。Phase 0.5-A は常に 1.0 固定
    """

    cooldown_until: float = 0.0
    consecutive_denials: int = 0
    threshold_multiplier: float = 1.0


class Dispatcher:
    """
    状態機械 + wake_event queue (Block 0)。

    通常フロー:
      1. ContinuousListener が wake_event 検知
      2. on_wake_detected(event) で queue に積む
      3. メインスレッドが wait_for_next_event() で次の event を取り出し RESPONDING へ
      4. メインスレッドが pipeline 実行
      5. on_pipeline_complete() で IDLE に戻る
      6. queue に残りがあれば 3 へ戻る

    Phase 0.5 への接続点:
      - HANDRAISING 状態 / on_interjection_candidate 等の API は予約のみ
        (Block 0 では NotImplementedError)
    """

    def __init__(
        self,
        *,
        on_queue_update: Callable[[list[QueuedWakeEvent]], None] | None = None,
        on_handraise_update: Callable[
            [dict[str, "HandraiseState"], dict[str, "CooldownState"]], None
        ] | None = None,
        on_bubble_update: Callable[[str, str, str, int | None], None] | None = None,
        max_events: int = DRAIN_MAX_EVENTS,
        max_age_sec: float = DRAIN_MAX_AGE_SEC,
    ) -> None:
        """
        Args:
            on_queue_update: queue 変化時 (add / evict / dequeue) に呼ばれる callback。
                ``dispatcher.queue.update`` イベントを bus に publish するために
                run_loop からセットされる想定。Lock 外で呼ばれるため、callback 内で
                長時間処理しても dispatcher は止まらない。
            on_handraise_update: 挙手状態 / cooldown 変化時に呼ばれる callback (Phase 0.5-A)。
                ``dispatcher.handraise.update`` イベントを bus に publish するために
                run_loop からセットされる想定。callback には _handraise_states と
                _cooldowns の shallow copy が渡される (Lock 外で呼ばれる)。
            on_bubble_update: bubble.update 発行が必要な時に呼ばれる callback (Phase 0.5-A)。
                引数 (character, step, text, ttl_ms) を受け取り、run_loop が
                ``build_bubble_update(...)`` で event を組み立てて publish する想定。
                Lock 外で呼ばれる。
            max_events:  queue の最大保持件数。超過分は古いものから破棄
            max_age_sec: enqueue から N 秒以上経過した event を drain 時に破棄
        """
        self._state = DispatcherState.IDLE
        self._lock = threading.Lock()
        # 「queue に新規 event が積まれた」または「pipeline 完了で次の event を
        # ドレインできるようになった」を待つ Condition。同じ Lock を共有する。
        self._event_available = threading.Condition(self._lock)
        self._wake_event_queue: deque[QueuedWakeEvent] = deque()

        # Phase 0.5 で使う dict。Block 0 では空のまま。
        # フェーズ 5a で _cooldowns 型を dict[str, float] → dict[str, CooldownState]
        # に変更 (consecutive_denials / threshold_multiplier の保持)。
        self._handraise_states: dict[str, HandraiseState] = {}
        self._cooldowns: dict[str, CooldownState] = {}

        self._on_queue_update = on_queue_update
        # Phase 0.5-A: 挙手状態 + bubble の publish callback
        self._on_handraise_update = on_handraise_update
        self._on_bubble_update = on_bubble_update
        self._max_events = max_events
        self._max_age_sec = max_age_sec

        # Phase 0.5-A: 挙手機能の設定 (環境変数から 1 回だけ読み込む)
        # テストでは monkeypatch.setenv した後に Dispatcher() を生成すれば反映される
        cfg = _get_handraise_config()
        self._use_handraise: bool = cfg["use_handraise"]
        self._lapse_sec: float = cfg["lapse_sec"]
        self._lapse_utterance_count: int = cfg["lapse_utterance_count"]

    # ─── 状態取得 (テスト・デバッグ用) ────────────────────────────────

    def get_state(self) -> DispatcherState:
        """現在の状態を返す。"""
        with self._lock:
            return self._state

    def get_queue_snapshot(self) -> list[QueuedWakeEvent]:
        """現在の queue 内容のコピーを返す (テスト・デバッグ用、副作用なし)。"""
        with self._lock:
            return list(self._wake_event_queue)

    # ─── 状態遷移 (run_loop が明示的に呼ぶ) ──────────────────────────

    def transition_to(self, state: DispatcherState) -> None:
        """
        状態を明示的に遷移させる。

        通常は wait_for_next_event() が IDLE→RESPONDING、on_pipeline_complete() が
        RESPONDING→IDLE を担うので、外部から呼ぶのはテスト・特殊ケースのみ。
        """
        with self._lock:
            old = self._state
            self._state = state
        logger.info("Dispatcher: %s → %s", old.value, state.value)

    # ─── イベント受信 (Listener / pipeline からのコールバック) ────────

    def on_wake_detected(self, event: WakeWordResult) -> None:
        """
        ContinuousListener (録音スレッド) から wake_event 検知時に呼ばれる。

        どの状態でも queue に積み、Condition.notify_all() でメインスレッドを起こす。
        IDLE 中なら wait_for_next_event() で即座に取り出され、RESPONDING 中なら
        on_pipeline_complete() で IDLE に戻った後に取り出される。
        """
        now = time.monotonic()
        queued = QueuedWakeEvent(event=event, enqueued_at=now)

        with self._lock:
            self._wake_event_queue.append(queued)
            # 上限超過分・期限切れをこの時点でも除去 (古い event が積まれた状態で
            # 新しい event が来た場合の整合性維持)
            self._evict_expired_unlocked(now=now)
            self._event_available.notify_all()
            current_state = self._state
            queue_copy = list(self._wake_event_queue)

        logger.info(
            "Dispatcher.on_wake_detected: state=%s slug=%s queue_size=%d",
            current_state.value, event.character_slug, len(queue_copy),
        )
        # publish は Lock 外で呼ぶ (callback の長時間処理が dispatcher を止めない)
        self._publish_queue_update(queue_copy)

    def on_pipeline_complete(self) -> None:
        """
        run_loop が pipeline (LLM + TTS + 再生) 完了時に呼ぶ。

        - 期限切れ event を破棄
        - 状態を IDLE に戻す
        - queue に残りがあればメインスレッドを起こす (次の wait_for_next_event が
          即座に dequeue できるように)
        """
        with self._lock:
            evicted = self._evict_expired_unlocked(now=time.monotonic())
            self._state = DispatcherState.IDLE
            queue_copy = list(self._wake_event_queue)
            if self._wake_event_queue:
                # queue に残り → 次の wait_for_next_event を起こす
                self._event_available.notify_all()

        logger.info(
            "Dispatcher.on_pipeline_complete: state→IDLE evicted=%d remaining=%d",
            evicted, len(queue_copy),
        )
        if evicted > 0:
            self._publish_queue_update(queue_copy)

    def wait_for_next_event(self, timeout: float | None = None) -> WakeWordResult | None:
        """
        次の wake_event を待機して取り出す (メインスレッドが呼ぶ)。

        queue から先頭を 1 件取り出し、状態を RESPONDING に遷移させて返す。
        queue が空の場合は Condition で wait し、新しい event が来るか
        timeout を待つ。

        Args:
            timeout: 待機タイムアウト秒数。None で無限待機。

        Returns:
            次の event。timeout で取得できなかった場合は None。
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        with self._lock:
            while True:
                # 待機ループに入る前 / 起こされた直後に期限切れを除去
                self._evict_expired_unlocked(now=time.monotonic())

                if self._wake_event_queue:
                    queued = self._wake_event_queue.popleft()
                    queue_copy = list(self._wake_event_queue)
                    self._state = DispatcherState.RESPONDING
                    # publish は Lock 解除後に行うため、ここではコピーだけ確保
                    break

                # queue 空 → wait
                if deadline is None:
                    self._event_available.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None  # timeout
                    self._event_available.wait(timeout=remaining)
                # spurious wakeup や notify_all で起こされた → ループ先頭で再チェック

        # Lock 外で publish + ログ
        logger.info(
            "Dispatcher.wait_for_next_event: dequeued slug=%s remaining=%d state→RESPONDING",
            queued.event.character_slug, len(queue_copy),
        )
        self._publish_queue_update(queue_copy)
        return queued.event

    # ─── drain ルール ─────────────────────────────────────────────

    def _evict_expired_unlocked(self, *, now: float) -> int:
        """
        期限切れ event と上限超過 event を queue から除去する (Lock 取得済み前提)。

        - 期限切れ: enqueued_at が ``now - max_age_sec`` より古い event を先頭から除去
        - 上限超過: queue 長が ``max_events`` を超えていれば古い順に除去

        Returns:
            除去した件数。
        """
        cutoff = now - self._max_age_sec
        evicted = 0

        # 期限切れ除去 (FIFO 性質から先頭が一番古い)
        while self._wake_event_queue and self._wake_event_queue[0].enqueued_at < cutoff:
            old = self._wake_event_queue.popleft()
            evicted += 1
            logger.info(
                "Dispatcher: evicted expired event slug=%s age_sec=%.1f",
                old.event.character_slug, now - old.enqueued_at,
            )

        # 上限超過分を古いものから除去 (上限 3 → 4 件目が来た時点で 1 件目を破棄)
        while len(self._wake_event_queue) > self._max_events:
            old = self._wake_event_queue.popleft()
            evicted += 1
            logger.info(
                "Dispatcher: evicted overflow event slug=%s",
                old.event.character_slug,
            )

        return evicted

    # ─── publish (queue.update イベント発行) ─────────────────────

    def _publish_queue_update(self, queue_copy: list[QueuedWakeEvent]) -> None:
        """
        queue 変化時に on_queue_update callback を呼び出す。

        Lock 外で呼ばれることを前提に、queue_copy はあらかじめコピー済みのリストを
        受け取る。callback で例外が出ても dispatcher 本体は止めない (warning ログ)。
        """
        if self._on_queue_update is None:
            return
        try:
            self._on_queue_update(queue_copy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dispatcher: on_queue_update callback failed: %s", exc)

    # ─── Phase 0.5-A 挙手 publish (bubble + handraise.update) ──────

    def _publish_handraise_update(self) -> None:
        """
        on_handraise_update callback を呼び出す (Phase 0.5-A)。

        Lock 外で呼ばれることを前提に、_handraise_states / _cooldowns の shallow copy
        を取得して callback に渡す (callback 内で dispatcher を再呼び出ししても
        deadlock しないように)。callback で例外が出ても dispatcher 本体は止めない。

        run_loop からセットされる callback は ``build_dispatcher_handraise_update``
        で event を組み立てて bus に publish する想定。
        """
        if self._on_handraise_update is None:
            return
        # Lock 取得して copy、Lock 外で callback 呼出 (dispatcher.queue.update と同パターン)
        with self._lock:
            states_copy = dict(self._handraise_states)
            cooldowns_copy = dict(self._cooldowns)
        try:
            self._on_handraise_update(states_copy, cooldowns_copy)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dispatcher: on_handraise_update callback failed: %s", exc,
            )

    def _publish_bubble_update(
        self,
        character: str,
        step: str,
        text: str,
        ttl_ms: int | None = None,
    ) -> None:
        """
        on_bubble_update callback を呼び出す (Phase 0.5-A)。

        Lock 外で呼ばれることを前提に、引数 (character, step, text, ttl_ms) を
        callback にそのまま渡す。run_loop 側で ``build_bubble_update(...)`` を
        呼び出して bus に publish する想定。callback で例外が出ても dispatcher
        本体は止めない。

        Args:
            character: キャラクター slug
            step:      "handraise" / "denied" / "lapsed" / "cancelled" 等
                       (events.py の build_bubble_update に渡される値)
            text:      bubble 表示テキスト
            ttl_ms:    自動消去ミリ秒。None なら次 step まで保持。Phase 0.5-A は
                       denied/lapsed=2000ms / handraise=None を想定
        """
        if self._on_bubble_update is None:
            return
        try:
            self._on_bubble_update(character, step, text, ttl_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dispatcher: on_bubble_update callback failed: %s", exc,
            )

    # ─── Phase 0.5 用 API (Block 0 では NotImplementedError) ──────
    #
    # Phase 0.5 着手時にここを実装する。シグネチャを予約しておくことで、
    # Phase 0.5 のテストや呼出し側を Block 0 段階から書き始められる
    # (NotImplementedError raises を期待値として確認する形)。
    #
    # 関連 Notion: 346e38612fe88190a79cd07c0d9c1484
    # ──────────────────────────────────────────────────────────

    def on_interjection_candidate(
        self,
        target_slug: str,
        transcript_snapshot: Any,  # TranscriptBuffer (循環 import 回避のため Any)
    ) -> None:
        """
        Phase 0.5: ``check_intent`` が ``interjection_candidate`` を返したときに呼ぶ。

        target_slug を HANDRAISING 状態に遷移させ、bubble.update(handraise) と
        BG LLM タスクを起動する想定。Block 0 では未実装 (NotImplementedError)。
        """
        raise NotImplementedError(
            "Phase 0.5 で実装。Block 0 では HANDRAISING 状態に遷移しない。"
        )

    def on_approval_granted(self, target_slug: str) -> None:
        """
        Phase 0.5: ルカの「〇〇、どうぞ」承認音声で呼ばれる。

        最新 transcript buffer (snapshot) で BG LLM を再生成し、RESPONDING へ
        遷移する想定。Block 0 では未実装。
        """
        raise NotImplementedError("Phase 0.5 で実装")

    def on_approval_denied(self, target_slug: str) -> None:
        """
        Phase 0.5: ルカの却下で呼ばれる。

        BG LLM タスクを cancel し、締めフレーズ bubble を表示して IDLE / cooldown
        に遷移する想定。Block 0 では未実装。
        """
        raise NotImplementedError("Phase 0.5 で実装")

    def on_lapse_timeout(self, target_slug: str) -> None:
        """
        Phase 0.5: 30 秒 or 3 utterance 経過で自動 lapse。

        BG LLM タスクを cancel し、lapsed bubble (「(遠慮しました)」) を表示する
        想定。Block 0 では未実装。
        """
        raise NotImplementedError("Phase 0.5 で実装")
