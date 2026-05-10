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

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .character_status import CharacterStatus, CharacterStatusManager
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


# ─── Phase 0.5-A bubble メッセージ取得 (denied/lapsed/cancelled の text 解決) ──
# pipeline.py に同名のヘルパーがあるが、dispatcher.py からの循環 import を回避する
# ため独立実装。フェーズ 7 で run_loop / pipeline と共通化する余地がある。
# テストでは ``monkeypatch.setattr("lab_lounge.dispatcher._load_bubble_messages",
# fake)`` で差し替える。
_BUBBLE_MESSAGES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "bubble_messages.json"
)


def _load_bubble_messages() -> dict[str, Any]:
    """data/bubble_messages.json を読み込んで {slug: {step: text}} の dict を返す。

    ロード失敗時は空 dict を返す (フォールバックで default テキストが使われる)。
    """
    try:
        return json.loads(_BUBBLE_MESSAGES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "bubble_messages.json 読み込み失敗: %s (default にフォールバック)", exc,
        )
        return {}


# default フォールバック (octamaid 等で denied/lapsed が定義されてない場合 / 読込失敗時)。
# 実テキストは data/bubble_messages.json で各キャラ口調に合わせて定義されている。
_DEFAULT_BUBBLE_TEXTS: dict[str, str] = {
    "denied": "(また今度…)",
    "lapsed": "(静まりました)",
    "cancelled": "(撤回)",
}


def _get_bubble_text(
    messages: dict[str, Any],
    character_slug: str,
    step: str,
) -> str:
    """messages から指定 step / キャラのテキストを取得する。

    キャラ別エントリが無い、または step が無い場合は default フォールバック文字列。
    """
    char_entry = messages.get(character_slug, {})
    text = char_entry.get(step)
    if not text:
        text = _DEFAULT_BUBBLE_TEXTS.get(step, "")
    return text


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
      bg_result:             BG LLM 生成結果。``HandraiseBgResult`` または None。
                             フェーズ 7 で実体投入 (フェーズ 5b は None のまま)
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
        on_handraise_started: Callable[[str, "Path | None", bool], None] | None = None,
        on_handraise_phrase_pending_release: Callable[[str, "Path | None"], None] | None = None,
        on_handraise_close: Callable[[str, str], None] | None = None,
        on_approval_replay: Callable[[str, Any], None] | None = None,
        status_manager: CharacterStatusManager | None = None,
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
            on_handraise_started: 挙手状態が確定した直後 (= ``_start_handraise`` の
                Lock 解除後) に呼ばれる callback。引数 ``(slug, phrase_path, se_pending)``。
                run_loop が ``se_pending=False`` なら handraise wav を即再生、
                True なら ``on_handraise_phrase_pending_release`` を待つ。
            on_handraise_phrase_pending_release: ``on_pipeline_complete`` 内で
                応答完了時の保留 wav リリースに呼ばれる callback。引数
                ``(slug, phrase_path)``。RESPONDING 中に挙手したキャラの wav を
                IDLE 復帰時に再生する経路。
            on_handraise_close: 却下 (``on_approval_denied``) / lapse
                (``on_lapse_timeout``) 時、Lock 解除後に呼ばれる close 通知 callback
                (Phase 0.5-B-β-2)。引数 ``(target_slug, reason)`` で、reason は
                ``"denied"`` / ``"lapsed"``。run_loop は本 callback で ask_character
                の bg_tts キャンセル + playback queue drain を実行し、案 A の音声漏れ
                (= 挙手中に流れた対話 TTS が却下後も再生キューに残る問題) を最小化する。
                None 時は通知スキップ (= 後方互換、Phase 0.5-A 以前と同じ挙動)。
            on_approval_replay: 承認 (``on_approval_granted``) 時に呼ばれる callback
                (Phase 0.5-F-1 で導入、F-3 で wiring されて以降は唯一の承認経路)。引数
                ``(target_slug, transcript_snapshot)``。run_loop は内部で
                ``dispatcher.on_wake_detected(WakeWordResult(transcript=...))`` を
                呼んで callout 経路と統合する想定。None 時は承認時に no-op (= 後方互換)。
            status_manager: 全キャラのステータス (Ready/Thinking/ToolCalling/Raisehand/
                Talking) を一元管理する CharacterStatusManager (Phase 0.5-B-α)。
                本クラスは handraise 経路 (start / approval_granted / approval_denied /
                lapse_timeout) で Raisehand / Ready を反映する。HUD dashboard は
                Manager の subscriber を経由して character.status.update event を
                購読する。None 時は status 反映スキップ (= 後方互換、既存テストは
                引数なしで動作)。
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
        # Phase 0.5-A フェーズ 7: 物理通知 callback (run_loop が注入)
        self._on_handraise_started = on_handraise_started
        self._on_handraise_phrase_pending_release = on_handraise_phrase_pending_release
        # Phase 0.5-B-β-2: 却下/lapse 時の close 通知 callback。run_loop が
        # ask_character bg_tts キャンセル + playback queue drain を実行する。
        self._on_handraise_close = on_handraise_close
        # Phase 0.5-F-1: 案 R 経路の callback。F-3 で wiring されて以降は
        # 唯一の承認経路 (= 旧 on_handraise_approved は F-4-c/f-2 で削除済)。
        self._on_approval_replay = on_approval_replay
        # Phase 0.5-B-α: 全キャラ状態を一元管理する Manager (handraise 経路で
        # Raisehand / Ready を反映)。None 時は status 反映スキップ (後方互換)。
        self._status_manager = status_manager
        self._max_events = max_events
        self._max_age_sec = max_age_sec
        # Phase 0.5-D-e-1 (= 中間実走 11 回目修正、timeout 30s 不足対処):
        # bg_completed.wait の timeout 値 (秒)。テスト時には monkeypatch で
        # 0.1s 等に短縮して時間効率を保つ。
        #
        # 【WHY: 30s → 60s に延長】
        # 中間実走 11 回目で観察された実測レイテンシ:
        #   - chisame 単段 (Gemini gemini-3.1-pro-preview): 49 秒
        #   - mimi 多段階 ask_character ×2 (gpt-5.5): 50-60 秒
        # 30s では構造的に間に合わず、毎回 fallback パスへ落ち、buffer drain で
        # mimi 導入セリフ + bridge filler + chisame chunk 1 が破棄される。
        # 60s に延長することで多段階 / Gemini を救済 (= bg_result=ready 確率向上)。
        #
        # 【環境変数 L2_APPROVAL_BG_TIMEOUT_SEC で override 可能】
        # 配信運用で動的調整できるよう環境変数化。別キャラ追加 / 別 LLM 採用時に
        # コード変更なしで調整できる柔軟性を確保。値は Dispatcher() 生成時に
        # 評価されるため、配信開始時には決定する。
        # 不正値 (= 数値変換不能) は 60.0 にフォールバック (= 配信中断回避優先)。
        _bg_timeout_raw = os.environ.get("L2_APPROVAL_BG_TIMEOUT_SEC", "60.0")
        try:
            self._approval_bg_completed_timeout: float = float(_bg_timeout_raw)
        except ValueError:
            logger.warning(
                "L2_APPROVAL_BG_TIMEOUT_SEC=%r は数値ではないため 60.0 にフォールバックします",
                _bg_timeout_raw,
            )
            self._approval_bg_completed_timeout = 60.0

        # Phase 0.5-A: 挙手機能の設定 (環境変数から 1 回だけ読み込む)
        # テストでは monkeypatch.setenv した後に Dispatcher() を生成すれば反映される
        cfg = _get_handraise_config()
        self._use_handraise: bool = cfg["use_handraise"]
        self._lapse_sec: float = cfg["lapse_sec"]
        self._lapse_utterance_count: int = cfg["lapse_utterance_count"]

        # Phase 0.5-D-e-4-2 (= 中間実走 12 検分):
        # 起動時に解決された timeout 値 + 環境変数の生値を INFO 出力。
        # 環境変数 typo / 不正値 fallback / 運用ミスでの 30s 戻り等を実走で
        # 早期発見できるようにする (= 起動時 1 行だけ出して noise を抑える)。
        logger.info(
            "Dispatcher 設定: bg_completed_timeout=%.1fs (env L2_APPROVAL_BG_TIMEOUT_SEC=%s)",
            self._approval_bg_completed_timeout,
            os.environ.get("L2_APPROVAL_BG_TIMEOUT_SEC", "(未設定)"),
        )

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

    def on_pipeline_complete(self, completed_slug: str | None = None) -> None:
        """
        run_loop が pipeline (LLM + TTS + 再生) 完了時に呼ぶ。

        - 期限切れ event を破棄
        - 状態を IDLE に戻す
        - queue に残りがあればメインスレッドを起こす (次の wait_for_next_event が
          即座に dequeue できるように)
        - Phase 0.5-A フェーズ 7: 応答中に挙手したキャラ (se_pending=True) の
          handraise wav を「IDLE に戻ったタイミング」で run_loop へ release 通知し、
          多重再生防止のため se_pending を False に巻き戻す

        Args:
            completed_slug: 完了した応答のキャラ slug。ログ強化 L-4 (Phase 0.5-A 後)
                            で追加。「どの応答の完了か」をログで識別できるようにする。
                            None 時は ? 表示 (= 旧経路 / 単独の状態リセット時)。
        """
        pending_releases: list[tuple[str, "Path | None"]] = []
        with self._lock:
            evicted = self._evict_expired_unlocked(now=time.monotonic())
            self._state = DispatcherState.IDLE
            # Phase 0.5-A フェーズ 7: se_pending=True かつ phrase_path 有を Lock 内で集める。
            # 多重通知防止のため se_pending を False に戻す (1 回だけリリース)。
            for slug, st in self._handraise_states.items():
                if st.se_pending and st.phrase_path is not None:
                    pending_releases.append((slug, st.phrase_path))
                    st.se_pending = False
            queue_copy = list(self._wake_event_queue)
            if self._wake_event_queue:
                # queue に残り → 次の wait_for_next_event を起こす
                self._event_available.notify_all()

        # ログ強化 L-4: 完了したキャラ slug を先頭に出して、複数並行する応答の
        # うちどの完了か識別容易に
        logger.info(
            "Dispatcher.on_pipeline_complete [character=%s]: state→IDLE "
            "evicted=%d remaining=%d pending_releases=%d",
            completed_slug or "?",
            evicted, len(queue_copy), len(pending_releases),
        )
        if evicted > 0:
            self._publish_queue_update(queue_copy)
        # Phase 0.5-A フェーズ 7: Lock 外で release callback を発火。
        # run_loop は通常応答完了 → 1.5 秒の空白 → handraise wav 再生という流れで
        # 視聴者の耳に立つ立て続け再生を回避する (env: L2_HANDRAISE_RESPONDING_PADDING_SEC)。
        for slug, path in pending_releases:
            if self._on_handraise_phrase_pending_release is not None:
                try:
                    self._on_handraise_phrase_pending_release(slug, path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "on_handraise_phrase_pending_release callback failed: slug=%s err=%s",
                        slug, exc,
                    )

    def flush_pending_handraise_releases(self) -> None:
        """挙手 wav の遅延再生を即時フラッシュする (Phase 0.5-A フェーズ 8-11)。

        通常応答の最終 chunk 物理再生完了直後 (= done_delay 5.0s 待機の前) に呼ばれること
        を想定。handraise wav の再生開始を 5 秒早めて視聴者の体感を改善する
        (実走 C-1 で観察した「sakura 挙手 wav が 6.5 秒遅れ」の解消)。

        本メソッドは「handraise wav release のみ」を担当する。state 遷移
        (RESPONDING → IDLE) や queue notify_all は ``on_pipeline_complete`` に残す。
        WHY: 物理再生完了直後はまだ done bubble 発行前で、State 上は RESPONDING のままが
        正しい (= 次の挙手判定で se_pending=True を維持できる正常な振る舞い)。

        冪等性: flush 内で ``se_pending=False`` に巻き戻すため、後続の
        ``on_pipeline_complete`` は pending_releases が空となり no-op として安全に動く。
        逆に flush が呼ばれない経路 (= worker 異常終了 / 例外で sentinel に到達せず等)
        で残った分は ``on_pipeline_complete`` 側のロジックが救済する (= 二重防御)。

        Lock 設計: Lock 内で pending_releases を集めて se_pending=False に巻き戻し、
        Lock 外で release callback を発火する (既存パターン踏襲、deadlock 回避)。
        """
        pending_releases: list[tuple[str, "Path | None"]] = []
        with self._lock:
            for slug, st in self._handraise_states.items():
                if st.se_pending and st.phrase_path is not None:
                    pending_releases.append((slug, st.phrase_path))
                    st.se_pending = False

        if pending_releases:
            logger.info(
                "Dispatcher.flush_pending_handraise_releases: pending=%d (early release)",
                len(pending_releases),
            )
        for slug, path in pending_releases:
            if self._on_handraise_phrase_pending_release is not None:
                try:
                    self._on_handraise_phrase_pending_release(slug, path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "flush_pending_handraise_releases callback failed: slug=%s err=%s",
                        slug, exc,
                    )

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
        category: str | None = None,
    ) -> None:
        """
        on_bubble_update callback を呼び出す (Phase 0.5-A)。

        Lock 外で呼ばれることを前提に、引数 (character, step, text, ttl_ms, category) を
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
            category:  Phase 0.5-A 8-10: V2 HUD 側で表示エリアを分岐させる種別。
                       Dispatcher が発行するのは挙手系のみのため、呼出元は
                       "handraise" を渡す。None なら events.py 側で payload に
                       含めず、受信側 default 解釈 (= speech 扱い、後方互換)。
        """
        if self._on_bubble_update is None:
            return
        try:
            self._on_bubble_update(character, step, text, ttl_ms, category)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dispatcher: on_bubble_update callback failed: %s", exc,
            )

    # ─── Phase 0.5-A 挙手 API (フェーズ 5b で実装解除) ────────────────
    #
    # 挙手フロー:
    #   1. on_segment_added(): BG Listener が segment 追加時に呼ぶ (フェーズ 6 で接続)
    #      - handraising キャラがあれば check_approval を試す
    #      - 該当なし or なければ check_intent (interjection_candidate モード)
    #      - 結果に応じて on_interjection_candidate / on_approval_*
    #   2. on_interjection_candidate(): _start_handraise() を呼んで挙手状態を作る
    #   3. on_approval_granted(): 承認音声 → 状態解除 + cooldown リセット
    #   4. on_approval_denied():  却下音声 → 状態解除 + cooldown 加算 + bubble denied
    #   5. on_lapse_timeout():    時間/utterance lapse → 状態解除 + bubble lapsed
    #
    # フェーズ 5 では bg_thread = None / bg_completed.set() 即時呼び (no-op)。
    # フェーズ 7 で BG LLM 本体 + handraise wav 再生を組み込む。
    #
    # 関連 Notion: 346e38612fe88190a79cd07c0d9c1484
    # ──────────────────────────────────────────────────────────

    def on_segment_added(
        self,
        segment: Any,            # TranscriptSegment (循環 import 回避で Any)
        buffer_full_text: str,
    ) -> bool:
        """BG Listener から segment 追加時に呼ばれる callback (Phase 0.5-A)。

        フェーズ 6 で BackgroundContinuousListener.start(on_segment_added=...)
        に接続される。状態に応じて check_intent / check_approval を呼び分け、
        結果に応じた API を呼び出す。

        判定の優先順位:
          1. ``self._use_handraise=False`` → 機能 off で即 return False
          2. handraising キャラあり → ``check_approval`` を試す
             - granted/denied なら on_approval_granted/denied を呼ぶ
             - None (関係ない発話) → utterance_count_since 加算 (lapse 判定)
          3. 上記で確定しなければ ``check_intent(text, character_slug=None)`` で
             interjection_candidate 判定を試す
             - interjection_candidate なら on_interjection_candidate を呼ぶ

        Args:
            segment:           TranscriptSegment (現状未使用、フェーズ 7 で
                              segment.text などを使う可能性あり)
            buffer_full_text:  TranscriptBuffer.full_text() (LLM 判定対象テキスト)

        Returns:
            bool: True=本 segment を「挙手系処理として確定的に消費」した。
                  Listener 側で ``_evaluate_wake`` を skip して wake_event 経路への
                  二重発火を防止する (= 案 W'-2)。

                  True を返す経路:
                    - granted: check_approval が承認 (= 「(キャラ名)、どうぞ」)
                    - denied:  check_approval が却下 (= 「いや、いいわ」)
                    - 自動 lapse 発火: 今回の発話で utterance_count が閾値到達
                    - interjection_candidate 新規挙手確定

                  False を返す経路:
                    - 機能 off (= L2_USE_HANDRAISE=false)
                    - handraising キャラあり / approval=None / lapse 発火なし
                      (= +1 加算のみ。自然な雑談中に名前呼びで別ターンを発火させたい)
                    - interjection_candidate 既存の slug への重複 (= 冪等 no-op、
                      state 変化なし → wake 判定を阻害しない)
                    - 上記いずれにも該当しない通常発話 (= unknown)

            **WHY (戻り値設計)**:
              snappy-bentley 実走で観察された 3 重発火パターンでは、ルカが
              「さくらさん、どうぞ」と発話した瞬間、その同一 segment が:
                (a) 本関数で check_approval=granted → on_approval_granted 呼出
                (b) 同時に Listener._evaluate_wake → router.route(name_hint=sakura)
                    → wake_event_queue 投入 → run_loop の (3) 通常応答ターン発火
              という二重消費を起こしていた。Dispatcher が「発話を一意に解釈して
              状態遷移を行った」segment と、「Dispatcher が触らなかった」segment を
              戻り値で区別することで、Listener 側で wake 判定を選択的に skip 可能。
        """
        if not self._use_handraise:
            return False  # 機能 off → wake 経路は通常通り走らせる

        # 関数内 import で循環回避 (router.py 側で dispatcher を import する将来拡張に備える)
        from . import router as _router

        # 「processed」= 本 segment を挙手系で消費したか。複数の経路で True に
        # 倒し得るので bool 1 つに集約 (= or 結合と同等)。
        processed = False

        # handraising キャラがあれば check_approval を先に試す
        with self._lock:
            candidate_slugs = list(self._handraise_states.keys())

        if candidate_slugs:
            approval = _router.check_approval(buffer_full_text, candidate_slugs)
            if approval is not None:
                if approval.granted:
                    self.on_approval_granted(approval.target_slug)
                else:
                    self.on_approval_denied(approval.target_slug)
                return True  # 承認/却下確定 → wake skip (W'-2)

            # check_approval が None (関係ない発話) → utterance_count_since 加算
            # 同時に複数の slug が utterance lapse 条件超過する可能性があるため、
            # Lock 内で集めてから Lock 外で順次 on_lapse_timeout を呼ぶ
            slugs_to_lapse: list[str] = []
            with self._lock:
                for slug, state in self._handraise_states.items():
                    state.utterance_count_since += 1
                    if state.utterance_count_since >= self._lapse_utterance_count:
                        slugs_to_lapse.append(slug)
            if slugs_to_lapse:
                for slug in slugs_to_lapse:
                    self.on_lapse_timeout(slug)
                # WHY: 自動 lapse 発火は dispatcher の状態遷移なので wake skip 対象。
                # ただし「+1 加算のみで lapse なし」は processed=False のまま (= 自然な
                # 雑談中に名前呼びで別ターンを発火させたい設計)。
                processed = True

        # interjection_candidate 判定 (handraising キャラ無し or 該当発話なし)
        intent = _router.check_intent(buffer_full_text, character_slug=None)
        if intent.intent == "interjection_candidate" and intent.target_slug:
            with self._lock:
                if intent.target_slug in self._handraise_states:
                    # 既に handraising 中 → 冪等 no-op。state 変化していないので
                    # wake 判定は阻害しない方針。ただし lapse 発火由来の processed=True
                    # は保つ (= 同一 segment で別 slug の lapse があった場合も skip)。
                    return processed
            self.on_interjection_candidate(
                intent.target_slug,
                transcript_snapshot=buffer_full_text,
            )
            return True  # 新規挙手確定 → wake skip (W'-2)

        return processed

    def _create_lapse_timer(
        self,
        target_slug: str,
        delay_sec: float,
    ) -> threading.Timer:
        """lapse タイマーを生成する (Phase 0.5-A)。

        threading.Timer のテスト非決定性を回避するため、生成を委譲メソッド化して
        ``monkeypatch.setattr(dispatcher_instance, "_create_lapse_timer", fake)``
        で FakeTimer に差し替えられるようにする。
        """
        return threading.Timer(delay_sec, self.on_lapse_timeout, args=[target_slug])

    def _start_handraise(
        self,
        target_slug: str,
        transcript_snapshot: Any,
    ) -> None:
        """挙手状態を作成 + lapse_timer 起動 + BG LLM 起動 + 通知発火 (Phase 0.5-A)。

        冪等性: target_slug が既に handraising 中なら no-op。

        フェーズ 7 で BG LLM スレッド (= ``self._bg_runner``) を起動。bg_runner が
        None の場合 (= テスト用 / 未注入) は ``state.bg_completed.set()`` で即座に
        完了状態にし、承認時に run_loop の fallback (= 同期再生成) パスに流れる。

        bubble.update(handraise) は ttl_ms=None で発行 (承認/却下/lapse まで保持)。
        ``on_handraise_started`` callback は Lock 解除後に発火し、run_loop が
        ``se_pending`` を見て handraise wav を即再生 / 保留する。

        Phase 0.5-B-β-3 commit 3: 既に talking 状態のキャラは挙手対象から除外する
        (= 同一キャラが応答中に raisehand に遷移するカオス防止、シナリオ 3 で観察)。
        """
        # 関数内 import で循環回避 + filler.py の副作用を起動時に避ける
        from .filler import select_filler_phrase

        # Phase 0.5-B-β-3 commit 3: 既に talking 状態のキャラは挙手対象から除外。
        # シナリオ 3 で観察した「自分が発話中なのに raisehand に遷移」のカオス的
        # フローを防ぐ。例えば chisame が量子コンピューターを応答中にルカが
        # 「論理と感情の両方の意見が聞きたい」と発話 → chisame 自身が候補に →
        # talking->raisehand へ遷移、という現象。応答中のキャラは「既に話す権利
        # を持っている」ため、追加で挙手するのは設計上不自然。
        # status_manager 未注入時 (= 後方互換、テスト等) はチェックスキップ。
        if self._status_manager is not None:
            current = self._status_manager.get_status(target_slug)
            if current == CharacterStatus.TALKING:
                logger.info(
                    "Dispatcher._start_handraise skip: slug=%s already talking",
                    target_slug,
                )
                return

        with self._lock:
            if target_slug in self._handraise_states:
                return  # 冪等

            # filler の handraise セクションから wav パス + テキストを取得
            path, phrase, _ = select_filler_phrase(target_slug, category="handraise")
            phrase_text = phrase.text if phrase else ""

            state = HandraiseState(
                target_slug=target_slug,
                started_at=time.monotonic(),
                transcript_snapshot=transcript_snapshot,
                phrase=phrase_text,
                phrase_path=path,
                trace_id=str(uuid4()),
                # 応答中 (RESPONDING) なら se_pending=True
                # (フェーズ 7: on_pipeline_complete で release callback を発火する経路)
                se_pending=(self._state == DispatcherState.RESPONDING),
            )
            # F-4-d で旧 bg_runner 経路廃止 (案 R = `on_approval_replay`)。bg_completed
            # を即時 set して承認時の wait を即 return させ、callout 経路に統合する。
            state.bg_completed.set()
            # lapse_timer 起動 (委譲メソッド経由でテスト容易性確保)
            state.lapse_timer = self._create_lapse_timer(target_slug, self._lapse_sec)
            state.lapse_timer.start()
            self._handraise_states[target_slug] = state

        logger.info(
            "Dispatcher._start_handraise: slug=%s phrase=%r trace_id=%s se_pending=%s",
            target_slug, phrase_text, state.trace_id, state.se_pending,
        )
        # Phase 0.5-B-α: 挙手状態を CharacterStatusManager に反映 (HUD 用)。
        # publish 順序: status.update → bubble.update(handraise) → handraise.update。
        # HUD 側は status.update を先に観察してから bubble の詳細を処理する流れと整合。
        #
        # Phase 0.5-D-d-1: RAISEHAND → RAISEHAND_PROGRESSING (= bg_runner 起動と同
        # タイミング、HUD で「先行思考中」ローディング表示)。
        #
        # bg_runner=None または bg_runner 起動失敗の経路では bg_completed が既に
        # set() 済 (上の Lock 内 / except 経路) なので、その場合は直後に
        # RAISEHAND_READY も反映して HUD を「準備完了」表示に進める (= 構造的に
        # 「PROGRESSING → READY」の自然遷移を維持、テスト経路 + 起動失敗時のフォール
        # バック経路含む)。実 BG LLM が動く経路では _bg_set_result の完了時に
        # RAISEHAND_READY が反映される (= こちらは独立経路、無関係)。
        if self._status_manager is not None:
            self._status_manager.set_status(
                target_slug, CharacterStatus.RAISEHAND_PROGRESSING,
            )
            # bg_completed が既に set 済 (= bg_runner=None / 起動失敗) なら READY 反映
            with self._lock:
                state_now = self._handraise_states.get(target_slug)
                bg_already_completed = (
                    state_now is not None and state_now.bg_completed.is_set()
                )
            if bg_already_completed:
                self._status_manager.set_status(
                    target_slug, CharacterStatus.RAISEHAND_READY,
                )
        # publish + 物理通知も Lock 外 (callback の長時間処理が dispatcher を止めない)
        self._publish_bubble_update(target_slug, "handraise", phrase_text, ttl_ms=None, category="handraise")
        self._publish_handraise_update()
        # フェーズ 7: handraise wav の物理再生は run_loop の責務。dispatcher は
        # 「IDLE 中なら即再生して」という意思を se_pending=False で伝えるだけ。
        # se_pending=True の場合 run_loop はスキップし、後の on_pipeline_complete
        # 内で release callback を待つ (RESPONDING → IDLE 遷移時に再生される経路)。
        if self._on_handraise_started is not None:
            try:
                self._on_handraise_started(
                    target_slug, state.phrase_path, state.se_pending,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_handraise_started callback failed: %s", exc)

    def on_interjection_candidate(
        self,
        target_slug: str,
        transcript_snapshot: Any,  # TranscriptBuffer (循環 import 回避のため Any)
    ) -> None:
        """check_intent が interjection_candidate を返したときに呼ぶ (Phase 0.5-A)。

        target_slug を挙手中状態にし、bubble.update(handraise) と
        dispatcher.handraise.update を発行する。BG LLM 起動はフェーズ 7 で実装。
        """
        if not self._use_handraise:
            return
        self._start_handraise(target_slug, transcript_snapshot)

    def on_approval_granted(self, target_slug: str) -> None:
        """ルカの「〇〇、どうぞ」承認音声で呼ばれる (Phase 0.5-A)。

        handraising 状態を解除し、cooldown の consecutive_denials をリセット。
        実際の TTS / 再生 + bubble.update("answering") はフェーズ 7 で run_loop が
        引き取る (BG LLM 結果 or 最新 buffer での再生成と TTS 開始タイミングを
        同期させるため、dispatcher 側では bubble の next step を発行しない)。

        Phase 0.5-A フェーズ 7: state pop の直前に bg_result / transcript_snapshot
        を抽出して、Lock 解除後に ``on_handraise_approved`` callback で run_loop に
        引き渡す。run_loop は bg_result.chunks を専用 mini playback worker で再生する。
        bg_result が None の場合は run_loop が同期 fallback (再生成) に流す。

        【Phase 0.5-D-d-2: daemon thread 化】
        承認時に BG LLM 未完了 (= bg_completed が未 set) の場合、daemon thread 内で
        最大 30 秒 wait してから本処理 (= ``_approve_after_bg_complete``) を実行する。
        これにより:
        - approval 早すぎで BG LLM 進行中なら、完了を待ってから state pop + callback
          (= bg_result が ready で本来の streaming spawn 経路を取れる)
        - 完了済みなら wait は即 return (= 既存テストへの影響なし)
        - timeout 30 秒経過で諦めて本処理 (= bg_result=None で fallback パス、真の救済)

        さらに progressing 経路では ``on_approval_progressing`` callback を発火して
        run_loop が bridge filler を即時再生する (= 30 秒沈黙の配信事故レベル対処)。
        ただし ``state.se_pending=True`` (= RESPONDING 中) のキャラは発火しない
        (= 通常応答 TTS との 3 重音声重なり UX 崩壊を防ぐ、★R4 対処)。

        【WHY: daemon thread にする理由】
        wake_event 処理スレッドが 30 秒ブロックされるのを避ける (= 次の wake_event
        を処理できなくなるのを防ぐ)。bg_completed 即 set 済 (= 既存挙動) の場合は
        wait() が即 return するため、daemon thread 起動コスト数 ms のみで結果同じ。
        既存テストは polling pattern で対応可能 (= max 0.5s の polling で抜ける)。
        """
        if not self._use_handraise:
            return
        with self._lock:
            state = self._handraise_states.get(target_slug)
            if state is None:
                return  # 冪等 (既に granted/denied/lapse 済)
            bg_completed_event = state.bg_completed

        # F-4-f-2: progressing 経路 + on_approval_progressing 廃止 (= bg_runner も
        # _start_handraise で即 set されるため bg_completed_event は wait() で即 return)。
        # bg_completed_event は下の wait() で使うため残す。
        timeout_sec = self._approval_bg_completed_timeout

        def _wait_and_approve() -> None:
            bg_completed_event.wait(timeout=timeout_sec)
            self._approve_after_bg_complete(target_slug)

        threading.Thread(
            target=_wait_and_approve,
            name=f"approval-wait-{target_slug}",
            daemon=True,
        ).start()

    def _approve_after_bg_complete(self, target_slug: str) -> None:
        """bg_completed.wait 完了後の本処理 (Phase 0.5-D-d-2 → F-4-f-2 で簡素化)。

        state pop + cancel_event.set + cooldown reset + READY 反映 +
        handraise.update publish + on_approval_replay callback 発火を行う。

        ``state is None`` の場合は冪等 no-op (= wait 中に granted/denied/lapse が別
        経路で発生した場合の race ガード)。

        F-4-f-2 で `on_handraise_approved` 経路 + `bg_result` 抽出を完全削除。
        承認時の応答経路は `on_approval_replay` (案 R) が唯一の経路。
        """
        transcript_snapshot = None
        with self._lock:
            state = self._handraise_states.get(target_slug)
            if state is None:
                return  # 冪等 (wait 中に別経路で削除された)
            state.cancel_event.set()  # BG LLM ベストエフォート cleanup
            if state.lapse_timer is not None:
                state.lapse_timer.cancel()
            # 承認なので連続却下カウントをリセット
            if target_slug in self._cooldowns:
                self._cooldowns[target_slug].consecutive_denials = 0
            transcript_snapshot = state.transcript_snapshot
            del self._handraise_states[target_slug]

        logger.info("Dispatcher.on_approval_granted: slug=%s", target_slug)
        # Phase 0.5-B-α: 承認時点で Raisehand_Ready → Ready に戻す。
        # WHY: 承認 → TTS chunks 生成 → 再生開始までに 2-3 秒の gap がある。その間
        # Ready (= ニュートラル) を維持する方が HUD の精度が上がる。Talking への
        # 上書きは run_loop の playback worker (案 R 経路では callout 経路) で行われる。
        if self._status_manager is not None:
            self._status_manager.set_status(target_slug, CharacterStatus.READY)
        self._publish_handraise_update()
        # Phase 0.5-F-1: 案 R 経路 (= raisehand を callout 経路に統合)。F-3 で
        # wiring されて以降は唯一の承認経路。
        if self._on_approval_replay is not None:
            try:
                self._on_approval_replay(target_slug, transcript_snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_approval_replay callback failed: %s", exc)

    def on_approval_denied(self, target_slug: str) -> None:
        """ルカの却下 (「いや、いいわ」) で呼ばれる (Phase 0.5-A)。

        cancel_event.set() で BG LLM 中断、cooldown の consecutive_denials を +1、
        bubble.update(denied) を ttl_ms=2000 で発行。
        threshold_multiplier の適用は Phase 0.5-B 以降 (現状は 1.0 固定)。
        """
        if not self._use_handraise:
            return
        with self._lock:
            state = self._handraise_states.get(target_slug)
            if state is None:
                return  # 冪等
            state.cancel_event.set()
            if state.lapse_timer is not None:
                state.lapse_timer.cancel()
            # cooldown 加算 (Phase 0.5-A は consecutive_denials のみ、threshold_multiplier=1.0 固定)
            cd = self._cooldowns.setdefault(target_slug, CooldownState())
            cd.consecutive_denials += 1
            denials_after = cd.consecutive_denials
            del self._handraise_states[target_slug]

        logger.info(
            "Dispatcher.on_approval_denied: slug=%s consecutive_denials=%d",
            target_slug, denials_after,
        )
        # Phase 0.5-B-α: 却下 → Raisehand → Ready
        if self._status_manager is not None:
            self._status_manager.set_status(target_slug, CharacterStatus.READY)
        # Phase 0.5-B-β-2: close 通知 (= run_loop が ask_character の bg_tts キャンセル
        # + playback queue drain を実行する経路)。bubble.update 発行より前に呼ぶこと
        # で、視聴者向け描画より先に音声 cleanup を起動する (= 表示が「却下」に切り
        # 替わった瞬間にすでに残音声の破棄要求が出ている状態を作る)。callback 内
        # の例外は Lock 外なので dispatcher を止めない (= warning ログのみ)。
        if self._on_handraise_close is not None:
            try:
                self._on_handraise_close(target_slug, "denied")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "on_handraise_close callback failed (denied, slug=%s): %s",
                    target_slug, exc,
                )
        # bubble.update("denied") の text を bubble_messages から取得 (フォールバックあり)
        messages = _load_bubble_messages()
        text = _get_bubble_text(messages, target_slug, "denied")
        self._publish_bubble_update(target_slug, "denied", text, ttl_ms=2000, category="handraise")
        self._publish_handraise_update()

    def on_lapse_timeout(self, target_slug: str) -> None:
        """lapse_timer 発火 or utterance_count_since 閾値超えで呼ばれる (Phase 0.5-A)。

        consecutive_denials は変動なし (lapse は「却下」とは異なる扱い)。
        bubble.update(lapsed) を ttl_ms=2000 で発行。

        環境変数 ``L2_HANDRAISE_LAPSE_SEC`` (default 300 = 5 分) と
        ``L2_HANDRAISE_LAPSE_UTTERANCE_COUNT`` (default 8) で閾値を制御する。
        """
        if not self._use_handraise:
            return
        with self._lock:
            state = self._handraise_states.get(target_slug)
            if state is None:
                return  # 冪等 (既に granted/denied/lapse 済)
            state.cancel_event.set()
            # lapse_timer 自身からの呼び出しなのでキャンセル不要 (発火後は cancel しても no-op)
            del self._handraise_states[target_slug]

        logger.info("Dispatcher.on_lapse_timeout: slug=%s", target_slug)
        # Phase 0.5-B-α: lapse → Raisehand → Ready (consecutive_denials は変動なし)
        if self._status_manager is not None:
            self._status_manager.set_status(target_slug, CharacterStatus.READY)
        # Phase 0.5-B-β-2: close 通知 (= on_approval_denied と同様)。reason="lapsed"
        # で run_loop に「タイムアウト由来の close」と区別を伝える (= 将来的に
        # cooldown / metric を分けたい場合の基盤)。
        if self._on_handraise_close is not None:
            try:
                self._on_handraise_close(target_slug, "lapsed")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "on_handraise_close callback failed (lapsed, slug=%s): %s",
                    target_slug, exc,
                )
        messages = _load_bubble_messages()
        text = _get_bubble_text(messages, target_slug, "lapsed")
        self._publish_bubble_update(target_slug, "lapsed", text, ttl_ms=2000, category="handraise")
        self._publish_handraise_update()
