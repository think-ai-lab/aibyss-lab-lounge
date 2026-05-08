"""
character_status.py — キャラクターステータス管理 (Phase 0.5-B-α)

責務:
  - 全キャラクター (mimi / chisame / sakura / octamaid / ruka 等) の内部状態を
    一元管理する。状態は CharacterStatus enum の 5 値:
      Ready / Thinking / ToolCalling / Raisehand / Talking
  - 状態 + metadata を atomic に取得できる get_snapshot() を提供
    (HUD dashboard が「全キャラの現在状態」を一覧表示するための土台)
  - 状態変化時に subscriber callback を発火 (= bus への character.status.update
    publish 経路を run_loop が closure で繋ぐ)
  - Talking 時の metadata で「立ち絵 (pose)」+「発話全文 (text)」を運ぶ
    (= ルカ要件: HUD dashboard で発話内容を視認できるようにする)

【スレッドセーフティ】
  - 全 public メソッドは self._lock (RLock) で保護される。
  - subscriber callback は Lock 外で呼ぶ (callback 内で Manager を再帰参照する
    パターンに対応するため、また publish の同期実行が長引いても他スレッドが
    ブロックされないように)。
  - RLock を採用する理由: subscribe callback が Manager を再帰呼び出す可能性が
    高い (Phase 0.5-B-β で interjection_candidate フィルタが status を読みながら
    interjection を判断する経路で必須)。再帰呼出時に Lock 取得して deadlock しない
    ようにする。

【設計記録】
  Phase 0.5-B-α の設計議論は plans/soft-bubbling-willow.md を参照。
  追加要件 (Talking 時の立ち絵 + 発話全文) は同 plan の「ルカからの追加要件」を参照。
"""

import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── CharacterStatus enum (Phase 0.5-B-α) ─────────────────────────────


class CharacterStatus(str, Enum):
    """
    キャラクターの内部状態。

    str を継承する WHY: Redis Stream の payload に直接シリアライズする際に
    `event["payload"]["status"] = status.value` でなく、`status` を渡しても
    正しく文字列として扱われる (= JSON 化時に Enum 名ではなく value が出る)。
    比較も `status == "thinking"` のように文字列直接比較が動く。

    各値の意味:
      READY:        応答可能、スタンバイ状態 (default)
      THINKING:     LLM 推論中 (graph._generation_node の入口で反映、Phase 0.5-B-α
                    commit 5 で配線)
      TOOL_CALLING: ask_character / web_search / retrieve_memory 等のツール実行中
                    (BubbleToolCallbackHandler.on_tool_start で反映、commit 5)
      RAISEHAND:    挙手中 (Dispatcher._start_handraise で反映、
                    on_approval_granted/denied/lapse で Ready に戻る、commit 3)
      TALKING:      TTS chunk 物理再生中 (run_loop の _on_tts_chunk 第 1 chunk +
                    _spawn_handraise_response_playback で反映、
                    _bg_cleanup_pipeline / worker finally で Ready に戻る、commit 4)
    """

    READY = "ready"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    RAISEHAND = "raisehand"
    TALKING = "talking"


# ─── CharacterStatusManager (Phase 0.5-B-α) ───────────────────────────


# subscriber callback シグネチャ:
#   (slug: str, new_status: CharacterStatus, old_status: CharacterStatus,
#    metadata: dict[str, Any] | None) -> None
StatusChangedCallback = Callable[
    [str, "CharacterStatus", "CharacterStatus", "dict[str, Any] | None"], None,
]


class CharacterStatusManager:
    """
    全キャラのステータス + metadata 追跡 + HUD dashboard 連携 (Phase 0.5-B-α)。

    ・状態と metadata を RLock 保護下で更新 (race-free、subscriber 内再帰呼出可)
    ・状態 / metadata 変化時に subscriber 全員に
      (slug, new_status, old_status, metadata) で通知 (Lock 外発火)
    ・get_snapshot() で全キャラの現在状態 + metadata を atomic に取得 (HUD 用)

    使用パターン:
        manager = CharacterStatusManager()
        manager.subscribe(my_callback)
        manager.set_status("mimi", CharacterStatus.THINKING)
        # → my_callback("mimi", THINKING, READY, None) が発火
        manager.set_status(
            "mimi",
            CharacterStatus.TALKING,
            metadata={"pose": "smile", "text": "..."},
        )
        # → my_callback("mimi", TALKING, THINKING, {"pose": ..., "text": ...}) 発火

    Args:
      on_status_changed: 状態変化時の callback (旧 / 新 / metadata の 4 引数)。
                         内部 _subscribers list の最初の要素として登録される。
                         後付けで subscribe() を呼べば追加 callback を登録可能。
                         None 時は subscriber なしで開始 (= publish 連携なし、
                         単純な状態保持のみで動作)。
    """

    def __init__(
        self,
        on_status_changed: StatusChangedCallback | None = None,
    ) -> None:
        # RLock 採用: subscriber callback が Manager を再帰参照する設計に対応
        # (Phase 0.5-B-β で interjection_candidate フィルタが status を読みながら
        # interjection を判断する経路で再帰呼出が発生する想定)
        self._lock = threading.RLock()
        self._statuses: dict[str, CharacterStatus] = {}
        self._metadata: dict[str, dict[str, Any] | None] = {}
        self._subscribers: list[StatusChangedCallback] = []
        if on_status_changed is not None:
            self._subscribers.append(on_status_changed)

    # ─── 状態取得 (テスト・HUD・後続 phase 用) ──────────────────────────

    def get_status(self, slug: str) -> CharacterStatus:
        """
        単一キャラの現在状態を返す (metadata 含まず)。

        未登録 slug は READY デフォルト。
        WHY: 起動直後で 1 度も set_status されていないキャラの問い合わせで例外を
        出さないため (= HUD 起動時に全キャラ状態を取りに来る経路で安全)。
        """
        with self._lock:
            return self._statuses.get(slug, CharacterStatus.READY)

    def get_metadata(self, slug: str) -> dict[str, Any] | None:
        """
        単一キャラの現在 metadata を返す。未登録 / metadata なしは None。
        """
        with self._lock:
            return self._metadata.get(slug)

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        """
        全キャラの状態 + metadata を atomic に取得する (HUD dashboard 用)。

        Returns:
            {slug: {"status": status.value, "metadata": metadata_dict_or_None}}

            外側 dict は新規生成 (= shallow copy)、内側 metadata dict は
            内部参照と共有される。WHY: metadata は immutable に扱う前提
            (= subscriber は受け取った dict を mutate しない)。コピーコスト
            最小化を優先。subscribe callback も同じ規約。

            「現在 set_status されていないキャラ」は snapshot に含まれない
            (= 未登録 = READY デフォルトと暗黙合意)。HUD 側で「全キャラ一覧」を
            出したい場合は characters.py から slug 一覧を取得して、未登録分は
            Ready 扱いで描画する設計。
        """
        with self._lock:
            return {
                slug: {
                    "status": status.value,
                    "metadata": self._metadata.get(slug),
                }
                for slug, status in self._statuses.items()
            }

    # ─── 状態更新 (Dispatcher / run_loop / graph から呼ばれる) ──────────

    def set_status(
        self,
        slug: str,
        status: CharacterStatus,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        状態 + metadata を更新し、変化時に subscriber 全員に通知する。

        冪等性: 同 status かつ同 metadata なら no-op + publish 抑止。
        ただし status 不変でも metadata が変わる場合は publish する
        (= 例えば Talking 中に text が累積するケースを将来サポート)。
        本 phase では metadata は Talking 反映時の 1 回のみで実用上問題ないが、
        将来の拡張で「chunk ごとに text を累積して publish」する場合に有効。

        callback 順序: Lock 内で旧値・新値・metadata をコピーしてから Lock 外で
        subscriber を回す (= deadlock 回避、subscriber の処理時間で他スレッドを
        ブロックしない)。subscribe() で _subscribers に追加された callback も
        同じ snapshot から発火。

        callback 例外: warning ログで握り潰し、次の subscriber には影響なし。
        Phase 0.5-A の Dispatcher._publish_xxx パターンと同じ fail-open 設計。

        Args:
            slug:     キャラクター slug (mimi/chisame/sakura/octamaid/ruka 等)
            status:   遷移先の状態
            metadata: status 固有の追加情報。Talking 時は {"pose": str, "text": str}
                      を含める (= ルカ要件: HUD dashboard で立ち絵 + 発話全文を表示)。
                      他状態では本 phase では None。将来の拡張余地として残してある。
        """
        with self._lock:
            old_status = self._statuses.get(slug, CharacterStatus.READY)
            old_metadata = self._metadata.get(slug)
            # 冪等性: status + metadata 一致なら no-op (= 連続 chunk 再生で
            # 毎チャンク Talking が set されても publish が無駄に発火しない)
            if old_status is status and old_metadata == metadata:
                return
            self._statuses[slug] = status
            self._metadata[slug] = metadata
            # Lock 内で subscriber list を snapshot (= 通知中に subscribe()
            # された callback には今回の通知は届かないが、それは race として
            # 自然 = 「subscribe したタイミング以降の変化を受け取る」設計)
            subscribers_snapshot = list(self._subscribers)

        # ログ強化 (Phase 0.5-A の L-1〜L-4 系の [character=xxx] 形式と統一)
        if metadata:
            # metadata の中身を全て出すと長文 (text=2000 字とか) になる可能性
            # あるため、info ログには key 一覧のみ、debug ログに詳細を出す
            logger.info(
                "[character=%s] status: %s -> %s (metadata_keys=%s)",
                slug, old_status.value, status.value, sorted(metadata.keys()),
            )
            logger.debug(
                "[character=%s] metadata: %s", slug, metadata,
            )
        else:
            logger.info(
                "[character=%s] status: %s -> %s",
                slug, old_status.value, status.value,
            )

        # Lock 外で subscriber に通知 (deadlock 回避、再帰参照可)
        for callback in subscribers_snapshot:
            try:
                callback(slug, status, old_status, metadata)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CharacterStatusManager: subscriber callback failed "
                    "[character=%s status=%s->%s]: %s",
                    slug, old_status.value, status.value, exc,
                )

    # ─── subscribe (後付け callback 登録) ─────────────────────────────

    def subscribe(self, callback: StatusChangedCallback) -> None:
        """
        状態変化を購読する callback を後付けで登録する。

        本 phase では購読解除を提供しない (= シンプルさ優先)。Phase 0.5-B-γ で
        必要になったら handle 返却に拡張可能 (= 既存呼出は無関係、後方互換)。

        登録 callback は (slug, new_status, old_status, metadata) で発火される。
        Lock 内で _subscribers list に append、本 callback も Lock 外で呼ばれる。

        WHY: __init__ の on_status_changed と統合実装 (= 内部 _subscribers list
        に追加するだけ)。code path が分岐しない。
        """
        with self._lock:
            self._subscribers.append(callback)


__all__ = ["CharacterStatus", "CharacterStatusManager", "StatusChangedCallback"]
