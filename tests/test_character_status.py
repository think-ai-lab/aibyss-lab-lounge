"""
test_character_status.py — CharacterStatus enum + CharacterStatusManager 単体テスト

Phase 0.5-B-α 新規。Manager は他モジュールに依存しない純粋データクラス + Lock の
ため、本テストファイルも他のテストモジュールに依存しない (= dispatcher / run_loop /
graph 等の mock 不要、純粋単体で完結)。
"""

import json
import threading
from unittest.mock import MagicMock

from lab_lounge.character_status import (
    CharacterStatus,
    CharacterStatusManager,
)


# ─── TestCharacterStatusEnum ────────────────────────────────────────


class TestCharacterStatusEnum:
    def test_seven_values(self):
        """7 値の .value が plan で確定したラベルと一致する。

        Phase 0.5-D-d-1 で RAISEHAND_PROGRESSING / RAISEHAND_READY を追加 (= 5 → 7 値)。
        BG LLM 進捗の HUD 視覚化 + 承認時の待ち合わせ用に挙手中状態を細分化。
        """
        assert CharacterStatus.READY.value == "ready"
        assert CharacterStatus.THINKING.value == "thinking"
        assert CharacterStatus.TOOL_CALLING.value == "tool_calling"
        assert CharacterStatus.RAISEHAND.value == "raisehand"
        assert CharacterStatus.RAISEHAND_PROGRESSING.value == "raisehand_progressing"
        assert CharacterStatus.RAISEHAND_READY.value == "raisehand_ready"
        assert CharacterStatus.TALKING.value == "talking"

    def test_str_inheritance(self):
        """str 継承で文字列直接比較が動く (= payload シリアライズ前提)。"""
        assert CharacterStatus.READY == "ready"
        assert CharacterStatus.THINKING == "thinking"
        assert isinstance(CharacterStatus.TALKING, str)

    def test_str_inheritance_for_progressing_ready(self):
        """Phase 0.5-D-d-1: 新ステータスも str 継承が機能する (= V2 HUD payload 比較で動作)。"""
        assert CharacterStatus.RAISEHAND_PROGRESSING == "raisehand_progressing"
        assert CharacterStatus.RAISEHAND_READY == "raisehand_ready"
        assert isinstance(CharacterStatus.RAISEHAND_PROGRESSING, str)
        assert isinstance(CharacterStatus.RAISEHAND_READY, str)

    def test_json_serializable(self):
        """JSON 化時に value が出る (= Enum 名ではなく文字列値)。"""
        # str 継承なので json.dumps で .value 相当の文字列が出る
        assert json.dumps(CharacterStatus.THINKING) == '"thinking"'
        assert json.dumps(CharacterStatus.RAISEHAND) == '"raisehand"'

    def test_json_serializable_progressing_ready(self):
        """Phase 0.5-D-d-1: 新ステータスも JSON 化で value が出る (= event payload 互換)。"""
        assert json.dumps(CharacterStatus.RAISEHAND_PROGRESSING) == '"raisehand_progressing"'
        assert json.dumps(CharacterStatus.RAISEHAND_READY) == '"raisehand_ready"'


# ─── TestCharacterStatusManagerInit ──────────────────────────────────


class TestCharacterStatusManagerInit:
    def test_default_empty(self):
        """default 初期空 (= 内部 _statuses dict が空)。"""
        m = CharacterStatusManager()
        assert m.get_snapshot() == {}

    def test_unregistered_slug_returns_ready(self):
        """未登録 slug は READY デフォルト (= HUD 起動時の安全策)。"""
        m = CharacterStatusManager()
        assert m.get_status("mimi") == CharacterStatus.READY
        assert m.get_metadata("mimi") is None

    def test_init_callback_none_works(self):
        """on_status_changed=None で初期化しても set_status が動く (= subscriber なしでの動作)。"""
        m = CharacterStatusManager(on_status_changed=None)
        m.set_status("mimi", CharacterStatus.THINKING)
        assert m.get_status("mimi") == CharacterStatus.THINKING


# ─── TestSetStatus ───────────────────────────────────────────────────


class TestSetStatus:
    def test_state_change(self):
        """status の遷移が反映される。"""
        m = CharacterStatusManager()
        m.set_status("mimi", CharacterStatus.THINKING)
        assert m.get_status("mimi") == CharacterStatus.THINKING
        m.set_status("mimi", CharacterStatus.TALKING)
        assert m.get_status("mimi") == CharacterStatus.TALKING

    def test_callback_fires_on_change(self):
        """callback が (slug, new, old, metadata) で発火する。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        m.set_status("mimi", CharacterStatus.THINKING)
        callback.assert_called_once_with(
            "mimi", CharacterStatus.THINKING, CharacterStatus.READY, None,
        )

    def test_idempotent_no_publish(self):
        """同 status + 同 metadata の連続呼出は 1 回のみ publish (冪等)。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        m.set_status("mimi", CharacterStatus.TALKING)
        m.set_status("mimi", CharacterStatus.TALKING)  # 冪等 no-op
        m.set_status("mimi", CharacterStatus.TALKING)  # 冪等 no-op
        assert callback.call_count == 1

    def test_callback_exception_does_not_break_next(self):
        """callback 例外でも他 subscriber に影響しない (fail-open)。"""
        bad_callback = MagicMock(side_effect=RuntimeError("boom"))
        good_callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=bad_callback)
        m.subscribe(good_callback)
        # 例外を上げず、両 subscriber が呼ばれる
        m.set_status("mimi", CharacterStatus.THINKING)
        bad_callback.assert_called_once()
        good_callback.assert_called_once()

    def test_callback_receives_metadata(self):
        """metadata 引数が callback の 4 つ目に渡る。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        meta = {"pose": "smile", "text": "こんにちは"}
        m.set_status("mimi", CharacterStatus.TALKING, metadata=meta)
        callback.assert_called_once_with(
            "mimi", CharacterStatus.TALKING, CharacterStatus.READY, meta,
        )


# ─── TestSetStatusMetadata ──────────────────────────────────────────


class TestSetStatusMetadata:
    def test_metadata_update_publishes(self):
        """同 status でも metadata が変わると publish する (= text 累積等の将来用途)。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        m.set_status("mimi", CharacterStatus.TALKING, metadata={"pose": "smile"})
        callback.reset_mock()
        # 同 status だが metadata に text を追加
        m.set_status(
            "mimi", CharacterStatus.TALKING,
            metadata={"pose": "smile", "text": "..."},
        )
        callback.assert_called_once()

    def test_metadata_none_to_dict_publishes(self):
        """metadata=None → dict への変化で publish する。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        m.set_status("mimi", CharacterStatus.TALKING, metadata=None)
        callback.reset_mock()
        m.set_status("mimi", CharacterStatus.TALKING, metadata={"pose": "smile"})
        callback.assert_called_once()

    def test_metadata_same_dict_no_publish(self):
        """同 status + 同内容 metadata の連続呼出は 1 回のみ publish (冪等性確認)。"""
        callback = MagicMock()
        m = CharacterStatusManager(on_status_changed=callback)
        meta = {"pose": "neutral", "text": "..."}
        m.set_status("mimi", CharacterStatus.TALKING, metadata=meta)
        # dict() で複製しても同内容なので冪等扱い
        m.set_status("mimi", CharacterStatus.TALKING, metadata=dict(meta))
        assert callback.call_count == 1


# ─── TestGetSnapshot ─────────────────────────────────────────────────


class TestGetSnapshot:
    def test_atomic_snapshot_includes_status_and_metadata(self):
        """全キャラの status + metadata が snapshot に含まれる。"""
        m = CharacterStatusManager()
        m.set_status("mimi", CharacterStatus.THINKING)
        m.set_status(
            "chisame", CharacterStatus.TALKING,
            metadata={"pose": "neutral", "text": "..."},
        )
        m.set_status("sakura", CharacterStatus.RAISEHAND)

        snap = m.get_snapshot()
        assert snap["mimi"] == {"status": "thinking", "metadata": None}
        assert snap["chisame"] == {
            "status": "talking",
            "metadata": {"pose": "neutral", "text": "..."},
        }
        assert snap["sakura"] == {"status": "raisehand", "metadata": None}

    def test_snapshot_is_shallow_copy(self):
        """snapshot を mutate しても内部状態に影響しない (= 外側 dict は新規)。"""
        m = CharacterStatusManager()
        m.set_status("mimi", CharacterStatus.THINKING)
        snap = m.get_snapshot()
        del snap["mimi"]
        # 内部状態は維持される
        assert m.get_status("mimi") == CharacterStatus.THINKING
        # 再度 snapshot 取得すると含まれる
        assert "mimi" in m.get_snapshot()


# ─── TestSubscribe ───────────────────────────────────────────────────


class TestSubscribe:
    def test_subscribe_after_init(self):
        """後付け subscribe で callback 追加できる。"""
        callback = MagicMock()
        m = CharacterStatusManager()
        m.subscribe(callback)
        m.set_status("mimi", CharacterStatus.THINKING)
        callback.assert_called_once_with(
            "mimi", CharacterStatus.THINKING, CharacterStatus.READY, None,
        )

    def test_multiple_subscribers_all_fire(self):
        """複数 subscriber 全員に発火する (= __init__ + subscribe() 両方)。"""
        cb1 = MagicMock()
        cb2 = MagicMock()
        m = CharacterStatusManager(on_status_changed=cb1)
        m.subscribe(cb2)
        m.set_status("mimi", CharacterStatus.THINKING)
        cb1.assert_called_once()
        cb2.assert_called_once()
