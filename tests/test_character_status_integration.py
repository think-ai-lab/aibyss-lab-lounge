"""
test_character_status_integration.py — CharacterStatus 統合シナリオテスト (Phase 0.5-B-α)

個別 commit の単体テストでは捉えにくい end-to-end / 並行シナリオを統合検証する。
Manager + Dispatcher + run_loop の組み合わせで状態遷移と metadata の流れを確認。

シナリオ:
1. 挙手承認 → Talking (with metadata) → Ready の lifecycle
2. 通常応答中の concurrent handraise (= 2 キャラ独立管理)
3. get_snapshot による全キャラ状態の dashboard view
4. 長文 metadata.text の保持 (= truncation なし)
5. publish 順序 (= status.update → handraise.update の HUD 前提)
"""

import threading
from unittest.mock import MagicMock

from lab_lounge.character_status import CharacterStatus, CharacterStatusManager
from lab_lounge.dispatcher import Dispatcher
from lab_lounge.run_loop import _spawn_handraise_response_playback


class _FakeTimer:
    """threading.Timer の決定論的差し替え (= test_dispatcher.py と同じパターン)。"""

    def __init__(self, delay: float, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True


def _make_dispatcher_with_status(monkeypatch, manager: CharacterStatusManager) -> Dispatcher:
    """status_manager 注入 + lapse_timer を差し替えた Dispatcher を生成する helper。"""
    d = Dispatcher(status_manager=manager)
    monkeypatch.setattr(
        d, "_create_lapse_timer",
        lambda slug, sec: _FakeTimer(sec, lambda: None),
    )
    return d


# ─── シナリオ 1: 挙手 → Talking → Ready の lifecycle ─────────────


class TestHandraiseToTalkingFlow:
    """挙手 → 承認 → Talking (with metadata) → Ready の遷移を統合検証 (Phase 0.5-B-α)。

    Dispatcher が Raisehand を反映、run_loop の _spawn_handraise_response_playback が
    Talking metadata を含めて反映、worker 完了で Ready に戻る一連の流れを 1 件のテストで
    確認する。実 LLM / TTS / audio は使わず、chunks=[] で worker を即終了させる。
    """

    def test_handraise_approval_to_talking_with_metadata(self, monkeypatch):
        """Raisehand → Ready (一瞬) → Talking (metadata 付き) → Ready の遷移を検証。"""
        callback = MagicMock()
        manager = CharacterStatusManager(on_status_changed=callback)

        # Phase 1: Dispatcher で挙手 → Raisehand
        d = _make_dispatcher_with_status(monkeypatch, manager)
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        assert manager.get_status("mimi") == CharacterStatus.RAISEHAND

        # Phase 2: 承認 → Raisehand → Ready
        d.on_approval_granted("mimi")
        assert manager.get_status("mimi") == CharacterStatus.READY

        # Phase 3: TTS chunks 再生開始 → Talking (with metadata)
        talking_metadata = {
            "pose": "smile",
            "text": "こんにちは、ルカさま。AI に心はあるのか、面白い問いですね…",
        }
        t = _spawn_handraise_response_playback(
            slug="mimi",
            chunks=[],  # 空 → worker 即終了
            trace_id="trace1",
            session_stream_id="s1",
            session_id_root="ss1",
            status_manager=manager,
            talking_metadata=talking_metadata,
        )
        t.join(timeout=2.0)

        # Phase 4: worker finally で Ready に戻る
        assert manager.get_status("mimi") == CharacterStatus.READY

        # callback 履歴で全遷移を検証
        # READY → RAISEHAND → READY → TALKING → READY の 4 回 publish
        calls = callback.call_args_list
        assert len(calls) == 4

        # 1: ready → raisehand
        assert calls[0].args[1] == CharacterStatus.RAISEHAND
        assert calls[0].args[2] == CharacterStatus.READY
        # 2: raisehand → ready (承認時)
        assert calls[1].args[1] == CharacterStatus.READY
        assert calls[1].args[2] == CharacterStatus.RAISEHAND
        # 3: ready → talking (with metadata)
        assert calls[2].args[1] == CharacterStatus.TALKING
        assert calls[2].args[2] == CharacterStatus.READY
        assert calls[2].args[3] == talking_metadata
        # 4: talking → ready (worker finally)
        assert calls[3].args[1] == CharacterStatus.READY
        assert calls[3].args[2] == CharacterStatus.TALKING


# ─── シナリオ 2: 並行 handraise (= 2 キャラ独立管理) ─────────────


class TestConcurrentCharacterStates:
    """通常応答中の chisame=Talking + 並行で mimi=Raisehand を同時保持できる検証。

    HUD dashboard の「全キャラ一覧」要件: 1 キャラずつではなく、複数キャラの状態を
    同時に追跡する設計が必要。Manager._statuses dict による独立管理を確認。
    """

    def test_two_characters_independent_states(self, monkeypatch):
        """chisame=Talking と mimi=Raisehand を同時に保持できる。"""
        manager = CharacterStatusManager()
        d = _make_dispatcher_with_status(monkeypatch, manager)

        # Step 1: chisame が応答中 (= Talking 状態を直接 set、通常応答経路の simulation)
        manager.set_status(
            "chisame", CharacterStatus.TALKING,
            metadata={"pose": "neutral", "text": "AI 倫理について…"},
        )

        # Step 2: 並行で mimi が挙手
        d.on_interjection_candidate("mimi", transcript_snapshot=None)

        # 両キャラの状態が独立に保持されていること
        assert manager.get_status("chisame") == CharacterStatus.TALKING
        assert manager.get_status("mimi") == CharacterStatus.RAISEHAND

        # metadata も独立保持
        chisame_meta = manager.get_metadata("chisame")
        assert chisame_meta is not None
        assert chisame_meta["pose"] == "neutral"
        assert manager.get_metadata("mimi") is None  # Raisehand は metadata なし

    def test_get_snapshot_dashboard_view(self, monkeypatch):
        """HUD dashboard 用: 全キャラの状態を 1 回の get_snapshot で取得できる。"""
        manager = CharacterStatusManager()
        d = _make_dispatcher_with_status(monkeypatch, manager)

        # 複数キャラの状態を作る
        manager.set_status(
            "chisame", CharacterStatus.TALKING,
            metadata={"pose": "smile", "text": "..."},
        )
        d.on_interjection_candidate("mimi", transcript_snapshot=None)
        manager.set_status("sakura", CharacterStatus.THINKING)

        # snapshot で全キャラ取得
        snap = manager.get_snapshot()

        # 3 キャラすべて含まれる
        assert set(snap.keys()) == {"chisame", "mimi", "sakura"}
        assert snap["chisame"]["status"] == "talking"
        assert snap["chisame"]["metadata"]["pose"] == "smile"
        assert snap["mimi"]["status"] == "raisehand"
        assert snap["mimi"]["metadata"] is None
        assert snap["sakura"]["status"] == "thinking"
        assert snap["sakura"]["metadata"] is None


# ─── シナリオ 3: 長文 metadata.text の保持 ────────────────────────


class TestMetadataTextPreservation:
    """LLM response の長文テキストが metadata.text に full で保持される検証。

    ルカ要件「(全文でOK)」を踏まえ、Manager / event payload で truncation が
    起きないことを確認。bus.py の _summarize_event でログ表示時に truncate するが、
    それは「ログの可読性」目的で、event payload 自体は変更しない設計。
    """

    def test_long_text_preserved_in_metadata(self):
        """500 文字超の発話 text が metadata.text に full で保持される。"""
        manager = CharacterStatusManager()

        # 配信中によくある長文応答 (深海ラボの研究談義の想定)
        long_text = (
            "深海というのは、本当に未知に満ちた場所ですよね。"
            "光の届かない場所で生体発光が美しく輝いて、"
            "AI 技術もまさにそういう「深く潜るほど発見がある」世界だと感じます。"
            "ルカさまと一緒にこの好奇心の海を泳ぐのが、わたくしの何よりの喜びですわ。"
        ) * 3  # ~600 文字

        manager.set_status(
            "mimi", CharacterStatus.TALKING,
            metadata={"pose": "smile", "text": long_text},
        )

        # metadata に full text が保持されている (= truncate されない)
        meta = manager.get_metadata("mimi")
        assert meta is not None
        assert meta["text"] == long_text
        # 数百文字レベルの長文が full で保持されることを明示
        assert len(meta["text"]) > 200


# ─── シナリオ 4: publish 順序検証 ───────────────────────────


class TestPublishOrder:
    """status.update → handraise.update の publish 順序検証 (HUD 前提)。

    Dispatcher の _start_handraise が Phase 0.5-B-α で:
      1. status_manager.set_status(RAISEHAND) → publish character.status.update
      2. _publish_bubble_update("handraise") → publish bubble.update
      3. _publish_handraise_update() → publish dispatcher.handraise.update

    の順で発行する。HUD は status.update を先に観察してから handraise の詳細を
    処理する想定なので、この順序が崩れると HUD ロジックが破綻する。
    """

    def test_status_published_before_handraise_update(self, monkeypatch):
        """挙手時、status_manager の publish が handraise.update より先。"""
        publish_log: list[str] = []

        # 全 callback を spy 化して順序を記録
        manager = CharacterStatusManager(
            on_status_changed=lambda slug, new, old, meta: publish_log.append(
                f"status.update:{slug}:{new.value}",
            ),
        )

        d = Dispatcher(
            status_manager=manager,
            on_handraise_update=lambda states, cooldowns: publish_log.append(
                "handraise.update",
            ),
            on_bubble_update=lambda c, s, t, ttl, cat: publish_log.append(
                f"bubble.update:{s}",
            ),
        )
        monkeypatch.setattr(
            d, "_create_lapse_timer",
            lambda slug, sec: _FakeTimer(sec, lambda: None),
        )

        d.on_interjection_candidate("mimi", transcript_snapshot=None)

        # 順序: status.update (raisehand) → bubble.update(handraise) → handraise.update
        assert publish_log == [
            "status.update:mimi:raisehand",
            "bubble.update:handraise",
            "handraise.update",
        ]
