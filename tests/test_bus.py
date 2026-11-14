"""
test_bus.py — bus.publish テスト

redis.from_url をモックして実際の Redis 接続なしで検証する。
Redis に届かないときの上限は、OS が割り当てる範囲の閉じたポートへ本物のクライアントで接続して測る。
"""

import json
import logging
import socket
import time
from unittest.mock import MagicMock, patch

import pytest
import redis
from redis.retry import Retry

import lab_lounge.bus as bus_mod
import lab_lounge.pipeline as pipeline_mod
from lab_lounge.bus import BusUnavailableError, _summarize_event, publish, publish_best_effort

SAMPLE_EVENT = {
    "ver": "0.1",
    "event_id": "11111111-2222-4333-8444-555555555555",
    "ts": "2026-03-08T10:00:00.000Z",
    "stream_id": "stream-bus-001",
    "session_id": "sess-bus-001",
    "trace_id": "trace-bus-001",
    "type": "utterance.final",
    "source": "lab-lounge",
    "payload": {"text": "テスト"},
}


@pytest.fixture()
def mock_redis_client():
    client = MagicMock()
    # decode_responses=False のとき xadd は bytes を返す
    client.xadd.return_value = b"1741000000000-0"
    with patch.object(bus_mod.redis, "from_url", return_value=client) as _:
        yield client


class TestPublish:
    def test_xadd_called_once(self, mock_redis_client):
        publish(SAMPLE_EVENT)
        mock_redis_client.xadd.assert_called_once()

    def test_xadd_field_name_is_event(self, mock_redis_client):
        """C2 との規約: フィールド名は "event" 固定"""
        publish(SAMPLE_EVENT)
        _, kwargs = mock_redis_client.xadd.call_args
        # xadd(stream, fields) の positional arg として渡される場合も考慮
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        assert "event" in fields

    def test_xadd_value_is_valid_json(self, mock_redis_client):
        """フィールド値が Event Envelope JSON 文字列になっている"""
        publish(SAMPLE_EVENT)
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        parsed = json.loads(fields["event"])
        assert parsed["event_id"] == SAMPLE_EVENT["event_id"]

    def test_xadd_stream_key_default(self, mock_redis_client, monkeypatch):
        monkeypatch.delenv("REDIS_STREAM_KEY", raising=False)
        publish(SAMPLE_EVENT)
        call_args = mock_redis_client.xadd.call_args
        stream = call_args.args[0] if call_args.args else call_args.kwargs.get("name")
        assert stream == "aibyss:events"

    def test_returns_msg_id(self, mock_redis_client):
        """bytes の msg_id が str として返る"""
        msg_id = publish(SAMPLE_EVENT)
        assert msg_id == "1741000000000-0"

    def test_client_closed_after_publish(self, mock_redis_client):
        publish(SAMPLE_EVENT)
        mock_redis_client.close.assert_called_once()

    def test_japanese_text_preserved(self, mock_redis_client):
        """日本語テキストが UTF-8 JSON バイト列として送られ、decode しても元の文字列に戻る"""
        japanese_event = dict(SAMPLE_EVENT)
        japanese_event["payload"] = {"text": "今日の天気を教えて"}
        publish(japanese_event)
        call_args = mock_redis_client.xadd.call_args
        fields = call_args.args[1] if call_args.args else call_args.kwargs.get("fields", {})
        raw = fields["event"]
        # bytes として受け取り UTF-8 decode → JSON parse しても日本語が壊れない
        assert isinstance(raw, bytes), "publish は UTF-8 bytes を送るべき"
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["payload"]["text"] == "今日の天気を教えて"

    def test_log_includes_summary(self, mock_redis_client, caplog):
        """ログ強化 L-1: publish 時のログに type 別 summary が含まれる。

        bubble.update なら character / step / category が、調査時の手掛かりとして
        パッと分かる形で出ること。
        """
        bubble_event = {
            **SAMPLE_EVENT,
            "type": "bubble.update",
            "payload": {
                "character": "mimi",
                "step": "thinking",
                "text": "考えていますわ",
                "category": "speech_status",
            },
        }
        with caplog.at_level(logging.INFO, logger="lab_lounge.bus"):
            publish(bubble_event)
        # 「published type=bubble.update [character=mimi step=thinking category=speech]」のような形
        msg = "\n".join(rec.message for rec in caplog.records)
        assert "type=bubble.update" in msg
        assert "character=mimi" in msg
        assert "step=thinking" in msg
        assert "category=speech" in msg


# ─── ログ強化 L-1: _summarize_event の type 別挙動 ─────────────────


class TestSummarizeEvent:
    """_summarize_event が type 別に payload から手掛かりを抽出することを検証する。

    Phase 0.5-A 後のログ強化 L-1: 実走ログで「どのキャラの動作か」を即座に
    把握できるようにする目的。
    """

    def test_utterance_final_includes_text_preview(self):
        """utterance.final はルカ発話 → text 先頭でターン識別。"""
        ev = {
            "type": "utterance.final",
            "payload": {"text": "最近のAI倫理について気になっています"},
        }
        s = _summarize_event(ev)
        assert "text=" in s
        assert "最近のAI倫理について" in s  # 先頭部分
        assert "chars=" in s

    def test_utterance_final_long_text_truncated(self):
        """30 文字超の text は ... で切り詰める (ログ可読性)。"""
        long_text = "あ" * 100
        ev = {"type": "utterance.final", "payload": {"text": long_text}}
        s = _summarize_event(ev)
        assert "..." in s
        assert "chars=100" in s

    def test_llm_final_includes_character_when_present(self):
        """llm.final に character フィールドがあれば summary に含まれる (L-2 後対応)。"""
        ev = {
            "type": "llm.final",
            "payload": {"character": "mimi", "model": "gpt-5.5", "text": "あら、ルカ"},
        }
        s = _summarize_event(ev)
        assert "character=mimi" in s
        assert "model=gpt-5.5" in s
        assert "text_len=" in s

    def test_llm_final_character_unknown_when_missing(self):
        """character フィールド無しは ? 表示 (= 「明示的に欠けている」と分かる)。"""
        ev = {"type": "llm.final", "payload": {"model": "x", "text": ""}}
        s = _summarize_event(ev)
        assert "character=?" in s

    def test_tts_done_uses_character_or_speaker(self):
        """tts.done は character > speaker の優先順位で character を表示。"""
        ev1 = {
            "type": "tts.done",
            "payload": {"character": "sakura", "speaker": "sakura", "duration_ms": 1234},
        }
        s1 = _summarize_event(ev1)
        assert "character=sakura" in s1
        assert "duration_ms=1234" in s1

        # character 無しなら speaker にフォールバック
        ev2 = {"type": "tts.done", "payload": {"speaker": "chisame", "duration_ms": 999}}
        s2 = _summarize_event(ev2)
        assert "character=chisame" in s2

    def test_bubble_update_includes_character_step_category(self):
        """bubble.update は character / step / category を全部表示。"""
        ev = {
            "type": "bubble.update",
            "payload": {
                "character": "sakura",
                "step": "handraise",
                "text": "あのぉ、ちょっといいですかぁ",
                "category": "raisehand",
            },
        }
        s = _summarize_event(ev)
        assert "character=sakura" in s
        assert "step=handraise" in s
        assert "category=raisehand" in s

    def test_bubble_update_default_category_speech_status(self):
        """category 省略時は default の "speech_status" 表示 (= Phase 0.5-E、受信側 default 解釈と一致)。"""
        ev = {
            "type": "bubble.update",
            "payload": {"character": "mimi", "step": "thinking", "text": "..."},
        }
        s = _summarize_event(ev)
        assert "category=speech_status" in s

    def test_dispatcher_queue_update_lists_slugs(self):
        """dispatcher.queue.update は queue 内の character_slug 一覧 + state を表示。"""
        ev = {
            "type": "dispatcher.queue.update",
            "payload": {
                "queue": [
                    {"character_slug": "mimi", "transcript": "..."},
                    {"character_slug": "chisame", "transcript": "..."},
                ],
                "state": "responding",
            },
        }
        s = _summarize_event(ev)
        assert "state=responding" in s
        assert "queue_size=2" in s
        assert "mimi" in s
        assert "chisame" in s

    def test_dispatcher_handraise_update_lists_slugs(self):
        """dispatcher.handraise.update は挙手中キャラ + cooldown キーを表示。"""
        ev = {
            "type": "dispatcher.handraise.update",
            "payload": {
                "handraise_states": [
                    {"target_slug": "sakura", "phrase": "..."},
                ],
                "cooldowns": {"mimi": {"consecutive_denials": 1}},
            },
        }
        s = _summarize_event(ev)
        assert "handraise_count=1" in s
        assert "sakura" in s
        assert "mimi" in s

    def test_unknown_type_returns_no_summary(self):
        """未知の type は (no summary) を返して後方互換維持。"""
        ev = {"type": "unknown.weird", "payload": {"some": "data"}}
        s = _summarize_event(ev)
        assert s == "(no summary)"

    def test_missing_payload_handled(self):
        """payload が無い event でも例外で死なない。"""
        ev = {"type": "bubble.update"}
        s = _summarize_event(ev)
        # payload 無し → デフォルト値 (?) でフォールバック
        assert "character=?" in s

    # ─── character.status.update (Phase 0.5-B-α) ──────────────────

    def test_character_status_update_includes_status_transition(self):
        """character.status.update: previous_status -> status の形式で summary。"""
        ev = {
            "type": "character.status.update",
            "payload": {
                "character": "mimi",
                "status": "thinking",
                "previous_status": "ready",
            },
        }
        s = _summarize_event(ev)
        assert "character=mimi" in s
        assert "status=ready->thinking" in s

    def test_character_status_update_includes_metadata_pose_and_text(self):
        """Talking 時の metadata (pose + text) が summary に含まれる。"""
        ev = {
            "type": "character.status.update",
            "payload": {
                "character": "mimi",
                "status": "talking",
                "previous_status": "thinking",
                "metadata": {"pose": "smile", "text": "こんにちは、ルカさま"},
            },
        }
        s = _summarize_event(ev)
        assert "pose=smile" in s
        assert "text=" in s
        assert "こんにちは" in s  # 先頭部分 (truncate されても残る)

    def test_character_status_update_long_text_truncated(self):
        """長文 text は preview chars (30) で truncate される。"""
        long_text = "深海" * 100  # 200 文字
        ev = {
            "type": "character.status.update",
            "payload": {
                "character": "mimi", "status": "talking",
                "metadata": {"pose": "smile", "text": long_text},
            },
        }
        s = _summarize_event(ev)
        assert "..." in s  # truncate suffix
        assert len(s) < 200  # 元の text より短い

    def test_character_status_update_default_unknowns(self):
        """payload 不完全時 character=? status=?->? で fallback。"""
        ev = {"type": "character.status.update", "payload": {}}
        s = _summarize_event(ev)
        assert "character=?" in s
        assert "status=?->?" in s


# ─── P3: Redis に届かないときの上限、失敗の後の待ち、fail-open ─────────────


def _closed_port() -> int:
    """何も待ち受けていない loopback のポートを返す。

    OS に割り当てさせたポートをすぐ閉じる。conftest は OS が割り当てる範囲のポートへの接続を通すので、
    本物の接続の失敗 (Windows では拒否まで約 2 秒) をそのまま測れる。実サービスのポートには触れない。
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def fail_open_default(monkeypatch):
    """本番の既定 (L2_BUS_FAIL_OPEN 未設定 = 真) に戻す。conftest はモックの漏れを赤くするため false にしている。"""
    monkeypatch.delenv("L2_BUS_FAIL_OPEN", raising=False)


@pytest.fixture()
def unreachable_redis():
    """XADD が接続の失敗になる Redis。from_url のモックを返し、接続を試した回数を数えられる。"""
    client = MagicMock()
    client.xadd.side_effect = redis.ConnectionError("Error 10061 connecting to 127.0.0.1:6379.")
    with patch.object(bus_mod.redis, "from_url", return_value=client) as from_url:
        yield from_url


class TestConnectionLimits:
    """接続と応答に上限を置き、再試行しない。"""

    def test_closed_port_fails_within_one_second(self, monkeypatch):
        """旧コードは閉じたポートで毎回約 2.05 秒待ってから ConnectionError になった (Windows、redis-py 7.3.0)。

        Windows は接続の上限 (0.5 秒) で TimeoutError、即座に拒否する OS では ConnectionError になる。
        """
        monkeypatch.setenv("REDIS_URL", f"redis://127.0.0.1:{_closed_port()}")
        started = time.monotonic()
        with pytest.raises((redis.ConnectionError, redis.TimeoutError)):
            publish(SAMPLE_EVENT)
        assert time.monotonic() - started < 1.0

    def test_connection_arguments_are_explicit(self, mock_redis_client):
        """既定に任せない引数。protocol は戻りの形、retry は待ちの積み上がりと二重の XADD に効く。"""
        publish(SAMPLE_EVENT)
        kwargs = bus_mod.redis.from_url.call_args.kwargs
        assert kwargs["decode_responses"] is False
        assert kwargs["protocol"] == 2
        assert kwargs["socket_connect_timeout"] == 0.5
        assert kwargs["socket_timeout"] == 1.0
        assert isinstance(kwargs["retry"], Retry)
        assert kwargs["retry"].get_retries() == 0

    def test_limits_follow_the_env(self, mock_redis_client, monkeypatch):
        monkeypatch.setenv("L2_BUS_CONNECT_TIMEOUT_SEC", "0.25")
        monkeypatch.setenv("L2_BUS_SOCKET_TIMEOUT_SEC", "0.75")
        publish(SAMPLE_EVENT)
        kwargs = bus_mod.redis.from_url.call_args.kwargs
        assert kwargs["socket_connect_timeout"] == 0.25
        assert kwargs["socket_timeout"] == 0.75

    def test_client_is_closed_even_when_xadd_fails(self, unreachable_redis):
        with pytest.raises(redis.ConnectionError):
            publish(SAMPLE_EVENT)
        unreachable_redis.return_value.close.assert_called_once()


class TestBackoffAfterFailure:
    """Redis に届かなかった後 L2_BUS_RETRY_AFTER_SEC 秒は、接続を試さずにすぐ諦める。"""

    def test_does_not_connect_right_after_a_failure(self, unreachable_redis):
        with pytest.raises(redis.ConnectionError):
            publish(SAMPLE_EVENT)
        started = time.monotonic()
        with pytest.raises(BusUnavailableError, match="L2_BUS_RETRY_AFTER_SEC"):
            publish(SAMPLE_EVENT)
        assert time.monotonic() - started < 0.1
        assert unreachable_redis.call_count == 1

    def test_backoff_error_is_a_connection_error(self):
        """呼び出し側の既存の except (Exception や redis.ConnectionError) が、今までどおり受けられる。"""
        assert issubclass(BusUnavailableError, redis.ConnectionError)

    def test_timeout_also_starts_the_wait(self):
        client = MagicMock()
        client.xadd.side_effect = redis.TimeoutError("Timeout reading from socket")
        with patch.object(bus_mod.redis, "from_url", return_value=client) as from_url:
            with pytest.raises(redis.TimeoutError):
                publish(SAMPLE_EVENT)
            with pytest.raises(BusUnavailableError):
                publish(SAMPLE_EVENT)
        assert from_url.call_count == 1

    def test_connects_again_after_the_wait(self, unreachable_redis, monkeypatch):
        monkeypatch.setenv("L2_BUS_RETRY_AFTER_SEC", "0.2")
        with pytest.raises(redis.ConnectionError):
            publish(SAMPLE_EVENT)
        time.sleep(0.3)
        with pytest.raises(redis.ConnectionError) as raised:
            publish(SAMPLE_EVENT)
        assert not isinstance(raised.value, BusUnavailableError)
        assert unreachable_redis.call_count == 2

    def test_zero_turns_the_wait_off(self, unreachable_redis, monkeypatch):
        monkeypatch.setenv("L2_BUS_RETRY_AFTER_SEC", "0")
        for _ in range(3):
            with pytest.raises(redis.ConnectionError) as raised:
                publish(SAMPLE_EVENT)
            assert not isinstance(raised.value, BusUnavailableError)
        assert unreachable_redis.call_count == 3

    def test_success_after_the_wait_publishes_normally(self, monkeypatch):
        """Redis が戻れば、次の publish から届く。"""
        monkeypatch.setenv("L2_BUS_RETRY_AFTER_SEC", "0.2")
        client = MagicMock()
        client.xadd.side_effect = [redis.ConnectionError("down"), b"1-0", b"2-0"]
        with patch.object(bus_mod.redis, "from_url", return_value=client):
            with pytest.raises(redis.ConnectionError):
                publish(SAMPLE_EVENT)
            time.sleep(0.3)
            assert publish(SAMPLE_EVENT) == "1-0"
            assert publish(SAMPLE_EVENT) == "2-0"

    def test_rejected_command_does_not_start_the_wait(self):
        """応答のエラーはすぐ返るので、次の publish を止める理由がない。"""
        client = MagicMock()
        client.xadd.side_effect = redis.ResponseError("OOM command not allowed when used memory > 'maxmemory'.")
        with patch.object(bus_mod.redis, "from_url", return_value=client) as from_url:
            for _ in range(2):
                with pytest.raises(redis.ResponseError):
                    publish(SAMPLE_EVENT)
        assert from_url.call_count == 2


class TestPublishBestEffort:
    """本流の event の publish。Redis の障害では、既定でターンを止めない。"""

    def test_returns_msg_id_on_success(self, mock_redis_client):
        assert publish_best_effort(SAMPLE_EVENT) == "1741000000000-0"

    def test_fail_open_by_default(self, fail_open_default, unreachable_redis, caplog):
        with caplog.at_level(logging.ERROR, logger="lab_lounge.bus"):
            assert publish_best_effort(SAMPLE_EVENT) is None
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert "type=utterance.final" in message
        assert SAMPLE_EVENT["event_id"] in message
        assert "ConnectionError" in message
        # 発話の本文はエラーのログに出さない (成功のログの要約だけが先頭を出す)
        assert "テスト" not in message

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_truthy_values_fail_open(self, unreachable_redis, monkeypatch, value):
        monkeypatch.setenv("L2_BUS_FAIL_OPEN", value)
        assert publish_best_effort(SAMPLE_EVENT) is None

    @pytest.mark.parametrize("value", ["false", "0", "no", ""])
    def test_fail_closed_raises(self, unreachable_redis, monkeypatch, value):
        """false で P3 より前の挙動 (例外でターンを止める) に戻す。空も、他の既定が真のフラグと同じく偽。"""
        monkeypatch.setenv("L2_BUS_FAIL_OPEN", value)
        with pytest.raises(redis.ConnectionError):
            publish_best_effort(SAMPLE_EVENT)

    def test_backoff_is_swallowed_too(self, fail_open_default, unreachable_redis):
        assert publish_best_effort(SAMPLE_EVENT) is None
        assert publish_best_effort(SAMPLE_EVENT) is None
        assert unreachable_redis.call_count == 1

    def test_rejected_command_is_swallowed(self, fail_open_default):
        client = MagicMock()
        client.xadd.side_effect = redis.ResponseError("MISCONF Redis is configured to save RDB snapshots")
        with patch.object(bus_mod.redis, "from_url", return_value=client):
            assert publish_best_effort(SAMPLE_EVENT) is None

    def test_code_errors_are_not_swallowed(self, fail_open_default, mock_redis_client):
        """JSON にできない event はコードの誤り。Redis の障害の ERROR ログに紛れさせない。"""
        broken = {**SAMPLE_EVENT, "payload": {"text": object()}}
        with pytest.raises(TypeError):
            publish_best_effort(broken)

    def test_closed_port_costs_one_wait_per_window(self, fail_open_default, monkeypatch):
        """本物のクライアントで: 1 回目だけ接続の上限まで待ち、続く publish は接続を試さずに済む。"""
        monkeypatch.setenv("REDIS_URL", f"redis://127.0.0.1:{_closed_port()}")
        started = time.monotonic()
        for _ in range(3):
            assert publish_best_effort(SAMPLE_EVENT) is None
        assert time.monotonic() - started < 1.0


class TestTurnContinuesWithoutRedis:
    """Redis が落ちていても、1 ターンが TTS の合成まで進む (P3 の目的: 声は止まらない)。"""

    _COMMON = dict(stream_id="stream-bus-002", session_id="sess-bus-002", trace_id="trace-bus-002")

    @pytest.fixture()
    def fake_synthesize(self, monkeypatch):
        """本物の TTS の経路を通し、合成だけを身代わりにする。"""
        from lab_lounge.tts import TTSResult

        monkeypatch.setenv("L2_USE_REAL_TTS", "true")
        monkeypatch.setenv("L2_TTS_PROVIDER", "edge_tts")
        monkeypatch.setenv("L2_TTS_VOICE", "ja-JP-NanamiNeural")
        monkeypatch.setenv("L2_TTS_SPEAKER", "Nanami")
        result = TTSResult(
            audio_url="file:///tmp/tts-bus-test.mp3", duration_ms=1200, voice="ja-JP-NanamiNeural",
            format="mp3", sample_rate=24000, speaker="Nanami",
        )
        with patch("lab_lounge.tts.synthesize", return_value=result) as synthesize:
            yield synthesize

    @pytest.fixture()
    def redis_down(self):
        """すべての publish が接続の失敗になる。本流の 3 件は bus.publish、吹き出しは pipeline.publish を通る。"""
        failing = MagicMock(side_effect=redis.ConnectionError("Redis is down"))
        with patch.object(bus_mod, "publish", failing), patch.object(pipeline_mod, "publish", failing):
            yield failing

    def test_turn_reaches_tts_when_every_publish_fails(self, fail_open_default, fake_synthesize, redis_down):
        result = pipeline_mod.run_pipeline("テスト", **self._COMMON)
        fake_synthesize.assert_called_once()
        assert [event["type"] for event in result.events] == ["utterance.final", "llm.final", "tts.done"]
        # 本流の 3 件と吹き出しの 2 件 (thinking / answering) のすべてで publish を試している
        assert redis_down.call_count == 5

    def test_fail_closed_stops_the_turn_before_tts(self, fake_synthesize, redis_down, monkeypatch):
        """L2_BUS_FAIL_OPEN=false で、P3 より前と同じく utterance.final の publish でターンが止まる。"""
        monkeypatch.setenv("L2_BUS_FAIL_OPEN", "false")
        with pytest.raises(redis.ConnectionError):
            pipeline_mod.run_pipeline("テスト", **self._COMMON)
        fake_synthesize.assert_not_called()
