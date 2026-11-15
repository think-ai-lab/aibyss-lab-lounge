"""
bus.py — Redis Streams Publisher

責務:
  - Event Envelope を JSON 文字列化して Redis Stream に XADD する
  - フィールド名は "event" 固定 (C2 Consumer との規約)
  - C2 への HTTP POST は絶対に行わない (Guardrail G-2)
  - Redis が落ちていても声 (ターン) を止めない

【Redis Streams メッセージ規約 (v0.1)】
  XADD <stream_key> * event '<Event Envelope JSON 文字列>'

【Redis に届かないとき】
  - publish は接続を L2_BUS_CONNECT_TIMEOUT_SEC (既定 0.5 秒)、応答を L2_BUS_SOCKET_TIMEOUT_SEC
    (既定 1.0 秒) で諦める。再試行はしない
  - 接続や応答で失敗した後 L2_BUS_RETRY_AFTER_SEC (既定 5 秒) の間は、接続を試さずにすぐ
    BusUnavailableError にする。1 ターンに publish は何度もある (本流の 3 件に吹き出しと状態の更新) ので、
    毎回待つと声が数秒遅れる
  - ターンの本流 (utterance.final / llm.final / tts.done) は publish_best_effort を使う。
    L2_BUS_FAIL_OPEN が真 (既定) なら ERROR ログを残して先へ進み、偽ならターンを例外で止める
  - 諦めた event は再送しない。正史 (C2) と HUD (V2) に穴が開くことは許容する
    (ルカの決定: Redis の障害時は声を止めない)
"""

import json
import logging
import os
import threading
import time
from typing import Any

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

logger = logging.getLogger(__name__)

# 日本語テキストをログに含めるとき、長すぎると 1 行が見にくいので先頭のみ。
# トラブル調査では「どの発話で何が起きたか」が分かれば十分なので 30 文字で打ち切る。
_LOG_TEXT_PREVIEW_CHARS = 30

# 「Redis に届かない、または答えない」とみなす例外。これが起きたら、しばらく接続を試さない。
# WHY: どれも次の publish が同じだけ待たされる失敗。Redis がコマンドを拒む応答のエラー
#      (ResponseError: メモリ不足や永続化の失敗など) はすぐ返るので、試さずに済ませる理由がなく、ここに含めない。
#      redis-py は socket の例外を ConnectionError / TimeoutError に包むが、包まれずに出る OSError も同じ扱いにする。
_UNREACHABLE_ERRORS = (redis.ConnectionError, redis.TimeoutError, OSError)

# 直前に Redis へ届かなかった時刻 (time.monotonic())。None なら、まだ失敗していないか、その後に成功した。
# WHY ロック: publish は録音、再生、Dispatcher などのスレッドから同時に呼ばれる。
_failure_lock = threading.Lock()
_last_failure_at: float | None = None


class BusUnavailableError(redis.ConnectionError):
    """直前の失敗から L2_BUS_RETRY_AFTER_SEC 秒以内なので、接続を試さずに諦めた。

    redis.ConnectionError の子にする理由: 呼び出し側の既存の except (Exception や redis.ConnectionError) が、
    今までの接続の失敗と同じように受けられるようにするため。
    """


def _redis_url() -> str:
    # WHY: Windows の localhost は ::1 を先に試すので、IPv4 だけで待ち受ける Redis だと
    #      接続ごとに約 2 秒失う。loopback の IPv4 を直接指す。
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")


def _stream_key() -> str:
    return os.environ.get("REDIS_STREAM_KEY", "aibyss:events")


def _connect_timeout_sec() -> float:
    """Redis への接続を待つ上限 (秒)。L2_BUS_CONNECT_TIMEOUT_SEC で変えられる。

    WHY: Windows は閉じたポートへの接続を拒むまでに約 2 秒かかる (実測 2.05 秒)。上限がないと、
         Redis が落ちている間は publish のたびにそのターンの声が 2 秒ずつ遅れる。loopback の Redis なら、
         接続は 1 ms もかからないので 0.5 秒で足りる。
    """
    return float(os.environ.get("L2_BUS_CONNECT_TIMEOUT_SEC", "0.5"))


def _socket_timeout_sec() -> float:
    """XADD の応答を待つ上限 (秒)。L2_BUS_SOCKET_TIMEOUT_SEC で変えられる。

    WHY: Redis が接続を受けたまま答えないと (CLIENT PAUSE やディスクの詰まり)、上限がなければ publish が返らない。
         待ち切れずに諦めた XADD は、Redis の側では遅れて入ることがある (再送しないので二重にはならない)。
    """
    return float(os.environ.get("L2_BUS_SOCKET_TIMEOUT_SEC", "1.0"))


def _retry_after_sec() -> float:
    """Redis に届かなかった後、接続を試さずに諦める時間 (秒)。L2_BUS_RETRY_AFTER_SEC で変えられる。0 で無効。"""
    return float(os.environ.get("L2_BUS_RETRY_AFTER_SEC", "5"))


def _fail_open() -> bool:
    """本流の publish が Redis の障害で失敗しても、ターンを続けるなら True (既定)。

    L2 の他の既定が真のフラグと同じ読み方。空は既定へ戻らず偽 (ターンを止める) になる。
    """
    return os.environ.get("L2_BUS_FAIL_OPEN", "true").lower() in ("1", "true", "yes")


def _raise_if_backing_off() -> None:
    """直前の失敗から L2_BUS_RETRY_AFTER_SEC 秒以内なら、接続を試さずに BusUnavailableError を送出する。"""
    with _failure_lock:
        failed_at = _last_failure_at
    if failed_at is None:
        return
    elapsed = time.monotonic() - failed_at
    wait = _retry_after_sec()
    if elapsed < wait:
        raise BusUnavailableError(
            f"{elapsed:.1f} 秒前に Redis へ届かなかったので、{wait:g} 秒たつまで接続を試しません"
            " (L2_BUS_RETRY_AFTER_SEC)"
        )


def _record_unreachable() -> None:
    """Redis に届かなかった時刻を記録する。以後 L2_BUS_RETRY_AFTER_SEC 秒は接続を試さない。"""
    global _last_failure_at
    with _failure_lock:
        _last_failure_at = time.monotonic()


def _record_reachable() -> None:
    """Redis に届いたので、失敗の記録を消す。"""
    global _last_failure_at
    with _failure_lock:
        _last_failure_at = None


def _reset_for_tests() -> None:
    """失敗の記録を消す。テストの間で状態を持ち越さないため (tests/conftest.py の autouse fixture から呼ぶ)。"""
    _record_reachable()


def _new_client(url: str) -> redis.Redis:
    """publish 1 回分のクライアントを作る。接続は最初のコマンドで張られる。

    明示する引数と WHY:
      - decode_responses=False: 送受信ともにバイト列を明示的に扱う
      - protocol=2: 既定に任せず RESP2 に固定する。C2 の consumer と同じく、redis-py の既定が変わっても
        戻りの形を変えないため
      - socket_connect_timeout / socket_timeout: _connect_timeout_sec と _socket_timeout_sec を参照
      - retry=Retry(NoBackoff(), 0): 再試行しない。redis-py 7.3 の既定は作り方で違い (from_url は 0 回、
        Redis(...) は 3 回)、版の更新で変わりうるので固定する。再試行すると待ちが回数分だけ積み上がり、
        応答を待ち切れなかった XADD を二重に入れるおそれもある
    """
    return redis.from_url(
        url,
        decode_responses=False,
        protocol=2,
        socket_connect_timeout=_connect_timeout_sec(),
        socket_timeout=_socket_timeout_sec(),
        retry=Retry(NoBackoff(), 0),
    )


def _summarize_event(event: dict[str, Any]) -> str:
    """ログ用に payload から「どのキャラの何の動作か」が分かる手掛かりを抽出する。

    Phase 0.5-A 後 ログ強化 L-1 で追加。実走ログで「published type=xxx event_id=yyy」
    だけだと「どのキャラの応答か」「何が再生されたか」がパッと分からず、調査効率が
    悪かった。各 event type ごとに調査時に欲しい識別子・コンテキストを整形する。

    payload に欲しい情報が無い type は ``(no summary)`` を返す (= 未知 type の
    後方互換、ログは type / event_id / msg_id のみで継続)。

    Args:
        event: Event Envelope dict (build_* 関数の戻り値)

    Returns:
        ログに含める短い summary 文字列 (例: ``character=mimi step=thinking category=speech``)。
        フィールド未定義時は ``?`` を入れる (= 「明示的に欠けている」と分かるよう)。
    """
    type_ = event.get("type", "")
    payload = event.get("payload", {}) or {}

    if type_ == "utterance.final":
        # ルカ発話なので character なし。発話テキスト先頭でターン識別。
        text = payload.get("text", "")
        preview = text[:_LOG_TEXT_PREVIEW_CHARS]
        suffix = "..." if len(text) > _LOG_TEXT_PREVIEW_CHARS else ""
        return f"text={preview!r}{suffix} chars={len(text)}"

    if type_ == "llm.final":
        # character は L-2 で payload に追加。それまでは ? の場合あり。
        char = payload.get("character", "?")
        model = payload.get("model", "?")
        text_len = len(payload.get("text", ""))
        return f"character={char} model={model} text_len={text_len}"

    if type_ == "tts.done":
        # speaker は voicepeak narrator slug 兼キャラ slug。L-2 で character 統一予定。
        char = payload.get("character") or payload.get("speaker", "?")
        duration = payload.get("duration_ms", "?")
        return f"character={char} duration_ms={duration}"

    if type_ == "bubble.update":
        # Phase 0.5-A 8-10 で category 追加済。speech / handraise の分岐軸。
        char = payload.get("character", "?")
        step = payload.get("step", "?")
        cat = payload.get("category", "speech_status")  # default 解釈と合わせる (Phase 0.5-E: bubble 3 系統分離)
        return f"character={char} step={step} category={cat}"

    if type_ == "dispatcher.queue.update":
        q = payload.get("queue", [])
        slugs = [item.get("character_slug") for item in q]
        state = payload.get("state", "?")
        return f"state={state} queue_size={len(q)} slugs={slugs}"

    if type_ == "dispatcher.handraise.update":
        states = payload.get("handraise_states", [])
        slugs = [s.get("target_slug") for s in states]
        cooldowns = payload.get("cooldowns", {})
        return (
            f"handraise_count={len(states)} slugs={slugs} "
            f"cooldown_slugs={list(cooldowns.keys())}"
        )

    if type_ == "character.status.update":
        # Phase 0.5-B-α 導入。Ready/Thinking/ToolCalling/Raisehand/Talking の遷移を
        # ログで観察可能にする。Talking 時は metadata の pose / text を簡略表示
        # (text は長文になりうるので _LOG_TEXT_PREVIEW_CHARS で truncate)。
        char = payload.get("character", "?")
        status = payload.get("status", "?")
        prev = payload.get("previous_status", "?")
        metadata = payload.get("metadata") or {}
        meta_parts = []
        if "pose" in metadata:
            meta_parts.append(f"pose={metadata['pose']}")
        if "text" in metadata:
            text = metadata["text"]
            preview = text[:_LOG_TEXT_PREVIEW_CHARS]
            suffix = "..." if len(text) > _LOG_TEXT_PREVIEW_CHARS else ""
            meta_parts.append(f"text={preview!r}{suffix}")
        meta_summary = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        return f"character={char} status={prev}->{status}{meta_summary}"

    return "(no summary)"


def publish(event: dict) -> str:
    """
    Event Envelope を Redis Stream に publish する。

    エンコーディング方針:
      - json.dumps(..., ensure_ascii=False) で Unicode 文字をそのまま JSON 化する
      - .encode("utf-8") で明示的に UTF-8 バイト列に変換してから Redis に送る
      - decode_responses=False で redis-py の内部エンコーディング推測に依存しない
      - 受け取った msg_id (bytes) は UTF-8 で decode して str として返す

    Redis に届かないとき (モジュールの説明も参照):
      - 接続と応答の上限で諦め、再試行しない
      - 失敗した後 L2_BUS_RETRY_AFTER_SEC 秒の間は、接続を試さずに BusUnavailableError を送出する

    Returns:
        XADD で返された Redis メッセージ ID 文字列。

    Raises:
        BusUnavailableError: 直前の失敗から L2_BUS_RETRY_AFTER_SEC 秒以内 (redis.ConnectionError の子)
        redis.ConnectionError / redis.TimeoutError: 接続できない、または上限の時間内に応答がない
        redis.RedisError: Redis がコマンドを拒んだ (メモリ不足など)
        ValueError: L2_BUS_* の値が数として読めない (preflight --static が起動前に FAIL にする)
    """
    _raise_if_backing_off()
    url = _redis_url()
    stream = _stream_key()
    client = _new_client(url)
    try:
        # locale 依存の encode を避け、UTF-8 バイト列として明示的に組み立てる
        payload_bytes = json.dumps(event, ensure_ascii=False).encode("utf-8")
        try:
            msg_id: bytes = client.xadd(stream, {"event": payload_bytes})
        except _UNREACHABLE_ERRORS:
            _record_unreachable()
            raise
        _record_reachable()
        msg_id_str = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
        # ログ強化 L-1: type 別 summary でキャラ識別 + 主要 context をログに残す
        summary = _summarize_event(event)
        logger.info(
            "published type=%s [%s] event_id=%s msg_id=%s",
            event.get("type"), summary, event.get("event_id"), msg_id_str,
        )
        return msg_id_str
    finally:
        client.close()


def publish_best_effort(event: dict) -> str | None:
    """ターンの本流の event を publish する。Redis の障害では、既定でターンを止めない。

    graph.py の utterance.final / llm.final / tts.done が使う。ほかの event (bubble.update など) は、
    呼び出し側が例外を握って WARNING にしている (今までどおり)。

    Args:
        event: Event Envelope dict

    Returns:
        XADD の msg_id。Redis の障害で諦め、L2_BUS_FAIL_OPEN が真 (既定) のときは None

    Raises:
        redis.RedisError: Redis の障害で、L2_BUS_FAIL_OPEN が偽のとき (ターンを止める、以前の挙動)
        そのほかの例外: event が JSON にできないなど、Redis の障害でないもの。既定でも握らない
            (コードの誤りを、Redis の障害の ERROR ログに紛れさせないため)

    WHY: 声 (TTS) は Redis を使わない。Redis が落ちたせいで発話を止めるより、正史 (C2) と HUD (V2) に
         穴が開くほうを選ぶ (ルカの決定)。諦めた event は再送しない。
    """
    try:
        return publish(event)
    except (redis.RedisError, OSError) as exc:
        if not _fail_open():
            raise
        logger.error(
            "Redis への publish を諦め、ターンを続けます (L2_BUS_FAIL_OPEN): type=%s event_id=%s error=%s: %s",
            event.get("type"), event.get("event_id"), type(exc).__name__, exc,
        )
        return None
