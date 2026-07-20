"""
tts.py — TTS アダプタ

責務:
  - TTS API 呼び出しをこのファイルに閉じ込める
  - provider 切替しやすい構造にする（llm.py と同構造）
  - synthesize() は TTSResult を返す
  - pipeline.py から呼ばれる

【対応 provider】
  "edge_tts"  : Microsoft Edge TTS (edge-tts パッケージ必要)
  "voicevox"  : VOICEVOX Engine HTTP API (ローカルサーバ必要)
  "voicepeak" : VOICEPEAK CLI (voicepeak コマンドが PATH に必要)

【前提パッケージ (real mode)】
  edge_tts:
    uv add --optional tts "edge-tts>=6.1"
    duration 計算に mutagen を使う（任意）: uv add --optional tts "mutagen>=1.47"
  voicevox:
    stdlib のみ (urllib)。VOICEVOX Engine が http://localhost:50021 で起動している必要がある。
    docker run -p 50021:50021 voicevox/voicevox_engine:latest

【audio_url の形式】
  ローカルファイルは pathlib.Path.as_uri() で file:///... 形式に変換する。
  将来 MinIO 等に切り替える場合は adapter 内の URL 組み立て部を差し替える。
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── 戻り値型 ─────────────────────────────────────────────────────

@dataclass
class TTSResult:
    """TTS 呼び出し結果。pipeline.py から参照する。"""

    audio_url: str
    duration_ms: int
    voice: str
    format: str
    sample_rate: int
    speaker: str = ""
    chunk_audio_urls: list[str] = field(default_factory=list)


# ─── Edge TTS adapter ────────────────────────────────────────────

def _call_edge_tts(
    text: str,
    *,
    voice: str,
    output_dir: str,
    speaker: str = "",
    **kwargs,
) -> TTSResult:
    """
    Microsoft Edge TTS (edge-tts) を使って音声合成する。

    edge-tts が未インストールの場合は ImportError を送出する。
    音声ファイルは output_dir に UUID ファイル名の mp3 として保存される。
    duration_ms は mutagen があれば実測値、なければ文字数ヒューリスティック。

    Args:
        voice: Edge TTS 音声名 (例: "ja-JP-NanamiNeural")
    """
    try:
        import edge_tts
    except ImportError as exc:
        raise ImportError(
            "edge-tts が必要です。"
            " uv add --optional tts edge-tts でインストールしてください。"
        ) from exc

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{uuid.uuid4()}.mp3"

    async def _save() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(filepath))

    # ThreadPoolExecutor でサブスレッドにて asyncio.run() を実行する。
    # 親スレッドのイベントループ状態に依存しない。
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, _save()).result()

    # duration: mutagen があれば実測値、なければ文字数ヒューリスティック
    duration_ms = _mp3_duration_ms(filepath) or max(1000, int(len(text) / 5 * 1000))
    logger.info("Edge TTS 生成完了: %s (%d bytes)", filepath.name, filepath.stat().st_size)

    return TTSResult(
        audio_url=filepath.as_uri(),
        duration_ms=duration_ms,
        voice=voice,
        format="mp3",
        sample_rate=24000,
        speaker=speaker or voice,
    )


def _mp3_duration_ms(filepath: Path) -> int | None:
    """mutagen で MP3 の再生時間を取得する。未インストールなら None を返す。"""
    try:
        from mutagen.mp3 import MP3
        return int(MP3(str(filepath)).info.length * 1000)
    except Exception:
        return None


# ─── VOICEVOX adapter ────────────────────────────────────────────

def _generate_voicevox_single_file(
    text: str,
    *,
    speaker_id: int,
    voicevox_url: str,
    filepath: Path,
) -> tuple[int, int]:
    """
    1 チャンク分の VOICEVOX 音声ファイルを生成し、(duration_ms, sample_rate) を返す。

    API フロー:
      1. POST /audio_query?text=<>&speaker=<id>  → audio_query JSON
      2. POST /synthesis?speaker=<id> + audio_query body → WAV bytes
    """
    import io
    import json as _json
    import urllib.parse
    import urllib.request
    import wave

    # 1. audio_query
    query_params = urllib.parse.urlencode({"text": text, "speaker": speaker_id})
    query_url = f"{voicevox_url}/audio_query?{query_params}"
    with urllib.request.urlopen(
        urllib.request.Request(query_url, method="POST")
    ) as resp:
        audio_query = _json.loads(resp.read().decode("utf-8"))

    # 2. synthesis
    synth_url = f"{voicevox_url}/synthesis?speaker={speaker_id}"
    body = _json.dumps(audio_query).encode("utf-8")
    req = urllib.request.Request(
        synth_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    with urllib.request.urlopen(req) as resp:
        wav_bytes = resp.read()

    # duration + sample_rate from WAV header
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
        sample_rate = wf.getframerate()

    filepath.write_bytes(wav_bytes)
    logger.info("VOICEVOX WAV 生成完了: %s (%d bytes)", filepath.name, len(wav_bytes))

    return duration_ms, sample_rate


def _call_voicevox(
    text: str,
    *,
    voice: str,
    output_dir: str,
    speaker: str = "",
    on_chunk_ready=None,
    **kwargs,
) -> TTSResult:
    """
    VOICEVOX Engine HTTP API を使って音声合成する（stdlib のみ）。

    voice には VOICEVOX のスピーカー ID（整数文字列）を渡す。
      例: "3" → ずんだもん
    VOICEVOX Engine のベース URL は環境変数 L2_TTS_VOICEVOX_URL で変更できる
    (デフォルト: http://localhost:50021)

    VOICEPEAK と同じ `_split_text_for_voicepeak` で 140 字分割し、チャンクごとに生成する。
    VOICEVOX 自体に文字数制限はないが、配信演出 (OBS セリフテロップ / ストリーミング再生) を
    VOICEPEAK と揃えるためにチャンク分割を適用する (Sprint Axis D 2026-04-14)。

    on_chunk_ready コールバックが渡された場合、各チャンク生成直後に
    (audio_url, chunk_text, is_last, speaker) を通知する（ストリーミング再生用）。

    Args:
        voice:           VOICEVOX スピーカー ID (整数文字列、例: "89")
        on_chunk_ready:  チャンク生成完了時コールバック
            (audio_url: str, chunk_text: str, is_last: bool, speaker: str) -> None
    """
    voicevox_url = os.environ.get("L2_TTS_VOICEVOX_URL", "http://localhost:50021")
    try:
        speaker_id = int(voice)
    except ValueError as exc:
        raise ValueError(
            f"voicevox provider の voice はスピーカー ID の整数文字列（例: '3'）にしてください: {voice!r}"
        ) from exc

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0.5-M: VOICEPEAK と同様に JSON parse → response field 抽出 + SAY 行抽出
    # (= _parse_voicepeak_json 経由)。 旧設計では VOICEVOX path がこの parse を呼ばず、
    # octamaid 用に LLM JSON 全文をそのまま VOICEVOX に投入していたため、
    # `{} "response" :` 等の JSON syntax を音声合成 + bubble 表示する事象が発生
    # (= run_loop_20260516_070847.log で観察)。
    # JSON でない / response field なしの場合は text そのまま返るため、後方互換維持。
    say_text, _emotion, _speed, _pose = _parse_voicepeak_json(text)

    # テキスト分割 (VOICEPEAK と同じロジック。140 字以内の短文は 1 チャンク)
    chunks = _split_text_for_voicepeak(say_text)
    logger.info("VOICEVOX チャンク分割: %d 個 (元テキスト %d 文字)", len(chunks), len(say_text))

    chunk_paths: list[Path] = []
    chunk_durations: list[int] = []
    sample_rate = 24000

    for i, chunk_text in enumerate(chunks):
        filepath = out_dir / f"{uuid.uuid4()}.wav"
        dur, sr = _generate_voicevox_single_file(
            chunk_text,
            speaker_id=speaker_id,
            voicevox_url=voicevox_url,
            filepath=filepath,
        )
        chunk_paths.append(filepath)
        chunk_durations.append(dur)
        sample_rate = sr
        logger.info(
            "VOICEVOX チャンク %d/%d 生成完了: %d ms (%d 文字)",
            i + 1, len(chunks), dur, len(chunk_text),
        )
        if on_chunk_ready:
            on_chunk_ready(filepath.as_uri(), chunk_text, i == len(chunks) - 1, speaker)

    # audio_url は最初のチャンクを代表値とする。
    # 結合 WAV は生成しない (ストリーミング再生では個別チャンクが使われるため)。
    audio_url = chunk_paths[0].as_uri() if chunk_paths else ""
    total_duration = sum(chunk_durations)

    return TTSResult(
        audio_url=audio_url,
        duration_ms=total_duration,
        voice=voice,
        format="wav",
        sample_rate=sample_rate,
        speaker=speaker or f"voicevox-{speaker_id}",
        chunk_audio_urls=[p.as_uri() for p in chunk_paths],
    )


# ─── VOICEPEAK adapter ──────────────────────────────────────────

# VOICEPEAK が "bad exception" を起こしやすい半角ASCII記号 → 全角変換テーブル
_VOICEPEAK_NORMALIZE: dict[str, str] = {
    "?": "？",
    "!": "！",
    "(": "（",
    ")": "）",
    "[": "［",
    "]": "］",
    "{": "｛",
    "}": "｝",
    "<": "＜",
    ">": "＞",
    '"': "＂",
    "&": "＆",
    "/": "／",
    "\\": "￥",
}


def _normalize_for_voicepeak(text: str) -> str:
    """VOICEPEAK に渡す前に半角ASCII記号を全角に変換する。"""
    for src, dst in _VOICEPEAK_NORMALIZE.items():
        text = text.replace(src, dst)
    return text


def _extract_say_lines(text: str) -> str:
    """
    octamaid 等の "SAY: ...\\nLOG: ..." 形式から SAY 行のみを抽出する (Phase 0.5-M)。

    octamaid の system prompt (= system_octamaid.txt) は次の 3 軸出力を要求する設計:
      - SAY: 読み上げ前提の短文 (= TTS で発声、bubble 表示対象)
      - LOG: 画面用の状態表示 (= 配信に出さない、メタ情報)
      - MODE: 現在の個体 (= 同上、必要時のみ)

    TTS / bubble には SAY 内容だけ流したいため、本関数で LOG / MODE 行を除去 +
    SAY 行の prefix を strip する。

    SAY: prefix が含まれない場合 (= mimi/chisame/sakura 等の他キャラ、もしくは
    octamaid が SAY: 形式に従わない自由応答) は元 text をそのまま返す
    (= 安全側挙動、既存挙動への regression なし)。

    複数 SAY: 行は半角スペースで連結 (= TTS で自然な間で発声)。
    """
    lines = text.split("\n")
    say_contents: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SAY:"):
            content = stripped[len("SAY:"):].strip()
            if content:
                say_contents.append(content)
    if say_contents:
        return " ".join(say_contents)
    return text


def _parse_voicepeak_json(
    text: str,
) -> tuple[str, dict[str, int] | None, int | None, str | None]:
    """
    LLM 応答が JSON 構造の場合、response / emotion / speed / pose を分離する。

    対応する JSON 形式:
        {"emotion": {"happy": 50, ...}, "speed": 100, "pose": "happy", "response": "テキスト"}

    全角記号に正規化済みの JSON も半角に戻してからパースを試みる。

    Phase 0.5-M: response field 抽出後、octamaid の "SAY: ...\\nLOG: ..." 形式から
    SAY 行のみ抽出する (= _extract_say_lines 経由、他キャラは影響なし)。

    Returns:
        (say_text, emotion_dict_or_None, speed_or_None, pose_or_None)
        JSON でない場合は (text, None, None, None) をそのまま返す。
    """
    import json as _json

    # 全角→半角の逆変換テーブル（正規化済み JSON のパース用）
    _REVERSE_NORMALIZE: dict[str, str] = {v: k for k, v in _VOICEPEAK_NORMALIZE.items()}

    raw = text
    for src, dst in _REVERSE_NORMALIZE.items():
        raw = raw.replace(src, dst)
    raw = raw.strip()

    # マークダウンコードブロック (```json ... ```) を除去
    if raw.startswith("```"):
        lines = raw.split("\n")
        # 先頭の ```json や ``` を除去
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # 末尾の ``` を除去
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        obj = _json.loads(raw)
    except (ValueError, TypeError):
        return text, None, None, None

    if not isinstance(obj, dict) or "response" not in obj:
        return text, None, None, None

    say_text = str(obj["response"])
    # Phase 0.5-M: octamaid 形式 ("SAY: ...\nLOG: ...") から SAY 行のみ抽出。
    # 他キャラ (mimi/chisame/sakura) は SAY: prefix なしなので変更なし (= 安全側)。
    say_text = _extract_say_lines(say_text)
    emotion = obj.get("emotion")
    if isinstance(emotion, dict):
        emotion = {str(k): int(v) for k, v in emotion.items()}
    else:
        emotion = None
    speed_val = obj.get("speed")
    if speed_val is not None:
        speed_val = int(speed_val)
    pose_val = obj.get("pose")
    if pose_val is not None:
        pose_val = str(pose_val).strip().lower()
        if not pose_val:
            pose_val = None

    return say_text, emotion, speed_val, pose_val


# ─── VOICEPEAK テキスト分割 ──────────────────────────────────────

_VOICEPEAK_MAX_CHARS = 140


def _split_text_for_voicepeak(
    text: str,
    max_chars: int = _VOICEPEAK_MAX_CHARS,
) -> list[str]:
    """
    VOICEPEAK の文字数制限 (140字) に合わせてテキストを自然な句読点で分割する。

    なるべく max_chars ギリギリまで活用しつつ、以下の優先順位で分割点を選ぶ:
      1. 句点・感嘆符・疑問符 (。！？!?\n) — 文末
      2. 読点・カンマ・セミコロン等 (、,，;；:： ) — 節の切れ目
      3. 上記なし → max_chars で強制カット

    140字以内のテキストは分割せずそのまま返す。
    """
    if len(text) <= max_chars:
        return [text]

    primary = set("。！？!?\n")
    secondary = set("、,，;；:： ")

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunk = remaining.strip()
            if chunk:
                chunks.append(chunk)
            break

        window = remaining[:max_chars]

        # 後方スキャンで最適な分割点を探す (max_chars ギリギリまで活用)
        best = -1
        for i in range(len(window) - 1, -1, -1):
            if window[i] in primary:
                best = i + 1
                break

        if best <= 0:
            for i in range(len(window) - 1, -1, -1):
                if window[i] in secondary:
                    best = i + 1
                    break

        if best <= 0:
            best = max_chars

        chunk = remaining[:best].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[best:].lstrip()

    return chunks if chunks else [text]


# ─── VOICEPEAK FIFO キュー ───────────────────────────────────────
#
# voicepeak.exe は同時に 1 プロセスしか実行できない。
# threading.Lock だと投入順序が保証されないため、FIFO キューで
# 先に投入されたジョブを先に実行する。
#
# フィラースレッドとメインスレッドが同時に VOICEPEAK を要求しても、
# 先に submit した方が先に合成される。

import concurrent.futures
import queue as _queue_mod

_voicepeak_queue: _queue_mod.Queue | None = None
_voicepeak_worker_thread: threading.Thread | None = None


def _decode_voicepeak_output(b: bytes | None) -> str:
    """VOICEPEAK の stdout/stderr bytes を安全にデコードする。

    Windows では CP932 が多いが、将来のバージョン変更に備えて複数エンコーディングを試す。
    """
    if not b:
        return ""
    for enc in ("cp932", "utf-8"):
        try:
            return b.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace").strip()


# VOICEPEAK 並列実行エラーのシグネチャ (stderr に小文字化して部分一致でチェック)
#
# 実際のエラーメッセージ (確認済み):
#   "In this version, up to 1 command line instance can be executed at same time."
#
# scripts/test_voicepeak_parallel_error.py で再現確認可能:
#   - returncode=1 / stderr に上記文字列 / wav 未生成
#   - 連続実行 (Test 4) では発生せず、真の並列実行 (Test 2/3) でのみ発生
_VOICEPEAK_BUSY_SIGNATURES = (
    # 公式エラーメッセージの特徴的部分 (将来バージョンで微変更しても拾えるように複数候補)
    "command line instance can be executed",
    "up to 1 command line instance",
    # 将来の VOICEPEAK バージョン or 別ロケール想定の候補
    "already running",
    "another instance",
    "instance is running",
    "すでに実行",
    "起動中",
    "実行中",
)


def _is_voicepeak_busy_error(stderr: str, stdout: str) -> bool:
    """VOICEPEAK の並列実行エラーかどうかを判定する。"""
    text = (stderr + " " + stdout).lower()
    return any(sig in text for sig in _VOICEPEAK_BUSY_SIGNATURES)


def _get_voicepeak_retry_wait_sec() -> float:
    """並列実行エラー / クラッシュ発生時のリトライ待機時間（秒）を返す。

    リトライ前にのみ挿入される (通常成功時は待機しない)。
    環境変数 L2_VOICEPEAK_RETRY_WAIT_SEC で上書き可能 (デフォルト: 1.0)。

    busy (並列実行エラー) と crash (非ゼロ exit) の両方で同じ値を使う
    (Phase 0.5-A フェーズ 8: 倍率撤廃。リトライ回数で確率カバーする方針に変更)。
    """
    return float(os.environ.get("L2_VOICEPEAK_RETRY_WAIT_SEC", "1.0"))


def _get_voicepeak_max_retries() -> int:
    """並列実行エラー / クラッシュ発生時の最大リトライ回数を返す。

    環境変数 L2_VOICEPEAK_MAX_RETRIES で上書き可能 (デフォルト: 8)。

    Phase 0.5-A フェーズ 8 で 2 → 8 に拡大。retry_wait を 2.0 → 1.0 に短縮した
    のと合わせて、最終的な完走確率を担保する (合計待機時間は 4s → 8s と微増)。
    """
    return int(os.environ.get("L2_VOICEPEAK_MAX_RETRIES", "8"))


def _voicepeak_worker_fn(q: _queue_mod.Queue) -> None:
    """VOICEPEAK キューワーカー。キューからジョブを取り出し順次実行する。

    並列実行エラー (`In this version, up to 1 command line instance ...`) を
    検出したら短い待機後にリトライする。通常成功時は無待機で次のジョブへ進む
    (固定クールダウンは入れない: レイテンシに直結するため)。
    """
    import subprocess as _sp

    while True:
        item = q.get()
        if item is None:
            break
        cmd_str, future = item

        max_retries = _get_voicepeak_max_retries()
        retry_wait = _get_voicepeak_retry_wait_sec()

        result = None
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                # text=False (bytes) で取得してから手動で UTF-8 / CP932 試行
                # Windows では VOICEPEAK 出力エンコーディングが不定のため、
                # decode 失敗を捕捉するより bytes のまま扱う
                result = _sp.run(cmd_str, capture_output=True, shell=True)
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "VOICEPEAK subprocess 例外 (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                break  # subprocess 起動失敗は再試行しない

            if result.returncode == 0:
                break  # 成功 → 即 next ジョブへ (待機なし)

            # 非ゼロ exit: 並列実行エラーかチェック
            stderr_text = _decode_voicepeak_output(result.stderr)
            stdout_text = _decode_voicepeak_output(result.stdout)
            is_busy = _is_voicepeak_busy_error(stderr_text, stdout_text)

            # Phase 0.5-A フェーズ 8: cmd_str 全文も warning に含める
            # (出力ファイル未生成 / クラッシュ時の引数再現に必要)。
            logger.warning(
                "VOICEPEAK 非ゼロ終了 (attempt %d/%d): returncode=%d busy=%s"
                "\n  stderr: %s\n  stdout: %s\n  cmd (full): %s",
                attempt + 1, max_retries + 1,
                result.returncode, is_busy,
                stderr_text or "(empty)",
                stdout_text or "(empty)",
                cmd_str,
            )

            if attempt < max_retries:
                # Phase 0.5-A フェーズ 8: busy と crash で wait を共通化 (倍率撤廃)。
                # 旧設計はクラッシュを重く扱って `retry_wait * 2` だったが、
                # max_retries を 4 倍 (2 → 8) に拡大したので、回数で確率カバーする
                # 方針に変更。クラッシュ後の応答開始遅延を短縮する効果。
                wait = retry_wait
                reason = "並列実行エラー" if is_busy else f"クラッシュ (returncode={result.returncode})"
                logger.info(
                    "VOICEPEAK %s検出 → %.1f 秒待機してリトライ",
                    reason, wait,
                )
                time.sleep(wait)
                continue
            break  # リトライ上限到達

        if last_exc is not None:
            future.set_exception(last_exc)
        else:
            future.set_result(result)


def _ensure_voicepeak_worker() -> _queue_mod.Queue:
    """VOICEPEAK ワーカースレッドを初回利用時に起動する。"""
    global _voicepeak_queue, _voicepeak_worker_thread
    if _voicepeak_queue is None:
        _voicepeak_queue = _queue_mod.Queue()
        _voicepeak_worker_thread = threading.Thread(
            target=_voicepeak_worker_fn,
            args=(_voicepeak_queue,),
            daemon=True,
        )
        _voicepeak_worker_thread.start()
        logger.info("VOICEPEAK FIFO ワーカー起動")
    return _voicepeak_queue


def _submit_voicepeak(cmd_str: str):
    """
    VOICEPEAK コマンドをキューに投入し、完了を待つ。

    FIFO 順序が保証される。先に投入されたジョブが先に実行される。

    Phase 0.5-A フェーズ 8 (出力ファイル未生成バグ調査):
    成功時 (returncode=0) は subprocess.CompletedProcess を返す。呼出側で
    stdout / stderr を参照することで「returncode=0 だが --out が無視されて
    出力ファイルが書かれない」現象の調査に使う。失敗時 (returncode != 0) は
    従来通り RuntimeError を投げる。

    Returns:
        subprocess.CompletedProcess (成功時のみ)

    Raises:
        FileNotFoundError: VOICEPEAK コマンドが見つからない
        RuntimeError: VOICEPEAK 実行エラー (returncode != 0)
    """
    q = _ensure_voicepeak_worker()
    future: concurrent.futures.Future = concurrent.futures.Future()
    q.put((cmd_str, future))

    try:
        result = future.result()  # ブロック: 完了まで待つ
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("VOICEPEAK 実行中に例外: %s", exc)
        logger.error("VOICEPEAK 実行中に例外 cmd (full): %s", cmd_str)
        raise RuntimeError(f"VOICEPEAK 実行エラー: {type(exc).__name__}: {exc}") from exc

    if result.returncode != 0:
        # ワーカー側で既に詳細ログは出力済み。ここでは例外メッセージのみ組み立てる。
        # Phase 0.5-A フェーズ 8: cmd (full) も warning に格上げ (出力ファイル未生成
        # バグの再現に必要)。
        stderr_text = _decode_voicepeak_output(result.stderr) or "(empty)"
        stdout_text = _decode_voicepeak_output(result.stdout) or "(empty)"
        logger.error(
            "VOICEPEAK 実行最終失敗: returncode=%d stderr: %s stdout: %s",
            result.returncode, stderr_text, stdout_text,
        )
        logger.error("VOICEPEAK 実行最終失敗 cmd (full): %s", cmd_str)
        raise RuntimeError(
            f"VOICEPEAK 実行エラー: returncode={result.returncode} "
            f"stderr={stderr_text!r} stdout={stdout_text!r}"
        )

    return result


def _log_voicepeak_output_missing_diagnostics(
    *,
    filepath,
    result,
    cmd_str: str,
    speaker: str | None,
    attempt: int,
    max_attempts: int,
) -> None:
    """VOICEPEAK 出力ファイル未生成バグの調査用 warning ログ (Phase 0.5-A フェーズ 8)。

    実走 (2026-05-08) で「returncode=0 だが期待した --out のファイルが生成されず、
    代わりに L2 ルート直下の output.wav に書かれる」という現象が観測された。
    再現性のないバグなので、次に発生した瞬間に原因を絞り込めるよう、以下の情報を
    すべて warning レベルで残す:

      - filepath の絶対パス (Windows パスや日本語混入の判別)
      - 期待ファイルの親ディレクトリの存在 / 書き込み権限の状況
      - 「cwd の output.wav」(VOICEPEAK のデフォルト出力先) の有無 + サイズ + mtime
        → 存在すれば --out 無視疑惑が確定する
      - subprocess の stdout / stderr (decoded)
      - cmd_str 全文 (引数のエスケープ / クォート問題の再現に必要)

    Args:
        filepath:    期待された出力ファイルパス (Path)
        result:      ``subprocess.CompletedProcess`` (returncode=0 だが file なし)
        cmd_str:     VOICEPEAK 実行コマンド全文
        speaker:     キャラ slug (ログ識別用)
        attempt:     試行回数 (0 = 初回、1 以上 = リトライ)
        max_attempts: 最大リトライ回数
    """
    from pathlib import Path as _Path
    import time as _time

    speaker_label = speaker or "(unknown)"
    abs_expected = filepath.resolve() if hasattr(filepath, "resolve") else _Path(filepath).resolve()
    parent = abs_expected.parent
    parent_status = "exists" if parent.is_dir() else "MISSING"

    # cwd / repo root の output.wav を確認 (VOICEPEAK がデフォルト出力先に書いた疑い)
    cwd = _Path.cwd()
    cwd_output = cwd / "output.wav"
    if cwd_output.is_file():
        st = cwd_output.stat()
        cwd_output_info = (
            f"cwd_output.wav 存在 (size={st.st_size} bytes, "
            f"mtime={_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(st.st_mtime))}, "
            f"path={cwd_output}) — VOICEPEAK が --out を無視した疑い"
        )
    else:
        cwd_output_info = f"cwd_output.wav 不在 (cwd={cwd})"

    # subprocess の stdout / stderr を decode
    if result is not None:
        stdout_text = _decode_voicepeak_output(result.stdout) or "(empty)"
        stderr_text = _decode_voicepeak_output(result.stderr) or "(empty)"
        returncode = result.returncode
    else:
        stdout_text = "(result is None)"
        stderr_text = "(result is None)"
        returncode = -1

    label = "初回検出" if attempt == 0 else f"リトライ後再検出 ({attempt}/{max_attempts})"

    logger.warning(
        "VOICEPEAK 出力ファイル未生成 [%s] speaker=%s\n"
        "  expected: name=%s abs=%s parent=%s (%s)\n"
        "  returncode=%d\n"
        "  stdout: %s\n"
        "  stderr: %s\n"
        "  %s\n"
        "  cmd (full): %s",
        label, speaker_label,
        filepath.name, abs_expected, parent, parent_status,
        returncode,
        stdout_text,
        stderr_text,
        cwd_output_info,
        cmd_str,
    )


def _generate_voicepeak_single_file(
    text: str,
    *,
    voice: str,
    filepath: Path,
    speed: int | None = None,
    emotion: dict[str, int] | None = None,
    speaker: str | None = None,
) -> tuple[int, int]:
    """
    VOICEPEAK CLI で 1 チャンク分の WAV を生成する。

    FIFO キューで排他制御される。投入順序が合成順序になる。

    Args:
        speaker: ログに出力するキャラクター slug (e.g., "mimi")。誰の発話かを
                 ログから即座に追えるようにするための識別情報。

    Returns:
        (duration_ms, sample_rate)
    """
    import subprocess
    import wave

    voicepeak_cmd = os.environ.get("L2_TTS_VOICEPEAK_PATH", "voicepeak")

    normalized_text = _normalize_for_voicepeak(text)
    if normalized_text != text:
        logger.debug("VOICEPEAK テキスト正規化: %r → %r", text, normalized_text)

    safe_text = normalized_text.replace('"', "'")
    # Phase 0.5-F-6-g (案 C): 改行 sanitize の最終防衛層 (= 案 A の二重防衛)。
    # _parse_voicepeak_json を経由しない呼出経路 (= filler 等で直接 _synthesize_voicepeak
    # に text を渡す経路) でも、subprocess 投入直前で必ず改行を除去することで
    # VOICEPEAK CLI 引数破壊を構造的に阻止する。詳細は _parse_voicepeak_json の
    # 同等処理を参照 (logs/runs/run_loop_20260515_002506.log で観察された事象)。
    safe_text = safe_text.replace("\\n", " ").replace("\n", " ")
    safe_voice = voice.replace('"', "'")

    cmd_str = (
        f'{subprocess.list2cmdline([voicepeak_cmd])}'
        f' --say "{safe_text}"'
        f' --narrator "{safe_voice}"'
        f' --out {subprocess.list2cmdline([str(filepath)])}'
    )
    if speed is not None:
        cmd_str += f" --speed {speed}"
    if emotion:
        emotion_expr = ",".join(f"{k}={v}" for k, v in emotion.items())
        cmd_str += f" --emotion {emotion_expr}"

    # 配信中のコンソール表示でファイルパス (ユーザーディレクトリ等) を漏らさないよう、
    # 「投入」サマリ行はファイル名・narrator・テキスト長のみを出す。
    # Phase 0.5-A フェーズ 8: voicepeak.exe 実行時の cmd_str 全文を info レベルで残す
    # (出力ファイル未生成バグの調査用。引数のエスケープ / クォート / 文字化けの再現に必要)。
    # 並行 / バックグラウンド合成中に「誰の何のチャンクか」を即座に追えるよう、
    # speaker と text 先頭 (40 文字) を含める。
    text_preview = safe_text if len(safe_text) <= 40 else safe_text[:40] + "…"
    logger.info(
        "VOICEPEAK 投入: speaker=%s narrator=%s text_len=%d out=%s text=%r",
        speaker or "(unknown)", voice, len(safe_text), filepath.name, text_preview,
    )
    logger.info("VOICEPEAK コマンド (full): %s", cmd_str)

    initial_result = _submit_voicepeak(cmd_str)

    # VOICEPEAK が returncode=0 でも出力ファイルを生成しないケースは
    # 「待っても永久に出ない」(= subprocess が正常終了したのにファイル書き込みが発生
    # しない不具合) なので、polling ではなく subprocess そのものを再実行する。
    # 1 秒間隔で最大 20 回まで再投入を試みる。
    # 環境変数:
    #   L2_VOICEPEAK_OUTPUT_RETRY_INTERVAL_SEC (default: 1.0) — 再実行間の待機秒数
    #   L2_VOICEPEAK_OUTPUT_RETRY_MAX_ATTEMPTS (default: 20)  — 最大再実行回数
    if not filepath.is_file():
        interval_sec = float(os.environ.get("L2_VOICEPEAK_OUTPUT_RETRY_INTERVAL_SEC", "1.0"))
        max_attempts = int(os.environ.get("L2_VOICEPEAK_OUTPUT_RETRY_MAX_ATTEMPTS", "20"))
        import time

        # Phase 0.5-A フェーズ 8: 出力ファイル未生成発生時の調査ログ強化。
        # VOICEPEAK が --out 引数を無視してデフォルト出力先 (cwd の output.wav) に
        # 書いている疑いを確認するため、cwd の output.wav を検出 + subprocess の
        # stdout/stderr を warning に出力 + cmd_str 全文を再掲する。
        _log_voicepeak_output_missing_diagnostics(
            filepath=filepath,
            result=initial_result,
            cmd_str=cmd_str,
            speaker=speaker,
            attempt=0,  # 0 = 初回検出 (リトライ前)
            max_attempts=max_attempts,
        )

        for attempt in range(1, max_attempts + 1):
            logger.warning(
                "VOICEPEAK 出力ファイル未生成: %s → %.1f 秒待機して subprocess 再実行 (%d/%d)",
                filepath.name, interval_sec, attempt, max_attempts,
            )
            time.sleep(interval_sec)
            retry_result = _submit_voicepeak(cmd_str)
            if filepath.is_file():
                logger.info(
                    "VOICEPEAK 再実行成功 (%d/%d): %s",
                    attempt, max_attempts, filepath.name,
                )
                break
            # 再投入後も未生成なら詳細を出す (毎回詳細 + cmd_str 再掲)
            _log_voicepeak_output_missing_diagnostics(
                filepath=filepath,
                result=retry_result,
                cmd_str=cmd_str,
                speaker=speaker,
                attempt=attempt,
                max_attempts=max_attempts,
            )

    if not filepath.is_file():
        raise FileNotFoundError(
            f"VOICEPEAK が出力ファイルを生成しませんでした: {filepath.name}"
        )

    with wave.open(str(filepath)) as wf:
        duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
        sample_rate = wf.getframerate()

    return duration_ms, sample_rate


def _concatenate_wavs(input_paths: list[Path], output_path: Path) -> None:
    """複数の WAV ファイルを 1 つに結合する。"""
    import wave

    with wave.open(str(input_paths[0]), "rb") as first:
        params = first.getparams()
        all_frames = first.readframes(first.getnframes())

    for p in input_paths[1:]:
        with wave.open(str(p), "rb") as wf:
            all_frames += wf.readframes(wf.getnframes())

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        out.writeframes(all_frames)


def _call_voicepeak(
    text: str,
    *,
    voice: str,
    output_dir: str,
    speaker: str = "",
    speed: int | None = None,
    on_chunk_ready=None,
    **kwargs,
) -> TTSResult:
    """
    VOICEPEAK CLI を使って音声合成する。

    voice にはナレーター名（例: "彩澄りりせ"）を渡す。
    VOICEPEAK コマンドのパスは環境変数 L2_TTS_VOICEPEAK_PATH で変更可能
    (デフォルト: "voicepeak")

    text が JSON 構造 (response/emotion/speed) の場合は自動的に分解し、
    --say / --emotion / --speed パラメーターに振り分ける。

    140字超のテキストは自然な句読点で分割し、チャンクごとに生成する。
    on_chunk_ready コールバックが渡された場合、各チャンク生成直後に
    (audio_url, chunk_text, is_last, speaker) を通知する（ストリーミング再生用）。

    Args:
        voice:           ナレーター名
        speed:           発話速度（50〜200。省略時は VOICEPEAK デフォルト）
        on_chunk_ready:  チャンク生成完了時コールバック
            (audio_url: str, chunk_text: str, is_last: bool, speaker: str) -> None
            - audio_url:  file:// URI
            - chunk_text: VOICEPEAK --say に渡した 140 字以内のテキスト
            - is_last:    最終チャンクなら True
            - speaker:    キャラクター slug ("mimi" / "chisame" / "sakura" / "octamaid")
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON 構造のパース（emotion / speed / pose / response の分離）
    # pose は TTS 内では使用しない（pose.update は graph.py / pipeline.py 側で発行）
    say_text, json_emotion, json_speed, _json_pose = _parse_voicepeak_json(text)
    # Phase 0.5-F-6-g (案 A): VOICEPEAK 経路限定で改行を sanitize。
    # LLM が response 内に改行を含めた場合、subprocess の --say 引数に literal
    # newline (= \n、1 文字) が渡され、Windows の引数解釈で --say の値が分断
    # されて --out 以降のオプションが無視される事象を防ぐ (logs/runs/
    # run_loop_20260515_002506.log で観察、VOICEPEAK が cwd の output.wav に出力)。
    # 加えて、LLM が誤って 2 文字の「\n」(= backslash + n) を含む応答を返す
    # ケースにも対応 (= 2 文字を空白に置換)。
    # 順序: \\n (= 2 文字 sequence) を先に置換、その後 \n (= literal newline) を
    # 置換。逆順だと \n を空白にした後の文字列に \\n が見つからない。
    # 【WHY: _parse_voicepeak_json 内ではなく呼出後で sanitize する】
    # _parse_voicepeak_json は HUD 表示用 (= bubble.update / character.status.update
    # の text 抽出、run_loop._extract_llm_response_text / ask_character.py 等)
    # にも使われる。HUD では改行を維持したいケースがあるため、関心事分離として
    # VOICEPEAK 経路でのみ sanitize する。
    say_text = say_text.replace("\\n", " ").replace("\n", " ")
    effective_speed = json_speed if json_speed is not None else speed

    # テキスト分割
    chunks = _split_text_for_voicepeak(say_text)
    text_preview_full = say_text if len(say_text) <= 60 else say_text[:60] + "…"
    logger.info(
        "VOICEPEAK チャンク分割: speaker=%s %d 個 (元テキスト %d 文字) text=%r",
        speaker or "(unknown)", len(chunks), len(say_text), text_preview_full,
    )

    chunk_paths: list[Path] = []
    chunk_durations: list[int] = []
    sample_rate = 24000

    for i, chunk_text in enumerate(chunks):
        filepath = out_dir / f"{uuid.uuid4()}.wav"
        dur, sr = _generate_voicepeak_single_file(
            chunk_text,
            voice=voice,
            filepath=filepath,
            speed=effective_speed,
            emotion=json_emotion,
            speaker=speaker,
        )
        chunk_paths.append(filepath)
        chunk_durations.append(dur)
        sample_rate = sr
        chunk_preview = chunk_text if len(chunk_text) <= 40 else chunk_text[:40] + "…"
        logger.info(
            "VOICEPEAK チャンク %d/%d 生成完了: speaker=%s %d ms (%d 文字) text=%r",
            i + 1, len(chunks), speaker or "(unknown)",
            dur, len(chunk_text), chunk_preview,
        )
        if on_chunk_ready:
            on_chunk_ready(filepath.as_uri(), chunk_text, i == len(chunks) - 1, speaker)

    # audio_url は最初のチャンクを代表値とする。
    # 結合 WAV は生成しない (ストリーミング再生では個別チャンクが使われるため)。
    # 全チャンクは chunk_audio_urls で参照する。
    audio_url = chunk_paths[0].as_uri() if chunk_paths else ""

    total_duration = sum(chunk_durations)

    return TTSResult(
        audio_url=audio_url,
        duration_ms=total_duration,
        voice=voice,
        format="wav",
        sample_rate=sample_rate,
        speaker=speaker or voice,
        chunk_audio_urls=[p.as_uri() for p in chunk_paths],
    )


# ─── irodori-TTS adapter (VoiceDesign / HTTP サイドカー経由) ──────────
#
# irodori-tts は torch + CUDA を要する重い ML スタックのため L2 本体には取り込まず、
# 別プロセスの HTTP サイドカー (sidecar/irodori_server.py) として起動する。ここは
# その stdlib HTTP クライアント (VOICEVOX adapter と同じく urllib のみで実装し、
# L2 に新規依存を増やさない)。
#
# provider "irodori_vd": VoiceDesign。声は固定 (self-ref アンカー + caption + seed)。
# 表現は pose (→ 本文末の正規絵文字 + caption 末尾サフィックス) と speed (→ ds) のみ。
# irodori は emotion 強度入力を持たないため emotion dict は使わない (pose-only)。
#
# キャラ別設定 (caption / アンカー / seed / cfg / pose_map) は voices.json を
# データ駆動で実行時ロードする (単一の真実源)。正典:
# T:\irodori-tts\reference_voices\voices.json (L2_IRODORI_VOICES_JSON で上書き可)。
# サイドカーは generic な合成エンドポイントに保つ (他ツールからも再利用可)。


def _voices_json_path() -> Path:
    """irodori 確定設定の正典 voices.json のパス (L2_IRODORI_VOICES_JSON で上書き可)。"""
    return Path(
        os.environ.get(
            "L2_IRODORI_VOICES_JSON",
            r"T:\irodori-tts\reference_voices\voices.json",
        )
    )


# voices.json は解決済みパスごとにキャッシュ (テストで別 fixture を指しても汚染しない)。
# 実行中に voices.json を編集した場合は L2 再起動で反映される (= .env と同様の扱い)。
_voices_cache: dict[str, dict] = {}


def _load_voices(path: Path | None = None) -> dict:
    """voices.json を読み込んで返す (パスでキャッシュ)。

    Returns:
        raw dict ({"common": {...}, "voices": {slug: {caption, ref_wav, seed, pose_map, ...}}})。
        ref_wav の絶対パス化は _resolve_irodori_voice 側で行う。

    Raises:
        FileNotFoundError: voices.json が存在しない (irodori_vd には必須)
    """
    import json as _json

    p = (path or _voices_json_path()).resolve()
    key = str(p)
    cached = _voices_cache.get(key)
    if cached is not None:
        return cached
    if not p.is_file():
        raise FileNotFoundError(
            f"irodori voices.json が見つかりません: {p}。"
            f" L2_IRODORI_VOICES_JSON で reference_voices/voices.json を指定してください。"
        )
    data = _json.loads(p.read_text(encoding="utf-8"))
    _voices_cache[key] = data
    return data


def _readings_path() -> Path:
    """irodori 読み辞書 readings.json のパス (L2_IRODORI_READINGS_JSON で上書き可)。"""
    return Path(
        os.environ.get(
            "L2_IRODORI_READINGS_JSON",
            r"T:\irodori-tts\reference_voices\readings.json",
        )
    )


# readings.json も解決済みパスごとにキャッシュ (voices と同型)。実行中の編集は L2 再起動で反映。
_readings_cache: dict[str, dict] = {}


def _load_readings(path: Path | None = None) -> dict:
    """読み辞書 readings.json を読み込んで返す (パスでキャッシュ)。

    voices.json と違い **欠損は致命ではない**: 辞書が無くても irodori は動く
    (読みが補正されないだけ) ので、ファイルが無ければ空 dict を返して passthrough する。

    Returns:
        {"global": {surface: reading}, "characters": {voice: {surface: reading}}, ...}
        または欠損時 {}。
    """
    import json as _json

    p = (path or _readings_path()).resolve()
    key = str(p)
    cached = _readings_cache.get(key)
    if cached is not None:
        return cached
    if not p.is_file():
        # 辞書なしでも動く (no-op)。ログだけ残して空をキャッシュ (毎回 stat しない)。
        logger.info("irodori readings.json が無いため読み辞書なしで動作します: %s", p)
        _readings_cache[key] = {}
        return {}
    data = _json.loads(p.read_text(encoding="utf-8"))
    _readings_cache[key] = data
    return data


def _apply_readings(text: str, voice: str) -> str:
    """喋るテキストに読み辞書を適用して返す (表層 → カタカナ読み)。

    irodori はエンジン側の読み辞書を持たず、英単語・略語・固有名詞を誤読する
    (実測: JSON→ジュウソン, Claude→キーエカ, 波心→なみごころ 等)。そこで
    **サイドカーへ送るテキストにだけ** 読み置換を適用する (HUD/字幕は元のまま保つ)。

    マージ規則:
      - readings["global"] を土台に readings["characters"][voice] (あれば) で上書き
        (キャラ別が優先)。

    置換規則 (longest-match-first):
      - 表層を長い順に並べた単一の正規表現で **1 パス置換** する。同じ位置では
        最長の表層が選ばれる ("Think-AI Lab" を "Lab" より優先)。単一パスなので
        置換後の読み (カタカナ) が再マッチして二重置換される事故も起きない。
      - re.escape で "A.I.byss" 等の記号を literal 化する。
      - _excluded_review (文脈依存の多音字) / _accent_meta は適用しない (loader が無視)。
      - 辞書が空 (ファイル欠損含む) なら text をそのまま返す (no-op)。

    注: 表層が 140 字チャンク境界を跨ぐ稀ケースは未対応 (実害ほぼなし)。
    """
    readings = _load_readings()
    if not readings:
        return text
    merged = dict(readings.get("global", {}))
    merged.update(readings.get("characters", {}).get(voice, {}))
    surfaces = [s for s in merged if s]  # 空文字 surface はパターンを壊すので除外
    if not surfaces:
        return text
    import re as _re

    # 長い表層を先の alternative に置く → 同位置で最長一致が選ばれる (longest-match-first)。
    pattern = _re.compile("|".join(_re.escape(s) for s in sorted(surfaces, key=len, reverse=True)))
    return pattern.sub(lambda m: merged[m.group(0)], text)


# 小数部を 1 桁ずつ読ませるためのカナ表 (3.14 → さんてん「いちよん」。「じゅうよん」にしない)。
_DECIMAL_DIGIT_KANA = {
    "0": "ゼロ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
    "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう",
}


def _normalize_decimals(text: str) -> str:
    """小数 (数字.数字) を irodori が読める形に正規化する ("5.5" → "5てんご")。

    irodori は g2p を持たず小数点 "." を読めない (実機確認: GPT-5.5 / Gemini Pro 3.1 等の
    小数が誤読される)。そこで喋るテキストの小数を:
      - 整数部 … 数字のまま (irodori は整数を読める: 12 → じゅうに)
      - 小数点 … 「てん」 (実測 probe7: "5てん5" は小数点として発声される)
      - 小数部 … 数字を 1 桁ずつカナ化 (小数は桁読み: 3.14 → さんてんいちよん。
                 "14" を「じゅうよん」と読ませない)
    に正規化する。数字に挟まれない "." (A.I.byss / 文末 等) は対象外。読み辞書と同じく
    **喋るテキストにだけ** 適用し、HUD/字幕には元の "5.5" を残す (呼出側で元 chunk を渡す)。
    """
    import re as _re

    def _repl(m) -> str:
        int_part, frac = m.group(1), m.group(2)
        return int_part + "てん" + "".join(_DECIMAL_DIGIT_KANA[d] for d in frac)

    return _re.sub(r"(\d+)\.(\d+)", _repl, text)


def _irodori_url() -> str:
    """サイドカーの /synthesize URL を返す (L2_TTS_IRODORI_URL で base 上書き可)。"""
    base = os.environ.get("L2_TTS_IRODORI_URL", "http://127.0.0.1:18080").rstrip("/")
    return f"{base}/synthesize"


def _speed_to_duration_scale(speed: int | None) -> float:
    """VOICEPEAK 互換の speed (既定 100) を irodori の duration_scale に変換する。

    duration_scale = max(0.85, 100/speed)。speed が大きい (速い) ほど短く (=<1)、
    小さいほど長く (=>1)。**0.85 を下限にクランプする**: ds<~0.7 は拡散が内容を詰め込めず
    後半が崩壊する (PoC ASR 実証: ds0.645 sim0.83 → ds0.85 sim0.98)。早口は ds≈0.85+caption。
    speed 未指定 / 不正は 1.0 (等倍)。
    """
    if not speed or speed <= 0:
        return 1.0
    return round(max(0.85, 100.0 / float(speed)), 3)


# 短文の末尾幻聴 (= 尺の過剰予測) 抑制パラメータ。すべて env で上書き可。
# WHY: irodori の duration predictor は短文の尺を過剰予測し (~3.4s floor)、余尺を
# 「それっぽい発話」(語尾の癖の反復) で埋める = 末尾幻聴。duration_scale では取り戻せない
# (実測 probe3: ds0.85 でも どうも/橋 は幻聴)。そこで短文だけ manual duration
# (SamplingRequest.seconds) で predictor をバイパスし、文字数ベースで尺を直接与える
# (実測 probe4: len×0.27s で どうも/橋 も完全クリーン、内容欠落なし)。
# 漢字密集の極短文は char 数が実モーラを過小評価し早口になりうるが、内容は保持され
# 幻聴の garble よりはるかに軽微 (ルカ判断で char ベース採用)。rate/閾値は実走で微調整。
_IRODORI_SHORT_CHARS = int(os.environ.get("L2_TTS_IRODORI_SHORT_CHARS", "12"))
_IRODORI_SEC_PER_CHAR = float(os.environ.get("L2_TTS_IRODORI_SEC_PER_CHAR", "0.26"))
_IRODORI_MIN_SEC = float(os.environ.get("L2_TTS_IRODORI_MIN_SEC", "0.6"))


def _short_text_seconds(spoken_text: str) -> float | None:
    """短文なら manual duration 秒、そうでなければ None (predictor 任せ) を返す。

    irodori は短文の尺を過剰予測して末尾を幻聴で埋める。閾値 (_IRODORI_SHORT_CHARS、
    既定 12 字) 以下のテキストは len×_IRODORI_SEC_PER_CHAR 秒 (下限 _IRODORI_MIN_SEC)
    の手動尺を返し、サイドカー側で predictor をバイパスさせる。閾値超は None
    (predictor は長文では尺が妥当なので任せる)。

    注: 文字数ベースの粗い見積もり。漢字密集の極短文は実モーラを過小評価しうるが、
    早口化 (内容保持) は幻聴 garble より軽微。env で rate/閾値を調整可能。
    """
    n = len(spoken_text.strip())
    if n == 0 or n > _IRODORI_SHORT_CHARS:
        return None
    return round(max(_IRODORI_MIN_SEC, n * _IRODORI_SEC_PER_CHAR), 2)


def _resolve_irodori_voice(voice: str) -> dict:
    """voice キーを voices.json から解決する (ref_wav は絶対パス化)。

    Returns:
        {"ref_wav": <abs path|None>, "caption": str, "seed": int|None,
         "cfg_scale_speaker": float, "num_steps": int, "t_schedule_mode": str}

    Raises:
        ValueError:          voices.json に未登録の voice
        FileNotFoundError:   voices.json が存在しない
    """
    data = _load_voices()
    voices = data.get("voices", {})
    entry = voices.get(voice)
    if entry is None:
        raise ValueError(
            f"irodori voice {voice!r} が voices.json にありません。登録済み: {sorted(voices)}"
        )
    common = data.get("common", {})
    realtime = common.get("realtime", {})
    ref_wav = None
    ref_wav_file = entry.get("ref_wav")
    if ref_wav_file:
        # ref_wav は voices.json のあるディレクトリ基準で解決する。
        ref_wav = str(_voices_json_path().resolve().parent / ref_wav_file)
    return {
        "ref_wav": ref_wav,
        "caption": entry.get("caption", ""),
        "seed": entry.get("seed"),
        "cfg_scale_speaker": float(common.get("cfg_scale_speaker", 5.0)),
        "num_steps": int(realtime.get("num_steps", 24)),
        "t_schedule_mode": str(realtime.get("t_schedule_mode", "sway")),
    }


def _irodori_control(voice: str, pose: str | None) -> tuple[str, str]:
    """キャラの pose を (本文末絵文字, caption サフィックス) に変換する (pose-only)。

    voices.json の pose_map[voice][pose] を引く。未対応 pose は ("", "") = neutral 扱い。
    emoji は ALLOWED_ANNOTATION_EMOJIS のみ (voices.json 側で担保)。

    irodori は声が固定で emotion 強度入力を持たないため、表現は pose と speed のみ
    (emotion dict は使わない)。
    """
    if not pose:
        return "", ""
    try:
        voices = _load_voices().get("voices", {})
    except FileNotFoundError:
        return "", ""
    pose_map = voices.get(voice, {}).get("pose_map", {})
    ctrl = pose_map.get(pose)
    if not ctrl:
        return "", ""
    return ctrl.get("emoji", ""), ctrl.get("caption_suffix", "")


def _generate_irodori_single_file(
    chunk_text: str,
    *,
    caption: str,
    ref_wav: str | None,
    duration_scale: float,
    num_steps: int,
    t_schedule_mode: str,
    cfg_scale_speaker: float,
    seed: int | None,
    filepath: Path,
    url: str,
    seconds: float | None = None,
) -> tuple[int, int]:
    """1 チャンク分を irodori サイドカー (VoiceDesign) に合成依頼し WAV を保存する。

    サイドカーは PCM16 WAV を返すので、VOICEVOX 経路と同様に stdlib `wave` で
    duration / sample_rate を読む。

    Args:
        seconds: 手動 duration (秒)。指定すると irodori の duration predictor を
            バイパスして尺を直接決める (短文の末尾幻聴抑制用)。None なら predictor。

    Returns:
        (duration_ms, sample_rate)
    """
    import io
    import json as _json
    import urllib.error
    import urllib.request
    import wave

    payload: dict = {
        "mode": "vd",
        "text": chunk_text,
        "caption": caption,
        "duration_scale": duration_scale,
        "num_steps": num_steps,
        "t_schedule_mode": t_schedule_mode,
        "cfg_scale_speaker": cfg_scale_speaker,
    }
    if ref_wav:
        payload["ref_wav"] = ref_wav
    if seed is not None:
        payload["seed"] = seed
    if seconds is not None:
        # 手動 duration: 短文の尺過剰予測 (末尾幻聴) を回避する。predictor をバイパス。
        payload["seconds"] = seconds

    timeout = float(os.environ.get("L2_TTS_IRODORI_TIMEOUT_SEC", "120"))
    body = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            wav_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        # サイドカーが返した JSON エラーメッセージを拾って例外に載せる
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"irodori サイドカー HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"irodori サイドカーに接続できません ({url}): {exc.reason}。"
            f" run_irodori_sidecar.ps1 で起動済みか確認してください。"
        ) from exc

    with wave.open(io.BytesIO(wav_bytes)) as wf:
        duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
        sample_rate = wf.getframerate()

    filepath.write_bytes(wav_bytes)
    logger.info(
        "irodori WAV 保存: %s (%d bytes, %d ms)", filepath.name, len(wav_bytes), duration_ms
    )
    return duration_ms, sample_rate


def _call_irodori(
    text: str,
    *,
    voice: str,
    output_dir: str,
    speaker: str = "",
    speed: int | None = None,
    on_chunk_ready=None,
    **kwargs,
) -> TTSResult:
    """irodori-TTS サイドカー (VoiceDesign) 経由で音声合成する (pose-only)。

    VOICEPEAK / VOICEVOX と同じく:
      - JSON 応答 (_parse_voicepeak_json) から say_text / speed / pose を抽出
        (emotion は irodori では未使用)
      - 140 字でチャンク分割し、各チャンクを個別合成 + on_chunk_ready 通知
      - 結合 WAV は作らず chunk_audio_urls で全チャンクを返す

    pose-only モデル (PoC 確定):
      - voice → voices.json (caption / self-ref アンカー / seed / cfg / num_steps / schedule)
      - pose → 本文末の正規絵文字 + caption 末尾サフィックス
      - speed → duration_scale = max(0.85, 100/speed)

    Args:
        voice:  irodori voice キー (voices.json の voices.<key>、例 "mimi")
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON 応答から本文 / speed / pose を分離 (emotion は irodori では未使用)。
    say_text, _json_emotion, json_speed, json_pose = _parse_voicepeak_json(text)
    effective_speed = json_speed if json_speed is not None else speed
    duration_scale = _speed_to_duration_scale(effective_speed)

    entry = _resolve_irodori_voice(voice)
    base_caption = entry["caption"]
    ref_wav = entry["ref_wav"]
    cfg_scale_speaker = entry["cfg_scale_speaker"]

    url = _irodori_url()
    # num_steps / schedule は voices.json (common.realtime) 由来。env で上書き可。
    num_steps = int(os.environ.get("L2_TTS_IRODORI_NUM_STEPS", str(entry["num_steps"])))
    t_schedule_mode = os.environ.get("L2_TTS_IRODORI_SCHEDULE", entry["t_schedule_mode"])
    # seed はキャラ毎固定 (voices.json)。env L2_TTS_IRODORI_SEED があれば全キャラ上書き。
    seed_env = os.environ.get("L2_TTS_IRODORI_SEED", "")
    seed = int(seed_env) if seed_env.strip() else entry["seed"]

    # pose → emoji (本文末) + caption サフィックス (pose-only。emotion は使わない)。
    # ターン単位で 1 回作り全チャンクに適用する。
    emoji, suffix = _irodori_control(voice, json_pose)
    caption_for_request = f"{base_caption}{suffix}" if suffix else base_caption

    chunks = _split_text_for_voicepeak(say_text)
    logger.info(
        "irodori チャンク分割: speaker=%s %d 個 (元 %d 文字) ds=%.3f pose=%s emoji=%r",
        speaker or "(unknown)", len(chunks), len(say_text), duration_scale,
        json_pose, emoji,
    )

    chunk_paths: list[Path] = []
    chunk_durations: list[int] = []
    sample_rate = 48000

    for i, chunk_text in enumerate(chunks):
        # 喋るテキストにだけ読み補正を適用 (HUD は元 chunk_text のまま)。
        # 読み辞書 → 小数正規化 → 末尾に pose 絵文字、の順で request_text を組む。
        spoken_text = _apply_readings(chunk_text, voice)
        spoken_text = _normalize_decimals(spoken_text)  # "5.5" → "5てんご"
        request_text = f"{spoken_text}{emoji}" if emoji else spoken_text
        # 短文は manual duration で predictor をバイパス (末尾幻聴抑制)。ただし pose 絵文字が
        # 付く場合は predictor に任せる (seconds=None): 絵文字は笑い声等の発声を生むため、
        # spoken_text 基準の短い尺だとその発声が途中で切れる (実測 probe23: 笑いが詰まる)。
        # 絵文字付きは予測器が発声分も含めて尺を取り、笑いが完走する (過剰予測=幻聴も出ない)。
        seconds = None if emoji else _short_text_seconds(spoken_text)
        filepath = out_dir / f"{uuid.uuid4()}.wav"
        dur, sr = _generate_irodori_single_file(
            request_text,
            caption=caption_for_request,
            ref_wav=ref_wav,
            duration_scale=duration_scale,
            num_steps=num_steps,
            t_schedule_mode=t_schedule_mode,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=seed,
            filepath=filepath,
            url=url,
            seconds=seconds,
        )
        chunk_paths.append(filepath)
        chunk_durations.append(dur)
        sample_rate = sr
        if on_chunk_ready:
            # chunk_text は元テキスト (絵文字なし) を渡す: HUD/ログ表示を綺麗に保つ。
            on_chunk_ready(filepath.as_uri(), chunk_text, i == len(chunks) - 1, speaker)

    audio_url = chunk_paths[0].as_uri() if chunk_paths else ""
    total_duration = sum(chunk_durations)

    return TTSResult(
        audio_url=audio_url,
        duration_ms=total_duration,
        voice=voice,
        format="wav",
        sample_rate=sample_rate,
        speaker=speaker or f"irodori-{voice}",
        chunk_audio_urls=[p.as_uri() for p in chunk_paths],
    )


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "edge_tts": _call_edge_tts,
    "voicevox": _call_voicevox,
    "voicepeak": _call_voicepeak,
    "irodori_vd": _call_irodori,
}


# emotion を JSON ({response, emotion}) で受け取る provider。
# VOICEPEAK のみ (irodori_vd は pose-only で emotion を使わない)。将来 VOICEPEAK 採用
# キャラがいる場合に、フィラー生成時の emotion JSON 包装判定に使う (filler.py が参照)。
_EMOTION_JSON_PROVIDERS = frozenset({"voicepeak"})


def provider_uses_emotion_json(provider: str) -> bool:
    """provider が emotion を JSON 経由で受け取るか (= フィラーで JSON 包装すべきか) を返す。"""
    return provider in _EMOTION_JSON_PROVIDERS


# ─── 公開 API ────────────────────────────────────────────────────

def synthesize(
    text: str,
    *,
    provider: str = "voicevox",
    voice: str,
    output_dir: str = "./data/audio",
    **kwargs,
) -> TTSResult:
    """
    テキストを音声合成して TTSResult を返す。

    Args:
        text:       合成するテキスト
        provider:   TTS プロバイダ（"edge_tts" or "voicevox"）
        voice:      音声識別子（provider 依存。edge_tts は "ja-JP-NanamiNeural" など）
        output_dir: 音声ファイルの保存先ディレクトリ
        **kwargs:   provider 固有のオプション（speaker 等）

    Returns:
        TTSResult

    Raises:
        ValueError:  未対応 provider、または voice が不正
        ImportError: provider のパッケージが未インストール (edge_tts のみ)
    """
    fn = _PROVIDERS.get(provider)
    if fn is None:
        supported = ", ".join(f'"{p}"' for p in _PROVIDERS)
        raise ValueError(
            f"未対応の provider: {provider!r}。対応プロバイダ: {supported}"
        )

    # 並行 / バックグラウンド合成中に「誰の何を合成中か」を即座に追えるよう、
    # speaker と text 先頭をログに含める。
    speaker_for_log = kwargs.get("speaker") or "(unknown)"
    text_preview = text if len(text) <= 40 else text[:40] + "…"
    logger.info(
        "TTS 開始: speaker=%s provider=%s voice=%s text_len=%d text=%r",
        speaker_for_log, provider, voice, len(text), text_preview,
    )
    result: TTSResult = fn(text, voice=voice, output_dir=output_dir, **kwargs)
    # audio_url は file:///T:/Users/... のような絶対パスになるため、配信中の
    # コンソール表示でユーザーディレクトリが漏れないようファイル名のみログ
    audio_filename = result.audio_url.rsplit("/", 1)[-1] if result.audio_url else ""
    logger.info(
        "TTS 完了: speaker=%s duration_ms=%d audio=%s",
        speaker_for_log, result.duration_ms, audio_filename,
    )
    logger.debug("TTS 完了 audio_url (full): %s", result.audio_url)
    return result
