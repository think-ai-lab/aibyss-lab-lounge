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

def _call_voicevox(
    text: str,
    *,
    voice: str,
    output_dir: str,
    speaker: str = "",
    **kwargs,
) -> TTSResult:
    """
    VOICEVOX Engine HTTP API を使って音声合成する（stdlib のみ）。

    voice には VOICEVOX のスピーカー ID（整数文字列）を渡す。
      例: "3" → ずんだもん
    VOICEVOX Engine のベース URL は環境変数 L2_TTS_VOICEVOX_URL で変更できる
    (デフォルト: http://localhost:50021)

    API フロー:
      1. POST /audio_query?text=<>&speaker=<id>  → audio_query JSON
      2. POST /synthesis?speaker=<id> + audio_query body → WAV bytes
    """
    import io
    import json as _json
    import urllib.parse
    import urllib.request
    import wave

    voicevox_url = os.environ.get("L2_TTS_VOICEVOX_URL", "http://localhost:50021")
    try:
        speaker_id = int(voice)
    except ValueError as exc:
        raise ValueError(
            f"voicevox provider の voice はスピーカー ID の整数文字列（例: '3'）にしてください: {voice!r}"
        ) from exc

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

    # duration from WAV header
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)

    # 保存
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{uuid.uuid4()}.wav"
    filepath.write_bytes(wav_bytes)
    logger.info("VOICEVOX WAV 生成完了: %s (%d bytes)", filepath.name, len(wav_bytes))

    return TTSResult(
        audio_url=filepath.as_uri(),
        duration_ms=duration_ms,
        voice=voice,
        format="wav",
        sample_rate=24000,
        speaker=speaker or f"voicevox-{speaker_id}",
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


def _parse_voicepeak_json(
    text: str,
) -> tuple[str, dict[str, int] | None, int | None, str | None]:
    """
    LLM 応答が JSON 構造の場合、response / emotion / speed / pose を分離する。

    対応する JSON 形式:
        {"emotion": {"happy": 50, ...}, "speed": 100, "pose": "happy", "response": "テキスト"}

    全角記号に正規化済みの JSON も半角に戻してからパースを試みる。

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
    """並列実行エラー発生時のリトライ待機時間（秒）を返す。

    並列実行エラー検知時のみ挿入される (通常成功時は待機しない)。
    環境変数 L2_VOICEPEAK_RETRY_WAIT_SEC で上書き可能 (デフォルト: 2.0)。
    """
    return float(os.environ.get("L2_VOICEPEAK_RETRY_WAIT_SEC", "2.0"))


def _get_voicepeak_max_retries() -> int:
    """並列実行エラー発生時の最大リトライ回数を返す。"""
    return int(os.environ.get("L2_VOICEPEAK_MAX_RETRIES", "2"))


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

            logger.warning(
                "VOICEPEAK 非ゼロ終了 (attempt %d/%d): returncode=%d busy=%s"
                "\n  stderr: %s\n  stdout: %s",
                attempt + 1, max_retries + 1,
                result.returncode, is_busy,
                stderr_text or "(empty)",
                stdout_text or "(empty)",
            )

            if is_busy and attempt < max_retries:
                logger.info(
                    "VOICEPEAK 並列実行エラー検出 → %.1f 秒待機してリトライ",
                    retry_wait,
                )
                time.sleep(retry_wait)
                continue
            break  # それ以外のエラー or リトライ上限到達

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


def _submit_voicepeak(cmd_str: str) -> None:
    """
    VOICEPEAK コマンドをキューに投入し、完了を待つ。

    FIFO 順序が保証される。先に投入されたジョブが先に実行される。

    Raises:
        FileNotFoundError: VOICEPEAK コマンドが見つからない
        RuntimeError: VOICEPEAK 実行エラー
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
        logger.debug("VOICEPEAK 実行中に例外 cmd (full): %s", cmd_str)
        raise RuntimeError(f"VOICEPEAK 実行エラー: {type(exc).__name__}: {exc}") from exc

    if result.returncode != 0:
        # ワーカー側で既に詳細ログは出力済み。ここでは例外メッセージのみ組み立てる
        stderr_text = _decode_voicepeak_output(result.stderr) or "(empty)"
        stdout_text = _decode_voicepeak_output(result.stdout) or "(empty)"
        logger.error(
            "VOICEPEAK 実行最終失敗: returncode=%d stderr: %s stdout: %s",
            result.returncode, stderr_text, stdout_text,
        )
        logger.debug("VOICEPEAK 実行最終失敗 cmd (full): %s", cmd_str)
        raise RuntimeError(
            f"VOICEPEAK 実行エラー: returncode={result.returncode} "
            f"stderr={stderr_text!r} stdout={stdout_text!r}"
        )


def _generate_voicepeak_single_file(
    text: str,
    *,
    voice: str,
    filepath: Path,
    speed: int | None = None,
    emotion: dict[str, int] | None = None,
) -> tuple[int, int]:
    """
    VOICEPEAK CLI で 1 チャンク分の WAV を生成する。

    FIFO キューで排他制御される。投入順序が合成順序になる。

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
    # ログにはファイル名・narrator・テキスト長のみを出す。完全なコマンドは debug レベルへ。
    logger.info(
        "VOICEPEAK 投入: narrator=%s text_len=%d out=%s",
        voice, len(safe_text), filepath.name,
    )
    logger.debug("VOICEPEAK コマンド (full): %s", cmd_str)

    _submit_voicepeak(cmd_str)

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
    audio_url を通知する（ストリーミング再生用）。

    Args:
        voice:           ナレーター名
        speed:           発話速度（50〜200。省略時は VOICEPEAK デフォルト）
        on_chunk_ready:  チャンク生成完了時コールバック (audio_url: str) -> None
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON 構造のパース（emotion / speed / pose / response の分離）
    # pose は TTS 内では使用しない（pose.update は graph.py / pipeline.py 側で発行）
    say_text, json_emotion, json_speed, _json_pose = _parse_voicepeak_json(text)
    effective_speed = json_speed if json_speed is not None else speed

    # テキスト分割
    chunks = _split_text_for_voicepeak(say_text)
    logger.info("VOICEPEAK チャンク分割: %d 個 (元テキスト %d 文字)", len(chunks), len(say_text))

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
        )
        chunk_paths.append(filepath)
        chunk_durations.append(dur)
        sample_rate = sr
        logger.info(
            "VOICEPEAK チャンク %d/%d 生成完了: %d ms (%d 文字)",
            i + 1, len(chunks), dur, len(chunk_text),
        )
        if on_chunk_ready:
            on_chunk_ready(filepath.as_uri())

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


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "edge_tts": _call_edge_tts,
    "voicevox": _call_voicevox,
    "voicepeak": _call_voicepeak,
}


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

    logger.info(
        "TTS 開始: provider=%s voice=%s text_len=%d",
        provider, voice, len(text),
    )
    result: TTSResult = fn(text, voice=voice, output_dir=output_dir, **kwargs)
    # audio_url は file:///T:/Users/... のような絶対パスになるため、配信中の
    # コンソール表示でユーザーディレクトリが漏れないようファイル名のみログ
    audio_filename = result.audio_url.rsplit("/", 1)[-1] if result.audio_url else ""
    logger.info(
        "TTS 完了: duration_ms=%d audio=%s",
        result.duration_ms, audio_filename,
    )
    logger.debug("TTS 完了 audio_url (full): %s", result.audio_url)
    return result
