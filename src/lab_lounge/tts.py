"""
tts.py — TTS アダプタ

責務:
  - TTS API 呼び出しをこのファイルに閉じ込める
  - provider 切替しやすい構造にする（llm.py と同構造）
  - synthesize() は TTSResult を返す
  - pipeline.py から呼ばれる

【対応 provider】
  "edge_tts" : Microsoft Edge TTS (edge-tts パッケージ必要)
  "voicevox"  : VOICEVOX Engine HTTP API (ローカルサーバ必要)

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

    return TTSResult(
        audio_url=filepath.as_uri(),
        duration_ms=duration_ms,
        voice=voice,
        format="wav",
        sample_rate=24000,
        speaker=speaker or f"voicevox-{speaker_id}",
    )


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "edge_tts": _call_edge_tts,
    "voicevox": _call_voicevox,
}


# ─── 公開 API ────────────────────────────────────────────────────

def synthesize(
    text: str,
    *,
    provider: str = "edge_tts",
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
    logger.info(
        "TTS 完了: duration_ms=%d audio_url=%s",
        result.duration_ms, result.audio_url,
    )
    return result
