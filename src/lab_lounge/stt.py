"""
stt.py — STT アダプタ

責務:
  - 音声ファイル → テキスト変換をこのファイルに閉じ込める
  - provider 切替しやすい構造にする（llm.py / tts.py と同構造）
  - transcribe_audio_file() は STTResult を返す
  - emitter.py から呼ばれる（pipeline.py は STT を直接呼ばない）

【対応 provider】
  "openai" : OpenAI Whisper API (openai パッケージ必要)
             同じ OPENAI_API_KEY で LLM / STT 両方使える

【前提パッケージ (real mode)】
  uv sync --extra stt
  環境変数: OPENAI_API_KEY=sk-...

【STTResult の各フィールド】
  text         : 書き起こしテキスト
  confidence   : 信頼度 (0.0-1.0 or None — Whisper API は返さない)
  lang         : 実際に認識した言語コード (例: "ja")
  duration_ms  : 音声ファイルの時間長 [ms]
  words        : 単語タイムスタンプ (optional)
                 [{"word": str, "start": float, "end": float}, ...]

【audio_query の形式】
  将来 timestamp_granularities=["word"] を有効化すると words が埋まる。
  現在は response_format="verbose_json" のみ使用し words は None のまま。
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── 戻り値型 ─────────────────────────────────────────────────────

@dataclass
class STTResult:
    """STT 呼び出し結果。emitter.py から参照する。"""

    text: str
    confidence: float | None
    lang: str
    duration_ms: int
    words: list | None = None


# ─── OpenAI Whisper API adapter ──────────────────────────────────

def _call_openai_whisper(
    path: str,
    *,
    lang: str,
    model: str = "whisper-1",
    **kwargs,
) -> STTResult:
    """
    OpenAI Whisper API を使って音声ファイルを文字起こしする。

    openai パッケージが未インストールの場合は ImportError を送出する。
    API キーは OPENAI_API_KEY 環境変数から取得する（LLM と共用可）。

    Args:
        path:  音声ファイルのパス (WAV / MP3 / M4A / OGG 等を受け付ける)
        lang:  言語コード (ISO 639-1, 例: "ja", "en")
        model: Whisper モデル名 (デフォルト: "whisper-1")
    """
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "openai パッケージが必要です。"
            " uv sync --extra stt でインストールしてください。"
        ) from exc

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    client = openai.OpenAI()  # OPENAI_API_KEY を環境変数から自動取得

    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=lang,
            response_format="verbose_json",
        )

    duration_raw = getattr(transcript, "duration", None)
    duration_ms = int(float(duration_raw) * 1000) if duration_raw is not None else _file_duration_ms(path)

    # verbose_json の words フィールド (timestamp_granularities が必要。現在は None)
    words_raw = getattr(transcript, "words", None)
    words = None
    if words_raw:
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in words_raw
        ]

    return STTResult(
        text=transcript.text,
        confidence=None,  # Whisper API は per-transcript confidence を返さない
        lang=getattr(transcript, "language", None) or lang,
        duration_ms=duration_ms,
        words=words,
    )


def _file_duration_ms(path: str) -> int:
    """
    音声ファイルの時間長 [ms] を推定する。
    wav は stdlib wave で実測。mp3 等は mutagen があれば使用。なければ 0 を返す。
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(p), "rb") as wf:
                return int(wf.getnframes() / wf.getframerate() * 1000)
        except Exception:
            pass

    try:
        from mutagen import File as MFile
        audio = MFile(str(p))
        if audio is not None and audio.info is not None:
            return int(audio.info.length * 1000)
    except Exception:
        pass

    return 0


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "openai": _call_openai_whisper,
}


# ─── 公開 API ────────────────────────────────────────────────────

def transcribe_audio_file(
    path: str,
    *,
    provider: str = "openai",
    lang: str = "ja",
    **kwargs,
) -> STTResult:
    """
    音声ファイルを文字起こしして STTResult を返す。

    Args:
        path:     音声ファイルのパス
        provider: STT プロバイダ（"openai"）
        lang:     認識言語 (ISO 639-1, 例: "ja", "en")
        **kwargs: provider 固有のオプション（model 等）

    Returns:
        STTResult

    Raises:
        ValueError:          未対応 provider
        FileNotFoundError:   ファイルが存在しない
        ImportError:         provider のパッケージが未インストール
    """
    fn = _PROVIDERS.get(provider)
    if fn is None:
        supported = ", ".join(f'"{p}"' for p in _PROVIDERS)
        raise ValueError(
            f"未対応の provider: {provider!r}。対応プロバイダ: {supported}"
        )

    if not Path(path).is_file():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    logger.info(
        "STT 開始: provider=%s lang=%s path=%s",
        provider, lang, path,
    )
    result: STTResult = fn(path, lang=lang, **kwargs)
    logger.info(
        "STT 完了: text_len=%d duration_ms=%d lang=%s",
        len(result.text), result.duration_ms, result.lang,
    )
    return result
