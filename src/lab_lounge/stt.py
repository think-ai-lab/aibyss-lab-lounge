"""
stt.py — STT アダプタ

責務:
  - 音声ファイル → テキスト変換をこのファイルに閉じ込める
  - provider 切替しやすい構造にする（llm.py / tts.py と同構造）
  - transcribe_audio_file() は STTResult を返す
  - emitter.py から呼ばれる（pipeline.py は STT を直接呼ばない）

【対応 provider】
  "openai"        : OpenAI Whisper API (openai パッケージ必要)
                    同じ OPENAI_API_KEY で LLM / STT 両方使える
  "faster-whisper": ローカル Whisper (faster-whisper パッケージ必要)
                    uv sync --extra stt-local

【prompt / initial_prompt について】
  transcribe_audio_file(path, prompt="ミミ様、ちさめさん、...") のように渡すと
  OpenAI Whisper では prompt パラメータ、faster-whisper では initial_prompt に
  転送される。キャラクター固有名詞を含めると転写精度が向上する。

【前提パッケージ (real mode)】
  uv sync --extra stt              → OpenAI Whisper API
  uv sync --extra stt-local        → faster-whisper (ローカル)
  環境変数: OPENAI_API_KEY=sk-...  (openai プロバイダのみ)

【STTResult の各フィールド】
  text         : 書き起こしテキスト
  confidence   : 信頼度 (0.0-1.0 or None — Whisper API は返さない)
  lang         : 実際に認識した言語コード (例: "ja")
  duration_ms  : 音声ファイルの時間長 [ms]
  words        : 単語タイムスタンプ (optional)
                 [{"word": str, "start": float, "end": float}, ...]
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
    prompt: str | None = None,
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

    create_kwargs: dict = dict(
        model=model,
        language=lang,
        response_format="verbose_json",
    )
    if prompt:
        create_kwargs["prompt"] = prompt

    with audio_path.open("rb") as audio_file:
        create_kwargs["file"] = audio_file
        transcript = client.audio.transcriptions.create(**create_kwargs)

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


# ─── faster-whisper ローカル provider ────────────────────────────────

# モデルインスタンスを process 内でキャッシュする
_faster_whisper_cache: dict = {}


def _get_faster_whisper_model(model_name: str, device: str, compute_type: str):
    """WhisperModel をキャッシュして返す。初回のみダウンロード+ロードが発生する。"""
    key = (model_name, device, compute_type)
    if key not in _faster_whisper_cache:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper が必要です。"
                " uv sync --extra stt-local でインストールしてください。"
            ) from exc
        logger.info(
            "faster-whisper モデルをロード中: model=%s device=%s compute_type=%s",
            model_name, device, compute_type,
        )
        _faster_whisper_cache[key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
    return _faster_whisper_cache[key]


def _call_faster_whisper(
    path: str,
    *,
    lang: str,
    model: str = "large-v3",
    device: str | None = None,
    compute_type: str | None = None,
    prompt: str | None = None,
    beam_size: int = 5,
    **kwargs,
) -> STTResult:
    """
    faster-whisper (ローカル) で音声ファイルを文字起こしする。

    Args:
        path:         音声ファイルのパス
        lang:         言語コード (例: "ja", "en")
        model:        Whisper モデル名 (デフォルト: "large-v3")
                      "large-v3" が最高精度。速度優先なら "medium" / "small"。
        device:       "cuda" (default) または "cpu"。L2_STT_DEVICE env で上書き可能。
        compute_type: GPU 時は "float16" 推奨。L2_STT_COMPUTE_TYPE env で上書き可能。
                      CPU 時は "int8"、GPU 時は "float16" が自動選択される。
        prompt:       初期プロンプト (固有名詞等)。initial_prompt に転送。
        beam_size:    ビームサーチ幅 (大きいほど精度向上・速度低下)
    """
    resolved_device = device or os.environ.get("L2_STT_DEVICE", "cuda")
    default_compute = "float16" if resolved_device == "cuda" else "int8"
    resolved_compute = compute_type or os.environ.get("L2_STT_COMPUTE_TYPE", default_compute)
    wmodel = _get_faster_whisper_model(model, resolved_device, resolved_compute)

    segments, info = wmodel.transcribe(
        path,
        language=lang,
        initial_prompt=prompt or None,
        beam_size=beam_size,
    )
    text = "".join(seg.text for seg in segments)
    duration_ms = int(info.duration * 1000) if info.duration else _file_duration_ms(path)

    return STTResult(
        text=text,
        confidence=None,
        lang=info.language or lang,
        duration_ms=duration_ms,
    )


# ─── プロバイダ登録テーブル ─────────────────────────────────────────

_PROVIDERS: dict = {
    "openai": _call_openai_whisper,
    "faster-whisper": _call_faster_whisper,
}


# ─── 公開 API ────────────────────────────────────────────────────

def transcribe_audio_file(
    path: str,
    *,
    provider: str = "openai",
    lang: str = "ja",
    prompt: str | None = None,
    **kwargs,
) -> STTResult:
    """
    音声ファイルを文字起こしして STTResult を返す。

    Args:
        path:     音声ファイルのパス
        provider: STT プロバイダ（"openai" | "faster-whisper"）
        lang:     認識言語 (ISO 639-1, 例: "ja", "en")
        prompt:   転写ヒント文字列。固有名詞を含めると精度向上。
                  openai → prompt パラメータ、faster-whisper → initial_prompt に転送。
        **kwargs: provider 固有のオプション（model / device / compute_type 等）

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

    # 配信中のコンソール表示でユーザーディレクトリ (例: C:\Users\xxx\AppData\Local\Temp)
    # を漏らさないため、ファイル名のみ出力。完全パスは debug レベルへ。
    logger.info(
        "STT 開始: provider=%s lang=%s file=%s",
        provider, lang, Path(path).name,
    )
    logger.debug("STT 開始 path (full): %s", path)
    result: STTResult = fn(path, lang=lang, prompt=prompt, **kwargs)
    logger.info(
        "STT 完了: text_len=%d duration_ms=%d lang=%s",
        len(result.text), result.duration_ms, result.lang,
    )
    return result
