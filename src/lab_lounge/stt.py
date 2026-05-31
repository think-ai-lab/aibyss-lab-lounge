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


# ─── Whisper hallucination 抑止 (Phase 0.5-A フェーズ 0) ────────────
#
# Whisper の典型的 hallucination パターン。訓練データに YouTube 動画の音声 +
# 字幕が大量に含まれており、配信終了の定型句が日本語データに偏在している。
# その結果、無音 / ノイズの入力に対して以下のような「らしい」フレーズを生成
# してしまう問題が知られている。
#
# パターンマッチで除外することで、配信中の環境ノイズによる誤検知を抑止する。
# faster-whisper のパラメータ (compression_ratio_threshold,
# log_prob_threshold, condition_on_previous_text=False) と併用してさらに
# 抑止精度を上げる。
#
# 環境変数:
#   L2_STT_HALLUCINATION_FILTER (default: true)
#       false にするとフィルタを無効化 (デバッグ用)。
#   L2_STT_HALLUCINATION_EXTRA_PATTERNS
#       カンマ区切りで追加パターンを指定可能 (運用ログで観測されたものを足す)。

_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    # YouTube 字幕由来の配信終了定型句
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴ありがとう",
    "ご覧いただきありがとうございました",
    "ご覧いただきありがとうございます",
    "ご覧いただきありがとう",
    "また次回お会いしましょう",
    "また次回",
    "おやすみなさい",
    # 字幕クレジット
    "字幕作成",
    "字幕:",
    "字幕:",
    "by H.",
    "Subtitled by",
    # Phase 0.5-A 実走で観測 (2026-05-07): メイドカフェ系定型句の誤検知。
    # 注意: "お嬢様" 単独は mimi の alias なので追加禁止 (mention 経路で誤検知される)。
    # 完全な定型句 (フレーズ全体) のみパターン化する。
    "お嬢様のお帰りの日",
    "お嬢様のお帰り",
)


def _get_hallucination_extra_patterns() -> tuple[str, ...]:
    """環境変数から追加パターンを読み込む (運用観測の追加対応用)."""
    raw = os.environ.get("L2_STT_HALLUCINATION_EXTRA_PATTERNS", "")
    if not raw:
        return ()
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _is_hallucination_filter_enabled() -> bool:
    """hallucination フィルタが有効かどうか。"""
    return os.environ.get("L2_STT_HALLUCINATION_FILTER", "true").lower() in (
        "true", "1", "yes",
    )


def _is_likely_hallucination(text: str, audio_duration_ms: int) -> bool:
    """
    Whisper の典型的 hallucination パターンを検出する。

    判定ルール:
      1. パターンマッチ: 完全一致 or 末尾一致 (定型句で終わる発話を除外)
      2. 短時間 + 長文の不整合: 1 秒未満の録音で 10 文字以上の出力は不審
      3. 長フレーズ反復: 半分のフレーズが 2 回以上繰り返される
         (例: "ご視聴ありがとうご視聴ありがとう...")
      4. 短句反復 (Phase 0.5-A フェーズ 8): 短句 (2-8 文字) が 3 回以上繰り返される
         (例: "お嬢様、お嬢様、お嬢様、お嬢様、お嬢様" → ルール 3 では検出不可)

    Args:
        text:              Whisper の出力テキスト
        audio_duration_ms: 録音の時間長 [ms] (短時間判定用)

    Returns:
        True なら hallucination とみなして空文字化推奨。
    """
    if not text:
        return False

    # 末尾の句読点 / 三点リーダ / 感嘆符を除去して正規化
    normalized = text.strip().rstrip("。.！!？?…")
    if not normalized:
        return False

    # ルール 1: パターンマッチ (完全一致 or 末尾一致)
    all_patterns = _HALLUCINATION_PATTERNS + _get_hallucination_extra_patterns()
    for pattern in all_patterns:
        if normalized == pattern or normalized.endswith(pattern):
            return True

    # ルール 2: 短時間録音 (1 秒未満) で 10 文字以上の出力は不審
    if 0 < audio_duration_ms < 1000 and len(normalized) >= 10:
        return True

    # ルール 3: 長フレーズ反復検出 (半分のフレーズが 2 回以上繰り返される)
    if len(normalized) >= 12:
        half = normalized[: len(normalized) // 2]
        if half and normalized.count(half) >= 2:
            return True

    # ルール 4 (Phase 0.5-A フェーズ 8): 短句反復検出
    # ルール 3 では検出できない短句反復 (例: "お嬢様、お嬢様、...") を検出する。
    # prefix_len 2-8 文字の prefix が 3 回以上出現すればハルシネーションとみなす。
    # 短い prefix での 3 回反復は普通の発話ではほぼ起き得ない (= 視聴者向け話で
    # "そうそうそう" のような同句連続も普通は 1-2 回程度)。
    if len(normalized) >= 6:
        max_prefix_len = min(9, len(normalized) // 3 + 1)
        for prefix_len in range(2, max_prefix_len):
            prefix = normalized[:prefix_len]
            if prefix and normalized.count(prefix) >= 3:
                return True

    return False


def _apply_hallucination_filter(result: STTResult) -> STTResult:
    """
    STT 結果に hallucination フィルタを適用する。

    検出時は text を空文字化して返す (duration_ms / lang は維持)。
    フィルタ無効化時はそのまま返す。
    """
    if not _is_hallucination_filter_enabled():
        return result
    if _is_likely_hallucination(result.text, result.duration_ms):
        logger.warning(
            "STT hallucination 検出 → 空文字に置換: text=%r duration_ms=%d",
            result.text, result.duration_ms,
        )
        return STTResult(
            text="",
            confidence=result.confidence,
            lang=result.lang,
            duration_ms=result.duration_ms,
            words=result.words,
        )
    return result


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

    【hallucination 抑止パラメータ (Phase 0.5-A フェーズ 0 で追加)】
      環境変数で上書き可能:
        L2_STT_COMPRESSION_RATIO_THRESHOLD (default 2.4):
            出力テキストの zlib 圧縮率がこの値を超えたら no-speech とみなす。
            反復出力 (例: "ご視聴ありがとうご視聴ありがとう...") を弾く。
        L2_STT_LOG_PROB_THRESHOLD (default -1.0):
            平均対数確率がこの値を下回ったら no-speech とみなす。
            低確信のテキストを弾く。
        L2_STT_NO_SPEECH_THRESHOLD (default 0.6):
            no_speech_prob がこの値を超えたら出力をスキップ。無音判定の閾値。
        L2_STT_CONDITION_ON_PREVIOUS_TEXT (default false):
            true にすると前のセグメントを context として使う (Whisper デフォルト)。
            false で hallucination の連鎖を防ぐ (Phase 0.5-A 推奨)。

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

    # hallucination 抑止パラメータ (環境変数で上書き可)
    compression_ratio_threshold = float(
        os.environ.get("L2_STT_COMPRESSION_RATIO_THRESHOLD", "2.4")
    )
    log_prob_threshold = float(
        os.environ.get("L2_STT_LOG_PROB_THRESHOLD", "-1.0")
    )
    no_speech_threshold = float(
        os.environ.get("L2_STT_NO_SPEECH_THRESHOLD", "0.6")
    )
    condition_on_previous_text = os.environ.get(
        "L2_STT_CONDITION_ON_PREVIOUS_TEXT", "false"
    ).lower() in ("true", "1", "yes")

    segments, info = wmodel.transcribe(
        path,
        language=lang,
        initial_prompt=prompt or None,
        beam_size=beam_size,
        # hallucination 抑止パラメータ群
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
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

    # Phase 0.5-A フェーズ 0: hallucination フィルタ適用
    # provider 共通で適用 (openai / faster-whisper どちらでも有効)。
    # 検出時は text を空文字化して、buffer 蓄積 / check_intent への流入を抑止する。
    result = _apply_hallucination_filter(result)

    logger.info(
        "STT 完了: text_len=%d duration_ms=%d lang=%s",
        len(result.text), result.duration_ms, result.lang,
    )
    return result
