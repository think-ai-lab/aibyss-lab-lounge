"""
wake_word.py — Porcupine ウェイクワード検知

責務:
  - Porcupine (pvporcupine) を使ってウェイクワードを検知する
  - 検知結果からキャラクター slug を解決する
  - マイクアクセスの排他制御は呼び出し元 (run_loop.py) の責務

【前提パッケージ】
  uv sync --extra wake    → pvporcupine>=3.0
  uv sync --extra mic     → sounddevice (マイク入力)

【環境変数】
  L2_PORCUPINE_ACCESS_KEY  — Picovoice アクセスキー (必須)
  L2_PORCUPINE_MODEL_DIR   — .ppn / .pv ファイルの配置先 (default: ./porcupine/)

【v3/v4 モデル混在】
  octamaid は v4 モデル、他は v3。同一 Porcupine インスタンスでの混在が
  不可能な場合、v4 モデルはスキップしてログ警告する。
"""

import logging
import os
import platform
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .characters import CharacterConfig, get_all_characters
from . import stt as _stt
from . import router as _router

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "porcupine"


@dataclass(frozen=True)
class WakeWordResult:
    """ウェイクワード検知結果。"""

    keyword: str                  # マッチしたキーワードテキスト (wake_word)
    character_slug: str           # 解決されたキャラクター slug
    keyword_index: int            # Porcupine 内の keyword index
    transcript: str | None = None # 転写テキスト (SpeechActivatedListener のみ使用)


def _resolve_model_dir() -> Path:
    """Porcupine モデルディレクトリを解決する。"""
    env_dir = os.environ.get("L2_PORCUPINE_MODEL_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return _DEFAULT_MODEL_DIR


def _get_platform_suffix() -> str:
    """現在のプラットフォームに対応する ppn ファイルのサフィックスを返す。"""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "mac"
    else:
        return "linux"


def _find_keyword_paths(
    characters: list[CharacterConfig],
    model_dir: Path,
) -> list[tuple[CharacterConfig, Path]]:
    """
    キャラクター一覧から Porcupine モデルファイルのパスを解決する。

    見つからないモデルはスキップしてログ警告を出す。
    """
    platform_suffix = _get_platform_suffix()
    results: list[tuple[CharacterConfig, Path]] = []

    for c in characters:
        if c.porcupine_model is None:
            continue

        model_path = model_dir / c.porcupine_model
        if model_path.is_file():
            results.append((c, model_path))
        else:
            # プラットフォーム違いの場合、同じキャラクターの別プラットフォームを探す
            logger.warning(
                "Porcupine モデルが見つかりません: %s (キャラクター: %s)",
                model_path,
                c.slug,
            )

    return results


class PorcupineListener:
    """Porcupine ウェイクワードリスナー。"""

    def __init__(self, *, access_key: str | None = None, model_dir: str | None = None):
        """
        Porcupine リスナーを初期化する。

        Args:
            access_key: Picovoice アクセスキー。省略時は L2_PORCUPINE_ACCESS_KEY env を使用。
            model_dir:  .ppn / .pv ファイルの配置先。省略時は L2_PORCUPINE_MODEL_DIR env。

        Raises:
            ImportError: pvporcupine が未インストール
            ValueError:  アクセスキーが設定されていない
            RuntimeError: Porcupine 初期化失敗
        """
        try:
            import pvporcupine
        except ImportError as exc:
            raise ImportError(
                "pvporcupine が必要です。"
                " uv sync --extra wake でインストールしてください。"
            ) from exc

        self._access_key = access_key or os.environ.get("L2_PORCUPINE_ACCESS_KEY", "")
        if not self._access_key:
            raise ValueError(
                "Porcupine アクセスキーが設定されていません。"
                " L2_PORCUPINE_ACCESS_KEY を設定してください。"
            )

        resolved_dir = Path(model_dir).resolve() if model_dir else _resolve_model_dir()
        characters = get_all_characters()
        keyword_entries = _find_keyword_paths(characters, resolved_dir)

        if not keyword_entries:
            raise RuntimeError("有効な Porcupine ウェイクワードモデルがありません。")

        # lang model
        lang_model = resolved_dir / "porcupine_params_ja.pv"
        if not lang_model.is_file():
            raise FileNotFoundError(
                f"Porcupine 日本語モデルが見つかりません: {lang_model}"
            )

        # キャラクター → index のマッピングを構築
        self._index_to_character: dict[int, CharacterConfig] = {}
        keyword_paths: list[str] = []

        for idx, (char, path) in enumerate(keyword_entries):
            self._index_to_character[idx] = char
            keyword_paths.append(str(path))

        logger.info(
            "Porcupine 初期化: %d キーワード (%s)",
            len(keyword_paths),
            ", ".join(c.slug for c, _ in keyword_entries),
        )

        try:
            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                keyword_paths=keyword_paths,
                model_path=str(lang_model),
            )
        except Exception as exc:
            raise RuntimeError(f"Porcupine 初期化失敗: {exc}") from exc

        self._frame_length = self._porcupine.frame_length
        self._sample_rate = self._porcupine.sample_rate

    @property
    def frame_length(self) -> int:
        """Porcupine が必要とするフレーム長 (samples)。"""
        return self._frame_length

    @property
    def sample_rate(self) -> int:
        """Porcupine が必要とするサンプルレート (Hz)。"""
        return self._sample_rate

    def listen_once(self, timeout_seconds: float = 30.0) -> WakeWordResult | None:
        """
        ウェイクワードを 1 回検知するまでマイクから読み取る。

        Args:
            timeout_seconds: タイムアウト秒数。超過で None を返す。

        Returns:
            WakeWordResult or None (タイムアウト)
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "sounddevice が必要です。"
                " uv sync --extra mic でインストールしてください。"
            ) from exc

        import time

        deadline = time.monotonic() + timeout_seconds
        logger.info("ウェイクワード待機開始 (timeout=%.0fs)", timeout_seconds)

        # InputStream で連続的に読み取る
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_length,
        ) as stream:
            while time.monotonic() < deadline:
                pcm, overflowed = stream.read(self._frame_length)
                if overflowed:
                    logger.debug("オーディオバッファオーバーフロー")

                # (N, 1) → (N,) の int16 配列
                keyword_index = self._porcupine.process(pcm[:, 0])

                if keyword_index >= 0:
                    char = self._index_to_character[keyword_index]
                    logger.info(
                        "ウェイクワード検知: %s → %s",
                        char.wake_word,
                        char.slug,
                    )
                    return WakeWordResult(
                        keyword=char.wake_word or char.slug,
                        character_slug=char.slug,
                        keyword_index=keyword_index,
                    )

        logger.info("ウェイクワード待機タイムアウト")
        return None

    def cleanup(self) -> None:
        """Porcupine リソースを解放する。"""
        if hasattr(self, "_porcupine") and self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
            logger.info("Porcupine リソース解放完了")

    def __del__(self) -> None:
        self.cleanup()


# ─── SpeechActivatedListener ─────────────────────────────────────

_DEFAULT_VAD_HOLD_FRAMES = 8      # 発話開始に必要な連続超閾値フレーム数
_DEFAULT_SILENCE_FRAMES = 24      # 録音終了に必要な連続無音フレーム数 (~1.5秒)
_DEFAULT_SAMPLE_RATE = 16000      # サンプルレート (Hz)
_DEFAULT_FRAME_SAMPLES = 512      # 1フレームのサンプル数 (~32ms)
_DEFAULT_MAX_RECORD_SECONDS = 30.0


def _build_character_prompt() -> str:
    """
    キャラクターの wake_word・display_name・aliases からプロンプト文字列を生成する。

    Whisper の prompt / initial_prompt に渡すことで固有名詞の転写精度を向上させる。
    例: "ミミ様、ミミ・オクタヴィア、ちさめさん、波心ちさめ、さくらさん、八重笠さくら、オクタメイド"
    """
    chars = get_all_characters()
    names: list[str] = []
    for c in chars:
        if c.wake_word:
            names.append(c.wake_word)
        names.append(c.display_name)
        names.extend(c.aliases)
    # 重複を除去しつつ順序を保持
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return "、".join(unique)


class SpeechActivatedListener:
    """
    VAD (RMS 閾値) + STT によるウェイクワード不要のリスナー。

    sounddevice で音量を監視し発話開始を検知 → 録音 → Whisper STT →
    router.route() でキャラクターを決定する。
    Porcupine モデルファイルなしで全キャラクターに対応可能。
    """

    def __init__(
        self,
        *,
        vad_threshold: float | None = None,
        device: int | str | None = None,
        stt_provider: str = "openai",
        stt_lang: str = "ja",
        stt_prompt: str | None = None,
        tmp_dir: str | None = None,
        max_record_seconds: float = _DEFAULT_MAX_RECORD_SECONDS,
    ) -> None:
        """
        Args:
            vad_threshold:      RMS 閾値。省略時は L2_SILENCE_THRESHOLD env (default 0.01)。
            device:             録音デバイスのインデックスまたは名前。
            stt_provider:       STT プロバイダ (デフォルト: "openai")。
            stt_lang:           STT 言語コード (デフォルト: "ja")。
            stt_prompt:         STT 転写ヒント。省略時はキャラクター名から自動生成。
            tmp_dir:            WAV 一時ファイルの保存先。
            max_record_seconds: 1 発話の最大録音秒数。
        """
        self._vad_threshold = vad_threshold if vad_threshold is not None else float(
            os.environ.get("L2_SILENCE_THRESHOLD", "0.01")
        )
        self._device = device
        self._stt_provider = stt_provider
        self._stt_lang = stt_lang
        self._stt_prompt = stt_prompt if stt_prompt is not None else _build_character_prompt()
        self._tmp_dir = tmp_dir
        self._max_record_seconds = max_record_seconds

    def listen_once(self, timeout_seconds: float = 30.0) -> "WakeWordResult | None":
        """
        発話を 1 回検知して WakeWordResult を返す。

        Phase 1: RMS が閾値を超えるフレームが _DEFAULT_VAD_HOLD_FRAMES 連続 → 発話開始
        Phase 2: 無音フレームが _DEFAULT_SILENCE_FRAMES 連続 or max_record_seconds → 録音終了
        その後: WAV 書き出し → STT → route() → WakeWordResult

        Args:
            timeout_seconds: Phase 1 のタイムアウト秒数。超過で None を返す。
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "sounddevice が必要です。"
                " uv sync --extra mic でインストールしてください。"
            ) from exc

        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "soundfile が必要です。"
                " uv sync --extra mic でインストールしてください。"
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "numpy が必要です。"
                " uv sync --extra mic でインストールしてください。"
            ) from exc

        import time

        sample_rate = _DEFAULT_SAMPLE_RATE
        frame_samples = _DEFAULT_FRAME_SAMPLES
        threshold = self._vad_threshold

        deadline = time.monotonic() + timeout_seconds
        max_record_frames = int(self._max_record_seconds * sample_rate / frame_samples)

        logger.info("音声待機開始 (timeout=%.0fs threshold=%.4f)", timeout_seconds, threshold)

        onset_count = 0
        silence_count = 0
        recording: list = []
        recording_started = False
        # onset フレームをプリバッファ: 検知確定時に録音先頭に含める
        # → 発話の冒頭が欠落するのを防ぐ
        onset_buffer: deque = deque(maxlen=_DEFAULT_VAD_HOLD_FRAMES)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=self._device,
        ) as stream:
            while time.monotonic() < deadline:
                pcm, overflowed = stream.read(frame_samples)
                if overflowed:
                    logger.debug("オーディオバッファオーバーフロー")

                # RMS 計算 (int16 → float 正規化)
                frame = pcm[:, 0].astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(frame ** 2)))

                if not recording_started:
                    # Phase 1: 発話開始検知
                    onset_buffer.append(pcm[:, 0].copy())  # 常にバッファ

                    if rms >= threshold:
                        onset_count += 1
                    else:
                        onset_count = 0

                    if onset_count >= _DEFAULT_VAD_HOLD_FRAMES:
                        recording_started = True
                        silence_count = 0
                        # onset フレームを録音先頭に含める（冒頭欠落防止）
                        recording = list(onset_buffer)
                        logger.info("発話開始検知")
                else:
                    # Phase 2: 録音中
                    recording.append(pcm[:, 0].copy())

                    if rms < threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if (
                        silence_count >= _DEFAULT_SILENCE_FRAMES
                        or len(recording) >= max_record_frames
                    ):
                        logger.info(
                            "発話終了検知 (frames=%d silence=%d)",
                            len(recording),
                            silence_count,
                        )
                        break
            else:
                logger.info("音声待機タイムアウト")
                return None

        if not recording:
            logger.info("録音データなし")
            return None

        # WAV 書き出し
        audio_data = np.concatenate(recording)
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=self._tmp_dir,
            delete=False,
        )
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            sf.write(tmp_path, audio_data, sample_rate, subtype="PCM_16")

            # STT
            stt_result = _stt.transcribe_audio_file(
                tmp_path,
                provider=self._stt_provider,
                lang=self._stt_lang,
                prompt=self._stt_prompt or None,
            )
            transcript = stt_result.text.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info("転写結果: %r", transcript)

        # キャラクター判定
        decision = _router.route(transcript)
        char_slug = decision.speaker

        # キャラクター設定から wake_word を取得
        from .characters import get_all_characters as _get_chars
        chars = {c.slug: c for c in _get_chars()}
        char = chars.get(char_slug)
        keyword = (char.wake_word if char and char.wake_word else None) or char_slug

        logger.info("音声ルーティング: %r → %s", transcript, char_slug)
        return WakeWordResult(
            keyword=keyword,
            character_slug=char_slug,
            keyword_index=0,
            transcript=transcript,
        )

    def cleanup(self) -> None:
        """no-op。PorcupineListener との互換性のため存在する。"""
