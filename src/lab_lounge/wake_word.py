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
_DEFAULT_SILENCE_FRAMES = 24      # RMS 用: 録音終了に必要な連続無音フレーム数 (~768ms at 32ms/frame)
_WEBRTC_SILENCE_FRAMES = 75       # WebRTC 用: ~1.5秒 (75 × 20ms)
_DEFAULT_SAMPLE_RATE = 16000      # サンプルレート (Hz)
_DEFAULT_FRAME_SAMPLES = 512      # 1フレームのサンプル数 (~32ms)
_DEFAULT_MAX_RECORD_SECONDS = 30.0

# WebRTC VAD は 10ms/20ms/30ms のフレームのみ対応。
# 16kHz × 20ms = 320 サンプル。512 サンプル (32ms) は非対応のため、
# WebRTC 使用時はフレームサイズを 320 に変更する。
_WEBRTC_FRAME_SAMPLES = 320       # 20ms at 16kHz


def _get_vad_backend() -> str:
    """VAD バックエンドを返す。"""
    return os.environ.get("L2_VAD_BACKEND", "rms")


def _create_vad_checker(backend: str, threshold: float):
    """
    VAD 判定関数を返すファクトリ。

    Returns:
        (is_speech(pcm_int16_mono) -> bool, frame_samples, silence_frames)
    """
    if backend == "webrtc":
        try:
            import webrtcvad
        except ImportError as exc:
            logger.warning("webrtcvad 未インストール。RMS にフォールバック: %s", exc)
            return _create_vad_checker("rms", threshold)

        aggressiveness = int(os.environ.get("L2_VAD_AGGRESSIVENESS", "2"))
        vad = webrtcvad.Vad(aggressiveness)
        logger.info("WebRTC VAD 初期化 (aggressiveness=%d)", aggressiveness)

        def _check_webrtc(pcm_int16_mono) -> bool:
            return vad.is_speech(pcm_int16_mono.tobytes(), sample_rate=_DEFAULT_SAMPLE_RATE)

        return _check_webrtc, _WEBRTC_FRAME_SAMPLES, _WEBRTC_SILENCE_FRAMES

    # デフォルト: RMS
    import numpy as np

    def _check_rms(pcm_int16_mono) -> bool:
        frame = pcm_int16_mono.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms >= threshold

    return _check_rms, _DEFAULT_FRAME_SAMPLES, _DEFAULT_SILENCE_FRAMES


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
        vad_backend = _get_vad_backend()
        is_speech, frame_samples, silence_frames = _create_vad_checker(vad_backend, self._vad_threshold)

        deadline = time.monotonic() + timeout_seconds
        max_record_frames = int(self._max_record_seconds * sample_rate / frame_samples)

        logger.info(
            "音声待機開始 (timeout=%.0fs vad=%s threshold=%.4f)",
            timeout_seconds, vad_backend, self._vad_threshold,
        )

        onset_count = 0
        silence_count = 0
        recording: list = []
        recording_started = False
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

                mono = pcm[:, 0]
                speech = is_speech(mono)

                if not recording_started:
                    onset_buffer.append(mono.copy())

                    if speech:
                        onset_count += 1
                    else:
                        onset_count = 0

                    if onset_count >= _DEFAULT_VAD_HOLD_FRAMES:
                        recording_started = True
                        silence_count = 0
                        recording = list(onset_buffer)
                        logger.info("発話開始検知")
                else:
                    recording.append(mono.copy())

                    if not speech:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if (
                        silence_count >= silence_frames
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

        # Phase 0 名前ゲート: キャラクター名が検出されなかった場合は無視
        if decision.reason == "default":
            logger.info("名前ゲート: キャラクター名未検出。無視します: %r", transcript)
            return None

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


# ─── SherpaStreamingListener ──────────────────────────────────────

_DEFAULT_SHERPA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "sherpa-models"


def _find_sherpa_model_files(
    model_dir: Path,
    quantized: bool,
) -> tuple[Path, Path, Path]:
    """
    Sherpa モデルディレクトリからエンコーダ・デコーダ・ジョイナーを自動検出する。

    エポック番号はモデルによって異なる (epoch-99, epoch-35 等) ため、
    glob で自動検出する。quantized=True の場合は .int8.onnx を優先。
    """
    suffix = ".int8" if quantized else ""
    pattern = f"encoder-epoch-*-avg-*{suffix}.onnx"
    candidates = sorted(model_dir.glob(pattern))

    if not candidates:
        # quantized 指定でも見つからなければ fp16 → fp32 の順で探す
        for fallback in (".fp16", ""):
            fb_pattern = f"encoder-epoch-*-avg-*{fallback}.onnx"
            candidates = sorted(model_dir.glob(fb_pattern))
            if candidates:
                suffix = fallback
                break

    if not candidates:
        return (
            model_dir / f"encoder-epoch-99-avg-1{suffix}.onnx",
            model_dir / f"decoder-epoch-99-avg-1{suffix}.onnx",
            model_dir / f"joiner-epoch-99-avg-1{suffix}.onnx",
        )

    # エポック部分を抽出して decoder/joiner にも適用
    encoder = candidates[0]
    stem = encoder.name.replace("encoder-", "").replace(".onnx", "")
    # stem = "epoch-35-avg-1.int8" etc.
    decoder = model_dir / f"decoder-{stem}.onnx"
    joiner = model_dir / f"joiner-{stem}.onnx"

    return encoder, decoder, joiner


class SherpaStreamingListener:
    """
    Sherpa-ONNX OfflineRecognizer + VAD によるリスナー。

    Sherpa-ONNX の日本語 Zipformer (ReazonSpeech) モデルを使い、
    VAD で発話区間を検出 → 即座にローカル認識 → router.route() で
    キャラクターを決定する。WAV ファイル I/O なし・API コールなし。

    注: 日本語モデルは Offline モデルのため、発話区間が完了してから
    認識を実行する (真のストリーミングではない)。
    """

    def __init__(
        self,
        *,
        vad_threshold: float | None = None,
        device: int | str | None = None,
        model_dir: str | None = None,
        provider: str | None = None,
        quantized: bool = True,
        max_record_seconds: float = _DEFAULT_MAX_RECORD_SECONDS,
    ) -> None:
        """
        Args:
            vad_threshold:      RMS 閾値。省略時は L2_SILENCE_THRESHOLD env。
            device:             マイクデバイス。
            model_dir:          Sherpa モデルディレクトリ。省略時は L2_SHERPA_MODEL_DIR env。
            provider:           ONNX Runtime provider ("cuda" / "cpu")。省略時は L2_SHERPA_PROVIDER env。
            quantized:          True=int8 モデル、False=fp32 モデル。
            max_record_seconds: 1 発話の最大録音秒数。
        """
        try:
            import sherpa_onnx as _sherpa
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx が必要です。"
                " uv sync --extra stt-sherpa でインストールしてください。"
            ) from exc

        self._vad_threshold = vad_threshold if vad_threshold is not None else float(
            os.environ.get("L2_SILENCE_THRESHOLD", "0.01")
        )
        self._device = device
        self._max_record_seconds = max_record_seconds

        # モデルパス解決
        mdir = Path(
            model_dir
            or os.environ.get("L2_SHERPA_MODEL_DIR")
            or str(_DEFAULT_SHERPA_MODEL_DIR)
        )
        # サブディレクトリにモデルが配置されている場合を自動検出
        if not (mdir / "tokens.txt").is_file():
            subdirs = [d for d in mdir.iterdir() if d.is_dir() and (d / "tokens.txt").is_file()]
            if len(subdirs) == 1:
                mdir = subdirs[0]
                logger.info("Sherpa モデルサブディレクトリ検出: %s", mdir.name)

        encoder, decoder, joiner = _find_sherpa_model_files(mdir, quantized)
        tokens = mdir / "tokens.txt"

        for p in (encoder, decoder, joiner, tokens):
            if not p.is_file():
                raise FileNotFoundError(
                    f"Sherpa モデルファイルが見つかりません: {p}"
                )

        prov = provider or os.environ.get("L2_SHERPA_PROVIDER", "cuda")

        logger.info(
            "Sherpa-ONNX 初期化: provider=%s encoder=%s",
            prov,
            encoder.name,
        )
        self._recognizer = _sherpa.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=2,
            sample_rate=_DEFAULT_SAMPLE_RATE,
            feature_dim=80,
            provider=prov,
        )
        logger.info("Sherpa-ONNX 初期化完了")

    def listen_once(self, timeout_seconds: float = 30.0) -> "WakeWordResult | None":
        """
        発話を 1 回検知して WakeWordResult を返す。

        Phase 1: RMS ベース VAD で発話開始検知 (onset pre-buffer 付き)
        Phase 2: 無音検知で録音終了
        Phase 3: Sherpa-ONNX で認識 → route() → 名前ゲート

        WAV ファイルの書き出しは行わない。
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "sounddevice が必要です。"
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

        logger.info(
            "Sherpa 音声待機開始 (timeout=%.0fs threshold=%.4f)",
            timeout_seconds,
            threshold,
        )

        onset_count = 0
        silence_count = 0
        recording: list = []
        recording_started = False
        onset_buffer: deque = deque(maxlen=_DEFAULT_VAD_HOLD_FRAMES)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            device=self._device,
        ) as stream:
            while time.monotonic() < deadline:
                pcm, overflowed = stream.read(frame_samples)
                if overflowed:
                    logger.debug("オーディオバッファオーバーフロー")

                frame = pcm[:, 0]
                rms = float(np.sqrt(np.mean(frame ** 2)))

                if not recording_started:
                    onset_buffer.append(frame.copy())

                    if rms >= threshold:
                        onset_count += 1
                    else:
                        onset_count = 0

                    if onset_count >= _DEFAULT_VAD_HOLD_FRAMES:
                        recording_started = True
                        silence_count = 0
                        recording = list(onset_buffer)
                        logger.info("発話開始検知")
                else:
                    recording.append(frame.copy())

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
                logger.info("Sherpa 音声待機タイムアウト")
                return None

        if not recording:
            logger.info("録音データなし")
            return None

        # Sherpa-ONNX で認識 (WAV ファイル不要)
        audio_data = np.concatenate(recording)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio_data)
        self._recognizer.decode_stream(s)
        transcript = s.result.text.strip()

        logger.info("Sherpa 転写結果: %r", transcript)

        # キャラクター判定 (名前ゲート付き)
        decision = _router.route(transcript)

        if decision.reason == "default":
            logger.info("名前ゲート: キャラクター名未検出。無視します: %r", transcript)
            return None

        char_slug = decision.speaker

        from .characters import get_all_characters as _get_chars
        chars = {c.slug: c for c in _get_chars()}
        char = chars.get(char_slug)
        keyword = (char.wake_word if char and char.wake_word else None) or char_slug

        logger.info("Sherpa ルーティング: %r → %s", transcript, char_slug)
        return WakeWordResult(
            keyword=keyword,
            character_slug=char_slug,
            keyword_index=0,
            transcript=transcript,
        )

    def cleanup(self) -> None:
        """Sherpa-ONNX リソースを解放する。"""
        if hasattr(self, "_recognizer"):
            self._recognizer = None
            logger.info("Sherpa-ONNX リソース解放完了")


# ─── ContinuousListener ──────────────────────────────────────────


class ContinuousListener:
    """
    常時文字起こし + 循環バッファによるリスナー。

    SpeechActivatedListener と異なり、キャラクター名が検出されない
    発話もバッファに保持し、名前検出時に直前の文脈として活用する。
    faster-whisper でセグメント単位に transcript を生成する。
    """

    def __init__(
        self,
        *,
        vad_threshold: float | None = None,
        device: int | str | None = None,
        stt_provider: str = "faster-whisper",
        stt_lang: str = "ja",
        stt_prompt: str | None = None,
        tmp_dir: str | None = None,
        max_record_seconds: float = _DEFAULT_MAX_RECORD_SECONDS,
        context_window_sec: float | None = None,
        context_max_chars: int | None = None,
    ) -> None:
        from .transcript_buffer import TranscriptBuffer

        self._vad_threshold = vad_threshold if vad_threshold is not None else float(
            os.environ.get("L2_SILENCE_THRESHOLD", "0.01")
        )
        self._device = device
        self._stt_provider = stt_provider
        self._stt_lang = stt_lang
        self._stt_prompt = stt_prompt if stt_prompt is not None else _build_character_prompt()
        self._tmp_dir = tmp_dir
        self._max_record_seconds = max_record_seconds

        window = context_window_sec if context_window_sec is not None else float(
            os.environ.get("L2_CONTEXT_WINDOW_SEC", "30")
        )
        max_ch = context_max_chars if context_max_chars is not None else int(
            os.environ.get("L2_CONTEXT_MAX_CHARS", "2000")
        )
        self._buffer = TranscriptBuffer(window_sec=window, max_chars=max_ch)

        # VAD チェッカーの初期化（listen_once / _record_one_segment で再利用）
        vad_backend = _get_vad_backend()
        self._is_speech, self._vad_frame_samples, self._silence_frames = _create_vad_checker(
            vad_backend, self._vad_threshold
        )

    def _record_one_segment(
        self,
        stream,
        remaining_sec: float,
        *,
        np_module,
        sf_module,
    ):
        """
        1 発話セグメントを録音・転写して TranscriptSegment を返す。

        Phase 1: RMS onset 検知 (onset_buffer 付き)
        Phase 2: 無音検知で録音終了
        Phase 3: temp WAV → STT → 削除

        Args:
            stream:        開いている sd.InputStream
            remaining_sec: 残りタイムアウト秒数
            np_module:     numpy モジュール
            sf_module:     soundfile モジュール

        Returns:
            TranscriptSegment or None (タイムアウト or 空録音)
        """
        from .transcript_buffer import TranscriptSegment
        import time
        import numpy as np

        sf = sf_module
        sample_rate = _DEFAULT_SAMPLE_RATE
        is_speech = self._is_speech
        frame_samples = self._vad_frame_samples
        max_record_frames = int(self._max_record_seconds * sample_rate / frame_samples)

        deadline = time.monotonic() + remaining_sec

        onset_count = 0
        silence_count = 0
        recording: list = []
        recording_started = False
        onset_buffer: deque = deque(maxlen=_DEFAULT_VAD_HOLD_FRAMES)

        while time.monotonic() < deadline:
            pcm, overflowed = stream.read(frame_samples)
            if overflowed:
                logger.debug("オーディオバッファオーバーフロー")

            mono = pcm[:, 0]
            speech = is_speech(mono)

            if not recording_started:
                onset_buffer.append(mono.copy())

                if speech:
                    onset_count += 1
                else:
                    onset_count = 0

                if onset_count >= _DEFAULT_VAD_HOLD_FRAMES:
                    recording_started = True
                    silence_count = 0
                    recording = list(onset_buffer)
                    logger.info("発話開始検知 (continuous)")
            else:
                recording.append(mono.copy())

                if not speech:
                    silence_count += 1
                else:
                    silence_count = 0

                if (
                    silence_count >= self._silence_frames
                    or len(recording) >= max_record_frames
                ):
                    logger.info(
                        "発話終了検知 (continuous, frames=%d)",
                        len(recording),
                    )
                    break

        if not recording:
            return None

        # WAV 書き出し → STT → 削除
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

            stt_result = _stt.transcribe_audio_file(
                tmp_path,
                provider=self._stt_provider,
                lang=self._stt_lang,
                prompt=self._stt_prompt or None,
            )
            transcript = stt_result.text.strip()
            duration_ms = stt_result.duration_ms
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not transcript:
            logger.debug("空の転写結果。スキップします。")
            return None

        import time as _time
        return TranscriptSegment(
            text=transcript,
            timestamp=_time.monotonic(),
            duration_ms=duration_ms,
        )

    def listen_once(self, timeout_seconds: float = 30.0) -> "WakeWordResult | None":
        """
        常時文字起こしで発話を蓄積し、キャラクター名が検出されたら
        文脈付きで WakeWordResult を返す。

        内部ループで複数セグメントを処理する。名前が検出されない
        セグメントはバッファに蓄積され、次のセグメントの文脈となる。
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

        deadline = time.monotonic() + timeout_seconds

        logger.info(
            "常時文字起こし開始 (timeout=%.0fs threshold=%.4f buffer=%d segments)",
            timeout_seconds,
            self._vad_threshold,
            len(self._buffer),
        )

        with sd.InputStream(
            samplerate=_DEFAULT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self._vad_frame_samples,
            device=self._device,
        ) as stream:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                segment = self._record_one_segment(
                    stream,
                    remaining,
                    np_module=np,
                    sf_module=sf,
                )
                if segment is None:
                    continue

                self._buffer.add(segment)
                logger.info(
                    "バッファ蓄積: %r (segments=%d chars=%d)",
                    segment.text,
                    len(self._buffer),
                    self._buffer.total_chars,
                )

                # バッファ全文でキャラクター判定
                decision = _router.route(self._buffer.full_text())

                if decision.reason != "default":
                    context = self._buffer.extract_context()
                    char_slug = decision.speaker

                    # LLM 意図ゲート (Phase 2) — L2_USE_INTENT_GATE=true のときのみ
                    # 「言及」と判定された場合はバッファを保持して次のセグメントを待つ
                    # (跨ぎ発話への対応: 「ミミ様のことなんだけど…聞いていい？」)
                    if _router.is_intent_gate_enabled():
                        intent = _router.check_intent(context, char_slug)
                        if intent == "mention":
                            logger.info(
                                "意図ゲート: 言及と判定してスキップ: char=%s (バッファ保持)",
                                char_slug,
                            )
                            continue  # バッファは保持したまま次のセグメント待ち
                        # "callout" or "unknown" → fall through (fail-open)
                        logger.info("意図ゲート: %s (char=%s) → 通過", intent, char_slug)

                    self._buffer.clear()

                    from .characters import get_all_characters as _get_chars
                    chars = {c.slug: c for c in _get_chars()}
                    char = chars.get(char_slug)
                    keyword = (char.wake_word if char and char.wake_word else None) or char_slug

                    logger.info(
                        "文脈付きルーティング: %s (context=%d chars)",
                        char_slug,
                        len(context),
                    )
                    return WakeWordResult(
                        keyword=keyword,
                        character_slug=char_slug,
                        keyword_index=0,
                        transcript=context,
                    )

                logger.info("名前ゲート: バッファ蓄積のみ: %r", segment.text)

        logger.info("常時文字起こしタイムアウト")
        return None

    def cleanup(self) -> None:
        """バッファをクリアする。"""
        self._buffer.clear()
