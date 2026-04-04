"""
audio_io.py — 録音・再生アダプタ

責務:
  - マイク録音と音声再生を sounddevice + soundfile に閉じ込める
  - 実デバイス依存部分を薄く保ち、run_once.py からテストしやすくする
  - 無音チェック (RMS) を録音時に行い SilenceError を送出する
  - play_audio_file() は失敗時にログを出して False を返す（例外を外に出さない）

【前提パッケージ (mic mode)】
  uv sync --extra mic
    sounddevice>=0.4  (録音 + 再生 + numpy 依存)
    soundfile>=0.12   (WAV 読み書き)

【再生フォーマット】
  soundfile が対応するフォーマット (WAV, AIFF, FLAC 等) のみ再生可能。
  MP3 は soundfile では非対応。
  - VOICEVOX (WAV) 利用時: 問題なし ✓
  - edge_tts (MP3) 利用時: 再生失敗 → 警告ログ + ファイルパス表示

【無音しきい値】
  L2_SILENCE_THRESHOLD 環境変数で上書き可 (デフォルト 0.005 / float32 RMS)。
  適切な値は録音デバイスや環境によって異なる。
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# 無音判定しきい値 (float32 RMS)
_SILENCE_RMS_THRESHOLD = 0.005

# 録音デフォルト設定
_DEFAULT_SAMPLE_RATE = 16000  # Hz — Whisper が推奨するサンプリングレート
_DEFAULT_CHANNELS = 1         # モノラル


# ─── 例外型 ──────────────────────────────────────────────────────

class RecordError(RuntimeError):
    """録音デバイスエラーなど、録音に失敗した場合に送出する。"""


class SilenceError(RecordError):
    """録音結果が無音と判定された場合に送出する。"""


# ─── 録音 ────────────────────────────────────────────────────────

def record_to_file(
    seconds: float,
    *,
    output_path: str | None = None,
    tmp_dir: str | None = None,
    device: int | str | None = None,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    channels: int = _DEFAULT_CHANNELS,
) -> str:
    """
    マイクから指定秒数録音して WAV ファイルに保存する。

    無音チェックはファイル書き込み前に行う。
    無音と判定した場合は SilenceError を送出し、ファイルは作成しない。

    Args:
        seconds:     録音秒数
        output_path: 保存先ファイルパス (省略時は tmpdir に UUID 名で自動生成)
        tmp_dir:     output_path 未指定時の保存先ディレクトリ (省略時は OS tmpdir)
        device:      オーディオデバイス (インデックスまたは名前。省略時はデフォルト)
        sample_rate: サンプリング周波数 (デフォルト 16000 Hz)
        channels:    チャンネル数 (デフォルト 1 = モノラル)

    Returns:
        保存した WAV ファイルのパス文字列

    Raises:
        ImportError:  sounddevice / soundfile が未インストール
        SilenceError: 無音と判定した場合 (ファイルは作成しない)
        RecordError:  録音デバイスエラーなど
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "sounddevice と soundfile が必要です。"
            " uv sync --extra mic でインストールしてください。"
        ) from exc

    logger.info(
        "録音開始: %.1f 秒  device=%s  sample_rate=%d",
        seconds, device, sample_rate,
    )
    try:
        audio_data = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:
        raise RecordError(f"録音デバイスエラー: {exc}") from exc

    logger.info("録音完了")

    # 無音チェック — ファイル書き込み前に行うことでゴミファイルを残さない
    threshold = float(os.environ.get("L2_SILENCE_THRESHOLD", str(_SILENCE_RMS_THRESHOLD)))
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    logger.debug("録音 RMS=%.4f  threshold=%.4f", rms, threshold)
    if rms < threshold:
        raise SilenceError(
            f"無音を検出しました (RMS={rms:.4f} < threshold={threshold})"
        )

    # ファイル保存 (無音でなかったときのみ)
    if output_path is None:
        if tmp_dir:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="l2_rec_", dir=tmp_dir
        )
        os.close(fd)

    sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
    logger.debug("録音ファイル保存: %s", output_path)
    return output_path


# ─── 再生 ────────────────────────────────────────────────────────

def play_audio_file(path_or_url: str) -> bool:
    """
    音声ファイルを再生する。

    sounddevice + soundfile で再生を試みる。
    失敗時 (未インストール・非対応フォーマット等) は警告ログを出して False を返す。
    例外は呼び出し元に伝播しない。

    Args:
        path_or_url: ローカルファイルパスまたは file:// URI

    Returns:
        True:  再生成功
        False: 再生失敗 (例外は外に出さない)
    """
    if not path_or_url:
        return False

    path = _uri_to_path(path_or_url)

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        logger.warning(
            "sounddevice/soundfile が未インストールのため再生をスキップします。"
            " TTS ファイル: %s  (uv sync --extra mic でインストール可能)",
            path,
        )
        return False

    if not Path(path).exists():
        logger.debug("再生スキップ: ファイルが存在しません (ダミー TTS または未生成): %s", path_or_url)
        return False

    try:
        data, samplerate = sf.read(path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
        return True
    except Exception as exc:
        logger.warning("再生失敗: %s  (%s)", path, exc)
        return False


def play_audio_interruptible(
    path_or_url: str,
    stop_event: "threading.Event",
    poll_interval: float = 0.05,
) -> bool:
    """
    音声を再生する。stop_event が set されたら即座に停止する。

    sounddevice.play() + ポーリングループで stop_event を監視。
    sd.stop() で再生を中断する。

    Args:
        path_or_url:    ローカルファイルパスまたは file:// URI
        stop_event:     このイベントが set されたら再生を中断
        poll_interval:  ポーリング間隔（秒）

    Returns:
        True:  再生完了（中断なし）
        False: 中断された、またはエラー
    """
    import time

    if not path_or_url:
        return False

    path = _uri_to_path(path_or_url)

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        logger.warning("sounddevice/soundfile が未インストールのため再生をスキップします。")
        return False

    if not Path(path).exists():
        logger.debug("再生スキップ: ファイルが存在しません: %s", path_or_url)
        return False

    try:
        data, samplerate = sf.read(path, dtype="float32")
        sd.play(data, samplerate)

        # ポーリングで stop_event を監視
        while sd.get_stream() and sd.get_stream().active:
            if stop_event.is_set():
                sd.stop()
                logger.debug("フィラー再生中断: %s", path)
                return False
            time.sleep(poll_interval)

        return True
    except Exception as exc:
        logger.warning("再生失敗: %s  (%s)", path, exc)
        return False


# ─── ヘルパー ────────────────────────────────────────────────────

def _uri_to_path(path_or_url: str) -> str:
    """
    file:// URI またはパス文字列をローカルファイルパスに変換する。

    Windows の file:///C:/... 形式にも対応。
    file:// 以外の場合はそのまま返す。
    """
    if not path_or_url.startswith("file://"):
        return path_or_url

    from urllib.parse import unquote, urlparse

    parsed = urlparse(path_or_url)
    path = unquote(parsed.path)

    # Windows: /C:/... → C:/...
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]

    return path
