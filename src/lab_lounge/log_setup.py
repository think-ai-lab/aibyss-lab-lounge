"""
log_setup.py — ログ設定の集約

責務:
  - コンソール + ファイル出力の両方を設定する
  - セッションごとに新しいログファイルを作成（タイムスタンプ付き）
  - 環境変数でファイル出力・ログディレクトリ・ログレベルを制御

【環境変数】
  L2_LOG_TO_FILE  — ファイル出力を有効化 (デフォルト: "true")
  L2_LOG_DIR      — ログファイルの配置ディレクトリ (デフォルト: "./logs/runs")
  L2_LOG_LEVEL    — ログレベル (デフォルト: "INFO")

【出力ファイル】
  {L2_LOG_DIR}/run_loop_YYYYMMDD_HHMMSS.log
  セッション（プロセス）ごとに新規作成。ローテーションは行わない。
"""

import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def _is_file_logging_enabled() -> bool:
    """L2_LOG_TO_FILE が有効かどうか。デフォルトは true。"""
    return os.environ.get("L2_LOG_TO_FILE", "true").lower() in ("true", "1", "yes")


def _get_log_dir() -> str:
    """ログ出力ディレクトリを返す。デフォルトは ./logs/runs。"""
    return os.environ.get("L2_LOG_DIR", "./logs/runs")


def _get_log_level() -> int:
    """環境変数からログレベルを取得する。デフォルトは INFO。"""
    name = os.environ.get("L2_LOG_LEVEL", "INFO").upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(
    *,
    session_name: str = "run_loop",
    log_dir: str | None = None,
) -> Path | None:
    """
    コンソール + ファイルの両方にログを出力する設定を行う。

    Args:
        session_name: ログファイル名のプレフィックス（例: "run_loop"）
        log_dir:      ログ出力ディレクトリ。None なら環境変数/デフォルトを使う

    Returns:
        作成されたログファイルのパス。ファイル出力が無効なら None。
    """
    level = _get_log_level()

    # ルートロガーをリセット（basicConfig との競合を避ける）
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ファイルハンドラー（オプション）
    if not _is_file_logging_enabled():
        return None

    dir_path = Path(log_dir or _get_log_dir())
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        root.warning("ログディレクトリ作成失敗: %s (%s)。ファイル出力を無効化。", dir_path, exc)
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dir_path / f"{session_name}_{timestamp}.log"

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as exc:
        root.warning("ログファイル作成失敗: %s (%s)。ファイル出力を無効化。", log_path, exc)
        return None

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("ログファイル出力: %s", log_path)
    return log_path
