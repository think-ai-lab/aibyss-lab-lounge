"""
log_setup.py — ログ設定の集約

責務:
  - コンソール + ファイル出力の両方を設定する
  - セッションごとに新しいログファイルを作成（タイムスタンプ付き）
  - 環境変数でファイル出力・ログディレクトリ・ログレベルを制御
  - 第三者ライブラリの機密情報漏洩を抑制（配信中のコンソール表示対策）

【環境変数】
  L2_LOG_TO_FILE  — ファイル出力を有効化 (デフォルト: "true")
  L2_LOG_DIR      — ログファイルの配置ディレクトリ (デフォルト: "./logs/runs")
  L2_LOG_LEVEL    — ログレベル (デフォルト: "INFO")

【出力ファイル】
  {L2_LOG_DIR}/run_loop_YYYYMMDD_HHMMSS.log
  セッション（プロセス）ごとに新規作成。ローテーションは行わない。

【機密情報抑制】
  以下の第三者ライブラリは INFO レベルでパスワード/host/port/URL を出力するため、
  WARNING 以上のみ表示するように制限する:
    - obsws_python  : OBS WebSocket の host/port/password を平文出力
    - httpx         : HTTP リクエストの URL 全文を出力 (API キーが含まれる可能性)
    - openai/anthropic/google : API リクエストの詳細
"""

import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# 機密情報を出力する第三者ライブラリ — INFO 以下を抑制
# (配信中のコンソール表示で password / host / port / API URL が漏れることを防ぐ)
_SENSITIVE_LIBRARY_LOGGERS = (
    "obsws_python",                # host / port / password を INFO で出力
    "obsws_python.baseclient",     # 同上
    "obsws_python.reqs",           # RPC バージョン情報
    "httpx",                        # HTTP リクエスト URL を INFO で出力
    "httpcore",                     # httpx の下位レイヤ
    "openai._base_client",          # API エンドポイント URL
    "anthropic._base_client",       # 同上
    "google_genai.models",          # AFC 設定など
    "google.genai",                 # 同上
)


def _suppress_sensitive_loggers() -> None:
    """機密情報を出力する第三者ライブラリの logger を WARNING に設定する。

    INFO 以下のメッセージ (host/port/password/URL を含む) が出力されなくなる。
    エラーや警告は引き続き表示される。
    """
    for name in _SENSITIVE_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


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

    # 第三者ライブラリの機密情報出力を抑制（ファイル出力の有無に関わらず）
    _suppress_sensitive_loggers()

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
