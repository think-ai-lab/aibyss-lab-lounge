"""
log_setup.py — ログ設定の集約

責務:
  - コンソール + ファイル出力の両方を設定する
  - セッションごとに新しいログファイルを作成（タイムスタンプ付き）
  - 環境変数でファイル出力・ログディレクトリ・ログレベルを制御
  - 第三者ライブラリの機密情報漏洩を抑制（配信中のコンソール表示対策）

【環境変数】
  L2_LOG_TO_FILE    — ファイル出力を有効化 (デフォルト: "true")
  L2_LOG_DIR        — ログファイルの配置ディレクトリ (デフォルト: "./logs/runs")
  L2_LOG_LEVEL      — ログレベル (デフォルト: "INFO")
  L2_LLM_DEBUG_HTTP — LLM SDK の HTTP 詳細 (retry 判定の status code 等) を DEBUG 出力
                      (デフォルト: "false"。実際に出すには L2_LOG_LEVEL=DEBUG も必要)

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
#
# 注意: openai/anthropic の `_base_client` は「リトライ行 (= retry 原因の可視化に有用)」
# と「リクエスト詳細 (= URL/ヘッダ等、キー漏洩リスク)」の両方を出すため、ここでは
# 一律抑制せず _LLM_RETRY_LOGGERS で個別制御する (下記参照)。httpx/httpcore は URL 全文を
# 出すため WARNING のまま据え置く (キー漏洩防止)。
_SENSITIVE_LIBRARY_LOGGERS = (
    "obsws_python",                # host / port / password を INFO で出力
    "obsws_python.baseclient",     # 同上
    "obsws_python.reqs",           # RPC バージョン情報
    "httpx",                        # HTTP リクエスト URL を INFO で出力
    "httpcore",                     # httpx の下位レイヤ
    "google_genai.models",          # AFC 設定など
    "google.genai",                 # 同上
)

# LLM SDK のリトライログを出すロガー。timeout / overloaded (429/5xx) で自動 retry が
# 起きた際、これらは INFO で `"Retrying request to <path> in N seconds"` を出す
# (= 配信フリーズ調査時に「何回・なぜ待たされたか」を追える)。この retry 行は
# URL の**パスのみ** (例 /chat/completions) で API キーを含まないため INFO 解放は安全。
# 一方、status code 等の詳細は DEBUG で出る (= L2_LLM_DEBUG_HTTP=true 時のみ解放)。
_LLM_RETRY_LOGGERS = (
    "openai._base_client",
    "anthropic._base_client",
)


def _is_llm_debug_http() -> bool:
    """L2_LLM_DEBUG_HTTP が有効か。デフォルト false。

    true のとき LLM SDK の HTTP 詳細 (retry 判定の status code 等) を DEBUG で出す。
    調査時のみ有効化する想定 (常用するとログが冗長になり、稀に URL 詳細も出るため)。
    """
    return os.environ.get("L2_LLM_DEBUG_HTTP", "false").lower() in ("true", "1", "yes")


def _suppress_sensitive_loggers() -> None:
    """機密情報を出力する第三者ライブラリの logger を WARNING に設定する。

    INFO 以下のメッセージ (host/port/password/URL を含む) が出力されなくなる。
    エラーや警告は引き続き表示される。
    """
    for name in _SENSITIVE_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _configure_llm_retry_loggers() -> None:
    """LLM SDK のリトライログを安全な範囲で可視化する。

    - 通常: INFO に設定 → retry 行 (URL パスのみ、キー非含有) が出る。
    - L2_LLM_DEBUG_HTTP=true: DEBUG に設定 → status code 等の詳細も出る (調査用)。

    timeout / retry に至った原因 (overloaded / connection error 等) をログから追える
    ようにするための設定。WHY は _LLM_RETRY_LOGGERS のコメント参照。
    """
    level = logging.DEBUG if _is_llm_debug_http() else logging.INFO
    for name in _LLM_RETRY_LOGGERS:
        logging.getLogger(name).setLevel(level)


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
    # LLM SDK のリトライログを安全な範囲で可視化する (retry 原因の追跡用)。
    # 注: DEBUG 詳細 (L2_LLM_DEBUG_HTTP=true) を実際に出すにはハンドラ側も DEBUG が
    # 必要なため L2_LOG_LEVEL=DEBUG と併用する。INFO の retry 行は既定レベルで出る。
    _configure_llm_retry_loggers()

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
