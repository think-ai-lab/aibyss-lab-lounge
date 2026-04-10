"""
test_log_setup.py — setup_logging() のテスト

ファイル出力・環境変数制御・コンソールハンドラーの検証。
"""

import logging
from pathlib import Path

import pytest

from lab_lounge.log_setup import setup_logging


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """各テストの前後でルートロガーのハンドラーをクリアする。"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


class TestSetupLoggingDefaults:
    def test_creates_log_file_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="test_default")
        assert log_path is not None
        assert log_path.is_file()
        assert log_path.name.startswith("test_default_")
        assert log_path.name.endswith(".log")

    def test_returns_none_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_TO_FILE", "false")
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="test_disabled")
        assert log_path is None
        # ファイルが作られていない
        assert list(tmp_path.iterdir()) == []

    def test_console_handler_always_added(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_TO_FILE", "false")
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_console")
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_file_handler_added_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_file_handler")
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1


class TestSetupLoggingEnvVars:
    def test_log_dir_env_var(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom"
        monkeypatch.setenv("L2_LOG_DIR", str(custom_dir))
        log_path = setup_logging(session_name="test_dir")
        assert log_path is not None
        assert log_path.parent == custom_dir
        assert custom_dir.is_dir()

    def test_log_level_env_var_debug(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("L2_LOG_LEVEL", "DEBUG")
        setup_logging(session_name="test_debug")
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_env_var_warning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("L2_LOG_LEVEL", "WARNING")
        setup_logging(session_name="test_warning")
        assert logging.getLogger().level == logging.WARNING

    def test_log_level_defaults_to_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        monkeypatch.delenv("L2_LOG_LEVEL", raising=False)
        setup_logging(session_name="test_default_level")
        assert logging.getLogger().level == logging.INFO

    def test_log_level_invalid_falls_back_to_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("L2_LOG_LEVEL", "NONSENSE")
        setup_logging(session_name="test_invalid")
        assert logging.getLogger().level == logging.INFO


class TestSetupLoggingFileContent:
    def test_log_messages_written_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="test_content")
        assert log_path is not None

        logger = logging.getLogger("lab_lounge.test")
        logger.info("テストメッセージ")

        # FileHandler の flush を強制
        for h in logging.getLogger().handlers:
            h.flush()

        content = log_path.read_text(encoding="utf-8")
        assert "テストメッセージ" in content
        assert "lab_lounge.test" in content
        assert "INFO" in content

    def test_setup_logging_info_message_in_file(self, tmp_path, monkeypatch):
        """setup_logging 自身が出力する 'ログファイル出力:' メッセージがファイルに含まれる。"""
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="test_self_log")
        assert log_path is not None

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_path.read_text(encoding="utf-8")
        assert "ログファイル出力" in content


class TestSetupLoggingSessionName:
    def test_session_name_in_filename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="custom_session")
        assert log_path is not None
        assert log_path.name.startswith("custom_session_")

    def test_timestamp_format_in_filename(self, tmp_path, monkeypatch):
        """ファイル名にタイムスタンプ (YYYYMMDD_HHMMSS) が含まれる。"""
        import re
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        log_path = setup_logging(session_name="test_ts")
        assert log_path is not None
        # test_ts_20260410_181024.log
        pattern = re.compile(r"test_ts_\d{8}_\d{6}\.log")
        assert pattern.match(log_path.name)


class TestSetupLoggingErrorHandling:
    def test_returns_none_on_invalid_log_dir(self, monkeypatch):
        """書き込み不可能なパスを指定しても例外にならず None を返す。"""
        # Windows で NUL デバイスを使って失敗をシミュレート
        # (pathlib.mkdir は存在しないドライブだと OSError)
        monkeypatch.setenv("L2_LOG_DIR", "Z:\\nonexistent\\invalid\\path")
        log_path = setup_logging(session_name="test_fail")
        # ファイル作成失敗時でも例外を投げず None を返す
        assert log_path is None
        # コンソールハンドラーは残っている
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1


class TestSensitiveLoggerSuppression:
    """配信中の機密情報漏洩を防ぐため、第三者ライブラリ logger が抑制される。"""

    @pytest.fixture(autouse=True)
    def _save_third_party_logger_levels(self):
        """テスト前後で第三者 logger の level を保存・復元する。"""
        from lab_lounge.log_setup import _SENSITIVE_LIBRARY_LOGGERS
        saved = {
            name: logging.getLogger(name).level
            for name in _SENSITIVE_LIBRARY_LOGGERS
        }
        # テスト前は NOTSET (継承) に戻す
        for name in _SENSITIVE_LIBRARY_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
        yield
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)

    def test_obsws_python_logger_set_to_warning(self, tmp_path, monkeypatch):
        """obsws_python の logger が WARNING 以上に抑制される (host/port/password 漏洩防止)。"""
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_obsws")
        assert logging.getLogger("obsws_python").level == logging.WARNING
        assert logging.getLogger("obsws_python.baseclient").level == logging.WARNING
        assert logging.getLogger("obsws_python.reqs").level == logging.WARNING

    def test_httpx_logger_set_to_warning(self, tmp_path, monkeypatch):
        """httpx の logger が WARNING 以上に抑制される (API URL 漏洩防止)。"""
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_httpx")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_llm_provider_loggers_set_to_warning(self, tmp_path, monkeypatch):
        """OpenAI / Anthropic / Google GenAI の logger が WARNING 以上に抑制される。"""
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_llm")
        assert logging.getLogger("openai._base_client").level == logging.WARNING
        assert logging.getLogger("anthropic._base_client").level == logging.WARNING
        assert logging.getLogger("google_genai.models").level == logging.WARNING
        assert logging.getLogger("google.genai").level == logging.WARNING

    def test_suppression_applies_when_file_logging_disabled(self, monkeypatch):
        """L2_LOG_TO_FILE=false のときも第三者 logger 抑制が機能する。"""
        monkeypatch.setenv("L2_LOG_TO_FILE", "false")
        setup_logging(session_name="test_no_file")
        assert logging.getLogger("obsws_python").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_warning_messages_still_pass_through(self, tmp_path, monkeypatch):
        """WARNING / ERROR メッセージは抑制されず流れることを確認。"""
        monkeypatch.setenv("L2_LOG_DIR", str(tmp_path))
        setup_logging(session_name="test_warn")

        logger = logging.getLogger("obsws_python")
        # 直接 logger の判定を確認 (caplog は setup_logging 後の handler 構成と相性が悪い)
        # WARNING は通る、INFO はブロックされる
        assert logger.isEnabledFor(logging.WARNING) is True
        assert logger.isEnabledFor(logging.ERROR) is True
        assert logger.isEnabledFor(logging.INFO) is False
        assert logger.isEnabledFor(logging.DEBUG) is False
