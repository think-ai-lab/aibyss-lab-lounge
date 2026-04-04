"""
conftest.py — テスト共通フィクスチャ

- AIBYSS_SCHEMA_PATH を sibling workspace から自動解決して設定する
- キャッシュされた validator をリセットしてテスト間の汚染を防ぐ
- L2_ モード環境変数を未設定状態にリセットし .env ファイルの影響を受けないようにする
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_schema_path(monkeypatch):
    """スキーマパスを確実に設定し、テスト間で validator キャッシュをリセットする。"""
    here = Path(__file__).resolve().parent
    candidate = here.parents[2] / "aibyss-workspace" / "specs" / "event-envelope-0.1.schema.json"
    if candidate.is_file():
        monkeypatch.setenv("AIBYSS_SCHEMA_PATH", str(candidate))

    # validator キャッシュをテスト前後にクリア
    from lab_lounge.events import _reset_validator_cache
    _reset_validator_cache()
    yield
    _reset_validator_cache()


@pytest.fixture(autouse=True)
def reset_l2_mode_vars(monkeypatch):
    """
    L2_ モード環境変数を未設定状態にリセットする。

    .env ファイルに real mode が設定されていても、テストはコードデフォルト
    (dummy mode) で動作させる。real mode が必要なテストは各 fixture で明示的に
    setenv する。
    """
    for var in ("L2_USE_REAL_LLM", "L2_USE_REAL_TTS", "L2_USE_REAL_STT",
                "L2_DEFAULT_SPEAKER", "L2_WAKE_BACKEND",
                "L2_SHERPA_MODEL_DIR", "L2_SHERPA_PROVIDER",
                "L2_CONTEXT_WINDOW_SEC", "L2_CONTEXT_MAX_CHARS",
                "L2_FILLER_ENABLED", "L2_LLM_FILLER_MODEL"):
        monkeypatch.delenv(var, raising=False)

