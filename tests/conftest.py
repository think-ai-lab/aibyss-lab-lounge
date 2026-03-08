"""
conftest.py — テスト共通フィクスチャ

- AIBYSS_SCHEMA_PATH を sibling workspace から自動解決して設定する
- キャッシュされた validator をリセットしてテスト間の汚染を防ぐ
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
