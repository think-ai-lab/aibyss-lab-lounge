"""
test_observability.py — observability ヘルパーのテスト

LangSmith 有効/無効判定と run_metadata 組み立てを検証する。
実際の LangSmith API は呼ばない。langsmith パッケージも不要。
"""

import pytest

from lab_lounge.observability import build_run_metadata, is_langsmith_enabled


# ─── is_langsmith_enabled テスト ──────────────────────────────────

class TestIsLangsmithEnabled:
    def test_default_is_false(self, monkeypatch):
        """LANGSMITH_TRACING 未設定のときは False"""
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        assert is_langsmith_enabled() is False

    def test_true_when_set_to_true(self, monkeypatch):
        """LANGSMITH_TRACING=true のとき True"""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        assert is_langsmith_enabled() is True

    def test_case_insensitive_upper(self, monkeypatch):
        """大文字 TRUE でも True"""
        monkeypatch.setenv("LANGSMITH_TRACING", "TRUE")
        assert is_langsmith_enabled() is True

    def test_numeric_one(self, monkeypatch):
        """LANGSMITH_TRACING=1 でも True"""
        monkeypatch.setenv("LANGSMITH_TRACING", "1")
        assert is_langsmith_enabled() is True

    def test_yes(self, monkeypatch):
        """LANGSMITH_TRACING=yes でも True"""
        monkeypatch.setenv("LANGSMITH_TRACING", "yes")
        assert is_langsmith_enabled() is True

    def test_false_when_explicitly_false(self, monkeypatch):
        """LANGSMITH_TRACING=false のとき False"""
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        assert is_langsmith_enabled() is False

    def test_empty_string_is_false(self, monkeypatch):
        """空文字のとき False"""
        monkeypatch.setenv("LANGSMITH_TRACING", "")
        assert is_langsmith_enabled() is False


# ─── build_run_metadata テスト ───────────────────────────────────

class TestBuildRunMetadata:
    def test_returns_dict(self):
        meta = build_run_metadata(stream_id="s1", session_id="ss1", trace_id="t1")
        assert isinstance(meta, dict)

    def test_stream_id_value(self):
        meta = build_run_metadata(stream_id="s1", session_id="ss1", trace_id="t1")
        assert meta["aibyss.stream_id"] == "s1"

    def test_session_id_value(self):
        meta = build_run_metadata(stream_id="s1", session_id="ss1", trace_id="t1")
        assert meta["aibyss.session_id"] == "ss1"

    def test_trace_id_value(self):
        meta = build_run_metadata(stream_id="s1", session_id="ss1", trace_id="t1")
        assert meta["aibyss.trace_id"] == "t1"

    def test_source_is_lab_lounge(self):
        """aibyss.source は常に "aibyss-lab-lounge" になること"""
        meta = build_run_metadata(stream_id="s", session_id="ss", trace_id="t")
        assert meta["aibyss.source"] == "aibyss-lab-lounge"

    def test_all_required_keys_present(self):
        """必須キーが全て含まれること"""
        meta = build_run_metadata(stream_id="s", session_id="ss", trace_id="t")
        required = {"aibyss.stream_id", "aibyss.session_id", "aibyss.trace_id", "aibyss.source"}
        assert required.issubset(meta.keys())

    def test_distinct_ids_are_preserved(self):
        """異なる ID を渡したとき混在しないこと"""
        meta = build_run_metadata(
            stream_id="stream-abc",
            session_id="session-xyz",
            trace_id="trace-123",
        )
        assert meta["aibyss.stream_id"] == "stream-abc"
        assert meta["aibyss.session_id"] == "session-xyz"
        assert meta["aibyss.trace_id"] == "trace-123"
