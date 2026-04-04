"""
test_transcript_buffer.py — TranscriptBuffer のテスト

純粋なデータ構造のテスト。音声・STT モックは不要。
"""

import pytest

from lab_lounge.transcript_buffer import TranscriptBuffer, TranscriptSegment


# ─── TestTranscriptSegment ────────────────────────────────────────

class TestTranscriptSegment:
    def test_dataclass_fields(self):
        seg = TranscriptSegment(text="テスト", timestamp=100.0, duration_ms=1500)
        assert seg.text == "テスト"
        assert seg.timestamp == 100.0
        assert seg.duration_ms == 1500

    def test_text_and_timestamp(self):
        seg = TranscriptSegment(text="こんにちは", timestamp=200.5, duration_ms=2000)
        assert seg.text == "こんにちは"
        assert seg.timestamp == pytest.approx(200.5)


# ─── TestTranscriptBufferInit ─────────────────────────────────────

class TestTranscriptBufferInit:
    def test_default_init(self):
        buf = TranscriptBuffer()
        assert buf._window_sec == 30.0
        assert buf._max_chars == 2000
        assert len(buf) == 0

    def test_custom_window_and_max_chars(self):
        buf = TranscriptBuffer(window_sec=60.0, max_chars=500)
        assert buf._window_sec == 60.0
        assert buf._max_chars == 500

    def test_empty_buffer_len_zero(self):
        buf = TranscriptBuffer()
        assert len(buf) == 0

    def test_empty_full_text_is_empty_string(self):
        buf = TranscriptBuffer()
        assert buf.full_text() == ""


# ─── TestTranscriptBufferAdd ──────────────────────────────────────

class TestTranscriptBufferAdd:
    def test_add_single_segment(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="テスト", timestamp=100.0, duration_ms=1000))
        assert len(buf) == 1

    def test_add_multiple_segments_preserves_order(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="最初", timestamp=100.0, duration_ms=1000))
        buf.add(TranscriptSegment(text="次", timestamp=101.0, duration_ms=1000))
        buf.add(TranscriptSegment(text="最後", timestamp=102.0, duration_ms=1000))
        assert len(buf) == 3
        assert buf.full_text() == "最初\n次\n最後"

    def test_full_text_joins_with_newline(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        assert buf.full_text() == "A\nB"

    def test_total_chars_correct(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="ABC", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="DE", timestamp=101.0, duration_ms=500))
        assert buf.total_chars == 5


# ─── TestTranscriptBufferEviction ─────────────────────────────────

class TestTranscriptBufferEviction:
    def test_evict_old_by_time(self):
        buf = TranscriptBuffer(window_sec=10.0)
        buf.add(TranscriptSegment(text="古い", timestamp=100.0, duration_ms=1000))
        buf.add(TranscriptSegment(text="新しい", timestamp=105.0, duration_ms=1000))

        removed = buf.evict_old(now=115.0)
        assert removed == 1
        assert len(buf) == 1
        assert buf.full_text() == "新しい"

    def test_evict_old_by_chars(self):
        buf = TranscriptBuffer(max_chars=5)
        buf.add(TranscriptSegment(text="ABC", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="DEF", timestamp=101.0, duration_ms=500))
        # total_chars = 6 > max_chars=5 → 先頭を除去
        assert len(buf) == 1
        assert buf.full_text() == "DEF"

    def test_evict_old_keeps_recent(self):
        buf = TranscriptBuffer(window_sec=10.0)
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="B", timestamp=105.0, duration_ms=500))

        removed = buf.evict_old(now=108.0)
        assert removed == 0
        assert len(buf) == 2

    def test_evict_returns_count(self):
        buf = TranscriptBuffer(window_sec=5.0)
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        buf.add(TranscriptSegment(text="C", timestamp=102.0, duration_ms=500))

        removed = buf.evict_old(now=110.0)
        assert removed == 3
        assert len(buf) == 0


# ─── TestTranscriptBufferExtractContext ───────────────────────────

class TestTranscriptBufferExtractContext:
    def test_extract_context_returns_all_text(self):
        buf = TranscriptBuffer(max_chars=100)
        buf.add(TranscriptSegment(text="文脈A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="文脈B", timestamp=101.0, duration_ms=500))
        assert buf.extract_context() == "文脈A\n文脈B"

    def test_extract_context_truncates_long_text(self):
        buf = TranscriptBuffer(max_chars=10)
        buf.add(TranscriptSegment(text="AAAAAAAAAA", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="BBBBBBBBBB", timestamp=101.0, duration_ms=500))
        context = buf.extract_context()
        assert len(context) <= 21  # max_chars + newline boundary

    def test_extract_context_single_segment(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="単一セグメント", timestamp=100.0, duration_ms=500))
        assert buf.extract_context() == "単一セグメント"

    def test_extract_context_empty_buffer(self):
        buf = TranscriptBuffer()
        assert buf.extract_context() == ""


# ─── TestTranscriptBufferClear ────────────────────────────────────

class TestTranscriptBufferClear:
    def test_clear_empties_buffer(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        buf.clear()
        assert len(buf) == 0

    def test_clear_resets_total_chars(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="ABC", timestamp=100.0, duration_ms=500))
        buf.clear()
        assert buf.total_chars == 0
        assert buf.full_text() == ""
