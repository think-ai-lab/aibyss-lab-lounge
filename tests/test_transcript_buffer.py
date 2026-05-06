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


# ─── TestTranscriptBufferSnapshot ─────────────────────────────────


class TestTranscriptBufferSnapshot:
    """snapshot() — Block 0 で追加。Phase 0.5 の挙手承認時に最新 buffer を
    BG LLM へ渡す用途を想定。元 buffer への以降の変更がスナップショットに
    反映されないこと（独立性）を検証する。"""

    def test_snapshot_returns_separate_instance(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        snap = buf.snapshot()
        assert snap is not buf

    def test_snapshot_preserves_content(self):
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        buf.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        snap = buf.snapshot()
        assert len(snap) == 2
        assert snap.full_text() == "A\nB"

    def test_snapshot_preserves_settings(self):
        buf = TranscriptBuffer(window_sec=60.0, max_chars=500)
        snap = buf.snapshot()
        assert snap._window_sec == 60.0
        assert snap._max_chars == 500

    def test_snapshot_independent_after_original_add(self):
        """元 buffer に追加しても snapshot に反映されない。"""
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        snap = buf.snapshot()
        buf.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        assert len(snap) == 1
        assert snap.full_text() == "A"
        assert len(buf) == 2

    def test_snapshot_independent_after_original_clear(self):
        """元 buffer を clear しても snapshot に反映されない。"""
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        snap = buf.snapshot()
        buf.clear()
        assert len(snap) == 1
        assert len(buf) == 0

    def test_snapshot_of_empty_buffer(self):
        buf = TranscriptBuffer()
        snap = buf.snapshot()
        assert len(snap) == 0
        assert snap.full_text() == ""

    def test_modifying_snapshot_does_not_affect_original(self):
        """逆方向の独立性: snapshot に追加しても元 buffer は変わらない。"""
        buf = TranscriptBuffer()
        buf.add(TranscriptSegment(text="A", timestamp=100.0, duration_ms=500))
        snap = buf.snapshot()
        snap.add(TranscriptSegment(text="B", timestamp=101.0, duration_ms=500))
        assert len(buf) == 1
        assert buf.full_text() == "A"
        assert len(snap) == 2


# ─── TestTranscriptBufferThreadSafety ─────────────────────────────


class TestTranscriptBufferThreadSafety:
    """マルチスレッドアクセスでの整合性を確認する。

    Block 0 では BackgroundContinuousListener (録音スレッド) と run_loop メイン
    スレッドの両方から呼ばれるため、deque 破壊・deadlock が起きないことを
    検証する。
    """

    def test_concurrent_add_keeps_all_segments(self):
        """並列 add で deque 破壊が起きず、全セグメントが残る。"""
        import threading

        buf = TranscriptBuffer(window_sec=1000.0, max_chars=1_000_000)
        n_threads = 10
        n_per_thread = 100

        def writer(thread_idx: int) -> None:
            for i in range(n_per_thread):
                buf.add(TranscriptSegment(
                    text=f"t{thread_idx}_{i}",
                    timestamp=100.0 + thread_idx * 0.01 + i * 0.001,
                    duration_ms=500,
                ))

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        for t in threads:
            assert not t.is_alive(), "writer スレッドがタイムアウト"
        assert len(buf) == n_threads * n_per_thread

    def test_concurrent_add_and_clear_no_deadlock(self):
        """add と clear の並列実行で deadlock / 例外が起きない。"""
        import threading
        import time as _time

        buf = TranscriptBuffer(window_sec=1000.0, max_chars=1_000_000)
        stop_event = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                i = 0
                while not stop_event.is_set():
                    buf.add(TranscriptSegment(
                        text=f"x{i}",
                        timestamp=100.0 + i * 0.001,
                        duration_ms=500,
                    ))
                    i += 1
            except Exception as exc:
                errors.append(exc)

        def clearer() -> None:
            try:
                while not stop_event.is_set():
                    buf.clear()
            except Exception as exc:
                errors.append(exc)

        w = threading.Thread(target=writer)
        c = threading.Thread(target=clearer)
        w.start()
        c.start()

        _time.sleep(0.2)
        stop_event.set()
        w.join(timeout=2.0)
        c.join(timeout=2.0)

        assert not errors, f"スレッドで例外: {errors}"
        assert not w.is_alive() and not c.is_alive()

    def test_concurrent_add_and_extract_context(self):
        """add と extract_context の並列実行で deadlock / 例外が起きない。"""
        import threading
        import time as _time

        buf = TranscriptBuffer(window_sec=1000.0, max_chars=1_000_000)
        stop_event = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                i = 0
                while not stop_event.is_set():
                    buf.add(TranscriptSegment(
                        text=f"x{i}",
                        timestamp=100.0 + i * 0.001,
                        duration_ms=500,
                    ))
                    i += 1
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                while not stop_event.is_set():
                    buf.extract_context()
            except Exception as exc:
                errors.append(exc)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()

        _time.sleep(0.2)
        stop_event.set()
        w.join(timeout=2.0)
        r.join(timeout=2.0)

        assert not errors, f"スレッドで例外: {errors}"
        assert not w.is_alive() and not r.is_alive()

    def test_concurrent_add_and_snapshot(self):
        """add と snapshot の並列実行で deadlock / 例外が起きず、snapshot が一貫した状態を返す。"""
        import threading
        import time as _time

        buf = TranscriptBuffer(window_sec=1000.0, max_chars=1_000_000)
        stop_event = threading.Event()
        errors: list[Exception] = []
        snapshots: list[TranscriptBuffer] = []

        def writer() -> None:
            try:
                i = 0
                while not stop_event.is_set():
                    buf.add(TranscriptSegment(
                        text=f"x{i}",
                        timestamp=100.0 + i * 0.001,
                        duration_ms=500,
                    ))
                    i += 1
            except Exception as exc:
                errors.append(exc)

        def snapshotter() -> None:
            try:
                while not stop_event.is_set():
                    snap = buf.snapshot()
                    snapshots.append(snap)
            except Exception as exc:
                errors.append(exc)

        w = threading.Thread(target=writer)
        s = threading.Thread(target=snapshotter)
        w.start()
        s.start()

        _time.sleep(0.2)
        stop_event.set()
        w.join(timeout=2.0)
        s.join(timeout=2.0)

        assert not errors, f"スレッドで例外: {errors}"
        # スナップショットが取れていること（いくつ取れたかは環境依存）
        assert len(snapshots) > 0
