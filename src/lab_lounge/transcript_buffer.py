"""
transcript_buffer.py — 転写テキストの循環バッファ

責務:
  - 発話セグメントの転写テキストを時間付きで蓄積する
  - 直近 N 秒 / 最大 M 文字の範囲でセグメントを保持する
  - キャラクター名検出時に文脈テキストを抽出する

ContinuousListener が使用する純粋なデータ構造。
音声・STT・ルーター等の外部依存は一切持たない。
"""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """1 回の発話区間の転写結果。"""

    text: str
    timestamp: float   # time.monotonic() at transcription completion
    duration_ms: int   # audio duration from STTResult.duration_ms


class TranscriptBuffer:
    """
    直近 N 秒の転写セグメントを保持する循環バッファ。

    時間ウィンドウ (`window_sec`) と文字数上限 (`max_chars`) の
    両方で古いセグメントを自動的に除去する。
    """

    def __init__(
        self,
        *,
        window_sec: float = 30.0,
        max_chars: int = 2000,
    ) -> None:
        self._segments: deque[TranscriptSegment] = deque()
        self._window_sec = window_sec
        self._max_chars = max_chars

    def add(self, segment: TranscriptSegment) -> None:
        """セグメントをバッファ末尾に追加し、古いセグメントを除去する。"""
        self._segments.append(segment)
        self.evict_old(now=segment.timestamp)

    def evict_old(self, *, now: float | None = None) -> int:
        """
        時間超過・文字数超過のセグメントを先頭から除去する。

        Returns:
            除去されたセグメント数。
        """
        if now is None:
            now = time.monotonic()
        removed = 0
        cutoff = now - self._window_sec

        # 時間ベース除去
        while self._segments and self._segments[0].timestamp < cutoff:
            self._segments.popleft()
            removed += 1

        # 文字数ベース除去
        while self._segments and self.total_chars > self._max_chars:
            self._segments.popleft()
            removed += 1

        return removed

    def full_text(self) -> str:
        """全セグメントのテキストを改行で結合して返す。"""
        return "\n".join(seg.text for seg in self._segments)

    def extract_context(self) -> str:
        """
        パイプラインに渡す文脈テキストを抽出する。

        max_chars 以内に収まるよう先頭を切り詰める。
        切り詰め時はセグメント境界（改行）で揃える。
        """
        text = self.full_text()
        if len(text) <= self._max_chars:
            return text

        # 末尾（最新の文脈）を優先して保持
        truncated = text[-self._max_chars :]
        nl = truncated.find("\n")
        if 0 < nl < len(truncated) // 2:
            truncated = truncated[nl + 1 :]
        return truncated

    def clear(self) -> None:
        """バッファを空にする。"""
        self._segments.clear()

    def __len__(self) -> int:
        """バッファ内のセグメント数を返す。"""
        return len(self._segments)

    @property
    def total_chars(self) -> int:
        """バッファ内の全テキストの合計文字数を返す。"""
        return sum(len(seg.text) for seg in self._segments)
