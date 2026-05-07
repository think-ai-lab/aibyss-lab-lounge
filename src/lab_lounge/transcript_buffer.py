"""
transcript_buffer.py — 転写テキストの循環バッファ（スレッドセーフ）

責務:
  - 発話セグメントの転写テキストを時間付きで蓄積する
  - 直近 N 秒 / 最大 M 文字の範囲でセグメントを保持する
  - キャラクター名検出時に文脈テキストを抽出する
  - 任意の時点のスナップショットを取得する

ContinuousListener / BackgroundContinuousListener が使用する純粋なデータ構造。
音声・STT・ルーター等の外部依存は一切持たない。

【スレッドセーフティ】
  - 全 public メソッドは内部 ``threading.Lock`` で保護される。
  - BackgroundContinuousListener (録音スレッド) からの ``add()`` / ``evict_old()`` と、
    run_loop メインスレッドからの ``extract_context()`` / ``clear()`` / ``snapshot()`` の
    同時アクセスに対応する。
  - 取得する Lock は self._lock のみ（他クラスの Lock と組み合わせない）ため、
    取得順序起因の deadlock 懸念はない。
  - メソッド間で再入が必要なケース（add → evict_old）は ``_*_unlocked()`` 内部
    メソッド経由にして、Lock の再取得を回避している。
"""

import copy
import threading
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
    直近 N 秒の転写セグメントを保持する循環バッファ（スレッドセーフ）。

    時間ウィンドウ (``window_sec``) と文字数上限 (``max_chars``) の
    両方で古いセグメントを自動的に除去する。

    全メソッドは内部 Lock で保護されており、複数スレッドから同時に呼び出し可能。
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
        # 全 public メソッドを保護する Lock。Block 0 (録音常時化) で
        # BackgroundContinuousListener (別スレッド) と run_loop メインスレッドの
        # 両方から呼ばれるため、shared state へのアクセスを直列化する。
        self._lock = threading.Lock()

    def add(self, segment: TranscriptSegment) -> None:
        """セグメントをバッファ末尾に追加し、古いセグメントを除去する。"""
        with self._lock:
            self._segments.append(segment)
            # Lock 取得済みのため、内部実装 (_evict_old_unlocked) を直接呼ぶ。
            self._evict_old_unlocked(now=segment.timestamp)

    def evict_old(self, *, now: float | None = None) -> int:
        """
        時間超過・文字数超過のセグメントを先頭から除去する。

        Returns:
            除去されたセグメント数。
        """
        with self._lock:
            return self._evict_old_unlocked(now=now)

    def _evict_old_unlocked(self, *, now: float | None = None) -> int:
        """
        ``evict_old`` の内部実装。Lock 取得済み前提。

        ``add()`` 等の Lock 取得済みメソッドから呼び出されるため、再入を避ける。
        外部から直接呼ばないこと（直接呼ぶ場合は呼び出し側で Lock 管理が必要）。
        """
        if now is None:
            now = time.monotonic()
        removed = 0
        cutoff = now - self._window_sec

        # 時間ベース除去
        while self._segments and self._segments[0].timestamp < cutoff:
            self._segments.popleft()
            removed += 1

        # 文字数ベース除去（_total_chars_unlocked で再計算）
        while self._segments and self._total_chars_unlocked() > self._max_chars:
            self._segments.popleft()
            removed += 1

        return removed

    def full_text(self) -> str:
        """全セグメントのテキストを改行で結合して返す。"""
        with self._lock:
            return "\n".join(seg.text for seg in self._segments)

    def extract_context(self) -> str:
        """
        パイプラインに渡す文脈テキストを抽出する。

        max_chars 以内に収まるよう先頭を切り詰める。
        切り詰め時はセグメント境界（改行）で揃える。
        """
        with self._lock:
            text = "\n".join(seg.text for seg in self._segments)
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
        with self._lock:
            self._segments.clear()

    def snapshot(self) -> "TranscriptBuffer":
        """
        現在の状態のディープコピーを返す。

        Phase 0.5 の挙手システムで「許可時点で最新 buffer を渡す」(Notion T7) 要件で
        使うことを想定。スナップショット後の元 buffer への変更は新インスタンスに
        反映されない（独立したコピー）。

        WHY ``copy.deepcopy``:
          ``TranscriptSegment`` は str / float / int の immutable な値のみで構成
          されるため、現状は deque を shallow copy するだけでも実用上は安全。
          ただし、将来 ``TranscriptSegment`` に mutable フィールド（list 等）が
          追加されたときに silently 共有される事故を防ぐため、保守的に deepcopy する。
          Snapshot は配信中の頻繁操作ではない（挙手承認時のみ）ため、コストは
          無視できる。

        Returns:
            新しい ``TranscriptBuffer`` インスタンス
            （同じ window_sec / max_chars / 現在のセグメントのディープコピー）。
        """
        with self._lock:
            new_buf = TranscriptBuffer(
                window_sec=self._window_sec,
                max_chars=self._max_chars,
            )
            # deque ごとディープコピー。元 buffer への変更（add/clear/evict）は
            # 新 buffer に影響しない。
            new_buf._segments = copy.deepcopy(self._segments)
            return new_buf

    def __len__(self) -> int:
        """バッファ内のセグメント数を返す。"""
        with self._lock:
            return len(self._segments)

    @property
    def total_chars(self) -> int:
        """バッファ内の全テキストの合計文字数を返す。"""
        with self._lock:
            return self._total_chars_unlocked()

    def _total_chars_unlocked(self) -> int:
        """
        ``total_chars`` の内部実装。Lock 取得済み前提。

        ``_evict_old_unlocked`` 等の Lock 取得済みメソッドから呼び出されるため、
        Lock の再取得を避ける。
        """
        return sum(len(seg.text) for seg in self._segments)
