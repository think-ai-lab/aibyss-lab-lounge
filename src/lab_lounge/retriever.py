"""
retriever.py — Retriever インターフェースと LocalRetriever 実装

責務:
  - Retriever 抽象基底クラスを定義する（将来 C2 API への差し替えを容易にする）
  - LocalRetriever: ローカルの index ファイルを使いコサイン類似度検索を行う
  - pipeline.py からのみ呼び出す

【index ファイル構成】
  {kb_path}/chunks.json     — チャンクメタデータ
                              [{"doc_id": ..., "text": ..., "source": ...}, ...]
  {kb_path}/embeddings.npy  — numpy 行列 (N, D) float32

【将来の差し替え】
  C2 の REST API（GET /retrieve）に差し替える場合は Retriever を継承した
  C2Retriever を実装するだけでよい。pipeline.py 側の変更は不要。

【前提パッケージ (LocalRetriever)】
  numpy>=1.26, openai>=1.0 が必要。
  uv add --optional rag numpy openai でインストールしてください。
  OPENAI_API_KEY 環境変数が必要。
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


# ─── 戻り値型 ─────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    """検索結果 1 件。"""
    doc_id: str
    text: str
    score: float   # cosine similarity (0.0〜1.0)
    source: str    # 元ファイル名


# ─── 抽象インターフェース ─────────────────────────────────────────

class Retriever(ABC):
    """
    Retriever 抽象基底クラス。

    将来 C2 の REST API（GET /retrieve）に差し替えるときは
    このクラスを継承した C2Retriever を作成するだけでよい。
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """
        query に関連するドキュメントを top_k 件返す。

        Args:
            query:  検索クエリ（STT テキストをそのまま渡す）
            top_k:  返す件数

        Returns:
            スコア降順の RetrievedDoc リスト
        """


# ─── ローカル実装 ─────────────────────────────────────────────────

class LocalRetriever(Retriever):
    """
    ローカルの index ファイルを使うコサイン類似度検索。

    build_index.py で生成した chunks.json / embeddings.npy を読み込む。
    """

    def __init__(self, index_path: str) -> None:
        """
        Args:
            index_path: index ファイルの格納ディレクトリ（例: "./data/index"）

        Raises:
            ImportError:     numpy が未インストール
            FileNotFoundError: index ファイルが見つからない
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "numpy が必要です。"
                " uv add --optional rag numpy でインストールしてください。"
            ) from exc

        chunks_file = os.path.join(index_path, "chunks.json")
        embeddings_file = os.path.join(index_path, "embeddings.npy")

        if not os.path.exists(chunks_file):
            raise FileNotFoundError(
                f"chunks.json が見つかりません: {chunks_file}\n"
                "scripts/build_index.py を実行してください。"
            )
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(
                f"embeddings.npy が見つかりません: {embeddings_file}\n"
                "scripts/build_index.py を実行してください。"
            )

        with open(chunks_file, encoding="utf-8") as f:
            self._chunks: list[dict] = json.load(f)

        self._embeddings = np.load(embeddings_file).astype(np.float32)

        logger.info(
            "LocalRetriever 初期化完了: %d チャンク, embedding shape=%s",
            len(self._chunks),
            self._embeddings.shape,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """
        query をベクトル化してコサイン類似度上位 top_k 件を返す。

        Args:
            query:  検索クエリ
            top_k:  返す件数

        Returns:
            スコア降順の RetrievedDoc リスト

        Raises:
            ImportError: openai / numpy が未インストール
        """
        import numpy as np

        t0 = time.monotonic()
        query_vec = _embed(query)
        q = np.array(query_vec, dtype=np.float32)

        # コサイン類似度 = dot(q_norm, E_norm^T)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normed = self._embeddings / np.where(norms == 0, 1.0, norms)
        q_norm = q / (np.linalg.norm(q) or 1.0)
        scores = normed @ q_norm  # shape (N,)

        top_indices = scores.argsort()[::-1][:top_k]
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info("retrieve 完了: latency_ms=%d top_k=%d", latency_ms, top_k)

        return [
            RetrievedDoc(
                doc_id=self._chunks[i]["doc_id"],
                text=self._chunks[i]["text"],
                score=float(scores[i]),
                source=self._chunks[i]["source"],
            )
            for i in top_indices
        ]


# ─── 埋め込みヘルパー ─────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """
    OpenAI Embeddings API でテキストをベクトル化する。

    OPENAI_API_KEY 環境変数が必要。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai が必要です。"
            " uv add --optional rag openai でインストールしてください。"
        ) from exc

    client = OpenAI()
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding
