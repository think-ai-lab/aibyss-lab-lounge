"""
retriever.py — Retriever インターフェースと LocalRetriever / C2Retriever / CompositeRetriever

責務:
  - Retriever 抽象基底クラスを定義する（将来 C2 API への差し替えを容易にする）
  - LocalRetriever: ローカルの index ファイルを使いコサイン類似度検索を行う
  - C2Retriever: C2 の GET /retrieve を HTTP で呼び出す純粋なクライアント
  - CompositeRetriever: 複数 Retriever を束ねスコア正規化してマージする
  - pipeline.py / graph.py からのみ呼び出す

【index ファイル構成】
  {kb_path}/chunks.json     — チャンクメタデータ
                              [{"doc_id": ..., "text": ..., "source": ...}, ...]
  {kb_path}/embeddings.npy  — numpy 行列 (N, D) float32

【C2Retriever 設計原則】
  L2 は C2 の embedding モデル・vector DB の種類・インデックス構造を一切知らない。
  純粋な HTTP クライアントとして振る舞い、C2 側が LIKE → sqlite-vec →
  将来の pgvector へ切り替えても本クラスは無変更。
  詳細: docs/design/lab-lounge-retriever.md

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
from dataclasses import dataclass, replace

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


# ═══════════════════════════════════════════════════════════════════
# C2Retriever — C2 の GET /retrieve を呼ぶ純粋な HTTP クライアント
# (Sprint Axis B Block 4 / CR-D)
# ═══════════════════════════════════════════════════════════════════


class C2Retriever(Retriever):
    """
    C2 (Coral Chronicle) の GET /retrieve を呼んで過去の会話履歴を取得する。

    設計原則:
      - L2 は embedding モデル・vector DB の知識を一切持たない
      - 純粋な HTTP クライアント
      - C2 側が LIKE → sqlite-vec へ切り替えても本クラスは無変更
      - fail-open: C2 到達不能 / タイムアウト / パース失敗 → 空リストを返し、
                    呼び出し元 (CompositeRetriever or graph ノード) が LocalRetriever
                    結果だけで応答を継続できるようにする

    L2_USE_C2_RETRIEVER=true で有効化される。
    """

    def __init__(
        self,
        base_url: str,
        top_k: int = 3,
        timeout_s: float = 0.6,
        exclude_event_ids: list[str] | None = None,
    ) -> None:
        """
        Args:
            base_url:   C2 のベース URL (例: "http://localhost:8100")
            top_k:      retrieve() のデフォルト取得件数
            timeout_s:  HTTP リクエストのタイムアウト秒数
            exclude_event_ids: 検索結果から除外する event_id のリスト。
                L2 は現ターンの utterance.final.event_id を渡して
                ナルシシスティック RAG (自己参照) を防止する。
        """
        self._base_url = base_url.rstrip("/")
        self._default_top_k = top_k
        self._timeout_s = timeout_s
        self._exclude_event_ids = list(exclude_event_ids) if exclude_event_ids else []

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """
        C2 の GET /retrieve を呼び出して RetrievedDoc リストを返す。

        fail-open:
          - 接続エラー / タイムアウト / 5xx / JSON パース失敗 → 空リスト + warning ログ
          - L2 の応答を止めない
        """
        try:
            import httpx
        except ImportError as exc:
            logger.warning("httpx が未インストール → C2Retriever 無効: %s", exc)
            return []

        url = f"{self._base_url}/retrieve"
        params: dict[str, str | int] = {"q": query, "k": top_k}
        if self._exclude_event_ids:
            params["exclude_event_ids"] = ",".join(self._exclude_event_ids)

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.warning("C2Retriever タイムアウト (timeout_s=%s)", self._timeout_s)
            return []
        except httpx.HTTPError as exc:
            logger.warning("C2Retriever HTTP エラー: %s", exc)
            return []
        except Exception as exc:  # JSON パース失敗等
            logger.warning("C2Retriever 予期せぬエラー: %s", exc)
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
        docs: list[RetrievedDoc] = []
        for r in results:
            try:
                docs.append(
                    RetrievedDoc(
                        doc_id=str(r["event_id"]),
                        text=r.get("excerpt") or "",
                        score=float(r.get("score", 0.0)),
                        source=f"c2:{r.get('type', 'unknown')}",
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("C2Retriever: レスポンス項目のパース失敗: %s (r=%s)", exc, r)
                continue

        return docs


# ═══════════════════════════════════════════════════════════════════
# RecentC2Retriever — C2 の GET /retrieve/recent を呼ぶ時系列クライアント
# (Sprint Axis B Block 5 / TD-1 早期対応)
# ═══════════════════════════════════════════════════════════════════


class RecentC2Retriever(Retriever):
    """
    C2 (Coral Chronicle) の GET /retrieve/recent を呼んで時系列リコールを行う。

    LIKE 検索 (C2Retriever) では「さっきの話は?」のような query-to-past 不一致な
    会話メモリ要求に対応できないため、query を無視して時系列で直近 N 件を
    取得するクライアントとして別クラスにした。

    【スコープ】
      - stream_id 指定時 (session scope): そのストリーム内の直近 N 件
        (現在の run_loop セッション内の会話メモリ)
      - stream_id 未指定時 (global scope, default): 全ストリーム横断の直近 N 件
        (セッションまたぎメモリ。過去の run_loop セッションで話した内容も参照可能)

    設計原則:
      - query は無視する (時系列ベース、過去発話とのキーワード一致に依存しない)
      - L2 は HTTP クライアントに徹し、C2 内部の実装 (LIKE 検索等) を意識しない
      - fail-open: C2 到達不能 / タイムアウト → 空リスト + warning ログ

    L2_USE_C2_RECENT=true (default) で CompositeRetriever に組み込まれる。
    スコープは L2_C2_RECENT_SCOPE=global (default) / session で制御。
    """

    def __init__(
        self,
        base_url: str,
        stream_id: str | None = None,
        top_k: int = 5,
        timeout_s: float = 0.6,
        exclude_event_ids: list[str] | None = None,
    ) -> None:
        """
        Args:
            base_url:   C2 のベース URL (例: "http://localhost:8100")
            stream_id:  対象ストリーム ID。None の場合は全ストリーム横断 (global scope)。
            top_k:      retrieve() のデフォルト取得件数
            timeout_s:  HTTP リクエストのタイムアウト秒数
            exclude_event_ids: 結果から除外する event_id のリスト (自己参照防止)
        """
        self._base_url = base_url.rstrip("/")
        # 空文字列も None 扱いにする (env から空文字が渡る可能性への防御)
        self._stream_id = stream_id if stream_id else None
        self._default_top_k = top_k
        self._timeout_s = timeout_s
        self._exclude_event_ids = list(exclude_event_ids) if exclude_event_ids else []

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """
        C2 の GET /retrieve/recent を呼び出して直近の会話履歴を返す。

        query は無視される (時系列ベース)。top_k と default_top_k の大きい方を使う。
        """
        try:
            import httpx
        except ImportError as exc:
            logger.warning("httpx が未インストール → RecentC2Retriever 無効: %s", exc)
            return []

        url = f"{self._base_url}/retrieve/recent"
        params: dict[str, str | int] = {
            "k": max(top_k, self._default_top_k),
        }
        if self._stream_id:
            params["stream_id"] = self._stream_id
        if self._exclude_event_ids:
            params["exclude_event_ids"] = ",".join(self._exclude_event_ids)

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.warning("RecentC2Retriever タイムアウト (timeout_s=%s)", self._timeout_s)
            return []
        except httpx.HTTPError as exc:
            logger.warning("RecentC2Retriever HTTP エラー: %s", exc)
            return []
        except Exception as exc:
            logger.warning("RecentC2Retriever 予期せぬエラー: %s", exc)
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
        docs: list[RetrievedDoc] = []
        for r in results:
            try:
                docs.append(
                    RetrievedDoc(
                        doc_id=str(r["event_id"]),
                        text=r.get("excerpt") or "",
                        score=float(r.get("score", 1.0)),
                        source=f"c2-recent:{r.get('type', 'unknown')}",
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("RecentC2Retriever: レスポンス項目のパース失敗: %s (r=%s)", exc, r)
                continue

        return docs


# ═══════════════════════════════════════════════════════════════════
# CompositeRetriever — 複数 Retriever を束ねてマージする
# (Sprint Axis B Block 4 / CR-D)
# ═══════════════════════════════════════════════════════════════════


class CompositeRetriever(Retriever):
    """
    複数の Retriever を順番に呼び出し、Reciprocal Rank Fusion (RRF) でマージする。

    Sprint Axis C Follow-up 2 (TD-6) で min-max 正規化 → RRF に切替。
    RRF はランクベースなので、異なるスコア尺度 (text-embedding-3-small vs
    voyage-4-large vs time-based 1.0) を自然に統合でき、同率 1.0 の人工的タイが
    起きない。同一 doc_id が複数 Retriever から返された場合は RRF スコアが合算され、
    "複数の情報源が合意した doc" が上位に浮かぶ。

    - 1 つの Retriever が例外を投げても他方の結果は返す (fail-open)
    - 各 sub-retriever に max(top_k*3, 10) を渡してプール拡大 (TD-7)
    - 最終的に top_k 件に絞る
    """

    def __init__(self, retrievers: list[Retriever]) -> None:
        """
        Args:
            retrievers: 組み合わせる Retriever のリスト (例: [LocalRetriever, C2Retriever])
        """
        if not retrievers:
            raise ValueError("CompositeRetriever には最低 1 つの Retriever が必要です")
        self._retrievers = retrievers

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """
        各 Retriever から候補を取得し、RRF でマージして上位 top_k を返す。

        sub_top_k = max(top_k * 3, 10) で各 Retriever に多めの候補を要求し、
        RRF マージプールを豊かにする。
        """
        sub_top_k = max(top_k * 3, 10)
        all_results: list[list[RetrievedDoc]] = []
        for retriever in self._retrievers:
            try:
                docs = retriever.retrieve(query, sub_top_k)
            except Exception as exc:
                logger.warning(
                    "CompositeRetriever: %s が例外発生 → 空リスト扱い: %s",
                    type(retriever).__name__,
                    exc,
                )
                continue
            all_results.append(docs)

        return _rrf_merge(all_results, top_k)


# ─── Reciprocal Rank Fusion (TD-6) ─────────────────────────────────

RRF_K = 60  # 標準 RRF 定数 (論文 "Reciprocal Rank Fusion outperforms Condorcet...")


def _rrf_merge(
    results_per_retriever: list[list[RetrievedDoc]],
    top_k: int,
) -> list[RetrievedDoc]:
    """
    Reciprocal Rank Fusion で複数 Retriever の結果をマージする。

    各 Retriever の結果は score 降順ソート済み前提 (Retriever.retrieve の契約)。
    同一 doc_id が複数 Retriever から返された場合は RRF スコアを合算する
    (ボーナス: 複数 Retriever が合意した doc は自動的に上位に浮かぶ)。

    RRF_score(doc) = sum(1 / (RRF_K + rank_i)) for each retriever i that returned doc
    """
    if not results_per_retriever:
        return []

    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}

    for retriever_results in results_per_retriever:
        for rank, doc in enumerate(retriever_results, start=1):
            rrf_score = 1.0 / (RRF_K + rank)
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + rrf_score
            # 同一 doc_id が複数 Retriever から返された場合、最初に出現した doc を保持
            # (text / source は最初の Retriever のものを採用)
            if doc.doc_id not in docs:
                docs[doc.doc_id] = doc

    # スコア降順でソートし、top_k 件に絞る
    sorted_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)
    return [
        replace(docs[did], score=scores[did])
        for did in sorted_ids[:top_k]
    ]
