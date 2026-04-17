"""
test_retriever.py — kb_loader / LocalRetriever ユニットテスト

OpenAI API は呼び出さない。_embed をモックして純粋に
ロード・チャンク化・コサイン類似度の動作を検証する。
"""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from lab_lounge.kb_loader import DocChunk, load_kb
from lab_lounge.retriever import LocalRetriever, RetrievedDoc


# ─── kb_loader テスト ──────────────────────────────────────────────

class TestLoadKb:
    def test_returns_list_of_doc_chunks(self, tmp_path):
        (tmp_path / "test.md").write_text("# タイトル\n\nテスト本文です。", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, DocChunk) for c in chunks)

    def test_chunk_has_required_fields(self, tmp_path):
        (tmp_path / "doc.md").write_text("# 見出し\n\n本文テキスト。", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        c = chunks[0]
        assert c.doc_id
        assert c.text
        assert c.source == "doc.md"

    def test_doc_id_format(self, tmp_path):
        (tmp_path / "think_ai_lab.md").write_text("# テスト\n\n内容です。", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        for c in chunks:
            assert c.doc_id.startswith("think_ai_lab_")

    def test_multiple_files_loaded(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nA の内容。", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\n\nB の内容。", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        sources = {c.source for c in chunks}
        assert "a.md" in sources
        assert "b.md" in sources

    def test_files_loaded_alphabetically(self, tmp_path):
        (tmp_path / "z.md").write_text("# Z\n\nZ の内容。", encoding="utf-8")
        (tmp_path / "a.md").write_text("# A\n\nA の内容。", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        sources = [c.source for c in chunks]
        a_first = sources.index("a.md")
        z_first = sources.index("z.md")
        assert a_first < z_first

    def test_empty_directory_returns_empty_list(self, tmp_path):
        chunks = load_kb(str(tmp_path))
        assert chunks == []

    def test_non_md_files_ignored(self, tmp_path):
        (tmp_path / "doc.md").write_text("# テスト\n\n内容。", encoding="utf-8")
        (tmp_path / "doc.txt").write_text("テキストファイル", encoding="utf-8")
        (tmp_path / "doc.json").write_text("{}", encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        assert all(c.source == "doc.md" for c in chunks)

    def test_short_paragraphs_merged(self, tmp_path):
        # MIN_CHARS 未満の段落は前のチャンクに結合される
        content = "# 見出し\n\n長いメインの段落テキストで、これはMIN_CHARSを超えるはずです。\n\n短い。"
        (tmp_path / "doc.md").write_text(content, encoding="utf-8")
        chunks = load_kb(str(tmp_path))
        # 「短い。」が単独チャンクにならず結合されていること
        texts = [c.text for c in chunks]
        assert not any(t.strip() == "短い。" for t in texts)

    def test_chunk_ids_are_unique(self, tmp_path):
        (tmp_path / "doc.md").write_text(
            "# 見出し1\n\nパラグラフ1の内容です。\n\n# 見出し2\n\nパラグラフ2の内容です。",
            encoding="utf-8",
        )
        chunks = load_kb(str(tmp_path))
        ids = [c.doc_id for c in chunks]
        assert len(ids) == len(set(ids))


# ─── LocalRetriever テスト ────────────────────────────────────────

def _make_fake_index(tmp_path, chunks: list[dict], embeddings: list[list[float]]) -> str:
    """テスト用の fake index ディレクトリを作る。"""
    index_dir = str(tmp_path / "index")
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    np.save(os.path.join(index_dir, "embeddings.npy"), np.array(embeddings, dtype=np.float32))
    return index_dir


FAKE_CHUNKS = [
    {"doc_id": "doc_0", "text": "Think-AI Lab. はAITuberプロジェクトです。", "source": "think_ai_lab.md"},
    {"doc_id": "doc_1", "text": "ミミ・オクタヴィアは深海貴族のAITuberです。", "source": "ai_agents.md"},
    {"doc_id": "doc_2", "text": "波心ちさめはジンベエザメがモチーフです。", "source": "ai_agents.md"},
]

# 各チャンクの embedding: 3次元で方向が明確に分かれているベクトル
FAKE_EMBEDDINGS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


class TestLocalRetrieverInit:
    def test_loads_index_successfully(self, tmp_path):
        index_dir = _make_fake_index(tmp_path, FAKE_CHUNKS, FAKE_EMBEDDINGS)
        retriever = LocalRetriever(index_dir)
        assert retriever is not None

    def test_raises_if_chunks_json_missing(self, tmp_path):
        index_dir = str(tmp_path / "empty_index")
        os.makedirs(index_dir)
        np.save(os.path.join(index_dir, "embeddings.npy"), np.zeros((1, 3), dtype=np.float32))
        with pytest.raises(FileNotFoundError, match="chunks.json"):
            LocalRetriever(index_dir)

    def test_raises_if_embeddings_npy_missing(self, tmp_path):
        index_dir = str(tmp_path / "no_emb_index")
        os.makedirs(index_dir)
        with open(os.path.join(index_dir, "chunks.json"), "w") as f:
            json.dump(FAKE_CHUNKS, f)
        with pytest.raises(FileNotFoundError, match="embeddings.npy"):
            LocalRetriever(index_dir)

    def test_raises_if_numpy_missing(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy not installed")
            return real_import(name, *args, **kwargs)

        index_dir = _make_fake_index(tmp_path, FAKE_CHUNKS, FAKE_EMBEDDINGS)
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", mock_import)
            with pytest.raises(ImportError, match="numpy"):
                LocalRetriever(index_dir)


class TestLocalRetrieverRetrieve:
    @pytest.fixture()
    def retriever(self, tmp_path):
        index_dir = _make_fake_index(tmp_path, FAKE_CHUNKS, FAKE_EMBEDDINGS)
        return LocalRetriever(index_dir)

    def _mock_embed(self, vec: list[float]):
        """指定ベクトルを返す _embed モック。"""
        return patch("lab_lounge.retriever._embed", return_value=vec)

    def test_returns_list_of_retrieved_docs(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedDoc) for r in results)

    def test_top_k_limits_results(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=2)
        assert len(results) == 2

    def test_top_k_1_returns_one_result(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=1)
        assert len(results) == 1

    def test_most_similar_doc_returned_first(self, retriever):
        # クエリが doc_0 方向 → doc_0 が1位
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=3)
        assert results[0].doc_id == "doc_0"

    def test_second_most_similar_doc(self, retriever):
        # クエリが doc_1 方向 → doc_1 が1位、doc_0 と doc_2 がそれ以下
        with self._mock_embed([0.0, 1.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=3)
        assert results[0].doc_id == "doc_1"

    def test_score_is_float_between_0_and_1(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_scores_descending(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_text_and_source(self, retriever):
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=1)
        assert results[0].text
        assert results[0].source

    def test_exact_match_score_is_1(self, retriever):
        # doc_0 の embedding と同じベクトルをクエリにすると score ≈ 1.0
        with self._mock_embed([1.0, 0.0, 0.0]):
            results = retriever.retrieve("テスト", top_k=1)
        assert abs(results[0].score - 1.0) < 1e-5


# ═══════════════════════════════════════════════════════════════════
# C2Retriever tests (Sprint Axis B Block 4 / CR-D)
# ═══════════════════════════════════════════════════════════════════

import logging  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from lab_lounge.retriever import (  # noqa: E402
    C2Retriever,
    CompositeRetriever,
    _rrf_merge,
)


def _mock_c2_response(results: list[dict], timed_out: bool = False, elapsed_ms: int = 10) -> MagicMock:
    """httpx.Client.get の戻り値モック。"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock(return_value=None)
    mock_resp.json = MagicMock(return_value={
        "results": results,
        "timed_out": timed_out,
        "elapsed_ms": elapsed_ms,
    })
    return mock_resp


class TestC2Retriever:
    """C2Retriever の HTTP クライアント動作を httpx モックで検証する。"""

    def test_returns_empty_on_connection_error(self, caplog):
        """httpx.ConnectError → 空リスト + warning ログ (fail-open)。"""
        import httpx

        retriever = C2Retriever("http://localhost:8100")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=httpx.ConnectError("connection refused"))

        with patch("httpx.Client", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.retriever"):
                result = retriever.retrieve("テスト", top_k=3)

        assert result == []
        assert any("C2Retriever" in r.getMessage() for r in caplog.records)

    def test_returns_empty_on_timeout(self, caplog):
        """httpx.TimeoutException → 空リスト + warning ログ。"""
        import httpx

        retriever = C2Retriever("http://localhost:8100", timeout_s=0.1)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("httpx.Client", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.retriever"):
                result = retriever.retrieve("テスト", top_k=3)

        assert result == []
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("タイムアウト" in m for m in warning_msgs)

    def test_parses_c2_response(self):
        """モックレスポンスを正しく RetrievedDoc にマッピングする。"""
        retriever = C2Retriever("http://localhost:8100")
        mock_resp = _mock_c2_response([
            {
                "event_id": "uuid-1",
                "type": "utterance.final",
                "ts": "2026-04-11T01:00:00Z",
                "stream_id": "s-1",
                "score": 1.0,
                "excerpt": "ミミ様、今日の天気は?",
            },
            {
                "event_id": "uuid-2",
                "type": "llm.final",
                "ts": "2026-04-11T01:00:02Z",
                "stream_id": "s-1",
                "score": 1.0,
                "excerpt": "晴れですわ",
            },
        ])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            docs = retriever.retrieve("天気", top_k=3)

        assert len(docs) == 2
        assert docs[0].doc_id == "uuid-1"
        assert docs[0].text == "ミミ様、今日の天気は?"
        assert docs[0].score == 1.0
        assert docs[0].source == "c2:utterance.final"
        assert docs[1].doc_id == "uuid-2"
        assert docs[1].source == "c2:llm.final"

    def test_passes_top_k_param(self):
        """top_k が HTTP query の k パラメータに渡される。"""
        retriever = C2Retriever("http://localhost:8100")
        mock_resp = _mock_c2_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("テスト", top_k=7)

        # call_args をチェック
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs.get("params", {}).get("k") == 7
        assert call_kwargs.get("params", {}).get("q") == "テスト"

    def test_source_prefix_c2_utterance(self):
        """source が 'c2:utterance.final' 形式になる。"""
        retriever = C2Retriever("http://localhost:8100")
        mock_resp = _mock_c2_response([
            {
                "event_id": "uuid-1",
                "type": "utterance.final",
                "ts": "2026-04-11T01:00:00Z",
                "stream_id": "s-1",
                "score": 1.0,
                "excerpt": "hello",
            },
        ])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            docs = retriever.retrieve("x")

        assert docs[0].source == "c2:utterance.final"

    def test_base_url_trailing_slash_stripped(self):
        """base_url の末尾 / は取り除かれる。"""
        retriever = C2Retriever("http://localhost:8100/")
        assert retriever._base_url == "http://localhost:8100"


# ═══════════════════════════════════════════════════════════════════
# CompositeRetriever tests (Sprint Axis B Block 4 / CR-D)
# ═══════════════════════════════════════════════════════════════════


class _StubRetriever:
    """CompositeRetriever テスト用のスタブ。"""
    def __init__(self, docs: list[RetrievedDoc] | Exception):
        self._docs_or_exc = docs

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        if isinstance(self._docs_or_exc, Exception):
            raise self._docs_or_exc
        return self._docs_or_exc[:top_k]


class TestCompositeRetriever:
    """CompositeRetriever の RRF マージ動作を検証する (TD-6 で min-max → RRF)。"""

    def test_requires_at_least_one_retriever(self):
        with pytest.raises(ValueError, match="最低 1 つ"):
            CompositeRetriever([])

    def test_merges_results_from_multiple_retrievers(self):
        """2 retriever の結果が結合される。"""
        local = _StubRetriever([
            RetrievedDoc(doc_id="l1", text="ローカル 1", score=0.9, source="kb:a.md"),
            RetrievedDoc(doc_id="l2", text="ローカル 2", score=0.7, source="kb:a.md"),
        ])
        c2 = _StubRetriever([
            RetrievedDoc(doc_id="c1", text="C2 1", score=0.8, source="c2:utterance.final"),
            RetrievedDoc(doc_id="c2", text="C2 2", score=0.6, source="c2:llm.final"),
        ])
        composite = CompositeRetriever([local, c2])

        docs = composite.retrieve("テスト", top_k=10)

        ids = {d.doc_id for d in docs}
        assert ids == {"l1", "l2", "c1", "c2"}

    def test_sorts_by_rrf_score_descending(self):
        """RRF マージ後にスコア降順でソートされる。"""
        local = _StubRetriever([
            RetrievedDoc(doc_id="l1", text="l1", score=0.9, source="kb"),
            RetrievedDoc(doc_id="l2", text="l2", score=0.7, source="kb"),
        ])
        c2 = _StubRetriever([
            RetrievedDoc(doc_id="c1", text="c1", score=0.8, source="c2"),
            RetrievedDoc(doc_id="c2", text="c2", score=0.6, source="c2"),
        ])
        composite = CompositeRetriever([local, c2])

        docs = composite.retrieve("テスト", top_k=10)

        # スコア降順に並ぶ
        scores = [d.score for d in docs]
        assert scores == sorted(scores, reverse=True)
        # top 2 は l1 と c1 (各 Retriever の rank 1 同士)
        top_ids = {docs[0].doc_id, docs[1].doc_id}
        assert top_ids == {"l1", "c1"}

    def test_caps_at_top_k(self):
        """結果が top_k 件を超えない。"""
        local = _StubRetriever([
            RetrievedDoc(doc_id=f"l{i}", text=f"l{i}", score=1.0 - i * 0.1, source="kb")
            for i in range(5)
        ])
        c2 = _StubRetriever([
            RetrievedDoc(doc_id=f"c{i}", text=f"c{i}", score=0.9 - i * 0.1, source="c2")
            for i in range(5)
        ])
        composite = CompositeRetriever([local, c2])

        docs = composite.retrieve("テスト", top_k=3)

        assert len(docs) == 3

    def test_one_retriever_failing_does_not_break_all(self, caplog):
        """1 つの retriever が例外を投げても他方の結果は返る。"""
        failing = _StubRetriever(RuntimeError("intentional failure"))
        working = _StubRetriever([
            RetrievedDoc(doc_id="w1", text="working", score=0.8, source="kb"),
        ])
        composite = CompositeRetriever([failing, working])

        with caplog.at_level(logging.WARNING, logger="lab_lounge.retriever"):
            docs = composite.retrieve("テスト", top_k=10)

        assert len(docs) == 1
        assert docs[0].doc_id == "w1"
        # warning ログに例外が記録される
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("_StubRetriever" in m or "intentional" in m for m in warning_msgs)


class TestRRFMerge:
    """_rrf_merge (Reciprocal Rank Fusion) の動作を直接検証。TD-6 で min-max から切替。"""

    def test_empty_input(self):
        """入力なしは空リスト。"""
        assert _rrf_merge([], top_k=3) == []

    def test_single_retriever_ranks(self):
        """1 Retriever × 3 docs: RRF スコアが rank 順に降順。"""
        docs = [
            RetrievedDoc("a", "a", 0.9, "s"),
            RetrievedDoc("b", "b", 0.5, "s"),
            RetrievedDoc("c", "c", 0.1, "s"),
        ]
        merged = _rrf_merge([docs], top_k=3)
        assert len(merged) == 3
        assert merged[0].doc_id == "a"
        assert merged[1].doc_id == "b"
        assert merged[2].doc_id == "c"
        # RRF スコアが降順
        assert merged[0].score > merged[1].score > merged[2].score
        # 具体値: 1/(60+1) ≈ 0.01639
        assert abs(merged[0].score - 1.0 / 61) < 1e-6

    def test_two_retrievers_no_overlap(self):
        """2 Retriever が全く異なる doc を返す: 各 Retriever の rank 1 が上位に。"""
        r1 = [RetrievedDoc("a", "a", 0.9, "s1"), RetrievedDoc("b", "b", 0.5, "s1")]
        r2 = [RetrievedDoc("c", "c", 0.8, "s2"), RetrievedDoc("d", "d", 0.4, "s2")]
        merged = _rrf_merge([r1, r2], top_k=4)
        # a と c はどちらも rank 1 (同じ RRF score) → 安定ソートで出現順
        assert merged[0].doc_id in ("a", "c")
        assert merged[1].doc_id in ("a", "c")
        assert {merged[0].doc_id, merged[1].doc_id} == {"a", "c"}

    def test_dedup_boosts_shared_doc(self):
        """同一 doc_id が 2 Retriever から返された場合: RRF スコアが合算される。"""
        # Retriever 1: shared at rank 1, unique_a at rank 2
        # Retriever 2: unique_b at rank 1, shared at rank 2
        r1 = [RetrievedDoc("shared", "shared", 0.9, "s1"), RetrievedDoc("a", "a", 0.5, "s1")]
        r2 = [RetrievedDoc("b", "b", 0.8, "s2"), RetrievedDoc("shared", "shared", 0.4, "s2")]
        merged = _rrf_merge([r1, r2], top_k=3)
        # shared: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # a: 1/(60+2) = 0.01613
        # b: 1/(60+1) = 0.01639
        # → shared > b > a
        assert merged[0].doc_id == "shared"
        assert merged[0].score > merged[1].score

    def test_top_k_caps_output(self):
        """結果件数は top_k で制限される。"""
        r1 = [RetrievedDoc(f"d{i}", f"d{i}", 0.9 - i * 0.1, "s") for i in range(5)]
        merged = _rrf_merge([r1], top_k=2)
        assert len(merged) == 2

    def test_insertion_order_does_not_dominate(self):
        """
        min-max 時代の回帰テスト: Local が先に来ても、
        複数 Retriever が合意した doc は Local の top よりも上位に来る。
        """
        local = [RetrievedDoc("local_top", "local", 0.95, "kb")]
        semantic = [RetrievedDoc("c2_top", "c2", 0.12, "c2")]
        recent = [RetrievedDoc("c2_top", "c2", 1.0, "c2:recent")]
        # c2_top は semantic + recent の 2 Retriever から rank 1 で返される
        # → 合算スコア 2/(60+1) ≈ 0.0328 > local_top の 1/(60+1) ≈ 0.0164
        merged = _rrf_merge([local, semantic, recent], top_k=2)
        assert merged[0].doc_id == "c2_top"
        assert merged[1].doc_id == "local_top"


# ═══════════════════════════════════════════════════════════════════
# C2Retriever exclude_event_ids (Block 5 / TD-1)
# ═══════════════════════════════════════════════════════════════════


class TestC2RetrieverExcludeEventIds:
    """C2Retriever の exclude_event_ids パラメータが HTTP query に渡ることを検証。"""

    def test_passes_exclude_event_ids_to_http_params(self):
        """コンストラクタの exclude_event_ids が HTTP query に反映される。"""
        retriever = C2Retriever(
            "http://localhost:8100",
            exclude_event_ids=["uuid-aaa", "uuid-bbb"],
        )
        mock_resp = _mock_c2_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("テスト", top_k=3)

        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["params"]["exclude_event_ids"] == "uuid-aaa,uuid-bbb"

    def test_empty_exclude_list_not_sent(self):
        """exclude_event_ids が None または [] のときはパラメータが付かない。"""
        retriever = C2Retriever("http://localhost:8100")
        mock_resp = _mock_c2_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("テスト")

        call_kwargs = mock_client.get.call_args.kwargs
        assert "exclude_event_ids" not in call_kwargs.get("params", {})


# ═══════════════════════════════════════════════════════════════════
# RecentC2Retriever (Block 5 / TD-1)
# ═══════════════════════════════════════════════════════════════════


from lab_lounge.retriever import RecentC2Retriever  # noqa: E402


def _mock_recent_response(results: list[dict]) -> MagicMock:
    """httpx.Client.get の戻り値モック (recent 形式)。"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock(return_value=None)
    mock_resp.json = MagicMock(return_value={
        "results": results,
        "timed_out": False,
        "elapsed_ms": 8,
    })
    return mock_resp


class TestRecentC2Retriever:
    """RecentC2Retriever の HTTP クライアント動作を httpx モックで検証する。"""

    def test_accepts_none_stream_id_for_global_scope(self):
        """stream_id=None は許容される (global scope、セッションまたぎメモリ)。"""
        retriever = RecentC2Retriever("http://localhost:8100")
        assert retriever._stream_id is None

    def test_empty_stream_id_treated_as_none(self):
        """空文字列の stream_id も None 扱い (env から空文字が渡る可能性への防御)。"""
        retriever = RecentC2Retriever("http://localhost:8100", stream_id="")
        assert retriever._stream_id is None

    def test_explicit_stream_id_stored(self):
        """stream_id 明示指定は保持される (session scope)。"""
        retriever = RecentC2Retriever("http://localhost:8100", stream_id="sess-1")
        assert retriever._stream_id == "sess-1"

    def test_global_scope_omits_stream_id_param(self):
        """stream_id=None のとき HTTP query に stream_id パラメータが付かない。"""
        retriever = RecentC2Retriever("http://localhost:8100")
        mock_resp = _mock_recent_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("ignored")

        call_kwargs = mock_client.get.call_args.kwargs
        params = call_kwargs.get("params", {})
        assert "stream_id" not in params
        assert "k" in params  # k は常に送られる

    def test_returns_empty_on_connection_error(self, caplog):
        """httpx.ConnectError → 空リスト + warning (fail-open)。"""
        import httpx

        retriever = RecentC2Retriever("http://localhost:8100", stream_id="stream-1")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=httpx.ConnectError("refused"))

        with patch("httpx.Client", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.retriever"):
                result = retriever.retrieve("query ignored", top_k=3)

        assert result == []
        assert any("RecentC2Retriever" in r.getMessage() for r in caplog.records)

    def test_returns_empty_on_timeout(self, caplog):
        """タイムアウト → 空リスト + warning。"""
        import httpx

        retriever = RecentC2Retriever(
            "http://localhost:8100", stream_id="stream-1", timeout_s=0.1
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("httpx.Client", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.retriever"):
                result = retriever.retrieve("q")

        assert result == []
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("タイムアウト" in m for m in warning_msgs)

    def test_parses_recent_response(self):
        """モックレスポンスを正しく RetrievedDoc にマッピングする。"""
        retriever = RecentC2Retriever("http://localhost:8100", stream_id="stream-1")
        mock_resp = _mock_recent_response([
            {
                "event_id": "uuid-1",
                "type": "utterance.final",
                "ts": "2026-04-11T14:50:00Z",
                "stream_id": "stream-1",
                "stream_idx": 5,
                "score": 1.0,
                "excerpt": "ミミ様、今日の天気は?",
            },
            {
                "event_id": "uuid-2",
                "type": "llm.final",
                "ts": "2026-04-11T14:50:02Z",
                "stream_id": "stream-1",
                "stream_idx": 4,
                "score": 1.0,
                "excerpt": "晴れですわ",
            },
        ])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            docs = retriever.retrieve("query ignored")

        assert len(docs) == 2
        assert docs[0].doc_id == "uuid-1"
        assert docs[0].source == "c2-recent:utterance.final"
        assert docs[1].source == "c2-recent:llm.final"

    def test_passes_stream_id_param(self):
        """stream_id が HTTP query に渡される。"""
        retriever = RecentC2Retriever(
            "http://localhost:8100", stream_id="my-stream-xyz", top_k=7
        )
        mock_resp = _mock_recent_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("ignored", top_k=3)

        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["params"]["stream_id"] == "my-stream-xyz"
        # top_k と default_top_k の大きい方 = 7
        assert call_kwargs["params"]["k"] == 7

    def test_passes_exclude_event_ids_param(self):
        """exclude_event_ids が HTTP query にカンマ区切りで渡される。"""
        retriever = RecentC2Retriever(
            "http://localhost:8100",
            stream_id="stream-1",
            exclude_event_ids=["uuid-current"],
        )
        mock_resp = _mock_recent_response([])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            retriever.retrieve("ignored")

        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["params"]["exclude_event_ids"] == "uuid-current"

    def test_source_prefix_c2_recent(self):
        """source が 'c2-recent:<type>' 形式になる。"""
        retriever = RecentC2Retriever("http://localhost:8100", stream_id="stream-1")
        mock_resp = _mock_recent_response([
            {
                "event_id": "uuid-1",
                "type": "utterance.final",
                "ts": "2026-04-11T00:00:00Z",
                "stream_id": "stream-1",
                "stream_idx": 0,
                "score": 1.0,
                "excerpt": "hello",
            },
        ])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("httpx.Client", return_value=mock_client):
            docs = retriever.retrieve("q")

        assert docs[0].source == "c2-recent:utterance.final"


# ═══════════════════════════════════════════════════════════════════
# Sprint Axis D Block 2: AITuber 相互知識の常時保持
# 実 kb index にキャラクター相互知識が索引されていることの検証
# (OpenAI API 呼び出しなし、chunks.json のメタデータレベル検証のみ)
# ═══════════════════════════════════════════════════════════════════


class TestCharacterKnowledgeInIndex:
    """本番 kb index (data/index/chunks.json) にキャラクター相互知識が
    索引されていることを検証する。

    前提:
    - docs/kb/ai_agents.md, think_ai_lab.md が存在する
    - scripts/build_index.py で索引済み
    - index がない環境 (初回セットアップ前等) では skip される
    """

    @pytest.fixture()
    def real_chunks(self):
        """本番 index の chunks.json を読み込む (なければ skip)。"""
        import json
        import os
        index_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "index", "chunks.json"
        )
        if not os.path.exists(index_path):
            pytest.skip(f"kb index not found: {index_path}. Run scripts/build_index.py first.")
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)

    def test_ai_agents_md_indexed(self, real_chunks):
        """ai_agents.md が index に含まれる。"""
        sources = {c["source"] for c in real_chunks}
        assert "ai_agents.md" in sources, (
            f"ai_agents.md missing from index. Found sources: {sources}"
        )

    def test_think_ai_lab_md_indexed(self, real_chunks):
        """think_ai_lab.md が index に含まれる。"""
        sources = {c["source"] for c in real_chunks}
        assert "think_ai_lab.md" in sources, (
            f"think_ai_lab.md missing from index. Found sources: {sources}"
        )

    def test_all_five_characters_mentioned_in_kb(self, real_chunks):
        """5 キャラ全員が kb の少なくとも 1 チャンクに含まれる。"""
        all_text = " ".join(c["text"] for c in real_chunks)
        for char_name in ["ミミ", "ちさめ", "さくら", "ルカ", "オクタメイド"]:
            assert char_name in all_text, (
                f"character '{char_name}' not found in kb — "
                f"クロスキャラクター参照が機能しない可能性"
            )

    def test_wake_words_documented(self, real_chunks):
        """各キャラのウェイクワードが kb に記述されている。"""
        all_text = " ".join(c["text"] for c in real_chunks)
        for wake_word in ["ミミ様", "ちさめさん", "さくらさん", "オクタメイド"]:
            assert wake_word in all_text, (
                f"wake word '{wake_word}' not documented in kb"
            )

    def test_four_quadrant_design_documented(self, real_chunks):
        """think_ai_lab.md に 4 象限バランス設計 (抽象↔具体 × 論理↔感情) が記述されている。"""
        tal_chunks = [c for c in real_chunks if c["source"] == "think_ai_lab.md"]
        tal_text = " ".join(c["text"] for c in tal_chunks)
        assert "抽象" in tal_text
        assert "具体" in tal_text
        assert "論理" in tal_text
        assert "感情" in tal_text
        # 4 人の役割割り当てが think_ai_lab.md にある
        for char in ["ルカ", "ミミ", "ちさめ", "さくら"]:
            assert char in tal_text, f"{char} の役割が think_ai_lab.md に記載なし"

    def test_naming_rules_documented(self, real_chunks):
        """呼称ルールが ai_agents.md に記述されている (アビスメイト含む)。"""
        agents_chunks = [c for c in real_chunks if c["source"] == "ai_agents.md"]
        agents_text = " ".join(c["text"] for c in agents_chunks)
        # アビスメイト (視聴者呼称) と呼称関連ワード
        assert "アビスメイト" in agents_text, "視聴者呼称 アビスメイト が kb に記載なし"

    def test_chisame_ramble_mode_documented(self, real_chunks):
        """ちさめ暴走モード (専門話題で早口になる特性) が kb に記述されている。

        他キャラが「ちさめさん、暴走してる?」と参照する時に必要。
        """
        all_text = " ".join(c["text"] for c in real_chunks)
        assert "暴走" in all_text, "ちさめ暴走モードが kb に記載なし"

    def test_conversation_flow_documented(self, real_chunks):
        """会話フロー (ルカ→ミミ→ちさめ→さくら→ルカ) が think_ai_lab.md にある。

        各キャラが自分の発言順序・役割を把握するために必要。
        """
        tal_chunks = [c for c in real_chunks if c["source"] == "think_ai_lab.md"]
        tal_text = " ".join(c["text"] for c in tal_chunks)
        # 基本フローに 4 キャラ全員が登場する段落があるはず
        # (think_ai_lab.md の「会話の流れ」セクション)
        flow_chunks = [c for c in tal_chunks
                       if all(name in c["text"] for name in ["ルカ", "ミミ", "ちさめ", "さくら"])]
        assert len(flow_chunks) > 0, (
            "4 人全員が登場するチャンクがない — 会話フローの相互認識ができない"
        )
