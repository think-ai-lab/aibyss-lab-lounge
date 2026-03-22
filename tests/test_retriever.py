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
