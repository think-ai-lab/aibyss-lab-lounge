"""
test_rag_pipeline.py — RAG パイプライン統合テスト

外部 API（OpenAI / Redis）はモックする。
検証観点:
  - RAG on/off の切り替え（L2_ENABLE_RAG）
  - fallback: retriever 失敗時に非 RAG で継続
  - llm.final payload に rag_used / answer_mode / retrieved_doc_* が入る
  - 既存パイプライン（L2_ENABLE_RAG=false）が壊れない regression
  - debug artifacts が L2_DEBUG_ARTIFACTS=true のときのみ書き出される
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import lab_lounge.pipeline as pipeline_mod
from lab_lounge.pipeline import run_pipeline
from lab_lounge.llm import LLMResult
from lab_lounge.retriever import RetrievedDoc


# ─── 共通定数 ─────────────────────────────────────────────────────

COMMON = dict(
    stream_id="stream-rag-test",
    session_id="sess-rag-test",
    trace_id="trace-rag-test",
)

FAKE_LLM_RESULT = LLMResult(
    text="AITuberプロジェクトです。",
    model="gpt-4o-mini",
    input_tokens=20,
    output_tokens=10,
    latency_ms=300,
    finish_reason="stop",
)

FAKE_DOCS = [
    RetrievedDoc(doc_id="doc_0", text="Think-AI Lab. の説明文。", score=0.85, source="think_ai_lab.md"),
    RetrievedDoc(doc_id="doc_1", text="ミミ・オクタヴィアの説明文。", score=0.72, source="ai_agents.md"),
]


# ─── フィクスチャ ─────────────────────────────────────────────────

@pytest.fixture()
def mock_publish():
    with patch.object(pipeline_mod, "publish", return_value="1-0") as m:
        yield m


@pytest.fixture()
def mock_run_graph():
    with patch("lab_lounge.graph.run_graph", return_value=FAKE_LLM_RESULT) as m:
        yield m


@pytest.fixture()
def mock_real_llm(monkeypatch):
    monkeypatch.setenv("L2_USE_REAL_LLM", "true")
    monkeypatch.setenv("L2_LLM_PROVIDER", "openai")
    monkeypatch.setenv("L2_LLM_MODEL", "gpt-4o-mini")


@pytest.fixture()
def fake_index(tmp_path):
    """LocalRetriever が読める最小 fake index を作る。"""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    chunks = [
        {"doc_id": "doc_0", "text": "Think-AI Lab. の説明文。", "source": "think_ai_lab.md"},
        {"doc_id": "doc_1", "text": "ミミ・オクタヴィアの説明文。", "source": "ai_agents.md"},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    (index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(str(index_dir / "embeddings.npy"), embeddings)
    return str(index_dir)


# ─── RAG off — regression ─────────────────────────────────────────

class TestRagOff:
    """L2_ENABLE_RAG=false（デフォルト）で既存動作が壊れないことを確認する。"""

    def test_still_three_events(self, mock_publish):
        result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_rag_used_false_in_dummy_mode(self, mock_publish):
        result = run_pipeline("テスト", **COMMON)
        # dummy mode では llm_meta が空 → build_llm_final のデフォルト値
        assert result.events[1]["payload"]["rag_used"] is False

    def test_answer_mode_fallback_in_dummy_mode(self, mock_publish):
        result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["answer_mode"] == "fallback"

    def test_retrieved_doc_count_zero_in_dummy_mode(self, mock_publish):
        result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["retrieved_doc_count"] == 0

    def test_rag_used_false_in_real_llm_mode(self, mock_publish, mock_real_llm, mock_run_graph):
        result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["rag_used"] is False

    def test_answer_mode_fallback_in_real_llm_mode(self, mock_publish, mock_real_llm, mock_run_graph):
        result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["answer_mode"] == "fallback"

    def test_no_retriever_called_when_rag_off(self, mock_publish, mock_real_llm, mock_run_graph):
        with patch("lab_lounge.retriever.LocalRetriever") as mock_cls:
            run_pipeline("テスト", **COMMON)
        mock_cls.assert_not_called()


# ─── RAG on — grounded ────────────────────────────────────────────

class TestRagOn:
    """L2_ENABLE_RAG=true で RAG が動き payload に反映されることを確認する。"""

    @pytest.fixture(autouse=True)
    def enable_rag(self, monkeypatch, fake_index):
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        monkeypatch.setenv("L2_KB_PATH", fake_index)
        monkeypatch.setenv("L2_RAG_TOP_K", "2")

    def test_rag_used_true_when_docs_found(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["rag_used"] is True

    def test_answer_mode_grounded_when_docs_found(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["answer_mode"] == "grounded"

    def test_retrieved_doc_count_matches_top_k(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["retrieved_doc_count"] == 2

    def test_retrieved_doc_ids_in_payload(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        doc_ids = result.events[1]["payload"]["retrieved_doc_ids"]
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == 2

    def test_retrieval_latency_ms_is_recorded(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["retrieval_latency_ms"] >= 0

    def test_run_graph_called_with_context(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        """run_graph に context が渡ること（RAG コンテキストが LLM に届く）。"""
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            run_pipeline("テスト", **COMMON)
        _, kwargs = mock_run_graph.call_args
        assert kwargs.get("context") is not None
        assert isinstance(kwargs["context"], str)

    def test_still_three_events(self, mock_publish, mock_real_llm, mock_run_graph):
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_no_stream_idx_in_events(self, mock_publish, mock_real_llm, mock_run_graph):
        """Guardrail G-1: RAG on でも stream_idx を含めない。"""
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            result = run_pipeline("テスト", **COMMON)
        for ev in result.events:
            assert "stream_idx" not in ev


# ─── fallback — retriever 失敗時 ──────────────────────────────────

class TestRagFallback:
    """retriever が例外を投げたとき非 RAG で会話が継続することを確認する。"""

    @pytest.fixture(autouse=True)
    def enable_rag(self, monkeypatch, fake_index):
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        monkeypatch.setenv("L2_KB_PATH", fake_index)

    def test_pipeline_completes_on_retriever_error(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", side_effect=RuntimeError("embed error")):
            result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_rag_used_false_on_retriever_error(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", side_effect=RuntimeError("embed error")):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["rag_used"] is False

    def test_answer_mode_fallback_on_retriever_error(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        with patch("lab_lounge.retriever._embed", side_effect=RuntimeError("embed error")):
            result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["answer_mode"] == "fallback"

    def test_run_graph_called_without_context_on_error(
        self, mock_publish, mock_real_llm, mock_run_graph
    ):
        """fallback 時は context=None で run_graph が呼ばれること。"""
        with patch("lab_lounge.retriever._embed", side_effect=RuntimeError("embed error")):
            run_pipeline("テスト", **COMMON)
        _, kwargs = mock_run_graph.call_args
        assert kwargs.get("context") is None

    def test_pipeline_completes_on_missing_index(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch
    ):
        monkeypatch.setenv("L2_KB_PATH", "/nonexistent/path")
        result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_rag_used_false_on_missing_index(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch
    ):
        monkeypatch.setenv("L2_KB_PATH", "/nonexistent/path")
        result = run_pipeline("テスト", **COMMON)
        assert result.events[1]["payload"]["rag_used"] is False


# ─── debug artifacts ──────────────────────────────────────────────

class TestDebugArtifacts:
    """L2_DEBUG_ARTIFACTS の on/off で artifacts が書き出されるかを確認する。"""

    @pytest.fixture()
    def log_dir(self, tmp_path):
        return str(tmp_path / "logs")

    def test_artifacts_written_when_enabled(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch, log_dir
    ):
        monkeypatch.setenv("L2_DEBUG_ARTIFACTS", "true")
        monkeypatch.setenv("L2_DEBUG_LOG_DIR", log_dir)
        run_pipeline("テスト", **COMMON)
        assert os.path.exists(os.path.join(log_dir, "stt_output.json"))
        assert os.path.exists(os.path.join(log_dir, "llm_prompt.txt"))
        assert os.path.exists(os.path.join(log_dir, "llm_response.txt"))

    def test_artifacts_not_written_when_disabled(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch, log_dir
    ):
        monkeypatch.setenv("L2_DEBUG_ARTIFACTS", "false")
        monkeypatch.setenv("L2_DEBUG_LOG_DIR", log_dir)
        run_pipeline("テスト", **COMMON)
        assert not os.path.exists(os.path.join(log_dir, "stt_output.json"))

    def test_stt_output_contains_text(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch, log_dir
    ):
        monkeypatch.setenv("L2_DEBUG_ARTIFACTS", "true")
        monkeypatch.setenv("L2_DEBUG_LOG_DIR", log_dir)
        run_pipeline("確認テキスト", **COMMON)
        data = json.loads(open(os.path.join(log_dir, "stt_output.json"), encoding="utf-8").read())
        assert data["text"] == "確認テキスト"

    def test_retrieval_json_written_when_rag_on(
        self, mock_publish, mock_real_llm, mock_run_graph, monkeypatch, log_dir, fake_index
    ):
        monkeypatch.setenv("L2_DEBUG_ARTIFACTS", "true")
        monkeypatch.setenv("L2_DEBUG_LOG_DIR", log_dir)
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        monkeypatch.setenv("L2_KB_PATH", fake_index)
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            run_pipeline("テスト", **COMMON)
        assert os.path.exists(os.path.join(log_dir, "retrieval.json"))
        data = json.loads(open(os.path.join(log_dir, "retrieval.json"), encoding="utf-8").read())
        assert data["rag_enabled"] is True
        assert "results" in data
