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
    """L2_ENABLE_RAG=true で retrieve_memory ツールが登録される
    ことを確認する（Sprint Axis D Block 3: Agent ツール化後）。

    旧テスト (rag_used/answer_mode/retrieved_doc_count 等) は
    _retrieval_node 削除に伴い廃止。RAG 検索は Agent が自律的に
    retrieve_memory ツールを呼ぶため、ツール登録の検証のみ行う。
    ツール呼び出し自体の検証は test_graph_agent.py / test_retrieve_memory.py
    および E2E 手動テストに委ねる。
    """

    @pytest.fixture(autouse=True)
    def enable_rag(self, monkeypatch, fake_index):
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        monkeypatch.setenv("L2_KB_PATH", fake_index)
        monkeypatch.setenv("L2_RAG_TOP_K", "2")

    def test_still_three_events(self, mock_publish, mock_real_llm, mock_run_graph):
        result = run_pipeline("テスト", **COMMON)
        assert len(result.events) == 3

    def test_no_stream_idx_in_events(self, mock_publish, mock_real_llm, mock_run_graph):
        """Guardrail G-1: RAG on でも stream_idx を含めない。"""
        result = run_pipeline("テスト", **COMMON)
        for ev in result.events:
            assert "stream_idx" not in ev

    def test_rag_enabled_flag_controls_tool_registration(self, monkeypatch):
        """L2_ENABLE_RAG=true で _is_rag_enabled() が True を返す。"""
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is True


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

    # Sprint Axis D Block 3: _retrieval_node 削除に伴い retrieval.json は
    # 生成されなくなった。RAG 検索は Agent がツールとして呼ぶため、
    # debug artifacts は LangSmith トレースに委ねる。


# ═══════════════════════════════════════════════════════════════════
# L2_USE_C2_RETRIEVER — CompositeRetriever 組み込み (Block 4 / CR-D)
# ═══════════════════════════════════════════════════════════════════


class TestRAGPipelineWithC2Retriever:
    """L2_USE_C2_RETRIEVER=true で CompositeRetriever が使われることを検証する。

    graph.py の _build_retriever_from_env ヘルパに着目して env 分岐のロジックを確認する。
    pipeline.run_pipeline 全体を呼ぶのは検証ノイズが多いため、
    ヘルパを直接呼ぶユニットレベルで検証する。
    """

    def test_l2_use_c2_retriever_enabled_uses_composite(self, monkeypatch, fake_index):
        """env true で CompositeRetriever が構築される (Recent を無効化して C2 のみ検証)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import CompositeRetriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")
        monkeypatch.setenv("L2_C2_RETRIEVER_TOP_K", "3")
        monkeypatch.setenv("L2_USE_C2_RECENT", "false")  # 本テストは C2Retriever のみを検証

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        assert isinstance(retriever, CompositeRetriever)
        # 2 つの retriever (Local + C2) が含まれる
        assert len(retriever._retrievers) == 2

    def test_l2_use_c2_retriever_disabled_uses_local_only(self, monkeypatch, fake_index):
        """env 未設定 (default false) で LocalRetriever 単独。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import LocalRetriever, CompositeRetriever

        monkeypatch.delenv("L2_USE_C2_RETRIEVER", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        assert isinstance(retriever, LocalRetriever)
        assert not isinstance(retriever, CompositeRetriever)

    def test_l2_use_c2_retriever_false_uses_local_only(self, monkeypatch, fake_index):
        """env=false でも LocalRetriever 単独。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import LocalRetriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "false")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        assert isinstance(retriever, LocalRetriever)

    def test_c2_unreachable_falls_back_to_local(self, monkeypatch, fake_index):
        """C2 HTTP 失敗時、LocalRetriever 結果だけで応答が返る。"""
        import httpx
        from lab_lounge.graph import _build_retriever_from_env

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        # C2 HTTP を接続失敗にする
        mock_httpx_client = MagicMock()
        mock_httpx_client.__enter__ = MagicMock(return_value=mock_httpx_client)
        mock_httpx_client.__exit__ = MagicMock(return_value=False)
        mock_httpx_client.get = MagicMock(side_effect=httpx.ConnectError("refused"))

        with patch("httpx.Client", return_value=mock_httpx_client), \
             patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            docs = retriever.retrieve("テスト", top_k=3)

        # C2 は空リストを返すが LocalRetriever は結果を返す
        assert len(docs) > 0
        # 全結果が kb: プレフィックス (LocalRetriever 由来)
        for d in docs:
            assert d.source != "c2:unknown"
            assert not d.source.startswith("c2:")

    def test_c2_url_default_value(self, monkeypatch, fake_index):
        """L2_C2_URL 未設定時のデフォルト値 http://localhost:8100 が使われる。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import C2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.delenv("L2_C2_URL", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        # 2 つ目の retriever が C2Retriever で base_url がデフォルト
        c2_retriever = retriever._retrievers[1]
        assert isinstance(c2_retriever, C2Retriever)
        assert c2_retriever._base_url == "http://localhost:8100"


# ═══════════════════════════════════════════════════════════════════
# L2_C2_URL_READONLY — dev プロファイルで prod を read-only 参照
# (Sprint Axis B Block 4 / Phase 4.5 profile separation)
# ═══════════════════════════════════════════════════════════════════


class TestC2RetrieverReadonlyProfile:
    """L2_C2_URL_READONLY で副 C2 を read-only 参照する設計を検証する。

    dev プロファイルが自身の C2 (primary) に書きつつ、prod C2 (readonly)
    からも過去の会話を参照できるかを確認する。

    【注】本クラスのテストは C2Retriever (LIKE 検索) の primary/readonly 構成に
    焦点を当てているので、RecentC2Retriever (別クラス) を無効化して
    retrievers の数を決定的にする。Recent の挙動は TestRAGPipelineWithRecentRetriever で検証。
    """

    @pytest.fixture(autouse=True)
    def disable_recent_retriever(self, monkeypatch):
        """本クラス全体で RecentC2Retriever を無効化する。"""
        monkeypatch.setenv("L2_USE_C2_RECENT", "false")

    def test_readonly_url_unset_uses_single_c2(self, monkeypatch, fake_index):
        """L2_C2_URL_READONLY 未設定 → C2Retriever は primary 1 個のみ (2 retrievers)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import C2Retriever, LocalRetriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")
        monkeypatch.delenv("L2_C2_URL_READONLY", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        # Local + C2 primary の 2 個のみ
        assert len(retriever._retrievers) == 2
        assert isinstance(retriever._retrievers[0], LocalRetriever)
        assert isinstance(retriever._retrievers[1], C2Retriever)
        assert retriever._retrievers[1]._base_url == "http://localhost:8100"

    def test_readonly_url_set_adds_second_c2(self, monkeypatch, fake_index):
        """L2_C2_URL_READONLY 設定 → C2Retriever を 2 つ (primary + readonly) 追加 (3 retrievers)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import C2Retriever, LocalRetriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8101")           # dev C2
        monkeypatch.setenv("L2_C2_URL_READONLY", "http://localhost:8100")  # prod C2

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        # Local + C2 primary (dev) + C2 readonly (prod) の 3 個
        assert len(retriever._retrievers) == 3
        assert isinstance(retriever._retrievers[0], LocalRetriever)
        assert isinstance(retriever._retrievers[1], C2Retriever)
        assert retriever._retrievers[1]._base_url == "http://localhost:8101"
        assert isinstance(retriever._retrievers[2], C2Retriever)
        assert retriever._retrievers[2]._base_url == "http://localhost:8100"

    def test_readonly_url_empty_string_ignored(self, monkeypatch, fake_index):
        """L2_C2_URL_READONLY='' (空文字列) は未設定扱い。"""
        from lab_lounge.graph import _build_retriever_from_env

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")
        monkeypatch.setenv("L2_C2_URL_READONLY", "")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        # 空文字は無視されるので 2 retrievers
        assert len(retriever._retrievers) == 2

    def test_readonly_url_whitespace_only_ignored(self, monkeypatch, fake_index):
        """L2_C2_URL_READONLY='   ' (空白のみ) は未設定扱い (strip)。"""
        from lab_lounge.graph import _build_retriever_from_env

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")
        monkeypatch.setenv("L2_C2_URL_READONLY", "   ")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index)

        assert len(retriever._retrievers) == 2


# ═══════════════════════════════════════════════════════════════════
# RecentC2Retriever 統合 (Block 5 / TD-1)
# ═══════════════════════════════════════════════════════════════════


class TestRAGPipelineWithRecentRetriever:
    """L2_USE_C2_RECENT と L2_C2_RECENT_SCOPE の組み合わせで RecentC2Retriever が
    CompositeRetriever に組み込まれることを検証。"""

    def test_recent_retriever_global_scope_default(
        self, monkeypatch, fake_index
    ):
        """default (L2_C2_RECENT_SCOPE 未設定) は global scope で stream_id=None。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import (
            C2Retriever, LocalRetriever, RecentC2Retriever,
        )

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8100")
        monkeypatch.delenv("L2_USE_C2_RECENT", raising=False)  # default=true
        monkeypatch.delenv("L2_C2_RECENT_SCOPE", raising=False)  # default=global
        monkeypatch.delenv("L2_C2_URL_READONLY", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(
                fake_index, stream_id="stream-test-123"
            )

        # Local + C2 primary + Recent C2 (global scope) の 3 個
        assert len(retriever._retrievers) == 3
        assert isinstance(retriever._retrievers[0], LocalRetriever)
        assert isinstance(retriever._retrievers[1], C2Retriever)
        assert isinstance(retriever._retrievers[2], RecentC2Retriever)
        # global scope → stream_id は無視されて None
        assert retriever._retrievers[2]._stream_id is None

    def test_recent_retriever_session_scope_uses_stream_id(
        self, monkeypatch, fake_index
    ):
        """L2_C2_RECENT_SCOPE=session で stream_id が Recent に渡される。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_RECENT_SCOPE", "session")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(
                fake_index, stream_id="stream-test-123"
            )

        # Recent C2 が追加され、stream_id が反映されている
        recent_retrievers = [
            r for r in retriever._retrievers if isinstance(r, RecentC2Retriever)
        ]
        assert len(recent_retrievers) == 1
        assert recent_retrievers[0]._stream_id == "stream-test-123"

    def test_recent_retriever_disabled_when_env_false(self, monkeypatch, fake_index):
        """L2_USE_C2_RECENT=false で RecentC2Retriever が追加されない。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_USE_C2_RECENT", "false")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(
                fake_index, stream_id="stream-test-123"
            )

        # Local + C2 primary の 2 個のみ (Recent は追加されない)
        assert len(retriever._retrievers) == 2
        assert not any(isinstance(r, RecentC2Retriever) for r in retriever._retrievers)

    def test_recent_retriever_global_scope_works_without_stream_id(
        self, monkeypatch, fake_index
    ):
        """global scope では stream_id=None でも RecentC2Retriever が追加される
        (セッションまたぎメモリの核心)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.delenv("L2_USE_C2_RECENT", raising=False)
        monkeypatch.delenv("L2_C2_RECENT_SCOPE", raising=False)  # default=global

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            # stream_id=None で呼ぶ (新しい run_loop の Turn 1 相当)
            retriever = _build_retriever_from_env(fake_index, stream_id=None)

        # global scope では stream_id 不要なので Recent が追加される
        recent_retrievers = [
            r for r in retriever._retrievers if isinstance(r, RecentC2Retriever)
        ]
        assert len(recent_retrievers) == 1
        assert recent_retrievers[0]._stream_id is None

    def test_recent_retriever_session_scope_skipped_without_stream_id(
        self, monkeypatch, fake_index
    ):
        """session scope で stream_id=None のとき Recent は追加されない (scope 不成立)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_RECENT_SCOPE", "session")

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(fake_index, stream_id=None)

        # session scope かつ stream_id なしなので Recent は追加されない
        assert not any(
            isinstance(r, RecentC2Retriever) for r in retriever._retrievers
        )

    def test_unknown_scope_falls_back_to_global(self, monkeypatch, fake_index, caplog):
        """不正な L2_C2_RECENT_SCOPE は warning 出して global にフォールバック。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_RECENT_SCOPE", "invalid-value")

        import logging
        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            with caplog.at_level(logging.WARNING, logger="lab_lounge.graph"):
                retriever = _build_retriever_from_env(
                    fake_index, stream_id="stream-1"
                )

        # warning が出ている
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("L2_C2_RECENT_SCOPE" in m for m in warning_msgs)

        # global にフォールバックしているので stream_id は None
        recent_retrievers = [
            r for r in retriever._retrievers if isinstance(r, RecentC2Retriever)
        ]
        assert len(recent_retrievers) == 1
        assert recent_retrievers[0]._stream_id is None

    def test_recent_retriever_not_added_for_readonly_c2(
        self, monkeypatch, fake_index
    ):
        """readonly C2 に対しては Recent を追加しない (dev stream_id は prod に存在しない)。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import (
            C2Retriever, RecentC2Retriever,
        )

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8101")
        monkeypatch.setenv("L2_C2_URL_READONLY", "http://localhost:8100")
        monkeypatch.delenv("L2_USE_C2_RECENT", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(
                fake_index, stream_id="dev-stream-1"
            )

        # Local + C2(primary=dev) + Recent(primary=dev) + C2(readonly=prod) の 4 個
        assert len(retriever._retrievers) == 4
        # Recent は 1 つだけ (primary のみ)
        recent_retrievers = [
            r for r in retriever._retrievers if isinstance(r, RecentC2Retriever)
        ]
        assert len(recent_retrievers) == 1
        assert recent_retrievers[0]._base_url == "http://localhost:8101"

    def test_exclude_event_ids_propagates_to_all_c2_retrievers(
        self, monkeypatch, fake_index
    ):
        """exclude_event_ids が全ての C2Retriever/RecentC2Retriever に伝播する。"""
        from lab_lounge.graph import _build_retriever_from_env
        from lab_lounge.retriever import C2Retriever, RecentC2Retriever

        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        monkeypatch.setenv("L2_C2_URL", "http://localhost:8101")
        monkeypatch.setenv("L2_C2_URL_READONLY", "http://localhost:8100")
        monkeypatch.delenv("L2_USE_C2_RECENT", raising=False)

        with patch("lab_lounge.retriever._embed", return_value=[1.0, 0.0]):
            retriever = _build_retriever_from_env(
                fake_index,
                stream_id="stream-1",
                exclude_event_ids=["uuid-current-turn"],
            )

        for r in retriever._retrievers:
            if isinstance(r, (C2Retriever, RecentC2Retriever)):
                assert r._exclude_event_ids == ["uuid-current-turn"]
