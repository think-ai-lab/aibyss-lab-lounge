"""
test_graph.py — graph.run_graph テスト

langgraph / LLM 呼び出しをモックして実 API なしで検証する。
langgraph がインストールされていなくても動作する。
"""

import builtins
import pytest
from unittest.mock import MagicMock, patch

from lab_lounge.llm import LLMResult


FAKE_RESULT = LLMResult(
    text="グラフ応答テスト",
    model="gpt-4o-mini",
    input_tokens=20,
    output_tokens=10,
    latency_ms=300,
    finish_reason="stop",
)


class TestRunGraph:
    def test_run_graph_returns_llm_result(self):
        """_build_simple_graph をモックして run_graph が LLMResult を返すことを確認する"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("テスト", model="gpt-4o-mini", provider="openai")

        assert isinstance(result, LLMResult)
        assert result.text == "グラフ応答テスト"

    def test_run_graph_invoke_called_with_correct_state(self):
        """graph.invoke に正しい初期 state が渡る"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト入力",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("テスト入力", model="gpt-4o-mini", provider="openai")

        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["text"] == "テスト入力"
        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["provider"] == "openai"
        assert call_args["result"] is None

    def test_run_graph_returns_correct_text(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("x", model="m")

        assert result.text == "グラフ応答テスト"
        assert result.model == "gpt-4o-mini"

    def test_run_graph_raises_if_result_is_none(self):
        """Graph が result=None を返した場合は RuntimeError"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "テスト",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "result": None,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            with pytest.raises(RuntimeError):
                run_graph("テスト", model="gpt-4o-mini")

    def test_build_simple_graph_raises_import_error_without_langgraph(self):
        """langgraph が未インストールの場合、ImportError を送出する"""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langgraph" in name:
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        from lab_lounge.graph import _build_simple_graph
        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="langgraph"):
                _build_simple_graph()


class TestRunGraphWithMetadata:
    """run_metadata を渡したとき config["metadata"] に反映されるテスト"""

    def test_run_metadata_passed_to_invoke_config(self):
        """run_metadata が graph.invoke の config["metadata"] に渡ること"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }
        meta = {"aibyss.trace_id": "t1", "aibyss.stream_id": "s1"}

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("x", model="m", run_metadata=meta)

        positional_args = mock_graph.invoke.call_args[0]
        config_arg = positional_args[1]
        assert config_arg == {"metadata": meta}

    def test_no_run_metadata_passes_none_config(self):
        """run_metadata=None のとき config は None で invoke されること"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            run_graph("x", model="m")

        positional_args = mock_graph.invoke.call_args[0]
        config_arg = positional_args[1]
        assert config_arg is None

    def test_run_metadata_does_not_affect_result(self):
        """run_metadata を渡しても LLMResult の内容は変わらないこと"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "text": "x",
            "model": "m",
            "provider": "openai",
            "result": FAKE_RESULT,
        }
        meta = {"aibyss.trace_id": "t1"}

        with patch("lab_lounge.graph._build_simple_graph", return_value=mock_graph):
            from lab_lounge.graph import run_graph
            result = run_graph("x", model="m", run_metadata=meta)

        assert result.text == FAKE_RESULT.text
        assert result.model == FAKE_RESULT.model


# ═══════════════════════════════════════════════════════════════════
# Pipeline Graph (マルチノード) テスト
# ═══════════════════════════════════════════════════════════════════


class TestBuildPipelineGraph:
    """_build_pipeline_graph のグラフ構造を検証する。"""

    def test_graph_has_four_nodes(self):
        from lab_lounge.graph import _build_pipeline_graph
        graph = _build_pipeline_graph()
        node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert node_names == {"routing", "retrieval", "generation", "tts"}

    def test_graph_has_correct_edge_count(self):
        from lab_lounge.graph import _build_pipeline_graph
        graph = _build_pipeline_graph()
        edges = graph.get_graph().edges
        # __start__→routing, routing→retrieval, retrieval→generation,
        # generation→tts, tts→__end__ = 5 edges
        assert len(edges) == 5


class TestPipelineGraphNodes:
    """各ノード関数の単体テスト。"""

    @pytest.fixture()
    def mock_publish(self):
        with patch("lab_lounge.pipeline.publish", return_value="1-0") as m:
            yield m

    @pytest.fixture()
    def base_state(self):
        """ノードテスト用のベース状態。"""
        from lab_lounge.graph import PipelineGraphState
        return PipelineGraphState(
            text="テスト入力",
            common=dict(stream_id="s1", session_id="ss1", trace_id="t1"),
            speaker_hint=None,
            utterance_meta=None,
            use_real_llm=False,
            llm_provider="openai",
            llm_model="gpt-5.4-mini",
            enable_rag=False,
            rag_top_k=3,
            kb_path="./data/index",
            use_real_tts=False,
            tts_provider="voicevox",
            tts_voice="89",
            tts_speaker="octamaid",
            tts_output_dir="./data/audio",
            system_prompt=None,
            on_tts_chunk_ready=None,
            character_slug="",
            rag_context=None,
            rag_used=False,
            retrieved_doc_ids=[],
            retrieval_latency_ms=0,
            answer_mode="fallback",
            llm_text="",
            llm_meta={},
            tts_meta={},
            events=[],
        )

    def test_routing_node_sets_character_slug(self, mock_publish, base_state):
        from lab_lounge.graph import _routing_node
        result = _routing_node(base_state)
        assert result["character_slug"] == "octamaid"

    def test_routing_node_emits_utterance_event(self, mock_publish, base_state):
        from lab_lounge.graph import _routing_node
        result = _routing_node(base_state)
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "utterance.final"

    def test_retrieval_node_skips_when_disabled(self, mock_publish, base_state):
        from lab_lounge.graph import _retrieval_node
        # routing 済みの状態をシミュレート
        base_state["character_slug"] = "octamaid"
        base_state["events"] = [{"event_id": "utt-1", "type": "utterance.final"}]
        result = _retrieval_node(base_state)
        assert result["rag_used"] is False
        assert result["answer_mode"] == "fallback"

    def test_generation_node_dummy_mode(self, mock_publish, base_state):
        from lab_lounge.graph import _generation_node
        base_state["character_slug"] = "octamaid"
        base_state["events"] = [{"event_id": "utt-1", "type": "utterance.final"}]
        result = _generation_node(base_state)
        assert result["llm_text"] == "ダミー応答: テスト入力"
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "llm.final"

    def test_tts_node_dummy_mode(self, mock_publish, base_state):
        from lab_lounge.graph import _tts_node
        base_state["character_slug"] = "octamaid"
        base_state["tts_speaker"] = "octamaid"
        base_state["llm_text"] = "ダミー応答: テスト入力"
        base_state["events"] = [
            {"event_id": "utt-1", "type": "utterance.final"},
            {"event_id": "llm-1", "type": "llm.final"},
        ]
        result = _tts_node(base_state)
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "tts.done"
        assert result["tts_meta"]["speaker"] == "octamaid"


class TestPipelineGraphFullInvoke:
    """run_pipeline_graph の統合テスト（ダミーモード）。"""

    @pytest.fixture()
    def mock_publish(self):
        with patch("lab_lounge.pipeline.publish", return_value="1-0"):
            yield

    def test_full_invoke_returns_three_events(self, mock_publish):
        from lab_lounge.graph import run_pipeline_graph, PipelineGraphState
        state: PipelineGraphState = {
            "text": "統合テスト",
            "common": dict(stream_id="s1", session_id="ss1", trace_id="t1"),
            "speaker_hint": None,
            "utterance_meta": None,
            "use_real_llm": False,
            "llm_provider": "openai",
            "llm_model": "gpt-5.4-mini",
            "enable_rag": False,
            "rag_top_k": 3,
            "kb_path": "./data/index",
            "use_real_tts": False,
            "tts_provider": "voicevox",
            "tts_voice": "89",
            "tts_speaker": "octamaid",
            "tts_output_dir": "./data/audio",
            "system_prompt": None,
            "on_tts_chunk_ready": None,
            "character_slug": "",
            "rag_context": None,
            "rag_used": False,
            "retrieved_doc_ids": [],
            "retrieval_latency_ms": 0,
            "answer_mode": "fallback",
            "llm_text": "",
            "llm_meta": {},
            "tts_meta": {},
            "events": [],
        }
        final = run_pipeline_graph(state)
        assert len(final["events"]) == 3
        types = [e["type"] for e in final["events"]]
        assert types == ["utterance.final", "llm.final", "tts.done"]

    def test_full_invoke_sets_character_slug(self, mock_publish):
        from lab_lounge.graph import run_pipeline_graph, PipelineGraphState
        state: PipelineGraphState = {
            "text": "テスト",
            "common": dict(stream_id="s1", session_id="ss1", trace_id="t1"),
            "speaker_hint": None,
            "utterance_meta": None,
            "use_real_llm": False,
            "llm_provider": "openai",
            "llm_model": "gpt-5.4-mini",
            "enable_rag": False,
            "rag_top_k": 3,
            "kb_path": "./data/index",
            "use_real_tts": False,
            "tts_provider": "voicevox",
            "tts_voice": "89",
            "tts_speaker": "octamaid",
            "tts_output_dir": "./data/audio",
            "system_prompt": None,
            "on_tts_chunk_ready": None,
            "character_slug": "",
            "rag_context": None,
            "rag_used": False,
            "retrieved_doc_ids": [],
            "retrieval_latency_ms": 0,
            "answer_mode": "fallback",
            "llm_text": "",
            "llm_meta": {},
            "tts_meta": {},
            "events": [],
        }
        final = run_pipeline_graph(state)
        assert final["character_slug"] == "octamaid"

    def test_build_pipeline_graph_raises_import_error_without_langgraph(self):
        """langgraph が未インストールの場合、ImportError を送出する。"""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langgraph" in name:
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        from lab_lounge.graph import _build_pipeline_graph
        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                _build_pipeline_graph()
