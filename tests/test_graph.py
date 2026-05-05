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

    def test_graph_has_three_nodes(self):
        """Sprint Axis D Block 3: retrieval ノード削除後は 3 ノード。"""
        from lab_lounge.graph import _build_pipeline_graph
        graph = _build_pipeline_graph()
        node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert node_names == {"routing", "generation", "tts"}

    def test_graph_has_correct_edge_count(self):
        from lab_lounge.graph import _build_pipeline_graph
        graph = _build_pipeline_graph()
        edges = graph.get_graph().edges
        # __start__→routing, routing→generation,
        # generation→tts, tts→__end__ = 4 edges
        assert len(edges) == 4


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
            stream_context=None,
            on_tts_chunk_ready=None,
            on_pose_ready=None,
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

    # Sprint Axis D Block 3: _retrieval_node は削除されたためテストも削除
    # RAG 検索は _generation_node 内の Agent がツールとして自律呼び出しする

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

    def test_tts_node_calls_set_pose_with_pose_from_json(
        self, mock_publish, base_state
    ):
        """LLM JSON に pose=happy が含まれれば set_pose が happy で呼ばれる。"""
        from lab_lounge.graph import _tts_node
        base_state["character_slug"] = "mimi"
        base_state["tts_speaker"] = "mimi"
        base_state["llm_text"] = (
            '{"emotion": {"happy": 80}, "speed": 100, "pose": "happy",'
            ' "response": "うれしいですわ"}'
        )
        base_state["events"] = [
            {"event_id": "utt-1", "type": "utterance.final"},
            {"event_id": "llm-1", "type": "llm.final"},
        ]
        with patch("lab_lounge.obs.set_pose") as mock_set_pose:
            _tts_node(base_state)
        mock_set_pose.assert_called_once_with("mimi", "happy")

    def test_tts_node_calls_set_pose_neutral_when_json_missing_pose(
        self, mock_publish, base_state
    ):
        """LLM JSON に pose フィールドがなければ set_pose が neutral で呼ばれる。"""
        from lab_lounge.graph import _tts_node
        base_state["character_slug"] = "mimi"
        base_state["tts_speaker"] = "mimi"
        base_state["llm_text"] = '{"emotion": {"happy": 50}, "response": "テスト"}'
        base_state["events"] = [
            {"event_id": "utt-1", "type": "utterance.final"},
            {"event_id": "llm-1", "type": "llm.final"},
        ]
        with patch("lab_lounge.obs.set_pose") as mock_set_pose:
            _tts_node(base_state)
        mock_set_pose.assert_called_once_with("mimi", "neutral")

    def test_tts_node_calls_set_pose_neutral_for_non_json_text(
        self, mock_publish, base_state
    ):
        """ダミーモード（非 JSON テキスト）でも neutral で set_pose が呼ばれる。"""
        from lab_lounge.graph import _tts_node
        base_state["character_slug"] = "octamaid"
        base_state["tts_speaker"] = "octamaid"
        base_state["llm_text"] = "ダミー応答: テスト"
        base_state["events"] = [
            {"event_id": "utt-1", "type": "utterance.final"},
            {"event_id": "llm-1", "type": "llm.final"},
        ]
        with patch("lab_lounge.obs.set_pose") as mock_set_pose:
            _tts_node(base_state)
        mock_set_pose.assert_called_once_with("octamaid", "neutral")

    def test_tts_node_does_not_publish_pose_update_event(
        self, mock_publish, base_state
    ):
        """pose.update Redis イベントは発行されない (L2 単独で OBS 制御するため)。"""
        from lab_lounge.graph import _tts_node
        base_state["character_slug"] = "mimi"
        base_state["tts_speaker"] = "mimi"
        base_state["llm_text"] = '{"pose": "happy", "response": "テスト"}'
        base_state["events"] = [
            {"event_id": "utt-1", "type": "utterance.final"},
            {"event_id": "llm-1", "type": "llm.final"},
        ]
        with patch("lab_lounge.obs.set_pose"):
            _tts_node(base_state)
        # publish 呼び出しに pose.update が含まれないことを確認
        published_types = [
            call.args[0].get("type") for call in mock_publish.call_args_list
        ]
        assert "pose.update" not in published_types


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
            "stream_context": None,
            "on_tts_chunk_ready": None,
            "on_pose_ready": None,
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
            "stream_context": None,
            "on_tts_chunk_ready": None,
            "on_pose_ready": None,
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


# ─────────────────────────────────────────────────────────────────
# Stream context (今日の配信内容) の結合テスト
# ─────────────────────────────────────────────────────────────────


class TestComposeSystemPrompt:
    """_compose_system_prompt のユニットテスト。

    キャラ素体プロンプト + 配信文脈 → 拡張 system_prompt の組み立てを検証する。
    """

    def test_no_stream_context_returns_character_prompt_unchanged(self):
        """配信文脈が None なら キャラ素体をそのまま返す (後方互換)。"""
        from lab_lounge.graph import _compose_system_prompt
        assert _compose_system_prompt("キャラ素体", None) == "キャラ素体"

    def test_empty_stream_context_returns_character_prompt_unchanged(self):
        """空文字列の配信文脈も結合しない (空セクションで LLM を惑わせない)。"""
        from lab_lounge.graph import _compose_system_prompt
        assert _compose_system_prompt("キャラ素体", "") == "キャラ素体"

    def test_merges_with_separator_and_heading(self):
        """配信文脈ありなら "## 本日の配信" 見出しで結合される。"""
        from lab_lounge.graph import _compose_system_prompt
        result = _compose_system_prompt("キャラ素体", "今日は大神プレイ")
        assert result is not None
        assert "キャラ素体" in result
        assert "## 本日の配信" in result
        assert "今日は大神プレイ" in result
        # 順序: キャラ素体 → 区切り → 配信文脈
        assert result.index("キャラ素体") < result.index("## 本日の配信")
        assert result.index("## 本日の配信") < result.index("今日は大神プレイ")
        # 既存 "## 参照情報" と同じ "\n\n---\n\n" セパレータを使う
        assert "\n\n---\n\n" in result

    def test_no_character_prompt_returns_stream_context_with_heading(self):
        """キャラ素体が None でも配信文脈だけは見出し付きで返す
        (FileNotFoundError 等のフォールバック動作)。"""
        from lab_lounge.graph import _compose_system_prompt
        result = _compose_system_prompt(None, "今日の配信内容")
        assert result is not None
        assert result.startswith("## 本日の配信")
        assert "今日の配信内容" in result

    def test_both_none_returns_none(self):
        """両方 None なら None (システムプロンプトなしで実行)。"""
        from lab_lounge.graph import _compose_system_prompt
        assert _compose_system_prompt(None, None) is None


class TestRoutingNodeStreamContextMerge:
    """routing ノードが state["stream_context"] を system_prompt にマージする検証。"""

    @pytest.fixture()
    def mock_publish(self):
        with patch("lab_lounge.pipeline.publish", return_value="1-0"):
            yield

    def _make_state(self, stream_context: str | None):
        """最小 PipelineGraphState を組み立てる。"""
        from lab_lounge.graph import PipelineGraphState
        return PipelineGraphState(
            text="テスト",
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
            stream_context=stream_context,
            on_tts_chunk_ready=None,
            on_pose_ready=None,
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

    def test_routing_merges_stream_context_into_system_prompt(self, mock_publish):
        """routing 後の system_prompt にキャラ素体と配信文脈の両方が含まれる。"""
        from lab_lounge.graph import _routing_node
        state = self._make_state(stream_context="今日は1年ぶりの配信、大神をプレイ")
        result = _routing_node(state)

        merged = result["system_prompt"]
        assert merged is not None
        # オクタメイドのキャラ素体テキスト由来の語が含まれている
        assert "オクタメイド" in merged
        # 配信文脈見出しと本文が含まれている
        assert "## 本日の配信" in merged
        assert "今日は1年ぶりの配信、大神をプレイ" in merged

    def test_routing_without_stream_context_keeps_system_prompt_unchanged(
        self, mock_publish
    ):
        """配信文脈なし時は従来通りキャラ素体のみが system_prompt になる (後方互換)。"""
        from lab_lounge.graph import _routing_node
        state = self._make_state(stream_context=None)
        result = _routing_node(state)

        merged = result["system_prompt"]
        assert merged is not None
        assert "オクタメイド" in merged
        # 配信文脈見出しが入っていないこと (= キャラ素体のみ)
        assert "## 本日の配信" not in merged
