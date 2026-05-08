"""
test_graph_agent.py — Agent 化した graph のテスト
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lab_lounge.graph import (
    _is_tools_enabled,
    run_graph,
)
from lab_lounge.llm import LLMResult


class TestToolsEnabled:
    """ツールモードの有効/無効テスト。"""

    def test_default_disabled(self):
        assert _is_tools_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_ENABLE_TOOLS", "true")
        assert _is_tools_enabled() is True

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("L2_ENABLE_TOOLS", "false")
        assert _is_tools_enabled() is False


class TestRunGraphSimpleMode:
    """従来互換の単一ノードモード（ツール無効）。"""

    def test_simple_mode_returns_llm_result(self, monkeypatch):
        """ツール無効時は従来の単一ノード構成で動作する。"""
        mock_result = LLMResult(
            text="テスト応答",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph.call_llm", return_value=mock_result):
            result = run_graph(
                "テスト入力",
                model="test-model",
                provider="openai",
            )

        assert result.text == "テスト応答"
        assert result.model == "test-model"

    def test_simple_mode_passes_system_prompt(self, monkeypatch):
        """システムプロンプトが LLM に渡される。"""
        mock_result = LLMResult(
            text="応答",
            model="m",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph.call_llm", return_value=mock_result) as mock_llm:
            run_graph(
                "入力",
                model="m",
                provider="openai",
                system_prompt="テストプロンプト",
            )

        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("system_prompt") == "テストプロンプト"


class TestRunGraphAgentMode:
    """Agent モード（ツール有効）。"""

    def test_agent_mode_with_no_tools_falls_back(self, monkeypatch):
        """ツール読み込み失敗時は単一ノードにフォールバック。"""
        monkeypatch.setenv("L2_ENABLE_TOOLS", "true")

        mock_result = LLMResult(
            text="フォールバック応答",
            model="test",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            finish_reason="stop",
        )

        with patch("lab_lounge.graph._load_mcp_tools", return_value=[]), \
             patch("lab_lounge.graph.call_llm", return_value=mock_result):
            result = run_graph(
                "テスト",
                model="test",
                provider="openai",
            )

        assert result.text == "フォールバック応答"


# ═══════════════════════════════════════════════════════════════════
# Sprint Axis D Block 3: tool_routing ガイダンス + BubbleToolCallbackHandler
# ═══════════════════════════════════════════════════════════════════
#
# TD-2 の context wrapper テストは削除。Agent 化により RAG context は
# LLM が自分で retrieve_memory ツールを呼んで取得するため、事前注入が不要に。
# 代わりに tool_routing ガイダンスの存在と BubbleToolCallbackHandler の
# 動作を検証する。


class TestToolRoutingGuidance:
    """Skills 定義ファイルベースのガイダンス生成を検証する。

    Sprint Axis D Block 4: _TOOL_ROUTING_GUIDANCE ハードコードを
    skills/common/tool_routing.md に移行。
    """

    def test_skills_contain_retrieve_memory(self):
        """Skills プロンプトに retrieve_memory ガイダンスが含まれる。"""
        from lab_lounge.skill_loader import build_skills_prompt
        result = build_skills_prompt("mimi")
        assert "retrieve_memory" in result
        assert "web_search" in result

    def test_skills_encourage_retrieve_memory_when_unsure(self):
        """「迷ったら retrieve_memory」の指示が Skills に含まれる。"""
        from lab_lounge.skill_loader import build_skills_prompt
        result = build_skills_prompt("mimi")
        assert "迷った" in result


class TestBubbleToolCallbackHandler:
    """BubbleToolCallbackHandler のツール呼び出し時 bubble 発行を検証する。"""

    def test_retrieve_memory_publishes_searching_bubble(self):
        """retrieve_memory_tool → bubble(searching) が発行される。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("mimi", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "retrieve_memory_tool"}, "test query")

        assert len(published) == 1
        assert published[0]["payload"]["step"] == "searching"
        assert published[0]["payload"]["character"] == "mimi"

    def test_web_search_publishes_searching_bubble(self):
        """web_search_tool → bubble(searching) が発行される (text は web_search キー)。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("chisame", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "web_search_tool"}, "test query")

        assert len(published) == 1
        assert published[0]["payload"]["step"] == "searching"

    def test_unknown_tool_does_not_publish(self):
        """未知のツール名では bubble を発行しない。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        published: list[dict] = []

        handler = BubbleToolCallbackHandler("mimi", {
            "stream_id": "s1", "session_id": "ss1", "trace_id": "t1",
        })

        with patch("lab_lounge.bus.publish", side_effect=lambda e: published.append(e)):
            handler.on_tool_start({"name": "unknown_tool"}, "test")

        assert len(published) == 0

    # ─── Phase 0.5-B-α: status_manager 連携テスト ─────────────

    def test_on_tool_start_sets_tool_calling(self):
        """on_tool_start で status_manager に TOOL_CALLING が反映される。"""
        from lab_lounge.character_status import CharacterStatus, CharacterStatusManager
        from lab_lounge.graph import BubbleToolCallbackHandler

        manager = CharacterStatusManager()
        handler = BubbleToolCallbackHandler(
            "mimi",
            {"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
            status_manager=manager,
        )

        with patch("lab_lounge.bus.publish", side_effect=lambda e: None):
            handler.on_tool_start({"name": "retrieve_memory_tool"}, "test")

        assert manager.get_status("mimi") == CharacterStatus.TOOL_CALLING

    def test_on_tool_end_sets_thinking_back(self):
        """on_tool_end で status_manager に THINKING が反映される (= TOOL_CALLING → THINKING)。"""
        from lab_lounge.character_status import CharacterStatus, CharacterStatusManager
        from lab_lounge.graph import BubbleToolCallbackHandler

        manager = CharacterStatusManager()
        manager.set_status("mimi", CharacterStatus.TOOL_CALLING)

        handler = BubbleToolCallbackHandler(
            "mimi",
            {"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
            status_manager=manager,
        )
        # LangChain は引数構造が揺れるため *args/**kwargs を受ける設計。引数なしで呼べる
        handler.on_tool_end()

        assert manager.get_status("mimi") == CharacterStatus.THINKING

    def test_unknown_tool_no_status_change(self):
        """TOOL_MESSAGE_KEY に無いツールでは status 変化なし (= bubble.update と同様の判定)。"""
        from lab_lounge.character_status import CharacterStatus, CharacterStatusManager
        from lab_lounge.graph import BubbleToolCallbackHandler

        manager = CharacterStatusManager()
        manager.set_status("mimi", CharacterStatus.THINKING)

        handler = BubbleToolCallbackHandler(
            "mimi",
            {"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
            status_manager=manager,
        )

        with patch("lab_lounge.bus.publish", side_effect=lambda e: None):
            handler.on_tool_start({"name": "unknown_tool"}, "test")

        # status は THINKING のまま (= TOOL_CALLING に変化しない)
        assert manager.get_status("mimi") == CharacterStatus.THINKING

    def test_status_manager_none_no_op(self):
        """status_manager=None で例外なく動く (= 既存呼出経路の互換性確認)。"""
        from lab_lounge.graph import BubbleToolCallbackHandler

        handler = BubbleToolCallbackHandler(
            "mimi",
            {"stream_id": "s1", "session_id": "ss1", "trace_id": "t1"},
        )  # status_manager=None (default)

        with patch("lab_lounge.bus.publish", side_effect=lambda e: None):
            handler.on_tool_start({"name": "retrieve_memory_tool"}, "test")
            handler.on_tool_end()
        # 例外なく完了


class TestAskCharacterToolRegistration:
    """ask_character ツール登録を検証する (Phase 3)。"""

    def test_ask_character_registered(self):
        """ask_character_tool がツールリストに含まれる。"""
        from lab_lounge.graph import _load_mcp_tools
        tools = _load_mcp_tools()
        tool_names = {t.name for t in tools}
        assert "ask_character_tool" in tool_names

    def test_bubble_handler_has_ask_character(self):
        """BubbleToolCallbackHandler に ask_character のマッピングがある。"""
        from lab_lounge.graph import BubbleToolCallbackHandler
        assert "ask_character_tool" in BubbleToolCallbackHandler.TOOL_MESSAGE_KEY


class TestRetrieveMemoryToolRegistration:
    """L2_ENABLE_RAG による retrieve_memory ツール登録の制御を検証する。"""

    def test_rag_enabled_registers_retrieve_memory(self, monkeypatch):
        """L2_ENABLE_RAG=true で retrieve_memory_tool が登録される。"""
        monkeypatch.setenv("L2_ENABLE_RAG", "true")
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is True

    def test_rag_disabled_does_not_register(self, monkeypatch):
        """L2_ENABLE_RAG=false で retrieve_memory は登録されない。"""
        monkeypatch.setenv("L2_ENABLE_RAG", "false")
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is False

    def test_rag_default_is_disabled(self, monkeypatch):
        """L2_ENABLE_RAG 未設定ではデフォルト false。"""
        monkeypatch.delenv("L2_ENABLE_RAG", raising=False)
        from lab_lounge.graph import _is_rag_enabled
        assert _is_rag_enabled() is False


class TestGetCharacterResponseSchema:
    """Phase 0.5-A フェーズ 8: キャラ別 Pydantic スキーマ動的生成。

    create_agent の response_format に渡すことで、LLM が strict JSON のみを
    返すようにする。VOICEPEAK CLI の引数破壊バグの根本対策。
    """

    def test_mimi_schema_has_5_emotion_keys(self):
        """mimi のスキーマは 5 つの emotion キー (happy/fun/angry/sad/sulky)。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("mimi")
        assert schema.__name__ == "MimiResponse"
        assert set(schema.model_fields.keys()) == {"response", "emotion", "speed", "pose"}
        emotion_schema = schema.model_fields["emotion"].annotation
        assert set(emotion_schema.model_fields.keys()) == {
            "happy", "fun", "angry", "sad", "sulky",
        }

    def test_chisame_schema_has_chisame_emotion_keys(self):
        """chisame のスキーマは bosoboso/doyaru/honwaka/angry/teary。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("chisame")
        assert schema.__name__ == "ChisameResponse"
        emotion_schema = schema.model_fields["emotion"].annotation
        assert set(emotion_schema.model_fields.keys()) == {
            "bosoboso", "doyaru", "honwaka", "angry", "teary",
        }

    def test_sakura_schema_has_sakura_emotion_keys(self):
        """sakura のスキーマは happy/sad/angry/whisper/cool。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("sakura")
        assert schema.__name__ == "SakuraResponse"
        emotion_schema = schema.model_fields["emotion"].annotation
        assert set(emotion_schema.model_fields.keys()) == {
            "happy", "sad", "angry", "whisper", "cool",
        }

    def test_octamaid_schema_has_no_emotion(self):
        """octamaid (voicepeak_emotion_keys 空) は emotion フィールド無し。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("octamaid")
        assert schema.__name__ == "OctamaidResponse"
        assert set(schema.model_fields.keys()) == {"response", "speed", "pose"}
        assert "emotion" not in schema.model_fields

    def test_ruka_schema_has_no_emotion(self):
        """ruka も emotion 無し (voicepeak_emotion_keys 空)。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("ruka")
        assert "emotion" not in schema.model_fields

    def test_unknown_slug_returns_generic_schema(self):
        """未登録 slug は GenericResponse でフォールバック (KeyError 吸収)。"""
        from lab_lounge.graph import _get_character_response_schema
        schema = _get_character_response_schema("nonexistent")
        assert schema.__name__ == "GenericResponse"
        assert set(schema.model_fields.keys()) == {"response", "speed", "pose"}

    def test_caches_same_class_for_same_slug(self):
        """@lru_cache で同一 slug は同一クラスを返す (LangChain schema 同一性)。"""
        from lab_lounge.graph import _get_character_response_schema
        s1 = _get_character_response_schema("mimi")
        s2 = _get_character_response_schema("mimi")
        assert s1 is s2

    def test_speed_validation(self):
        """speed フィールドは 50-200 の範囲制約付き。"""
        from lab_lounge.graph import _get_character_response_schema
        from pydantic import ValidationError
        Schema = _get_character_response_schema("mimi")
        # 範囲内 OK
        Schema(
            response="test",
            emotion={"happy": 50, "fun": 0, "angry": 0, "sad": 0, "sulky": 0},
            speed=100,
            pose="neutral",
        )
        # 範囲外 NG
        try:
            Schema(
                response="test",
                emotion={"happy": 50, "fun": 0, "angry": 0, "sad": 0, "sulky": 0},
                speed=500,  # 200 超過
                pose="neutral",
            )
            raise AssertionError("ValidationError が出るはず")
        except ValidationError:
            pass


class TestRunAgentStructuredResponse:
    """Phase 0.5-A フェーズ 8: _run_agent の structured_response 経路検証。

    create_agent (新 API) で response_format=Pydantic を有効化した場合、
    state['structured_response'] に Pydantic instance が入る。これを優先的に
    取得して LLMResult.text に JSON 文字列化することで、既存パイプライン
    (_tts_node の _parse_voicepeak_json) と互換性を維持する。
    """

    def test_structured_response_is_used_as_text(self):
        """state['structured_response'] が Pydantic ならそれを JSON 文字列化して text に。"""
        from lab_lounge.graph import _run_agent, _get_character_response_schema

        Schema = _get_character_response_schema("mimi")
        structured_instance = Schema(
            response="わたくしの見解ですわ",
            emotion={"happy": 50, "fun": 0, "angry": 0, "sad": 0, "sulky": 0},
            speed=100,
            pose="happy",
        )
        # Mock agent: invoke が structured_response 含む dict を返す
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "(無視される)"
        mock_msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [mock_msg],
            "structured_response": structured_instance,
        }

        result = _run_agent(mock_agent, "テスト", "mimi-model")

        # text が JSON 文字列化されている
        assert "わたくしの見解ですわ" in result.text
        assert '"response":' in result.text
        assert '"emotion":' in result.text
        # token usage は messages から取得
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_falls_back_to_messages_when_no_structured(self):
        """structured_response 無しなら messages の最後の AI メッセージを使う (旧経路互換)。"""
        from lab_lounge.graph import _run_agent

        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "メッセージから取得"
        mock_msg.usage_metadata = {}

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [mock_msg],
            # structured_response キー無し
        }

        result = _run_agent(mock_agent, "テスト", "mimi-model")

        assert result.text == "メッセージから取得"

    def test_falls_back_when_structured_serialize_fails(self):
        """structured_response があっても model_dump_json が失敗したらフォールバック。"""
        from lab_lounge.graph import _run_agent

        # model_dump_json も model_dump も無いオブジェクト
        broken_structured = MagicMock(spec=[])  # 属性なし

        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "フォールバック値"
        mock_msg.usage_metadata = {}

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [mock_msg],
            "structured_response": broken_structured,
        }

        result = _run_agent(mock_agent, "テスト", "mimi-model")

        # フォールバック先 (messages の最後の AI) が使われる
        assert result.text == "フォールバック値"


class TestBuildAgentGraphWithStructuredOutput:
    """Phase 0.5-A フェーズ 8: _build_agent_graph が response_format を渡す。"""

    def test_passes_response_format_when_character_slug_set(self, monkeypatch):
        """character_slug 指定時に create_agent の response_format に Pydantic スキーマが渡される。"""
        from lab_lounge import graph as graph_mod

        # _load_mcp_tools が空でない tool list を返すよう mock
        mock_tool = MagicMock()
        mock_tool.name = "fake_tool"
        # ログ強化 L-2 で _load_mcp_tools(character_slug=...) のシグネチャに変わったため
        # **kwargs を受け取れるようにする (mock の引数互換維持)
        monkeypatch.setattr(graph_mod, "_load_mcp_tools", lambda **_kw: [mock_tool])

        # _get_llm_for_agent も mock (実 LLM 呼出を避ける)
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        monkeypatch.setattr(graph_mod, "_get_llm_for_agent", lambda *a: mock_llm)

        # langchain.agents.create_agent を spy
        captured_kwargs = {}

        def spy_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(name="agent")

        with patch("langchain.agents.create_agent", spy_create_agent):
            graph_mod._build_agent_graph(
                provider="anthropic",
                model="claude-test",
                system_prompt="prompt",
                character_slug="mimi",
            )

        # response_format に Pydantic スキーマが渡されている
        assert "response_format" in captured_kwargs
        schema = captured_kwargs["response_format"]
        assert schema.__name__ == "MimiResponse"
        # system_prompt も渡される (旧 prompt= からのリネーム)
        assert "system_prompt" in captured_kwargs
        # tools も渡される
        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"] == [mock_tool]

    def test_no_response_format_when_character_slug_none(self, monkeypatch):
        """character_slug=None なら response_format は渡されない (旧経路互換)。"""
        from lab_lounge import graph as graph_mod

        mock_tool = MagicMock()
        mock_tool.name = "fake_tool"
        # ログ強化 L-2 で _load_mcp_tools(character_slug=...) のシグネチャに変わったため
        # **kwargs を受け取れるようにする (mock の引数互換維持)
        monkeypatch.setattr(graph_mod, "_load_mcp_tools", lambda **_kw: [mock_tool])
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        monkeypatch.setattr(graph_mod, "_get_llm_for_agent", lambda *a: mock_llm)

        captured_kwargs = {}

        def spy_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("langchain.agents.create_agent", spy_create_agent):
            graph_mod._build_agent_graph(
                provider="anthropic",
                model="claude-test",
                system_prompt="prompt",
                character_slug=None,
            )

        # response_format が渡されない
        assert "response_format" not in captured_kwargs
