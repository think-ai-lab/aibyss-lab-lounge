"""
pipeline.py — 開発用テキストパイプライン

責務:
  - 入力テキストから utterance.final → llm.final → tts.done の 3 イベントを
    順番に組み立て、Event Bus (Redis Streams) に publish する
  - 各イベントに links で因果関係を持たせる:
      llm.final.links   = [utterance.final.event_id]
      tts.done.links    = [llm.final.event_id]
  - publish 以外の副作用を持たない（C2 への HTTP 呼び出し禁止 Guardrail G-2）
  - STT 呼び出しは emitter.py 側の責務。
    音声から変換したテキスト + メタデータ (utterance_meta) を受け取るだけ。

【パイプライン処理フロー（RAG オン時）】
  utterance.final  (seq=0, links=なし)
       ↓ links=[utterance.event_id]
  [Retrieve: 知識ベース検索 — L2_ENABLE_RAG=true のときのみ実行]
       ↓ 失敗/タイムアウト時は non-RAG で継続（fallback）
  llm.final        (seq=1)  ← rag_used / answer_mode をメタデータに付与
       ↓ links=[llm.event_id]
  tts.done         (seq=2)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bus import publish
from .events import build_bubble_update

logger = logging.getLogger(__name__)

# ─── 吹き出しメッセージ ──────────────────────────────────────────

_BUBBLE_MESSAGES: dict | None = None


def _load_bubble_messages() -> dict:
    """data/bubble_messages.json を読み込む（キャッシュ付き）。"""
    global _BUBBLE_MESSAGES
    if _BUBBLE_MESSAGES is None:
        import json
        msg_path = Path(__file__).resolve().parent.parent.parent / "data" / "bubble_messages.json"
        if msg_path.is_file():
            _BUBBLE_MESSAGES = json.loads(msg_path.read_text(encoding="utf-8"))
        else:
            logger.warning("bubble_messages.json が見つかりません: %s", msg_path)
            _BUBBLE_MESSAGES = {}
    return _BUBBLE_MESSAGES


def _publish_bubble(
    step: str,
    character_slug: str,
    common: dict,
    links: list[str] | None = None,
    *,
    category: str = "speech",
) -> None:
    """bubble.update イベントを発行する。

    category はデフォルト "speech" (通常応答パス: thinking/searching/answering 等)。
    挙手系 (handraise/denied/lapsed/cancelled) を発行する経路は dispatcher が直接
    build_bubble_update を呼ぶため、本 helper は通常応答用に最適化する (Phase 0.5-A 8-10)。
    """
    messages = _load_bubble_messages()
    char_msgs = messages.get(character_slug, {})
    text = char_msgs.get(step, "")

    try:
        bubble = build_bubble_update(
            character=character_slug,
            step=step,
            text=text,
            links=links,
            category=category,
            **common,
        )
        publish(bubble)
        logger.info("bubble.update published: step=%s character=%s", step, character_slug)
    except Exception as exc:
        logger.warning("bubble.update 発行失敗: %s", exc)


# ─── LLM モード設定 ──────────────────────────────────────────────

def _get_llm_mode(character=None) -> tuple[bool, str, str]:
    """
    LLM 実行モードを返す。キャラクター設定があればそちらを優先。

    優先順位: キャラクター設定 > 環境変数 > デフォルト

    Returns:
        (use_real, provider, model)
          use_real: True なら graph.py 経由で real LLM を呼ぶ
          provider: "openai" など
          model:    モデル名
    """
    use_real = os.environ.get("L2_USE_REAL_LLM", "false").lower() in ("true", "1", "yes")

    if character and getattr(character, "llm_model", ""):
        provider = character.llm_provider
        model = character.llm_model
    else:
        provider = os.environ.get("L2_LLM_PROVIDER", "openai")
        model = os.environ.get("L2_LLM_MODEL", "gpt-5.4-mini")

    return use_real, provider, model


# ─── RAG モード設定 ──────────────────────────────────────────────

def _get_rag_mode() -> tuple[bool, int, str]:
    """
    環境変数から RAG 実行モードを読み取る。

    Returns:
        (enable_rag, top_k, kb_path)
          enable_rag: True なら知識ベース検索を実行する
          top_k:      取得する文書数
          kb_path:    index ファイルの配置パス
    """
    enable_rag = os.environ.get("L2_ENABLE_RAG", "false").lower() in ("true", "1", "yes")
    top_k = int(os.environ.get("L2_RAG_TOP_K", "5"))  # TD-7: 3→5 (4 Retriever 構成で top_k=3 は狭すぎた)
    kb_path = os.environ.get("L2_KB_PATH", "./data/index")
    return enable_rag, top_k, kb_path


# ─── TTS モード設定 ──────────────────────────────────────────────

def _get_tts_mode() -> tuple[bool, str, str, str, str]:
    """
    環境変数から TTS 実行モードを読み取る。

    Returns:
        (use_real, provider, voice, speaker, output_dir)
          use_real:   True なら tts.py 経由で real TTS を呼ぶ
          provider:   "edge_tts" など
          voice:      音声識別子 (provider 依存)
          speaker:    人可読スピーカー名（メタデータ用）
          output_dir: 音声ファイル保存先
    """
    use_real = os.environ.get("L2_USE_REAL_TTS", "false").lower() in ("true", "1", "yes")
    provider = os.environ.get("L2_TTS_PROVIDER", "voicevox")
    voice = os.environ.get("L2_TTS_VOICE", "89")
    speaker = os.environ.get("L2_TTS_SPEAKER", "Voidoll")
    output_dir = os.environ.get("L2_TTS_OUTPUT_DIR", "./data/audio")
    return use_real, provider, voice, speaker, output_dir


@dataclass
class PipelineResult:
    stream_id: str
    session_id: str
    trace_id: str
    speaker: str  # ルーティングされたキャラクター slug
    events: list[dict[str, Any]]  # publish した順に格納


def run_pipeline(
    text: str,
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    utterance_meta: dict[str, Any] | None = None,
    speaker_hint: str | None = None,
    on_tts_chunk_ready=None,
    on_pose_ready=None,
    stream_context: str | None = None,
    suppress_bubble_answering: bool = False,
    disable_tools: list[str] | None = None,
    status_manager=None,
) -> PipelineResult:
    """
    テキストを受け取り 3 イベントを publish する。

    LangGraph パイプライングラフ (routing → generation → tts) を使用。

    Args:
        text:            発話テキスト（utterance.final の payload.text）
        stream_id:       ストリーム識別子
        session_id:      セッション識別子
        trace_id:        トレース識別子
        utterance_meta:  STT 結果から抽出したメタデータ (optional)。
                         キー: confidence / lang / duration_ms / words
                         省略時は build_utterance_final() のデフォルト値を使う。
        speaker_hint:    ウェイクワード検知結果のキャラクター slug / 名前 (optional)。
                         Router に渡され、応答キャラクターを決定する。
        stream_context:  「今日の配信内容」Markdown 本文 (optional)。
                         run_loop 起動時に1度ロードされ、配信中の全ターンで
                         同じ値が渡される。routing ノードでキャラ素体に
                         "## 本日の配信" として重ねられる。
                         None なら配信文脈なしで動作 (後方互換)。
        suppress_bubble_answering: Phase 0.5-A フェーズ 7 で追加。True にすると
                         _generation_node 内の bubble.update("answering") 発行を
                         抑制する。挙手 BG 先行生成では承認時に run_loop が
                         TTS 開始時刻と同期して bubble を発行する設計のため、
                         graph 側の二重発行を避ける。デフォルト False で既存挙動。
        disable_tools:   Phase 0.5-A 案 W'-3 + バグ 3 修正 (案 A) で追加。
                         Agent から除外するツール名のリスト (例: ["ask_character"])。
                         fallback パス (= _approved_synthesize_fallback) で
                         ask_character ツールを除外し、並行する他キャラ TTS との
                         deadlock を回避する (logs/runs/run_loop_20260508_181051.log
                         で観察されたハングの対処)。デフォルト None で既存挙動。

    Returns:
        PipelineResult（publish 済みイベント一覧を含む）
    """
    return _run_pipeline_graph(
        text,
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        utterance_meta=utterance_meta,
        speaker_hint=speaker_hint,
        on_tts_chunk_ready=on_tts_chunk_ready,
        on_pose_ready=on_pose_ready,
        stream_context=stream_context,
        suppress_bubble_answering=suppress_bubble_answering,
        disable_tools=disable_tools,
        status_manager=status_manager,
    )


def _run_pipeline_graph(
    text: str,
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    utterance_meta: dict[str, Any] | None = None,
    speaker_hint: str | None = None,
    on_tts_chunk_ready=None,
    on_pose_ready=None,
    stream_context: str | None = None,
    suppress_bubble_answering: bool = False,
    disable_tools: list[str] | None = None,
    status_manager=None,
) -> PipelineResult:
    """LangGraph パイプライングラフ経由で実行する。"""
    from .graph import run_pipeline_graph, PipelineGraphState

    common = dict(stream_id=stream_id, session_id=session_id, trace_id=trace_id)

    # 設定読み取り (キャラクター未確定 → routing ノードがキャラ別に上書き)
    use_real_llm, llm_provider, llm_model = _get_llm_mode()
    enable_rag, rag_top_k, kb_path = _get_rag_mode()
    use_real_tts, tts_provider, tts_voice, tts_speaker, tts_output_dir = _get_tts_mode()

    initial_state: PipelineGraphState = {
        "text": text,
        "common": common,
        "speaker_hint": speaker_hint,
        "utterance_meta": utterance_meta,
        "use_real_llm": use_real_llm,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "enable_rag": enable_rag,
        "rag_top_k": rag_top_k,
        "kb_path": kb_path,
        "use_real_tts": use_real_tts,
        "tts_provider": tts_provider,
        "tts_voice": tts_voice,
        "tts_speaker": tts_speaker,
        "tts_output_dir": tts_output_dir,
        "system_prompt": None,
        "stream_context": stream_context,
        "on_tts_chunk_ready": on_tts_chunk_ready,
        "on_pose_ready": on_pose_ready,
        "suppress_bubble_answering": suppress_bubble_answering,
        "disable_tools": disable_tools,
        # Phase 0.5-B-α: status_manager は run_loop から透過渡し。
        # _generation_node が Thinking、BubbleToolCallbackHandler が ToolCalling を反映。
        "status_manager": status_manager,
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

    final_state = run_pipeline_graph(initial_state)

    return PipelineResult(
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        speaker=final_state["character_slug"],
        events=final_state["events"],
    )



# レガシーパイプライン (_run_pipeline_legacy) は Sprint Axis D で廃止。
# LangGraph は必須依存。パイプラインは _run_pipeline_graph のみ使用。


# ─── Phase 0.5-A 案 W'-1: LLM-only / TTS-only パイプライン ───────


def run_pipeline_llm_only(
    text: str,
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    utterance_meta: dict[str, Any] | None = None,
    speaker_hint: str | None = None,
    stream_context: str | None = None,
    status_manager=None,
    on_tts_chunk_ready=None,
    on_pose_ready=None,
) -> PipelineResult:
    """LLM のみ先行実行 (TTS ノード抜き、Phase 0.5-A 案 W'-1)。

    挙手 BG 先行生成で使う。承認確率に賭けて LLM だけ走らせ、TTS 合成は承認後に
    run_loop が ``run_pipeline_tts_only(llm_result)`` を呼ぶ設計。これにより
    却下/lapse 時に VOICEPEAK FIFO への投入を回避し、3 重発火時の VOICEPEAK
    競合を緩和する (= 案 W'-1 の核心)。

    【WHY: run_pipeline と分けた理由】
    既存 run_pipeline は引数 ``suppress_bubble_answering=True`` で挙動を切り替えて
    いたが、「TTS を skip するか否か」は graph 構造そのものに影響する設計判断
    (= TTS ノードが呼ばれるか否か) なので、引数フラグではなく独立 API にした方が
    責務が明確。既存 run_pipeline の挙動には一切手を入れない (= 通常応答ターン
    経路の回帰なし)。

    【WHY: 当初は callback 引数を絞っていた】(Phase 0.5-A 案 W'-1)
    LLM-only モードでは graph._tts_node が走らないため、最終応答 TTS の callback
    は意味を持たなかった。``suppress_bubble_answering`` は内部で True 固定
    (= BG 経路では承認時に run_loop が answering bubble を発行する設計、graph 側
    の二重発行を防ぐ)。

    【WHY: on_tts_chunk_ready / on_pose_ready を後追いで受けるようにした】
    (Phase 0.5-B-β-1)
    ask_character ツール経由の対話 TTS は graph._tts_node を経由せず、ツール内で
    直接 ``tts.synthesize(on_chunk_ready=...)`` を呼ぶ独立経路。BG LLM 経路でも
    callback を渡せば ask_character の対話 TTS が playback queue に届くようになる
    (= A1 修正、ミミ様導入セリフ + 協働応答 TTS が再生される)。最終応答 TTS は
    依然 graph 構造 (= _tts_node 不在) で suppress される (= 案 W'-1 不変)。
    pipeline.py:404-407 の設計者コメント「LLM-only モードでは ask_character ツール
    起動の余地がある」と整合。

    Args:
        on_tts_chunk_ready: ask_character 内の対話 TTS chunk 投入用 callback
                            (= run_loop の ``_on_tts_chunk``)。signature は
                            ``(url: str, chunk_text: str, is_last: bool, character: str) -> None``。
                            None (default) なら従来挙動 (= 対話 TTS スキップ)。
        on_pose_ready:      target キャラの pose 切替予約 callback
                            (= run_loop の ``_on_pose_ready``)。signature は
                            ``(slug: str, pose: str) -> None``。None なら従来挙動。

    Returns:
        PipelineResult: events に [utterance.final, llm.final] の 2 件を含む。
                        tts.done は含まれない。承認時に ``run_pipeline_tts_only``
                        にこの result を渡して TTS を実行する。
    """
    return _run_pipeline_graph_llm_only(
        text,
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        utterance_meta=utterance_meta,
        speaker_hint=speaker_hint,
        stream_context=stream_context,
        status_manager=status_manager,
        on_tts_chunk_ready=on_tts_chunk_ready,
        on_pose_ready=on_pose_ready,
    )


def _run_pipeline_graph_llm_only(
    text: str,
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    utterance_meta: dict[str, Any] | None = None,
    speaker_hint: str | None = None,
    stream_context: str | None = None,
    status_manager=None,
    on_tts_chunk_ready=None,
    on_pose_ready=None,
) -> PipelineResult:
    """LLM-only パイプライングラフ経由で実行する。

    Phase 0.5-B-β-1 で ``on_tts_chunk_ready`` / ``on_pose_ready`` を受け付け開始。
    本 commit (β-1-1) では signature のみ拡張し、initial_state への伝播は β-1-2 で
    実装する (= signature 拡張と内部 wiring を分けて段階的に検証)。
    """
    from .graph import run_pipeline_graph_llm_only, PipelineGraphState

    common = dict(stream_id=stream_id, session_id=session_id, trace_id=trace_id)

    use_real_llm, llm_provider, llm_model = _get_llm_mode()
    enable_rag, rag_top_k, kb_path = _get_rag_mode()
    # TTS 設定は initial_state 型整合のため埋める (= LLM-only モードでは _tts_node
    # が呼ばれないため値は使われない、dead value 容認)。
    use_real_tts, tts_provider, tts_voice, tts_speaker, tts_output_dir = _get_tts_mode()

    initial_state: PipelineGraphState = {
        "text": text,
        "common": common,
        "speaker_hint": speaker_hint,
        "utterance_meta": utterance_meta,
        "use_real_llm": use_real_llm,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "enable_rag": enable_rag,
        "rag_top_k": rag_top_k,
        "kb_path": kb_path,
        "use_real_tts": use_real_tts,
        "tts_provider": tts_provider,
        "tts_voice": tts_voice,
        "tts_speaker": tts_speaker,
        "tts_output_dir": tts_output_dir,
        "system_prompt": None,
        "stream_context": stream_context,
        # LLM-only モードでは callback は呼ばれない (= TTS ノード不在のため)
        "on_tts_chunk_ready": None,
        "on_pose_ready": None,
        # WHY: 承認時に run_loop が answering bubble を発行する設計のため、graph 側で
        # 二重発行しないよう内部固定で抑制する
        "suppress_bubble_answering": True,
        # disable_tools: LLM-only モードでは ask_character ツール起動の余地が
        # あるが、本セッションではデフォルト None で既存挙動 (= bg_result=ready
        # 経路では ask_character 使える)。fallback パスは別途 disable_tools を渡す。
        "disable_tools": None,
        # Phase 0.5-B-α: status_manager は run_loop の bg_runner から透過渡し。
        # BG LLM 経路でも _generation_node が Thinking 反映する。
        "status_manager": status_manager,
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

    final_state = run_pipeline_graph_llm_only(initial_state)

    return PipelineResult(
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        speaker=final_state["character_slug"],
        events=final_state["events"],
    )


def run_pipeline_tts_only(
    llm_result: PipelineResult,
    *,
    on_tts_chunk_ready=None,
    on_pose_ready=None,
) -> PipelineResult:
    """LLM 結果を再利用して TTS のみ実行 (Phase 0.5-A 案 W'-1)。

    initial_state に llm_result.events から utterance.final + llm.final を注入
    し、character_slug / llm_text / common (stream_id/session_id/trace_id) を
    再構築。TTS-only graph (= tts → END) を invoke することで _tts_node 内の
    全ロジック (pose 切替 / wait_bg / build_tts_done) を再利用する。

    【WHY: tts.synthesize 直接呼出ではなく graph 再利用】
    _tts_node 内には pose 切替 (_parse_voicepeak_json + set_pose) /
    ask_character 協働 TTS 完了待ち (wait_bg_tts_complete) /
    build_tts_done event publish などの周辺ロジックが詰まっている。
    tts.synthesize 直接呼出ではこれらを再実装する必要があり、コード重複と
    保守負荷が増える。グラフ再利用なら新規実装ゼロ。

    Args:
        llm_result:        run_pipeline_llm_only() の戻り値。events には
                           [utterance.final, llm.final] が入っている前提。
        on_tts_chunk_ready: TTS chunk 生成時 callback (= run_loop の playback queue
                           投入)
        on_pose_ready:     pose 予約 callback (= run_loop の _pending_poses 投入)

    Returns:
        PipelineResult: events に initial の 2 件 + tts.done の合計 3 件を含む
    """
    from .graph import run_pipeline_graph_tts_only, PipelineGraphState

    common = dict(
        stream_id=llm_result.stream_id,
        session_id=llm_result.session_id,
        trace_id=llm_result.trace_id,
    )

    # events から llm.final.text を抽出 (= _tts_node が state["llm_text"] として読む)。
    # WHY: _tts_node:1112 で llm_event_id = state["events"][1]["event_id"] と
    # 「2 番目の event」を期待しているため、events 配列を含めて events[1] が
    # llm.final になる構造を保つ必要がある。
    llm_text = ""
    for ev in llm_result.events:
        if ev.get("type") == "llm.final":
            llm_text = ev.get("payload", {}).get("text", "")
            break

    # WHY (バグ 1 修正、実走 logs/runs/run_loop_20260508_180426.log で発覚):
    # TTS 設定はキャラ別に異なる (= sakura: voicepeak Haruno Sora、chisame:
    # voicepeak Miyamai Moca、mimi: voicepeak Asumi Ririse、octamaid: voicevox
    # Voidoll)。環境変数の _get_tts_mode() デフォルト値 (= L2_TTS_PROVIDER=voicevox
    # / L2_TTS_VOICE=89 / L2_TTS_SPEAKER=Voidoll) をそのまま使うと、「sakura の
    # 応答が octamaid (Voidoll) voice で合成される」バグになる。実走で実際に
    # 「TTS 開始: speaker=Voidoll」が観察され、ルカが「オクタメイドが応答した」
    # と報告した症状の正体。通常応答パスの _routing_node:781-783 と同じく
    # character config から正しい設定を引いて initial_state に詰める。
    from .characters import get_character
    character = get_character(llm_result.speaker)
    use_real_tts, _, _, _, tts_output_dir = _get_tts_mode()

    initial_state: PipelineGraphState = {
        # text / utterance_meta は TTS-only では使われないが型整合のため埋める
        "text": "",
        "common": common,
        "speaker_hint": None,
        "utterance_meta": None,
        # LLM 設定は使われない
        "use_real_llm": False,
        "llm_provider": "",
        "llm_model": "",
        "enable_rag": False,
        "rag_top_k": 0,
        "kb_path": "",
        # TTS 設定 (_tts_node が読む) — character config からキャラ別に引く
        "use_real_tts": use_real_tts,
        "tts_provider": character.tts_provider,
        "tts_voice": character.tts_voice,
        "tts_speaker": character.slug,
        "tts_output_dir": tts_output_dir,
        "system_prompt": None,
        "stream_context": None,
        "on_tts_chunk_ready": on_tts_chunk_ready,
        "on_pose_ready": on_pose_ready,
        # _tts_node 内では使われないが型整合
        "suppress_bubble_answering": True,
        # disable_tools は TTS-only モードでは使われないが型整合のため None
        "disable_tools": None,
        # Phase 0.5-B-α: TTS-only モードでは _generation_node を通らないため
        # status_manager は使われない (= 型整合のため None で埋める)。
        # Talking 反映は run_loop の _spawn_handraise_response_playback 内で別途行う。
        "status_manager": None,
        # _tts_node が読む値群
        "character_slug": llm_result.speaker,
        "llm_text": llm_text,
        # rag/answer_mode/llm_meta/tts_meta は TTS-only では使われない
        "rag_context": None,
        "rag_used": False,
        "retrieved_doc_ids": [],
        "retrieval_latency_ms": 0,
        "answer_mode": "fallback",
        "llm_meta": {},
        "tts_meta": {},
        # events は LLM 結果 (utterance.final + llm.final) をコピーして注入
        "events": list(llm_result.events),
    }

    final_state = run_pipeline_graph_tts_only(initial_state)

    return PipelineResult(
        stream_id=llm_result.stream_id,
        session_id=llm_result.session_id,
        trace_id=llm_result.trace_id,
        speaker=final_state["character_slug"],
        events=final_state["events"],
    )
