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
import re
import time
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
    category: str = "speech_status",
) -> None:
    """bubble.update イベントを発行する。

    category はデフォルト "speech" (= 旧名称、後方互換)。Phase 0.5-E (= bubble 3 系統分離)
    で以下の新 category に分離予定:
      - "speech_status":  ステータス遷移 (thinking/searching/answering/done)、本 helper の
                          中心用途、bubble_messages.json の固定メッセージを表示
      - "speech_content": 発話内容 (= chunk text)、speaking 中のみ更新、run_loop の
                          `_publish_bubble_safe` 経路で別途発行 (本 helper 対象外)
      - "raisehand":      挙手系 (handraise/denied/lapsed/cancelled)、dispatcher が直接
                          build_bubble_update を呼ぶため本 helper は使わない

    本 helper は通常応答パスの speech_status 系を対象に最適化される。caller 側で
    category を明示すれば override 可能 (= future-proof: 新 step 追加時に必要なら明示)。
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
        disable_tools:   Agent から除外するツール名のリスト (例: ["ask_character"])。
                         デフォルト None で既存挙動。

    Returns:
        PipelineResult（publish 済みイベント一覧を含む）

    終わりに 1 行 "TURN SUMMARY ..." を INFO で出す (成功でも例外でも)。基準計測で、
    1 ターンが何を呼び、何トークン、何秒かを grep 1 回で読むための行。実発話の全文は含めない
    (件数・名前・時間だけ)。例外はそのまま呼び出し元へ再送出する。
    """
    from .graph import begin_turn_stats, end_turn_stats

    stats, token = begin_turn_stats()
    t0 = time.monotonic()
    outcome = "ok"
    speaker: str | None = None
    try:
        result = _run_pipeline_graph(
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
        speaker = result.speaker
        return result
    except BaseException as exc:
        # BaseException: KeyboardInterrupt などで止まったターンも「何で終わったか」を残す。再送出する。
        outcome = type(exc).__name__
        raise
    finally:
        try:
            end_turn_stats(token)
        except Exception:  # noqa: BLE001
            # reset の例外で、ターンの戻り値や、本来の例外を置き換えない。
            pass
        try:
            logger.info(
                "%s", format_turn_summary(
                    stats, speaker=speaker, trace_id=trace_id, outcome=outcome,
                    total_ms=int((time.monotonic() - t0) * 1000),
                ),
            )
        except Exception:  # noqa: BLE001
            # 集計の整形で配信のターンを落とさない (ログは補助の情報)。
            pass


def _summary_value(value: Any, *, extra_banned: str = "") -> str:
    """TURN SUMMARY の値を、空白も ``=`` も含まない 1 語にする (空は ``-``)。

    WHY: 行は ``key=value`` を空白で区切る。話者や道具の名前、例外の型に空白や ``=`` が混ざると、
    後で機械で読むときに行が割れる。extra_banned は、その値の中で区切りに使う文字 (道具の ``,`` と ``:``)。
    """
    text = re.sub(rf"[\s={re.escape(extra_banned)}]", "_", str(value)) if value is not None else ""
    return text or "-"


def format_turn_summary(
    stats: Any,
    *,
    speaker: str | None,
    trace_id: str,
    outcome: str,
    total_ms: int,
) -> str:
    """TurnStats を、機械で読める 1 行 (key=value を空白で区切る) に整える。

    WHY 1 行 key=value: 基準計測が正規表現 1 本で読めるようにするため。値に空白を入れない
    (道具は ``名前:回数`` をカンマで並べ、無ければ ``-``)。tokens の形は step の行
    (``tokens=in:X/out:Y``) と同じにして、読み手を 1 つで済ませる。

    集計の範囲:
      - steps / tokens / llm_ms は本応答のモデルの分。導入セリフなど道具の中の LLM は
        aux (呼び出し数と tokens) に分ける。協働先の Agent の分は含めない (別スレッドで走る)。
      - llm_ms は step の所要の合計。単一ノード (path=simple) は、その 1 回の call_llm の分を
        tokens / llm_ms に入れる (steps は 0 のまま。steps は Agent の step の数)。
      - Agent が失敗して単一ノードへ流れた分 (path=fallback) は、本応答と混ぜずに
        fallback=in:X/out:Y と fallback_ms に分ける。失敗した Agent の step と tokens は本応答の側に残る。
      - handler が付かない呼び出し (common が無く、Agent に BubbleToolCallbackHandler を付けない) では、
        steps / tokens / aux / tools は 0 や ``-`` になる。経路 (path) と話者は _run_agent が記録する。

    tokens の読み方:
      - cache_read は in の内数 (in の中の、キャッシュから読んだ分)。足し合わせない。
      - reasoning は out の内数 (out の中の、推論に使った分)。足し合わせない。

    Args:
        stats: graph.TurnStats
        speaker: 結果の話者 (成功時)。例外のときは None で、集計に入っている話者を使う
        trace_id: ターンのトレース ID (他のログと突き合わせる鍵)
        outcome: "ok" または例外の型名
        total_ms: run_pipeline 全体の所要 (ミリ秒)
    """
    tools = ",".join(
        f"{_summary_value(name, extra_banned=',:')}:{n}" for name, n in stats.tools.items()
    ) or "-"
    return " ".join([
        "TURN SUMMARY",
        f"trace={_summary_value(trace_id)}",
        f"speaker={_summary_value(speaker or stats.speaker or '?')}",
        f"path={_summary_value(stats.path)}",
        f"source={_summary_value(stats.source)}",
        f"tools={tools}",
        f"steps={stats.steps}",
        f"tokens=in:{stats.input_tokens}/out:{stats.output_tokens}",
        f"reasoning={stats.reasoning_tokens}",
        f"cache_read={stats.cache_read_tokens}",
        f"aux={stats.aux_calls}",
        f"aux_tokens=in:{stats.aux_input_tokens}/out:{stats.aux_output_tokens}",
        f"llm_ms={stats.llm_ms}",
        f"fallback=in:{stats.fallback_input_tokens}/out:{stats.fallback_output_tokens}",
        f"fallback_ms={stats.fallback_ms}",
        f"total_ms={total_ms}",
        f"result={_summary_value(outcome)}",
    ])


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


