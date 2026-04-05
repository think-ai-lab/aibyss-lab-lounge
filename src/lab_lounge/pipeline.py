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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bus import publish
from .characters import get_character, load_system_prompt
from .debug import (
    write_llm_prompt,
    write_llm_response,
    write_retrieval,
    write_stt_output,
)
from .events import build_bubble_update, build_llm_final, build_tts_done, build_utterance_final
from .observability import build_run_metadata
from .router import route

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
) -> None:
    """bubble.update イベントを発行する。"""
    messages = _load_bubble_messages()
    char_msgs = messages.get(character_slug, {})
    text = char_msgs.get(step, "")

    try:
        bubble = build_bubble_update(
            character=character_slug,
            step=step,
            text=text,
            links=links,
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
    top_k = int(os.environ.get("L2_RAG_TOP_K", "3"))
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
) -> PipelineResult:
    """
    テキストを受け取り 3 イベントを publish する。

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

    Returns:
        PipelineResult（publish 済みイベント一覧を含む）
    """
    common = dict(stream_id=stream_id, session_id=session_id, trace_id=trace_id)

    # 0. ルーティング — どのキャラクターが応答するか決定
    decision = route(text, name_hint=speaker_hint)
    character = get_character(decision.speaker)
    try:
        system_prompt = load_system_prompt(character)
    except FileNotFoundError:
        logger.warning(
            "システムプロンプトが見つかりません: %s。プロンプトなしで続行。",
            character.system_prompt_file,
        )
        system_prompt = None

    # 1. utterance.final — STT メタデータがあれば反映する
    utt_kwargs: dict[str, Any] = utterance_meta or {}
    utt = build_utterance_final(text=text, seq=0, **utt_kwargs, **common)
    publish(utt)
    write_stt_output(text, utterance_meta)

    # ─── bubble: searching ───
    _publish_bubble("searching", character.slug, common, links=[utt["event_id"]])

    # 2. Retrieve (optional) — L2_ENABLE_RAG=true のときのみ実行
    rag_context: str | None = None
    rag_used = False
    retrieved_doc_ids: list[str] = []
    retrieval_latency_ms = 0
    answer_mode = "fallback"

    enable_rag, top_k, kb_path = _get_rag_mode()
    if enable_rag:
        try:
            from .retriever import LocalRetriever
            _t0 = time.monotonic()
            _retriever = LocalRetriever(kb_path)
            _docs = _retriever.retrieve(text, top_k=top_k)
            retrieval_latency_ms = int((time.monotonic() - _t0) * 1000)
            if _docs:
                rag_context = "\n\n---\n\n".join(d.text for d in _docs)
                retrieved_doc_ids = [d.doc_id for d in _docs]
                _retrieval_scores = [d.score for d in _docs]
                rag_used = True
                answer_mode = "grounded"
                logger.info(
                    "RAG 検索完了: latency_ms=%d docs=%d ids=%s",
                    retrieval_latency_ms,
                    len(_docs),
                    retrieved_doc_ids,
                )
            else:
                _retrieval_scores = []
            write_retrieval(retrieved_doc_ids, _retrieval_scores, retrieval_latency_ms, rag_enabled=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG 検索失敗 (fallback): %s", exc)
            rag_context = None
            rag_used = False
            answer_mode = "fallback"
            write_retrieval([], [], retrieval_latency_ms, rag_enabled=True)

    # ─── bubble: thinking ───
    _publish_bubble("thinking", character.slug, common, links=[utt["event_id"]])

    # 3. llm.final — utterance.final を links で参照
    use_real, provider, llm_model = _get_llm_mode(character)
    if use_real:
        # real mode: graph.py 経由 (lazy import — ダミーモードでは langgraph 不要)
        from .graph import run_graph as _run_graph
        _run_meta = build_run_metadata(
            stream_id=stream_id,
            session_id=session_id,
            trace_id=trace_id,
            rag_used=rag_used,
            answer_mode=answer_mode,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieved_doc_count=len(retrieved_doc_ids),
            retrieved_doc_ids=retrieved_doc_ids,
        )
        write_llm_prompt(text, rag_context)
        _llm_result = _run_graph(
            text,
            model=llm_model,
            provider=provider,
            context=rag_context,
            system_prompt=system_prompt,
            run_metadata=_run_meta,
        )
        write_llm_response(_llm_result.text)
        llm_text = _llm_result.text
        llm_meta: dict[str, Any] = dict(
            model=_llm_result.model,
            input_tokens=_llm_result.input_tokens,
            output_tokens=_llm_result.output_tokens,
            latency_ms=_llm_result.latency_ms,
            finish_reason=_llm_result.finish_reason,
            rag_used=rag_used,
            answer_mode=answer_mode,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieved_doc_count=len(retrieved_doc_ids),
            retrieved_doc_ids=retrieved_doc_ids,
        )
    else:
        # dummy mode: 後方互換のため "ダミー応答: {text}" を維持する
        llm_text = f"ダミー応答: {text}"
        llm_meta = {}  # build_llm_final のデフォルト値を使う
    llm = build_llm_final(text=llm_text, seq=1, links=[utt["event_id"]], **llm_meta, **common)
    publish(llm)

    # ─── bubble: answering ───
    _publish_bubble("answering", character.slug, common, links=[llm["event_id"]])

    # 4. tts.done — llm.final を links で参照
    use_real_tts, _env_tts_provider, _env_tts_voice, _env_tts_speaker, tts_output_dir = _get_tts_mode()
    # キャラクター設定を優先。env は fallback
    tts_provider = character.tts_provider
    tts_voice = character.tts_voice
    tts_speaker = character.slug
    if use_real_tts:
        # real mode: tts.py 経由 (lazy import — ダミーモードでは edge-tts 不要)
        from .tts import synthesize as _synthesize
        _tts_result = _synthesize(
            llm_text,
            provider=tts_provider,
            voice=tts_voice,
            speaker=tts_speaker,
            output_dir=tts_output_dir,
            on_chunk_ready=on_tts_chunk_ready,
        )
        tts_meta: dict[str, Any] = dict(
            audio_url=_tts_result.audio_url,
            duration_ms=_tts_result.duration_ms,
            voice=_tts_result.voice,
            format=_tts_result.format,
            sample_rate=_tts_result.sample_rate,
            speaker=_tts_result.speaker,
        )
    else:
        tts_meta = dict(speaker=tts_speaker)  # ダミーモードでも speaker slug を記録
    tts = build_tts_done(text=llm_text, seq=2, links=[llm["event_id"]], **tts_meta, **common)
    publish(tts)

    # ─── bubble: done ───
    _publish_bubble("done", character.slug, common, links=[tts["event_id"]])

    return PipelineResult(
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        speaker=character.slug,
        events=[utt, llm, tts],
    )
