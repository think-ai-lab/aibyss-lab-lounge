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

【パイプライン処理フロー】
  utterance.final  (seq=0, links=なし)
       ↓ links=[utterance.event_id]
  llm.final        (seq=1)
       ↓ links=[llm.event_id]
  tts.done         (seq=2)
"""

import os
from dataclasses import dataclass
from typing import Any

from .bus import publish
from .events import build_llm_final, build_tts_done, build_utterance_final
from .observability import build_run_metadata


# ─── LLM モード設定 ──────────────────────────────────────────────

def _get_llm_mode() -> tuple[bool, str, str]:
    """
    環境変数から LLM 実行モードを読み取る。

    Returns:
        (use_real, provider, model)
          use_real: True なら graph.py 経由で real LLM を呼ぶ
          provider: "openai" など
          model:    モデル名
    """
    use_real = os.environ.get("L2_USE_REAL_LLM", "false").lower() in ("true", "1", "yes")
    provider = os.environ.get("L2_LLM_PROVIDER", "openai")
    model = os.environ.get("L2_LLM_MODEL", "gpt-4o-mini")
    return use_real, provider, model


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
    events: list[dict[str, Any]]  # publish した順に格納


def run_pipeline(
    text: str,
    *,
    stream_id: str,
    session_id: str,
    trace_id: str,
    utterance_meta: dict[str, Any] | None = None,
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

    Returns:
        PipelineResult（publish 済みイベント一覧を含む）
    """
    common = dict(stream_id=stream_id, session_id=session_id, trace_id=trace_id)

    # 1. utterance.final — STT メタデータがあれば反映する
    utt_kwargs: dict[str, Any] = utterance_meta or {}
    utt = build_utterance_final(text=text, seq=0, **utt_kwargs, **common)
    publish(utt)

    # 2. llm.final — utterance.final を links で参照
    use_real, provider, llm_model = _get_llm_mode()
    if use_real:
        # real mode: graph.py 経由 (lazy import — ダミーモードでは langgraph 不要)
        from .graph import run_graph as _run_graph
        _run_meta = build_run_metadata(
            stream_id=stream_id,
            session_id=session_id,
            trace_id=trace_id,
        )
        _llm_result = _run_graph(text, model=llm_model, provider=provider, run_metadata=_run_meta)
        llm_text = _llm_result.text
        llm_meta: dict[str, Any] = dict(
            model=_llm_result.model,
            input_tokens=_llm_result.input_tokens,
            output_tokens=_llm_result.output_tokens,
            latency_ms=_llm_result.latency_ms,
            finish_reason=_llm_result.finish_reason,
            rag_used=False,
        )
    else:
        # dummy mode: 後方互換のため "ダミー応答: {text}" を維持する
        llm_text = f"ダミー応答: {text}"
        llm_meta = {}  # build_llm_final のデフォルト値を使う
    llm = build_llm_final(text=llm_text, seq=1, links=[utt["event_id"]], **llm_meta, **common)
    publish(llm)

    # 3. tts.done — llm.final を links で参照
    use_real_tts, tts_provider, tts_voice, tts_speaker, tts_output_dir = _get_tts_mode()
    if use_real_tts:
        # real mode: tts.py 経由 (lazy import — ダミーモードでは edge-tts 不要)
        from .tts import synthesize as _synthesize
        _tts_result = _synthesize(
            llm_text,
            provider=tts_provider,
            voice=tts_voice,
            speaker=tts_speaker,
            output_dir=tts_output_dir,
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
        tts_meta = {}  # build_tts_done のデフォルト値を使う
    tts = build_tts_done(text=llm_text, seq=2, links=[llm["event_id"]], **tts_meta, **common)
    publish(tts)

    return PipelineResult(
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        events=[utt, llm, tts],
    )
