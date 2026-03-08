"""
pipeline.py — 開発用テキストパイプライン

責務:
  - 入力テキストから utterance.final → llm.final → tts.done の 3 イベントを
    順番に組み立て、Event Bus (Redis Streams) に publish する
  - 各イベントに links で因果関係を持たせる:
      llm.final.links   = [utterance.final.event_id]
      tts.done.links    = [llm.final.event_id]
  - publish 以外の副作用を持たない（C2 への HTTP 呼び出し禁止 Guardrail G-2）

【パイプライン処理フロー】
  utterance.final  (seq=0, links=なし)
       ↓ links=[utterance.event_id]
  llm.final        (seq=1)
       ↓ links=[llm.event_id]
  tts.done         (seq=2)
"""

from dataclasses import dataclass
from typing import Any

from .bus import publish
from .events import build_llm_final, build_tts_done, build_utterance_final


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
) -> PipelineResult:
    """
    テキストを受け取り 3 イベントを publish する。

    Args:
        text:       発話テキスト（utterance.final の payload.text）
        stream_id:  ストリーム識別子
        session_id: セッション識別子
        trace_id:   トレース識別子

    Returns:
        PipelineResult（publish 済みイベント一覧を含む）
    """
    common = dict(stream_id=stream_id, session_id=session_id, trace_id=trace_id)

    # 1. utterance.final
    utt = build_utterance_final(text=text, seq=0, **common)
    publish(utt)

    # 2. llm.final — utterance.final を links で参照
    llm_text = f"ダミー応答: {text}"
    llm = build_llm_final(text=llm_text, seq=1, links=[utt["event_id"]], **common)
    publish(llm)

    # 3. tts.done — llm.final を links で参照
    tts = build_tts_done(text=llm_text, seq=2, links=[llm["event_id"]], **common)
    publish(tts)

    return PipelineResult(
        stream_id=stream_id,
        session_id=session_id,
        trace_id=trace_id,
        events=[utt, llm, tts],
    )
