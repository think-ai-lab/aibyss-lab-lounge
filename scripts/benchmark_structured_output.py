"""
benchmark_structured_output.py -- LangGraph create_react_agent の response_format
レイテンシ実測 (Phase 0.5-A フェーズ 8 調査)

「response_format=Pydantic を指定すると agent ループ完了後に追加 LLM 呼出が発生し、
レイテンシが +1-3 秒される」という仮説を実機で検証する。

【使い方】
  uv run python scripts/benchmark_structured_output.py

【測定方法】
  同じ system_prompt + user input を、以下の 2 通りで N 回ずつ呼出して latency を比較:
    A: response_format なし (現状の実装)
    B: response_format=Pydantic (キャラ応答スキーマで強制)

  ツール (web_search 等) は呼ばれにくい入力にして、ツール呼出ループによる
  ばらつきを抑える (= 純粋に「応答 LLM + 構造化 LLM」のレイテンシを測る)。

【出力】
  各実行ごとの latency + 平均 + 中央値 + 分布。
  bg_result の B/A 比 (= structured_response が +X% 遅い) を表示。
"""

import logging
import os
import sys
import statistics
import time
from pathlib import Path

# Windows cp932 環境でも日本語 / 絵文字を出力できるように stdout を UTF-8 化
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# プロジェクトルートを sys.path に追加
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# .env を明示パスで読み込み (uv run 経由で cwd 推定が外れる対策)。
# override=True で PowerShell session に残った古い env (空文字 / 旧値) を強制上書き。
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=True)
_dev_env = _REPO_ROOT / ".env.dev"
if _dev_env.is_file():
    load_dotenv(_dev_env, override=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── 検証パラメータ ──────────────────────────────────────────────

# 実 API 呼出を行うキャラとモデル。L2 で問題が出た sakura (claude-sonnet-4-6) を
# 主軸にする。GPT/Gemini は別ラン。
CHARACTERS = [
    # (slug, provider, model, voice_emotion_keys)
    ("sakura", "anthropic", "claude-sonnet-4-6",
     ("happy", "sad", "angry", "whisper", "cool")),
    # ("mimi", "openai", "gpt-5.5", ("happy", "fun", "angry", "sad", "sulky")),
    # ("chisame", "google", "gemini-3.1-pro-preview",
    #  ("bosoboso", "doyaru", "honwaka", "angry", "teary")),
]

# 各 agent を呼び出す回数 (回数を増やすほど分散が安定するが API コスト増)
N_ITERATIONS = 5

# 同じ入力テキスト (ツールを呼ばずに即応答する想定)
TEST_INPUT = "最近のAI倫理について気になっているのですが、どう考えるべきでしょうか?"


def _build_pydantic_schema(slug: str, emotion_keys: tuple[str, ...]):
    """キャラ別の応答 Pydantic スキーマを動的構築する。

    emotion フィールドはキャラごとにキー集合が異なる (sakura:
    happy/sad/angry/whisper/cool / mimi: happy/fun/angry/sad/sulky 等)。
    """
    from pydantic import BaseModel, Field, create_model

    # emotion サブスキーマを動的に生成 (キャラ別キー)
    emotion_fields = {
        k: (int, Field(default=0, ge=0, le=100, description=f"{k} 強度 (0-100)"))
        for k in emotion_keys
    }
    EmotionSchema = create_model(
        f"{slug.capitalize()}Emotion",
        __doc__=f"{slug} の emotion フィールド (5 キー)",
        **emotion_fields,
    )

    # メイン応答スキーマ
    ResponseSchema = create_model(
        f"{slug.capitalize()}Response",
        __doc__=f"{slug} のキャラ応答 (response/emotion/speed/pose)",
        response=(str, Field(..., description="ユーザーへの応答テキスト")),
        emotion=(EmotionSchema, Field(..., description="感情パラメータ")),
        speed=(int, Field(default=100, ge=50, le=200,
                          description="発話速度 (50-200、100 が標準)")),
        pose=(str, Field(default="neutral", description="OBS 立ち絵 pose 名")),
    )
    return ResponseSchema


def _build_agent(provider: str, model: str, response_format=None, *, use_new_api: bool = True):
    """ReAct Agent を構築する。

    use_new_api=True (default): LangChain v1 系 ``langchain.agents.create_agent`` を使う。
                                Anthropic Claude 4.x の新仕様 (assistant prefill 不可)
                                に対応した tool-based structured output を使う。
    use_new_api=False:          旧 ``langgraph.prebuilt.create_react_agent`` を使う
                                (deprecated、Claude 4.x で response_format 利用時に
                                400 エラーが出る)。

    response_format=None なら通常応答。response_format=Pydantic で structured output。
    """
    # LLM プロバイダごとに ChatModel を取得
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, max_tokens=2048)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model)
    else:
        raise ValueError(f"未対応 provider: {provider}")

    # ダミーツール (= 呼ばれにくい retrieve_memory 風ツール、agent 構造を保つ)
    from langchain_core.tools import tool

    @tool
    def retrieve_memory(query: str) -> str:
        """過去の会話履歴を検索する。一般的な話題には不要。"""
        return "(関連メモなし)"

    tools = [retrieve_memory]

    system_prompt = (
        "あなたは AI キャラの応答生成器です。"
        "ユーザーの質問に対して、自分の意見を含む応答を簡潔に (200 文字程度) 返してください。"
        "技術的な裏付けが必要な場合のみ retrieve_memory ツールを使用してください。"
        "通常の会話・意見表明には不要です。"
    )

    if use_new_api:
        # 新 API: langchain.agents.create_agent (LangChain v1 系の正式 API)
        from langchain.agents import create_agent
        kwargs = {
            "model": llm,
            "tools": tools,
            "system_prompt": system_prompt,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return create_agent(**kwargs)
    else:
        # 旧 API: langgraph.prebuilt.create_react_agent (deprecated)
        from langgraph.prebuilt import create_react_agent
        kwargs = {
            "model": llm,
            "tools": tools,
            "prompt": system_prompt,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return create_react_agent(**kwargs)


def _run_agent(agent, user_text: str) -> tuple[float, dict]:
    """agent を 1 回呼び出して latency と最終 state を返す。"""
    start = time.monotonic()
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_text}],
        })
    except Exception as exc:
        elapsed = time.monotonic() - start
        return elapsed, {"error": str(exc)}
    elapsed = time.monotonic() - start
    return elapsed, result


def _summary(label: str, latencies: list[float]) -> dict:
    """latency 統計を辞書で返す。"""
    return {
        "label": label,
        "n": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) >= 2 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
    }


def _print_summary(summary: dict, samples: list[float]) -> None:
    print(f"\n  [{summary['label']}] n={summary['n']}")
    print(f"    mean   = {summary['mean']:.3f}s")
    print(f"    median = {summary['median']:.3f}s")
    print(f"    stdev  = {summary['stdev']:.3f}s")
    print(f"    min    = {summary['min']:.3f}s")
    print(f"    max    = {summary['max']:.3f}s")
    print(f"    samples= {[f'{x:.3f}' for x in samples]}")


def main():
    print("=" * 70)
    print("LangGraph create_react_agent: response_format レイテンシ検証")
    print("=" * 70)
    print(f"入力テキスト: {TEST_INPUT!r}")
    print(f"反復回数:     {N_ITERATIONS} 回 / 各設定")
    print()

    for slug, provider, model, emotion_keys in CHARACTERS:
        print(f"\n{'=' * 70}")
        print(f"  キャラ: {slug} (provider={provider}, model={model})")
        print(f"{'=' * 70}")

        # スキーマ構築
        ResponseSchema = _build_pydantic_schema(slug, emotion_keys)
        print(f"  Pydantic スキーマ: {ResponseSchema.__name__}")
        print(f"    fields: {list(ResponseSchema.model_fields.keys())}")
        print(f"    emotion keys: {emotion_keys}")

        # ─── A: response_format なし ───
        print(f"\n  [A] response_format なし (現状実装)")
        agent_a = _build_agent(provider, model, response_format=None)

        latencies_a: list[float] = []
        for i in range(N_ITERATIONS):
            elapsed, result = _run_agent(agent_a, TEST_INPUT)
            if "error" in result:
                print(f"    [{i+1}] {elapsed:.3f}s ERROR: {result['error']}")
            else:
                last_msg = result["messages"][-1]
                content_preview = (
                    str(last_msg.content)[:50] if hasattr(last_msg, "content")
                    else str(last_msg)[:50]
                )
                print(f"    [{i+1}] {elapsed:.3f}s -- {content_preview!r}...")
            latencies_a.append(elapsed)

        # ─── B: response_format=Pydantic ───
        print(f"\n  [B] response_format=Pydantic (構造化出力強制)")
        agent_b = _build_agent(provider, model, response_format=ResponseSchema)

        latencies_b: list[float] = []
        for i in range(N_ITERATIONS):
            elapsed, result = _run_agent(agent_b, TEST_INPUT)
            if "error" in result:
                print(f"    [{i+1}] {elapsed:.3f}s ERROR: {result['error']}")
            else:
                structured = result.get("structured_response")
                if structured is not None:
                    response_text = (
                        structured.response[:50] if hasattr(structured, "response")
                        else str(structured)[:50]
                    )
                    print(f"    [{i+1}] {elapsed:.3f}s -- {response_text!r}...")
                else:
                    print(f"    [{i+1}] {elapsed:.3f}s -- (structured_response 無し)")
            latencies_b.append(elapsed)

        # ─── サマリ ───
        sum_a = _summary("A: response_format なし", latencies_a)
        sum_b = _summary("B: response_format=Pydantic", latencies_b)
        _print_summary(sum_a, latencies_a)
        _print_summary(sum_b, latencies_b)

        # 比較
        diff_mean = sum_b["mean"] - sum_a["mean"]
        ratio = (sum_b["mean"] / sum_a["mean"] - 1) * 100 if sum_a["mean"] > 0 else 0
        print(f"\n  === 比較 (B - A) ===")
        print(f"    Δmean   = {diff_mean:+.3f}s ({ratio:+.1f}%)")
        print(f"    Δmedian = {sum_b['median'] - sum_a['median']:+.3f}s")

    print("\n" + "=" * 70)
    print("検証完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
