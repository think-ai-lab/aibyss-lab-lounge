#!/usr/bin/env python3
"""
bench_ttft.py — マルチプロバイダ TTFT / レイテンシ ベンチマーク

A.I.byss Suite の LLM 呼び出しレイヤ（LangChain）経由で、
各モデルの TTFT (Time To First Token) と総生成時間を計測する。

対象モデル（2026-06-11 に各社公式ドキュメントで ID の実在を確認済み、MODELS で変更可能）:
  Anthropic : claude-haiku-4-5 / claude-sonnet-4-6 / claude-opus-4-8 / claude-fable-5
  OpenAI    : gpt-5.5 / gpt-5.4-mini
  Google    : gemini-3.1-pro-preview / gemini-3.5-flash

必要パッケージ:
  pip install langchain-anthropic langchain-openai langchain-google-genai python-dotenv

必要環境変数（スクリプトと同階層またはカレントの .env から自動読み込み）:
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY

.env の例:
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  GOOGLE_API_KEY=...

  ※ 既にシェルで設定済みの環境変数が優先される（.env では上書きしない）
  ※ --env オプションで .env のパスを明示指定可能

使い方:
  python bench_ttft.py                  # 全モデル × 全プロンプト × N回
  python bench_ttft.py --trials 3      # 試行回数変更
  python bench_ttft.py --only fable    # モデル名部分一致でフィルタ
  python bench_ttft.py --prompt deep   # プロンプト種別フィルタ
  python bench_ttft.py --csv out.csv   # CSV 出力先

計測定義:
  TTFT      = リクエスト送信開始から「最初の可視テキストチャンク」受信まで
  total     = リクエスト送信開始からストリーム終了まで
  ※ reasoning/thinking トークンは可視テキストに含めない（first_chunk と区別して記録）
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# .env 読み込み
# ---------------------------------------------------------------------------


def load_env(env_path: str | None = None) -> None:
    """python-dotenv で .env を読み込む。

    - env_path 指定時はそのファイルを使用（存在しなければエラー終了）
    - 未指定時は カレント → スクリプト同階層 の順に探索
    - 既存の環境変数は上書きしない（override=False）
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        sys.exit(
            "python-dotenv が見つかりません: pip install python-dotenv"
        )

    if env_path:
        if not os.path.exists(env_path):
            sys.exit(f".env が見つかりません: {env_path}")
        load_dotenv(env_path, override=False)
        print(f"[env] loaded: {env_path}")
        return

    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    for path in candidates:
        if os.path.exists(path):
            load_dotenv(path, override=False)
            print(f"[env] loaded: {path}")
            return
    print("[env] .env が見つからないため、シェルの環境変数のみ使用します")


REQUIRED_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def check_keys(providers: set[str]) -> None:
    missing = [REQUIRED_KEYS[p] for p in providers if not os.getenv(REQUIRED_KEYS[p])]
    if missing:
        sys.exit(f"環境変数が未設定です: {', '.join(missing)} (.env を確認してください)")


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

MAX_TOKENS = 200          # 出力上限（比較条件を揃える）
TEMPERATURE = 0.7          # ※ 非対応モデルには渡さない（NO_TEMPERATURE_MODELS 参照）
SLEEP_BETWEEN_CALLS = 2.0  # レートリミット相互干渉の回避
WARMUP = True              # 各モデル1回ウォームアップ（集計から除外）

# temperature を渡すと API が 400 を返すモデル群。
#   - Anthropic: Opus 4.7 以降 / Fable 5 では sampling パラメータ
#     (temperature / top_p / top_k) が API から削除された
#   - OpenAI: gpt-5 系 reasoning モデルは temperature の変更を受け付けない
# これらはプロバイダ既定値で計測する（temperature は TTFT にほぼ影響しないため
# 比較条件としては許容、と判断）。
NO_TEMPERATURE_MODELS = {
    "claude-opus-4-8",
    "claude-fable-5",
    "gpt-5.5",
    "gpt-5.4-mini",
}

# provider: "anthropic" | "openai" | "google"
MODELS = [
    {"provider": "anthropic", "model": "claude-haiku-4-5"},
    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    {"provider": "anthropic", "model": "claude-opus-4-8"},
    {"provider": "anthropic", "model": "claude-fable-5"},
    {"provider": "openai",    "model": "gpt-5.5"},
    {"provider": "openai",    "model": "gpt-5.4-mini"},
    {"provider": "google",    "model": "gemini-3.1-pro-preview"},
    {"provider": "google",    "model": "gemini-3.5-flash"},
]

SYSTEM_PROMPT = (
    "あなたは配信のAIキャラクターです。視聴者の発話に日本語で自然に応答してください。"
)

# さくらの運用シナリオに対応した3種
PROMPTS = {
    "greeting": "こんにちは、今日もよろしくお願いします！",
    "explain": "ハルシネーションってなんですか？簡単に教えてください。",
    "deep": (
        "最近、仕事でずっとうまくいかなくて、自分には価値がないんじゃないかって"
        "思ってしまいます。頑張っても報われない気がして、つらいです。"
        "こういうとき、どう考えたらいいんでしょうか。"
    ),
}

# ---------------------------------------------------------------------------
# クライアント生成（LangChain 経由 = 本番と同じ抽象化レイヤ）
# ---------------------------------------------------------------------------


def build_llm(provider: str, model: str):
    # temperature は対応モデルにのみ渡す（非対応モデルは API が 400 を返す。
    # 一覧と理由は NO_TEMPERATURE_MODELS のコメント参照）
    sampling: dict = {}
    if model not in NO_TEMPERATURE_MODELS:
        sampling["temperature"] = TEMPERATURE

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=MAX_TOKENS, **sampling)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # reasoning 系モデルは最小推論に寄せて TTFT 条件を揃える。
        # reasoning_effort は langchain-openai 1.x では第一級引数のため直接渡す
        # （model_kwargs 経由だと UserWarning が出る）。
        return ChatOpenAI(
            model=model,
            max_tokens=MAX_TOKENS,
            reasoning_effort="low",
            **sampling,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, max_output_tokens=MAX_TOKENS, **sampling
        )
    raise ValueError(f"unknown provider: {provider}")


# ---------------------------------------------------------------------------
# 計測
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    model: str
    prompt_kind: str
    ttft_ms: float | None      # 最初の「可視テキスト」まで
    first_chunk_ms: float | None  # 最初のチャンク（thinking 含む）まで
    total_ms: float | None
    out_chars: int
    error: str = ""


def extract_text(chunk) -> str:
    """LangChain のチャンクから可視テキストのみを取り出す。"""
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Anthropic: {"type": "text", "text": ...} / thinking ブロックは除外
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def run_trial(llm, model: str, prompt_kind: str, prompt: str) -> Trial:
    messages = [("system", SYSTEM_PROMPT), ("human", prompt)]
    t0 = time.perf_counter()
    ttft = None
    first_chunk = None
    chars = 0
    try:
        for chunk in llm.stream(messages):
            now = time.perf_counter()
            if first_chunk is None:
                first_chunk = (now - t0) * 1000
            text = extract_text(chunk)
            if text:
                if ttft is None:
                    ttft = (now - t0) * 1000
                chars += len(text)
        total = (time.perf_counter() - t0) * 1000
        return Trial(model, prompt_kind, ttft, first_chunk, total, chars)
    except Exception as e:  # noqa: BLE001
        return Trial(model, prompt_kind, None, None, None, 0, error=repr(e))


# ---------------------------------------------------------------------------
# 集計・出力
# ---------------------------------------------------------------------------


def pctile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(trials: list[Trial]) -> list[dict]:
    rows = []
    keys = sorted({(t.model, t.prompt_kind) for t in trials})
    for model, kind in keys:
        ok = [t for t in trials if t.model == model and t.prompt_kind == kind and not t.error]
        err = [t for t in trials if t.model == model and t.prompt_kind == kind and t.error]
        ttfts = [t.ttft_ms for t in ok if t.ttft_ms is not None]
        totals = [t.total_ms for t in ok if t.total_ms is not None]
        rows.append({
            "model": model,
            "prompt": kind,
            "n_ok": len(ok),
            "n_err": len(err),
            "ttft_med_ms": round(statistics.median(ttfts)) if ttfts else None,
            "ttft_p95_ms": round(pctile(ttfts, 0.95)) if ttfts else None,
            "total_med_ms": round(statistics.median(totals)) if totals else None,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'model':<28} {'prompt':<9} {'n':>3} {'err':>3} {'TTFT中央値':>10} {'TTFT p95':>9} {'total中央値':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<28} {r['prompt']:<9} {r['n_ok']:>3} {r['n_err']:>3} "
            f"{str(r['ttft_med_ms']) + 'ms' if r['ttft_med_ms'] else '---':>10} "
            f"{str(r['ttft_p95_ms']) + 'ms' if r['ttft_p95_ms'] else '---':>9} "
            f"{str(r['total_med_ms']) + 'ms' if r['total_med_ms'] else '---':>11}"
        )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="multi-provider TTFT benchmark")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--only", type=str, default="", help="モデル名の部分一致フィルタ")
    ap.add_argument("--prompt", type=str, default="", help="greeting / explain / deep")
    ap.add_argument("--csv", type=str, default="bench_ttft_results.csv")
    ap.add_argument("--env", type=str, default="", help=".env ファイルのパス（省略時は自動探索）")
    args = ap.parse_args()

    load_env(args.env or None)

    targets = [m for m in MODELS if args.only.lower() in m["model"].lower()]
    prompts = {k: v for k, v in PROMPTS.items() if (not args.prompt or k == args.prompt)}
    if not targets or not prompts:
        sys.exit("フィルタに一致するモデル/プロンプトがありません")

    check_keys({m["provider"] for m in targets})

    all_trials: list[Trial] = []
    raw_rows: list[dict] = []

    for spec in targets:
        try:
            llm = build_llm(spec["provider"], spec["model"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {spec['model']}: クライアント生成失敗 {e!r}")
            continue

        if WARMUP:
            print(f"[warmup] {spec['model']}")
            run_trial(llm, spec["model"], "warmup", PROMPTS["greeting"])
            time.sleep(SLEEP_BETWEEN_CALLS)

        for kind, prompt in prompts.items():
            for i in range(args.trials):
                t = run_trial(llm, spec["model"], kind, prompt)
                all_trials.append(t)
                raw_rows.append(t.__dict__)
                status = f"err={t.error}" if t.error else f"ttft={t.ttft_ms:.0f}ms total={t.total_ms:.0f}ms"
                print(f"[{spec['model']}] {kind} #{i + 1}: {status}")
                time.sleep(SLEEP_BETWEEN_CALLS)

    # 生データ CSV
    if raw_rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\n生データ: {args.csv}")

    print("\n===== 集計 =====")
    print_table(summarize(all_trials))


if __name__ == "__main__":
    main()
