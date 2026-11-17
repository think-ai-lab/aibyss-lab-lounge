#!/usr/bin/env python3
"""
bench_ttft.py — マルチプロバイダ TTFT / レイテンシ ベンチマーク

【補助の道具】モデルを切り替えるときの目安を見るための道具で、採否の根拠にしない。
  - 本番の Agent（ツール、構造化出力、system prompt）を通さず、短いプロンプトの TTFT だけを測る。
    本番と同じ経路の時間は `python -m lab_lounge.probe agent`（step ごとの時間と tokens）で測る。
  - 1 つでも失敗（[skip] を含む）があれば終了コード 1 で終わる。黙って表が欠けたまま緑にしないため。
  - **支出の記録（lab_lounge.spend）と上限の外**で動く。呼び出しの数は `--trials` × プロンプト × モデルで決まり、
    費用は記録されない。試す前に `--only` と `--trials` で絞る。価格表（spend prices）に無いモデルも流れる。
  - 継承した LLM の接続先（ANTHROPIC_BASE_URL など）は、最初に環境から外す（本番と別の相手を測らないため）。
    `.env` がそれらを指定していれば、終了コード 2 で止まる。

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
from pathlib import Path

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

# temperature を渡さないもの。プロバイダ既定値で計測する（temperature は TTFT にほぼ影響しないため
# 比較条件としては許容、と判断）。
#   - Anthropic の Claude 全部: 新しいモデル（Opus 4.7 以降、Fable 5 など）では sampling パラメータ
#     (temperature / top_p / top_k) が API から削除され、渡すと 400 になる。モデルごとの境目をこの表で
#     追うと、後継のモデルを足すたびに表の更新漏れで落ちる。Claude には一律で送らない
#   - OpenAI: gpt-5 系 reasoning モデルは temperature の変更を受け付けない
NO_TEMPERATURE_PROVIDERS = {"anthropic"}
NO_TEMPERATURE_MODELS = {
    "gpt-5.5",
    "gpt-5.4-mini",
}


def sends_temperature(provider: str, model: str) -> bool:
    """このモデルへ temperature を渡すか。Claude には渡さない。"""
    return provider not in NO_TEMPERATURE_PROVIDERS and model not in NO_TEMPERATURE_MODELS

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
    # temperature は対応モデルにのみ渡す（Claude と非対応モデルには渡さない。
    # 理由は NO_TEMPERATURE_PROVIDERS のコメント参照）
    sampling: dict = {}
    if sends_temperature(provider, model):
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
        if ttft is None:
            # 見える文字が 1 つも来ずに終わった（考えるだけで上限に達した、など）。TTFT が無い試行を成功に数えると、
            # 表の書式や中央値が None で落ちて、集計が全部失われる。失敗として数える。
            return Trial(model, prompt_kind, None, first_chunk, total, chars, error="NoVisibleText")
        return Trial(model, prompt_kind, ttft, first_chunk, total, chars)
    except Exception as e:  # noqa: BLE001
        return Trial(model, prompt_kind, None, None, None, 0, error=describe_error(e))


def describe_error(exc: BaseException) -> str:
    """例外を、型の名前と（あれば）HTTP の状態だけで表す。

    WHY: repr(e) は、API が返したメッセージ（要求の断片や鍵の一部を含みうる）をそのまま CSV と画面へ出す。
         プロンプトは今は固定の合成の文だが、将来の取り違えに備えて、型と状態だけに絞る。
    """
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return type(exc).__name__ + (f"({status})" if isinstance(status, int) else "")


def clean_inherited_endpoints(env_path: str | None) -> None:
    """継承した LLM の接続先（ANTHROPIC_BASE_URL など）を環境から外す。lab_lounge が無い環境では警告だけ。

    WHY: ツールなどから起動すると接続先を継承していることがあり、本番と別の相手を測ってしまう。
         `.env` がそれらを指定していれば、clean_llm_env が終了コード 2 で止める。
    """
    try:
        from lab_lounge.preflight import clean_llm_env
    except ImportError:
        print("[env] lab_lounge が import できないので、継承した接続先は外していません")
        return
    candidates = [env_path] if env_path else [
        os.path.join(os.getcwd(), ".env"), os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")]
    clean_llm_env([Path(path) for path in candidates if path and os.path.exists(path)])


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


class CsvSink:
    """生データ CSV を、1 試行ごとに書いて flush する。最初の試行まで、ファイルは作らない。

    WHY 1 試行ごと: 最後にまとめて書くと、途中で落ちた（例外、Ctrl+C）ときに、それまでの測定が全部失われる。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None:
            self._file.close()

    def write(self, trial: "Trial") -> None:
        if self._file is None:
            self._file = open(self.path, "w", newline="", encoding="utf-8")  # noqa: SIM115 - __exit__ で閉じる
            self._writer = csv.DictWriter(self._file, fieldnames=list(trial.__dict__.keys()))
            self._writer.writeheader()
        self._writer.writerow(trial.__dict__)
        self._file.flush()


def main() -> int:
    """ベンチを流す。失敗（[skip]、ウォームアップの失敗、試行のエラー）が 1 つでもあれば 1、なければ 0 を返す。"""
    ap = argparse.ArgumentParser(description="multi-provider TTFT benchmark")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--only", type=str, default="", help="モデル名の部分一致フィルタ")
    ap.add_argument("--prompt", type=str, default="", help="greeting / explain / deep")
    ap.add_argument("--csv", type=str, default="bench_ttft_results.csv")
    ap.add_argument("--env", type=str, default="", help=".env ファイルのパス（省略時は自動探索）")
    args = ap.parse_args()

    clean_inherited_endpoints(args.env or None)
    load_env(args.env or None)

    targets = [m for m in MODELS if args.only.lower() in m["model"].lower()]
    prompts = {k: v for k, v in PROMPTS.items() if (not args.prompt or k == args.prompt)}
    if not targets or not prompts:
        sys.exit("フィルタに一致するモデル/プロンプトがありません")

    check_keys({m["provider"] for m in targets})

    all_trials: list[Trial] = []
    # 生データ CSV は 1 試行ごとに書く（途中で落ちても、そこまでの分が残る）。
    with CsvSink(args.csv) as sink:
        failures = run_all(targets, prompts, args.trials, all_trials, sink)  # [skip]、ウォームアップ、試行のエラーの合計
    if all_trials:
        print(f"\n生データ: {args.csv}")

    print("\n===== 集計 =====")
    print_table(summarize(all_trials))
    if failures:
        print(f"\n失敗が {failures} 件あります（[skip]、ウォームアップ、試行のエラーの合計）。この表は欠けています")
        return 1
    return 0


def run_all(targets: list[dict], prompts: dict[str, str], trials: int, all_trials: list, sink: CsvSink) -> int:
    """全モデル × 全プロンプトを流し、失敗の数（[skip]、ウォームアップ、試行のエラーの合計）を返す。"""
    failures = 0
    for spec in targets:
        try:
            llm = build_llm(spec["provider"], spec["model"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {spec['model']}: クライアント生成失敗 {describe_error(e)}")
            failures += 1
            continue

        if WARMUP:
            print(f"[warmup] {spec['model']}")
            warm = run_trial(llm, spec["model"], "warmup", PROMPTS["greeting"])
            if warm.error:
                print(f"[warmup-error] {spec['model']}: {warm.error}")
                failures += 1
            time.sleep(SLEEP_BETWEEN_CALLS)

        for kind, prompt in prompts.items():
            for i in range(trials):
                t = run_trial(llm, spec["model"], kind, prompt)
                all_trials.append(t)
                sink.write(t)
                failures += bool(t.error)
                status = f"err={t.error}" if t.error else f"ttft={t.ttft_ms:.0f}ms total={t.total_ms:.0f}ms"
                print(f"[{spec['model']}] {kind} #{i + 1}: {status}")
                time.sleep(SLEEP_BETWEEN_CALLS)
    return failures


if __name__ == "__main__":
    sys.exit(main())
