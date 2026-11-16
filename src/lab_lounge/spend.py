"""
spend.py — 有料の計測で使う、価格表と支出の記録と上限（計測の道具の共通部品）

責務:
  - モデルごとの単価（出典の URL と確認日つき）を持ち、usage から金額を出す
  - LLM の API への HTTP の応答から usage だけを読み、1 呼び出しごとに支出の記録へ足す
  - 送る前に金額を多めに見積もり、上限を超えそうなら送らずに止める
  - 計測の道具（リプレイ、計測）の最上位で、例外と終了を「本文を出さない形」に整える

本番の経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない
（本番のモジュールがこれを import しないことはテストで固定している）。

【WHY HTTP の層で usage を写すか】
  - SDK のクラスの形（response.usage の型）ではなく、API の応答の形に依る。SDK を上げても同じ読み方で済む
  - SDK が内部で再試行した分も、HTTP の 1 往復ごとに数えられる（SDK の戻り値は最後の 1 回分しか持たない）
  - router と filler の raw の呼び出しは本文しか返さないので、呼び出し側では tokens を拾えない
  嬉しくない点: SDK が httpx / httpx2 以外の HTTP クライアント（aiohttp など）を使うと拾えない。
  そのため、包みを入れた直後に自己テストを流し、「有料の呼び出しをしたのに usage が 0 件」なら止める。

【使い方】
  # 支出の記録の要約（金額と件数だけ。本文は元から記録していない）
  uv run --frozen --all-extras python -m lab_lounge.spend show

  # 価格表と出典
  uv run --frozen --all-extras python -m lab_lounge.spend prices

  計測の道具からは、次のように使う::

      ledger = SpendLedger(default_spend_dir())
      with UsageMeter(ledger, run="label", max_usd=1.0) as meter:
          ...                      # 本物の関数を呼ぶ（SDK が httpx で API へ送る）
          meter.check_capture(n)   # n 回呼んだのに usage が 0 件なら UsageCaptureLost

【出力の約束】
  支出の記録（data/probe/spent.json と calls.jsonl、gitignore 済み）には、時刻、実行の名前、モデル名、
  tokens、金額、状態だけを書く。要求と応答の本文、ヘッダ、URL の query は保存しない。
  例外とログも、本文を含みうるので画面へ素で出さない（run_tool と quiet_failures）。

【終了コード】（run_tool を通した道具に共通）
  0: 正常に終わった / 1: 失敗した、想定外のエラー / 4: 上限、価格表にないモデル、usage の拾い損ねで止めた
  64: コマンドラインの誤り / 130: Ctrl+C で止めた
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

# ─── 定数 ─────────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SPEND_STOP = 4
EXIT_USAGE = 64

# 有料の計測すべてを合わせた上限（USD）。支出の記録のファイルに書かれた値があれば、そちらを使う。
# WHY ファイルの値を優先: 上限を上げるのは人の判断にしたい。コマンドの引数で黙って上げられないようにする。
TOTAL_CAP_USD = 5.0

# 送る前の見積もりで、最大出力が要求に書かれていないときに仮定する出力の tokens。
# WHY: 書かれていない要求の出力は、モデルの上限まで伸びうる。大きめに見て、上限の判定を甘くしない。
DEFAULT_ASSUMED_MAX_OUTPUT = 8192

# 送る前の見積もりで、要求の本文の何 bytes を 1 token とみなすか。
# WHY 2: 日本語の UTF-8 は 1 文字 3 bytes で、だいたい 1〜1.5 文字が 1 token。英語は 4 文字が 1 token 前後。
#     どちらでも bytes/2 は実際より多い側に出るので、上限の判定が甘くならない。
_BYTES_PER_TOKEN_ESTIMATE = 2

# 2xx の応答がこの回数続いて、usage を 1 件も拾えていなければ、次を送る前に止める。
_ZERO_USAGE_LIMIT = 3

# 流れてくる応答（stream）を、usage を読むために貯める上限。超えたら貯めるのをやめ、拾えなかったと数える。
_TEE_LIMIT_BYTES = 8 * 1024 * 1024

# 自己テストで使うモデル名。価格表には載せず、支出の記録にも書かない。
_SELFTEST_MODEL = "spend-selftest-model"
_SELFTEST_INPUT_TOKENS = 7

# 3 社の API のホスト。ここへの POST で、知らない道（batches など）は数えられないので送らない。
_PROVIDER_HOSTS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "generativelanguage.googleapis.com": "google",
}
_GEMINI_ROUTE = re.compile(r"/models/(?P<model>[^/:]+):(?P<method>generateContent|streamGenerateContent)$")

# 画面と記録に出してよい、モデル名や実行の名前の形（空白のない短い ASCII）。
_SHOWABLE = re.compile(r"[A-Za-z0-9_.:@/\-]{1,80}")
_RUN_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


# ─── 止める例外 ───────────────────────────────────────────────────

class SpendStop(BaseException):
    """有料の計測を止める例外の親。

    WHY BaseException から作る: L2 の判定の関数（router の _call_router_llm など）は、配信を止めないために
    ``except Exception`` で失敗を握り、None を返す。Exception の子にすると、上限で止めたつもりが
    「判定の失敗」として飲まれ、次の行の呼び出しがそのまま続く。BaseException なら抜けて最上位まで届く。

    文言には、自分で作った数字とモデル名だけを入れる（本文を入れない）。
    """


class BudgetExceeded(SpendStop):
    """送る前の見積もりで、上限（実行ごと、または全体）を超えそうになった。"""


class UnpricedModel(SpendStop):
    """価格表にないモデル、またはモデルの分からない要求を送ろうとした。"""


class UsageCaptureLost(SpendStop):
    """有料の呼び出しをしたのに usage を拾えていない（包みが効いていない）。"""


class LedgerBusy(SpendStop):
    """別の有料の実行が、支出の記録を使っている。"""


class MeterFailure(SpendStop):
    """包みの中で想定外の失敗が起きた（支出の記録が書けない、など）。

    WHY SpendStop にする: 包みは SDK の send の中で動く。そこから Exception を出すと、SDK はそれを接続の失敗と
         みなして再試行し、同じ要求をもう一度送って課金が重なる。BaseException なら再試行されずに止まる。
    """


# ─── 価格表 ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rate:
    """1 つの価格帯の単価（USD / 100 万 tokens）。

    Attributes:
        input:          キャッシュに当たらない入力
        output:         出力（reasoning / thinking も出力として数える）
        cache_read:     キャッシュから読んだ入力
        cache_write_5m: キャッシュへ書いた入力（5 分）。None なら書き込みの課金がない（入力の単価で数える）
        cache_write_1h: キャッシュへ書いた入力（1 時間）。None なら cache_write_5m と同じに数える
    """

    input: float
    output: float
    cache_read: float
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None


@dataclass(frozen=True)
class Price:
    """モデル 1 つの価格。

    Attributes:
        provider:        anthropic / openai / google
        rate:            ふつうの価格帯
        source_url:      確かめた公式の価格表
        as_of:           確かめた日（YYYY-MM-DD）
        long_rate:       長い入力の価格帯。無ければ None
        long_over_tokens: 入力（キャッシュを含む）がこれを超えると long_rate
        note:            条件の補足（音声の入力は別の単価、など）
    """

    provider: str
    rate: Rate
    source_url: str
    as_of: str
    long_rate: Rate | None = None
    long_over_tokens: int | None = None
    note: str = ""

    def rate_for(self, prompt_tokens: int) -> Rate:
        """入力の tokens（キャッシュを含む）に応じた価格帯を返す。"""
        if self.long_rate is not None and self.long_over_tokens is not None and prompt_tokens > self.long_over_tokens:
            return self.long_rate
        return self.rate


_ANTHROPIC_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
_OPENAI_SOURCE = "https://developers.openai.com/api/docs/pricing"
_GOOGLE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"

# 公式の価格表で確かめた単価（標準の層、グローバルの経路）。確かめられない値は入れない。
# 表にないモデルは流さない（UnpricedModel）。一時的な上書きは --price で渡し、支出の記録に残す。
# WHY 支出の上限はこの表に依る: 表がずれると上限の判定もずれる。実際の請求は、各社の管理画面で確かめる。
PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(
        "anthropic", Rate(1.00, 5.00, 0.10, 1.25, 2.00), _ANTHROPIC_SOURCE, "2026-11-16",
    ),
    "claude-sonnet-4-6": Price(
        "anthropic", Rate(3.00, 15.00, 0.30, 3.75, 6.00), _ANTHROPIC_SOURCE, "2026-11-16",
        note="1M の文脈まで同じ単価",
    ),
    "gpt-5.5": Price(
        "openai", Rate(5.00, 30.00, 0.50), _OPENAI_SOURCE, "2026-11-16",
        long_rate=Rate(10.00, 45.00, 1.00), long_over_tokens=272_000,
    ),
    "gpt-5.4-mini": Price("openai", Rate(0.75, 4.50, 0.075), _OPENAI_SOURCE, "2026-11-16"),
    "gpt-5.4-nano": Price("openai", Rate(0.20, 1.25, 0.02), _OPENAI_SOURCE, "2026-11-16"),
    "gemini-3.1-pro-preview": Price(
        "google", Rate(2.00, 12.00, 0.20), _GOOGLE_SOURCE, "2026-11-16",
        long_rate=Rate(4.00, 18.00, 0.40), long_over_tokens=200_000,
        note="キャッシュの保管料（時間あたり）は含まない。明示のキャッシュは使っていない",
    ),
    "gemini-3.1-flash-lite": Price(
        "google", Rate(0.25, 1.50, 0.025), _GOOGLE_SOURCE, "2026-11-16",
        note="テキストの単価。音声の入力は別（L2 は送らない）",
    ),
}


def lookup_price(model: str | None, prices: dict[str, Price]) -> Price:
    """モデル名から価格を引く。日付の付いた版名（-20251001、-2026-03-05）と models/ の接頭辞は外して引く。

    Raises:
        UnpricedModel: 表にない、またはモデルが分からない
    """
    if not model:
        raise UnpricedModel("モデルの分からない要求は送りません")
    name = model.removeprefix("models/")
    for candidate in (name, re.sub(r"-\d{8}$", "", name), re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)):
        if candidate in prices:
            return prices[candidate]
    raise UnpricedModel(f"価格表にないモデルです: {_showable(model)}")


def parse_price_override(text: str) -> tuple[str, Price]:
    """``--price model=in,out[,cache_read[,cache_write]]`` の値を読む（USD / 100 万 tokens）。

    WHY: 価格表にない新しいモデルを、表を直す前に 1 回だけ試したいときの口。使ったことは支出の記録に残る。

    Raises:
        ValueError: 形が違う、数が負
    """
    name, sep, numbers = text.partition("=")
    name = name.strip()
    if not sep or not _SHOWABLE.fullmatch(name):
        raise ValueError("--price は model=in,out[,cache_read[,cache_write]] の形で渡してください")
    try:
        values = [float(v) for v in numbers.split(",")]
    except ValueError:
        raise ValueError("--price の数が読めません（model=in,out[,cache_read[,cache_write]]）") from None
    if not 2 <= len(values) <= 4 or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError("--price の数は 2〜4 個の 0 以上の数にしてください")
    rate = Rate(values[0], values[1], values[2] if len(values) > 2 else values[0],
                values[3] if len(values) > 3 else None)
    provider = "anthropic" if name.startswith("claude") else "google" if name.startswith("gemini") else "openai"
    return name, Price(provider, rate, "--price", datetime.now(UTC).date().isoformat())


# ─── usage ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Usage:
    """1 回の応答の tokens を、3 社で共通の形にしたもの。

    Attributes:
        input:          キャッシュに当たらない入力
        output:         出力（reasoning を含む）
        reasoning:      出力のうちの reasoning / thinking（記録だけ。金額は output で数える）
        cache_read:     キャッシュから読んだ入力
        cache_write_5m: キャッシュへ書いた入力（5 分）
        cache_write_1h: キャッシュへ書いた入力（1 時間）
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def prompt_tokens(self) -> int:
        """入力の合計（キャッシュを含む）。長い入力の価格帯を選ぶのに使う。"""
        return self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h


def _int(value: Any) -> int:
    """tokens の値を 0 以上の int にする。数でなければ 0。"""
    return max(int(value), 0) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _sub(obj: Any, key: str) -> dict[str, Any]:
    """dict の下の dict を返す。無ければ空の dict。"""
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if isinstance(value, dict) else {}


def normalize_usage(provider: str, raw: dict[str, Any]) -> Usage | None:
    """各社の usage の dict を Usage にする。数える鍵が 1 つも無ければ None。

    3 社で「入力」の意味が違うので、ここでそろえる:
        - Anthropic: input_tokens はキャッシュの読み書きを含まない
        - OpenAI:    prompt_tokens / input_tokens はキャッシュの読みを含む（cached_tokens を引く）。
                     completion_tokens / output_tokens は reasoning を含む
        - Gemini:    promptTokenCount はキャッシュの読みを含む。candidatesTokenCount は thoughts を含まないので足す。
                     道具の結果の入力（toolUsePromptTokenCount）は入力に足す
    """
    if provider == "anthropic":
        if "input_tokens" not in raw and "output_tokens" not in raw:
            return None
        written = _int(raw.get("cache_creation_input_tokens"))
        hour = min(_int(_sub(raw, "cache_creation").get("ephemeral_1h_input_tokens")), written)
        return Usage(
            input=_int(raw.get("input_tokens")), output=_int(raw.get("output_tokens")),
            cache_read=_int(raw.get("cache_read_input_tokens")), cache_write_5m=written - hour, cache_write_1h=hour,
        )
    if provider == "openai":
        if "prompt_tokens" in raw or "completion_tokens" in raw:  # Chat Completions
            prompt, out = _int(raw.get("prompt_tokens")), _int(raw.get("completion_tokens"))
            cached = _int(_sub(raw, "prompt_tokens_details").get("cached_tokens"))
            reasoning = _int(_sub(raw, "completion_tokens_details").get("reasoning_tokens"))
        elif "input_tokens" in raw or "output_tokens" in raw:  # Responses
            prompt, out = _int(raw.get("input_tokens")), _int(raw.get("output_tokens"))
            cached = _int(_sub(raw, "input_tokens_details").get("cached_tokens"))
            reasoning = _int(_sub(raw, "output_tokens_details").get("reasoning_tokens"))
        else:
            return None
        cached = min(cached, prompt)
        return Usage(input=prompt - cached, output=out, reasoning=reasoning, cache_read=cached)
    if provider == "google":
        if "promptTokenCount" not in raw and "candidatesTokenCount" not in raw:
            return None
        prompt = _int(raw.get("promptTokenCount"))
        cached = min(_int(raw.get("cachedContentTokenCount")), prompt)
        thoughts = _int(raw.get("thoughtsTokenCount"))
        return Usage(
            input=prompt - cached + _int(raw.get("toolUsePromptTokenCount")),
            output=_int(raw.get("candidatesTokenCount")) + thoughts, reasoning=thoughts, cache_read=cached,
        )
    return None


def _raw_usage(provider: str, obj: Any) -> dict[str, Any] | None:
    """応答（または stream の 1 件）から usage の dict を探す。

    Anthropic の stream は message_start の message.usage と message_delta の usage に、
    OpenAI の Responses の stream は response.completed の response.usage に分かれて入る。
    """
    if not isinstance(obj, dict):
        return None
    if provider == "google":
        found = obj.get("usageMetadata")
    else:
        found = obj.get("usage")
        if not isinstance(found, dict):
            found = _sub(obj, "message").get("usage") or _sub(obj, "response").get("usage")
    return found if isinstance(found, dict) else None


def _wire_model(obj: Any) -> str | None:
    """応答に書かれたモデル名（実際に答えた版）を探す。出してよい形でなければ None。"""
    if not isinstance(obj, dict):
        return None
    for value in (obj.get("model"), obj.get("modelVersion"), _sub(obj, "message").get("model"),
                  _sub(obj, "response").get("model")):
        if isinstance(value, str) and _SHOWABLE.fullmatch(value):
            return value
    return None


def usage_from_payload(provider: str, data: bytes, content_type: str) -> tuple[Usage | None, str | None]:
    """応答の本文（JSON か SSE）から usage と応答のモデル名を読む。本文はどこにも残さない。

    SSE は data: の行を順に読み、usage の鍵を後のもので上書きしていく（各社とも、後の件が累計か最終値）。

    Returns:
        (Usage か None, 応答のモデル名か None)
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, None
    objects: list[Any] = []
    if "event-stream" in content_type.lower():
        for line in text.splitlines():
            if line.startswith("data:"):
                with contextlib.suppress(ValueError):
                    objects.append(json.loads(line[5:].strip()))
    else:
        with contextlib.suppress(ValueError):
            loaded = json.loads(text)
            # Gemini の REST は、stream でなくても配列で返すことがある。
            objects.extend(loaded if isinstance(loaded, list) else [loaded])
    merged: dict[str, Any] = {}
    model = None
    for obj in objects:
        raw = _raw_usage(provider, obj)
        if raw:
            merged.update({k: v for k, v in raw.items() if v is not None})
        model = _wire_model(obj) or model
    return (normalize_usage(provider, merged) if merged else None), model


def cost_usd(price: Price, usage: Usage) -> float:
    """usage の金額（USD）を返す。"""
    rate = price.rate_for(usage.prompt_tokens)
    write_5m = rate.cache_write_5m if rate.cache_write_5m is not None else rate.input
    write_1h = rate.cache_write_1h if rate.cache_write_1h is not None else write_5m
    return (
        usage.input * rate.input + usage.output * rate.output + usage.cache_read * rate.cache_read
        + usage.cache_write_5m * write_5m + usage.cache_write_1h * write_1h
    ) / 1_000_000


def estimate_usd(price: Price, body_bytes: int, max_output: int) -> float:
    """送る前の、多めの見積もり（USD）。入力は bytes/2 tokens、出力は最大出力まで出たとみなす。"""
    tokens = math.ceil(body_bytes / _BYTES_PER_TOKEN_ESTIMATE)
    return cost_usd(price, Usage(input=tokens, output=max_output))


def _max_output(body: dict[str, Any]) -> int | None:
    """要求に書かれた最大出力の tokens を返す。書かれていなければ None。"""
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if _int(body.get(key)):
            return _int(body.get(key))
    for config_key, key in (("generationConfig", "maxOutputTokens"), ("generation_config", "max_output_tokens")):
        if _int(_sub(body, config_key).get(key)):
            return _int(_sub(body, config_key).get(key))
    return None


# ─── 支出の記録 ───────────────────────────────────────────────────

def default_spend_dir() -> Path:
    """支出の記録を置く場所（リポジトリの data/probe/、gitignore 済み）。"""
    return Path(__file__).resolve().parents[2] / "data" / "probe"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _replace_with_retry(source: Path, target: Path, attempts: int = 6) -> None:
    """os.replace を、Windows の一時的な PermissionError のときだけ短く待って繰り返す。

    WHY: ウイルス対策やエディタが一瞬ファイルを開いていると、Windows の置き換えは PermissionError になる。
         1 回で諦めると、課金した呼び出しを記録できずに止まる（MeterFailure）。
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * 2 ** attempt)


class SpendLedger:
    """有料の計測の支出を、ファイルに 1 呼び出しごとに書き足す。

    ファイル:
        - spent.json:  累計、実行ごと、モデルごとの金額。1 回ごとに一時ファイルへ書いて os.replace で入れ替える
        - calls.jsonl: 1 呼び出し 1 行（時刻、実行、モデル、tokens、金額、状態）。本文は書かない
        - spent.lock:  有料の実行が使っている間だけある（同時に 2 つ流さない）

    WHY 1 回ごとに原子的に書く: 計測は途中で止まりうる（上限、Ctrl+C、時間切れの os._exit）。
         最後にまとめて書くと、止まった回の支出が記録から消え、次の回の上限の判定が甘くなる。
    """

    def __init__(self, directory: Path, *, total_cap_usd: float = TOTAL_CAP_USD) -> None:
        self.directory = directory
        self.path = directory / "spent.json"
        self.calls_path = directory / "calls.jsonl"
        self.lock_path = directory / "spent.lock"
        self._default_cap = total_cap_usd
        self._locked = False
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        """記録を読む。無ければ空の記録を作る（まだ書かない）。"""
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema") != 1:
                raise ValueError("spent.json の schema が 1 ではありません")
            return data
        return {"schema": 1, "total_cap_usd": self._default_cap, "total_usd": 0.0, "by_run": {}, "by_model": {},
                "price_overrides": [], "updated": None}

    @property
    def total_cap_usd(self) -> float:
        return float(self.data.get("total_cap_usd", self._default_cap))

    @property
    def total_usd(self) -> float:
        return float(self.data.get("total_usd", 0.0))

    def run_usd(self, run: str) -> float:
        """実行（名前）ごとの累計。同じ名前で再開した分も含む。"""
        return float(self.data["by_run"].get(run, {}).get("usd", 0.0))

    def acquire(self) -> None:
        """有料の実行の間だけ、ロックのファイルを作る。

        Raises:
            LedgerBusy: すでにある（別の実行中か、前の実行が消し損ねた）
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LedgerBusy(
                f"別の有料の実行が支出の記録を使っています（data/probe/spent.lock、{self._lock_owner()}）。"
                "その実行がもう終わっていれば、そのファイルを消してから流し直してください",
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} since={_now()}\n")
        self._locked = True
        # WHY 読み直す: 作ったときに読んだ値は、ロックを取るまでの間に終わった別の実行の額を含まない。
        #      そのまま書くと、その額を上書きで消してしまう。
        self.data = self._load()

    def _lock_owner(self) -> str:
        """ロックのファイルの持ち主（pid と時刻）を、出してよい形で返す。"""
        try:
            text = self.lock_path.read_text(encoding="utf-8")
        except OSError:
            return "持ち主は読めません"
        pid = re.search(r"pid=(\d+)", text)
        since = re.search(r"since=([0-9T:+\-]+)", text)
        return f"pid {pid[1] if pid else '?'}、{since[1] if since else '?'} から"

    def release(self) -> None:
        """ロックのファイルを消す。自分で作ったときだけ。"""
        if self._locked:
            with contextlib.suppress(FileNotFoundError):
                self.lock_path.unlink()
            self._locked = False

    def _write(self) -> None:
        """spent.json を原子的に書く。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data["updated"] = _now()
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=1, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(tmp, self.path)

    def note_override(self, model: str, price: Price) -> None:
        """--price の一時的な上書きを記録に残す。"""
        self.data.setdefault("price_overrides", []).append({
            "ts": _now(), "model": model, "input": price.rate.input, "output": price.rate.output,
            "cache_read": price.rate.cache_read, "cache_write": price.rate.cache_write_5m,
        })
        self._write()

    def add(self, *, run: str, model: str, usd: float, usage: Usage | None, status: str,
            model_on_wire: str | None = None, http_status: int | None = None) -> None:
        """1 呼び出しを記録する（calls.jsonl へ 1 行、spent.json を書き直す）。

        Args:
            run:           実行の名前
            model:         要求のモデル名（価格表の鍵）
            usd:           数えた金額。usage を拾えなかったときは送る前の見積もり
            usage:         拾えた tokens。拾えなければ None
            status:        usage / http_error / no_usage / transport_error / stream_unfinished
            model_on_wire: 応答に書かれたモデル名
            http_status:   HTTP の状態コード
        """
        ts = _now()
        line = {"ts": ts, "run": run, "model": model, "model_on_wire": model_on_wire, "status": status,
                "http_status": http_status, "usd": round(usd, 9),
                "tokens": None if usage is None else {
                    "input": usage.input, "input_total": usage.prompt_tokens, "output": usage.output,
                    "reasoning": usage.reasoning, "cache_read": usage.cache_read,
                    "cache_write_5m": usage.cache_write_5m, "cache_write_1h": usage.cache_write_1h,
                }}
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.calls_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        self.data["total_usd"] = self.total_usd + usd
        by_run = self.data["by_run"].setdefault(run, {"usd": 0.0, "calls": 0, "uncounted": 0, "first_ts": ts})
        by_run["usd"] += usd
        by_run["calls"] += 1
        by_run["uncounted"] += status in ("no_usage", "transport_error", "stream_unfinished")
        by_run["last_ts"] = ts
        by_model = self.data["by_model"].setdefault(
            model, {"usd": 0.0, "calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
        )
        by_model["usd"] += usd
        by_model["calls"] += 1
        if usage is not None:
            by_model["input"] += usage.input
            by_model["output"] += usage.output
            by_model["cache_read"] += usage.cache_read
            by_model["cache_write"] += usage.cache_write_5m + usage.cache_write_1h
        self._write()


# ─── HTTP の層の包み ──────────────────────────────────────────────

@dataclass
class MeterStats:
    """包みが見た呼び出しの数。画面に出してよい（数だけ）。

    by_provider は提供元（anthropic / openai / google）ごとの sent / counted / ok_responses。
    WHY 提供元ごと: 1 社だけが httpx を通らない（google-genai の非同期は aiohttp を使いうる）と、全体では
         拾えているように見えたまま、その社だけ上限にもかからず数えられない。
    """

    sent: int = 0
    counted: int = 0
    uncounted: int = 0
    http_errors: int = 0
    ok_responses: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, provider: str, key: str) -> None:
        """提供元ごとの数を 1 つ足す。"""
        counts = self.by_provider.setdefault(provider, {"sent": 0, "counted": 0, "ok_responses": 0})
        counts[key] += 1


@dataclass
class _Call:
    """送る前に決めた、1 呼び出しの情報。本文は持たない。"""

    provider: str
    model: str
    price: Price | None
    reserved: float
    selftest_key: str | None = None
    settled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_ACTIVE: UsageMeter | None = None
_ACTIVE_LOCK = threading.Lock()

# スレッドごとの、最後に数えた呼び出し（last_call が返す）。
# WHY: 同期の SDK は、呼んだスレッドの中で HTTP を送って受ける。呼び出し側（リプレイの 1 行）が、
#      自分の呼び出しに答えたモデルの名前を、本文に触れずに知るため。
_THREAD = threading.local()

# os._exit の前に必ず流す後始末（支出の記録のロックを消す、など）。hard_exit と run_tool が流す。
_EXIT_CLEANUPS: list[Callable[[], None]] = []


def provider_of(model: str, prices: dict[str, Price] | None = None) -> str:
    """モデル名の提供元（anthropic / openai / google）を、価格表から返す。

    Raises:
        UnpricedModel: 価格表に無い
    """
    return lookup_price(model, PRICES if prices is None else prices).provider


def tokens_of(usage: Usage) -> dict[str, int]:
    """Usage を、計測の道具が表に出す tokens の dict にする。

    入力は 2 つの名前で区別する:
        - input_uncached: キャッシュに当たらなかった入力（Anthropic の input_tokens と同じ意味）
        - input_total:    入力の合計（input_uncached + cache_read + cache_creation。LangChain の usage_metadata の
                          input_tokens と同じ意味）
    WHY 2 つ: 「in」の意味が SDK と LangChain で違い、キャッシュの効果を比べるときに混ざるため。
    input は input_uncached と同じ値（前からの名前。新しいコードは名前の付いた方を使う）。
    """
    creation = usage.cache_write_5m + usage.cache_write_1h
    return {"input_uncached": usage.input, "input_total": usage.prompt_tokens, "input": usage.input,
            "output": usage.output, "reasoning": usage.reasoning, "cache_read": usage.cache_read,
            "cache_creation": creation}


def last_call() -> dict[str, Any] | None:
    """このスレッドで最後に数えた呼び出しの情報を返し、消す。無ければ None。

    鍵: model、model_on_wire、provider、status、usd、tokens（tokens_of の dict。usage を拾えなかったときは None）。

    WHY tokens も返す: router と filler の raw の呼び出しは本文しか返さないので、計測の道具は呼び出しごとの
         tokens（キャッシュの読み書きを含む）をここからしか知れない。本文には触れない。
    """
    last = getattr(_THREAD, "last", None)
    _THREAD.last = None
    return last


def _classify(request: Any) -> tuple[str, str | None, bool] | None:
    """要求が 3 社の生成の API かどうかを見る。

    Returns:
        (provider, 道から読んだモデル名, stream を頼んだか)。生成の API でなければ None

    Raises:
        UnpricedModel: 3 社のホストへの、知らない道の POST（数えられないので送らない）
    """
    if str(request.method).upper() != "POST":
        return None
    path = request.url.path
    gemini = _GEMINI_ROUTE.search(path)
    if gemini:
        return "google", gemini["model"], gemini["method"] == "streamGenerateContent"
    if path.endswith("/v1/messages"):
        return "anthropic", None, False
    if path.endswith("/chat/completions") or path.endswith("/responses"):
        return "openai", None, False
    host = (request.url.host or "").lower()
    if host in _PROVIDER_HOSTS:
        raise UnpricedModel(f"{_PROVIDER_HOSTS[host]} の、数え方を知らない道へは送りません")
    return None


def _request_body(request: Any) -> tuple[dict[str, Any], int]:
    """要求の本文を dict と bytes 数で返す。読めなければ ({}, Content-Length)。本文は返した先でも保存しない。"""
    try:
        content = request.content
    except Exception:  # noqa: BLE001 - stream の本文（未読）は数えられない
        length = request.headers.get("content-length", "")
        return {}, int(length) if length.isdigit() else 0
    try:
        body = json.loads(content) if content else {}
    except ValueError:
        body = {}
    return (body if isinstance(body, dict) else {}), len(content)


class UsageMeter:
    """有料の計測の間だけ、httpx（と httpx2）の Client.send を包み、usage を支出の記録へ写す。

    with の中だけ差し替え、出るときに必ず戻す。同時に 2 つは入れない。

    送る前:
        - 3 社の生成の API への POST だけを数える（ほかの要求はそのまま通す）
        - モデルを価格表で引き、無ければ UnpricedModel
        - 見積もり（bytes/2 の入力 + 最大出力）を予約し、上限を超えるなら BudgetExceeded
        - 2xx が続いたのに usage を 1 件も拾えていなければ UsageCaptureLost
    受けた後:
        - usage を読めたら、その金額で記録する
        - 2xx 以外で usage が無ければ 0 円（各社とも失敗した要求は課金しない）
        - 2xx で usage を読めない、送信の例外、stream を最後まで読まずに閉じた、は見積もりの金額で記録する
          （WHY: 実際に課金されたかが分からないので、多い側で数えて上限を甘くしない）

    Args:
        ledger:  支出の記録
        run:     実行の名前（支出の記録の鍵。英数字と _.- で 64 字まで）
        max_usd: この実行の上限。同じ名前の過去の分も含めて数える
        prices:  価格表（既定は PRICES）
        overrides: --price の一時的な上書き。使ったことを記録に残す
        assumed_max_output: 最大出力の書かれていない要求で仮定する出力の tokens
        modules: 包む HTTP クライアントのモジュール名（入っていないものは飛ばす）
    """

    def __init__(self, ledger: SpendLedger, run: str, *, max_usd: float, prices: dict[str, Price] | None = None,
                 overrides: dict[str, Price] | None = None,
                 assumed_max_output: int = DEFAULT_ASSUMED_MAX_OUTPUT,
                 modules: tuple[str, ...] = ("httpx", "httpx2")) -> None:
        if not _RUN_NAME.fullmatch(run):
            raise ValueError("実行の名前は、英数字と _.- で 64 字までにしてください")
        if not (max_usd > 0 and math.isfinite(max_usd)):
            raise ValueError("max_usd は正の数にしてください")
        self.ledger = ledger
        self.run = run
        self.max_usd = max_usd
        self.overrides = dict(overrides or {})
        self.prices = {**(PRICES if prices is None else prices), **self.overrides}
        self.assumed_max_output = assumed_max_output
        self.module_names = modules
        self.stats = MeterStats()
        self.wrapped: list[str] = []
        self._pending = 0.0
        self._lock = threading.RLock()
        self._originals: list[tuple[type, str, Any]] = []
        self._selftest_seen: dict[str, Usage | None] = {}
        # 送ったがまだ決着していない呼び出し（id(_Call) → _Call）。打ち切りのときに見積もりで記録する。
        self._inflight: dict[int, _Call] = {}

    # ── 入れる、戻す ──

    def __enter__(self) -> UsageMeter:
        global _ACTIVE
        with _ACTIVE_LOCK:
            if _ACTIVE is not None:
                raise RuntimeError("UsageMeter はすでに使われています（入れ子にできません）")
            _ACTIVE = self
        try:
            self.ledger.acquire()
            # hard_exit は後に足したものから流すので、決着していない呼び出しの記録がロックを消すより先に走る。
            _EXIT_CLEANUPS.append(self.ledger.release)
            _EXIT_CLEANUPS.append(self.settle_inflight)
            for model, price in self.overrides.items():
                self.ledger.note_override(model, price)
            self._install()
            self._selftest()
        except BaseException:
            self._uninstall()
            self.ledger.release()
            self._forget_cleanups()
            with _ACTIVE_LOCK:
                _ACTIVE = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        global _ACTIVE
        try:
            self.settle_inflight()
            self._uninstall()
        finally:
            self.ledger.release()
            self._forget_cleanups()
            with _ACTIVE_LOCK:
                _ACTIVE = None

    def _forget_cleanups(self) -> None:
        for cleanup in (self.ledger.release, self.settle_inflight):
            with contextlib.suppress(ValueError):
                _EXIT_CLEANUPS.remove(cleanup)

    def settle_inflight(self) -> None:
        """送ったまま決着していない呼び出しを、見積もりの額で記録する（status は inflight_at_exit）。

        WHY: 打ち切り（Ctrl+C、時間切れの hard_exit、上限で止めた後に残った並列の呼び出し）のとき、相手はすでに
             受け取って課金しうる。記録しないと、次の実行の上限の判定が甘くなる。後から応答が来ても二重には数えない。
        """
        with self._lock:
            pending = list(self._inflight.values())
        for call in pending:
            self._settle(call, status="inflight_at_exit")

    def _install(self) -> None:
        """入っているモジュールの Client.send / AsyncClient.send を包む。"""
        for name in self.module_names:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            for cls_name, is_async in (("Client", False), ("AsyncClient", True)):
                cls = getattr(module, cls_name)
                original = cls.__dict__["send"]
                self._originals.append((cls, "send", original))
                setattr(cls, "send", _wrap_async(original, module) if is_async else _wrap_sync(original, module))
            self.wrapped.append(name)
        if not self.wrapped:
            raise UsageCaptureLost("包める HTTP クライアント（httpx / httpx2）がありません")

    def _uninstall(self) -> None:
        """包む前の send に戻す。"""
        while self._originals:
            cls, attr, original = self._originals.pop()
            setattr(cls, attr, original)

    def _selftest(self) -> None:
        """包みが効くことを、ネットへ出ずに確かめる（読み終えた応答と、stream の応答の 2 通り）。

        WHY: SDK や httpx の更新で send の呼ばれ方が変わると、包みが黙って素通りになる。課金の前に気づきたい。
        """
        for name in self.wrapped:
            module = importlib.import_module(name)
            payload = json.dumps({"usage": {"input_tokens": _SELFTEST_INPUT_TOKENS, "output_tokens": 1}}).encode()

            class _Chunks(module.SyncByteStream):  # type: ignore[name-defined, misc]
                def __iter__(self) -> Iterator[bytes]:
                    yield payload

            for streamed in (False, True):
                key = f"{name}/{'stream' if streamed else 'read'}"

                def handler(_request: Any, streamed: bool = streamed) -> Any:
                    headers = {"content-type": "application/json"}
                    if streamed:
                        return module.Response(200, headers=headers, stream=_Chunks())
                    return module.Response(200, headers=headers, content=payload)

                with module.Client(transport=module.MockTransport(handler)) as client:
                    request = client.build_request(
                        "POST", "https://selftest.invalid/v1/messages",
                        json={"model": _SELFTEST_MODEL, "max_tokens": 1}, headers={"x-spend-selftest": key},
                    )
                    response = client.send(request, stream=streamed)
                    response.read()
                    response.close()
                usage = self._selftest_seen.get(key)
                if usage is None or usage.input != _SELFTEST_INPUT_TOKENS:
                    raise UsageCaptureLost(f"{key} の自己テストで usage を拾えませんでした")

    # ── 呼び出しごと ──

    def check_capture(self, expected_calls: int | dict[str, int]) -> None:
        """呼び出し側が「有料の呼び出しをこれだけした」と分かっている時点で呼ぶ。

        Args:
            expected_calls: 提供元ごとの回数（{"anthropic": 3, "google": 2}。provider_of で引く）。
                            int なら提供元を問わない全体の回数（提供元の分かる呼び出し側は dict を使う）

        包みがその提供元の usage を 1 件も拾えていない（SDK が別の HTTP クライアントを使った）なら止める。

        Raises:
            UsageCaptureLost
        """
        with self._lock:
            if isinstance(expected_calls, int):
                if expected_calls > 0 and self.stats.counted == 0:
                    raise UsageCaptureLost(
                        f"{expected_calls} 回呼んだのに usage を 1 件も拾えていません（包みが見たのは {self.stats.sent} 回）",
                    )
                return
            for provider, expected in sorted(expected_calls.items()):
                counts = self.stats.by_provider.get(provider, {})
                if expected > 0 and counts.get("counted", 0) == 0:
                    raise UsageCaptureLost(
                        f"{_showable(provider)} を {expected} 回呼んだのに usage を 1 件も拾えていません"
                        f"（包みが見たのは {counts.get('sent', 0)} 回）",
                    )

    def _before(self, request: Any) -> _Call | None:
        """送る前の判定と予約。生成の API でなければ None。"""
        found = _classify(request)
        if found is None:
            return None
        provider, path_model, _streamed = found
        body, size = _request_body(request)
        model = path_model or body.get("model")
        model = model if isinstance(model, str) else None
        if model == _SELFTEST_MODEL:
            return _Call(provider, model, None, 0.0, selftest_key=request.headers.get("x-spend-selftest"))
        price = lookup_price(model, self.prices)
        assert model is not None
        estimate = estimate_usd(price, size, _max_output(body) or self.assumed_max_output)
        with self._lock:
            counts = self.stats.by_provider.get(provider, {})
            if counts.get("ok_responses", 0) >= _ZERO_USAGE_LIMIT and counts.get("counted", 0) == 0:
                raise UsageCaptureLost(
                    f"{provider} の 2xx の応答が {counts['ok_responses']} 回続いたのに usage を拾えていません")
            spent_total = self.ledger.total_usd + self._pending
            spent_run = self.ledger.run_usd(self.run) + self._pending
            room = min(self.ledger.total_cap_usd - spent_total, self.max_usd - spent_run)
            if estimate > room:
                raise BudgetExceeded(
                    f"上限で止めました: 次の見積もり ${estimate:.4f} > 残り ${max(room, 0):.4f}"
                    f"（実行 {self.run} ${self.ledger.run_usd(self.run):.4f} / ${self.max_usd:.2f}、"
                    f"全体 ${self.ledger.total_usd:.4f} / ${self.ledger.total_cap_usd:.2f}）",
                )
            self._pending += estimate
            self.stats.sent += 1
            self.stats.bump(provider, "sent")
            call = _Call(provider, model, price, estimate)
            self._inflight[id(call)] = call
        return call

    def _settle(self, call: _Call, *, status: str, http_status: int | None = None,
                data: bytes | None = None, content_type: str = "") -> None:
        """応答を数えて記録する。1 呼び出しにつき 1 回だけ。"""
        with call.lock:
            if call.settled:
                return
            call.settled = True
        usage, wire_model = (None, None)
        if data is not None:
            usage, wire_model = usage_from_payload(call.provider, data, content_type)
        if call.selftest_key is not None:
            self._selftest_seen[call.selftest_key] = usage
            return
        assert call.price is not None
        if usage is not None:
            status, usd = "usage", cost_usd(call.price, usage)
        elif status in ("response", "stream_unfinished") and http_status is not None \
                and not 200 <= http_status < 300:
            status, usd = "http_error", 0.0
        else:
            status = "no_usage" if status == "response" else status
            usd = call.reserved
        with self._lock:
            self._pending -= call.reserved
            self._inflight.pop(id(call), None)
            if http_status is not None and 200 <= http_status < 300:
                self.stats.ok_responses += 1
                self.stats.bump(call.provider, "ok_responses")
            if status == "usage":
                self.stats.counted += 1
                self.stats.bump(call.provider, "counted")
            elif status == "http_error":
                self.stats.http_errors += 1
            else:
                self.stats.uncounted += 1
            self.ledger.add(run=self.run, model=call.model, usd=usd, usage=usage, status=status,
                            model_on_wire=wire_model, http_status=http_status)
        _THREAD.last = {"model": call.model, "model_on_wire": wire_model, "provider": call.provider,
                        "status": status, "usd": usd, "tokens": None if usage is None else tokens_of(usage)}

    def _after_response(self, call: _Call, response: Any, module: Any) -> None:
        """応答を受けた後。読み終えていれば今数え、stream なら読み終えたときに数える。"""
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        try:
            data = response.content
        except module.ResponseNotRead:
            response.stream = _tee(module, response, lambda raw, done: _guarded(
                self._settle_stream, call, response, raw, done, status, content_type))
            return
        self._settle(call, status="response", http_status=status, data=data, content_type=content_type)

    def _settle_stream(self, call: _Call, response: Any, raw: bytes | None, done: bool, status: int,
                       content_type: str) -> None:
        """stream を読み終えた（または途中で閉じた）ときに数える。"""
        if not done or raw is None:
            self._settle(call, status="stream_unfinished", http_status=status)
            return
        self._settle(call, status="response", http_status=status, data=_decode(response, raw),
                     content_type=content_type)


def _decode(response: Any, raw: bytes) -> bytes | None:
    """stream で貯めた生の bytes を、Content-Encoding に従って戻す。戻せなければ None。

    WHY: stream の層には圧縮されたままの bytes が流れる（展開は iter_bytes の側でする）。
         httpx の展開の部品は非公開で、しかも応答ごとに 1 つを使い回す（読み終えた後は使えない）。
         標準の zlib で戻せる gzip / deflate だけを戻し、ほかの形は「拾えなかった」に倒す。
    """
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding in ("", "identity"):
        return raw
    if encoding not in ("gzip", "deflate"):
        return None
    try:
        # wbits=MAX_WBITS|32 は gzip と zlib の頭を自動で見分ける。生の deflate は -MAX_WBITS で読み直す。
        return zlib.decompress(raw, zlib.MAX_WBITS | 32)
    except zlib.error:
        try:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
        except zlib.error:
            return None


def _tee(module: Any, response: Any, on_done: Callable[[bytes | None, bool], None]) -> Any:
    """応答の stream を、流れる bytes を貯める stream に包む。読み終えたら on_done(bytes, True)。

    読み終える前に閉じられたら on_done(None, False)。上限を超えて貯められなかったら on_done(None, True)。
    """
    inner = response.stream
    buffer = bytearray()
    state = {"finished": False, "overflow": False}

    def keep(chunk: bytes) -> None:
        if not state["overflow"]:
            if len(buffer) + len(chunk) > _TEE_LIMIT_BYTES:
                state["overflow"] = True
                buffer.clear()
            else:
                buffer.extend(chunk)

    def finish(done: bool) -> None:
        if state["finished"]:
            return
        state["finished"] = True
        on_done(None if state["overflow"] or not done else bytes(buffer), done)

    if isinstance(inner, module.AsyncByteStream) and not isinstance(inner, module.SyncByteStream):
        class _TeeAsync(module.AsyncByteStream):  # type: ignore[misc, name-defined]
            async def __aiter__(self):  # type: ignore[no-untyped-def]
                async for chunk in inner:
                    keep(chunk)
                    yield chunk
                finish(True)

            async def aclose(self) -> None:
                try:
                    await inner.aclose()
                finally:
                    finish(False)

        return _TeeAsync()

    class _TeeSync(module.SyncByteStream):  # type: ignore[misc, name-defined]
        def __iter__(self) -> Iterator[bytes]:
            for chunk in inner:
                keep(chunk)
                yield chunk
            finish(True)

        def close(self) -> None:
            try:
                inner.close()
            finally:
                finish(False)

    return _TeeSync()


def _guarded(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """包みの中の処理を流し、想定外の Exception は MeterFailure に変える（SDK に再試行させない）。"""
    try:
        return function(*args, **kwargs)
    except SpendStop:
        raise
    except Exception as exc:  # noqa: BLE001 - 型を変えて SDK の except Exception を抜けさせる
        raise MeterFailure(f"包みの中で失敗しました: {describe_failure(exc)}") from exc


def _wrap_sync(original: Callable[..., Any], module: Any) -> Callable[..., Any]:
    """Client.send の包み。包みの外（UsageMeter が無いとき）はそのまま通す。"""
    @functools.wraps(original)
    def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        meter = _ACTIVE
        call = _guarded(meter._before, request) if meter is not None else None
        if call is None:
            return original(client, request, *args, **kwargs)
        assert meter is not None
        try:
            response = original(client, request, *args, **kwargs)
        except BaseException:
            _guarded(meter._settle, call, status="transport_error")
            raise
        _guarded(meter._after_response, call, response, module)
        return response

    return send


def _wrap_async(original: Callable[..., Any], module: Any) -> Callable[..., Any]:
    """AsyncClient.send の包み。"""
    @functools.wraps(original)
    async def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        meter = _ACTIVE
        call = _guarded(meter._before, request) if meter is not None else None
        if call is None:
            return await original(client, request, *args, **kwargs)
        assert meter is not None
        try:
            response = await original(client, request, *args, **kwargs)
        except BaseException:
            _guarded(meter._settle, call, status="transport_error")
            raise
        _guarded(meter._after_response, call, response, module)
        return response

    return send


# ─── 本文を出さない後始末（計測の道具に共通） ─────────────────────

def _showable(value: object) -> str:
    """出してよい形（空白のない短い ASCII）ならそのまま、そうでなければ長さだけを返す。"""
    text = str(value)
    return text if _SHOWABLE.fullmatch(text) else f"<{len(text)} 文字>"


def emit(line: str) -> None:
    """道具の出力をここへ集める。行は自分で組み立てた集計だけにする（本文を渡さない）。

    WHY: Windows のコンソールの文字コードで出せない文字があっても、集計の行を落とさない。
    """
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def describe_failure(exc: BaseException) -> str:
    """例外を、型と、このパッケージの中の行番号だけの 1 行にする。

    WHY: 例外の文言には本文が入りうる（ast.literal_eval や JSON の例外は入力の断片を、API の例外は
         応答の本文を含む）。型と自分のコードの行番号だけでも、どこで何が起きたかは辿れる。
    """
    package = Path(__file__).resolve().parent
    places = []
    for frame in traceback.extract_tb(exc.__traceback__):
        path = Path(frame.filename)
        with contextlib.suppress(ValueError):
            places.append(f"{path.resolve().relative_to(package.parent).as_posix()}:{frame.lineno}")
    shown = ", ".join(places[-3:])
    return f"{type(exc).__name__}" + (f" ({shown})" if shown else "")


@contextlib.contextmanager
def quiet_failures() -> Iterator[None]:
    """with の中で、捕まえ損ねた例外（本体とスレッド）を型と行番号だけで出す。出るときに戻す。"""
    import logging

    previous_hook, previous_thread_hook = sys.excepthook, threading.excepthook
    previous_unraisable = sys.unraisablehook
    previous_last_resort = logging.lastResort

    def hook(_type: type[BaseException], exc: BaseException, _tb: object) -> None:
        emit(f"[失敗] {describe_failure(exc)}")

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None:
            emit(f"[失敗] スレッドで {describe_failure(args.exc_value)}")

    def unraisable_hook(unraisable: Any) -> None:
        # WHY: __del__ や後始末の中の例外は、既定では文言ごと標準エラーへ出る（本文を含みうる）。
        emit(f"[失敗] 後始末で {getattr(unraisable.exc_type, '__name__', '例外')}")

    sys.excepthook, threading.excepthook = hook, thread_hook
    sys.unraisablehook = unraisable_hook
    # WHY: root logger に handler が無いとき、logging は WARNING 以上を stderr へ素で出す（本文を含みうる）。
    logging.lastResort = None
    try:
        yield
    finally:
        sys.excepthook, threading.excepthook = previous_hook, previous_thread_hook
        sys.unraisablehook = previous_unraisable
        logging.lastResort = previous_last_resort


def hard_exit(code: int) -> NoReturn:
    """後始末（支出の記録のロックなど）を流してから、残りのスレッドを待たずに終える。

    WHY: 時間切れで打ち切るとき、普通の終了ではスレッドの join を待つ間も課金が続きうる。
         ただ os._exit だけだとロックのファイルが残り、次の有料の実行が LedgerBusy で止まる。
    """
    for cleanup in reversed(_EXIT_CLEANUPS):
        with contextlib.suppress(Exception):
            cleanup()
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def run_tool(main: Callable[[], int], *, name: str) -> NoReturn:
    """計測の道具の入口。main を流し、結果に応じた終了コードで即座に終える。

    WHY os._exit: 打ち切った呼び出しのスレッドが残ると、普通の終了では interpreter が join を待ち、
         その間も API を呼び続けて課金が続きうる。結果を出したら残りを待たずに終える。
         例外は型と行番号だけを出す（describe_failure）。SpendStop の文言は自分で作った数字だけなので出す。
    """
    code = EXIT_FAILED
    try:
        with quiet_failures():
            try:
                code = main()
            except SpendStop as exc:
                code = EXIT_SPEND_STOP
                emit(f"[{name}] 止めました: {type(exc).__name__}: {exc}")
            except SystemExit as exc:
                # sys.exit() と sys.exit(None) は正常の終わり。文言の付いた sys.exit は値を含みうるので出さずに 1 にする。
                code = EXIT_OK if exc.code is None else exc.code if isinstance(exc.code, int) else EXIT_FAILED
            except KeyboardInterrupt:
                code = 130
                emit(f"[{name}] Ctrl+C で止めました")
            except Exception as exc:  # noqa: BLE001 - 最後の砦。値を出さずに型と行番号だけを知らせる
                code = EXIT_FAILED
                emit(f"[{name}] 想定外のエラー: {describe_failure(exc)}")
    finally:
        hard_exit(code)


# ─── コマンドライン ───────────────────────────────────────────────

def _cmd_show(directory: Path) -> int:
    """支出の記録の要約を出す（金額と件数だけ）。"""
    ledger = SpendLedger(directory)
    emit(f"全体: ${ledger.total_usd:.4f} / 上限 ${ledger.total_cap_usd:.2f}"
         f"（更新 {ledger.data.get('updated') or 'なし'}）")
    for run, row in sorted(ledger.data["by_run"].items()):
        emit(f"  実行 {_showable(run)}: ${row['usd']:.4f}  呼び出し {row['calls']}"
             f"（usage を拾えず見積もりで数えた {row.get('uncounted', 0)}）")
    for model, row in sorted(ledger.data["by_model"].items()):
        emit(f"  モデル {_showable(model)}: ${row['usd']:.4f}  呼び出し {row['calls']}"
             f"  in {row['input']} out {row['output']} cache_read {row['cache_read']} cache_write {row['cache_write']}")
    for item in ledger.data.get("price_overrides", []):
        emit(f"  --price の上書き: {_showable(item.get('model'))} ({item.get('ts')})")
    return EXIT_OK


def _cmd_prices() -> int:
    """価格表と出典を出す。"""
    emit("モデル / 入力 / 出力 / キャッシュ読み / キャッシュ書き(5m) — USD / 100 万 tokens")
    for model, price in PRICES.items():
        rate = price.rate
        write = "-" if rate.cache_write_5m is None else f"{rate.cache_write_5m:g}"
        line = f"  {model}: {rate.input:g} / {rate.output:g} / {rate.cache_read:g} / {write}"
        if price.long_rate is not None:
            line += (f"（入力 {price.long_over_tokens:,} 超: {price.long_rate.input:g} / {price.long_rate.output:g}"
                     f" / {price.long_rate.cache_read:g}）")
        emit(line)
        emit(f"    出典 {price.source_url}（{price.as_of} 確認）{'  ' + price.note if price.note else ''}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """``python -m lab_lounge.spend <show|prices>``。"""
    parser = argparse.ArgumentParser(prog="python -m lab_lounge.spend", description="有料の計測の支出と価格表")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="支出の記録の要約（金額と件数だけ）")
    show.add_argument("--dir", type=Path, default=None, help="支出の記録の場所（既定は data/probe/）")
    sub.add_parser("prices", help="価格表と出典")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.command == "show":
        return _cmd_show(args.dir or default_spend_dir())
    return _cmd_prices()


if __name__ == "__main__":
    run_tool(main, name="spend")
