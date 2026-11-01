"""
preflight.py — 配信前の点検

責務:
  - 配信を始める前に「いま何が動いていて、何が壊れているか」を経路ごとの行で返し、最後に判定の行を出す
  - 本番と同じ関数を、フォールバックの内側で直接呼んで判定する
  - 本番経路からは呼ばれない。このモジュールを消しても配信の挙動は変わらない

【WHY】
  L2 は配信を止めないために、LLM の失敗を except で握りつぶして定型の経路へ落とす。
  そのため SDK やモデルが壊れても、配信は続いたまま挙手や協働だけが無音で消え、モックの
  テストも緑のままになる。ここでは None や例外を「正常な失敗」として扱わず、すべて FAIL にする。

【使い方】
  # 課金なし。設定の嘘を見つける（実効モデル、キーの有無、版の一致、配信文脈の鮮度など）
  uv run --frozen --all-extras --no-sync python -m lab_lounge.preflight --static --profile dev

  # 少額の課金あり。全経路へ実際に 1 回ずつ流して受理を確かめる
  uv run --frozen --all-extras --no-sync python -m lab_lounge.preflight --paid --profile dev

  版の一致を見るときは --no-sync か .venv の python を直接使う。uv が実行前に同期すると、
  壊れた環境が黙って直されて検査の意味がなくなる。

【終了コード】
  0:  必須の経路がすべて PASS（WARN は含んでよい）
  1:  必須の経路に FAIL がある。想定外のエラーで止まったときも 1（要約の代わりに「想定外のエラー」の行を出す）
  2:  実行を拒否した（本番ストリームへの書き込み、.env による LLM 接続先の上書き）
  3:  前提サービスがないので、課金する前に止めた
  64: コマンドラインの誤り
  130: Ctrl+C で止めた
  最後の「判定:」の行は、同じ結果を言葉で示す（終了コードは画面に出ないので）。

【出力の約束】
  API キーは set / unset だけを出し、値、発話本文、接続先の host は出さない。
  点検の出力は配信中のコンソールや記録に貼られることがあるため。API の応答やモデルの出力に由来する
  例外も、本文に入力のエコーや発話が入りうるので、画面には型と status、短い type / code / param だけを
  出す。全文は gitignore 済みの logs/preflight/ に残す。
  手元の例外も、文言に載った相手の応答と URL は切り、設定された接続先の host は [HOST] に置き換える
  （loopback は調査の手がかりなので残す）。

  [FAIL*] の行と要約の「（別枠）」は、終了コードに影響しない FAIL の印（配信に出ないキャラなど）。

【--paid の副作用】
  - dev ストリームへ preflight-<時刻> の stream_id で publish するので、dev の C2 と V2 の HUD に
    点検の発話が入る。C2 の recent は stream_id で絞らないので、以後の dev の記憶にも混ざりうる
  - data/audio/ に WAV が残る（gitignore 済み）。音は鳴らさず、OBS の立ち絵も切り替えない
  - --allow-prod を付けると、これらが本番の正史に恒久的に残る
"""

import argparse
import ast
import concurrent.futures
import contextvars
import importlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_MISSING_SERVICE = 3
EXIT_USAGE = 64

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PROD_STREAM = "aibyss:events"

# 配信に出るキャラ。残り（ruka / aruka）は別枠で記録し、終了コードには影響させない。
REQUIRED_CHARACTERS = ("mimi", "chisame", "sakura", "octamaid")

# 協働の点検で流す組。--collab one は先頭だけ、all は全部。
# どのプロバイダも呼び出し側と協働先の両方に 1 回ずつ現れるように選ぶ。
# 先頭は、強制 tool_choice と 2 段目の応答で壊れやすい Anthropic を呼び出し側にした組。
_COLLAB_PAIRS = (("sakura", "chisame"), ("mimi", "sakura"), ("chisame", "mimi"))

# 親プロセスから継承すると LLM の向き先や認証を変えてしまう環境変数。
# WHY: Claude Code などのツールから起動したプロセスは、ツール自身の接続先を環境変数で引き継ぐ。
#      そのまま流すと、点検は通っても本番とは別の接続先を測ったことになる。
INHERITED_LLM_VARS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "GOOGLE_GEMINI_BASE_URL",
)

# LangSmith はこの 4 つのどれでも有効になる（langsmith は *_TRACING_V2 を先に見る）。
_TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)

_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "VOYAGE_API_KEY",
    "L2_PORCUPINE_ACCESS_KEY",
    "L2_OBS_WS_PASSWORD",
)

# SDK の例外文には、マスクされたキーの断片や認証ヘッダが混ざることがある。形で見分けて伏せる。
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-*]{4,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{6,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-]+)*"),
    re.compile(r"AIza[0-9A-Za-z_\-]{6,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"lsv2_[A-Za-z0-9_\-]{6,}"),
    re.compile(r"pa-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"(?i)\b(authorization|x-api-key|api[_-]?key|token|password|secret)\s*[=:]\s*\S+"),
    re.compile(r"(?<=://)[^/\s@]+@"),
)
_LAN_ADDRESS = re.compile(
    r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}\b"
)

# 接続先の URL を持つ設定。host はライブラリや本番の例外文に入るので、型ではなく設定の値で伏せる。
_ENDPOINT_ENV_NAMES = (
    "REDIS_URL",
    "L2_C2_URL",
    "L2_C2_URL_READONLY",
    "L2_TTS_VOICEVOX_URL",
    "L2_TTS_IRODORI_URL",
    "L2_OBS_WS_URL",
)

# 値を表示してよい設定の許可リスト。秘匿値が混ざらないよう、名前を明示して絞る。
_BOOL_FLAGS = (
    "L2_USE_REAL_LLM",
    "L2_USE_REAL_TTS",
    "L2_USE_REAL_STT",
    "L2_ENABLE_TOOLS",
    "L2_ENABLE_RAG",
    "L2_USE_C2_RETRIEVER",
    "L2_USE_C2_RECENT",
    "L2_USE_INTENT_GATE",
    "L2_USE_LLM_ROUTER",
    "L2_USE_HANDRAISE",
    "L2_FILLER_ENABLED",
)
_OTHER_FLAGS = (
    "L2_STT_PROVIDER",
    "L2_WAKE_BACKEND",
    "L2_VAD_BACKEND",
    "L2_DEFAULT_SPEAKER",
    "L2_LLM_REASONING_EFFORT",
    "L2_LLM_MAX_OUTPUT_TOKENS",
)

# 配信の経路が int() / float() でそのまま読む設定。壊れていると、読む経路が例外で止まる。
# 今の設定でその経路を通るかどうかは _numeric_env_use が決める。ベンチや単発の道具だけが読むものは入れない。
_NUMERIC_ENV = (
    ("L2_LLM_TIMEOUT_SEC", float),
    ("L2_ROUTER_LLM_TIMEOUT_SEC", float),
    ("L2_LLM_MAX_RETRIES", int),
    ("L2_AGENT_INVOKE_TIMEOUT_SEC", float),
    ("L2_C2_RETRIEVER_TOP_K", int),
    ("L2_C2_RETRIEVER_BUDGET_MS", float),
    ("L2_C2_RECENT_TOP_K", int),
    ("L2_RAG_TOP_K", int),
    # tts.py が import の時点で読むので、ダミー TTS でも全ターンが例外になる。
    ("L2_TTS_IRODORI_SHORT_CHARS", int),
    ("L2_TTS_IRODORI_SEC_PER_CHAR", float),
    ("L2_TTS_IRODORI_MIN_SEC", float),
    # irodori の合成で読む。
    ("L2_TTS_IRODORI_TIMEOUT_SEC", float),
    ("L2_TTS_IRODORI_NUM_STEPS", int),
    ("L2_TTS_IRODORI_SEED", int),
    # VOICEPEAK の合成で読む。次の 2 つはワーカーのスレッドの中なので、例外で死ぬと合成の呼び出しが永久に待つ。
    ("L2_VOICEPEAK_RETRY_WAIT_SEC", float),
    ("L2_VOICEPEAK_MAX_RETRIES", int),
    ("L2_VOICEPEAK_OUTPUT_RETRY_INTERVAL_SEC", float),
    ("L2_VOICEPEAK_OUTPUT_RETRY_MAX_ATTEMPTS", int),
    # run_loop の起動で読む。
    ("L2_RECORD_SECONDS", float),
    ("L2_WAKE_TIMEOUT", float),
    # リスナーの生成で読む。
    ("L2_SILENCE_THRESHOLD", float),
    ("L2_VAD_AGGRESSIVENESS", int),
    ("L2_CONTEXT_WINDOW_SEC", float),
    ("L2_CONTEXT_MAX_CHARS", int),
    # Dispatcher の生成で読む。
    ("L2_HANDRAISE_LAPSE_SEC", float),
    ("L2_HANDRAISE_LAPSE_UTTERANCE_COUNT", int),
    # faster-whisper の転写のたびに読む。bg-continuous では録音のスレッドが例外で無音のまま止まる。
    ("L2_STT_COMPRESSION_RATIO_THRESHOLD", float),
    ("L2_STT_LOG_PROB_THRESHOLD", float),
    ("L2_STT_NO_SPEECH_THRESHOLD", float),
)

# 本番が空を「指定なし」と読む設定。.env.example の説明どおり空のまま置けるので、壊れた値に数えない。
_NUMERIC_ENV_BLANK_OK = ("L2_TTS_IRODORI_SEED",)

# llm.py の get_llm_reasoning_effort の説明にある有効値。
_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "none")

# 既定が真のフラグ。本番は os.environ.get(name, "true") で読むので、空を書くと既定へ戻らず偽になる。
_DEFAULT_TRUE_FLAGS = (
    "L2_USE_HANDRAISE",
    "L2_FILLER_ENABLED",
    "L2_USE_C2_RECENT",
    "L2_STT_HALLUCINATION_FILTER",
    "L2_LOG_TO_FILE",
)

# python-dotenv がダブルクォートの中の \a \b \f \n \r \t \v から作る文字。
_DOTENV_CONTROL_CHARS = re.compile(r"[\a\b\f\n\r\t\v]")

_PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "tavily": ("TAVILY_API_KEY",),
}

_DEFAULT_PORTS = {"http": 80, "https": 443, "redis": 6379, "ws": 80, "wss": 443}

# 配信文脈が古いと、過去の配信の話を「今日の話」として全キャラの prompt に入れてしまう。
_STREAM_CONTEXT_STALE_DAYS = 14

# 出力が途中で切れたことを示す finish_reason。切れた JSON は本番で別の経路へ落ちる。
_TRUNCATED_FINISH = {"length", "max_tokens", "MAX_TOKENS"}

# WHY: create_agent の既定は上限が事実上ない。応答が構造化出力の検証を通らないと再試行を
#      続けるので、壁時計の打ち切りの後も裏で API を呼び続けて課金が止まらない。
#      挨拶のターンは 2 段階、協働の 1 往復は 4 段階で足りる。余裕を残しつつ、検証に落ち続けたときの
#      課金の上限を抑える値にする。
_AGENT_RECURSION_LIMIT = 10

# timeout のない経路の壁時計。Agent の 90 秒 + フォールバックの 80 秒 + TTS を上回る値にする。
_TAVILY_DEADLINE_SEC = 30
_EMBED_DEADLINE_SEC = 30
_PIPELINE_DEADLINE_SEC = 420

_MAX_DETAIL_CHARS = 200

# 配信前の合否に数えられない実行の行に入れる文言。判定の行はこれを目印に、合格へ「ただし」を添える。
_NOT_FOR_GATE = "配信前の合格には使えません"

# 判定の行に出す言葉。数字だけだと、窓の表示や貼られた出力から意味を読めない。
_VERDICT_WORDS = {
    EXIT_OK: "合格",
    EXIT_FAILED: "不合格",
    EXIT_REFUSED: "拒否",
    EXIT_MISSING_SERVICE: "前提サービスなし",
    EXIT_USAGE: "引数の誤り",
}

_ROUTER_PROBE_SYSTEM = "次の発話の宛先を mimi か none の 1 語で答えてください。"
_ROUTER_PROBE_USER = "ミミ様、聞こえますか"
_ROUTER_PROBE_ANSWERS = ("mimi", "none")
_FILLER_PROBE_USER = "（独り言）"

# API の応答に由来する例外を出すモジュール。本文に入力のエコーやモデルの出力が入りうる。
# langchain と pydantic は、構造化出力の検証に落ちた応答（発話）を本文に含める。
_API_ERROR_MODULES = {
    "openai",
    "anthropic",
    "google",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "requests",
    "tavily",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_tavily",
    "pydantic",
    "pydantic_core",
}

# pydantic の検証エラーの input_value= には、検証に落ちた値がそのまま入る。langchain はこの本文を
# 手元の ValueError に包んで投げることがあるので、例外の型では見分けられない。
# tts.py は VOICEPEAK の stderr / stdout も同じ形で手元の RuntimeError へ入れる。CLI の出力は本文を含みうる。
_INPUT_VALUE = re.compile(r"\b(input_value|stderr|stdout)=.*")

# 手元の例外の文言に続く、相手の応答やモデルの出力。ここから後ろを切る。全文は問題ログに残る。
#   HTTP nnn:        tts.py がサイドカーの応答本文を続ける（サーバは入力をエコーしうる）
#   invalid literal: tts.py の int() がモデルの返した値を続ける（speed や emotion の欄）
_PEER_TEXT = re.compile(r"(HTTP \d{3}|invalid literal for [^:]+):.*", re.S)

# 手元の例外の文言に入る接続先の URL。設定にない接続先の host も、これで伏せられる。
# scheme の長さを区切るのは、区切り文字の多い長い文字列で照合の回数が 2 乗に増えないようにするため。
_URL_IN_TEXT = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]{0,30}://[^\s)\"']+")

# ログの引数のうち、そのまま表示してよい形。
_SHOWABLE_LOG_ARG = re.compile(r"[\x21-\x7e]{1,40}")

# モデルが返したツール名をそのまま表示しない。崩れた応答では名前の欄に本文が入りうる。
_TOOL_NAME = re.compile(r"[A-Za-z0-9_\-]{1,64}")


def _is_truthy(value: str | None) -> bool:
    """L2 の他のフラグと同じ規則で真偽を読む。

    WHY: 本番は strip しないので、前後に空白のある値は偽になる。ここだけ甘く読むと、本番がダミーで
         動く設定を緑にしてしまう。
    """
    return (value or "").lower() in ("1", "true", "yes")


def _memory_tool_enabled() -> bool:
    """本番で retrieve_memory ツールが動くかどうかを返す。

    WHY: _retrieval_node はグラフに配線されていないので（graph.py の _build_pipeline_graph）、記憶の検索は
         Agent のツールからしか呼ばれない。Agent は L2_ENABLE_TOOLS が真のときだけ作られ、retrieve_memory は
         L2_ENABLE_RAG が真のときだけ登録される。どちらかが偽なら、埋め込みも C2 も呼ばれない。
    """
    return _is_truthy(os.environ.get("L2_ENABLE_TOOLS")) and _is_truthy(os.environ.get("L2_ENABLE_RAG"))


def _c2_retriever_enabled() -> bool:
    """本番が C2 の検索を呼ぶかどうかを返す。

    L2_USE_C2_RETRIEVER は graph.py の _build_retriever_from_env と同じ規則で読む（"true" だけが真）。
    本番で C2 を読むのは retrieve_memory だけなので、ツールが動かない設定では値によらず呼ばれない。
    """
    use_c2 = os.environ.get("L2_USE_C2_RETRIEVER", "false").lower() == "true"
    return use_c2 and _memory_tool_enabled()


def _handraise_enabled() -> bool:
    """dispatcher.py の設定と同じ規則で読む（既定は有効）。"""
    return os.environ.get("L2_USE_HANDRAISE", "true").lower() in ("1", "true", "yes")


# ─── 出力 ─────────────────────────────────────────────────────────

def _home_pattern() -> re.Pattern[str] | None:
    """ホームのパスを見つける正規表現を返す。伏せても意味がないときは None を返す。

    WHY: OSError の本文はパスを repr で持つので、Windows では区切りが 2 文字になる。/ で書かれたパスや、
         小文字になったパスも例外文に入る。str(Path.home()) の全文一致では当たらないので、区切りと
         大文字小文字によらず見つける。
         UID だけを指定したコンテナは HOME=/ になり、置き換えると区切りごと ~ になって表が読めなくなる。
         ホームが決まらない環境で例外を出すと、どの行も表示できない。どちらも置き換えない。
    """
    try:
        home = str(Path.home()).rstrip("\\/")
    except RuntimeError:
        return None
    # 残りが空かドライブだけならルート。Windows で USERPROFILE が空だと "." になる。
    if home in ("", ".") or re.fullmatch(r"[A-Za-z]:", home):
        return None
    return re.compile(r"[\\/]+".join(re.escape(part) for part in re.split(r"[\\/]+", home)), re.IGNORECASE)


def redact(text: str) -> str:
    """出力前に秘匿値、キーの形をした文字列、LAN アドレス、ホームのパスを伏せる。

    Args:
        text: 表示しようとしている文字列（例外文を含みうる）

    Returns:
        伏せた後の文字列
    """
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        # 極端に短い値を置換すると無関係な文字まで消えるので、最低長を設ける。
        # 空白を除いて数えるのは、空白だけの値を strip した空文字で置換すると、全文字の間に印が入るため。
        if not value or len(value.strip()) < 4:
            continue
        # errors: POSIX の非 UTF-8 の環境変数は孤立サロゲートになり、既定の quote は例外を出して全出力が止まる。
        quoted = urllib.parse.quote(value, safe="", errors="replace")
        quoted_plus = urllib.parse.quote_plus(value, errors="replace")
        for variant in (value, value.strip(), quoted, quoted_plus):
            text = text.replace(variant, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = _LAN_ADDRESS.sub("[LAN]", text)
    home = _home_pattern()
    return home.sub("~", text) if home else text


def _endpoint_hosts() -> list[str]:
    """設定された接続先の host を、長い順に返す。loopback は伏せる意味がないので除く。

    WHY: box と box.lan を設定していて短い方を先に置き換えると、[HOST].lan と host の一部が残る。
    """
    values = (os.environ.get(name, "") for name in _ENDPOINT_ENV_NAMES)
    hosts = {location[0] for value in values if (location := _url_location(value)) and not _is_loopback_url(value)}
    return sorted(hosts, key=len, reverse=True)


def _one_line(text: str) -> str:
    """伏せてから 1 行に整えて切り詰める。

    WHY: 順序が逆だと、切り詰めで途中までになった秘匿値が全文一致に当たらず、先頭が残る。
         host は例外文と理由の行にだけ掛ける。redact に入れると、host が c2 や redis のときに行の名前や
         固定文まで置き換わり、全文を残す問題ログからも消える。前後が名前の続きなら別の語なので残す。
         ASCII: 日本語の文字が直後に続くと \\w に一致して、host を見逃すため。
    """
    text = redact(text)
    for host in _endpoint_hosts():
        text = re.sub(rf"(?<![\w.-]){re.escape(host)}(?![\w-])", "[HOST]", text, flags=re.IGNORECASE | re.ASCII)
    return " ".join(text.split())[:_MAX_DETAIL_CHARS]


def _emit(line: str) -> None:
    """すべての出力をここへ集め、伏せ忘れの経路を作らない。

    WHY: redact が例外を出すと、最後の砦の行も含めて何も表示されないまま終わり、原因を読めない。
         伏せられなかった行は値を含みうるので、そのままは出さず、落ちたことだけを 1 行で知らせる。
    """
    try:
        text = redact(line)
    except Exception as exc:  # noqa: BLE001 - 出力の失敗で点検の結果を読めなくしない
        text = f"[preflight] 伏せる処理に失敗したので、この行を省きました ({type(exc).__name__})"
    print(text, flush=True)


def _is_api_error(exc: BaseException) -> bool:
    """例外が API の応答（モデルの出力を含む）に由来するかどうかを返す。"""
    if getattr(exc, "status_code", None) is not None or getattr(exc, "response", None) is not None:
        return True
    # __mro__: langgraph の ToolInvocationError のように、本文を持つ例外の子が別のモジュールにあるため。
    if any(cls.__module__.split(".")[0] in _API_ERROR_MODULES for cls in type(exc).__mro__):
        return True
    # langchain_openai は応答の error を dict や pydantic のまま ValueError へ渡す（説明の文ではなく応答そのもの）。
    return isinstance(exc, ValueError) and bool(exc.args) and not isinstance(exc.args[0], str)


def _describe_exc(exc: BaseException) -> str:
    """例外を 1 行に要約する。

    API の応答に由来する例外は本文を出さない。モデレーションや入力検証のエラーは、発話や
    system prompt の一部を本文に含めて返すことがあるため。SDK の使い方が壊れたときの
    TypeError などは本文が要るので、手元で起きた例外は本文も出す。ただし文言に載った相手の応答と
    接続先の URL は切る。全文は呼び出し側の note_exception が問題ログへ残す。
    """
    name = type(exc).__name__
    if not _is_api_error(exc):
        text = _PEER_TEXT.sub(r"\1", _INPUT_VALUE.sub(r"\1=<略>", str(exc)))
        message = _one_line(_URL_IN_TEXT.sub("<URL>", text))
        return f"{name}: {message}" if message else name

    fields = [name]
    status = getattr(exc, "status_code", None)
    if status is None:
        # httpx の HTTPStatusError は status を response の下に持つ。C2 の 422（k が上限超え）と 500 を読み分けるのに要る。
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        fields.append(f"status={status}")
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    for attr in ("type", "code", "param"):
        value = getattr(exc, attr, None)
        if value is None and isinstance(error, dict):
            value = error.get(attr)
        if isinstance(value, (str, int)) and value != "" and len(str(value)) < 64:
            fields.append(f"{attr}={value}")
    return _one_line(" ".join(fields))


@dataclass
class Check:
    """点検 1 行ぶんの結果。"""

    name: str
    status: str  # PASS / FAIL / WARN / SKIP / INFO
    detail: str
    required: bool


class Report:
    """点検の結果を集め、1 行ずつその場で表示する。

    その場で出すのは、API を順に呼ぶ間も進み具合が見えるようにするため。
    """

    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, name: str, status: str, detail: str = "", *, required: bool = True) -> None:
        """結果を 1 行記録する。

        Args:
            name:     経路の名前（例: "main/mimi"）
            status:   PASS / FAIL / WARN / SKIP / INFO
            detail:   補足。秘匿値や発話本文を入れない
            required: False なら FAIL でも終了コードに影響させない（別枠のキャラ、副 C2、--no-pipeline の TTS）
        """
        self.checks.append(Check(name, status, detail, required))
        mark = f"{status}*" if status == "FAIL" and not required else status
        _emit(f"[{mark:<5}] {name}: {detail}" if detail else f"[{mark:<5}] {name}")

    def has_required_failure(self) -> bool:
        """必須の経路に FAIL があるかを返す。"""
        return any(c.status == "FAIL" and c.required for c in self.checks)

    def exit_code(self) -> int:
        """終了コードを返す。"""
        return EXIT_FAILED if self.has_required_failure() else EXIT_OK

    def print_summary(self) -> None:
        """件数、失敗の一覧、トークン数の合計を表示する。"""
        counts = {s: sum(c.status == s for c in self.checks) for s in ("PASS", "FAIL", "WARN", "SKIP")}
        _emit("")
        _emit(
            f"PASS {counts['PASS']} / FAIL {counts['FAIL']} / WARN {counts['WARN']} / SKIP {counts['SKIP']}"
            f" / 本応答の tokens in={self.input_tokens} out={self.output_tokens}"
        )
        for check in self.checks:
            if check.status == "FAIL":
                note = "" if check.required else "（別枠）"
                _emit(f"  FAIL{note} {check.name}")


def _record_message(record: logging.LogRecord) -> str:
    """ログの本文を返す。書式が合わなくても例外にしない。

    WHY: ここで例外を出すと、ログを出した本番の呼び出し側まで巻き込んで点検の結果が変わる。
    """
    try:
        return record.getMessage()
    except Exception:  # noqa: BLE001 - 書式の誤りの KeyError や、引数の __str__ が投げる例外も受ける
        return str(record.msg)


class _LogCapture(logging.Handler):
    """lab_lounge のログをメモリへ集め、画面には出さない。

    WHY: 本番のログには発話本文が全文で入る（text=%r）。判定には使うが表示はしない。
         None を返す関数の失敗理由もログにしか残らないので、FAIL の説明に使う。
         WARNING 以上の全文は、調査できるように gitignore 済みのファイルへ残す。
    """

    def __init__(self, problem_log: Path | None = None) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []
        self.problem_log = problem_log
        self.wrote_problems = False

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("lab_lounge"):
            return
        self.records.append(record)
        if record.levelno >= logging.WARNING:
            self.write_problem(f"{record.levelname} {record.name}: {_record_message(record)}")

    def note_exception(self, name: str, exc: BaseException) -> None:
        """画面には要約しか出さない例外の全文を、問題ログへ残す。"""
        self.write_problem(f"{name}: {type(exc).__name__}: {exc}")

    def write_problem(self, line: str) -> None:
        """問題ログへ 1 行書く。伏せてから書き、書けなくても点検は止めない。"""
        if self.problem_log is None:
            return
        try:
            self.problem_log.parent.mkdir(parents=True, exist_ok=True)
            # errors: 発話の孤立サロゲートで書き込みが例外になると、本番の呼び出し側まで巻き込む。
            with self.problem_log.open("a", encoding="utf-8", errors="replace") as stream:
                stream.write(redact(line) + "\n")
            self.wrote_problems = True
        except OSError:
            # ログが書けなくても点検の判定は続ける。
            pass

    def mark(self) -> int:
        """今の位置を返す。これ以降のログだけを見るために使う。"""
        return len(self.records)

    def messages_since(self, mark: int) -> list[str]:
        """mark 以降のログ本文を返す。判定に使い、表示はしない。"""
        return [_record_message(r) for r in self.records[mark:]]

    def problem_counts(self, mark: int) -> tuple[int, int]:
        """mark 以降の (ERROR 以上の件数, WARNING の件数) を返す。本文は発話を含みうるので返さない。"""
        levels = [record.levelno for record in self.records[mark:]]
        errors = sum(level >= logging.ERROR for level in levels)
        return errors, sum(level >= logging.WARNING for level in levels) - errors

    def last_problem(self, mark: int) -> str:
        """mark 以降で最後の WARNING 以上のログを、表示してよい形にして返す。無ければ空文字。"""
        for record in reversed(self.records[mark:]):
            if record.levelno >= logging.WARNING:
                return _safe_log_message(record)
        return ""

    def problems_since(self, mark: int) -> list[str]:
        """mark 以降の WARNING 以上のログを、表示してよい形にして重複なく返す。"""
        return list(dict.fromkeys(_safe_log_message(r) for r in self.records[mark:] if r.levelno >= logging.WARNING))


def _safe_log_message(record: logging.LogRecord) -> str:
    """ログを表示してよい 1 行にする。例外は要約へ、長い文字列は長さへ置き換える。

    WHY: ログの引数には API の例外本文や発話が入る。定型文の部分だけで原因は読める。
    """
    def safe(arg: Any) -> Any:
        if isinstance(arg, BaseException):
            return _describe_exc(arg)
        # slug やモデル名のような空白のない短い ASCII だけを見せる。発話と、発話を含む dict や list の repr は
        # 40 文字以下でもこれに当たらない。数値はこの形なのでそのまま残り、status=%d のような書式も崩れない。
        shown = str(arg)
        return arg if _SHOWABLE_LOG_ARG.fullmatch(shown) else f"<{len(shown)} 文字>"

    args = record.args
    try:
        text = str(record.msg) % tuple(safe(a) for a in args) if isinstance(args, tuple) else str(record.msg)
    except Exception:  # noqa: BLE001 - 理由の表示のために点検を止めない
        text = str(record.msg)
    return _one_line(text)


@contextmanager
def _captured_logs(problem_log: Path | None = None) -> Iterator[_LogCapture]:
    """root logger と warnings を差し替え、点検中のログをコンソールへ流さない。"""
    root = logging.getLogger()
    capture = _LogCapture(problem_log)
    previous_handlers, previous_level = list(root.handlers), root.level
    root.handlers = [capture]
    root.setLevel(logging.INFO)
    # pydantic の警告は input_value= に値を含むので、stderr へ素で出さない。
    logging.captureWarnings(True)
    try:
        yield capture
    finally:
        logging.captureWarnings(False)
        root.handlers = previous_handlers
        root.setLevel(previous_level)


# ─── 環境の準備 ───────────────────────────────────────────────────

def _url_location(value: str) -> tuple[str, int | None] | None:
    """URL の host と port を返す。形が不正なら None。

    WHY: パスワードに / や # を含む URL は urlsplit の port で ValueError になり、その例外文に
         パスワードが全文で出る。ここで捕まえて、値を外へ出さない。
    """
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    return (host, port) if host else None


def _is_loopback_url(value: str) -> bool:
    """値が loopback を指す URL かどうかを返す。"""
    location = _url_location(value)
    return location is not None and location[0] in ("127.0.0.1", "::1", "localhost")


def clean_llm_env(env_files: Iterable[Path] = (), *, allow_loopback: bool = False) -> list[str]:
    """継承した LLM の接続先を環境から外す。.env が指定していれば実行を拒否する。

    preflight の prepare_env から呼ぶ。LLM を呼ぶ道具を増やすときも、冒頭で同じように呼ぶ。

    Args:
        env_files:      これから読む env ファイル
        allow_loopback: True かつ pytest の実行中に限り、loopback を指す値を残す
                        （テストで SDK を偽サーバへ向けるための口で、main() は使わない）

    Returns:
        環境から外した変数名（値は返さない）

    Raises:
        SystemExit: .env が接続先を上書きしている（終了コード 2）
    """
    for path in env_files:
        # 空の値も止める。SDK は base_url="" を既定へ戻さず、呼び出しごとに接続エラーになる。
        # Windows の環境変数は大文字小文字を区別しないので、小文字の綴りも同じ変数になる。
        # 値のない行（None）は本番の読み込みでも環境に入らないので止めない。
        found = [k for k, v in dotenv_values(path).items() if k.upper() in INHERITED_LLM_VARS and v is not None]
        if found:
            _emit(f"[preflight] {path.name} が {', '.join(found)} を設定しています。外してから流してください。")
            raise SystemExit(EXIT_REFUSED)

    keep_loopback = allow_loopback and "PYTEST_CURRENT_TEST" in os.environ
    removed: list[str] = []
    for name in INHERITED_LLM_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        if keep_loopback and _is_loopback_url(value):
            continue
        del os.environ[name]
        removed.append(name)
    if removed:
        _emit(f"[preflight] 継承した変数を外しました: {', '.join(removed)}")
    return removed


def env_files_for(profile: str, repo_root: Path) -> list[Path]:
    """プロファイルに応じて読む env ファイルを返す。

    run_dev.ps1 / run_prod.ps1 と同じ規則で、dev は .env の上に .env.dev を重ねる。
    PYTHON_DOTENV_DISABLED が立っていれば何も読まない（テストの隔離用）。
    """
    if os.environ.get("PYTHON_DOTENV_DISABLED", "").strip().lower() in ("1", "true", "t", "yes", "y"):
        return []
    names = [".env", ".env.dev"] if profile == "dev" else [".env"]
    return [repo_root / name for name in names if (repo_root / name).is_file()]


def prepare_env(profile: str, repo_root: Path) -> list[Path]:
    """接続先を掃除してから env ファイルを読み込む。

    Args:
        profile:   "dev" または "prod"
        repo_root: env ファイルのあるリポジトリのルート

    Returns:
        読み込んだ env ファイル

    Raises:
        SystemExit: .env が LLM の接続先を上書きしている（終了コード 2）
    """
    files = env_files_for(profile, repo_root)
    clean_llm_env(files)
    # WHY: run スクリプトと同じくファイルを正とする。ファイルにある名前については、前の実行の
    #      環境変数が残っていても同じ結果になるように、既存の値を上書きする。
    for path in files:
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ[key] = value
    return files


def silence_side_outputs(*, trace: bool = False) -> dict[str, str | None]:
    """課金する実行の前に、点検の外へ実発話が出る経路を止める。

    Args:
        trace: True なら LangSmith への送信は止めない

    Returns:
        止める前の値。本番の設定を表へ出すのに使う

    WHY: 点検の 1 ターンを本番の LangSmith プロジェクトに混ぜず、実発話も送らない。
         L2_DEBUG_ARTIFACTS は追跡済みの logs/ を上書きするので、公開リポジトリへ混ざりうる。
         SDK の import が langsmith のキャッシュへ値を入れるので、run_static より先に呼ぶ必要がある。
    """
    production = {name: os.environ.get(name) for name in ("L2_DEBUG_ARTIFACTS", *_TRACING_VARS)}
    os.environ["L2_DEBUG_ARTIFACTS"] = "false"
    if not trace:
        for name in _TRACING_VARS:
            os.environ[name] = "false"
    return production


def _report_silenced_outputs(report: Report, production: dict[str, str | None]) -> None:
    """--paid が止めた設定を、本番の値で表へ出す。

    WHY: run_static は止めた後の環境を読むので、そのままだと本番で実発話が logs/ や LangSmith へ出る設定が
         --paid の表から消える。
    """
    if _is_truthy(production.get("L2_DEBUG_ARTIFACTS")):
        report.add("flag/L2_DEBUG_ARTIFACTS", "WARN", "本番は true。追跡済みの logs/ に実発話が書かれます（この実行では止めています）")
    # --trace のときは止めていないので、run_static の行がそのまま本番の状態を示している。
    enabled = [name for name in _TRACING_VARS if _is_truthy(production.get(name)) and not _is_truthy(os.environ.get(name))]
    if enabled:
        report.add("flag/LANGSMITH", "INFO", f"本番は有効 ({', '.join(enabled)})。この実行では止めています")


# ─── --static: 課金なしの点検 ─────────────────────────────────────

def effective_filler_model(character: Any) -> str:
    """filler.py の _generate_filler_text と同じ優先順位でモデルを決める。"""
    return character.filler_model or os.environ.get("L2_LLM_FILLER_MODEL", "gpt-5.4-nano")


def _canonical(name: str) -> str:
    """パッケージ名を比較できる形にそろえる（PEP 503）。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def _marker_holds(marker: str) -> bool:
    """環境マーカーを評価する。評価できないものは「対象外」として扱う。

    WHY: 誤って不足と報告すると、正常な環境で配信の起動を止めてしまう。
    """
    try:
        from packaging.markers import Marker
    except ImportError:
        return False
    try:
        return Marker(marker).evaluate()
    except Exception:  # noqa: BLE001 - packaging は形の誤りと評価の失敗で別々の例外を出す
        return False


def locked_closure(lock_data: dict[str, Any]) -> dict[str, set[str]]:
    """uv.lock から、全 extras を入れたときに必要なパッケージと版を求める。

    Args:
        lock_data: uv.lock を tomllib で読んだ dict

    Returns:
        {正規化したパッケージ名: 許容する版の集合}。プロジェクト自身は含まない。
        Python の版で分岐するパッケージは複数の版を持つ。
    """
    packages: dict[str, list[dict[str, Any]]] = {}
    root: dict[str, Any] | None = None
    for package in lock_data.get("package", []):
        packages.setdefault(_canonical(package["name"]), []).append(package)
        if package.get("source", {}).get("editable") == ".":
            root = package
    if root is None:
        return {}

    def direct_deps(package: dict[str, Any], extras: Iterable[str] | None = None) -> list[dict[str, Any]]:
        optional = package.get("optional-dependencies", {})
        chosen = optional.keys() if extras is None else extras
        deps = list(package.get("dependencies", []))
        for extra in chosen:
            deps.extend(optional.get(extra, []))
        return deps

    wanted: dict[str, set[str]] = {}
    seen: set[tuple[str, str | None, tuple[str, ...]]] = set()
    stack = direct_deps(root)
    while stack:
        dep = stack.pop()
        marker = dep.get("marker")
        if marker and not _marker_holds(marker):
            continue
        name = _canonical(dep["name"])
        extras = tuple(dep.get("extra", ()))
        key = (name, dep.get("version"), extras)
        if key in seen:
            continue
        seen.add(key)
        for package in packages.get(name, []):
            if dep.get("version") and package["version"] != dep["version"]:
                continue
            wanted.setdefault(name, set()).add(package["version"])
            stack.extend(direct_deps(package, extras))
    return wanted


def _check_versions(report: Report, repo_root: Path) -> None:
    """実際に入っている版と uv.lock を突き合わせる。

    WHY: 素の `uv sync` は extras を消す。音声や OBS の依存が消えても、L2 は握りつぶして
         起動してしまうので、起動前にここで止める。
    """
    lock_path = repo_root / "uv.lock"
    if not lock_path.is_file():
        report.add("deps/uv.lock", "WARN", "uv.lock がないので版の一致を確かめられません")
        return
    try:
        wanted = locked_closure(tomllib.loads(lock_path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001 - 壊れた lock も、形の変わった lock も同じ扱いにする
        # 例外のまま抜けると、残りの静的な点検が 1 つも流れない。
        report.add("deps/uv.lock", "FAIL", f"lock を読めません ({type(exc).__name__})")
        return
    if not wanted:
        # 0 件のまま PASS にすると、検知器が何も見ていない状態を見逃す。
        report.add("deps/uv.lock", "FAIL", "lock からプロジェクトの依存を読み取れません")
        return
    missing: list[str] = []
    mismatched: list[str] = []
    for name, versions in sorted(wanted.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if installed not in versions:
            mismatched.append(f"{name} {installed} (lock: {'/'.join(sorted(versions))})")
    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"不足 {len(missing)} 件: {', '.join(missing[:8])}")
        if mismatched:
            parts.append(f"版違い {len(mismatched)} 件: {', '.join(mismatched[:8])}")
        report.add("deps/uv.lock", "FAIL", f"{' / '.join(parts)}。uv sync --frozen --all-extras で直ります")
        return
    report.add("deps/uv.lock", "PASS", f"{len(wanted)} 件が lock と一致")


def _check_imports(report: Report, providers: set[str], capture: _LogCapture | None = None) -> None:
    """本番が遅延 import する SDK を実際に import する。

    WHY: 失敗しても本番は握りつぶすので、ここで見ないと気づけない。
         - anthropic 0.125 以降は名前の長いファイルを含み、LongPathsEnabled=0 の Windows で venv の
           パスが約 126 文字を超えると import が失敗する。版は一致するので _check_versions では
           見えない。router と filler は「パッケージなし」の WARNING で None を返すだけになる
         - create_react_agent が消えると、協働先がログなしでツールなしの呼び出しへ落ちる
    """
    modules = {"openai": ("openai", "langchain_openai"),
               "anthropic": ("anthropic", "langchain_anthropic"),
               "google": ("google.genai", "langchain_google_genai")}
    targets = [name for provider in sorted(providers) for name in modules.get(provider, ())]
    if _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        targets.extend(("langchain.agents", "langchain_tavily", "langgraph.prebuilt"))

    failed = False
    for target in targets:
        try:
            module = importlib.import_module(target)
            if target == "langgraph.prebuilt":
                getattr(module, "create_react_agent")
        except Exception as exc:  # noqa: BLE001 - import の失敗の種類は問わない
            failed = True
            if capture is not None:
                # 画面が型の名前だけになる例外（pydantic など）でも、調査の手がかりを残す。
                capture.note_exception(f"import/{target}", exc)
            report.add(f"import/{target}", "FAIL", _describe_exc(exc))
    if not failed:
        report.add("import", "PASS", f"{len(targets)} 件")


def _intent_route_enabled() -> bool:
    """意図判定のモデルが使われるかどうかを返す。

    WHY: 挙手と承認の判定は L2_USE_INTENT_GATE を見ずに同じモデルを呼ぶ。ゲートが無効でも
         挙手が有効なら、この経路が壊れると挙手だけが無音で消える。
    """
    from .router import is_intent_gate_enabled

    return is_intent_gate_enabled() or _handraise_enabled()


def _check_models(report: Report, characters: list[Any]) -> set[str]:
    """経路ごとの実効モデルを表示し、使うプロバイダの集合を返す。"""
    from .filler import _detect_provider, is_filler_enabled
    from .pipeline import _get_llm_mode
    from .router import _INTENT_GATE_MODEL, _LLM_ROUTER_MODEL, _is_llm_router_enabled

    providers: set[str] = set()
    for character in characters:
        _, provider, model = _get_llm_mode(character)
        filler = effective_filler_model(character)
        providers.add(provider)
        # 無効なら呼ばれないので、使わない経路のキーがないだけで --paid を止めない。
        if is_filler_enabled():
            providers.add(_detect_provider(filler))
        report.add(f"model/{character.slug}", "INFO", f"main={provider}:{model} filler={filler}")

    # ask_character.py はこれがあるとキャラの設定を無視し、全員の協働先を L2_LLM_PROVIDER のこのモデルにする。
    # ツールが無効なら読まれないので、使わない経路のキーがないだけで止めない。
    override = os.environ.get("L2_ASK_CHARACTER_MODEL_OVERRIDE", "")
    if override and _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        override_provider = os.environ.get("L2_LLM_PROVIDER", "openai")
        providers.add(override_provider)
        report.add("model/collab-override", "WARN", f"協働先は全員 {override_provider}:{override} を使います")

    router_model = os.environ.get("L2_LLM_ROUTER_MODEL", _LLM_ROUTER_MODEL)
    intent_model = os.environ.get("L2_INTENT_GATE_MODEL", _INTENT_GATE_MODEL)
    report.add("model/router", "INFO", f"{router_model} enabled={_is_llm_router_enabled()}")
    report.add("model/intent", "INFO", f"{intent_model} enabled={_intent_route_enabled()}")
    if _is_llm_router_enabled():
        providers.add(_detect_provider(router_model))
    if _intent_route_enabled():
        providers.add(_detect_provider(intent_model))
    # 埋め込みは OpenAI 固定（retriever._embed）なので、--characters で OpenAI のキャラを外しても要る。
    if _memory_tool_enabled():
        providers.add("openai")
    return providers


def _check_keys(report: Report, providers: set[str], file_names: set[str]) -> None:
    """API キーの有無を確かめる。値は読み上げない。

    Args:
        report:     結果の記録先
        providers:  使うプロバイダ
        file_names: env ファイルに書かれている名前。空なら出どころは判定しない

    WHY: ツールやターミナルから継承したキーで緑になると、Explorer から起動する本番では
         キーがなくて落ちる。env ファイルにないキーは WARN で知らせる。
    """
    real_llm = _is_truthy(os.environ.get("L2_USE_REAL_LLM"))
    targets = set(providers)
    # 検索も実 LLM の Agent 経路からしか呼ばれないので、LLM のキーと同じ規則で見る。
    if _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        targets.add("tavily")
    for provider in sorted(targets):
        names = _PROVIDER_KEYS.get(provider, ())
        # 本番の LangChain は最初に定義された名前だけを読む。空の GOOGLE_API_KEY は GEMINI_API_KEY を隠し、
        # Agent の構築が ValidationError で落ちる（raw の genai は空を飛ばすので、filler だけは動く）。
        first = next((name for name in names if name in os.environ), None)
        present = [first] if first and os.environ[first] else []
        if not present:
            status = "FAIL" if real_llm else "WARN"
            if any(os.environ.get(name) for name in names):
                # キーはあるので、足すのではなく空の行を消す必要がある。
                report.add(f"key/{provider}", status, f"unset ({first} が空で、残りの名前は読まれません。この行を消してください)")
            else:
                report.add(f"key/{provider}", status, f"unset ({' / '.join(names)})")
        elif file_names and not any(name in file_names for name in present):
            report.add(f"key/{provider}", "WARN", "set（env ファイルになく、親プロセスから継承）")
        else:
            report.add(f"key/{provider}", "PASS", "set")

    for name in ("LANGSMITH_API_KEY", "L2_PORCUPINE_ACCESS_KEY", "L2_OBS_WS_PASSWORD"):
        report.add(f"key/{name}", "INFO", "set" if os.environ.get(name) else "unset")


def _check_silently_replaced_values(report: Report) -> None:
    """本番が例外にせず、黙って別の値で動く設定を知らせる。値の行は _check_flags が先に出している。"""
    from .characters import CHARACTER_REGISTRY
    from .llm import get_llm_reasoning_effort

    speaker = os.environ.get("L2_DEFAULT_SPEAKER")
    # 空は既定と同じ octamaid になるので知らせない。
    if speaker and speaker not in CHARACTER_REGISTRY:
        report.add("flag/L2_DEFAULT_SPEAKER", "WARN", "未知のキャラです。本番は octamaid へ落とします")
    tokens = (os.environ.get("L2_LLM_MAX_OUTPUT_TOKENS") or "").strip()
    if tokens:
        # 0 と負の値は本番の説明どおりの「上限なし」なので知らせない。読めない値だけが無言で安全弁を外す。
        try:
            int(tokens)
        except ValueError:
            report.add("flag/L2_LLM_MAX_OUTPUT_TOKENS", "WARN", "整数として読めないので、出力の上限なしで動きます")
    effort = get_llm_reasoning_effort()
    if effort and effort not in _REASONING_EFFORTS:
        report.add("flag/L2_LLM_REASONING_EFFORT", "WARN", "想定外の値です。API が拒否すると、ツールなしの経路と導入セリフが落ちます")
    for name in _DEFAULT_TRUE_FLAGS:
        if os.environ.get(name) == "":
            report.add(f"flag/{name}", "WARN", "空は既定（真）へ戻らず、偽として読まれます")


def _check_flags(report: Report, file_names: Iterable[str] = ()) -> None:
    """動作を切り替えるフラグを許可リストの範囲で表示し、読めない値を知らせる。

    Args:
        report:     結果の記録先
        file_names: env ファイルに書かれている名前。空なら出どころは判定しない

    WHY: PowerShell の中で run_*.ps1 を流すと、値がその端末に残る。継承した値で測って緑でも、
         Explorer などから起動する本番ではフラグが偽になり、ツールや協働が無音で消える。
    """
    for name in _BOOL_FLAGS:
        value = os.environ.get(name)
        if value is None:
            report.add(f"flag/{name}", "INFO", "(未設定)")
        elif value.lower() not in ("true", "false", "1", "0", "yes", "no", ""):
            # L2_USE_REAL_LLM=ture は黙って偽になり、全キャラがダミー応答を返す。前後の空白も同じ。
            report.add(f"flag/{name}", "WARN", f"{value} は真偽として読めません。偽として扱われます")
        else:
            report.add(f"flag/{name}", "INFO", value)
    # graph.py はこの 2 つを "true" だけで読むので、1 や yes は偽になり、記憶の経路だけが無音で消える。
    for name in ("L2_USE_C2_RETRIEVER", "L2_USE_C2_RECENT"):
        value = os.environ.get(name, "")
        if _is_truthy(value) and value.lower() != "true":
            report.add(f"flag/{name}", "WARN", "true 以外の真は graph.py で偽として扱われます")
    # 記憶の検索は Agent のツールからしか呼ばれない。緑のままだと、RAG が効いていると思い込む。
    if _is_truthy(os.environ.get("L2_ENABLE_RAG")) and not _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        report.add("flag/L2_ENABLE_RAG", "WARN", "L2_ENABLE_TOOLS が偽なので、RAG は効きません")
    for name in _OTHER_FLAGS:
        value = os.environ.get(name)
        report.add(f"flag/{name}", "INFO", value if value is not None else "(未設定)")
    if _is_truthy(os.environ.get("L2_DEBUG_ARTIFACTS")):
        report.add("flag/L2_DEBUG_ARTIFACTS", "WARN", "true。追跡済みの logs/ に実発話が書かれます")
    if os.environ.get("L2_WAKE_BACKEND") == "porcupine" and not os.environ.get("L2_PORCUPINE_ACCESS_KEY"):
        report.add("flag/L2_WAKE_BACKEND", "WARN", "porcupine ですが L2_PORCUPINE_ACCESS_KEY がありません")
    _check_silently_replaced_values(report)
    names_in_files = set(file_names)
    inherited = [
        name for name in _BOOL_FLAGS
        if names_in_files and _is_truthy(os.environ.get(name)) and name not in names_in_files
    ]
    if inherited:
        report.add("flag/inherited", "WARN", f"env ファイルになく、親プロセスから継承しています: {', '.join(inherited)}")


def _check_env_quoting(report: Report, file_names: Iterable[str]) -> None:
    """ダブルクォートの中の円記号が制御文字に変わった値を知らせる。値は表示しない。

    WHY: "C:\\reference\\voices.json" の \\r と \\v は、python-dotenv（この点検と run_loop の load_dotenv）では
         制御文字になり、run_*.ps1 では書いたとおりに読まれる。同じファイルでも起動の仕方で値が変わる。
         引用符を外せば、どちらでも同じ値になる。
    """
    names = sorted(name for name in file_names if _DOTENV_CONTROL_CHARS.search(os.environ.get(name) or ""))
    if names:
        report.add("env/quoting", "WARN", f"{', '.join(names)} の値に制御文字があります。ダブルクォートの中の円記号が変わった形です。引用符を外してください")


def _check_real_modes(
    report: Report, profile: str, file_names: set[str], *, paid: bool, with_pipeline: bool = True,
) -> None:
    """実 LLM と実 TTS が有効かを確かめる。

    dev の --static はダミー起動を許す（OSS 利用者のダミー起動の想定）。--paid はダミーのままだと
    課金だけして本番の経路を通らず、本番ストリームでダミーが正当な場面もないので FAIL にする。
    真の値が env ファイルになく親プロセスから継承しているだけなら、Explorer から起動する本番では
    ダミーになるので WARN で知らせる。

    L2_NO_PLAY が真だと run_loop は再生スレッドもフィラーも作らず、音を一切出さない。.env.example に
    行があるので試験で真にしたまま残りやすい。本番では FAIL、dev は試験で使うので WARN にする。

    Args:
        report:        結果の記録先
        profile:       "dev" または "prod"
        file_names:    env ファイルに書かれている名前
        paid:          --paid の実行か
        with_pipeline: False（--paid --no-pipeline）なら TTS を通らないので、L2_USE_REAL_TTS は見ない
    """
    if _is_truthy(os.environ.get("L2_NO_PLAY")):
        report.add("mode/L2_NO_PLAY", "FAIL" if profile == "prod" else "WARN", "真なので、音声を再生しません")
    if not (paid or profile == "prod"):
        return
    for name in ("L2_USE_REAL_LLM", "L2_USE_REAL_TTS") if with_pipeline else ("L2_USE_REAL_LLM",):
        if not _is_truthy(os.environ.get(name)):
            report.add(f"mode/{name}", "FAIL", "真ではないので、本番の経路を通りません")
        elif file_names and name not in file_names:
            report.add(f"mode/{name}", "WARN", "env ファイルになく、親プロセスから継承しています")


def _tts_engine_use(provider: str, characters: list[Any] | None) -> tuple[bool, bool]:
    """実 TTS でこのエンジンを使うかどうかを返す。

    Args:
        provider:   CharacterConfig.tts_provider の値
        characters: 点検するキャラ。None なら全キャラ

    Returns:
        (使うか, 必須のキャラが使うか)
    """
    if not _is_truthy(os.environ.get("L2_USE_REAL_TTS")):
        return False, False
    if characters is None:
        from .characters import get_all_characters

        characters = get_all_characters()
    users = [character.slug for character in characters if character.tts_provider == provider]
    return bool(users), any(slug in REQUIRED_CHARACTERS for slug in users)


def _raw_llm_routes_enabled() -> bool:
    """ルーター、意図判定、filler のどれかが動くかを返す。どれも get_llm_timeout_config("router") を読む。"""
    from .filler import is_filler_enabled
    from .router import _is_llm_router_enabled

    return _is_llm_router_enabled() or _intent_route_enabled() or is_filler_enabled()


def _numeric_env_use(name: str, characters: list[Any] | None) -> tuple[bool, bool]:
    """_NUMERIC_ENV の設定を、今の設定で本番が読むかどうかを返す。

    Args:
        name:       設定の名前
        characters: 点検するキャラ。None なら全キャラ

    Returns:
        (読むか, 読む経路が必須か)。必須でなければ、FAIL でも終了コードに影響させない

    WHY: 使わない経路の値で配信の起動を止めない。ここで挙げていない名前は、どの構成でも読む
         （tts.py の import、1 ターンの _get_rag_mode、run_loop の起動、リスナーと Dispatcher の生成）。
         wake backend は起動時の引数で決まり、ここからは見えないので、リスナーと Dispatcher の値は読むものとして扱う。
         挙手の 2 つは L2_USE_HANDRAISE が偽でも Dispatcher の生成で読むので、このフラグでは分けない。
    """
    real_llm = _is_truthy(os.environ.get("L2_USE_REAL_LLM"))
    if name == "L2_LLM_TIMEOUT_SEC":
        return real_llm, True
    if name == "L2_ROUTER_LLM_TIMEOUT_SEC":
        return _raw_llm_routes_enabled(), True
    if name == "L2_LLM_MAX_RETRIES":
        return real_llm or _raw_llm_routes_enabled(), True
    if name == "L2_AGENT_INVOKE_TIMEOUT_SEC":
        return real_llm and _is_truthy(os.environ.get("L2_ENABLE_TOOLS")), True
    if name in ("L2_C2_RETRIEVER_TOP_K", "L2_C2_RETRIEVER_BUDGET_MS", "L2_C2_RECENT_TOP_K"):
        return _c2_retriever_enabled(), True
    if name in ("L2_TTS_IRODORI_TIMEOUT_SEC", "L2_TTS_IRODORI_NUM_STEPS", "L2_TTS_IRODORI_SEED"):
        return _tts_engine_use("irodori_vd", characters)
    if name.startswith("L2_VOICEPEAK_"):
        return _tts_engine_use("voicepeak", characters)
    if name == "L2_VAD_AGGRESSIVENESS":
        from .wake_word import _get_vad_backend

        return _get_vad_backend() == "webrtc", True
    if name.startswith("L2_STT_"):
        return os.environ.get("L2_STT_PROVIDER", "openai") == "faster-whisper", True
    return True, True


def _check_numeric_env(report: Report, characters: list[Any] | None = None) -> None:
    """本番が int() / float() でそのまま読む設定を確かめる。値は表示しない。

    WHY: 壊れた値は読む経路を例外で止めるか、握りつぶされて機能を無音で落とす。今の設定で読む経路なら FAIL、
         読まない経路なら誤りだけを WARN で知らせる。
    """
    for name, parse in _NUMERIC_ENV:
        value = os.environ.get(name)
        if value is None or (name in _NUMERIC_ENV_BLANK_OK and not value.strip()):
            continue
        try:
            parse(value)
        except ValueError:
            used, required = _numeric_env_use(name, characters)
            if used:
                report.add(f"flag/{name}", "FAIL", f"{parse.__name__} として読めません", required=required)
            else:
                report.add(f"flag/{name}", "WARN", f"{parse.__name__} として読めません（今の設定では本番が読みません）")


def _check_personas(report: Report, characters: list[Any]) -> None:
    """人格の prompt を、本番と同じ関数で読めるかを確かめる。

    WHY: prompt がないと graph.py は WARNING だけで続行し、人格なしで応答する。空のファイルも同じで、
         この場合は --paid の本応答も通ってしまう。graph.py は FileNotFoundError しか受けないので、
         UTF-8 でないファイルはそのキャラのターンをすべて例外で落とす。
    """
    from .characters import _PROMPTS_DIR, load_system_prompt

    for character in characters:
        name = f"persona/{character.slug}"
        required = character.slug in REQUIRED_CHARACTERS
        try:
            text = load_system_prompt(character)
        except FileNotFoundError:
            report.add(name, "FAIL", f"{character.system_prompt_file} もサンプルもありません", required=required)
            continue
        except (OSError, UnicodeDecodeError) as exc:
            report.add(name, "FAIL", f"{character.system_prompt_file} を読めません ({type(exc).__name__})", required=required)
            continue
        # BOM は strip で落ちない。PowerShell 5 の Out-File は空でも BOM を書く。
        if not text.replace("\ufeff", "").strip():
            report.add(name, "FAIL", f"{character.system_prompt_file} が空です", required=required)
        elif not (_PROMPTS_DIR / character.system_prompt_file).is_file():
            report.add(name, "WARN", f"{character.system_prompt_file} がなくサンプルで代替します")


def _check_tts_config(
    report: Report, characters: list[Any], capture: _LogCapture | None = None, *, with_pipeline: bool = True,
) -> None:
    """通信せずに分かる TTS の設定を確かめる。合成はしない。

    Args:
        report:        結果の記録先
        characters:    点検するキャラ
        capture:       読めなかった理由の全文を残す先。画面の行はパスや登録名を含まない定型文にする
        with_pipeline: False（--paid --no-pipeline）なら TTS を通らないので、行だけ残して合否に数えない
    """
    if not _is_truthy(os.environ.get("L2_USE_REAL_TTS")):
        return
    from .tts import _resolve_irodori_voice

    for character in characters:
        name = f"tts-config/{character.slug}"
        required = character.slug in REQUIRED_CHARACTERS and with_pipeline
        if character.tts_provider == "irodori_vd":
            try:
                voice = _resolve_irodori_voice(character.tts_voice)
            except Exception as exc:  # noqa: BLE001 - 壊れた JSON や形の違う JSON も、このキャラの行に留める
                if capture is not None:
                    capture.note_exception(name, exc)
                if isinstance(exc, FileNotFoundError):
                    report.add(name, "FAIL", "voices.json がありません。L2_IRODORI_VOICES_JSON を確かめてください", required=required)
                else:
                    report.add(
                        name, "FAIL",
                        f"voice={character.tts_voice} を voices.json から読めません ({type(exc).__name__})",
                        required=required,
                    )
                continue
            ref_wav = voice.get("ref_wav")
            if ref_wav and not Path(ref_wav).is_file():
                report.add(name, "FAIL", f"voice={character.tts_voice} の参照音声がありません", required=required)
        elif character.tts_provider == "voicepeak":
            if shutil.which(os.environ.get("L2_TTS_VOICEPEAK_PATH", "voicepeak")) is None:
                report.add(name, "FAIL", "voicepeak の実行ファイルがありません", required=required)


def _check_readings(report: Report, characters: list[Any], *, with_pipeline: bool = True) -> None:
    """irodori の読み辞書を、本番と同じ関数で読めるかを確かめる。

    Args:
        report:        結果の記録先
        characters:    点検するキャラ
        with_pipeline: False（--paid --no-pipeline）なら TTS を通らないので、行だけ残して合否に数えない

    WHY: 壊れた辞書は _call_irodori の中で合成の前に例外になり、irodori のキャラの発話がすべて落ちる。
         辞書がないと本番は INFO のログだけで続け、英単語や固有名詞の誤読が無音で始まる。声は出るし、
         辞書は任意で OSS の利用者には普通にないので、ないだけなら WARN に留める。INFO にしないのは、
         手元で辞書を失ったときに、要約に数えられない行だと見落とすため。
    """
    irodori = [character for character in characters if character.tts_provider == "irodori_vd"]
    if not irodori or not _is_truthy(os.environ.get("L2_USE_REAL_TTS")):
        return
    from .tts import _apply_readings, _load_readings

    try:
        readings = _load_readings()
        # 形の誤りは読み込みでは分からず、置換の準備で初めて例外になるので、本番の関数まで通す。
        for character in irodori:
            _apply_readings("", character.tts_voice)
        # 値の型の誤りは、その語を読み上げるまで例外にならない。null は黙って語を消す。
        if readings:
            tables = [readings.get("global", {}), *readings.get("characters", {}).values()]
            if any(not isinstance(value, str) for table in tables for value in table.values()):
                raise TypeError("読みが文字列ではありません")
    except Exception as exc:  # noqa: BLE001 - 壊れた JSON も、読めない文字コードも、形の誤りも同じ扱いにする
        required = with_pipeline and any(character.slug in REQUIRED_CHARACTERS for character in irodori)
        report.add("tts-config/readings", "FAIL", f"読み辞書を読めません ({type(exc).__name__})", required=required)
        return
    if not readings:
        report.add("tts-config/readings", "WARN", "読み辞書がないか空です。英単語や固有名詞を誤読します")


def _check_filler_cache(report: Report, characters: list[Any]) -> None:
    """配信に出るキャラの filler の opener の WAV があるかを確かめる。

    WHY: run_filler_loop は opener がないと LLM を呼ぶ前に終わるので、--paid の filler の行が PASS でも、
         応答を待つ間は無音になる。WAV は git に入らず自動でも作られないので、新しい環境では必ずこの状態に
         なる。主機能は動くので WARN に留める。ダミーの TTS は応答の音声を作らず待ち時間もないので見ない。
    """
    from .filler import is_filler_enabled, select_filler_path

    if not is_filler_enabled() or not _is_truthy(os.environ.get("L2_USE_REAL_TTS")):
        return
    # 別枠のキャラは見ない。ルーティングされないか、フレーズの定義がなく WAV を作れない。
    missing = [
        character.slug for character in characters
        if character.slug in REQUIRED_CHARACTERS and select_filler_path(character.slug, "opener")[0] is None
    ]
    if missing:
        report.add(
            "filler-cache", "WARN",
            f"opener の WAV がなく、応答を待つ間が無音になります: {', '.join(missing)}。scripts/generate_filler_cache.py で作れます",
        )


def _check_kb_index(report: Report) -> None:
    """seed の索引を読めるかを確かめる。通信も課金もしない。

    WHY: _build_retriever_from_env は LocalRetriever を最初に作るので、索引がないと C2 の分も含めて記憶の検索が
         全滅する。retrieve_memory はそれを文字列にして LLM へ返すだけなので、声は止まらない。
         実 LLM でなければ Agent を通らず索引も読まれないので、WARN に留める。
    """
    if not _memory_tool_enabled():
        return
    from .retriever import LocalRetriever

    try:
        LocalRetriever(os.environ.get("L2_KB_PATH", "./data/index"))
    except Exception as exc:  # noqa: BLE001 - 読めない理由は問わない
        status = "FAIL" if _is_truthy(os.environ.get("L2_USE_REAL_LLM")) else "WARN"
        report.add("rag/index", status, f"seed の索引を読めません ({type(exc).__name__})。scripts/build_index.py で作れます")


def _report_redis_url_shape(report: Report, value: str) -> bool:
    """REDIS_URL を本番の redis-py と同じ読み方で確かめる。接続はしない。

    WHY: urlsplit で host が取れるかどうかでは本番とずれる。unix ソケットと host の省略は本番が受け、
         http:// や読めないクエリは本番が拒否して、全ターンの publish が落ちる。

    Returns:
        行を出したら True。host と port で読める普通の URL なら False（残りは呼び出し側が見る）
    """
    import redis

    try:
        pool = redis.ConnectionPool.from_url(value)
    except (ValueError, redis.RedisError):
        # 例外文はパスワードを含みうるので出さない。
        report.add("url/REDIS_URL", "FAIL", "URL の形が不正です（値は表示しません）")
        return True
    if pool.connection_class is redis.UnixDomainSocketConnection:
        if hasattr(socket, "AF_UNIX"):
            report.add("url/REDIS_URL", "INFO", "unix ソケット")
        else:
            # Windows の Python には AF_UNIX がなく、本番の publish が毎回 AttributeError になる。
            report.add("url/REDIS_URL", "FAIL", "この環境の Python は unix ソケットを使えません")
        return True
    if not pool.connection_kwargs.get("host"):
        # redis-py は省略した host を localhost と読むので、localhost と同じ遅れが出る。
        report.add("url/REDIS_URL", "WARN", "host の省略は localhost と同じで ::1 を先に試して遅れます。127.0.0.1 にしてください")
        return True
    return False


def _check_endpoints(report: Report, characters: list[Any] | None = None) -> None:
    """接続先の形だけを確かめる。host は表示せず、loopback かどうかと port だけを出す。

    形の誤りは、今の設定で本番が読む接続先だけ FAIL にする。読まないものは WARN で知らせる。
    使わない接続先の誤りで配信の起動を止めないため。

    Args:
        report:     結果の記録先
        characters: 点検するキャラ。None なら全キャラ

    WHY: Windows の localhost は ::1 を先に試すので、IPv4 だけで待ち受ける相手には接続ごとに
         約 2 秒失う。C2 の RAG は 600 ms の予算なので、これだけで検索なしに落ちる。
    """
    c2_used = _c2_retriever_enabled()
    voicevox_used, voicevox_required = _tts_engine_use("voicevox", characters)
    irodori_used, irodori_required = _tts_engine_use("irodori_vd", characters)
    endpoints = (
        # (名前, 既定, 今の設定で本番が読むか, 読む経路が必須か)
        ("REDIS_URL", "redis://127.0.0.1:6379", True, True),
        ("L2_C2_URL", "http://127.0.0.1:8100", c2_used, True),
        ("L2_C2_URL_READONLY", "", c2_used, True),
        ("L2_TTS_VOICEVOX_URL", "http://127.0.0.1:15021", voicevox_used, voicevox_required),
        ("L2_TTS_IRODORI_URL", "http://127.0.0.1:18080", irodori_used, irodori_required),
    )
    for name, default, used, required in endpoints:
        value = os.environ.get(name, default)
        # 本番も os.environ.get で読むので、空の値は既定に戻らず、接続の時点で壊れる（不正な URL と同じ）。
        # 既定が空の L2_C2_URL_READONLY だけは、本番と同じく空白も含めて「使わない」と読む。
        if not value.strip() and not default:
            continue
        if name == "REDIS_URL" and _report_redis_url_shape(report, value):
            continue
        location = _url_location(value)
        if location is None:
            if used:
                report.add(f"url/{name}", "FAIL", "URL の形が不正です（値は表示しません）", required=required)
            else:
                report.add(f"url/{name}", "WARN", "URL の形が不正です（今の設定では本番が読みません。値は表示しません）")
            continue
        host, port = location
        # tts.py は VOICEVOX の URL だけ末尾の / を落とさないので、//audio_query へ送る。FastAPI の経路には
        # 合わず 404 になるはず（Starlette の照合で確認。実物の Engine では未確認なので WARN に留める）。
        # 生死の確認は / を落とした URL を見るので、--paid の前提サービスの確認では分からない。
        if name == "L2_TTS_VOICEVOX_URL" and value.endswith("/"):
            report.add(f"url/{name}", "WARN", "末尾の / を外してください。本番は //audio_query へ送ります")
        if host == "localhost":
            report.add(f"url/{name}", "WARN", "localhost は ::1 を先に試して遅れます。127.0.0.1 にしてください")
        # macOS / Linux では 50021 が VOICEVOX の既定で、そのまま使える。
        elif name == "L2_TTS_VOICEVOX_URL" and port == 50021 and sys.platform == "win32":
            report.add(f"url/{name}", "WARN", "50021 は Windows の除外レンジです。15021 にしてください")
        else:
            kind = "loopback" if _is_loopback_url(value) else "remote"
            report.add(f"url/{name}", "INFO", f"{kind}:{port or _DEFAULT_PORTS.get(urlsplit(value).scheme)}")


def _check_stream(report: Report, profile: str) -> None:
    """publish 先のストリームが、プロファイルの想定どおりかを確かめる。"""
    from .bus import _stream_key

    stream = _stream_key()
    if not stream.strip():
        # 本番は既定へ戻さず、空の名前のまま publish する。C2 と HUD が読むストリームとは別になる。
        report.add("stream", "WARN", "REDIS_STREAM_KEY が空です。C2 と HUD に発話が届きません")
    elif profile == "dev" and stream == PROD_STREAM:
        report.add("stream", "WARN", f"{stream}。dev の発話が本番の正史に入ります")
    elif profile == "prod" and stream != PROD_STREAM:
        report.add("stream", "WARN", f"{stream}。本番の正史に入りません")
    else:
        report.add("stream", "INFO", stream)


def _check_run_script_refusals(report: Report, profile: str, repo_root: Path) -> None:
    """run_dev.ps1 / run_prod.ps1 が起動を拒否する組を知らせる。

    WHY: 点検が緑でも、スクリプトから起動すると exit 1 で止まる。python を直接起動する構成もあるので WARN に留める。
         スクリプトは未設定の L2_C2_URL を拒否しないので、既定の値を補わずに同じ条件で見る。
    """
    c2_url = os.environ.get("L2_C2_URL", "")
    if profile == "dev":
        if not (repo_root / ".env.dev").is_file():
            report.add("run-script/.env.dev", "WARN", ".env.dev がありません。run_dev.ps1 は起動を拒否します")
        if re.search(r":8100/?$", c2_url):
            report.add("run-script/L2_C2_URL", "WARN", "dev で本番のポート 8100 を指しています。run_dev.ps1 は起動を拒否します")
    elif re.search(r":8101/?$", c2_url):
        report.add("run-script/L2_C2_URL", "WARN", "prod で dev のポート 8101 を指しています。run_prod.ps1 は起動を拒否します")


def _check_loopback_proxy(report: Report) -> None:
    """手元のサービスへの接続がプロキシを通る設定を知らせる。プロキシの URL は表示しない。

    WHY: 本番の C2Retriever (httpx) と TTS (urllib) は環境のプロキシを読むので、NO_PROXY がないと 127.0.0.1 への
         接続もプロキシへ向かう。run スクリプトの /health の確認はプロキシを通らないので、起動の確認は緑のまま
         本番だけが繋がらないことがある。--static は run スクリプトが毎回流すので、ここで見えるようにする。
    """
    if urllib.request.getproxies().get("http") and not urllib.request.proxy_bypass("127.0.0.1"):
        report.add("env/proxy", "WARN", "プロキシの設定があり、NO_PROXY に 127.0.0.1 がありません。C2 と TTS への接続がプロキシを通ります")


def _check_event_schema(report: Report) -> None:
    """イベントのスキーマを、本番と同じ関数で解決できるかを確かめる。

    WHY: 本番は publish の前に必ず検証するので、スキーマが読めないと最初の build_utterance_final で
         RuntimeError になり、run_loop はそれを「Pipeline 失敗」のログにして次へ進む。ダミー起動でも
         全ターンが落ちるので常に FAIL にする。--paid まで待つと、router、filler、本応答の課金の後で分かる。
    """
    try:
        from .events import _get_validator

        _get_validator()
    except Exception as exc:  # noqa: BLE001 - 見つからないのも、壊れた JSON も、スキーマとして不正なのも同じ扱いにする
        # 例外文には手元のパスが入るので、型だけを出す。
        report.add(
            "schema", "FAIL",
            f"イベントのスキーマを読めません ({type(exc).__name__})。aibyss-workspace を兄弟に置くか、AIBYSS_SCHEMA_PATH を設定してください",
        )


def _display_path(path: Path, repo_root: Path) -> str:
    """リポジトリの中ならルートからの相対で表示する（ユーザー名を出さないため）。"""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _check_stream_context(report: Report, repo_root: Path) -> None:
    """配信文脈の鮮度を確かめる。本文は読み上げない。"""
    from .stream_context import _DEFAULT_PATH

    env_path = os.environ.get("L2_STREAM_CONTEXT_FILE")
    path = Path(env_path) if env_path else _DEFAULT_PATH
    shown = _display_path(path, repo_root)
    if not path.is_file():
        report.add("stream_context", "INFO", f"{shown} なし（prompt に足さない）")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # 本番の load_stream_context も同じ読み方で、起動時に例外で落ちる。
        report.add("stream_context", "FAIL", f"{shown} を読めません ({type(exc).__name__})")
        return
    if not text.strip():
        report.add("stream_context", "INFO", f"{shown} は空（prompt に足さない）")
        return
    age_days = (time.time() - path.stat().st_mtime) / 86400
    updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    if age_days > _STREAM_CONTEXT_STALE_DAYS:
        report.add(
            "stream_context", "WARN",
            f"{shown} は {updated} 更新（{age_days:.0f} 日前）。今日の話として全キャラの prompt に入ります",
        )
    else:
        report.add("stream_context", "INFO", f"{shown} は {updated} 更新")


def _tracing_state() -> str:
    """LangSmith への送信が有効かどうかを、4 つの変数すべてを見て返す。"""
    enabled = [name for name in _TRACING_VARS if _is_truthy(os.environ.get(name))]
    return f"有効 ({', '.join(enabled)})" if enabled else "無効"


def _check_obs(report: Report, characters: list[Any]) -> None:
    """OBS への接続と立ち絵の有無を、本番と同じ関数で確かめる。

    OBS がなくても配信の主機能は動くので、結果は WARN までに留める。
    ただし URL の形が不正なときだけは、本番の init_obs が例外で落ちて起動できないので FAIL にする。
    接続先と例外文は host / port を含むので表示しない（obs.py と同じ方針）。

    WHY: グループがあっても中のソース名が規約と違うと、set_pose は INFO のログだけで何もしない。切替の可否を
         決める _get_item_id で見ると、グループの有無だけを見るより本番の判定に近づく。neutral は全キャラにある
         基本の pose で、未知の pose の落ち先でもあるので、代表としてこれを見る。
    """
    url = os.environ.get("L2_OBS_WS_URL")
    if not url:
        report.add("obs", "WARN", "L2_OBS_WS_URL が未設定。立ち絵の切替は動きません")
        return
    try:
        import obsws_python  # noqa: F401
    except ImportError:
        report.add("obs", "WARN", "obsws-python がありません")
        return

    from . import obs

    try:
        obs._parse_ws_url(url)
    except ValueError:
        report.add("obs", "FAIL", "L2_OBS_WS_URL の形が不正です。本番は起動できません（値は表示しません）")
        return

    obs.init_obs()
    try:
        if obs._client is None:
            # init_obs は失敗を握りつぶし、host を含まない例外の型だけを WARNING のログに残す。
            report.add("obs", "WARN", "接続できません（理由は logs/preflight/ のログにあります）")
            return
        missing = [
            character.slug for character in characters
            if obs._get_item_id(character.slug, f"{character.slug}_neutral") is None
        ]
        if missing:
            report.add("obs", "WARN", f"接続できましたが、立ち絵（<slug>_neutral）がありません: {', '.join(missing)}")
        else:
            report.add("obs", "PASS", "接続でき、立ち絵グループが揃っています")
    finally:
        # 接続とキャッシュを残さず、起動前と同じ状態へ戻す。
        obs.disconnect_obs()


def run_static(
    report: Report,
    repo_root: Path,
    profile: str,
    characters: list[Any],
    file_names: set[str],
    *,
    paid: bool = False,
    with_pipeline: bool = True,
    capture: _LogCapture | None = None,
) -> None:
    """課金なしの点検をすべて流す。

    Args:
        report:        結果の記録先
        repo_root:     .env と uv.lock のあるルート
        profile:       "dev" または "prod"
        characters:    点検するキャラ
        file_names:    env ファイルに書かれている名前
        paid:          --paid の前の点検なら True
        with_pipeline: False（--paid --no-pipeline）なら、実 TTS の有無と TTS の設定を合否に数えない
        capture:       点検が例外で止まったときの全文を残す先

    WHY: SDK の import の失敗（ImportError 以外）や、読み込むファイルの形の違いで 1 つの点検が例外になっても、
         残りの行と要約を失わない。起動前のゲートは、何が壊れているかを読めることが大事なので。
    """
    def run(name: str, check: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """1 つの点検を流す。例外は FAIL の 1 行にして None を返し、残りの点検へ進む。"""
        try:
            return check(report, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 想定外の例外も FAIL の 1 行にする
            # 例外文は値を含みうるので、_cli と同じく画面には型だけを出し、全文は問題ログへ残す。
            if capture is not None:
                capture.note_exception(name, exc)
            report.add(name, "FAIL", f"点検を続けられません ({type(exc).__name__})")
            return None

    run("deps/uv.lock", _check_versions, repo_root)
    providers = run("model", _check_models, characters)
    if providers is None:
        # どの SDK が原因かを import の行で見せるため、全プロバイダを見る。
        providers = {"openai", "anthropic", "google"}
    run("import", _check_imports, providers, capture)
    run("key", _check_keys, providers, file_names)
    run("flag", _check_flags, file_names)
    run("env/quoting", _check_env_quoting, file_names)
    run("flag/numeric", _check_numeric_env, characters)
    run("mode", _check_real_modes, profile, file_names, paid=paid, with_pipeline=with_pipeline)
    run("persona", _check_personas, characters)
    run("tts-config", _check_tts_config, characters, capture, with_pipeline=with_pipeline)
    run("tts-config/readings", _check_readings, characters, with_pipeline=with_pipeline)
    run("filler-cache", _check_filler_cache, characters)
    run("rag/index", _check_kb_index)
    run("url", _check_endpoints, characters)
    run("stream", _check_stream, profile)
    run("run-script", _check_run_script_refusals, profile, repo_root)
    run("env/proxy", _check_loopback_proxy)
    run("schema", _check_event_schema)
    run("stream_context", _check_stream_context, repo_root)
    report.add("flag/LANGSMITH", "INFO", _tracing_state())
    run("obs", _check_obs, characters)


# ─── --paid: 全経路へ実際に流す ───────────────────────────────────

def _call_name(character: Any) -> str:
    """本番でキャラへ呼びかけるときの名前を返す。wake word のないキャラ（ruka）は表示名にする。"""
    return character.wake_word or character.display_name


def _probe_text(character: Any) -> str:
    """本番と同じ「ルカからの発言」の形で、ツールが要らない短い発話を作る。"""
    return f"[ルカからの発言]\n{_call_name(character)}、配信前の点検です。ツールは使わず、ひとことだけ挨拶してください。"


def _is_list_repr(text: str) -> bool:
    """応答が list の repr かどうかを返す。

    WHY: llm.py の str(response.content) は、thinking が動くモデルで list の repr を返す。
         本番ではそれをそのまま読み上げてしまう。
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return False
    try:
        return isinstance(ast.literal_eval(stripped), list)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        # literal_eval は壊れた入力でこれらも出す。読み上げのチャンク 1 つで点検全体を止めない。
        return False


def _check_raw_llm(
    report: Report,
    capture: _LogCapture,
    name: str,
    model: str,
    call: Callable[[], str | None],
    *,
    required: bool = True,
    validate: Callable[[str], str | None] | None = None,
) -> None:
    """raw SDK の経路を 1 回呼ぶ。

    Args:
        report:   結果の記録先
        capture:  ログの捕捉先
        name:     経路の名前
        model:    表示するモデル名
        call:     呼び出し本体
        required: False なら別枠
        validate: 応答を受けて、問題があれば理由を返す関数

    WHY: _call_router_llm / _call_filler_llm は失敗を None に変える。本番ではそれで会話が続くが、
         点検では None こそが知りたい失敗なので FAIL にする。理由はログから拾う。
    """
    mark = capture.mark()
    started = time.monotonic()
    try:
        text = call()
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception(name, exc)
        report.add(name, "FAIL", f"{model}: {_describe_exc(exc)}", required=required)
        return
    elapsed = time.monotonic() - started
    if not text or not text.strip():
        reason = capture.last_problem(mark) or "空応答"
        report.add(name, "FAIL", f"{model}: None が返りました ({reason})", required=required)
        return
    problem = validate(text) if validate else None
    if problem:
        report.add(name, "FAIL", f"{model}: {problem}", required=required)
        return
    report.add(name, "PASS", f"{model} {elapsed:.1f}s len={len(text)}", required=required)


def _validate_router_answer(text: str) -> str | None:
    """本番の _route_by_llm が受け取れる答えかどうかを確かめる。"""
    # 本番は == で比べるので、strip せずに同じ比べ方をする。
    if text in _ROUTER_PROBE_ANSWERS:
        return None
    # 拒否文や max_tokens で切れた語は、本番では候補と一致せず捨てられる。
    return f"候補にない応答です (len={len(text)})"


def check_router(report: Report, capture: _LogCapture) -> None:
    """ルーターと意図判定の raw 経路を確かめる。"""
    from .router import (
        _INTENT_GATE_MODEL,
        _LLM_ROUTER_MODEL,
        _call_router_llm,
        _is_llm_router_enabled,
    )

    routes = (
        ("router", _is_llm_router_enabled(), os.environ.get("L2_LLM_ROUTER_MODEL", _LLM_ROUTER_MODEL)),
        ("intent", _intent_route_enabled(), os.environ.get("L2_INTENT_GATE_MODEL", _INTENT_GATE_MODEL)),
    )
    for name, enabled, model in routes:
        if not enabled:
            report.add(name, "SKIP", f"{model}（無効）")
            continue
        _check_raw_llm(
            report, capture, name, model,
            lambda m=model: _call_router_llm(m, _ROUTER_PROBE_SYSTEM, _ROUTER_PROBE_USER),
            validate=_validate_router_answer,
        )


def _validate_filler_answer(text: str) -> str | None:
    """filler の応答を確かめ、問題があれば理由を返す。

    WHY: max_tokens で切れた JSON は本番の _parse_voicepeak_json を通らず、断片がそのまま読み上げの
         材料になる。一方、プレーンな文やコードブロック付きの JSON は本番でも読めるので通す。
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    if not re.match(r'\{\s*"', stripped):
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        return f"JSON として読めません (len={len(text)})"
    # {" で始まるので data は必ず dict。本番は str() して読むので、null は「None」と読み上げてしまう。
    response = data.get("response")
    if not (isinstance(response, str) and response.strip()):
        return "JSON に response がありません"
    return None


def check_filler(report: Report, capture: _LogCapture, character: Any, *, required: bool) -> None:
    """キャラごとの filler の raw 経路を、本番と同じ prompt で確かめる。"""
    from .filler import _build_filler_prompt, _call_filler_llm, is_filler_enabled

    name = f"filler/{character.slug}"
    model = effective_filler_model(character)
    if not is_filler_enabled():
        report.add(name, "SKIP", f"{model}（無効）", required=required)
        return
    _check_raw_llm(
        report, capture, name, model,
        lambda: _call_filler_llm(model, _build_filler_prompt(character.slug), _FILLER_PROBE_USER),
        required=required,
        validate=_validate_filler_answer,
    )


def _api_kind(metadata: dict[str, Any]) -> str:
    """応答がどの API を通ったかを response_metadata から読む。

    WHY: 「gpt-5.5 の Chat + tools が通るか」を行で答えるには、Chat Completions を通った証拠が要る。
         SDK はモデルや引数で Responses へ振り替えることがあるので、設定ではなく応答を見る。
         Responses の応答にだけ object=response が入る。

    Args:
        metadata: AIMessage の response_metadata

    Returns:
        OpenAI は "chat" か "responses"、他のプロバイダは model_provider の値、読めなければ "?"
    """
    if metadata.get("object") == "response":
        return "responses"
    provider = metadata.get("model_provider")
    if provider == "openai":
        return "chat"
    return str(provider) if provider else "?"


def _summarize_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    """agent.invoke の結果から、判定に使う値だけを取り出す。本文は長さしか残さない。"""
    structured = result.get("structured_response")
    response = getattr(structured, "response", None)
    if isinstance(structured, dict):
        response = structured.get("response")

    ai_messages = [m for m in result.get("messages", []) if getattr(m, "type", "") == "ai"]
    input_tokens = output_tokens = 0
    tools: list[str] = []
    for message in ai_messages:
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += usage.get("input_tokens") or 0
        output_tokens += usage.get("output_tokens") or 0
        # 構造化出力を tool で受ける戦略では <Slug>Response という tool が呼ばれるので除く。
        tools.extend(
            call["name"] if _TOOL_NAME.fullmatch(call["name"]) else "<不正な名前>"
            for call in getattr(message, "tool_calls", None) or []
            if not call["name"].endswith("Response")
        )
    last_metadata = (getattr(ai_messages[-1], "response_metadata", None) or {}) if ai_messages else {}
    return {
        "source": "structured_response" if structured is not None else "messages",
        "response_len": len(response.strip()) if isinstance(response, str) else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "steps": len(ai_messages),
        "tools": tools,
        "finish": last_metadata.get("finish_reason") or last_metadata.get("stop_reason") or "",
        "api": _api_kind(last_metadata),
    }


class _AgentAudit:
    """Agent の各 LLM 呼び出しを数える。

    WHY: GraphRecursionError で終わると戻り値がないので、課金した回数も、検証に落ちた応答の中身も失う。
         理由を調べるためにもう一度課金しなくて済むよう、最後の応答を問題ログへ残せる形で持つ。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_output = ""

    def callback(self) -> Any:
        """agent.invoke の config へ渡すコールバックを作る。

        WHY: BaseCallbackHandler を継承するのは、LangChain が handler に求める属性が更新で増えても、
             属性の不足で例外が握られて無音で数えなくなるのを避けるため。関数の中で読むのは、
             langchain_core の import が壊れた環境でも、点検の行に留めて残りを流すため。
        """
        from langchain_core.callbacks import BaseCallbackHandler

        audit = self

        class _Handler(BaseCallbackHandler):
            """LLM の終了だけを受ける。残りの属性は親クラスに任せる。"""

            def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
                """LLM の 1 回ぶんの応答を受け取る。"""
                audit.count(response)

        return _Handler()

    def count(self, response: Any) -> None:
        """LLM の 1 回ぶんの応答（LLMResult）を数える。"""
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens") or 0
                self.output_tokens += usage.get("output_tokens") or 0
                self.last_output = repr(getattr(message, "tool_calls", None) or getattr(message, "content", ""))

    def record_failure(self, report: Report, capture: _LogCapture, name: str) -> str:
        """失敗した実行の分を集計へ足し、最後の応答を問題ログへ残す。

        Returns:
            行に添える短い文字列。本文は含めない
        """
        report.input_tokens += self.input_tokens
        report.output_tokens += self.output_tokens
        if self.last_output:
            capture.write_problem(f"{name}: 最後の応答 ({self.calls} 回目): {self.last_output}")
        return f" (LLM {self.calls} 回)" if self.calls else ""


def _invoke_agent_directly(
    agent: Any,
    text: str,
    *,
    slug: str,
    provider: str,
    model: str,
    audit: _AgentAudit,
    session_id: str | None = None,
) -> dict[str, Any]:
    """本番の壁時計ガードを通して agent.invoke を呼ぶ。_run_agent のフォールバックは通さない。

    audit には、例外で終わっても課金した回数と最後の応答が残る。
    session_id を渡すと、打ち切ったときに本番と同じく協働先の TTS を止める。
    """
    from .graph import _get_agent_invoke_timeout, _invoke_agent_with_timeout

    return _invoke_agent_with_timeout(
        agent,
        {"messages": [{"role": "user", "content": text}]},
        {"recursion_limit": _AGENT_RECURSION_LIMIT, "callbacks": [audit.callback()]},
        timeout_sec=_get_agent_invoke_timeout(),
        character_slug=slug,
        provider=provider,
        model=model,
        handler=None,
        session_id=session_id,
    )


def _agent_prompt(character: Any, stream_context: str | None) -> str | None:
    """_routing_node と同じ形で system prompt を組み立てる。"""
    from .characters import load_system_prompt
    from .graph import _compose_system_prompt

    return _compose_system_prompt(load_system_prompt(character), stream_context, now=datetime.now())


def check_agent(
    report: Report,
    capture: _LogCapture,
    character: Any,
    stream_context: str | None,
    *,
    required: bool,
) -> bool:
    """本応答を、_run_agent のフォールバックを通さずに確かめる。

    WHY: _run_agent は失敗すると call_llm（ツールなし）へ落とすので、ask_character が毎ターン
         消えていても声は出る。_build_agent_graph と agent.invoke を直接呼び、構造化出力が
         返らなければ FAIL にする。ツールの一部が消えた場合もログで捕まえる。

    Returns:
        PASS なら True。呼び出し側はこれを見て、同じ Agent を通すパイプラインを省く
    """
    from .graph import _build_agent_graph, _is_tools_enabled
    from .llm import call_llm
    from .pipeline import _get_llm_mode

    slug = character.slug
    name = f"main/{slug}"
    _, provider, model = _get_llm_mode(character)
    mark = capture.mark()
    started = time.monotonic()
    audit = _AgentAudit()
    try:
        prompt = _agent_prompt(character, stream_context)
        if not _is_tools_enabled():
            # ツールが無効なら本番も単一ノードなので、同じ call_llm を確かめる。
            result = call_llm(_probe_text(character), model=model, provider=provider,
                              system_prompt=prompt, caller_slug=slug)
            report.input_tokens += result.input_tokens
            report.output_tokens += result.output_tokens
            if result.finish_reason in _TRUNCATED_FINISH:
                problem = f"出力が切れました ({result.finish_reason})"
            else:
                # 単一ノードは構造化出力を通らないので、本番が読み上げる本文を pipeline と同じ判定で見る。
                problem = next(iter(_spoken_problems(_spoken_text(result.text))), None)
            if problem:
                report.add(name, "FAIL", f"{model}: {problem}", required=required)
                return False
            report.add(name, "PASS", f"{model} {time.monotonic() - started:.1f}s tools=off", required=required)
            return True
        agent = _build_agent_graph(provider, model, prompt, character_slug=slug)
        if agent is None:
            reason = capture.last_problem(mark) or "ツールが 0 件"
            report.add(name, "FAIL", f"{model}: 単一ノードへ落ちます ({reason})", required=required)
            return False
        result = _invoke_agent_directly(
            agent, _probe_text(character), slug=slug, provider=provider, model=model, audit=audit,
        )
        # 戻り値の形は SDK の更新で変わりうるので、読むところまで try の中に入れる。
        summary = _summarize_agent_result(result)
        report.input_tokens += summary["input_tokens"]
        report.output_tokens += summary["output_tokens"]
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception(name, exc)
        calls = audit.record_failure(report, capture, name)
        report.add(name, "FAIL", f"{model}: {_describe_exc(exc)}{calls}", required=required)
        return False

    detail = (
        f"{model} {time.monotonic() - started:.1f}s api={summary['api']} source={summary['source']}"
        f" in={summary['input_tokens']} out={summary['output_tokens']} steps={summary['steps']}"
        # 渡したツールではなく、モデルが呼んだツール。本応答は常にツールを渡して流す。
        f" called={','.join(summary['tools']) or '-'}"
    )
    problems = _agent_problems(summary, capture.messages_since(mark))
    if problems:
        report.add(name, "FAIL", f"{detail} / {' / '.join(problems)}", required=required)
        return False
    report.add(name, "PASS", detail, required=required)
    return True


def _agent_problems(summary: dict[str, Any], logs: list[str]) -> list[str]:
    """本応答の 1 ターンの問題を挙げる。"""
    problems = []
    if summary["source"] != "structured_response":
        problems.append("構造化出力が返りません")
    elif not summary["response_len"]:
        problems.append("応答が空です")
    if summary["finish"] in _TRUNCATED_FINISH:
        problems.append(f"出力が切れました ({summary['finish']})")
    if any("ツール読み込み失敗" in message for message in logs):
        problems.append("ツールの読み込みに失敗しています")
    return problems


def _with_deadline(call: Callable[[], Any], seconds: float) -> Any:
    """call を別スレッドで流し、seconds を超えたら TimeoutError にする。

    WHY: langchain_tavily と VOICEVOX と bus.publish は timeout なしで通信し、埋め込みは SDK の既定で
         最長 30 分待つ。相手が応答を止めると、残りの行も要約も出ないまま点検が固まる。
         固まった worker は、コマンドラインでは _cli の os._exit で消える。
    """
    ctx = contextvars.copy_context()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(ctx.run, call)
    try:
        return future.result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        # 3.11 以降は組み込みの TimeoutError と同じ型。call 自身が投げたものは、そのまま理由として出す。
        if future.done():
            raise
        raise TimeoutError(f"{seconds:.0f}s を超過") from None
    finally:
        executor.shutdown(wait=False)


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    """GET が 2xx を返すかどうかを返す。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - 設定された URL のみ
            return 200 <= response.status < 300
    except Exception:  # noqa: BLE001 - 届かない理由は問わない
        return False


def tts_engine_status(character: Any) -> tuple[bool, str]:
    """キャラの TTS エンジンが使える状態かを返す。合成はしない。

    Returns:
        (使えるか, エンジン名)
    """
    provider = character.tts_provider
    if provider == "irodori_vd":
        base = os.environ.get("L2_TTS_IRODORI_URL", "http://127.0.0.1:18080").rstrip("/")
        return _http_ok(f"{base}/health"), "irodori"
    if provider == "voicevox":
        base = os.environ.get("L2_TTS_VOICEVOX_URL", "http://127.0.0.1:15021").rstrip("/")
        return _http_ok(f"{base}/version"), "voicevox"
    if provider == "voicepeak":
        return shutil.which(os.environ.get("L2_TTS_VOICEPEAK_PATH", "voicepeak")) is not None, "voicepeak"
    if provider == "edge_tts":
        # 外部のサービスなのでローカルのエンジンがなく、確かめる相手がない。
        return True, provider
    # 本番の synthesize は未知の provider を ValueError にする。
    return False, provider


def missing_services(characters: list[Any], *, with_pipeline: bool) -> list[str]:
    """課金の前に、前提サービスが揃っているかを確かめる。

    Args:
        characters:    点検するキャラ
        with_pipeline: パイプラインの 1 ターンも流すか

    Returns:
        足りないサービスの名前
    """
    missing: list[str] = []
    # WHY: Redis の consumer が止まっても C2 は listen を続けるので、TCP の接続だけでは「いる」と誤る。
    #      /health は consumer の生死と読み取りの鮮度を見て 503 を返す。
    c2_url = os.environ.get("L2_C2_URL", "http://127.0.0.1:8100").rstrip("/")
    if _c2_retriever_enabled() and not _http_ok(f"{c2_url}/health"):
        missing.append("C2")
    if not with_pipeline:
        return missing

    import redis

    from .bus import _redis_url

    try:
        redis.Redis.from_url(_redis_url(), socket_connect_timeout=1, socket_timeout=1).ping()
    except (redis.RedisError, ValueError):
        missing.append("Redis")
    for character in characters:
        if character.slug not in REQUIRED_CHARACTERS:
            continue
        ok, engine = tts_engine_status(character)
        if not ok and engine not in missing:
            missing.append(engine)
    return missing


def check_tavily(report: Report, capture: _LogCapture) -> None:
    """Tavily の実検索を 1 回流し、結果の中身で判定する。

    WHY: langchain_tavily は HTTP エラーを例外にせず {"error": ...} を返し、本番の _web_search_impl は
         それを str() にする。文字列の非空で判定すると、キー切れやクレジット切れも PASS になる。
         戻り値の形を見るために、本番と同じ引数（課金を抑えるので件数だけ 1）で TavilySearch を直接呼ぶ。
    """
    if not _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        report.add("tavily", "SKIP", "L2_ENABLE_TOOLS が無効")
        return
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=1, topic="general", search_depth="basic")
        result = _with_deadline(lambda: tool.invoke("A.I.byss"), _TAVILY_DEADLINE_SEC)
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception("tavily", exc)
        report.add("tavily", "FAIL", _describe_exc(exc))
        return
    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        if isinstance(error, BaseException):
            capture.note_exception("tavily", error)
            # TavilySearch はどの例外もここへ包む。HTTP エラーは手元の ValueError（Error 401: <detail.error>）で届く。
            # API の応答の本文なので status だけを出す。接続の失敗は host を本文に含むが、_describe_exc が型だけにする。
            matched = re.match(r"Error (\d{3})\b", str(error)) if isinstance(error, ValueError) else None
            shown = f"{type(error).__name__} status={matched.group(1)}" if matched else _describe_exc(error)
            report.add("tavily", "FAIL", shown)
        else:
            report.add("tavily", "FAIL", "error が返りました")
        return
    if not (isinstance(result, dict) and result.get("results")):
        report.add("tavily", "FAIL", "結果が空です")
        return
    report.add("tavily", "PASS", f"results={len(result['results'])}")


def check_embeddings(report: Report, capture: _LogCapture) -> None:
    """OpenAI の埋め込みを 1 回流す。L2 の seed corpus の検索が使う経路。

    WHY: ツールが無効なら本番は埋め込みを呼ばないので、使わない経路に課金せず、キーがなくても FAIL にしない。
    """
    if not _memory_tool_enabled():
        report.add("embeddings", "SKIP", "L2_ENABLE_TOOLS か L2_ENABLE_RAG が無効")
        return
    from .retriever import EMBEDDING_MODEL, _embed

    try:
        vector = _with_deadline(lambda: _embed("preflight"), _EMBED_DEADLINE_SEC)
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception("embeddings", exc)
        report.add("embeddings", "FAIL", f"{EMBEDDING_MODEL}: {_describe_exc(exc)}")
        return
    dim = len(vector) if vector else 0
    report.add("embeddings", "PASS" if vector else "FAIL", f"{EMBEDDING_MODEL} dim={dim}")


def _check_c2_endpoint(report: Report, name: str, base_url: str, *, required: bool) -> None:
    """C2 の /retrieve を呼び、意味検索が動いているかを確かめる。

    WHY: C2Retriever は失敗を空リストに変えるので、呼び出し元では検索が死んでいても分からない。
         search_mode が like なら、C2 の embedding が落ちているか、ベクトルがまだ 0 件で、語句一致に退化している。
         5 秒待つ別のクライアントなので、本番の予算切れはここでは見えない（check_memory で見る）。
    """
    query = urllib.parse.urlencode({"q": "preflight", "k": 1})
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/retrieve?{query}", timeout=5) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - どんな失敗も報告する
        status = "FAIL" if required else "WARN"
        report.add(name, status, f"届きません ({type(exc).__name__})", required=required)
        return
    if not isinstance(data, dict):
        report.add(name, "FAIL" if required else "WARN", "応答の形が不正です", required=required)
        return
    mode = data.get("search_mode")
    status = "PASS" if mode == "semantic" else "WARN"
    report.add(name, status, f"search_mode={mode}", required=required)


def check_c2(report: Report) -> None:
    """主 C2 と副 C2 の検索を確かめる。"""
    if not _c2_retriever_enabled():
        report.add("c2", "SKIP", "L2_ENABLE_TOOLS か L2_ENABLE_RAG か L2_USE_C2_RETRIEVER が無効")
        return
    _check_c2_endpoint(report, "c2", os.environ.get("L2_C2_URL", "http://127.0.0.1:8100"), required=True)
    readonly = os.environ.get("L2_C2_URL_READONLY", "").strip()
    if readonly:
        # 副 C2 は任意（dev から prod を参照する）。止まっていても本応答は続く。
        _check_c2_endpoint(report, "c2/readonly", readonly, required=False)


# _retrieve_memory_impl が失敗のときに返す文の先頭。本番はこの文をそのまま LLM へ渡す。
_MEMORY_ERROR_PREFIX = "記憶検索でエラーが発生しました"
# CompositeRetriever が子の例外を飲み込んだときのログの文言。C2 の子は通信の失敗を自分で空リストへ変えるので、
# これは seed の検索の失敗を指す。
_MEMORY_SWALLOWED = "例外発生"
# 戻り値の 1 件ごとの見出し。本文は読まず、ここだけを数える。
_MEMORY_HEADER = re.compile(r"^\[(\d+)\] \(source: (.*), score: [^)\n]*\)$", re.MULTILINE)


def _memory_summary(answer: str) -> tuple[int, str]:
    """retrieve_memory の戻り値から、件数と出どころの内訳だけを取り出す。

    Args:
        answer: _retrieve_memory_impl の戻り値（実発話を含むので表示しない）

    Returns:
        (件数, "c2=2 c2-recent=1 seed=2" の形の内訳)

    WHY: 番号が続く見出しだけを数えるので、本文に見出しと似た行があっても数がずれにくく、ずれても本文は漏れない。
         seed の source は手元のファイル名なので、C2 以外はまとめて seed と数える。
    """
    origins: dict[str, int] = {}
    total = 0
    for number, source in _MEMORY_HEADER.findall(answer):
        if int(number) != total + 1:
            continue
        total += 1
        origin = source.split(":", 1)[0] if source.startswith(("c2:", "c2-recent:")) else "seed"
        origins[origin] = origins.get(origin, 0) + 1
    return total, " ".join(f"{name}={origins[name]}" for name in sorted(origins))


def check_memory(report: Report, capture: _LogCapture, stream_id: str | None = None) -> None:
    """retrieve_memory を本番の入口で 1 回流す。

    Args:
        report:    結果の記録先
        capture:   ログの捕捉先
        stream_id: 点検の stream_id。L2_C2_RECENT_SCOPE=session では、これがないと /retrieve/recent を呼ばない

    WHY: 本番は失敗を文字列や空リストに変えて LLM へ渡すので、声は止まらず記憶だけが無音で消える。
         embeddings と c2 の行は部品を別々に見るだけで、c2 の行は 5 秒待つ別のクライアントを使う。
         本番の組み立て、既定 600 ms の予算、k、/retrieve/recent を通るのはここだけ。

         FAIL: エラー文が返った、または seed の検索が例外になった。C2 が有効だと seed の例外は CompositeRetriever が
               飲み込むが、C2 が無効な構成ならエラー文になる同じ壊れ方なので、C2 の有無で判定を変えない。
         WARN: C2 の打ち切りや HTTP エラー。1 回の遅さで配信を止めないため。
         戻り値は実発話を含むので表示しない。ms と件数と出どころの内訳だけを出す。
    """
    if not _memory_tool_enabled():
        report.add("memory", "SKIP", "L2_ENABLE_TOOLS か L2_ENABLE_RAG が無効")
        return
    from .mcp_servers.retrieve_memory import (
        _retrieve_memory_impl,
        reset_retrieval_context,
        set_retrieval_context,
    )

    # 本番の _generation_node と同じ形で渡す。今のターンの event_id は点検にないので、どれにも当たらない値にする。
    set_retrieval_context(stream_id=stream_id, exclude_event_ids=[str(uuid4())])
    mark = capture.mark()
    started = time.monotonic()
    try:
        # 埋め込みは SDK の既定で最長 30 分待つ。文脈は _with_deadline が引き継ぐ。
        answer = _with_deadline(lambda: _retrieve_memory_impl("preflight"), _EMBED_DEADLINE_SEC)
        # 戻り値の形は本番の変更で変わりうるので、読むところまで try の中に入れる。
        returned_error = answer.startswith(_MEMORY_ERROR_PREFIX)
        docs, origins = _memory_summary(answer)
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception("memory", exc)
        report.add("memory", "FAIL", _describe_exc(exc))
        return
    finally:
        reset_retrieval_context()

    elapsed_ms = int((time.monotonic() - started) * 1000)
    problems = capture.problems_since(mark)
    reasons = " / ".join(problems)
    # 引数には発話が入りうるので、定型文の部分だけを見る。
    if returned_error or any(_MEMORY_SWALLOWED in str(record.msg) for record in capture.records[mark:]):
        report.add("memory", "FAIL", f"{elapsed_ms}ms / {reasons or 'エラー文が返りました'}")
        return
    summary = f"{elapsed_ms}ms docs={docs} {origins}".rstrip()
    if problems:
        report.add("memory", "WARN", f"{summary} / {reasons}")
    elif not docs:
        report.add("memory", "WARN", f"{summary} / 検索は通りましたが 1 件も返りません")
    else:
        report.add("memory", "PASS", summary)


def check_tts_engine(report: Report, character: Any, *, required: bool) -> bool:
    """キャラの TTS エンジンの生死を記録し、使えるかどうかを返す。"""
    ok, engine = tts_engine_status(character)
    report.add(f"tts/{character.slug}", "PASS" if ok else "FAIL", engine, required=required)
    return ok


def _spoken_text(raw: str) -> str:
    """tts.done の text から、本番が読み上げる本文を取り出す。

    Agent 経路の text は JSON 全文なので、本番と同じ解釈を通してから判定する。
    """
    from .tts import _parse_voicepeak_json

    return _parse_voicepeak_json(raw)[0]


def _spoken_problems(text: str) -> list[str]:
    """読み上げる本文の問題を挙げる。pipeline、協働、単一ノードの本応答で同じ判定を使う。

    WHY: 本番はこの文字列をそのまま TTS へ渡す。{"response": null} は str() されて「None」と読み上げ、
         切れた JSON や response のない JSON は _parse_voicepeak_json を素通りして断片を読み上げる。
    """
    stripped = text.strip()
    if not stripped or stripped == "None":
        return ["読み上げる本文が空です"]
    if stripped.startswith(("{", "```")) or _is_list_repr(stripped):
        return ["JSON の断片が読み上げに入っています"]
    if stripped.startswith("エラー:"):
        return ["エラー文が読み上げに入っています"]
    return []


def _pipeline_problems(result: Any, slug: str, logs: list[str]) -> list[str]:
    """パイプラインの 1 ターンの問題を挙げる。"""
    types = [event.get("type") for event in result.events]
    problems = []
    if result.speaker != slug:
        problems.append(f"別のキャラへ流れました ({result.speaker})")
    if any("Agent 実行失敗" in message for message in logs):
        problems.append("Agent が失敗してフォールバックしました")
    if any("ask_character 協働先 Agent 失敗" in message for message in logs):
        problems.append("協働先の Agent が失敗しました")
    if any("ツール読み込み失敗" in message for message in logs):
        problems.append("ツールの読み込みに失敗しています")
    if _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        if not any("source=structured_response" in message for message in logs):
            problems.append("構造化出力が返りません")
    if "tts.done" not in types:
        problems.append("tts.done がありません")
    for event in result.events:
        if event.get("type") != "tts.done":
            continue
        text = (event.get("payload") or {}).get("text")
        # 協働先の失敗はログの判定で見る。ここは主応答に混ざった場合を見る。
        problems.extend(_spoken_problems(_spoken_text(text) if isinstance(text, str) else ""))
        # 0 フレームの音声でも本番の合成は例外にならず、声が出ないまま 1 ターンが終わる。
        # 欠落は数えない（本番の tts.done は必ず持つ値で、形が変わっただけで赤くしないため）。
        if (event.get("payload") or {}).get("duration_ms") == 0:
            problems.append("音声の長さが 0 です")
    return problems


def check_pipeline(
    report: Report,
    capture: _LogCapture,
    character: Any,
    stream_id: str,
    stream_context: str | None,
    *,
    required: bool,
) -> None:
    """emitter と同じ 1 ターンを流し、途中で無音のフォールバックがなかったかを確かめる。

    直接呼びと違って Redis への publish と TTS まで通る。_run_agent が内部で握りつぶすので、
    結果だけでなくログも見て判定する。
    """
    from .pipeline import run_pipeline
    from .router import _ROUTING_EXCLUDED_SLUGS

    name = f"pipeline/{character.slug}"
    if character.slug in _ROUTING_EXCLUDED_SLUGS:
        # ルーティングの対象外なので、speaker_hint を渡しても既定のキャラへ流れる。本番にない経路。
        report.add(name, "SKIP", "ルーティングの対象外", required=required)
        return

    mark = capture.mark()
    started = time.monotonic()
    try:
        result = _with_deadline(
            lambda: run_pipeline(
                f"{_call_name(character)}、配信前の点検です。ひとことだけ挨拶してください。",
                stream_id=stream_id,
                session_id=str(uuid4()),
                trace_id=str(uuid4()),
                speaker_hint=character.slug,
                stream_context=stream_context,
                # None だと _tts_node が OBS の立ち絵を即時に切り替え、最後の pose のまま残す。
                on_pose_ready=lambda *_args, **_kwargs: None,
            ),
            _PIPELINE_DEADLINE_SEC,
        )
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception(name, exc)
        report.add(name, "FAIL", _describe_exc(exc), required=required)
        return

    for event in result.events:
        if event.get("type") == "llm.final":
            payload = event.get("payload") or {}
            report.input_tokens += payload.get("input_tokens") or 0
            report.output_tokens += payload.get("output_tokens") or 0

    problems = _pipeline_problems(result, character.slug, capture.messages_since(mark))
    # 文言を挙げずにレベルで見るので、本番のログの文言が増えても追従が要らない。全文は問題ログに残る。
    errors, warnings = capture.problem_counts(mark)
    if errors:
        problems.append(f"ERROR のログ {errors} 件")
    detail = f"{time.monotonic() - started:.1f}s events={len(result.events)}"
    if problems:
        report.add(name, "FAIL", f"{detail} / {' / '.join(problems)}", required=required)
    elif warnings:
        # 本番が握りつぶした失敗（記憶の検索、bubble の発行など）。1 回の揺れでは配信を止めないので FAIL にしない。
        report.add(name, "WARN", f"{detail} / 警告のログ {warnings} 件", required=required)
    else:
        report.add(name, "PASS", detail, required=required)


def _collab_problems(logs: list[str], target_slug: str) -> list[str]:
    """協働の経路の問題をログから挙げる。

    WHY: 協働は失敗を黙って固定文やツールなしの呼び出しに変えるので、戻り値では判定できない。
         「通った」ことを示すログが無ければ FAIL にして、文言が変わった場合も安全側へ倒す。
    """
    problems = []
    if any("ask_character 協働先 Agent 失敗" in message for message in logs):
        problems.append("協働先の Agent が失敗しました")
    if any("導入セリフ LLM 生成失敗" in message for message in logs):
        problems.append("導入セリフの生成に失敗しました")
    elif not any("導入セリフ生成完了" in message for message in logs):
        problems.append("導入セリフが固定文へ落ちています")
    elif any(re.search(r"""導入セリフ生成完了: \[\w+\] \[(?:\{|'|"|\])""", message) for message in logs):
        problems.append("導入セリフが list の repr です")
    if not any(f"Agent 実行完了 [character={target_slug}]" in message for message in logs):
        # create_react_agent が消えると、ログなしでツールなしの call_llm へ落ちる。
        problems.append("協働先が Agent を通っていません")
    if any("Agent 実行失敗" in message for message in logs):
        problems.append("Agent が失敗してフォールバックしました")
    if any("導入セリフ TTS 失敗" in message or "協働応答 TTS 合成失敗" in message for message in logs):
        problems.append("協働の TTS に失敗しました")
    return problems


def _chunk_problems(chunks: list[tuple[str, str]], target_slug: str) -> list[str]:
    """協働で読み上げる本文の問題を挙げる。

    WHY: 本番はこの文字列をそのまま TTS へ渡す。_run_agent は finish_reason を "stop" に固定するので、
         max_tokens で切れた JSON やエラー文は、ここで中身を見ないと分からない。
    """
    problems = []
    if not any(character == target_slug for character, _ in chunks):
        problems.append("協働先の TTS が 1 つも届きません")
    for _, text in chunks:
        problems.extend(_spoken_problems(text))
    return sorted(set(problems))


def check_collab(
    report: Report,
    capture: _LogCapture,
    caller: Any,
    target: Any,
    stream_id: str,
    stream_context: str | None,
) -> None:
    """協働の経路を 1 往復流す。

    呼び出し側の Agent に ask_character を実際に呼ばせるので、ツール結果を受けた 2 段目の応答、
    導入セリフ、協働先の Agent、協働先の TTS を一度に通せる。音は鳴らさず、再生キューの代わりに
    チャンクを数えるだけにする。

    WHY: 通常の点検の発話は挨拶なのでツールを呼ばない。協働だけが壊れても緑のままになる。
    """
    from .graph import _build_agent_graph
    from .mcp_servers.ask_character import (
        _ask_character_impl,
        reset_ask_character_context,
        set_ask_character_context,
        wait_bg_tts_complete,
    )
    from .pipeline import _get_llm_mode

    name = f"collab/{caller.slug}->{target.slug}"
    _, provider, model = _get_llm_mode(caller)
    question = "深海で一番好きな生き物は何ですか"
    text = (
        f"[ルカからの発言]\n{_call_name(caller)}、配信前の点検です。"
        f"ask_character ツールで {target.slug} に「{question}」と聞いて、"
        "返事を受けてひとことだけまとめてください。"
    )
    session_id = str(uuid4())
    common = {"stream_id": stream_id, "session_id": session_id, "trace_id": str(uuid4())}
    chunks: list[tuple[str, str]] = []
    audit = _AgentAudit()
    summary: dict[str, Any] | None = None

    def collect_chunk(_url: str, chunk_text: str, _is_last: bool, character: str) -> None:
        # bridge filler は本文が空文字なので数えない。
        if chunk_text != "":
            chunks.append((character, chunk_text))

    mark = capture.mark()
    started = time.monotonic()
    set_ask_character_context(
        on_tts_chunk=collect_chunk,
        tts_output_dir=os.environ.get("L2_TTS_OUTPUT_DIR", "./data/audio"),
        common=common,
        caller_slug=caller.slug,
    )
    try:
        agent = _build_agent_graph(provider, model, _agent_prompt(caller, stream_context), character_slug=caller.slug)
        if agent is None:
            report.add(name, "FAIL", f"{model}: ツールが 0 件です")
            return
        result = _invoke_agent_directly(
            agent, text, slug=caller.slug, provider=provider, model=model, session_id=session_id, audit=audit,
        )
        summary = _summarize_agent_result(result)
        report.input_tokens += summary["input_tokens"]
        report.output_tokens += summary["output_tokens"]
        tool_called = "ask_character_tool" in summary["tools"]
        if not tool_called:
            # モデルの判断には揺れがあるので、協働の経路だけは直接呼んで確かめる。
            # 壁時計のない経路なので、ここにも打ち切りを置く。文脈は _with_deadline が引き継ぐ。
            _with_deadline(lambda: _ask_character_impl(target.slug, question), _PIPELINE_DEADLINE_SEC)
        wait_bg_tts_complete(session_id)
    except Exception as exc:  # noqa: BLE001 - どんな失敗も FAIL として報告する
        capture.note_exception(name, exc)
        # Agent が返った後の失敗なら、トークンはもう数えてある。2 重に足さない。
        calls = audit.record_failure(report, capture, name) if summary is None else ""
        report.add(name, "FAIL", f"{model}: {_describe_exc(exc)}{calls}")
        return
    finally:
        reset_ask_character_context()

    problems = _collab_problems(capture.messages_since(mark), target.slug)
    problems.extend(_agent_problems(summary, []))
    problems.extend(_chunk_problems(chunks, target.slug))
    detail = (
        f"{model} {time.monotonic() - started:.1f}s steps={summary['steps']}"
        f" called={','.join(summary['tools']) or '-'} chunks={len(chunks)}"
    )
    if problems:
        report.add(name, "FAIL", f"{detail} / {' / '.join(problems)}")
    elif not tool_called:
        report.add(name, "WARN", f"{detail} / モデルが呼ばなかったので、2 段目の応答は未確認です")
    else:
        report.add(name, "PASS", detail)


def _collab_pairs(characters: list[Any], mode: str) -> list[tuple[Any, Any]]:
    """--collab に応じて、流す組を返す。選ばれていないキャラを含む組は除く。"""
    by_slug = {character.slug: character for character in characters}
    pairs = [
        (by_slug[caller], by_slug[target])
        for caller, target in _COLLAB_PAIRS
        if caller in by_slug and target in by_slug
    ]
    return pairs[:1] if mode == "one" else pairs


def run_paid(
    report: Report,
    capture: _LogCapture,
    characters: list[Any],
    *,
    with_pipeline: bool,
    collab: str = "one",
) -> None:
    """全経路へ実際に 1 回ずつ流す。安い経路から順に流し、早く壊れた箇所を見せる。"""
    from .stream_context import load_stream_context

    stream_context = load_stream_context()
    stream_id = f"preflight-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report.add("stream_id", "INFO", stream_id)

    check_router(report, capture)
    for character in characters:
        check_filler(report, capture, character, required=character.slug in REQUIRED_CHARACTERS)
    check_tavily(report, capture)
    check_embeddings(report, capture)
    check_c2(report)
    check_memory(report, capture, stream_id)
    failed_main: set[str] = set()
    for character in characters:
        required = character.slug in REQUIRED_CHARACTERS
        # --no-pipeline は TTS を前提にしない（missing_services も見ない）ので、行だけ残して合否に数えない。
        engine_ok = check_tts_engine(report, character, required=required and with_pipeline)
        agent_ok = check_agent(report, capture, character, stream_context, required=required)
        if not agent_ok:
            failed_main.add(character.slug)
        if not with_pipeline:
            continue
        if not agent_ok:
            # 同じ Agent を通すので情報が増えない。_run_agent には recursion_limit がないので、壊れた
            # 応答が続くと壁時計まで待たされ、打ち切った後も裏で API を呼び続けて課金が増える。
            report.add(f"pipeline/{character.slug}", "SKIP", "本応答が FAIL なので流しません", required=required)
            continue
        if not engine_ok and not required:
            # 別枠のキャラの TTS が落ちていると分かっているなら、無駄に課金しない。
            report.add(f"pipeline/{character.slug}", "SKIP", "TTS エンジンが使えません", required=False)
            continue
        check_pipeline(report, capture, character, stream_id, stream_context, required=required)

    if not with_pipeline:
        report.add("collab", "SKIP", "--no-pipeline では Redis を前提にしないので流しません")
        return
    if not _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        # ツールが無効なら本番は単一ノードで、ask_character を使わない。
        report.add("collab", "SKIP", "L2_ENABLE_TOOLS が無効")
        return
    if collab == "none":
        report.add("collab", "WARN", f"--collab none。協働の経路は未確認なので、{_NOT_FOR_GATE}")
        return
    pairs = _collab_pairs(characters, collab)
    if not pairs:
        report.add("collab", "SKIP", "選んだキャラに協働の組がありません")
    for caller, target in pairs:
        if caller.slug in failed_main:
            # pipeline と同じ理由で流さない。_COLLAB_PAIRS の呼び出し側はどれも必須なので、終了コードはすでに 1。
            report.add(f"collab/{caller.slug}->{target.slug}", "SKIP", "呼び出し側の本応答が FAIL なので流しません")
            continue
        check_collab(report, capture, caller, target, stream_id, stream_context)


def _note_unmeasured_routes(report: Report, characters: list[Any]) -> None:
    """流していない必須の経路があれば、配信前の合格に使えないと知らせる。

    WHY: --characters で絞ると測っていないキャラは FAIL にならず、ツールが無効なら構造化出力と協働を
         通らない。どちらも終了コードは 0 なので、印がないと全経路の合格と読み違える。
    """
    selected = {character.slug for character in characters}
    omitted = [slug for slug in REQUIRED_CHARACTERS if slug not in selected]
    if omitted:
        report.add("characters", "WARN", f"{', '.join(omitted)} を流していないので、{_NOT_FOR_GATE}")
    if not _is_truthy(os.environ.get("L2_ENABLE_TOOLS")):
        report.add("tools", "WARN", f"L2_ENABLE_TOOLS が無効で、構造化出力と協働を通っていないので、{_NOT_FOR_GATE}")


# ─── CLI ──────────────────────────────────────────────────────────

class _Parser(argparse.ArgumentParser):
    """コマンドラインの誤りを、実行の拒否（2）と区別できる終了コードで返す。"""

    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    """コマンドラインの定義を返す。"""
    # WHY: argparse の既定は前方一致の省略を受けるので、--all が --allow-prod になって本番の拒否が外れる。
    parser = _Parser(prog="preflight", description="A.I.byss Lab Lounge — 配信前の点検", allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true", help="課金なしの点検")
    mode.add_argument("--paid", action="store_true", help="全経路へ実際に流す（少額の課金あり）")
    # WHY: 既定を持たせると、本番の起動前に dev の設定を測って緑を返す事故が起きる。
    parser.add_argument(
        "--profile", choices=("dev", "prod"), required=True,
        help="読む env ファイル。dev は .env + .env.dev、prod は .env のみ",
    )
    parser.add_argument("--allow-prod", action="store_true", help="--paid で本番ストリームへの書き込みを許可する")
    parser.add_argument("--characters", help="点検するキャラ（カンマ区切り。省略時は全キャラ）")
    parser.add_argument("--no-pipeline", action="store_true", help="--paid で publish と TTS まで通す 1 ターンと協働を省く")
    parser.add_argument(
        "--collab", choices=("one", "all", "none"), default="one",
        help="--paid で流す協働の組。one は sakura から chisame への 1 組、all はプロバイダを一巡する 3 組",
    )
    parser.add_argument("--trace", action="store_true", help="--paid で LangSmith への送信を止めない")
    return parser


def _select_characters(raw: str | None) -> list[Any]:
    """--characters を解釈する。省略時は全キャラ。

    Raises:
        KeyError: 未知のキャラ（main が終了コード 64 にする）
    """
    from .characters import get_all_characters, get_character

    if raw is None:
        return get_all_characters()
    return [get_character(slug.strip()) for slug in raw.split(",") if slug.strip()]


def _emit_verdict(code: int, report: Report | None = None) -> None:
    """最後の結果の行を出す。

    Args:
        code:   終了コード
        report: 流していない経路を調べる相手。点検を流す前に止まったときは None
    """
    line = f"判定: {_VERDICT_WORDS.get(code, '中断')} (exit {code})"
    if report is not None and code == EXIT_OK:
        # 終了コードは 0 でも、流していない経路があれば配信前の合格とは読ませない。
        gaps = [check.name for check in report.checks if _NOT_FOR_GATE in check.detail]
        if gaps:
            line += f"。ただし{_NOT_FOR_GATE}: {', '.join(gaps)}"
    _emit(line)


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    """点検を実行し、最後に判定の行を出して終了コードを返す。

    WHY: 終了コードは画面に出ないので、窓の表示や貼られた出力からは合否を読めない。途中で止まる経路が
         増えても出し忘れないよう、点検の本体の外側の 1 か所で出す。

    Args:
        argv:      コマンドライン引数（省略時は sys.argv）
        repo_root: .env と uv.lock のあるルート（テストで差し替える）

    Returns:
        終了コード（モジュール冒頭の説明を参照）

    Raises:
        SystemExit: コマンドラインの誤り（64）、.env が LLM の接続先を上書きしている（2）
    """
    report = Report()
    try:
        code = _run_checks(report, argv, repo_root)
    except SystemExit as exc:
        # --help の 0 は点検ではないので出さない。数字でない SystemExit は _cli が受ける。
        if isinstance(exc.code, int) and exc.code != EXIT_OK:
            _emit_verdict(exc.code)
        raise
    _emit_verdict(code, report)
    return code


def _run_checks(report: Report, argv: list[str] | None, repo_root: Path | None) -> int:
    """引数を読んで点検を流し、結果を report へ集めて終了コードを返す。"""
    # WHY: 出力先が cp932 の環境（リダイレクトなど）でも、日本語の判定結果で落ちないようにする。
    #      --help は parse_args の中で出力するので、それより先に済ませる。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)

    root = repo_root or _REPO_ROOT
    # L2_TTS_OUTPUT_DIR や L2_STREAM_CONTEXT_FILE は作業ディレクトリ相対なので、本番と揃える。
    os.chdir(root)
    inherited = [name for name in INHERITED_LLM_VARS if name in os.environ]
    files = prepare_env(args.profile, root)
    file_names = {name for path in files for name in dotenv_values(path)}
    production_outputs = silence_side_outputs(trace=args.trace) if args.paid else {}

    problem_log = root / "logs" / "preflight" / f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with _captured_logs(problem_log) as capture:
        from .bus import _stream_key

        stream = _stream_key()
        _emit(f"== preflight {'--paid' if args.paid else '--static'} profile={args.profile} stream={stream} ==")
        _emit(f"env: {', '.join(p.name for p in files) or '(なし)'}")

        # 本番の正史へ点検の行を入れないよう、何より先に止める。
        if args.paid and stream == PROD_STREAM and not args.allow_prod:
            # WHY: dev で本番へ向くのは .env.dev がないか REDIS_STREAM_KEY の誤りで、--allow-prod は答えではない。
            hint = (
                ".env.dev がないか、REDIS_STREAM_KEY が本番を指しています（雛形は .env.dev.example）。"
                "--allow-prod は本番の正史へ書くので、dev の点検では付けないでください。"
                if args.profile == "dev"
                else "本当に流すなら --allow-prod を付けてください。"
            )
            _emit(f"{PROD_STREAM} へ書き込むので止めました。{hint}")
            return EXIT_REFUSED

        try:
            characters = _select_characters(args.characters)
        except KeyError as exc:
            parser.error(f"未知のキャラです: {exc}")
        if not characters:
            # 測っていないのに緑を返さない。空の変数を渡した誤りでもあるので、全キャラへ広げもしない。
            parser.error("--characters にキャラが 1 つもありません")

        removed = [name for name in inherited if name not in os.environ]
        if removed:
            # WHY: 点検の中では外すが、同じシェルから起動する本番には残る。緑でも本番は別の相手へ行く。
            report.add(
                "env/inherited", "WARN",
                f"{', '.join(removed)} を外して測りました。同じシェルから本番を起動すると接続先や認証が変わります",
            )
        def print_results() -> None:
            """要約と、問題ログの場所を出す。課金の前に止まる経路でも同じ形にする。"""
            report.print_summary()
            if capture.wrote_problems:
                # import の失敗などは画面が型の名前だけなので、全文の場所を示さないと原因を読みに行けない。
                _emit(f"詳細: {_display_path(problem_log, root)}")

        # --no-pipeline は --paid の省略なので、--static のゲートを緩めない。
        run_static(
            report, root, args.profile, characters, file_names,
            paid=args.paid, with_pipeline=not (args.paid and args.no_pipeline), capture=capture,
        )
        _report_silenced_outputs(report, production_outputs)
        if args.paid:
            if report.has_required_failure():
                print_results()
                _emit("設定に FAIL があるので、課金する前に止めました。")
                return report.exit_code()
            missing = missing_services(characters, with_pipeline=not args.no_pipeline)
            if missing:
                print_results()
                _emit(f"前提サービスがないので、課金する前に止めました: {', '.join(missing)}")
                return EXIT_MISSING_SERVICE
            run_paid(report, capture, characters, with_pipeline=not args.no_pipeline, collab=args.collab)
            if args.no_pipeline:
                report.add("no-pipeline", "WARN", f"publish と TTS を通していないので、{_NOT_FOR_GATE}")
            _note_unmeasured_routes(report, characters)
        print_results()
        return report.exit_code()


def _cli() -> None:
    """コマンドラインの入口。

    WHY: 想定外の例外で traceback を素で出すと、URL のパスワードなどが値のまま stderr に残る。
         また、打ち切った Agent の worker は interpreter の終了時に join されるので、普通に終わると
         結果を出した後もプロセスが残り、API を呼び続けることがある。結果を出したら即座に終える。
    """
    logging.lastResort = None
    code = EXIT_FAILED
    try:
        try:
            code = main()
        except SystemExit as exc:
            if isinstance(exc.code, int):
                code = exc.code
            else:
                # sys.exit("文言") の文言は値を含みうるので、止まったことだけを知らせる。
                _emit("[preflight] 想定外の終了: SystemExit")
                _emit_verdict(code)
        except KeyboardInterrupt:
            code = 130
            _emit_verdict(code)
        except Exception as exc:  # noqa: BLE001 - 最後の砦。値を出さずに型だけを知らせる
            _emit(f"[preflight] 想定外のエラー: {type(exc).__name__}")
            _emit_verdict(code)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # 閉じたパイプへの書き込みや stdout がない環境で例外が出ても、必ずここを通して worker を残さない。
        os._exit(code)


if __name__ == "__main__":
    _cli()
