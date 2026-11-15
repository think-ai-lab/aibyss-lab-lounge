"""
decision_log.py — 判定 (ルーター / 意図ゲート / 挙手候補 / 承認) の入出力を JSONL で残す (オプトイン)

責務:
  - router.py の 4 つの判定が LLM に渡した入力と、返ってきた生の応答、解釈した結果を、
    1 判定 1 行の JSON として追記する
  - 古いファイルを保持日数で消す
  - 書けなくても判定を止めない (ログは補助の情報。配信のターンを落とさない)

WHY 残すか:
  判定のログ (INFO) は入力を 30 文字に切り、使ったトークンと時間も捨てている。判定の精度を測る、
  別のモデルへ替えた結果を比べる (リプレイ) には、「何を聞いて、何が返り、どう解釈したか」の全文が要る。

WHY 既定はオフ:
  入力の全文は **実発話そのもの** (配信者や視聴者の声の文字起こし) なので、平文で残る。
  ログの保存先 logs/ は .gitignore 済みだが、実発話を平文で溜めるので、使う人が分かったうえで
  明示的にオンにする (.env に L2_DECISION_LOG=true)。preflight --static はオンのとき WARN を出し、
  --paid は点検の発話がリプレイの入力に混ざらないよう強制でオフにする。

【環境変数】
  L2_DECISION_LOG                — true で有効 (デフォルト: "false")
  L2_DECISION_LOG_RETENTION_DAYS — 何日前までのファイルを残すか (デフォルト: 30)

【出力ファイル】
  logs/decisions/decisions_YYYYMMDD.jsonl  (カレントディレクトリからの相対。1 日 1 ファイル)

【1 行の中身】
  ts                 時刻 (ISO 8601、ローカルのオフセット付き)
  kind               "route" / "intent" / "interjection" / "approval"
  model              判定に使ったモデル名
  character          判定対象のキャラ slug (意図ゲートのみ。ほかは null)
  candidates         候補の slug の一覧 (無ければ null)
  input              LLM に渡した発話の全文
  raw                LLM の生の応答 (呼び出しに失敗したときは null)
  result             解釈した結果 (文字列か dict。解釈できなかったときは null)
  elapsed_ms         _call_router_llm の前後で測った所要時間
  error              呼び出しが失敗したときの例外の型名 (成功なら null)
  system_prompt_sha  system prompt の SHA-256 の先頭 16 文字 (全文は残さない。版の見分け用)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS = 30
_FILE_PATTERN = re.compile(r"^decisions_(\d{8})\.jsonl$")

# 書き込みの直列化と、「その日の最初の書き込みだけ掃除する」「警告は 1 回だけ」の状態。
# WHY lock: 判定は dispatcher と背景のスレッドの両方から呼ばれうる。1 行が混ざらないようにする。
_lock = threading.Lock()
_cleaned_for: date | None = None
_warned = False


def is_enabled() -> bool:
    """判定ログが有効か。デフォルト false。

    WHY 毎回環境変数を読む: preflight --paid が実行中に強制でオフにする。値をキャッシュすると、
    すでに import 済みのプロセスでその切り替えが効かない。
    """
    return os.environ.get("L2_DECISION_LOG", "false").strip().lower() in ("true", "1", "yes")


def _retention_days() -> int:
    """保持日数。読めない値や負の値は既定の 30 日にする (掃除で全消しにしないため)。"""
    raw = os.environ.get("L2_DECISION_LOG_RETENTION_DAYS", "")
    try:
        days = int(raw)
    except ValueError:
        return _DEFAULT_RETENTION_DAYS
    return days if days >= 0 else _DEFAULT_RETENTION_DAYS


def _log_dir() -> Path:
    """書き先のディレクトリ (カレントディレクトリからの相対)。"""
    return Path("logs") / "decisions"


def prompt_hash(system_prompt: str) -> str:
    """system prompt の SHA-256 の先頭 16 文字。全文は残さず、版が変わったことだけ分かるようにする。"""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _to_jsonable(value: Any) -> Any:
    """dataclass (IntentResult / ApprovalResult) を dict にし、ほかはそのまま返す。"""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _remove_old_files(directory: Path, today: date) -> None:
    """保持日数より古い decisions_YYYYMMDD.jsonl を消す。

    WHY ファイル名の日付で見る: mtime は、コピーや同期で変わる。名前の日付なら、
    「その日のログ」という意味が崩れない。名前の形に合わないファイルは触らない。
    """
    oldest_kept = today - timedelta(days=_retention_days())
    for path in directory.iterdir():
        match = _FILE_PATTERN.match(path.name)
        if match is None:
            continue
        try:
            file_date = datetime.strptime(match[1], "%Y%m%d").date()
        except ValueError:
            continue
        if file_date < oldest_kept:
            try:
                path.unlink()
            except OSError as exc:
                logger.debug("判定ログ: 古いファイルを消せませんでした: %s (%s)", path.name, exc)


def record(
    kind: str,
    *,
    model: str,
    system_prompt: str,
    input_text: str,
    raw: str | None,
    result: Any,
    elapsed_ms: int,
    character: str | None = None,
    candidates: list[str] | None = None,
    error: str | None = None,
) -> None:
    """判定 1 件を 1 行で追記する。オフのときは何もしない。例外は外へ出さない。

    Args:
        kind: "route" / "intent" / "interjection" / "approval"
        model: 判定に使ったモデル名
        system_prompt: 判定に渡した system prompt (全文は残さず、ハッシュだけ残す)
        input_text: 判定に渡した発話の全文
        raw: LLM の生の応答。呼び出しに失敗したときは None
        result: 解釈した結果 (文字列、dict、dataclass、None)
        elapsed_ms: LLM 呼び出しの所要時間 (ミリ秒)
        character: 判定対象のキャラ slug (意図ゲートのみ)
        candidates: 候補の slug の一覧
        error: 呼び出しが失敗したときの例外の型名

    WHY 例外を握る: 書けない理由 (ディスクの空き、権限、ファイルを別のソフトが開いている) は
    判定の正しさと関係ない。判定が止まると、配信の応答が止まる。警告は 1 回だけ出す
    (毎判定で出すと、コンソールが警告で埋まる)。
    """
    if not is_enabled():
        return
    global _cleaned_for, _warned
    try:
        now = datetime.now().astimezone()
        row = {
            "ts": now.isoformat(timespec="milliseconds"),
            "kind": kind,
            "model": model,
            "character": character,
            "candidates": candidates,
            "input": input_text,
            "raw": raw,
            "result": _to_jsonable(result),
            "elapsed_ms": elapsed_ms,
            "error": error,
            "system_prompt_sha": prompt_hash(system_prompt),
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        directory = _log_dir()
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            today = now.date()
            if _cleaned_for != today:
                # その日の最初の書き込みで 1 回だけ掃除する (毎回ディレクトリを走査しない)。
                # 掃除に失敗しても、今の 1 行は書く (掃除はおまけ)。
                _cleaned_for = today
                try:
                    _remove_old_files(directory, today)
                except OSError as exc:
                    logger.debug("判定ログ: 古いファイルの掃除に失敗: %s", exc)
            with open(directory / f"decisions_{today:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:  # noqa: BLE001
        if not _warned:
            _warned = True
            logger.warning(
                "判定ログを書けませんでした (判定は続けます。以降の失敗は警告を省きます): %s: %s",
                type(exc).__name__, exc,
            )
