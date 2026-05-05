"""
stream_context.py — 配信単位の文脈ロード

責務:
  - 「今日の配信内容」を記したファイル (Markdown) をディスクから読み込む
  - 配信開始時 (run_loop / run_once 起動時) に1度だけ呼ばれる前提
  - 読み込んだ文脈は graph.py の routing ノードで system_prompt にマージされる
    (キャラ素体 → ## 本日の配信 → ## 参照情報 という3層構造の中段)

【設計判断】
  characters.py から分離した理由:
    characters.py は「AITuber キャラクターレジストリ」が責務。
    配信文脈は「セッション/配信レベルの情報」でキャラとは独立した関心事。
    L2 リポの慣習 (router.py / kb_loader.py / skill_loader.py 等) に倣い、
    新しい関心事は新モジュールに切り出す。

【ファイル解決順序】
  1. 引数 path (テスト・明示指定用)
  2. 環境変数 L2_STREAM_CONTEXT_FILE (絶対パスまたは作業ディレクトリ相対)
  3. デフォルト: <repo_root>/data/stream_context/current.md

【後方互換性】
  ファイル未存在 / 空ファイル時は None を返す。
  呼び出し側 (graph.py routing) は None なら system_prompt にマージしないので、
  この機能を導入する以前と完全に同じ挙動になる。
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# デフォルトパス: src/lab_lounge/stream_context.py から見て
# 2 階層上 (= リポジトリルート) の data/stream_context/current.md
_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "stream_context"
    / "current.md"
)


def load_stream_context(path: Path | None = None) -> str | None:
    """
    配信文脈 Markdown ファイルを読み込む。

    Args:
        path: 明示パス。
              None の場合は 環境変数 L2_STREAM_CONTEXT_FILE → デフォルトパス
              の順で解決する。

    Returns:
        ファイル本文 (前後空白 strip 後)。
        ファイル未存在 / 空文字列なら None。

    【None 返却の WHY】
        未使用時に "## 本日の配信" 見出しだけが空っぽで system_prompt に
        差し込まれると、LLM が「設問が欠落している」「文脈が抜けている」と
        誤解しがち (空表題は強い注意焦点になる)。
        None を返せば後段でマージをスキップでき、従来挙動と完全互換になる。
    """
    if path is None:
        env_path = os.environ.get("L2_STREAM_CONTEXT_FILE")
        path = Path(env_path) if env_path else _DEFAULT_PATH

    if not path.is_file():
        logger.info(
            "配信文脈ファイルなし: %s (system_prompt に追加マージしない)",
            path,
        )
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        logger.info(
            "配信文脈ファイルが空: %s (system_prompt に追加マージしない)",
            path,
        )
        return None

    logger.info("配信文脈をロード: %s (%d 文字)", path, len(text))
    return text
