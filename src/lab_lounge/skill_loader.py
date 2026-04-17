"""
skill_loader.py — Skills 定義ファイルの読み込みモジュール

Sprint Axis D Block 4: Skills 定義 v0.1

Skills は Agent の行動判断基準を宣言的に定義する Markdown ファイル群。
システムプロンプトに注入する形で Agent に「いつ・どのツールを・どう使うか」の規範を与える。

2 層構造:
  - 共通 Skill (skills/common/*.md): 全 AITuber が共有するツール選択ルール・行動パターン
  - キャラ別 Skill (skills/characters/{slug}.md): キャラ固有の判断バイアス

使い方:
  from lab_lounge.skill_loader import build_skills_prompt
  skills_text = build_skills_prompt("mimi")
  # → 共通 Skill + ミミ固有バイアス を結合した文字列
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# ─── キャッシュ ─────────────────────────────────────────────────────
# ファイル内容は実行中に変わらない想定。初回読み込み後はキャッシュする。
_common_cache: str | None = None
_character_cache: dict[str, str] = {}


def _read_file(path: Path) -> str:
    """UTF-8 でファイルを読み込む。存在しなければ空文字を返す。"""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.debug("Skill ファイル未検出 (skip): %s", path)
        return ""
    except Exception as exc:
        logger.warning("Skill ファイル読み込み失敗: %s (%s)", path, exc)
        return ""


def load_common_skills(skills_dir: Path | None = None) -> str:
    """
    skills/common/*.md を全て読み込み、結合した文字列を返す。

    ファイルはアルファベット順に読み込まれる（tool_routing が先、comfort が後など）。
    結果はキャッシュされ、2 回目以降はファイル I/O を行わない。

    Args:
        skills_dir: Skills ルートディレクトリ。省略時は src/lab_lounge/skills/

    Returns:
        結合された共通 Skill テキスト。ファイルなしの場合は空文字。
    """
    global _common_cache
    if _common_cache is not None and skills_dir is None:
        return _common_cache

    base = (skills_dir or _SKILLS_DIR) / "common"
    if not base.is_dir():
        logger.debug("共通 Skill ディレクトリなし: %s", base)
        return ""

    parts: list[str] = []
    for md_file in sorted(base.glob("*.md")):
        content = _read_file(md_file)
        if content:
            parts.append(content)

    result = "\n\n".join(parts)

    if skills_dir is None:
        _common_cache = result

    logger.info("共通 Skill 読み込み完了: %d ファイル (%d chars)", len(parts), len(result))
    return result


def load_character_skill(slug: str, skills_dir: Path | None = None) -> str:
    """
    skills/characters/{slug}.md を読み込み、文字列を返す。

    ファイルが存在しなければ空文字を返す（fail-open）。
    結果はキャッシュされ、同一 slug の 2 回目以降はファイル I/O を行わない。

    Args:
        slug:       キャラクター slug (e.g., "mimi", "chisame")
        skills_dir: Skills ルートディレクトリ。省略時は src/lab_lounge/skills/

    Returns:
        キャラ別 Skill テキスト。ファイルなしの場合は空文字。
    """
    if slug in _character_cache and skills_dir is None:
        return _character_cache[slug]

    base = (skills_dir or _SKILLS_DIR) / "characters"
    path = base / f"{slug}.md"
    result = _read_file(path)

    if skills_dir is None:
        _character_cache[slug] = result

    if result:
        logger.info("キャラ別 Skill 読み込み完了: %s (%d chars)", slug, len(result))
    return result


def build_skills_prompt(slug: str, skills_dir: Path | None = None) -> str:
    """
    共通 Skill + キャラ別 Skill を結合した文字列を返す。

    graph.py の _build_agent_graph から呼ばれ、system_prompt と合わせて
    create_react_agent の prompt に渡される。

    Args:
        slug:       キャラクター slug
        skills_dir: Skills ルートディレクトリ（テスト用）

    Returns:
        結合された Skills テキスト。Skill なしの場合は空文字。
    """
    common = load_common_skills(skills_dir)
    character = load_character_skill(slug, skills_dir)

    parts = [p for p in [common, character] if p]
    return "\n\n".join(parts)


def reset_cache() -> None:
    """テスト用: キャッシュをクリアする。"""
    global _common_cache
    _common_cache = None
    _character_cache.clear()
