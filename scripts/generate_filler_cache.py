"""
generate_filler_cache.py — フィラー音声一括生成スクリプト

全キャラクター（または指定キャラクター）のフィラーフレーズを
TTS で合成し、data/filler_cache/ にキャッシュする。

【使い方】
  # 全キャラクター一括生成
  uv run python scripts/generate_filler_cache.py

  # 特定キャラクターのみ
  uv run python scripts/generate_filler_cache.py --slug mimi

  # 強制再生成（既存キャッシュを上書き）
  uv run python scripts/generate_filler_cache.py --force

  # ドライラン（生成せず対象を表示）
  uv run python scripts/generate_filler_cache.py --dry-run

【前提】
  uv sync --extra tts
  VOICEPEAK がインストール済み（voicepeak キャラクター用）
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フィラー音声を一括生成してキャッシュする",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="特定キャラクターのみ生成（省略時は全キャラクター）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存キャッシュを上書きして再生成",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="生成せず対象を表示するだけ",
    )
    args = parser.parse_args()

    from lab_lounge.characters import get_all_characters, get_character
    from lab_lounge.filler import (
        ensure_filler_cache,
        get_filler_duration_ms,
        load_filler_phrases,
    )

    # 対象キャラクターを決定
    if args.slug:
        char = get_character(args.slug)
        if char is None:
            print(f"キャラクターが見つかりません: {args.slug}", file=sys.stderr)
            raise SystemExit(1)
        slugs = [args.slug]
    else:
        slugs = [c.slug for c in get_all_characters()]

    total_generated = 0

    for slug in slugs:
        phrase_set = load_filler_phrases(slug)
        if len(phrase_set) == 0:
            print(f"  [{slug}] フレーズなし — スキップ")
            continue

        print(f"\n{'='*60}")
        print(
            f"  [{slug}] opener={len(phrase_set.opener)} "
            f"continue={len(phrase_set.continue_)} "
            f"bridge={len(phrase_set.bridge)} "
            f"closer={len(phrase_set.closer)} "
            f"handraise={len(phrase_set.handraise)} "
            f"total={len(phrase_set)}"
        )
        print(f"{'='*60}")

        for cat, phrase in phrase_set.all_phrases:
            emotion_str = f"  ({phrase.emotion})" if phrase.emotion else ""
            print(f"  [{cat:>8}] {phrase.text}{emotion_str}")

        if args.dry_run:
            continue

        # 生成実行
        paths_by_cat = ensure_filler_cache(slug, force=args.force)

        for cat in ("opener", "continue", "bridge", "closer", "handraise"):
            for p in paths_by_cat.get(cat, []):
                duration = get_filler_duration_ms(p)
                print(f"  → [{cat:>8}] {p.name}  ({duration} ms)")
                total_generated += 1

    if not args.dry_run:
        print(f"\n完了: {total_generated} ファイル生成")


if __name__ == "__main__":
    main()
