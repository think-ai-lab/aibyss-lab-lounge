"""
router.py — 発話ルーティング

責務:
  - 発話テキストやウェイクワード検知結果から、どのキャラクターが応答すべきかを決定する
  - LLM 呼び出しの前に実行される軽量なモジュール（LLM は呼ばない）
  - LangGraph ノードではなく独立モジュールとして実装

【ルーティング優先順位】
  1. name_hint (ウェイクワード検知結果) → 一致するキャラクターを選択
  2. テキスト内のキャラクター名マッチ (wake_word, display_name, aliases)
  3. デフォルトキャラクター (L2_DEFAULT_SPEAKER)
"""

import logging
import re
from dataclasses import dataclass

from .characters import (
    CharacterConfig,
    get_all_characters,
    get_default_character,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """ルーティング結果。"""

    speaker: str  # character slug
    reason: str   # "name_hint" | "text_match" | "default"


def _match_by_hint(name_hint: str, characters: list[CharacterConfig]) -> CharacterConfig | None:
    """name_hint (ウェイクワード検知結果) でキャラクターを検索する。"""
    hint_lower = name_hint.strip()
    for c in characters:
        # slug 一致
        if hint_lower == c.slug:
            return c
        # wake_word 一致
        if c.wake_word and hint_lower == c.wake_word:
            return c
        # display_name 一致
        if hint_lower == c.display_name:
            return c
        # aliases 一致
        if hint_lower in c.aliases:
            return c
    return None


def _match_by_text(text: str, characters: list[CharacterConfig]) -> CharacterConfig | None:
    """テキスト内のキャラクター名を検索する。最初にマッチしたキャラクターを返す。"""
    for c in characters:
        # wake_word でマッチ（最も意図的な呼びかけ）
        if c.wake_word and c.wake_word in text:
            return c

    for c in characters:
        # display_name でマッチ
        if c.display_name in text:
            return c

    for c in characters:
        # aliases でマッチ
        for alias in c.aliases:
            if alias in text:
                return c

    return None


def route(
    text: str,
    *,
    name_hint: str | None = None,
    last_speaker: str | None = None,
) -> RoutingDecision:
    """
    発話テキストからルーティング先キャラクターを決定する。

    Args:
        text:         ユーザー発話テキスト
        name_hint:    ウェイクワード検知結果の slug または名前 (optional)
        last_speaker: 前回応答したキャラクターの slug (optional, 将来の round-robin 用)

    Returns:
        RoutingDecision (speaker slug + reason)
    """
    characters = get_all_characters()

    # 1. name_hint (ウェイクワード検知)
    if name_hint:
        matched = _match_by_hint(name_hint, characters)
        if matched:
            logger.info("ルーティング: name_hint=%r → %s", name_hint, matched.slug)
            return RoutingDecision(speaker=matched.slug, reason="name_hint")
        logger.warning("name_hint=%r に一致するキャラクターなし。テキストマッチへ。", name_hint)

    # 2. テキスト内のキャラクター名マッチ
    matched = _match_by_text(text, characters)
    if matched:
        logger.info("ルーティング: テキストマッチ → %s", matched.slug)
        return RoutingDecision(speaker=matched.slug, reason="text_match")

    # 3. デフォルト
    default = get_default_character()
    logger.info("ルーティング: デフォルト → %s", default.slug)
    return RoutingDecision(speaker=default.slug, reason="default")
