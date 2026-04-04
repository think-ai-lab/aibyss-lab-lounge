"""
router.py — 発話ルーティング

責務:
  - 発話テキストやウェイクワード検知結果から、どのキャラクターが応答すべきかを決定する
  - LLM 呼び出しの前に実行される軽量なモジュール
  - LangGraph ノードではなく独立モジュールとして実装

【ルーティング優先順位】
  1. name_hint (ウェイクワード検知結果) → 一致するキャラクターを選択
  2. テキスト内のキャラクター名マッチ (単一名 → 即確定)
  3. 複数キャラクター名検出 → LLM ルーティング (L2_USE_LLM_ROUTER=true 時)
  4. デフォルトキャラクター (L2_DEFAULT_SPEAKER)
"""

import logging
import os
import re
from dataclasses import dataclass

from .characters import (
    CharacterConfig,
    get_all_characters,
    get_default_character,
)

logger = logging.getLogger(__name__)

_LLM_ROUTER_MODEL = os.environ.get("L2_LLM_ROUTER_MODEL", "claude-haiku-4-5-20251001")


@dataclass(frozen=True)
class RoutingDecision:
    """ルーティング結果。"""

    speaker: str  # character slug
    reason: str   # "name_hint" | "text_match" | "llm_router" | "default"


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


def _all_names(c: CharacterConfig) -> list[str]:
    """キャラクターの全呼称リストを返す（wake_word, display_name, aliases）。"""
    names: list[str] = []
    if c.wake_word:
        names.append(c.wake_word)
    names.append(c.display_name)
    names.extend(c.aliases)
    return names


def _find_all_matches(text: str, characters: list[CharacterConfig]) -> list[CharacterConfig]:
    """テキスト内で名前が出現するキャラクターを全て返す（重複なし、出現順）。"""
    seen: set[str] = set()
    result: list[CharacterConfig] = []
    for c in characters:
        if c.slug in seen:
            continue
        for name in _all_names(c):
            if name in text:
                seen.add(c.slug)
                result.append(c)
                break
    return result


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


def _is_llm_router_enabled() -> bool:
    """LLM ルーター機能が有効かどうか。"""
    return os.environ.get("L2_USE_LLM_ROUTER", "").lower() in ("1", "true", "yes")


def _detect_provider(model: str) -> str:
    """モデル名からプロバイダーを自動判定する。"""
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    return "openai"


def _call_router_llm(
    model: str,
    system_prompt: str,
    user_text: str,
) -> str | None:
    """
    ルーター用の軽量 LLM 呼び出し。

    モデル名からプロバイダーを自動判定し、適切なクライアントを使用する。
    将来モデルを変更しても、モデル名の変更だけで対応可能。
    """
    provider = _detect_provider(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=20,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
            )
            return resp.content[0].text.strip().lower()

        elif provider == "google":
            import google.genai as genai
            client = genai.Client()
            resp = client.models.generate_content(
                model=model,
                contents=user_text,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=20,
                    temperature=0,
                ),
            )
            return resp.text.strip().lower()

        else:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=20,
                temperature=0,
            )
            return resp.choices[0].message.content.strip().lower()

    except ImportError as exc:
        logger.warning("LLM ルーター: %s パッケージなし: %s", provider, exc)
        return None
    except Exception as exc:
        logger.error("LLM ルーター呼び出し失敗: %s", exc)
        return None


def _route_by_llm(
    text: str,
    candidates: list[CharacterConfig],
) -> CharacterConfig | None:
    """
    LLM に「誰に話しかけていますか？」と聞いて判定する。

    L2_LLM_ROUTER_MODEL でモデルを指定可能。
    モデル名からプロバイダー（OpenAI / Anthropic / Google）を自動判定する。
    応答は slug のみを返すよう指示し、パース失敗時は None を返す。
    """
    slug_to_names = {
        c.slug: _all_names(c) for c in candidates
    }
    candidates_desc = "\n".join(
        f"- {slug}: {', '.join(names)}" for slug, names in slug_to_names.items()
    )
    valid_slugs = [c.slug for c in candidates]

    system_prompt = (
        "あなたは発話ルーティングの判定器です。\n"
        "ユーザーの発話テキストを読み、話しかけている相手のキャラクターを判定してください。\n"
        "「言及」ではなく「呼びかけ」の対象を選んでください。\n"
        f"候補:\n{candidates_desc}\n\n"
        f"回答は slug のみを1つ返してください（{', '.join(valid_slugs)} のいずれか）。"
    )

    model = os.environ.get("L2_LLM_ROUTER_MODEL", _LLM_ROUTER_MODEL)
    answer = _call_router_llm(model, system_prompt, text)

    if answer is None:
        return None

    logger.info("LLM ルーター応答: %r (model=%s)", answer, model)

    for c in candidates:
        if answer == c.slug:
            return c

    logger.warning("LLM ルーター応答 %r が候補に一致しない。", answer)
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
    all_matches = _find_all_matches(text, characters)

    if len(all_matches) == 1:
        # 単一名 → 即確定
        matched = all_matches[0]
        logger.info("ルーティング: テキストマッチ(単一) → %s", matched.slug)
        return RoutingDecision(speaker=matched.slug, reason="text_match")

    if len(all_matches) >= 2 and _is_llm_router_enabled():
        # 3. 複数名検出 → LLM ルーティング
        logger.info(
            "複数キャラクター検出: %s → LLM ルーターへ委譲",
            [c.slug for c in all_matches],
        )
        llm_result = _route_by_llm(text, all_matches)
        if llm_result:
            logger.info("ルーティング: LLM ルーター → %s", llm_result.slug)
            return RoutingDecision(speaker=llm_result.slug, reason="llm_router")
        # LLM 失敗 → 従来パターンマッチでフォールバック
        logger.warning("LLM ルーター失敗。パターンマッチへフォールバック。")

    # 複数名でも LLM 無効/失敗時、または 0 マッチ時 → 従来ロジック
    if all_matches:
        matched = _match_by_text(text, characters)
        if matched:
            logger.info("ルーティング: テキストマッチ → %s", matched.slug)
            return RoutingDecision(speaker=matched.slug, reason="text_match")

    # 4. デフォルト
    default = get_default_character()
    logger.info("ルーティング: デフォルト → %s", default.slug)
    return RoutingDecision(speaker=default.slug, reason="default")
