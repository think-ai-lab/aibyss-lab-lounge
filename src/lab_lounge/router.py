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
from pathlib import Path
from typing import Literal

from .characters import (
    CharacterConfig,
    get_all_characters,
    get_default_character,
)

logger = logging.getLogger(__name__)

_LLM_ROUTER_MODEL = os.environ.get("L2_LLM_ROUTER_MODEL", "claude-haiku-4-5-20251001")

# ログ強化 L-3 (Phase 0.5-A 後): ルーティング/判定ログに「どの発話に対する判定か」
# が分かるように発話テキスト先頭をログに含める。長すぎると 1 行が見にくいので
# 30 文字で打ち切る。
_LOG_TEXT_PREVIEW_CHARS = 30


def _text_preview(text: str | None) -> str:
    """ログ用の発話テキスト preview を整形する。

    repr で改行・特殊文字をエスケープし、30 文字超は ... で切り詰め。
    None / 空文字は空文字を返す (= ログでは省略される側で安全)。
    """
    if not text:
        return ""
    if len(text) > _LOG_TEXT_PREVIEW_CHARS:
        return repr(text[:_LOG_TEXT_PREVIEW_CHARS]) + "..."
    return repr(text)


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
) -> CharacterConfig | Literal["none"] | None:
    """
    LLM に「誰に話しかけていますか？」と聞いて判定する。

    L2_LLM_ROUTER_MODEL でモデルを指定可能。
    モデル名からプロバイダー（OpenAI / Anthropic / Google）を自動判定する。

    Returns:
        CharacterConfig - ルーティング先キャラクター
        "none"          - LLM が「呼びかけなし（全て言及）」と判定
        None            - LLM 呼び出し失敗 / パース失敗（パターンマッチへフォールバック）
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
        "呼びかけが存在しない場合（全て言及のみ）は 'none' と答えてください。\n"
        f"候補:\n{candidates_desc}\n\n"
        f"回答は以下のいずれか一語のみ: {', '.join(valid_slugs)}, none"
    )

    model = os.environ.get("L2_LLM_ROUTER_MODEL", _LLM_ROUTER_MODEL)
    answer = _call_router_llm(model, system_prompt, text)

    if answer is None:
        return None

    # ログ強化 L-3: text preview を含めて「何の発話への応答か」を識別容易に
    logger.info(
        "LLM ルーター応答: %r (model=%s text=%s)",
        answer, model, _text_preview(text),
    )

    if answer == "none":
        return "none"

    for c in candidates:
        if answer == c.slug:
            return c

    logger.warning(
        "LLM ルーター応答 %r が候補に一致しない (text=%s)。",
        answer, _text_preview(text),
    )
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

    # ログ強化 L-3: ルーティング系ログに発話 text preview を含めて「どの発話に対する
    # 判定か」を識別容易にする (= 連続するルーティング呼出を log で trace しやすく)
    text_p = _text_preview(text)

    # 1. name_hint (ウェイクワード検知)
    if name_hint:
        matched = _match_by_hint(name_hint, characters)
        if matched:
            logger.info(
                "ルーティング: name_hint=%r → %s (text=%s)",
                name_hint, matched.slug, text_p,
            )
            return RoutingDecision(speaker=matched.slug, reason="name_hint")
        logger.warning(
            "name_hint=%r に一致するキャラクターなし。テキストマッチへ。 (text=%s)",
            name_hint, text_p,
        )

    # 2. テキスト内のキャラクター名マッチ
    all_matches = _find_all_matches(text, characters)

    if len(all_matches) == 1:
        # 単一名 → 即確定
        matched = all_matches[0]
        logger.info(
            "ルーティング: テキストマッチ(単一) → %s (text=%s)",
            matched.slug, text_p,
        )
        return RoutingDecision(speaker=matched.slug, reason="text_match")

    if len(all_matches) >= 2 and _is_llm_router_enabled():
        # 3. 複数名検出 → LLM ルーティング
        logger.info(
            "複数キャラクター検出: %s → LLM ルーターへ委譲 (text=%s)",
            [c.slug for c in all_matches], text_p,
        )
        llm_result = _route_by_llm(text, all_matches)
        if isinstance(llm_result, CharacterConfig):
            logger.info(
                "ルーティング: LLM ルーター → %s (text=%s)",
                llm_result.slug, text_p,
            )
            return RoutingDecision(speaker=llm_result.slug, reason="llm_router")
        if llm_result == "none":
            # LLM が「呼びかけなし（全て言及）」と判定 → デフォルト扱い
            # 意図ゲートは呼ばれず、ContinuousListener はバッファ保持のまま次へ
            default = get_default_character()
            logger.info(
                "ルーティング: LLM ルーター → 呼びかけなし → デフォルト(%s) (text=%s)",
                default.slug, text_p,
            )
            return RoutingDecision(speaker=default.slug, reason="default")
        # llm_result is None (API 失敗 / パース失敗) → 従来パターンマッチでフォールバック
        logger.warning(
            "LLM ルーター失敗。パターンマッチへフォールバック。 (text=%s)", text_p,
        )

    # 複数名でも LLM 無効/失敗時、または 0 マッチ時 → 従来ロジック
    if all_matches:
        matched = _match_by_text(text, characters)
        if matched:
            logger.info(
                "ルーティング: テキストマッチ → %s (text=%s)",
                matched.slug, text_p,
            )
            return RoutingDecision(speaker=matched.slug, reason="text_match")

    # 4. デフォルト
    default = get_default_character()
    logger.info("ルーティング: デフォルト → %s (text=%s)", default.slug, text_p)
    return RoutingDecision(speaker=default.slug, reason="default")


# ─── LLM 意図ゲート (Phase 2 + Phase 0.5-A) ─────────────────────
#
# Phase 2 (Axis B): check_intent で「呼びかけ vs 言及」を判定 (character_slug 指定時)
# Phase 0.5-A: check_intent を IntentResult 返却に拡張、character_slug=None で
#              「自発介入候補キャラ」を判定する interjection_candidate モードを追加。
#              check_approval を新規追加し、handraising 中のルカ承認/却下を判定。
# ────────────────────────────────────────────────────────────

_INTENT_GATE_MODEL = os.environ.get("L2_INTENT_GATE_MODEL", "claude-haiku-4-5-20251001")


@dataclass(frozen=True)
class IntentResult:
    """check_intent() の戻り値 (Phase 0.5-A で導入)。

    intent:
      - "callout"               — 特定キャラへの直接の呼びかけ (Phase 2)
      - "mention"               — キャラについての言及のみ (Phase 2)
      - "unknown"               — LLM 失敗 / パース失敗 (fail-open)
      - "interjection_candidate"— 名前ヒントなしで自発介入候補のキャラを検出
                                  (Phase 0.5-A 挙手システム)
    target_slug:
      - callout/mention 時は呼出元の character_slug
      - interjection_candidate 時は LLM が判定した候補 slug
      - unknown 時は呼出元 slug or None
    confidence: 0.0-1.0、現状は 1.0 / 0.0 の二値 (将来 LLM 出力に拡張)
    """

    intent: Literal["callout", "mention", "unknown", "interjection_candidate"]
    target_slug: str | None
    confidence: float


@dataclass(frozen=True)
class ApprovalResult:
    """check_approval() の戻り値 (Phase 0.5-A handraising 中のみ)。

    granted=True  → ルカが承認 (「ミミ、どうぞ」「いいよ」)
    granted=False → ルカが却下 (「いや、いいわ」「やめて」)
    どちらでもない発話の場合は check_approval が ``None`` を返す (本 dataclass 未使用)。
    """

    granted: bool
    target_slug: str
    confidence: float


def is_intent_gate_enabled() -> bool:
    """LLM 意図ゲートが有効かどうか。"""
    return os.environ.get("L2_USE_INTENT_GATE", "").lower() in ("1", "true", "yes")


def check_intent(
    text: str,
    character_slug: str | None = None,
) -> IntentResult:
    """
    発話文脈の意図を LLM で判定する (Phase 2 互換 + Phase 0.5-A 拡張)。

    モード:
      - character_slug 指定時 (Phase 2 互換):
          特定キャラへの呼びかけ (callout) / 言及 (mention) / 不明 (unknown) を判定。
          ContinuousListener / BackgroundContinuousListener が、ウェイクワード
          検知後の文脈チェックで使う。
      - character_slug=None 時 (Phase 0.5-A):
          挙手候補キャラを判定。各キャラの関心領域に触れる発話があれば
          interjection_candidate を返す。Dispatcher.on_segment_added が、
          handraising キャラ無し + IDLE/RESPONDING 中に呼ぶ。

    L2_INTENT_GATE_MODEL でモデル指定可能。プロバイダ自動判定。

    Args:
        text:           判定対象の発話テキスト (バッファ抽出 or 1 segment)
        character_slug: Phase 2 互換モードでは判定対象キャラの slug、
                        Phase 0.5-A interjection_candidate モードでは None

    Returns:
        IntentResult (frozen dataclass)。LLM 失敗時は intent="unknown" で fail-open。
    """
    if character_slug is not None:
        return _check_intent_for_character(text, character_slug)
    return _check_intent_interjection_candidate(text)


def _check_intent_for_character(
    text: str,
    character_slug: str,
) -> IntentResult:
    """Phase 2 互換: 特定キャラへの callout/mention/unknown を判定する。

    ロジックは Phase 2 (Axis B) と同一だが、戻り値を str → IntentResult に変更。
    ContinuousListener / BackgroundContinuousListener から呼ばれる。
    """
    system_prompt = (
        "あなたは発話意図判定器です。\n"
        f"ユーザーの発話に「特定のキャラクター（{character_slug}）への直接の呼びかけ」"
        "が含まれているかを判定してください。\n\n"
        "判定基準:\n"
        "- 呼びかけ (callout): ユーザーがキャラクターに直接話しかけている、"
        "質問・依頼している\n"
        "- 言及 (mention): キャラクターについて話しているだけ、"
        "第三者に説明している、引用している\n\n"
        "回答は以下のいずれか一語のみ:\n"
        "- callout\n"
        "- mention"
    )

    model = os.environ.get("L2_INTENT_GATE_MODEL", _INTENT_GATE_MODEL)
    answer = _call_router_llm(model, system_prompt, text)

    if answer is None:
        logger.warning("意図ゲート: LLM 呼び出し失敗 → unknown (fail-open)")
        return IntentResult(intent="unknown", target_slug=character_slug, confidence=0.0)

    # ログ強化 L-3: text preview 追加で「何の発話への判定か」識別容易に
    logger.info(
        "意図ゲート応答: %r (model=%s char=%s text=%s)",
        answer, model, character_slug, _text_preview(text),
    )

    if answer == "callout":
        return IntentResult(intent="callout", target_slug=character_slug, confidence=1.0)
    if answer == "mention":
        return IntentResult(intent="mention", target_slug=character_slug, confidence=1.0)

    logger.warning("意図ゲート: 予期しない応答 %r → unknown (fail-open)", answer)
    return IntentResult(intent="unknown", target_slug=character_slug, confidence=0.0)


def _load_character_interest_area(slug: str) -> str | None:
    """``skills/characters/<slug>.md`` から ``**担当エリア**:`` 行を抽出する (Phase 0.5-A フェーズ 8)。

    interjection_candidate プロンプトに各キャラの担当領域を含めるためのヘルパー。
    md ファイルが Single Source Of Truth (SSOT) で、router.py 側でハードコードする
    DRY 違反を避ける。md のヘッダ近辺だけを走査するため I/O コストは最小。

    Args:
        slug: キャラクター slug (例: "mimi")

    Returns:
        担当エリアの記述 (例: "抽象 × 感情（美学・価値観・ノブレスオブリージュ）")。
        md ファイルが見つからない / `**担当エリア**:` 行が無い場合は None。
    """
    skills_dir = Path(__file__).parent / "skills" / "characters"
    md_file = skills_dir / f"{slug}.md"
    if not md_file.is_file():
        return None
    try:
        content = md_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("interest_area 読込失敗: slug=%s err=%s", slug, exc)
        return None
    # 先頭 10 行のみ走査 (担当エリアは通常 line 3 付近の見出し直下)
    for line in content.splitlines()[:10]:
        stripped = line.strip()
        if stripped.startswith("**担当エリア**:"):
            return stripped.replace("**担当エリア**:", "").strip()
    return None


_HANDRAISE_EXCLUDED_SLUGS: frozenset[str] = frozenset({
    # Notion §C1 確定: 補助員ボット、応答待機専用キャラなので挙手しない
    "octamaid",
    # Phase 0.5-A フェーズ 8 実走で除外確定 (2026-05-07): 配信者本人 (ルカ) の
    # キャラクター。AI が「ルカとして自発介入する」のは設計上不自然 (配信者の意思を
    # AI が代弁する形になり、自律エージェント感の趣旨に反する)。
    "ruka",
})


def _check_intent_interjection_candidate(text: str) -> IntentResult:
    """Phase 0.5-A: 自発介入候補のキャラを判定する。

    候補は ``_HANDRAISE_EXCLUDED_SLUGS`` を除く挙手対応キャラ (mimi / chisame / sakura)。
    LLM に「自分の関心領域に触れる発話があるキャラ」を判定させ、該当があれば
    interjection_candidate + target_slug を返す (該当なしは unknown)。

    除外スラグ:
      - octamaid: 補助員ボット (Notion §C1 確定)
      - ruka:     配信者本人 (Phase 0.5-A フェーズ 8 実走で確定)

    Phase 0.5-A フェーズ 8: 各キャラの担当エリア (skills/characters/<slug>.md の
    `**担当エリア**:` 行) をプロンプトに含めて、LLM の判定精度を改善した。
    実走で「AI 倫理について…」発話に対して候補なし (none) と保守的に判定された
    問題への対応。
    """
    characters = get_all_characters()
    # 挙手対象から除外: octamaid (補助員) + ruka (配信者本人)。詳細は
    # ``_HANDRAISE_EXCLUDED_SLUGS`` の docstring を参照。
    candidate_chars = [
        c for c in characters if c.slug not in _HANDRAISE_EXCLUDED_SLUGS
    ]
    candidate_slugs = [c.slug for c in candidate_chars]

    if not candidate_slugs:
        logger.warning("interjection_candidate: 候補キャラなし → unknown")
        return IntentResult(intent="unknown", target_slug=None, confidence=0.0)

    # 各候補キャラに担当エリアを添える (skills/characters/<slug>.md からロード)。
    # 担当エリアが取得できないキャラは display_name のみで提示 (グレースフルデグレード)。
    candidates_lines: list[str] = []
    for c in candidate_chars:
        interest = _load_character_interest_area(c.slug)
        if interest:
            candidates_lines.append(f"- {c.slug} ({c.display_name}) — 担当: {interest}")
        else:
            candidates_lines.append(f"- {c.slug} ({c.display_name})")
    candidates_desc = "\n".join(candidates_lines)

    system_prompt = (
        "あなたは自発介入候補判定器です。\n"
        "ユーザー (ルカ) と他キャラの会話の文脈で、特定キャラへの呼びかけは無いが、\n"
        "あるキャラが「自分の関心領域・専門分野」として自発介入したそうな話題が\n"
        "発話に含まれているかを判定してください。\n\n"
        f"候補キャラとそれぞれの担当エリア:\n{candidates_desc}\n\n"
        "判定基準:\n"
        "- 発話の話題が候補キャラの担当エリアに**触れている / 関連している**かを判定する。\n"
        "  問題提起の段階 (まだ具体性がなくても) でも、話題が明示されていれば候補有。\n"
        "  例:「最近のAI倫理について気になっている」 → 倫理担当の sakura、価値観担当の\n"
        "  mimi が候補。「データの裏付けを取りたい」 → 論理担当の chisame が候補。\n"
        "- 既にキャラ名で呼びかけ済みなら none (= callout 経路に流れる)。\n"
        "- 担当エリアと**まったく無関係**な雑談 / 挨拶のみ none。\n"
        "- 複数候補が該当する場合は最も中心的な 1 名を選ぶ。\n\n"
        "**出力形式 (厳守)**: 候補 slug 1 語のみ。理由 / 説明 / 改行 / 装飾文字\n"
        "(`**`、`:`、引用符、空行) は一切付けない。\n"
        f"許容値: {', '.join(candidate_slugs)}, none"
    )

    model = os.environ.get("L2_INTENT_GATE_MODEL", _INTENT_GATE_MODEL)
    answer = _call_router_llm(model, system_prompt, text)

    if answer is None:
        logger.warning("interjection_candidate: LLM 呼び出し失敗 → unknown (fail-open)")
        return IntentResult(intent="unknown", target_slug=None, confidence=0.0)

    # ログ強化 L-3: text preview + 候補 slugs を追加して「何の発話に対する候補判定か」
    # 「候補から漏れた slug が無いか」を識別容易に
    logger.info(
        "interjection_candidate 応答: %r (model=%s candidates=%s text=%s)",
        answer, model, candidate_slugs, _text_preview(text),
    )

    # Phase 0.5-A フェーズ 8 実走対応: LLM が指示を完全には守らず理由付きで返すケース
    # (例: "none\n\n**理由**: ...") に備えて、最初のトークンで候補を判定する。
    # 空白 / 改行で split → 句読点 / 装飾文字を rstrip → 候補と照合。
    stripped = answer.strip()
    first_token = ""
    if stripped:
        first_token = stripped.split()[0].rstrip(
            "。.,、!?:;-`'\"*）)】」"
        ).lstrip("`'\"*（(【「")
    logger.debug("interjection_candidate: first_token=%r", first_token)

    if first_token == "none":
        return IntentResult(intent="unknown", target_slug=None, confidence=0.0)

    if first_token in candidate_slugs:
        return IntentResult(
            intent="interjection_candidate",
            target_slug=first_token,
            confidence=1.0,
        )

    logger.warning(
        "interjection_candidate: 候補外 first_token=%r (raw=%r) → unknown",
        first_token, answer,
    )
    return IntentResult(intent="unknown", target_slug=None, confidence=0.0)


def check_approval(
    text: str,
    candidate_slugs: list[str],
) -> ApprovalResult | None:
    """Phase 0.5-A: handraising 中の承認/却下を判定する。

    ルカの発話を:
      - 承認 (granted): 「(キャラ名)、どうぞ」「いいよ」「話して」など発話を促す
      - 却下 (denied):  「いや、いいわ」「やめて」「後で」など発話を断る
      - 関係なし:       上記いずれでもない別の話題 → ``None`` を返す
                       (呼出元 Dispatcher は通常の意図ゲート処理に流す)

    候補が複数 (複数挙手連鎖) の場合も対応。LLM はキャラ名を識別して
    「どのキャラへの承認/却下か」を返す。

    L2_INTENT_GATE_MODEL でモデル指定可能。

    Args:
        text:            ルカの発話テキスト
        candidate_slugs: 現在 handraising 中のキャラ slug のリスト (1 件以上)

    Returns:
        ApprovalResult | None: 承認/却下のいずれかなら ApprovalResult、
                              どちらでもない場合 / LLM 失敗時は ``None`` で fail-open
                              (呼出元は通常の意図ゲート処理に流す)。
    """
    if not candidate_slugs:
        return None

    # キャラ名 (display_name + nickname + aliases) も提示することで、LLM が
    # 「ミミ、どうぞ」のような自然な日本語を slug に変換しやすくなる。
    characters = get_all_characters()
    slug_to_names: dict[str, list[str]] = {}
    for c in characters:
        if c.slug not in candidate_slugs:
            continue
        names: list[str] = [c.display_name]
        if c.nickname:
            names.append(c.nickname)
        names.extend(c.aliases)
        slug_to_names[c.slug] = [n for n in names if n]

    candidates_with_names = "\n".join(
        f"- {slug}: {', '.join(slug_to_names.get(slug, [slug]))}"
        for slug in candidate_slugs
    )

    system_prompt = (
        "あなたは挙手承認判定器です。\n"
        "AI キャラ (挙手中) に対して、ユーザー (ルカ) の発話を:\n"
        "- 承認 (granted): 「(キャラ名)、どうぞ」「いいよ」「話して」など発話を促す\n"
        "- 却下 (denied):  「いや、いいわ」「やめて」「後で」など発話を断る\n"
        "- 関係なし (none): 上記いずれでもない別の話題\n"
        "を判定してください。\n\n"
        f"挙手中の候補:\n{candidates_with_names}\n\n"
        "回答は次のいずれか一行のみ:\n"
        "- granted:<slug>\n"
        "- denied:<slug>\n"
        "- none\n"
        f"(slug は {', '.join(candidate_slugs)} のいずれか)"
    )

    model = os.environ.get("L2_INTENT_GATE_MODEL", _INTENT_GATE_MODEL)
    answer = _call_router_llm(model, system_prompt, text)

    if answer is None:
        logger.warning("check_approval: LLM 呼び出し失敗 → None (fail-open: 通常処理へ)")
        return None

    # ログ強化 L-3: text preview を追加して「ルカの何の発話に対する承認/却下判定か」
    # 識別容易に
    logger.info(
        "check_approval 応答: %r (model=%s candidates=%s text=%s)",
        answer, model, candidate_slugs, _text_preview(text),
    )

    if answer == "none":
        return None

    if ":" in answer:
        verdict, _, slug = answer.partition(":")
        slug = slug.strip()
        verdict = verdict.strip()
        if slug not in candidate_slugs:
            logger.warning("check_approval: 候補外 slug %r → None", slug)
            return None
        if verdict == "granted":
            return ApprovalResult(granted=True, target_slug=slug, confidence=1.0)
        if verdict == "denied":
            return ApprovalResult(granted=False, target_slug=slug, confidence=1.0)

    logger.warning("check_approval: 予期しない応答 %r → None", answer)
    return None
