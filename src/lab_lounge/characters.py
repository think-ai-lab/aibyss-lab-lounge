"""
characters.py — AITuber キャラクターレジストリ

責務:
  - AITuber キャラクター定義を一元管理する
  - キャラクター slug による検索、デフォルトキャラクターの解決
  - システムプロンプトファイルの読み込み

【キャラクター一覧】
  mimi     — ミミ・オクタヴィア (AI 解説担当)
  chisame  — 波心ちさめ (情報・知識担当)
  sakura   — 八重笠さくら (感情・道徳担当)
  octamaid — オクタメイド (補助員ボット)
  ruka     — 坂東ルカ (メインパーソナリティ / ホスト)

【設計判断】
  環境変数は L2_DEFAULT_SPEAKER (デフォルトキャラクター slug) のみ。
  キャラクター固有の設定はこのファイルで定義し、環境変数爆発を防ぐ。
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# システムプロンプトの配置ディレクトリ
_PROMPTS_DIR = Path(__file__).resolve().parent / "system_prompts"


@dataclass(frozen=True)
class CharacterConfig:
    """AITuber キャラクター設定。"""

    slug: str
    display_name: str
    wake_word: str | None
    tts_provider: str
    tts_voice: str
    system_prompt_file: str
    llm_provider: str = "openai"
    llm_model: str = ""          # 空文字 = 環境変数 L2_LLM_MODEL のデフォルトを使用
    filler_model: str = ""       # 空文字 = 環境変数 L2_LLM_FILLER_MODEL のデフォルトを使用
    porcupine_model: str | None = None
    aliases: list[str] = field(default_factory=list)
    # 呼びかけ用の親しみのある通り名 (例: 「ちさめ」「さくら」)。空文字なら
    # display_name を使う。ask_character の導入セリフ生成時に target に直接
    # 語りかける際、フルネーム ("波心ちさめ") よりも自然になるため使用する。
    nickname: str = ""
    # VOICEPEAK ナレーター固有の emotion キー名。
    # フィラー LLM プロンプトに含めて JSON 出力を誘導する。
    # 空タプル = emotion 非対応 (voicevox 等) → フィラーはプレーンテキスト生成。
    voicepeak_emotion_keys: tuple[str, ...] = ()


# ─── キャラクター定義 ────────────────────────────────────────────────

CHARACTER_REGISTRY: dict[str, CharacterConfig] = {
    "mimi": CharacterConfig(
        slug="mimi",
        display_name="ミミ・オクタヴィア",
        nickname="ミミ",
        wake_word="ミミ様",
        # TTS: irodori VoiceDesign（確定版）。サイドカー起動が前提
        #   （aibyss.ps1 start -Irodori）。VOICEPEAK に戻すには
        #   tts_provider="voicepeak" / tts_voice="Asumi Ririse"。
        #   tts_voice は reference_voices/voices.json の voices.<key>（例 "mimi"）。
        tts_provider="irodori_vd",
        tts_voice="mimi",
        system_prompt_file="system_mimi.txt",
        llm_provider="openai",
        llm_model="gpt-5.5",
        filler_model="gpt-5.4-mini",
        porcupine_model="mimi-sama_ja_windows_v4_0_0.ppn",
        aliases=["ミミ", "お嬢様"],
        # irodori pose-only 化に伴い emotion 軸オフ（pose + speed で表現）。VOICEPEAK に
        # 戻す場合は ("happy","fun","angry","sad","sulky") を復元 + プロンプトの emotion 復元。
        voicepeak_emotion_keys=(),
    ),
    "chisame": CharacterConfig(
        slug="chisame",
        display_name="波心ちさめ",
        nickname="ちさめ",
        wake_word="ちさめさん",
        # TTS: irodori VoiceDesign（確定版）。VOICEPEAK に戻すには
        #   tts_provider="voicepeak" / tts_voice="Miyamai Moca"。
        tts_provider="irodori_vd",
        tts_voice="chisame",
        system_prompt_file="system_chisame.txt",
        llm_provider="google",
        llm_model="gemini-3.1-pro-preview",
        filler_model="gemini-3.1-flash-lite",
        porcupine_model="chisame-san_ja_windows_v3_0_0.ppn",
        aliases=["ちさめ"],
        # irodori pose-only 化に伴い emotion 軸オフ。VOICEPEAK に戻す場合は
        # ("bosoboso","doyaru","honwaka","angry","teary") を復元 + プロンプトの emotion 復元。
        voicepeak_emotion_keys=(),
    ),
    "sakura": CharacterConfig(
        slug="sakura",
        display_name="八重笠さくら",
        nickname="さくら",
        wake_word="さくらさん",
        # TTS: irodori VoiceDesign（確定版）。VOICEPEAK に戻すには
        #   tts_provider="voicepeak" / tts_voice="Haruno Sora"。
        tts_provider="irodori_vd",
        tts_voice="sakura",
        system_prompt_file="system_sakura.txt",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        filler_model="claude-haiku-4-5",
        porcupine_model="sakura-san_ja_windows_v3_0_0.ppn",
        aliases=["さくら", "桜さん", "桜"],
        # irodori pose-only 化に伴い emotion 軸オフ。VOICEPEAK に戻す場合は
        # ("happy","sad","angry","whisper","cool") を復元 + プロンプトの emotion 復元。
        voicepeak_emotion_keys=(),
    ),
    "octamaid": CharacterConfig(
        slug="octamaid",
        display_name="オクタメイド",
        nickname="オクタメイド",
        wake_word="オクタメイド",
        tts_provider="voicevox",
        tts_voice="89", # Voidoll
        system_prompt_file="system_octamaid.txt",
        porcupine_model="octamaid_ja_windows_v4_0_0.ppn",
        aliases=["オクタメイド(仮)"],
    ),
    "ruka": CharacterConfig(
        slug="ruka",
        display_name="坂東ルカ",
        nickname="ルカ",
        wake_word=None,
        tts_provider="voicepeak",
        tts_voice="Frimomen",  # 暫定
        system_prompt_file="system_ruka.txt",
        porcupine_model=None,
        aliases=["ルカ"],
    ),
    "aruka": CharacterConfig(
        slug="aruka",
        display_name="アルカ",
        nickname="アルカ",
        # 暫定 wake word。.ppn 未整備のため Porcupine 検知は porcupine_model=None で当面オフ
        # （ルーティングは name/aliases のテキスト一致で可能）。
        wake_word="アルカさん",
        # TTS: irodori VoiceDesign（確定版、AI ホスト）。VOICEVOX に戻すなら
        #   tts_provider="voicevox" / tts_voice=<冥鳴ひまりの style id>。
        tts_provider="irodori_vd",
        tts_voice="aruka",
        system_prompt_file="system_aruka.txt",
        porcupine_model=None,
        aliases=["アルカ", "Aルカ", "アルカさん"],
        # アルカは emotion パラメータを持たない（LLM 出力は {speed, pose, response}）。
        # 感情は pose + speed + 文体で表現。空タプル = フィラーは emotion JSON を生成しない。
        voicepeak_emotion_keys=(),
    ),
}


# ─── 公開 API ────────────────────────────────────────────────────────


def get_character(slug: str) -> CharacterConfig:
    """
    slug でキャラクターを取得する。

    Raises:
        KeyError: 未知の slug
    """
    config = CHARACTER_REGISTRY.get(slug)
    if config is None:
        available = ", ".join(f'"{s}"' for s in CHARACTER_REGISTRY)
        raise KeyError(f"未知のキャラクター slug: {slug!r}。利用可能: {available}")
    return config


def get_default_character() -> CharacterConfig:
    """
    デフォルトキャラクターを返す。

    環境変数 L2_DEFAULT_SPEAKER で slug を指定可能 (default: "octamaid")。
    """
    slug = os.environ.get("L2_DEFAULT_SPEAKER", "octamaid")
    try:
        return get_character(slug)
    except KeyError:
        logger.warning(
            "L2_DEFAULT_SPEAKER=%r は未知の slug です。octamaid にフォールバックします。",
            slug,
        )
        return CHARACTER_REGISTRY["octamaid"]


def get_all_characters() -> list[CharacterConfig]:
    """全キャラクターをリストで返す。"""
    return list(CHARACTER_REGISTRY.values())


def load_system_prompt(config: CharacterConfig) -> str:
    """
    キャラクターのシステムプロンプトをファイルから読み込む。

    Args:
        config: 対象キャラクターの設定

    Returns:
        プロンプトテキスト

    Raises:
        FileNotFoundError: プロンプトファイルが見つからない
    """
    path = _PROMPTS_DIR / config.system_prompt_file
    if not path.is_file():
        raise FileNotFoundError(
            f"システムプロンプトが見つかりません: {path}"
        )
    return path.read_text(encoding="utf-8")
