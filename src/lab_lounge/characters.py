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
    porcupine_model: str | None = None
    aliases: list[str] = field(default_factory=list)


# ─── キャラクター定義 ────────────────────────────────────────────────

CHARACTER_REGISTRY: dict[str, CharacterConfig] = {
    "mimi": CharacterConfig(
        slug="mimi",
        display_name="ミミ・オクタヴィア",
        wake_word="ミミ様",
        tts_provider="voicepeak",
        tts_voice="Asumi Ririse",
        system_prompt_file="system_mimi.txt",
        porcupine_model="mimi-sama_ja_windows_v4_0_0.ppn",
        aliases=["ミミ", "お嬢様"],
    ),
    "chisame": CharacterConfig(
        slug="chisame",
        display_name="波心ちさめ",
        wake_word="ちさめさん",
        tts_provider="voicepeak",
        tts_voice="Miyamai Moca",
        system_prompt_file="system_chisame.txt",
        porcupine_model="chisame-san_ja_windows_v3_0_0.ppn",
        aliases=["ちさめ"],
    ),
    "sakura": CharacterConfig(
        slug="sakura",
        display_name="八重笠さくら",
        wake_word="さくらさん",
        tts_provider="voicepeak",
        tts_voice="Haruno Sora",
        system_prompt_file="system_sakura.txt",
        porcupine_model="sakura-san_ja_windows_v3_0_0.ppn",
        aliases=["さくら", "桜さん", "桜"],
    ),
    "octamaid": CharacterConfig(
        slug="octamaid",
        display_name="オクタメイド",
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
        wake_word=None,
        tts_provider="voicepeak",
        tts_voice="Frimomen",  # 暫定
        system_prompt_file="system_ruka.txt",
        porcupine_model=None,
        aliases=["ルカ"],
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
