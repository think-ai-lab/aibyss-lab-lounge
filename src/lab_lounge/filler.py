"""
filler.py — フィラー音声管理

責務:
  - キャラクター別フィラーフレーズの読み込み (data/filler_phrases/{slug}.txt)
  - フィラー WAV の事前生成・キャッシュ管理 (data/filler_cache/{slug}/)
  - ルールベースのフレーズ選択（ランダム、直前と被らない）
  - フィラー再生ループ（opener → continue 連結再生）

【フレーズファイル形式】
  data/filler_phrases/mimi.txt に [section] ヘッダーで分類して記述。

    [opener]
    ふふ、良い質問ですわねぇ	happy=50

    [continue]
    えーと、そうですわねぇ……	happy=50
    わたくしの考えでは……	happy=50

    [closer]
    もう少しだけお待ちくださいね	happy=50

  - [opener]: 最初の 1 フレーズ（呼びかけへの反応）
  - [continue]: 2 フレーズ目以降（考え中の独り言、つなぎ）— ループ再生
  - [closer]: 予約（現在未使用）
  - ヘッダーなしの行は全て opener 扱い（後方互換）
  - 空行・先頭 # はスキップ
  - タブ区切りで emotion を付加可能（VOICEPEAK 用）

【キャッシュ】
  data/filler_cache/{slug}/{category}_{index:02d}.wav に TTS 生成済み WAV を保存。
  例: opener_00.wav, continue_00.wav, continue_01.wav
  フレーズが変更されたら --force で再生成が必要。

【環境変数】
  L2_FILLER_ENABLED — フィラー音声の有効化 (デフォルト: true)
"""

import json
import logging
import os
import random
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path

# Phase 0.5-J: google.genai を module top で eager import することで、複数スレッド間の
# 並列 import (= filler スレッド + 本命 LLM スレッド経由の langchain-google-genai が
# 同 ms 内で google.genai.types を import) による circular import race condition を
# 構造的に回避する。
#
# 旧設計 (= 関数内 lazy import `import google.genai as genai`) では、セッション内で
# Gemini を初めて使う瞬間に「partially initialized module 'google.genai.types'」
# エラーが間欠的に発生 (= 2026-04-11 起票の Notion 課題、2026-05-15 実走で実害確認:
# logs/runs/run_loop_20260515_121353.log、chisame の callout ターンが丸ごと skip)。
#
# eager import により program startup 時 (= 単一スレッド) に 1 回だけ import される
# ため、その後の並列呼出時には既にキャッシュ済モジュールが返り、race window が消失。
try:
    import google.genai as _GOOGLE_GENAI  # noqa: F401  (function 内で参照)
    from google.genai import types as _GOOGLE_GENAI_TYPES  # noqa: F401  (function 内で参照)
except ImportError:
    _GOOGLE_GENAI = None  # type: ignore[assignment]
    _GOOGLE_GENAI_TYPES = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_FILLER_PHRASES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filler_phrases"
_FILLER_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filler_cache"

_VALID_SECTIONS = {"opener", "continue", "bridge", "closer", "handraise"}
# "handraise" は Phase 0.5-A で追加。挙手 (interjection_candidate) 検知時に
# L2 側で再生される短いキャラ声フレーズ。bubble.update(handraise) の text にも
# 同じフレーズ実体が使われる (wav 再生とテキスト表示が同期)。


def is_filler_enabled() -> bool:
    """フィラー音声が有効かどうかを返す。"""
    return os.environ.get("L2_FILLER_ENABLED", "true").lower() in ("true", "1", "yes")


@dataclass
class FillerPhrase:
    """フィラーフレーズとオプションの emotion 値。"""
    text: str
    emotion: dict[str, int] | None = None


@dataclass
class FillerPhraseSet:
    """カテゴリ別に分類されたフィラーフレーズセット。"""
    opener: list[FillerPhrase] = field(default_factory=list)
    continue_: list[FillerPhrase] = field(default_factory=list)
    bridge: list[FillerPhrase] = field(default_factory=list)
    closer: list[FillerPhrase] = field(default_factory=list)
    handraise: list[FillerPhrase] = field(default_factory=list)
    # ↑ Phase 0.5-A 追加。挙手機能用の短いフレーズ (wav 兼 SE)。

    @property
    def all_phrases(self) -> list[tuple[str, FillerPhrase]]:
        """(category, phrase) ペアのフラットリスト（キャッシュ生成用）。"""
        items: list[tuple[str, FillerPhrase]] = []
        for cat, lst in [("opener", self.opener),
                         ("continue", self.continue_),
                         ("bridge", self.bridge),
                         ("closer", self.closer),
                         ("handraise", self.handraise)]:
            for p in lst:
                items.append((cat, p))
        return items

    def __len__(self) -> int:
        return (
            len(self.opener) + len(self.continue_) + len(self.bridge)
            + len(self.closer) + len(self.handraise)
        )


def _parse_emotion(emotion_str: str) -> dict[str, int]:
    """
    "happy=80,sad=0" 形式の emotion 文字列をパースする。
    """
    result: dict[str, int] = {}
    for pair in emotion_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        try:
            result[key.strip()] = int(val.strip())
        except ValueError:
            continue
    return result


def load_filler_phrases(slug: str) -> FillerPhraseSet:
    """
    data/filler_phrases/{slug}.txt からフレーズを読み込む。

    [opener], [continue], [closer] セクションヘッダーで分類。
    ヘッダーなしの行は全て opener 扱い（後方互換）。
    ファイルが存在しない場合は空の FillerPhraseSet を返す。
    """
    phrases_file = _FILLER_PHRASES_DIR / f"{slug}.txt"
    if not phrases_file.is_file():
        logger.debug("フィラーフレーズファイルなし: %s", phrases_file)
        return FillerPhraseSet()

    lines = phrases_file.read_text(encoding="utf-8").splitlines()
    result = FillerPhraseSet()
    current_section = "opener"  # デフォルト（後方互換）

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # セクションヘッダー検出
        if stripped.startswith("[") and stripped.endswith("]"):
            section_name = stripped[1:-1].strip().lower()
            if section_name in _VALID_SECTIONS:
                current_section = section_name
            else:
                logger.warning("不明なセクション: %s (無視)", stripped)
            continue

        # フレーズ行をパース
        parts = stripped.split("\t", 1)
        text = parts[0].strip()
        emotion = None
        if len(parts) > 1 and parts[1].strip():
            emotion = _parse_emotion(parts[1].strip())

        phrase = FillerPhrase(text=text, emotion=emotion)

        if current_section == "opener":
            result.opener.append(phrase)
        elif current_section == "continue":
            result.continue_.append(phrase)
        elif current_section == "bridge":
            result.bridge.append(phrase)
        elif current_section == "closer":
            result.closer.append(phrase)
        elif current_section == "handraise":
            result.handraise.append(phrase)

    total = len(result)
    logger.debug(
        "フィラーフレーズ読み込み: %s "
        "(opener=%d continue=%d bridge=%d closer=%d handraise=%d total=%d)",
        slug,
        len(result.opener), len(result.continue_), len(result.bridge),
        len(result.closer), len(result.handraise), total,
    )
    if not result.continue_:
        logger.warning(
            "[%s] [continue] セクションなし — フィラーループは opener のみで終了します",
            slug,
        )
    return result


def get_filler_duration_ms(path: Path) -> int:
    """WAV ファイルの再生時間をミリ秒で返す。"""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0
            return int(frames / rate * 1000)
    except Exception:
        return 0


def get_cached_filler_paths(slug: str) -> dict[str, list[Path]]:
    """
    キャッシュ済みフィラー WAV パスをカテゴリ別に返す。

    Returns:
        {"opener": [...], "continue": [...], "bridge": [...], "closer": [...],
         "handraise": [...]}

    NOTE: カテゴリリストは ``_VALID_SECTIONS`` と同期させる。フェーズ 2 で
    handraise セクションが追加されたが、本関数の cat tuple への反映が漏れて
    いたため、Phase 0.5-A 実走で挙手 wav が認識されない不具合となっていた。
    """
    cache_dir = _FILLER_CACHE_DIR / slug
    result: dict[str, list[Path]] = {}
    # Phase 0.5-A フェーズ 2 で追加された "handraise" を含む全カテゴリを走査
    for cat in ("opener", "continue", "bridge", "closer", "handraise"):
        if cache_dir.is_dir():
            result[cat] = sorted(cache_dir.glob(f"{cat}_*.wav"))
        else:
            result[cat] = []
    return result


def ensure_filler_cache(
    slug: str,
    *,
    force: bool = False,
) -> dict[str, list[Path]]:
    """
    キャラクターのフィラー WAV を事前生成しキャッシュする。

    Returns:
        カテゴリ別の WAV パス辞書。
    """
    from .characters import get_character
    from .tts import synthesize

    phrase_set = load_filler_phrases(slug)
    # Phase 0.5-A フェーズ 2 で追加された "handraise" を含む全カテゴリを初期化。
    # 5 カテゴリのうち 1 つでも欠けると、phrase_set.all_phrases の反復中に
    # KeyError (counters[cat]) で abort する不具合があった (実走で発見)。
    _empty: dict[str, list[Path]] = {
        "opener": [], "continue": [], "bridge": [], "closer": [], "handraise": [],
    }
    if len(phrase_set) == 0:
        logger.warning("フィラーフレーズが定義されていません: %s", slug)
        return dict(_empty)

    char = get_character(slug)
    if char is None:
        logger.warning("キャラクターが見つかりません: %s", slug)
        return dict(_empty)

    cache_dir = _FILLER_CACHE_DIR / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --force 時は旧キャッシュを全削除
    if force:
        for old_file in cache_dir.glob("*.wav"):
            old_file.unlink()

    result_paths: dict[str, list[Path]] = {
        "opener": [], "continue": [], "bridge": [], "closer": [], "handraise": [],
    }
    counters: dict[str, int] = {
        "opener": 0, "continue": 0, "bridge": 0, "closer": 0, "handraise": 0,
    }

    for cat, entry in phrase_set.all_phrases:
        idx = counters[cat]
        counters[cat] += 1
        target = cache_dir / f"{cat}_{idx:02d}.wav"

        if target.is_file() and not force:
            logger.debug("キャッシュ済みスキップ: %s", target)
            result_paths[cat].append(target)
            continue

        # emotion 付きの場合は JSON 形式でテキストを渡す
        if entry.emotion and char.tts_provider == "voicepeak":
            tts_text = json.dumps(
                {"response": entry.text, "emotion": entry.emotion},
                ensure_ascii=False,
            )
        else:
            tts_text = entry.text

        logger.info(
            "フィラー生成中: [%s/%s] %r (emotion=%s) → %s",
            slug, cat, entry.text, entry.emotion, target.name,
        )
        try:
            tts_result = synthesize(
                tts_text,
                provider=char.tts_provider,
                voice=char.tts_voice,
                output_dir=str(cache_dir),
            )
            src = Path(tts_result.audio_url.replace("file:///", "").replace("file://", ""))
            if src.is_file() and src != target:
                target.unlink(missing_ok=True)
                src.rename(target)
            elif not target.is_file():
                logger.warning("TTS 出力ファイルが見つかりません: %s", src)
                continue

            duration = get_filler_duration_ms(target)
            logger.info(
                "フィラー生成完了: [%s/%s] %s (%d ms)",
                slug, cat, target.name, duration,
            )
            result_paths[cat].append(target)

        except Exception as exc:
            logger.error("フィラー生成失敗: [%s/%s] %r — %s", slug, cat, entry.text, exc)
            if entry.emotion:
                logger.info("emotion なしでリトライ: [%s] %r", slug, entry.text)
                try:
                    tts_result = synthesize(
                        entry.text,
                        provider=char.tts_provider,
                        voice=char.tts_voice,
                        output_dir=str(cache_dir),
                    )
                    src = Path(tts_result.audio_url.replace("file:///", "").replace("file://", ""))
                    if src.is_file() and src != target:
                        target.unlink(missing_ok=True)
                        src.rename(target)
                    if target.is_file():
                        duration = get_filler_duration_ms(target)
                        logger.info("リトライ成功: [%s] %s (%d ms)", slug, target.name, duration)
                        result_paths[cat].append(target)
                        continue
                except Exception as retry_exc:
                    logger.error("リトライも失敗: [%s] %r — %s", slug, entry.text, retry_exc)

    return result_paths


def select_filler_path(
    slug: str,
    category: str = "opener",
    *,
    last_index: int = -1,
) -> tuple[Path | None, int]:
    """
    指定カテゴリのフィラー WAV をランダム選択する（直前と被らない制約付き）。
    """
    paths_by_cat = get_cached_filler_paths(slug)
    paths = paths_by_cat.get(category, [])
    if not paths:
        return None, -1

    if len(paths) == 1:
        return paths[0], 0

    candidates = [i for i in range(len(paths)) if i != last_index]
    idx = random.choice(candidates)
    return paths[idx], idx


def select_filler_phrase(
    slug: str,
    category: str = "opener",
    *,
    last_index: int = -1,
) -> tuple[Path | None, FillerPhrase | None, int]:
    """
    ``select_filler_path`` の拡張版。Path と元の FillerPhrase の両方を返す。

    Phase 0.5-A の挙手機能で導入。handraise wav 再生時に bubble.update の text
    として元フレーズを表示する必要があるため、wav パスだけでなく FillerPhrase
    オブジェクト (text + emotion) も同時に取得できるようにする。

    既存 ``select_filler_path`` の呼出側は変更せず、新規の挙手フローのみ
    こちらを使う想定。

    Args:
        slug:        キャラクター slug
        category:    "opener" / "continue" / "bridge" / "closer" / "handraise"
        last_index:  直前選択 index (重複回避)

    Returns:
        (Path or None, FillerPhrase or None, index)
        wav が見つからない場合は (None, None, -1)。
        wav はあるが phrase が見つからない場合 (キャッシュとフレーズ定義の
        ズレ) は (Path, None, index) を返し、呼出側が text="" でフォールバック
        できるようにする。
    """
    paths_by_cat = get_cached_filler_paths(slug)
    paths = paths_by_cat.get(category, [])
    if not paths:
        return None, None, -1

    phrase_set = load_filler_phrases(slug)
    phrases_by_cat: dict[str, list[FillerPhrase]] = {
        "opener": phrase_set.opener,
        "continue": phrase_set.continue_,
        "bridge": phrase_set.bridge,
        "closer": phrase_set.closer,
        "handraise": phrase_set.handraise,
    }
    phrases = phrases_by_cat.get(category, [])

    if len(paths) == 1:
        phrase = phrases[0] if phrases else None
        return paths[0], phrase, 0

    candidates = [i for i in range(len(paths)) if i != last_index]
    idx = random.choice(candidates)
    phrase = phrases[idx] if idx < len(phrases) else None
    return paths[idx], phrase, idx


_FILLER_PROMPTS: dict[str, str] = {
    "mimi": (
        "あなたはミミ・オクタヴィアです。深海貴族のAITuberで、上品で優雅な口調で話します。\n"
        "「ですわ」「ですわねぇ」「ございますわ」のような語尾を使います。\n"
        "今、視聴者の質問について考え中です。考えをまとめている最中の自然な独り言を 1 文だけ生成してください。\n"
        "30文字以内。テキストのみ出力。JSON不要。"
    ),
    "chisame": (
        "あなたは波心ちさめです。冷静・論理的なAITuberで、落ち着いた丁寧語で話します。\n"
        "「……そうですね」「確認しますね」のような口調です。\n"
        "今、視聴者の質問について考え中です。考えをまとめている最中の自然な独り言を 1 文だけ生成してください。\n"
        "30文字以内。テキストのみ出力。JSON不要。"
    ),
    "sakura": (
        "あなたは八重笠さくらです。優しくのんびりしたAITuberで、柔らかい口調で話します。\n"
        "「ですねぇ」「ですよぉ」「かなぁ」のような語尾を使います。\n"
        "今、視聴者の質問について考え中です。考えをまとめている最中の自然な独り言を 1 文だけ生成してください。\n"
        "30文字以内。テキストのみ出力。JSON不要。"
    ),
    "octamaid": (
        "あなたはオクタメイドです。事務的で丁寧なAIアシスタントで、敬語で話します。\n"
        "「ございます」「いたします」のような語尾を使います。\n"
        "今、視聴者の質問について考え中です。考えをまとめている最中の自然な独り言を 1 文だけ生成してください。\n"
        "30文字以内。テキストのみ出力。JSON不要。"
    ),
}

_FILLER_DEFAULT_PROMPT = (
    "あなたはAITuberです。今、視聴者の質問について考え中です。\n"
    "考えをまとめている最中の自然な独り言を 1 文だけ生成してください。\n"
    "30文字以内。テキストのみ出力。JSON不要。"
)


def _build_filler_prompt(slug: str) -> str:
    """キャラクター設定からフィラー用システムプロンプトを構築する。

    voicepeak_emotion_keys が設定されているキャラクターの場合、
    JSON 形式 (response + emotion) の出力を指示するプロンプトを返す。
    未設定 (voicevox 等) の場合は従来通りプレーンテキスト指示。
    """
    from .characters import get_character

    try:
        char = get_character(slug)
    except KeyError:
        return _FILLER_DEFAULT_PROMPT

    base = _FILLER_PROMPTS.get(slug, _FILLER_DEFAULT_PROMPT)

    if not char.voicepeak_emotion_keys:
        return base  # emotion 非対応 → プレーンテキストのまま

    # JSON 形式指示に切り替え: 旧指示を除去して JSON フォーマットを追加
    base = base.replace("30文字以内。テキストのみ出力。JSON不要。", "").rstrip()
    emotion_template = ", ".join(f'"{k}": 0' for k in char.voicepeak_emotion_keys)
    return (
        base + "\n"
        f'出力は以下の JSON で返してください:\n'
        f'{{"response": "独り言テキスト(30文字以内)", "emotion": {{{emotion_template}}}, "pose": "neutral"}}\n'
        f"emotion の各値は 0〜100。キャラクターの性格と発話内容に合った値を設定してください。\n"
        f"pose は neutral / happy / angry / sad / fun のいずれか。\n"
        f"JSON のみ出力。"
    )


def _detect_provider(model: str) -> str:
    """モデル名からプロバイダーを自動判定する。"""
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    return "openai"


def _call_filler_llm(
    model: str,
    system_prompt: str,
    user_text: str,
) -> str | None:
    """
    フィラー用 LLM 呼び出し。モデル名からプロバイダーを自動判定。
    """
    provider = _detect_provider(model)

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0.9,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
            )
            return resp.content[0].text.strip()

        elif provider == "google":
            # Phase 0.5-J: module top の eager import を参照 (= 旧 lazy import を削除)。
            # _GOOGLE_GENAI が None なら package 未インストール、明示的 ImportError を投げる。
            if _GOOGLE_GENAI is None or _GOOGLE_GENAI_TYPES is None:
                raise ImportError("google.genai is required for provider=google")
            client = _GOOGLE_GENAI.Client()
            resp = client.models.generate_content(
                model=model,
                contents=user_text,
                config=_GOOGLE_GENAI_TYPES.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=150,
                    temperature=0.9,
                ),
            )
            return resp.text.strip()

        else:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_completion_tokens=150,
                temperature=0.9,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None

    except ImportError as exc:
        logger.warning("LLM フィラー: %s パッケージなし: %s", provider, exc)
        return None
    except Exception as exc:
        logger.warning("LLM フィラー呼び出し失敗: [%s] %s", provider, exc)
        return None


def _generate_filler_text(slug: str, *, user_text: str = "") -> str | None:
    """
    LLM でキャラクターらしいフィラー独り言を生成する。

    ユーザーの入力テキストが提供された場合、それに応じた文脈的な
    独り言を生成する。

    キャラクターの filler_model を使用。未設定なら L2_LLM_FILLER_MODEL env。
    モデル名からプロバイダー（OpenAI / Anthropic / Google）を自動判定。
    エラー時は None を返す（呼び出し元でフォールバック処理）。
    """
    from .characters import get_character

    try:
        char = get_character(slug)
    except KeyError:
        return None

    system_prompt = _build_filler_prompt(slug)

    # キャラクター設定 > 環境変数 > デフォルト
    if char.filler_model:
        model = char.filler_model
    else:
        model = os.environ.get("L2_LLM_FILLER_MODEL", "gpt-5.4-nano")

    if user_text:
        user_msg = f"ユーザーが「{user_text}」と聞いています。（独り言）"
    else:
        user_msg = "（独り言）"

    text = _call_filler_llm(model, system_prompt, user_msg)

    if text:
        # JSON レスポンス (emotion 付き) の場合はストリップしない
        # TTS の _parse_voicepeak_json() がそのまま解析する
        if not text.startswith("{"):
            text = text.strip('"').strip("「」")
        logger.info("LLM フィラー生成: [%s] %r (model=%s)", slug, text, model)
        return text if text else None

    logger.warning("LLM フィラー生成: 空応答 [%s] (model=%s)", slug, model)
    return None


def run_filler_loop(slug: str, stop_event: threading.Event, *, user_text: str = "") -> None:
    """
    ハイブリッドフィラー再生（並行 LLM+TTS）。

    Phase 1: opener 再生と同時に LLM+TTS を並行準備
    Phase 2: opener 終了後、準備済みなら即再生 / 未完了なら bridge 再生で待機
    Phase 3: LLM 生成フィラーを再生

    フレーズは最後まで再生する（途中中断しない）。
    LLM フィラーは 1 回のみ（繰り返さない）。

    Args:
        slug: キャラクター slug
        stop_event: 停止シグナル
        user_text: ユーザーの入力テキスト（文脈的フィラー生成に使用）
    """
    from .audio_io import play_audio_file
    from .characters import get_character
    from .tts import synthesize
    import time

    opener_path, _ = select_filler_path(slug, "opener")
    if opener_path is None:
        logger.warning("フィラー opener なし: %s。終了。", slug)
        return

    # LLM+TTS を並行準備するスレッド
    filler_ready = threading.Event()
    filler_audio = [None]  # [0] = audio_path or None
    filler_pose = [None]   # [0] = pose value (LLM JSON から抽出)

    def _prepare_filler():
        char = get_character(slug)
        filler_text = _generate_filler_text(slug, user_text=user_text)

        if filler_text and char:
            # LLM JSON から pose を抽出 (emotion 対応と同様に tts の parser を再利用)
            from .tts import _parse_voicepeak_json
            _, _, _, _pose = _parse_voicepeak_json(filler_text)
            filler_pose[0] = _pose
            logger.info("フィラー continue (LLM): [%s] %r", slug, filler_text)
            try:
                out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "audio"
                out_dir.mkdir(parents=True, exist_ok=True)
                tts_result = synthesize(
                    filler_text,
                    provider=char.tts_provider,
                    voice=char.tts_voice,
                    output_dir=str(out_dir),
                )
                filler_audio[0] = tts_result.audio_url.replace("file:///", "").replace("file://", "")
            except Exception as exc:
                logger.warning("フィラー TTS 失敗: [%s] %s", slug, exc)
                path, _ = select_filler_path(slug, "continue")
                if path:
                    filler_audio[0] = str(path)
        else:
            path, _ = select_filler_path(slug, "continue")
            if path:
                logger.info("フィラー continue (cached): [%s] %s", slug, path.name)
                filler_audio[0] = str(path)
        filler_ready.set()

    # Phase 1: opener 再生 + LLM+TTS 並行開始
    prep_thread = threading.Thread(target=_prepare_filler, daemon=True)
    prep_thread.start()

    logger.info("フィラー opener 再生: [%s] %s", slug, opener_path.name)
    play_audio_file(str(opener_path))

    if stop_event.is_set():
        logger.debug("フィラー終了（opener 後）: %s", slug)
        return

    # Phase 2: opener 終了後、LLM+TTS が準備できていなければ bridge で待機
    if not filler_ready.is_set():
        last_bridge_idx = -1
        while not filler_ready.is_set() and not stop_event.is_set():
            bridge_path, idx = select_filler_path(slug, "bridge", last_index=last_bridge_idx)
            if bridge_path is None:
                filler_ready.wait(timeout=0.5)
                continue
            last_bridge_idx = idx
            logger.info("フィラー bridge 再生: [%s] %s", slug, bridge_path.name)
            play_audio_file(str(bridge_path))
            # bridge 間に間を空ける（立て続けの再生を防止）。
            # 中間実走 3 回目 (logs/runs/run_loop_20260509_184653.log) で観察された
            # 「ブリッジフレーズがしつこい」(= 3 秒固定で連続再生されて視聴者が
            # 単調に感じる) 問題への対処として、4〜8 秒のランダムインターバルに
            # 変更する。lower=4 で「再生直後の即時連発」を防ぎ、upper=8 で
            # 「待ち時間が長すぎる空白」も防ぐ範囲。
            if not filler_ready.is_set():
                filler_ready.wait(timeout=random.uniform(4.0, 8.0))

    if stop_event.is_set():
        logger.debug("フィラー終了（bridge 後）: %s", slug)
        return

    # Phase 3: LLM 生成フィラーを再生 (pose があれば立ち絵も切替)
    # LLM フィラーは caller の声で再生される独り言なので、立ち絵も caller の
    # ものに切り替えるのが自然 (前キャラの立ち絵で caller の声が流れる方が
    # 違和感が大きい)。立ち絵切替は LLM フィラーの再生直前に行う。
    if filler_audio[0]:
        if filler_pose[0]:
            from .obs import set_pose
            set_pose(slug, filler_pose[0])
        time.sleep(0.3)
        play_audio_file(filler_audio[0])

    if stop_event.is_set():
        logger.debug("フィラー終了（continue 後）: %s", slug)
        return

    # Phase 4: 本命到着まで bridge filler + LLM continue で待機（最大 60 秒）
    #
    # Phase 0.5-D-3 follow-up 2: 「ブリッジフレーズがしつこい」問題への (A)+(D) 対処
    # 中間実走 3 / 4 回目で観察された「同じ bridge フレーズが 3〜4 秒間隔で繰り返さ
    # れる機械感」への根本対処。
    # - (A) bridge filler は最大 2 回まで再生 (= 同じキャッシュ wav の繰り返し感を緩和)
    # - (D) bridge 2 回後 → 10 秒経過するごとに LLM continue を再生成・再生
    #   (= 動的フレーズで多様性、LLM コストは Phase 4 全体で最大 ~4 回程度)
    #
    # Phase 2 (= LLM 推論待ち中の bridge) は短時間 (1-2 回程度) で済むので 4〜8 秒
    # ランダム化のみ。Phase 4 (= post-continue 本命待ち、最大 60 秒) は長時間化する
    # ことが多いので段階制御で「機械的繰り返し感」を構造的に抑制する。
    last_bridge_idx2 = -1
    phase4_deadline = time.monotonic() + 60.0
    bridge_play_count = 0
    MAX_BRIDGE_PLAYS = 2  # (A) bridge filler 再生回数の上限

    while not stop_event.is_set() and time.monotonic() < phase4_deadline:
        if bridge_play_count < MAX_BRIDGE_PLAYS:
            # (A) bridge filler を最大 2 回まで再生
            bridge_path, idx = select_filler_path(slug, "bridge", last_index=last_bridge_idx2)
            if bridge_path is None:
                stop_event.wait(timeout=0.5)
                continue
            last_bridge_idx2 = idx
            # bridge 間のインターバル: 4〜8 秒ランダム (= 上の Phase 2 と同じ範囲)
            stop_event.wait(timeout=random.uniform(4.0, 8.0))
            if stop_event.is_set():
                break
            logger.info("フィラー bridge 再生 (post-continue): [%s] %s", slug, bridge_path.name)
            play_audio_file(str(bridge_path))
            bridge_play_count += 1
            continue

        # (D) bridge 2 回後は 10 秒待ち → LLM continue を動的再生成・再生
        # WHY: 既存 bridge wav (= キャッシュから 2-3 種類のローテーション) の繰り返しを
        # 避け、状況に応じた多様な発話で「機械的繰り返し感」を解消する。LLM 呼出は
        # 10 秒間隔に制限することで Phase 4 内で最大 ~4 回程度に抑える (= コスト制限的)。
        stop_event.wait(timeout=10.0)
        if stop_event.is_set():
            break

        char = get_character(slug)
        additional_filler_text = _generate_filler_text(slug, user_text=user_text)
        if additional_filler_text and char:
            from .tts import _parse_voicepeak_json
            _, _, _, additional_pose = _parse_voicepeak_json(additional_filler_text)
            logger.info(
                "フィラー continue (LLM, additional): [%s] %r",
                slug, additional_filler_text[:60],
            )
            try:
                out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "audio"
                out_dir.mkdir(parents=True, exist_ok=True)
                tts_result = synthesize(
                    additional_filler_text,
                    provider=char.tts_provider,
                    voice=char.tts_voice,
                    output_dir=str(out_dir),
                )
                additional_audio = tts_result.audio_url.replace("file:///", "").replace("file://", "")
                if additional_pose:
                    from .obs import set_pose
                    set_pose(slug, additional_pose)
                time.sleep(0.3)
                play_audio_file(additional_audio)
            except Exception as exc:
                logger.warning(
                    "追加 LLM フィラー TTS 失敗 (= 無音待機継続): [%s] %s",
                    slug, exc,
                )
        else:
            # LLM 失敗 / 空応答時は無音待機継続 (= bridge 連発を避ける)
            logger.debug("追加 LLM フィラー生成失敗、無音待機継続: %s", slug)

    logger.debug("フィラー終了: %s", slug)
