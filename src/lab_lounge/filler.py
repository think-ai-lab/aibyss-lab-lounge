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

logger = logging.getLogger(__name__)

_FILLER_PHRASES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filler_phrases"
_FILLER_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "filler_cache"

_VALID_SECTIONS = {"opener", "continue", "closer"}


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
    closer: list[FillerPhrase] = field(default_factory=list)

    @property
    def all_phrases(self) -> list[tuple[str, FillerPhrase]]:
        """(category, phrase) ペアのフラットリスト（キャッシュ生成用）。"""
        items: list[tuple[str, FillerPhrase]] = []
        for cat, lst in [("opener", self.opener),
                         ("continue", self.continue_),
                         ("closer", self.closer)]:
            for p in lst:
                items.append((cat, p))
        return items

    def __len__(self) -> int:
        return len(self.opener) + len(self.continue_) + len(self.closer)


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
        elif current_section == "closer":
            result.closer.append(phrase)

    total = len(result)
    logger.debug(
        "フィラーフレーズ読み込み: %s (opener=%d continue=%d closer=%d total=%d)",
        slug, len(result.opener), len(result.continue_), len(result.closer), total,
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
        {"opener": [Path, ...], "continue": [Path, ...], "closer": [Path, ...]}
    """
    cache_dir = _FILLER_CACHE_DIR / slug
    result: dict[str, list[Path]] = {}
    for cat in ("opener", "continue", "closer"):
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
    if len(phrase_set) == 0:
        logger.warning("フィラーフレーズが定義されていません: %s", slug)
        return {"opener": [], "continue": [], "closer": []}

    char = get_character(slug)
    if char is None:
        logger.warning("キャラクターが見つかりません: %s", slug)
        return {"opener": [], "continue": [], "closer": []}

    cache_dir = _FILLER_CACHE_DIR / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --force 時は旧キャッシュを全削除
    if force:
        for old_file in cache_dir.glob("*.wav"):
            old_file.unlink()

    result_paths: dict[str, list[Path]] = {"opener": [], "continue": [], "closer": []}
    counters: dict[str, int] = {"opener": 0, "continue": 0, "closer": 0}

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
                max_tokens=60,
                temperature=0.9,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
            )
            return resp.content[0].text.strip()

        elif provider == "google":
            import google.genai as genai
            client = genai.Client()
            resp = client.models.generate_content(
                model=model,
                contents=user_text,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=60,
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
                max_completion_tokens=60,
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


def _generate_filler_text(slug: str) -> str | None:
    """
    LLM でキャラクターらしいフィラー独り言を生成する。

    キャラクターの filler_model を使用。未設定なら L2_LLM_FILLER_MODEL env。
    モデル名からプロバイダー（OpenAI / Anthropic / Google）を自動判定。
    エラー時は None を返す（呼び出し元でフォールバック処理）。
    """
    from .characters import get_character

    try:
        char = get_character(slug)
    except KeyError:
        return None

    system_prompt = _FILLER_PROMPTS.get(slug, _FILLER_DEFAULT_PROMPT)

    # キャラクター設定 > 環境変数 > デフォルト
    if char.filler_model:
        model = char.filler_model
    else:
        model = os.environ.get("L2_LLM_FILLER_MODEL", "gpt-5.4-nano")

    text = _call_filler_llm(model, system_prompt, "（独り言）")

    if text:
        text = text.strip('"').strip("「」")
        logger.info("LLM フィラー生成: [%s] %r (model=%s)", slug, text, model)
        return text if text else None

    logger.warning("LLM フィラー生成: 空応答 [%s] (model=%s)", slug, model)
    return None


def run_filler_loop(slug: str, stop_event: threading.Event) -> None:
    """
    ハイブリッドフィラー再生。

    Phase 1: 事前生成 opener を即再生（ゼロレイテンシ）
    Phase 2: LLM で 1 回だけフィラーテキスト生成 → TTS → 再生
             （opener 再生中に LLM 呼び出しが並行で走る）
             LLM/TTS 失敗時は事前生成 continue を 1 つ再生

    フレーズは最後まで再生する（途中中断しない）。
    LLM フィラーは 1 回のみ（繰り返さない）。
    """
    from .audio_io import play_audio_file
    from .characters import get_character
    from .tts import synthesize
    import tempfile
    import time

    # Phase 1: 事前生成 opener を即再生
    opener_path, _ = select_filler_path(slug, "opener")
    if opener_path is None:
        logger.warning("フィラー opener なし: %s。終了。", slug)
        return

    logger.info("フィラー opener 再生: [%s] %s", slug, opener_path.name)
    play_audio_file(str(opener_path))

    if stop_event.is_set():
        logger.debug("フィラー終了（opener 後）: %s", slug)
        return

    # Phase 2: LLM フィラー 1 回のみ
    char = get_character(slug)
    filler_text = _generate_filler_text(slug)

    if filler_text and char:
        logger.info("フィラー continue (LLM): [%s] %r", slug, filler_text)
        try:
            tmp_dir = Path(__file__).resolve().parent.parent.parent / "data" / "audio"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            tts_result = synthesize(
                filler_text,
                provider=char.tts_provider,
                voice=char.tts_voice,
                output_dir=str(tmp_dir),
            )
            audio_path = tts_result.audio_url.replace("file:///", "").replace("file://", "")

            # 生成済みフィラーは常に最後まで再生する（ぶつ切り防止）
            time.sleep(0.3)
            play_audio_file(audio_path)

            Path(audio_path).unlink(missing_ok=True)

        except Exception as exc:
            logger.warning("フィラー TTS 失敗: [%s] %s", slug, exc)
            # フォールバック
            path, _ = select_filler_path(slug, "continue")
            if path:
                time.sleep(0.3)
                play_audio_file(str(path))
    else:
        # LLM 失敗 → 事前生成 continue を 1 つ再生
        path, _ = select_filler_path(slug, "continue")
        if path:
            logger.info("フィラー continue (cached): [%s] %s", slug, path.name)
            time.sleep(0.3)
            play_audio_file(str(path))

    logger.debug("フィラー終了: %s", slug)
