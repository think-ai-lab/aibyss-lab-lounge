#!/usr/bin/env python3
"""dict.vdc2 (VOICEPEAK エクスポート辞書) → readings.json (irodori 読み辞書) 変換。

irodori はエンジン側の読み辞書を持たないため、L2 側で「喋るテキスト」に読み置換を適用する
(tts._apply_readings)。その辞書の初期データを、ルカの VOICEPEAK 辞書エクスポートから生成する。

変換ルール:
  - 各エントリの ``sur``(表層)→ ``pron``(カタカナ読み)を ``global`` に入れる。
  - **文脈依存の短い多音字**(方=かた/ほう、十分=じゅうぶん/じっぷん 等)は ``_excluded_review``
    に隔離する。素朴な文字列置換では一方の読みを強制して他方を壊すため(irodori は文脈で
    そこそこ読み分けるので任せる)。loader はこのキーを無視する。
  - VOICEPEAK 固有の ``accentType`` / ``overwriteAccents`` は ``_accent_meta`` として保持する。
    irodori はアクセント制御の入力を持たないため **inert(無視)** だが、VOICEPEAK 復帰や
    将来のアクセント対応モデル用に情報を失わないよう残す。
  - 重複 ``sur`` は後勝ちで dedup。

使い方:
    uv run python scripts/import_vdc2_readings.py \
        --vdc2 "D:\\work\\vp_cache\\dict.vdc2" \
        --out  "T:\\irodori-tts\\reference_voices\\readings.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 文脈依存で読みが変わる短い多音字 → naive 置換では一方を壊すので global に入れない。
# (例: 方→カタ を全置換すると「あちらの方(=ほう)」が壊れる。irodori に任せる。)
# ルカが精査して増減できるよう _excluded_review に隔離する。
_AMBIGUOUS_SURFACES: set[str] = {"方", "十分", "品"}


def convert(vdc2_path: Path, out_path: Path) -> dict:
    """dict.vdc2 を読み、readings.json 構造の dict を返す + ファイルに書く。"""
    entries = json.loads(vdc2_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"想定外の vdc2 形式 (JSON 配列のはず): {vdc2_path}")

    global_map: dict[str, str] = {}
    excluded: dict[str, str] = {}
    accent_meta: dict[str, dict] = {}

    for e in entries:
        sur = e.get("sur")
        pron = e.get("pron")
        if not sur or not pron:
            continue
        # 文脈依存の多音字は隔離、それ以外は global (後勝ち dedup)
        if sur in _AMBIGUOUS_SURFACES:
            excluded[sur] = pron
        else:
            global_map[sur] = pron
        # アクセント情報は inert メタとして保持
        meta = {k: e[k] for k in ("accentType", "overwriteAccents", "pos", "priority") if k in e}
        if meta:
            accent_meta[sur] = meta

    out = {
        "_comment": (
            "irodori 読み辞書。global = 表層 → カタカナ読み (喋るテキストに longest-match-first で置換)。"
            " characters = キャラ別上書き (任意)。_excluded_review = 文脈依存の多音字 (適用しない)。"
            " _accent_meta = VOICEPEAK 由来のアクセント情報 (irodori は無視 / 将来用)。"
            " 生成元: VOICEPEAK エクスポート辞書 (scripts/import_vdc2_readings.py)。"
        ),
        "global": dict(sorted(global_map.items(), key=lambda kv: -len(kv[0]))),
        "characters": {},
        "_excluded_review": excluded,
        "_accent_meta": accent_meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[import_vdc2] global={len(global_map)} 件 / _excluded_review={len(excluded)} 件 "
        f"/ _accent_meta={len(accent_meta)} 件 -> {out_path}"
    )
    if excluded:
        print(f"[import_vdc2] 隔離した多音字 (要精査): {sorted(excluded)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="VOICEPEAK dict.vdc2 -> irodori readings.json")
    ap.add_argument("--vdc2", default=r"D:\work\vp_cache\dict.vdc2", help="入力 .vdc2 (JSON)")
    ap.add_argument(
        "--out",
        default=r"T:\irodori-tts\reference_voices\readings.json",
        help="出力 readings.json",
    )
    args = ap.parse_args()
    convert(Path(args.vdc2), Path(args.out))


if __name__ == "__main__":
    main()
