"""
kb_loader.py — Knowledge Base ドキュメント読み込み・チャンク化

責務:
  - aibyss-workspace/docs/kb/ 以下の Markdown ファイルを読み込む
  - ドキュメントを適切なサイズのチャンクに分割する
  - build_index.py から呼ばれる。pipeline.py からは直接呼ばない

【チャンク化戦略】
  1. Markdown のヘッダー行（# / ## / ###）を起点にセクション分割
  2. さらに空行（\n\n）区切りのパラグラフ単位に細分化
  3. MIN_CHARS 未満のチャンクは前のチャンクに結合
  4. MAX_CHARS を超えるチャンクは強制分割
"""

import os
import re
from dataclasses import dataclass

MAX_CHARS = 800   # 1 チャンクの最大文字数（概算）
MIN_CHARS = 50    # これ未満は単独チャンクとしない


@dataclass
class DocChunk:
    """分割済みドキュメントチャンク。"""
    doc_id: str    # "{ファイル名(拡張子なし)}_{連番}"
    text: str      # チャンクのテキスト
    source: str    # 元ファイル名（例: "think_ai_lab.md"）


def load_kb(kb_dir: str) -> list[DocChunk]:
    """
    kb_dir 内の全 .md ファイルを読み込みチャンク化して返す。

    Args:
        kb_dir: Knowledge Base ディレクトリのパス

    Returns:
        DocChunk のリスト（ファイル名アルファベット順）
    """
    chunks: list[DocChunk] = []
    md_files = sorted(f for f in os.listdir(kb_dir) if f.endswith(".md"))
    for filename in md_files:
        filepath = os.path.join(kb_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        chunks.extend(_chunk_document(content, filename))
    return chunks


def _chunk_document(content: str, filename: str) -> list[DocChunk]:
    """
    Markdown テキストをチャンク化する。

    ヘッダー行（# / ## / ###）を起点にセクション分割したあと、
    空行区切りでパラグラフに細分化する。
    """
    base = os.path.splitext(filename)[0]

    # ヘッダー行（行頭の # ）を起点にセクション分割
    sections = re.split(r"(?=^#{1,3} )", content, flags=re.MULTILINE)

    raw_paragraphs: list[str] = []
    for section in sections:
        # 空行で分割してパラグラフへ
        paras = [p.strip() for p in re.split(r"\n{2,}", section)]
        raw_paragraphs.extend(p for p in paras if p)

    # MIN_CHARS 未満は前のパラグラフに結合
    merged: list[str] = []
    for para in raw_paragraphs:
        if merged and len(para) < MIN_CHARS:
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)

    # MAX_CHARS を超えるものを強制分割
    final: list[str] = []
    for para in merged:
        if len(para) <= MAX_CHARS:
            final.append(para)
        else:
            for i in range(0, len(para), MAX_CHARS):
                chunk = para[i : i + MAX_CHARS].strip()
                if chunk:
                    final.append(chunk)

    return [
        DocChunk(doc_id=f"{base}_{idx}", text=text, source=filename)
        for idx, text in enumerate(final)
        if text
    ]
