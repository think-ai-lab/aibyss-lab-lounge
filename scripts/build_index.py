#!/usr/bin/env python
"""
build_index.py — seed corpus から RAG 用 index を生成する

使用方法:
  uv run python scripts/build_index.py

オプション:
  --kb-dir      Knowledge Base ディレクトリ (default: ../../aibyss-workspace/docs/kb)
  --output-dir  index 出力先ディレクトリ    (default: ./data/index)
  --batch-size  embedding バッチサイズ       (default: 64)

前提:
  - OPENAI_API_KEY 環境変数が設定済みであること
  - uv add --optional rag openai numpy でパッケージがインストール済みであること

成果物:
  {output_dir}/chunks.json    — チャンクメタデータ
  {output_dir}/embeddings.npy — numpy 埋め込み行列 (N, 1536)
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# スクリプトから src/ を参照できるように sys.path を調整
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
sys.path.insert(0, _SRC_DIR)

DEFAULT_KB_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "aibyss-workspace", "docs", "kb")
)
DEFAULT_OUTPUT_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "data", "index")
)

EMBEDDING_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge Base index を生成する")
    parser.add_argument(
        "--kb-dir",
        default=DEFAULT_KB_DIR,
        help=f"Knowledge Base ディレクトリ (default: {DEFAULT_KB_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"index 出力先ディレクトリ (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="embedding バッチサイズ (default: 64)",
    )
    return parser.parse_args()


def embed_batch(texts: list[str], client) -> list[list[float]]:
    """OpenAI Embeddings API でバッチ埋め込みを取得する。"""
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def main() -> None:
    args = parse_args()
    kb_dir = os.path.normpath(args.kb_dir)
    output_dir = os.path.normpath(args.output_dir)

    # .env 読み込み（OPENAI_API_KEY 等）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # ─── 前提チェック ───────────────────────────────────────────────
    if not os.path.isdir(kb_dir):
        logger.error("kb-dir が見つかりません: %s", kb_dir)
        sys.exit(1)

    try:
        import numpy as np
        from openai import OpenAI
    except ImportError as exc:
        logger.error(
            "依存パッケージが未インストールです: %s\n"
            "  uv add --optional rag openai numpy を実行してください。",
            exc,
        )
        sys.exit(1)

    # ─── チャンク読み込み ───────────────────────────────────────────
    from lab_lounge.kb_loader import load_kb

    logger.info("KB 読み込み中: %s", kb_dir)
    chunks = load_kb(kb_dir)
    logger.info("チャンク数: %d", len(chunks))

    if not chunks:
        logger.error("チャンクが 0 件です。kb-dir を確認してください。")
        sys.exit(1)

    # ─── 埋め込み生成 ───────────────────────────────────────────────
    client = OpenAI()
    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []
    batch_size = args.batch_size
    total_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(
        "埋め込み生成開始: model=%s chunks=%d batches=%d",
        EMBEDDING_MODEL,
        len(texts),
        total_batches,
    )

    t0 = time.monotonic()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_batch(batch, client)
        all_embeddings.extend(vecs)
        logger.info("  batch %d/%d 完了", i // batch_size + 1, total_batches)

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info("埋め込み生成完了: latency_ms=%d", latency_ms)

    # ─── 保存 ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    chunks_path = os.path.join(output_dir, "chunks.json")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")

    chunk_data = [
        {"doc_id": c.doc_id, "text": c.text, "source": c.source}
        for c in chunks
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    matrix = np.array(all_embeddings, dtype=np.float32)
    np.save(embeddings_path, matrix)

    logger.info("保存完了:")
    logger.info("  %s  (%d チャンク)", chunks_path, len(chunks))
    logger.info("  %s  shape=%s", embeddings_path, matrix.shape)
    logger.info("")
    logger.info("次のステップ: L2_ENABLE_RAG=true で emitter を実行してください")
    logger.info("  uv run python -m lab_lounge.emitter \"Think-AI Lab.について教えて\"")


if __name__ == "__main__":
    main()
