from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import setup_logging

DEFAULT_EMBEDDINGS_PATH = Path("data/bug_embeddings.npy")
DEFAULT_METADATA_PATH = Path("data/bug_metadata.csv")
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_TOP_K = 10

BUG_ID_COLUMN = "bug_id"
BUG_TITLE_COLUMN = "bug_title"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve similar bugs from the historical bug embedding index")
    parser.add_argument("query", type=str, help="Query text used to search for similar bugs")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the saved bug embedding matrix",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the saved bug metadata CSV",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name used for query embeddings",
    )
    return parser


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required_columns = [BUG_ID_COLUMN, BUG_TITLE_COLUMN]
    missing_columns = [column for column in required_columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata file is missing required columns: {missing_columns}")
    return metadata


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index


def build_query_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return SentenceTransformer(model_name, device=device)


def embed_query(model: SentenceTransformer, query_text: str) -> np.ndarray:
    query = f"query: {query_text.strip()}"
    query_embedding = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    return query_embedding


def pretty_print_results(results: pd.DataFrame) -> None:
    if results.empty:
        print("No similar bugs found.")
        return
    print(results.to_string(index=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()

    logger.info("Loading bug embeddings from %s", args.embeddings)
    embeddings = load_embeddings(args.embeddings)
    logger.info("Embedding shape: %s", embeddings.shape)

    logger.info("Loading bug metadata from %s", args.metadata)
    metadata = load_metadata(args.metadata).reset_index(drop=True)

    index = build_index(embeddings)
    logger.info("Index size: %s", index.ntotal)

    model = build_query_model(args.model_name)
    query_embedding = embed_query(model, args.query)

    top_k = max(1, int(args.top_k))
    query_start = time.perf_counter()
    scores, indices = index.search(query_embedding, top_k)
    query_runtime = time.perf_counter() - query_start

    rows: List[dict] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0 or index_position >= len(metadata):
            continue
        row = metadata.iloc[int(index_position)]
        rows.append(
            {
                "bug_id": row[BUG_ID_COLUMN],
                "bug_title": row[BUG_TITLE_COLUMN],
                "similarity_score": float(score),
            }
        )

    results = pd.DataFrame(rows, columns=["bug_id", "bug_title", "similarity_score"])
    pretty_print_results(results)

    logger.info("Query runtime seconds: %.4f", query_runtime)
    logger.info("Total runtime seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
