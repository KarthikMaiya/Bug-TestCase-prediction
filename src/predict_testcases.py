from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import setup_logging

DEFAULT_EMBEDDINGS_PATH = Path("data/bug_embeddings.npy")
DEFAULT_METADATA_PATH = Path("data/bug_metadata.csv")
DEFAULT_MASTER_PATH = Path("data/master_dataset.csv")
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_TOP_K_BUGS = 10
DEFAULT_TOP_K_TESTCASES = 10
DEFAULT_HYBRID_TOP_K_BUGS = 50
DEFAULT_RRF_K = 60

BUG_ID_COLUMN = "bug_id"
BUG_TITLE_COLUMN = "bug_title"
TESTCASE_ID_COLUMN = "testcase_id"
TESTCASE_TITLE_COLUMN = "testcase_title"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict testcase recommendations for a new bug")
    parser.add_argument("query", type=str, help="Bug text used to retrieve similar historical bugs")
    parser.add_argument("-k", "--top-k-bugs", type=int, default=DEFAULT_TOP_K_BUGS, help="Number of similar bugs to retrieve")
    parser.add_argument(
        "--top-k-testcases",
        type=int,
        default=DEFAULT_TOP_K_TESTCASES,
        help="Number of testcase recommendations to return",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the saved historical bug embeddings",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the saved bug metadata CSV",
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER_PATH,
        help="Path to the bug↔testcase mapping dataset",
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
        raise ValueError(f"Bug metadata file is missing required columns: {missing_columns}")
    metadata = metadata.reset_index(drop=True)
    if "normalized_bug_title" not in metadata.columns:
        metadata["normalized_bug_title"] = metadata[BUG_TITLE_COLUMN].apply(normalize_bug_title)
    if "bug_text_enriched" not in metadata.columns:
        metadata["bug_text_enriched"] = metadata.apply(build_enriched_bug_text, axis=1)
    return metadata


def load_master_dataset(master_path: Path) -> pd.DataFrame:
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master dataset file: {master_path}")
    master = pd.read_csv(master_path)
    required_columns = [BUG_ID_COLUMN, TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN]
    missing_columns = [column for column in required_columns if column not in master.columns]
    if missing_columns:
        raise ValueError(f"Master dataset file is missing required columns: {missing_columns}")
    return master


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index


def build_query_model(model_name: str) -> tuple[SentenceTransformer, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(model_name, device=device)
    return model, device


def normalize_bug_title(title: object) -> str:
    import re

    if title is None:
        return ""
    s = str(title).strip().lower()
    tokens = [t for t in re.split(r"\s+", s) if t]
    cleaned_tokens = []
    for t in tokens:
        if re.fullmatch(r"\d+", t):
            continue
        if re.search(r"\d", t) and re.search(r"[a-zA-Z]", t):
            t = re.sub(r"\d+", "", t)
            if not t:
                continue
        cleaned_tokens.append(t)
    normalized = " ".join(cleaned_tokens)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_enriched_bug_text(row: pd.Series) -> str:
    original_title = str(row.get(BUG_TITLE_COLUMN, "") or "").strip()
    normalized_title = normalize_bug_title(original_title)
    tags = str(row.get("bug_tags", "") or "").strip()
    severity = str(row.get("severity", "") or "").strip()
    description = str(row.get("bug_description", "") or "").strip()

    token_parts: list[str] = []
    if severity:
        token_parts.append(f"[SEV_{severity.upper().replace(' ', '_')}]")
    if tags:
        for tag in [value.strip() for value in tags.split(",") if value.strip()]:
            token_parts.append(f"[TAG_{tag.upper().replace(' ', '_')}]")

    parts = [
        f"Normalized Title: {normalized_title}",
        f"Original Title: {original_title}",
        f"Tags: {tags}",
        f"Severity: {severity}",
        f"Description: {description}",
    ]
    if token_parts:
        parts.append(" ".join(token_parts))
    return "\n".join(parts)


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", str(text)) if token.strip()]


def build_bm25_index(metadata: pd.DataFrame) -> tuple[BM25Okapi, list[str]]:
    corpus_texts = metadata["bug_text_enriched"].fillna("").astype(str).tolist()
    tokenized_corpus = [tokenize_text(text) for text in corpus_texts]
    return BM25Okapi(tokenized_corpus), corpus_texts


def build_enriched_query_text(query: str, tags: str | None = None, severity: str | None = None, description: str | None = None) -> str:
    # Treat `query` as the original title when provided interactively
    orig_title = query.strip()
    normalized = normalize_bug_title(orig_title)
    tags = (tags or "").strip()
    severity = (severity or "").strip()
    description = (description or "").strip()
    parts = [f"Normalized Title: {normalized}", f"Original Title: {orig_title}", f"Tags: {tags}", f"Severity: {severity}", f"Description: {description}"]
    token_parts = []
    if severity:
        token_parts.append(f"[SEV_{severity.upper().replace(' ', '_')}]")
    if tags:
        for tag in [t.strip() for t in tags.split(",") if t.strip()]:
            token_parts.append(f"[TAG_{tag.upper().replace(' ', '_')}]")
    if token_parts:
        parts.append(" ".join(token_parts))
    return "\n".join(parts)


def embed_query(model: SentenceTransformer, query_text: str) -> np.ndarray:
    enriched = build_enriched_query_text(query_text)
    query = f"query: {enriched}"
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


def pretty_print_dataframe(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    if df.empty:
        print("<empty>")
        return
    print(df.to_string(index=False))


def retrieve_similar_bugs(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    metadata: pd.DataFrame,
    top_k_bugs: int,
) -> pd.DataFrame:
    scores, indices = index.search(query_embedding, max(1, top_k_bugs))
    rows: List[dict] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0 or index_position >= len(metadata):
            continue
        row = metadata.iloc[int(index_position)]
        rows.append(
            {
                BUG_ID_COLUMN: row[BUG_ID_COLUMN],
                BUG_TITLE_COLUMN: row[BUG_TITLE_COLUMN],
                "similarity_score": float(score),
            }
        )
    return pd.DataFrame(rows, columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "similarity_score"])


def retrieve_similar_bugs_bm25(
    bm25_index: BM25Okapi,
    query_text: str,
    metadata: pd.DataFrame,
    top_k_bugs: int,
) -> pd.DataFrame:
    scores = bm25_index.get_scores(tokenize_text(query_text))
    order = np.argsort(np.asarray(scores))[::-1][: max(1, top_k_bugs)]
    rows: List[dict] = []
    for rank, index_position in enumerate(order, start=1):
        if index_position < 0 or index_position >= len(metadata):
            continue
        row = metadata.iloc[int(index_position)]
        rows.append(
            {
                BUG_ID_COLUMN: row[BUG_ID_COLUMN],
                BUG_TITLE_COLUMN: row[BUG_TITLE_COLUMN],
                "bm25_score": float(scores[int(index_position)]),
                "bm25_rank": int(rank),
            }
        )
    return pd.DataFrame(rows, columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])


def retrieve_hybrid_similar_bugs(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    bm25_index: BM25Okapi,
    query_text: str,
    metadata: pd.DataFrame,
    top_k_bugs: int = DEFAULT_HYBRID_TOP_K_BUGS,
    exclude_bug_id: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> pd.DataFrame:
    dense_df = retrieve_similar_bugs(index, query_embedding, metadata, top_k_bugs)
    dense_df = dense_df.copy()
    dense_df["dense_rank"] = dense_df.index + 1

    bm25_df = retrieve_similar_bugs_bm25(bm25_index, query_text, metadata, top_k_bugs)
    bm25_df = bm25_df.copy()
    bm25_df["bm25_rank"] = bm25_df.index + 1

    combined = dense_df.merge(bm25_df, on=[BUG_ID_COLUMN, BUG_TITLE_COLUMN], how="outer")
    if exclude_bug_id is not None:
        combined = combined[combined[BUG_ID_COLUMN] != exclude_bug_id]

    if combined.empty:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"])

    dense_rank = combined["dense_rank"].fillna(10**9).astype(float)
    bm25_rank = combined["bm25_rank"].fillna(10**9).astype(float)
    combined["hybrid_score"] = (1.0 / (rrf_k + dense_rank)) + (1.0 / (rrf_k + bm25_rank))
    combined = combined.sort_values(
        ["hybrid_score", "dense_score", "bm25_score", BUG_ID_COLUMN],
        ascending=[False, False, False, True],
    ).head(max(1, top_k_bugs)).reset_index(drop=True)
    return combined[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"]]


def aggregate_testcase_candidates(
    similar_bugs: pd.DataFrame,
    master: pd.DataFrame,
    top_k_testcases: int,
    score_column: str = "similarity_score",
) -> pd.DataFrame:
    if similar_bugs.empty:
        return pd.DataFrame(
            columns=[
                TESTCASE_ID_COLUMN,
                TESTCASE_TITLE_COLUMN,
                "similarity_sum",
                "support_count",
                "rank_bonus",
                "final_score",
            ]
        )

    ranked_bugs = similar_bugs.reset_index(drop=True).copy()
    ranked_bugs["rank_position"] = ranked_bugs.index.astype(float)

    merged = ranked_bugs[[BUG_ID_COLUMN, score_column, "rank_position"]].merge(
        master[[BUG_ID_COLUMN, TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN]],
        on=BUG_ID_COLUMN,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                TESTCASE_ID_COLUMN,
                TESTCASE_TITLE_COLUMN,
                "similarity_sum",
                "support_count",
                "rank_bonus",
                "final_score",
            ]
        )

    merged["rank_bonus_component"] = 1.0 / (merged["rank_position"] + 1.0)

    aggregated = (
        merged.groupby([TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN], as_index=False)
        .agg(
            similarity_sum=(score_column, "sum"),
            support_count=(BUG_ID_COLUMN, "nunique"),
            rank_bonus=("rank_bonus_component", "sum"),
        )
        .assign(
            final_score=lambda frame: (
                0.70 * frame["similarity_sum"]
                + 0.20 * frame["support_count"]
                + 0.10 * frame["rank_bonus"]
            )
        )
        .sort_values(["final_score", "similarity_sum", "support_count", TESTCASE_ID_COLUMN], ascending=[False, False, False, True])
        .head(max(1, top_k_testcases))
        .reset_index(drop=True)
    )
    return aggregated[[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "similarity_sum", "support_count", "rank_bonus", "final_score"]]


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

    logger.info("Loading master dataset from %s", args.master)
    master = load_master_dataset(args.master)

    index = build_index(embeddings)
    logger.info("Index size: %s", index.ntotal)
    bm25_index, _ = build_bm25_index(metadata)

    model, device = build_query_model(args.model_name)
    logger.info("Selected device: %s", device)

    query_embedding = embed_query(model, args.query)
    query_text = build_enriched_query_text(args.query)

    bug_start = time.perf_counter()
    similar_bugs = retrieve_hybrid_similar_bugs(
        index=index,
        query_embedding=query_embedding,
        bm25_index=bm25_index,
        query_text=query_text,
        metadata=metadata,
        top_k_bugs=max(args.top_k_bugs, DEFAULT_HYBRID_TOP_K_BUGS),
    )
    bug_runtime = time.perf_counter() - bug_start
    pretty_print_dataframe("Top similar bugs", similar_bugs)

    testcase_start = time.perf_counter()
    testcase_recommendations = aggregate_testcase_candidates(similar_bugs, master, args.top_k_testcases, score_column="hybrid_score")
    testcase_runtime = time.perf_counter() - testcase_start
    pretty_print_dataframe("Top testcase recommendations", testcase_recommendations)

    logger.info("Bug retrieval runtime seconds: %.4f", bug_runtime)
    logger.info("Testcase aggregation runtime seconds: %.4f", testcase_runtime)
    logger.info("Total runtime seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
