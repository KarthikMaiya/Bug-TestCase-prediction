from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from typing import Optional

from reranker import Reranker

from config import setup_logging

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback only used if tqdm is unavailable
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


DEFAULT_EMBEDDINGS_PATH = Path("data/bug_embeddings.npy")
DEFAULT_METADATA_PATH = Path("data/bug_metadata.csv")
DEFAULT_MASTER_PATH = Path("data/master_dataset.csv")
DEFAULT_OUTPUT_PATH = Path("evaluation_results.csv")
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_TOP_K_BUGS = 10
DEFAULT_TOP_K_TESTCASES = 10

BUG_ID_COLUMN = "bug_id"
BUG_TITLE_COLUMN = "bug_title"
BUG_DESCRIPTION_COLUMN = "bug_description"
BUG_TAGS_COLUMN = "bug_tags"
SEVERITY_COLUMN = "severity"
PRIORITY_COLUMN = "priority"
TESTCASE_ID_COLUMN = "testcase_id"
TESTCASE_TITLE_COLUMN = "testcase_title"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate testcase recommendation accuracy")
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write per-bug evaluation results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name used for query embeddings",
    )
    parser.add_argument(
        "--top-k-bugs",
        type=int,
        default=DEFAULT_TOP_K_BUGS,
        help="Number of similar bugs to retrieve per query",
    )
    parser.add_argument(
        "--top-k-testcases",
        type=int,
        default=DEFAULT_TOP_K_TESTCASES,
        help="Number of testcase recommendations to keep per query",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="If set, apply cross-encoder reranking to candidate testcases",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name to use for reranking",
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
    return metadata.reset_index(drop=True)


def normalize_bug_title(title: object) -> str:
    if title is None or pd.isna(title):
        return ""
    text = str(title).strip().lower()
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = []
    for token in re.split(r"\s+", text):
        if not token:
            continue
        # drop residual testcase / numeric prefixes such as 66804 or tc123
        if re.fullmatch(r"\d+", token):
            continue
        if re.search(r"\d", token) and re.search(r"[a-z]", token):
            token = re.sub(r"\d+", "", token)
            if not token:
                continue
        tokens.append(token)
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def format_metadata_tokens(tags: str, severity: str) -> str:
    token_parts: list[str] = []
    if severity:
        token_parts.append(f"[SEV_{severity.upper().replace(' ', '_')}]")
    if tags:
        for tag in [value.strip() for value in str(tags).split(",") if value.strip()]:
            token_parts.append(f"[TAG_{tag.upper().replace(' ', '_')}]")
    return " ".join(token_parts)


def build_enriched_bug_text(bug_row: pd.Series) -> str:
    original_title = str(bug_row.get(BUG_TITLE_COLUMN, "") or "").strip()
    normalized_title = normalize_bug_title(original_title)
    tags = str(bug_row.get(BUG_TAGS_COLUMN, "") or "").strip()
    severity = str(bug_row.get(SEVERITY_COLUMN, "") or "").strip()
    description = str(bug_row.get(BUG_DESCRIPTION_COLUMN, "") or "").strip()
    metadata_tokens = format_metadata_tokens(tags, severity)

    parts = [
        f"Normalized Title: {normalized_title}",
        f"Original Title: {original_title}",
        f"Tags: {tags}",
        f"Severity: {severity}",
        f"Description: {description}",
    ]
    if metadata_tokens:
        parts.append(metadata_tokens)
    return "\n".join(parts)


def ensure_enriched_embeddings(master_path: Path, embeddings_path: Path, metadata_path: Path, model_name: str, logger: logging.Logger) -> None:
    needs_rebuild = False

    if not embeddings_path.exists() or not metadata_path.exists():
        needs_rebuild = True
    else:
        try:
            metadata = pd.read_csv(metadata_path, nrows=5)
            required_columns = {"normalized_bug_title", "bug_text_enriched"}
            if not required_columns.issubset(set(metadata.columns)):
                needs_rebuild = True
        except Exception:
            needs_rebuild = True

    if not needs_rebuild:
        return

    build_script = Path(__file__).resolve().parent / "build_bug_embeddings.py"
    logger.info("Enriched embeddings are missing or stale; rebuilding via %s", build_script)
    subprocess.run(
        [
            sys.executable,
            str(build_script),
            "--input",
            str(master_path),
            "--embeddings-output",
            str(embeddings_path),
            "--metadata-output",
            str(metadata_path),
            "--model-name",
            model_name,
        ],
        check=True,
    )


def tokenize_text(text: object) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", str(text or "")) if token.strip()]


def build_bm25_index(metadata: pd.DataFrame) -> BM25Okapi:
    if "bug_text_enriched" in metadata.columns:
        corpus_texts = metadata["bug_text_enriched"].fillna("").astype(str).tolist()
    else:
        corpus_texts = [build_enriched_bug_text(row) for _, row in metadata.iterrows()]
    tokenized_corpus = [tokenize_text(text) for text in corpus_texts]
    return BM25Okapi(tokenized_corpus)


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


def build_query_model(model_name: str, logger: logging.Logger) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
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


def build_bug_query_text(bug_row: pd.Series) -> str:
    return build_enriched_bug_text(bug_row)


def retrieve_similar_bugs(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    metadata: pd.DataFrame,
    top_k_bugs: int,
    exclude_bug_id: int | None = None,
) -> pd.DataFrame:
    scores, indices = index.search(query_embedding, max(1, top_k_bugs))
    rows: list[dict[str, object]] = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0 or index_position >= len(metadata):
            continue
        row = metadata.iloc[int(index_position)]
        bug_id = int(row[BUG_ID_COLUMN])
        if exclude_bug_id is not None and bug_id == exclude_bug_id:
            continue
        rows.append(
            {
                BUG_ID_COLUMN: bug_id,
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
    exclude_bug_id: int | None = None,
) -> pd.DataFrame:
    scores = np.asarray(bm25_index.get_scores(tokenize_text(query_text)), dtype=np.float32)
    if scores.size == 0:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])

    candidate_order = np.argsort(scores)[::-1]
    rows: list[dict[str, object]] = []
    rank = 0
    for index_position in candidate_order:
        if index_position < 0 or index_position >= len(metadata):
            continue
        row = metadata.iloc[int(index_position)]
        bug_id = int(row[BUG_ID_COLUMN])
        if exclude_bug_id is not None and bug_id == exclude_bug_id:
            continue
        rank += 1
        rows.append(
            {
                BUG_ID_COLUMN: bug_id,
                BUG_TITLE_COLUMN: row[BUG_TITLE_COLUMN],
                "bm25_score": float(scores[int(index_position)]),
                "bm25_rank": rank,
            }
        )
        if rank >= max(1, top_k_bugs):
            break

    return pd.DataFrame(rows, columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])


def retrieve_hybrid_similar_bugs(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    bm25_index: BM25Okapi,
    query_text: str,
    metadata: pd.DataFrame,
    top_k_bugs: int,
    exclude_bug_id: int | None = None,
    rrf_k: int = 60,
) -> pd.DataFrame:
    dense_df = retrieve_similar_bugs(index, query_embedding, metadata, top_k_bugs, exclude_bug_id=exclude_bug_id).copy()
    dense_df["dense_rank"] = np.arange(1, len(dense_df) + 1, dtype=np.int32)

    bm25_df = retrieve_similar_bugs_bm25(
        bm25_index=bm25_index,
        query_text=query_text,
        metadata=metadata,
        top_k_bugs=top_k_bugs,
        exclude_bug_id=exclude_bug_id,
    ).copy()
    bm25_df["bm25_rank"] = np.arange(1, len(bm25_df) + 1, dtype=np.int32)

    combined = dense_df.merge(bm25_df, on=[BUG_ID_COLUMN, BUG_TITLE_COLUMN], how="outer")
    if combined.empty:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "similarity_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"])

    combined["dense_rank"] = combined["dense_rank"].fillna(10**9).astype(float)
    combined["bm25_rank"] = combined["bm25_rank"].fillna(10**9).astype(float)
    combined["hybrid_score"] = (1.0 / (rrf_k + combined["dense_rank"])) + (1.0 / (rrf_k + combined["bm25_rank"]))
    combined = combined.sort_values(
        ["hybrid_score", "similarity_score", "bm25_score", BUG_ID_COLUMN],
        ascending=[False, False, False, True],
    ).head(max(1, top_k_bugs)).reset_index(drop=True)
    return combined[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "similarity_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"]]


def aggregate_testcase_candidates(
    similar_bugs: pd.DataFrame,
    master: pd.DataFrame,
    top_k_testcases: int,
    score_column: str = "similarity_score",
) -> pd.DataFrame:
    if similar_bugs.empty:
        return pd.DataFrame(columns=[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"])

    candidate_links = master[[BUG_ID_COLUMN, TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN]].drop_duplicates()
    merged = similar_bugs[[BUG_ID_COLUMN, score_column]].merge(
        candidate_links,
        on=BUG_ID_COLUMN,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"])

    aggregated = (
        merged.groupby([TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN], as_index=False)
        .agg(
            aggregated_score=(score_column, "sum"),
            supporting_bug_count=(BUG_ID_COLUMN, "nunique"),
        )
        .sort_values(["aggregated_score", "supporting_bug_count", TESTCASE_ID_COLUMN], ascending=[False, False, True])
        .head(max(1, top_k_testcases))
        .reset_index(drop=True)
    )
    return aggregated


def predict_testcases(
    *,
    bug_row: pd.Series,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    master: pd.DataFrame,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    top_k_bugs: int,
    top_k_testcases: int,
    exclude_bug_id: int | None = None,
) -> pd.DataFrame:
    # This legacy function is preserved for backward compatibility. Use
    # predict_testcases with reranker support below by importing this module.
    query_text = build_bug_query_text(bug_row)
    query_embedding = embed_query(model, query_text)

    similar_bugs = retrieve_similar_bugs(
        index=index,
        query_embedding=query_embedding,
        metadata=metadata,
        top_k_bugs=top_k_bugs,
        exclude_bug_id=exclude_bug_id,
    )

    filtered_master = master
    if exclude_bug_id is not None:
        filtered_master = master[master[BUG_ID_COLUMN] != exclude_bug_id]

    recommendations = aggregate_testcase_candidates(similar_bugs, filtered_master, top_k_testcases, score_column="similarity_score")
    if recommendations.empty:
        return recommendations[[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"]]

    return recommendations[[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"]]


def _normalize_to_unit(values: list[float]) -> list[float]:
    import numpy as _np

    if not values:
        return []
    arr = _np.array(values, dtype=float)
    if arr.size == 1:
        return [1.0]
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return [1.0 for _ in arr.tolist()]
    scaled = (arr - mn) / (mx - mn)
    return scaled.tolist()


def predict_testcases(
    *,
    bug_row: pd.Series,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    master: pd.DataFrame,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi | None,
    retrieval_mode: str,
    top_k_bugs: int,
    top_k_testcases: int,
    exclude_bug_id: int | None = None,
    reranker: Optional[Reranker] = None,
    use_reranker: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (retrieval_df, final_df).

    - retrieval_df: candidates ordered by retrieval aggregated_score
    - final_df: candidates ordered by final_score (possibly reranked)
    """
    query_text = build_bug_query_text(bug_row)
    query_embedding = embed_query(model, query_text)

    if retrieval_mode == "dense":
        similar_bugs = retrieve_similar_bugs(
            index=index,
            query_embedding=query_embedding,
            metadata=metadata,
            top_k_bugs=top_k_bugs,
            exclude_bug_id=exclude_bug_id,
        )
        score_column = "similarity_score"
    elif retrieval_mode == "hybrid":
        if bm25_index is None:
            raise ValueError("bm25_index is required for hybrid retrieval")
        similar_bugs = retrieve_hybrid_similar_bugs(
            index=index,
            query_embedding=query_embedding,
            bm25_index=bm25_index,
            query_text=query_text,
            metadata=metadata,
            top_k_bugs=max(top_k_bugs, 50),
            exclude_bug_id=exclude_bug_id,
        )
        score_column = "hybrid_score"
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    filtered_master = master
    if exclude_bug_id is not None:
        filtered_master = master[master[BUG_ID_COLUMN] != exclude_bug_id]

    retrieval_df = aggregate_testcase_candidates(similar_bugs, filtered_master, top_k_testcases, score_column=score_column)
    if retrieval_df.empty:
        empty = pd.DataFrame(columns=[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"])
        return empty, empty

    retrieval_df = retrieval_df[[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "aggregated_score", "supporting_bug_count"]].copy()

    # prepare final_df with normalized retrieval score
    final_df = retrieval_df.copy()
    retrieval_scores = final_df["aggregated_score"].astype(float).tolist()
    final_df["retrieval_norm"] = _normalize_to_unit(retrieval_scores)

    if use_reranker and reranker is not None and not final_df.empty:
        pairs = [(query_text, str(t)) for t in final_df[TESTCASE_TITLE_COLUMN].tolist()]
        try:
            reranker_raw = reranker.score_pairs(pairs)
        except Exception:
            reranker_raw = [0.0 for _ in pairs]
        import numpy as _np

        reranker_arr = _np.array(reranker_raw, dtype=float)
        reranker_scores = (1.0 / (1.0 + _np.exp(-reranker_arr))).tolist()
        final_df["reranker_score"] = reranker_scores
        final_df["reranker_norm"] = _normalize_to_unit(final_df["reranker_score"].astype(float).tolist())
        final_df["final_score"] = (0.5 * final_df["retrieval_norm"].astype(float) + 0.5 * final_df["reranker_norm"].astype(float))
        final_df["ranking_method"] = "reranked"
    else:
        final_df["reranker_score"] = [None for _ in range(len(final_df))]
        final_df["reranker_norm"] = [None for _ in range(len(final_df))]
        final_df["final_score"] = final_df["retrieval_norm"]
        final_df["ranking_method"] = "retrieval"

    final_df = final_df.sort_values("final_score", ascending=False).head(max(1, top_k_testcases)).reset_index(drop=True)
    return retrieval_df.reset_index(drop=True), final_df.reset_index(drop=True)


def recall_at_k(predicted_ids: list[int], ground_truth_ids: set[int], k: int) -> int:
    return int(any(testcase_id in ground_truth_ids for testcase_id in predicted_ids[:k]))


def reciprocal_rank(predicted_ids: list[int], ground_truth_ids: set[int]) -> float:
    for rank, testcase_id in enumerate(predicted_ids, start=1):
        if testcase_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def format_id_list(values: Iterable[int]) -> str:
    return json.dumps([int(value) for value in values])


def compute_metrics(predicted_ids: list[int], ground_truth_ids: set[int]) -> dict[str, float | int]:
    hit_1 = recall_at_k(predicted_ids, ground_truth_ids, 1)
    hit_3 = recall_at_k(predicted_ids, ground_truth_ids, 3)
    hit_5 = recall_at_k(predicted_ids, ground_truth_ids, 5)
    hit_10 = recall_at_k(predicted_ids, ground_truth_ids, 10)
    mrr = reciprocal_rank(predicted_ids, ground_truth_ids)
    return {
        "hit@1": int(hit_1),
        "hit@3": int(hit_3),
        "hit@5": int(hit_5),
        "hit@10": int(hit_10),
        "mrr": float(mrr),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()

    logger.info("Loading bug embeddings from %s", args.embeddings)
    ensure_enriched_embeddings(args.master, args.embeddings, args.metadata, args.model_name, logger)
    embeddings = load_embeddings(args.embeddings)
    logger.info("Embedding shape: %s", embeddings.shape)

    logger.info("Loading bug metadata from %s", args.metadata)
    metadata = load_metadata(args.metadata)

    logger.info("Loading master dataset from %s", args.master)
    master = load_master_dataset(args.master)

    index = build_index(embeddings)
    logger.info("Index size: %s", index.ntotal)

    bm25_index = build_bm25_index(metadata)
    logger.info("Built BM25 index from enriched bug text")

    model = build_query_model(args.model_name, logger)

    evaluation_rows: list[dict[str, object]] = []
    metric_names = ["baseline", "reranked", "hybrid", "hybrid_reranked"]
    metric_totals: dict[str, dict[str, float]] = {
        name: {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0} for name in metric_names
    }

    unique_bug_rows = (
        master[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, BUG_DESCRIPTION_COLUMN, BUG_TAGS_COLUMN, SEVERITY_COLUMN, PRIORITY_COLUMN]]
        .drop_duplicates(subset=[BUG_ID_COLUMN])
        .reset_index(drop=True)
    )

    bug_metadata_ids = set(metadata[BUG_ID_COLUMN].dropna().astype(int).tolist())

    logger.info("Evaluating %d unique bugs", len(unique_bug_rows))
    # initialize reranker if requested
    reranker = None
    if args.use_reranker:
        try:
            reranker = Reranker(model_name=args.reranker_model)
        except Exception as e:
            logger.error("Failed to initialize reranker: %s", e)
            reranker = None

    for _, bug_row in tqdm(unique_bug_rows.iterrows(), total=len(unique_bug_rows), desc="Evaluating", unit="bug"):
        bug_id = int(bug_row[BUG_ID_COLUMN])
        if bug_id not in bug_metadata_ids:
            logger.warning("Skipping bug_id %s because it is missing from bug metadata", bug_id)
            continue

        ground_truth_ids = set(
            master.loc[master[BUG_ID_COLUMN] == bug_id, TESTCASE_ID_COLUMN]
            .dropna()
            .astype(int)
            .tolist()
        )
        if not ground_truth_ids:
            continue

        prediction_start = time.perf_counter()
        dense_retrieval_df, dense_final_df = predict_testcases(
            bug_row=bug_row,
            embeddings=embeddings,
            metadata=metadata,
            master=master,
            model=model,
            index=index,
            bm25_index=None,
            retrieval_mode="dense",
            top_k_bugs=args.top_k_bugs,
            top_k_testcases=args.top_k_testcases,
            exclude_bug_id=bug_id,
            reranker=None,
            use_reranker=False,
        )
        dense_prediction_runtime = time.perf_counter() - prediction_start

        dense_ids = dense_retrieval_df[TESTCASE_ID_COLUMN].dropna().astype(int).tolist()[: args.top_k_testcases] if not dense_retrieval_df.empty else []
        dense_metrics = compute_metrics(dense_ids, ground_truth_ids)
        for metric_name, metric_value in dense_metrics.items():
            metric_totals["baseline"][metric_name] += float(metric_value)

        if args.use_reranker:
            reranked_prediction_start = time.perf_counter()
            _, dense_reranked_df = predict_testcases(
                bug_row=bug_row,
                embeddings=embeddings,
                metadata=metadata,
                master=master,
                model=model,
                index=index,
                bm25_index=None,
                retrieval_mode="dense",
                top_k_bugs=args.top_k_bugs,
                top_k_testcases=args.top_k_testcases,
                exclude_bug_id=bug_id,
                reranker=reranker,
                use_reranker=True,
            )
            dense_reranked_runtime = time.perf_counter() - reranked_prediction_start
            dense_reranked_ids = dense_reranked_df[TESTCASE_ID_COLUMN].dropna().astype(int).tolist() if not dense_reranked_df.empty else []
            dense_reranked_metrics = compute_metrics(dense_reranked_ids, ground_truth_ids)
            for metric_name, metric_value in dense_reranked_metrics.items():
                metric_totals["reranked"][metric_name] += float(metric_value)
        else:
            dense_reranked_df = dense_final_df.copy()
            dense_reranked_ids = dense_ids
            dense_reranked_metrics = dense_metrics
            dense_reranked_runtime = 0.0
            for metric_name, metric_value in dense_reranked_metrics.items():
                metric_totals["reranked"][metric_name] += float(metric_value)

        hybrid_prediction_start = time.perf_counter()
        hybrid_retrieval_df, hybrid_final_df = predict_testcases(
            bug_row=bug_row,
            embeddings=embeddings,
            metadata=metadata,
            master=master,
            model=model,
            index=index,
            bm25_index=bm25_index,
            retrieval_mode="hybrid",
            top_k_bugs=max(args.top_k_bugs, 50),
            top_k_testcases=args.top_k_testcases,
            exclude_bug_id=bug_id,
            reranker=None,
            use_reranker=False,
        )
        hybrid_prediction_runtime = time.perf_counter() - hybrid_prediction_start

        hybrid_ids = hybrid_retrieval_df[TESTCASE_ID_COLUMN].dropna().astype(int).tolist() if not hybrid_retrieval_df.empty else []
        hybrid_metrics = compute_metrics(hybrid_ids, ground_truth_ids)
        for metric_name, metric_value in hybrid_metrics.items():
            metric_totals["hybrid"][metric_name] += float(metric_value)

        if args.use_reranker:
            hybrid_reranked_start = time.perf_counter()
            _, hybrid_reranked_df = predict_testcases(
                bug_row=bug_row,
                embeddings=embeddings,
                metadata=metadata,
                master=master,
                model=model,
                index=index,
                bm25_index=bm25_index,
                retrieval_mode="hybrid",
                top_k_bugs=max(args.top_k_bugs, 50),
                top_k_testcases=args.top_k_testcases,
                exclude_bug_id=bug_id,
                reranker=reranker,
                use_reranker=True,
            )
            hybrid_reranked_runtime = time.perf_counter() - hybrid_reranked_start
            hybrid_reranked_ids = hybrid_reranked_df[TESTCASE_ID_COLUMN].dropna().astype(int).tolist() if not hybrid_reranked_df.empty else []
            hybrid_reranked_metrics = compute_metrics(hybrid_reranked_ids, ground_truth_ids)
            for metric_name, metric_value in hybrid_reranked_metrics.items():
                metric_totals["hybrid_reranked"][metric_name] += float(metric_value)
        else:
            hybrid_reranked_df = hybrid_final_df.copy()
            hybrid_reranked_ids = hybrid_ids
            hybrid_reranked_metrics = hybrid_metrics
            hybrid_reranked_runtime = 0.0
            for metric_name, metric_value in hybrid_reranked_metrics.items():
                metric_totals["hybrid_reranked"][metric_name] += float(metric_value)

        active_df = hybrid_reranked_df if args.use_reranker else hybrid_final_df
        active_ids = hybrid_reranked_ids if args.use_reranker else hybrid_ids
        active_metrics = hybrid_reranked_metrics if args.use_reranker else hybrid_metrics

        row = {
            BUG_ID_COLUMN: bug_id,
            "ground_truth_testcases": format_id_list(sorted(ground_truth_ids)),
            "baseline_predicted_testcases": format_id_list(dense_ids),
            "dense_reranked_predicted_testcases": format_id_list(dense_reranked_ids),
            "hybrid_predicted_testcases": format_id_list(hybrid_ids),
            "hybrid_reranked_predicted_testcases": format_id_list(hybrid_reranked_ids),
            "predicted_testcases": format_id_list(active_ids),
            "hit@1": int(active_metrics["hit@1"]),
            "hit@3": int(active_metrics["hit@3"]),
            "hit@5": int(active_metrics["hit@5"]),
            "hit@10": int(active_metrics["hit@10"]),
            "mrr": float(active_metrics["mrr"]),
            "ranking_method": "hybrid+reranked" if args.use_reranker else "hybrid",
            "retrieval_scores": format_id_list(active_df["aggregated_score"].astype(float).tolist() if not active_df.empty else []),
            "reranker_scores": format_id_list(active_df["reranker_score"].astype(float).tolist() if (not active_df.empty and "reranker_score" in active_df.columns) else []),
            "final_scores": format_id_list(active_df["final_score"].astype(float).tolist() if (not active_df.empty and "final_score" in active_df.columns) else []),
        }
        evaluation_rows.append(row)

        logger.debug(
            "Evaluated bug_id %s dense=%.4fs hybrid=%.4fs dense_reranked=%.4fs hybrid_reranked=%.4fs",
            bug_id,
            dense_prediction_runtime,
            hybrid_prediction_runtime,
            dense_reranked_runtime,
            hybrid_reranked_runtime,
        )

    results = pd.DataFrame(evaluation_rows)
    if results.empty:
        raise RuntimeError("No evaluation rows were produced. Check that master_dataset.csv contains valid bug↔testcase mappings.")

    results.to_csv(args.output, index=False)

    evaluated_bug_count = len(results)
    logger.info("Evaluation complete for %d bugs", evaluated_bug_count)
    for label, totals in metric_totals.items():
        recall_at_1 = totals["hit@1"] / evaluated_bug_count
        recall_at_3 = totals["hit@3"] / evaluated_bug_count
        recall_at_5 = totals["hit@5"] / evaluated_bug_count
        recall_at_10 = totals["hit@10"] / evaluated_bug_count
        mean_reciprocal_rank = totals["mrr"] / evaluated_bug_count

        display_label = {
            "baseline": "BASELINE",
            "reranked": "RERANKED",
            "hybrid": "HYBRID",
            "hybrid_reranked": "HYBRID+RERANKED",
        }[label]
        logger.info(display_label)
        logger.info("Recall@1: %.4f", recall_at_1)
        logger.info("Recall@3: %.4f", recall_at_3)
        logger.info("Recall@5: %.4f", recall_at_5)
        logger.info("Recall@10: %.4f", recall_at_10)
        logger.info("MRR: %.4f", mean_reciprocal_rank)

    logger.info("Saved per-bug results to %s", args.output)
    logger.info("Total runtime seconds: %.2f", time.perf_counter() - start_time)

    # PART A: Failure analysis for hit@10 == 0
    try:
        failures = results[results["hit@10"] == 0].copy()
        total_failures = len(failures)
        failure_rate_percent = 100.0 * total_failures / evaluated_bug_count
        logger.info("Total failures (hit@10==0): %d (%.2f%%)", total_failures, failure_rate_percent)

        # print first 20 failures
        first20 = failures.head(20)
        print("First 20 failures (bug_id, bug_title, ground_truth_testcases, predicted_testcases):")
        # join to get bug_title and extra fields
        master_ext = master[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, BUG_DESCRIPTION_COLUMN, SEVERITY_COLUMN, BUG_TAGS_COLUMN]].drop_duplicates(subset=[BUG_ID_COLUMN])
        failures = failures.merge(master_ext, left_on=BUG_ID_COLUMN, right_on=BUG_ID_COLUMN, how="left")
        for _, row in first20.merge(master_ext, on=BUG_ID_COLUMN, how="left").iterrows():
            print(row[BUG_ID_COLUMN], row.get(BUG_TITLE_COLUMN, ""), row.get("ground_truth_testcases", ""), row.get("predicted_testcases", ""))

        # diagnostics
        missing_description = failures[failures[BUG_DESCRIPTION_COLUMN].isna() | (failures[BUG_DESCRIPTION_COLUMN].str.strip() == "")]
        pct_missing_description = 100.0 * len(missing_description) / total_failures if total_failures else 0.0

        # most common failed testcase ids (from ground truth lists)
        import json as _json
        from collections import Counter

        gt_counts = Counter()
        for s in failures["ground_truth_testcases"].dropna().tolist():
            try:
                ids = _json.loads(s)
                gt_counts.update([int(x) for x in ids])
            except Exception:
                continue

        top_failed_testcases = gt_counts.most_common(10)

        severity_counts = failures[SEVERITY_COLUMN].fillna("<missing>").value_counts().to_dict()

        # common title token patterns
        import re as _re

        token_counter = Counter()
        for t in failures[BUG_TITLE_COLUMN].fillna(""):
            tokens = [tok.lower() for tok in _re.findall(r"\w+", str(t)) if len(tok) > 2]
            token_counter.update(tokens)
        common_title_tokens = token_counter.most_common(20)

        # infer likely causes (simple heuristics)
        causes = Counter()
        # heuristic: if description missing -> sparse metadata
        if pct_missing_description > 20.0:
            causes["sparse metadata"] += 1
        # heuristic: if many failures share same ground-truth testcase -> mapping ambiguity / noisy mapping
        if any(count > 10 for _, count in top_failed_testcases):
            causes["mapping ambiguity or noisy testcase mapping"] += 1
        # heuristic: if top title tokens are generic like "error", "failed", etc -> generic title
        generic_tokens = {"error", "failed", "issue", "test", "problem"}
        common_tokens_set = {t for t, _ in common_title_tokens[:10]}
        if common_tokens_set & generic_tokens:
            causes["duplicate/generic title"] += 1
        # default add semantic retrieval miss
        causes["semantic retrieval miss"] += 1

        # write report
        report_lines = []
        report_lines.append(f"Total evaluated bugs: {evaluated_bug_count}")
        report_lines.append(f"Total failures (hit@10==0): {total_failures}")
        report_lines.append(f"Failure rate percent: {failure_rate_percent:.2f}%")
        report_lines.append("")
        report_lines.append("Top failed ground-truth testcase ids:")
        for tc_id, cnt in top_failed_testcases:
            report_lines.append(f"- {tc_id}: {cnt}")
        report_lines.append("")
        report_lines.append(f"Percent failures with missing description: {pct_missing_description:.2f}%")
        report_lines.append("")
        report_lines.append("Failure severity distribution:")
        for sev, cnt in severity_counts.items():
            report_lines.append(f"- {sev}: {cnt}")
        report_lines.append("")
        report_lines.append("Common title tokens:")
        for tok, cnt in common_title_tokens[:20]:
            report_lines.append(f"- {tok}: {cnt}")
        report_lines.append("")
        report_lines.append("Inferred likely causes (heuristic):")
        for cause, cnt in causes.items():
            report_lines.append(f"- {cause}: score {cnt}")
        report_lines.append("")
        report_lines.append("Representative failures (first 20):")
        for _, r in failures.head(20).iterrows():
            report_lines.append(f"Bug {r[BUG_ID_COLUMN]}: {r.get(BUG_TITLE_COLUMN)} | severity={r.get(SEVERITY_COLUMN)} | ground_truth={r.get('ground_truth_testcases')} | predicted={r.get('predicted_testcases')}")

        with open("failure_analysis_report.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines))

        logger.info("Saved diagnostic report to failure_analysis_report.txt")
    except Exception as e:
        logger.exception("Failed to produce failure analysis: %s", e)


if __name__ == "__main__":
    main()