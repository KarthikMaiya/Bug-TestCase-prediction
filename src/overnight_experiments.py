from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import setup_logging
from reranker import Reranker

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback only used if tqdm is unavailable
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_PATH = ROOT / "data" / "master_dataset.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "overnight_experiments"
DEFAULT_RESULTS_PATH = ROOT / "experiment_results.csv"
DEFAULT_BEST_CONFIG_PATH = ROOT / "best_config.json"
DEFAULT_LOG_PATH = ROOT / "overnight_log.txt"
DEFAULT_SUMMARY_PATH = ROOT / "overnight_summary.txt"
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_KS = [10, 20, 50]
DEFAULT_AGGREGATION_WEIGHTS = [
    {"similarity": 0.7, "support": 0.2, "rank": 0.1},
    {"similarity": 0.6, "support": 0.3, "rank": 0.1},
    {"similarity": 0.5, "support": 0.3, "rank": 0.2},
]
DEFAULT_RERANKER_WEIGHTS = [0.25, 0.50, 0.75]
RRF_K = 60

BUG_ID_COLUMN = "bug_id"
BUG_TITLE_COLUMN = "bug_title"
BUG_DESCRIPTION_COLUMN = "bug_description"
BUG_TAGS_COLUMN = "bug_tags"
SEVERITY_COLUMN = "severity"
PRIORITY_COLUMN = "priority"
TESTCASE_ID_COLUMN = "testcase_id"
TESTCASE_TITLE_COLUMN = "testcase_title"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_set: str
    top_k_dense: int
    top_k_bm25: int
    aggregation_weights: dict[str, float]
    reranker_weight: float
    use_normalized_titles: bool
    use_metadata_tokens: bool
    use_bm25: bool
    use_reranker: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class ExperimentOutcome:
    experiment_name: str
    timestamp: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    runtime_seconds: float


class TeeLogger:
    def __init__(self, log_path: Path, logger: logging.Logger) -> None:
        self.log_path = log_path
        self.logger = logger
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        if not message:
            return
        self._file.write(message)
        self._file.flush()
        self.logger.info(message.rstrip("\n"))

    def close(self) -> None:
        self._file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run overnight auto-experiments for testcase recommendation")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER_PATH, help="Path to master_dataset.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Directory for cached embeddings and metadata")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Path to experiment_results.csv")
    parser.add_argument("--best-config", type=Path, default=DEFAULT_BEST_CONFIG_PATH, help="Path to best_config.json")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="Path to overnight_log.txt")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="Path to overnight_summary.txt")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model for bug embeddings")
    parser.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL, help="Cross-encoder model for reranking")
    parser.add_argument("--max-stale-experiments", type=int, default=8, help="Stop after this many consecutive non-improving experiments")
    parser.add_argument("--top-k-list", type=int, nargs="*", default=DEFAULT_TOP_KS, help="Dense/BM25 top-k search values")
    return parser


def normalize_bug_title(title: object) -> str:
    if title is None or pd.isna(title):
        return ""
    text = str(title).strip().lower()
    text = re.sub(r"\b\d+\b", " ", text)
    tokens: list[str] = []
    for token in re.split(r"\s+", text):
        if not token:
            continue
        if re.search(r"\d", token) and re.search(r"[a-z]", token):
            token = re.sub(r"\d+", "", token)
            if not token:
                continue
        if re.fullmatch(r"\d+", token):
            continue
        tokens.append(token)
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def tokenize_text(text: object) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", str(text or "")) if token.strip()]


def build_bug_text(row: pd.Series, *, use_normalized_title: bool, use_metadata_tokens: bool) -> str:
    original_title = str(row.get(BUG_TITLE_COLUMN, "") or "").strip()
    normalized_title = normalize_bug_title(original_title)
    tags = str(row.get(BUG_TAGS_COLUMN, "") or "").strip()
    severity = str(row.get(SEVERITY_COLUMN, "") or "").strip()
    description = str(row.get(BUG_DESCRIPTION_COLUMN, "") or "").strip()

    title_line = normalized_title if use_normalized_title else original_title.lower()
    parts = [
        f"Normalized Title: {normalized_title}",
        f"Original Title: {original_title}",
        f"Tags: {tags}",
        f"Severity: {severity}",
        f"Description: {description}",
    ]
    if not use_normalized_title:
        parts[0] = f"Normalized Title: {title_line}"

    if use_metadata_tokens:
        token_parts: list[str] = []
        if severity:
            token_parts.append(f"[SEV_{severity.upper().replace(' ', '_')}]")
        if tags:
            for tag in [value.strip() for value in str(tags).split(",") if value.strip()]:
                token_parts.append(f"[TAG_{tag.upper().replace(' ', '_')}]")
        if token_parts:
            parts.append(" ".join(token_parts))

    return "\n".join(parts)


def load_master_dataset(master_path: Path) -> pd.DataFrame:
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master dataset file: {master_path}")
    master = pd.read_csv(master_path)
    required_columns = [BUG_ID_COLUMN, TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, BUG_TITLE_COLUMN, BUG_DESCRIPTION_COLUMN, BUG_TAGS_COLUMN, SEVERITY_COLUMN]
    missing = [column for column in required_columns if column not in master.columns]
    if missing:
        raise ValueError(f"Master dataset is missing required columns: {missing}")
    return master.reset_index(drop=True)


def ensure_embeddings_artifacts(
    master: pd.DataFrame,
    artifacts_dir: Path,
    model_name: str,
    use_normalized_titles: bool,
    use_metadata_tokens: bool,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rep_key = json.dumps(
        {
            "model_name": model_name,
            "use_normalized_titles": use_normalized_titles,
            "use_metadata_tokens": use_metadata_tokens,
            "rows": len(master),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(rep_key.encode("utf-8")).hexdigest()[:16]
    run_dir = artifacts_dir / digest
    embeddings_path = run_dir / "bug_embeddings.npy"
    metadata_path = run_dir / "bug_metadata.csv"

    if embeddings_path.exists() and metadata_path.exists():
        try:
            cached = pd.read_csv(metadata_path, nrows=5)
            if {"bug_text", "normalized_bug_title"}.issubset(set(cached.columns)):
                return embeddings_path, metadata_path
        except Exception:
            pass

    logger.info("Rebuilding embeddings for representation hash %s", digest)
    run_dir.mkdir(parents=True, exist_ok=True)

    unique_bugs = master[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, BUG_DESCRIPTION_COLUMN, BUG_TAGS_COLUMN, SEVERITY_COLUMN, PRIORITY_COLUMN]].drop_duplicates(subset=[BUG_ID_COLUMN]).reset_index(drop=True)
    unique_bugs["normalized_bug_title"] = unique_bugs[BUG_TITLE_COLUMN].apply(normalize_bug_title)
    unique_bugs["bug_text"] = unique_bugs.apply(
        lambda row: build_bug_text(row, use_normalized_title=use_normalized_titles, use_metadata_tokens=use_metadata_tokens),
        axis=1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    model = SentenceTransformer(model_name, device=device)
    texts = unique_bugs["bug_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    unique_bugs.to_csv(metadata_path, index=False)
    return embeddings_path, metadata_path


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index


def build_bm25_index(metadata: pd.DataFrame) -> BM25Okapi:
    corpus = metadata["bug_text"].fillna("").astype(str).tolist()
    tokenized = [tokenize_text(text) for text in corpus]
    return BM25Okapi(tokenized)


def build_query_model(model_name: str) -> tuple[SentenceTransformer, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(model_name, device=device)
    return model, device


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


def build_query_text(row: pd.Series, *, use_normalized_titles: bool, use_metadata_tokens: bool) -> str:
    return build_bug_text(row, use_normalized_title=use_normalized_titles, use_metadata_tokens=use_metadata_tokens)


def retrieve_dense_candidates(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    metadata: pd.DataFrame,
    top_k_dense: int,
    exclude_bug_id: int | None = None,
) -> pd.DataFrame:
    scores, indices = index.search(query_embedding, max(1, top_k_dense))
    rows: list[dict[str, object]] = []
    rank = 0
    for score, index_position in zip(scores[0], indices[0]):
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
                "dense_score": float(score),
                "dense_rank": int(rank),
            }
        )
    return pd.DataFrame(rows, columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "dense_rank"])


def retrieve_bm25_candidates(
    bm25_index: BM25Okapi,
    query_text: str,
    metadata: pd.DataFrame,
    top_k_bm25: int,
    exclude_bug_id: int | None = None,
) -> pd.DataFrame:
    scores = np.asarray(bm25_index.get_scores(tokenize_text(query_text)), dtype=np.float32)
    if scores.size == 0:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])
    order = np.argsort(scores)[::-1]
    rows: list[dict[str, object]] = []
    rank = 0
    for idx in order:
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata.iloc[int(idx)]
        bug_id = int(row[BUG_ID_COLUMN])
        if exclude_bug_id is not None and bug_id == exclude_bug_id:
            continue
        rank += 1
        rows.append(
            {
                BUG_ID_COLUMN: bug_id,
                BUG_TITLE_COLUMN: row[BUG_TITLE_COLUMN],
                "bm25_score": float(scores[int(idx)]),
                "bm25_rank": int(rank),
            }
        )
        if rank >= max(1, top_k_bm25):
            break
    return pd.DataFrame(rows, columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])


def retrieve_hybrid_candidates(
    dense_df: pd.DataFrame,
    bm25_df: pd.DataFrame,
    top_k: int,
    rrf_k: int = RRF_K,
) -> pd.DataFrame:
    dense = dense_df.copy()
    bm25 = bm25_df.copy()
    if dense.empty and bm25.empty:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"])
    dense[BUG_ID_COLUMN] = dense[BUG_ID_COLUMN].astype(str)
    bm25[BUG_ID_COLUMN] = bm25[BUG_ID_COLUMN].astype(str)
    dense[BUG_TITLE_COLUMN] = dense[BUG_TITLE_COLUMN].fillna("").astype(str)
    bm25[BUG_TITLE_COLUMN] = bm25[BUG_TITLE_COLUMN].fillna("").astype(str)

    logging.getLogger(__name__).info("Hybrid merge dense dtypes: %s", dense.dtypes.to_dict())
    logging.getLogger(__name__).info("Hybrid merge bm25 dtypes: %s", bm25.dtypes.to_dict())
    logging.getLogger(__name__).info("Hybrid merge dense shape: %s", dense.shape)
    logging.getLogger(__name__).info("Hybrid merge bm25 shape: %s", bm25.shape)

    try:
        combined = dense.merge(bm25, on=[BUG_ID_COLUMN, BUG_TITLE_COLUMN], how="outer", suffixes=("_dense", "_bm25"))
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.exception("Hybrid merge failed after key normalization: %s", exc)
        logger.info("Dense dtypes before failure: %s", dense.dtypes.to_dict())
        logger.info("BM25 dtypes before failure: %s", bm25.dtypes.to_dict())
        logger.info("Dense sample rows before failure: %s", dense.head(5).to_dict(orient="records"))
        logger.info("BM25 sample rows before failure: %s", bm25.head(5).to_dict(orient="records"))
        raise RuntimeError("Hybrid retrieval merge failed after key normalization; see logs for dense/bm25 dtypes and sample rows.") from exc

    if combined.empty:
        return pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"])

    combined[BUG_ID_COLUMN] = pd.to_numeric(combined[BUG_ID_COLUMN], errors="coerce")
    combined = combined.dropna(subset=[BUG_ID_COLUMN]).copy()
    combined[BUG_ID_COLUMN] = combined[BUG_ID_COLUMN].astype(int)

    # Restore original score columns if pandas added suffixes during merge.
    if "dense_score" not in combined.columns and "dense_score_dense" in combined.columns:
        combined["dense_score"] = combined["dense_score_dense"]
    if "bm25_score" not in combined.columns and "bm25_score_bm25" in combined.columns:
        combined["bm25_score"] = combined["bm25_score_bm25"]
    if "dense_rank" not in combined.columns and "dense_rank_dense" in combined.columns:
        combined["dense_rank"] = combined["dense_rank_dense"]
    if "bm25_rank" not in combined.columns and "bm25_rank_bm25" in combined.columns:
        combined["bm25_rank"] = combined["bm25_rank_bm25"]

    combined["dense_rank"] = combined["dense_rank"].fillna(10**9).astype(float)
    combined["bm25_rank"] = combined["bm25_rank"].fillna(10**9).astype(float)
    combined["hybrid_score"] = (1.0 / (rrf_k + combined["dense_rank"])) + (1.0 / (rrf_k + combined["bm25_rank"]))
    combined = combined.sort_values(
        ["hybrid_score", "dense_score", "bm25_score", BUG_ID_COLUMN],
        ascending=[False, False, False, True],
    ).head(max(1, top_k)).reset_index(drop=True)
    return combined[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "dense_score", "bm25_score", "dense_rank", "bm25_rank", "hybrid_score"]]


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 1:
        return [1.0]
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return [1.0 for _ in arr.tolist()]
    return ((arr - mn) / (mx - mn)).tolist()


def build_candidate_score(
    frame: pd.DataFrame,
    aggregation_weights: dict[str, float],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = frame.copy()
    scored["rank_bonus_component"] = 1.0 / (scored.index.astype(float) + 1.0)
    agg = (
        scored.groupby([TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN], as_index=False)
        .agg(
            similarity_sum=("hybrid_score", "sum"),
            support_count=(BUG_ID_COLUMN, "nunique"),
            rank_bonus=("rank_bonus_component", "sum"),
        )
    )
    agg["final_score"] = (
        aggregation_weights["similarity"] * agg["similarity_sum"]
        + aggregation_weights["support"] * agg["support_count"]
        + aggregation_weights["rank"] * agg["rank_bonus"]
    )
    return agg.sort_values(["final_score", "similarity_sum", "support_count", TESTCASE_ID_COLUMN], ascending=[False, False, False, True]).reset_index(drop=True)


def rerank_candidates(
    query_text: str,
    candidates: pd.DataFrame,
    reranker: Reranker | None,
    reranker_weight: float,
) -> pd.DataFrame:
    if candidates.empty or reranker is None:
        frame = candidates.copy()
        frame["reranker_score"] = np.nan
        frame["ranking_method"] = "retrieval"
        frame["final_score"] = frame["final_score"] if "final_score" in frame.columns else frame.get("similarity_sum", pd.Series(dtype=float))
        return frame

    frame = candidates.copy()
    pairs = [(query_text, str(title)) for title in frame[TESTCASE_TITLE_COLUMN].tolist()]
    try:
        raw_scores = reranker.score_pairs(pairs)
    except Exception:
        raw_scores = [0.0 for _ in pairs]
    frame["reranker_score"] = normalize_scores([float(v) for v in raw_scores])
    retrieval_scores = normalize_scores(frame["final_score"].astype(float).tolist())
    if len(retrieval_scores) != len(frame):
        retrieval_scores = [0.0 for _ in range(len(frame))]
    frame["retrieval_score"] = retrieval_scores
    frame["final_score"] = (1.0 - reranker_weight) * frame["retrieval_score"] + reranker_weight * frame["reranker_score"]
    frame["ranking_method"] = "reranked"
    return frame.sort_values(["final_score", TESTCASE_ID_COLUMN], ascending=[False, True]).reset_index(drop=True)


def recall_at_k(predicted_ids: list[int], ground_truth_ids: set[int], k: int) -> int:
    return int(any(testcase_id in ground_truth_ids for testcase_id in predicted_ids[:k]))


def reciprocal_rank(predicted_ids: list[int], ground_truth_ids: set[int]) -> float:
    for rank, testcase_id in enumerate(predicted_ids, start=1):
        if testcase_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_metrics(predicted_ids: list[int], ground_truth_ids: set[int]) -> dict[str, float]:
    return {
        "Recall@1": float(recall_at_k(predicted_ids, ground_truth_ids, 1)),
        "Recall@3": float(recall_at_k(predicted_ids, ground_truth_ids, 3)),
        "Recall@5": float(recall_at_k(predicted_ids, ground_truth_ids, 5)),
        "Recall@10": float(recall_at_k(predicted_ids, ground_truth_ids, 10)),
        "MRR": float(reciprocal_rank(predicted_ids, ground_truth_ids)),
    }


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _read_git_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def build_experiment_grid() -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    set1, set2, set3 = DEFAULT_AGGREGATION_WEIGHTS
    for top_k_dense in DEFAULT_TOP_KS:
        for top_k_bm25 in DEFAULT_TOP_KS:
            for aggregation_weights in (set1, set2, set3):
                for reranker_weight in DEFAULT_RERANKER_WEIGHTS:
                    configs.extend(
                        [
                            ExperimentConfig("dense_only", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, False, False, False),
                            ExperimentConfig("dense_reranker", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, False, False, True),
                            ExperimentConfig("dense_normalized_titles", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, True, False, False, False),
                            ExperimentConfig("dense_metadata_tokens", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, True, False, False),
                            ExperimentConfig("dense_bm25_hybrid", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, False, True, False),
                            ExperimentConfig("dense_bm25_reranker", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, False, True, True),
                            ExperimentConfig("dense_bm25_reranker_metadata", top_k_dense, top_k_bm25, aggregation_weights, reranker_weight, False, True, True, True),
                        ]
                    )
    # preserve ordering while deduplicating exact duplicates
    unique: dict[str, ExperimentConfig] = {}
    for config in configs:
        unique[config.to_json()] = config
    return list(unique.values())


def evaluate_config(
    *,
    config: ExperimentConfig,
    master: pd.DataFrame,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    reranker: Reranker | None,
    query_model: SentenceTransformer,
    logger: logging.Logger,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    unique_bug_rows = master[[BUG_ID_COLUMN, BUG_TITLE_COLUMN, BUG_DESCRIPTION_COLUMN, BUG_TAGS_COLUMN, SEVERITY_COLUMN, PRIORITY_COLUMN]].drop_duplicates(subset=[BUG_ID_COLUMN]).reset_index(drop=True)
    evaluation_rows: list[dict[str, Any]] = []
    totals = {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}
    evaluated = 0

    for _, bug_row in tqdm(unique_bug_rows.iterrows(), total=len(unique_bug_rows), desc=config.experiment_set, unit="bug"):
        bug_id = int(bug_row[BUG_ID_COLUMN])
        ground_truth_ids = set(master.loc[master[BUG_ID_COLUMN] == bug_id, TESTCASE_ID_COLUMN].dropna().astype(int).tolist())
        if not ground_truth_ids:
            continue
        evaluated += 1

        query_text = build_query_text(
            bug_row,
            use_normalized_titles=config.use_normalized_titles,
            use_metadata_tokens=config.use_metadata_tokens,
        )
        query_embedding = embed_query(query_model, query_text)

        dense_candidates = retrieve_dense_candidates(index, query_embedding, metadata, config.top_k_dense, exclude_bug_id=bug_id)
        bm25_candidates = retrieve_bm25_candidates(bm25_index, query_text, metadata, config.top_k_bm25, exclude_bug_id=bug_id) if config.use_bm25 else pd.DataFrame(columns=[BUG_ID_COLUMN, BUG_TITLE_COLUMN, "bm25_score", "bm25_rank"])
        if config.use_bm25:
            hybrid_candidates = retrieve_hybrid_candidates(dense_candidates, bm25_candidates, top_k=max(config.top_k_dense, config.top_k_bm25))
        else:
            hybrid_candidates = dense_candidates.copy()
            hybrid_candidates["hybrid_score"] = hybrid_candidates["dense_score"]
            hybrid_candidates["dense_rank"] = hybrid_candidates["dense_rank"].astype(float)
            hybrid_candidates["bm25_score"] = 0.0
            hybrid_candidates["bm25_rank"] = np.nan

        ranked_bugs = hybrid_candidates.sort_values(["hybrid_score", "dense_score", BUG_ID_COLUMN], ascending=[False, False, True]).reset_index(drop=True)
        candidate_links = master[[BUG_ID_COLUMN, TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN]].drop_duplicates()
        testcase_candidates = ranked_bugs.merge(candidate_links, on=BUG_ID_COLUMN, how="inner")
        if testcase_candidates.empty:
            predicted_ids: list[int] = []
            active_rows = pd.DataFrame(columns=[TESTCASE_ID_COLUMN, TESTCASE_TITLE_COLUMN, "final_score"])
        else:
            aggregated = build_candidate_score(testcase_candidates, config.aggregation_weights)
            active_rows = aggregated.copy()
            if config.use_reranker:
                active_rows = rerank_candidates(query_text, active_rows, reranker, config.reranker_weight)
            predicted_ids = active_rows[TESTCASE_ID_COLUMN].dropna().astype(int).tolist()[:10]

        metrics = compute_metrics(predicted_ids, ground_truth_ids)
        for metric_name, metric_value in metrics.items():
            totals[metric_name] += float(metric_value)

        evaluation_rows.append(
            {
                BUG_ID_COLUMN: bug_id,
                "ground_truth_testcases": safe_json_dumps(sorted(ground_truth_ids)),
                "predicted_testcases": safe_json_dumps(predicted_ids),
                "ranking_method": "hybrid+reranked" if config.use_bm25 and config.use_reranker else ("hybrid" if config.use_bm25 else ("reranked" if config.use_reranker else "dense")),
                "hybrid_score_top": float(active_rows["final_score"].iloc[0]) if not active_rows.empty else 0.0,
                "experiment_set": config.experiment_set,
            }
        )

    if evaluated == 0:
        raise RuntimeError("No evaluation rows were produced for this configuration.")

    averages = {metric: value / evaluated for metric, value in totals.items()}
    parameters = asdict(config)
    return averages, parameters, evaluation_rows


def update_best(best: dict[str, Any] | None, config: ExperimentConfig, metrics: dict[str, float], experiment_name: str) -> dict[str, Any]:
    candidate = {
        "experiment_name": experiment_name,
        "config": asdict(config),
        "metrics": metrics,
    }
    if best is None:
        return candidate
    if metrics["Recall@1"] > best["metrics"]["Recall@1"]:
        return candidate
    if metrics["Recall@1"] == best["metrics"]["Recall@1"] and metrics["Recall@5"] > best["metrics"]["Recall@5"]:
        return candidate
    if metrics["Recall@1"] == best["metrics"]["Recall@1"] and metrics["Recall@5"] == best["metrics"]["Recall@5"] and metrics["MRR"] > best["metrics"]["MRR"]:
        return candidate
    return best


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting overnight experiment pipeline")
    logger.info("Git branch: %s", _read_git_branch())

    log_tee = TeeLogger(args.log_path, logger)
    start_time = time.perf_counter()

    try:
        master = load_master_dataset(args.master)
        experiment_grid = build_experiment_grid()
        logger.info("Generated %d candidate experiments", len(experiment_grid))

        query_model, _ = build_query_model(args.model_name)
        shared_reranker: Reranker | None = None
        if any(config.use_reranker for config in experiment_grid):
            try:
                shared_reranker = Reranker(model_name=args.reranker_model)
            except Exception as exc:  # pragma: no cover - depends on model availability
                logger.error("Shared reranker initialization failed: %s", exc)

        outcomes: list[ExperimentOutcome] = []
        best: dict[str, Any] | None = None
        stale_experiments = 0
        skipped_configs = 0

        for experiment_number, config in enumerate(experiment_grid, start=1):
            if stale_experiments >= args.max_stale_experiments:
                skipped_configs = len(experiment_grid) - experiment_number + 1
                logger.info("Early stopping triggered after %d stale experiments; skipping %d remaining configs", stale_experiments, skipped_configs)
                break

            experiment_name = f"{config.experiment_set}_dense{config.top_k_dense}_bm25{config.top_k_bm25}_agg{config.aggregation_weights['similarity']:.1f}-{config.aggregation_weights['support']:.1f}-{config.aggregation_weights['rank']:.1f}_rw{config.reranker_weight:.2f}"
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("Experiment %d/%d: %s", experiment_number, len(experiment_grid), experiment_name)
            logger.info("Config: %s", config.to_json())

            experiment_start = time.perf_counter()
            embeddings_path, metadata_path = ensure_embeddings_artifacts(
                master=master,
                artifacts_dir=args.artifacts_dir,
                model_name=args.model_name,
                use_normalized_titles=config.use_normalized_titles,
                use_metadata_tokens=config.use_metadata_tokens,
                logger=logger,
            )
            embeddings = load_embeddings(embeddings_path)
            metadata = pd.read_csv(metadata_path)
            index = build_index(embeddings)
            bm25_index = build_bm25_index(metadata)

            metrics, parameters, evaluation_rows = evaluate_config(
                config=config,
                master=master,
                embeddings=embeddings,
                metadata=metadata,
                index=index,
                bm25_index=bm25_index,
                reranker=shared_reranker if config.use_reranker else None,
                query_model=query_model,
                logger=logger,
            )
            runtime_seconds = time.perf_counter() - experiment_start

            outcome = ExperimentOutcome(
                experiment_name=experiment_name,
                timestamp=timestamp,
                parameters=parameters,
                metrics=metrics,
                runtime_seconds=runtime_seconds,
            )
            outcomes.append(outcome)

            best_before = best
            best = update_best(best, config, metrics, experiment_name)
            improved = best_before is None or best != best_before
            stale_experiments = 0 if improved else stale_experiments + 1

            log_tee.write(
                f"experiment={experiment_number} name={experiment_name} runtime={runtime_seconds:.2f}s metrics={safe_json_dumps(metrics)} best_so_far={safe_json_dumps(best)}\n"
            )
            logger.info("Best so far: %s", safe_json_dumps(best))

        if outcomes:
            rows = [
                {
                    "experiment_name": outcome.experiment_name,
                    "timestamp": outcome.timestamp,
                    "parameters": safe_json_dumps(outcome.parameters),
                    "metrics": safe_json_dumps(outcome.metrics),
                    "runtime_seconds": outcome.runtime_seconds,
                }
                for outcome in outcomes
            ]
            pd.DataFrame(rows).to_csv(args.results, index=False)
        else:
            pd.DataFrame(columns=["experiment_name", "timestamp", "parameters", "metrics", "runtime_seconds"]).to_csv(args.results, index=False)

        if best is None:
            raise RuntimeError("No experiments completed successfully.")

        args.best_config.parent.mkdir(parents=True, exist_ok=True)
        with args.best_config.open("w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2, sort_keys=True)

        ranked = sorted(outcomes, key=lambda item: (item.metrics["Recall@1"], item.metrics["Recall@5"], item.metrics["MRR"]), reverse=True)
        top_10 = ranked[:10]
        summary_lines = [
            f"Total experiments completed: {len(outcomes)}",
            f"Skipped configs after early stopping: {skipped_configs}",
            "",
            "Top 10 experiments ranked by Recall@1:",
        ]
        for rank, outcome in enumerate(top_10, start=1):
            summary_lines.append(
                f"{rank}. {outcome.experiment_name} | Recall@1={outcome.metrics['Recall@1']:.4f} | Recall@5={outcome.metrics['Recall@5']:.4f} | MRR={outcome.metrics['MRR']:.4f} | runtime={outcome.runtime_seconds:.2f}s"
            )
        summary_lines.extend(
            [
                "",
                "Recommend next actions:",
                "- Inspect the top Recall@1 and MRR configurations for stability across metric tradeoffs.",
                "- Re-run the best-performing configuration on a larger holdout slice if available.",
                "- Review failure_analysis_report.txt for the most common miss patterns before changing the model stack.",
            ]
        )
        with args.summary_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(summary_lines))

        logger.info("Saved experiment results to %s", args.results)
        logger.info("Saved best config to %s", args.best_config)
        logger.info("Saved summary to %s", args.summary_path)
        logger.info("Total runtime seconds: %.2f", time.perf_counter() - start_time)
    finally:
        log_tee.close()


if __name__ == "__main__":
    main()
