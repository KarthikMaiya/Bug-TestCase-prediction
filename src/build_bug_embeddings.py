from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import setup_logging

DEFAULT_INPUT_PATH = Path("data/master_dataset.csv")
DEFAULT_EMBEDDINGS_PATH = Path("data/bug_embeddings.npy")
DEFAULT_METADATA_PATH = Path("data/bug_metadata.csv")
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"

BUG_ID_COLUMN = "bug_id"
BUG_TEXT_COLUMNS = ["bug_title", "bug_tags", "severity", "bug_description"]


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def build_bug_text(row: pd.Series) -> str:
    title = normalize_text(row.get("bug_title"))
    tags = normalize_text(row.get("bug_tags"))
    severity = normalize_text(row.get("severity"))
    description = normalize_text(row.get("bug_description"))

    return "\n".join(
        [
            f"Title: {title}" if title else "Title: ",
            f"Tags: {tags}" if tags else "Tags: ",
            f"Severity: {severity}" if severity else "Severity: ",
            f"Description: {description}" if description else "Description: ",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build bug embeddings from the master dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input master dataset CSV")
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Output NumPy array path for bug embeddings",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Output CSV path for deduplicated bug metadata",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name",
    )
    return parser


def load_master_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")
    return pd.read_csv(input_path)


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = [BUG_ID_COLUMN, *BUG_TEXT_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input dataset is missing required columns: {missing_columns}")


def deduplicate_bugs(df: pd.DataFrame) -> pd.DataFrame:
    bugs = df.copy()
    bugs[BUG_ID_COLUMN] = pd.to_numeric(bugs[BUG_ID_COLUMN], errors="coerce")
    bugs = bugs.dropna(subset=[BUG_ID_COLUMN])
    bugs[BUG_ID_COLUMN] = bugs[BUG_ID_COLUMN].astype(int)
    bugs = bugs.drop_duplicates(subset=[BUG_ID_COLUMN], keep="first").reset_index(drop=True)
    return bugs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    logger.info("Selected device: %s", device)
    if device == "cuda":
        logger.info("GPU name: %s", torch.cuda.get_device_name(0))

    logger.info("Loading master dataset from %s", args.input)
    df = load_master_dataset(args.input)
    logger.info("Rows loaded: %s", len(df))

    validate_columns(df)
    bugs = deduplicate_bugs(df)
    logger.info("Unique bugs: %s", len(bugs))

    bug_metadata = bugs[[BUG_ID_COLUMN, *BUG_TEXT_COLUMNS]].copy()
    bug_metadata[BUG_ID_COLUMN] = bug_metadata[BUG_ID_COLUMN].astype(int)
    bug_metadata["bug_text"] = bug_metadata.apply(build_bug_text, axis=1)

    model = SentenceTransformer(args.model_name, device=device)
    texts: List[str] = bug_metadata["bug_text"].tolist()
    logger.info("Total texts: %s", len(texts))
    logger.info("Batch size: %s", 32)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    logger.info("Embedding shape: %s", embeddings.shape)

    args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.embeddings_output, embeddings)
    bug_metadata.to_csv(args.metadata_output, index=False)

    logger.info("Saved embeddings to %s", args.embeddings_output)
    logger.info("Saved bug metadata to %s", args.metadata_output)
    logger.info("Runtime seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
