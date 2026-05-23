from __future__ import annotations

import argparse
import html
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import setup_logging
from validation import build_validation_report, write_validation_report

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

BUG_COLUMNS = [
    "bug_id",
    "bug_title",
    "bug_description",
    "bug_tags",
    "priority",
    "severity",
    "area_path",
    "iteration_path",
    "state",
]

TESTCASE_COLUMNS = [
    "testcase_id",
    "testcase_title",
    "testcase_description",
    "test_steps",
    "expected_results",
    "preconditions",
]

MASTER_COLUMNS = [
    "bug_id",
    "bug_title",
    "bug_description",
    "bug_tags",
    "priority",
    "severity",
    "testcase_id",
    "testcase_title",
    "testcase_description",
    "test_steps",
]

MAPPING_CANDIDATE_PATHS = [
    Path("data/incident_data.csv"),
    Path("data/Incident data.csv"),
    Path("Incident data.csv"),
]


def normalize_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def preview_dataframe(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label} preview (first 5 rows):")
    if df.empty:
        print("<empty>")
        return
    print(df.head(5).to_string(index=False))


def find_mapping_source() -> Optional[Path]:
    for candidate in MAPPING_CANDIDATE_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_mapping_data(mapping_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading mapping data from %s", mapping_path)
    encodings = ["utf-8","utf-8-sig","cp1252","latin1"]

    for enc in encodings:
        try:
            mapping = pd.read_csv(mapping_path, encoding=enc)
            logger.info(f"Loaded mapping using encoding={enc}")
            break
        except UnicodeDecodeError:
            logger.warning(f"Failed encoding={enc}")
    else:
        raise ValueError("Unable to decode mapping CSV.")

    required_columns = {"WorkItemId", "LinkedWorkItemId", "LinkedType"}
    missing_columns = [column for column in required_columns if column not in mapping.columns]
    if missing_columns:
        raise ValueError(
            f"Mapping source is missing required columns: {missing_columns}"
        )

    mapping = mapping[["WorkItemId", "LinkedWorkItemId", "LinkedType"]].copy()
    mapping = mapping.dropna(subset=["WorkItemId", "LinkedWorkItemId"])
    mapping["WorkItemId"] = pd.to_numeric(mapping["WorkItemId"], errors="coerce")
    mapping["LinkedWorkItemId"] = pd.to_numeric(mapping["LinkedWorkItemId"], errors="coerce")
    mapping = mapping.dropna(subset=["WorkItemId", "LinkedWorkItemId"])
    mapping["WorkItemId"] = mapping["WorkItemId"].astype(int)
    mapping["LinkedWorkItemId"] = mapping["LinkedWorkItemId"].astype(int)
    return mapping


def build_real_master_dataset(
    bugs: pd.DataFrame,
    testcases: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    mapping_path = find_mapping_source()
    if mapping_path is None:
        logger.warning("No real bug↔testcase mapping source available.")
        raise ValueError("No real bug↔testcase mapping source available.")

    mapping = load_mapping_data(mapping_path, logger)

    bug_ids = set(pd.to_numeric(bugs["bug_id"], errors="coerce").dropna().astype(int))
    testcase_ids = set(pd.to_numeric(testcases["testcase_id"], errors="coerce").dropna().astype(int))

    bug_to_testcase = mapping[
        mapping["WorkItemId"].isin(bug_ids) & mapping["LinkedWorkItemId"].isin(testcase_ids)
    ].copy()
    testcase_to_bug = mapping[
        mapping["WorkItemId"].isin(testcase_ids) & mapping["LinkedWorkItemId"].isin(bug_ids)
    ].copy()

    if bug_to_testcase.empty and testcase_to_bug.empty:
        logger.warning("No real bug↔testcase mapping source available.")
        raise ValueError("No real bug↔testcase mapping source available.")

    bug_to_testcase = bug_to_testcase.rename(
        columns={"WorkItemId": "bug_id", "LinkedWorkItemId": "testcase_id", "LinkedType": "link_type"}
    )
    testcase_to_bug = testcase_to_bug.rename(
        columns={"LinkedWorkItemId": "bug_id", "WorkItemId": "testcase_id", "LinkedType": "link_type"}
    )

    mappings = pd.concat([bug_to_testcase, testcase_to_bug], ignore_index=True)
    mappings = mappings[["bug_id", "testcase_id", "link_type"]].drop_duplicates()

    logger.info("Real mapping rows found: %s", len(mappings))

    master = mappings.merge(bugs, on="bug_id", how="inner").merge(testcases, on="testcase_id", how="inner")
    master = master[MASTER_COLUMNS].drop_duplicates().reset_index(drop=True)
    master = master.dropna(subset=["bug_id", "testcase_id"], how="any")
    return master


def clean_dataframe(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_text)
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge bugs and test cases into a master dataset")
    parser.add_argument(
        "--bugs",
        type=Path,
        default=Path("data/bugs.csv"),
        help="Input bug CSV path",
    )
    parser.add_argument(
        "--testcases",
        type=Path,
        default=Path("data/testcases.csv"),
        help="Input test case CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/master_dataset.csv"),
        help="Output master dataset CSV path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()
    logger.info("Loading bug data from %s", args.bugs)
    bugs = pd.read_csv(args.bugs)
    logger.info("Loading test case data from %s", args.testcases)
    testcases = pd.read_csv(args.testcases)

    missing_bug_columns = [column for column in BUG_COLUMNS if column not in bugs.columns]
    missing_testcase_columns = [column for column in TESTCASE_COLUMNS if column not in testcases.columns]
    if missing_bug_columns:
        raise ValueError(f"Bug dataset is missing required columns: {missing_bug_columns}")
    if missing_testcase_columns:
        raise ValueError(f"Test case dataset is missing required columns: {missing_testcase_columns}")

    bugs = clean_dataframe(bugs, ["bug_title", "bug_description", "bug_tags", "priority", "severity"])
    testcases = clean_dataframe(
        testcases,
        ["testcase_title", "testcase_description", "test_steps", "expected_results", "preconditions"],
    )

    master = build_real_master_dataset(bugs, testcases, logger)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)

    validation_report = build_validation_report(bugs, testcases, master)
    write_validation_report(validation_report, args.output.parent / "validation_report.txt")

    missing_counts = {
        column: int(master[column].isna().sum() + (master[column] == "").sum())
        for column in MASTER_COLUMNS
        if column in master.columns
    }
    logger.info("Master dataset rows: %s", len(master))
    logger.info("Master dataset missing field counts: %s", missing_counts)
    logger.info("Saved master dataset to %s", args.output)
    preview_dataframe(master, "Master dataset")
    logger.info("Extraction duration seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
