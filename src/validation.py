from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _null_percentages(df: pd.DataFrame) -> Dict[str, float]:
    return {
        column: round(float(df[column].isna().mean() * 100), 2)
        for column in df.columns
    }


def _duplicate_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df.duplicated(subset=[column]).sum())


def _empty_title_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().eq("").sum())


def _missing_description_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().eq("").sum())


def build_validation_report(
    bugs: pd.DataFrame,
    testcases: pd.DataFrame,
    master: pd.DataFrame,
) -> str:
    sections = []

    def add_section(title: str, lines: Iterable[str]) -> None:
        sections.append(title)
        sections.extend(list(lines))
        sections.append("")

    add_section(
        "BUG VALIDATION",
        [
            f"rows: {len(bugs)}",
            f"duplicate bug_id count: {_duplicate_count(bugs, 'bug_id')}",
            f"empty bug_title count: {_empty_title_count(bugs, 'bug_title')}",
            f"missing bug_description count: {_missing_description_count(bugs, 'bug_description')}",
            f"null percentages: {_null_percentages(bugs)}",
        ],
    )
    add_section(
        "TEST CASE VALIDATION",
        [
            f"rows: {len(testcases)}",
            f"duplicate testcase_id count: {_duplicate_count(testcases, 'testcase_id')}",
            f"empty testcase_title count: {_empty_title_count(testcases, 'testcase_title')}",
            f"missing testcase_description count: {_missing_description_count(testcases, 'testcase_description')}",
            f"null percentages: {_null_percentages(testcases)}",
        ],
    )
    add_section(
        "MASTER DATASET VALIDATION",
        [
            f"rows: {len(master)}",
            f"duplicate bug_id count: {_duplicate_count(master, 'bug_id')}",
            f"duplicate testcase_id count: {_duplicate_count(master, 'testcase_id')}",
            f"empty bug_title count: {_empty_title_count(master, 'bug_title')}",
            f"empty testcase_title count: {_empty_title_count(master, 'testcase_title')}",
            f"missing bug_description count: {_missing_description_count(master, 'bug_description')}",
            f"missing testcase_description count: {_missing_description_count(master, 'testcase_description')}",
            f"null percentages: {_null_percentages(master)}",
        ],
    )
    return "\n".join(sections).strip() + "\n"


def write_validation_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")