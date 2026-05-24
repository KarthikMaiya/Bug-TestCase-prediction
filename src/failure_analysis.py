from __future__ import annotations

import argparse
import ast
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import setup_logging


DEFAULT_EVALUATION_PATH = Path("evaluation_results.csv")
DEFAULT_REPORT_PATH = Path("failure_analysis_report.txt")

BUG_ID_COLUMN = "bug_id"
GROUND_TRUTH_COLUMN = "ground_truth_testcases"
PREDICTED_COLUMN = "predicted_testcases"
HIT_AT_1_COLUMN = "hit@1"
HIT_AT_5_COLUMN = "hit@5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze retrieval evaluation failures")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=DEFAULT_EVALUATION_PATH,
        help="Path to evaluation_results.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write the diagnostic report",
    )
    return parser


def load_evaluation_results(evaluation_path: Path) -> pd.DataFrame:
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {evaluation_path}")
    frame = pd.read_csv(evaluation_path)
    required_columns = [BUG_ID_COLUMN, GROUND_TRUTH_COLUMN, PREDICTED_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"evaluation_results.csv is missing required columns: {missing_columns}")
    return frame


def parse_id_list(value: object) -> list[int]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.split(",") if part.strip()]

    if isinstance(parsed, list):
        return [int(item) for item in parsed if str(item).strip()]
    if isinstance(parsed, tuple):
        return [int(item) for item in parsed if str(item).strip()]
    return [int(parsed)]


def format_list_preview(values: Iterable[int], limit: int = 20) -> str:
    items = list(values)
    if not items:
        return "[]"
    preview = items[:limit]
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    return f"{preview}{suffix}"


def classify_failure(ground_truth: list[int]) -> list[str]:
    causes = ["A. Semantic retrieval miss"]
    if len(ground_truth) > 1:
        causes.append("B. Mapping ambiguity")
    return causes


def most_common_patterns(counter: Counter[int], limit: int = 10) -> list[tuple[int, int]]:
    return counter.most_common(limit)


def write_report(
    report_path: Path,
    total_rows: int,
    failed_rows: pd.DataFrame,
    failed_row_records: list[dict[str, object]],
    gt_counter: Counter[int],
    predicted_counter: Counter[int],
    multi_gt_count: int,
    logger: logging.Logger,
) -> None:
    failure_count = len(failed_rows)
    failure_percentage = (failure_count / total_rows * 100.0) if total_rows else 0.0
    multi_gt_percentage = (multi_gt_count / failure_count * 100.0) if failure_count else 0.0

    lines: list[str] = []
    lines.append("Phase 5.5 — Failure Analysis")
    lines.append("")
    lines.append(f"Total rows analyzed: {total_rows}")
    lines.append(f"Failure count (hit@10 == 0): {failure_count}")
    lines.append(f"Failure percentage: {failure_percentage:.2f}%")
    lines.append(f"Failures with multiple ground-truth testcase_ids: {multi_gt_count} ({multi_gt_percentage:.2f}%)")
    lines.append("")

    lines.append("Top findings")
    lines.append("- All failures are semantic retrieval misses by definition: none of the ground-truth testcase_ids appear in the top-10 predictions.")
    lines.append(f"- {multi_gt_count} failed bugs have multiple ground-truth testcase_ids, which is a proxy for mapping ambiguity.")
    lines.append("- The current CSV does not contain bug titles, descriptions, or testcase text, so duplicate/generic titles, missing bug_description, and noisy testcase titles cannot be measured directly here.")
    lines.append("")

    lines.append("Representative failed rows")
    sample_rows = failed_rows.head(20)
    if sample_rows.empty:
        lines.append("- No failed rows found.")
    else:
        for _, row in sample_rows.iterrows():
            lines.append(
                f"- bug_id={int(row[BUG_ID_COLUMN])} | ground_truth_testcases={row[GROUND_TRUTH_COLUMN]} | predicted_testcases={row[PREDICTED_COLUMN]}"
            )
    lines.append("")

    lines.append("Summary statistics")
    lines.append(f"- % failures with multiple ground-truth testcase_ids: {multi_gt_percentage:.2f}%")
    lines.append("- Most common failed testcase_ids (from ground truth):")
    for testcase_id, count in most_common_patterns(gt_counter, 10):
        lines.append(f"  - testcase_id {testcase_id}: {count}")
    lines.append("- Most common predicted testcase_ids among failures:")
    for testcase_id, count in most_common_patterns(predicted_counter, 10):
        lines.append(f"  - testcase_id {testcase_id}: {count}")
    lines.append("- Most common failed bug title patterns: N/A in evaluation_results.csv")
    lines.append("- % failures with missing bug_description: N/A in evaluation_results.csv")
    lines.append("- Noisy testcase titles: N/A in evaluation_results.csv")
    lines.append("")

    lines.append("Likely failure causes")
    lines.append("- A. Semantic retrieval miss: high confidence; applies to every failed row.")
    lines.append("- B. Mapping ambiguity: moderate confidence when a failed bug has multiple ground-truth testcase_ids.")
    lines.append("- C. Duplicate / generic bug titles: not observable from evaluation_results.csv alone.")
    lines.append("- D. Missing bug_description / sparse metadata: not observable from evaluation_results.csv alone.")
    lines.append("- E. Noisy testcase titles: not observable from evaluation_results.csv alone.")
    lines.append("")

    lines.append("Representative examples for next review")
    for record in failed_row_records[:10]:
        lines.append(
            f"- bug_id={record[BUG_ID_COLUMN]} | gt={record[GROUND_TRUTH_COLUMN]} | pred={record[PREDICTED_COLUMN]} | causes={', '.join(record['causes'])}"
        )
    lines.append("")

    lines.append("Recommendations")
    lines.append("- Increase candidate recall before aggregation, for example by retrieving a larger bug pool or adding a second semantic pass.")
    lines.append("- Add richer bug metadata into the query representation to reduce pure retrieval misses.")
    lines.append("- If the evaluation CSV is regenerated with text columns, rerun this analysis to quantify title duplication, sparse descriptions, and testcase title noise.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved diagnostic report to %s", report_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()

    logger.info("Loading evaluation results from %s", args.evaluation)
    results = load_evaluation_results(args.evaluation)

    results = results.copy()
    results["ground_truth_list"] = results[GROUND_TRUTH_COLUMN].apply(parse_id_list)
    results["predicted_list"] = results[PREDICTED_COLUMN].apply(parse_id_list)
    results["computed_hit@10"] = results.apply(
        lambda row: int(any(testcase_id in set(row["ground_truth_list"]) for testcase_id in row["predicted_list"][:10])),
        axis=1,
    )

    failed_rows = results[results["computed_hit@10"] == 0].copy()
    total_rows = len(results)
    failure_count = len(failed_rows)
    failure_percentage = (failure_count / total_rows * 100.0) if total_rows else 0.0

    logger.info("Failure count (hit@10 == 0): %d", failure_count)
    logger.info("Failure percentage: %.2f%%", failure_percentage)

    print(f"Total failure count: {failure_count}")
    print(f"Failure percentage: {failure_percentage:.2f}%")
    print("Sample 20 failed rows:")
    if failed_rows.empty:
        print("<none>")
    else:
        sample = failed_rows.head(20)[[BUG_ID_COLUMN, GROUND_TRUTH_COLUMN, PREDICTED_COLUMN]]
        print(sample.to_string(index=False))

    failed_row_records: list[dict[str, object]] = []
    gt_counter: Counter[int] = Counter()
    predicted_counter: Counter[int] = Counter()
    multi_gt_count = 0

    for _, row in failed_rows.iterrows():
        ground_truth_list = row["ground_truth_list"]
        predicted_list = row["predicted_list"]
        gt_counter.update(ground_truth_list)
        predicted_counter.update(predicted_list)
        if len(ground_truth_list) > 1:
            multi_gt_count += 1
        failed_row_records.append(
            {
                BUG_ID_COLUMN: int(row[BUG_ID_COLUMN]),
                GROUND_TRUTH_COLUMN: row[GROUND_TRUTH_COLUMN],
                PREDICTED_COLUMN: row[PREDICTED_COLUMN],
                "causes": classify_failure(ground_truth_list),
            }
        )

    print("Most common failed testcase_ids:")
    if gt_counter:
        for testcase_id, count in gt_counter.most_common(10):
            print(f"{testcase_id}: {count}")
    else:
        print("<none>")

    write_report(
        report_path=args.report,
        total_rows=total_rows,
        failed_rows=failed_rows,
        failed_row_records=failed_row_records,
        gt_counter=gt_counter,
        predicted_counter=predicted_counter,
        multi_gt_count=multi_gt_count,
        logger=logger,
    )

    logger.info("Total runtime seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()