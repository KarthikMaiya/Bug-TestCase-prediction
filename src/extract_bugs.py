from __future__ import annotations

import argparse
import html
import logging
import re
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

from azure_client import AzureDevOpsClient
from config import load_config, setup_logging

BUG_FIELDS = [
    "System.Id",
    "System.Title",
    "System.Description",
    "System.Tags",
    "Microsoft.VSTS.Common.Priority",
    "Microsoft.VSTS.Common.Severity",
    "System.AreaPath",
    "System.IterationPath",
    "System.State",
]

OUTPUT_COLUMNS = [
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

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def chunked(values: List[int], size: int) -> Iterator[List[int]]:
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


def normalize_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def normalize_tags(value: object) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    tags = [part.strip() for part in text.split(";") if part.strip()]
    return "; ".join(tags) if tags else None


def safe_field(fields: Dict[str, object], field_name: str, *, tags: bool = False) -> Optional[str]:
    value = fields.get(field_name)
    if tags:
        return normalize_tags(value)
    return normalize_text(value)


def preview_dataframe(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label} preview (first 5 rows):")
    if df.empty:
        print("<empty>")
        return
    print(df.head(5).to_string(index=False))


def extract_bug_rows(client: AzureDevOpsClient, logger: logging.Logger) -> List[Dict[str, object]]:
    wiql = (
        "SELECT [System.Id] FROM WorkItems "
        "WHERE [System.TeamProject] = @project "
        "AND [System.WorkItemType] = 'Bug' "
        "ORDER BY [System.Id]"
    )
    query_result = client.wiql(wiql)
    work_items = query_result.get("workItems", []) or []
    bug_ids = [int(item["id"]) for item in work_items if item.get("id") is not None]

    logger.info("Found %s bug work items", len(bug_ids))
    logger.info("WIQL query returned %s bug IDs", len(bug_ids))

    rows: List[Dict[str, object]] = []
    missing_counts = {column: 0 for column in OUTPUT_COLUMNS if column != "bug_id"}

    if not bug_ids:
        logger.warning("WIQL returned no bug IDs; writing empty bug dataset")
        return rows

    for batch_index, batch_ids in enumerate(chunked(bug_ids, client.config.page_size), start=1):
        logger.info(
            "Fetching bug batch %s with %s items (request_count=%s)",
            batch_index,
            len(batch_ids),
            client.request_count,
        )
        try:
            batch_items = client.get_work_items(batch_ids, fields=BUG_FIELDS)
        except Exception:
            logger.exception("Failed to fetch bug batch %s", batch_index)
            raise
        for item in batch_items:
            fields = item.get("fields", {}) or {}
            row = {
                "bug_id": item.get("id") or fields.get("System.Id"),
                "bug_title": safe_field(fields, "System.Title"),
                "bug_description": safe_field(fields, "System.Description"),
                "bug_tags": safe_field(fields, "System.Tags", tags=True),
                "priority": safe_field(fields, "Microsoft.VSTS.Common.Priority"),
                "severity": safe_field(fields, "Microsoft.VSTS.Common.Severity"),
                "area_path": safe_field(fields, "System.AreaPath"),
                "iteration_path": safe_field(fields, "System.IterationPath"),
                "state": safe_field(fields, "System.State"),
            }
            for column in missing_counts:
                if row.get(column) in (None, ""):
                    missing_counts[column] += 1
            rows.append(row)

        logger.info("Collected %s bug rows so far", len(rows))

    logger.info("Missing bug field counts: %s", missing_counts)
    return rows


def write_bugs_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.drop_duplicates(subset=["bug_id"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Bug work items from Azure DevOps")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bugs.csv"),
        help="Output CSV path for bug data",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()
    config = load_config(logger=logger)

    logger.info("Starting bug extraction for org=%s project=%s", config.org, config.project)
    with AzureDevOpsClient(config, logger=logger) as client:
        rows = extract_bug_rows(client, logger)

    write_bugs_csv(rows, args.output)
    bugs_df = pd.read_csv(args.output)
    preview_dataframe(bugs_df, "Bugs CSV")
    logger.info("Saved %s bug rows to %s", len(rows), args.output)
    logger.info("Azure API request count: %s", client.request_count if 'client' in locals() else 0)
    logger.info("Extraction duration seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
