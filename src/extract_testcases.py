from __future__ import annotations

import argparse
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from azure_client import AzureDevOpsClient, AzureDevOpsError
from config import load_config, setup_logging

TESTCASE_REQUIRED_FIELDS = ["System.Id", "System.Title"]
TESTCASE_OPTIONAL_FIELDS = [
    "System.Description",
    "Microsoft.VSTS.TCM.Steps",
    "Microsoft.VSTS.TCM.Parameters",
    "Microsoft.VSTS.TCM.Preconditions",
]
TESTCASE_FIELDS = TESTCASE_REQUIRED_FIELDS + TESTCASE_OPTIONAL_FIELDS

OUTPUT_COLUMNS = [
    "testcase_id",
    "testcase_title",
    "testcase_description",
    "test_steps",
    "expected_results",
    "preconditions",
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


def safe_field(fields: Dict[str, object], field_name: str) -> Optional[str]:
    return normalize_text(fields.get(field_name))


def preview_dataframe(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label} preview (first 5 rows):")
    if df.empty:
        print("<empty>")
        return
    print(df.head(5).to_string(index=False))


def _is_bad_request_error(error: Exception) -> bool:
    message = str(error).lower()
    return "status 400" in message or "400 bad request" in message or "bad request" in message


def _detect_unsupported_optional_field(error: Exception, fields: List[str]) -> Optional[str]:
    message = str(error).lower()
    for field_name in fields:
        if field_name.lower() in message:
            return field_name
    return None


def _resolve_supported_testcase_fields(
    client: AzureDevOpsClient,
    logger: logging.Logger,
    probe_ids: List[int],
) -> List[str]:
    supported_fields = list(TESTCASE_FIELDS)
    unsupported_fields: List[str] = []

    if not probe_ids:
        logger.info("Supported testcase fields: %s", supported_fields)
        logger.info("Unsupported testcase fields removed: %s", unsupported_fields)
        return supported_fields

    while True:
        try:
            logger.info(
                "Probing testcase fields before batch request: ids_count=%s fields=%s",
                len(probe_ids),
                supported_fields,
            )
            client.get_work_items(probe_ids, fields=supported_fields)
            break
        except AzureDevOpsError as error:
            if not _is_bad_request_error(error):
                raise

            unsupported_field = _detect_unsupported_optional_field(error, supported_fields)
            if unsupported_field is None:
                for candidate in reversed(TESTCASE_OPTIONAL_FIELDS):
                    if candidate in supported_fields:
                        unsupported_field = candidate
                        break

            if unsupported_field is None or unsupported_field in TESTCASE_REQUIRED_FIELDS:
                raise

            logger.warning(
                "Removing unsupported testcase field after Azure 400: %s",
                unsupported_field,
            )
            supported_fields.remove(unsupported_field)
            unsupported_fields.append(unsupported_field)

    logger.info("Supported testcase fields: %s", supported_fields)
    logger.info("Unsupported testcase fields removed: %s", unsupported_fields)
    return supported_fields


def strip_namespaces(xml_text: str) -> str:
    return re.sub(r"xmlns(:\w+)?=\"[^\"]*\"", "", xml_text)


def collect_element_text(element: ET.Element) -> str:
    text_parts: List[str] = []
    for node in element.iter():
        if node.text:
            cleaned = normalize_text(node.text)
            if cleaned:
                text_parts.append(cleaned)
    return " ".join(text_parts).strip()


def parse_test_steps(step_xml: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not step_xml:
        return None, None

    xml_text = html.unescape(str(step_xml)).strip()
    if not xml_text:
        return None, None

    try:
        root = ET.fromstring(strip_namespaces(xml_text))
    except ET.ParseError:
        cleaned = normalize_text(xml_text)
        return cleaned, None

    action_segments: List[str] = []
    expected_segments: List[str] = []

    for step_index, step in enumerate(root.findall(".//step"), start=1):
        values = [collect_element_text(child) for child in list(step)]
        values = [value for value in values if value]
        if not values:
            continue
        action = values[0]
        expected = values[1] if len(values) > 1 else None
        action_segments.append(f"{step_index}. {action}")
        if expected:
            expected_segments.append(f"{step_index}. {expected}")

    if not action_segments:
        fallback = normalize_text(root.text or xml_text)
        return fallback, None

    return "\n".join(action_segments), "\n".join(expected_segments) if expected_segments else None


def extract_testcase_rows(client: AzureDevOpsClient, logger: logging.Logger) -> List[Dict[str, object]]:
    wiql = (
        "SELECT [System.Id] FROM WorkItems "
        "WHERE [System.TeamProject] = @project "
        "AND [System.WorkItemType] = 'Test Case' "
        "ORDER BY [System.Id]"
    )
    query_result = client.wiql(wiql)
    work_items = query_result.get("workItems", []) or []
    testcase_ids = [int(item["id"]) for item in work_items if item.get("id") is not None]

    logger.info("Found %s test case work items", len(testcase_ids))
    logger.info("WIQL query returned %s testcase IDs", len(testcase_ids))

    rows: List[Dict[str, object]] = []
    missing_counts = {column: 0 for column in OUTPUT_COLUMNS if column != "testcase_id"}

    if not testcase_ids:
        logger.warning("WIQL returned no test case IDs; writing empty test case dataset")
        return rows

    supported_fields = _resolve_supported_testcase_fields(
        client,
        logger,
        testcase_ids[:1],
    )

    for batch_index, batch_ids in enumerate(chunked(testcase_ids, client.config.page_size), start=1):
        logger.info(
            "Fetching test case batch %s with %s items (request_count=%s)",
            batch_index,
            len(batch_ids),
            client.request_count,
        )
        try:
            batch_items = client.get_work_items(batch_ids, fields=supported_fields)
        except Exception:
            logger.exception("Failed to fetch test case batch %s", batch_index)
            raise
        for item in batch_items:
            fields = item.get("fields", {}) or {}
            raw_steps = fields.get("Microsoft.VSTS.TCM.Steps")
            test_steps, expected_results = parse_test_steps(raw_steps)
            row = {
                "testcase_id": item.get("id") or fields.get("System.Id"),
                "testcase_title": safe_field(fields, "System.Title"),
                "testcase_description": safe_field(fields, "System.Description"),
                "test_steps": test_steps,
                "expected_results": expected_results,
                "preconditions": safe_field(fields, "Microsoft.VSTS.TCM.Preconditions"),
            }
            for column in missing_counts:
                if row.get(column) in (None, ""):
                    missing_counts[column] += 1
            rows.append(row)

        logger.info("Collected %s test case rows so far", len(rows))

    logger.info("Missing test case field counts: %s", missing_counts)
    return rows


def write_testcases_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.drop_duplicates(subset=["testcase_id"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Test Case work items from Azure DevOps")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/testcases.csv"),
        help="Output CSV path for test case data",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.perf_counter()
    config = load_config(logger=logger)

    logger.info("Starting test case extraction for org=%s project=%s", config.org, config.project)
    with AzureDevOpsClient(config, logger=logger) as client:
        rows = extract_testcase_rows(client, logger)

    write_testcases_csv(rows, args.output)
    testcases_df = pd.read_csv(args.output)
    preview_dataframe(testcases_df, "Test cases CSV")
    logger.info("Saved %s test case rows to %s", len(rows), args.output)
    logger.info("Azure API request count: %s", client.request_count if 'client' in locals() else 0)
    logger.info("Extraction duration seconds: %.2f", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
