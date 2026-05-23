from __future__ import annotations

import difflib
import logging
from typing import List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth

from config import load_config, setup_logging

PROJECTS_API_VERSIONS = ("7.1", "7.0", "6.0")


def build_projects_url(org: str) -> str:
    return f"https://dev.azure.com/{org}/_apis/projects"


def _response_body_preview(response: requests.Response, limit: int = 2000) -> str:
    body = response.text.strip()
    if len(body) > limit:
        return body[:limit] + "..."
    return body


def get_project_names(
    session: requests.Session,
    org: str,
    api_version: str,
    logger: logging.Logger,
) -> Tuple[List[str], str]:
    url = build_projects_url(org)
    params = {
        "api-version": api_version,
        "$top": 1000,
    }
    logger.info("Testing projects endpoint: %s", url)
    logger.info("Using api-version=%s", api_version)

    try:
        response = session.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        logger.error("Network failure while calling %s: %s", url, exc)
        raise

    request_url = response.url
    logger.info("Projects request URL: %s", request_url)
    logger.info("Projects response status code: %s", response.status_code)

    if response.status_code == 401:
        logger.error("Response body: %s", _response_body_preview(response))
        raise PermissionError("Azure DevOps returned 401 Unauthorized. Check AZURE_PAT.")
    if response.status_code == 403:
        logger.error("Response body: %s", _response_body_preview(response))
        raise PermissionError("Azure DevOps returned 403 Forbidden. The PAT may not have project listing permissions.")
    if response.status_code == 404:
        logger.error("Response body: %s", _response_body_preview(response))
        raise FileNotFoundError("Azure DevOps returned 404 Not Found for the projects endpoint. Check the organization name.")
    if response.status_code >= 400:
        logger.error("Response body: %s", _response_body_preview(response))
        raise RuntimeError(
            f"Unexpected Azure DevOps response {response.status_code} for {request_url}"
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Unexpected Azure DevOps response %s: %s", response.status_code, _response_body_preview(response))
        raise RuntimeError(f"Failed to list projects: {exc}") from exc

    payload = response.json()
    values = payload.get("value", []) if isinstance(payload, dict) else []
    project_names = [item.get("name", "") for item in values if isinstance(item, dict) and item.get("name")]
    return project_names, request_url


def print_project_names(project_names: List[str]) -> None:
    print("Available projects:")
    for name in project_names:
        print(f"- {name}")


def verify_project_exists(project_name: str, project_names: List[str]) -> Optional[List[str]]:
    normalized = project_name.strip().lower()
    matches = [name for name in project_names if name.strip().lower() == normalized]
    if matches:
        return matches
    return None


def suggest_close_matches(project_name: str, project_names: List[str], limit: int = 5) -> List[str]:
    return difflib.get_close_matches(project_name, project_names, n=limit, cutoff=0.4)


def main() -> None:
    logger = setup_logging()
    config = load_config(logger=logger)

    print(f"org: {config.org}")
    print(f"project: {config.project}")
    print(f"api_version: {config.api_version}")

    session = requests.Session()
    session.auth = HTTPBasicAuth("", config.pat)
    session.headers.update({"Accept": "application/json"})

    try:
        project_names: List[str] = []
        used_url = ""
        last_error: Optional[Exception] = None

        for api_version in PROJECTS_API_VERSIONS:
            logger.info("Attempting projects endpoint with api-version=%s", api_version)
            try:
                project_names, used_url = get_project_names(session, config.org, api_version, logger)
                logger.info("Projects endpoint succeeded with api-version=%s", api_version)
                break
            except (PermissionError, FileNotFoundError):
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Projects endpoint failed with api-version=%s: %s", api_version, exc)
        else:
            if last_error is not None:
                raise last_error
    except PermissionError as exc:
        logger.error(str(exc))
        return
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    except requests.RequestException:
        logger.exception("Network failure during connectivity diagnostic")
        return
    except Exception:
        logger.exception("Unexpected failure while testing Azure DevOps connectivity")
        return
    finally:
        session.close()

    logger.info("Projects request URL used: %s", used_url)
    print_project_names(project_names)

    if verify_project_exists(config.project, project_names):
        logger.info("Configured project exists: %s", config.project)
        print(f"Configured project found: {config.project}")
        return

    logger.warning("Configured project not found: %s", config.project)
    print(f"Configured project not found: {config.project}")

    matches = suggest_close_matches(config.project, project_names)
    if matches:
        print("Closest matching projects:")
        for match in matches:
            print(f"- {match}")
    else:
        print("No close project name matches found.")


if __name__ == "__main__":
    main()
