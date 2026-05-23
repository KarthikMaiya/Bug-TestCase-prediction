from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

LOGGER_NAME = "testcasepred"
DEFAULT_API_VERSION = "7.1"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_PAGE_SIZE = 100
REQUIRED_ENV_VARS = ("AZURE_PAT", "AZURE_ORG", "AZURE_PROJECT")
ENV_ALIASES = {
    "AZURE_ORG": ("AZURE_DEVOPS_ORG",),
    "AZURE_PROJECT": ("AZURE_DEVOPS_PROJECT",),
}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a consistent application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@dataclass(frozen=True)
class AzureConfig:
    pat: str
    org: str
    project: str
    api_version: str = DEFAULT_API_VERSION
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    page_size: int = DEFAULT_PAGE_SIZE

    @property
    def base_url(self) -> str:
        return f"https://dev.azure.com/{self.org}/{self.project}/_apis"


def _load_dotenv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def get_env_value(name: str, required: bool = True) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        if required:
            raise ValueError(f"Missing required environment variable: {name}")
        return None
    stripped = value.strip()
    if required and not stripped:
        raise ValueError(f"Environment variable {name} is empty")
    return stripped or None


def validate_environment(logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    logger = logger or setup_logging()
    _load_dotenv()

    resolved: Dict[str, str] = {}
    missing: list[str] = []

    for name in REQUIRED_ENV_VARS:
        value = get_env_value(name, required=False)
        if value:
            resolved[name] = value
            continue

        for alias in ENV_ALIASES.get(name, ()):
            alias_value = get_env_value(alias, required=False)
            if alias_value:
                logger.warning(
                    "Environment variable %s is missing; using legacy alias %s",
                    name,
                    alias,
                )
                resolved[name] = alias_value
                os.environ[name] = alias_value
                break
        else:
            missing.append(name)

    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return resolved


def load_config(logger: Optional[logging.Logger] = None) -> AzureConfig:
    logger = logger or setup_logging()
    values = validate_environment(logger=logger)
    _load_dotenv()
    return AzureConfig(
        pat=values["AZURE_PAT"],
        org=values["AZURE_ORG"],
        project=values["AZURE_PROJECT"],
    )
