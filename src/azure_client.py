from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests
from requests import Response, Session
from requests.auth import HTTPBasicAuth

from config import AzureConfig, setup_logging

FALLBACK_API_VERSIONS = ("7.1", "7.0", "6.0")


class AzureDevOpsError(RuntimeError):
    """Raised when Azure DevOps returns an unrecoverable error."""


@dataclass
class AzureResponse:
    status_code: int
    payload: Any
    headers: Dict[str, str]


class AzureDevOpsClient:
    """Reusable Azure DevOps REST client with retries and pagination."""

    def __init__(self, config: AzureConfig, logger=None) -> None:
        self.config = config
        self.logger = logger or setup_logging()
        self.session = self._build_session(config.pat)
        self.request_count = 0

    @staticmethod
    def _build_session(pat: str) -> Session:
        session = requests.Session()
        session.auth = HTTPBasicAuth("", pat)
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        return session

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "AzureDevOpsClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        expected_status: Iterable[int] = (200,),
    ) -> AzureResponse:
        request_params = dict(params or {})
        endpoint = self._endpoint_name(url)
        requested_api_version = request_params.get("api-version")
        api_version_attempts = self._api_version_attempts(requested_api_version)

        if not api_version_attempts:
            response = self._request_with_transport_retries(
                method=method,
                url=url,
                params=request_params,
                json=json,
                endpoint=endpoint,
                api_version=None,
            )
            self.logger.info(
                "Azure request endpoint=%s api-version=%s status=%s request_count=%s fallback_attempt=1/1",
                endpoint,
                requested_api_version,
                response.status_code,
                self.request_count,
            )

            if response.status_code in expected_status:
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                return AzureResponse(
                    status_code=response.status_code,
                    payload=payload,
                    headers=dict(response.headers),
                )

            if response.status_code in {401, 403}:
                raise AzureDevOpsError(
                    "Unauthorized Azure DevOps response. Check that AZURE_PAT is valid and has access to the project."
                )

            raise AzureDevOpsError(self._build_error_message(response))

        fallback_index = 0

        while True:
            current_api_version = api_version_attempts[fallback_index]
            if current_api_version is not None:
                request_params["api-version"] = current_api_version

            response = self._request_with_transport_retries(
                method=method,
                url=url,
                params=request_params,
                json=json,
                endpoint=endpoint,
                api_version=current_api_version,
            )

            self.logger.info(
                "Azure request endpoint=%s api-version=%s status=%s request_count=%s fallback_attempt=%s/%s",
                endpoint,
                current_api_version,
                response.status_code,
                self.request_count,
                fallback_index + 1,
                len(api_version_attempts) if api_version_attempts else 1,
            )

            if response.status_code in expected_status:
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                return AzureResponse(
                    status_code=response.status_code,
                    payload=payload,
                    headers=dict(response.headers),
                )

            if response.status_code == 400 and requested_api_version is not None:
                next_index = fallback_index + 1
                if next_index < len(api_version_attempts):
                    self.logger.warning(
                        "Azure request retry endpoint=%s api-version=%s status=%s retry_attempt=%s/%s",
                        endpoint,
                        current_api_version,
                        response.status_code,
                        next_index,
                        len(api_version_attempts),
                    )
                    fallback_index = next_index
                    continue

            if response.status_code in {401, 403}:
                raise AzureDevOpsError(
                    "Unauthorized Azure DevOps response. Check that AZURE_PAT is valid and has access to the project."
                )

            if response.status_code == 400:
                raise AzureDevOpsError(self._build_error_message(response))

            raise AzureDevOpsError(self._build_error_message(response))

    def _request_with_transport_retries(
        self,
        *,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        json: Optional[Dict[str, Any]],
        endpoint: str,
        api_version: Optional[str],
    ) -> Response:
        last_error: Optional[Exception] = None
        retry_statuses = {429, 500, 502, 503, 504}

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.request_count += 1
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    timeout=self.config.timeout_seconds,
                )

                self.logger.info(
                    "Azure transport endpoint=%s api-version=%s status=%s transport_attempt=%s/%s request_count=%s",
                    endpoint,
                    api_version,
                    response.status_code,
                    attempt,
                    self.config.max_retries,
                    self.request_count,
                )

                if response.status_code in {401, 403, 400}:
                    return response

                if response.status_code in retry_statuses:
                    wait_seconds = self._retry_delay(response, attempt)
                    self.logger.warning(
                        "Azure transport retry endpoint=%s api-version=%s status=%s retry_attempt=%s/%s wait=%.1fs request_count=%s",
                        endpoint,
                        api_version,
                        response.status_code,
                        attempt,
                        self.config.max_retries,
                        wait_seconds,
                        self.request_count,
                    )
                    time.sleep(wait_seconds)
                    continue

                if 200 <= response.status_code < 300:
                    return response

                return response

            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                wait_seconds = self._retry_delay(None, attempt)
                self.logger.warning(
                    "Azure transport retry endpoint=%s api-version=%s after error=%s retry_attempt=%s/%s wait=%.1fs request_count=%s",
                    endpoint,
                    api_version,
                    exc,
                    attempt,
                    self.config.max_retries,
                    wait_seconds,
                    self.request_count,
                )
                time.sleep(wait_seconds)

        raise AzureDevOpsError(
            f"Failed to call Azure DevOps after {self.config.max_retries} attempts: {method} {url}"
        ) from last_error

    def _retry_delay(self, response: Optional[Response], attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return min(2 ** (attempt - 1), 30)

    @staticmethod
    def _endpoint_name(url: str) -> str:
        return url.split("/_apis/", 1)[-1] if "/_apis/" in url else url

    @staticmethod
    def _api_version_attempts(requested_api_version: Optional[str]) -> List[str]:
        if not requested_api_version:
            return []

        attempts: List[str] = []
        if requested_api_version:
            attempts.append(requested_api_version)
        for version in FALLBACK_API_VERSIONS:
            if version not in attempts:
                attempts.append(version)
        return attempts

    def _build_error_message(self, response: Response) -> str:
        detail = response.text.strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        return (
            f"Azure DevOps request failed with status {response.status_code} "
            f"for {response.request.method if response.request else 'GET'} {response.url}: {detail}"
        )

    def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        expected_status: Iterable[int] = (200,),
    ) -> AzureResponse:
        url = self._build_url(path)
        return self._request(
            "GET",
            url,
            params=params,
            expected_status=expected_status,
        )

    def post(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        expected_status: Iterable[int] = (200, 201),
    ) -> AzureResponse:
        url = self._build_url(path)
        return self._request(
            "POST",
            url,
            params=params,
            json=payload,
            expected_status=expected_status,
        )

    def paginate(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        item_key: str = "value",
        continuation_param: str = "continuationToken",
    ) -> Iterator[Dict[str, Any]]:
        base_params = dict(params or {})
        next_continuation_token: Optional[str] = None

        while True:
            request_params = dict(base_params)
            if next_continuation_token:
                request_params[continuation_param] = next_continuation_token

            response = self.get(path, params=request_params)
            payload = response.payload
            if isinstance(payload, dict) and item_key in payload:
                items = payload.get(item_key) or []
            elif isinstance(payload, list):
                items = payload
            else:
                raise AzureDevOpsError(
                    f"Pagination expected list payload for {path}, received {type(payload).__name__}"
                )

            for item in items:
                yield item

            next_continuation_token = self._continuation_token(response)
            if not next_continuation_token:
                break

    @staticmethod
    def _continuation_token(response: AzureResponse) -> Optional[str]:
        header_token = response.headers.get("x-ms-continuationtoken") or response.headers.get(
            "continuationtoken"
        )
        if header_token:
            return header_token

        payload = response.payload
        if isinstance(payload, dict):
            for key in ("continuationToken", "continuationtoken", "$continuationToken"):
                token = payload.get(key)
                if token:
                    return str(token)
        return None

    def wiql(self, query: str) -> Dict[str, Any]:
        response = self.post(
            "wit/wiql",
            params={"api-version": self.config.api_version},
            payload={"query": query},
        )
        if not isinstance(response.payload, dict):
            raise AzureDevOpsError("WIQL response was not a JSON object")
        work_items = response.payload.get("workItems")
        if work_items is None:
            response.payload["workItems"] = []
        return response.payload

    def get_work_items(
        self,
        ids: List[int],
        fields: Optional[List[str]] = None,
        expand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []

        payload: Dict[str, Any] = {"ids": ids}
        fields_enabled = fields is not None
        expand_enabled = expand is not None and not fields_enabled

        if fields_enabled:
            payload["fields"] = fields or []
            payload.pop("expand", None)
            payload.pop("$expand", None)
        elif expand_enabled:
            payload["expand"] = expand
            payload.pop("fields", None)

        self.logger.info(
            "Azure workitemsbatch request ids_count=%s fields_enabled=%s expand_enabled=%s payload=%s fields_value=%s expand_value=%s",
            len(ids),
            fields_enabled,
            expand_enabled,
            payload,
            fields,
            expand,
        )

        response = self.post(
            "wit/workitemsbatch",
            params={"api-version": self.config.api_version},
            payload=payload,
        )
        if not isinstance(response.payload, dict):
            raise AzureDevOpsError("Work items batch response was not a JSON object")
        return response.payload.get("value", []) or []

    def get_work_item(self, work_item_id: int, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"api-version": self.config.api_version}
        if fields:
            params["fields"] = ",".join(fields)
        response = self.get(f"wit/workitems/{work_item_id}", params=params)
        if not isinstance(response.payload, dict):
            raise AzureDevOpsError("Work item response was not a JSON object")
        return response.payload

    def _build_url(self, path: str) -> str:
        cleaned_path = path.lstrip("/")
        return f"{self.config.base_url}/{cleaned_path}"
