"""looot tool spec: search the catalog, inspect an endpoint, run it, read a run, read the balance."""

import json
import os
import uuid
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.looot.ai"
MAX_OUTPUT_CHARS = 20_000


class LoootError(Exception):
    """A non-2xx answer. `code` is the gateway's error code, e.g. insufficient_balance."""

    def __init__(
        self, status: int, code: str, message: str, request_id: Optional[str] = None
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.request_id = request_id


def _clean(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = ("true" if value else "false") if isinstance(value, bool) else value
    return out


def _check(response: httpx.Response, method: str, path: str) -> Any:
    try:
        data: Any = response.json() if response.content else None
    except ValueError:
        data = response.text
    if response.is_error:
        err = data.get("error") if isinstance(data, dict) else None
        err = err if isinstance(err, dict) else {}
        raise LoootError(
            response.status_code,
            err.get("code") or f"http_{response.status_code}",
            err.get("message")
            or f"{method} {path} failed with HTTP {response.status_code}",
            err.get("requestId"),
        )
    return data


def _render(data: Any) -> str:
    """Compact JSON, cut at MAX_OUTPUT_CHARS so one answer cannot fill the model's context."""
    text = json.dumps(data, separators=(",", ":"), default=str)
    if len(text) > MAX_OUTPUT_CHARS:
        return (
            text[:MAX_OUTPUT_CHARS]
            + f"... [truncated, {len(text) - MAX_OUTPUT_CHARS} more characters]"
        )
    return text


class LoootToolSpec(BaseToolSpec):
    """
    looot tool spec.

    One token and one prepaid balance give an agent 4,200+ data APIs. Search, inspect,
    get_run and balance are free. run spends credit, and a failed call costs nothing.
    """

    spec_functions = ["search_catalog", "inspect", "run", "get_run", "balance"]

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        max_cost_usd: Optional[float] = 1.0,
        timeout: float = 75.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            token: A looot agent token. Defaults to the LOOOT_TOKEN env var.
            base_url: Defaults to LOOOT_BASE_URL, then https://api.looot.ai.
            max_cost_usd: Cap in USD for one run when looot falls back to another provider.
                None sends no cap.
            timeout: HTTP timeout in seconds.
            transport: An httpx transport. Tests pass httpx.MockTransport.

        """
        self.token = (
            token if token is not None else os.environ.get("LOOOT_TOKEN", "")
        ).strip() or None
        self.base_url = (
            base_url or os.environ.get("LOOOT_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_cost_usd = max_cost_usd
        self.timeout = timeout
        self._transport = transport

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
    ) -> str:
        if not self.token:
            raise LoootError(
                401,
                "missing_token",
                f"{method} {path} needs a token: set LOOOT_TOKEN or pass token=",
            )
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.token}",
        }
        with httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=self._transport
        ) as http:
            response = http.request(
                method,
                path,
                params=_clean(params or {}),
                json=json_body,
                headers=headers,
            )
        return _render(_check(response, method, path))

    def search_catalog(
        self,
        query: str,
        limit: int = 10,
        provider: Optional[str] = None,
        category: Optional[str] = None,
        prefer: Optional[str] = None,
    ) -> str:
        """
        Search the looot catalog of 4,200+ data API endpoints by the task you want done. Free.

        Call this first, then inspect, then run.

        Args:
            query: What you want to do, in plain words, e.g. 'find the work email of a person at a company'.
            limit: Max number of rows (1 to 50).
            provider: Only this provider id.
            category: Only this category.
            prefer: How to rank the rows: balanced, cheapest, fastest or reliable.

        Returns:
            Compact JSON with ranked rows: endpoint ids, prices and job ids.

        """
        params = {
            "q": query,
            "limit": limit,
            "provider": provider,
            "category": category,
            "prefer": prefer,
        }
        return self._request("GET", "/v1/catalog/search", params)

    def inspect(self, endpoint_id: str) -> str:
        """
        Read the exact input fields, output shape and maximum price of one endpoint. Free.

        Use the field names it lists as the input of run.

        Args:
            endpoint_id: An endpoint id from a search row, or a job id such as 'job:people.email.find'.

        Returns:
            Compact JSON describing the endpoint.

        """
        return self._request("GET", f"/v1/operations/{quote(endpoint_id, safe='')}")

    def run(
        self,
        endpoint_id: str,
        input: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        wait: int = 20,
    ) -> str:
        """
        Run an endpoint or job and spend prepaid credit. A failed call costs nothing.

        Check the price with inspect first. If status is not 'completed', call get_run
        later with the runId.

        Args:
            endpoint_id: An endpoint id from search, or a job such as 'job:people.email.find' to let looot pick the provider.
            input: The input object, using the field names inspect lists.
            idempotency_key: Reuse the same key to retry without paying twice. Leave empty for a new key.
            wait: Seconds to wait for the result (0 to 60). 0 returns a run id at once.

        Returns:
            Compact JSON with the run: status, result, actualCost.

        """
        fallback: Dict[str, Any] = {"enabled": True}
        if self.max_cost_usd:
            fallback["maxCostUsd"] = self.max_cost_usd
        body = {
            "endpointId": endpoint_id,
            "input": input,
            "idempotencyKey": idempotency_key or f"li-{uuid.uuid4()}",
            "fallback": fallback,
        }
        return self._request("POST", "/v1/runs", {"wait": wait}, body)

    def get_run(self, run_id: str) -> str:
        """
        Read one run: status, result, cost and error. Free.

        Args:
            run_id: The runId returned by run.

        Returns:
            Compact JSON with the run.

        """
        return self._request("GET", f"/v1/runs/{quote(run_id, safe='')}")

    def balance(self) -> str:
        """
        Read the available and reserved credit and the top-up link. Free.

        Returns:
            Compact JSON with the balance.

        """
        return self._request("GET", "/v1/balance")
