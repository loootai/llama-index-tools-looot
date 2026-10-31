import json

import httpx
import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.looot import LoootError, LoootToolSpec


def make(handler, **kw):
    seen = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    spec = LoootToolSpec(
        token="tok_test", transport=httpx.MockTransport(wrapped), **kw
    )
    return spec, seen


def ok(payload):
    return lambda request: httpx.Response(200, json=payload)


def test_class_inherits_base_tool_spec():
    assert issubclass(LoootToolSpec, BaseToolSpec)
    assert LoootToolSpec.spec_functions == [
        "search_catalog",
        "inspect",
        "run",
        "get_run",
        "balance",
    ]


def test_to_tool_list_has_five_tools():
    tools = LoootToolSpec(token="t").to_tool_list()
    assert [t.metadata.name for t in tools] == LoootToolSpec.spec_functions


def test_search_catalog():
    spec, seen = make(ok({"rows": [{"endpointId": "serper-search"}]}))
    out = spec.search_catalog("google results", limit=3, prefer="cheapest")
    assert json.loads(out)["rows"][0]["endpointId"] == "serper-search"
    req = seen[0]
    assert req.method == "GET" and req.url.path == "/v1/catalog/search"
    assert req.url.params["q"] == "google results"
    assert req.url.params["limit"] == "3"
    assert req.url.params["prefer"] == "cheapest"
    assert "provider" not in req.url.params
    assert req.headers["authorization"] == "Bearer tok_test"


def test_inspect_encodes_job_id():
    spec, seen = make(ok({"id": "job:people.email.find"}))
    spec.inspect("job:people.email.find")
    assert seen[0].url.raw_path.decode().startswith(
        "/v1/operations/job%3Apeople.email.find"
    )


def test_run_sends_body_wait_and_default_cap():
    spec, seen = make(ok({"runId": "r1", "status": "completed"}))
    out = spec.run("job:people.email.find", {"domain": "example.com"}, wait=5)
    assert json.loads(out)["status"] == "completed"
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/runs"
    assert req.url.params["wait"] == "5"
    body = json.loads(req.content)
    assert body["endpointId"] == "job:people.email.find"
    assert body["input"] == {"domain": "example.com"}
    assert body["idempotencyKey"].startswith("li-")
    assert body["fallback"] == {"enabled": True, "maxCostUsd": 1.0}


def test_run_custom_cap_and_idempotency_key():
    spec, seen = make(ok({"runId": "r1"}), max_cost_usd=0.25)
    spec.run("job:company.enrich", {"domain": "example.com"}, idempotency_key="abc")
    body = json.loads(seen[0].content)
    assert body["idempotencyKey"] == "abc"
    assert body["fallback"]["maxCostUsd"] == 0.25


def test_run_no_cap_when_none():
    spec, seen = make(ok({"runId": "r1"}), max_cost_usd=None)
    spec.run("job:company.enrich", {})
    assert json.loads(seen[0].content)["fallback"] == {"enabled": True}


def test_get_run():
    spec, seen = make(ok({"runId": "r 1", "status": "running"}))
    spec.get_run("r 1")
    assert seen[0].method == "GET"
    assert seen[0].url.raw_path.decode() == "/v1/runs/r%201"


def test_balance():
    spec, seen = make(ok({"availableUsd": 4.2}))
    assert json.loads(spec.balance())["availableUsd"] == 4.2
    assert seen[0].url.path == "/v1/balance"


def test_error_raises_looot_error():
    def handler(request):
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": "insufficient_balance",
                    "message": "Top up first",
                    "requestId": "req_1",
                }
            },
        )

    spec, _ = make(handler)
    with pytest.raises(LoootError) as exc:
        spec.run("job:company.enrich", {})
    assert exc.value.status == 402
    assert exc.value.code == "insufficient_balance"
    assert exc.value.request_id == "req_1"


def test_missing_token_makes_no_request(monkeypatch):
    monkeypatch.delenv("LOOOT_TOKEN", raising=False)
    spec = LoootToolSpec()
    with pytest.raises(LoootError) as exc:
        spec.balance()
    assert exc.value.code == "missing_token"


def test_long_output_is_truncated():
    spec, _ = make(ok({"blob": "x" * 30_000}))
    assert "truncated" in spec.search_catalog("anything")
