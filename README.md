# llama-index-tools-looot

LlamaIndex tool spec for [looot](https://looot.ai). One token and one prepaid balance give an agent 4,200+ data APIs from 300+ providers: work emails, people and company search, Google results, web pages, news. The agent searches the catalog, reads the price, then runs. A failed call costs nothing.

## Install

Until the package is on PyPI, install from the git tag:

    pip install "llama-index-tools-looot @ git+https://github.com/loootai/llama-index-tools-looot@v0.1.0"

Get a token at https://looot.ai (Settings, Agent tokens) and set it:

    export LOOOT_TOKEN=...

Top up from $5 before the first paid run. Search, inspect, get_run and balance are free.

## One call

```python
from llama_index.tools.looot import LoootToolSpec

spec = LoootToolSpec()  # reads LOOOT_TOKEN
print(spec.balance())
print(spec.search_catalog("find the work email of a person at a company", limit=3))
```

## Use with an agent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.looot import LoootToolSpec

tools = LoootToolSpec(max_cost_usd=1.0).to_tool_list()
agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4.1"))
print(await agent.run("Find the work email of the CEO of example.com"))
```

## Tools

| Function | What it does | Cost |
| --- | --- | --- |
| `search_catalog(query, limit, provider, category, prefer)` | Find endpoints by the task you want done | free |
| `inspect(endpoint_id)` | Input fields, output shape, maximum price | free |
| `run(endpoint_id, input, idempotency_key, wait)` | Run an endpoint or a `job:...` id | spends credit, failed calls cost nothing |
| `get_run(run_id)` | Read a run: status, result, cost, error | free |
| `balance()` | Available and reserved credit, top-up link | free |

`run` sends a fallback cap of `max_cost_usd` (default 1.0) so a job that falls back to another provider cannot spend more than that. Pass `max_cost_usd=None` for no cap. Failures raise `LoootError` with `status`, `code` and `request_id`.

Docs: https://docs.looot.ai. Remote MCP server: https://api.looot.ai/mcp.

## Develop

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest
    git config core.hooksPath .githooks

MIT license.
