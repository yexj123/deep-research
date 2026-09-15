---
description: This project's web-layer patterns — streaming the agent to the browser over SSE, checkpointing/persistence, and the frontend-framework tradeoff. Load when writing or reviewing an API route, streaming code, or frontend code.
---

# Web layer conventions for this project

The agent itself (state schema, nodes, `Send` fan-out, recursion control) is
unchanged and lives in the `langgraph-conventions` skill. This skill covers
the app wrapped around it — the part that makes it feel like Open WebUI
instead of a script you run from the terminal.

## Streaming the agent to the browser

`graph.astream()` accepts a list of `stream_mode`s and interleaves them on
one iterator — this is the whole trick, don't reach for websockets or a
task queue before you've tried it. This project uses `version="v2"`
(D-030): every chunk is a dict with `type`, `ns` and `data`.

```python
async for chunk in graph.astream(
    {"question": question},
    config={"configurable": {"thread_id": thread_id}, "recursion_limit": 150},
    context=RunContext(provider=provider),   # per-run provider (D-015)
    stream_mode=["updates", "custom", "messages"],
    version="v2",
):
    match chunk["type"]:
        case "updates":
            ...   # chunk["data"] == {node_name: that node's update}
        case "custom":
            ...   # progress you emit yourself, see below
        case "messages":
            token, metadata = chunk["data"]   # metadata["langgraph_node"] names the node
```

- `updates` — one event per finished node. This is your "decompose →
  searching subtopic 3/5 → checking gaps" progress trail for free.
- `custom` — anything you push yourself with `get_stream_writer()` (from
  `langgraph.config`) inside a node, e.g. `writer({"status": "found 4 new
  papers on X"})`. Use this for progress that doesn't map to a whole node
  finishing — mid-search updates, papers as they're found.
- `messages` — token-by-token text from the synthesize node specifically,
  the same feel as an LLM chat response streaming in.

Standard production pattern: one FastAPI route starts a run and returns a
`thread_id`; a second route (`GET /runs/{thread_id}/stream`) wraps the
`astream` loop above and yields Server-Sent Events. SSE over websockets
here — it's one-directional (server → browser), simpler, and plays nicer
with HTTP infra than a websocket does. Reach for a websocket only if you
add a feature that needs the browser to talk back mid-run (e.g. "steer the
agent" controls) — not needed for v1.

## Persistence

`AsyncSqliteSaver` (package `langgraph-checkpoint-sqlite`, import from
`langgraph.checkpoint.sqlite.aio`) gives the graph resumable runs keyed by
`thread_id`, and the same SQLite file can hold your saved-review history —
don't stand up a second database until there's a real reason to. Set
`LANGGRAPH_STRICT_MSGPACK=true` (or pass an explicit
`allowed_msgpack_modules`) when constructing the checkpointer — it restricts
what a checkpoint can deserialize into, which matters once this file is
something other people run from a cloned repo.

## Frontend — decide this together, don't default silently

Two real options, genuinely different tradeoffs:

- **Match Open WebUI's shape (SvelteKit SPA + REST/SSE)**: closer to the
  thing you're modeling this on, but a second toolchain, a build step, and
  time spent on frontend plumbing instead of the agent.
- **Server-rendered (FastAPI + Jinja2 + htmx, SSE wired straight into a
  swapped DOM fragment)**: one language, one process, live-updating UI
  without a JS framework — the pragmatic default under a real deadline,
  with less to point to if "matching Open WebUI's feel" is itself part of
  the point.

Pick based on how much of the remaining time you want to spend on frontend
versus the agent — there's no wrong answer, just say which and I'll reason
from there instead of assuming.

## Auth — don't build it yet

Single-user by default; if this leaves localhost, gate it behind one
password/env var, not a user system. Open WebUI needs real multi-tenant
auth because it's shared infrastructure for many people's chats — this
isn't, unless and until you decide it should be.
