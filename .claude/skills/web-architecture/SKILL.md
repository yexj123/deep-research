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
    config={"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT},
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
don't stand up a second database until there's a real reason to.

**Never use `AsyncSqliteSaver.from_conn_string()` here (D-082).** Verified
2026-09-21: it takes no `serde` argument and its body is `cls(conn)`, so it
silently discards the allowlist from `build_serializer()` — the one thing
D-014 exists to enforce. Build it by hand instead, and call `setup()` once:

```python
async with aiosqlite.connect(db_path) as conn:
    checkpointer = AsyncSqliteSaver(conn, serde=build_serializer())
    await checkpointer.setup()
```

Without the allowlist a `Source` still restores **today**, but logs
`Deserializing unregistered type ... will be blocked in a future version`.
It's not a break now; it's a break later, announced by a line that's easy to
miss in server logs.

**Resumability is real and worth designing around (D-081).** Closing the
stream mid-run — what FastAPI does when the browser goes away — leaves a
checkpoint whose `next` names the pending nodes, and `astream(None, config)`
continues from there. The pending `Send` fan-out survives too. So a dropped
connection costs the in-flight node, not the run, and no background task
registry is needed.

## Frontend — decided (D-080): htmx, plus a small EventSource

Server-rendered Jinja2 in `api/`, htmx for the page, form, run history and
the node-progress trail. **The token stream is NOT htmx.** Verified
2026-09-21: the SSE extension (`htmx-ext-sse`, separate from core) documents
`sse-connect`, `sse-swap`, `hx-trigger="sse:<name>"` and `sse-close`, and
shows **no example combining `sse-swap` with `hx-swap`** — appending streamed
tokens is undocumented. Pure htmx would mean re-swapping the whole review
block per token.

And JS was always required regardless: htmx swaps HTML, the review is
markdown accumulating token by token, so something must re-render it. "htmx
means no JavaScript" is false for this app — don't repeat it.

So: `updates` → `sse-swap` on a progress element (exactly what the extension
documents); `messages` → ~20 lines of vanilla `EventSource` that buffers text
and re-renders markdown. SvelteKit was rejected as a second toolchain for a
project whose contribution is the agent.

## Auth — don't build it yet

Single-user by default; if this leaves localhost, gate it behind one
password/env var, not a user system. Open WebUI needs real multi-tenant
auth because it's shared infrastructure for many people's chats — this
isn't, unless and until you decide it should be.
