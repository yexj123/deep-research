# Open WebUI for Deep Research & Recursive Literature Review

A self-hosted web app — same idea as Open WebUI, but the "chat" is a
LangGraph agent: give it a research question, it searches literature,
recursively decomposes into subtopics when it finds gaps, and synthesizes
a cited review. Python + LangGraph + FastAPI + the Anthropic API. Built for
thesis and personal use, published on GitHub.

## Timeline

Past the hackathon — building this independently now, normal pace. No more
MVP-triage framing; everything below assumes there's room to do it properly.

## How we work together

I write the implementation — across the agent, the API, and the frontend.
You're the architect and reviewer, not the typist.

- Default to explaining: the concept, 2-3 real approaches with tradeoffs,
  your recommendation and *why* — then stop. I write the code.
- The full behavioral spec is the active output style at
  `.claude/output-styles/research-mentor.md`. `.claude/settings.json` backs
  it up by requiring my confirmation on every Edit/Write/MultiEdit/
  NotebookEdit — a decline is expected behavior, not a bug to route around.
- When reviewing code I've written: cite the file/line, explain *why* it
  matters, show a corrected snippet in your response. Don't edit the file.
- I need to be able to defend every line — to a judge, an advisor, or a
  stranger's GitHub issue — so optimize for my understanding over shipping
  speed.
- CLAUDE.md instructions are context, not enforcement — if something here
  really has to hold, it's backed by a permission rule or tool restriction,
  not just this paragraph.

## Tech stack

- Backend: Python 3.12, FastAPI, the LangGraph agent below
- Streaming: Server-Sent Events wrapping `graph.astream(..., stream_mode=
  ["updates","custom","messages"])` — full pattern in `web-architecture`
- Persistence: SQLite via `langgraph-checkpoint-sqlite` (`AsyncSqliteSaver`)
- Frontend: **TODO** — decide together, tradeoffs in `web-architecture`
- Package manager: **TODO**

## Legal / licensing ground rules (settled — don't re-litigate per session)

- MIT license on this repo.
- Users bring their own API keys (Anthropic, and arXiv/Semantic Scholar if
  used) — never bake mine in, never commit a `.env`.
- arXiv/Semantic Scholar metadata: free to use. Full PDFs: fetch to read,
  don't cache-and-redistribute — most papers don't permit it.
- No Google Scholar scraping, anywhere, ever.

## Architecture

**Agent (LangGraph)** — the research/recursion logic itself:

```
intake → decompose (planner) → Send() fan-out per subtopic
    → search+read (worker, parallel) → gap_check (conditional)
        → [gaps remain & depth < max] → back to decompose
        → [else] → synthesize → END
```

Two-layer recursion control (a `depth` field in state is the real exit;
`recursion_limit` in the invoke config is only a backstop), the `Send`
fan-out template, and citation grounding all live in the
`langgraph-conventions` skill — this is just the shape.

**Web layer** — new on top of the agent, unchanged agent internals:

```
Browser --SSE--> FastAPI route --astream()--> LangGraph agent
                       |
                  AsyncSqliteSaver (checkpoint + saved-review history)
```

One route starts a run and hands back a `thread_id`; a second streams
`updates` (node progress), `custom` (progress you emit yourself), and
`messages` (token-by-token synthesis text) over one SSE connection. No
multi-user auth until there's an actual reason for one — full pattern,
including the checkpoint security note, lives in the `web-architecture`
skill.

## Repository layout

```
src/deep_research/
├── agent/            # unchanged agent design — state.py, graph.py, nodes/, sources/
├── api/              # FastAPI: main.py, routes/{runs,reviews}.py, deps.py
├── persistence/       # checkpointer.py (AsyncSqliteSaver)
└── config.py
frontend/              # shape depends on the SvelteKit-vs-htmx decision — see web-architecture
tests/{agent,api}/, conftest.py
notebooks/              # thesis-side evaluation, not shipped code
```

`src/` layout, not flat — avoids import-path footguns and matches how the
package will actually get installed. Full reasoning and the two frontend
shapes (htmx has no `frontend/` at all — templates live in `api/`; SvelteKit
gets a real one) live in chat, ask me to restate if this file outlives that
conversation.

## Commands

- Backend run: **TODO** (something like `uvicorn app.main:app --reload`)
- Test: `pytest`
- Frontend: **TODO**

## Skills & subagents in this project

- `/judge-prep` — prep for the AI Judge interview, a thesis-advisor
  walkthrough, or explaining the repo to a stranger.
- `/demo-check` — run before any live demo: server starts, a run actually
  streams end to end, nothing's silently broken.
- `langgraph-advisor` subagent (read-only) — agent-architecture questions.
- `code-reviewer` subagent (read-only) — reviews the LangGraph agent code.
- `api-reviewer` subagent (read-only) — reviews FastAPI/streaming/frontend
  code: auth gaps, exposed keys, dropped SSE connections.

Claude Code's own auto memory (`MEMORY.md`, on by default) picks up
corrections and running context as we go — this file is for what should
hold every session, not a log of what happened in one.
