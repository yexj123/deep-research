# Progress

What's done, what's next, and whose job each item is. Design reasoning
lives in [`decisions.md`](decisions.md); this file only tracks status.

**Last updated:** 2026-09-16 · **Current milestone:** 2 (one real source: arXiv)

---

## Do next (you)

- [x] `.claude/settings.json`: removed `"MultiEdit"`, since Claude Code reported it "matches no known tool"
      (duplicate `"Edit"` also removed; `CLAUDE.md` and the output style updated to match)
- [ ] **Fully quit and reopen VS Code once.** Claude Code gets its environment from
      VS Code, and it needs the new `UV_CACHE_DIR=E:\uv-cache` variable.
- [ ] *(Optional)* Delete the old 55 MB uv cache on C::
      `uv cache clean --cache-dir "$env:LOCALAPPDATA\uv\cache"`
- [ ] **Before writing milestone 2 code**, settle its open decisions (table below).

## Upcoming milestones

| # | Milestone | Open decisions to settle first (see `decisions.md` → Open) |
|---|---|---|
| 2 | One real source (arXiv), single subtopic, citation shape | **All decided (D-040 – D-050).** Next: plan the milestone 2 code (files, tests, saved arXiv responses for `MockTransport`) |
| 3 | `decompose` + `Send` fan-out, reducers | arXiv rate limiter (≤1 req / 3 s) · models per role · treating HTTP 4xx and 5xx differently |
| 4 | `gap_check` + depth recursion, retry cap, paper overlap | making failures visible · `recursion_limit` value |
| 5 | Web layer: FastAPI + SSE + `AsyncSqliteSaver` | frontend (SvelteKit or htmx) · public entry function |

**Reminders for when these come up:**
- **Milestone 2 onward:** add every custom Pydantic model or dataclass stored in state to
  `allowed_msgpack_modules` (D-014).
- **Milestone 2:** replace the milestone 1 `SYSTEM_PROMPT` in `nodes/synthesize.py`. The new one
  must cite retrieved sources using exactly `[arXiv:<arxiv_id>]` (D-046).
- **Milestone 2:** `uv add defusedxml` (D-047); add `Source` to `allowed_msgpack_modules` (D-014)
  and write its checkpoint round-trip test (D-050).
- **Milestone 3:** capture what `updates` chunks look like when several `Send` workers finish
  in the same step (`docs/langgraph-outputs.md` §4).
- **Milestone 5:** if you choose SvelteKit, add `node_modules/`, `.svelte-kit/` and the build output
  folder to `.gitignore`.

---

## Done

### Milestone 1: `intake → synthesize` (commit `95fd66c`)
The graph compiles, the provider arrives through runtime context, tokens stream in
`messages` mode, and state survives the checkpointer. `uv run pytest`: 14 passed,
1 deselected. The integration test passed against the real OpenAI API (2026-09-16).

- [x] `agent/state.py`: `ResearchState` dataclass (D-013)
- [x] `agent/context.py`: `RunContext` with the per-run `provider` (D-015)
- [x] `agent/config.py`: key variables, model names (`gpt-4o`, `deepseek-flash`),
      `LLM_TIMEOUT_SECONDS = 60.0`, `LLM_MAX_RETRIES = 2` (D-039)
- [x] `agent/llm.py`: client factory with clear errors and explicit limits (D-029, D-034, D-038)
- [x] `agent/nodes/intake.py`: context, provider and question checks (D-033)
- [x] `agent/nodes/synthesize.py`: milestone 1 system prompt; returns the reply as `review`
- [x] `agent/graph.py`: `build_graph(model_factory, checkpointer)` (D-032)
- [x] `tests/agent/`: fake-model graph tests, factory tests including the limits, and an
      opt-in integration test (D-031, D-036, D-037)

### Docs, license and test policy (commit `82fada6`)
- [x] MIT `LICENSE` and license metadata in `pyproject.toml` (checked in a built wheel)
- [x] Integration tests deselected by default (`-m "not integration"`)
- [x] `docs/decisions.md` D-032 to D-039; `docs/progress.md`; `docs/langgraph-outputs.md`
      (output shapes captured from this graph)
- [x] README: setup, tests, running integration tests

### Milestone 0: project skeleton (commit `a9505d1`)
- [x] uv project with `src/` layout (`uv_build`), `uv.lock`, Python 3.12 pinned
- [x] Dependencies: `langgraph`, `pydantic`, `langchain-openai`, `langchain-deepseek`,
      `langchain-core`; dev: `pytest`, `pytest-asyncio`
- [x] pytest config: `testpaths`, `--strict-markers`, `integration` marker
- [x] Smoke test, so `uv run pytest` exits 0
- [x] README (setup, bring-your-own-keys, tests)
- [x] Removed `uv init`'s placeholder `main()` and script entry

### Design log and conventions (commits `b5b933c`, `a9505d1`)
- [x] Decision log `docs/decisions.md` with an Open section
- [x] `CLAUDE.md` requires every decision to be logged when it's made
- [x] Skills updated to match the decisions (`langgraph-conventions`,
      `web-architecture` v2 streaming example)
- [x] Corrected the reducer note: a missing reducer **raises** `InvalidUpdateError`;
      it doesn't silently overwrite

### Checked against installed versions (langgraph 1.2.11)
- [x] `context_schema` + `Runtime[Ctx]` gives the node the context and a dataclass state
- [x] `ainvoke()` inside a node still streams tokens in `messages` mode;
      `GenericFakeChatModel` streams token by token
- [x] `ainvoke(version="v2").value` returns the dataclass; `get_state().values` is a dict
- [x] Calling the graph without `context=` makes `runtime.context` `None` → `AttributeError` later on
- [x] `ChatDeepSeek` subclasses `BaseChatOpenAI` (raises `openai.*` errors)
- [x] Without explicit limits, the OpenAI client under `ChatOpenAI` has `timeout=None`

### Repo and environment (commit `fecf66d` and machine setup)
- [x] Claude Code config moved into `.claude/` (agents, output style, skills, settings)
- [x] `git init -b main`; `.gitignore` covers `.env*`, SQLite + WAL files, PDFs
- [x] PowerShell and `uv run pytest` permission allow rules
- [x] `uv` installed and resolving on Claude Code's PATH
- [x] uv cache moved to `E:\uv-cache` (no more hardlink warning)
