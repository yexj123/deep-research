# Progress

What's done, what's next, and whose job each item is. Design reasoning
lives in [`decisions.md`](decisions.md); this file only tracks status.

**Last updated:** 2026-09-15 · **Current milestone:** 1 (`intake → synthesize`)

---

## Do next (you)

- [x] Moved `agent/state.py` into `src/deep_research/agent/`
- [x] `__init__.py` files in `agent/`, `agent/nodes/`, `tests/`, `tests/agent/`
      (add one to `tests/api/` when it exists)
- [ ] **Fully quit and reopen VS Code once.** Claude Code gets its environment from
      VS Code, and it needs the new `UV_CACHE_DIR=E:\uv-cache` variable.
- [ ] *(Optional)* Delete the old 55 MB uv cache on C::
      `uv cache clean --cache-dir "$env:LOCALAPPDATA\uv\cache"`
- [ ] Commit the pending changes: `LICENSE`, `pyproject.toml`, `docs/`, and the skill
      edit (ask Claude, or commit them yourself).

## Milestone 1: `intake → synthesize` (you write)

**Goal:** the graph compiles, the provider arrives through runtime context,
tokens stream in `messages` mode, and state survives the checkpointer. No search yet.

All paths below are under `src/deep_research/`.

**Status:** done. `uv run pytest` gives 14 passed, 1 deselected, and the integration
test passes against the real OpenAI API.

- [x] `agent/state.py`: `ResearchState` dataclass (D-013)
- [x] `agent/context.py`: `RunContext` dataclass with `provider: ProviderType`
      *(optional: `frozen=True`)*
- [x] `agent/config.py`: `API_KEY_ENV_VARS`, `MODEL_NAMES` (`gpt-4o`, `deepseek-flash`)
- [x] `agent/llm.py`: client factory; clear errors for a missing key or model name (D-029, D-034)
- [x] `agent/nodes/intake.py`: context, provider and question checks; returns `{"question": ...}` (D-033)
- [x] `agent/nodes/synthesize.py`: milestone 1 system prompt; returns `{"review": reply.text}`
- [x] `agent/graph.py`: `build_graph(model_factory, checkpointer)`, nodes named
      `intake` / `synthesize` (D-032)
- [x] `tests/agent/`: async tests, each with `@pytest.mark.asyncio`, using
      `GenericFakeChatModel` (12 passing):
  - [x] streamed tokens arrive as several `messages` chunks, with `langgraph_node == "synthesize"`
  - [x] `updates` chunks arrive in node order
  - [x] `ainvoke(..., version="v2").value` is a `ResearchState`
  - [x] the checkpoint (`get_state(config).values`, a plain dict) contains `review`
  - [x] the provider from the context reaches the factory
  - [x] an empty question raises; a missing or invalid context raises
  - [x] the factory raises when the API key is missing (use `monkeypatch.delenv`)
- [x] `match="intake:"` moved into `pytest.raises(...)` in the three validation tests
- [x] Integration tests are opt-in: `addopts` has `-m "not integration"` (D-036)
- [x] `LLM_TIMEOUT_SECONDS = 60.0` / `LLM_MAX_RETRIES = 2` passed to both chat models
      (D-038, D-039), with a parametrized unit test that fails if they're removed
- [x] Integration test written (D-037); passed against the real OpenAI API
      (`uv run pytest -m integration`, run 2026-09-16)
- [ ] `uv run pytest` passes. Then ask Claude or `code-reviewer` for a review.
- [ ] Commit milestone 1.

## Later milestones

| # | Milestone | Open decisions to settle first (see `decisions.md` → Open) |
|---|---|---|
| 2 | One real source (arXiv), single subtopic, citation shape | `Source` object shape · prompt-injection defenses (citation IDs restricted to the retrieved set) |
| 3 | `decompose` + `Send` fan-out, reducers | arXiv rate limiter (≤1 req / 3 s) · models per role · treating HTTP 4xx and 5xx differently |
| 4 | `gap_check` + depth recursion, retry cap, paper overlap | making failures visible · `recursion_limit` value |
| 5 | Web layer: FastAPI + SSE + `AsyncSqliteSaver` | frontend (SvelteKit or htmx) · public entry function |

**Reminders for when these come up:**
- **Milestone 2 onward:** add every custom Pydantic model or dataclass stored in state to
  `allowed_msgpack_modules` (D-014).
- **Milestone 5:** if you choose SvelteKit, add `node_modules/`, `.svelte-kit/` and the build output
  folder to `.gitignore`.

---

## Done

### Milestone 0: project skeleton (commit `a9505d1`)
- [x] uv project with `src/` layout (`uv_build`), `uv.lock`, Python 3.12 pinned
- [x] Dependencies: `langgraph`, `pydantic`, `langchain-openai`, `langchain-deepseek`,
      `langchain-core`; dev: `pytest`, `pytest-asyncio`
- [x] pytest config: `testpaths`, `--strict-markers`, `integration` marker
- [x] Smoke test, so `uv run pytest` exits 0
- [x] README (setup, bring-your-own-keys, tests)
- [x] Removed `uv init`'s placeholder `main()` and script entry

### Design (commits `b5b933c`, `a9505d1`, plus pending changes)
- [x] Decision log `docs/decisions.md`, D-001 to D-035, with an Open section
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

### Repo and environment
- [x] Claude Code config moved into `.claude/` (agents, output style, skills, settings),
      commit `fecf66d`
- [x] `git init -b main`; `.gitignore` covers `.env*`, SQLite + WAL files, PDFs
- [x] PowerShell and `uv run pytest` permission allow rules
- [x] MIT `LICENSE` and license metadata in `pyproject.toml`; checked in a built wheel
      *(not committed yet)*
- [x] `uv` installed and resolving on Claude Code's PATH
- [x] uv cache moved to `E:\uv-cache` (no more hardlink warning)
