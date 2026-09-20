# Progress

What's done, what's next, and whose job each item is. Design reasoning
lives in [`decisions.md`](decisions.md); this file only tracks status.

**Last updated:** 2026-09-20 · **Current milestone:** 2 (one real source: arXiv)

**Who writes what:** you write the implementation (`src/`); Claude writes every test
(since 2026-09-19) and keeps the docs (since 2026-09-20) — `docs/*.md` and the README.
Backed by `CLAUDE.md`, the output style, and `Edit(/tests/**)`, `Edit(/docs/**)`,
`Edit(/README.md)` in `.claude/settings.json`.

---

## Do next (you)

- [x] `.claude/settings.json`: removed `"MultiEdit"`, since Claude Code reported it "matches no known tool"
      (duplicate `"Edit"` also removed; `CLAUDE.md` and the output style updated to match; commit `532473d`)
- [x] **Fully quit and reopen VS Code once** for `UV_CACHE_DIR=E:\uv-cache` (the variable is set in the environment)
- [ ] *(Optional)* Delete the old 55 MB uv cache on C::
      `uv cache clean --cache-dir "$env:LOCALAPPDATA\uv\cache"`
- [x] **Review the milestone 2 code** — walked through on 2026-09-20 (see "Review outcome" below).
      "No papers found" confirmed as D-060; two findings became D-061 and D-062
- [x] **Apply the three review fixes** — `extra="forbid"` on `Source` (D-061), the corrected
      `arxiv.py` comment, and `CITATION_BRACKET` in `check_citations.py` (D-062)
- [x] Run `uv run pytest` → **66 passed, 1 deselected**
- [x] Run the integration test → **1 passed** (2026-09-20, see below)
- [x] Two comment fixes in `check_citations.py` from the review (module docstring, `CITATION_BRACKET`)
- [x] Commit — milestone 2 is `60611ac`; the docs-ownership change is `2db3260`

## Milestone 2: one real source, arXiv

**Goal:** `intake → search → synthesize → check_citations`. The question is searched on arXiv,
the review cites only retrieved papers, and a `Source` survives a checkpoint.

**Status (2026-09-20):** done, committed as `60611ac`. `uv run pytest`:
**66 passed, 1 deselected**. `uv run pytest -m integration`: **1 passed** in 11.17s against the real
OpenAI API and the real arXiv API.

**What the integration run proved (2026-09-20, `gpt-4o`):** arXiv returned papers for
"What is attention in transformer models?"; the review streamed in more than one `messages` chunk;
the joined stream matched the saved `review` exactly; at least one `[arXiv:<id>]` citation was
present; and `citation_violations == []`. The version-suffix failure predicted during review
(`[arXiv:2411.18583v1]`, which D-044 correctly counts as a violation) did **not** occur — the model
followed the marker format. Caveat worth stating in any write-up: that is **one** run. It shows
prompt adherence is achievable, not that it's reliable; measuring how often it holds is
`notebooks/` work for the thesis, not a unit test.

**Design:** D-040 – D-062. How the files relate is explained in [`code-map.md`](code-map.md);
what one run actually does, step by step with real values, is in [`walkthrough.md`](walkthrough.md).

### Review outcome (2026-09-20)

The 09-19 code was walked through file by file. Three things came out of it:

- **D-060** — "No papers found" confirmed as written and moved out of **Open**. Carries a
  milestone-5 consequence: that run streams nothing in `messages` mode.
- **D-061** — `Source` silently ignored unknown fields. Verified: `Source(**valid, abstract="x")`
  constructed fine and dropped `abstract`. Fail-fast hole at the parsing boundary; fixed with
  `extra="forbid"`.
- **D-062** — **the significant one.** `[arXiv:A, B]` and `[arXiv: A]` extracted *nothing*, so a
  review citing only in those forms reported `citation_violations == []` — identical to a fully
  grounded review. A false negative in the project's headline feature. `check_citations` now finds
  every `[arXiv:...]` bracket and records the ones it can't parse.

Reviewed and deliberately left alone: the bare `except ValueError` in `parse_feed`. It does catch
caller bugs alongside bad data, but `test_arxiv.py`'s `skipped == 0` assertion on the real fixture
is an adequate guard, and error-type introspection would couple production code to Pydantic
internals. Only the misleading comment was corrected.

Raised to **Open**, not fixed: `build_search_query` destroys non-ASCII terms
("Schrödinger" → `all:schr AND all:dinger`), which yields zero results and is recorded as a success.

**Written by Claude on 2026-09-19** (at your request: "complete what's left"). Reviewed 2026-09-20 —
outcome above. Kept here as the record of what wasn't yours:
- Fixes to your code:
  - `state.py` (field names and types, `default_factory`);
  - `config.py` (removed the copies of `arxiv.py` constants; added D-052's settings);
  - `models.py` (the ID pattern, D-056);
  - `arxiv.py` (`entry_to_source` passed `id=`/`abstract=`, so every entry was silently skipped; the
    link fallback that could pick the PDF was removed; `parse_feed` returns a tuple);
  - `check_citations.py` (`source.arxiv_id`; missing context raises `RuntimeError`, because an `assert`
    in a validator becomes a `ValidationError` and would record every citation as a violation);
  - `graph.py` (argument order `model_factory, http_client, checkpointer`; direct `httpx` import).
- New code: the `synthesize.py` prompt, `format_papers` and `NO_SOURCES_REVIEW` (D-046, D-055, D-060).
- Tests: every stub filled in, plus tests for D-058, D-059, the missing-context `RuntimeError`, the
  empty-search path, a failed search, and a control for the round-trip test. The `checkpointer`
  fixture now uses `build_serializer()`.

**Order:**
- [x] 1. `state.py` fields · `persistence/checkpointer.py`
- [x] 2. `models.py` → `test_source_model.py` green
- [x] 3. `arxiv.py` (`split_versioned_id` → `entry_to_source` → `parse_feed` → `search_arxiv`) → `test_arxiv.py` green
- [x] 4. `check_citations.py` → `test_check_citations.py` green
- [x] 5. `build_search_query` + arXiv settings + `search.py`
- [x] 6. `synthesize.py` prompt + `graph.py` wiring → update `test_graph.py`; write `test_checkpoint_roundtrip.py`
- [x] 7. Reviewed (2026-09-20) → D-060, D-061, D-062; tests for the last two written and red
- [x] 8. Applied the three `src/` fixes → `uv run pytest`: **66 passed, 1 deselected**
- [x] 9. Integration test run 2026-09-20 → **1 passed** in 11.17s, first attempt
- [x] 10. Committed as `60611ac` (milestone 2) and `2db3260` (docs ownership)

## Upcoming milestones

| # | Milestone | Open decisions to settle first (see `decisions.md` → Open) |
|---|---|---|
| 2 | One real source (arXiv), single subtopic, citation shape | **Done, ready to commit**: see the Milestone 2 section above (D-040 – D-062) |
| 3 | `decompose` + `Send` fan-out, reducers | arXiv rate limiter (≤1 req / 3 s) · models per role · treating HTTP 4xx and 5xx differently |
| 4 | `gap_check` + depth recursion, retry cap, paper overlap | making failures visible · `recursion_limit` value |
| 5 | Web layer: FastAPI + SSE + `AsyncSqliteSaver` | frontend (SvelteKit or htmx) · public entry function |

**Reminders for when these come up:**
- **Milestone 2 onward:** add every custom Pydantic model or dataclass stored in state to
  `allowed_msgpack_modules` (D-014), with a round-trip test (D-050).
- **Milestone 5:** a zero-result run streams **nothing** in `messages` mode, because D-060 returns
  `NO_SOURCES_REVIEW` without calling a model. The review arrives only in the `synthesize` `updates`
  chunk, so the SSE layer must render a review that never produced a token — otherwise an empty
  search looks like a hung UI.
- **Milestone 3:** `ValueError` from `build_search_query` isn't on the worker catch list (D-059).
  Decide whether a subtopic with no terms counts as a failed subtopic.
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
