# Progress

What's done, what's next, and whose job each item is. Design reasoning
lives in [`decisions.md`](decisions.md); this file only tracks status.

**Last updated:** 2026-09-21 · **Current milestone:** 4 (`gap_check` + depth recursion)

**Who writes what:** you write the implementation (`src/`); Claude writes every test
(since 2026-09-19) and keeps the docs (since 2026-09-20) — `docs/*.md` and the README.
Backed by `CLAUDE.md`, the output style, and `Edit(/tests/**)`, `Edit(/docs/**)`,
`Edit(/README.md)` in `.claude/settings.json`.

---

## Do next (you)

- [ ] *(Optional)* Delete the old 55 MB uv cache on C::
      `uv cache clean --cache-dir "$env:LOCALAPPDATA\uv\cache"`
- [x] Fixed the four truncated comments in `state.py`
- [x] Deleted `nodes/search.py`, superseded by `research_worker`
- [ ] **Re-run the paid integration test** — milestone 4 changed the graph again, so the
      2026-09-20 run no longer covers it. Now makes up to `MAX_DEPTH + 2` planner calls plus the
      review: `uv run pytest -m integration`
- [ ] Commit milestone 4, then record the hash here

## Milestone 4: `gap_check` + depth recursion

**Goal:** the run recurses — `gap_check` decides whether another round is worth it, `depth`
bounds it, and `recursion_limit` is a backstop that should never fire.

**Status (2026-09-21):** code complete, not committed. `uv run pytest`:
**148 passed, 1 deselected** (up from 131). Decisions: D-075, D-076, D-077 (settling O-4).

**Done:**
- [x] `nodes/gap_check.py` — increments `depth` and nothing else (D-076)
- [x] `graph.py`: `route_after_gap_check` and the cycle back to `decompose` (D-075)
- [x] `state.py`: `seen_before_round`, the per-round baseline (D-075)
- [x] `decompose` writes `seen_before_round` at the start of each round
- [x] `config.py`: `MAX_DEPTH = 2`, `RECURSION_LIMIT = 15`
- [x] `test_gap_check.py` (8) and `test_recursion.py` (6); `test_integration.py` now passes
      `recursion_limit` in its config

**Measured 2026-09-21:**
- **`recursion_limit` minimum is 13**, found by bisection on the real graph shape — 12 raises
  `GraphRecursionError`. `RECURSION_LIMIT = 15` leaves two steps of headroom, and a test asserts
  a full-depth run fits. The skill previously advised 150, ~11× the real need.
- **The limit is independent of `MAX_SUBTOPICS`:** a `Send` fan-out is one super-step however
  wide. More subtopics cost wall-clock (D-064's rate limiting), never recursion budget.

**The behaviour change that rippled through the tests:** a productive round *always* triggers a
second `decompose` call — `gap_check` routes back, and only then does the explored filter empty
the plan and end the run. Every test scripting one plan broke, because the planner received the
review prose. `one_round_replies()` in `fakes.py` now encodes that property in one place.

**Known follow-up:** D-022's ≥60% per-subtopic overlap check is now superseded for control flow
(D-075 does the job at round level). Under Open: retire it, or keep the ratio as thesis evidence.

## Milestone 3: `decompose` + `Send` fan-out

**Goal:** the planner proposes subtopics, `Send` dispatches one worker per subtopic in parallel,
and the reducers merge their results without losing or double-counting anything.

**Status (2026-09-20):** done, committed as `b4a4dff`. `uv run pytest`: **131 passed, 1 deselected**
(up from 66 at milestone 2). Decisions: D-064 – D-074.

**Done:**
- [x] `state.py`: `pending_subtopics`, `explored_subtopics`, `failed_subtopics`, `seen_paper_ids`,
      `sources`, `skipped_entries`, `depth`, with the reducers from D-067 → `test_reducers.py` (14)
- [x] `sources/rate_limit.py`: `ArxivRateLimiter` (D-064), passed into `build_graph`
- [x] `nodes/research_worker.py`: one arXiv search per subtopic, `SubtopicTask` payload (D-071),
      catch list per D-072 and D-065 → `test_research_worker.py` (14)
- [x] `nodes/decompose.py`: planner + the three hard filters (D-070, D-073) → `test_decompose.py` (16)
- [x] `graph.py`: `route_subtopics` conditional edge (D-069), rewired
      `intake → decompose → Send → research_worker → synthesize → check_citations`
- [x] `config.py`: `MAX_SUBTOPICS = 3`, `ARXIV_MIN_INTERVAL_SECONDS = 3.0`
- [x] Fan-out behaviour and reducers under parallel writes → `test_fanout.py` (14)
- [x] `test_graph.py` updated for the milestone 3 contract (node order, the limiter argument,
      two scripted model replies)

- [x] Integration test re-run against the milestone 3 graph (see below)
- [x] Committed as `b4a4dff`

**What the milestone 3 integration run proved (2026-09-20, `gpt-4o`, 16.58s):** a real planner
decomposed the question into more than one subtopic; every subtopic's worker succeeded
(`explored_subtopics == pending_subtopics`, `failed_subtopics` empty); the review streamed in
several `messages` chunks and the joined text equalled the saved review exactly; at least one
`[arXiv:<id>]` citation was present and `citation_violations == []`. Run with the **real**
`ArxivRateLimiter`, so arXiv's 1-request-per-3-seconds rule was honoured. Same caveat as milestone
2: one run shows the behaviour is achievable, not that it is reliable.

**Three things measured while building this** (all in `decisions.md`):
- **An empty `Send` list ends the run silently** — no error, no downstream node, no review.
  Reachable whenever every proposal is already explored, hence D-069's guard.
- **`Send` payloads are checkpointed**, so a custom payload class comes back as a plain `dict`
  on *resume* with only a logged warning (D-071). Invisible to any start-to-finish test.
- **A token-bucket limiter allows 2 concurrent arXiv connections**; arXiv permits one (D-064).

**Known follow-up:** `nodes/search.py` is now unused — `research_worker` replaced it. Delete it and
its `code-map.md` entry, or keep it deliberately.

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
| 3 | `decompose` + `Send` fan-out, reducers | **Done, `b4a4dff`** (D-064 – D-074) |
| 4 | `gap_check` + depth recursion, retry cap, paper overlap | **O-4** `recursion_limit` value · **O-5** making failures visible |
| 5 | Web layer: FastAPI + SSE + `AsyncSqliteSaver` | **O-6** frontend (htmx or SvelteKit) · **O-7** public entry function |

Every open item now carries options, tradeoffs and a recommendation in
[`decisions.md` → Open](decisions.md#open-proposed-not-decided), numbered `O-1` … `O-10`.
Not milestone-gated: **O-8** prompt injection · **O-9** accent spellings · **O-10** non-English stopwords.

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
