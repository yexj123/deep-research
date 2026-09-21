# Progress

What's done, what's next, and whose job each item is. Design reasoning
lives in [`decisions.md`](decisions.md); this file only tracks status.

**Last updated:** 2026-09-21 · **Current milestone:** 5 (web layer: FastAPI + SSE + SQLite)

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
- [x] Re-ran the paid integration test against the milestone 4 graph — **1 passed** in 35.34s
- [x] Committed milestone 4 as `f34d0d8`
- [ ] **Decide D-022's fate** — its per-subtopic overlap check is superseded for control flow by
      D-075; retire it, or keep the ratio as thesis evidence (see Open)
- [ ] **`notebooks/`: measure ungrounded citations against depth** (D-079, Open) — needed before
      tuning `MAX_DEPTH` or `MAX_SUBTOPICS`, or those numbers are guesses

## Milestone 5: web layer

**Goal:** the agent reachable over HTTP — start a run, stream it to a browser, read it back —
on a real SQLite checkpoint file.

**Status (2026-09-21):** complete. `uv run pytest`: **207 passed, 1 deselected** (up from 148).
Decisions: D-080 – D-086, settling O-5, O-6, O-7 and O-8's XSS half. Backend `7112477`,
frontend `ac20b98`.

**Done:**
- [x] Dependencies: `fastapi`, `uvicorn[standard]`, `langgraph-checkpoint-sqlite`, `jinja2` —
      plus **`httpx` as an explicit dependency**, which `src/` has imported all along while only
      arriving transitively via `langchain-openai`
- [x] `persistence/checkpointer.py`: `open_checkpointer()` (D-082)
- [x] `persistence/runs.py`: the runs table, same SQLite file as the checkpoints (D-007)
- [x] `agent/runner.py`: `stream_run`, `get_run_state`, `get_review` — settles O-7 (D-081, D-084)
- [x] `api/main.py`: `create_app()` + lifespan owning all four graph dependencies
- [x] `api/routes/runs.py`: POST /runs, GET /runs, GET /runs/{id}, GET /runs/{id}/stream (D-083)
- [x] `state.py`: `citations_checked`, the terminal completion marker (D-084)
- [x] `tests/api/` (13) over `httpx.ASGITransport`; `tests/agent/test_runner.py` (6)

- [x] Frontend: `templates/index.html`, `_history.html`, `static/app.js`, `static/app.css`
- [x] `api/rendering.py` + `routes/pages.py` — server-rendered markdown, escaped (D-085)
- [x] Verified against a real server: page, assets and the history fragment all serve, and a
      question containing `<b>` renders escaped
- [x] Frontend committed as `ac20b98`
- [x] **O-5 — the run reports what it did not cover** (D-086): `agent/coverage.py`,
      `empty_subtopics` in state, live `custom` progress from the workers, and a
      "Coverage and limitations" panel
- [x] **Sidebar of past runs** (D-087): click to reopen a run; unfinished runs offer to start
      or resume, which D-081 makes cheap
- [ ] **Run it end to end in a browser** — the one thing tests cannot reach: that tokens
      appear progressively and the review renders. `/demo-check` covers this.

**Where htmx actually ended up.** D-080 expected it to handle the page, the form, history and
the progress trail. In practice the run flow needs `fetch` + `EventSource` anyway, so the form
and the trail are plain JS and **htmx handles the history list alone** (`hx-get` + a
`refresh-history` event). That is a smaller footprint than the decision anticipated. It still
earns its place — the alternative is hand-rolling a fetch-and-swap — but the honest framing is
"one useful helper", not "the frontend framework".

**O-5 closed the last instance of this project's recurring failure shape.** Of the three kinds
of loss, only one was even recorded: `failed_subtopics` had no reader, `skipped_entries` had no
reader, and a **zero-result subtopic was recorded nowhere at all** — D-021 marks it explored, so
it read identically to a productive one. A review missing a third of its subtopics looked
exactly like a complete one. The run now reports what it explored, what found nothing, what
failed and how often, how many entries were skipped, and **why it stopped** — because "the depth
limit cut this off" and "the search converged" mean very different things to a reader.

**Five things measured while building this:**
- **`get_stream_writer()` raises outside a runnable context** (`Called get_config outside of a
  runnable context`), which broke every direct-call worker test. Fixed with a no-op fallback,
  on principle rather than for convenience: progress reporting must never be able to break a
  run, and the worker's contract should stay testable without graph scaffolding.
- **`markdown-it-py`'s default is unsafe.** `MarkdownIt()` ships with `html=True`, so
  `<script>` in a review passes straight through to the browser. Since the review is written by
  a model that just read untrusted abstracts, that is a live XSS. Fixed with `html=False`
  (D-085), and pinned by tests precisely because it is a default being overridden.
- **`AsyncSqliteSaver.from_conn_string()` takes no `serde`** — using it silently discards the
  D-014 allowlist. Not a break today; a break later, announced only by a log line (D-082).
- **`next == ()` does not mean "finished"** — with the production stream modes, an interrupted
  run has an empty `next` too. D-081's discriminator was wrong; D-084 replaced it with a
  terminal marker. Resume itself works regardless.
- **A disconnect cannot be tested through `httpx.ASGITransport`** — it drives the response
  generator to completion regardless of the client breaking out. A test written there passed
  without exercising anything, so resume coverage lives in `test_runner.py` instead.

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

**What the milestone 4 integration runs showed (2026-09-21, `gpt-4o`, real API):** the recursion
works end to end — three rounds, ~9 subtopics, all workers succeeding, the streamed text equalling
the saved review, and `depth` stopping the run rather than `recursion_limit`.

Two runs, one finding, and a correction. Run A produced a **fabricated citation**
(`2113.11460` — month 13, not a possible arXiv ID) which `check_citations` caught: grounding
working exactly as D-046 designed, since the format check passes and the retrieval check fails.
That made the old `citation_violations == []` assertion wrong in kind — it treated the checker
succeeding as a code failure — so D-078 changed the test to assert the *checker* works and report
violations instead. Run B, identical in every respect, produced none. So the honest claim is
"ungrounded citations happen and vary run to run", not "hallucination rises with depth" — a trend
drawn from two consecutive runs that the third contradicted (D-079).

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
| 5 | Web layer: FastAPI + SSE + `AsyncSqliteSaver` | **Done, `ac20b98`** (D-080 – D-087) |
| 6 | **Evaluation harness** — a baseline before anything changes | **O-11** framework + a frozen question set |
| 7 | **Local-first corpus (BM25/FTS5)** | **O-13** `MIN_LOCAL_PAPERS` (measure with O-11) · full-text fetch trigger · staleness policy |
| 8 | **In-band claim checker** | **O-12** — after 7, since chunks give it tighter context |

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


---

## Baseline recorded (2026-09-21)

Ten questions, `gpt-4o`, abstracts only. Raw data in `tests/eval/recordings/`, scores in
`tests/eval/results.json`. Full analysis in D-090; the headline numbers:

| | |
|---|---|
| Papers supplied / cited | **797 / 81 — 10.2% used** |
| Cited per run | 6-12, median 8, **independent of how many were supplied** |
| Runs stopping on the depth ceiling | **10 of 10** |
| Faithfulness / relevancy | 0.970 ± 0.041 / 0.990 ± 0.020 |
| Ungrounded citations | 1 in 81 (~1.2%) |
| Planner JSON failures | 1 run in 10 died; succeeded on retry |

**What it changes.** Context pruning moves ahead of the corpus work: ~262k tokens were paid
for and unused across ten runs, and since the cited count doesn't scale with supply, a top-N
cut costs nothing measurable. The semantic exit is revealed as dead code — `depth` does 100%
of the work, inverting D-009 — and the cause is planner paraphrasing that
`normalize_subtopic` cannot catch, which resurrects D-022. And run-to-run variance is large
enough (4 of 18 cited papers shared between identical runs) that any O-13 comparison needs
repeats, not one run per arm.

## Next: measure before building (O-11 → O-13)

Designed 2026-09-21, not started. The sequencing is the point: **O-11 first**, because
without a baseline taken before the corpus work, "full text improved the reviews" is
unfalsifiable.

**The gap O-11 fills.** `citation_violations` measures citation *validity* exactly and for
free — most RAG projects have nothing that good. Nothing measures **claim support**: "Smith
showed X [arXiv:1234]" where the ID is real, the paper was retrieved, and the paper never says
X. That gap widens with retrieval, since a chunk can be topically adjacent without supporting
anything.

**The tension worth naming.** This project's habit is to prefer code it can defend line by
line over a dependency (D-042, D-083), which argues for a hand-written claim check. What
outweighs it here is **comparability**: a thesis number measured with a recognized framework
can be compared to published work; one from a bespoke judge invites "how do you know it's
right?".

**Already confirmed for O-13:** SQLite FTS5 is compiled into the bundled SQLite (3.49.1), so
lexical retrieval needs no dependency at all and lives in the same file as the checkpoints,
following D-007 rather than fighting it. And arXiv's terms explicitly contemplate **building
indexes** over full text; what they prohibit is *serving* PDFs. That keeps a local cache
legitimate for a single-user app (D-008) and means **the cache must go if this is ever
deployed for others** — a constraint to accept now rather than discover later.

**Sequenced before O-13:** O-8's nonce delimiter. A full paper is roughly 40× more
attacker-controllable text than an abstract, so the `</papers>` gap should close first.

### Milestone 6 shape (O-13, designed 2026-09-21)

A corpus that grows from the research actually done. Each **subtopic** — not each run — checks
the local corpus first and falls back to arXiv only when the corpus doesn't cover it, indexing
anything new on the way through. Putting the branch in `research_worker` rather than at the top
of the run keeps `decompose` intact and lets one run answer two subtopics locally while
fetching for a third.

**Two tiers, which is what makes the cold start bearable.** Abstracts are indexed for every
paper any search returns — free, since they're already retrieved. Full text is fetched only
for selected papers. Downloading a PDF for every result would be ~30 s per subtopic before a
word is written; this way the corpus has value from the first run, and today's
abstract-only behaviour remains the floor for comparison.

**The sufficiency test is the whole design, and similarity is the wrong metric for it.** Cosine
scores aren't calibrated, aren't comparable across queries, and — fatally — ten near-identical
chunks score beautifully while the field holds two hundred papers. A QA system can answer from
one good passage; a literature review cannot, because breadth *is* the product. So sufficiency
counts **distinct papers above a floor**, not chunk scores, mirroring D-028's ≥3-results rule.
Both thresholds get measured with O-11 rather than guessed — picking them by intuition would
be the `recursion_limit = 150` mistake again.

**Staleness is a correctness problem.** arXiv grows ~100 GB/month, so a corpus that answered
well in March silently misses April's key paper. The fix isn't to defeat the cache, it's to
report it: D-086's coverage gains "answered N of M subtopics from the corpus, indexed between
X and Y, no new search performed." Same principle as reporting *why* a run stopped.

**Retrieval is BM25 via SQLite FTS5, not embeddings** (revised 2026-09-21). Exact jargon is
what academic search runs on — verified that `'mamba'` and `'FlashAttention'` rank correctly
under BM25, which is precisely what a 384-dim model blurs. FTS5 is compiled into the bundled
SQLite, so there is no fastembed, no ONNX runtime and no model download. The strongest
argument is consistency: arXiv matches lexically, so a lexical local tier makes "the corpus
doesn't cover this" mean the same thing in both places — with embeddings locally and keywords
remotely, a local miss might mean only that two retrieval methods disagreed. Dense retrieval
is demoted to a **measured follow-on**, RRF hybrid to a decision after that.

*(`sqlite-vec` was recommended earlier partly because it was already installed. Installed is
not the same as warranted — recorded in O-13 as a reasoning error.)*

**Two behaviours locked down in O-13 because they invert silently:**

- `bm25()` returns **negative** scores, lower being better, so `ORDER BY bm25(t)` ascending is
  best-first and adding `DESC` returns the worst matches with no error. Retrieval converts to
  `relevance = -bm25(...)` at the boundary so nothing above it reasons about negatives. And
  **absolute score thresholds are unusable**: a term in more than half the corpus gets
  degenerate IDF and scores collapse toward zero regardless of match quality — measured,
  `model` at 25/30 docs scores −0.000 while `attention` at 3/30 scores −1.820. Sufficiency
  counts distinct *papers* instead.
- **Lowercasing is a security property.** FTS5's word operators are case-sensitive:
  `attention AND transformer` matched 2 documents as an operator, `attention and transformer`
  matched 0 as a plain term. The existing lowercasing therefore neutralizes `AND`/`OR`/`NOT`/
  `NEAR`, and someone preserving capitalization to be tidier would silently reintroduce
  operator injection. Punctuation stripping (D-051) removes the symbolic operators and stops
  an unterminated quote raising `OperationalError`.
