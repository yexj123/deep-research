# Code map

What each file does, what it uses, and what uses it. Built from the actual imports on
2026-09-21. Design reasons are in [`decisions.md`](decisions.md) (the `D-` numbers).

**Status tags**
- **[M5]**: new in milestone 5 (the web layer)
- **[M4]**: added in milestone 4
- **[M3]**: added in milestone 3
- **[M1]**: implemented and tested (milestone 1)
- **[M2]**: added in milestone 2
- **[M1 → M2]** / **[M1 → M3]**: from an earlier milestone, changed since

`agent/nodes/search.py` was **deleted** at milestone 3: `research_worker` replaced it with the same
search, one per subtopic and with a catch list.

---

## 1. The big picture: four layers

```text
  web              api/main.py (create_app + lifespan) · api/routes/runs.py [M5]
                     │ owns the dependencies for the process, formats SSE
                     ▼
  entry point      agent/runner.py [M5]  (stream_run: start / resume / replay)
                     │
                     ▼
  callers          tests/  · api/
                     │ build the dependencies and call build_graph(...)
                     ▼
  wiring           agent/graph.py
                     │ registers the nodes and connects them
                     ▼
  nodes            agent/nodes/intake.py · decompose.py · research_worker.py · gap_check.py
                   synthesize.py · check_citations.py
                     │ read state, return partial updates
                     ▼
  building blocks  agent/llm.py (chat models) · agent/sources/arxiv.py (arXiv client and parser)
                   agent/sources/rate_limit.py (ArxivRateLimiter, D-064)
                     │
                     ▼
  definitions      agent/context.py · agent/config.py · agent/state.py · agent/sources/models.py
                   persistence/checkpointer.py (serializer settings)
```

**Rules the code follows:**
1. **Imports only point downward.** A definition never imports a node, and a node never imports `graph.py`.
2. **Nodes never import each other.** Only `graph.py` knows the order they run in.
3. **Nodes never create their own dependencies.** The model factory, HTTP client and checkpointer are
   created by the caller and passed into `build_graph` (D-032, D-049).
4. **Only `llm.py` constructs real LLM clients, and only `arxiv.py` talks to arXiv.** That's why tests
   can swap in fakes at exactly those two points.

---

## 2. Import map

`A --> B` means **A imports B**.

```mermaid
flowchart TD
    graph["agent/graph.py"]
    intake["nodes/intake.py"]
    synth["nodes/synthesize.py"]
    decomp["nodes/decompose.py [M3]"]
    worker["nodes/research_worker.py [M3]"]
    gap["nodes/gap_check.py [M4]"]
    ratelim["sources/rate_limit.py [M3]"]
    cites["nodes/check_citations.py [M2]"]
    llm["agent/llm.py"]
    arxiv["sources/arxiv.py [M2]"]
    config["agent/config.py"]
    context["agent/context.py"]
    state["agent/state.py"]
    models["sources/models.py [M2]"]
    ckpt["persistence/checkpointer.py [M2]"]

    exits["agent/exits.py [D-096]"]
    graph --> exits
    exits --> config
    graph --> context
    graph --> llm
    graph --> intake
    graph --> decomp
    graph --> worker
    graph --> gap
    graph --> ratelim
    graph --> synth
    graph --> cites
    graph --> state
    intake --> context
    intake --> state
    decomp --> arxiv
    decomp --> config
    decomp --> context
    decomp --> llm
    decomp --> state
    worker --> arxiv
    worker --> config
    worker --> ratelim
    gap --> state
    synth --> context
    synth --> llm
    synth --> models
    synth --> state
    cites --> state
    llm --> config
    llm --> context
    config --> context
    arxiv --> models
    state --> models
    ckpt --> models
```

The last arrow matters: `persistence/checkpointer.py` builds its allowlist entry from the `Source` class
(`Source.__module__`, `Source.__name__`), so a move or rename is picked up automatically (D-057).
`test_checkpoint_roundtrip.py` still matters: it catches a *new* state type that was never added (D-014, D-050).

---

## 3. File by file: `src/deep_research/`

### `__init__.py` [M1]
The package itself, holding only a docstring. `tests/test_smoke.py` imports it to prove the package is installed.

### `agent/context.py` [M1]
**Defines:** `ProviderType = Literal["openai", "deepseek"]` and `RunContext` (a dataclass with `provider`).
**Uses:** nothing from the project.
**Used by:**
- `config.py`: `ProviderType` is the key type of its dicts.
- `llm.py`: the factory takes a `ProviderType`.
- `nodes/intake.py`: builds `VALID_PROVIDERS` from `ProviderType` with `get_args`, and checks `runtime.context`.
- `nodes/synthesize.py`: reads `runtime.context.provider` to choose the model.
- `graph.py`: passes `context_schema=RunContext` to `StateGraph`.
- tests: pass `context=RunContext(provider=...)` on every run.

**Why it's separate from state:** the context is per-run input that isn't checkpointed; state is research
data that is (D-015, D-032).

### `agent/config.py` [M1 → M4]
**Defines:**
- `API_KEY_ENV_VARS`: which environment variable holds each provider's key;
- `MODEL_NAMES`: which model each provider uses;
- `LLM_TIMEOUT_SECONDS = 60.0` and `LLM_MAX_RETRIES = 2` (D-039).

**Uses:** `context.ProviderType`.
**Used by:**
- `llm.py` uses the four LLM constants;
- `tests/agent/test_llm.py` imports `API_KEY_ENV_VARS` and the two limits, to check the built clients against the same values.

**Milestone 2 added:**
- `ARXIV_MAX_RESULTS = 10`, used by `nodes/research_worker.py`;
- `ARXIV_TIMEOUT_SECONDS = 30.0`, used by whoever creates the `httpx.AsyncClient`: the integration test now, the web layer later (D-052).

**Milestone 3 added:**
- `ARXIV_MIN_INTERVAL_SECONDS = 3.0`, used by whoever creates the `ArxivRateLimiter` (D-064);
- `MAX_SUBTOPICS = 3`, used by `nodes/decompose.py` for both the prompt and the filter cap (D-070).

**Milestone 4 added:**
- `MAX_DEPTH = 2`, used by `graph.route_after_gap_check` — 0-indexed, so 3 search passes (D-026).
  Since D-096 this is a **ceiling, not a target**: the adaptive exits normally stop a run after
  one round, so most runs never reach it;
- `EMPTY_ROUND_RATIO = 0.5` [D-096, D-098], used by `exits._came_back_empty`. **The only
  constant in this file not backed by a recorded experiment**, and named rather than inlined
  so that is visible; the bounds it must stay within are asserted in `test_adaptive_exit.py`;
- `RECURSION_LIMIT = 15`, used by the **caller** as invoke config. The graph never reads it: leave
  it out of the config and LangGraph silently uses its default of 25 (D-077).

### `agent/state.py` [M1 → M4]
**Also defines the reducers** (D-067), kept beside the fields they serve so one file explains the
whole state contract: `normalize_subtopic` (casefold + strip, D-017), `merge_subtopics` (dedup on
the normalized form, storing the original text) and `merge_sources` (dedup on `arxiv_id`, keep
first). Both copy before appending -- a reducer must never mutate its left argument.

**Milestone 4 fields:** `depth` (rounds completed, written by `gap_check`, D-076) and
`seen_before_round` (**no reducer**, overwritten by `decompose` — the baseline `gap_check`
compares against to learn what a round added, D-075).

**D-096 field:** `empty_before_round` (**no reducer**, overwritten by `decompose` alongside
`seen_before_round`). `empty_subtopics` accumulates across rounds, so the total cannot answer
"did *this* round come back empty" — without the baseline, two harmless dead ends in early
rounds would latch the exit on permanently.

**Milestone 3 fields:** `pending_subtopics` (**no reducer**, overwritten each round),
`explored_subtopics`, `failed_subtopics` (`operator.add`, duplicates are the attempt count),
`seen_paper_ids` (`set[str]`, `operator.or_`), `depth`. `sources` and `skipped_entries` gained
reducers because `research_worker` writes them in parallel.

**Defines:** `ResearchState`, the graph's state dataclass:
- `question` (no default, because it's the required input);
- `review = ""`;
- `sources: list[Source]`, `skipped_entries: int = 0` and `citation_violations: list[str]` (D-054). The
  list fields use `field(default_factory=list)`. `citation_violations` holds *every* citation that
  couldn't be verified: unknown IDs, and unparseable markers quoted verbatim (D-062).

**Uses:** `Source` from `sources/models.py`.
**Used by:**
- every node, which reads it;
- `graph.py`, which passes it to `StateGraph(ResearchState, …)`;
- `test_graph.py`, which checks that `.value` is a `ResearchState`;
- `test_check_citations.py`, which builds one directly.

### `agent/llm.py` [M1]
**Defines:**
- `ModelFactory`: the type `Callable[[ProviderType], BaseChatModel]`;
- `get_chat_model(provider)`: the real factory.

**Uses:**
- the four constants from `config.py`: env var name → key, model name, timeout, retries;
- `context.ProviderType`;
- `ChatOpenAI` and `ChatDeepSeek`.

**Used by:**
- `nodes/synthesize.py` and `graph.py` import **only the `ModelFactory` type**, never `get_chat_model`;
- `test_integration.py` passes `get_chat_model` into `build_graph`;
- `test_llm.py` calls it directly.

**Key point:** nodes never call `get_chat_model` themselves. The caller chooses the factory, so tests
pass `RecordingFactory` instead and no test ever needs a key (D-029, D-032).

### `agent/nodes/intake.py` [M1]
**Defines:** `intake(state, runtime)`, a plain `def` because it does no I/O.
**Uses:** `context.ProviderType` / `RunContext`, `state.ResearchState`.
**Reads:** `state.question`, `runtime.context`. **Writes:** `question` (stripped).
**Raises** a `ValueError` starting `"intake: …"` for a missing context, an invalid provider or a blank question (D-033).
**Registered by:** `graph.py`, as the node named `"intake"`.

### `agent/nodes/decompose.py` [M3]
**Defines:** `SYSTEM_PROMPT` (asks for JSON), `SubtopicPlan` (the Pydantic reply model),
`MAX_FAILURES = 2`, `_is_searchable`, `_keep_worth_researching`, and `make_decompose(model_factory)`.
**Reads:** `state.question`, `state.explored_subtopics`, `state.failed_subtopics`,
`runtime.context.provider`. **Writes:** `pending_subtopics`.
**The division of labour:** the planner *proposes*; this node *decides*. The explored list in the
prompt is a soft filter (D-022); the hard filters are the normalized-match check, the N=2 retry cap
counted on normalized subtopics (D-020), and dropping subtopics `build_search_query` can't use
(D-059, D-073). Validation is explicit (`model_validate_json`) and catches nothing, so an
unparseable plan fails the run (D-070).
**Registered by:** `graph.py`, as `"decompose"`.

### `agent/nodes/research_worker.py` [M3]
**Defines:** `SubtopicTask` (a **`TypedDict`** — `Send` payloads are checkpointed, so a custom class
would come back as a plain `dict` on resume, D-071), `EXTERNAL_FAILURES`, `_is_external_status`, and
`make_research_worker(http_client, limiter)`.
**Reads:** its `Send` payload only — a worker never sees full state.
**Writes (success):** `sources`, `skipped_entries` (its own delta), `explored_subtopics`,
`seen_paper_ids`. **Writes (failure):** `failed_subtopics` only — never explored (D-018).
**Catch list:** `httpx.TransportError`, `pydantic.ValidationError`, `ParseError`,
`DefusedXmlException`, `ArxivAPIError`; plus `HTTPStatusError` **only** for 429/5xx — every other
4xx re-raises, because it means we sent something wrong (D-065).
Every request runs inside `async with limiter:` (D-064).
**Registered by:** `graph.py`, as `"research_worker"`, reached only via `Send`.

### `agent/nodes/gap_check.py` [M4]
**Defines:** `gap_check(state)`, a plain `def` that returns `{"depth": state.depth + 1}` and
nothing else.
**Reads:** `state.depth`. **Writes:** `depth`.
**What it is not:** a gap finder. `decompose` finds gaps — it receives `explored_subtopics` and is
asked for something new (D-070). This node only counts the round, and calls no model. The routing
decision lives in `graph.py`'s `route_after_gap_check`, because which node runs next is ordering
knowledge (rule 2 above).
**Registered by:** `graph.py`, as `"gap_check"`.

### `agent/sources/rate_limit.py` [M3]
**Defines:** `ArxivRateLimiter(min_interval)`, an async context manager holding an
`asyncio.Semaphore(1)` **across** the request plus monotonic-clock spacing (D-064).
**Uses:** stdlib only. **Used by:** `nodes/research_worker.py`, and whoever builds the graph.
**Why not a token bucket:** one paces request *starts*, which allows 2 concurrent connections when a
request outlasts the interval (measured). arXiv permits one. Note the semaphore binds to the first
event loop that uses it, so the web layer must create it inside the app lifespan.

### `agent/nodes/synthesize.py` [M1 → M2]
**Defines:**
- `new_fence()`: a per-run `secrets.token_hex(8)` delimiter token (D-097);
- `system_prompt(fence)`: requires citations in exactly the form `[arXiv:<arxiv_id>]` (D-046), and
  names the fenced block as data, not instructions (D-055, D-097). Replaced the old
  `SYSTEM_PROMPT` constant, which could not carry a per-run token;
- `format_papers(sources, fence="")`: the papers block, one `[arXiv:<id>] <title>` line plus
  abstract per paper. **The unfenced form is a compatibility contract** -- `tests/eval/test_record.py`
  builds `retrieval_context` with it and all 80 recordings contain that text (D-088);
- `NO_SOURCES_REVIEW`: returned without calling the model when `sources` is empty (D-060);
- `make_synthesize(model_factory)`, which returns the async `synthesize(state, runtime)` node.

**Uses:** the `ModelFactory` type from `llm.py`, `context.RunContext`, `state.ResearchState`, `sources/models.Source`.
**Reads:** `state.question`, `state.sources`, `runtime.context.provider`. **Writes:** `review`.
Its model call streams tokens in `messages` mode automatically; the node doesn't stream them itself.
**Registered by:** `graph.py`, as `"synthesize"`, via `make_synthesize(model_factory)`.

### `agent/nodes/check_citations.py` [M2]
**Defines:**
- `CITATION_BRACKET`, the regex for *any* `[arXiv:...]` bracket;
- `CITATION_MARKER`, the stricter regex for a well-formed `[arXiv:<id>]`;
- `Citation`, a Pydantic model whose validator checks the ID against `known_ids` passed in as validation context;
- `check_citations(state)`, a plain `def`.

**Uses:** `state.ResearchState`, Pydantic.
**Reads:** `state.review`, `state.sources`. **Writes:** `citation_violations`.
**How it reads the review (D-062):** it finds every `CITATION_BRACKET`, then tries to parse each with
`CITATION_MARKER`. A bracket that parses is validated against the retrieved IDs; one that doesn't is
recorded verbatim. So a citation it can't read is never mistaken for a verified one.
**Its contract with `synthesize`:** the prompt defines the one canonical marker format; anything else
is reported rather than accepted (D-046, D-062).
**Registered by:** `graph.py`, as `"check_citations"`.

### `agent/coverage.py` [M5]
**Defines:** `Coverage` (explored / empty / failed / skipped_entries / papers / rounds /
`stopped_because`, plus `is_complete`) and `summarize_coverage(values)`.
**Why it exists (settles O-5):** three kinds of loss were invisible — failed subtopics and
skipped entries had no reader, and a zero-result subtopic was recorded nowhere, so it read
exactly like a productive one (D-021). A review missing a third of its subtopics looked complete.
**In the agent, not `api/`:** "what did this run fail to cover" is a research fact the thesis
notebooks want. Pure functions over state, so it tests with no graph and no HTTP.
**Used by:** `api/routes/runs.py`, `api/rendering.py`, `tests/agent/test_coverage.py`.

### `agent/exits.py` [D-096]
**Defines:** `exit_reason(...)` — which of four conditions ends a run, or `None` to continue —
plus `REASONS` and `describe()` for the sentence a reader sees.
**Uses:** `config.MAX_DEPTH`, `config.SYNTHESIS_TOP_N`. Nothing else, so both callers can
import it without either importing the other.
**Used by:** `graph.route_after_gap_check` (routes on it) and `coverage._stop_reason`
(reports it).

**Why it is its own module.** The rule and its explanation lived in two places and drifted
twice: D-094 (coverage compared against an unpatched `MAX_DEPTH`, so ceiling stops were
recorded as convergence) and D-096 (coverage knew nothing of the two new exits, so a full
prompt was reported as convergence). Neither crashed; both produced a confident wrong sentence
in the panel readers use to judge a review. Taking primitives rather than `ResearchState` is
what lets `graph.py` pass dataclass fields and `coverage.py` pass a checkpoint dict while
imports still point downward (rule 1).

### `persistence/corpus.py` [D-100]
**Defines:** `SCHEMA` (papers, chunks, external-content `chunks_fts`, sync triggers),
`connect`, `index_sources`, `search`, `covering_papers`, `load_sources`, `stats`, and the
`ABSTRACT` / `FULL_TEXT` tier constants.
**Uses:** `sources/models.Source`, `sqlite3`. No new dependency — FTS5 is in the bundled
SQLite (3.49.1).
**Used by:** `nodes/research_worker.py` (indexes each search's results, D-101) and
`api/main.py`'s lifespan, which opens it on the same file as the checkpoints (D-007). Nothing
*reads* it yet -- local-first retrieval is the next increment.

**Two things to know before editing it.** `bm25()` returns the *negative* of the standard
score, so `ORDER BY bm25(t)` ascending is best-first and `DESC` silently returns the worst
matches; `search` converts at the boundary so nothing above it sees a negative. And `section`
is `NOT NULL DEFAULT ''` because SQLite allows unlimited NULLs in a unique index — nullable
would let the same abstract be indexed twice and inflate its own BM25 term frequencies.

### `agent/runner.py` [M5]
**Defines:** `RunState` (NOT_STARTED / INTERRUPTED / FINISHED), `get_run_state`, `stream_run`,
`get_review`, `STREAM_MODES`.
**Why it exists (settles O-7):** everything a caller must get right lives here rather than in a
route — the run context (D-033), `recursion_limit` (D-077), the stream modes, and the
start/resume/replay decision (D-081). In the agent package, not `api/`, so the decision is
testable without HTTP and usable from a script.
**The trap it encodes:** `next == ()` does **not** mean finished — an interrupted run has an empty
`next` too under the production stream modes. `citations_checked`, written only by the terminal
node, is the reliable signal (D-084).
**Used by:** `api/routes/runs.py`, `tests/agent/test_runner.py`.

### `api/main.py` [M5]
**Defines:** `create_app(...)` and the `lifespan` context manager. `DEFAULT_DB_PATH`.
**Owns the four graph dependencies for the process** (D-032, D-049, D-064). They are built in the
lifespan, not at import: the HTTP client and SQLite connection are context managers, and the rate
limiter's semaphore binds to the first event loop that touches it.
**Why a factory, not a module-level `app`:** tests inject a fake model factory, an
`httpx.MockTransport` client and a zero-delay limiter without monkeypatching. Run it with
`uvicorn deep_research.api.main:create_app --factory`.

### `api/routes/runs.py` [M5]
**Defines:** `CreateRun` / `RunCreated` (Pydantic at the boundary, D-013), `_sse`, `_event_for`,
and the four routes: `POST /runs`, `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/stream`.
**`_event_for` is deliberately not a passthrough:** raw `updates` chunks carry `Source` objects,
which are not JSON-serializable, and the browser has no use for full state. It emits `node`,
`progress`, `token` and `done` events instead — and filters `messages` to `synthesize`, or the
planner's JSON reaches the user's review pane (D-080).
**SSE is hand-rolled** (D-083): `event: <type>
data: <json>

` in a `StreamingResponse`.

### `persistence/runs.py` [M5]
**Defines:** `Run`, `init_runs_table`, `record_run`, `get_run`, `list_runs`.
**Why:** a checkpoint holds a run's *state*, but nothing records a run before it has executed a
node — and `POST /runs` returns a thread_id without running anything (D-081). Same SQLite file as
the checkpoints (D-007).

### `agent/graph.py` [M1 → M4]
**Defines:** `build_graph(model_factory, http_client, limiter, checkpointer)` (D-032, D-049, D-064),
`route_subtopics(state)` and `route_after_gap_check(state)`. Wiring: START → `intake` →
`decompose` → *(conditional)* → `research_worker` (one per subtopic, via `Send`) → `gap_check` →
*(conditional)* back to `decompose` **or** on to `synthesize` → `check_citations` → END.
`route_after_gap_check` continues only when `depth` is not spent **and** the round added a paper
not already in `seen_paper_ids` (D-075). It reads the depth `gap_check` just incremented: a
conditional edge sees the node's update already applied.
`route_subtopics` is **not a node**: it is the conditional edge out of `decompose`. It returns the
node name `"synthesize"` when `pending_subtopics` is empty, and `Send` objects otherwise — an empty
`Send` list ends the run silently, with no error and no review (D-069).
**Uses:** `ResearchState`, `RunContext`, the `ModelFactory` type, `httpx`, `intake`, `make_search`,
`make_synthesize`, `check_citations`.
**Used by:** `test_graph.py`, `test_checkpoint_roundtrip.py`, `test_integration.py`, and later the web layer.
Node names are part of the streaming contract: they appear as `metadata["langgraph_node"]` and as
`updates` keys (see `langgraph-outputs.md`).

### `agent/sources/models.py` [M2]
**Defines:**
- `ARXIV_ID_PATTERN`;
- `Source`, a frozen Pydantic dataclass: `arxiv_id`, `version`, `title`, `authors`, `summary`, `published`, `url` (D-043, D-044), with the ID validator (D-041, D-056) and the version and title validators (D-058). Declared with `config=ConfigDict(extra="forbid")`, so an unknown keyword raises instead of being silently dropped (D-061).

**Uses:** Pydantic only. It depends on nothing else in the project, so any module can import it.
**Used by:**
- `sources/arxiv.py`, which builds `Source`s;
- `state.py` (the `sources` field) and `nodes/synthesize.py` (`format_papers`);
- `persistence/checkpointer.py`, which builds its allowlist entry from the class (D-057);
- `tests/agent/fakes.py` (`make_source`).

### `agent/sources/arxiv.py` [M2]
**Defines:**
- `ARXIV_API_URL`, `NAMESPACES`, `STOPWORDS`, `PUNCTUATION_PATTERN`, `_ENTRY_ID_PATTERN`;
- `ArxivAPIError`, `ArxivSearchResult` (`sources` is a tuple);
- `build_search_query` (D-051, D-059), `split_versioned_id`, `entry_to_source` (D-058), `parse_feed`, `search_arxiv`.

**Uses:** `sources/models.Source`, `httpx`, `defusedxml`.
**Used by:** `nodes/research_worker.py`, `nodes/decompose.py` (`build_search_query`, for the
searchability filter, D-073), `test_arxiv.py`.
**Call chain inside the file:**
1. `search_arxiv` makes the HTTP GET and calls `raise_for_status`, then `parse_feed`;
2. `parse_feed` runs `defusedxml` with `forbid_dtd=True`, checks for the error feed, then calls `entry_to_source` once per entry;
3. `entry_to_source` calls `split_versioned_id`, then `Source(...)`, which runs the validation.

`search_arxiv` receives its `client` as a parameter and never creates one (D-049).

### `persistence/checkpointer.py` [M2 → M5]
**Milestone 5 added** `open_checkpointer(db_path)`: builds `AsyncSqliteSaver(conn, serde=...)` by
hand and calls `setup()`. **Never `from_conn_string()`** — it takes no `serde`, so it silently
discards the allowlist that D-014 exists to enforce (D-082).

**Defines:** `ALLOWED_MSGPACK_MODULES` (the custom state types as `(module, class)` pairs, built from
each class, D-057) and `build_serializer()`, which turns on strict mode.
**Uses:** `JsonPlusSerializer` from LangGraph, `sources/models.Source`.
**Used by:** `tests/agent/conftest.py` (the `checkpointer` fixture), `test_checkpoint_roundtrip.py`, and
the web layer's `AsyncSqliteSaver` at milestone 5.
Every checkpointer must be created with `serde=build_serializer()`. Otherwise the allowlist isn't applied.

---

## 4. File by file: `tests/`

| File | What it is | Uses |
|---|---|---|
Claude writes and maintains every file here (since 2026-09-19; see `CLAUDE.md`).

| File | What it is | Uses |
|---|---|---|
| `agent/conftest.py` | Fixtures: `fake_factory`, `checkpointer` (a fresh `InMemorySaver` with `serde=build_serializer()`), `arxiv_ok` (an arXiv stub serving `search_ok.xml`, closed after the test) | `fakes.py`, `persistence/checkpointer.build_serializer` |
| `agent/fakes.py` | Test doubles: `RecordingFactory` (stands in for `get_chat_model` and records providers; its default reply cites `2411.18583`), `make_arxiv_stub` (an `httpx.MockTransport` client that records requests), `load_arxiv_fixture`, `make_source` | `context.ProviderType`, `sources/models.Source`, `httpx`, LangChain's `GenericFakeChatModel` |
| `agent/fixtures/arxiv/*.xml` | 3 real arXiv responses + 4 derived ones (origins in the README there) | read by `load_arxiv_fixture` |
| `test_smoke.py` | Is the package importable? | `deep_research` |
| `agent/test_graph.py` [M1 → M2] | The whole graph with the fake model and the arXiv stub: streaming, node order, the query sent, sources and violations in state, the empty search, a failed search, final state, checkpoint, provider routing, `intake` validation | `build_graph`, `RunContext`, `ResearchState`, `NO_SOURCES_REVIEW`, `fakes` |
| `agent/test_llm.py` | The factory: missing keys, client type, explicit limits | `get_chat_model`, `config` constants |
| `agent/test_integration.py` [M1 → M2] | **Opt-in, paid**: one real OpenAI + arXiv run. The streamed text must equal the saved review, with at least one citation and no violations | `build_graph`, `get_chat_model`, `RunContext`, `ARXIV_TIMEOUT_SECONDS`, `CITATION_MARKER` |
| `agent/test_source_model.py` [M2] | `Source` validation: ID formats, tuple conversion, frozen, version and title (D-058), and that an unknown field raises rather than being dropped (D-061) | `make_source` |
| `agent/test_arxiv.py` [M2] | Parser, query builder and client on the saved responses, with no network | `sources/arxiv.py`, `fakes` |
| `agent/test_check_citations.py` [M2] | Citation grounding, calling the node directly; missing context fails loudly; unparseable markers are recorded, ordinary brackets aren't, and the no-bracket blind spot is pinned as a known limit (D-062) | `check_citations`, `Citation`, `ResearchState`, `make_source` |
| `agent/test_reducers.py` [M3] | The merge reducers as pure functions: dedup keys, first-seen order, and that neither mutates its left argument (D-067) | `state.py` reducers, `make_source` |
| `agent/test_decompose.py` [M3] | The planner's three hard filters, JSON validation, and `route_subtopics`' empty-plan guard (D-069, D-070, D-073) | `make_decompose`, `route_subtopics`, `fakes` |
| `agent/test_research_worker.py` [M3] | The worker called directly: success contract, the 429/5xx-vs-4xx split, and that a failure never marks a subtopic explored (D-065, D-072) | `make_research_worker`, `fakes` |
| `api/test_runs.py` [M5] | The routes over `httpx.ASGITransport`: create, list, stream, replay, 404s, and that every streamed event is JSON-serializable (D-081, D-083) | `create_app`, `fakes` |
| `agent/test_runner.py` [M5] | start / resume / replay as a unit. Here rather than in `api/` because ASGITransport drives the response generator to completion, so an HTTP-level disconnect test would pass without exercising anything (D-084) | `stream_run`, `get_run_state` |
| `agent/test_coverage.py` [M5] | The coverage summary: what was lost, attempt counts normalized like the retry cap, and that every run reports why it stopped (O-5, D-086) | `summarize_coverage` |
| `api/test_rendering.py` [M5] | Model-authored markdown must not become live HTML: raw HTML escaped, dangerous link schemes not linkified, real formatting still works (D-085) | `render_review` |
| `api/test_pages.py` [M5] | The page and its assets are served, and the history fragment escapes the question (D-080, D-085) | `create_app` |
| `agent/test_gap_check.py` [M4] | The stopping rule as a pure function: depth accounting, and the two exits (D-075, D-076) | `gap_check`, `route_after_gap_check` |
| `agent/test_papers_fence.py` [D-097] | The `<papers>` block cannot be closed by its own contents: hostile titles and abstracts, the fence being unguessable and per-run, the prompt naming the same token, and the *unfenced* form staying byte-identical for the eval recorder | `format_papers`, `new_fence`, `system_prompt` |
| `agent/test_worker_indexing.py` [D-101] | The worker seeds the corpus, idempotently across subtopics; `corpus=None` behaves exactly as before; a broken corpus does **not** fail the subtopic but **is** reported on the progress stream; and a TypeError in indexing still crashes | `make_research_worker`, `persistence/corpus` |
| `agent/test_corpus.py` [D-100] | Indexing, BM25 retrieval and the traps: ascending-is-best-first, relevance never negative, idempotent re-indexing, sufficiency counting distinct papers rather than chunks, the tier filter, and six hostile subtopics that cannot break the query — with a control proving the raw form would have raised | `persistence/corpus`, `build_fts_query` |
| `agent/test_exits.py` [D-096] | That routing and reporting can never disagree: each of the four exits fires on a state built for it, the router stops whenever any fires, and the coverage panel names the one that actually did. Exists because that invariant broke twice (D-094, D-096) | `exit_reason`, `describe`, `summarize_coverage`, `route_after_gap_check` |
| `agent/test_adaptive_exit.py` [D-096] | The two measured exits: stop when the prompt is already full (`SYNTHESIS_TOP_N` reached) and when most of a round's searches came back empty. Includes the boundary in the direction that costs quality, the per-round baseline that stops cumulative empties latching the exit on, and the thin-question case that must still get its second round | `route_after_gap_check`, `make_source` |
| `agent/test_recursion.py` [M4] | The whole cycle: a full-depth run fits RECURSION_LIMIT, 12 is one step too few, and each early exit (D-009, D-077) | `build_graph`, `make_arxiv_feed` |
| `agent/test_fanout.py` [M3] | The whole graph fanning out: one worker per subtopic, reducers under parallel writes, partial failure, and which nodes stream (D-067, D-068, D-069) | `build_graph`, `fakes` |
| `agent/test_checkpoint_roundtrip.py` [M2] | A `Source` comes back from a checkpoint as a `Source`, plus the control case (empty allowlist → `dict`) | `build_graph`, `build_serializer`, `JsonPlusSerializer`, `fakes` |
| `eval/recording.py` [O-11] | The recording format: `Recording`, `arm_name(settings)` (the arm is *derived* from settings so a run can never be filed under a configuration that did not produce it), `save`, `load_all(arm)`, `arms()` | `questions.json`, `recordings/` |
| `eval/questions.json` [O-11] | The frozen, **append-only** ten-question set — broad single-area survey prompts. Rewording one invalidates every earlier recording (D-088) | read by `load_questions("broad")` |
| `eval/questions-narrow.json` [O-14] | Ten **intersection** questions, each spanning two or three of the same areas the broad set covers separately. Built to be the case recursion should win, so D-094's null result can be attributed to the feature rather than to the questions | read by `load_questions("narrow")` |
| `eval/compare.py` [D-094] | Paired comparison of two arms, run as `python -m tests.eval.compare <arm-a> <arm-b>`. Pairs by question id, reports the mean paired difference in standard errors, and refuses arms that share no questions | `recording.load_all`, `results.json` |
| `eval/test_compare.py` [D-094] | The analysis arithmetic on synthetic arms with hand-computable answers: pairing by id not position, unshared questions excluded, zero-variance metrics reporting no SE rather than infinity — plus a check that the module still reproduces the numbers D-094's prose quotes | `compare.py` |
| `eval/test_question_sets.py` [D-088, O-14] | The sets stay well-formed: ids unique within *and across* sets (they key `results.json`), an unknown set name raises rather than silently recording the default, and each file states its own append-only rule | `recording.load_questions` |
| `eval/metrics.py` [D-093] | `numeric_density` and `citation_density` (deterministic, free) plus `specificity_metric` (G-Eval). The free measurement stays primary (D-088) | `deepeval.GEval` |
| `eval/test_record.py` [O-11] | **Opt-in, paid** (`-m record`): runs the real agent over the question set and saves one `Recording` per question. Holds the shared arXiv spacing, the outage assertion (every-subtopic-failed is an outage, not data) and `monkeypatch_depth`, which must patch **every** module binding `MAX_DEPTH` (D-094) | `build_graph`, `summarize_coverage`, `recording.py` |
| `eval/test_review_quality.py` [O-11] | **Opt-in, paid** (`-m eval`): scores saved recordings per arm and writes `results.json`. Records citation violations rather than asserting on them — asserting would assert the *model* behaved (D-078) | `recording.load_all`, `metrics.py`, `deepeval` |
| `eval/test_metrics.py` [D-093] | Tests the *instrument*: specificity must separate a deliberately concrete review from a deliberately vague one by >0.3 (measured 0.950 vs 0.222). A metric that scores everything ~0.95 is a number generator | `metrics.py` |
| `eval/test_recordings_are_consistent.py` [D-094] | Free, no judge, no network: every committed recording must be filed under the arm its `settings` describe, may not exceed its own depth ceiling, and **may not report convergence after using its whole depth budget** — the harness bug that briefly reversed D-094's finding | `recording.arm_name`, `recordings/*/*.json` |

---

## 5. One run, step by step (milestone 4)

```text
caller (test or web layer)
  1. model_factory = get_chat_model            (tests: RecordingFactory())
  2. http_client   = httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS)   (tests: make_arxiv_stub(...).client)
  3. limiter      = ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS)       (tests: NullLimiter())
  4. checkpointer  = InMemorySaver(serde=build_serializer())            (web layer: AsyncSqliteSaver)
  5. graph = build_graph(model_factory, http_client, limiter, checkpointer)
  6. graph.astream({"question": ...},
                   {"configurable": {"thread_id": ...}, "recursion_limit": RECURSION_LIMIT},
                   context=RunContext(provider=...), stream_mode=[...], version="v2")

inside the graph (a checkpoint is saved after every step)
  intake            reads question, runtime.context   → writes question (stripped)
  decompose         reads question, explored, failed   → model_factory(provider).ainvoke(...)
                                                        → writes pending_subtopics (filtered)
  route_subtopics   (conditional edge, not a node)     → Send per subtopic, or "synthesize"
  research_worker   reads its Send payload only        → async with limiter: search_arxiv(...)
    (one per subtopic, in parallel, one super-step)     → writes sources, skipped_entries,
                                                          explored_subtopics, seen_paper_ids
                                                        → on failure: failed_subtopics only
  gap_check         reads depth                       → writes depth + 1 (D-076)
  route_after_gap   (conditional edge, not a node)     → back to decompose while depth is
                                                         left AND the round added a paper
                                                         not already seen; else synthesize
  synthesize        reads question, sources, provider  → writes review (tokens stream)
                    (no sources: writes NO_SOURCES_REVIEW; no model is built)
  check_citations  reads review, sources              → writes citation_violations
```

### Which node writes which state field

| Field | Written by | Read by |
|---|---|---|
| `question` | the input; cleaned by `intake` | `search`, `synthesize` |
| `pending_subtopics` [M3] | `decompose` (overwrite) | `route_subtopics` |
| `sources` [M2] | `research_worker` ‖ | `synthesize` (data block), `check_citations` (`known_ids`) |
| `skipped_entries` [M2] | `research_worker` ‖ | nobody yet (shown in the report later) |
| `explored_subtopics` [M3] | `research_worker` ‖ | `decompose` (filter + prompt) |
| `failed_subtopics` [M3] | `research_worker` ‖ | `decompose` (N=2 retry cap) |
| `seen_paper_ids` [M3] | `research_worker` ‖ | the `Send` payload; `route_after_gap_check` (D-075) |
| `depth` [M4] | `gap_check` | `route_after_gap_check` (D-076) |
| `seen_before_round` [M4] | `decompose` (overwrite) | `route_after_gap_check` (D-075) |
| `empty_subtopics` [M5] | `research_worker` ‖ | `summarize_coverage` (O-5, D-086) |
| `citations_checked` [M5] | `check_citations` | `get_run_state` (D-084) |
| `review` | `synthesize` | `check_citations` |
| `citation_violations` [M2] | `check_citations` | nobody yet (shown in the report / UI later). Unknown IDs *and* unparseable markers (D-062) |

At milestone 2 each field had exactly one writer, so no reducers were needed. **That invariant ends
at milestone 3:** the fields marked ‖ are written by parallel `Send` workers in the same step and
each needs a reducer, or LangGraph raises `InvalidUpdateError` (D-067).

---

## 6. The four dependencies passed into `build_graph`

| Dependency | Unit tests | Integration test | Web layer (milestone 5) |
|---|---|---|---|
| model factory | `RecordingFactory()` (`fake_factory` fixture) | `get_chat_model` | `get_chat_model` |
| HTTP client [M2] | `make_arxiv_stub(...).client` (`arxiv_ok` fixture) | `httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS)` | one shared client, opened at startup |
| arXiv limiter [M3] | `NullLimiter()` (`limiter` fixture, zero delay) | `ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS)` | one limiter, created inside the app lifespan |
| checkpointer | `InMemorySaver(serde=build_serializer())` (`checkpointer` fixture) | the same fixture | `AsyncSqliteSaver` with `serde=build_serializer()` |

The graph code is identical in all three columns. Only the dependencies passed in change.

---

## 7. Every constant: where it's defined and used

| Constant | Defined in | Used by | Decision |
|---|---|---|---|
| `API_KEY_ENV_VARS` | `agent/config.py` | `llm.py`, `test_llm.py` | D-034 |
| `MODEL_NAMES` | `agent/config.py` | `llm.py` | D-029 |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | `agent/config.py` | `llm.py`, `test_llm.py` | D-038, D-039 |
| `ARXIV_MAX_RESULTS` [M2] | `agent/config.py` | `nodes/research_worker.py` | D-052 |
| `ARXIV_MIN_INTERVAL_SECONDS` [M3] | `agent/config.py` | whoever builds the `ArxivRateLimiter` | D-064 |
| `MAX_SUBTOPICS` [M3] | `agent/config.py` | `nodes/decompose.py` (prompt + filter cap) | D-070 |
| `MAX_DEPTH` [M4] | `agent/config.py` | `graph.route_after_gap_check` **and** `coverage._stop_reason` | D-025, D-026, D-076, D-094 |
| `SYNTHESIS_TOP_N` [D-091] | `agent/config.py` | `synthesize` (ranks the prompt) **and** `graph.route_after_gap_check` (the sufficiency exit) | D-091, D-092, D-096 |
| `RECURSION_LIMIT` [M4] | `agent/config.py` | the **caller**, as invoke config — not the graph | D-077 |
| `MAX_FAILURES` [M3] | `nodes/decompose.py` | the N=2 retry cap | D-020 |
| `ARXIV_TIMEOUT_SECONDS` [M2] | `agent/config.py` | whoever creates the HTTP client (`test_integration.py` now) | D-052 |
| `VALID_PROVIDERS` | `nodes/intake.py` (built from `ProviderType`) | `intake` | D-033 |
| `SYSTEM_PROMPT` | `nodes/synthesize.py` | `synthesize` | D-046, D-055 |
| `NO_SOURCES_REVIEW` [M2] | `nodes/synthesize.py` | `synthesize`, `test_graph.py` | D-060 |
| `ARXIV_API_URL`, `NAMESPACES`, `_ENTRY_ID_PATTERN` | `sources/arxiv.py` | `arxiv.py` only (fixed by arXiv, not settings) | D-042, D-044 |
| `STOPWORDS`, `PUNCTUATION_PATTERN` | `sources/arxiv.py` | `build_search_query` | D-051, D-059, D-063 |
| `ARXIV_ID_PATTERN` | `sources/models.py` | `Source` validator | D-041, D-056 |
| `CITATION_MARKER` | `nodes/check_citations.py` | `check_citations`; must match `SYSTEM_PROMPT` | D-046 |
| `CITATION_BRACKET` [M2] | `nodes/check_citations.py` | `check_citations`: finds citation attempts the strict marker can't parse | D-062 |
| `ALLOWED_MSGPACK_MODULES` | `persistence/checkpointer.py` | `build_serializer` | D-014 |

**Settings vs. constants:** values you might tune (model names, limits, result counts) go in
`config.py`. Values fixed by an external format (arXiv's URL, XML namespaces, the ID pattern) stay next
to the code that depends on them.

---

## Keeping this file current

Update it in the same change when a file is added, moved, or starts or stops importing another file.

Its companion is [`walkthrough.md`](walkthrough.md): this file is the static structure (what imports
what), that one is the dynamic behavior (what one run actually does, with captured real values).
A change to the graph shape or a state field usually touches both.
