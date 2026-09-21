# Design decisions

For each design decision in this repo, this file records what was chosen, why,
and what was rejected. It's the answer to "why did you do it this way?"

**Rules**
- Add an entry when the decision is made, not after.
- A recommendation nobody has confirmed goes under **Open**. It moves into
  the log, with a new ID, only once it's decided.
- Never delete an entry. If a decision changes, add a new entry and mark the
  old one `Superseded by D-NNN`.
- Skill files (`.claude/skills/`) hold the working rules. This file holds
  the reasoning.

---

## Settled before 2026-09-15 (carried over from CLAUDE.md)

- **D-001 — MIT license.**
- **D-002 — Users bring their own API keys.** No keys are built in, and `.env` is never committed.
- **D-003 — Paper handling.** arXiv/Semantic Scholar metadata may be used freely.
  Full PDFs are fetched to read, never cached for redistribution.
- **D-004 — No Google Scholar scraping**, anywhere.
- **D-005 — `src/` layout.** It avoids import-path pitfalls and matches how the package gets installed.
- **D-006 — SSE, not WebSockets, for v1.** Streaming only goes server → browser. Revisit if
  the browser needs to steer a run while it's going.
- **D-007 — One SQLite file** (`AsyncSqliteSaver`) holds both checkpoints and saved-review
  history.
- **D-008 — Single-user, no auth, for v1.** If it leaves localhost, gate it behind one password.
- **D-009 — Two-layer recursion control.** A `depth` field in state is the real exit;
  `recursion_limit` is only a backstop. Hitting `GraphRecursionError` means
  the depth exit is broken.

## 2026-09-15

### D-010 — Repository hygiene
- **Decision:** git on `main`. `.gitignore` covers `.env*` (except
  `.env.example`), SQLite files including their `-wal`/`-shm` side files, and `*.pdf`.
- **Why:** WAL side files can hold checkpoint data that hasn't reached the
  main database file yet. Ignoring PDFs backs up D-003 at the git level.

### D-011 — Package manager: uv
- **Decision:** uv, with `uv.lock` committed.
- **Why:** someone who clones the repo gets the exact versions with `uv sync`.
  Conda's main strength (native/CUDA dependencies) doesn't matter for this stack.
- **Rejected:** conda + `pip install -e` (no lockfile); conda for Python with
  uv inside (two tools to document).

### D-012 — Build order: agent first
- **Decision:** build the agent before the web layer: linear graph → one
  real source (arXiv) → `Send` fan-out → recursion → web layer. From the
  first graph, compile with `MemorySaver` and drive tests through `astream`.
- **Why:** the recursion and citation logic is the uncertain part and the part the thesis depends on. The
  web layer is designed as a thin wrapper, and testing through `astream` from the start
  surfaces serialization and streaming problems early.
- **Rejected:** walking skeleton (forces the frontend decision early and builds
  around a fake agent); source clients first (wraps APIs before knowing what
  the nodes need).

### D-013 — State typing: Pydantic at boundaries, dataclasses inside
- **Decision:** the top-level graph state is a dataclass. External data (LLM
  structured output, arXiv/Semantic Scholar responses) is parsed into
  Pydantic models with an explicit `Model.model_validate(...)` in the node
  that receives it.
- **Why:** when the state itself is Pydantic, LangGraph only validates the
  input to the first node, so later nodes and outputs go unchecked. Pydantic is also slower than
  dataclasses.
- **Rejected:** all Pydantic (validation doesn't run where it's needed); all
  dataclasses (external data is never validated).

### D-014 — Allowlist every custom type stored in state
- **Decision:** every custom Pydantic model or dataclass stored in state is
  added to `allowed_msgpack_modules` in the same change.
- **Why:** in strict msgpack mode, a type that isn't allowlisted can't be
  loaded back when a run resumes from SQLite.
- **Correction (2026-09-16, confirmed on langgraph 1.2.11):** there is **no error**. A blocked type
  is restored as a **plain `dict`**, with only a logged warning ("Blocked deserialization of
  … not in allowed_msgpack_modules"), and the first attribute access fails far from the
  cause. Without strict mode, the type is restored but LangGraph warns that this "will be
  blocked in a future version." Separately, a plain dataclass's `tuple` fields come back as
  `list`s; a Pydantic dataclass converts them back when it's restored.
- **Scope extended (2026-09-20, D-071):** this rule is not limited to types stored in state.
  A pending `Send` payload is checkpointed too, and a custom payload class hits the identical
  silent-`dict` failure — but only on a **resumed** run, so no start-to-finish test catches it.
  `Send` payloads are `TypedDict`s for that reason.

### D-015 — LLM providers: OpenAI default, DeepSeek selectable per run
- **Decision:** the UI chooses the provider for each run. The choice reaches nodes
  through LangGraph runtime `context`, not state and not a module-level global.
- **Why:** a global client would let concurrent runs overwrite each other's
  choice. State is checkpointed and meant for research data.
- **Known difference (from DeepSeek docs):** DeepSeek's JSON output is
  `json_object` only, so the JSON isn't guaranteed to match the schema, and it
  may return empty content. Expect more validation failures on DeepSeek.
  ~~LangChain's docs also say DeepSeek's reasoning model doesn't support
  tool calling or structured output.~~
- **Correction (2026-09-20, verified against DeepSeek's docs):** the struck-out line has
  **expired**. `deepseek-chat` and `deepseek-reasoner` were retired on 2026-07-24; the current
  models are `deepseek-flash` (DeepSeek-V4.1-Flash, the one configured here) and `deepseek-v4-pro`,
  and **both support JSON output and tool calls**. The rest of this entry still holds: JSON is
  `json_object` only, so schema conformance isn't guaranteed, and the docs still warn about
  occasional empty content. This correction is what settles D-066 — there is no longer a capability
  gap to guard against, only validation failures to catch.
- **Lesson worth keeping:** a provider constraint written into this log is a fact with an expiry
  date. Re-verify one before building anything around it.

### D-016 — LLM client: LangChain chat models
- **Decision:** `langchain-openai` (`ChatOpenAI`) and `langchain-deepseek`
  (`ChatDeepSeek`).
- **Why:** `stream_mode="messages"` streams tokens only from LangChain chat
  models, and the SSE design depends on that.
- **Rejected:** raw `openai` SDK with DeepSeek's `base_url` (tokens would
  have to be pushed by hand through `custom` mode).

### D-017 — Tracking explored subtopics
- **Decision:** `explored_subtopics: Annotated[list[str], <dedup reducer>]`,
  deduplicated after `casefold` + `strip`. `pending_subtopics` has no
  reducer and is overwritten each round.
- **Why:** workers write explored subtopics in parallel, so the key needs a reducer.
  `operator.add` would keep duplicates, and an accumulating pending list would
  dispatch old subtopics again.

### D-018 — A subtopic counts as explored only on success
- **Decision:** a worker marks its subtopic explored only when the search
  succeeds. A failed subtopic stays unexplored and can be proposed again.
- **Why:** a temporary failure shouldn't permanently drop a subtopic.

### D-019 — Worker failures handled with try/except inside the worker
- **Decision:** the worker catches failures itself, applies its own backoff,
  and returns a state update (`failed_subtopics`) instead of raising.
- **Why:** it's lower level, so every retry and failure path is written out and understood.
- **Rejected:**
  - `RetryPolicy` + `error_handler` (LangGraph ≥1.2): the default `retry_on`
    skips `ValueError` (which includes Pydantic `ValidationError`), `OSError`, and
    httpx errors other than 5xx (which includes arXiv's 429). How the handler's update behaves inside a `Send`
    branch wasn't confirmed.
  - Crash and resume from the checkpoint: one flaky call ends a long run, and the
    failure never becomes data.

### D-020 — Retry cap: N = 2
- **Decision:** `failed_subtopics: Annotated[list[str], operator.add]`. Each
  duplicate entry is one failed attempt. `decompose` skips any subtopic that has
  failed 2 times.
- **Why:** without a cap, a subtopic that always fails is proposed again every round
  until the depth limit.

### D-021 — Zero search results count as success
- **Decision:** an empty search marks the subtopic explored.
- **Why:** in a literature review, "nothing published on X" is a finding
  (a possible gap). Treating it as a failure would retry it and then quietly
  drop it.

### D-022 — Subtopic comparison: normalized match plus paper overlap (X = 0.6)
- **Decision:** a proposed subtopic is skipped if its normalized form
  (`casefold` + `strip`) is already in `explored_subtopics` (D-017), **or**
  if ≥ 60% of its search results are papers already in `seen_paper_ids`.
  Overlap = |results ∩ seen_paper_ids| / |results|. The explored list is also
  included in the planner prompt as a soft filter.
- **Why:** the planner is an LLM and rephrases, so exact string matching misses
  repeats. arXiv IDs are canonical, so comparing papers is exact. It also gives a
  rule you can measure and state.
- **Rejected:** embedding similarity (every run would depend on an embeddings
  API even when chat uses DeepSeek, plus a similarity threshold to tune).
- **Consequences:**
  - Zero results would make the overlap 0/0. Zero results follow D-021 and
    never reach the overlap check.
  - A `Send` worker only receives its payload, so `seen_paper_ids` has to be
    included in the payload.
  - `seen_paper_ids` has parallel writers, so it needs a deduplicating reducer.

### D-023 — Worker exception scope: expected external failures only
- **Decision:** the worker catches only expected external or I/O failures,
  such as httpx status errors and timeouts, and Pydantic `ValidationError` on
  external data. Programming errors crash immediately. `BaseException` is never
  caught, so `asyncio.CancelledError` propagates cleanly.
- **Why:** a bug caught as "search failed" would be retried and then silently
  dropped by the retry cap (D-020). This is the fail-fast rule.

### D-024 — N counts failures at the worker level
- **Decision:** each entry in `failed_subtopics` is one worker run that still
  failed after its own internal backoff. Individual HTTP retries don't count
  toward N = 2.
- **Why:** otherwise one 429 with two backoff attempts would use up the whole cap.

### D-025 — `max_depth = 2` as the baseline
- **Decision:** `max_depth = 2` to start, to be tuned in the thesis evaluation.

### D-026 — `depth` is 0-indexed
- **Decision:** `depth = 0` is the first search pass. `gap_check` routes back
  to `decompose` while `depth < max_depth`. With `max_depth = 2`, that's up to
  **3 search passes**: the first plus 2 recursive rounds.
- **Why:** the N = 2 retry cap (D-020) needs room to apply. A subtopic that
  fails in passes 1 and 2 is skipped in pass 3.

### D-027 — The worker's exact catch list
- **Decision:** `except (httpx.TransportError, httpx.HTTPStatusError, pydantic.ValidationError)`.
  This refines D-023. No manual backoff around LLM calls.
- **Why:** `TransportError` covers timeouts *and* connection, read and protocol
  errors. `TimeoutException` alone would let a DNS failure crash the run.
  LLM calls raise `openai.APIError` subclasses, not httpx ones
  (`ChatDeepSeek` subclasses `BaseChatOpenAI`, confirmed in the installed
  packages), and the OpenAI client already retries them.
- **Note:** `HTTPStatusError` is raised only when the code calls
  `response.raise_for_status()`.
- *Catch list extended for XML parsing by D-048.*

### D-028 — Paper overlap needs at least 3 results
- **Decision:** the 60% overlap rule (D-022) applies only when the search
  returns ≥ 3 papers. Below that, only the normalized-match check applies.
  Zero results still follow D-021.
- **Why:** with 1 or 2 results, a single paper already read means 50–100% overlap, which is noise.

### D-029 — A hand-written client factory
- **Decision:** a small factory maps `Literal["openai", "deepseek"]` to
  `ChatOpenAI` / `ChatDeepSeek`.
- **Why:** it's the point where tests swap in `GenericFakeChatModel`. It's typed,
  and it avoids pulling in the whole `langchain` package.
- **Rejected:** `init_chat_model` (no clean way to swap in a fake; extra dependency).

### D-030 — Streaming uses `version="v2"`
- **Decision:** `astream(..., version="v2")` and `ainvoke(..., version="v2")`.
- **Why:** every stream chunk is `{"type", "ns", "data"}`, and
  `ainvoke(..., version="v2").value` comes back as the state dataclass
  (confirmed against langgraph 1.2.11). `graph.get_state(config).values` is still a
  plain dict.

### D-031 — Test strategy
- **Decision:** the default tests use `GenericFakeChatModel`, with no keys, no
  cost and no network. Real-API tests are marked `@pytest.mark.integration` and
  skipped when that provider's key isn't set. `--strict-markers` is on, so a mistyped
  marker is an error instead of a test that silently never runs.
- **Why:** the default test run must not depend on your wallet or a provider's uptime.
- *When integration tests run was revised by D-036.*

### D-032 — Dependencies are passed into `build_graph`
- **Decision:** `build_graph(model_factory, checkpointer)`. Nodes get the
  factory by wrapping it at build time. The runtime context stays plain run data
  (`provider`).
- **Why the factory is a parameter:** tests pass a factory that returns
  `GenericFakeChatModel`, and every dependency is visible in the signature.
- **Why the checkpointer is a parameter:** the thing that owns the checkpointer's
  *lifetime* isn't the graph. Tests need a fresh `InMemorySaver` per test. At
  the web-layer milestone, `AsyncSqliteSaver` is an async context manager that
  the FastAPI app opens at startup and closes at shutdown, with its strict-msgpack
  settings. If `build_graph` created its own checkpointer, both cases would
  need `graph.py` rewritten.
- **Rejected:** putting the factory in `RunContext` (mixes run data from the UI with
  code wiring); monkeypatching in tests (breaks on import paths and hides the
  dependency).

### D-033 — `intake` checks the run context
- **Decision:** `intake` raises a descriptive `ValueError` right away if
  `runtime.context` is `None` or `provider` isn't one of the allowed values.
- **Why:** without `context=`, LangGraph passes `None` and the run fails later
  with `AttributeError` (confirmed on langgraph 1.2.11). A `Literal` type hint
  on a dataclass isn't checked when the program runs, so `RunContext(provider="gemini")`
  constructs without error. The check in `intake` covers every way into the graph.

### D-034 — A missing API key fails in the factory
- **Decision:** the client factory raises an error naming the missing
  environment variable (`OPENAI_API_KEY` / `DEEPSEEK_API_KEY`) before any model
  call.
- **Why:** it fails at the point where the key is needed, with a message that
  tells the user what to set, instead of a provider auth error in the middle of a run.

### D-035 — pytest-asyncio stays in strict mode
- **Decision:** `strict` (the default). Every async test carries
  `@pytest.mark.asyncio`.
- **Why:** there are few async tests, and the marker on each one shows its intent.
- **Rejected:** `auto`.

### D-036 — Integration tests are opt-in (revises D-031)
- **Decision:** pytest `addopts` includes `"-m", "not integration"`. Plain `uv run pytest`
  never calls a real API. `uv run pytest -m integration` runs those tests on purpose,
  and they still skip when the key isn't set.
- **Why:** under D-031 alone, a machine with `OPENAI_API_KEY` set made a paid, slow
  call that can fail when the network does on every test run. Confirmed in a scratch copy: the
  command-line `-m integration` overrides the `-m` in `addopts`.
- **Rejected:** an opt-in variable (`RUN_INTEGRATION=1`) in the `skipif`: one more thing to remember,
  when the `integration` marker already exists.
- **Consequence:** a syntax error in an integration test file still breaks the default run,
  because pytest imports every test file before deselecting anything.

### D-037 — Integration test checks: final state plus real streaming
- **Decision:** the integration test runs the graph once with `get_chat_model` and checks
  (a) more than one `messages` chunk from `synthesize`, (b) a non-empty
  `review` in the saved state, and (c) the joined streamed text equals the saved
  `review`. It never compares against fixed text.
- **Why:** a real model words things differently every run, so only properties can be
  checked. Streaming is what milestone 5 depends on, and the fake model always
  streams, so only a real provider can prove it.
- **Rejected:** checking only the final state (doesn't test streaming); checking content
  such as the "no sources consulted" disclaimer (flaky; that's prompt evaluation for
  `notebooks/`).

### D-038 — The client factory sets explicit timeout and retry limits
- **Decision:** `get_chat_model` passes `timeout=` and `max_retries=` to both
  `ChatOpenAI` and `ChatDeepSeek`, with values from `agent/config.py` (values under Open).
- **Why:** when they're not set, the underlying OpenAI client is built with
  `timeout=None` (confirmed on langchain-openai 1.6.2), so there's no time limit, and a
  hung call blocks the run or a test indefinitely. Both classes accept the same two
  parameters (confirmed).

### D-039 — LLM limits: 60 s timeout, 2 retries
- **Decision:** `LLM_TIMEOUT_SECONDS = 60.0` and `LLM_MAX_RETRIES = 2` in `agent/config.py`.
- **Why:** the timeout applies to each network wait, not the whole call. While streaming,
  it's the longest silence allowed between pieces of data; without streaming, it's the wait
  for the complete reply, and a review of up to 500 words should fit well within 60 s. 2 retries
  matches the OpenAI client's default, now stated explicitly. The names carry an `LLM_`
  prefix and a unit because milestone 2 adds an arXiv HTTP timeout to the same config.
- **Tested:** `test_client_has_explicit_timeout_and_retries` checks both providers. With the
  limits removed from the factory, it fails with `assert None == 60.0` (confirmed in a scratch copy).

### D-040 — `Source` is a dataclass
- **Decision:** the paper record stored in state (`Source`) is a dataclass, not a TypedDict or a
  plain Pydantic `BaseModel`. Which *kind* of dataclass, and its fields, are under Open.
- **Why:** it follows D-013 (dataclasses inside the graph) and must be allowlisted (D-014).

### D-041 — Only papers with a valid arXiv ID are kept, checked by Pydantic field validators
- **Decision:** paper IDs are validated with Pydantic `field_validator`s when arXiv data is
  parsed. An entry without a valid arXiv ID never becomes a `Source`.
- **Why:** the paper ID is how everything else refers to a paper: dedup and paper
  overlap (D-022), `seen_paper_ids`, and citations. A malformed ID has to be stopped where the data
  comes in (D-013).
- **Valid formats** (arXiv identifier docs): new scheme `YYMM.NNNN` (0704–1412) or
  `YYMM.NNNNN` (from 1501), optional `vN`; old scheme (before April 2007)
  `archive[.XX]/YYMMNNN`, e.g. `hep-th/9901001`, `math.GT/0309136`. The API
  returns the ID as a URL with a version, e.g. `<id>http://arxiv.org/abs/1706.03762v7</id>`
  (confirmed with a real request).
- **Not covered:** this checks an ID's *format*. It doesn't check that a citation in the review
  points to a paper that was actually retrieved; see Open → "Citation grounding".

### D-042 — The arXiv API is called with `httpx`
- **Decision:** call the arXiv API directly with `httpx`, not the `arxiv` package.
- **Why:** it keeps D-027's exception list correct (`httpx.TransportError`,
  `httpx.HTTPStatusError`), and every request is visible and testable.
- **Facts (confirmed 2026-09-16):** `https://export.arxiv.org/api/query` works over HTTPS
  (status 200, `application/atom+xml`). Terms of use: "no more than one request every three
  seconds" and "a single connection at a time". Metadata may be stored and shared (CC0), but not
  e-prints (PDFs).

### D-043 — `Source` is a frozen Pydantic dataclass (refines D-040)
- **Decision:** `@pydantic.dataclasses.dataclass(frozen=True)` with `field_validator`s. It's a
  single type that is both the validated boundary record (D-041) and the dataclass stored in state (D-040).
- **Why (confirmed on langgraph 1.2.11 / pydantic 2.13.5):** it's a real dataclass
  (`dataclasses.is_dataclass` → `True`), and invalid data raises `ValidationError` (a `ValueError`).
  After a checkpoint round trip it comes back as `Source` with `tuple` fields still `tuple`s, because
  validation runs again on restore. `frozen=True` makes it immutable and hashable, so it can go in sets.
- **Rejected:** a Pydantic `BaseModel` for parsing plus a plain `@dataclass` for state (two types
  and a conversion function, and the plain dataclass's `tuple` fields come back as `list`s after a
  checkpoint, which breaks hashing); a plain dataclass validated in `__post_init__` (validation
  written by hand).
- **Reminder:** add `Source` to `allowed_msgpack_modules` (D-014).

### D-044 — `Source` fields and the canonical arXiv ID
- **Decision:** `arxiv_id: str` (canonical, **no version**), `version: int`, `title: str`,
  `authors: tuple[str, ...]`, `summary: str`, `published: datetime`, `url: str` (the abstract page).
  `arxiv_id` is derived from the feed's `<id>` (e.g. `http://arxiv.org/abs/1706.03762v7`) by
  stripping the `http://arxiv.org/abs/` prefix and the `vN` suffix; `N` becomes `version`.
- **Why:** dedup, paper overlap (D-022), `seen_paper_ids` and citations all need one stable ID
  per paper. A new version of the same paper mustn't count as a different paper.

### D-045 — Invalid entries are skipped and counted; only a broken feed fails the search
- **Decision:** an entry that fails validation is skipped and counted, and the valid ones are kept.
  The whole search fails only when the feed can't be parsed or is arXiv's error feed (a single
  entry titled "Error", per the API manual).
- **Why:** one malformed entry shouldn't throw away a page of good results, but the skipped
  count stays visible. A broken feed means there's no data to work with, so it counts as a failed
  search (D-018/D-020).

### D-046 — Citation grounding happens in its own node
- **Decision:** `synthesize` writes prose with inline markers in the form `[arXiv:<arxiv_id>]`.
  A separate node after it extracts every marker and validates each ID with a Pydantic model given
  `context={"known_ids": <retrieved IDs>}`, then records any unknown IDs in state. Graph:
  `… → synthesize → <citation check> → END`.
- **Why:** the review still streams as readable prose (milestone 5). The check is deterministic
  and testable, and a violation becomes data the report can show. Checking the format alone (D-041)
  would accept a well-formed ID the model invented. Validation context works as intended (confirmed:
  an ID outside `known_ids` is rejected with "not a retrieved paper").
- **Rejected:** structured output validated against the retrieved IDs (token streaming would
  stream JSON instead of prose); checking inside `synthesize` (mixes generating and checking in one node,
  and hides violations from state).
- **Consequence:** the milestone 2 system prompt must require the exact marker format.

### D-047 — Parse arXiv XML with `defusedxml`, with `forbid_dtd=True`
- **Decision:** parse with `defusedxml.ElementTree.fromstring(..., forbid_dtd=True)`. Add
  `defusedxml` as a dependency when milestone 2 starts.
- **Why:** the Python docs say expat versions below 2.7.2 "may be vulnerable" to billion-laughs,
  quadratic-blowup and large-token attacks, and this environment has expat 2.7.1. Confirmed on
  defusedxml 0.7.1: by default it rejects entities and external references (`EntitiesForbidden`)
  but still accepts a plain DOCTYPE; `forbid_dtd=True` rejects that too (`DTDForbidden`). A real
  arXiv feed has neither.
- **Rejected:** stdlib `xml.etree.ElementTree` (no protection beyond expat's);
  `feedparser` (never raises on malformed input, which goes against fail-fast).

### D-048 — The worker catch list includes XML failures (extends D-027)
- **Decision:** the worker catches `(httpx.TransportError, httpx.HTTPStatusError,
  pydantic.ValidationError, xml.etree.ElementTree.ParseError, defusedxml.DefusedXmlException)`.
- **Why:** both new exceptions come from broken or unexpected external data, not from bugs.
  D-027's list catches neither (confirmed): truncated XML raises
  `xml.etree.ElementTree.ParseError`, which is **not** a `ValueError`; `DefusedXmlException` **is** a
  `ValueError` but not a `ValidationError`.

### D-049 — An `httpx.AsyncClient` is passed into `build_graph`
- **Decision:** `build_graph` takes an `httpx.AsyncClient`, just as it takes the model factory
  and checkpointer (D-032). Whoever calls `build_graph` owns the client's lifetime.
- **Why:** tests pass a client built on `httpx.MockTransport` (available in httpx 0.28.1)
  that serves saved arXiv responses, so they're deterministic and need no network. The web layer
  opens one client at startup and closes it at shutdown.
- **Rejected:** creating a client inside the search function (no shared connection, and tests
  need monkeypatching); a module-level client (unclear lifetime, and it can end up tied to the wrong
  event loop across tests).

### D-050 — A checkpoint round-trip test for `Source`
- **Decision:** a test saves state holding a `Source` through a checkpointer configured with the
  real serializer settings (allowlist included), loads it back, and asserts the value is still a
  `Source` with `tuple` fields.
- **Why:** a type that isn't allowlisted comes back as a plain `dict` with no error (D-014
  correction). Without this test, a missing allowlist entry would only show up as an
  `AttributeError` somewhere far away.

### D-051 — Milestone 2 builds the arXiv query from key terms
- **Decision:** `build_search_query` strips punctuation and arXiv query syntax characters, drops
  stopwords, and joins the remaining terms as `all:<term> AND all:<term> …`. Deterministic, no LLM call.
- **Why:** measured 2026-09-16, the raw question (`all:What is attention in transformer models?`)
  matched 327,597 papers, while `all:attention AND all:transformer AND all:models` matched 14,338
  more relevant ones. Removing syntax characters also stops a question containing `:`, `"` or
  parentheses from breaking the query.
- **Rejected:** the raw question (very noisy); an LLM rewriting the question (an extra paid call, and
  the milestone 3 planner will write queries anyway).
- **Known limit:** a long question can AND together enough terms to return zero results, which
  counts as a success (D-021).

### D-052 — arXiv settings: 10 results, 30 s client timeout
- **Decision:** `ARXIV_MAX_RESULTS = 10` and `ARXIV_TIMEOUT_SECONDS = 30.0` in `agent/config.py`.
  The timeout is applied when the `httpx.AsyncClient` is created, by whoever creates it (D-049).

### D-053 — At milestone 2, a failed search fails the run
- **Decision:** the `search` node catches nothing. Catching and recording failures (D-019, D-048)
  arrives with the parallel workers at milestone 3, and `ArxivAPIError` joins D-048's catch list then.
- **Why:** with a single search there's nothing to continue with after a failure, so recording it
  would just hide it. Failing loudly is the honest behavior here.

### D-054 — Milestone 2 layout and names
- **Decision:**
  - `Source` lives in `agent/sources/models.py`, so its allowlist entry is
    `("deep_research.agent.sources.models", "Source")`;
  - the arXiv client and parser live in `agent/sources/arxiv.py`;
  - the serializer settings live in `persistence/checkpointer.py`;
  - the new nodes are named `search` and `check_citations`;
  - the new `ResearchState` fields are `sources: list[Source]`, `skipped_entries: int` and
    `citation_violations: list[str]`.
- **Why:** each source client gets its own module under `sources/`, and the serializer is shared by
  tests now and the web layer later (milestone 5).

### D-055 — Retrieved papers go into the prompt as a delimited data block
- **Decision:** `synthesize` passes the retrieved papers in the human message as a clearly delimited
  block (ID, title, summary per paper). The system prompt says that block is data, not instructions.
- **Why:** paper abstracts are untrusted third-party text. This is the first concrete step on
  "Prompt-injection defenses" (still Open for the rest).

### D-056 — arXiv IDs use the simple digit rule
- **Decision:** a new-scheme ID has 4 or 5 digits after the dot for any YYMM.
- **Why:** it's simple and enough for the tests. **Rejected:** tying the digit count to YYMM (4 digits for
  0704–1412, 5 from 1501), which is more exact but adds complexity for little gain.

## 2026-09-19

### D-057 — The allowlist entry is built from the class (refines D-054)
- **Decision:** `ALLOWED_MSGPACK_MODULES` holds `(Source.__module__, Source.__name__)`, so
  `persistence/checkpointer.py` imports `Source`. The value is still
  `("deep_research.agent.sources.models", "Source")`.
- **Why:** an entry typed as strings goes stale silently when `Source` moves or is renamed. One
  built from the class follows it. It matches what the serializer records (confirmed in langgraph
  1.2.11 `jsonplus.py`: `(obj.__class__.__module__, obj.__class__.__name__, …)`).
- **Cost:** `persistence` now imports from `agent.sources`. There's no cycle, because `models.py`
  imports nothing from the project.
- **Still needed:** the round-trip test (D-050). It's what catches a *new* state type that was never
  added.

### D-058 — More reasons an entry is skipped: version, title, authors, abstract link
- **Decision:** besides the ID (D-041), an entry is skipped and counted (D-045) when:
  - `version < 1` (`Source` validator);
  - the title is empty after whitespace is collapsed (`Source` validator; runs of whitespace
    become one space);
  - there are no author names (`entry_to_source`);
  - there's no `<link rel="alternate">` (`entry_to_source`). There's no fallback to another link.
- **Why:** a paper with no title or authors is useless in the prompt and the report, and arXiv
  numbers versions from 1. In a real entry the next link is the PDF (`rel="related"`), so "fall back
  to any link" would store the PDF as the abstract page.
- **Alternative (not taken):** `Annotated[int, Field(ge=1)]` for the version, which has the same effect.

### D-059 — `build_search_query` raises `ValueError` when no terms are left
- **Decision:** a question made only of stopwords and punctuation (e.g. "What is it?") raises
  `ValueError`, which fails the run at milestone 2 (D-053).
- **Why:** arXiv would otherwise receive an empty query. Failing loudly names the cause.
- **Consequence at milestone 3:** `ValueError` isn't on D-048's catch list, so a planner subtopic with
  no terms would crash the whole run instead of counting as a failed subtopic. Revisit then.
- **Known limit:** `PUNCTUATION_PATTERN` (`[^a-zA-Z0-9\s]`) treats non-ASCII letters as punctuation,
  so "Schrödinger" becomes `all:schr AND all:dinger`.

## 2026-09-20

### D-060 — No papers found: a fixed review, with no model call
- **Decision:** when `sources` is empty, `synthesize` returns the fixed `NO_SOURCES_REVIEW` string
  and never builds or calls a model. Promotes the behavior implemented provisionally on 2026-09-19.
- **Why:** it costs nothing, and more importantly there is no model output that could invent a
  citation or answer the question from general knowledge. Zero results is a success (D-021), so this
  is the success path for "nothing published on X" — a finding the review should state plainly.
- **Rejected:** calling the model with an empty `<papers>` block and prompting it to report that
  nothing was found. That is a paid call whose correctness depends on the model obeying the prompt,
  to produce a sentence we can write ourselves.
- **Consequence (carry to milestone 5):** this run streams **nothing** in `messages` mode, because no
  model is invoked. The review arrives only in the `synthesize` `updates` chunk. The SSE layer must
  render a review that never produced a token, or the UI will look hung on an empty search.

### D-061 — `Source` forbids unknown fields (refines D-043)
- **Decision:** `Source` is declared `@dataclass(frozen=True, config=ConfigDict(extra="forbid"))`.
- **Why:** Pydantic dataclasses default to `extra="ignore"`. Verified 2026-09-20:
  `Source(**valid_fields, abstract="x")` constructed successfully and silently discarded `abstract`.
  A typo'd or stale field name was therefore accepted in silence whenever the required fields
  happened to also be present. That breaks fail-fast (D-023) at the parsing boundary, which is
  exactly where D-013 puts validation. The 2026-09-19 `id=`/`abstract=` bug was caught only because
  the typo *also* removed a required field, producing a `missing` error — luck, not design.
- **Effect:** an unknown keyword now raises `ValidationError` with type `unexpected_keyword_argument`
  and the offending name in `loc`. `parse_feed` still counts it as a skipped entry (D-045); what
  changes is that the error names the wrong field, and `test_arxiv.py`'s `skipped == 0` assertion
  turns it into a red test instead of a silent zero-result search.
- **Rejected:** inspecting `ValidationError.errors()[*]["type"]` inside `parse_feed` to re-raise
  caller bugs (`missing`) while skipping bad data (`value_error`). It couples production code to
  Pydantic's internal error-type strings to defend against something the test suite already catches.
- **Tested:** `test_unknown_field_is_rejected` and `test_unknown_field_error_names_the_offending_field`.
  Both fail with `DID NOT RAISE` against the pre-fix code (confirmed 2026-09-20).

### D-062 — An unparseable citation marker is a violation (closes an Open item)
- **Decision:** `check_citations` matches every `[arXiv:...]` bracket with `CITATION_BRACKET`, then
  tries to parse each one with `CITATION_MARKER`. A bracket that doesn't parse is recorded in
  `citation_violations` verbatim, e.g. `"[arXiv:A, B]"`. Promotes the Open item "Citations the check
  can't see".
- **Why:** verified 2026-09-20 against the node — `[arXiv:A, B]` and `[arXiv: A]` extracted
  *nothing*, so a review citing only in those forms returned `citation_violations == []`, the same
  result as a perfectly grounded review. That is a false negative in the feature the whole project
  is built on: an unreadable citation must never be indistinguishable from a verified one. The
  prompt forbids both forms, but relying on that is relying on model compliance — the exact
  dependency D-046 built a deterministic checker to remove.
- **Consequence:** `citation_violations` now means "every citation that could not be verified", not
  "well-formed IDs that weren't retrieved". Malformed entries are recognizable by their brackets.
  `state.py` and `code-map.md` are updated to match.
- **Rejected:** parsing grouped IDs (`[arXiv:A, B]`) into separate citations and validating each. It
  would silently accept a format the prompt forbids, so the prompt and the checker would drift
  apart; flagging it keeps one canonical form.
- **Known limit (tested):** a citation with no brackets at all, e.g. a bare `arXiv:1706.03762` in
  prose, is still not detected — `arXiv:` occurs in ordinary text, so matching it unbracketed would
  produce false positives. `test_known_limit_citation_without_brackets_is_invisible` marks that
  boundary.

### D-063 — `build_search_query` keeps non-ASCII letters (closes an Open item, refines D-051)
- **Decision:** `PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")`. Python 3's `\w` is Unicode-aware by
  default, so accented Latin, CJK and Cyrillic letters survive as search terms while arXiv's query
  syntax is still stripped.
- **Why:** the old `[^a-zA-Z0-9\s]` treated every non-ASCII letter as punctuation, so "Schrödinger"
  became `all:schr AND all:dinger`. Measured against the real API on 2026-09-20:

  | query sent | `totalResults` | top hit |
  |---|---|---|
  | `all:schrödinger` | **20,184** | *Optomechanical Schrödinger Cats* |
  | `all:schrodinger` | 4,900 | *…nonlinear Schrodinger equation…* |
  | `all:schr AND all:dinger` (old) | 903 | *…Korteweg-de Vries…* — unrelated |

  The old behavior was not merely lossy: it returned confident nonsense. Two meaningless fragments
  ANDed together still match ~900 papers, none of them the right ones. Under D-021 a near-empty
  result is recorded as a *success*, so this failed silently on much of physics and most
  non-English author names.
- **Confirmed:** arXiv accepts UTF-8 in `search_query` — httpx percent-encodes `ö` as `%C3%B6` and
  the API returns HTTP 200 with relevant results. arXiv does **not** fold accents internally; the
  accented and unaccented forms are genuinely different queries, which is why the fix must preserve
  the accent rather than normalize it.
- **Verified:** `[^\w\s]` still strips every arXiv syntax character — `:`, `"`, `(`, `)`, `+`, `-`,
  `[`, `]`, `^` — so D-051's injection protection is unchanged.
- **Rejected:** NFKD normalization to ASCII, then the old pattern. It measures *worse* (4,900 vs
  20,184), `ß` doesn't decompose so German terms are still shredded, and CJK survives the fold only
  to be deleted by the ASCII pattern. Also rejected: blacklisting arXiv's syntax characters
  explicitly — more precise, but it's a list that must be kept in sync with their query grammar and
  fails open if they add an operator.
- **Known limits:** `\w` also keeps `_`, which is harmless inside a term. `STOPWORDS` is still
  English and ASCII only, so a non-English question keeps its stopwords — they become search terms
  rather than being dropped. That's a smaller problem than deleting the letters.

### D-064 — arXiv access is serialized by a limiter passed into `build_graph` (settles O-1)
- **Decision:** a small `ArxivRateLimiter` — an `asyncio.Semaphore(1)` held **across** the request,
  plus monotonic-clock spacing — is created by the caller and passed into `build_graph`, alongside
  the model factory, HTTP client and checkpointer (D-032, D-049). Tests pass `min_interval=0`.
- **Why the semaphore is held across the request, not just around the start:** measured 2026-09-20.
  A token-bucket limiter paces request *starts*, so when a request outlasts the interval the next
  one begins before it finishes — peak **2 concurrent connections** in a 4-worker simulation.
  arXiv's terms allow one connection at a time *and* one request per three seconds (D-042). Holding
  the semaphore for the whole request satisfies both at once, and the effective spacing becomes
  `max(min_interval, request_duration)` with no extra code.
- **Rejected:** `langchain_core.rate_limiters.InMemoryRateLimiter`, which is already a dependency and
  needs no new code — but it is a token bucket, so it produced the 2-concurrent result above and does
  not meet arXiv's terms. Also rejected: `httpx.Limits(max_connections=1)`, which serializes but adds
  no spacing, so it solves half the problem in a place nobody looks; and a module-level limiter inside
  `sources/arxiv.py`, because a module-level `asyncio.Semaphore` binds to the first event loop that
  touches it and then fails across tests.
- **The reframe worth keeping:** arXiv's terms make parallel *searching* non-compliant however it is
  implemented. The parallelism milestone 3 buys is in the LLM work (reading, summarizing, gap
  analysis); arXiv access is a queue that parallel workers take turns on. Serialization here is a
  policy requirement, **not** a performance bug to optimize away later.

### D-065 — Retry 429 and 5xx; re-raise every other 4xx (refines D-027, D-048; settles O-2)
- **Decision:** the worker treats `429` and `5xx` as external failures that count toward the N=2
  retry cap (D-020). Every other `HTTPStatusError` — notably `400` and `404` — is re-raised and
  crashes the run.
- **Why:** confirmed 2026-09-20 — arXiv serves its error feed with **HTTP 400**
  (`test_search_arxiv_raises_on_http_400`), so `raise_for_status()` fires before `parse_feed` runs.
  Under the previous catch-all, a malformed query from a bug in `build_search_query` — exactly the
  class D-059 warns about — was retried twice and then recorded as "subtopic failed". A bug wore a
  network failure's clothes, which is precisely what D-023's fail-fast rule exists to prevent. A 4xx
  means *we* sent something wrong.
- **Rejected:** an explicit status map (`{429, 5xx}` retry, `{400, 404}` crash, else crash). More
  precise and easier to test per status, but a table to maintain while there is still only one
  source. Revisit when a second source lands.
- **Known consequence:** `ArxivAPIError` remains unreachable through `search_arxiv`, since arXiv's
  error feed arrives with a 400 that `raise_for_status()` catches first. It is exercised only by
  tests calling `parse_feed` directly. Whether to raise it *before* `raise_for_status()`, so arXiv's
  own error text ("incorrect id format") reaches the log instead of a bare `400 Bad Request`, is
  left open.

### D-066 — One model for every role; Pydantic validation is the structured-output guard (settles O-3)
- **Decision:** no per-role model table and no capability matrix. Every role uses the model chosen
  for the run's provider. Where a node needs structured data, it asks for JSON and validates the
  reply with Pydantic at the boundary (D-013); the resulting `ValidationError` is already on the
  worker's catch list (D-048) and drives the retry.
- **Why the guard this originally proposed is unnecessary:** the premise expired. D-015 recorded that
  DeepSeek's reasoning model supports neither tool calling nor structured output. Verified against
  DeepSeek's docs on 2026-09-20: `deepseek-chat` and `deepseek-reasoner` were **retired 2026-07-24**,
  and both current models (`deepseek-flash`, `deepseek-v4-pro`) support JSON output *and* tool calls.
  The configured `deepseek-flash` is correct and current.
- **Why the guard was also impossible:** verified 2026-09-20 — `with_structured_output(...)` binds
  without error on *any* chat model, including one that cannot honor it. LangChain does not expose
  model capabilities, so a "startup check" would have to be a hand-maintained table of
  (provider, model) → capability. That table is the same kind of artifact that just went stale inside
  D-015, and it would fail in the more dangerous direction: wrongly refusing to start.
- **What still holds from D-015:** DeepSeek's JSON is `json_object` only, so schema conformance is
  not guaranteed, and the docs still warn the API "may occasionally return empty content". Both are
  validation failures, which is exactly what Pydantic at the boundary is for. Expect more retries on
  DeepSeek than on OpenAI.
- **Rejected:** `MODELS_BY_ROLE` in config, which would allow tuning cost against quality per role
  (a cheap planner, a stronger synthesizer). Deferred, not dismissed — revisit with real token-cost
  numbers from the thesis evaluation runs rather than by guessing now.

### D-067 — Milestone 3 state fields and their reducers
- **Decision:**

  | Field | Writers | Reducer |
  |---|---|---|
  | `pending_subtopics: list[str]` | `decompose` | **none** — overwritten each round (D-017) |
  | `explored_subtopics: Annotated[list[str], merge_subtopics]` | workers, parallel | dedup on `casefold` + `strip`, **storing the original text** |
  | `failed_subtopics: Annotated[list[str], operator.add]` | workers, parallel | plain concat — duplicates are the attempt count (D-020) |
  | `seen_paper_ids: Annotated[set[str], operator.or_]` | workers, parallel | set union |
  | `sources: Annotated[list[Source], merge_sources]` | workers, parallel | dedup on `arxiv_id`, keep first |
  | `skipped_entries: Annotated[int, operator.add]` | workers, parallel | sum of per-worker **deltas** |
  | `depth: int` | `gap_check` | none — single writer |

- **Why `sources` and `skipped_entries` need reducers now:** at milestone 2 each state field had
  exactly one writer, which is why no reducers existed. `search` becoming a parallel `Send` worker
  ends that invariant. Without a reducer LangGraph raises `InvalidUpdateError` — loudly, not
  silently (D-014's correction note).
- **Why `sources` dedups on `arxiv_id` rather than on the `Source` object:** two subtopics can
  retrieve the same paper, and they can retrieve *different versions* of it. `Source` is frozen and
  hashable, so a `set[Source]` would compile and still keep v1 and v2 as two entries for one paper,
  defeating D-044's canonical-ID rule. Keying on `arxiv_id` collapses them correctly. First-seen
  order is preserved so the `<papers>` block is stable across runs, which a `set` would not give.
- **Why the original subtopic text is stored, not the normalized form:** normalization is a
  *comparison* concern. Storing `"attention in bert"` would leak lowercased text into the planner
  prompt, the review's coverage section and the UI. The reducer normalizes to compare and keeps what
  the planner wrote.
- **Confirmed 2026-09-20:** a `set[str]` in state survives a checkpoint round trip through
  `build_serializer()` and comes back as a `set`, so `seen_paper_ids` needs no allowlist entry —
  `set` is a builtin, not a custom type (D-014 applies only to custom classes).
- **Sharp edge:** with `operator.add` on `skipped_entries`, a node must return its own **delta**
  (`{"skipped_entries": result.skipped}`), never a running total, or the count compounds. Pinned by
  a test.
- **Reducers must not mutate their left argument.** LangGraph may still hold a reference to it.
  `operator.add` and `operator.or_` both build new objects; `merge_subtopics` and `merge_sources`
  copy before appending. A reducer like `lambda a, b: a.extend(b) or a` is wrong twice over — it
  mutates, and it returns `None`.

### D-068 — `Send` dispatch order is observed, not guaranteed; tests assert order-insensitively
- **Decision:** assertions on fields written by parallel workers compare sorted lists or sets. One
  dedicated test asserts exact dispatch order and is documented as pinning observed behavior.
- **Why:** measured 2026-09-20 on langgraph 1.2.11 — five workers sleeping 50/10/40/20/30 ms
  produced dispatch order (`a,b,c,d,e`) on three consecutive runs, not completion order; `b`
  finished first and landed second. So ordering *is* currently deterministic, which is what makes
  the `<papers>` block reproducible for the thesis evaluation. But it is **not a documented
  LangGraph contract**, and a minor release could change it.
- **Why one pinning test rather than none:** if the behavior changes, the failure should be one
  clearly-labelled test naming the assumption, not several unrelated-looking tests going red at
  once. Reproducibility is worth knowing about; it is not worth depending on silently.
- **Rejected:** asserting exact order everywhere (couples the whole suite to an undocumented
  detail); asserting nothing about order (loses the reproducibility signal entirely).

### D-069 — An empty subtopic list routes to `synthesize`, never to an empty fan-out
- **Decision:** `route_subtopics` returns the node name `"synthesize"` when `pending_subtopics` is
  empty, and a list of `Send` objects otherwise. It lives in `graph.py`, because it encodes node
  ordering (code-map rule 2: only `graph.py` knows what runs when).
- **Why:** measured 2026-09-20 — a conditional edge returning `[]` produces **no error and no
  downstream node**. The graph ran the planner and stopped, with no review written. An empty list
  is reachable in a later round when every proposed subtopic is already explored or has hit the
  N=2 retry cap (D-020), which is a legitimate *success* state, not a failure. Falling off the end
  silently is the same shape as D-062 and D-063: reporting completion while producing nothing.
- **Confirmed:** one conditional edge may return a bare node name in one branch and `Send` objects
  in another — `pending=[]` routed `plan → synthesize`, `pending=["a","b"]` routed
  `plan → worker:a, worker:b → synthesize`.
- **Rejected:** having `decompose` raise when it filters everything out (it is a valid end state,
  not an error); adding a separate guard node (an extra super-step, and it splits the guard from
  the routing it guards).

### D-070 — `decompose`: explicit JSON validation, and a planner failure crashes the run
- **Decision:** the prompt asks for JSON; the node parses it with
  `SubtopicPlan.model_validate_json(reply.text)`. `decompose` catches nothing at milestone 3, so an
  unparseable plan raises `ValidationError` and fails the run.
- **Why explicit validation over `with_structured_output`:** D-013 requires external data to be
  validated *in the node that receives it*, and a model's reply is external data. It also behaves
  identically on both providers, where `with_structured_output` routes through tool calling and
  binds without error even on a model that cannot honor it (D-066). Keeping the parse in the node
  means the `ValidationError` is ours to route.
- **Why a failure crashes:** the same reasoning as D-053 for milestone 2's `search`. With a single
  planner there is nothing to continue with, so catching would hide it. `decompose` is **not** a
  `Send` worker, so D-048's catch list does not apply to it.
- **Revisit at milestone 4:** once `gap_check` can re-plan, a repair retry (re-asking with the
  validation error in the prompt) becomes worthwhile — D-015 notes DeepSeek occasionally returns
  empty content, so the expected value is real. Rejected for now as a second paid call and an
  untested path.
- **Rejected:** falling back to searching the original question as one subtopic. It keeps the run
  alive but silently downgrades a recursive review to a milestone-2 one — the failure shape this
  project has now hit three times (D-062, D-063, D-069).
- **The division of labour worth stating:** the planner *proposes*; the code *decides*. The
  explored list in the prompt is a soft filter (D-022); the hard filters are the normalized-match
  check and the N=2 retry cap (D-020), both applied in `decompose` after the model replies. The
  third filter — ≥60% paper overlap — cannot run here because it needs search results, so it stays
  in the worker (D-022, D-028).

### D-071 — `Send` payloads are checkpointed, so they are plain dicts (`TypedDict`) (extends D-014)
- **Decision:** a `Send` payload is a `TypedDict` (`SubtopicTask`), never a dataclass or Pydantic
  model. At runtime it is an ordinary `dict`, so msgpack handles it natively and no allowlist entry
  is needed, while the type checker still sees the shape.
- **Why:** measured 2026-09-20. A pending `Send` is saved in the checkpoint, so its payload goes
  through the same serializer as state. With a custom frozen dataclass payload, a **resumed** run
  logged `Blocked deserialization of Payload - not in allowed_msgpack_modules` and handed the
  worker a plain `dict`. `payload.subtopic` would then raise `AttributeError`.
- **Why this is worse than the state version of the same bug:** it is invisible in every test that
  runs a graph start-to-finish. It only appears after an interrupt-and-resume, which is exactly the
  path the web layer uses at milestone 5.
- **D-014's scope was too narrow.** It says "every custom type *stored in state*". `Send` payloads
  are subject to the identical rule. A `TypedDict` sidesteps it entirely, which is why it is
  preferred over allowlisting a payload class.
- **Rejected:** a plain untyped `dict` (same runtime safety, no type checking); a dataclass payload
  plus an allowlist entry (works, but adds a type to the allowlist purely for transport, and the
  failure mode when someone forgets is silent).

### D-072 — `research_worker`: contract and catch list
- **Decision:** `make_research_worker(http_client, limiter)` returns an async node taking a
  `SubtopicTask` payload (D-071). On success it returns `sources`, `skipped_entries` (its own
  delta, D-067), `explored_subtopics: [subtopic]` and `seen_paper_ids`. On a caught failure it
  returns **only** `failed_subtopics: [subtopic]` — never `explored_subtopics`, because a subtopic
  counts as explored only on success (D-018).
- **Catch list** — D-048 plus D-053's promise that `ArxivAPIError` joins here, refined by D-065:
  `httpx.TransportError`, `pydantic.ValidationError`, `xml.etree.ElementTree.ParseError`,
  `defusedxml.DefusedXmlException` and `ArxivAPIError` are recorded as failures.
  `httpx.HTTPStatusError` is recorded **only** for `429` and `5xx`; every other 4xx is re-raised
  (D-065). `BaseException` is never caught, so `asyncio.CancelledError` propagates (D-023).
- **Every arXiv request is made inside `async with limiter:`** (D-064). The limiter wraps the
  request, not just its start.
- **Zero results is a success** (D-021): the subtopic is marked explored with an empty `sources`.

### D-073 — `decompose` also drops subtopics `build_search_query` can't use (settles a D-059 item)
- **Decision:** alongside the explored-match and N=2 retry-cap filters, `decompose` drops any
  proposed subtopic for which `build_search_query` raises `ValueError` — i.e. one made only of
  stopwords and punctuation (D-059).
- **Why here and not in the worker:** `ValueError` is not on the worker's catch list (D-048), so an
  all-stopword subtopic would crash the whole run. Catching it in the worker instead would burn two
  retry-cap attempts (D-020) on something that fails identically every time, since the failure is
  deterministic. `decompose` is the place where "don't dispatch it" is an available response.
- **Rejected:** letting it crash (one bad subtopic out of three kills a run that could have
  delivered the other two); catching `ValueError` in the worker (wastes the retry cap, and widening
  that catch would also swallow `ValidationError` from genuine bugs).

### D-074 — The ≥60% paper-overlap check is deferred to milestone 4
- **Decision:** the worker does **not** implement D-022's overlap rule at milestone 3. It writes
  `seen_paper_ids` so the data is ready, and the `Send` payload carries `seen_paper_ids` so the
  plumbing is tested, but no overlap comparison is made yet.
- **Why:** D-022 was written as "a proposed subtopic is *skipped* if ≥60% of its results are already
  seen", which assumes the check runs before dispatch. It cannot — overlap needs search results, so
  by the time it is computable the search is already paid for. Once redundant subtopics keep their
  papers (the ~40% that are new are real findings the search already bought) and are marked
  explored like any other, the comparison produces **no observable difference** in the returned
  update. Implementing it now would be dead code.
- **Its real consumer is `gap_check`** deciding whether a round found anything new, which arrives at
  milestone 4 along with the state field needed to record redundancy. Building it then means the
  field is justified by an actual reader rather than added speculatively.
- **Consequence:** D-022's wording should be read as "don't *re-explore* it", not "discard its
  results". D-028's ≥3-results floor and D-021's zero-results path are unaffected and still apply
  when the check lands.

## 2026-09-21

### D-075 — `gap_check` is a termination check, not a gap finder
- **Decision:** `gap_check` routes back to `decompose` only when **both** hold: `depth` is not
  spent, **and** the round just finished contributed at least one paper not already in
  `seen_paper_ids`. Otherwise it routes to `synthesize`. It calls no model.
- **The division of labour:** `decompose` is the gap finder — it receives `explored_subtopics` and
  is asked for something genuinely new (D-070). `gap_check` only answers "is another round worth
  it?". Keeping discovery and termination in separate nodes is what makes the stopping rule
  statable in one sentence, which the thesis evaluation needs.
- **Why deterministic rather than an LLM judge:** it is free, reproducible, and testable with no
  model. An LLM asked "are there still gaps?" can answer yes indefinitely, which would leave
  `depth` as the *mechanism* rather than the backstop — the inversion D-009 warns against. Rejected
  for now, not forever: revisit if measurement shows the run stops while obvious gaps remain.
- **How a round's contribution is measured:** state accumulates across rounds, so `gap_check` cannot
  read "what this round added" from totals. A field with `operator.add` **cannot be reset by a
  node** either, because `reducer(current, 0) == current` — returning zero is a no-op. So
  `decompose` writes `seen_before_round = len(state.seen_paper_ids)` at the start of each round, in
  the same no-reducer, overwritten-each-round style as `pending_subtopics` (D-017), and `gap_check`
  compares the current length against it.
- **Consequence for D-022's paper-overlap rule:** the per-subtopic ≥60% overlap check now has a very
  weak case. Its purpose was to avoid re-exploring a redundant subtopic, but it cannot run before
  dispatch (D-074), and this round-level rule already stops the recursion when a whole round is
  redundant. The overlap *ratio* may still be worth recording as thesis evidence; the
  control-flow use is superseded. See Open → "Retire or repurpose D-022's overlap check".
- **Consequence:** `route_subtopics` keeps returning `"synthesize"` for an empty plan (D-069) rather
  than routing to `gap_check`. A round with nothing to research is itself the "no gaps left" signal,
  and sending it round the loop would buy another paid planner call to learn the same thing.

### D-076 — `gap_check` increments `depth`, then routes on the new value
- **Decision:** `gap_check` returns `{"depth": state.depth + 1}` and decides using the incremented
  value. `depth` therefore means **rounds completed**, and a full run ends with `depth == 3` when
  `max_depth == 2`.
- **Why here:** recursion control stays in one node. `gap_check` both decides whether to loop and
  owns the counter that stops it; splitting them across `decompose` and `gap_check` would mean
  reading two files to answer "why did this run stop?".
- **Why increment-then-check:** the final checkpoint records what actually happened (three rounds
  completed) rather than what was about to happen. Both orderings give the same three search passes
  (D-026); only the recorded value differs.
- **Single writer, so no reducer** (D-067). `depth` is never written by a parallel worker.
- **Verified 2026-09-21** on a graph of this exact shape: `intake → decompose → Send → worker →
  gap_check → {decompose | synthesize} → check_citations` produced three search passes
  (`d=0, 1, 2`) and ended at `depth == 3`.

### D-077 — `recursion_limit = 15` (settles O-4)
- **Decision:** runs are invoked with `recursion_limit = 15`, and a test asserts a full-depth run
  completes under it.
- **Why 15:** measured 2026-09-21 by bisection on the real graph shape — **13 is the minimum**
  (12 raises `GraphRecursionError`). The count is `intake` (1) + 3 rounds × [`decompose` + workers +
  `gap_check`] (9) + `synthesize` + `check_citations` (2) = 12 super-steps, and LangGraph needs
  super-steps **+ 1**. 15 leaves two steps of headroom without being generous enough to let a
  runaway loop burn many paid calls.
- **Independent of `MAX_SUBTOPICS`:** a `Send` fan-out is a single super-step however wide it is, so
  changing subtopics-per-round costs wall-clock (rate limiting, D-064) but never recursion budget.
  Worth knowing before anyone "fixes" a `GraphRecursionError` by raising the subtopic count.
- **Why a test, not just a number:** `recursion_limit` is the backstop for a broken semantic exit
  (D-009). A number nobody checks silently stops protecting anything once the graph grows a node.
  Hitting `GraphRecursionError` means the `depth` exit is broken — raise the depth logic, not the
  limit.
- **Supersedes** the `langgraph-conventions` skill's previous advice to "set it explicitly and
  generously, e.g. 150", which predates any measurement and is ~11× the real need.

### D-078 — The integration test asserts the citation checker works, not that the model behaved
- **Decision:** the integration test requires (a) at least one citation of a paper that *was*
  retrieved, and (b) that every recorded violation is genuinely absent from `sources`. It prints
  any violations rather than failing on them. It no longer asserts `citation_violations == []`.
- **Why:** observed 2026-09-21 on a real three-round run — `gpt-4o` cited **`2113.11460`**, a
  fabricated ID, and `check_citations` caught it. That is D-046 working exactly as designed: a
  well-formed ID the model invented passes the format check (D-041) and fails the retrieval check.
  The old assertion treated that success as a failure.
- **The distinction that matters:** citation grounding exists *because* models cite from memory.
  Asserting the model never does conflates "the checker works" with "this run got lucky", and
  makes the only test that talks to a real API randomly red — useless as a pre-demo gate.
- **What still fails the build:** zero valid citations (the review cited nothing real), or a
  violation that *was* retrieved. The second is a false positive and means the checker is broken,
  which is a code bug.
- **Rejected:** keeping `== []` (depends on behavior we don't control); strengthening the prompt
  first and keeping the strict assertion (a paid run per attempt, and prompt adherence can never be
  guaranteed — D-015 already notes DeepSeek is worse here).
- **Note on `2113.11460`:** month 13, so not even a possible arXiv ID. D-056 chose the simple
  digit rule over tying digit count to `YYMM`, and this is that tradeoff appearing in the wild. It
  did not matter — grounding caught it regardless. The format check was never the defense, which
  is the point of D-046.

### D-079 — Recorded: ungrounded citations occur, and vary run to run
- **Observations (2026-09-21, `gpt-4o`, real API, identical question and code):**

  | Run | Shape | Ungrounded citations |
  |---|---|---|
  | milestone 2 | 1 search, 3 papers | 0 |
  | milestone 4, run A | 3 rounds, ~9 subtopics | **1** (`2113.11460`, fabricated) |
  | milestone 4, run B | 3 rounds, ~9 subtopics | 0 |

- **The correction worth keeping.** Run A alone suggested "hallucination rises with recursion
  depth", and that reading was written down before run B existed. Run B — same code, same
  question, same model — produced none. So the honest claim is narrower: **ungrounded citations
  happen, and they vary between identical runs.** Whether depth changes the *rate* is unresolved,
  and two runs at one depth cannot resolve it.
- **Why this is logged anyway:** it is the first real evidence that the grounding check catches
  something a reader would otherwise have believed, and it is a standing warning against drawing a
  trend from consecutive runs — which is exactly what nearly happened here.
- **Competing explanations, untested:** more rounds means a longer `<papers>` block and more
  distance between a claim and its abstract; or plain sampling variance. Only N runs per depth
  separates them.
- **Consequence:** `citation_violations` is **data**, not an assertion (D-078) — which is what
  D-046 designed it to be. See Open → "Measure hallucination rate against recursion depth".

### D-080 — Frontend: htmx, plus a small `EventSource` for the review pane (settles O-6)
- **Decision:** server-rendered Jinja2 templates in `api/`, htmx for the page, the form, run
  history and the node-progress trail. The token stream is handled by ~20 lines of vanilla JS
  using `EventSource` directly, buffering text and re-rendering markdown with a small library.
  No `frontend/` directory, no build step, no `node_modules`.
- **Why the earlier "htmx, no JavaScript" framing was wrong** (verified 2026-09-21 against the
  current docs): the SSE extension is `htmx-ext-sse@2.2.4`, separate from htmx 2.0.x core, and its
  documented attributes are `sse-connect`, `sse-swap`, `hx-trigger="sse:<name>"` and `sse-close`.
  The docs show **no example combining `sse-swap` with `hx-swap`**, so appending streamed tokens
  is undocumented. Within pure htmx the alternative is replace-semantics — re-swapping the whole
  review block on every token, hundreds of times for a 500-word review.
- **And JavaScript was always required anyway:** htmx swaps HTML; the review is markdown
  accumulating token by token. Either the server re-renders the accumulated review on every chunk,
  or the client does. "No JS" was never achievable for this feature, and claiming it was an
  advantage of htmx was wrong.
- **Why still htmx for the rest:** the node-progress trail *is* a natural `sse-swap` target — one
  event per finished node, replacing a progress element, which is exactly what the extension
  documents. Run history and the question form are ordinary form posts. Each tool does what it is
  actually good at.
- **Rejected:** SvelteKit — a component model genuinely handles accumulating tokens and
  re-rendering markdown better, and it is closer to Open WebUI's real shape, but it costs a second
  toolchain, a build step and a deploy story for a project whose contribution is the agent.
  Also rejected: dropping token streaming entirely, which would discard the `messages` stream that
  D-016 chose LangChain chat models for and D-037's integration test exists to prove.

### D-081 — The run lifecycle: `POST` creates, `GET` streams, resumes or replays
- **Decision:** `POST /runs` records the question and provider and returns a `thread_id` without
  executing anything. `GET /runs/{thread_id}/stream` drives the run and decides from the
  checkpoint which of three states it is in:

  | Snapshot | Meaning | Action |
  |---|---|---|
  | `created_at is None` | never started | `astream({"question": ...})` |
  | `next` non-empty | interrupted mid-run | `astream(None)` — resume |
  | `created_at` set, `next == ()` | finished | replay the saved review, **do not re-run** |

- **Why no background task registry:** verified 2026-09-21 — abandoning a stream mid-run (closing
  the async generator, which is what FastAPI does on client disconnect) leaves a checkpoint whose
  `next` is `('research_worker', 'research_worker')`, and `astream(None, config)` resumes from
  exactly there through to completion. **The pending `Send` fan-out survives**, which is D-071
  paying off. A dropped browser connection therefore costs at most the in-flight node, not the run.
- **Why the third state matters:** a completed run also has `next == ()`, so `next` alone cannot
  distinguish "finished" from "never started". Without the `created_at` check, a browser
  reconnecting to a finished run would re-execute the whole graph and re-bill the account.
- **Rejected:** running the graph inside `POST /runs` as a background task with the stream route
  following it. It needs a task registry, and "follow a graph another coroutine is running" is not
  something `astream` offers — it would mean a pub/sub layer or polling checkpoints. Resume gives
  the same disconnect-safety with no new machinery.

### D-082 — `AsyncSqliteSaver` is constructed by hand, never via `from_conn_string`
- **Decision:** the app builds its checkpointer as
  `AsyncSqliteSaver(conn, serde=build_serializer())` over an `aiosqlite.connect(...)` connection
  opened in the FastAPI lifespan, and calls `await checkpointer.setup()` once.
- **Why not `from_conn_string`:** verified 2026-09-21 — its signature is
  `from_conn_string(conn_string)` with **no `serde` parameter**; its body is `cls(conn)`. Using it
  silently discards the allowlist from `build_serializer()`, which is the one thing D-014 exists to
  enforce. The convenience method quietly bypasses the safety decision.
- **What actually happens without the allowlist** (measured, both paths, through a real SQLite
  file and a *fresh connection*): `Source` still restores correctly **today**, but with
  `Deserializing unregistered type ... This will be blocked in a future version`. So it is not a
  break now — it is a break later, announced only by a log line that is easy to miss on a server.
  That matches D-014's correction exactly.
- **Also confirmed:** `setup()` exists and must be called (it creates the tables), and
  `seen_paper_ids` round-trips as a real `set` through SQLite, not just through `InMemorySaver`
  (D-067).

### D-083 — SSE is hand-rolled with `StreamingResponse`
- **Decision:** the stream endpoint is an async generator yielding
  `event: <type>\ndata: <json>\n\n`, returned as
  `StreamingResponse(..., media_type="text/event-stream")`. No `sse-starlette`.
- **Why:** it is roughly fifteen lines, and the framing is the part a judge is most likely to ask
  about. CLAUDE.md's standing requirement is that every line be defensible; adding a dependency to
  hide the format works against that. Disconnect handling needs no library either — the generator
  is closed, and D-081 showed that leaves a resumable checkpoint.
- **Rejected:** `sse-starlette`, which handles framing, keepalive pings and disconnect detection.
  Genuinely fiddly parts, and worth revisiting if keepalive turns out to matter behind a proxy —
  but not worth a dependency for a localhost single-user app (D-008).

### D-084 — `citations_checked` is the completion signal (revises D-081)
- **Decision:** `ResearchState` gains `citations_checked: bool = False`, written only by
  `check_citations`, the terminal node. `get_run_state` returns `FINISHED` when it is true, and
  `INTERRUPTED` otherwise. `next` is not consulted at all.
- **Why D-081's discriminator was wrong.** It used `next == ()` to mean finished. Measured
  2026-09-21, breaking a stream immediately after `decompose` and closing the generator:

  | `stream_mode` | resulting `next` |
  |---|---|
  | `["updates"]` | `('research_worker', 'research_worker')` |
  | `["updates", "messages"]` | `()` |
  | `["updates", "custom", "messages"]` (production) | `()` |

  So an interrupted run and a finished run are indistinguishable by `next` — and the production
  stream modes are precisely the case that produces the ambiguous value. Under D-081 as written,
  a browser reconnecting after a mid-run disconnect would have been told the run was finished and
  handed an empty review.
- **Why `review` alone is not enough either:** `check_citations` runs *after* `synthesize`, so a
  run interrupted between them has a review and an empty `citation_violations`. Treating that as
  finished would report "no ungrounded citations" for a review that was never checked — D-062's
  failure shape reached by a different route. Only a field the terminal node writes can mean
  "this run completed".
- **Confirmed:** resume works regardless of `next`. `astream(None, config)` on a `next == ()`
  interrupted checkpoint re-ran `decompose` and continued to completion, with 2 arXiv requests
  rather than 4 — so a step-boundary interruption costs re-running the last node, which is what
  D-081 claimed. The resumability story survives; only the state detection was wrong.
- **Cost:** one bool in state. No reducer (single writer), no allowlist entry (a builtin).

### D-085 — Review markdown is rendered on the server, with raw HTML escaped (settles O-8's second half)
- **Decision:** `api/rendering.py` renders the review with
  `MarkdownIt("commonmark", {"html": False, "linkify": False})`. The browser never parses
  markdown: the `done` event carries `review_html`, already rendered and escaped.
- **Why this is a real path, not a hypothetical one:** the review is written by a model that has
  just read arXiv abstracts -- untrusted third-party text (D-055). An abstract carrying markup the
  model copies into its answer becomes live HTML the moment the page assigns it to `innerHTML`.
- **`html=False` is NOT the default.** Verified 2026-09-21: plain `MarkdownIt()` ships with
  `html=True`, and `Hello <script>alert('xss')</script>` came out verbatim. With the option off it
  renders as escaped, visible text. The tests pin this precisely because it is a default being
  overridden -- drop the option and the app is exploitable with no other symptom.
- **Confirmed safe by default:** `javascript:` and `data:` URLs in markdown link syntax are not
  turned into links at all. Pinned anyway, since it is a default being relied on.
- **Why server-side rather than in the browser:** the escaping lives in Python where it is
  testable, and the page needs no client-side markdown library or sanitizer -- two fewer
  dependencies, and no client-side sanitizer to configure wrongly.
- **While streaming, tokens are plain text** (`textContent`, never `innerHTML`), so no HTML is
  parsed on the streaming path at all. Formatting arrives with the `done` event. The tradeoff is
  that formatting appears at the end rather than progressively; the gain is that an entire
  vulnerability class is absent from the hot path.
- **Also escaped:** the run history fragment, via Jinja2 autoescaping -- the review is not the
  only model- or user-supplied string reaching the DOM.

### D-086 — Coverage is computed from state and reported separately (settles O-5)
- **Decision:** `agent/coverage.py` derives a `Coverage` summary from final state — what was
  explored, what found nothing, what failed and how often, how many entries were skipped, and
  **why the run stopped**. The API exposes it on `GET /runs/{id}` and renders it into a
  "Coverage and limitations" panel in the `done` event. Workers additionally emit `custom`
  events for live progress.
- **The gap this closes.** Of the three kinds of loss, only one was even recorded:
  `failed_subtopics` had no reader, `skipped_entries` had no reader, and a **zero-result
  subtopic was recorded nowhere at all** — D-021 marks it explored, so it read identically to a
  productive one. A review missing a third of its subtopics looked exactly like a complete one.
  That is the same failure shape as D-062, D-063 and D-069: success reported for less work than
  the reader assumes.
- **New state field:** `empty_subtopics`, written by the worker on a successful search that
  found nothing, with the same dedup reducer as `explored_subtopics` (D-067).
- **"Why the run stopped" is reported, not just "it stopped".** Hitting the depth ceiling means
  the run was *cut off* and more rounds might have found more; a round finding nothing new means
  the search converged. Reporting those identically would hide the distinction that most affects
  whether a reader should trust the review's completeness.
- **Computed in the agent, rendered in the API.** "What did this run fail to cover" is a
  research fact the thesis notebooks want, not a presentation detail. `Coverage` is a pure
  function of state, so it is testable with no graph and no HTTP.
- **Built from state, never asked of the model** — the same principle as D-070 and D-073: the
  code knows exactly what was lost, and a model asked to confess its own gaps may simply not.
- **Reported alongside the review, not inside it.** Appending to `review` would break the
  invariant that the streamed text equals the saved review (D-037's integration assertion) and
  would feed the coverage section to `check_citations`. Keeping them separate means `review` is
  exactly what the model wrote, and the *document* is review + coverage.
- **Empty for a clean run,** so a review that lost nothing is not padded with a list of nothing.
- **`get_stream_writer()` falls back to a no-op** when there is no runnable context (confirmed:
  it raises `RuntimeError: Called get_config outside of a runnable context`). Two reasons:
  progress reporting must never be able to break a run — a node whose research fails because its
  telemetry could not initialize is badly designed — and it keeps the worker callable directly,
  which is what `test_research_worker.py` relies on.

### D-087 — A sidebar of past runs, loaded as server-rendered fragments
- **Decision:** a persistent left sidebar lists every run. Clicking one issues
  `hx-get="/runs/{id}/view"` targeting `#loaded-run`; the route returns the rendered review,
  coverage and citation results as an HTML fragment. A run that is not finished offers to be
  started or resumed.
- **Why server-rendered rather than JSON + client rendering:** the review's markdown stays
  escaped in Python (D-085) and the page needs no second rendering path. It is also htmx's
  second real job here — before this it only refreshed the history list, which made D-080's
  choice look thinner than it was.
- **Why `#live-run` and `#loaded-run` are separate containers:** htmx replaces its target
  wholesale, so a single shared container would drop the elements the streaming code holds
  references to mid-run. Only one is visible at a time — two reviews on screen is a good way
  to misread which one you are looking at.
- **Opening a run must not execute it.** `GET /runs/{id}/view` only reads the checkpoint; a
  test asserts the model is not built. Reopening a finished run to re-read it would otherwise
  repeat every paid call (D-081's third state, the same trap in a new place).
- **Wording distinguishes never-started from interrupted.** `POST /runs` records a run without
  executing it, so "this run didn't finish" would report a failure that never happened. Not
  started offers *Start it*; interrupted offers *Resume it*.
- **No per-run status in the sidebar list:** that would need a checkpoint read per row (N+1)
  to show something the run view states on open. Revisit if the list grows large enough for
  status-at-a-glance to matter.
- **Not built:** follow-up questions on an existing run. A sidebar entry here is a finished
  piece of research, not a continuing conversation — see Open → "Multi-turn research sessions".

### D-088 — The evaluation harness records first, scores second (settles O-11)
- **Decision:** DeepEval, driven from pytest under two new markers, both deselected by default
  (D-036's pattern). `-m record` runs the real agent over a frozen ten-question set and saves
  a `Recording` per question; `-m eval` loads those recordings and scores them. Judge pinned to
  `gpt-4o-mini` and printed in every assertion message.
- **Why the two halves are separate:** the same reason `tests/agent/fixtures/arxiv/` exists —
  capture real output once, use it repeatedly. Recording is ~35 s and several paid calls per
  question; scoring reads a file. Fusing them would mean re-running the agent every time a
  metric changed, and would leave **no fixed point to compare against** when O-13 lands.
- **What it adds that the agent does not already measure:** `citation_violations` answers "is
  this cited ID a paper we retrieved?" exactly and for free (D-046, D-062). It does not answer
  *claim support* — "Smith et al. showed X [arXiv:1234.5678]" where the ID is real, the paper
  was retrieved, and the paper never says X. Faithfulness is exactly that metric.
- **The deterministic checks stay primary.** They run first in the file and need no judge. A
  measurement that needs no model is stronger evidence than a judged score; these complement
  it rather than replace it.
- **Why DeepEval over rolling our own, against this project's usual instinct:** D-042 chose
  `httpx` over the `arxiv` package and D-083 hand-rolled SSE, both to keep every line
  defensible. The claim check could have been ~50 lines the same way. What outweighs it is
  **comparability** — a thesis number measured with a recognized framework compares to
  published work, while a bespoke judge invites "how do you know it's right?". Worth recording
  that this is a deliberate exception, not an oversight.
- **Also rejected:** Ragas as the runner. Its synthetic question generator is the better tool
  for *building* a question set, but DeepEval is pytest-native, which matches 207 existing
  tests, `--strict-markers` and the `integration` marker convention with no new workflow.
- **Rules that keep the numbers meaningful,** each mirroring a mistake this log already
  records: `questions.json` is **append-only** (a reworded question silently invalidates every
  earlier recording); each `Recording` stores its `settings`, including `retrieval_unit`, so
  abstract-era and full-text-era recordings are distinguishable; thresholds are **floors, not
  quality gates**, because a red test would mean "the model wrote a worse review today", which
  is not a code regression.

### D-089 — The first baseline recording, and what it overturned

**Recorded 2026-09-21**, one question (`attention`) against the real OpenAI and arXiv APIs,
`gpt-4o`, `max_depth=2`, `max_subtopics=3`, `retrieval_unit="abstract"`. Scored with
`gpt-4o-mini`.

| | |
|---|---|
| Faithfulness | **0.944** |
| Answer relevancy | **1.000** |
| Ungrounded citations | **0** |
| Papers in the synthesis prompt | **82** |
| Distinct papers actually cited | **10** |
| Subtopics explored | 9, none empty, none failed |
| Stop reason | depth limit, not convergence |

Four findings, three of which contradict something already written down.

**1. The cost model in O-13 was wrong by roughly 12×.** It estimated "abstracts ≈ 2.5k
tokens", assuming ten papers. A real run accumulates 3 subtopics × 3 rounds × 10 results and
deduplicates to **82 papers ≈ 29k tokens** in a single synthesis prompt. The estimate assumed
one round of one subtopic and was never checked against a run.

**This inverts the argument for retrieval.** Full text in context would be 82 × ~8k ≈ **656k
tokens**, which does not fit any current context window. So retrieval is not a cost
optimization at this corpus size — it is the only way full text is possible at all.

**2. Only 10 of 82 supplied papers were cited — 88% of the context went unused.** That is
~25k tokens paid for on every run to no effect, and it points at a cheaper first win than
anything in O-13: **select which abstracts reach the prompt**, rather than passing everything
the recursion found. BM25 over the run's own papers, ranked against the question, needs no
corpus, no embeddings and no PDFs. See Open → "Rank the synthesis context".

**3. The run stopped on the depth ceiling, not on convergence.** `stopped_because` reported
*"the depth limit was reached after 3 rounds; more rounds might have found more"*. So D-075's
semantic exit never fired: every round found papers it had not seen. With ten results per
search on a broad question there are essentially always new papers, which makes "did this
round add anything?" too weak a stopping rule — and D-009 is explicit that `depth` should be
the backstop, not the mechanism. See Open → "The semantic exit is not firing".

**4. The baseline quality is already high, which is a problem for the RAG hypothesis.**
Faithfulness 0.944 and relevancy 1.000 leave little headroom. If full-text retrieval is
justified, it will not be by these two metrics on questions like this one — it is more likely
to show up in *depth* of the review (method detail, numbers, limitations) than in whether
claims are supported. Worth knowing **before** building O-13 rather than after: it suggests
the evaluation needs a metric that captures depth, and that the honest thesis claim may be
about what a review can *say*, not about whether it is faithful.

**Why this entry exists.** Every number above was an estimate in a previous entry, written
confidently. One recording, one question, 39 seconds and a few cents replaced four of them.
That is the argument for O-11 preceding O-13, and it is worth pointing at when asked why the
evaluation harness came first.

### D-090 — The ten-question baseline, and the four things it settles

**Recorded and scored 2026-09-21.** Ten questions, `gpt-4o`, `max_depth=2`, `max_subtopics=3`,
`retrieval_unit="abstract"`; judged by `gpt-4o-mini`. Raw data in `tests/eval/recordings/` and
`tests/eval/results.json`.

| | supplied | cited | used | stop |
|---|---|---|---|---|
| attention | 87 | 12 | 13.8% | depth |
| diffusion | 88 | 6 | 6.8% | depth |
| distillation | 89 | 8 | 9.0% | depth |
| federated | 83 | 7 | 8.4% | depth |
| gnn | 79 | 10 | 12.7% | depth |
| long-context | 59 | 9 | 15.3% | depth |
| moe | 85 | 7 | 8.2% | depth |
| quantization | 83 | 10 | 12.0% | depth |
| rag | 73 | 6 | 8.2% | depth |
| speculative | 71 | 6 | 8.5% | depth |
| **total** | **797** | **81** | **10.2%** | **10/10 depth** |

Faithfulness **0.970 ± 0.041** (min 0.875), relevancy **0.990 ± 0.020** (min 0.950).

---

**1. The context waste is universal, and worse than the single sample showed.** 797 papers
supplied, 81 cited — **10.2% used**, in a tight band of 6.8–15.3%. `attention` at 13.8% was
one of the *better* cases. Roughly **262k tokens paid for and unused across ten runs**.

The number that matters for a fix: **papers cited is 6–12, median 8, and does not scale with
papers supplied.** A run given 89 papers cites 8; a run given 59 cites 9. So a top-N cut is
not a tradeoff between cost and coverage — above roughly 20 papers the extra context buys
nothing measurable. Proposed `SYNTHESIS_TOP_N ≈ 20–25`, leaving headroom over the observed
maximum of 12 because BM25 ranking will not perfectly predict which papers the model cites.

**2. The semantic exit never fires. `depth` is doing 100% of the work.** All ten runs stopped
on the depth ceiling; not one converged. D-009 is explicit that `depth` is the backstop and
the semantic rule is the mechanism — the reality is exactly inverted, so D-075's "did this
round add a new paper?" is effectively dead code.

**3. The cause is planner paraphrasing, and it makes finding 2 inevitable.** Two consecutive
runs of the *same question with the same code* shared **2 of 9 subtopics** and **4 of 18
cited papers**. The planner readily produces near-duplicates — run 1 explored "Multi-head
attention in transformers", run 2 "Multi-head attention"; run 1 "Attention visualization in
transformers", run 2 "Attention visualization techniques".

`normalize_subtopic` is `casefold` + `strip` (D-017), so it catches none of these. Every round
therefore proposes subtopics that look new, which retrieve papers that are new, which keeps
the semantic exit from ever firing. **Findings 2 and 3 are one finding.**

This also resurrects D-022. Its ≥60% paper-overlap rule existed precisely to catch a
rephrased subtopic that returns papers already seen; D-074 deferred it and D-075 marked its
control-flow use superseded. That was wrong — D-075's round-level rule cannot catch a
*subtopic-level* duplicate, because a round containing one genuine subtopic and two paraphrases
still adds new papers overall.

**4. Run-to-run variance is large enough to invalidate N=1 comparisons.** 4 of 18 distinct
cited papers shared between identical runs; review length differed by 12%. Any O-13 comparison
drawing conclusions from one run per arm would be measuring sampling noise. Repeats are not
optional for the thesis claim.

---

**Two numbers this produces for earlier open items:**

- **Ungrounded citation rate: 1 in 81 citations, ~1.2%**, at `max_depth=2`. D-079 asked for a
  rate and had two anecdotes; this is the first real measurement, though still at a single
  depth setting.
- **Planner JSON failure: 1 run in 10 died** with `ValidationError` on an unparseable plan,
  and the same question succeeded on retry. A full-depth run makes up to four planner calls,
  so a ~3% per-call failure rate implies roughly **11% of runs dying**. D-070 chose "crash the
  run" with a note to revisit once `gap_check` could re-plan; that revisit now has evidence
  behind it rather than a hypothetical.

**The ceiling effect is confirmed across ten questions, not one.** Faithfulness 0.970 and
relevancy 0.990 leave no room for O-13 to demonstrate improvement on either. A specificity
metric is not a nice-to-have — without it the comparison **could not detect a difference even
if one existed**, and the thesis would report "no measurable effect" from an instrument
incapable of measuring it.

### D-092 — Pruning measured: 76% fewer tokens, no detectable quality change

**Both arms recorded and scored 2026-09-21**, same ten questions, same settings apart from
`SYNTHESIS_TOP_N`. Raw data in `tests/eval/recordings/abstract-all/` and
`.../broad-abstract-top20-d2-fixed/`; scores in `tests/eval/results.json`. (Both were renamed by
D-094 and O-14, which added the depth and question-set dimensions to the arm name.)

| | abstract-all | abstract-top20 | change |
|---|---|---|---|
| Papers shown (total) | 797 | 200 | **−75%** |
| Prompt tokens (~k) | 292 | 71 | **−76%** |
| Papers cited (total) | 81 | 85 | +5% |
| Ungrounded citations | 1 | 0 | — |
| Faithfulness (mean) | 0.970 | 0.983 | +0.012 |
| Relevancy (mean) | 0.990 | 0.982 | −0.009 |

Papers *retrieved* is unchanged at ~81 per run. Only what reaches the prompt differs, so the
recursion is doing the same work; the saving is entirely in synthesis.

**The claim this supports: pruning to 20 papers costs 76% fewer prompt tokens with no
detectable change in review quality.**

**The claim it does NOT support: that pruning improves quality.** Both deltas are firmly
inside the noise. The baseline faithfulness standard deviation is 0.041 across ten questions,
so the standard error of the mean is about 0.013 — the observed +0.012 is under one standard
error. Relevancy's −0.009 against sd 0.020 is about 1.4 standard errors. Neither is
distinguishable from run-to-run variation, and D-090 already established that variation is
large: two identical runs of the same question shared only 4 of 18 cited papers.

The per-question swings say the same thing more plainly. `federated` went 7 → 12 cited,
`moe` 7 → 12, while `rag` went 6 → 3 and `quantization` 10 → 7. Movement in both directions of
that size is what noise looks like, not an effect.

**The one result that is more than a cost saving:** citations did not fall. Cutting 75% of the
context produced *at least as many* citations (81 → 85). Whatever the 62 discarded papers were
contributing, it was not citations — which is precisely what D-090 predicted from the observed
independence of cited count and supplied count, and is now confirmed by intervention rather
than inferred from correlation.

A plausible mechanism, untested and not claimed: with 20 papers the model engages with each,
while with 82 it skims. Testing that properly would need the specificity metric below.

**What this settles about the evaluation itself.** Faithfulness and relevancy could not
distinguish these two arms — the instrument has no resolution here, exactly as D-090 warned
when the baseline came in at 0.970/0.990. If a 75% context cut is invisible to both metrics,
so is full-text retrieval. **The specificity metric is not optional before O-13**, or that
milestone will report "no measurable effect" from an instrument incapable of measuring one.

**Cost of the finding:** two ten-question recording sweeps (~6 minutes each) and one scoring
sweep, for a permanent ~76% reduction in the most expensive call the system makes.

### D-093 — A specificity metric, validated before use, and what it says about O-13

**The problem it solves.** Faithfulness (0.970) and relevancy (0.990) could not distinguish a
**75% context cut** (D-092). Both sit at their ceiling, so neither would detect full-text
retrieval either, and O-13 would report "no measurable effect" from an instrument incapable of
measuring one.

**Two halves**, following D-088's rule that the free exact measurement stays primary:

- `numeric_density` and `citation_density` -- deterministic, no judge, no cost, no variance.
  Crude, but they cannot drift.
- A **G-Eval specificity criterion** scoring concrete, checkable claims (reported numbers,
  named methods, experimental conditions, stated limitations) over topic-level description
  ("several approaches have been proposed").

**The instrument was tested before being trusted.** Two reviews of the same question, same
length, same confident register, citing the same four papers, differing *only* in
concreteness:

| | score |
|---|---|
| Deliberately concrete | **0.950** |
| Deliberately vague | **0.222** |

A 0.73 gap, against a demanded minimum of 0.3. A metric that returns ~0.95 for everything is a
number generator, not a measurement -- which is precisely what faithfulness became here -- so
this check is a permanent test rather than a one-off sanity run.

---

**Scored across both arms** (paired differences, since the arms share questions):

| metric | abstract-all | abstract-top20 | delta | SE units |
|---|---|---|---|---|
| specificity | 0.805 | 0.786 | −0.019 | −1.3 |
| numeric_density | 0.654 | 0.527 | −0.127 | −0.5 |
| citation_density | 1.858 | 2.071 | +0.213 | +1.2 |
| faithfulness | 0.970 | 0.983 | +0.012 | +0.6 |
| relevancy | 0.990 | 0.982 | −0.009 | −0.5 |

**Every delta is under 2 standard errors**, so pruning changed nothing measurable on any of
the five. That strengthens D-092's claim rather than weakening it: 76% fewer tokens, and now
five metrics agree there is no quality cost rather than two.

---

**Two findings that matter more than the comparison.**

**1. Specificity has headroom; faithfulness does not.** Real reviews score ~0.80 on an
instrument that puts concrete at 0.95 and vague at 0.22. So there is genuine room for O-13 to
move this number, unlike faithfulness at 0.97 or relevancy at 0.99. **This is the metric the
full-text comparison should be judged on**, and it now has a calibrated scale behind it rather
than an assumption.

**2. The reviews contain almost no numbers.** `numeric_density` averages 0.65 claims per 100
words, and **five of ten reviews contain none at all**. That is direct evidence for O-13's
premise: abstracts rarely state the measurements, so a review written from abstracts cannot
report them. If full text is worth its cost, this is where it should show first -- and it is
the *cheapest* metric of the five, needing no judge, no key and no money.

The cheapest instrument being the most discriminating for the hypothesis was not the expected
outcome, and it is worth keeping: whatever O-13 does to specificity, `numeric_density` can be
re-measured on every recording forever at zero cost.

### D-094 — Recursion depth measured: 2.8× the retrieval, no measurable quality gain

**The headline feature does not pay for itself on this question set, and the semantic exit
cannot be fixed as specified.** Both halves were measured today, and both contradict what was
written before the measurement — including what I wrote in D-090.

**What was run.** `max_depth` became a recordable dimension: `arm_name` now carries `-d{depth}`
(`tests/eval/recording.py`), and `EVAL_MAX_DEPTH` overrides the constant for a sweep. Three
arms, same ten questions, same `SYNTHESIS_TOP_N=20`, same judge — 1, 2 and 3 rounds.

| | d0 (1 round) | d1 (2 rounds) | d2 (3 rounds) |
|---|---|---|---|
| arXiv searches | 3.0 | 6.0 | 9.1 |
| Papers retrieved | 28.6 | 56.6 | **81.3** |
| **Papers in the prompt** | **20.0** | **20.0** | **20.0** |
| Papers cited | 8.0 | 8.7 | 8.5 |
| Specificity | 0.80 | 0.79 | 0.79 |
| Faithfulness | 0.98 | 0.97 | 0.98 |
| Relevancy | 1.00 | 0.96 | 0.98 |

Paired differences against d0, in standard errors: specificity −1.0 / −1.7, faithfulness
−0.6 / +0.2, relevancy −2.2 / −1.3. Only one exceeds 2 SE, and it is **non-monotonic** —
d1 worse than both d0 and d2 — which is the shape of noise, not of a dose-response. Every
other delta is inside the noise, and the two nominally *largest* effects favour **fewer**
rounds.

**The claim: three rounds retrieve 2.8× the papers and issue 3× the arXiv requests for no
quality difference this harness can detect.** Not "recursion is useless" — see the limits
below — but the burden of proof has moved, and it now sits on the feature.

**The mechanism, which the table makes obvious in one row.** `retrieval_context` is **20 at
every depth**, because D-091 ranks and truncates. So extra rounds cannot enlarge the synthesis
prompt; they can only change *which* twenty papers win it. A deeper run picks its top 20 from
81 candidates instead of 28 — a strictly better-informed choice — and the reviews are
indistinguishable. That is a much sharper result than "more search doesn't help": **BM25 over
28 papers already finds as good a top 20 as BM25 over 81.**

This also means D-091 and recursion are **substitutes, not complements**, which nobody
intended. Pruning was justified as a cost saving (D-092); it turns out to have quietly capped
the only channel through which depth could have acted.

---

**The semantic exit cannot be fixed the way it was scoped, and D-090's diagnosis was wrong.**

D-090 attributed the 10/10 ceiling stops to the planner paraphrasing itself, and the prescribed
fix was lexical: catch rephrased subtopics with overlap or Jaccard distance rather than an LLM
judge. Measured before building it:

- Within-run subtopic Jaccard is **median 0.14, p99 0.50**, and every 0.50 pair inspected is
  two genuinely distinct topics, not a rephrasing. **The planner does not paraphrase within a
  run.**
- Replaying recorded subtopics round by round, rounds are **almost entirely disjoint**:
  `attention` round 2 found 30 papers, **27 of them new**.

So there is nearly nothing for a lexical filter to catch, and no novelty rule — D-075's
round-level test, D-022's subtopic-level paper overlap, or a paraphrase filter — can fire when
27 of 30 papers are genuinely new. **The exit is not too weak; the premise that a broad arXiv
query ever runs out of new papers is false.** A filter built to the original spec would have
been dead code that passed its own tests, which is why it was measured first.

**Rejected on the evidence:** the lexical paraphrase filter (nothing to catch); a minimum-new-
papers threshold (would fire only on an arbitrary cutoff unrelated to whether anything was
*learned*); and tuning `max_depth` while the exit is broken — D-090 said settle the exit before
tuning the ceiling, and that ordering is now inverted, because there is no exit to settle.

**Not decided here: the new default for `MAX_DEPTH`.** The measurement says d0 is as good as
d2 on these ten questions, but the questions are all broad survey prompts ("how does X work"),
which is the case least likely to need decomposition. Changing the default is a `src/` change
and a product decision, and it deserves the narrow-question arm below first. Logged as an Open
item rather than silently applied.

**Limits of this result, stated plainly because the conclusion is unwelcome:** n=10, one run
per cell, no narrow or multi-hop questions, and three of the five metrics are at their ceiling
(D-092 already established they cannot resolve a 75% context cut). Specificity is the one with
headroom (D-093), and it moved −1.7 SE toward *fewer* rounds. This is evidence of **no
detectable effect**, which is not the same as evidence of no effect — but it is the only
evidence that exists, and it cost three recording sweeps.

---

**A defect in the measuring instrument, found by the same sweep.** `MAX_DEPTH` is bound by
value into both `graph.py` (routing) and `coverage.py` (reporting). `monkeypatch_depth()`
patched only `graph.py`, so the d0 and d1 runs routed correctly but `_stop_reason` compared
against the unpatched ceiling and wrote *"round N found no papers that earlier rounds hadn't
already seen"* into all twenty recordings — the signature of the semantic exit **firing**, in
the arms whose whole purpose was measuring that it never does. Read at face value it reverses
the finding.

This is D-062, D-069 and D-084 for the fourth time — **the system reporting more than it
earned** — and the first time in the evaluation harness rather than the agent. The recorder now
patches every module binding the constant, and
`tests/eval/test_recordings_are_consistent.py` asserts the invariant on committed JSON at zero
cost: a run that used its whole depth budget may never describe itself as converged. The
twenty affected recordings had the derived field recomputed with the production `_stop_reason`
itself, per arm; no agent output was altered, and nothing needed re-running, because the field
is a function of `rounds` and `max_depth` alone.

### D-095 — Depth fails on the questions built to need it, and the failure has a mechanism

**O-14's first question is answered: no.** D-094's null result was not an artifact of asking
broad survey questions. Ten *intersection* questions — each spanning two or three of the same
areas the broad set covers separately, so a single arXiv query cannot cover them — were
recorded at one round and three (`narrow-abstract-top20-d0-fixed`, `-d2`) and scored with the same
pinned judge.

| metric | d0 (1 round) | d2 (3 rounds) | delta | SE |
|---|---|---|---|---|
| papers retrieved | 27.5 | 63.3 | +35.8 | +4.9 |
| searches | 3.0 | 10.1 | +7.1 | +18.8 |
| **empty subtopics** | **0.0** | **1.1** | **+1.1** | **+2.9** |
| papers cited | 6.7 | 6.3 | −0.4 | −0.5 |
| specificity | 0.79 | 0.76 | −0.04 | −1.6 |
| numeric_density | 0.51 | 0.16 | −0.35 | −1.4 |
| faithfulness | 0.96 | 0.97 | +0.01 | +0.4 |

**Pooled with D-094's broad arms — 20 paired questions, the strongest statement available:**

| metric | d0 | d2 | delta | SE |
|---|---|---|---|---|
| papers retrieved | 28.05 | 72.30 | +44.25 | +10.7 |
| papers cited | 7.35 | 7.40 | +0.05 | **+0.1** |
| **specificity** | **0.80** | **0.77** | **−0.03** | **−2.2** |
| **empty subtopics** | **0.00** | **0.60** | **+0.60** | **+2.7** |
| numeric_density | 0.66 | 0.35 | −0.31 | −1.4 |

Reproduce with:

```sh
uv run python -m tests.eval.compare \
    broad-abstract-top20-d0-fixed broad-abstract-top20-d2-fixed \
    narrow-abstract-top20-d0-fixed narrow-abstract-top20-d2-fixed
```

**Two metrics clear 2 SE, and both say depth is worse.** Citations are identical to one
decimal place. Specificity — the one metric with measured headroom (D-093: 0.95 concrete,
0.22 vague) — is *lower* after three rounds, consistently in both question sets independently
(−1.7 broad, −1.6 narrow) before pooling made it −2.2.

**Honest reading of the statistics, because this is the claim a reader will attack.** The
table reports 13 metrics, so at n=20 a single crossing of 2 SE is unremarkable on its own.
What makes specificity more than that is the *replication*: the same direction, at similar
magnitude, in two independently recorded question sets, with `numeric_density` pointing the
same way (−0.7 and −1.4). The defensible claim is **"three rounds are no better than one, and
plausibly slightly worse."** Not "recursion harms quality" — that needs more than 20 questions.

---

**The mechanism, which is the genuinely new finding.** `empty_subtopics` counts subtopics that
returned zero papers. D-021 records those as a *success*, so they are invisible in every other
metric — and they are where the deep rounds go. One round never came up empty, in either set.
Three rounds came up empty **10.9% of the time on narrow questions**, against 1.1% on broad.

The `spec-quant` run shows it exactly. Round 1 decomposes the intersection question correctly:

```
speculative decoding in language models
post-training quantization methods
interaction effects in neural network optimization
```

Rounds 2 and 3 then drill further into the intersection — and two of those searches return
nothing at all:

```
interactions between decoding and quantization in neural networks   -> 0 papers
performance trade-offs in dual application of ... and quantization  -> 0 papers
```

**The planner is not malfunctioning. It is decomposing correctly into literature that has not
been written.** The papers that end up cited come from the broad round-1 searches; the deep
rounds ask increasingly specific questions of an increasingly empty shelf. This is why depth
fails *hardest* on exactly the questions designed to need it — the narrower the intersection,
the less exists at it.

That reframes the whole feature. Recursive decomposition was justified on the assumption that
a hard question hides more literature deeper down. Measured, the opposite holds on this
corpus: **depth finds less per search the deeper it goes, and the synthesis prompt is capped at
20 papers anyway (D-094), so what it does find mostly gets discarded before the model sees it.**

**What this does NOT settle:** whether a much larger `SYNTHESIS_TOP_N` would let depth pay off
(D-092 measured pruning as free, but never at depth 0 vs 2 jointly), and whether a corpus with
denser coverage than arXiv abstracts would behave differently — which is O-13's question, now
noticeably more interesting than depth.

**Cost of the finding:** two recording sweeps (~9.5 min) and one scoring sweep (~11 min).

### D-096 — Adaptive exits: two thirds of the searches removed, no quality cost

**What was chosen.** Two new conditions in `route_after_gap_check`, and **`MAX_DEPTH` left at
2**. Depth becomes adaptive rather than fixed: the ceiling stays a backstop, and a run that
has what it needs stops on its own.

1. **The synthesis prompt is already full.** `rank_sources` truncates to `SYNTHESIS_TOP_N`
   (D-091), so once a run holds that many papers another round can only reshuffle which ones
   the model sees. This is D-094's mechanism turned into a rule.
2. **The round came back empty** — at least half of its subtopics returned zero papers. This
   is D-095's mechanism turned into a rule.

**Measured on the real agent**, both question sets re-recorded and scored (20 paired
questions, `*-d2-adaptive` against `*-d2-fixed`):

| metric | fixed ceiling | adaptive | delta | SE |
|---|---|---|---|---|
| arXiv searches | 9.60 | **3.25** | −6.35 | **−23.2** |
| papers retrieved | 72.30 | 29.00 | −43.30 | −10.9 |
| rounds | 3.00 | **1.05** | −1.95 | −39.0 |
| empty subtopics | 0.60 | 0.10 | −0.50 | −2.4 |
| papers cited | 7.40 | 7.95 | +0.55 | +1.0 |
| specificity | 0.77 | 0.78 | +0.01 | +1.2 |
| faithfulness | 0.98 | 0.98 | +0.01 | +0.5 |

**−66% of the arXiv traffic and −60% of the retrieval, with every quality metric flat or
nominally better.** Against the cheapest arm (`d0-fixed`) every single metric is within 2 SE,
so the adaptive run matches one-round cost *and* one-round quality while keeping the recursion
available.

**19 of 20 runs stop after one round.** The twentieth, `fed-privacy`, found only 19 papers in
its first round — one short of the cap — and correctly took a second, reaching 47. That single
case is the entire reason `MAX_DEPTH` was not simply set to 0.

---

**Why not `MAX_DEPTH = 0`, which D-095 recommended and which is what was asked for.** It buys
the same saving. What it does not buy is the *reason*: a constant records a conclusion, a rule
records the mechanism, and the run can then explain itself in the coverage panel ("enough
papers were found to fill the synthesis context"). It would also cap `fed-privacy` and any
future question like it at a round that demonstrably had not finished. **Rejected on the
grounds that it hides why, not on the numbers** — the numbers are equivalent.

**Why the requested yield exit could not work alone.** The instruction was to stop when a
search comes back empty. Measured first: **round 1 never comes back empty** — 0 of 20
questions, in both sets. Empties appear only in rounds 2 and 3, which is *after* the cost has
been spent. An empty-based rule alone would therefore have fired only once the run was nearly
over. It is implemented anyway, because it is correct and it matters the moment `MAX_DEPTH` is
raised, but the exit that actually does the work is sufficiency. Had this not been measured
first, the shipped feature would have been the one that cannot fire — the D-094 mistake again.

**Honest flag: one ungrounded citation appeared** (`rag`, broad set) where the fixed arms had
none across 40 recordings. One occurrence in 20 runs is not a rate, and nothing in the change
touches citation grounding — but it is recorded here rather than omitted, per D-078, and is
worth watching on the next sweep.

---

**A defect found while verifying this, of the same class as D-094's.** The first adaptive
recordings reported *"round 1 found no papers that earlier rounds hadn't already seen"* — the
signature of D-075's semantic exit — when they had in fact stopped with a full prompt.
`coverage._stop_reason` kept its own copy of the stopping rules and knew nothing about the new
exits.

That is the second time routing and reporting drifted (D-094 was the first, over `MAX_DEPTH`),
and both produced a fluent, confident, wrong sentence in the panel a reader uses to judge a
review. The rules now live in **`agent/exits.py`**, imported by `graph.py` (which routes on it)
and `coverage.py` (which reports it), taking primitives so neither imports the other and the
code-map's downward-imports rule still holds. `tests/agent/test_exits.py` pins the invariant:
each exit fires on a state built for it, the router stops whenever any fires, and the panel
names the one that actually did.

**Cost of the finding:** two recording sweeps and one scoring sweep (~25 min).

### D-097 — A per-run nonce fence around the `<papers>` block (settles O-8's remaining half)

**The hole.** `format_papers` wrapped untrusted arXiv text in a *fixed* `<papers>` delimiter.
D-055 recorded the gap when the block was introduced: an abstract containing the literal
closing tag ends the data block early, and everything after it reads to the model as though it
came from the system prompt. Nothing about the resulting review looks wrong — which is the
attack, and the same shape as every other bug this project has found.

**Decision.** The delimiter carries a random per-run token from `secrets.token_hex(8)`:
`<papers-a1b2c3d4e5f60718>` … `</papers-a1b2c3d4e5f60718>`, and the system prompt names that
exact token. Untrusted text cannot forge it without guessing 16 hex characters.

**Per run, not per process.** A process-wide constant would leak the moment one review quoted
an abstract, unlocking every later run on the same server.

**The payload is not scrubbed.** A paper whose abstract really contains `</papers>` is
legitimate data and the model must see what it said. Stripping it would corrupt the evidence
to defend the frame; the fence defends the frame without touching the data.

**Rejected alternatives.** *Escaping the tag inside abstracts* — changes the text the model
reads and silently alters `retrieval_context`, so every faithfulness score shifts. *Dropping
the entry* — a D-021-shaped loss, invisible and unfalsifiable. *A JSON payload instead of
tags* — genuinely unforgeable, but it rewrites the prompt format the citation rules and all 80
recordings are built around, for a defence the nonce already provides.

**What this does NOT fix, stated so the fence is not mistaken for a complete defence.** An
abstract that writes "ignore your instructions" in plain prose is unaffected by any delimiter.
That residual risk is carried by the prompt framing (D-055), by citations being validated
against papers the run actually retrieved (D-046), and by the rendered review being escaped
(D-085). **Prompt injection is mitigated here, not solved.**

**Why now, before O-13.** A full paper is roughly 40x more attacker-controllable text than an
abstract. Closing this after the corpus lands means closing it against a much larger surface.

**Compatibility constraint that shaped the API.** `format_papers(sources)` with no fence keeps
its exact current output, because `tests/eval/test_record.py` builds `retrieval_context` with
it and all 80 committed recordings contain that text. Faithfulness is judged against it, so
changing the unfenced form would quietly make new recordings incomparable to the baseline they
exist to be compared against — a D-088 violation no test would otherwise catch. The recorder
passes no fence deliberately, and `test_papers_fence.py` pins the unfenced string exactly.

## Open (proposed, not decided)

**Settled 2026-09-20:** O-1 → D-064, O-2 → D-065, O-3 → D-066.
**Settled 2026-09-21:** O-4 → D-077, O-5 → D-086, O-6 → D-080, O-7 → `agent/runner.py`
(see D-081), O-8's XSS half → D-085. The remaining numbering is unchanged so earlier references
stay valid.

- **Multi-turn research sessions.** The sidebar (D-087) lists past runs to reopen and read.
  It does not let you ask a follow-up on one, which is what "session" means in ChatGPT or
  Claude. Doing that properly is a milestone, not a UI change, and the open questions are
  real: does a follow-up start a new `thread_id` or extend the existing one; do `sources` and
  `explored_subtopics` carry over (probably yes — reusing `seen_paper_ids` is most of the
  value); does `depth` reset (probably yes, or a third question can never search); does the
  planner see earlier rounds' subtopics as explored, and does the new review replace or extend
  the old one. Note the graph currently ends at `check_citations`, so it has no notion of a
  second question against existing state.

- ~~**Rank the synthesis context.**~~ **Settled → D-091, measured in D-092: −76% prompt
  tokens, citations unchanged.** Original note kept for the reasoning: Measured in D-089: a real run put
  **82 papers** in the synthesis prompt and the review cited **10**. That is ~25k tokens per run
  bought for nothing, and it is fixable without a corpus, embeddings or PDFs — rank the run's
  own papers against the question with BM25 and pass the top N. Proposed: reuse the FTS5
  machinery O-13 needs anyway, over a temporary in-memory index of just this run's papers, so
  the ranking code is written once and serves both. **Measured across ten questions (D-090): 10.2% of supplied papers are cited, and the cited
  count is 6-12 regardless of how many are supplied.** So `SYNTHESIS_TOP_N` around 20-25 keeps
  headroom over the observed maximum while cutting roughly three quarters of the prompt.
  Build it as a switchable parameter, not a replacement, so both arms are reproducible from
  one codebase. **Likely worth more than full text**, and far cheaper to build.

- **The semantic exit is not firing.** D-075 stops a run when a round adds no paper the earlier
  rounds had not seen. Measured in D-089: the run ended on the **depth ceiling** instead, having
  found new papers in all three rounds — which D-009 explicitly says should not be how runs
  normally end. With ten results per search on a broad question there are nearly always new
  papers, so the rule is too weak to fire. Proposed alternatives, none decided: require a
  *minimum number* of new papers rather than one; compare new papers against those actually
  *cited* rather than merely retrieved; or accept that broad questions legitimately run to depth
  and say so in the coverage report. **Measured across ten questions (D-090): 10 of 10 runs stopped on the depth ceiling; not one
  converged.** And the cause is now known — the planner paraphrases, two identical runs shared
  only 2 of 9 subtopics, and `normalize_subtopic` (casefold + strip) catches none of
  "Multi-head attention in transformers" vs "Multi-head attention". So every round looks new,
  retrieves new papers, and the exit never fires. **This resurrects D-022:** its subtopic-level
  paper-overlap rule is exactly what catches a rephrased subtopic, and D-075's round-level rule
  provably cannot, since a round with one real subtopic and two paraphrases still adds papers.
  Settle before `max_depth` is tuned — tuning a ceiling that is doing all the work is tuning
  the wrong thing.

  **Closed 2026-09-21 → D-094, which overturned the diagnosis above — including the part I
  wrote.** The planner does *not* paraphrase within a run (subtopic Jaccard median 0.14; the
  0.50 outliers are genuinely distinct topics), and rounds are near-disjoint in what they
  retrieve (`attention` round 2: 30 papers found, 27 new). No novelty rule can fire on that
  data — not D-075's round-level test, not D-022's subtopic paper-overlap, not a lexical
  paraphrase filter. The exit is not too weak; **the premise that a broad arXiv query runs out
  of new papers is false.** Closed as unbuildable as specified. What replaces it is O-14.

- ~~**O-14 — What should `MAX_DEPTH` default to, and should the exit be value-based?**~~
  **Settled 2026-09-21 → D-096.** Neither, as it turned out: the ceiling stays at 2 and the
  exits became *adaptive*, which buys the same saving while keeping the reason visible and
  the one thin question in twenty able to take a second round. The recommendation below to
  lower `MAX_DEPTH` was **not** taken; the original text is kept because the reasoning that
  led there is still the reasoning behind the rules that replaced it. Opened by
  D-094, which measured 1, 2 and 3 rounds and found no quality difference the harness can
  detect, at 2.8× the retrieval. Two questions, in order:

  1. ~~**Does depth help on questions built to need it?**~~ **Answered 2026-09-21 → D-095:
     no, and it fails hardest exactly there.** Ten intersection questions recorded at d0 and
     d2: citations unchanged (+0.1 SE pooled), specificity *lower* (−2.2 SE pooled), and
     10.9% of deep searches returning zero papers because the planner decomposes correctly
     into literature that does not exist. Original note kept: the frozen ten are all broad survey
     prompts — the case least likely to need decomposition, so D-094 may be measuring the
     easy case rather than the feature. **If depth shows no gain there either, recursion is
     decoration and the honest thesis claim says so.**
  2. **If it does help, the exit has to be value-based, not novelty-based.** Every rule
     considered so far asks "did this round find anything *new*", which D-094 shows is always
     yes. The question that actually matters is "did this round change the *review*" — e.g.
     stop when a round contributes no paper that survives BM25 ranking into the top `N`
     (`rank_sources` already computes exactly this, and it is free and deterministic, so it
     fits D-088's rule that the free exact measurement stays primary).

     **This rule's firing rate is currently an estimate, not a measurement, and must not be
     quoted as one.** The intuition is that with 81 papers retrieved and 20 reaching the
     prompt, most of a later round is discarded — but a recording stores only the *final*
     top 20, not which round each paper entered in, so nothing on disk can confirm it. Testing
     it needs per-round paper ids in the recording and a re-record. D-094 exists precisely
     because a plausible written estimate was wrong four times over (D-089); this one gets the
     same treatment before it earns a decision number.

  **Recommendation after D-095, for your decision — this is `src/`, so it stays here until you
  confirm it.** The evidence base is now 20 paired questions across two independently recorded
  sets, not the n=10 that made me withhold a recommendation before.

  **a. `MAX_DEPTH = 1`** (from 2 — i.e. two rounds, not three). The pure evidence supports
  **0**: one round matches three on every quality metric and beats it on specificity. The
  reason to keep one refinement round anyway is that *every* question measured so far is a
  literature-review prompt answered from abstracts, and d1 was never recorded on the narrow
  set — so 0 would be extrapolating past the data in the direction that happens to be
  convenient. `MAX_DEPTH = 1` halves the cost of a run against today's default, keeps the
  architecture's recursion live and demonstrable, and is inside what was measured. If the
  narrow d1 arm later matches d0, dropping to 0 becomes defensible.

  **b. Replace the novelty exit with a yield exit** — stop a round when its searches come back
  empty. D-095 makes this the *only* rule with measured support: `empty_subtopics` is the one
  effect of depth that clears 2 SE, it is free and deterministic, and it fires precisely when
  the planner has started querying literature that does not exist. Concretely, in `gap_check`:
  stop if a round's subtopics returned zero papers for at least half of them. Note this needs
  `empty_subtopics` to be readable *per round*, which state does not currently expose — it is
  an accumulating list, so the round's slice has to be derived the way `seen_before_round`
  already does it for `seen_paper_ids` (D-075).

  **c. Rejected: the value-based (top-N contribution) exit** proposed above. It is more
  complex than (b), its firing rate is still unmeasured, and D-095 gives (b) a mechanism it
  does not have. Keep it in mind only if `SYNTHESIS_TOP_N` is ever raised enough that depth
  starts paying off.

  **What would overturn (a):** a question type genuinely unlike these twenty — a question
  whose answer depends on tracing citations across papers rather than surveying a topic. Worth
  noting as the honest boundary of the claim rather than pretending 20 questions settle
  everything.

- ~~**Evaluation needs a depth metric.**~~ **Settled → D-093**, and it found the number O-13
  should be judged on: reviews score ~0.80 specificity with headroom to 0.95, and contain
  almost no numbers at all (five of ten have none). Original note kept: D-089 measured faithfulness 0.944
  and relevancy 1.000 on the abstract-only baseline, which leaves almost no headroom for O-13 to
  demonstrate improvement on either. If full text is worth its cost, the gain is in what the
  review can *say* — method detail, reported numbers, stated limitations — not in whether its
  claims are supported. Proposed: a G-Eval criterion scoring specificity (does the review cite
  concrete methods and results, or only describe topics?), so the comparison can actually
  distinguish the two retrieval units. Without it the thesis claim risks being "no measurable
  difference", from a measurement that could not have detected one.

- **Measure hallucination rate against recursion depth (`notebooks/`).** D-079 records one
  observation in each direction; a rate needs N runs per depth on the same questions, counting
  `len(citation_violations)` and the number of citations. Proposed: a small harness that runs the
  graph at `max_depth` 0, 1 and 2 over a fixed question set, since `max_depth` is already tunable
  (D-025). Outputs a number the thesis can state: "ungrounded citations per 100 citations, by
  depth". Worth doing before tuning `MAX_DEPTH` or `MAX_SUBTOPICS` — otherwise those are guesses.
  Note the run is paid, so the question set should be small and fixed.

- **Retire or repurpose D-022's ≥60% paper-overlap check.** D-075's round-level rule ("did this
  round add a paper we hadn't seen?") supersedes its control-flow purpose, and D-074 established it
  cannot run before dispatch anyway. Proposed: mark the overlap *skip* superseded, and decide
  separately whether to record the overlap ratio per subtopic as thesis evidence about how much
  subtopics duplicate each other. Cost of recording it: one more state field and reducer, with no
  consumer in the agent itself.

Each item lists the real options with their tradeoffs and a recommendation. A recommendation
here is **not** a decision — it moves into the log with a new ID only once confirmed.
Grouped by the milestone that forces the choice.

---

### Milestone 3 — all settled 2026-09-20

O-1 (arXiv rate limiter) → **D-064** · O-2 (4xx vs 5xx) → **D-065** · O-3 (models per role) → **D-066**.
Nothing open blocks the `Send` fan-out.

### Milestone 4 — all settled

O-4 → **D-077** (recursion_limit = 15) · O-5 → **D-086** (coverage reporting).

### Milestone 5 — all settled 2026-09-21

O-6 → **D-080** (htmx + a small EventSource) · O-7 → **`agent/runner.py`**, whose shape came out
of D-081 and D-084. Nothing open blocks the web layer.

### Proposed next: measure first, then build (O-11 → O-13)

These three are sequenced deliberately. O-11 establishes a baseline, O-13 is the change worth
measuring, and O-12 sits between them as a product feature rather than a metric. Building
O-13 first would make "full text improved the reviews" unfalsifiable.

#### O-11 — An evaluation harness, before any RAG work

**Problem.** The project measures citation *validity* exactly, for free, with no LLM judge:
`citation_violations` answers "is this cited ID a paper we retrieved?" (D-046, D-062). Most
RAG projects have nothing this good. But nothing measures the next question down — **claim
support**:

> "Smith et al. showed X [arXiv:1234.5678]" — valid ID, paper retrieved, and the paper never
> says X.

That gap gets worse with retrieval, because a chunk can be topically adjacent without
supporting the claim. It is also what D-079's Open item needs and does not have: a harness
for measuring hallucination rate against depth.

| Option | Pros | Cons |
|---|---|---|
| **A. DeepEval** | **pytest-native**, which matches this repo directly — 207 tests, `--strict-markers`, and an existing `@pytest.mark.integration` convention for paid tests. An `@pytest.mark.eval` marker needs no new workflow. Covers agentic metrics, and this is an agent, not only a RAG pipeline | Another dependency, and its judge prompts are a black box you did not write |
| **B. Ragas** | Four RAG metrics needing no ground-truth labels; a good **synthetic question generator**, which is exactly what the fixed question set below requires | RAG-shaped rather than agent-shaped; less natural to run from pytest |
| **C. Roll your own claim check** | ~50 lines, defensible line by line, cheaper, and consistent with this project's pattern of preferring understood code to a dependency (D-042 chose `httpx` over the `arxiv` package; D-083 hand-rolled SSE) | For a thesis, "faithfulness measured with Ragas" is comparable to published work; "we wrote our own judge" invites *how do you know it's right?* |

**Recommendation: A for running metrics, optionally B to generate the question set.** Worth
naming the tension honestly: C is what this project would normally do, and the thing that
outweighs it is **comparability** — a thesis number that cannot be compared to other work is
worth much less than one that can.

**Design points that matter more than the framework choice:**
- **A fixed, committed question set (10–20 questions).** Results are only comparable across
  changes if the input is identical. Generated once, reviewed by hand, then frozen.
- **`@pytest.mark.eval`, deselected by default**, exactly as D-036 did for `integration`.
  Evaluation is paid and slow; it must never run on `uv run pytest`.
- **Pin and record the judge model.** A different judge produces different numbers, so an
  unrecorded judge makes two runs incomparable — the same mistake as an unrecorded `max_depth`.
- **The deterministic metrics stay primary.** `citation_violations` and D-086's coverage are
  exact and free; the LLM metrics complement them. Do not replace a measurement that needs no
  judge with one that does.

**Cost:** roughly $0.001–0.003 per test case for five metrics with `gpt-4o-mini` as judge, so
a 20-question set is cents per run. Cheap enough to run often, expensive enough not to run on
every commit.

#### O-12 — An in-band claim checker (the "reviewer agent")

**Problem.** `check_citations` verifies the ID. Nothing verifies the claim.

| Option | Pros | Cons |
|---|---|---|
| **A. One node, one call** — after `check_citations`, extract cited sentences and ask the model in a single call whether each is supported by its cited paper; record `unsupported_claims` | Exactly parallel to `check_citations`, and slots straight into D-086's coverage reporting: ID-level grounding *and* claim-level grounding, both surfaced | One more paid call per run, and it is an LLM judging an LLM — non-deterministic where the rest of this pipeline is not |
| **B. One call per claim** | More accurate; each judgement sees only one claim | N× the cost and latency for a review that may make twenty claims |
| **C. Don't build it; rely on O-11 offline** | No runtime cost, no new node | The *reader* never learns which claims are unsupported — only the thesis does. That is precisely the "reported success for less work than assumed" shape this project keeps fixing |

**Recommendation: A, after O-13.** Chunks give a claim checker far tighter context to judge
against than a 200-word abstract does, so it will work better once full text exists.

**Two things to keep straight:**
- **This is a product feature, not a thesis metric.** `unsupported_claims` is produced by the
  same system being evaluated, so using it as your quality number is self-assessment. O-11's
  independent judge is what the thesis reports; this is what the reader sees.
- **Keep it to claim support only.** A reviewer that critiques structure or completeness would
  be an LLM judging things code can already check, which cuts against D-070 (planner proposes,
  code decides) and D-075 (deterministic termination over an LLM judge). Claim support is the
  one thing here that genuinely cannot be checked deterministically — that is what justifies a
  model doing it.

#### O-13 — Milestone 6: a local-first corpus, BM25 first

**The shape.** A personal corpus that grows from the research actually done. Each *subtopic*
is answered from local papers when the corpus covers it, and arXiv is consulted only when it
doesn't — with anything newly fetched indexed on the way through.

```
subtopic
  ├─ sanitize into terms  →  FTS5 MATCH over the corpus, ranked by bm25()
  │
  ├─ corpus covers it?  ──yes──►  use corpus papers, no network
  │
  └───────────────────── no ───►  arXiv search
                                    ├─ index every result's abstract (free, no download)
                                    ├─ fetch full text for the top few only
                                    └─ use corpus hits ∪ new results
```

**Per subtopic, inside `research_worker` — not per run.** Checking the corpus at the top of a
run would bypass `decompose` entirely. In the worker it needs no new node, and a single run
can answer two subtopics locally while fetching for a third.

**On fallback, augment rather than replace.** Corpus hits below the bar are still real papers.

---

##### Why BM25 rather than embeddings, and why that order

**Decided: SQLite FTS5 with `bm25()` ranking, no embedding model.** Dense retrieval is demoted
to a *measured follow-on*, and hybrid fusion (RRF) to a decision after that.

1. **Exact jargon is what academic search runs on.** Verified 2026-09-21: `'mamba'` and
   `'FlashAttention'` rank their papers correctly under BM25. A 384-dim, 22M-parameter model
   blurs precisely these tokens, and literature search is full of them — *LoRA*, *Chinchilla*,
   *RWKV*, author names, method names.
2. **Zero dependencies.** FTS5 is compiled into the bundled SQLite (confirmed: 3.49.1, FTS5
   present). No fastembed, no ONNX runtime, no ~130 MB model download, and the asymmetric
   query/passage prefix trap never enters the critical path.
3. **It mirrors arXiv's own retrieval semantics** — the strongest argument. `build_search_query`
   already extracts keywords and arXiv matches lexically. If the local tier matches lexically
   too, then *"the corpus doesn't cover this"* means something consistent. With embeddings
   locally and keywords remotely, a local miss might mean only that **two retrieval methods
   disagreed**, and the sufficiency test would be measuring that disagreement rather than
   coverage.
4. **It is debuggable.** You can see which terms matched. A cosine score cannot tell you that.

**A reasoning error worth recording:** `sqlite-vec` was recommended partly *because it was
already installed*. Installed is not the same as warranted, and convenience stood in for
justification until the BM25 option was raised.

**Where dense retrieval genuinely wins**, and therefore what the follow-on experiment tests:
paraphrase and synonymy (*"attention mechanism"* vs *"scaled dot-product attention"*), and
long full-text chunks where the relevant passage never repeats the query terms. Both are real;
neither is obviously load-bearing for short, keyword-dense abstracts. Build BM25, measure with
O-11, and add vectors only if the numbers justify them — the same method that put the eval
harness before the corpus work.

---

##### LOCKED: `bm25()` score directionality

SQLite's `bm25()` returns the **negative** of the standard BM25 score, so that plain
`ORDER BY bm25(t)` sorts best-first. Measured 2026-09-21 on a 30-document corpus:

| term | docs matched | top `bm25()` |
|---|---|---|
| `attention` | 3 / 30 | **−1.820** |
| `proteins` | 5 / 30 | −1.355 |
| `model` | 25 / 30 | **−0.000** |

**Rules, to stop this being inverted silently:**

- **`ORDER BY bm25(table)` ascending is best-first.** Adding `DESC` returns the *worst*
  matches, with no error and plausible-looking output.
- **Convert at the boundary.** The retrieval function returns `relevance = -bm25(...)`, so
  every caller and every threshold above it is "higher is better". Negative scores must not
  escape the module that queries FTS5.
- **Absolute score thresholds are unusable** — and this is the substantive finding, not a
  style point. A term in more than half the corpus gets degenerate IDF and scores collapse
  toward zero **regardless of match quality**. The same query therefore scores differently on
  a cold corpus than a mature one, and differently for common versus rare jargon. Any
  `relevance > X` sufficiency rule would drift as the corpus grows, in a direction nobody
  would notice.

**Therefore sufficiency counts distinct papers, not scores.** A subtopic is covered locally
when at least `MIN_LOCAL_PAPERS` *distinct* papers appear in the top-k FTS5 results. Counting
papers measures breadth; counting chunks measures redundancy; thresholding scores measures
corpus size. This mirrors D-028, which already requires ≥3 results before the paper-overlap
rule means anything. `MIN_LOCAL_PAPERS` and `k` get **measured with O-11**, not guessed —
choosing them by intuition would be `recursion_limit = 150` again (D-077).

---

##### LOCKED: query sanitization, shared with the arXiv path

FTS5 has its own query language, so an unsanitized subtopic is an injection risk — not
theoretical: `'"unterminated'` raises `sqlite3.OperationalError: unterminated string`,
crashing the query outright (measured 2026-09-21).

**Refactor:** one extraction step, two formatters.

```
subtopic ──► search_terms()  ──┬──► build_search_query(terms)  →  all:a AND all:b   (arXiv)
             (clean + strip    └──► build_fts_query(terms)     →  a b               (FTS5)
              stopwords, lower)
```

`search_terms` is today's `build_search_query` cleaning, lifted out unchanged: strip
`PUNCTUATION_PATTERN`, drop stopwords, lowercase (D-051, D-059, D-063).

**Two properties that make this safe, both verified:**

- **Punctuation stripping removes every symbolic FTS5 operator** — `"` `*` `(` `)` `:` `^` —
  which is what prevents the `OperationalError` above and blocks `column:` filters. The same
  pattern already protects the arXiv path (D-051), so one fix serves both backends.
- **Lowercasing is a security property here, not just normalization.** FTS5's word operators
  are case-sensitive: `'attention AND transformer'` matched 2 documents as an operator, while
  `'attention and transformer'` matched 0 — it became an ordinary search term. So lowercasing
  neutralizes `AND`, `OR`, `NOT` and `NEAR`. **This must be commented and tested**, because
  someone "improving" the code by preserving the planner's capitalization would silently
  reintroduce operator injection with nothing failing.

`or`, `not` and `near` are not in `STOPWORDS` (`and` is), so they survive as harmless
barewords that add a non-matching term to the implicit AND. Worth adding to the stopword list
for that reason alone.

---

##### The two tiers (what makes the cold start bearable)

Downloading a PDF per search result is ~30 s per subtopic and ~90 s per round before a word is
written (D-064 applies to downloads too). So:

| Tier | Indexed | When | Cost |
|---|---|---|---|
| **Abstracts** | Every paper any search returns | Always, automatically | Free — already retrieved |
| **Full text** | Selected papers only | Papers the review cited, or on request | 3 s + extraction each |

Every arXiv search seeds the abstract tier, so the corpus is useful from the first run rather
than after a bulk-download phase. It also keeps today's behaviour as the comparison floor,
which is what makes D-088's `retrieval_unit` recordings a fair test rather than two variables
changing at once.

---

##### Staleness is a correctness problem, not a performance one

arXiv grows ~100 GB/month, so a corpus that answered a subtopic well in March **silently**
misses April's key paper. That is this project's recurring failure shape arriving somewhere
new (D-062, D-063, D-069, O-5).

The fix is not to defeat the cache but to say so. D-086's coverage section gains:

> *Answered 2 of 3 subtopics from the local corpus (7 papers, indexed between 2026-03-04 and
> 2026-08-21). No new arXiv search was performed for those subtopics.*

Open sub-question: whether a time rule should force a refresh, or whether reporting is enough.

---

##### Schema — one SQLite file, as D-007 already decided

```sql
papers (arxiv_id PK, title, authors, summary, published, url, indexed_at, has_full_text)
chunks (id PK, arxiv_id, tier, section, text)          -- tier: 'abstract' | 'full_text'
chunks_fts USING fts5(text, content='chunks', content_rowid='id',
                      tokenize='porter unicode61')      -- external-content index
```

`tokenize='porter unicode61'` keeps non-ASCII terms working, consistent with D-063. `tier`
lets retrieval be restricted to abstracts for a fair comparison against today's baseline.
No vector table until the follow-on experiment justifies one.

---

##### Legal position (re-read 2026-09-21)

arXiv prohibits **storing and serving** e-prints from your servers and asks that users be
directed to arXiv for downloads — but *"if you build indexes or tools based on the full-text,
you must link back to arXiv"*, so index building is explicitly contemplated. A gitignored
local cache on a single-user localhost app (D-008) is on the right side of that line and
matches D-003.

**The constraint to accept now:** if this is ever deployed for other people, the full-text
cache has to go. The abstract tier is unaffected — arXiv metadata may be stored and shared
(D-042).

---

##### What is still genuinely open

1. **`MIN_LOCAL_PAPERS` and `k`** — measure with O-11, don't guess.
2. **Full-text fetch trigger.** Papers the review cited? A user action? Both?
3. **Staleness policy** — report only, or force a refresh after N days?
4. **Chunking** for the full-text tier: section-aware beats fixed-size but is harder, and
   references should be dropped entirely.
5. **Corpus scope in coverage reporting.** "This run explored X" becomes "the corpus holds Y,
   this run added Z" — better, but D-086 has to be redesigned around it rather than patched.
6. **Dense retrieval as a follow-on**, then RRF hybrid if the measurement supports it.

**Sequenced before this:** O-8's nonce delimiter. A full paper is ~40× more
attacker-controllable text than an abstract.

### Ongoing / not milestone-gated

#### O-8 — Prompt-injection defenses

**Decided so far:** citation IDs restricted to the retrieved set (D-046); an unparseable marker
reported rather than ignored (D-062); papers passed as a delimited data block (D-055).

**The known gap:** an abstract containing the literal text `</papers>` closes D-055's block early,
so the rest of that abstract is read as instructions rather than data.

| Option | Pros | Cons |
|---|---|---|
| **A. A random nonce in the delimiter** — `<papers-a3f91c>` ... `</papers-a3f91c>`, generated per call | Untrusted text cannot guess the delimiter, which closes the gap completely rather than filtering for it. Standard practice | The prompt looks slightly odd; the nonce must be threaded from builder to prompt |
| **B. Strip or escape `</papers>` in abstracts** | A two-line change | A blacklist: it fixes this one string and nothing else. Casing and whitespace variants slip past |
| **C. Drop tag delimiters entirely** — a numbered list framed as data | No closing tag to forge | Weaker visual boundary for the model, and no evidence it resists injection better |

**Recommendation: A.** Cheap, complete for this gap, and easy to test — feed an abstract containing
`</papers>` and assert the block still encloses every paper.

**Still open beyond this:** no tools with side effects (easy to hold now, worth writing down before
tools exist), and sanitizing model output before the browser renders it — a review is
model-authored markdown going into a page, so it is an XSS path unless rendered safely. That one
belongs with milestone 5.

#### O-9 — Search both spellings of an accented term (from D-063)

**Problem.** `all:schrödinger` (20,184 results) and `all:schrodinger` (4,900) are different sets.
Authors spell it both ways and older ASCII-era submissions use the bare form. Neither contains the
other, so keeping only the accented form still misses papers.

| Option | Pros | Cons |
|---|---|---|
| **A. `(all:<term> OR all:<folded>)` when a term has non-ASCII letters**, folding via NFKD | Covers both spellings; targeted, since it only fires on affected terms | A more complex query string to debug. `ß` does not decompose under NFKD, so German is still partly broken. Unmeasured — it may add noise as well as recall |
| **B. Status quo (D-063): accented form only** | Simple, and already 22x better than the old behavior | Still misses ~4,900 papers on the motivating example |
| **C. Measure first in `notebooks/`, then decide** | The thesis needs a recall measurement anyway; this is exactly that experiment | Leaves the gap open in the meantime |

**Recommendation: C, then A if the numbers justify it.** This is a recall question, and guessing at
recall is how you end up defending a number you never measured.

#### O-10 — Stopwords are English and ASCII only

**Problem.** `STOPWORDS` holds English words, so `квантовые вычисления` keeps both tokens and ANDs
them into the query.

| Option | Pros | Cons |
|---|---|---|
| **A. Accept it, document it** | Zero work. The failure is *narrowing*, not corrupting — unlike D-063's bug, the terms are real words from the question | A long non-English question ANDs many low-value terms and can return nothing, which D-021 records as a success |
| **B. Drop tokens shorter than 3 characters** | Language-agnostic, catches most function words | Crude: kills legitimate short terms like "AI", "ML", "3D" |
| **C. Per-script stopword lists** | Correct | Real scope: a list per language, plus language detection, for a use case that may never arrive |

**Recommendation: A.** The honest framing is that this app is English-first today. Revisit only if
non-English questions become a real use case — and note that a *long* question is the sharper
version of this problem, English or not, since D-051 already warns that ANDing many terms can
return nothing.

---

### Summary

| # | Item | Recommendation | Needed by |
|---|---|---|---|
| ~~O-1~~ | arXiv rate limiter | **Settled → D-064** | ~~M3~~ |
| ~~O-2~~ | 4xx vs 5xx | **Settled → D-065** | ~~M3~~ |
| ~~O-3~~ | Models per role | **Settled → D-066** | ~~M3~~ |
| ~~O-4~~ | `recursion_limit` | **Settled → D-077** (15, measured minimum 13) | ~~M4~~ |
| ~~O-5~~ | Failures visible | **Settled → D-086** | ~~M4~~ |
| ~~O-6~~ | Frontend | **Settled → D-080** (htmx + a small EventSource) | ~~M5~~ |
| ~~O-7~~ | Public entry function | **Settled** — `agent/runner.py` (D-081, D-084) | ~~M5~~ |
| ~~O-8~~ | Prompt injection | **Settled → D-085 (XSS half) and D-097 (nonce fence).** Instruction-following injection remains mitigated, not solved | ~~Any time~~ |
| O-9 | Accent spellings | Measure recall first, then decide | Any time |
| O-10 | Non-English stopwords | Accept and document | Any time |
| ~~O-11~~ | Evaluation harness | **Settled → D-088** | ~~before O-13~~ |
| **O-12** | **In-band claim checker** | One node, one call; a product feature, not a thesis metric | After O-13 |
| **O-13** | **Local-first corpus, BM25 first** | SQLite FTS5, no embedding model; sufficiency counted in distinct *papers*; dense retrieval demoted to a measured follow-on | Milestone 6 |
| ~~O-14~~ | ~~`MAX_DEPTH` default + a yield-based exit~~ | **Settled → D-096.** Adaptive exits instead of a lower ceiling: −66% searches, quality flat, 19/20 runs stop after one round | ~~Now~~ |
