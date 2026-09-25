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

### D-098 — Three judgement calls in D-096, settled after review

D-096 shipped with three things decided by reasoning rather than measurement. Reviewed and
settled on 2026-09-24; each is recorded with what *would* overturn it, because none rests on
an experiment.

**1. The empty-round threshold becomes a named constant, and stays at one half.**

It was `empties_this_round * 2 >= dispatched` — a judgement disguised as arithmetic. It is now
`EMPTY_ROUND_RATIO = 0.5` in `config.py`, carrying the argument for the value and what would
settle it. At `MAX_SUBTOPICS = 3` the behaviour is unchanged: 2 of 3 empty stops the run.

Kept at 0.5 because the two neighbouring values are both wrong in a known direction: **1.0**
(all empty) would almost never fire, which is the D-095 behaviour the rule exists to prevent —
one productive subtopic of three kept those runs drilling. **1/3 or below** stops on a single
dead end, which is normal and not evidence of anything; the agent would quit on its first
unlucky query. `test_adaptive_exit.py` now asserts `1/MAX_SUBTOPICS < EMPTY_ROUND_RATIO < 1.0`,
so a future retune has to stay inside those bounds or fail a test.

**What would settle it:** record one arm at 0.34 and one at 1.0 and compare papers retrieved
against specificity. Deliberately *not* run — no recorded run has been observed stopping on
this rule at all, so there is nothing yet to tune against, and D-079's warning about rates
from thin data applies.

**2. The empty-round exit is kept, despite never having been observed firing.**

The case against was this project's own rule, the one that killed D-094's paraphrase filter:
don't ship what you have measured cannot fire. On reflection that rule does not apply, and the
distinction is worth stating precisely.

The paraphrase filter was **structurally** unable to fire — rounds are near-disjoint, so no
novelty rule can ever trigger, at any data. This one is merely **pre-empted**: `PROMPT_FULL`
stops 19 of 20 runs before it is reached. But `PROMPT_FULL` can only fire on a question with
*plenty* of literature. A **thin** question never reaches the cap, so nothing else stops it,
and it recurses to the ceiling drilling an empty shelf — precisely D-095's failure. The two
rules are complements covering opposite cases, not alternatives.

`fed-privacy` came within one paper of being that run: 19 retrieved in round 1, cap 20.

**What would overturn it:** a full sweep in which no run ever stops here *and* no run under
the cap ever wastes a round on empty searches. Then it is dead weight and should go.

**3. `MAX_DEPTH` stays at 2.**

Lowering it to 0 was the original request and D-095's recommendation. Rejected now for a
sharper reason than before: with the adaptive exits in place, `MAX_DEPTH = 0` would make exits
2, 3 and 4 **unreachable** — there is no second round to prevent — deleting the mechanism that
was just measured and built. The ceiling now costs essentially nothing (runs average 1.05
rounds and nothing has reached 3), while lowering it reintroduces the risk of capping a run
that legitimately needed more.

Lowering to 1 was also considered: no adaptive run has used a third round, so 1 would be
sufficient for everything observed. Rejected because a backstop nothing reaches is not
costing anything, and a backstop set too low is a silent quality loss — the asymmetry favours
leaving headroom.

**The honest framing for all three:** these are reasoned defaults with stated falsifiers, not
measured results. They are in the log at the same level of confidence they deserve, which is
lower than D-094's or D-095's.

### D-099 — The stop reason is always shown; the loss list stays conditional (settles O-15)

**Decision.** `render_coverage` is called unconditionally at both call sites
(`routes/runs.py`, `routes/pages.py`). The `is_complete` gate moves *inside* the panel, where
it now chooses the heading -- "Coverage" for a clean run, "Coverage and limitations" when
something was lost -- rather than deciding whether the panel exists at all.

**Why it changed.** D-086 hid the panel when nothing was lost, which was right while it only
ever listed losses. **D-096 changed what it has to say.** Runs now stop after a single round,
and a reader who watches three searches and then synthesis has no way to learn why it did not
go deeper. The sentence existed, was tested, and was never shown -- on precisely the runs that
provoke the question. Found by `/demo-check`, not by any test, because every test asserted the
gate rather than asking what a person would see.

On a clean run the panel now reads:

> **Coverage** — Searched 3 subtopic(s) over 1 round(s), finding 30 paper(s). The run stopped
> because enough papers were found to fill the synthesis context, so further rounds could only
> have changed which papers were used, not how many.

That is the answer to "why only one round?", which is the first thing a judge asks about the
project's headline feature.

**The distinction this draws, which D-086 conflated:** *what a run covered* is information for
every run; *what it lost* is information only when there is some. One panel, two rules.

**Rejected:** showing the panel only when the stop reason is *not* the depth ceiling
(backwards -- the ceiling is the case a reader most needs to see), and leaving it (the feature
most in need of explanation would have none in the UI).

**One nonsense case had to be handled first.** `exits.describe(None, ...)` returns "the run has
not finished" for an interrupted run, which would have rendered as "The run stopped because the
run has not finished." Harmless while hidden; on screen once unconditional. `NOT_FINISHED` is
now a named constant and the panel phrases that case separately.

**Also fixed, cosmetic:** `/favicon.ico` 404'd, the only console error on the page. An inline
SVG data URI in `index.html` stops the browser requesting it at all -- no route, no binary in
the repo. Console verified at **0 errors, 0 warnings**, down from 1 error.

### D-100 — The corpus store and its FTS5 retrieval (O-13, first increment)

**What landed.** `persistence/corpus.py` — the schema O-13 specified, abstract indexing, BM25
retrieval and the sufficiency helper — plus `arxiv.build_fts_query`, the second formatter over
the existing `search_terms` extraction. **No worker integration and no PDF fetching yet**, so
agent behaviour is unchanged and every recorded arm stays comparable.

**Why this slice first.** It is the part with no judgement calls left in it: the schema,
`bm25()` directionality and the sanitization refactor were all locked in the O-13 note before
any code existed. The parts that still need *measurement* — `MIN_LOCAL_PAPERS`, the top-k, and
whether full text pays for itself — come after, so they can be measured against a corpus that
already exists rather than guessed alongside it.

**Four decisions made while building it**, none of which the O-13 note had settled:

1. **`build_fts_query` ORs; `build_search_query` ANDs.** The asymmetry is deliberate. arXiv
   ANDs to narrow millions of papers (D-051); a personal corpus holds hundreds, so ANDing
   every term returns nothing and *"not covered"* would mean *"the corpus is small"* rather
   than *"the corpus lacks this topic"* — making the sufficiency rule measure corpus size.
   BM25 already ranks multi-term matches above single-term ones, so OR costs no precision at
   the top of the list; it only stops the tail being truncated to empty.

2. **Chunk text is `title + summary`, not the summary alone.** The method name a search is
   most likely to use ("FlashAttention", "Mamba") is usually in the title and only paraphrased
   in the abstract. Verified: a summary-only index misses `FlashAttention` entirely.

3. **`section` is `NOT NULL DEFAULT ''`, not nullable.** SQLite permits unlimited NULLs in a
   unique index, so a nullable `section` would let `UNIQUE(arxiv_id, tier, section)` pass while
   the same abstract was indexed repeatedly. That is a *correctness* bug, not tidiness: BM25
   would then rank a paper highly for having been **found often** rather than for matching the
   query, and nothing would look wrong.

4. **The module is synchronous.** These are microsecond queries on a local file, and staying
   sync keeps it pure and testable — the choice `agent/ranking.py` already makes. A caller
   inside the event loop that finds them slow should use `asyncio.to_thread` rather than
   making the module async.

**What the tests pin**, chosen for failure modes that are silent rather than loud: `bm25()`
ascending is best-first and relevance never escapes negative; re-indexing is idempotent;
sufficiency counts distinct *papers* (a ten-chunk full-text paper is one paper of coverage,
not ten); the tier filter isolates an abstracts-only arm; and six hostile subtopic forms
cannot break the query — with a **control test** proving the raw form really would have raised
`OperationalError`, so the defence is not being asserted against something harmless.

**Still open, and deliberately unmeasured:** `MIN_LOCAL_PAPERS` and the top-k. Choosing them
by intuition would be `recursion_limit = 150` again (D-077).

### D-101 — The worker seeds the corpus (O-13, second increment)

**What landed.** `research_worker` indexes every search result on the way through, and
`build_graph` grows an optional `corpus` argument that the API lifespan now supplies. Verified
end to end: one real run through the HTTP route leaves **3 papers, 3 chunks** in the corpus.

**Still no behaviour change.** Nothing *reads* the corpus yet — the review is written from the
same papers as before. That is the point: the corpus has to exist and fill before
`MIN_LOCAL_PAPERS` and the top-k can be measured rather than guessed (D-077's lesson).

**Why indexing here is free.** These abstracts were already fetched and paid for by a search
the run needed anyway. Without this they are used once and discarded, and the corpus would
require a bulk-download phase before it was useful at all.

**Three decisions, all about failure:**

1. **A corpus write must never fail a subtopic.** The search already succeeded and the papers
   are in hand; the run completes fine having never touched the corpus. Marking the subtopic
   failed would burn one of its two retries (D-020) and drop it from the review — turning a
   disk problem into missing research. Same reasoning as `_progress_writer`: storage, like
   reporting, is strictly additive.

2. **But the catch is `sqlite3.Error` only.** A `TypeError` from a malformed `Source` is a bug
   and must crash (D-023). A blanket `except Exception` would turn it into a progress message
   and hide it for as long as the corpus appeared to work.

3. **And the failure is reported, not swallowed.** A corpus that silently stops filling is
   this project's recurring shape arriving somewhere new (D-062, D-069, O-5): nothing looks
   wrong, and months later the local tier is mysteriously empty. It goes to `custom` progress
   rather than state, because it is an infrastructure problem, not a research finding.

**`corpus=None` is the baseline arm, not a convenience default.** The eval recorder passes
nothing, which is exactly what keeps the 80 committed recordings comparable to anything
recorded after the corpus lands — the switchable-parameter pattern D-091 used for
`SYNTHESIS_TOP_N` (D-088).

**A concurrency property worth stating, because it is load-bearing and invisible.** Workers
run in parallel under `Send`, sharing one connection. `index_sources` contains no `await`, so
the event loop cannot interleave two of them and the writes are effectively atomic. **Making
the corpus module async would introduce exactly the interleaving it currently cannot have** —
which is a second, stronger reason for D-100's sync decision than the one recorded there.

**Lifetime:** the corpus opens on its own synchronous connection to the same file as the
checkpoints and the runs table (D-007), and closes in a `finally` alongside them.

### D-102 — The test suite is kept off the network

**Symptom.** `uv run pytest` took anywhere between **14 and 97 seconds** for the same 581
tests, and the "slow test" moved between runs. Individual tests reported ~11.1 s, consistent to
the centisecond, but ran in 0.4 s in isolation.

**Cause.** A developer `.env` sets `LANGSMITH_TRACING_V2=true` with a LangSmith key that no
longer authorizes. Nothing in `src/` or `tests/` loads that file — the LangSmith SDK finds it
itself — so the suite had a network dependency that no test asked for and no file mentioned.
Every LangChain call was traced, and the flush blocked ~11 s on
`LangSmithAuthError: 401 Unauthorized` before giving up.

**Fix.** `tests/conftest.py` sets every tracing flag to `false` and *removes* the API keys at
import time, before LangChain loads. `python-dotenv` does not override variables that already
exist, so this wins over `.env` without editing it. Development tracing is unaffected: it only
applies under pytest. **Measured: 97 s → 13.5 s**, top duration 1.36 s.

**Two reasons this belongs in the suite rather than in a README note:**

- **A test suite that depends on an external service is not a test suite.** It fails, or
  crawls, for reasons unrelated to the code under test, and on a machine with no network it
  does both.
- **Traces carry the content.** Research questions, retrieved abstracts and generated reviews
  were leaving the machine on every `pytest` run, including the paid eval sweeps. That sits
  badly against this project's rule that users bring their own keys and nothing of theirs is
  baked in — and it was nobody's decision; it arrived through a file borrowed from another
  project (`LANGSMITH_PROJECT=langchain_academy`).

**Every spelling is disabled, not the one that looked right.** The flag actually in use was
`LANGSMITH_TRACING_V2`, which is neither of LangSmith's two documented names — so guessing
`LANGSMITH_TRACING` alone, as the first attempt did, changed nothing and made the cause look
like something else.

**The cost of not noticing sooner.** Three separate investigations went hunting for a code
regression that did not exist — first blaming the corpus connection, then SQLite lock
contention, then the rate limiter. Worth recording as a debugging lesson: **a duration
constant to the centisecond is a timeout, not computation**, and a slow test that is fast in
isolation is environmental.

### D-103 — The corpus stores `version`, and gains a migration

**Bug, introduced by D-100 and found on review.** `Source` has a validated `version` field.
The first corpus schema did not store it, so `load_sources` hardcoded `version=1`. A paper
indexed at v3 came back claiming v1 **while its `url` still ended `v3`** — internally
inconsistent, wrong in a way that reads as entirely normal.

It would have stayed wrong indefinitely, because **nothing downstream uses `version` today**:
citations key on the canonical id (D-044), ranking uses title and summary, and the review
never mentions it. A field that nothing reads is a field no test notices is broken.

**Fixed:** a `version` column, written by `index_sources` and read by `load_sources`, plus a
test asserting `version` and `url` cannot contradict each other.

**And a migration, which is the part worth more than the fix.** `CREATE TABLE IF NOT EXISTS`
does nothing to a table that already exists, so every corpus built by D-100 would have raised
`no such column: version` on the next read. `migrate()` runs on every `connect`, is
idempotent, and adds the column in place. Rebuilding instead would have been simpler and
wrong: it silently discards the `indexed_at` history the staleness report depends on (O-13).

Rows written before the fix cannot be repaired, only re-indexed — which is why the column
default is `1` and why that is stated rather than quietly assumed.

**Two smaller things settled in the same pass:**

- `covering_papers` dedups with `dict.fromkeys` rather than a list membership check — one pass
  instead of quadratic, and it states the intent (preserve first-seen order) more directly.
- **First seen wins on re-indexing, deliberately.** A paper already in the corpus is left alone
  even when arXiv serves a newer version, matching `merge_sources` (D-067). A corpus can
  therefore hold v1's abstract months after v3 appeared. That belongs in O-13's open staleness
  question rather than being fixed quietly, since refreshing means re-indexing chunks and
  invalidating BM25 statistics on every search — a cost worth paying only with evidence.

### D-104 — The sufficiency test O-13 specified cannot work; term coverage can

**O-13 specified the rule in advance:** *"a subtopic is covered locally when at least
`MIN_LOCAL_PAPERS` distinct papers appear in the top-k FTS5 results."* Measured against a
1669-paper corpus rebuilt from the 80 committed recordings, that rule **returns k every time,
for every query, in every corpus** — including when the ML corpus is asked about medieval
Flemish guilds. It cannot discriminate at all, because `build_fts_query` ORs the terms and any
non-trivial corpus contains something matching something.

This is D-094 happening again: a rule specified from reasoning, measured before being built,
and found unable to fire as intended. **That is now twice, which is worth noticing as a
pattern rather than an accident** — both times the rule was about *detecting absence*, and
both times the signal it keyed on was present almost everywhere.

**The test that works: term coverage.** Count how many of the top-k papers match at least half
the question's terms. A paper matching "attention" alone is noise; one matching "attention",
"transformer" *and* "models" is about the subtopic.

| population | papers in top-20 | matching >= half the terms |
|---|---|---|
| 20 in-domain questions | always 20 | **5 to 20** |
| 4 out-of-domain questions | always 20 | **0 to 1** |

`MIN_LOCAL_PAPERS = 3` sits in the middle of that gap, with margin on both sides. The tightest
in-domain margin is `spec-quant` at 5 — an intersection question, which is the right place for
the threshold to be tightest, since those are the subtopics most likely to be genuinely
uncovered.

**Half, not all.** `AND` over every term returns nothing on almost every query: the median
across 20 questions was **0** even on a same-topic corpus. Too strict to be a coverage test.

---

**The first version of this experiment was wrong, and how it was wrong is the lesson.** Its
negative population was *"the other nineteen questions"* — but every question in the frozen
set is machine learning, so a corpus of nineteen ML topics genuinely does hold attention
papers. That population was never a true negative. It scored **higher** than the positive one,
purely by being 20x larger: raw counts measure corpus size, exactly as O-13 warned raw scores
would.

The control that makes the measurement mean anything is a question from a genuinely different
domain, and the first design had none. Recorded because the mistake is easy to repeat: **a
negative control drawn from the same distribution as the positive is not a control.**

`OUT_OF_DOMAIN` in `tests/eval/corpus_coverage.py` is frozen for that reason — changing those
four questions changes what "not covered" means, the same way rewording a frozen evaluation
question invalidates every recording (D-088).

**Cost:** zero. The whole measurement runs off committed recordings with no API key and no
network — `uv run python -m tests.eval.corpus_coverage`.

**Still not wired in.** The corpus can now answer "do I cover this?", but nothing asks it yet.
Local-first retrieval is the next increment, and it changes agent behaviour, so it gets its
own arm and its own comparison.

### D-105 — A covered subtopic skips arXiv rather than augmenting it (O-13, third increment)

**O-13 said "on fallback, augment rather than replace". The measurements overturned the
replace half of that.** Two facts, both from data already on disk:

1. **Two runs of the same question retrieve ~89% different papers.** Mean Jaccard overlap 11%,
   median 8%, min 0%, measured across 20 questions and 80 recordings. arXiv's answer to the
   same question is close to a fresh draw each time.
2. **Those runs produce statistically indistinguishable reviews.** D-094 and D-095: every
   quality metric within 2 SE across four arms.

Together: **paper identity does not drive review quality — topical relevance does.** A review
written from the corpus's papers should therefore be as good as one written from today's
arXiv results, because "today's results" were already near-arbitrary among the relevant set.

That makes augmenting the worse option on its own terms. It keeps the network call, adds
papers that `rank_sources` truncates away at `SYNTHESIS_TOP_N` (D-091), and buys nothing any
instrument here can detect. **Skipping is the only version of local-first with a measurable
benefit**, and the benefit is the call itself: one request and one rate-limit interval per
covered subtopic, against a limiter that serializes at 3 s (D-064).

**The fallback stays pure arXiv**, not arXiv-plus-corpus-leftovers. One variable per arm
(D-088): `LOCAL_FIRST` toggles exactly whether a *covered* subtopic uses the network. Mixing
corpus hits into uncovered subtopics as well would change two things at once and make the
recorded comparison unreadable. Augment-on-fallback remains available as a later experiment.

**Off by default.** This changes what reaches the review, so it gets a recorded arm before it
becomes the default — the D-091 discipline, and the reason the 80 committed recordings stay
comparable.

**Reported, never hidden.** `local_subtopics` flows into `Coverage.local` and the panel says:

> **Answered from the local corpus:** *tiling attention kernels*. No new arXiv search was
> performed for those subtopics, so papers published since they were last indexed are not
> represented.

It sits in the **always-visible** part of the panel (D-099), not the loss list — answering
from the corpus is not a failure, but a reader judging how current a review is has to know.
arXiv grows ~100 GB a month, and silence is this project's recurring failure (D-062, D-069).

**Two ordering details that are load-bearing:**

- `build_search_query` runs **before** the corpus check, so a malformed subtopic fails the
  same way whether or not the corpus could have answered it. After the check, a
  stopwords-only subtopic would be served locally while crashing the remote path — two
  contracts for one input (D-073).
- A corpus *read* failure returns `[]` and falls through to arXiv. The D-101 rule applied to
  reads: a broken corpus costs latency, never the subtopic.

**What is still unmeasured, and is the next paid step:** whether a locally-answered review is
actually as good. The argument above is an inference from two measurements, not a measurement
of the thing itself. It needs a `LOCAL_FIRST=true` arm recorded against the current default.

### D-106 — Coverage also requires the corpus to hold every term

**A threshold validated on one input distribution, applied to another, is not validated.**
D-104 measured the half-the-terms rule on whole *questions* -- 7 to 12 terms each. Production
feeds it *subtopics*, which the planner writes at 2 to 4 terms. At that length "half the
terms" can be a single generic word.

Caught before the paid arm, by re-running D-104's control on subtopic-length inputs:

| probe | terms | strongly matching |
|---|---|---|
| `self-attention mechanism` | 3 | 20 |
| `coral reef bleaching` | 3 | 0 |
| **`CRISPR off-target effects`** | 4 | **20** |

`CRISPR off-target effects` scored **full coverage against a machine-learning corpus**, because
*target* and *effects* are ordinary ML words while *crispr* matched nothing at all. The rule
was measuring incidental vocabulary overlap.

**Fix: every query term must appear somewhere in the corpus**, checked with one `LIMIT 1`
query per term, short-circuiting on the first absent one. A corpus that holds no paper
containing *crispr* does not cover a CRISPR subtopic, however many papers mention *target*.

Re-measured across 21 probes -- 10 in-domain (subtopics and questions), 11 out-of-domain:
**every in-domain probe has all terms present; every out-of-domain probe has at least one
absent.** No overlap, including the CRISPR case the half-rule missed.

The check respects `tier`, so a term appearing only in full text cannot make an
abstracts-only arm look covered -- otherwise the tier filter leaks and the comparison arm
answers locally on the strength of text it was configured not to use (D-088).

**Why this was found at all:** the coverage rate on real recorded subtopics came out at
**100%**, which is a number worth distrusting rather than celebrating. Checking *why* it was
100% is what surfaced the distribution mismatch. It is still 100% after the fix, because the
seeded corpus genuinely holds these questions' papers -- but now for the right reason.

### D-107 — Local-first measured: 65 arXiv requests become 1, quality unchanged

**Both question sets recorded with `LOCAL_FIRST` on and scored against the current default.**
20 paired questions, same judge, one variable changed.

| metric | arXiv-only | local-first | delta | SE |
|---|---|---|---|---|
| **arXiv requests (total)** | **65** | **1** | −64 | — |
| **wall clock (both sweeps)** | **703 s** | **248 s** | **−65%** | — |
| papers cited | 7.95 | 7.90 | −0.05 | −0.1 |
| specificity | 0.78 | 0.78 | −0.00 | −0.4 |
| faithfulness | 0.98 | 0.99 | +0.01 | +1.1 |
| relevancy | 0.98 | 0.99 | +0.01 | +0.9 |
| ungrounded citations | 0.05 | 0.00 | −0.05 | −1.0 |

**Every quality metric inside 2 SE, and the two that moved at all moved upward.** Reproduce
with `uv run python -m tests.eval.compare broad-abstract-top20-d2-adaptive
broad-abstract-local-top20-d2-adaptive narrow-abstract-top20-d2-adaptive
narrow-abstract-local-top20-d2-adaptive`.

**Decision: `LOCAL_FIRST` defaults to on.**

**Why that is safe rather than optimistic.** An empty corpus covers nothing, so a first run
falls through to arXiv and behaves exactly as before — the feature cannot fire before it has
evidence to fire on. The switch works in **both** directions so the arXiv-only arm stays
reproducible from this codebase (D-088).

**The fallback path was exercised, not just the happy one.** `gnn` answered two subtopics from
the corpus and went to arXiv for the third — that is the one remaining request.

**`papers_retrieved` went up, 29 → 51.8**, because the corpus returns up to
`LOCAL_SEARCH_TOP_K` per subtopic where arXiv returns `ARXIV_MAX_RESULTS`. `papers_in_prompt`
stayed pinned at 20: ranking discards the surplus, D-094's mechanism again. More candidates,
same prompt, same review.

---

**What this experiment cannot see, stated plainly because the result is favourable.**

1. **It is the best case.** The corpus was seeded from earlier runs of *these exact
   questions*, so coverage was 100%. A corpus grown from adjacent rather than identical
   research will cover less, and the fallback will carry more of the load.
2. **It is one moment in time.** A topic researched repeatedly could stay answered from cache
   indefinitely while arXiv moves on — and this arm, recorded in a single afternoon, cannot
   distinguish a fresh corpus from a frozen one. The coverage panel reports which subtopics
   were answered locally (D-105) so the reader can judge, but nothing forces a refresh.

The second is the real open risk, and it is logged as **O-16** rather than defended against
with a guessed time limit — `recursion_limit = 150` again (D-077).

**Cost of the finding:** two recording sweeps (~4 min, down from ~12) and one scoring sweep
(~9 min).

### D-109 - The full-text tier: one glyph, 235 lost papers, and a retrieval shift

**The bug first, because it is the transferable part.** pypdf renders some glyphs --
mathematical bold, certain ligatures -- as **unpaired surrogates** (the U+D800-U+DFFF range).
Python holds those in a `str` without complaint; they are not encodable UTF-8, so SQLite
raised `UnicodeEncodeError` on insert. The enrichment pass died at **paper 126 of 360**: one
character in one PDF cost every paper after it, roughly an hour of rate-limited downloading.

That is D-053's shape exactly -- bad external data becoming an exception that takes down the
run rather than data the run records and continues past. Fixed at two levels on purpose:

1. **`extract_text` strips lone surrogates.** *Dropped*, not replaced with U+FFFD: they are
   decoding noise carrying no meaning, and a replacement character would be an indexable token
   matching nothing that also appears in any excerpt a reader sees. An `isascii()` fast path
   means almost every paper pays nothing.
2. **The pass catches `UnicodeError` and `DatabaseError` too.** Second line of defence, because
   a long unattended pass must degrade *per paper* -- any one paper must cost only itself.

A test pins that **real non-ASCII survives**. D-063 already cost this project a bug where
cleaning destroyed non-ASCII terms, and author names and titles routinely carry accents and
CJK; losing those would make papers unsearchable by the words that identify them.

**Then the same bug destroyed this file.** The script writing this entry put a literal
surrogate into its own text and called `Path.write_text`, which **truncates before it
encodes** -- so the encode failed, `decisions.md` was left at zero bytes, and a `git add -A`
committed the deletion of 2703 lines and 97 decisions. Recovered with `git checkout HEAD~1`,
which is the only reason this paragraph can be written at all.

Two lessons, both cheap and both mine:

- **A failed write must not destroy the input.** Documents are now written to a temporary file
  and `os.replace`d, which is atomic -- either the old content or the new, never neither.
- **`git add -A` after a script that raised is how a truncation becomes a commit.** The diff
  said `2703 deletions` and nothing read it.

**Enrichment result:** 365 papers, **12,220 chunks**, 3 failures -- all PDFs over the 30 MB
cap (35-42 MB, figure-heavy). The cap did its job and the pass continued.

---

**What full text does to retrieval**, measured on the same 1669 papers with and without it:

| | |
|---|---|
| top-20 overlap with the abstracts-only corpus | **48%** |
| enriched share of the top-20 | **64%** |
| enriched share of the corpus | 22% |

**Full text changes half the retrieved set**, and gives enriched papers nearly three times
their share of the results. The mechanism is not subtle: a paper with ~33 chunks has 33
chances to match where an abstract has 1, and `covering_papers` needs only one good chunk.

**That is the hypothesis working, not a bug** -- a paper whose *methods* section discusses the
query is relevant even when its abstract never says so, and finding those is the entire point
of the tier. But it must be stated plainly: **the arm compares "corpus with full text" against
"corpus without", not full text in causal isolation.** The enriched papers are exactly those
that appeared in earlier top-20s, so enrichment is not random. That is ecologically valid --
production would enrich the same papers for the same reason -- and it is not a clean causal
claim, and the difference matters when the number is quoted.

**A measurement I reported and had to retract.** At 125 of 365 papers enriched I reported
30-45% over-representation on two questions and 0-5% on others, calling it possible bias. It
was an artifact of **enrichment order**: the pass works through papers in recorded-context
order, so early questions were enriched and later ones were not. The number measured progress,
not bias. Worth recording because a partial-run measurement looks exactly like a finished one.

### D-110 - Retrieval and synthesis must read the same text

**The first full-text arm did not test full text, and the number proves it.** `load_sources`
returns `Source` objects whose `summary` is the abstract, so full text changed only *which
papers retrieval found* -- the review still read abstracts. The recordings said so plainly:
context entries averaged **1390 characters against the abstract arm's 1326**, the same text.
It was noticed only because `papers_retrieved` fell unexpectedly (51.8 to 39.8) and that
needed explaining.

**Scored anyway, because the accident turned out to be informative.** Renamed
`ftretrieval-local` -- full-text-*informed retrieval*, abstracts in the prompt:

| metric | abstracts | ft-retrieval | delta | SE |
|---|---|---|---|---|
| **faithfulness** | **0.99** | **0.96** | **-0.03** | **-3.1** |
| numeric_density | 0.51 | 0.25 | -0.26 | -1.2 |
| specificity | 0.78 | 0.78 | +0.00 | +0.1 |
| papers cited | 7.90 | 7.85 | -0.05 | -0.1 |
| papers retrieved | 51.80 | 39.80 | -12.00 | -6.2 |

**Faithfulness clears 2 SE in the harmful direction** -- the first metric in this whole
evaluation to do so. The mechanism follows directly from the gap: retrieval selected papers
whose *body* matched the query, then handed the model only their *abstracts*, which do not
support the topic the paper was selected for. The model cited them anyway.

**The finding is general and worth more than the arm: retrieval and synthesis must see the
same text.** A retriever that selects on evidence the writer never reads produces sources that
look relevant to the machine and do not support the claim for the reader. That is this
project's recurring failure shape -- confident output, less support than it appears -- arriving
in the one place nobody had looked for it.

`papers_retrieved` falling is the same mechanism seen from the other side: `covering_papers`
deduplicates by paper, so one enriched paper occupying several chunk slots in the top-k
crowds out distinct papers. Full text buys depth per paper at the cost of breadth.

---

**The fix.** `Source` gains an `excerpt` field rather than overwriting `summary`. The abstract
is what arXiv published and stays what it says; `excerpt` is what *this query* found inside the
paper. A `Source` whose `summary` silently became something else would be a record that lies
about its own provenance, and empty is the honest default for every paper the corpus has only
seen the abstract of.

`matching_excerpts` picks each paper's best passages by the same BM25 ranking as everything
else, so a paper contributes the part of itself that answers *this* query rather than its
opening paragraphs. Passages carry their section label: "the Results section says 2.1x" is a
stronger claim for a reader, and for O-12's claim checker, than an unattributed sentence.

`format_papers` uses `excerpt or summary`, so every paper without full text stays byte
identical and the abstracts-only arm remains comparable (D-088).

**The cost is stated, not hidden.** `EXCERPT_MAX_CHARS = 4000` is two chunks' worth. Measured
on a real subtopic: **15,157 characters of abstracts becomes 42,891 with excerpts, 2.8x**.
D-092 cut prompt tokens by 76% through pruning; excerpts spend some of that back, and the arm
records it rather than assuming it away. `0` disables them, keeping the baseline switchable.

### D-111 - Full text measured: 2.8x the prompt, no measurable gain (settles O-13)

**Both arms recorded and scored**, 20 paired questions, with retrieval and synthesis finally
reading the same text (D-110). Against the abstracts baseline:

| metric | abstracts | full text | delta | SE |
|---|---|---|---|---|
| prompt characters (one subtopic) | 15,157 | **42,891** | **2.8x** | - |
| specificity | 0.78 | 0.79 | +0.01 | +0.9 |
| numeric_density | 0.51 | 0.72 | +0.21 | +1.0 |
| faithfulness | 0.99 | 0.97 | -0.02 | **-1.9** |
| papers cited | 7.90 | 8.30 | +0.40 | +0.6 |
| papers retrieved | 51.80 | 37.85 | -13.95 | **-8.9** |

**Nothing clears 2 SE in full text's favour.** Faithfulness is nominally *down*. Breadth falls
27%, because one enriched paper occupies several top-k slots and crowds out distinct papers --
depth bought at the cost of coverage.

**This is a null from an instrument capable of detecting the effect.** D-093 built specificity
precisely because faithfulness and relevancy sit at their ceiling, and validated it at 0.95
concrete against 0.22 vague. Real reviews score ~0.78, so there was room to move. It did not
move. `numeric_density` -- the metric D-093 predicted would respond first, since abstracts
rarely state measurements -- rose 41% and stayed inside the noise, with reviews containing no
numbers at all going 10/20 to 8/20.

**Decision: `EXCERPT_MAX_CHARS = 0`.** Full text is built, measured, and off. Setting it to
4000 reproduces this arm exactly.

---

**The one effect that does clear 2 SE, and what it is against.** Excerpts beat
*full-text-informed retrieval with abstracts in the prompt* on numeric_density, 0.25 to 0.72
(**+2.8 SE**), and recover faithfulness from 0.96 to 0.97. But that comparison is against a
**broken configuration** -- one this project created by accident and has now made unreachable.
It validates D-110's diagnosis; it is not an argument for full text.

**So the coupling is enforced in code, not documented as a caution.** With
`EXCERPT_MAX_CHARS = 0`, `_local_answer` restricts retrieval to the abstract tier. A corpus
that has read papers in full therefore cannot select on that text while the model reads only
abstracts -- the pairing that measured *worse than having no full text at all*. Two tests pin
both halves, because a fix that merely disables full text everywhere is not the same as making
retrieval and synthesis agree.

---

**What O-13 cost and what it returned.** Roughly 40 minutes of enrichment, ~1 GB of PDFs, four
recording sweeps and four scoring sweeps. It returns:

- **A negative result on its headline hypothesis**, measured with an instrument built for the
  purpose and shown to have headroom.
- **D-110's general finding** -- retrieval and synthesis must read the same text -- which is
  worth more than the hypothesis was, is not specific to this project, and was only found
  because an accident produced a configuration nobody would have chosen to test.
- **A local corpus that pays for itself on its own terms** (D-107): 65 arXiv requests to 1,
  wall clock down 65%, quality flat. That part stays on.

**The honest framing for the thesis:** three of this project's four headline ideas --
recursive decomposition (D-094, D-095), deeper search (D-096), and full text (D-111) -- were
measured and did not improve review quality. The fourth, the local corpus, did pay, in cost
rather than quality. A system whose own evaluation overturns three of its four premises is a
more defensible artifact than one that never asked.

### D-112 - Ungrounded citations are too rare to have a depth rate (closes D-079's open item)

D-079 recorded one observation in each direction and asked for a rate. **140 runs later the
rate exists, and it is 0.18 per 100 citations** - 2 ungrounded citations in 1103, across
every arm ever recorded. The measurement cost nothing: `citation_violations` is a
deterministic field already in every committed recording (D-046, D-086).

| rounds actually run | runs | citations | ungrounded | per 100 |
|---|---|---|---|---|
| 1 | 99 | 776 | 1 | 0.1 |
| 2 | 11 | 98 | 0 | 0.0 |
| 3 | 30 | 229 | 1 | 0.4 |

**The depth question is unanswerable, and that is the answer.** Two events cannot support a
rate comparison: the apparent 4x between one round and three is one citation against one
citation. Reporting "0.4 vs 0.1 per 100" as a depth effect would be exactly the thin-data
inference D-079 itself warned about.

**What the number does settle** is more useful than what it does not:

- **Ungrounded citations are rare enough not to be the risk worth designing against.** Two in
  1103 is 0.18%. The project's earlier framing treated hallucinated IDs as a central hazard;
  measured, they are a footnote.
- **The two that did occur were both caught**, which is the property that matters. D-046
  verifies the *ID*, not the claim, so the guarantee is narrow and exact: no review cites a
  paper the run did not retrieve. That guarantee held 1103 times.
- **The remaining risk is claim support, not ID validity** - "Smith showed X [arXiv:1234]"
  where the ID is real, the paper was retrieved, and the paper never says X. That is O-12's
  territory, and this measurement is the argument for it being the *next* thing rather than
  more citation-ID work.

**Not re-run per arm as a quality gate.** D-078 settled that asserting `citation_violations
== []` asserts the *model* behaved; this is recorded, not enforced.

### D-113 - The claim checker (settles O-12)

`check_citations` verifies the **ID**. Measured across 140 runs, that guarantee held 1103
times against 2 violations -- **0.18 per 100 citations** (D-112). ID validity is, empirically,
not the risk. What remains is:

> *"Smith et al. showed a 2.1x speedup [arXiv:1234.5678]"* -- the ID is real, the paper was
> retrieved, the model was shown it, and the paper never says that.

**One node, one call, after synthesize** (O-12's option A). Cited sentences are extracted,
numbered, and judged in a single model call against the papers they cite.

**Verified against a real model**, with one claim the abstract supports and one invented:

> `The system achieves a 94.3% ROUGE-L score on PubMed abstracts [arXiv:2411.18583].`
> -- flagged: *"No mention of ROUGE-L score or 94.3% in the evidence."*

The supported claim passed. That is the entire feature working on its first live call.

---

**Five decisions, each of which had a wrong option that looked reasonable:**

**1. It reads exactly what the writer read.** Evidence is `excerpt or summary` for the papers
in `synthesized_from` -- the same text `format_papers` put in the synthesis prompt. D-110
measured what a mismatch costs: retrieval selecting on full text while the model read
abstracts dropped faithfulness **3.1 standard errors**. A checker judging different text would
manufacture that defect in reverse, flagging supported claims because it read something else.

**2. Failure is recorded, never fatal.** An unparseable judgement leaves
`claims_checked = False` and a `claim_check_error`, and the run completes. This is the
**inverse** of D-070: an unparseable *plan* means no research happened, so crashing is
correct; an unparseable *judgement* means a finished, ID-verified review went unverified, and
discarding it to report that would be a far worse trade.

**3. "Not checked" and "checked, found none" are distinguishable.** Both would otherwise be an
empty list -- the exact ambiguity D-084 removed for citations, reappearing one node later.

**4. Citations to papers never shown are dropped, not judged.** `check_citations` already
reports those. Judging them here would name one defect twice on the same sentence, and the
second name would blame the writer for something the retriever did.

**5. The evidence block carries D-097's nonce fence.** A checker is a *more* attractive
injection target than a writer: text that talks its way past the thing verifying it defeats
the verification, not merely the prose.

---

**It runs before `check_citations`, not after.** D-084 made `citations_checked` the terminal
completion marker; a node after it would mean a run reporting itself finished while work
remained. Ordering is free here, so the marker keeps its meaning.

**A node on the terminal path costs a super-step.** The measured minimum `recursion_limit`
moved from 13 to 14, so `RECURSION_LIMIT = 15` now carries **one** step of headroom rather
than two. `test_recursion.py` is the only thing that noticed -- the constant itself did not
change, so nothing else would have.

**Reported first in the coverage panel**, ahead of the other entries: they say what the review
did not cover; this says part of what it *did* say may not be backed by the paper it credits.
The wording tells the reader it is a model judging a model, and to treat it as a prompt to
check rather than a verdict.

**This is a product feature, not a thesis metric.** `unsupported_claims` is produced by the
same system being evaluated, so quoting it as a quality number is self-assessment. O-11's
independent judge is what the thesis reports; this is what the reader sees. It is therefore
**not** added to `compare.py` or `results.json`.

**Cost:** one extra model call per run, and a fourth streaming node. `api/routes/runs.py`
filters the `messages` stream with an **allowlist** (only `synthesize` passes), which is why
adding it was safe -- a denylist naming the known non-review nodes would have leaked its JSON
into the user's review the day it landed.

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

#### ~~O-8 — Prompt-injection defenses~~ — **settled → D-085 (XSS) and D-097 (nonce fence)**

**Settled 2026-09-24.** Option A below was taken: `new_fence()` generates a per-run
`secrets.token_hex(8)` token, `format_papers(sources, fence)` wraps the block in it, and
`system_prompt(fence)` names that exact token. The options table is kept because the reasoning
against B (a blacklist that catches one string) and C (a weaker boundary with no evidence
behind it) is still the reasoning for A. **Instruction-following injection in plain prose
remains mitigated, not solved** -- see D-097.

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

- ~~**O-15 — the stop reason is invisible on a clean run.**~~ **Settled 2026-09-24 → D-099.**
  Option A was taken. Original note kept for the reasoning. Found by `/demo-check` on
  2026-09-24. `runs.py:145` and `pages.py:64` send `coverage_html` only when
  `coverage.is_complete` is false — D-086's choice, so a clean run is not cluttered with a
  list of nothing. That was right when the panel only ever reported *losses*.

  D-096 changed what it has to say. Runs now stop after **one** round, and a viewer who
  watches three searches and then synthesis has no way to learn why it did not go deeper.
  "Enough papers were found to fill the synthesis context" reads as confidence; its absence
  invites "did it give up?". The sentence exists, is tested, and is never shown.

  | Option | Pros | Cons |
  |---|---|---|
  | **A. Always show `stopped_because`, keep the loss list conditional** | One line of prose costs nothing and answers the obvious question. Separates "why it stopped" from "what it lost", which D-086 conflated | The panel appears on every run, so it must read well when there is nothing wrong |
  | **B. Show the panel whenever the stop reason is *not* the depth ceiling** | Only surfaces the reassuring cases | Backwards: the ceiling case is the one a reader most needs to see |
  | **C. Leave it** | No work | The feature most in need of explaining to a judge is the one with no explanation in the UI |

  **Recommendation: A**, and it is small — `is_complete` already distinguishes the two, so it
  is a template change plus deciding whether the line reads as a caption or a sentence. Worth
  doing before any demo where someone asks why it only searched once.

- **O-16 — a corpus that answers a topic forever never refreshes it.** Opened by D-107.
  `LOCAL_FIRST` is on, coverage was 100% on the measured questions, and nothing in the design
  ever re-fetches a subtopic the corpus already covers. arXiv grows ~100 GB a month, so a
  topic researched in March can stay answered from March's papers indefinitely — silently,
  which is this project's recurring failure shape (D-062, D-069, O-5).

  D-105 makes it *visible* (the panel names locally-answered subtopics and says papers
  published since are not represented), but visible is not the same as handled.

  | Option | Pros | Cons |
  |---|---|---|
  | **A. Report only (today)** | Zero work; the reader can judge and re-run | A reader who does not read the panel gets a stale review that looks current |
  | **B. Max age per paper** — coverage ignores chunks indexed more than N days ago | Bounded staleness, cheap to implement (`indexed_at` is already stored) | N is unmeasured, and guessing it is exactly what D-077 warns against |
  | **C. Refresh on a schedule** — re-search covered subtopics every N runs | Keeps the corpus alive without per-run cost | Same unmeasured N, plus surprise latency on an arbitrary run |
  | **D. Always search, use the corpus only to enrich** | No staleness at all | Gives up the entire measured benefit (D-105 rejected this on evidence) |

  **Recommendation: A now, B when there is a number.** The experiment that produces one needs
  *time to pass*, not compute: re-record a covered question against a corpus seeded weeks or
  months earlier and compare specificity and citation recency. That is genuinely future work,
  and saying so is better than shipping a guessed N.

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
| ~~O-12~~ | ~~In-band claim checker~~ | **Settled -> D-113.** One node, one call, reading exactly what the writer read; failure recorded, never fatal | ~~After O-13~~ |
| ~~O-15~~ | ~~Stop reason invisible on a clean run~~ | **Settled → D-099** | ~~Before a demo~~ |
| **O-16** | **A covered topic never refreshes** | Report only (D-105) until a measured max age exists; the experiment needs time to pass, not compute | When the corpus is months old |
| ~~O-13~~ | ~~Local-first corpus, BM25 first~~ | **Settled.** The corpus pays (D-107: 65 arXiv requests to 1, -65% wall clock, quality flat). Full text does not (D-111: 2.8x the prompt, nothing past 2 SE) | ~~Milestone 6~~ |
| ~~O-14~~ | ~~`MAX_DEPTH` default + a yield-based exit~~ | **Settled → D-096.** Adaptive exits instead of a lower ceiling: −66% searches, quality flat, 19/20 runs stop after one round | ~~Now~~ |
