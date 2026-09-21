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

## Open (proposed, not decided)

**Settled 2026-09-20:** O-1 → D-064, O-2 → D-065, O-3 → D-066.
**Settled 2026-09-21:** O-4 → D-077. The remaining numbering is unchanged so earlier references
stay valid.

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

### Needed for milestone 4 (`gap_check` + recursion)

O-4 settled as **D-077**.

#### O-5 — Making failures visible

**Problem.** Failed and zero-result subtopics are currently invisible in the output. A review
silently missing a third of its subtopics looks identical to a complete one — the same failure
shape as D-062.

| Option | Pros | Cons |
|---|---|---|
| **A. State fields only** (`failed_subtopics` exists already), rendered by whatever reads state | Checkpointed, deterministic, testable without a browser. Single source of truth | Nothing surfaces until the run ends |
| **B. A `custom` stream event per failure** | Live feedback during a long run, which is the main UX complaint a demo will hit | Ephemeral — a reconnecting browser misses it. Not testable without streaming |
| **C. Both: state is the record, `custom` is the notification** | Live *and* durable | Two code paths that can disagree about what counts as a failure |

**Recommendation: A first, then C at milestone 5,** with `custom` events derived from the same
state update so they cannot drift. Add a "Coverage and limitations" section to the review listing
failed and zero-result subtopics — for a literature review that section is a *finding*, not an
apology, and it is the kind of honesty an advisor rewards.

---

### Needed for milestone 5 (web layer)

#### O-6 — Frontend: SvelteKit or htmx

**Problem.** The app is fundamentally "stream text into a page, show node progress, list saved
reviews". Both stacks can do it.

| Option | Pros | Cons |
|---|---|---|
| **htmx** (no `frontend/`; templates live in `api/`) | No second toolchain, no build step, no `node_modules`, nothing to deploy separately. SSE maps directly onto the existing design (D-006). Far less surface to defend | Streaming *markdown* is the hard part and still needs a JS library, so "no JavaScript" is not quite true. Run history and re-render logic get awkward as state grows |
| **SvelteKit** (a real `frontend/`) | A proper component model for token-by-token rendering and run history. A more polished demo | A whole second toolchain, build step and deploy story for a project whose thesis value is entirely in the agent. More code you must be able to defend |
| **Server-rendered HTML + a small vanilla JS `EventSource`** | Smallest possible dependency set; the SSE client is ~20 lines you fully understand | You hand-roll what a framework gives free; grows into a bad framework if the UI expands |

**Recommendation: htmx, and verify the SSE story before committing.** The deciding argument is
that every hour on the frontend is an hour not spent on recursion and citation grounding, which is
what the thesis is actually about. **Verify first** (do not take this from memory): that htmx's SSE
extension can append streamed tokens into a live-rendering markdown block, since that is the one
requirement that could rule it out.

#### O-7 — A public entry function

**Problem.** `intake` validates the run context (D-033), but every caller still assembles
`RunContext` and the config dict by hand.

| Option | Pros | Cons |
|---|---|---|
| **A. A `run_research(question, provider, thread_id)` wrapper** owning context, config and stream modes | One place to get it right; the API route stays thin; the signature documents what a run needs | Another layer to keep in sync with the graph; tests that want raw `astream` bypass it anyway |
| **B. Status quo — rely on `intake`** | Nothing to build; the graph stays the only interface | The web layer will rebuild the same context in at least two routes (start and resume), so the duplication is guaranteed rather than hypothetical |

**Recommendation: A at milestone 5, not before.** Its real shape only becomes clear once the routes
exist, and writing it now means guessing at the signature.

---

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
| O-5 | Failures visible | State first, `custom` events later; add a limitations section | Milestone 4 |
| O-6 | Frontend | htmx — but verify its SSE + streaming-markdown story first | Milestone 5 |
| O-7 | Public entry function | Build it at milestone 5, once the routes exist | Milestone 5 |
| O-8 | Prompt injection | Nonce delimiter; output sanitizing at milestone 5 | Any time |
| O-9 | Accent spellings | Measure recall first, then decide | Any time |
| O-10 | Non-English stopwords | Accept and document | Any time |
