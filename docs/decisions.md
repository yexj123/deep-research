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

## Open (proposed, not decided)

**Settled 2026-09-20:** O-1 → D-064, O-2 → D-065, O-3 → D-066. The remaining numbering is unchanged
so earlier references stay valid.

Each item lists the real options with their tradeoffs and a recommendation. A recommendation
here is **not** a decision — it moves into the log with a new ID only once confirmed.
Grouped by the milestone that forces the choice.

---

### Milestone 3 — all settled 2026-09-20

O-1 (arXiv rate limiter) → **D-064** · O-2 (4xx vs 5xx) → **D-065** · O-3 (models per role) → **D-066**.
Nothing open blocks the `Send` fan-out.

### Needed for milestone 4 (`gap_check` + recursion)

#### O-4 — The `recursion_limit` backstop value

**Problem.** `max_depth = 2` is the real exit (D-009, D-025); `recursion_limit` is only the
backstop for when the depth exit is broken. Its value was never chosen.

**Measured 2026-09-20:** the milestone-2 graph (4 sequential nodes) needs `recursion_limit=5`;
4 raises `GraphRecursionError`. So the limit must be **super-steps + 1**, one more than intuition
suggests. A `Send` fan-out is a single super-step regardless of worker count.

Projected milestone-4 arithmetic — `intake` (1) + 3 rounds x [`decompose` + workers + `gap_check`]
(9) + `synthesize` + `check_citations` (2) = **12 super-steps, so >= 13**.

| Option | Pros | Cons |
|---|---|---|
| **A. 15 — tight, just above the real need** | A broken depth exit trips it almost immediately, which is the entire point of a backstop | Any graph change needs the number revisited, or a legitimate run dies |
| **B. 25 — LangGraph's default** | Nothing to justify; comfortable headroom | Roughly 2x the real need, so a runaway loop burns about twice as many paid calls before stopping |
| **C. Derive it: `max_depth * 3 + 4`** | Self-adjusting when `max_depth` is tuned in the thesis evaluation | Hides a magic formula that silently goes wrong if the per-round node count changes |

**Recommendation: A (15), with a test asserting a normal full-depth run completes under it.** That
test is what makes the number defensible and catches the graph outgrowing it. Re-measure when the
milestone-4 graph actually exists — the arithmetic above is projection, not measurement.

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
| O-4 | `recursion_limit` | 15, plus a test that a full-depth run fits | Milestone 4 |
| O-5 | Failures visible | State first, `custom` events later; add a limitations section | Milestone 4 |
| O-6 | Frontend | htmx — but verify its SSE + streaming-markdown story first | Milestone 5 |
| O-7 | Public entry function | Build it at milestone 5, once the routes exist | Milestone 5 |
| O-8 | Prompt injection | Nonce delimiter; output sanitizing at milestone 5 | Any time |
| O-9 | Accent spellings | Measure recall first, then decide | Any time |
| O-10 | Non-English stopwords | Accept and document | Any time |
