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
  LangChain's docs also say DeepSeek's reasoning model doesn't support
  tool calling or structured output.

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

## Open (proposed, not decided)

- **A public entry function.** Revisit at the web-layer milestone: a wrapper that
  requires `context`, in addition to the check in `intake` (D-033).
- **Models per role.** The planner, gap checker and synthesizer each get a
  configured model. The app refuses to start if a role that needs structured
  output gets a model without it. The factory itself is decided (D-029).
- **Treat 4xx and 5xx differently (D-027).** Catching every `HTTPStatusError` also
  catches 400/404 errors caused by bugs in your own request code, which would then
  count toward the retry cap instead of crashing. Proposed: treat 429 and 5xx as
  external failures and re-raise other 4xx.
- **Making failures visible.** Proposed: a `custom` stream event, plus a coverage
  limitations section in the review listing failed and zero-result subtopics.
- **Prompt-injection defenses.** Citation IDs restricted to the retrieved set are decided
  (D-046). Still proposed: no tools with side effects; untrusted text in delimited data
  sections; sanitize LLM output before it's rendered.
- **arXiv rate limit** (≤1 request every 3 s) under parallel `Send`: where the
  limiter lives.
- **`recursion_limit` value.** `max_depth` is set (D-025); the backstop value
  isn't.
- **Frontend: SvelteKit or htmx** (at the web-layer milestone; if SvelteKit, add
  `node_modules/`, `.svelte-kit/` and the build output folder to `.gitignore`).
