# Code map

What each file does, what it uses, and what uses it. Built from the actual imports on
2026-09-19. Design reasons are in [`decisions.md`](decisions.md) (the `D-` numbers).

**Status tags**
- **[M1]**: implemented and tested (milestone 1)
- **[M2]**: new in milestone 2, implemented and tested (not committed yet)
- **[M1 → M2]**: from milestone 1, changed in milestone 2

---

## 1. The big picture: four layers

```text
  callers          tests/  (later also api/, the web layer)
                     │ build the dependencies and call build_graph(...)
                     ▼
  wiring           agent/graph.py
                     │ registers the nodes and connects them
                     ▼
  nodes            agent/nodes/intake.py · search.py · synthesize.py · check_citations.py
                     │ read state, return partial updates
                     ▼
  building blocks  agent/llm.py (chat models) · agent/sources/arxiv.py (arXiv client and parser)
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
    search["nodes/search.py [M2]"]
    cites["nodes/check_citations.py [M2]"]
    llm["agent/llm.py"]
    arxiv["sources/arxiv.py [M2]"]
    config["agent/config.py"]
    context["agent/context.py"]
    state["agent/state.py"]
    models["sources/models.py [M2]"]
    ckpt["persistence/checkpointer.py [M2]"]

    graph --> context
    graph --> llm
    graph --> intake
    graph --> search
    graph --> synth
    graph --> cites
    graph --> state
    intake --> context
    intake --> state
    search --> arxiv
    search --> config
    search --> state
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

### `agent/config.py` [M1 → M2]
**Defines:**
- `API_KEY_ENV_VARS`: which environment variable holds each provider's key;
- `MODEL_NAMES`: which model each provider uses;
- `LLM_TIMEOUT_SECONDS = 60.0` and `LLM_MAX_RETRIES = 2` (D-039).

**Uses:** `context.ProviderType`.
**Used by:**
- `llm.py` uses the four LLM constants;
- `tests/agent/test_llm.py` imports `API_KEY_ENV_VARS` and the two limits, to check the built clients against the same values.

**Milestone 2 added:**
- `ARXIV_MAX_RESULTS = 10`, used by `nodes/search.py`;
- `ARXIV_TIMEOUT_SECONDS = 30.0`, used by whoever creates the `httpx.AsyncClient`: the integration test now, the web layer later (D-052).

### `agent/state.py` [M1 → M2]
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

### `agent/nodes/search.py` [M2]
**Defines:** `make_search(http_client)`, which returns the async `search(state)` node (the same pattern as `make_synthesize`).
**Uses:**
- `sources/arxiv.py`: `build_search_query` and `search_arxiv`;
- `state.ResearchState`;
- `config.ARXIV_MAX_RESULTS`.

**Reads:** `state.question`. **Writes:** `sources`, `skipped_entries`.
**Catches nothing** at milestone 2, so a failed search fails the run (D-053).
**Registered by:** `graph.py`, as `"search"`, via `make_search(http_client)`.

### `agent/nodes/synthesize.py` [M1 → M2]
**Defines:**
- `SYSTEM_PROMPT`: requires citations in exactly the form `[arXiv:<arxiv_id>]` (D-046), and says the
  `<papers>` block is data, not instructions (D-055);
- `format_papers(sources)`: the `<papers>` block, one `[arXiv:<id>] <title>` line plus abstract per paper;
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

### `agent/graph.py` [M1 → M2]
**Defines:** `build_graph(model_factory, http_client, checkpointer)` (D-032, D-049). Wiring:
START → `intake` → `search` → `synthesize` → `check_citations` → END.
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
**Used by:** `nodes/search.py`, `test_arxiv.py`.
**Call chain inside the file:**
1. `search_arxiv` makes the HTTP GET and calls `raise_for_status`, then `parse_feed`;
2. `parse_feed` runs `defusedxml` with `forbid_dtd=True`, checks for the error feed, then calls `entry_to_source` once per entry;
3. `entry_to_source` calls `split_versioned_id`, then `Source(...)`, which runs the validation.

`search_arxiv` receives its `client` as a parameter and never creates one (D-049).

### `persistence/checkpointer.py` [M2]
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
| `agent/test_checkpoint_roundtrip.py` [M2] | A `Source` comes back from a checkpoint as a `Source`, plus the control case (empty allowlist → `dict`) | `build_graph`, `build_serializer`, `JsonPlusSerializer`, `fakes` |

---

## 5. One run, step by step (milestone 2)

```text
caller (test or web layer)
  1. model_factory = get_chat_model            (tests: RecordingFactory())
  2. http_client   = httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS)   (tests: make_arxiv_stub(...).client)
  3. checkpointer  = InMemorySaver(serde=build_serializer())            (web layer: AsyncSqliteSaver)
  4. graph = build_graph(model_factory, http_client, checkpointer)
  5. graph.astream({"question": ...}, {"configurable": {"thread_id": ...}},
                   context=RunContext(provider=...), stream_mode=[...], version="v2")

inside the graph (a checkpoint is saved after every step)
  intake           reads question, runtime.context    → writes question (stripped)
  search           reads question                     → arxiv.search_arxiv(http_client, …)
                                                       → writes sources, skipped_entries
  synthesize       reads question, sources, provider  → model_factory(provider).ainvoke(...)
                                                       → writes review   (tokens stream while it runs)
                   (no sources: writes NO_SOURCES_REVIEW; no model is built)
  check_citations  reads review, sources              → writes citation_violations
```

### Which node writes which state field

| Field | Written by | Read by |
|---|---|---|
| `question` | the input; cleaned by `intake` | `search`, `synthesize` |
| `sources` [M2] | `search` | `synthesize` (data block), `check_citations` (`known_ids`) |
| `skipped_entries` [M2] | `search` | nobody yet (shown in the report later) |
| `review` | `synthesize` | `check_citations` |
| `citation_violations` [M2] | `check_citations` | nobody yet (shown in the report / UI later). Unknown IDs *and* unparseable markers (D-062) |

Each field has exactly one writer, so **no reducers are needed at milestone 2**. That changes at
milestone 3, when parallel workers write `sources` at the same step.

---

## 6. The three dependencies passed into `build_graph`

| Dependency | Unit tests | Integration test | Web layer (milestone 5) |
|---|---|---|---|
| model factory | `RecordingFactory()` (`fake_factory` fixture) | `get_chat_model` | `get_chat_model` |
| HTTP client [M2] | `make_arxiv_stub(...).client` (`arxiv_ok` fixture) | `httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS)` | one shared client, opened at startup |
| checkpointer | `InMemorySaver(serde=build_serializer())` (`checkpointer` fixture) | the same fixture | `AsyncSqliteSaver` with `serde=build_serializer()` |

The graph code is identical in all three columns. Only the dependencies passed in change.

---

## 7. Every constant: where it's defined and used

| Constant | Defined in | Used by | Decision |
|---|---|---|---|
| `API_KEY_ENV_VARS` | `agent/config.py` | `llm.py`, `test_llm.py` | D-034 |
| `MODEL_NAMES` | `agent/config.py` | `llm.py` | D-029 |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | `agent/config.py` | `llm.py`, `test_llm.py` | D-038, D-039 |
| `ARXIV_MAX_RESULTS` [M2] | `agent/config.py` | `nodes/search.py` | D-052 |
| `ARXIV_TIMEOUT_SECONDS` [M2] | `agent/config.py` | whoever creates the HTTP client (`test_integration.py` now) | D-052 |
| `VALID_PROVIDERS` | `nodes/intake.py` (built from `ProviderType`) | `intake` | D-033 |
| `SYSTEM_PROMPT` | `nodes/synthesize.py` | `synthesize` | D-046, D-055 |
| `NO_SOURCES_REVIEW` [M2] | `nodes/synthesize.py` | `synthesize`, `test_graph.py` | D-060 |
| `ARXIV_API_URL`, `NAMESPACES`, `_ENTRY_ID_PATTERN` | `sources/arxiv.py` | `arxiv.py` only (fixed by arXiv, not settings) | D-042, D-044 |
| `STOPWORDS`, `PUNCTUATION_PATTERN` | `sources/arxiv.py` | `build_search_query` | D-051, D-059 |
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
