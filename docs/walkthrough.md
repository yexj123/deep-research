# Walkthrough

Two workflows, end to end, with real values captured from the code on 2026-09-20
(milestone 2, commit `60611ac`):

1. **[The agent run](#1-the-agent-run)** — what happens when someone asks a question.
2. **[The development cycle](#2-the-development-cycle)** — how a change gets made, using
   D-062 from this repo's history as the worked example.

Design reasoning lives in [`decisions.md`](decisions.md); what each file does is in
[`code-map.md`](code-map.md). This file is the narrative that connects them.

---

## 1. The agent run

Question in, cited review out. At milestone 2 the graph is linear —
`intake → search → synthesize → check_citations` — and every value below is real,
captured by running the graph against the saved arXiv fixture with the fake model.

### Step 0 — The caller builds three dependencies

Nothing in the graph constructs its own model client, HTTP client or checkpointer
(D-032, D-049). The caller owns all three, which is why the same graph code runs in
tests, in the integration test, and later in the web layer:

```python
graph = build_graph(
    model_factory=get_chat_model,                              # tests: RecordingFactory()
    http_client=httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS),
    checkpointer=InMemorySaver(serde=build_serializer()),       # web layer: AsyncSqliteSaver
)

graph.astream(
    {"question": "What is attention in transformer models?"},
    {"configurable": {"thread_id": "..."}},
    context=RunContext(provider="openai"),   # per-run, never a module global (D-015)
    stream_mode=["updates", "messages"],
    version="v2",
)
```

The provider arrives through runtime `context`, not state and not a global, so two
concurrent runs can't overwrite each other's choice.

### Step 1 — `intake`

Reads `state.question` and `runtime.context`. Writes back the stripped question.

Its real job is to fail early: it raises `ValueError` if the context is missing or the
provider isn't `openai`/`deepseek` (D-033). Without that check, calling the graph with no
`context=` passes `None` through and dies later with an `AttributeError` far from the
cause. A `Literal` type hint isn't enforced at runtime, so `RunContext(provider="gemini")`
constructs happily — this is the only place that's caught.

### Step 2 — `search`

Builds a deterministic arXiv query from the question. No LLM call:

```
"What is attention in transformer models?"
  → build_search_query()
  → "all:attention AND all:transformer AND all:models"
```

Stopwords (`what`, `is`, `in`) and punctuation are dropped, and each remaining term is
ANDed. Measured on 2026-09-16: the raw question matched **327,597** papers, this query
**14,338** far more relevant ones (D-051).

The request that actually goes out:

```
https://export.arxiv.org/api/query
    ?search_query=all%3Aattention+AND+all%3Atransformer+AND+all%3Amodels
    &start=0
    &max_results=10
```

The response is Atom XML, parsed by `parse_feed` with `defusedxml` and
`forbid_dtd=True` — defusedxml accepts a bare `<!DOCTYPE>` by default, and this
environment's expat is below 2.7.2 (D-047). Each `<entry>` becomes a `Source`:

```python
Source(
    arxiv_id  = '2411.18583',          # canonical, version stripped (D-044)
    version   = 1,
    title     = 'Automated Literature Review Using NLP Techniques and ...',
    authors   = ('Nurshat Fateh Ali', 'Md. Mahdi Mohtasim', ...),   # 4 total
    summary   = 'This research presents and compares multiple approaches ...',
    published = datetime(2024, 11, 27, 18, 27, 7, tzinfo=UTC),
    url       = 'https://arxiv.org/abs/2411.18583v1',
)
```

An entry that fails validation is **skipped and counted**, not fatal — one malformed
entry shouldn't discard a good page of results. Only an unparseable feed or arXiv's error
feed fails the search (D-045).

Writes `sources` and `skipped_entries`. At this milestone `search` catches nothing: with a
single search there is nothing to continue with, so recording a failure would only hide it
(D-053).

### Step 3 — `synthesize`

Two messages go to the model. The system prompt defines the citation contract; the human
message carries the question plus the papers as a delimited data block:

```
<papers>
[arXiv:2411.18583] Automated Literature Review Using NLP Techniques and LLM-Based ...
This research presents and compares multiple approaches to automate the generation of ...

[arXiv:2502.00306] ...
</papers>
```

Two things are deliberate here:

- **The marker before each title is the only citation format allowed.** The prompt requires
  `[arXiv:<id>]` exactly — no version suffix, no URL, one paper per marker — because
  `check_citations` has to be able to parse it in step 4.
- **The prompt states that everything inside `<papers>` is data, not instructions**
  (D-055). Abstracts are untrusted third-party text. This is the first concrete
  prompt-injection defense; the rest is still Open.

**If `sources` is empty, no model is built and no call is made** — `synthesize` returns the
fixed `NO_SOURCES_REVIEW` (D-060). Zero results is a success (D-021), and there's no model
output that could invent a citation or answer from general knowledge. The cost: that run
streams nothing in `messages` mode.

### Step 4 — `check_citations`

Extracts every citation from the review and validates each ID against the papers actually
retrieved, using Pydantic **validation context**:

```python
Citation.model_validate({"arxiv_id": cited}, context={"known_ids": known_ids})
```

That lifts the check from "is this a well-formed arXiv ID" (D-041, which a plausible
hallucination passes) to "is this one of the papers we actually read."

It finds **citation attempts**, not just citations (D-062). `CITATION_BRACKET` matches any
`[arXiv:...]` bracket; `CITATION_MARKER` then tries to parse each one:

| The model writes | Parsed? | Result |
|---|---|---|
| `[arXiv:2411.18583]` | yes, and retrieved | clean |
| `[arXiv:9999.99999]` | yes, not retrieved | violation: `9999.99999` |
| `[arXiv:2411.18583v1]` | yes, but not canonical | violation: `2411.18583v1` |
| `[arXiv:A, B]` | no | violation: `[arXiv:A, B]` |
| `[1]`, `[see Table 2]` | not a citation | ignored |

The fourth row is why the permissive pattern exists. Matching only what the strict pattern
understands would make an unreadable citation indistinguishable from *no* citation, so a
review full of malformed markers would report zero violations — a false negative in the
feature this whole project is about.

### The stream, as the browser will see it

Real chunk order from one run:

```
updates   intake           -> wrote ['question']
updates   search           -> wrote ['sources', 'skipped_entries']
messages  synthesize       -> 'Retrieval'
messages  synthesize       -> ' '
messages  synthesize       -> 'grounds'
   ... one chunk per token ...
messages  synthesize       -> '[arXiv:2411.18583].'
updates   synthesize       -> wrote ['review']
updates   check_citations  -> wrote ['citation_violations']
```

**The `messages` chunks arrive before `synthesize`'s `updates` chunk.** Tokens stream
*while* the node runs; the state write lands when it finishes. That ordering is what
milestone 5's SSE design depends on — `updates` drives node-progress UI, `messages` drives
the text appearing live.

(The review text above comes from the fake model used in tests. Against real OpenAI the
wording differs every run, which is why the integration test asserts *properties* — that
the joined stream equals the saved review, that a citation exists, that violations are
empty — never fixed text.)

Final state, checkpointed after every step:

```python
question            = 'What is attention in transformer models?'
review              = "Retrieval grounds a model's answer in papers it has just read [arXiv:2411.18583]."
sources             = [3 Source objects]
skipped_entries     = 0
citation_violations = []
```

Each field has exactly **one** writer, which is why no reducers exist yet. That changes at
milestone 3, when parallel `Send` workers all write `sources` in the same step — and a
missing reducer there raises `InvalidUpdateError` rather than silently overwriting.

---

## 2. The development cycle

The working agreement: **you write `src/`, Claude writes `tests/` and `docs/`.** The split
isn't about capability — it's that you have to defend every line of the implementation to a
judge or an advisor, so writing it is the point.

Here is one real cycle, D-062, exactly as it happened on 2026-09-20.

### 1. A question surfaces during review

Reading `check_citations.py` during a walkthrough raised a doubt: `CITATION_MARKER` only
matches `[arXiv:<id>]`. What happens to `[arXiv:A, B]`?

### 2. Verify instead of assuming

Rather than reasoning about the regex, the node was run against five realistic model
outputs. `[arXiv:A, B]` and `[arXiv: A]` extracted **nothing** and produced
`citation_violations == []` — byte-identical to a perfectly grounded review.

This step is what turned a vague doubt into a finding. The bug was invisible from reading
the code, because the code did exactly what it said; it was the *meaning* of the empty list
that was wrong.

### 3. Decide, and log the decision

The finding was a genuine design fork — fix it now, defer it, or handle it differently — so
it went back to you. Once decided, it was logged as **D-062** in `decisions.md`: what was
chosen, why, **what was rejected** (parsing grouped IDs, which would let the prompt and the
checker drift apart), and the known limit that remains (a citation with no brackets at all).

A recommendation you haven't confirmed goes under **Open**, never into the log.

### 4. Claude writes the failing tests

Seven tests, each with a docstring naming the behavior it pins and the `D-` number it
checks. They went red for the right reason — `DID NOT RAISE`, and a violations list that
came back empty:

```
7 failed, 59 passed, 1 deselected
```

A test that fails against your code is a **review finding**, reported with the file and
line. It is never "fixed" by weakening the test.

### 5. You write the implementation

Three edits in `src/`, from snippets and explanation — not from Claude editing the files.
Then:

```
66 passed, 1 deselected
```

### 6. Review the implementation, not just the test result

Green tests prove the tests pass, not that the code is right. The review of your edit
caught a stale module docstring and a missing comment on `CITATION_BRACKET` — the least
self-evident line in the milestone, and the one a stranger on GitHub would ask about.

### 7. Docs move in the same change

`decisions.md` (the entry), `code-map.md` (the new constant, the changed meaning of
`citation_violations`), `progress.md` (status), and the `langgraph-conventions` skill —
because CLAUDE.md requires a settled Open item to update any skill that states the rule.
Docs written later are docs written wrong.

### 8. Prove it end to end

```
uv run pytest -m integration     # 1 passed, 11.17s
```

One paid OpenAI call and one real arXiv request. Unit tests use a fake model that always
streams, so only a real provider can prove streaming actually works.

### 9. Commit, with the reasoning in the message

The commit message carries the *why*, including the verified evidence
(`Source(**valid, abstract="x")` constructed fine and dropped the field). `git log` is the
one piece of documentation that can't drift from the code.

### The shape of it

```
notice → verify → decide → log → failing test → implement → review → docs → prove → commit
         ↑                                                     |
         └──────── a finding sends you back here ──────────────┘
```

Two rules make it work:

- **Verify before asserting.** Every claim in `decisions.md` that says "confirmed" was run,
  not remembered. Library APIs move, and D-014's correction is an entry that exists purely
  because an earlier assumption turned out to be wrong.
- **The decision log is written when the decision is made.** Reconstructed reasoning is
  reasoning you can't defend.

---

## Keeping this file current

Update it in the same change as the code, like [`code-map.md`](code-map.md) and
[`progress.md`](progress.md).

This file is the one most likely to rot quietly, because §1 contains **values captured from
a real run**, not prose. A wrong query string or a stale stream order reads as authoritative
and is worse than no example at all. Re-capture whenever any of these change:

| Change | What goes stale in §1 |
|---|---|
| A node added, removed or renamed | the step headings, the stream-order block |
| A `ResearchState` field added or renamed | the final-state block, the one-writer note |
| `build_search_query` or the arXiv params | the query string and the request URL |
| `Source`'s fields | the `Source(...)` block |
| The `synthesize` prompt or `format_papers` | the `<papers>` block |
| Stream modes, or where tokens come from | the stream-order block and the note under it |
| A new milestone's graph shape | most of §1 — rewrite rather than patch |

**How the values were captured:** the graph was run against the saved
`tests/agent/fixtures/arxiv/search_ok.xml` fixture with `RecordingFactory` as the model
factory — no network, no API cost — printing each node's update, every `messages` chunk,
the final state, and the URL the stub received. Anything in §1 can be regenerated that way;
nothing in it should be written from memory.

§2 changes far less often. Revisit it when the working agreement moves — as it did on
2026-09-19 (tests) and 2026-09-20 (docs).
