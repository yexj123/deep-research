# Walkthrough

Two workflows, end to end, with real values captured from the code on 2026-09-21
(milestone 4):

1. **[The agent run](#1-the-agent-run)** — what happens when someone asks a question.
2. **[The development cycle](#2-the-development-cycle)** — how a change gets made, using
   D-062 from this repo's history as the worked example.

Design reasoning lives in [`decisions.md`](decisions.md); what each file does is in
[`code-map.md`](code-map.md). This file is the narrative that connects them.

---

## 1. The agent run

Question in, cited review out. At milestone 4 the graph recurses — the planner proposes
subtopics, workers search them in parallel, and `gap_check` decides whether another round is
worth it. Every value below is real, captured by running the graph against the saved arXiv
fixture with the fake model.

```text
START → intake → decompose → (Send per subtopic) → research_worker → gap_check
                     ↑                                                   │
                     └── another round could still change the review ─────┘
                                                                         │
                                             synthesize ←────────────────┘
                                                  │
                                    check_claims → check_citations → END
```

### Step 0 — The caller builds four dependencies

Nothing in the graph constructs its own model client, HTTP client, rate limiter or
checkpointer (D-032, D-049, D-064). The caller owns all four, which is why the same graph
code runs in tests, in the integration test, and later in the web layer:

```python
graph = build_graph(
    model_factory=get_chat_model,                              # tests: RecordingFactory()
    http_client=httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS),
    limiter=ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS),      # tests: NullLimiter()
    checkpointer=InMemorySaver(serde=build_serializer()),       # web layer: AsyncSqliteSaver
)

graph.astream(
    {"question": "What is attention in transformer models?"},
    {"configurable": {"thread_id": "..."}, "recursion_limit": RECURSION_LIMIT},
    context=RunContext(provider="openai"),   # per-run, never a module global (D-015)
    stream_mode=["updates", "messages"],
    version="v2",
)
```

Two things that are easy to get wrong here. The provider arrives through runtime `context`,
not state and not a global, so concurrent runs can't overwrite each other's choice. And
`recursion_limit` is **invoke config, not graph config** — leave it out and LangGraph
silently uses its default of 25 instead of the measured 15 (D-077).

### Step 1 — `intake`

Reads `state.question` and `runtime.context`. Writes back the stripped question.

Its real job is to fail early: it raises `ValueError` if the context is missing or the
provider isn't `openai`/`deepseek` (D-033). Without that check, calling the graph with no
`context=` passes `None` through and dies later with an `AttributeError` far from the cause.
A `Literal` type hint isn't enforced at runtime, so `RunContext(provider="gemini")`
constructs happily — this is the only place that's caught.

### Step 2 — `decompose` (the planner)

Asks the model to break the question into searchable subtopics, then **decides which to
dispatch**. That split is the point: the planner *proposes*, the code *decides*.

```python
{"subtopics": ["attention mechanisms", "positional encoding"]}
```

The reply is parsed with `SubtopicPlan.model_validate_json(...)` — explicit validation in the
node that receives it, because a model's reply is external data (D-013, D-070). It catches
nothing, so an unparseable plan fails the run.

Three hard filters then run in code. The explored list also goes into the prompt, but only as
a hint — the planner is an LLM and will rephrase, repeat, or ignore it:

| Filter | Why it's enforced in code |
|---|---|
| normalized form already in `explored_subtopics` | the prompt hint is advisory (D-017, D-022) |
| `failed_subtopics` count ≥ 2, counted on the **normalized** form | the N=2 retry cap (D-020); raw counting lets a rephrasing reset the budget |
| `build_search_query` would raise | `ValueError` isn't on the worker's catch list, so dispatching one would crash the run (D-059, D-073) |

Writes `pending_subtopics` (no reducer — overwritten each round, D-017) and
`seen_before_round`, the baseline `gap_check` compares against later.

### Step 3 — `route_subtopics` (a conditional edge, not a node)

```python
Send("research_worker", {"subtopic": topic, "seen_paper_ids": state.seen_paper_ids})
```

One `Send` per subtopic, dispatched in parallel. The payload is a plain dict (a `TypedDict` at
the type level) because **`Send` payloads are checkpointed** — a custom class comes back as a
plain `dict` on *resume*, with only a logged warning (D-071).

If `pending_subtopics` is empty it returns the node name `"synthesize"` instead. An empty
`Send` list ends the run silently — no error, no downstream node, no review (D-069).

### Step 4 — `research_worker` (one per subtopic, in parallel)

Builds a query from **its subtopic**, not the question:

```
"attention mechanisms" → build_search_query() → "all:attention AND all:mechanisms"

https://export.arxiv.org/api/query
    ?search_query=all%3Aattention+AND+all%3Amechanisms&start=0&max_results=10
```

Every request runs inside `async with limiter:` — held across the whole request, not just its
start, because arXiv allows one connection at a time *and* one request per three seconds
(D-064). Parallel workers therefore queue for arXiv; the concurrency `Send` buys is in the
LLM work.

Each `<entry>` becomes a `Source`, parsed with `defusedxml` and `forbid_dtd=True` (D-047):

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

**On success** it writes `sources`, `skipped_entries` (its own delta), `explored_subtopics`
and `seen_paper_ids`. **On failure** it writes `failed_subtopics` and nothing else — a failed
subtopic must stay unexplored so it can be retried (D-018).

What counts as a failure is deliberately narrow: transport errors, bad XML, validation
failures, arXiv's error feed, and `HTTPStatusError` **only** for 429 and 5xx. Every other 4xx
is re-raised, because a 400 means *we* sent something wrong — arXiv answers a malformed query
that way, and absorbing it would retry a bug twice and then drop it silently (D-065).

### Step 5 — `gap_check`

One line: `{"depth": state.depth + 1}`. It finds no gaps and calls no model — `decompose` is
the gap finder. `gap_check` only counts the round; `route_after_gap_check` then decides:

```python
reason = exit_reason(depth=..., seen_count=..., seen_before_round=...,
                     source_count=..., dispatched=..., empties_this_round=...)
return "synthesize" if reason else "decompose"
```

The four conditions live in `agent/exits.py`, which is also what the coverage panel reads —
they were defined twice and drifted twice, each time reporting a stop reason the run did not
have (D-094, D-096). In order:

1. **The depth ceiling** (D-009) — the hard backstop; hitting it means the run was cut off.
2. **No new papers at all** (D-075) — correct by construction, and measured as almost never
   firing.
3. **The synthesis prompt is already full** (D-096) — ranking truncates to `SYNTHESIS_TOP_N`
   (D-091), so past that point another round can only change *which* papers the model sees.
   **This is the one that actually fires: 19 of 20 real runs stop here, after a single round.**
4. **The round came back empty** (D-096) — most of its searches returned nothing, so the topic
   is exhausted rather than under-explored.

Together these make depth *adaptive*: `MAX_DEPTH` is a ceiling, not a target. Measured over
twenty questions, three rounds retrieved 2.6× the papers of one for no quality difference
(D-094, D-095), so the exits recover ~67% of the searches without touching the constant.

State accumulates across rounds, so "what did *this* round add" isn't readable from totals —
and a field with `operator.add` can't be reset by a node, since `reducer(current, 0) == current`.
Hence `seen_before_round`, written by `decompose` at the start of each round (D-075).

### Step 6 — `synthesize`

Two messages go to the model. The papers arrive as a delimited data block:

```
<papers>
[arXiv:2411.18583] Automated Literature Review Using NLP Techniques and LLM-Based ...
This research presents and compares multiple approaches to automate the generation of ...
</papers>
```

The marker before each title is the only citation format allowed, and the prompt states that
everything inside `<papers>` is **data, not instructions** (D-055) — abstracts are untrusted
third-party text.

**If `sources` is empty, no model is built and no call is made**: it returns the fixed
`NO_SOURCES_REVIEW` (D-060). Zero results is a success (D-021), and there's no model output
that could invent a citation. The cost: that run streams nothing in `messages` mode.

### Step 7 — `check_claims`

`check_citations` (next) verifies the **ID**. This verifies the **claim** — the one thing here
that cannot be checked deterministically, which is why it is the only place a model judges
anything (D-113).

It extracts every sentence carrying a citation, numbers them, and asks in **one call** whether
each is supported by the paper it cites:

```
Claims:
1. Automated literature review can be approached with retrieval-augmented generation [arXiv:2411.18583].
2. The system achieves a 94.3% ROUGE-L score on PubMed abstracts [arXiv:2411.18583].

<evidence-a1b2c3d4e5f60718>
[arXiv:2411.18583] Automated Literature Review Using NLP Techniques
This research presents and compares multiple approaches to automate...
</evidence-a1b2c3d4e5f60718>
```

Captured from a real call against that input, claim 2 came back:

> *"No mention of ROUGE-L score or 94.3% in the evidence."*

**Three things about this step are deliberate and easy to get wrong:**

- **The evidence is `excerpt or summary` for papers in `synthesized_from`** — exactly what
  step 6 put in front of the writer. D-110 measured a 3.1 SE faithfulness drop when retrieval
  and synthesis read different text; a checker reading different text again would flag
  supported claims for the same reason, pointing the other way.
- **A parse failure is recorded, not raised.** `claims_checked` goes False with a
  `claim_check_error`, and the run finishes. The inverse of `decompose` (D-070): an
  unparseable *plan* means no research happened, so crashing is right; an unparseable
  *judgement* means a finished, ID-verified review went unverified.
- **It runs before `check_citations`** so `citations_checked` stays the last field any node
  writes — D-084 made it the terminal completion marker.

The evidence block carries the same nonce fence as the synthesis prompt (D-097). A checker is
a *more* attractive injection target than a writer: text that talks its way past the thing
verifying it defeats the verification, not just the prose.

### Step 8 — `check_citations`

Finds every citation **attempt**, then tries to parse each one:

| The model writes | Parsed? | Result |
|---|---|---|
| `[arXiv:2411.18583]` | yes, and retrieved | clean |
| `[arXiv:9999.99999]` | yes, not retrieved | violation: `9999.99999` |
| `[arXiv:2411.18583v1]` | yes, but not canonical | violation: `2411.18583v1` |
| `[arXiv:A, B]` | no | violation: `[arXiv:A, B]` |
| `[1]`, `[see Table 2]` | not a citation | ignored |

The fourth row is why a permissive bracket pattern exists alongside the strict marker.
Matching only what the strict pattern understands would make an unreadable citation
indistinguishable from *no* citation, so a review full of malformed markers would report zero
violations — a false negative in the feature this project is about (D-062).

Validation uses Pydantic **context** (`context={"known_ids": ...}`), which lifts the check from
"is this a well-formed arXiv ID" to "is this one of the papers we actually read". Missing
context raises `RuntimeError`, not `ValueError`: Pydantic converts `ValueError` and
`AssertionError` raised in a validator into `ValidationError`, which the node records as a
violation — so a caller bug would mark *every* citation ungrounded.

### The stream, as the browser will see it

Real chunk order from one run (two subtopics, one search round):

```
updates   intake           -> wrote ['question']
messages  decompose        -> 9 token chunks          ← the planner streams too
updates   decompose        -> wrote ['pending_subtopics', 'seen_before_round']
updates   research_worker  -> wrote ['sources', 'skipped_entries', 'explored_subtopics', 'seen_paper_ids']
updates   research_worker  -> wrote ['sources', 'skipped_entries', 'explored_subtopics', 'seen_paper_ids']
updates   gap_check        -> wrote ['depth']
messages  decompose        -> 9 token chunks          ← round 2's planner
updates   decompose        -> wrote ['pending_subtopics', 'seen_before_round']
messages  synthesize       -> 23 token chunks
updates   synthesize       -> wrote ['review']
updates   check_citations  -> wrote ['citation_violations']
```

Three things worth reading off this trace:

1. **`messages` chunks arrive before their node's `updates` chunk.** Tokens stream *while* the
   node runs; the state write lands when it finishes. That ordering is what the milestone-5
   SSE design depends on — `updates` drives node progress, `messages` drives live text.
2. **`decompose` streams as well as `synthesize`.** A consumer that doesn't filter on
   `metadata["langgraph_node"]` will render the planner's raw JSON into the user's review.
3. **`decompose` runs twice for a one-round result.** `gap_check` saw new papers and routed
   back; the second plan re-proposed the same subtopics, the explored filter dropped them all,
   and the empty plan routed to `synthesize`. That second planner call is the real cost of
   recursion when there's nothing new to find.

The two `research_worker` entries are a **single super-step** — a `Send` fan-out is one step
however wide, which is why `recursion_limit` doesn't depend on `MAX_SUBTOPICS`.

(The review text is from the fake model used in tests. Against real OpenAI the wording differs
every run, which is why the integration test asserts *properties* — that the joined stream
equals the saved review, that a citation exists, that violations are empty — never fixed text.)

Final state, checkpointed after every step:

```python
question            = 'What is attention in transformer models?'
review              = "Retrieval grounds a model's answer in papers it has just read [arXiv:2411.18583]."
pending_subtopics   = []                       # emptied by round 2's filter — that's the exit
explored_subtopics  = ['attention mechanisms', 'positional encoding']
empty_subtopics     = []
failed_subtopics    = []
seen_paper_ids      = {'2510.22344', '2502.00306', '2411.18583'}
sources             = [3 Source objects]       # both workers found the same 3; merge_sources dedups
skipped_entries     = 0
depth               = 1                        # one round completed
seen_before_round   = 3                        # the baseline round 2 was measured against
empty_before_round  = 0                        # the same baseline for zero-result searches (D-096)
local_subtopics     = []                       # nothing answered from the corpus here (D-105)
unsupported_claims  = []                       # every cited sentence checked out (D-113)
claims_checked      = True                     # checked, not merely empty (D-084's lesson)
citation_violations = []
synthesized_from    = ['2411.18583', '2510.22344', '2502.00306']
citations_checked   = True                     # the terminal completion marker (D-084)
```

and the coverage summary built from it:

```python
stopped_because = "a round found no papers that earlier rounds hadn't already seen"
```

Note which exit fired: only **3** papers, far under `SYNTHESIS_TOP_N = 20`, so D-096's
sufficiency exit stays quiet and this stays a two-round run. A real run retrieves ~30 papers
in round 1 and stops there instead — the fixture is small on purpose, and it is worth knowing
that it exercises a path production almost never takes.

Six of these fields are written by **parallel** workers, so each needs a reducer or LangGraph
raises `InvalidUpdateError` (D-067). `pending_subtopics`, `depth`, `seen_before_round` and
`empty_before_round` have single writers and deliberately have none, as do `unsupported_claims` and
`claims_checked`.

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
| A node added, removed or renamed | the step headings, the graph diagram, the stream-order block |
| A `ResearchState` field added or renamed | the final-state block, the one-writer note |
| `build_search_query` or the arXiv params | the query string and the request URL |
| `Source`'s fields | the `Source(...)` block |
| The `synthesize` prompt or `format_papers` | the `<papers>` block |
| Stream modes, or where tokens come from | the stream-order block and the note under it |
| A new milestone's graph shape | most of §1 — rewrite rather than patch (done for milestone 4) |
| A new reducer, or a field changing writer | the final-state block and the note under it |
| A node added to the terminal path | the diagram, step headings, the stream-order block, **and `test_recursion.py`'s measured minimum** -- a node there costs one super-step (D-113 moved it 13 -> 14) |
| A stopping rule added or changed (`agent/exits.py`) | step 5's condition list, the graph diagram's cycle label, and `stopped_because` in the final-state block |

**How the values were captured:** the graph was run against the saved
`tests/agent/fixtures/arxiv/search_ok.xml` fixture with `RecordingFactory` as the model
factory — no network, no API cost — printing each node's update, every `messages` chunk,
the final state, and the URL the stub received. Anything in §1 can be regenerated that way;
nothing in it should be written from memory.

§2 changes far less often. Revisit it when the working agreement moves — as it did on
2026-09-19 (tests) and 2026-09-20 (docs).
