---
description: This project's LangGraph implementation patterns — state schema shape, node template, Send fan-out template, recursion-depth guard, checkpointing. Load when writing or reviewing a node, edge, or the state schema.
---

# LangGraph conventions for this project

## State schema

- **Decided**: Pydantic at the boundaries, dataclasses inside. The top-level
  graph state is a dataclass. Anything arriving from outside the process
  (LLM structured output, arXiv/Semantic Scholar responses) is a Pydantic
  model, validated explicitly with `Model.model_validate(...)` in the node
  that receives it. Reason: when the state itself is Pydantic, LangGraph only
  validates the input to the *first* node, so validation has to happen at
  each boundary on purpose.
- **`Source` is the exception to the two-type split:** it's a single
  `@pydantic.dataclasses.dataclass(frozen=True)`, validated when it's built from
  arXiv data *and* stored in state (D-043). Fields: `arxiv_id` (no version),
  `version`, `title`, `authors: tuple[str, ...]`, `summary`, `published: datetime`, `url` (D-044).
  Entries that fail validation are skipped and counted (D-045).
- **Allowlist every custom type that ends up in state.** With
  `LANGGRAPH_STRICT_MSGPACK` / `allowed_msgpack_modules`, a checkpoint can
  only deserialize custom Pydantic models and dataclasses that are
  explicitly allowlisted. A type missing from the list does **not** raise an error: it
  comes back as a plain `dict` with only a logged warning, and the next attribute
  access fails far from the cause (confirmed on langgraph 1.2.11; D-014). When you
  add a model to state, add it to the allowlist in the same change.
- A key needs an explicit reducer (e.g. `Annotated[list[str], operator.add]`)
  in two cases:
  1. **More than one node writes it in the same step.** This means parallel `Send` branches
     *or* static fan-out edges. Without a reducer, LangGraph raises
     `InvalidUpdateError: At key 'x': Can receive only one value per step`
     (confirmed on langgraph 1.2.11). It fails loudly; it does not silently overwrite.
  2. **The value should accumulate across sequential steps** (e.g.
     `explored_subtopics` across rounds). Without a reducer, each write
     replaces the previous value.
- **Run context:** calling the graph without `context=` makes `runtime.context`
  `None`, and `Literal` hints aren't checked at runtime. `intake` raises
  `ValueError` if the context is missing or the provider is invalid (D-033).
- **Dependencies:** `build_graph` receives the model factory, the checkpointer and an
  `httpx.AsyncClient`. Nodes never construct their own model client, HTTP client or
  checkpointer (D-032, D-049). Tests pass an `httpx.MockTransport`-backed client.
- **Checkpoint round trip:** every custom type in state gets a test that saves it through
  the real serializer settings and asserts it comes back as the same type (D-050). The allowlist
  in `persistence/checkpointer.py` builds each entry from the class itself (D-057), so a move or
  rename is picked up; the test is what catches a new type that was never added. The `checkpointer`
  test fixture uses `build_serializer()`, so every graph test restores state the way production does.
- **Milestone 2 names (D-054):** nodes `intake → search → synthesize → check_citations`; state fields
  `sources`, `skipped_entries`, `citation_violations`. At milestone 2 the `search` node catches
  nothing (D-053). Retrieved papers reach the model as a delimited data block (D-055).

## Subtopic bookkeeping

- **Decided**: a subtopic counts as explored only when its worker *succeeds*.
  The worker writes `explored_subtopics`; if the search fails, the subtopic
  stays unexplored and can be proposed again in a later round.
- `explored_subtopics` accumulates across rounds and has parallel writers, so
  it uses a deduplicating reducer (normalized with casefold + strip).
- `pending_subtopics` has **no reducer** and is overwritten each round.
  With `operator.add`, round N would dispatch the subtopics from every
  earlier round again.
- Worker failures are handled with `try/except` **inside the worker**, which
  returns `{"failed_subtopics": [subtopic]}` instead of raising (D-019).
- `failed_subtopics: Annotated[list[str], operator.add]`. Duplicates are
  intentional because each entry is one failed attempt. `decompose` skips a subtopic
  that has failed **2** times (D-020).
- Zero search results is a **success**: mark the subtopic explored (D-021).
- Skip a subtopic if its normalized form is already explored, or if ≥ 60% of
  its results are already in `seen_paper_ids` (D-022). The overlap check applies only
  when there are ≥ 3 results (D-028), and zero results never reach it. Pass `seen_paper_ids` in the `Send` payload, and give it a
  deduplicating reducer.
- The worker catches exactly
  `(httpx.TransportError, httpx.HTTPStatusError, pydantic.ValidationError,
  xml.etree.ElementTree.ParseError, defusedxml.DefusedXmlException)`
  (D-023, D-027, D-048). arXiv XML is parsed with
  `defusedxml.ElementTree.fromstring(..., forbid_dtd=True)` (D-047). Never catch `BaseException`, let programming errors crash,
  and add no backoff around LLM calls, since the OpenAI client already retries. N counts
  failed worker runs, not HTTP retries (D-024).
- `depth` is 0-indexed. Recurse while `depth < max_depth`. With `max_depth = 2`,
  that's 3 search passes (D-025, D-026).
- Reasoning and open questions (subtopic comparison, which exceptions to
  catch) are in `docs/decisions.md`.

## Node template

**TODO**: fill this in with a real node once the first one is written, as a
concrete example future sessions (and Claude Code) can pattern-match against.

## Fan-out (recursive subtopic search)

Use `Send` from `langgraph.types` to dispatch one worker per subtopic at
runtime — the count isn't known until the planner runs, so a static edge
can't express it:

```python
from langgraph.types import Send

def route_subtopics(state: ResearchState) -> list[Send]:
    return [
        # A worker only sees its Send payload, so pass what it needs (D-022).
        Send("research_worker", {"subtopic": t, "seen_paper_ids": state.seen_paper_ids})
        for t in state.pending_subtopics  # state is a dataclass: attribute access
    ]
```

## Recursion / depth control

Two layers, not one:

1. **Semantic exit** — a `depth` field in state; the `gap_check` conditional
   edge routes to synthesis once `depth >= max_depth` or the last pass found
   no new gaps. This is the real stop condition and should trigger well
   before the graph gets anywhere near the hard limit.
2. **Safety net** — `recursion_limit` in the invoke config (LangGraph's
   default is 25, too low for a multi-round recursive fan-out; set it
   explicitly and generously, e.g.
   `graph.invoke(input, config={"recursion_limit": 150})`). If you ever hit
   `GraphRecursionError` (from `langgraph.errors`), that means layer 1 didn't
   fire — fix the semantic exit, don't just raise the number again.

## Citation grounding

No claim in the synthesized report without a `Source` it came from still in
state (D-046):

1. `synthesize` writes prose with inline markers in exactly this form:
   `[arXiv:<arxiv_id>]`, using the versionless ID (D-044). The system prompt
   has to require that format.
2. A separate node after `synthesize` extracts every marker and validates
   each ID with a Pydantic model given `context={"known_ids": <retrieved IDs>}`
   (the `field_validator` reads `info.context`). Unknown IDs are recorded in
   state, not silently dropped.
3. Checking the format (D-041) is not grounding: a well-formed ID the model
   invented passes a format check but must fail step 2.
4. **Find citation *attempts*, not just citations (D-062).** Match every
   `[arXiv:...]` bracket with a permissive pattern, then try to parse each one
   with the strict marker. A bracket that won't parse — `[arXiv:A, B]`,
   `[arXiv: A]` — is recorded verbatim as a violation. Extracting only what the
   strict pattern matches makes an unreadable citation indistinguishable from no
   citation, so a review full of malformed markers reports zero violations. Never
   let "I couldn't read it" render as "I verified it".
5. Missing validation context must raise `RuntimeError`, not `ValueError` or an
   `assert`: Pydantic converts the latter two into `ValidationError`, which the
   node records as a violation — so a caller bug would mark *every* citation
   ungrounded instead of crashing.

## Reference implementations

Worth reading, not copying, both MIT licensed:
[`langchain-ai/open_deep_research`](https://github.com/langchain-ai/open_deep_research)
(production-shaped: clarify → brief → supervisor/researcher subgraphs →
report) and
[`langchain-ai/deep_research_from_scratch`](https://github.com/langchain-ai/deep_research_from_scratch)
(notebook-by-notebook build-up — closer in spirit to how we're doing this).
