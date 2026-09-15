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
- **Allowlist every custom type that ends up in state.** With
  `LANGGRAPH_STRICT_MSGPACK` / `allowed_msgpack_modules`, a checkpoint can
  only deserialize custom Pydantic models and dataclasses that are
  explicitly allowlisted. Anything missing from the list breaks resuming a
  run from SQLite. When you add a model to state, add it to the allowlist
  in the same change.
- Every key a parallel `Send` branch writes to needs an explicit reducer
  (e.g. `Annotated[list[str], operator.add]`), or concurrent writes silently
  clobber each other instead of merging.

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
  its results are already in `seen_paper_ids` (D-022). Zero results never
  reach the overlap check. Pass `seen_paper_ids` in the `Send` payload, and give it a
  deduplicating reducer.
- The worker catches only expected external or I/O failures (httpx errors,
  `ValidationError` on external data). Never catch `BaseException`, and let
  programming errors crash (D-023). N counts failed worker runs, not HTTP
  retries (D-024).
- `max_depth = 2` baseline (D-025).
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
        Send("research_worker", {"subtopic": t})
        for t in state["pending_subtopics"]
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

No claim in the synthesized report without a source object it came from
still attached in state. **TODO**: fill in the exact source-tracking shape
once decided (e.g. a `Source` dataclass carried alongside each finding).

## Reference implementations

Worth reading, not copying, both MIT licensed:
[`langchain-ai/open_deep_research`](https://github.com/langchain-ai/open_deep_research)
(production-shaped: clarify → brief → supervisor/researcher subgraphs →
report) and
[`langchain-ai/deep_research_from_scratch`](https://github.com/langchain-ai/deep_research_from_scratch)
(notebook-by-notebook build-up — closer in spirit to how we're doing this).
