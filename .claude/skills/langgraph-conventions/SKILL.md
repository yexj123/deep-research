---
description: This project's LangGraph implementation patterns — state schema shape, node template, Send fan-out template, recursion-depth guard, checkpointing. Load when writing or reviewing a node, edge, or the state schema.
---

# LangGraph conventions for this project

## State schema

- **TODO**: TypedDict or Pydantic model — decide and note the choice here.
- Every key a parallel `Send` branch writes to needs an explicit reducer
  (e.g. `Annotated[list[str], operator.add]`), or concurrent writes silently
  clobber each other instead of merging.

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
