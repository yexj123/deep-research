"""The public way to drive a run (settles O-7).

Everything a caller needs to get right is here rather than in a route: the run context
(D-033), the invoke config including `recursion_limit` (D-077), the stream modes, and the
three-state start/resume/replay decision (D-081).

Kept in the agent package, not `api/`, so the decision is testable without HTTP and usable
from a script or notebook.
"""

from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from deep_research.agent.config import RECURSION_LIMIT
from deep_research.agent.context import ProviderType, RunContext

STREAM_MODES = ["updates", "custom", "messages"]


class RunState(str, Enum):
    """Which of D-081's three states a thread_id is in."""

    NOT_STARTED = "not_started"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"


def _config(thread_id: str) -> dict[str, Any]:
    # recursion_limit is invoke config, not graph config: leave it out and LangGraph uses its
    # default of 25 rather than the measured 15 (D-077).
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


async def get_run_state(graph: CompiledStateGraph, thread_id: str) -> RunState:
    """Classify a thread from its checkpoint (D-081, revised by D-084).

    Two things make this trickier than it looks, both measured 2026-09-21:

    - `created_at is None` is the only reliable "nothing ever ran" signal. `values` and `next`
      are both empty for a fresh thread *and* for a finished one.
    - **`next == ()` does NOT mean finished.** A run interrupted at a step boundary also has an
      empty `next`, and which happens depends on the stream modes: breaking after decompose
      leaves `next=('research_worker', 'research_worker')` with `stream_mode=["updates"]`, but
      `next=()` with `["updates", "custom", "messages"]` -- the production set. So `next` is
      useless here.

    `citations_checked` is written only by the terminal node, which makes it the one honest
    completion signal. `review` would be wrong too: a run interrupted between synthesize and
    check_citations has a review and an empty `citation_violations`, and calling that finished
    would report "no ungrounded citations" without ever having checked (D-062's failure shape).
    """
    snapshot = await graph.aget_state(_config(thread_id))
    if snapshot.created_at is None:
        return RunState.NOT_STARTED
    return RunState.FINISHED if snapshot.values.get("citations_checked") else RunState.INTERRUPTED


async def stream_run(
    graph: CompiledStateGraph,
    thread_id: str,
    question: str,
    provider: ProviderType,
    model: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """Stream a run, starting it or resuming it as the checkpoint requires (D-081).

    Yields raw LangGraph chunks (`{"type", "ns", "data"}`, D-030). Formatting them as SSE is
    the API layer's job -- this stays usable from a script.

    A FINISHED thread yields nothing: the caller reads the saved review from the checkpoint
    instead. Re-running it would repeat every paid call to produce a review that already exists.

    On a caller disconnect this generator is closed, which leaves the checkpoint resumable --
    the pending `Send` fan-out survives (D-081). Nothing here needs to catch that.
    """
    state = await get_run_state(graph, thread_id)
    if state is RunState.FINISHED:
        return

    # None resumes from the checkpoint; the question starts a fresh run.
    graph_input = None if state is RunState.INTERRUPTED else {"question": question}

    async for chunk in graph.astream(
        graph_input,
        _config(thread_id),
        context=RunContext(provider=provider, model=model),
        stream_mode=STREAM_MODES,
        version="v2",
    ):
        yield chunk


async def get_review(graph: CompiledStateGraph, thread_id: str) -> dict[str, Any]:
    """The finished state of a run, as a plain dict (`aget_state().values`, D-030)."""
    return (await graph.aget_state(_config(thread_id))).values
