"""Graph-level recursion tests for milestone 4 (D-009, D-075, D-076, D-077).

These run the whole cycle:

    intake -> decompose -> Send -> research_worker -> gap_check
                  ^                                      |
                  +----------- depth left & new papers --+

What they pin down is that the run ends for a *stated reason*. A GraphRecursionError here
would mean the semantic exit is broken, which is exactly the inversion D-009 warns against --
so these assert the recursion_limit is never the thing that stops a run.
"""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from deep_research.agent.config import MAX_DEPTH, RECURSION_LIMIT
from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.state import ResearchState
from tests.agent.fakes import (
    DEFAULT_REPLY,
    NullLimiter,
    RecordingFactory,
    make_arxiv_feed,
    make_arxiv_router_stub,
    plan_reply,
)

# One distinct plan per round, then the review. Model builds happen in this order:
# decompose(round 0), decompose(round 1), decompose(round 2), synthesize -- gap_check and
# the workers call no model.
THREE_ROUND_PLANS = [
    plan_reply("alpha topic"),
    plan_reply("beta topic"),
    plan_reply("gamma topic"),
    DEFAULT_REPLY,
]

# Each round's subtopic finds papers the earlier rounds didn't, so the run stays productive
# and only depth can stop it (D-075).
FRESH_PAPERS_PER_ROUND = {
    "alpha": (make_arxiv_feed("2401.00001", "2401.00002"), 200),
    "beta": (make_arxiv_feed("2402.00001", "2402.00002"), 200),
    "gamma": (make_arxiv_feed("2403.00001", "2403.00002"), 200),
}


def _config(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


async def _run(routes, replies, thread: str, recursion_limit: int = RECURSION_LIMIT):
    """Run the graph once against a routing arXiv stub. Returns the final ResearchState."""
    stub = make_arxiv_router_stub(routes, fallback=(make_arxiv_feed(), 200))
    async with stub.client:
        graph = build_graph(
            RecordingFactory(replies=replies), stub.client, NullLimiter(), InMemorySaver()
        )
        output = await graph.ainvoke(
            {"question": "What is attention in transformer models?"},
            _config(thread, recursion_limit),
            context=RunContext(provider="openai"),
            version="v2",
        )
    return output.value


# ---- the full-depth run (D-026, D-076, D-077) ----------------------------------------


@pytest.mark.asyncio
async def test_a_full_depth_run_completes_within_the_recursion_limit() -> None:
    """Three productive rounds finish under RECURSION_LIMIT (D-077).

    This is the test that keeps the number honest, and it has already earned its place:
    adding `check_claims` (D-113) moved the measured minimum from 13 to 14, and this file is
    what said so. RECURSION_LIMIT = 15 now carries **one** step of headroom rather than two,
    so the next node added to the terminal path needs the constant raised with it.
    """
    state = await _run(FRESH_PAPERS_PER_ROUND, THREE_ROUND_PLANS, "full-depth")

    assert isinstance(state, ResearchState)
    assert state.depth == MAX_DEPTH + 1, "three rounds completed, counted after each (D-076)"
    assert sorted(state.explored_subtopics) == ["alpha topic", "beta topic", "gamma topic"]
    assert len(state.sources) == 6, "two papers per round, all distinct"


@pytest.mark.asyncio
async def test_the_depth_exit_fires_before_the_recursion_limit() -> None:
    """The semantic exit stops the run, not the backstop (D-009, D-077).

    Run with recursion_limit at the measured minimum: if depth weren't stopping it first,
    this would raise GraphRecursionError. Hitting that in real use means the depth logic is
    broken -- fix the exit, don't raise the number.
    """
    state = await _run(FRESH_PAPERS_PER_ROUND, THREE_ROUND_PLANS, "min-limit", recursion_limit=14)
    assert state.depth == MAX_DEPTH + 1


@pytest.mark.asyncio
async def test_thirteen_is_one_step_too_few() -> None:
    """Pins the measured minimum: 13 super-steps + 1 is required (D-077, D-113).

    The control for the test above. Without it, RECURSION_LIMIT could drift far above what's
    needed and nothing would notice. Also documents that LangGraph needs super-steps + 1.

    **It was 12 until `check_claims` landed.** A node on the terminal path costs one
    super-step, and this pair of tests is the only thing that makes that visible -- the
    constant itself does not change, so nothing else would have noticed the headroom halving.
    """
    with pytest.raises(GraphRecursionError):
        await _run(FRESH_PAPERS_PER_ROUND, THREE_ROUND_PLANS, "too-few", recursion_limit=13)


# ---- stopping early (D-075) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_round_that_adds_no_new_papers_ends_the_run() -> None:
    """Every round returning the same papers stops the run early (D-075).

    Round 0 is productive by definition -- it goes from zero papers to some -- so the exit
    can only fire after round 1, which searches a different subtopic but retrieves nothing
    unseen. The run therefore ends at depth 2, short of MAX_DEPTH + 1 == 3: one round of
    recursion was spent proving there was nothing new, and the third was skipped.
    """
    same_papers = (make_arxiv_feed("2401.00001", "2401.00002"), 200)
    routes = {"alpha": same_papers, "beta": same_papers, "gamma": same_papers}
    state = await _run(routes, THREE_ROUND_PLANS[:2] + [DEFAULT_REPLY], "no-new-papers")

    assert state.depth == 2, "stopped after round 1, not after all three"
    assert sorted(state.explored_subtopics) == ["alpha topic", "beta topic"]
    assert len(state.sources) == 2, "round 1 re-retrieved the same papers; merge_sources dedups"


@pytest.mark.asyncio
async def test_a_first_round_finding_nothing_ends_the_run() -> None:
    """An empty first round costs one round, not three (D-021, D-075).

    "Nothing published on X" is a finding. Recursing on it would re-plan twice more and find
    nothing again, at the price of two extra paid calls.
    """
    state = await _run({}, THREE_ROUND_PLANS, "empty-first")

    assert state.depth == 1
    assert state.sources == []
    assert state.explored_subtopics == ["alpha topic"], "zero results is still a success (D-021)"


@pytest.mark.asyncio
async def test_the_planner_sees_what_earlier_rounds_explored() -> None:
    """explored_subtopics accumulates across rounds so the planner isn't asked to repeat (D-017).

    merge_subtopics is what makes this hold: each round's worker adds its own subtopic, and
    the list survives into the next round's prompt.
    """
    state = await _run(FRESH_PAPERS_PER_ROUND, THREE_ROUND_PLANS, "accumulates")
    assert len(state.explored_subtopics) == 3, "one per round, accumulated (D-067)"
