"""Graph-level fan-out tests for milestone 3 (D-067, D-068, D-069, D-072).

These run the whole graph, so they cover what the node-level tests can't: that Send actually
dispatches one worker per subtopic, and that the reducers merge parallel updates correctly.

Ordering: assertions on fields written by parallel workers compare sorted values or sets,
because Send dispatch order is observed behavior and not a documented LangGraph contract
(D-068). The one test that does assert order is labelled as pinning that assumption.
"""

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.nodes.synthesize import NO_SOURCES_REVIEW
from deep_research.agent.state import ResearchState
from tests.agent.fakes import (
    DEFAULT_REPLY,
    ArxivStub,
    NullLimiter,
    RecordingFactory,
    load_arxiv_fixture,
    make_arxiv_router_stub,
    plan_reply,
)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _query_terms(stub: ArxivStub) -> set[str]:
    """The search_query of every request the stub received."""
    return {r.url.params["search_query"] for r in stub.requests}


# ---- dispatch ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_worker_runs_per_subtopic(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Two planned subtopics produce two arXiv searches, one per worker (D-069, D-072).

    The default plan proposes "attention mechanisms" and "positional encoding", so the
    queries are built from the subtopics rather than from the question.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    await graph.ainvoke(
        {"question": "What is attention in transformer models?"},
        _config("fanout-two"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert len(arxiv_ok.requests) == 2
    assert _query_terms(arxiv_ok) == {
        "all:attention AND all:mechanisms",
        "all:positional AND all:encoding",
    }


@pytest.mark.asyncio
async def test_every_search_goes_through_the_rate_limiter(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Parallel workers still serialize their arXiv access (D-064).

    arXiv allows one connection at a time, so max_concurrent must stay 1 however many
    workers are dispatched. This is the rule that a token-bucket limiter fails.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-limiter"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert limiter.entered == 2
    assert limiter.max_concurrent == 1, "arXiv permits one connection at a time (D-064)"


# ---- reducers under parallel writes --------------------------------------------------


@pytest.mark.asyncio
async def test_papers_found_by_two_subtopics_are_stored_once(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Both workers get the same 3 papers, and merge_sources keeps 3, not 6 (D-067).

    This is the dedup reducer doing its job across parallel branches: without it the
    <papers> block would list every paper twice.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-dedup"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert sorted(s.arxiv_id for s in output.value.sources) == [
        "2411.18583",
        "2502.00306",
        "2510.22344",
    ]


@pytest.mark.asyncio
async def test_explored_subtopics_collects_both_workers(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Each successful worker adds its own subtopic (D-018, D-067). Order-insensitive (D-068)."""
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-explored"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert sorted(output.value.explored_subtopics) == [
        "attention mechanisms",
        "positional encoding",
    ]


@pytest.mark.asyncio
async def test_seen_paper_ids_is_the_union_across_workers(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """seen_paper_ids unions the parallel updates and survives the checkpoint as a set (D-067)."""
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    config = _config("fanout-seen")
    await graph.ainvoke(
        {"question": "What is attention?"},
        config,
        context=RunContext(provider="openai"),
        version="v2",
    )

    seen = (await graph.aget_state(config)).values["seen_paper_ids"]
    assert isinstance(seen, set), "a set must round-trip as a set, not a list"
    assert seen == {"2411.18583", "2502.00306", "2510.22344"}


@pytest.mark.asyncio
async def test_skipped_entries_sums_across_workers(
    fake_factory: RecordingFactory, limiter: NullLimiter, checkpointer: InMemorySaver
) -> None:
    """Each worker contributes its own delta, and operator.add sums them (D-067).

    Both workers get search_one_invalid_id.xml (1 bad entry each), so the total is 2. A
    worker returning a running total instead of a delta would compound here.
    """
    stub = make_arxiv_router_stub({}, fallback=(load_arxiv_fixture("search_one_invalid_id.xml"), 200))
    async with stub.client:
        graph = build_graph(RecordingFactory(replies=[plan_reply("alpha topic", "beta topic"), DEFAULT_REPLY]), stub.client, limiter, checkpointer)
        output = await graph.ainvoke(
            {"question": "What is attention?"},
            _config("fanout-skipped"),
            context=RunContext(provider="openai"),
            version="v2",
        )

    assert output.value.skipped_entries == 2


# ---- partial failure -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_worker_failing_does_not_stop_the_others(
    limiter: NullLimiter, checkpointer: InMemorySaver
) -> None:
    """A failed subtopic is recorded while the healthy one still contributes papers (D-019, D-072).

    This is the payoff for handling failures inside the worker: one flaky search doesn't
    end a long run, and the failure becomes data rather than an exception.
    """
    stub = make_arxiv_router_stub(
        {"alpha": ("", 503)},
        fallback=(load_arxiv_fixture("search_ok.xml"), 200),
    )
    factory = RecordingFactory(replies=[plan_reply("alpha topic", "beta topic"), DEFAULT_REPLY])
    async with stub.client:
        graph = build_graph(factory, stub.client, limiter, checkpointer)
        output = await graph.ainvoke(
            {"question": "What is attention?"},
            _config("fanout-partial"),
            context=RunContext(provider="openai"),
            version="v2",
        )

    assert output.value.failed_subtopics == ["alpha topic"]
    assert output.value.explored_subtopics == ["beta topic"]
    assert len(output.value.sources) == 3


@pytest.mark.asyncio
async def test_a_failed_subtopic_is_not_marked_explored(
    limiter: NullLimiter, checkpointer: InMemorySaver
) -> None:
    """Every worker failing leaves explored empty, so the subtopics can be retried (D-018).

    If a failure marked a subtopic explored, a temporary outage would permanently drop it.
    """
    stub = make_arxiv_router_stub({}, fallback=("", 503))
    factory = RecordingFactory(replies=[plan_reply("alpha topic", "beta topic"), DEFAULT_REPLY])
    async with stub.client:
        graph = build_graph(factory, stub.client, limiter, checkpointer)
        output = await graph.ainvoke(
            {"question": "What is attention?"},
            _config("fanout-all-fail"),
            context=RunContext(provider="openai"),
            version="v2",
        )

    assert sorted(output.value.failed_subtopics) == ["alpha topic", "beta topic"]
    assert output.value.explored_subtopics == []
    assert output.value.review == NO_SOURCES_REVIEW, "no papers means the fixed review (D-060)"


@pytest.mark.asyncio
async def test_a_4xx_from_one_worker_fails_the_whole_run(
    limiter: NullLimiter, checkpointer: InMemorySaver
) -> None:
    """A 400 is our bug, so it propagates out of the fan-out instead of being recorded (D-065).

    The contrast with the 503 test above is the point: transient failures become data,
    our own mistakes crash.
    """
    stub = make_arxiv_router_stub({"alpha": ("", 400)}, fallback=(load_arxiv_fixture("search_ok.xml"), 200))
    factory = RecordingFactory(replies=[plan_reply("alpha topic", "beta topic"), DEFAULT_REPLY])
    async with stub.client:
        graph = build_graph(factory, stub.client, limiter, checkpointer)
        with pytest.raises(httpx.HTTPStatusError):
            await graph.ainvoke(
                {"question": "What is attention?"},
                _config("fanout-4xx"),
                context=RunContext(provider="openai"),
                version="v2",
            )


# ---- the empty fan-out guard (D-069) -------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_plan_still_reaches_synthesize(
    arxiv_ok: ArxivStub, limiter: NullLimiter, checkpointer: InMemorySaver
) -> None:
    """A planner proposing nothing must not end the run silently (D-069).

    Measured 2026-09-20: a conditional edge returning [] produces no error and no downstream
    node. Without the guard this run would finish with review == "" and no indication why.
    """
    factory = RecordingFactory(replies=[plan_reply(), DEFAULT_REPLY])
    graph = build_graph(factory, arxiv_ok.client, limiter, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-empty-plan"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert arxiv_ok.requests == [], "nothing to research means no searches"
    assert output.value.review == NO_SOURCES_REVIEW
    assert output.value.citation_violations == []


# ---- stream shape --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_updates_arrive_as_one_step(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Both workers report under the same node name, in one super-step (D-068).

    Captured for docs/langgraph-outputs.md: a Send fan-out is a single super-step however
    many workers it dispatches, which is the arithmetic behind the recursion_limit estimate.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)

    node_order: list[str] = []
    async for chunk in graph.astream(
        {"question": "What is attention?"},
        _config("fanout-stream"),
        context=RunContext(provider="openai"),
        stream_mode=["updates"],
        version="v2",
    ):
        if chunk["type"] == "updates":
            node_order.extend(chunk["data"].keys())

    assert node_order.count("research_worker") == 2
    assert node_order[0] == "intake"
    assert node_order[-1] == "check_citations"


@pytest.mark.asyncio
async def test_the_planner_also_streams_in_messages_mode(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """decompose's tokens reach the messages stream too, so consumers must filter by node.

    At milestone 2 every messages chunk came from synthesize. From milestone 3 the planner's
    JSON streams as well -- the web layer must filter on langgraph_node, or raw JSON appears
    in the user's review.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)

    nodes_that_streamed: set[str] = set()
    async for chunk in graph.astream(
        {"question": "What is attention?"},
        _config("fanout-messages"),
        context=RunContext(provider="openai"),
        stream_mode=["messages"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            nodes_that_streamed.add(chunk["data"][1]["langgraph_node"])

    assert nodes_that_streamed == {"decompose", "synthesize"}


# ---- ordering, pinned deliberately (D-068) -------------------------------------------


@pytest.mark.asyncio
async def test_send_dispatch_order_is_preserved(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """Parallel worker updates apply in dispatch order, not completion order (D-068).

    PINS OBSERVED BEHAVIOR, NOT A CONTRACT. LangGraph does not document this. It is what
    makes the <papers> block reproducible run to run, which the thesis evaluation relies on,
    so it is worth knowing if it changes -- but every other test here is order-insensitive
    so that a change breaks this one test and nothing else.
    """
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-order"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert output.value.explored_subtopics == ["attention mechanisms", "positional encoding"]


@pytest.mark.asyncio
async def test_final_state_is_a_research_state_after_fan_out(
    fake_factory: RecordingFactory,
    arxiv_ok: ArxivStub,
    limiter: NullLimiter,
    checkpointer: InMemorySaver,
) -> None:
    """A fan-out run still returns a ResearchState, with the reducer'd fields populated."""
    graph = build_graph(fake_factory, arxiv_ok.client, limiter, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("fanout-final"),
        context=RunContext(provider="openai"),
        version="v2",
    )

    assert isinstance(output.value, ResearchState)
    assert output.value.review == DEFAULT_REPLY
    assert output.value.pending_subtopics == ["attention mechanisms", "positional encoding"]
