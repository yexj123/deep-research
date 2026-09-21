"""stream_run tests: the start / resume / replay decision (D-081, O-7).

Tested here rather than through the API on purpose. `httpx.ASGITransport` drives a
StreamingResponse generator to completion regardless of the client breaking out of the
iteration -- measured 2026-09-21: after "disconnecting" two lines in, the run had already
finished. A disconnect test at the HTTP level would pass without exercising anything.

Closing the async generator is what FastAPI actually does when a client goes away, and that is
directly reproducible here.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.graph import build_graph
from deep_research.agent.runner import RunState, get_review, get_run_state, stream_run
from deep_research.persistence.checkpointer import build_serializer
from tests.agent.fakes import (
    DEFAULT_REPLY,
    NullLimiter,
    RecordingFactory,
    load_arxiv_fixture,
    make_arxiv_stub,
    one_round_replies,
)

QUESTION = "What is attention in transformer models?"


def build(factory: RecordingFactory, stub):
    return build_graph(factory, stub.client, NullLimiter(), InMemorySaver(serde=build_serializer()))


async def drain(graph, thread_id: str, factory: RecordingFactory) -> list[dict]:
    chunks = []
    async for chunk in stream_run(graph, thread_id, QUESTION, "openai"):
        chunks.append(chunk)
    return chunks


# ---- the three states (D-081) --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fresh_thread_is_not_started() -> None:
    """created_at is None only when nothing was ever checkpointed (D-081)."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        graph = build(RecordingFactory(replies=one_round_replies()), stub)
        assert await get_run_state(graph, "never-run") is RunState.NOT_STARTED


@pytest.mark.asyncio
async def test_a_completed_thread_is_finished() -> None:
    """Finished means the terminal node ran, proven by citations_checked (D-084).

    `next` is not usable here: a finished run and one interrupted at a step boundary both
    have an empty `next`. `review` is not enough either -- a run stopped between synthesize
    and check_citations has one, and calling that finished would report zero citation
    violations without ever having checked.
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        factory = RecordingFactory(replies=one_round_replies())
        graph = build(factory, stub)
        await drain(graph, "done-thread", factory)
        assert await get_run_state(graph, "done-thread") is RunState.FINISHED


@pytest.mark.asyncio
async def test_an_abandoned_run_is_interrupted_and_keeps_its_pending_work() -> None:
    """Closing the stream mid-run leaves a resumable checkpoint (D-081).

    `next` names the nodes that had not run, including the pending Send fan-out -- those
    payloads survive because they are checkpointed too (D-071). This is what makes a dropped
    browser connection cost the in-flight node rather than the run.
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        factory = RecordingFactory(replies=one_round_replies())
        graph = build(factory, stub)

        agen = stream_run(graph, "abandoned", QUESTION, "openai")
        updates = 0
        async for chunk in agen:
            if chunk["type"] == "updates":
                updates += 1
                if updates == 2:  # intake, decompose -- stop before the workers
                    break
        await agen.aclose()  # what FastAPI does when the client goes away

        assert await get_run_state(graph, "abandoned") is RunState.INTERRUPTED
        snapshot = await graph.aget_state({"configurable": {"thread_id": "abandoned"}})
        assert not snapshot.values.get("citations_checked"), "the terminal node had not run"
        # Deliberately NOT asserting snapshot.next: with the production stream modes it is
        # empty here, and with ["updates"] alone it names the pending workers. That
        # difference is exactly why D-084 stopped using it (measured 2026-09-21).


@pytest.mark.asyncio
async def test_resuming_continues_instead_of_restarting() -> None:
    """A resumed run finishes without redoing the work already checkpointed (D-081).

    The proof is the arXiv request count: a restart would search both subtopics again, so the
    stub would see four requests rather than two.
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        factory = RecordingFactory(replies=one_round_replies())
        graph = build(factory, stub)

        agen = stream_run(graph, "resumed", QUESTION, "openai")
        updates = 0
        async for chunk in agen:
            if chunk["type"] == "updates":
                updates += 1
                if updates == 2:
                    break
        await agen.aclose()
        assert stub.requests == [], "the workers had not run yet"
        assert await get_run_state(graph, "resumed") is RunState.INTERRUPTED

        await drain(graph, "resumed", factory)

        values = await get_review(graph, "resumed")
        assert values["review"] == DEFAULT_REPLY
        assert len(stub.requests) == 2, "a restart would have searched both subtopics twice"
        assert sorted(values["explored_subtopics"]) == [
            "attention mechanisms",
            "positional encoding",
        ]


@pytest.mark.asyncio
async def test_a_finished_run_yields_nothing() -> None:
    """Replay produces no chunks at all, so no model call is repeated (D-081).

    The caller reads the saved review from the checkpoint instead. Re-running would repeat
    every paid call to produce a review that already exists.
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        factory = RecordingFactory(replies=one_round_replies())
        graph = build(factory, stub)
        await drain(graph, "replay", factory)

        models_after_first = factory.models_built
        requests_after_first = len(stub.requests)

        assert await drain(graph, "replay", factory) == []
        assert factory.models_built == models_after_first
        assert len(stub.requests) == requests_after_first


@pytest.mark.asyncio
async def test_stream_run_requests_all_three_stream_modes() -> None:
    """updates, custom and messages arrive on one iterator (D-030).

    `custom` has no producer yet -- no node calls get_stream_writer -- but it is requested now
    so adding one later needs no change here. See Open, "Making failures visible" (O-5).
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        factory = RecordingFactory(replies=one_round_replies())
        graph = build(factory, stub)
        kinds = {chunk["type"] for chunk in await drain(graph, "modes", factory)}

    assert {"updates", "messages"} <= kinds
