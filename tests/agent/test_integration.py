"""Integration test: one real run through the whole graph, with OpenAI and arXiv (D-036, D-037).

Makes a real, paid OpenAI call and a real arXiv request. Deselected by default; run it with
`uv run pytest -m integration`. Also skipped when OPENAI_API_KEY isn't set.
"""

import os

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.config import ARXIV_MIN_INTERVAL_SECONDS, ARXIV_TIMEOUT_SECONDS
from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import get_chat_model
from deep_research.agent.nodes.check_citations import CITATION_MARKER
from deep_research.agent.sources.rate_limit import ArxivRateLimiter


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_real_run_streams_a_review_citing_only_retrieved_papers(
    checkpointer: InMemorySaver,
) -> None:
    """A real run fans out to several subtopics, streams the review in several chunks, and
    every citation is a retrieved paper.

    A real model words its reply differently every run, so the test compares the
    run with itself instead of with fixed text.

    Milestone 3 adds a real planner call and a real fan-out, so this makes MAX_SUBTOPICS + 1
    paid calls (one per planner round plus the synthesis) and MAX_SUBTOPICS arXiv requests,
    spaced by the real limiter at ARXIV_MIN_INTERVAL_SECONDS. Expect it to take roughly
    MAX_SUBTOPICS x 3 seconds longer than the milestone 2 version.
    """
    config = _config("integration")
    tokens: list[str] = []

    # The caller owns the HTTP client, its timeout and the rate limiter (D-049, D-052, D-064).
    # The real limiter, not the test double: this run must honor arXiv's terms of use.
    limiter = ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS)
    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS) as http_client:
        graph = build_graph(get_chat_model, http_client, limiter, checkpointer)
        async for chunk in graph.astream(
            {"question": "What is attention in transformer models?"},
            config,
            context=RunContext(provider="openai"),
            stream_mode=["messages"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]
                # Keep only review tokens: from milestone 3, other nodes' LLM calls stream here too.
                if metadata["langgraph_node"] == "synthesize":
                    tokens.append(message_chunk.text)

    # Await first, then read .values: `await x.values` would read it from the coroutine.
    values = (await graph.aget_state(config)).values
    review = values["review"]

    assert values["sources"], "expected arXiv to return papers for this question"
    # Milestone 3: a real planner must actually decompose, and a real fan-out must run more
    # than one worker. With one subtopic this would silently degrade to a milestone 2 run.
    assert len(values["pending_subtopics"]) > 1, f"planner proposed: {values['pending_subtopics']}"
    assert sorted(values["explored_subtopics"]) == sorted(values["pending_subtopics"]), (
        f"some subtopics failed: {values['failed_subtopics']}"
    )
    assert len(tokens) > 1, "expected the review to stream as several chunks"
    assert "".join(tokens) == review
    assert CITATION_MARKER.search(review), "expected at least one [arXiv:<id>] citation"
    assert values["citation_violations"] == [], f"cited papers that weren't retrieved: {values['citation_violations']}"
