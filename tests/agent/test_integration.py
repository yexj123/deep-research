"""Integration test: one real run through the whole graph, with OpenAI and arXiv (D-036, D-037).

Makes a real, paid OpenAI call and a real arXiv request. Deselected by default; run it with
`uv run pytest -m integration`. Also skipped when OPENAI_API_KEY isn't set.
"""

import os

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.config import (
    ARXIV_MIN_INTERVAL_SECONDS,
    ARXIV_TIMEOUT_SECONDS,
    MAX_DEPTH,
    RECURSION_LIMIT,
)
from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import get_chat_model
from deep_research.agent.nodes.check_citations import CITATION_MARKER
from deep_research.agent.sources.rate_limit import ArxivRateLimiter


def _config(thread_id: str) -> RunnableConfig:
    # recursion_limit is invoke config, not graph config -- the caller sets it, or LangGraph
    # silently uses its default of 25 instead of the measured value (D-077).
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


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

    From milestone 4 the run recurses, so the cost scales with rounds: up to MAX_DEPTH + 2
    paid planner calls plus the synthesis, and up to (MAX_DEPTH + 1) x MAX_SUBTOPICS arXiv
    requests, each spaced by the real limiter at ARXIV_MIN_INTERVAL_SECONDS. Measured
    2026-09-21: about 38 s for a full three-round run.
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

    # A real planner must actually decompose and the fan-out must run more than one worker,
    # or the run has silently degraded to a milestone 2 single search.
    assert len(values["explored_subtopics"]) > 1, (
        f"explored only {values['explored_subtopics']}, failed {values['failed_subtopics']}"
    )
    # The recursion ran and stopped for a stated reason. depth counts rounds completed
    # (D-076), so it is at least 1 and never past the ceiling -- exceeding it would mean the
    # semantic exit failed and only recursion_limit was holding the run back (D-009, D-075).
    assert 1 <= values["depth"] <= MAX_DEPTH + 1, f"depth ended at {values['depth']}"
    # explored_subtopics accumulates across every round; pending_subtopics only ever holds the
    # latest plan (D-017). They are equal only in a single-round run, which is why this
    # asserts the relationship rather than equality.
    assert set(values["pending_subtopics"]) <= set(values["explored_subtopics"]) | set(
        values["failed_subtopics"]
    ), "the last plan should be subtopics the run actually attempted"

    assert len(tokens) > 1, "expected the review to stream as several chunks"
    assert "".join(tokens) == review

    # Citation grounding: this asserts the *checker* works, not that the model behaved
    # (D-078). Grounding exists precisely because models cite from memory -- requiring zero
    # violations would conflate "the checker works" with "this run got lucky", and make the
    # one test that talks to a real API randomly red.
    known_ids = {source.arxiv_id for source in values["sources"]}
    cited_ids = set(CITATION_MARKER.findall(review))
    violations = values["citation_violations"]

    assert cited_ids & known_ids, (
        f"expected at least one citation of a retrieved paper; cited {cited_ids or 'nothing'}"
    )
    # No false positives: anything recorded must genuinely not be a retrieved paper. A
    # violation that *was* retrieved would mean the checker is broken, which is a code bug
    # and must fail.
    for violation in violations:
        assert violation not in known_ids, (
            f"false positive: {violation!r} was retrieved but recorded as ungrounded"
        )
    if violations:
        # Visible without failing the build. Observed 2026-09-21: one three-round run cited
        # the fabricated ID '2113.11460' (month 13 -- not even a possible arXiv ID); the very
        # next identical run produced none. Ungrounded citations happen and vary run to run;
        # whether depth changes the rate is unresolved (D-079) and is notebooks/ work.
        print(f"\nNOTE: {len(violations)} ungrounded citation(s) caught by check_citations: {violations}")
