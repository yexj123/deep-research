"""Integration test: one real run through the whole graph, with OpenAI and arXiv (D-036, D-037).

Makes a real, paid OpenAI call and a real arXiv request. Deselected by default; run it with
`uv run pytest -m integration`. Also skipped when OPENAI_API_KEY isn't set.
"""

import os

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.config import ARXIV_TIMEOUT_SECONDS
from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import get_chat_model
from deep_research.agent.nodes.check_citations import CITATION_MARKER


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_real_run_streams_a_review_citing_only_retrieved_papers(
    checkpointer: InMemorySaver,
) -> None:
    """A real run streams the review in several chunks, the streamed text is exactly the
    review saved in the checkpoint, and every citation is a retrieved paper.

    A real model words its reply differently every run, so the test compares the
    run with itself instead of with fixed text.
    """
    config = _config("integration")
    tokens: list[str] = []

    # The caller owns the HTTP client and its timeout (D-049, D-052).
    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS) as http_client:
        graph = build_graph(get_chat_model, http_client, checkpointer)
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
    assert len(tokens) > 1, "expected the review to stream as several chunks"
    assert "".join(tokens) == review
    assert CITATION_MARKER.search(review), "expected at least one [arXiv:<id>] citation"
    assert values["citation_violations"] == [], f"cited papers that weren't retrieved: {values['citation_violations']}"
