"""Integration test: one real OpenAI run through the whole graph (D-036, D-037).

Makes a real, paid API call. Deselected by default; run it with
`uv run pytest -m integration`. Also skipped when OPENAI_API_KEY isn't set.
"""

import os

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import get_chat_model


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_real_openai_run_streams_and_saves_the_review(checkpointer: InMemorySaver) -> None:
    """A real run streams the review in several chunks, and the streamed text is
    exactly the review saved in the checkpoint.

    A real model words its reply differently every run, so the test compares the
    run with itself instead of with fixed text.
    """
    graph = build_graph(get_chat_model, checkpointer)
    config = _config("integration")

    tokens: list[str] = []
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
    review = (await graph.aget_state(config)).values["review"]

    assert len(tokens) > 1, "expected the review to stream as several chunks"
    assert review.strip(), "expected a non-empty review"
    assert "".join(tokens) == review
