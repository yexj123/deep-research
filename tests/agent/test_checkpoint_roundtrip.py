"""Checkpoint round-trip test for custom state types (D-014, D-050)."""

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.sources.models import Source
from deep_research.persistence.checkpointer import build_serializer
from tests.agent.fakes import (
    NullLimiter,
    RecordingFactory,
    load_arxiv_fixture,
    make_arxiv_stub,
    one_round_replies,
)


async def _first_restored_source(serde: JsonPlusSerializer) -> Any:
    """Run the graph on search_ok.xml, then load the first source back from the checkpoint."""
    checkpointer = InMemorySaver(serde=serde)
    config: RunnableConfig = {"configurable": {"thread_id": "round-trip"}}
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    factory = RecordingFactory(replies=one_round_replies())
    async with stub.client:
        graph = build_graph(factory, stub.client, NullLimiter(), checkpointer)
        await graph.ainvoke(
            {"question": "What is retrieval augmented generation?"},
            config,
            context=RunContext(provider="openai"),
            version="v2",
        )
    return (await graph.aget_state(config)).values["sources"][0]


@pytest.mark.asyncio
async def test_source_survives_a_checkpoint_round_trip() -> None:
    """A Source saved in a checkpoint comes back as a Source, with a tuple `authors`.

    Without an allowlist entry it comes back as a plain dict with only a warning (D-014),
    and this test is the only thing that would catch that.
    """
    restored = await _first_restored_source(build_serializer())

    assert isinstance(restored, Source)
    assert isinstance(restored.authors, tuple)


@pytest.mark.asyncio
async def test_without_the_allowlist_entry_a_source_comes_back_as_a_dict() -> None:
    """The control for the test above: it proves InMemorySaver really serializes, so the
    round trip above would fail if Source were missing from the allowlist (D-014 correction)."""
    restored = await _first_restored_source(JsonPlusSerializer(allowed_msgpack_modules=[]))

    assert isinstance(restored, dict)
