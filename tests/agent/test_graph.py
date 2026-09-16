"""Graph tests for milestone 1 (START -> intake -> synthesize -> END).

No network: `fake_factory` returns GenericFakeChatModel, and `checkpointer` is a
fresh InMemorySaver for each test (see tests/agent/conftest.py).
"""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.state import ResearchState
from tests.agent.fakes import RecordingFactory


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


# ---- Streaming ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_streams_token_by_token_from_synthesize(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """The reply arrives as several `messages` chunks, all from the synthesize node,
    and together they spell out the full reply."""
    graph = build_graph(fake_factory, checkpointer)

    tokens: list[str] = []
    async for chunk in graph.astream(
        {"question": "What is attention?"},
        _config("stream"),
        context=RunContext(provider="openai"),
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            message_chunk, metadata = chunk["data"]
            assert metadata["langgraph_node"] == "synthesize"
            tokens.append(message_chunk.content)

    assert len(tokens) > 1, "expected the reply to stream as several chunks"
    assert "".join(tokens) == fake_factory.reply


@pytest.mark.asyncio
async def test_updates_arrive_in_node_order(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """`updates` chunks are keyed by node name and arrive in graph order:
    intake first, then synthesize."""
    graph = build_graph(fake_factory, checkpointer)

    node_order: list[str] = []
    async for chunk in graph.astream(
        {"question": "What is attention?"},
        _config("updates"),
        context=RunContext(provider="openai"),
        stream_mode=["updates"],
        version="v2",
    ):
        if chunk["type"] == "updates":
            node_order.extend(chunk["data"].keys())

    assert node_order == ["intake", "synthesize"]


# ---- Final state and checkpoint ---------------------------------------------------


@pytest.mark.asyncio
async def test_final_state_is_a_research_state(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """With version="v2", `ainvoke(...).value` is a ResearchState holding the model's reply."""
    graph = build_graph(fake_factory, checkpointer)
    output = await graph.ainvoke(
        {"question": "Who published 'Attention Is All You Need'?"},
        _config("final-state"),
        context=RunContext(provider="openai"),
        version="v2",
    )
    assert isinstance(output.value, ResearchState)
    assert output.value.review == fake_factory.reply


@pytest.mark.asyncio
async def test_checkpoint_contains_the_review(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """The checkpoint saved under the run's thread_id holds the review.
    `aget_state(...).values` is a plain dict, not a ResearchState."""
    graph = build_graph(fake_factory, checkpointer)
    config = _config("checkpoint")
    await graph.ainvoke(
        {"question": "What is attention?"},
        config,
        context=RunContext(provider="openai"),
        version="v2",
    )
    snapshot = await graph.aget_state(config)
    assert isinstance(snapshot.values, dict)
    assert snapshot.values["review"] == fake_factory.reply


@pytest.mark.asyncio
async def test_provider_from_context_reaches_the_factory(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """The provider set in RunContext is the one the model factory receives (D-015)."""
    graph = build_graph(fake_factory, checkpointer)
    await graph.ainvoke(
        {"question": "What is attention?"},
        _config("provider"),
        context=RunContext(provider="deepseek"),
        version="v2",
    )
    assert fake_factory.providers == ["deepseek"]


# ---- Input and context validation in intake (D-033) -------------------------------


@pytest.mark.asyncio
async def test_blank_question_raises(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """A whitespace-only question is rejected by intake with ValueError."""
    graph = build_graph(fake_factory, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "   "},
            _config("blank-question"),
            context=RunContext(provider="openai"),
            version="v2",
        )


@pytest.mark.asyncio
async def test_missing_context_raises(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """Invoking without context= is rejected by intake with ValueError, instead of
    failing later with AttributeError."""
    graph = build_graph(fake_factory, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "What is attention?"},
            _config("missing-context"),
            version="v2",
        )


@pytest.mark.asyncio
async def test_invalid_provider_raises(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """A provider outside ProviderType is rejected by intake with ValueError.
    RunContext doesn't check its Literal type at runtime, so the check lives in intake."""
    graph = build_graph(fake_factory, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "Who published 'Attention Is All You Need'?"},
            _config("invalid-provider"),
            context=RunContext(provider="gemini"),  # type: ignore[arg-type]
            version="v2",
        )
