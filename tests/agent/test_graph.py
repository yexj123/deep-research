"""Graph tests for milestone 2 (START -> intake -> search -> synthesize -> check_citations -> END).

No network: `fake_factory` returns GenericFakeChatModel, `arxiv_ok` serves the saved 3-paper
arXiv response, and `checkpointer` is a fresh InMemorySaver with the real serializer settings
for each test (see tests/agent/conftest.py).
"""

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.context import RunContext
from deep_research.agent.graph import build_graph
from deep_research.agent.nodes.synthesize import NO_SOURCES_REVIEW
from deep_research.agent.state import ResearchState
from tests.agent.fakes import ArxivStub, RecordingFactory, load_arxiv_fixture, make_arxiv_stub


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


# ---- Streaming ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_streams_token_by_token_from_synthesize(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """The reply arrives as several `messages` chunks, all from the synthesize node,
    and together they spell out the full reply."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)

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
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """`updates` chunks are keyed by node name and arrive in graph order."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)

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

    assert node_order == ["intake", "search", "synthesize", "check_citations"]


# ---- Search, citations and the empty case ---------------------------------------------


@pytest.mark.asyncio
async def test_search_sends_the_key_terms_of_the_question(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """The search node sends one request, built from the cleaned question (D-051)."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
    await graph.ainvoke(
        {"question": "  What is attention?  "},
        _config("search-query"),
        context=RunContext(provider="openai"),
        version="v2",
    )
    assert len(arxiv_ok.requests) == 1
    assert arxiv_ok.requests[0].url.params["search_query"] == "all:attention"


@pytest.mark.asyncio
async def test_run_records_the_retrieved_sources_and_no_violations(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """The fake reply cites a retrieved paper, so the run ends with no citation violations."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is retrieval augmented generation?"},
        _config("grounded"),
        context=RunContext(provider="openai"),
        version="v2",
    )
    assert [s.arxiv_id for s in output.value.sources] == ["2411.18583", "2502.00306", "2510.22344"]
    assert output.value.skipped_entries == 0
    assert output.value.citation_violations == []


@pytest.mark.asyncio
async def test_invented_citation_ends_up_in_state(
    arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """A reply citing a paper that wasn't retrieved is recorded by check_citations (D-046)."""
    factory = RecordingFactory(reply="Transformers rely on attention [arXiv:1706.03762].")
    graph = build_graph(factory, arxiv_ok.client, checkpointer)
    output = await graph.ainvoke(
        {"question": "What is attention?"},
        _config("invented"),
        context=RunContext(provider="openai"),
        version="v2",
    )
    assert output.value.citation_violations == ["1706.03762"]


@pytest.mark.asyncio
async def test_no_papers_skips_the_model(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """Zero results is a success (D-021): the review is the fixed message and no model is built."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_empty.xml"))
    async with stub.client:
        graph = build_graph(fake_factory, stub.client, checkpointer)
        output = await graph.ainvoke(
            {"question": "What is qzxwvkjhgfdsa?"},
            _config("no-papers"),
            context=RunContext(provider="openai"),
            version="v2",
        )
    assert output.value.sources == []
    assert output.value.review == NO_SOURCES_REVIEW
    assert output.value.citation_violations == []
    assert fake_factory.providers == []


@pytest.mark.asyncio
async def test_failed_search_fails_the_run(
    fake_factory: RecordingFactory, checkpointer: InMemorySaver
) -> None:
    """At milestone 2 the search node catches nothing, so an HTTP error ends the run (D-053)."""
    stub = make_arxiv_stub(load_arxiv_fixture("error_feed.xml"), status_code=400)
    async with stub.client:
        graph = build_graph(fake_factory, stub.client, checkpointer)
        with pytest.raises(httpx.HTTPStatusError):
            await graph.ainvoke(
                {"question": "What is attention?"},
                _config("failed-search"),
                context=RunContext(provider="openai"),
                version="v2",
            )
    assert fake_factory.providers == []


# ---- Final state and checkpoint ---------------------------------------------------


@pytest.mark.asyncio
async def test_final_state_is_a_research_state(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """With version="v2", `ainvoke(...).value` is a ResearchState holding the model's reply."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
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
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """The checkpoint saved under the run's thread_id holds the review.
    `aget_state(...).values` is a plain dict, not a ResearchState."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
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
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """The provider set in RunContext is the one the model factory receives (D-015)."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
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
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """A whitespace-only question is rejected by intake with ValueError, before any search."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "   "},
            _config("blank-question"),
            context=RunContext(provider="openai"),
            version="v2",
        )
    assert arxiv_ok.requests == []


@pytest.mark.asyncio
async def test_missing_context_raises(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """Invoking without context= is rejected by intake with ValueError, instead of
    failing later with AttributeError."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "What is attention?"},
            _config("missing-context"),
            version="v2",
        )


@pytest.mark.asyncio
async def test_invalid_provider_raises(
    fake_factory: RecordingFactory, arxiv_ok: ArxivStub, checkpointer: InMemorySaver
) -> None:
    """A provider outside ProviderType is rejected by intake with ValueError.
    RunContext doesn't check its Literal type at runtime, so the check lives in intake."""
    graph = build_graph(fake_factory, arxiv_ok.client, checkpointer)
    with pytest.raises(ValueError, match="intake:"):
        await graph.ainvoke(
            {"question": "Who published 'Attention Is All You Need'?"},
            _config("invalid-provider"),
            context=RunContext(provider="gemini"),  # type: ignore[arg-type]
            version="v2",
        )
