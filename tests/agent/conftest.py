"""Shared fixtures for agent tests."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.persistence.checkpointer import build_serializer
from tests.agent.fakes import (
    ArxivStub,
    NullLimiter,
    RecordingFactory,
    load_arxiv_fixture,
    make_arxiv_stub,
    one_round_replies,
)


@pytest.fixture
def fake_factory() -> RecordingFactory:
    """Scripts the three model calls a one-round milestone 4 run makes, in order.

    decompose needs JSON it can validate (D-070); synthesize needs prose with a citation
    marker (D-046). The second plan is what stops the recursion: gap_check routes back after
    a productive round, the planner re-proposes the same subtopics, and the explored filter
    drops them all (D-069, D-075). See one_round_replies.
    """
    return RecordingFactory(replies=one_round_replies())


@pytest.fixture
def checkpointer() -> InMemorySaver:
    # Fresh for every test, so no checkpoint state leaks from one test into another.
    # The real serializer settings, so every graph test restores state the way production does.
    return InMemorySaver(serde=build_serializer())


@pytest.fixture
def limiter() -> NullLimiter:
    """A zero-delay arXiv limiter (D-064): the real 3 s spacing would make the suite unusable."""
    return NullLimiter()


@pytest_asyncio.fixture
async def arxiv_ok() -> AsyncIterator[ArxivStub]:
    """An arXiv client that answers every request with the real 3-paper response (search_ok.xml)."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        yield stub
