"""Shared fixtures for agent tests."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.persistence.checkpointer import build_serializer
from tests.agent.fakes import ArxivStub, RecordingFactory, load_arxiv_fixture, make_arxiv_stub


@pytest.fixture
def fake_factory() -> RecordingFactory:
    return RecordingFactory()


@pytest.fixture
def checkpointer() -> InMemorySaver:
    # Fresh for every test, so no checkpoint state leaks from one test into another.
    # The real serializer settings, so every graph test restores state the way production does.
    return InMemorySaver(serde=build_serializer())


@pytest_asyncio.fixture
async def arxiv_ok() -> AsyncIterator[ArxivStub]:
    """An arXiv client that answers every request with the real 3-paper response (search_ok.xml)."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        yield stub
