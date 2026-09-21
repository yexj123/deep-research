"""Fixtures for the API tests: a real app over a temp SQLite file, with fake agent dependencies.

The app is exercised through `httpx.ASGITransport`, so there is no server, no port and no
network -- but the routes, the lifespan, the real graph and a real SQLite checkpoint file all
run for real. Only the model and arXiv are faked.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio
from asgi_lifespan import LifespanManager

from deep_research.api.main import create_app
from tests.agent.fakes import (
    ArxivStub,
    NullLimiter,
    RecordingFactory,
    load_arxiv_fixture,
    make_arxiv_stub,
    one_round_replies,
)


@pytest_asyncio.fixture
async def arxiv_stub() -> AsyncIterator[ArxivStub]:
    """An arXiv client answering every request with the saved 3-paper response."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        yield stub


@pytest_asyncio.fixture
async def api(
    tmp_path: Path, arxiv_stub: ArxivStub
) -> AsyncIterator[tuple[httpx.AsyncClient, RecordingFactory]]:
    """An app on a fresh SQLite file, plus the factory so tests can inspect model calls.

    LifespanManager runs startup/shutdown: without it app.state.graph never exists, because
    the graph is built in the lifespan (D-032).
    """
    factory = RecordingFactory(replies=one_round_replies())
    app = create_app(
        model_factory=factory,
        db_path=str(tmp_path / "test.sqlite"),
        http_client=arxiv_stub.client,
        limiter=NullLimiter(),
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, factory
