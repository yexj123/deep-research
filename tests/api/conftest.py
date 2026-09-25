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
    make_arxiv_feed,
    make_arxiv_router_stub,
    plan_reply,
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
            # Attached so a test can replace a lifespan-built dependency -- the follow-up
            # rewriter, say (D-121). Reaching `client._transport.app` would work and would
            # couple the tests to httpx internals.
            client.app = app
            yield client, factory


@pytest_asyncio.fixture
async def api_with(tmp_path: Path):
    """Build an app whose arXiv stub can fail or return nothing, for coverage tests (O-5).

    A factory rather than a fixture per scenario: what varies is one arXiv response, and
    spelling that out at the call site keeps each test's setup visible in the test.
    """
    created: list[httpx.AsyncClient] = []

    async def build(*, alpha_status: int = 200, empty: bool = False):
        # `empty`: every search returns a valid feed with no entries, so both subtopics are
        # explored-but-empty (D-021). `alpha_status`: only "alpha topic" fails, so the run
        # has one failure and one success -- the mixed case a report has to be honest about.
        papers = make_arxiv_feed() if empty else load_arxiv_fixture("search_ok.xml")
        routes = {} if alpha_status == 200 else {"alpha": ("", alpha_status)}
        stub = make_arxiv_router_stub(routes, fallback=(papers, 200))
        created.append(stub.client)
        factory = RecordingFactory(
            replies=one_round_replies(plan=plan_reply("alpha topic", "beta topic"))
        )
        app = create_app(
            model_factory=factory,
            db_path=str(tmp_path / f"cov{len(created)}.sqlite"),
            http_client=stub.client,
            limiter=NullLimiter(),
        )
        manager = LifespanManager(app)
        await manager.__aenter__()
        managers.append(manager)
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        created.append(client)
        return client, factory

    managers: list[LifespanManager] = []
    yield build
    for manager in managers:
        await manager.__aexit__(None, None, None)
    for client in created:
        await client.aclose()
