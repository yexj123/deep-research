"""The FastAPI app: owns the four dependencies for the process's lifetime (D-032, D-049, D-064).

Single-user, no auth (D-008). If this ever leaves localhost, gate it behind one password rather
than building a user system.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from deep_research.agent.config import ARXIV_MIN_INTERVAL_SECONDS, ARXIV_TIMEOUT_SECONDS
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import ModelFactory, get_chat_model
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.api.routes import pages, runs
from deep_research.persistence.checkpointer import open_checkpointer
from deep_research.persistence.runs import init_runs_table

DEFAULT_DB_PATH = str(Path("deep_research.sqlite").resolve())
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the graph once, with dependencies that live as long as the process.

    The graph's four dependencies are created here because this is what owns their lifetime
    (D-032): the HTTP client and the SQLite connection are context managers, and the rate
    limiter's semaphore binds to the first event loop that touches it (D-064) -- so it must be
    created inside the running loop, not at import time.
    """
    async with open_checkpointer(app.state.db_path) as (checkpointer, conn):
        await init_runs_table(conn)
        async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS) as http_client:
            app.state.conn = conn
            app.state.graph = build_graph(
                app.state.model_factory,
                app.state.http_client or http_client,
                app.state.limiter or ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS),
                checkpointer,
            )
            yield


def create_app(
    *,
    model_factory: ModelFactory = get_chat_model,
    db_path: str = DEFAULT_DB_PATH,
    http_client: httpx.AsyncClient | None = None,
    limiter: ArxivRateLimiter | None = None,
) -> FastAPI:
    """Build the app. The keyword arguments exist so tests can inject fakes.

    Same pattern as `build_graph` (D-032): every dependency is visible in the signature, and a
    test swaps in a fake model factory, an httpx.MockTransport client and a zero-delay limiter
    without monkeypatching anything.
    """
    app = FastAPI(title="Deep Research", lifespan=lifespan)
    app.state.model_factory = model_factory
    app.state.db_path = db_path
    app.state.http_client = http_client
    app.state.limiter = limiter
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(runs.router)
    app.include_router(pages.router)
    return app
