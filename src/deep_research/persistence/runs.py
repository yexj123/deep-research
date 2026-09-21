"""The runs table: what question a thread_id was started with (D-007, D-081).

LangGraph's checkpoint holds the *state* of a run, but nothing records a run before it has
executed a single node. `POST /runs` returns a thread_id without running anything (D-081), so
the question has to live somewhere until the stream route asks for it.

Same SQLite file as the checkpoints (D-007): one database until there is a real reason for a
second. This table is also what the history route lists.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite

from deep_research.agent.context import ProviderType

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id  TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    provider   TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Run:
    """One recorded run. Not graph state -- see ResearchState for that."""

    thread_id: str
    question: str
    provider: ProviderType
    created_at: str


async def init_runs_table(conn: aiosqlite.Connection) -> None:
    """Create the runs table if it isn't there. Called once, in the app lifespan."""
    await conn.execute(CREATE_RUNS_TABLE)
    await conn.commit()


async def record_run(
    conn: aiosqlite.Connection, thread_id: str, question: str, provider: ProviderType
) -> Run:
    """Store a run before it executes, so the stream route can look up its question."""
    run = Run(
        thread_id=thread_id,
        question=question,
        provider=provider,
        created_at=datetime.now(UTC).isoformat(),
    )
    await conn.execute(
        "INSERT INTO runs (thread_id, question, provider, created_at) VALUES (?, ?, ?, ?)",
        (run.thread_id, run.question, run.provider, run.created_at),
    )
    await conn.commit()
    return run


async def get_run(conn: aiosqlite.Connection, thread_id: str) -> Run | None:
    """The recorded run, or None if this thread_id was never created."""
    async with conn.execute(
        "SELECT thread_id, question, provider, created_at FROM runs WHERE thread_id = ?",
        (thread_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return Run(*row) if row else None


async def list_runs(conn: aiosqlite.Connection, limit: int = 50) -> list[Run]:
    """Recent runs, newest first. The history list in the UI."""
    async with conn.execute(
        "SELECT thread_id, question, provider, created_at FROM runs "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [Run(*row) for row in rows]
