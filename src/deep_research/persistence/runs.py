"""The runs table: what question a thread_id was started with (D-007, D-081, D-121).

LangGraph's checkpoint holds the *state* of a run, but nothing records a run before it has
executed a single node. `POST /runs` returns a thread_id without running anything (D-081), so
the question has to live somewhere until the stream route asks for it.

Same SQLite file as the checkpoints (D-007): one database until there is a real reason for a
second. This table is also what the history route lists.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from deep_research.agent.context import ProviderType

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id  TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    provider   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- A follow-up gets its own thread and its own clean graph state (D-121). These two
    -- columns are the only thing tying turns together: `parent_thread_id` is what the
    -- follow-up was asked about, `session_id` groups a whole conversation for the UI.
    -- A first turn is its own session, so `session_id = thread_id`.
    parent_thread_id TEXT REFERENCES runs(thread_id),
    session_id TEXT NOT NULL DEFAULT '',
    -- The model the user typed, or '' for the provider's default (D-125). Stored rather than
    -- re-read from config, because `stream_run` resumes a run from its checkpoint days later:
    -- without this, resuming would silently finish a run on a different model than it started
    -- on, and the review would be a blend of two.
    model TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class Run:
    """One recorded run. Not graph state -- see ResearchState for that."""

    thread_id: str
    question: str
    provider: ProviderType
    created_at: str
    # None for a first question; the turn this one follows up on otherwise (D-121).
    parent_thread_id: str | None = None
    # The conversation this turn belongs to. Equals thread_id for a first question.
    session_id: str = ""
    # The model this run was started with, or None for the provider's default (D-125, D-126).
    model: str | None = None

    @property
    def is_follow_up(self) -> bool:
        return self.parent_thread_id is not None


def _run_from_row(row: tuple[Any, ...]) -> Run:
    """One row as a `Run`, restoring the absent model to `None` (D-126).

    SQLite stores `model` as `TEXT NOT NULL DEFAULT ''` -- the column shipped that way in
    D-125, and making it nullable would mean rebuilding the table for no behavioural gain.
    Python's absent value is `None`, so the two representations meet here, in the one place
    every query goes through, rather than being converted at four call sites that could drift.
    """
    thread_id, question, provider, created_at, parent_thread_id, session_id, model = row[:7]
    return Run(
        thread_id=thread_id,
        question=question,
        provider=provider,
        created_at=created_at,
        parent_thread_id=parent_thread_id,
        session_id=session_id,
        model=model or None,
    )


@dataclass(frozen=True)
class SessionHead:
    """A conversation as the sidebar sees it: its first question, and how far it got (D-121).

    Separate from `Run` because `turns` and `last_at` are aggregates over the session, not
    columns of the head run -- folding them into `Run` would put two fields on every row that
    are only ever populated by one query.
    """

    run: Run
    turns: int
    last_at: str


async def init_runs_table(conn: aiosqlite.Connection) -> None:
    """Create the runs table if it isn't there, and bring an older one up to date.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a database
    written before D-121 would be missing both columns and every read would fail. The same
    lesson as D-103, one table over.
    """
    await conn.execute(CREATE_RUNS_TABLE)
    async with conn.execute("PRAGMA table_info(runs)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}
    if "parent_thread_id" not in columns:
        await conn.execute("ALTER TABLE runs ADD COLUMN parent_thread_id TEXT")
    if "session_id" not in columns:
        await conn.execute("ALTER TABLE runs ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
        # Existing rows are each their own session, which is what they were.
        await conn.execute("UPDATE runs SET session_id = thread_id WHERE session_id = ''")
    if "model" not in columns:
        # No backfill: '' already means "the provider's default", which is exactly what every
        # run recorded before D-125 used. Writing a model name in would claim a choice the
        # user never made.
        await conn.execute("ALTER TABLE runs ADD COLUMN model TEXT NOT NULL DEFAULT ''")
    await conn.commit()


async def record_run(
    conn: aiosqlite.Connection,
    thread_id: str,
    question: str,
    provider: ProviderType,
    parent: Run | None = None,
    model: str | None = None,
) -> Run:
    """Store a run before it executes, so the stream route can look up its question.

    With a `parent`, this turn joins that turn's session. Without one it starts its own, so
    `session_id` is never empty and the UI needs no special case for a first question.
    """
    run = Run(
        thread_id=thread_id,
        question=question,
        provider=provider,
        created_at=datetime.now(UTC).isoformat(),
        parent_thread_id=parent.thread_id if parent else None,
        session_id=parent.session_id if parent else thread_id,
        # Normalized once, here, so "  " and "" and None all become the same absent value
        # rather than three variants the rest of the code has to keep distinguishing.
        model=(model or "").strip() or None,
    )
    await conn.execute(
        "INSERT INTO runs (thread_id, question, provider, created_at, parent_thread_id,"
        " session_id, model) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run.thread_id,
            run.question,
            run.provider,
            run.created_at,
            run.parent_thread_id,
            run.session_id,
            run.model or "",  # the column is NOT NULL; see _run_from_row
        ),
    )
    await conn.commit()
    return run


async def get_run(conn: aiosqlite.Connection, thread_id: str) -> Run | None:
    """The recorded run, or None if this thread_id was never created."""
    async with conn.execute(
        "SELECT thread_id, question, provider, created_at, parent_thread_id, session_id, model"
        " FROM runs WHERE thread_id = ?",
        (thread_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return _run_from_row(row) if row else None


async def list_runs(conn: aiosqlite.Connection, limit: int = 50) -> list[Run]:
    """Recent runs, newest first. The history list in the UI."""
    async with conn.execute(
        "SELECT thread_id, question, provider, created_at, parent_thread_id, session_id, model"
        " FROM runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [_run_from_row(row) for row in rows]


async def list_session_heads(conn: aiosqlite.Connection, limit: int = 50) -> list[SessionHead]:
    """One entry per conversation, most recently active first -- the sidebar (D-121).

    `list_runs` would list a follow-up as though it were an independent question, which is the
    confusion this feature exists to remove: the sidebar should list conversations, not turns.

    Ordered by the newest turn in each session rather than by when the session *started*, so
    following up on an old conversation brings it back to the top where you just left it.
    """
    async with conn.execute(
        "SELECT r.thread_id, r.question, r.provider, r.created_at, r.parent_thread_id,"
        "       r.session_id, r.model,"
        "       (SELECT COUNT(*) FROM runs t WHERE t.session_id = r.session_id),"
        "       (SELECT MAX(t.created_at) FROM runs t WHERE t.session_id = r.session_id)"
        " FROM runs r"
        # The head of a session is the turn that started it: `session_id` points at itself.
        " WHERE r.thread_id = r.session_id"
        # Ordinal 9 is the MAX(created_at) above. It moves whenever a column is added before
        # it -- as `model` just did -- so it is checked by test_a_session_is_ordered_by_its
        # _newest_turn rather than trusted.
        " ORDER BY 9 DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    # _run_from_row takes row[:7]; the two aggregates follow it.
    return [SessionHead(run=_run_from_row(row), turns=row[7], last_at=row[8]) for row in rows]


async def list_session(conn: aiosqlite.Connection, session_id: str) -> list[Run]:
    """Every turn of one conversation, oldest first -- the order they were asked in."""
    async with conn.execute(
        "SELECT thread_id, question, provider, created_at, parent_thread_id, session_id, model"
        " FROM runs WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [_run_from_row(row) for row in rows]
