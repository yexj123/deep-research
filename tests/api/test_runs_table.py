"""The runs table directly, for what the API can't reach (D-007, D-121).

Everything else about this table is covered through the routes, which is the right level.
The exception is the migration: a database written *before* D-121 exists only on a machine
that has been running this app for a while, and no route can create one. That is precisely
the case D-103 got wrong one table over -- `CREATE TABLE IF NOT EXISTS` does nothing to a
table that already exists, so a missing column stays missing and every read fails.
"""

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from deep_research.persistence.runs import (
    Run,
    get_run,
    init_runs_table,
    list_runs,
    list_session,
    list_session_heads,
    record_run,
)

# The schema exactly as it was before D-121 added the two session columns.
PRE_D121_SCHEMA = """
CREATE TABLE runs (
    thread_id  TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    provider   TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@pytest_asyncio.fixture
async def conn(tmp_path: Path):
    async with aiosqlite.connect(tmp_path / "runs.sqlite") as connection:
        yield connection


async def _heads(connection: aiosqlite.Connection) -> list[tuple[str, int]]:
    return [(h.run.question, h.turns) for h in await list_session_heads(connection)]


@pytest.mark.asyncio
async def test_a_pre_session_database_is_migrated_in_place(tmp_path: Path) -> None:
    """An existing database gains both columns, and its rows keep working (D-121).

    Without the ALTER TABLE in `init_runs_table`, the very first `get_run` on an upgraded
    install raises "no such column: parent_thread_id" -- on every run in the history.
    """
    path = tmp_path / "old.sqlite"
    async with aiosqlite.connect(path) as old:
        await old.execute(PRE_D121_SCHEMA)
        await old.execute(
            "INSERT INTO runs (thread_id, question, provider, created_at)"
            " VALUES ('t1', 'an old question', 'openai', '2026-09-01T10:00:00+00:00')"
        )
        await old.commit()

    async with aiosqlite.connect(path) as conn:
        await init_runs_table(conn)

        run = await get_run(conn, "t1")
        assert run is not None
        assert run.question == "an old question"
        assert run.parent_thread_id is None
        # Backfilled: a run that predates sessions was its own conversation, which is what
        # it was. Left at '' it would vanish from the sidebar, since no head would match it.
        assert run.session_id == "t1"
        assert await _heads(conn) == [("an old question", 1)]


@pytest.mark.asyncio
async def test_migrating_twice_changes_nothing(tmp_path: Path) -> None:
    """`init_runs_table` runs on every startup, so it has to be idempotent (D-121).

    An unguarded `ALTER TABLE ADD COLUMN` raises "duplicate column name" the second time,
    which would take the app down on its next restart rather than its first.
    """
    path = tmp_path / "twice.sqlite"
    async with aiosqlite.connect(path) as conn:
        await init_runs_table(conn)
        await record_run(conn, "t1", "a question", "openai")

    async with aiosqlite.connect(path) as conn:
        await init_runs_table(conn)
        run = await get_run(conn, "t1")
        assert run is not None and run.session_id == "t1"


@pytest.mark.asyncio
async def test_a_follow_up_joins_the_parents_session(conn) -> None:
    """`session_id` is inherited from the parent, not from the immediate predecessor.

    Turn 3 follows up on turn 2, but belongs to turn 1's session -- so a conversation stays
    one conversation however deep it goes (D-121).
    """
    await init_runs_table(conn)
    first = await record_run(conn, "t1", "first", "openai")
    second = await record_run(conn, "t2", "second", "openai", first)
    third = await record_run(conn, "t3", "third", "openai", second)

    assert second.session_id == "t1"
    assert third.session_id == "t1", "turn 3 belongs to turn 1's conversation, not turn 2's"
    assert third.parent_thread_id == "t2", "but it answers turn 2"
    assert [r.question for r in await list_session(conn, "t1")] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_only_the_first_turn_heads_a_session(conn) -> None:
    """A follow-up is not a sidebar entry (D-121).

    `WHERE thread_id = session_id` is the whole filter, which works because a first turn is
    its own session -- there is no nullable "is this a head" flag to get out of step.
    """
    await init_runs_table(conn)
    first = await record_run(conn, "t1", "first", "openai")
    await record_run(conn, "t2", "second", "openai", first)
    await record_run(conn, "t3", "unrelated", "openai")

    assert await _heads(conn) == [("unrelated", 1), ("first", 2)]
    assert len(await list_runs(conn)) == 3, "list_runs still sees every turn"


@pytest.mark.asyncio
async def test_a_session_is_ordered_by_its_newest_turn(conn) -> None:
    """Following up on an old conversation brings it back to the top (D-121).

    Ordering heads by their own `created_at` would bury the conversation you are working in
    under every question asked since it started.
    """
    await init_runs_table(conn)
    old = await record_run(conn, "t1", "older", "openai")
    await record_run(conn, "t2", "newer", "openai")
    await record_run(conn, "t3", "a follow-up to the older one", "openai", old)

    assert [q for q, _ in await _heads(conn)] == ["older", "newer"]


@pytest.mark.asyncio
async def test_is_follow_up_is_derived_not_stored(conn) -> None:
    """One source of truth for "is this a follow-up": the parent column (D-121).

    A stored boolean could disagree with `parent_thread_id`; a property cannot.
    """
    assert Run("t1", "q", "openai", "now").is_follow_up is False
    assert Run("t2", "q", "openai", "now", parent_thread_id="t1").is_follow_up is True
