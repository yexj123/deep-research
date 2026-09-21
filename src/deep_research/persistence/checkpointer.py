"""Checkpoint serialization settings and the app's checkpointer (D-014, D-050, D-082)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_research.agent.sources.models import Source

# Every custom type stored in state, as (module, class name). A type missing here comes back
# from a checkpoint as a plain dict, with only a logged warning (D-014 correction).
# Built from the class, not typed as strings, so moving or renaming Source can't leave a stale
# entry (D-057). It must match what the serializer records: (obj.__class__.__module__, obj.__class__.__name__).
ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    (Source.__module__, Source.__name__),
]


def build_serializer() -> JsonPlusSerializer:
    """The serializer every checkpointer must use: only allowlisted custom types are restored.

    Passing any allowlist, even an empty one, turns on strict mode for this serializer.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def open_checkpointer(db_path: str) -> AsyncIterator[tuple[AsyncSqliteSaver, aiosqlite.Connection]]:
    """Open the SQLite checkpointer for the app's lifetime (D-007, D-082).

    Built by hand rather than with `AsyncSqliteSaver.from_conn_string()`: that helper takes no
    `serde` argument -- its body is `cls(conn)` -- so it silently discards the allowlist above,
    which is the one thing D-014 exists to enforce. Without it a `Source` still restores today,
    but logs "Deserializing unregistered type ... will be blocked in a future version": not a
    break now, a break later, announced by a line that is easy to miss in server logs.

    Yields the connection too, because the same file holds the runs table (D-007): one database
    until there is a real reason for a second.
    """
    async with aiosqlite.connect(db_path) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=build_serializer())
        await checkpointer.setup()  # creates the checkpoint tables; required once per file
        yield checkpointer, conn
