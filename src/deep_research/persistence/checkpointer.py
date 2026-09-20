"""Checkpoint serialization settings, shared by tests and the web layer (D-014, D-050)."""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

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
