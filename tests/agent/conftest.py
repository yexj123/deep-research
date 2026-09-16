"""Shared fixtures for agent tests."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from tests.agent.fakes import RecordingFactory


@pytest.fixture
def fake_factory() -> RecordingFactory:
    return RecordingFactory()


@pytest.fixture
def checkpointer() -> InMemorySaver:
    # Fresh for every test, so no checkpoint state leaks from one test into another.
    return InMemorySaver()
