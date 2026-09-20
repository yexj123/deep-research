"""Test doubles and test data builders for agent tests."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from deep_research.agent.context import ProviderType
from deep_research.agent.sources.models import Source

# Cites the first paper in search_ok.xml, so a graph run on `arxiv_ok` has no citation violations.
DEFAULT_REPLY = "Retrieval grounds a model's answer in papers it has just read [arXiv:2411.18583]."
ARXIV_FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"


class RecordingFactory:
    """A ModelFactory for tests: returns a scripted fake model and records each provider it's asked for."""

    def __init__(self, reply: str = DEFAULT_REPLY) -> None:
        self.reply = reply
        self.providers: list[ProviderType] = []

    def __call__(self, provider: ProviderType) -> BaseChatModel:
        self.providers.append(provider)
        # A fresh iterator per call. GenericFakeChatModel consumes one item per
        # invoke, and an exhausted iterator would fail the second call.
        return GenericFakeChatModel(messages=iter([self.reply]))


def load_arxiv_fixture(name: str) -> str:
    """Return a saved arXiv response from tests/agent/fixtures/arxiv (see the README there)."""
    return (ARXIV_FIXTURES / name).read_text(encoding="utf-8")


@dataclass
class ArxivStub:
    """An httpx.AsyncClient that answers every request with one saved response and records the requests."""

    client: httpx.AsyncClient
    requests: list[httpx.Request] = field(default_factory=list)


def make_arxiv_stub(body: str, status_code: int = 200) -> ArxivStub:
    """Build an ArxivStub. Close it with `async with stub.client:` (or `await stub.client.aclose()`)."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code,
            text=body,
            headers={"content-type": "application/atom+xml; charset=utf-8"},
        )

    return ArxivStub(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests=requests)


def make_source(**overrides: Any) -> Source:
    """A valid Source built from the first entry of search_ok.xml; override any field."""
    fields: dict[str, Any] = {
        "arxiv_id": "2411.18583",
        "version": 1,
        "title": "Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation",
        "authors": ("Nurshat Fateh Ali", "Md. Mahdi Mohtasim", "Shakil Mosharrof", "T. Gopi Krishna"),
        "summary": "This research presents and compares multiple approaches to automate the generation of literature reviews.",
        "published": datetime(2024, 11, 27, 18, 27, 7, tzinfo=UTC),
        "url": "https://arxiv.org/abs/2411.18583v1",
    }
    fields.update(overrides)
    return Source(**fields)
