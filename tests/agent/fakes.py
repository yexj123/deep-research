"""Test doubles and test data builders for agent tests."""

import json
from collections.abc import Mapping, Sequence
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


def plan_reply(*subtopics: str) -> str:
    """A planner reply in the JSON shape SubtopicPlan expects (D-070)."""
    return json.dumps({"subtopics": list(subtopics)})


def claim_report(*unsupported: int, judged: int = 1) -> str:
    """A claim-checker reply in the JSON shape ClaimReport expects (O-12, D-113).

    `judged` claims, with the 1-based indices in `unsupported` marked false. Defaults to one
    supported claim, which is what the default review produces: a single cited sentence.
    """
    return json.dumps(
        {
            "judgements": [
                {
                    "claim": index,
                    "supported": index not in unsupported,
                    "why": "the evidence does not state this" if index in unsupported else "",
                }
                for index in range(1, judged + 1)
            ]
        }
    )


# Two subtopics, so the default graph test actually fans out (D-067, D-069).
DEFAULT_PLAN_REPLY = plan_reply("attention mechanisms", "positional encoding")


def one_round_replies(
    review: str = DEFAULT_REPLY,
    plan: str = DEFAULT_PLAN_REPLY,
    claims: str | None = None,
) -> list[str]:
    """Scripted replies for a run that does one search round and then stops.

    Milestone 4 calls decompose a **second** time after any productive round: gap_check sees
    new papers and routes back (D-075), and only then does the planner re-propose the same
    subtopics, which the explored filter drops -- leaving pending empty so route_subtopics
    goes to synthesize (D-069). Since D-113 a fourth model is built for the claim checker,
    so the order is: plan, plan, review, claims.

    A test that scripts only [plan, review] gets the *review prose* handed to the second
    decompose, which fails SubtopicPlan validation -- and one that stops at [plan, plan,
    review] hands that same prose to the claim checker, which then reports itself as having
    failed rather than as having found nothing. Both are silent-looking failures, which is
    what this helper exists to prevent.
    """
    return [plan, plan, review, claims if claims is not None else claim_report()]


class NullLimiter:
    """A no-delay stand-in for ArxivRateLimiter (D-064), so tests don't wait 3 seconds.

    Counts entries, so a test can assert every arXiv request went through the limiter
    without depending on wall-clock timing.
    """

    def __init__(self) -> None:
        self.entered = 0
        self.max_concurrent = 0
        self._active = 0

    async def __aenter__(self) -> None:
        self.entered += 1
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)

    async def __aexit__(self, *exc: object) -> None:
        self._active -= 1


class RecordingFactory:
    """A ModelFactory for tests: returns a scripted fake model and records each provider it's asked for.

    `reply` scripts a single answer used for every model built. `replies` scripts them in the
    order the graph builds models -- at milestone 3 that is decompose (JSON) then synthesize
    (prose), which need different text. The last entry repeats if more models are built, so a
    test only has to script the calls it cares about.

    `models_built` counts every model ever built; the script position is tracked separately,
    so `restart()` can rewind the script without rewriting history. They were one counter
    until D-121 needed a test that runs the graph twice: the second run picked up the script
    where the first left off, so its decompose was handed the *claim checker's* JSON and died
    on SubtopicPlan validation -- a confusing failure a long way from its cause.
    """

    def __init__(self, reply: str = DEFAULT_REPLY, replies: Sequence[str] | None = None) -> None:
        self.replies: list[str] = list(replies) if replies is not None else [reply]
        if not self.replies:
            raise ValueError("RecordingFactory needs at least one reply")
        self.providers: list[ProviderType] = []
        self.models_built = 0
        self._position = 0

    @property
    def reply(self) -> str:
        """The first scripted reply, for tests written before `replies` existed."""
        return self.replies[0]

    def restart(self) -> None:
        """Rewind to the top of the script, for a test that runs the graph more than once.

        `models_built` and `providers` are deliberately left alone: they record what actually
        happened, and a test asserting "opening a run executes nothing" still needs the real
        total across every run.
        """
        self._position = 0

    def __call__(self, provider: ProviderType) -> BaseChatModel:
        self.providers.append(provider)
        reply = self.replies[min(self._position, len(self.replies) - 1)]
        self._position += 1
        self.models_built += 1
        # A fresh iterator per call. GenericFakeChatModel consumes one item per
        # invoke, and an exhausted iterator would fail the second call.
        return GenericFakeChatModel(messages=iter([reply]))


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


_FEED_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" \
xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/api/synthetic</id>
  <title>arXiv Query: synthetic</title>
  <updated>2026-09-21T00:00:00Z</updated>
{entries}
</feed>
"""

_ENTRY_TEMPLATE = """  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <title>Synthetic paper {arxiv_id}</title>
    <updated>2026-09-21T00:00:00Z</updated>
    <published>2026-09-21T00:00:00Z</published>
    <link href="https://arxiv.org/abs/{arxiv_id}v1" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/{arxiv_id}v1" rel="related" type="application/pdf" title="pdf"/>
    <summary>A synthetic abstract for {arxiv_id}, used where a test needs specific IDs.</summary>
    <author><name>A. Tester</name></author>
  </entry>"""


def make_arxiv_feed(*arxiv_ids: str) -> str:
    """A minimal valid Atom feed containing exactly these paper IDs.

    The saved fixtures are real arXiv responses and stay that way. This builder covers tests
    that need *specific* IDs -- a multi-round run, for instance, where each round has to
    return papers the previous round didn't, or gap_check stops at round 1 (D-075).
    """
    entries = "\n".join(_ENTRY_TEMPLATE.format(arxiv_id=i) for i in arxiv_ids)
    return _FEED_TEMPLATE.format(entries=entries)


def make_arxiv_router_stub(
    routes: Mapping[str, tuple[str, int]], fallback: tuple[str, int]
) -> ArxivStub:
    """An arXiv stub that answers differently per subtopic, for fan-out tests.

    `routes` maps a term to (body, status_code); the first term found in the request's
    search_query wins. Needed because parallel workers each search a different subtopic,
    so a single canned response can't express "one worker succeeds, another fails".
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        query = request.url.params.get("search_query", "")
        body, status_code = next(
            ((b, s) for term, (b, s) in routes.items() if term in query), fallback
        )
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
