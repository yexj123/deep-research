"""research_worker tests (D-072). No network: the arXiv client is an httpx.MockTransport stub.

The worker is called directly with a Send payload, so these pin its return contract and its
catch list without running the fan-out. What a worker returns *on failure* is as important as
what it returns on success: a failed subtopic must not be marked explored (D-018), or it can
never be retried.
"""

import httpx
import pytest

from deep_research.agent.nodes.research_worker import make_research_worker
from tests.agent.fakes import NullLimiter, load_arxiv_fixture, make_arxiv_stub


async def run_worker(body: str, status_code: int = 200, subtopic: str = "attention transformers"):
    """Call the worker against one canned arXiv response; returns (update, stub, limiter)."""
    stub = make_arxiv_stub(body, status_code=status_code)
    limiter = NullLimiter()
    async with stub.client:
        node = make_research_worker(stub.client, limiter)
        update = await node({"subtopic": subtopic})
    return update, stub, limiter


# ---- success path --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_search_returns_sources_and_marks_explored() -> None:
    """The real 3-paper response yields sources and marks the subtopic explored (D-018, D-072)."""
    update, _, _ = await run_worker(load_arxiv_fixture("search_ok.xml"))

    assert len(update["sources"]) == 3
    assert update["explored_subtopics"] == ["attention transformers"]
    assert "failed_subtopics" not in update


@pytest.mark.asyncio
async def test_seen_paper_ids_are_the_canonical_ids_of_what_was_found() -> None:
    """seen_paper_ids carries versionless IDs, matching Source.arxiv_id (D-044, D-072).

    The next round's overlap check (D-022, deferred by D-074) compares against this set, so a
    versioned ID here would never match and the check would silently never fire.
    """
    update, _, _ = await run_worker(load_arxiv_fixture("search_ok.xml"))
    assert update["seen_paper_ids"] == {s.arxiv_id for s in update["sources"]}
    assert all("v" not in i.rsplit("/", 1)[-1] for i in update["seen_paper_ids"])


@pytest.mark.asyncio
async def test_skipped_entries_is_this_workers_delta_not_a_total() -> None:
    """The worker reports only what it skipped (D-067).

    skipped_entries uses operator.add across parallel workers, so returning a running total
    would compound. search_one_invalid_id.xml has exactly one bad entry.
    """
    update, _, _ = await run_worker(load_arxiv_fixture("search_one_invalid_id.xml"))
    assert update["skipped_entries"] == 1
    assert len(update["sources"]) == 2


@pytest.mark.asyncio
async def test_zero_results_is_a_success() -> None:
    """An empty search marks the subtopic explored with no sources (D-021, D-072).

    "Nothing published on X" is a finding, not a failure -- treating it as failure would retry
    it and then silently drop it at the retry cap.
    """
    update, _, _ = await run_worker(load_arxiv_fixture("search_empty.xml"))

    assert update["sources"] == []
    assert update["explored_subtopics"] == ["attention transformers"]
    assert "failed_subtopics" not in update


@pytest.mark.asyncio
async def test_the_search_runs_inside_the_rate_limiter() -> None:
    """Every arXiv request is wrapped by the limiter (D-064).

    arXiv allows one request per 3 s AND one connection at a time, so the limiter must be
    held across the request, not merely consulted before it.
    """
    _, stub, limiter = await run_worker(load_arxiv_fixture("search_ok.xml"))
    assert limiter.entered == 1
    assert len(stub.requests) == 1


@pytest.mark.asyncio
async def test_the_subtopic_is_what_gets_searched_not_the_question() -> None:
    """The query is built from the Send payload's subtopic (D-051, D-072).

    A worker only sees its payload, so searching anything else means the fan-out is wired wrong.
    """
    _, stub, _ = await run_worker(load_arxiv_fixture("search_ok.xml"), subtopic="positional encoding")
    assert "all%3Apositional+AND+all%3Aencoding" in str(stub.requests[0].url)


# ---- failure path --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_transport_error_is_recorded_as_a_failed_subtopic() -> None:
    """Network failures become data, not an exception (D-019, D-072)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    limiter = NullLimiter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        update = await make_research_worker(client, limiter)(
            {"subtopic": "attention"}
        )

    assert update["failed_subtopics"] == ["attention"]
    assert "explored_subtopics" not in update, "a failed subtopic must stay unexplored (D-018)"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 500, 503])
async def test_429_and_5xx_are_recorded_as_failures(status_code: int) -> None:
    """Rate limiting and server errors are external failures worth retrying (D-065)."""
    update, _, _ = await run_worker("", status_code=status_code)
    assert update["failed_subtopics"] == ["attention transformers"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 404])
async def test_other_4xx_crashes_instead_of_being_recorded(status_code: int) -> None:
    """A 4xx means we sent something wrong, so it must not look like a flaky network (D-065).

    arXiv answers a malformed query with 400. Recording it would retry a deterministic bug
    twice and then drop it silently at the retry cap (D-020).
    """
    with pytest.raises(httpx.HTTPStatusError):
        await run_worker("", status_code=status_code)


@pytest.mark.asyncio
async def test_malformed_xml_is_recorded_as_a_failure() -> None:
    """A truncated feed is bad external data, not a bug (D-048, D-072)."""
    update, _, _ = await run_worker(load_arxiv_fixture("truncated.xml"))
    assert update["failed_subtopics"] == ["attention transformers"]


@pytest.mark.asyncio
async def test_the_arxiv_error_feed_is_recorded_as_a_failure() -> None:
    """ArxivAPIError joins the catch list at milestone 3, as D-053 promised (D-072).

    Served with HTTP 200 here: with arXiv's real 400 the status check fires first (D-065),
    which is why this path is otherwise unreachable in production.
    """
    update, _, _ = await run_worker(load_arxiv_fixture("error_feed.xml"))
    assert update["failed_subtopics"] == ["attention transformers"]


@pytest.mark.asyncio
async def test_a_failed_worker_reports_nothing_else() -> None:
    """A failure returns only failed_subtopics (D-072).

    Returning sources=[] or skipped_entries=0 alongside would be harmless today, but it
    blurs the success/failure contract the reducers depend on.
    """
    update, _, _ = await run_worker(load_arxiv_fixture("truncated.xml"))
    assert set(update) == {"failed_subtopics"}
