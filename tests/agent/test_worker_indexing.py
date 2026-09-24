"""The worker indexes what it searches, and a corpus failure never costs a subtopic (D-101).

Indexing is **free**: these abstracts were already fetched and paid for, and without this they
are used once and discarded. That is what makes the corpus useful from the first run instead
of after a bulk-download phase.

The tests that matter most here are the ones about *not* breaking things:

- A corpus write failing must not mark a successful subtopic failed. The search already
  succeeded, the papers are in hand, and the run completes fine without ever touching the
  corpus -- so a locked database must not burn one of that subtopic's two retries (D-020).
- But the failure must be **reported**, not swallowed. A corpus that silently stops filling is
  this project's recurring shape (D-062, D-069, O-5): nothing looks wrong, and months later
  the local tier is mysteriously empty.
- And a *bug* must still crash. The catch is `sqlite3.Error` only, so a malformed `Source`
  raising TypeError propagates (D-023).
"""

import sqlite3

import pytest

from deep_research.agent.nodes.research_worker import make_research_worker
from deep_research.agent.sources.arxiv import build_fts_query
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.persistence.corpus import SCHEMA, covering_papers, stats
from tests.agent.fakes import load_arxiv_fixture, make_arxiv_stub


@pytest.fixture
def corpus():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    yield db
    db.close()


async def _run_worker(corpus, subtopic: str = "attention mechanisms", body: str | None = None):
    stub = make_arxiv_stub(body if body is not None else load_arxiv_fixture("search_ok.xml"))
    worker = make_research_worker(stub.client, ArxivRateLimiter(0.0), corpus)
    try:
        return await worker({"subtopic": subtopic, "seen_paper_ids": set()})
    finally:
        await stub.client.aclose()


@pytest.mark.asyncio
async def test_a_search_seeds_the_corpus(corpus) -> None:
    """Every result is indexed on the way through, at no extra network cost (O-13)."""
    update = await _run_worker(corpus)

    assert update["sources"], "fixture should return papers"
    assert stats(corpus)["papers"] == len(update["sources"])


@pytest.mark.asyncio
async def test_indexed_papers_are_findable_afterwards(corpus) -> None:
    """Indexing is only worth anything if retrieval then works end to end.

    Asserting the row count alone would pass against a corpus whose FTS triggers never fired,
    which is precisely the bug external-content FTS5 invites.
    """
    update = await _run_worker(corpus)
    first = update["sources"][0]

    found = covering_papers(corpus, build_fts_query(first.title), limit=10)
    assert first.arxiv_id in found


@pytest.mark.asyncio
async def test_two_subtopics_finding_the_same_papers_index_them_once(corpus) -> None:
    """Parallel workers share one corpus, and overlap between subtopics is normal (D-022).

    Without idempotence the same abstract is indexed once per subtopic that found it, and BM25
    would rank a paper highly for having been *found often* rather than for matching.
    """
    await _run_worker(corpus, "attention mechanisms")
    after_first = stats(corpus)["papers"]
    await _run_worker(corpus, "positional encoding")

    assert stats(corpus)["papers"] == after_first


@pytest.mark.asyncio
async def test_a_worker_without_a_corpus_behaves_exactly_as_before(corpus) -> None:
    """`corpus=None` is the switchable baseline arm, not an accident (D-091, D-088).

    The eval recorder passes nothing, which is what keeps the 80 committed recordings
    comparable to anything recorded after the corpus lands.
    """
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    worker = make_research_worker(stub.client, ArxivRateLimiter(0.0))
    try:
        update = await worker({"subtopic": "attention mechanisms", "seen_paper_ids": set()})
    finally:
        await stub.client.aclose()

    assert update["sources"]
    assert stats(corpus)["papers"] == 0, "no corpus was passed, so nothing may be written"


@pytest.mark.asyncio
async def test_a_broken_corpus_does_not_fail_the_subtopic(corpus) -> None:
    """The search succeeded; a storage problem must not undo that (D-020, D-023).

    Marking the subtopic failed would burn one of its two retries and remove it from the
    review, turning a disk problem into missing research.
    """
    corpus.close()  # every later write raises ProgrammingError, a sqlite3.Error

    update = await _run_worker(corpus)

    assert update["explored_subtopics"] == ["attention mechanisms"]
    assert "failed_subtopics" not in update
    assert update["sources"], "the papers the search found must still reach the run"


@pytest.mark.asyncio
async def test_a_corpus_failure_is_reported_on_the_progress_stream(corpus, monkeypatch) -> None:
    """A corpus that quietly stops filling is the exact failure O-5 exists to prevent.

    Reported through `custom` progress rather than as state: it is an infrastructure problem,
    not a research finding, and it must not change what the run contributes.
    """
    messages: list[dict] = []
    monkeypatch.setattr(
        "deep_research.agent.nodes.research_worker._progress_writer",
        lambda: messages.append,
    )
    corpus.close()

    await _run_worker(corpus)

    assert any("Could not index" in m.get("status", "") for m in messages), messages


@pytest.mark.asyncio
async def test_a_programming_error_in_indexing_still_crashes(corpus, monkeypatch) -> None:
    """The catch is narrow on purpose: bugs must be loud (D-023).

    A blanket `except Exception` here would turn a malformed `Source` into a progress message
    and hide it for as long as the corpus kept "working".
    """

    def boom(*_args, **_kwargs):
        raise TypeError("malformed source")

    monkeypatch.setattr("deep_research.agent.nodes.research_worker.index_sources", boom)

    with pytest.raises(TypeError, match="malformed source"):
        await _run_worker(corpus)


@pytest.mark.asyncio
async def test_a_failed_search_indexes_nothing(corpus) -> None:
    """Nothing was retrieved, so there is nothing to store -- and no empty rows to explain."""
    stub = make_arxiv_stub("not xml at all")
    worker = make_research_worker(stub.client, ArxivRateLimiter(0.0), corpus)
    try:
        update = await worker({"subtopic": "attention mechanisms", "seen_paper_ids": set()})
    finally:
        await stub.client.aclose()

    assert update == {"failed_subtopics": ["attention mechanisms"]}
    assert stats(corpus)["papers"] == 0
