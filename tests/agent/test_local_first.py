"""Answering a subtopic from the corpus instead of arXiv (O-13, D-105).

**The decision these pin.** When the corpus covers a subtopic, the worker uses it and does
*not* call arXiv. O-13 originally said "augment rather than replace", and the measurements
overturned that:

- Two runs of the same question retrieve **~89% different papers** (mean Jaccard 11%, median
  8%, across 20 questions and 80 recordings).
- Those runs produce **statistically indistinguishable reviews** (D-094, D-095 -- every
  quality metric within 2 SE).

So paper *identity* does not drive quality; topical relevance does. Augmenting would keep the
network call, add papers that ranking truncates away at `SYNTHESIS_TOP_N`, and buy nothing
measurable. Skipping buys the call.

**`LOCAL_FIRST` is on by default since D-107**, which measured it: 65 arXiv requests down to
1, wall clock down 65%, and every quality metric inside 2 SE. These tests still patch it
explicitly in both directions, because a default is not a contract -- the off state has to
stay reachable so the arXiv-only arm remains reproducible from this same codebase (D-088,
D-091).

**What is deliberately NOT tested here: that local answers are as good.** That needs a paid
arm and a judge, not a unit test. Asserting it here would be the D-078 mistake.
"""

import sqlite3

import pytest

from deep_research.agent.config import MIN_LOCAL_PAPERS
from deep_research.agent.nodes import research_worker as worker_module
from deep_research.agent.nodes.research_worker import make_research_worker
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.persistence.corpus import SCHEMA, index_sources
from tests.agent.fakes import load_arxiv_fixture, make_arxiv_stub, make_source

SUBTOPIC = "tiling attention kernels"

# Papers that strongly match SUBTOPIC: each carries most of its terms, which is what the
# coverage test requires (D-104). Enough of them to clear MIN_LOCAL_PAPERS with margin.
COVERED = [
    make_source(
        arxiv_id=f"2411.{20000 + i:05d}",
        title=f"Tiling attention kernels, part {i}",
        summary="IO-aware tiling of attention kernels for exact attention.",
    )
    for i in range(MIN_LOCAL_PAPERS + 2)
]


@pytest.fixture
def local_first(monkeypatch):
    """Enable the feature in the module that reads it.

    `LOCAL_FIRST` is bound into `research_worker`'s namespace at import (the D-094 lesson
    about module constants), so patching `config` alone would change nothing.
    """
    monkeypatch.setattr(worker_module, "LOCAL_FIRST", True)


@pytest.fixture
def corpus():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    yield db
    db.close()


async def _run(corpus, subtopic=SUBTOPIC):
    """Run the worker against a stub that records whether arXiv was called at all."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    worker = make_research_worker(stub.client, ArxivRateLimiter(0.0), corpus)
    try:
        update = await worker({"subtopic": subtopic})
    finally:
        await stub.client.aclose()
    return update, stub.requests


@pytest.mark.asyncio
async def test_a_covered_subtopic_does_not_call_arxiv(local_first, corpus) -> None:
    """The whole point: the network call is what skipping buys (D-105).

    Asserted on the request log rather than on timing, so it cannot pass by being merely fast.
    """
    index_sources(corpus, COVERED)
    update, requests = await _run(corpus)

    assert requests == [], "a covered subtopic must not reach arXiv"
    assert update["explored_subtopics"] == [SUBTOPIC]
    assert len(update["sources"]) >= MIN_LOCAL_PAPERS


@pytest.mark.asyncio
async def test_an_uncovered_subtopic_still_calls_arxiv(local_first, corpus) -> None:
    """The fallback, and the control for the test above.

    Without this, a worker that never used the corpus at all would pass the covered case only
    by accident of the fixture.
    """
    index_sources(corpus, COVERED)
    _, requests = await _run(corpus, subtopic="crystallography of perovskite lattices")

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_a_thinly_covered_subtopic_still_calls_arxiv(local_first, corpus) -> None:
    """Below `MIN_LOCAL_PAPERS` is not coverage (D-104).

    The boundary in the direction that costs a network call rather than the direction that
    silently answers from too little -- a review built on two cached papers would look exactly
    like one built on twenty.
    """
    index_sources(corpus, COVERED[: MIN_LOCAL_PAPERS - 1])
    _, requests = await _run(corpus)

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_the_feature_can_be_turned_off(corpus, monkeypatch) -> None:
    """Disabled, behaviour is exactly the pre-D-107 default (D-091, D-088).

    The switch has to work in *both* directions: the arXiv-only arm is recorded by turning
    this off, and it is what keeps the 80 earlier recordings comparable.
    """
    monkeypatch.setattr(worker_module, "LOCAL_FIRST", False)
    index_sources(corpus, COVERED)
    update, requests = await _run(corpus)

    assert len(requests) == 1, "with LOCAL_FIRST off, arXiv must still be called"
    assert "local_subtopics" not in update


@pytest.mark.asyncio
async def test_an_empty_corpus_falls_through_to_arxiv(corpus) -> None:
    """Cold start: the default being on must not break a first run (D-107).

    An empty corpus covers nothing, so the feature cannot fire before it has evidence to fire
    on. This is what makes defaulting it on safe rather than optimistic.
    """
    update, requests = await _run(corpus)

    assert len(requests) == 1
    assert "local_subtopics" not in update
    assert update["sources"]


@pytest.mark.asyncio
async def test_a_local_answer_is_reported_not_hidden(local_first, corpus) -> None:
    """A cached answer must be visible to the reader (O-5, D-105).

    The papers are real and the subtopic is genuinely explored, so nothing here is a loss --
    but a reader judging how current a review is has to know part of it came from a corpus
    that may be months old. Silence is this project's recurring failure (D-062, D-069).
    """
    index_sources(corpus, COVERED)
    update, _ = await _run(corpus)

    assert update["local_subtopics"] == [SUBTOPIC]


@pytest.mark.asyncio
async def test_local_papers_are_validated_sources(local_first, corpus) -> None:
    """Corpus papers flow into the same state, ranking and citation check as fetched ones.

    A dict-shaped row would fail somewhere far from here -- in `rank_sources`, or in the
    checkpoint round trip (D-043, D-014).
    """
    index_sources(corpus, COVERED)
    update, _ = await _run(corpus)

    paper = update["sources"][0]
    assert paper.arxiv_id and paper.title and paper.summary
    assert paper.published.tzinfo is not None


@pytest.mark.asyncio
async def test_a_broken_corpus_falls_back_to_arxiv(local_first, corpus) -> None:
    """A read failure costs latency, not the subtopic (the D-101 rule, applied to reads).

    The remote path still works, so a locked or corrupt corpus must degrade to today's
    behaviour rather than failing research that arXiv could have answered.
    """
    index_sources(corpus, COVERED)
    corpus.close()

    update, requests = await _run(corpus)

    assert len(requests) == 1
    assert "failed_subtopics" not in update


@pytest.mark.asyncio
async def test_a_malformed_subtopic_fails_the_same_way_with_or_without_the_corpus(
    local_first, corpus
) -> None:
    """The local path must not accept subtopics the remote path rejects (D-073).

    `build_search_query` runs before the corpus check on purpose. If it ran after, a subtopic
    of pure stopwords would be answered locally while the same subtopic crashed the arXiv
    path -- two different contracts for one input.
    """
    index_sources(corpus, COVERED)
    with pytest.raises(ValueError):
        await _run(corpus, subtopic="what is the")


@pytest.mark.asyncio
async def test_retrieval_uses_abstracts_when_synthesis_will_not_read_full_text(
    local_first, corpus, monkeypatch
) -> None:
    """The harmful configuration must be unreachable, not merely unused (D-110, D-111).

    Selecting papers on body text the model never sees measured faithfulness 0.99 -> 0.96
    (-3.1 SE): papers were chosen for full-text relevance, handed to the model as abstracts
    that did not support the topic, and cited anyway. Coupling the tier to
    `EXCERPT_MAX_CHARS` is what stops that pairing existing.
    """
    monkeypatch.setattr(worker_module, "EXCERPT_MAX_CHARS", 0)
    index_sources(corpus, COVERED)
    # A paper whose FULL TEXT matches but whose abstract does not mention the query at all.
    corpus.execute(
        "INSERT INTO chunks (arxiv_id, tier, section, text) VALUES (?, 'full_text', 'M', ?)",
        (COVERED[0].arxiv_id, "sparse mixture routing entropy collapse"),
    )
    corpus.commit()

    _, requests = await _run(corpus, subtopic="sparse mixture routing")

    assert len(requests) == 1, (
        "with excerpts off, full-text-only matches must not count as local coverage"
    )


@pytest.mark.asyncio
async def test_excerpts_reach_the_model_when_enabled(local_first, corpus, monkeypatch) -> None:
    """The other half of the coupling: with excerpts on, the model reads what was matched.

    Without this the fix could pass by disabling full text everywhere, which is not the same
    as making retrieval and synthesis agree.
    """
    monkeypatch.setattr(worker_module, "EXCERPT_MAX_CHARS", 4000)
    index_sources(corpus, COVERED)
    for paper in COVERED:
        corpus.execute(
            "INSERT INTO chunks (arxiv_id, tier, section, text) VALUES (?, 'full_text', 'M', ?)",
            (paper.arxiv_id, "tiling attention kernels measured at 2.1x speedup"),
        )
    corpus.commit()

    update, requests = await _run(corpus)

    assert requests == []
    assert any("2.1x speedup" in paper.excerpt for paper in update["sources"]), (
        "the passage that matched must be what the model is shown"
    )


@pytest.mark.asyncio
async def test_a_local_answer_never_claims_to_have_searched_arxiv(local_first, corpus, monkeypatch) -> None:
    """The progress trail must not announce a call that never happens (D-118).

    Found by /demo-check: the trail read "Searching arXiv for X" followed immediately by
    "Answered X from 20 local paper(s)", and a reader would reasonably conclude both
    happened. Reporting work that was not done is the same defect as hiding work that was --
    it just flatters instead of alarming.
    """
    messages: list[dict] = []
    monkeypatch.setattr(
        "deep_research.agent.nodes.research_worker._progress_writer", lambda: messages.append
    )
    index_sources(corpus, COVERED)

    _, requests = await _run(corpus)

    assert requests == [], "precondition: this subtopic is answered locally"
    trail = " | ".join(m.get("status", "") for m in messages)
    assert "Searching arXiv" not in trail, trail
    assert "local paper(s)" in trail


@pytest.mark.asyncio
async def test_an_arxiv_search_is_still_announced(local_first, corpus, monkeypatch) -> None:
    """The control: when the call does happen, the trail must say so.

    Without this, deleting the message entirely would pass the test above while leaving a
    long network wait unexplained on screen.
    """
    messages: list[dict] = []
    monkeypatch.setattr(
        "deep_research.agent.nodes.research_worker._progress_writer", lambda: messages.append
    )
    index_sources(corpus, COVERED)

    await _run(corpus, subtopic="crystallography of perovskite lattices")

    assert any("Searching arXiv" in m.get("status", "") for m in messages)
