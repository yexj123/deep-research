"""research_worker node: one arXiv search per subtopic, run in parallel via Send (D-072)."""

import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any, TypedDict
from xml.etree.ElementTree import ParseError

import httpx
import pydantic
from defusedxml import DefusedXmlException
from langgraph.config import get_stream_writer

from deep_research.agent.config import (
    ARXIV_MAX_RESULTS,
    EXCERPT_MAX_CHARS,
    LOCAL_FIRST,
    LOCAL_SEARCH_TOP_K,
    MIN_LOCAL_PAPERS,
)
from deep_research.agent.sources.arxiv import ArxivAPIError, build_search_query, search_arxiv
from deep_research.agent.sources.models import Source
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.persistence.corpus import (
    ABSTRACT,
    index_sources,
    matching_excerpts,
    load_sources,
    strongly_matching_papers,
)


class SubtopicTask(TypedDict):
    """The Send payload. A TypedDict, because Send payloads are checkpointed too (D-071).

    **`seen_paper_ids` used to travel here and no longer does (D-114).** It existed for
    D-022's paper-overlap rule -- skip a subtopic when >=60% of its results are already seen --
    which was specified, plumbed, and never implemented. D-094 then measured why it could not
    work: rounds are near-disjoint, with `attention` round 2 finding 30 papers of which **27
    were new**, a 10% overlap against a 60% threshold. The rule would have fired approximately
    never, and the set was being checkpointed on every fan-out to feed it.
    """

    subtopic: str


WorkerNode = Callable[[SubtopicTask], Awaitable[dict[str, Any]]]

# Bad external data, not bugs: each becomes a failed subtopic (D-048, D-053, D-072).
EXTERNAL_FAILURES = (
    httpx.TransportError,
    pydantic.ValidationError,
    ParseError,
    DefusedXmlException,
    ArxivAPIError,
)


def _progress_writer() -> Callable[[dict[str, Any]], None]:
    """The `custom` stream writer, or a no-op when there isn't one (O-5).

    `get_stream_writer()` raises RuntimeError outside a runnable context -- confirmed
    2026-09-21: "Called get_config outside of a runnable context". Two reasons this falls back
    rather than propagating:

    - **Progress must never be able to break a run.** A node whose research fails because its
      telemetry could not initialize is badly designed. Reporting is strictly additive.
    - It keeps the worker callable directly, so its contract stays testable without graph
      scaffolding -- which is what test_research_worker.py exists to do.

    The only cause of that RuntimeError is "not running inside a graph", so nothing diagnostic
    is being swallowed.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None


def _is_external_status(exc: httpx.HTTPStatusError) -> bool:
    """429 and 5xx are arXiv's problem; every other 4xx is ours (D-065)."""
    status = exc.response.status_code
    return status == 429 or status >= 500


def _index_into_corpus(
    corpus: sqlite3.Connection | None,
    sources: Sequence[Source],
    subtopic: str,
    writer: Callable[[dict[str, Any]], None],
) -> None:
    """Index this search's abstracts, if a corpus was supplied (O-13, D-101).

    **Free.** These abstracts were already fetched and paid for; without this they are used
    once and discarded. Indexing them is what makes the corpus useful from the first run
    rather than after a bulk-download phase.

    **Why a corpus write must not fail a subtopic**, and why the catch is narrow. The search
    already succeeded: the papers are in hand and the run can complete without ever touching
    the corpus, so a locked or full database is not a reason to mark the subtopic failed and
    burn one of its two retries (D-020). That is the same reasoning as `_progress_writer` --
    storage is additive, like reporting. But the catch is `sqlite3.Error` only: a `TypeError`
    from a malformed `Source` is a *bug* and must crash loudly (D-023).

    **The failure is reported, not swallowed.** A corpus that silently stops filling is this
    project's recurring shape (D-062, D-069, O-5): everything looks fine, and months later the
    local tier is empty for no visible reason.

    **Synchronous on purpose, and that is what makes it safe here.** Workers run in parallel
    under `Send`, sharing one connection. Because `index_sources` contains no `await`, the
    event loop cannot interleave two of them, so the writes are effectively atomic. Making
    this async would introduce exactly the interleaving it currently cannot have.
    """
    if corpus is None or not sources:
        return
    try:
        index_sources(corpus, sources)
    except sqlite3.Error as exc:
        writer({"status": f"Could not index results for {subtopic!r} ({exc})"})


def _local_answer(
    corpus: sqlite3.Connection | None, subtopic: str
) -> list[Source]:
    """Papers already held locally for this subtopic, or `[]` to go to arXiv (O-13, D-105).

    **Covered means term coverage, not paper count** (D-104): at least `MIN_LOCAL_PAPERS`
    papers in the top-k that match half the subtopic's terms. The plain count O-13 originally
    specified returns k for every query in every corpus and would make this always fire.

    **Why skipping the search rather than augmenting it.** Measured across 80 recordings, two
    runs of the same question retrieve ~89% different papers (mean Jaccard 11%), and those
    runs produce statistically indistinguishable reviews (D-094, D-095). Paper identity does
    not drive quality; topical relevance does. Augmenting would keep the network call, add
    papers that ranking truncates away at `SYNTHESIS_TOP_N`, and buy nothing measurable.

    A read failure returns `[]` rather than raising: the arXiv path still works, so a broken
    corpus should cost latency, not the subtopic (the D-101 rule, applied to reads).
    """
    if corpus is None or not LOCAL_FIRST:
        return []
    try:
        # **Retrieval must not select on text synthesis will never read** (D-110, D-111).
        # With excerpts off, restricting to the abstract tier is what makes the harmful
        # configuration unreachable rather than merely unused: full-text-informed retrieval
        # feeding abstracts to the model measured faithfulness 0.99 -> 0.96 (-3.1 SE),
        # because papers were chosen for body text the model never saw and cited anyway.
        tier = None if EXCERPT_MAX_CHARS else ABSTRACT
        ids = strongly_matching_papers(corpus, subtopic, LOCAL_SEARCH_TOP_K, tier)
        if len(ids) < MIN_LOCAL_PAPERS:
            return []
        papers = load_sources(corpus, ids)
        if not EXCERPT_MAX_CHARS:
            return papers
        # Attach the full-text passages that matched, so the model reads the paper rather than
        # only its abstract (D-110). Papers the corpus has not read keep an empty excerpt and
        # fall back to their abstract in `format_papers`.
        excerpts = matching_excerpts(corpus, subtopic, ids, EXCERPT_MAX_CHARS)
        return [
            replace(paper, excerpt=excerpts[paper.arxiv_id])
            if paper.arxiv_id in excerpts
            else paper
            for paper in papers
        ]
    except sqlite3.Error:
        return []


def make_research_worker(
    http_client: httpx.AsyncClient,
    limiter: ArxivRateLimiter,
    corpus: sqlite3.Connection | None = None,
) -> WorkerNode:
    """Build the worker with its client, limiter and corpus in a closure (D-032, D-049, D-064).

    `corpus=None` disables indexing entirely, which keeps today's behaviour reproducible from
    this same codebase rather than from git history -- the switchable-parameter pattern D-091
    used for `SYNTHESIS_TOP_N`, and what lets a no-corpus arm be recorded against a corpus arm
    with one variable changing (D-088).
    """

    async def research_worker(task: SubtopicTask) -> dict[str, Any]:
        subtopic = task["subtopic"]
        # Live progress for a long run (O-5). Derived from the same facts as the state
        # update below, so the two cannot drift: one place decides what happened.
        writer = _progress_writer()
        writer({"status": f"Searching arXiv for {subtopic!r}"})
        # Outside the try on purpose: decompose guarantees the subtopic is searchable (D-073),
        # so a ValueError here is a bug and must crash rather than burn the retry cap. Built
        # before the corpus check so a malformed subtopic fails the same way either way --
        # otherwise the local path would quietly accept subtopics the remote path rejects.
        query = build_search_query(subtopic)

        local = _local_answer(corpus, subtopic)
        if local:
            writer({"status": f"Answered {subtopic!r} from {len(local)} local paper(s)"})
            return {
                "sources": local,
                "explored_subtopics": [subtopic],  # genuinely explored, just not over HTTP
                "seen_paper_ids": {source.arxiv_id for source in local},
                "local_subtopics": [subtopic],  # reported, never hidden (D-105)
            }

        try:
            async with limiter:  # held across the request, not just its start (D-064)
                result = await search_arxiv(http_client, query, ARXIV_MAX_RESULTS)
        except httpx.HTTPStatusError as exc:
            if not _is_external_status(exc):
                raise
            writer({"status": f"Search failed for {subtopic!r}"})
            return {"failed_subtopics": [subtopic]}
        except EXTERNAL_FAILURES:
            writer({"status": f"Search failed for {subtopic!r}"})
            return {"failed_subtopics": [subtopic]}

        found = len(result.sources)
        writer(
            {
                "status": f"Found {found} paper(s) for {subtopic!r}"
                if found
                else f"No papers found for {subtopic!r}"
            }
        )

        # After the progress report and before the state update: the papers are already in
        # hand, so nothing here can change what this subtopic contributes to the run.
        _index_into_corpus(corpus, result.sources, subtopic, writer)

        update: dict[str, Any] = {
            "sources": list(result.sources),
            "skipped_entries": result.skipped,  # this worker's delta, never a total (D-067)
            "explored_subtopics": [subtopic],  # success only (D-018)
            "seen_paper_ids": {source.arxiv_id for source in result.sources},
        }
        if not found:
            # Explored, but nothing published: a finding worth reporting, not a silent
            # no-op (D-021, O-5).
            update["empty_subtopics"] = [subtopic]
        return update

    return research_worker