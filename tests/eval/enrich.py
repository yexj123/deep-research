"""Populate the corpus's full-text tier, for the O-13 comparison (D-109).

    uv run python -m tests.eval.enrich --limit 30        # pilot
    uv run python -m tests.eval.enrich                   # everything in the recorded contexts

**Why this is a script and not a graph node.** The production question -- *when* should a run
fetch full text -- only matters if full text helps. Building the node first and measuring
after would be the D-094 mistake a third time: two rules specified from reasoning have already
turned out unable to fire. So this populates a corpus offline, the arm gets recorded against
it, and the plumbing is designed only if the numbers justify it.

**The corpus is a file, not in-memory.** Abstracts rebuild from recordings in a second
(D-107), but full text costs a 3-second interval and a ~650 KB download per paper, so it has
to survive between this pass and the recording sweep. The file is gitignored and regenerable
by this command; **no PDF is ever written to disk**, only extracted chunks (D-008).

**Rate limited exactly like a search.** D-064's one-request-per-three-seconds applies to
downloads too, and a PDF is a far larger request than a metadata query. Sequential on purpose:
this is a background chore, and being polite to arXiv costs only wall-clock time nobody is
waiting on.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from deep_research.agent.config import ARXIV_MIN_INTERVAL_SECONDS
from deep_research.agent.sources.fulltext import (
    FullTextError,
    chunk_paper,
    extract_text,
    fetch_pdf,
)
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.persistence.corpus import (
    connect,
    index_full_text,
    index_sources,
    papers_without_full_text,
    stats,
)
from tests.eval.corpus_coverage import all_recorded_papers, parse_papers
from tests.eval.recording import load_all

CORPUS_PATH = Path(__file__).parent / "corpus-fulltext.sqlite"


# The arms the comparison will actually retrieve against: today's default configuration.
# Enriching every recorded paper is 1669 at three seconds each -- 83 minutes, most of it spent
# on papers from superseded arms (the pre-D-091 `abstract-all` arm alone contributed ~80 per
# question). Scoping to the current arms is the difference between 83 minutes and 20.
DEFAULT_ARMS = (
    "broad-abstract-local-top20-d2-adaptive",
    "narrow-abstract-local-top20-d2-adaptive",
)


def papers_in_recorded_contexts(arms: tuple[str, ...] | None = DEFAULT_ARMS) -> list[str]:
    """Papers that actually reached a synthesis prompt, in first-seen order.

    Only papers retrieval surfaces can affect a review, and `retrieval_context` is exactly the
    set that did. `arms=None` takes every recording, which is the whole corpus and rarely what
    is wanted.
    """
    seen: dict[str, None] = {}
    for recording in load_all():
        if arms is not None and recording.arm not in arms:
            continue
        for paper in parse_papers(recording.retrieval_context):
            seen.setdefault(paper.arxiv_id, None)
    return list(seen)


async def enrich(limit: int | None, corpus_path: Path = CORPUS_PATH) -> None:
    db = connect(str(corpus_path))
    try:
        index_sources(db, all_recorded_papers())
        todo = papers_without_full_text(db, papers_in_recorded_contexts())
        if limit is not None:
            todo = todo[:limit]

        print(f"corpus: {stats(db)}")
        print(f"to enrich: {len(todo)} paper(s)\n")
        if not todo:
            return

        limiter = ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS)
        failures = 0
        async with httpx.AsyncClient(
            headers={"User-Agent": "deep-research/0.1 (research tool; arXiv API terms)"}
        ) as client:
            for index, arxiv_id in enumerate(todo, start=1):
                try:
                    async with limiter:
                        pdf = await fetch_pdf(client, arxiv_id)
                    chunks = chunk_paper(extract_text(pdf))
                    added = index_full_text(
                        db, arxiv_id, [(chunk.section, chunk.text) for chunk in chunks]
                    )
                    print(f"  {index:>4}/{len(todo)}  {arxiv_id}  {added:>3} chunks")
                except FullTextError as exc:
                    # A withdrawn paper, a scan, a transport blip. Data, not a bug (D-053):
                    # the corpus simply has no full text for it, and the pass continues.
                    failures += 1
                    print(f"  {index:>4}/{len(todo)}  {arxiv_id}  SKIPPED: {exc}")

        print(f"\nfailed: {failures}/{len(todo)}")
        print(f"corpus: {stats(db)}")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="enrich at most N papers (pilot first)"
    )
    args = parser.parse_args()
    asyncio.run(enrich(args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
