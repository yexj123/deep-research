"""How fast does arXiv move on a topic? Picks the staleness question set on evidence (O-16).

O-16 asks what a corpus costs once it stops refreshing. That effect can only show up where
new papers actually appear: on a mature topic a month of arXiv changes almost nothing, so
measuring staleness there would be **underpowered by construction** -- D-095's mistake (testing
depth where it could not win) pointed the other way.

So the question set is chosen by measured publication velocity, not intuition. This counts
submissions in the last 30 days per candidate topic, and prints them ranked.

    uv run python -m tests.eval.velocity

Free: arXiv's API needs no key. ~3 s per topic, because the rate limit is respected (D-064).
"""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from defusedxml.ElementTree import fromstring

from deep_research.agent.sources.arxiv import ARXIV_API_URL, NAMESPACES, search_terms

WINDOW_DAYS = 30

# Deliberately a mix. The fast half is where staleness should bite; the slow half is the
# control that says the measurement can tell the difference at all -- without it, "the stale
# corpus missed N papers" has no baseline for "N is large".
CANDIDATES = [
    # expected fast
    "speculative decoding",
    "mixture of experts routing",
    "state space models sequence",
    "KV cache compression",
    "test time compute scaling",
    "retrieval augmented generation",
    "LLM agent tool use",
    "diffusion language models",
    # expected slow (controls)
    "support vector machine kernel",
    "hidden markov model speech recognition",
    "conditional random fields sequence labeling",
    "principal component analysis dimensionality",
]


def window() -> tuple[str, str]:
    """The last WINDOW_DAYS as arXiv's submittedDate bounds (YYYYMMDDHHMM)."""
    end = datetime.now(UTC)
    start = end - timedelta(days=WINDOW_DAYS)
    return start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")


def dated_query(topic: str, start: str, end: str) -> str:
    """The project's own term cleaning (D-051, D-059), plus a submission-date range.

    Built here rather than in `build_search_query` because a date filter is an evaluation
    concern: production never wants one, and adding a parameter only this script passes would
    put an unused branch in the retrieval path.
    """
    terms = " AND ".join(f"all:{term}" for term in search_terms(topic))
    return f"({terms}) AND submittedDate:[{start} TO {end}]"


async def total_results(client: httpx.AsyncClient, query: str) -> int:
    """arXiv's own match count, from opensearch:totalResults.

    Asking for the count rather than the papers: `max_results=1` returns the total in the feed
    header, so this is one cheap request per topic and no parsing of entries. **Not 0** --
    arXiv answers `max_results=0` with a 500, confirmed 2026-09-25.
    """
    response = await client.get(
        ARXIV_API_URL, params={"search_query": query, "start": 0, "max_results": 1}
    )
    response.raise_for_status()
    root = fromstring(response.text, forbid_dtd=True)  # D-047
    node = root.find("opensearch:totalResults", NAMESPACES)
    return int(node.text) if node is not None and node.text else 0


async def main() -> None:
    start, end = window()
    print(f"submissions between {start} and {end} ({WINDOW_DAYS} days)\n")

    rows: list[tuple[str, int]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for topic in CANDIDATES:
            count = await total_results(client, dated_query(topic, start, end))
            rows.append((topic, count))
            print(f"  {topic:<45} {count:>5}")
            await asyncio.sleep(3)  # arXiv asks for one request per three seconds

    print(f"\n{'ranked by velocity':<45} {'last ' + str(WINDOW_DAYS) + 'd':>10}")
    print("-" * 57)
    for topic, count in sorted(rows, key=lambda r: -r[1]):
        print(f"  {topic:<45} {count:>8}")


if __name__ == "__main__":
    asyncio.run(main())
