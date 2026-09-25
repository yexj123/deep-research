"""Does OpenAlex fit this agent's retrieval contract? Measured, not assumed (O-17).

Two questions decide whether a second source is cheap or a rewrite:

1. **Identity.** `Source.arxiv_id` is the primary key of the corpus, the `seen_paper_ids` set,
   `synthesized_from`, and the `[arXiv:<id>]` citation marker the whole project is built
   around (D-046). If OpenAlex results carry arXiv IDs, a second source is additive. If they
   do not, adding one means changing what a paper *is* everywhere.

2. **Relevance.** arXiv's API searches a CS/physics corpus; OpenAlex searches all of
   scholarship. A source that answers "speculative decoding" with 1973 media theory would
   feed the corpus and the review with off-topic papers -- and citation grounding cannot
   catch that, because those papers really were retrieved.

Run against the subtopics real runs actually produced:

    uv run python -m tests.eval.openalex_fit

Costs OpenAlex credits (~10 per subtopic), no API key, no LLM.
"""

import re
import time
from collections import Counter

import httpx

from tests.eval.recording import arms, load_all

MAILTO = "deep-research@example.invalid"
URL = "https://api.openalex.org/works"
PER_PAGE = 20

# arXiv registers a DataCite DOI for every submission, so an arXiv paper reachable through
# OpenAlex shows up as one of these. Checked against ids, the DOI and every location.
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)", re.I)
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#v]+)", re.I)

# What this agent is for. Anything else retrieved for a CS/ML subtopic is noise that reaches
# the synthesis prompt as though it were evidence.
PAPER_TYPES = {"article", "preprint", "conference-paper", "posted-content"}


def arxiv_id_of(work: dict) -> str | None:
    """The arXiv ID behind an OpenAlex work, if there is one reachable at all."""
    doi = work.get("doi") or ""
    if match := ARXIV_DOI.search(doi):
        return match.group(1)
    for value in (work.get("ids") or {}).values():
        if isinstance(value, str) and (match := ARXIV_DOI.search(value) or ARXIV_URL.search(value)):
            return match.group(1)
    for location in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            if (url := location.get(key)) and (match := ARXIV_URL.search(url)):
                return match.group(1)
    return None


def recorded_subtopics(limit: int) -> list[str]:
    """Subtopics real runs actually produced, deduplicated, oldest arm first.

    Using the planner's real output rather than invented queries: the question a source has to
    answer here is a *subtopic*, which is narrower and more jargon-heavy than a user question,
    and D-106 is the standing reminder that those two populations behave differently.
    """
    seen: dict[str, None] = {}
    for arm in sorted(arms()):
        for recording in load_all(arm):
            for subtopic in recording.coverage.get("explored", []):
                seen.setdefault(subtopic, None)
    return list(seen)[:limit]


def search(client: httpx.Client, subtopic: str) -> list[dict]:
    response = client.get(
        URL,
        params={
            "search": subtopic,
            "per-page": PER_PAGE,
            "mailto": MAILTO,
            "select": "id,ids,doi,title,publication_date,type,locations",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"]


def main(limit: int = 12) -> None:
    subtopics = recorded_subtopics(limit)
    if not subtopics:
        print("no recorded subtopics found")
        return

    totals = Counter()
    types = Counter()
    rows = []

    with httpx.Client(follow_redirects=True) as client:
        for subtopic in subtopics:
            try:
                works = search(client, subtopic)
            except httpx.HTTPStatusError as exc:
                print(f"! {subtopic[:50]:<50} {exc.response.status_code}")
                continue
            with_arxiv = sum(1 for w in works if arxiv_id_of(w))
            papers = sum(1 for w in works if w.get("type") in PAPER_TYPES)
            for work in works:
                types[work.get("type") or "?"] += 1
            totals["results"] += len(works)
            totals["arxiv"] += with_arxiv
            totals["papers"] += papers
            rows.append((subtopic, len(works), with_arxiv, papers))
            time.sleep(0.2)  # polite, and well inside the credit budget

    print(f"{'subtopic':<52} {'n':>3} {'arXiv':>6} {'paper-type':>11}")
    print("-" * 76)
    for subtopic, n, with_arxiv, papers in rows:
        print(f"{subtopic[:50]:<52} {n:>3} {with_arxiv:>6} {papers:>11}")
    print("-" * 76)
    n, a, p = totals["results"], totals["arxiv"], totals["papers"]
    print(f"{'POOLED':<52} {n:>3} {a:>6} {p:>11}")
    if n:
        print(f"\narXiv-identifiable: {a}/{n} = {100 * a / n:.1f}%")
        print(f"paper-type (article/preprint/conference): {p}/{n} = {100 * p / n:.1f}%")
    print("\nresult types:", dict(types.most_common()))


if __name__ == "__main__":
    main()
