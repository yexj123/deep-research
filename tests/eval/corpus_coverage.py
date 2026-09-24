"""Does the corpus cover this subtopic? Measuring a test that works (O-13 step 2, D-104).

**What this measures, and what it overturned.** O-13 specified the sufficiency test in
advance: *"a subtopic is covered locally when at least `MIN_LOCAL_PAPERS` distinct papers
appear in the top-k FTS5 results."* Measured against the 80 committed recordings, that test
**returns k every single time, for every query, in every corpus** -- including a corpus of
1669 machine-learning papers queried about medieval Flemish guilds. It cannot discriminate,
because `build_fts_query` ORs the terms and any non-trivial corpus contains *something*
matching *some* term.

That is D-094's lesson arriving again: the rule was specified from reasoning, and measuring it
first showed it could never fire the way it was meant to.

**The test that does work: term coverage.** Instead of counting papers in the top-k, count how
many of those papers match at least half the query's terms. A paper matching "attention" alone
is noise; one matching "attention", "transformer" *and* "models" is about the subtopic.

Measured on a 1669-paper ML corpus:

| query | papers in top-20 | matching >= half the terms |
|---|---|---|
| in-domain (4 questions) | 20, 20, 20, 20 | **20, 20, 20, 5** |
| out-of-domain (4 questions) | 20, 20, 20, 20 | **1, 0, 0, 0** |

The count column is constant; the coverage column separates cleanly.

**Why the first experiment failed, which is worth recording.** Its negative population was
"the other nineteen questions", and every question in the frozen set is machine learning. A
corpus of nineteen ML topics genuinely *does* hold attention papers -- so that population was
never a true negative, and it scored *higher* than the positive one simply by being 20x
larger. Raw counts measure corpus size, exactly as O-13 warned about raw scores. The control
that makes the measurement mean anything is a genuinely out-of-domain question, and the first
design had none.
"""

import re
import sqlite3
import statistics as st
from dataclasses import dataclass
from datetime import UTC, datetime

from deep_research.agent.sources.arxiv import build_fts_query, search_terms
from deep_research.agent.sources.models import Source
from deep_research.persistence.corpus import SCHEMA, index_sources, search
from tests.eval.recording import load_all

ENTRY = re.compile(r"^<papers>\n\[arXiv:([^\]]+)\] (.*?)\n(.*)\n</papers>$", re.S)
_PLACEHOLDER_DATE = datetime(2024, 1, 1, tzinfo=UTC)

# The negative control the first design lacked. Deliberately far from machine learning in
# vocabulary as well as subject: the mechanism being tested is whether the corpus holds the
# *words*, so a negative that shares jargon would prove nothing. Frozen like a question set --
# changing these changes what "out of domain" means.
OUT_OF_DOMAIN = (
    "How do CRISPR off-target effects arise in gene editing?",
    "What causes coral bleaching in tropical reef ecosystems?",
    "How did medieval guilds regulate apprenticeship in Flanders?",
    "What is the role of mycorrhizal fungi in forest nutrient cycling?",
)


def parse_papers(retrieval_context: list[str]) -> list[Source]:
    """Rebuild `Source` objects from a recording's formatted `<papers>` blocks.

    Free: the 80 recordings already hold every paper eight runs retrieved, so the threshold
    can be re-derived whenever they change, with no API call. Validated `Source`s rather than
    tuples, so the corpus is fed exactly what the agent would feed it.
    """
    papers = []
    for block in retrieval_context:
        match = ENTRY.match(block)
        if not match:
            continue
        arxiv_id, title, summary = match.groups()
        papers.append(
            Source(
                arxiv_id=arxiv_id,
                version=1,
                title=title,
                authors=("unknown",),
                summary=summary,
                published=_PLACEHOLDER_DATE,
                url=f"https://arxiv.org/abs/{arxiv_id}v1",
            )
        )
    return papers


def build_corpus(papers: list[Source]) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    index_sources(db, papers)
    return db


def all_recorded_papers() -> list[Source]:
    """Every paper across every committed recording, deduplicated by the corpus itself."""
    return [paper for rec in load_all() for paper in parse_papers(rec.retrieval_context)]


def recorded_questions() -> dict[str, str]:
    return {rec.id: rec.question for rec in load_all()}


@dataclass(frozen=True)
class Probe:
    """One question asked of one corpus."""

    question: str
    in_domain: bool
    papers_in_top_k: int
    strongly_matching: int


def strongly_matching(db: sqlite3.Connection, question: str, k: int = 20) -> Probe | None:
    """How many of the top-k papers match at least half the question's terms.

    Half rather than all: `AND` over every term returns nothing on almost every query
    (measured -- the median across 20 questions was **0** even on a same-topic corpus), so it
    is too strict to be a coverage test. Half is strict enough that a paper sharing one common
    word does not count, which is what the naive top-k count failed to exclude.

    Returns None when the question has no searchable terms (D-059).
    """
    try:
        terms = search_terms(question)
        query = build_fts_query(question)
    except ValueError:
        return None

    needed = (len(terms) + 1) // 2
    top = list(dict.fromkeys(hit.arxiv_id for hit in search(db, query, k)))

    matches: dict[str, int] = {}
    for term in terms:
        for hit in search(db, term, 10_000):
            if hit.arxiv_id in top:
                matches[hit.arxiv_id] = matches.get(hit.arxiv_id, 0) + 1

    return Probe(
        question=question,
        in_domain=True,  # set by the caller
        papers_in_top_k=len(top),
        strongly_matching=sum(1 for pid in top if matches.get(pid, 0) >= needed),
    )


def measure(k: int = 20) -> list[Probe]:
    """Probe one corpus of every recorded paper with in- and out-of-domain questions."""
    db = build_corpus(all_recorded_papers())
    try:
        probes: list[Probe] = []
        for question in recorded_questions().values():
            probe = strongly_matching(db, question, k)
            if probe:
                probes.append(probe)
        for question in OUT_OF_DOMAIN:
            probe = strongly_matching(db, question, k)
            if probe:
                probes.append(
                    Probe(probe.question, False, probe.papers_in_top_k, probe.strongly_matching)
                )
        return probes
    finally:
        db.close()


def summarize(probes: list[Probe]) -> str:
    inside = [p.strongly_matching for p in probes if p.in_domain]
    outside = [p.strongly_matching for p in probes if not p.in_domain]
    counts = {p.papers_in_top_k for p in probes}

    return "\n".join(
        [
            f"papers in top-k: {sorted(counts)}  <- constant, which is why counting fails",
            "",
            f"{'population':<14}{'n':>4}{'min':>6}{'median':>9}{'max':>6}",
            "-" * 39,
            f"{'in-domain':<14}{len(inside):>4}{min(inside):>6}{st.median(inside):>9.1f}{max(inside):>6}",
            f"{'out-of-domain':<14}{len(outside):>4}{min(outside):>6}{st.median(outside):>9.1f}{max(outside):>6}",
            "",
            f"separation: worst in-domain {min(inside)} vs best out-of-domain {max(outside)}"
            f"  -> gap {min(inside) - max(outside):+d}",
        ]
    )


def main() -> int:
    print(summarize(measure()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
