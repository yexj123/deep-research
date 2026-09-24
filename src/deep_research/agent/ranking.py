"""Rank a run's own papers against its question, with BM25 over SQLite FTS5.

**Why this exists.** Measured across ten baseline runs (D-090): 797 papers were placed in
synthesis prompts and 81 were cited -- 10.2%. The cited count is 6-12 and does **not** scale
with how many papers are supplied: 89 in gives 8 cited, 59 in gives 9. So the ~90% surplus is
not buying coverage, it is ~26k wasted tokens per run.

No corpus, no embeddings, no downloads: this indexes the run's *own* papers in an in-memory
FTS5 table and keeps the best N. The same machinery is what O-13's persistent corpus needs, so
writing it here is a down payment rather than a detour.

**FTS5 is in the bundled SQLite** (confirmed 3.49.1), so this adds no dependency.
"""

import sqlite3
from collections.abc import Sequence

from deep_research.agent.sources.arxiv import build_fts_query
from deep_research.agent.sources.models import Source

_SCHEMA = """
CREATE VIRTUAL TABLE papers USING fts5(
    title, summary, tokenize='porter unicode61'
)
"""


def _relevance_rows(sources: Sequence[Source], query: str) -> list[tuple[int, float]]:
    """(index into `sources`, relevance) for papers matching `query`, best first.

    **`bm25()` returns the NEGATIVE of the standard score** so that a plain `ORDER BY bm25(t)`
    sorts best-first. Two consequences this function exists to contain:

    - Adding `DESC` would return the *worst* matches, with no error and plausible output.
    - Negative scores must not escape this module, so the caller sees `relevance = -bm25(...)`
      and can reason in "higher is better" everywhere above it.
    """
    with sqlite3.connect(":memory:") as db:
        db.execute(_SCHEMA)
        db.executemany(
            "INSERT INTO papers(rowid, title, summary) VALUES (?, ?, ?)",
            [(i, s.title, s.summary) for i, s in enumerate(sources)],
        )
        return [
            (rowid, -score)
            for rowid, score in db.execute(
                "SELECT rowid, bm25(papers) FROM papers WHERE papers MATCH ? ORDER BY bm25(papers)",
                (query,),
            )
        ]


def rank_sources(
    sources: Sequence[Source], question: str, top_n: int | None
) -> list[Source]:
    """The `top_n` papers most relevant to `question`, best first.

    `top_n=None` returns every paper in its original order -- today's behaviour, kept as a
    switchable baseline arm so both can be recorded from one codebase rather than from git
    history (D-090).

    Papers that match nothing are appended after the ranked ones rather than dropped. BM25
    scores are corpus-relative: a term appearing in more than half the papers gets degenerate
    IDF and scores collapse toward zero regardless of match quality (measured: a term in 25 of
    30 documents scored -0.000). Silently discarding non-matchers would make the cut depend on
    that artifact instead of on relevance.
    """
    if top_n is None:
        return list(sources)

    try:
        query = build_fts_query(question)
    except ValueError:
        # A question of only stopwords cannot rank anything. decompose filters these before
        # dispatch (D-073), so reaching here means ranking has nothing to go on -- not that
        # the run is broken. Degrade to an unranked cut rather than failing a run that has
        # already paid for its results, and still honour top_n: returning everything here
        # would quietly restore the ~90% surplus this function exists to remove.
        return list(sources)[:top_n]

    # Rank even when nothing will be cut, so a function called rank_sources always ranks and
    # the prompt's ordering does not depend on how many papers a run happened to retrieve.
    ranked = _relevance_rows(sources, query)
    matched = {index for index, _ in ranked}
    chosen = [sources[index] for index, _ in ranked]
    unmatched = [s for index, s in enumerate(sources) if index not in matched]
    return (chosen + unmatched)[:top_n]
