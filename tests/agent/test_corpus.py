"""The local corpus: indexing, BM25 retrieval, and the traps it exists to contain (O-13).

Three of these tests pin properties that fail **silently** rather than loudly, which is why
they are worth their length:

- `bm25()` returns the negative of the standard score, so `ORDER BY ... DESC` returns the
  *worst* matches with no error and entirely plausible output.
- Re-indexing a paper would inflate its own term frequencies, so the corpus would rank a paper
  highly for having been *found often* rather than for matching the query.
- An unsanitized subtopic crashes the query outright (`OperationalError: unterminated
  string`), and a capitalized one silently becomes an FTS5 operator expression.

No network and no model: FTS5 is in the bundled SQLite, so this whole file runs in
milliseconds against `:memory:`.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from deep_research.agent.sources.arxiv import build_fts_query
from deep_research.persistence.corpus import (
    ABSTRACT,
    FULL_TEXT,
    SCHEMA,
    Hit,
    covering_papers,
    index_sources,
    load_sources,
    search,
    stats,
)
from tests.agent.fakes import make_source


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    yield conn
    conn.close()


def _paper(n: int, title: str, summary: str):
    return make_source(arxiv_id=f"2411.{10000 + n:05d}", title=title, summary=summary)


PAPERS = [
    _paper(1, "FlashAttention: fast exact attention", "IO-aware exact attention with tiling."),
    _paper(2, "Mamba: linear-time sequence modelling", "A selective state space model."),
    _paper(3, "LoRA: low-rank adaptation", "Freezes weights and injects trainable matrices."),
]


# ---- indexing ------------------------------------------------------------------------


def test_indexing_stores_papers_and_their_abstracts(db) -> None:
    """A search's results become a corpus with no extra network cost (O-13)."""
    assert index_sources(db, PAPERS) == 3
    s = stats(db)
    assert s["papers"] == 3 and s["chunks"] == 3
    assert s["full_text_papers"] == 0


def test_indexing_the_same_paper_twice_does_not_duplicate_it(db) -> None:
    """Idempotence, and it is a *correctness* property rather than a tidiness one.

    A paper found by three subtopics in one run would otherwise be indexed three times, and
    BM25 would then rank it highly for having been found often rather than for matching the
    query. Nothing about that would look wrong.
    """
    index_sources(db, PAPERS)
    assert index_sources(db, PAPERS) == 0, "second pass must add no papers"

    s = stats(db)
    assert s["papers"] == 3 and s["chunks"] == 3


def test_the_title_is_searchable_not_only_the_abstract(db) -> None:
    """The method name is usually in the title and only paraphrased in the abstract.

    "FlashAttention" appears in no summary here, so a corpus indexing summaries alone would
    miss the exact query a user is most likely to type.
    """
    index_sources(db, PAPERS)
    assert covering_papers(db, build_fts_query("FlashAttention"), limit=10) == [
        PAPERS[0].arxiv_id
    ]


# ---- retrieval, and the directionality trap ------------------------------------------


def test_results_come_back_best_first(db) -> None:
    """Ascending `bm25()` is best-first, because the score is negated (O-13, LOCKED).

    Written as a relevance ordering rather than a fixed id list so it fails for the right
    reason: if the ORDER BY were ever flipped to DESC, this is the assertion that catches it.
    """
    index_sources(db, PAPERS)
    hits = search(db, build_fts_query("attention"), limit=10)

    assert hits, "expected at least one match"
    assert [h.relevance for h in hits] == sorted((h.relevance for h in hits), reverse=True)


def test_relevance_is_positive_so_higher_is_better(db) -> None:
    """Negative scores must not escape this module.

    `bm25()` is negative by design. Leaking that outward would make every threshold above it
    read backwards, and a caller writing `relevance > 0.5` would silently filter everything.
    """
    index_sources(db, PAPERS)
    hits = search(db, build_fts_query("attention"), limit=10)
    assert all(h.relevance >= 0 for h in hits), [h.relevance for h in hits]


def test_a_rarer_term_outranks_a_common_one(db) -> None:
    """BM25 is doing real work, not just returning insertion order.

    Without this, every test above would pass against a stub that returned all rows.
    """
    index_sources(db, PAPERS)
    rare = search(db, build_fts_query("Mamba"), limit=10)
    assert len(rare) == 1 and rare[0].arxiv_id == PAPERS[1].arxiv_id


def test_a_query_matching_nothing_returns_nothing(db) -> None:
    """A miss is an empty list, not an error -- "not covered" is a normal answer (D-021)."""
    index_sources(db, PAPERS)
    assert search(db, build_fts_query("crystallography"), limit=10) == []


def test_limit_is_honoured(db) -> None:
    """The top-k bound the sufficiency rule counts within."""
    index_sources(db, PAPERS)
    assert len(search(db, build_fts_query("attention OR model OR adaptation"), limit=2)) <= 2


# ---- sufficiency counts papers, not chunks or scores ---------------------------------


def test_covering_papers_counts_distinct_papers_not_chunks(db) -> None:
    """One paper with many chunks is one paper's worth of coverage (O-13).

    Counting chunks would measure redundancy: a single full-text paper split into ten sections
    would read as ten papers of coverage and the corpus would declare a subtopic covered on
    the strength of one source.
    """
    index_sources(db, PAPERS)
    target = PAPERS[0].arxiv_id
    for i, section in enumerate(("intro", "method", "results")):
        db.execute(
            "INSERT INTO chunks (arxiv_id, tier, section, text) VALUES (?, ?, ?, ?)",
            (target, FULL_TEXT, section, f"tiling attention section {i}"),
        )
    db.commit()

    hits = search(db, build_fts_query("attention"), limit=20)
    assert sum(h.arxiv_id == target for h in hits) > 1, "fixture should produce many chunks"
    assert covering_papers(db, build_fts_query("attention"), limit=20).count(target) == 1


def test_covering_papers_preserves_best_first_order(db) -> None:
    """The caller's threshold cuts the top of this list, so order has to survive dedup."""
    index_sources(db, PAPERS)
    ids = covering_papers(db, build_fts_query("attention OR adaptation"), limit=10)
    assert ids and ids[0] == PAPERS[0].arxiv_id


# ---- the tier split, which keeps the O-13 comparison fair ----------------------------


def test_searching_one_tier_excludes_the_other(db) -> None:
    """An abstracts-only arm must be recordable against a full-text arm (D-088).

    Without this the comparison changes two variables at once and neither arm means anything.
    """
    index_sources(db, PAPERS)
    db.execute(
        "INSERT INTO chunks (arxiv_id, tier, section, text) VALUES (?, ?, 'method', ?)",
        (PAPERS[1].arxiv_id, FULL_TEXT, "a selective scan over attention baselines"),
    )
    db.commit()

    abstracts = search(db, build_fts_query("attention"), limit=20, tier=ABSTRACT)
    full = search(db, build_fts_query("attention"), limit=20, tier=FULL_TEXT)

    assert all(h.tier == ABSTRACT for h in abstracts)
    assert [h.arxiv_id for h in full] == [PAPERS[1].arxiv_id]
    assert PAPERS[1].arxiv_id not in [h.arxiv_id for h in abstracts]


# ---- sanitization: the injection surface -------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        '"unterminated',
        "attention AND transformer",
        "text:attention",
        "atten*",
        "(attention)",
        "attention NEAR transformer",
    ],
)
def test_a_hostile_subtopic_cannot_break_the_query(db, hostile: str) -> None:
    """Planner text reaches this query, so FTS5 syntax in it is an injection surface (O-13).

    Measured 2026-09-21: `'"unterminated'` raises `OperationalError: unterminated string`,
    crashing the query outright. The defence is `build_fts_query` -> `search_terms`, which
    strips symbolic operators and lowercases the word operators into ordinary terms. This test
    is the reason nobody can "tidy up" that lowercasing later: preserving the planner's
    capitalization would silently restore operator injection.
    """
    index_sources(db, PAPERS)
    search(db, build_fts_query(hostile), limit=10)  # must not raise


def test_the_raw_form_really_would_have_broken(db) -> None:
    """The control case: proves the test above is defending against something real.

    Without it, `test_a_hostile_subtopic_cannot_break_the_query` would pass just as well
    against an input that was never dangerous.
    """
    index_sources(db, PAPERS)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ('"unterminated',))


# ---- round trip ----------------------------------------------------------------------


def test_a_paper_from_the_corpus_is_a_validated_source(db) -> None:
    """Corpus papers must be indistinguishable downstream from freshly fetched ones (D-043).

    They flow into the same `sources` list, the same ranking and the same citation check, so a
    corpus row that is merely dict-shaped would fail somewhere far from here.
    """
    index_sources(db, PAPERS)
    restored = load_sources(db, [PAPERS[0].arxiv_id])

    assert len(restored) == 1
    assert restored[0].arxiv_id == PAPERS[0].arxiv_id
    assert restored[0].title == PAPERS[0].title
    assert restored[0].authors == PAPERS[0].authors
    assert isinstance(restored[0].published, datetime)
    assert restored[0].published.tzinfo is not None, "timezone must survive the round trip"


def test_the_paper_version_survives_the_round_trip(db) -> None:
    """A v3 paper must not come back claiming v1 (D-103).

    `version` is a validated field on `Source`, and the first version of this schema simply
    did not store it -- `load_sources` hardcoded 1. The result was a paper whose `version`
    said 1 while its `url` still ended `v3`: internally inconsistent, wrong in a way that
    reads as normal, and invisible until someone compared the two. Nothing downstream uses
    `version` today, which is exactly why it could have stayed wrong indefinitely.
    """
    versioned = make_source(arxiv_id="2411.20000", version=3, url="https://arxiv.org/abs/2411.20000v3")
    index_sources(db, [versioned])

    restored = load_sources(db, ["2411.20000"])[0]
    assert restored.version == 3
    assert restored.url.endswith("v3"), "version and url must not contradict each other"


def test_an_older_corpus_gains_the_version_column(tmp_path) -> None:
    """A corpus built before `version` existed must still open (D-103).

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so without a
    migration every corpus built by the first release would raise `no such column: version`
    on the next read -- or, worse, be silently rebuilt and lose the `indexed_at` history the
    staleness report depends on.
    """
    import sqlite3 as _sqlite3

    from deep_research.persistence.corpus import connect

    path = str(tmp_path / "old.sqlite")
    old = _sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE papers (
            arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT NOT NULL,
            summary TEXT NOT NULL, published TEXT NOT NULL, url TEXT NOT NULL,
            indexed_at TEXT NOT NULL, has_full_text INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    old.execute(
        "INSERT INTO papers VALUES ('2411.18583', 'T', '[]', 'S',"
        " '2024-11-27T00:00:00+00:00', 'https://arxiv.org/abs/2411.18583v1',"
        " '2026-01-01T00:00:00+00:00', 0)"
    )
    old.commit()
    old.close()

    migrated = connect(path)
    try:
        assert load_sources(migrated, ["2411.18583"])[0].version == 1
        assert stats(migrated)["indexed_from"] == "2026-01-01T00:00:00+00:00", (
            "the index history must survive the migration"
        )
    finally:
        migrated.close()


def test_migrating_twice_is_harmless(tmp_path) -> None:
    """`connect` runs the migration every time, so it has to be idempotent."""
    from deep_research.persistence.corpus import connect

    path = str(tmp_path / "twice.sqlite")
    for _ in range(3):
        db = connect(path)
        db.close()

    db = connect(path)
    try:
        index_sources(db, PAPERS)
        assert stats(db)["papers"] == 3
    finally:
        db.close()


def test_load_sources_keeps_the_order_asked_for(db) -> None:
    """Ranking order is the whole point of retrieval; `IN (...)` does not preserve it."""
    index_sources(db, PAPERS)
    wanted = [PAPERS[2].arxiv_id, PAPERS[0].arxiv_id]
    assert [s.arxiv_id for s in load_sources(db, wanted)] == wanted


def test_load_sources_skips_ids_the_corpus_does_not_have(db) -> None:
    """A missing id is a gap, not a crash -- and must not shift the others."""
    index_sources(db, PAPERS)
    got = load_sources(db, ["9999.99999", PAPERS[0].arxiv_id])
    assert [s.arxiv_id for s in got] == [PAPERS[0].arxiv_id]


def test_load_sources_of_nothing_is_empty(db) -> None:
    """Guards the `IN ()` syntax error an empty list would otherwise build."""
    assert load_sources(db, []) == []


# ---- staleness reporting -------------------------------------------------------------


def test_stats_reports_when_the_corpus_was_indexed(db) -> None:
    """The coverage panel has to be able to say how old its answer is (O-13, O-5).

    arXiv grows ~100 GB a month, so a corpus that answered well in March silently misses
    April's key paper. Reporting the window is what stops that being invisible.
    """
    index_sources(db, PAPERS)
    s = stats(db)
    assert s["indexed_from"] and s["indexed_to"]
    assert datetime.fromisoformat(str(s["indexed_to"])).tzinfo is not None


def test_stats_on_an_empty_corpus_does_not_crash(db) -> None:
    """First run, before anything is indexed: zero papers, no dates, no exception."""
    s = stats(db)
    assert s["papers"] == 0 and s["chunks"] == 0
    assert s["indexed_from"] is None
