"""The local paper corpus: abstracts indexed from every search, retrieved with FTS5 BM25 (O-13).

**What this is for.** Every arXiv search already returns abstracts the run pays for and then
throws away. Indexing them costs nothing extra and makes the corpus useful from the first run
rather than after a bulk-download phase. A later subtopic that the corpus already covers can
then be answered with no network call at all.

**Why BM25 and not embeddings.** Academic search runs on exact jargon -- *LoRA*, *Mamba*,
*FlashAttention*, author names -- which a small embedding model blurs. FTS5 is compiled into
the bundled SQLite (3.49.1 confirmed), so this adds no dependency, no model download and no
asymmetric query/passage prefix trap. The strongest argument is consistency: `arxiv.py`
matches lexically, so if the local tier matches lexically too, "the corpus does not cover this"
means something. With embeddings locally and keywords remotely, a local miss might mean only
that two retrieval methods disagreed. Dense retrieval is a *measured follow-on*, not a
rejected idea.

**Synchronous on purpose.** These are microsecond queries against a local file, and keeping
the module sync keeps it pure and testable -- the same choice `agent/ranking.py` makes. A
caller inside the event loop that ever finds these slow should wrap them in `asyncio.to_thread`
rather than making this module async.

Legal position (D-008, re-read 2026-09-21): arXiv permits building indexes over content and
prohibits *serving* e-prints. This indexes metadata and abstracts for a single user's own
retrieval; it must not become a redistribution endpoint.
"""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from deep_research.agent.sources.arxiv import build_fts_query, search_terms
from deep_research.agent.sources.models import Source

# `tier` values. Abstracts are free and indexed always; full text is fetched for selected
# papers only, and is what O-13's comparison actually tests.
ABSTRACT = "abstract"
FULL_TEXT = "full_text"

# `section` is NOT NULL with an empty default so that UNIQUE(arxiv_id, tier, section) actually
# constrains: SQLite permits unlimited NULLs in a unique index, so a nullable column here would
# let the same abstract be indexed twice per run and inflate every BM25 count silently.
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id      TEXT PRIMARY KEY,
    version       INTEGER NOT NULL DEFAULT 1,
    title         TEXT NOT NULL,
    authors       TEXT NOT NULL,
    summary       TEXT NOT NULL,
    published     TEXT NOT NULL,
    url           TEXT NOT NULL,
    indexed_at    TEXT NOT NULL,
    has_full_text INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    tier     TEXT NOT NULL,
    section  TEXT NOT NULL DEFAULT '',
    text     TEXT NOT NULL,
    UNIQUE (arxiv_id, tier, section)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `relevance` is always "higher is better" -- see `search`."""

    arxiv_id: str
    tier: str
    section: str
    relevance: float


def connect(path: str) -> sqlite3.Connection:
    """Open the corpus and make sure its schema exists and is current.

    `check_same_thread=False` because a caller may hand this to `asyncio.to_thread`; the
    connection is still used by one task at a time.
    """
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript(SCHEMA)
    migrate(db)
    return db


def migrate(db: sqlite3.Connection) -> None:
    """Bring an existing corpus up to the current schema.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a column
    added after the first release would be missing on every corpus built before it -- and
    `load_sources` would read `NULL` or fall back to a default, producing papers that are
    quietly wrong rather than absent. This is idempotent and runs on every `connect`.

    The corpus is a cache and could simply be rebuilt, but rebuilding silently discards the
    `indexed_at` history the staleness report depends on (O-13), so migrating is the cheaper
    honest option.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(papers)")}
    if "version" not in columns:
        # Added after D-100 shipped without it. DEFAULT 1 is wrong for any paper that was
        # actually at v2+, which is why the fix is a migration and not just a schema edit:
        # rows written before this point cannot be repaired, only re-indexed.
        db.execute("ALTER TABLE papers ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        db.commit()


def index_sources(db: sqlite3.Connection, sources: Sequence[Source]) -> int:
    """Index each paper's metadata and its abstract chunk. Returns papers newly added.

    **Idempotent.** A paper seen by three subtopics in one run, or by three runs over a month,
    is stored once: `INSERT OR IGNORE` on the primary key, and `UNIQUE(arxiv_id, tier, section)`
    on chunks. Without that the same abstract would be indexed repeatedly and inflate its own
    BM25 term frequencies -- a corpus that ranks a paper higher for having been found often
    rather than for matching the query, and nothing would look wrong.

    The chunk text is `title + summary` rather than the summary alone, because a title carries
    the method name a search is most likely to be for ("FlashAttention") while the abstract
    may only paraphrase it.

    **First seen wins, deliberately.** A paper already in the corpus is left alone even if
    arXiv now serves a newer version, matching `merge_sources`, which dedups on `arxiv_id` and
    keeps the first (D-067). The consequence is real and belongs in the staleness story rather
    than being fixed quietly here: a corpus can hold v1's abstract months after v3 appeared.
    Deciding whether to refresh on version change is part of O-13's open staleness question,
    and doing it here would mean re-indexing chunks and invalidating BM25 statistics on every
    search -- a cost worth paying only once there is evidence it matters.
    """
    now = datetime.now(UTC).isoformat()
    added = 0
    with db:
        for source in sources:
            cursor = db.execute(
                "INSERT OR IGNORE INTO papers"
                " (arxiv_id, version, title, authors, summary, published, url, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source.arxiv_id,
                    source.version,
                    source.title,
                    json.dumps(list(source.authors)),
                    source.summary,
                    source.published.isoformat(),
                    source.url,
                    now,
                ),
            )
            added += cursor.rowcount or 0
            db.execute(
                "INSERT OR IGNORE INTO chunks (arxiv_id, tier, section, text)"
                " VALUES (?, ?, '', ?)",
                (source.arxiv_id, ABSTRACT, f"{source.title}\n\n{source.summary}"),
            )
    return added


def index_full_text(
    db: sqlite3.Connection, arxiv_id: str, chunks: Sequence[tuple[str, str]]
) -> int:
    """Index a paper's full-text chunks as `(section, text)` pairs. Returns chunks added.

    The paper must already be in `papers` -- full text enriches a paper the corpus knows
    about, it never introduces one. A chunk referencing an unknown `arxiv_id` would violate
    the foreign key and, more importantly, would mean a paper searchable in full text but
    absent from `load_sources`: retrievable and uncitable at once.

    Idempotent on `(arxiv_id, tier, section)`, so re-enriching a paper is a no-op rather than
    a second copy inflating its own BM25 term frequencies (D-100). That is why
    `chunk_paper` numbers split sections -- three unnumbered "Methodology" chunks would
    collapse to one here, silently losing two thirds of the section.

    Sets `has_full_text`, which is what stops the enrichment pass re-fetching a paper it has
    already read.
    """
    if not chunks:
        return 0
    with db:
        known = db.execute(
            "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if not known:
            raise ValueError(f"cannot index full text for unknown paper {arxiv_id!r}")

        added = 0
        for section, text in chunks:
            cursor = db.execute(
                "INSERT OR IGNORE INTO chunks (arxiv_id, tier, section, text)"
                " VALUES (?, ?, ?, ?)",
                (arxiv_id, FULL_TEXT, section, text),
            )
            added += cursor.rowcount or 0
        db.execute("UPDATE papers SET has_full_text = 1 WHERE arxiv_id = ?", (arxiv_id,))
    return added


def papers_without_full_text(db: sqlite3.Connection, arxiv_ids: Sequence[str]) -> list[str]:
    """Which of `arxiv_ids` the corpus knows but has not read in full, in the order given.

    The enrichment pass's work list. Filtering here rather than at the call site keeps the
    "have I already read this?" question in one place -- asking it twice is how a pass ends up
    re-downloading papers it already has, at three seconds each.
    """
    if not arxiv_ids:
        return []
    placeholders = ",".join("?" * len(arxiv_ids))
    rows = {
        row[0]
        for row in db.execute(
            f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({placeholders})"
            " AND has_full_text = 0",
            list(arxiv_ids),
        )
    }
    return [arxiv_id for arxiv_id in arxiv_ids if arxiv_id in rows]


def search(
    db: sqlite3.Connection, query: str, limit: int, tier: str | None = None
) -> list[Hit]:
    """Chunks matching `query`, best first. `query` must already be FTS5-safe.

    **`bm25()` returns the NEGATIVE of the standard score**, so that a plain `ORDER BY bm25(t)`
    sorts best-first. Two consequences this function exists to contain, both of which fail
    silently rather than loudly:

    - Adding `DESC` returns the *worst* matches, with no error and entirely plausible output.
    - Negative scores must not escape this module. Callers get `relevance = -bm25(...)` and can
      reason in "higher is better" everywhere above it.

    Sanitization is the caller's job, via `arxiv.build_fts_query`, which strips the symbolic
    operators and lowercases away AND/OR/NOT/NEAR. Passing a raw subtopic here is an injection
    bug: `'"unterminated` raises `OperationalError` and crashes the query outright.

    `tier` restricts to one layer, which is what lets an abstracts-only arm be recorded against
    a full-text arm with one variable changing (D-088).
    """
    sql = (
        "SELECT c.arxiv_id, c.tier, c.section, -bm25(chunks_fts) AS relevance"
        " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
        " WHERE chunks_fts MATCH ?"
    )
    params: list[object] = [query]
    if tier is not None:
        sql += " AND c.tier = ?"
        params.append(tier)
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(limit)

    return [Hit(row[0], row[1], row[2], row[3]) for row in db.execute(sql, params)]


def covering_papers(
    db: sqlite3.Connection, query: str, limit: int, tier: str | None = None
) -> list[str]:
    """Distinct paper ids among the top `limit` chunks, best first.

    **Sufficiency counts distinct papers, never scores.** BM25 is corpus-relative: a term in
    more than half the corpus gets degenerate IDF and scores collapse toward zero regardless
    of match quality (measured: a term in 25 of 30 documents scored -0.000). So an absolute
    score threshold would drift as the corpus grows and would be measuring corpus size rather
    than coverage. Counting *papers* measures breadth; counting chunks would measure
    redundancy, since one full-text paper can contribute many chunks.

    This mirrors D-028, which already requires >=3 results before the paper-overlap rule means
    anything. The caller's own threshold on this list is what decides "covered", and that
    number gets measured with O-11 rather than guessed -- choosing it by intuition would be
    `recursion_limit = 150` again (D-077).
    """
    # dict.fromkeys dedups in one pass while preserving first-seen order, which is the ranking
    # order the caller's threshold cuts from. A list with an `in` check would be quadratic and
    # say the same thing less clearly.
    return list(dict.fromkeys(hit.arxiv_id for hit in search(db, query, limit, tier)))


def strongly_matching_papers(
    db: sqlite3.Connection, question: str, limit: int, tier: str | None = None
) -> list[str]:
    """Papers in the top-`limit` that match at least half of `question`'s terms (D-104).

    **This is the sufficiency test, and it replaces the one O-13 specified.** That test --
    "at least `MIN_LOCAL_PAPERS` distinct papers in the top-k" -- was measured against the 80
    committed recordings and **returns k every time, for every query, in every corpus**,
    including 1669 machine-learning papers asked about medieval Flemish guilds. `build_fts_query`
    ORs the terms, so any non-trivial corpus contains something matching something. Counting
    papers measures corpus size, exactly as O-13 warned raw scores would.

    Counting *term coverage* works. A paper matching "attention" alone is noise; one matching
    "attention", "transformer" and "models" is about the subtopic. Measured on that corpus:

    | | papers in top-20 | matching >= half the terms |
    |---|---|---|
    | 20 in-domain questions | always 20 | **5 to 20** |
    | 4 out-of-domain questions | always 20 | **0 to 1** |

    **Half, not all.** `AND` over every term returns nothing on almost every query -- the
    median across 20 questions was **0** even on a same-topic corpus -- so it is too strict to
    be a coverage test at all.

    Takes the raw question rather than a sanitized query because it needs the term *list*, not
    just the MATCH string; sanitization happens here, via the same `search_terms` both
    backends share. Returns `[]` for a question with no searchable terms (D-059).
    """
    try:
        terms = search_terms(question)
        query = build_fts_query(question)
    except ValueError:
        return []

    # **Every term must appear somewhere in the corpus** (D-106). The half-rule alone was
    # validated on whole questions (7-12 terms) but production feeds it *subtopics* (2-4),
    # where "half" can be one generic word. Measured: "CRISPR off-target effects" scored full
    # coverage against a machine-learning corpus, because "target" and "effects" are common ML
    # words while "crispr" matched nothing at all. Requiring the whole vocabulary to be present
    # separates all 10 in-domain probes from all 11 out-of-domain ones with no overlap.
    #
    # Cheap: one LIMIT 1 query per term, short-circuiting on the first absent one.
    if not all(search(db, term, 1, tier) for term in terms):
        return []

    top = covering_papers(db, query, limit, tier)
    if not top:
        return []

    needed = (len(terms) + 1) // 2
    placeholders = ",".join("?" * len(top))
    matches: dict[str, int] = dict.fromkeys(top, 0)
    for term in terms:
        # Restricted to the top-k rather than scanning the whole corpus per term: on a large
        # corpus an unrestricted per-term query returns tens of thousands of rows to discard.
        for row in db.execute(
            "SELECT DISTINCT c.arxiv_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
            f" WHERE chunks_fts MATCH ? AND c.arxiv_id IN ({placeholders})",
            [term, *top],
        ):
            matches[row[0]] += 1

    return [paper for paper in top if matches[paper] >= needed]


def load_sources(db: sqlite3.Connection, arxiv_ids: Sequence[str]) -> list[Source]:
    """Rebuild `Source` objects for the given ids, in the order asked for.

    Returns them as the validated `Source` type rather than raw rows, so a paper that came out
    of the corpus is indistinguishable downstream from one that came off the wire -- the same
    reason `arxiv.py` validates at the innermost step (D-043).
    """
    if not arxiv_ids:
        return []
    placeholders = ",".join("?" * len(arxiv_ids))
    rows = {
        row[0]: row
        for row in db.execute(
            "SELECT arxiv_id, version, title, authors, summary, published, url"
            f" FROM papers WHERE arxiv_id IN ({placeholders})",
            list(arxiv_ids),
        )
    }
    return [
        Source(
            arxiv_id=rows[i][0],
            version=rows[i][1],
            title=rows[i][2],
            authors=tuple(json.loads(rows[i][3])),
            summary=rows[i][4],
            published=datetime.fromisoformat(rows[i][5]),
            url=rows[i][6],
        )
        for i in arxiv_ids
        if i in rows
    ]


def stats(db: sqlite3.Connection) -> dict[str, object]:
    """Corpus size and index dates, for the coverage report (O-13's staleness note).

    arXiv grows by roughly 100 GB a month, so a corpus that answered a subtopic well in March
    silently misses April's key paper. The fix is not to defeat the cache but to *say so* --
    this project's recurring failure shape (D-062, D-069, O-5) arriving somewhere new.
    """
    papers, oldest, newest = db.execute(
        "SELECT COUNT(*), MIN(indexed_at), MAX(indexed_at) FROM papers"
    ).fetchone()
    chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    full_text = db.execute("SELECT COUNT(*) FROM papers WHERE has_full_text").fetchone()[0]
    return {
        "papers": papers,
        "chunks": chunks,
        "full_text_papers": full_text,
        "indexed_from": oldest,
        "indexed_to": newest,
    }
