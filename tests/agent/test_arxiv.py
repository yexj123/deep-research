"""arXiv client and feed parser tests (D-042, D-044, D-045, D-047, D-051). No network: saved responses only."""

from xml.etree.ElementTree import ParseError

import httpx
import pytest
from defusedxml import DefusedXmlException, DTDForbidden

from deep_research.agent.sources.arxiv import (
    ArxivAPIError,
    build_fts_query,
    build_search_query,
    parse_feed,
    search_arxiv,
    search_terms,
    split_versioned_id,
)
from tests.agent.fakes import load_arxiv_fixture, make_arxiv_stub


# ---- parse_feed ----------------------------------------------------------------------


def test_parse_feed_builds_sources_from_a_real_response() -> None:
    """The real 3-paper response becomes 3 Sources with canonical IDs and nothing skipped."""
    result = parse_feed(load_arxiv_fixture("search_ok.xml"))

    assert [s.arxiv_id for s in result.sources] == ["2411.18583", "2502.00306", "2510.22344"]
    assert [s.version for s in result.sources] == [1, 2, 1]
    assert result.sources[0].authors[0] == "Nurshat Fateh Ali"
    assert result.sources[0].url == "https://arxiv.org/abs/2411.18583v1"
    assert result.skipped == 0


def test_parse_feed_with_no_results_returns_no_sources() -> None:
    """search_empty.xml: sources == () and skipped == 0. Zero results is a success (D-021)."""
    result = parse_feed(load_arxiv_fixture("search_empty.xml"))

    assert result.sources == ()
    assert result.skipped == 0


def test_parse_feed_skips_and_counts_an_invalid_entry() -> None:
    """search_one_invalid_id.xml: 2 sources (2411.18583, 2510.22344) and skipped == 1 (D-045)."""
    result = parse_feed(load_arxiv_fixture("search_one_invalid_id.xml"))

    assert [s.arxiv_id for s in result.sources] == ["2411.18583", "2510.22344"]
    assert result.skipped == 1


def test_parse_feed_rejects_truncated_xml() -> None:
    """truncated.xml raises ParseError, which is not a ValueError (D-048)."""
    with pytest.raises(ParseError) as excinfo:
        parse_feed(load_arxiv_fixture("truncated.xml"))

    assert not isinstance(excinfo.value, ValueError)


def test_parse_feed_rejects_a_doctype() -> None:
    """with_doctype.xml raises DTDForbidden. This is the test that proves forbid_dtd=True is set (D-047)."""
    with pytest.raises(DTDForbidden):
        parse_feed(load_arxiv_fixture("with_doctype.xml"))


def test_parse_feed_rejects_entities() -> None:
    """with_entity.xml raises DTDForbidden, a DefusedXmlException (D-047).

    Not EntitiesForbidden: with forbid_dtd=True, the DOCTYPE holding the entity
    declaration is rejected before the entity is ever looked at (confirmed)."""
    with pytest.raises(DTDForbidden) as excinfo:
        parse_feed(load_arxiv_fixture("with_entity.xml"))

    assert isinstance(excinfo.value, DefusedXmlException)


def test_parse_feed_raises_on_the_error_feed() -> None:
    """error_feed.xml raises ArxivAPIError carrying arXiv's own <summary>."""
    with pytest.raises(ArxivAPIError, match="incorrect id format"):
        parse_feed(load_arxiv_fixture("error_feed.xml"))


# ---- split_versioned_id --------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [
        ("http://arxiv.org/abs/2411.18583v1", ("2411.18583", 1)),
        ("http://arxiv.org/abs/hep-th/9901001v2", ("hep-th/9901001", 2)),
    ],
)
def test_split_versioned_id(entry_id: str, expected: tuple[str, int]) -> None:
    """The prefix and the vN suffix are split off."""
    assert split_versioned_id(entry_id) == expected


@pytest.mark.parametrize("entry_id", ["http://arxiv.org/abs/2411.18583", "2411.18583v1"])
def test_split_versioned_id_rejects_missing_parts(entry_id: str) -> None:
    """No version suffix, or no URL prefix: ValueError."""
    with pytest.raises(ValueError, match="Invalid arXiv entry ID"):
        split_versioned_id(entry_id)


# ---- build_search_query --------------------------------------------------------------


def test_build_search_query_keeps_only_key_terms() -> None:
    """Stopwords and punctuation are dropped, and each remaining term is ANDed (D-051)."""
    query = build_search_query("What is attention in transformer models?")
    assert query == "all:attention AND all:transformer AND all:models"


def test_build_search_query_strips_query_syntax() -> None:
    """Quotes, colons and parentheses can't reach arXiv as query syntax (D-051)."""
    query = build_search_query('RAG: "dense" (retrieval)')
    assert query == "all:rag AND all:dense AND all:retrieval"


def test_build_search_query_rejects_a_question_of_only_stopwords() -> None:
    """Nothing left to search for: ValueError instead of an empty query (D-059)."""
    with pytest.raises(ValueError, match="no meaningful terms"):
        build_search_query("What is it?")


def test_build_search_query_keeps_accented_letters_whole() -> None:
    """"Schrödinger" stays one term instead of splitting into "schr" and "dinger" (D-063).

    The ASCII-only pattern treated every non-ASCII letter as punctuation. Measured against
    the real API on 2026-09-20: all:schrödinger matched 20,184 papers, while the split form
    all:schr AND all:dinger matched 903 whose top hit was an unrelated Korteweg-de Vries
    paper. The old behavior didn't just narrow the search, it matched the wrong papers.
    """
    assert build_search_query("Schrödinger equation") == "all:schrödinger AND all:equation"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Gödel incompleteness", "all:gödel AND all:incompleteness"),
        ("naïve Bayes classifier", "all:naïve AND all:bayes AND all:classifier"),
        ("量子コンピュータ", "all:量子コンピュータ"),
        ("квантовые вычисления", "all:квантовые AND all:вычисления"),
    ],
)
def test_build_search_query_preserves_non_ascii_scripts(question: str, expected: str) -> None:
    """Latin accents, CJK and Cyrillic all survive as terms (D-063).

    arXiv accepts UTF-8 in search_query (confirmed 2026-09-20: httpx percent-encodes it and
    the API returns HTTP 200 with relevant results), so there is no reason to strip them.
    """
    assert build_search_query(question) == expected


def test_build_search_query_still_strips_syntax_from_non_ascii_text() -> None:
    """Widening the pattern must not let arXiv query syntax through (D-063).

    The regression guard for D-063: the point of PUNCTUATION_PATTERN is stripping `:`, `"`
    and parentheses so a question can't alter the query's meaning (D-051). Keeping accented
    letters must not weaken that.
    """
    query = build_search_query('Schrödinger: "cat" (paradox)')
    assert query == "all:schrödinger AND all:cat AND all:paradox"


def test_build_search_query_lowercases_non_ascii_terms() -> None:
    """Non-ASCII terms are lowercased like ASCII ones, so the query is case-stable (D-063).

    Pins that the existing .lower() call is Unicode-aware, which matters now that accented
    characters actually reach it.
    """
    assert build_search_query("SCHRÖDINGER") == "all:schrödinger"


# ---- search_arxiv (MockTransport, no network) ----------------------------------------


@pytest.mark.asyncio
async def test_search_arxiv_sends_the_expected_request() -> None:
    """One GET to the API endpoint, with the query, start and max_results as parameters."""
    stub = make_arxiv_stub(load_arxiv_fixture("search_ok.xml"))
    async with stub.client:
        result = await search_arxiv(stub.client, "all:attention", 3)

    assert len(stub.requests) == 1
    url = stub.requests[0].url
    assert (url.host, url.path) == ("export.arxiv.org", "/api/query")
    assert url.params["search_query"] == "all:attention"
    assert url.params["max_results"] == "3"
    assert url.params["start"] == "0"
    assert len(result.sources) == 3


@pytest.mark.asyncio
async def test_search_arxiv_raises_on_http_400() -> None:
    """Serve error_feed.xml with status_code=400 (as arXiv really does): httpx.HTTPStatusError."""
    stub = make_arxiv_stub(load_arxiv_fixture("error_feed.xml"), status_code=400)
    async with stub.client:
        with pytest.raises(httpx.HTTPStatusError):
            await search_arxiv(stub.client, "all:attention", 3)


# ---- build_fts_query: the second formatter over one extraction (O-13) ----------------


def test_fts_query_ors_the_terms() -> None:
    """OR, not AND, and the asymmetry against arXiv is deliberate (O-13).

    arXiv ANDs to narrow millions of papers (D-051). A personal corpus holds hundreds, so
    ANDing every term returns nothing and "not covered" would mean "the corpus is small"
    rather than "the corpus lacks this topic" -- which would make the sufficiency rule
    measure corpus size instead of coverage.
    """
    assert build_fts_query("What is attention in transformer models?") == (
        "attention OR transformer OR models"
    )


def test_both_formatters_share_one_extraction() -> None:
    """The property that makes "the corpus does not cover this" mean something (O-13).

    If the two backends cleaned differently, a local miss might only mean they disagreed about
    what was being asked, and the sufficiency test would be measuring that disagreement.
    """
    question = "How do Mixture-of-Experts models scale?"
    terms = search_terms(question)
    assert build_fts_query(question) == " OR ".join(terms)
    assert build_search_query(question) == " AND ".join(f"all:{t}" for t in terms)


def test_fts_query_strips_the_operators_that_crash_fts5() -> None:
    """Symbolic FTS5 operators must not survive into a MATCH string (O-13).

    An unstripped quote raises `OperationalError: unterminated string` and crashes the query;
    a surviving `:` would allow a `column:` filter.
    """
    built = build_fts_query('"attention" (transformer) text:model atten*')
    for symbol in ('"', "(", ")", ":", "*"):
        assert symbol not in built, f"{symbol!r} survived into {built!r}"


def test_fts_query_lowercases_the_word_operators() -> None:
    """Lowercasing is a security property here, not normalization (O-13).

    FTS5's word operators are case-sensitive: `attention AND transformer` is an operator
    expression, `attention and transformer` is ordinary terms. Anyone "improving" this by
    preserving the planner's capitalization would silently reintroduce operator injection with
    nothing failing -- which is exactly what this test exists to prevent.
    """
    built = build_fts_query("attention NOT transformer NEAR encoder")
    assert "NOT" not in built and "NEAR" not in built
    assert "not" in built.split(" OR ") and "near" in built.split(" OR ")


def test_fts_query_rejects_a_question_of_only_stopwords() -> None:
    """Fails loudly rather than issuing an empty MATCH (D-023, D-059)."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="no meaningful terms"):
        build_fts_query("what is the")
