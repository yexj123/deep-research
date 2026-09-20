"""arXiv client and feed parser tests (D-042, D-044, D-045, D-047, D-051). No network: saved responses only."""

from xml.etree.ElementTree import ParseError

import httpx
import pytest
from defusedxml import DefusedXmlException, DTDForbidden

from deep_research.agent.sources.arxiv import (
    ArxivAPIError,
    build_search_query,
    parse_feed,
    search_arxiv,
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
