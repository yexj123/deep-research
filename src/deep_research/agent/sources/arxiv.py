"""arXiv API client and Atom feed parser (D-042, D-044, D-045, D-047)."""

import re
from dataclasses import dataclass
from xml.etree.ElementTree import Element

import httpx
from defusedxml.ElementTree import fromstring

from deep_research.agent.sources.models import Source

ARXIV_API_URL = "https://export.arxiv.org/api/query"
NAMESPACES: dict[str, str] = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "what", "how", "why", "who",
    "which", "when", "where", "can", "could", "should", "would",
}


# Everything that isn't a letter, digit or whitespace, including arXiv query syntax: " : ( ) (D-051).
# Unicode-aware `\w`, so accented Latin, CJK and Cyrillic letters survive as terms (D-063).
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")

# An entry <id>: abstract URL, canonical ID, "v" + version. Entry <id>s use http:// even when the
# API is called over HTTPS (captured 2026-09-16); https is accepted too. Used with fullmatch.
_ENTRY_ID_PATTERN = re.compile(r"https?://arxiv\.org/abs/(?P<id>.+?)v(?P<version>[1-9]\d*)")


class ArxivAPIError(Exception):
    """arXiv returned its error feed: a single entry titled "Error" (D-045)."""


@dataclass(frozen=True)
class ArxivSearchResult:
    sources: tuple[Source, ...]
    skipped: int  # entries dropped because they failed validation (D-045)


def search_terms(question: str) -> list[str]:
    """The searchable terms of a question: punctuation stripped, stopwords dropped, lowercased.

    Lifted out of `build_search_query` so the same cleaning serves every backend that takes a
    query. Two properties make it safe rather than merely tidy, and both are load-bearing:

    - **Punctuation stripping removes query-syntax characters** for arXiv (`:`, `"`, parens —
      D-051) *and* for SQLite FTS5 (`"`, `*`, `(`, `)`, `:`, `^`). An unstripped quote makes
      FTS5 raise `OperationalError: unterminated string`.
    - **Lowercasing is a security property, not normalization.** FTS5's word operators are
      case-sensitive: `attention AND transformer` is an operator expression, `attention and
      transformer` is three ordinary terms. Preserving the planner's capitalization to look
      tidier would silently reintroduce operator injection, with nothing failing.

    Raises ValueError when nothing is left, e.g. a question made only of stopwords (D-059).
    """
    cleaned = PUNCTUATION_PATTERN.sub(" ", question)
    terms = [token.lower() for token in cleaned.split() if token.lower() not in STOPWORDS]

    if not terms:
        raise ValueError(
            f"Search question {question!r} contains no meaningful terms after stopword removal."
        )
    return terms


def build_search_query(question: str) -> str:
    """Turn the user's question into an arXiv `search_query` string (D-051).

    "What is attention in transformer models?" -> "all:attention AND all:transformer AND all:models"
    """
    return " AND ".join(f"all:{term}" for term in search_terms(question))


def split_versioned_id(entry_id: str) -> tuple[str, int]:
    """Split an entry <id> into (canonical ID, version).

    "http://arxiv.org/abs/2411.18583v1"     -> ("2411.18583", 1)
    "http://arxiv.org/abs/hep-th/9901001v2" -> ("hep-th/9901001", 2)
    """
    match = _ENTRY_ID_PATTERN.fullmatch(entry_id.strip())
    if not match:
        raise ValueError(
            f"Invalid arXiv entry ID: {entry_id!r}. "
            "Must start with 'http(s)://arxiv.org/abs/' and end with a 'v<N>' suffix (e.g., 'v1')."
        )

    canonical_id = match.group("id")
    version = int(match.group("version"))
    return canonical_id, version


def entry_to_source(entry: Element) -> Source:
    """Build a Source from one <entry> element.

    Raises ValueError for a missing element, or ValidationError (a ValueError) for an invalid value.
    """
    raw_id = entry.findtext("atom:id", namespaces=NAMESPACES)
    if not raw_id:
        raise ValueError("Missing required <id> element in arXiv entry.")

    title = entry.findtext("atom:title", namespaces=NAMESPACES)
    if title is None:
        raise ValueError("Missing required <title> element in arXiv entry.")

    summary = entry.findtext("atom:summary", namespaces=NAMESPACES)
    if summary is None:
        raise ValueError("Missing required <summary> element in arXiv entry.")

    published = entry.findtext("atom:published", namespaces=NAMESPACES)
    if not published:
        raise ValueError("Missing required <published> element in arXiv entry.")

    canonical_id, version = split_versioned_id(raw_id)

    author_names = [
        name.strip()
        for elem in entry.findall("atom:author/atom:name", namespaces=NAMESPACES)
        if (name := elem.text) and name.strip()
    ]
    if not author_names:
        raise ValueError("Missing required <author>/<name> elements in arXiv entry.")

    # Only the abstract page. No fallback to "any link": the next one is the PDF (D-058).
    link_elem = entry.find("atom:link[@rel='alternate']", namespaces=NAMESPACES)
    url = link_elem.get("href") if link_elem is not None else None
    if not url:
        raise ValueError("Missing required <link rel='alternate'> href attribute in arXiv entry.")

    return Source(
        arxiv_id=canonical_id,
        version=version,
        title=title,
        authors=author_names,
        summary=summary,
        published=published,
        url=url,
    )


def parse_feed(xml_text: str) -> ArxivSearchResult:
    """Parse an arXiv Atom feed into sources, skipping and counting invalid entries (D-045).

    Raises:
        xml.etree.ElementTree.ParseError: the XML is malformed or truncated.
        defusedxml.DefusedXmlException: the XML contains a DTD or entities (D-047).
        ArxivAPIError: the feed is arXiv's error feed.
    """
    root = fromstring(xml_text, forbid_dtd=True)  # D-047: forbid_dtd=True is not the default
    entries = root.findall("atom:entry", NAMESPACES)

    # Detect arXiv's dedicated error feed (single entry with title "Error")
    if len(entries) == 1:
        entry_title = entries[0].findtext("atom:title", namespaces=NAMESPACES)
        if entry_title and entry_title.strip() == "Error":
            summary = entries[0].findtext("atom:summary", default="", namespaces=NAMESPACES).strip()
            raise ArxivAPIError(summary or "Unknown arXiv API error.")

    sources: list[Source] = []
    skipped_count = 0

    for entry in entries:
        try:
            sources.append(entry_to_source(entry))
        except ValueError:
            skipped_count += 1

    # A tuple, as the field declares: a list would make the frozen result mutable, and `== ()` false.
    return ArxivSearchResult(sources=tuple(sources), skipped=skipped_count)


async def search_arxiv(
    client: httpx.AsyncClient, query: str, max_results: int
) -> ArxivSearchResult:
    """Run one arXiv search. The caller owns `client` (D-049) and the rate limit (1 request / 3 s)."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    response = await client.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    return parse_feed(response.text)
