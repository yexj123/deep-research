"""Fetching and chunking arXiv full text (O-13, D-108).

**No arXiv PDF is committed as a fixture, and that is a licence decision, not a size one.**
arXiv permits building indexes over full text and prohibits storing and serving e-prints
(D-008). A PDF checked into a public MIT repository is redistribution, whatever it is labelled.
So the tests use two things instead:

- `make_pdf()`, which builds a real 563-byte PDF from scratch, for the extraction path.
- `tests/agent/fixtures/fulltext/*.txt`, the *extracted text* of two real papers, trimmed to
  12,000 characters. Text extracted for indexing is the thing arXiv's terms permit; it is also
  what the chunker actually consumes.

The two fixtures are deliberately different heading conventions -- IEEE Roman numerals and
LaTeX Arabic -- because a chunker that only handles one silently indexes the other as a single
undifferentiated blob.
"""

import io
from pathlib import Path

import httpx
import pytest

from deep_research.agent.sources.fulltext import (
    MAX_CHUNK_CHARS,
    FullTextError,
    chunk_paper,
    extract_text,
    fetch_pdf,
    split_sections,
)

FIXTURES = Path(__file__).parent / "fixtures" / "fulltext"
IEEE = (FIXTURES / "ieee-style.txt").read_text(encoding="utf-8")
LATEX = (FIXTURES / "latex-style.txt").read_text(encoding="utf-8")


def make_pdf(text: str) -> bytes:
    """A genuine minimal PDF carrying `text`, built rather than committed.

    Real enough that pypdf parses it through the same path a downloaded paper takes -- a
    stubbed-out byte string would test nothing about extraction.
    """
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(stream), stream),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj" % index + body + b"endobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---- fetching ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_pdf_is_fetched_from_the_versioned_url() -> None:
    """The version matters: /pdf/1234.5678v2 is a different document from v1.

    The corpus stores `version` (D-103) precisely so the right one can be fetched.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=make_pdf("hello"))

    async with _client(handler) as client:
        await fetch_pdf(client, "2411.18583", version=2)

    assert seen == ["https://arxiv.org/pdf/2411.18583v2"]


@pytest.mark.asyncio
async def test_a_withdrawn_paper_is_external_data_not_a_bug() -> None:
    """A 404 means the corpus has no full text for that paper, not that the run is broken.

    Same split as the arXiv search path (D-065): external failures become data, and only
    genuine programming errors crash.
    """
    async with _client(lambda _: httpx.Response(404)) as client:
        with pytest.raises(FullTextError):
            await fetch_pdf(client, "2411.18583")


@pytest.mark.asyncio
async def test_a_transport_error_is_external_data_too() -> None:
    """A dropped connection is arXiv's problem, handled the same way as a 404."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset")

    async with _client(handler) as client:
        with pytest.raises(FullTextError):
            await fetch_pdf(client, "2411.18583")


@pytest.mark.asyncio
async def test_an_oversized_response_is_refused() -> None:
    """A cap, because this is held in memory and a run fans out in parallel (D-067).

    Three workers each pulling an unbounded response is how a research tool becomes an
    out-of-memory crash on someone else's machine.
    """
    huge = b"%PDF-1.4\n" + b"\0" * (31 * 1024 * 1024)
    async with _client(lambda _: httpx.Response(200, content=huge)) as client:
        with pytest.raises(FullTextError, match="over the"):
            await fetch_pdf(client, "2411.18583")


# ---- extraction ----------------------------------------------------------------------


def test_text_is_extracted_from_a_real_pdf() -> None:
    """The happy path, through pypdf's actual parser rather than a stub."""
    assert "tiling" in extract_text(make_pdf("Attention is tiling and quantization"))


def test_a_pdf_with_no_text_layer_fails_loudly() -> None:
    """A scanned paper must not be indexed as an empty full-text entry.

    Silently storing "" would add a paper to the full-text tier that contributes nothing and
    can never be matched -- counted as covered, absent from every search. That is the
    invisible-loss shape this project keeps finding (D-021, O-5).
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(FullTextError, match="no text layer"):
        extract_text(buffer.getvalue())


def test_a_malformed_pdf_fails_loudly() -> None:
    """Truncated or non-PDF bytes are external data, not an unhandled crash."""
    with pytest.raises(FullTextError):
        extract_text(b"this is not a pdf at all")


# ---- section splitting, on two real heading conventions -------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [(IEEE, "INTRODUCTION"), (LATEX, "Introduction")],
    ids=["ieee-roman", "latex-arabic"],
)
def test_both_heading_conventions_are_split(text: str, expected: str) -> None:
    """Each paper follows one convention and scores zero on the other (D-108).

    That is what lets "whichever style matches more" pick correctly without knowing the venue.
    A chunker handling only one convention would index the other as a single blob and lose
    every section label.
    """
    headings = [heading for heading, _ in split_sections(text)]
    assert expected in headings
    assert len(headings) >= 3, headings


@pytest.mark.parametrize("text", [IEEE, LATEX], ids=["ieee", "latex"])
def test_the_title_and_abstract_are_kept_as_frontmatter(text: str) -> None:
    """Text before the first heading is the most query-like in the paper.

    Dropping it would discard the title -- the one place a method's name reliably appears,
    which is the same reason the abstract tier indexes title plus summary (D-100).
    """
    sections = dict(split_sections(text))
    assert "frontmatter" in sections
    assert sections["frontmatter"].strip()


def test_references_are_dropped() -> None:
    """A references section is a list of other papers' titles (D-108).

    Indexing it would make every paper match every query: BM25 would be measuring how many
    works a paper cites rather than what it is about.
    """
    text = (
        "1 Introduction\nWe study tiling.\n\n"
        "2 Method\nWe tile attention.\n\n"
        "3 References\n[1] Vaswani et al. Attention is all you need.\n"
    )
    headings = [heading for heading, _ in split_sections(text)]
    assert "Introduction" in headings and "Method" in headings
    assert not any("eference" in heading for heading in headings)


def test_a_paper_with_no_detectable_headings_is_still_indexable() -> None:
    """Parsing failure must degrade, not discard (D-021).

    A paper whose headings this cannot read is still worth searching; returning nothing would
    silently drop it from the corpus.
    """
    assert split_sections("Just prose with no headings at all.") == [
        ("body", "Just prose with no headings at all.")
    ]


# ---- chunking ------------------------------------------------------------------------


@pytest.mark.parametrize("text", [IEEE, LATEX], ids=["ieee", "latex"])
def test_no_chunk_exceeds_the_bound(text: str) -> None:
    """The bound is what keeps BM25 term frequencies meaningful (D-108).

    A 20,000-character section scores the same for one mention of a term as for twenty, so an
    unbounded chunk would rank on length rather than relevance.
    """
    chunks = chunk_paper(text)
    assert chunks
    assert max(len(chunk.text) for chunk in chunks) <= MAX_CHUNK_CHARS


@pytest.mark.parametrize("text", [IEEE, LATEX], ids=["ieee", "latex"])
def test_every_chunk_carries_a_section_label(text: str) -> None:
    """Provenance: a claim checker (O-12) needs to say *where* in the paper a claim came from."""
    assert all(chunk.section.strip() for chunk in chunk_paper(text))


def test_a_split_section_is_numbered() -> None:
    """A long section becomes several chunks, and they must stay distinguishable.

    Without the ordinal, three chunks of "Methodology" are indistinguishable in the corpus and
    `UNIQUE(arxiv_id, tier, section)` would collapse them to one (D-100).
    """
    long_section = "1 Methodology\n\n" + "\n\n".join(["Sentence about tiling."] * 400)
    labels = [chunk.section for chunk in chunk_paper(long_section)]

    assert len(labels) > 1
    assert labels[0] == "Methodology (1)"
    assert len(set(labels)) == len(labels), "section labels must be unique within a paper"


def test_chunks_do_not_overlap() -> None:
    """No overlap, deliberately (D-108).

    Overlap helps dense retrieval at boundaries; for FTS5 a term either appears or it does
    not, and repeating text across chunks would inflate its own term frequencies -- the same
    correctness bug as indexing a paper twice (D-100).
    """
    text = "1 Method\n\n" + "\n\n".join(f"Paragraph {i} about tiling." for i in range(200))
    joined = "".join(chunk.text for chunk in chunk_paper(text))

    assert joined.count("Paragraph 7 about tiling.") == 1


def test_an_oversized_paragraph_is_cut_rather_than_emitted_whole() -> None:
    """One runaway paragraph must not produce one runaway chunk.

    Extracted PDF text frequently loses paragraph breaks, so "split on blank lines" alone
    would leave a whole section as a single chunk on exactly the papers that extract worst.
    """
    chunks = chunk_paper("1 Method\n\n" + "x " * (MAX_CHUNK_CHARS * 2))
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= MAX_CHUNK_CHARS


def test_unpaired_surrogates_are_removed(  ) -> None:
    """pypdf emits lone surrogates for some glyphs, and SQLite cannot store them (D-109).

    Found the hard way: one mathematical-bold character in one PDF raised
    `UnicodeEncodeError` on insert and killed a 360-paper enrichment pass at paper 126 --
    an hour of rate-limited downloading lost to one glyph. Bad external data must become
    data, never an exception that takes down the run (D-053).
    """
    from deep_research.agent.sources.fulltext import _drop_surrogates

    cleaned = _drop_surrogates("bold \ud835 math and \ud83d more")
    assert "\ud835" not in cleaned
    assert cleaned.encode("utf-8"), "the result must be storable"
    assert "bold" in cleaned and "math" in cleaned, "surrounding text must survive"


def test_real_unicode_is_not_collateral_damage() -> None:
    """Stripping surrogates must not strip legitimate non-ASCII.

    D-063 already cost this project a bug where cleaning destroyed non-ASCII terms; author
    names and titles routinely carry accents and CJK, and losing them would make those papers
    unsearchable by the very words that identify them.
    """
    from deep_research.agent.sources.fulltext import _drop_surrogates

    assert _drop_surrogates("naïve café 東京 Müller") == "naïve café 東京 Müller"


def test_a_truncated_extraction_is_a_known_limitation() -> None:
    """pypdf can stop extracting partway and only *warn* (D-116).

    "Exceeded 5000 form XObject invocations while extracting text; further form content is
    skipped" appeared on 3 of 362 papers. It is a log line, not an exception, so extract_text
    returns a truncated string that looks complete and the paper is indexed as fully read.

    This test does not assert a fix -- there is none, and raising the limit trades memory
    against completeness on exactly the figure-heavy papers already near the size cap. It
    pins the *contract*: extraction returns whatever text it got, and callers must not treat
    a non-empty result as proof of a complete read.
    """
    text = extract_text(make_pdf("A short paper"))
    assert text, "a non-empty result means SOME text, never necessarily ALL of it"
