"""Fetching and chunking arXiv full text, for the corpus's second tier (O-13).

**Why this exists.** D-093 measured `numeric_density` at 0.65 claims per 100 words, with
**five of ten reviews containing no numbers at all** -- because abstracts rarely state the
measurements. D-094 and D-095 then showed that supplying *more abstracts* changes nothing.
So the remaining hypothesis is that better *sources* help where more sources did not, and
full text is the cheapest test of it.

**Legal position (D-008, CLAUDE.md, re-read 2026-09-21).** arXiv prohibits storing and
serving e-prints, and asks that users be directed to arXiv to download. It explicitly permits
*building indexes or tools based on the full text*. This module fetches a PDF, extracts text,
and stores chunks for one user's own retrieval -- an index, not a mirror. Two rules follow and
must not be quietly relaxed:

- **The PDF itself is never stored.** It is parsed in memory and discarded.
- **Nothing here may become a serving endpoint.** If this app is ever deployed for other
  people, the full-text tier has to go, not just be hidden behind a login.

**Why pypdf and not PyMuPDF.** PyMuPDF extracts better and faster, and is **AGPL-3.0** --
anyone deploying this MIT-licensed app over a network would have to release their own source.
That is a licence this repo cannot impose on its users (CLAUDE.md: MIT, settled). pypdf is
BSD-3-Clause, pure Python, and adequate here because arXiv PDFs are LaTeX-generated and carry
a real text layer: measured 26,676 characters from a 6-page paper in 0.5 s.
"""

import io
import re
from dataclasses import dataclass

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Sections whose *body* is citation lists rather than prose. Dropped on purpose: a references
# section names dozens of papers and methods, so indexing it would make every paper match
# every query -- BM25's term frequencies would measure how many works a paper cites rather
# than what it is about. Measured: references run to a third of a 30-page paper's characters.
SKIP_SECTIONS = re.compile(
    r"^(references|bibliography|acknowledge?ments?|appendix|supplementary)", re.I
)

# Two heading styles cover the arXiv papers sampled, and each paper matches exactly one --
# the other returns zero, so "whichever finds more" picks correctly without guessing a venue.
# Measured 2026-09-24 on three papers: IEEE-style Roman found 6/0/0, LaTeX-style Arabic 0/12/6.
HEADING_STYLES = (
    re.compile(r"^\s*(\d{1,2})\.?\s+([A-Z][A-Za-z][A-Za-z \-]{2,40})\s*$", re.M),
    re.compile(r"^\s*([IVX]{1,5})\.?\s+([A-Z][A-Z \-]{3,40})\s*$", re.M),
)

# A chunk is a retrieval unit, not a paragraph. Too large and BM25's term frequency dilutes --
# a 20,000-character section scores the same for one mention as for twenty. Too small and a
# term loses the context that makes it discriminating. Sections are split to this bound rather
# than indexed whole, because a 124,000-character paper over 6 sections averages 20k each.
MAX_CHUNK_CHARS = 2000

# No overlap between chunks, deliberately. Overlap helps dense retrieval at boundaries; for
# FTS5 a term either appears or it does not, and repeating text across chunks would inflate
# its own term frequencies -- the same correctness bug as re-indexing a paper twice (D-100).
PDF_TIMEOUT_SECONDS = 30.0
MAX_PDF_BYTES = 30 * 1024 * 1024


@dataclass(frozen=True)
class Chunk:
    """One indexable piece of a paper, with the section it came from."""

    section: str
    text: str


class FullTextError(RuntimeError):
    """A PDF could not be fetched or read. External data, not a bug (D-048, D-053)."""


async def fetch_pdf(client: httpx.AsyncClient, arxiv_id: str, version: int = 1) -> bytes:
    """Download one paper's PDF. Held in memory, never written to disk.

    The caller holds the rate limiter around this: D-064's one-request-per-three-seconds
    applies to downloads exactly as it does to searches, and a PDF is a much larger request
    than a metadata query.

    Raises `FullTextError` for anything external -- a 404 for a withdrawn paper, a transport
    error, a response too large to be a paper. Those are data problems, and the corpus simply
    has no full text for that paper.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"
    try:
        response = await client.get(
            url,
            timeout=PDF_TIMEOUT_SECONDS,
            follow_redirects=True,  # arXiv redirects /pdf/ to a versioned or mirrored URL
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.StreamError) as exc:
        raise FullTextError(f"could not fetch {url}: {exc}") from exc

    if len(response.content) > MAX_PDF_BYTES:
        raise FullTextError(f"{url} is {len(response.content)} bytes, over the {MAX_PDF_BYTES} cap")
    return response.content


def extract_text(pdf: bytes) -> str:
    """Text layer of a PDF, pages joined. Raises `FullTextError` if it cannot be read.

    An empty result is returned as an error rather than an empty string: arXiv PDFs are
    LaTeX-generated and always have a text layer, so nothing extracted means a scanned or
    malformed file, not a paper with no words. Silently indexing "" would add a paper to the
    full-text tier that contributes nothing and can never be matched -- present in the count,
    absent from every search.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError, OSError) as exc:
        raise FullTextError(f"could not read PDF: {exc}") from exc

    if not text.strip():
        raise FullTextError("PDF yielded no text layer (scanned or malformed)")
    return _drop_surrogates(text)


def _drop_surrogates(text: str) -> str:
    """Remove lone surrogates, which are extraction artifacts and cannot be stored.

    **Found the hard way (D-109).** pypdf renders some glyphs -- mathematical bold, certain
    ligatures -- as unpaired surrogates like `\\ud835`. Python holds those in a `str` happily,
    but they are not encodable UTF-8, so SQLite raises `UnicodeEncodeError` on insert. That
    killed an entire 360-paper enrichment pass at paper 126: **one malformed glyph in one PDF
    cost every paper after it**, which is precisely the shape D-053 exists to prevent -- bad
    external data must become data, never an exception that takes down the run.

    Dropped rather than replaced with U+FFFD: these are decoding noise carrying no meaning, and
    a replacement character would be an indexable token that matches nothing and appears in
    any excerpt shown to a reader.

    The fast path costs one `str.isascii()` check, so the overwhelming majority of papers pay
    almost nothing for it.
    """
    if text.isascii():
        return text
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split extracted text into `(heading, body)` pairs, references dropped.

    Picks whichever heading style matches more, because each paper follows one convention and
    the other scores zero -- so the choice needs no knowledge of the venue.

    Text before the first heading (title, authors, abstract) is kept under the heading
    `"frontmatter"`: it holds the title and abstract, which are the most query-like text in
    the paper. Returning `[("body", text)]` when no heading is found keeps every paper
    indexable rather than making chunking depend on parsing succeeding.
    """
    best: list[re.Match[str]] = []
    for pattern in HEADING_STYLES:
        found = list(pattern.finditer(text))
        if len(found) > len(best):
            best = found

    if not best:
        return [("body", text)]

    sections: list[tuple[str, str]] = []
    if text[: best[0].start()].strip():
        sections.append(("frontmatter", text[: best[0].start()]))

    for index, match in enumerate(best):
        heading = match.group(2).strip()
        end = best[index + 1].start() if index + 1 < len(best) else len(text)
        body = text[match.end() : end]
        if SKIP_SECTIONS.match(heading) or not body.strip():
            continue
        sections.append((heading, body))
    return sections


def chunk_paper(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a paper into indexable chunks, each labelled with its section.

    Splitting happens on blank lines where possible so a chunk ends at a paragraph rather than
    mid-sentence; a single paragraph longer than `max_chars` is cut at the bound, because one
    runaway paragraph must not produce one runaway chunk.

    The section label carries the ordinal when a section is split (`"Methodology (2)"`), so a
    chunk can be traced back to where in the paper it came from -- provenance the abstract
    tier does not need but a claim checker (O-12) will.
    """
    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        pieces = _split_to_size(body, max_chars)
        for ordinal, piece in enumerate(pieces, start=1):
            label = heading if len(pieces) == 1 else f"{heading} ({ordinal})"
            chunks.append(Chunk(section=label, text=piece))
    return chunks


def _split_to_size(body: str, max_chars: int) -> list[str]:
    """Paragraph-aligned pieces of at most `max_chars`, empty pieces dropped."""
    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            # One paragraph bigger than the bound: cut it rather than emit an oversized chunk.
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if len(current) + len(paragraph) + 2 > max_chars and current:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces
