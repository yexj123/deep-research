"""Source model tests (D-041, D-043, D-044, D-058, D-061). No network."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from tests.agent.fakes import make_source


def test_valid_new_scheme_id_builds_a_source() -> None:
    """A canonical new-scheme ID (no version) is accepted unchanged."""
    source = make_source(arxiv_id="2411.18583")
    assert source.arxiv_id == "2411.18583"


def test_four_digit_new_scheme_id_is_accepted() -> None:
    """IDs from 0704 to 1412 have 4 digits after the dot, e.g. "0706.0001"."""
    assert make_source(arxiv_id="0706.0001").arxiv_id == "0706.0001"


@pytest.mark.parametrize("arxiv_id", ["hep-th/9901001", "math.GT/0309136"])
def test_old_scheme_ids_are_accepted(arxiv_id: str) -> None:
    """Pre-2007 IDs: archive, optional .XX subject class, "/", 7 digits."""
    assert make_source(arxiv_id=arxiv_id).arxiv_id == arxiv_id


def test_versioned_id_is_rejected() -> None:
    """"2411.18583v1" raises ValidationError: the version belongs in `version` (D-044)."""
    with pytest.raises(ValidationError, match="not a canonical arXiv ID"):
        make_source(arxiv_id="2411.18583v1")


@pytest.mark.parametrize(
    "arxiv_id",
    ["", "not-a-real-id", "2411.185", "http://arxiv.org/abs/2411.18583"],
)
def test_malformed_ids_are_rejected(arxiv_id: str) -> None:
    """Each raises ValidationError with the validator's own message."""
    with pytest.raises(ValidationError, match="not a canonical arXiv ID"):
        make_source(arxiv_id=arxiv_id)


def test_authors_list_is_stored_as_a_tuple() -> None:
    """make_source(authors=["A", "B"]).authors is the tuple ("A", "B"): Pydantic converts it (D-043)."""
    source = make_source(authors=["A", "B"])
    assert isinstance(source.authors, tuple)
    assert source.authors == ("A", "B")


def test_source_is_frozen() -> None:
    """Assigning to a field raises dataclasses.FrozenInstanceError (not a Pydantic error)."""
    source = make_source()
    with pytest.raises(FrozenInstanceError):
        source.title = "changed"  # type: ignore[misc]


def test_version_below_one_is_rejected() -> None:
    """arXiv numbers versions from 1, so v0 can't come from a real <id> (D-058)."""
    with pytest.raises(ValidationError, match="version must be >= 1"):
        make_source(version=0)


def test_title_whitespace_is_collapsed() -> None:
    """Line breaks and runs of spaces in a title become single spaces (D-058)."""
    assert make_source(title="  Attention Is\n  All You Need ").title == "Attention Is All You Need"


def test_blank_title_is_rejected() -> None:
    """A title that is only whitespace is rejected, so the entry is skipped (D-045, D-058)."""
    with pytest.raises(ValidationError, match="title cannot be empty"):
        make_source(title=" \n ")


def test_unknown_field_is_rejected() -> None:
    """An unknown keyword raises instead of being silently dropped (D-061).

    Pydantic dataclasses default to extra="ignore", which let Source(..., abstract=...)
    construct successfully with `abstract` discarded. That is the shape of the 2026-09-19
    `id=`/`abstract=` bug, and it was only caught then because the typo also removed a
    required field. config=ConfigDict(extra="forbid") makes the typo itself the error.
    """
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        make_source(abstract="not a field on Source")


def test_unknown_field_error_names_the_offending_field() -> None:
    """The ValidationError points at the bad keyword by name (D-061).

    The reason for forbidding extras is diagnostic: parse_feed counts any ValidationError
    as a skipped entry (D-045), so the error text is what tells you which field name is
    wrong when the `skipped == 0` assertion in test_arxiv.py goes red.
    """
    with pytest.raises(ValidationError) as exc_info:
        make_source(abstract="not a field on Source")

    errors = exc_info.value.errors()
    assert [error["type"] for error in errors] == ["unexpected_keyword_argument"]
    assert [error["loc"] for error in errors] == [("abstract",)]
