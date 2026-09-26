"""The recency metric's date logic (O-16).

This is O-16's **primary** metric and the only one in the repo that does not depend on a
judge, so its arithmetic carries more weight than usual: a quiet off-by-one here would
misreport the headline number with nothing else to contradict it.

The interesting cases are all boundaries -- the seed month, the old id scheme, and the
year rollover that a naive string comparison gets wrong.
"""

from datetime import UTC, datetime

import pytest

from tests.eval.recency import after, cited_ids, submitted

SEED = datetime(2026, 9, 25, 17, 8, tzinfo=UTC)


@pytest.mark.parametrize(
    ("arxiv_id", "expected"),
    [
        ("2509.04244", (2025, 9)),
        ("2411.18583", (2024, 11)),
        ("2601.00001", (2026, 1)),
        ("2612.99999", (2026, 12)),
        ("2411.1858", (2024, 11)),  # 4-digit sequence, the older width (D-056)
    ],
)
def test_the_id_encodes_its_submission_month(arxiv_id: str, expected: tuple[int, int]) -> None:
    """YYMM comes from the id, not from stored metadata.

    It has to: the fresh arm cites papers held in no local record, so a metric reading
    `published` could only ever measure the stale arm.
    """
    assert submitted(arxiv_id) == expected


def test_the_old_id_scheme_has_no_parseable_date() -> None:
    """Pre-2007 ids like "hep-th/9901001" carry no YYMM, and must return None rather than
    guess. They are counted in their own column so they cannot silently inflate either side."""
    assert submitted("hep-th/9901001") is None
    assert after("hep-th/9901001", SEED) is None


def test_a_paper_from_after_the_seed_month_counts() -> None:
    """October 2026 is unambiguously after a September 2026 seed -- the case the November
    run is built on."""
    assert after("2610.00123", SEED) is True
    assert after("2701.00123", SEED) is True, "year rollover, which string compare gets wrong"


def test_a_paper_from_before_the_seed_does_not_count() -> None:
    assert after("2608.00123", SEED) is False
    assert after("2411.18583", SEED) is False


def test_the_seed_month_itself_counts_as_not_post_seed() -> None:
    """The one-month resolution limit, pinned deliberately rather than left implicit.

    A paper submitted 2026-09-26 -- the day after the seed -- carries the same `2609.` prefix
    as one from 2026-09-01, so the two are indistinguishable. Counting them as NOT post-seed
    undercounts the fresh arm's advantage, which is the conservative direction: it can make
    staleness look smaller than it is, never larger. If this ever flips to True, the headline
    number starts overstating the effect.
    """
    assert after("2609.99999", SEED) is False


def test_citations_are_read_from_the_review_text() -> None:
    """The ids come from the review's markers, because a `Recording` stores no id list.

    An earlier version of the purity check read a `synthesized_from` key that does not exist
    on the dataclass; every question then reported zero cited papers and the arm was declared
    clean **vacuously**. Parsing the text is what the data actually supports.
    """
    review = "Tiling helps [arXiv:2411.18583]. Routing differs [arXiv:2502.00306] [arXiv:2411.18583]."
    assert cited_ids(review) == {"2411.18583", "2502.00306"}


def test_a_review_citing_nothing_yields_no_ids() -> None:
    """The zero-sources path (D-060) must produce an empty set, not an error."""
    assert cited_ids("No papers were retrieved for this question.") == set()
