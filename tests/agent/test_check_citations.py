"""Citation grounding tests (D-046, D-062, D-084). No network, no graph: the node is called directly."""

import pytest

from deep_research.agent.nodes.check_citations import Citation, check_citations
from deep_research.agent.state import ResearchState
from tests.agent.fakes import make_source


def test_citations_of_retrieved_papers_have_no_violations() -> None:
    """Every cited ID is one of the retrieved sources, so nothing is recorded."""
    state = ResearchState(
        question="What is RAG?",
        review="RAG grounds answers in retrieved text [arXiv:2411.18583].",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": [], "citations_checked": True}


def test_invented_citation_is_recorded() -> None:
    """A well-formed ID that wasn't retrieved, e.g. [arXiv:1706.03762], is a violation."""
    state = ResearchState(
        question="What is RAG?",
        review="Transformers rely on attention [arXiv:1706.03762].",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": ["1706.03762"], "citations_checked": True}


def test_each_violation_is_recorded_once_in_order() -> None:
    """Review cites B, A, B (none retrieved): violations == [B, A]."""
    state = ResearchState(
        question="What is RAG?",
        review="First [arXiv:2502.00306], then [arXiv:1706.03762], then [arXiv:2502.00306] again.",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": ["2502.00306", "1706.03762"], "citations_checked": True}


def test_review_without_markers_has_no_violations() -> None:
    """No [arXiv:...] markers at all: {"citation_violations": [], "citations_checked": True}."""
    state = ResearchState(
        question="What is RAG?",
        review="RAG grounds answers in retrieved text.",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": [], "citations_checked": True}


def test_versioned_citation_is_a_violation() -> None:
    """[arXiv:2411.18583v1] isn't the canonical ID (D-044), even though 2411.18583 was retrieved."""
    state = ResearchState(
        question="What is RAG?",
        review="RAG grounds answers in retrieved text [arXiv:2411.18583v1].",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": ["2411.18583v1"], "citations_checked": True}


def test_citation_without_context_fails_loudly() -> None:
    """Missing validation context is a caller bug: RuntimeError, not a ValidationError that
    check_citations would quietly record as a violation."""
    with pytest.raises(RuntimeError, match="validation context"):
        Citation.model_validate({"arxiv_id": "2411.18583"})


# --- D-062: a citation the checker can't parse is recorded, never silently ignored ---


@pytest.mark.parametrize(
    "marker",
    [
        "[arXiv:2411.18583, 2502.00306]",  # grouped IDs; the prompt forbids this form
        "[arXiv: 2411.18583]",  # space after the colon
        "[arXiv:]",  # no ID at all
    ],
)
def test_unparseable_marker_is_recorded_verbatim(marker: str) -> None:
    """A malformed [arXiv:...] bracket becomes a violation, quoted as written (D-062).

    Before D-062 these extracted nothing, so a review citing only in these forms reported
    citation_violations == [] -- indistinguishable from a fully grounded review. A citation
    the checker cannot read must never read as a verified one.
    """
    state = ResearchState(
        question="What is RAG?",
        review=f"RAG grounds answers in retrieved text {marker}.",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": [marker], "citations_checked": True}


def test_malformed_and_invented_citations_are_recorded_together_in_order() -> None:
    """One unparseable marker and one unretrieved ID land in the same list, in reading order (D-062).

    Pins that citation_violations means "every citation that could not be verified", not
    only "well-formed IDs that weren't retrieved".
    """
    state = ResearchState(
        question="What is RAG?",
        review=(
            "Valid [arXiv:2411.18583], grouped [arXiv:A, B], then invented [arXiv:1706.03762]."
        ),
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {
        "citation_violations": ["[arXiv:A, B]", "1706.03762"],
        "citations_checked": True,
    }


def test_repeated_malformed_marker_is_recorded_once() -> None:
    """The same unparseable bracket twice is one violation (D-062).

    Matches how invented IDs are already deduplicated, so a violation count means
    "distinct bad citations", not "occurrences".
    """
    state = ResearchState(
        question="What is RAG?",
        review="First [arXiv: 2411.18583], later [arXiv: 2411.18583] again.",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {
        "citation_violations": ["[arXiv: 2411.18583]"],
        "citations_checked": True,
    }


def test_ordinary_brackets_are_not_citations() -> None:
    """Brackets that aren't [arXiv:...] are ignored, so D-062 adds no false positives (D-062).

    Guards the widened bracket match: a review using [1]-style numbering or bracketed
    asides must still report a clean run when its arXiv markers are all valid.
    """
    state = ResearchState(
        question="What is RAG?",
        review="As noted [1] and [see Table 2], RAG grounds answers [arXiv:2411.18583].",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": [], "citations_checked": True}


def test_known_limit_citation_without_brackets_is_invisible() -> None:
    """A bare "arXiv:<id>" with no brackets is still not seen at all (D-062, known limit).

    D-062 closes the case where a bracket exists but can't be parsed. It does not detect a
    citation written with no bracket, because the review is prose and "arXiv:" can appear in
    ordinary text. This test marks that boundary so it stays a documented limit rather than a
    surprise; the prompt requiring the bracketed form is the only defense.
    """
    state = ResearchState(
        question="What is RAG?",
        review="Transformers rely on attention, arXiv:1706.03762, as shown there.",
        sources=[make_source(arxiv_id="2411.18583")],
    )
    assert check_citations(state) == {"citation_violations": [], "citations_checked": True}


def test_check_citations_marks_the_run_as_checked() -> None:
    """The terminal node sets citations_checked, the only honest "run completed" signal (D-084).

    Without it, an empty citation_violations is ambiguous: it means "checked, found none"
    after this node runs, and "not checked yet" before. A run interrupted between synthesize
    and check_citations would otherwise be reported as finished with zero violations --
    D-062's failure shape, reached a different way.
    """
    state = ResearchState(question="q", review="No citations here.", sources=[])
    assert check_citations(state)["citations_checked"] is True
