"""Coverage summary tests (O-5). Pure functions over state: no graph, no network.

What these pin down is the project's recurring failure shape in its last remaining form: a
review missing a third of its subtopics currently reads exactly like a complete one. D-062
(unreadable citations reported as clean), D-063 (a shredded query returning confident
nonsense) and D-069 (an empty fan-out ending the run silently) were the same mistake. This is
the summary that makes the loss visible.
"""

import pytest

from deep_research.agent.config import MAX_DEPTH
from deep_research.agent.coverage import summarize_coverage
from tests.agent.fakes import make_source


def values(**overrides):
    base = {
        "explored_subtopics": ["attention mechanisms", "positional encoding"],
        "empty_subtopics": [],
        "failed_subtopics": [],
        "skipped_entries": 0,
        "sources": [make_source(arxiv_id="2411.18583")],
        "depth": 1,
    }
    return {**base, **overrides}


# ---- what was lost -------------------------------------------------------------------


def test_a_clean_run_is_complete() -> None:
    """Nothing failed, nothing empty, nothing skipped: is_complete is True (O-5).

    The UI shows the coverage panel only when this is False, so a clean run isn't cluttered
    with a list of nothing.
    """
    assert summarize_coverage(values()).is_complete


def test_a_subtopic_that_found_nothing_is_reported() -> None:
    """A zero-result subtopic is a finding, not a silent no-op (D-021, O-5).

    This is the loss that was previously unrecorded anywhere: zero results marks a subtopic
    explored, so it read identically to a productive one.
    """
    coverage = summarize_coverage(values(empty_subtopics=["quantum tokenizers"]))

    assert coverage.empty == ["quantum tokenizers"]
    assert not coverage.is_complete


def test_failed_subtopics_are_reported_with_their_attempt_counts() -> None:
    """Each entry in failed_subtopics is one attempt, so duplicates are the count (D-020)."""
    coverage = summarize_coverage(
        values(failed_subtopics=["flaky topic", "flaky topic", "other topic"])
    )

    assert coverage.failed == {"flaky topic": 2, "other topic": 1}
    assert not coverage.is_complete


def test_failures_are_counted_on_the_normalized_subtopic() -> None:
    """"Flaky topic" and "  FLAKY TOPIC " are one subtopic, as decompose treats them (D-020, D-073).

    Counting raw strings would report two separate failures for something the agent itself
    considers a single subtopic that spent both retry attempts.
    """
    coverage = summarize_coverage(values(failed_subtopics=["Flaky topic", "  FLAKY TOPIC "]))

    assert len(coverage.failed) == 1
    assert next(iter(coverage.failed.values())) == 2


def test_the_reported_spelling_is_the_planners_original_wording() -> None:
    """Normalized for counting, but reported as written (D-067).

    The same reason explored_subtopics stores original text: "flaky topic" in a report reads
    as sloppy when the planner wrote "Flaky Topic".
    """
    coverage = summarize_coverage(values(failed_subtopics=["Flaky Topic", "flaky topic"]))
    assert list(coverage.failed) == ["Flaky Topic"]


def test_skipped_entries_are_reported() -> None:
    """Malformed arXiv entries were counted but never read by anything (D-045, O-5)."""
    coverage = summarize_coverage(values(skipped_entries=3))

    assert coverage.skipped_entries == 3
    assert not coverage.is_complete


# ---- why the run stopped -------------------------------------------------------------


def test_hitting_the_depth_limit_says_more_might_exist() -> None:
    """Depth exhausted means the run was cut off, not that the search converged (D-076).

    Reporting this identically to "converged" would hide the difference that matters most to
    a reader deciding whether to trust the review's completeness.
    """
    reason = summarize_coverage(values(depth=MAX_DEPTH + 1)).stopped_because

    assert "depth limit" in reason
    assert "might have found more" in reason


def test_converging_says_the_round_found_nothing_new() -> None:
    """The semantic exit (D-075): the search stopped because it had stopped learning."""
    reason = summarize_coverage(values(depth=1)).stopped_because

    assert "no papers that earlier rounds hadn't already seen" in reason
    assert "depth limit" not in reason


def test_a_run_with_no_plan_says_so() -> None:
    """depth == 0 means route_subtopics went straight to synthesize (D-069).

    Distinct from both other reasons: nothing was researched at all, which a reader must not
    have to infer from an empty subtopic list.
    """
    assert "proposed no subtopics" in summarize_coverage(values(depth=0)).stopped_because


@pytest.mark.parametrize("depth", [0, 1, MAX_DEPTH + 1])
def test_every_run_reports_a_reason(depth: int) -> None:
    """There is no run whose ending is unexplained (O-5)."""
    assert summarize_coverage(values(depth=depth)).stopped_because


def test_missing_fields_do_not_crash_the_summary() -> None:
    """A not-started run has an almost-empty state dict (D-081).

    GET /runs/{id} summarizes coverage whatever the run's status, so this must degrade to
    zeros rather than raising.
    """
    coverage = summarize_coverage({})

    assert coverage.papers == 0
    assert coverage.explored == []
    assert coverage.stopped_because
