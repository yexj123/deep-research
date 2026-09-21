"""Routing and reporting must never disagree about why a run stopped (D-096).

This file exists because the same bug happened twice. `graph.route_after_gap_check` decided
when to stop; `coverage._stop_reason` kept its own copy of the rules to pick a sentence, and
the copies drifted:

- **D-094** — `coverage.py` compared against an unpatched `MAX_DEPTH`, so runs that hit their
  ceiling were recorded as having *converged*. Twenty recordings claimed the semantic exit had
  fired in the very arms built to prove it never does.
- **D-096** — two new exits were added to the router; `_stop_reason` knew nothing about them,
  so every adaptive run reported "round 1 found no papers that earlier rounds hadn't already
  seen" when it had in fact stopped with a full synthesis prompt.

Neither crashed. Both produced a confident, plausible, wrong sentence in the panel a reader
uses to judge how far to trust a review — this project's recurring failure shape (D-062,
D-069, D-084), aimed at the reader instead of the run.

The fix was structural: one `exits.exit_reason`, called by both. These tests pin the property
that made the fix worth making, so a third copy cannot quietly appear.
"""

import pytest

from deep_research.agent.config import MAX_DEPTH, MAX_SUBTOPICS, SYNTHESIS_TOP_N
from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.exits import (
    DEPTH_CEILING,
    NO_NEW_PAPERS,
    PROMPT_FULL,
    REASONS,
    ROUND_EMPTY,
    describe,
    exit_reason,
)
from deep_research.agent.graph import route_after_gap_check
from deep_research.agent.state import ResearchState
from tests.agent.fakes import make_source

# Each case is a state plus the exit it must trigger. Built as data so the router, the reason
# and the reported sentence are all checked against the *same* state -- the thing that was
# not true before.
CASES: dict[str, dict] = {
    DEPTH_CEILING: {"depth": MAX_DEPTH + 1, "n_sources": 1},
    NO_NEW_PAPERS: {"depth": 1, "n_sources": 3, "seen_before_round": 3},
    PROMPT_FULL: {"depth": 1, "n_sources": SYNTHESIS_TOP_N},
    ROUND_EMPTY: {"depth": 1, "n_sources": 2, "n_empty": MAX_SUBTOPICS},
}


def _ids(n: int) -> list[str]:
    return [f"2411.{10000 + i:05d}" for i in range(n)]


def _values(*, depth: int, n_sources: int, seen_before_round: int = 0, n_empty: int = 0) -> dict:
    """A state dict as a checkpoint holds it (D-030), so coverage and the router see one truth."""
    ids = _ids(n_sources)
    return {
        "question": "What is attention?",
        "depth": depth,
        "seen_paper_ids": set(ids),
        "seen_before_round": seen_before_round,
        "sources": [make_source(arxiv_id=i) for i in ids],
        "pending_subtopics": [f"subtopic {i}" for i in range(MAX_SUBTOPICS)],
        "empty_subtopics": [f"empty {i}" for i in range(n_empty)],
        "empty_before_round": 0,
    }


def _call(values: dict) -> str | None:
    return exit_reason(
        depth=values["depth"],
        seen_count=len(values["seen_paper_ids"]),
        seen_before_round=values["seen_before_round"],
        source_count=len(values["sources"]),
        dispatched=len(values["pending_subtopics"]),
        empties_this_round=len(values["empty_subtopics"]) - values["empty_before_round"],
    )


@pytest.mark.parametrize("expected", list(CASES), ids=list(CASES))
def test_each_exit_fires_on_the_state_built_for_it(expected: str) -> None:
    """The four exits are individually reachable.

    Without this, the consistency checks below could all pass with a single rule doing every
    job -- which is exactly the state the code was in before D-096.
    """
    assert _call(_values(**CASES[expected])) == expected


@pytest.mark.parametrize("expected", list(CASES), ids=list(CASES))
def test_the_router_stops_whenever_an_exit_fires(expected: str) -> None:
    """Any reason at all means "synthesize"; no reason means another round.

    Pins that `route_after_gap_check` is a thin translation of `exit_reason` and holds no
    rules of its own -- the property that keeps the two from drifting a third time.
    """
    values = _values(**CASES[expected])
    assert route_after_gap_check(ResearchState(**values)) == "synthesize"


@pytest.mark.parametrize("expected", list(CASES), ids=list(CASES))
def test_coverage_reports_the_exit_that_actually_fired(expected: str) -> None:
    """The sentence a reader sees must name the rule that stopped the run (D-094, D-096).

    This is the assertion both historical bugs would have failed: each produced a fluent,
    confident sentence describing an exit that had not fired.
    """
    values = _values(**CASES[expected])
    reported = summarize_coverage(values).stopped_because
    assert reported == describe(expected, depth=values["depth"])
    assert reported, "a finished run must always say why it stopped"


def test_a_ceiling_stop_warns_that_the_run_was_cut_off() -> None:
    """The distinction O-5 exists to preserve (D-086).

    "The depth limit cut this off" and "the search was already complete" are opposite claims
    about whether to trust the review's coverage. Reporting them in the same words is the
    failure this whole module guards.
    """
    cut_off = describe(DEPTH_CEILING, depth=MAX_DEPTH + 1)
    assert "depth limit" in cut_off and "might have found more" in cut_off
    for complete in (REASONS[PROMPT_FULL], REASONS[ROUND_EMPTY], REASONS[NO_NEW_PAPERS]):
        assert "might have found more" not in complete


def test_a_productive_round_under_the_ceiling_continues() -> None:
    """The control case: no exit fires, so the run recurses.

    Every test above asserts something stops. If nothing could ever continue, they would all
    pass against a router hard-wired to "synthesize".
    """
    values = _values(depth=1, n_sources=SYNTHESIS_TOP_N - 1)
    assert _call(values) is None
    assert route_after_gap_check(ResearchState(**values)) == "decompose"


def test_an_unfinished_run_does_not_claim_a_reason() -> None:
    """A resumable run has not stopped, and must not be described as though it had (D-084).

    `next == ()` already proved an unreliable "finished" signal once; a coverage panel that
    invents a stop reason for an interrupted run would repeat that mistake in the UI.
    """
    assert describe(None, depth=1) == "the run has not finished"


def test_every_reason_constant_has_a_sentence() -> None:
    """A new exit without a message would render as a KeyError in the coverage panel.

    Cheap insurance: the next exit added gets caught here rather than in a browser.
    """
    for reason in (NO_NEW_PAPERS, PROMPT_FULL, ROUND_EMPTY):
        assert REASONS[reason].strip()
    # DEPTH_CEILING is formatted rather than looked up, so it is checked via describe().
    assert describe(DEPTH_CEILING, depth=3).strip()
