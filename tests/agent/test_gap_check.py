"""gap_check tests (D-075, D-076) and the recursion routing out of it.

gap_check calls no model and does no I/O, so it tests as a pure function of state. What it
pins down is the stopping rule: the run must end for a stated reason, not because it ran out
of recursion budget (D-009).
"""

import pytest

from deep_research.agent.config import MAX_DEPTH
from deep_research.agent.graph import route_after_gap_check
from deep_research.agent.nodes.gap_check import gap_check
from deep_research.agent.state import ResearchState
from tests.agent.fakes import make_source


def _state(**overrides) -> ResearchState:
    """A state as it looks when a round has just finished."""
    base = {
        "question": "What is attention?",
        "depth": 0,
        "seen_before_round": 0,
        "seen_paper_ids": {"2411.18583"},
        "sources": [make_source(arxiv_id="2411.18583")],
    }
    return ResearchState(**{**base, **overrides})


# ---- depth accounting (D-076) --------------------------------------------------------


def test_gap_check_increments_depth() -> None:
    """depth counts rounds completed, so gap_check adds one (D-076).

    Single writer, so no reducer: depth is never written by a parallel worker (D-067).
    """
    assert gap_check(_state(depth=0))["depth"] == 1


def test_depth_is_the_only_field_gap_check_writes() -> None:
    """gap_check decides routing; it doesn't touch research data (D-075).

    Its update is deliberately minimal -- everything else it needs is already in state.
    """
    assert set(gap_check(_state())) == {"depth"}


# ---- the stopping rule (D-075) -------------------------------------------------------


def test_another_round_runs_when_the_last_one_found_new_papers() -> None:
    """A round that added papers, with depth left, routes back to decompose (D-075)."""
    state = _state(depth=1, seen_before_round=1, seen_paper_ids={"a", "b", "c"})
    assert route_after_gap_check(state) == "decompose"


def test_the_run_stops_when_a_round_adds_no_new_papers() -> None:
    """Zero new papers means another round would repeat the same work (D-075).

    This is the semantic exit. It must fire before recursion_limit does -- a
    GraphRecursionError would mean this rule is broken (D-009, D-077).
    """
    state = _state(depth=1, seen_before_round=3, seen_paper_ids={"a", "b", "c"})
    assert route_after_gap_check(state) == "synthesize"


def test_the_run_stops_once_depth_is_spent() -> None:
    """depth > MAX_DEPTH ends the run even when the last round was productive (D-026, D-076).

    With MAX_DEPTH = 2 and gap_check incrementing first, depth == 3 means three search
    passes have completed.
    """
    state = _state(depth=MAX_DEPTH + 1, seen_before_round=0, seen_paper_ids={"a", "b"})
    assert route_after_gap_check(state) == "synthesize"


def test_depth_wins_over_a_productive_round() -> None:
    """Both conditions must hold to continue; depth is the hard ceiling (D-009, D-075)."""
    productive = _state(depth=MAX_DEPTH + 1, seen_before_round=0, seen_paper_ids={"a", "b", "c"})
    assert route_after_gap_check(productive) == "synthesize"


@pytest.mark.parametrize("depth", range(MAX_DEPTH + 1))
def test_every_depth_below_the_ceiling_continues_on_new_papers(depth: int) -> None:
    """Rounds 0..MAX_DEPTH all continue when productive, giving MAX_DEPTH + 1 passes (D-026)."""
    state = _state(depth=depth, seen_before_round=0, seen_paper_ids={"a"})
    assert route_after_gap_check(state) == "decompose"


def test_a_round_that_only_failed_stops_the_run() -> None:
    """Every worker failing adds no papers, so the run ends rather than retrying forever (D-075).

    The failed subtopics stay unexplored (D-018) and are recorded, but without new papers
    there is no evidence another round would do better.
    """
    state = _state(
        depth=0,
        seen_before_round=1,
        seen_paper_ids={"2411.18583"},
        failed_subtopics=["alpha topic", "beta topic"],
    )
    assert route_after_gap_check(state) == "synthesize"


def test_the_first_round_finding_nothing_stops_immediately() -> None:
    """An empty first round ends the run at depth 1 instead of burning all three passes (D-075).

    Zero results is a success (D-021), so this is the honest "nothing published on X" path --
    it should cost one round, not three.
    """
    state = _state(depth=0, seen_before_round=0, seen_paper_ids=set(), sources=[])
    assert route_after_gap_check(state) == "synthesize"
