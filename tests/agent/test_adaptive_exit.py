"""The adaptive exits: stop when another round cannot change the review (D-096).

**Why these exist.** D-094 and D-095 measured `max_depth` 0, 1 and 2 over twenty questions in
two difficulty classes and found rounds 2 and 3 buy nothing: citations identical (+0.1 SE),
specificity slightly *lower* (-2.2 SE), at 2.6x the retrieval. Two mechanisms explain it, and
each one is a stopping rule the agent can check for free:

1. **The prompt is already full.** `rank_sources` truncates to `SYNTHESIS_TOP_N` (D-091), so
   once a run holds that many papers, another round cannot enlarge what the model sees -- only
   reshuffle which papers win. Measured: round 1 alone reaches 20 papers in **19 of 20**
   questions.
2. **The shelf is empty.** On intersection questions the planner decomposes *correctly* into
   literature that was never written, and 10.9% of deep searches return zero papers (D-095).
   A round that mostly came back empty is evidence the topic is exhausted, not that it needs
   deeper decomposition.

Together these make depth **adaptive** rather than fixed: `MAX_DEPTH` stays a ceiling, and the
common case stops after one round on its own. That is why neither `MAX_DEPTH` nor
`SYNTHESIS_TOP_N` was changed -- tuning the ceiling would have hidden the mechanism behind a
constant.

**What these tests do NOT assert:** that stopping early produces better reviews. It produces
*indistinguishable* ones more cheaply (D-095), and asserting quality here would be the D-078
mistake of asserting that the model behaved.
"""

import pytest

from deep_research.agent.config import MAX_DEPTH, MAX_SUBTOPICS, SYNTHESIS_TOP_N
from deep_research.agent.graph import route_after_gap_check
from deep_research.agent.state import ResearchState
from tests.agent.fakes import make_source


def _sources(n: int, start: int = 0) -> list:
    """n distinct sources, with ids that pass Source's validator."""
    return [make_source(arxiv_id=f"24{11 + (start + i) // 100:02d}.{10000 + (start + i):05d}") for i in range(n)]


def _state(**overrides) -> ResearchState:
    """State as `route_after_gap_check` sees it: after gap_check incremented `depth`.

    Defaults describe a round that *should* continue -- room left under the cap, new papers
    found, nothing empty -- so each test below changes exactly the one thing it is about.
    """
    sources = overrides.pop("sources", None)
    if sources is None:
        sources = _sources(5)
    base = {
        "question": "What is attention?",
        "depth": 1,
        "seen_before_round": 0,
        "seen_paper_ids": {s.arxiv_id for s in sources},
        "sources": sources,
        "pending_subtopics": [f"subtopic {i}" for i in range(MAX_SUBTOPICS)],
        "empty_subtopics": [],
        "empty_before_round": 0,
    }
    return ResearchState(**{**base, **overrides})


def test_the_default_state_continues() -> None:
    """The control case. Without it, every assertion below could pass for the wrong reason.

    If the default already stopped, a test asserting "stops" would prove nothing about the
    rule it names.
    """
    assert route_after_gap_check(_state()) == "decompose"


# ---- exit 1: the synthesis prompt is already full (D-091, D-094) ----------------------


def test_a_run_holding_top_n_papers_stops() -> None:
    """Another round cannot enlarge a prompt that ranking already truncates (D-094).

    This is the mechanism D-094 identified in one row of its table: `retrieval_context` is 20
    at every depth. Rounds 2 and 3 changed *which* twenty papers reached the model and nothing
    else, and produced no measurable difference for it.
    """
    assert SYNTHESIS_TOP_N is not None, "this exit is meaningless without a cap"
    assert route_after_gap_check(_state(sources=_sources(SYNTHESIS_TOP_N))) == "synthesize"


def test_one_paper_short_of_the_cap_still_continues() -> None:
    """The boundary, in the direction that costs money rather than the direction that saves it.

    Off by one here means a run that could still have grown its prompt stops early -- a silent
    quality loss, which is the failure mode this project keeps finding (D-086, D-094).
    """
    assert route_after_gap_check(_state(sources=_sources(SYNTHESIS_TOP_N - 1))) == "decompose"


def test_holding_more_than_the_cap_stops() -> None:
    """`>=`, not `==`: a parallel round can overshoot the cap in one super-step.

    Three workers writing concurrently (D-067) can take 19 papers to 31, so an equality check
    would sail past the exit and never fire again.
    """
    assert route_after_gap_check(_state(sources=_sources(SYNTHESIS_TOP_N + 11))) == "synthesize"


# ---- exit 2: the round came back empty (D-095) ---------------------------------------


def test_a_round_whose_searches_all_came_back_empty_stops() -> None:
    """The D-095 mechanism: the planner decomposed into literature that does not exist.

    A zero-result search is recorded as a *success* (D-021), so it is invisible in every other
    signal -- which is exactly why the run kept recursing into a void for three rounds.
    """
    state = _state(
        empty_before_round=0,
        empty_subtopics=[f"subtopic {i}" for i in range(MAX_SUBTOPICS)],
    )
    assert route_after_gap_check(state) == "synthesize"


def test_a_round_half_empty_stops() -> None:
    """Half is the threshold: most of the round found nothing, so the topic is exhausted.

    Demanding *all* empty would almost never fire -- one productive subtopic out of three is
    enough to keep a run drilling, which is the behaviour D-095 measured as wasteful.
    """
    state = _state(empty_before_round=0, empty_subtopics=["a", "b"])  # 2 of 3
    assert route_after_gap_check(state) == "synthesize"


def test_a_round_with_one_empty_subtopic_of_three_continues() -> None:
    """One dead end is normal and is not evidence the topic is exhausted.

    Stopping here would make the agent quit on its first unlucky query, which is a worse
    failure than the over-searching D-095 found.
    """
    assert route_after_gap_check(_state(empty_before_round=0, empty_subtopics=["a"])) == "decompose"


def test_empties_are_counted_per_round_not_cumulatively() -> None:
    """`empty_subtopics` accumulates, so totals cannot answer "what did THIS round find".

    This is the same trap `seen_before_round` exists for (D-075). Without the per-round
    baseline, two harmless empties in earlier rounds would permanently latch the exit on and
    every later run would stop after one round, regardless of what it found.
    """
    state = _state(
        empty_before_round=3,  # three earlier rounds each found one dead end
        empty_subtopics=["old a", "old b", "old c", "new one"],  # this round: 1 of 3
    )
    assert route_after_gap_check(state) == "decompose"


def test_a_round_that_dispatched_nothing_does_not_divide_by_zero() -> None:
    """An empty fan-out must not crash the router (D-069).

    `decompose` can legitimately return no subtopics when the planner proposes nothing new,
    and the ratio test would be 0/0. The run should end, not raise.
    """
    assert route_after_gap_check(_state(pending_subtopics=[])) == "synthesize"


# ---- the older exits still hold ------------------------------------------------------


def test_the_depth_ceiling_still_wins() -> None:
    """The ceiling is the backstop and must outrank every adaptive rule (D-009, D-026).

    The adaptive exits are heuristics over model-chosen subtopics; the ceiling is the only
    guarantee the run terminates.
    """
    state = _state(depth=MAX_DEPTH + 1, sources=_sources(1))
    assert route_after_gap_check(state) == "synthesize"


def test_a_round_that_added_no_new_paper_still_stops() -> None:
    """D-075's original exit survives, even though D-094 showed it never fires in practice.

    Keeping it costs nothing and it is the only rule that is *correct by construction* rather
    than by measurement: a round adding no paper genuinely cannot change the review.
    """
    sources = _sources(3)
    state = _state(sources=sources, seen_before_round=len(sources))
    assert route_after_gap_check(state) == "synthesize"


def test_the_exits_compose_without_contradicting_each_other() -> None:
    """Every rule points the same way: any one firing ends the run.

    A router where rules can disagree needs a documented precedence; this one does not, and
    that property is worth pinning so a later rule cannot quietly introduce one.
    """
    full_and_empty = _state(
        sources=_sources(SYNTHESIS_TOP_N),
        empty_before_round=0,
        empty_subtopics=["a", "b", "c"],
        depth=MAX_DEPTH + 1,
    )
    assert route_after_gap_check(full_and_empty) == "synthesize"


@pytest.mark.parametrize("n_sources", [1, 5, SYNTHESIS_TOP_N - 1])
def test_a_thin_run_still_gets_its_extra_round(n_sources: int) -> None:
    """The one case the adaptive exits must protect: a question with little literature.

    `moe-latency` retrieved 16 papers in one round (under the cap) -- the single question of
    twenty where a second round had something to add. If the exits fired here too, they would
    be `MAX_DEPTH = 0` wearing a disguise, and the recursion would be dead rather than
    adaptive.
    """
    assert route_after_gap_check(_state(sources=_sources(n_sources))) == "decompose"
