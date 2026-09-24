"""Why a run stops. One definition, read by the router and by the coverage report (D-096).

**Why this module exists.** The stopping rule lived in `graph.route_after_gap_check` and was
*described* by `coverage._stop_reason`, which reimplemented enough of it to pick a sentence.
Those two drifted apart twice:

- D-094: `coverage.py` compared against its own copy of `MAX_DEPTH`, so runs that hit their
  ceiling were recorded as having converged.
- D-096: two new exits were added to the router and `_stop_reason` knew nothing about them,
  so every adaptive run reported "round 1 found no papers that earlier rounds hadn't already
  seen" when it had actually stopped with a full prompt.

Both are the same failure: **the run reporting a reason it did not have.** That is this
project's recurring shape (D-062, D-069, D-084), and it is worse here than in the agent,
because the coverage panel is what a reader trusts when deciding whether to believe a review.

So the decision and its explanation come from one place. The functions take primitives rather
than `ResearchState`, so `graph.py` can pass a dataclass's fields and `coverage.py` can pass a
checkpoint dict without either importing the other -- imports stay pointing downward
(code-map rule 1).
"""

from deep_research.agent.config import EMPTY_ROUND_RATIO, MAX_DEPTH, SYNTHESIS_TOP_N

NO_SUBTOPICS = "no_subtopics"
DEPTH_CEILING = "depth_ceiling"
NO_NEW_PAPERS = "no_new_papers"
PROMPT_FULL = "prompt_full"
ROUND_EMPTY = "round_empty"

# What each exit means to a reader. "The depth limit cut this off" and "the search was already
# complete" are very different claims about how much to trust the review, which is the whole
# point of reporting a reason at all (D-086).
REASONS = {
    NO_SUBTOPICS: "the planner proposed no subtopics worth researching",
    NO_NEW_PAPERS: "a round found no papers that earlier rounds hadn't already seen",
    PROMPT_FULL: (
        "enough papers were found to fill the synthesis context, so further rounds could only "
        "have changed which papers were used, not how many"
    ),
    ROUND_EMPTY: (
        "a round's searches mostly returned nothing, so the topic appears to be exhausted "
        "rather than under-explored"
    ),
}


def exit_reason(
    *,
    depth: int,
    seen_count: int,
    seen_before_round: int,
    source_count: int,
    dispatched: int,
    empties_this_round: int,
) -> str | None:
    """Which exit fires, or None if another round is worth running.

    Ordered cheapest and most certain first. The order matters only for *reporting* -- the
    routing decision is the same whichever fires, and no two of these ever disagree about
    whether to stop.

    Keyword-only on purpose: six integers in a row is exactly the signature where a
    transposed pair produces a plausible wrong answer instead of an error.

    `NO_SUBTOPICS` is deliberately **not** decided here. "No round ever ran" is something only
    a finished run can be asked about; the router is called after `gap_check` has already
    incremented `depth`, so it never sees zero. Putting that case here made the router stop on
    a state it cannot reach and broke a test that legitimately described depth 0 -- routing
    rules and report-only interpretations are different things, and `summarize_coverage` owns
    the second kind.
    """
    if depth > MAX_DEPTH:
        return DEPTH_CEILING
    if seen_count == seen_before_round:
        # Correct by construction: a round that retrieved nothing unseen cannot change the
        # review (D-075). Measured as almost never firing (D-094), and kept because it is the
        # only exit that is true by definition rather than by measurement.
        return NO_NEW_PAPERS
    if SYNTHESIS_TOP_N is not None and source_count >= SYNTHESIS_TOP_N:
        # Ranking truncates to SYNTHESIS_TOP_N (D-091), so past this point another round can
        # only reshuffle which papers the model sees. Measured: one round alone reaches the
        # cap in 19 of 20 questions, and rounds 2-3 bought no quality for 2.6x the retrieval
        # (D-094, D-095). This is the exit that actually fires.
        return PROMPT_FULL
    if _came_back_empty(dispatched, empties_this_round):
        return ROUND_EMPTY
    return None


def _came_back_empty(dispatched: int, empties_this_round: int) -> bool:
    """Did most of this round's searches find nothing? (D-095)

    A zero-result search is recorded as a *success* (D-021), so it is invisible in every other
    signal -- which is how a run kept decomposing into literature that was never written for
    three full rounds. On intersection questions 10.9% of deep searches returned no papers at
    all, against 1.1% on broad ones: the narrower the intersection, the less exists at it, so
    drilling further finds less rather than more.

    **This rule covers the case rule 3 structurally cannot, which is why it is kept even
    though no recorded run has been observed stopping here (D-098).** `PROMPT_FULL` fires when
    a question has plenty of literature -- 19 of 20 questions. A *thin* question never reaches
    the cap, so nothing above stops it, and it recurses to the ceiling drilling an empty shelf:
    exactly D-095's failure. `fed-privacy` came within one paper of being that run. The two
    rules are complements, not alternatives.

    The threshold is `EMPTY_ROUND_RATIO`, named in `config.py` because it is a judgement
    rather than a measurement -- the argument for 0.5 is there, with what would settle it.
    """
    if not dispatched:
        # Unreachable today: `route_subtopics` sends an empty plan straight to synthesize
        # (D-069). Kept because the ratio would be 0/0, and because the safe answer to "did
        # this round find anything" when no round ran is to stop -- if that guard is ever
        # removed, this fails toward terminating rather than toward looping.
        return True
    return empties_this_round >= dispatched * EMPTY_ROUND_RATIO


def describe(reason: str | None, depth: int) -> str:
    """The sentence shown to a reader, for the exit that actually fired.

    `depth` is interpolated only for the ceiling, where the number is the point: "cut off
    after 3 rounds" tells a reader the review may be incomplete, which none of the other
    reasons do.
    """
    if reason == DEPTH_CEILING:
        return (
            f"the depth limit was reached after {depth} rounds (max_depth={MAX_DEPTH}); "
            "more rounds might have found more"
        )
    if reason is None:
        # A run that has not stopped: interrupted, resumable, or still going (D-084).
        return "the run has not finished"
    return REASONS[reason]
