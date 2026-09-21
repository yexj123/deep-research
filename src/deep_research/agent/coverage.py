"""What a run did *not* cover, derived from its final state (O-5).

The problem this solves: a review missing a third of its subtopics currently looks exactly
like a complete one. Failed subtopics are recorded but nobody reads them, malformed arXiv
entries are counted but nobody reads that either, and a subtopic that found zero papers is
marked explored (D-021) and so is indistinguishable from a productive one.

That is the failure shape this project has hit repeatedly -- D-062 (unreadable citations
reported as clean), D-063 (a shredded query returning confident nonsense), D-069 (an empty
fan-out ending the run silently). Each time the system reported success for less work than
the reader assumed.

Pure functions over state, in the agent package rather than `api/`, because "what did this
run fail to cover" is a research fact the thesis notebooks want, not a presentation detail.
Rendering it is the API's job.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from deep_research.agent.exits import NO_SUBTOPICS, REASONS, describe, exit_reason
from deep_research.agent.state import normalize_subtopic


@dataclass(frozen=True)
class Coverage:
    """The honest summary of a run: what it searched, what it missed, and why it stopped."""

    explored: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    failed: dict[str, int] = field(default_factory=dict)
    skipped_entries: int = 0
    papers: int = 0
    rounds: int = 0
    stopped_because: str = ""

    @property
    def is_complete(self) -> bool:
        """True when nothing was lost: no failures, no empty subtopics, no skipped entries.

        The UI shows the coverage panel only when this is False, so a clean run isn't
        cluttered with a list of nothing.
        """
        return not self.empty and not self.failed and self.skipped_entries == 0


def _stop_reason(values: dict[str, Any]) -> str:
    """Why the run ended. Different reasons mean different things to a reader (D-075, D-096).

    Hitting the depth ceiling means more rounds *might* have found more -- the run was cut
    off, not finished. A full synthesis context means further rounds could only have changed
    *which* papers were used. Reporting them identically would hide the difference that
    matters most to someone deciding how far to trust the review.

    The decision is **not** reimplemented here: `exits.exit_reason` is the same function
    `graph.route_after_gap_check` routes on. Two earlier versions of this function kept their
    own copy of the rules and both drifted -- D-094 reported ceiling stops as convergence,
    D-096 reported a full prompt as convergence -- so the reason a reader sees now comes from
    the rule that actually fired.
    """
    depth = values.get("depth", 0)
    if depth == 0:
        # gap_check never ran, so no round completed (D-069). Report-only: the router is
        # called *after* gap_check increments depth and so never sees this.
        return REASONS[NO_SUBTOPICS]
    return describe(
        exit_reason(
            depth=values.get("depth", 0),
            seen_count=len(values.get("seen_paper_ids", ())),
            seen_before_round=values.get("seen_before_round", 0),
            source_count=len(values.get("sources", ())),
            dispatched=len(values.get("pending_subtopics", ())),
            empties_this_round=(
                len(values.get("empty_subtopics", ())) - values.get("empty_before_round", 0)
            ),
        ),
        depth=values.get("depth", 0),
    )


def summarize_coverage(values: dict[str, Any]) -> Coverage:
    """Build a Coverage from `aget_state(...).values` (a plain dict, D-030).

    Failures are counted on the normalized subtopic, the same key `decompose` uses for the
    N=2 retry cap (D-020, D-073) -- otherwise "Flaky topic" and "flaky topic" would be
    reported as two different failures when the agent treats them as one.
    """
    failed_raw: list[str] = values.get("failed_subtopics", [])
    attempts = Counter(normalize_subtopic(subtopic) for subtopic in failed_raw)
    # Report the planner's original wording, not the normalized key (D-067).
    first_spelling = {normalize_subtopic(s): s for s in reversed(failed_raw)}

    return Coverage(
        explored=list(values.get("explored_subtopics", [])),
        empty=list(values.get("empty_subtopics", [])),
        failed={first_spelling[key]: count for key, count in attempts.items()},
        skipped_entries=values.get("skipped_entries", 0),
        papers=len(values.get("sources", [])),
        rounds=values.get("depth", 0),
        stopped_because=_stop_reason(values),
    )
