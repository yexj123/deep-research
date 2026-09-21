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

from deep_research.agent.config import MAX_DEPTH
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
    """Why the run ended. Different reasons mean different things to a reader (D-075, D-076).

    Hitting the depth ceiling means more rounds *might* have found more -- the run was cut
    off, not finished. A round finding nothing new means the search had converged. Reporting
    them identically would hide the difference that matters most.
    """
    depth = values.get("depth", 0)
    if depth == 0:
        return "the planner proposed no subtopics worth researching"
    if depth > MAX_DEPTH:
        return (
            f"the depth limit was reached after {depth} rounds "
            f"(max_depth={MAX_DEPTH}); more rounds might have found more"
        )
    return f"round {depth} found no papers that earlier rounds hadn't already seen"


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
