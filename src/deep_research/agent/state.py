"""Graph state: what every node reads, what the checkpointer saves after each step, and the
reducers that merge parallel writes (D-013, D-054, D-067)."""

import operator
from dataclasses import dataclass, field
from typing import Annotated

from deep_research.agent.sources.models import Source


def normalize_subtopic(text: str) -> str:
    """The comparison key for subtopic dedup: casefold + strip (D-017)."""
    return text.strip().casefold()


def merge_subtopics(current: list[str], update: list[str]) -> list[str]:
    """Append subtopics whose normalized form isn't present yet, keeping original text (D-067)."""
    seen = {normalize_subtopic(topic) for topic in current}
    merged = list(current)  # copy: never mutate the left argument, LangGraph may still hold it
    for topic in update:
        key = normalize_subtopic(topic)
        if key not in seen:
            seen.add(key)
            merged.append(topic)
    return merged


def merge_sources(current: list[Source], update: list[Source]) -> list[Source]:
    """Append papers whose arxiv_id isn't present yet, keeping the first seen (D-067).

    Keyed on arxiv_id, not on the Source object: Source is frozen and hashable, so a set
    would compile and still keep v1 and v2 of one paper as two entries (D-044).
    """
    seen = {source.arxiv_id for source in current}
    merged = list(current)
    for source in update:
        if source.arxiv_id not in seen:
            seen.add(source.arxiv_id)
            merged.append(source)
    return merged


@dataclass
class ResearchState:
    question: str
    review: str = ""
    # No reducer, overwritten each round: operator.add would re-dispatch every earlier
    # round's subtopics alongside the new ones (D-017).
    pending_subtopics: list[str] = field(default_factory=list)
    # Parallel writers from here down (D-067).
    explored_subtopics: Annotated[list[str], merge_subtopics] = field(default_factory=list)
    failed_subtopics: Annotated[list[str], operator.add] = field(default_factory=list)
    seen_paper_ids: Annotated[set[str], operator.or_] = field(default_factory=set)
    sources: Annotated[list[Source], merge_sources] = field(default_factory=list)
    # Each worker returns its own delta, never a running total (D-067).
    skipped_entries: Annotated[int, operator.add] = 0
    # Rounds completed. gap_check increments it, so a full run ends at MAX_DEPTH + 1
    # (D-026, D-076). Single writer, so no reducer.
    depth: int = 0
    # No reducer, overwritten by decompose each round: gap_check compares the current
    # len(seen_paper_ids) against this to learn what *this* round added (D-075). A field with
    # operator.add could not do this -- reducer(current, 0) == current, so a node cannot
    # reset one.
    seen_before_round: int = 0
    citation_violations: list[str] = field(default_factory=list)
    # Written only by check_citations, the terminal node, so it is the one reliable "this run
    # completed" signal (D-084). `next == ()` is not: a run interrupted at a step boundary
    # also has an empty `next`, and `review` alone would call a run finished that stopped
    # between synthesize and check_citations -- reporting zero citation violations without
    # ever having checked. Single writer, so no reducer.
    citations_checked: bool = False