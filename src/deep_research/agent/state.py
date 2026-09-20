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
    merged = list(current)          # copy: never
    for topic in update:
        key = normalize_subtopic(topic)
        if key not in seen:
            seen.add(key)
            merged.append(topic)
    return merged


def merge_sources(current: list[Source], update: list[Source]) -> list[Source]:
    """Append papers whose arxiv_id isn't present(D-067)."""
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
    # Overwritten each round: operator.add would nd (D-017).
    pending_subtopics: list[str] = field(default_factory=list)
    # Parallel writers from here down (D-067).
    explored_subtopics: Annotated[list[str], merge_subtopics] = field(default_factory=list)
    failed_subtopics: Annotated[list[str], operator.add] = field(default_factory=list)
    seen_paper_ids: Annotated[set[str], operator.or_] = field(default_factory=set)
    sources: Annotated[list[Source], merge_sources] = field(default_factory=list)
    # Each worker returns its own delta, never a running total (D-067).
    skipped_entries: Annotated[int, operator.add] = 0
    depth: int = 0
    citation_violations: list[str] = field(default_factory=list)