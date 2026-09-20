"""Graph state: what every node reads, and what the checkpointer saves after each step (D-013, D-054)."""

from dataclasses import dataclass, field

from deep_research.agent.sources.models import Source


@dataclass
class ResearchState:
    question: str
    review: str = ""
    # Mutable defaults need default_factory: a plain `= []` would be one list shared by every instance.
    sources: list[Source] = field(default_factory=list)
    skipped_entries: int = 0  # arXiv entries dropped because they failed validation (D-045)
    # every citation that couldn't be verified: unknown IDs, and markers the checker can't parse,
    # quoted verbatim (D-046, D-062)
    citation_violations: list[str] = field(default_factory=list)  
