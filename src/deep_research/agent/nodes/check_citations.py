"""check_citations node: every citation in the review must be a retrieved paper (D-046, D-062)."""

import re
from typing import Any

from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from deep_research.agent.state import ResearchState

# The citation format the synthesize prompt must require (D-046). Group 1 is the ID.
CITATION_MARKER: re.Pattern[str] = re.compile(r"\[arXiv:([^\]\s]+)\]")
# Every [arXiv:...] bracket, parseable or not. Anything this matches that CITATION_MARKER
# cannot fullmatch is a citation the checker can't read, and is recorded rather than
# silently dropped (D-062).
CITATION_BRACKET: re.Pattern[str] = re.compile(r"\[arXiv:[^\]]*\]")

class Citation(BaseModel):
    """One citation, validated against the retrieved IDs passed in as validation context."""

    arxiv_id: str

    @field_validator("arxiv_id")
    @classmethod
    def _was_retrieved(cls, value: str, info: ValidationInfo) -> str:
        # Missing context is a bug in the caller, not bad data, so it must fail loudly. Pydantic
        # turns ValueError *and* AssertionError raised here into a ValidationError, which
        # check_citations would record as a violation for every citation. RuntimeError propagates.
        if info.context is None or "known_ids" not in info.context:
            raise RuntimeError("Citation needs validation context: context={'known_ids': ...}")
        known_ids: set[str] = info.context["known_ids"]

        if value not in known_ids:
            raise ValueError(f"not a retrieved paper: {value!r}")
        return value


def check_citations(state: ResearchState) -> dict[str, Any]:
    """Record every citation that can't be verified against the retrieved papers (D-046, D-062)."""
    known_ids: set[str] = {source.arxiv_id for source in state.sources}

    violations: list[str] = []
    seen: set[str] = set()

    for bracket in CITATION_BRACKET.finditer(state.review):
        text = bracket.group(0)
        match = CITATION_MARKER.fullmatch(text)

        if match is None:
            entry = text  # malformed: record the bracket verbatim, e.g. "[arXiv:A, B]"
        else:
            cited = match.group(1)
            try:
                Citation.model_validate({"arxiv_id": cited}, context={"known_ids": known_ids})
                continue  # verified: nothing to record
            except ValidationError:
                entry = cited  # well-formed ID, but not one we retrieved

        if entry not in seen:
            seen.add(entry)
            violations.append(entry)

    # citations_checked is the terminal marker: an empty citation_violations means "checked,
    # found none" only once this is True (D-084).
    return {"citation_violations": violations, "citations_checked": True}