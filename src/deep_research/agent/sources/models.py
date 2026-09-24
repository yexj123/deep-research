"""Paper records stored in graph state (D-040, D-041, D-043, D-044)."""

import re
from datetime import datetime

from pydantic import field_validator
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict 

# The canonical arXiv ID, with no version suffix (D-041, D-044). Always checked with fullmatch.
#   New scheme: YYMM.NNNN or YYMM.NNNNN; 4 or 5 digits are accepted for any YYMM (D-056).
#   Old scheme (before April 2007): archive, optional ".XX" subject class, "/", YYMMNNN,
#   e.g. "hep-th/9901001", "math.GT/0309136".
ARXIV_ID_PATTERN: re.Pattern[str] = re.compile(
    r"\d{4}\.\d{4,5}"  # new scheme
    r"|[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7}"  # old scheme
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Source:
    """One arXiv paper.

    A Pydantic dataclass: validated when it's built from arXiv data, and again when it's
    restored from a checkpoint, which also turns tuple fields back into tuples (D-043).
    It must be listed in persistence.checkpointer.ALLOWED_MSGPACK_MODULES (D-014).
    """

    arxiv_id: str  # canonical, no version, e.g. "2411.18583"
    version: int  # from the "vN" suffix, e.g. 1
    title: str
    authors: tuple[str, ...]
    summary: str
    published: datetime
    url: str  # abstract page, e.g. "https://arxiv.org/abs/2411.18583v1"
    # Full-text passages that matched the query, when the corpus has read the paper (D-110).
    # A separate field rather than overwriting `summary`: the abstract is what arXiv published
    # and stays what it says, while `excerpt` is what *this query* found inside the paper. A
    # `Source` whose `summary` silently became something else would be a record that lies
    # about its own provenance -- and empty here is the honest default for every paper the
    # corpus has only ever seen the abstract of.
    excerpt: str = ""

    @field_validator("arxiv_id")
    @classmethod
    def _arxiv_id_is_canonical(cls, value: str) -> str:
        # A versioned ID such as "2411.18583v1" is rejected: the version belongs in `version`.
        if ARXIV_ID_PATTERN.fullmatch(value):
            return value
        raise ValueError(
            f"not a canonical arXiv ID (e.g. '2411.18583', 'hep-th/9901001'; no version): {value!r}"
        )

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"arXiv version must be >= 1, got {value!r}")
        return value

    @field_validator("title")
    @classmethod
    def _collapse_title_whitespace(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            raise ValueError("Paper title cannot be empty after stripping whitespace")
        return cleaned
