"""The evaluation recording format (O-11).

Evaluation is split into two halves on purpose, mirroring `tests/agent/fixtures/arxiv/`:
capture real output once, score it repeatedly.

- **Recording** runs the real agent over the frozen question set. Slow (~35 s per question)
  and paid, so it happens rarely and explicitly.
- **Scoring** loads those recordings and judges them. Also paid, but cheap and fast.

Why the split matters beyond cost: the same recordings can be re-scored with a different judge
or a new metric and stay comparable, and a recording is the *baseline artifact* that a later
full-text run (O-13) gets compared against. Score-and-run in one step would mean re-running
the agent every time a metric changed, and no fixed point to compare to.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECORDINGS_DIR = Path(__file__).parent / "recordings"
QUESTIONS_FILE = Path(__file__).parent / "questions.json"


@dataclass(frozen=True)
class Recording:
    """One agent run, saved for scoring.

    `settings` is recorded because a run made with a different `max_depth` or model is not
    comparable to one made with another -- the same reason D-079 insists two runs at one depth
    cannot establish a rate. A recording that does not say how it was produced is not evidence.
    """

    id: str
    question: str
    review: str
    # The abstracts that actually went into synthesize. This is DeepEval's `retrieval_context`:
    # faithfulness asks whether each claim in the review is supported by these.
    retrieval_context: list[str] = field(default_factory=list)
    # Deterministic measurements the agent already makes -- free, exact, no judge (D-046, D-086).
    citation_violations: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = ""

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()


def load_questions() -> list[dict[str, str]]:
    """The frozen question set. Append-only: rewording one invalidates earlier recordings."""
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]


def save(recording: Recording) -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDINGS_DIR / f"{recording.id}.json"
    path.write_text(json.dumps(asdict(recording), indent=2) + "\n", encoding="utf-8")
    return path


def load_all() -> list[Recording]:
    """Every saved recording, sorted by id so test ordering is stable."""
    if not RECORDINGS_DIR.exists():
        return []
    return [
        Recording(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(RECORDINGS_DIR.glob("*.json"))
    ]
