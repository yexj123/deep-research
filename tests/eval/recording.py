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

# Question sets are separate files rather than one file with a `set` field, so that adding a
# set physically cannot touch the frozen one (D-088). "broad" is the original ten single-area
# survey prompts; "narrow" is O-14's intersection questions, built to need decomposition.
QUESTION_SETS = {
    "broad": QUESTIONS_FILE,
    "narrow": Path(__file__).parent / "questions-narrow.json",
    # O-16's staleness set, split into a fast arm and a slow control by measured arXiv
    # velocity. Nothing is recorded against it until the seeded corpus is genuinely old --
    # see docs/staleness-experiment.md.
    "staleness": Path(__file__).parent / "questions-staleness.json",
}
DEFAULT_QUESTION_SET = "broad"


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
    # How many papers the run retrieved in total. retrieval_context holds only those that
    # reached the prompt, so this is what makes pruning's effect visible in the artifact.
    papers_retrieved: int = 0
    settings: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = ""

    @property
    def arm(self) -> str:
        """Which configuration produced this, used as the directory name.

        Arms live side by side rather than overwriting each other: a comparison needs both
        present at once, and keeping the baseline only in git history means it is one
        careless re-record away from being unrecoverable in practice.
        """
        return arm_name(self.settings)

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()


def load_questions(question_set: str = DEFAULT_QUESTION_SET) -> list[dict[str, str]]:
    """One frozen question set. Append-only: rewording one invalidates earlier recordings."""
    if question_set not in QUESTION_SETS:
        raise ValueError(f"unknown question set {question_set!r}; have {sorted(QUESTION_SETS)}")
    return json.loads(QUESTION_SETS[question_set].read_text(encoding="utf-8"))["questions"]


def arm_name(settings: dict[str, Any]) -> str:
    """A short, stable name for one configuration: "broad-abstract-top20-d2", ...

    Derived from `settings` rather than passed in, so a recording can never be filed under an
    arm that does not match how it was produced. That mislabelling would be invisible and
    would silently corrupt every comparison drawn from it.

    The question set is part of the name and is *always* written, with no default-is-omitted
    special case: two arms differing only in which questions they asked are not comparable,
    and a conditional prefix is exactly the shape of the D-094 mislabelling bug. Older
    recordings were renamed to carry the `broad-` prefix rather than grandfathered.
    """
    question_set = settings.get("question_set", DEFAULT_QUESTION_SET)
    unit = settings.get("retrieval_unit", "abstract")
    top_n = settings.get("synthesis_top_n")
    depth = settings.get("max_depth", 2)
    # How the run decided to stop. "fixed" is the depth ceiling plus D-075's new-papers test;
    # "adaptive" adds D-096's exits. This belongs in the name because it changes how many
    # rounds actually run -- an adaptive run under max_depth=2 usually stops after one, so
    # without it a new recording would silently overwrite the d2 baseline it is meant to be
    # compared against.
    exit_rule = settings.get("exit_rule", "fixed")
    # How far into the staleness experiment this run happened (O-16). Absent for every other
    # question set, so the 140 committed arm names are untouched.
    #
    # It belongs in the name for the same reason the question set does: the corpus's *age* is
    # the independent variable, so a run two months after the seed is not the same
    # configuration as one made the day after, however identical the rest of the settings are.
    # Without it November's run would write to the same directory as the baseline and silently
    # overwrite it -- `save()` overwrites by path -- destroying the comparison it was recorded
    # for. That is the D-094 mislabelling along a new axis, and the same consistency test
    # catches it.
    age = settings.get("days_since_seed")
    age_suffix = "" if age is None else f"-age{age}d"
    return (
        f"{question_set}-{unit}-{'all' if top_n is None else f'top{top_n}'}-d{depth}"
        f"-{exit_rule}{age_suffix}"
    )


def save(recording: Recording) -> Path:
    directory = RECORDINGS_DIR / recording.arm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{recording.id}.json"
    path.write_text(json.dumps(asdict(recording), indent=2) + "\n", encoding="utf-8")
    return path


def load_all(arm: str | None = None) -> list[Recording]:
    """Recordings for one arm, or every arm. Sorted, so test ordering is stable."""
    if not RECORDINGS_DIR.exists():
        return []
    pattern = f"{arm}/*.json" if arm else "*/*.json"
    return [
        Recording(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(RECORDINGS_DIR.glob(pattern))
    ]


def arms() -> list[str]:
    """Every arm that currently has recordings."""
    if not RECORDINGS_DIR.exists():
        return []
    return sorted(d.name for d in RECORDINGS_DIR.iterdir() if d.is_dir())
