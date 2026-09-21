"""Score recorded runs (O-11). Paid, but cheap and fast -- it re-runs nothing.

    uv run pytest -m record   # produce recordings (slow, expensive)
    uv run pytest -m eval     # score them (fast, cents)

**What this adds that the agent does not already measure.** `citation_violations` answers
"is this cited ID a paper we retrieved?" exactly, for free, with no judge (D-046, D-062).
It does not answer the question below it:

    "Smith et al. showed X [arXiv:1234.5678]"
    -- valid ID, paper retrieved, and the paper never says X.

That is claim support, and nothing in the pipeline catches it. Faithfulness is precisely that
metric: it breaks the review into individual claims and checks each against the abstracts the
review was written from.

**The deterministic checks stay primary.** They are exact and free; these are a complement,
not a replacement. Never swap a measurement that needs no judge for one that does.
"""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.eval.recording import Recording, load_all

# Pinned, and recorded in the assertion output. A different judge produces different numbers,
# so an unrecorded judge makes two evaluation runs incomparable -- the same mistake as an
# unrecorded max_depth (D-079). gpt-4o-mini keeps a ten-question sweep at cents.
JUDGE_MODEL = "gpt-4o-mini"

# Thresholds are deliberately low. These are a BASELINE to compare O-13 against, not a quality
# gate: a red build here would mean "the model wrote a worse review today", which is not a
# regression in the code. Raise them only once there is evidence of what this system actually
# scores.
FAITHFULNESS_FLOOR = 0.5
RELEVANCY_FLOOR = 0.5

RESULTS_FILE = Path(__file__).parent / "results.json"

# Scores accumulate here and are written once at session end. A passing test only says "above
# the floor", which is useless as a baseline: when O-13 lands, both runs would pass and the
# comparison would be impossible. The number is the artifact, not the green tick.
_SCORES: dict[str, dict[str, float]] = {}


def _record_score(recording_id: str, metric: str, score: float) -> None:
    _SCORES.setdefault(recording_id, {})[metric] = round(score, 4)


@pytest.fixture(scope="session", autouse=True)
def write_results() -> Iterator[None]:
    """Write the scores after the session, merging with whatever was there before.

    Merging rather than overwriting means `-k attention` updates one entry instead of wiping
    the rest -- the same reason recordings are one file per question.
    """
    yield
    if not _SCORES:
        return
    existing: dict[str, Any] = {}
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    runs = existing.get("runs", {})
    for recording_id, scores in _SCORES.items():
        runs.setdefault(recording_id, {}).update(scores)
    RESULTS_FILE.write_text(
        json.dumps(
            {"judge": JUDGE_MODEL, "scored_at": datetime.now(UTC).isoformat(), "runs": runs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


RECORDINGS = load_all()
needs_recordings = pytest.mark.skipif(
    not RECORDINGS, reason="no recordings yet -- run `uv run pytest -m record` first"
)
needs_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set (the judge needs it)"
)


def _case(recording: Recording) -> LLMTestCase:
    return LLMTestCase(
        input=recording.question,
        actual_output=recording.review,
        retrieval_context=recording.retrieval_context,
    )


# ---- deterministic: free, exact, no judge --------------------------------------------


@needs_recordings
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: r.id)
def test_every_citation_was_actually_retrieved(recording: Recording) -> None:
    """The agent's own grounding check, replayed over the recorded set (D-046).

    Listed first on purpose: this is the measurement that needs no model and costs nothing,
    and it is stronger evidence than any judged score. A violation here is the agent citing a
    paper it never retrieved.
    """
    _record_score(recording.id, "ungrounded_citations", len(recording.citation_violations))
    _record_score(recording.id, "papers_in_context", len(recording.retrieval_context))
    assert recording.citation_violations == [], (
        f"{recording.id}: ungrounded citations {recording.citation_violations}"
    )


@needs_recordings
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: r.id)
def test_the_run_actually_researched_something(recording: Recording) -> None:
    """A recording with no papers cannot be scored meaningfully (D-021, D-086).

    Zero results is a legitimate agent outcome, but it makes faithfulness vacuous -- there is
    nothing for a claim to be supported by. Better to fail here than to report a score for a
    review written from nothing.
    """
    assert recording.retrieval_context, f"{recording.id}: no papers, nothing to judge against"


# ---- judged: the claim-support gap ---------------------------------------------------


@needs_recordings
@needs_key
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: r.id)
def test_claims_are_supported_by_the_retrieved_papers(recording: Recording) -> None:
    """Faithfulness: is each claim in the review supported by the abstracts it cites?

    This is the gap `check_citations` cannot close. A review can cite only real, retrieved
    papers and still assert things those papers never said -- the citation is valid and the
    claim is invented.
    """
    metric = FaithfulnessMetric(threshold=FAITHFULNESS_FLOOR, model=JUDGE_MODEL)
    metric.measure(_case(recording))
    _record_score(recording.id, "faithfulness", metric.score or 0.0)

    assert metric.score is not None and metric.score >= FAITHFULNESS_FLOOR, (
        f"{recording.id}: faithfulness {metric.score:.2f} "
        f"(judge={JUDGE_MODEL}) -- {metric.reason}"
    )


@needs_recordings
@needs_key
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: r.id)
def test_the_review_answers_the_question_asked(recording: Recording) -> None:
    """Answer relevancy: a faithful review of the wrong topic is still a bad review.

    Worth measuring separately because the failure modes are independent -- decomposition can
    drift into adjacent subtopics while every individual claim stays perfectly supported.
    """
    metric = AnswerRelevancyMetric(threshold=RELEVANCY_FLOOR, model=JUDGE_MODEL)
    metric.measure(_case(recording))
    _record_score(recording.id, "relevancy", metric.score or 0.0)

    assert metric.score is not None and metric.score >= RELEVANCY_FLOOR, (
        f"{recording.id}: relevancy {metric.score:.2f} "
        f"(judge={JUDGE_MODEL}) -- {metric.reason}"
    )
