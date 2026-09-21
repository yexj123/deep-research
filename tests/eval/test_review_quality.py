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

from tests.eval.metrics import citation_density, numeric_density, specificity_metric
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
_SCORES: dict[str, dict[str, dict[str, float]]] = {}


def _record_score(recording: "Recording", metric: str, score: float) -> None:
    # Keyed by arm first: two arms scoring the same question must not overwrite each other,
    # which is the entire point of recording them separately.
    _SCORES.setdefault(recording.arm, {}).setdefault(recording.id, {})[metric] = round(score, 4)


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
    for arm, per_question in _SCORES.items():
        for recording_id, scores in per_question.items():
            runs.setdefault(arm, {}).setdefault(recording_id, {}).update(scores)
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
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
def test_ungrounded_citations_are_recorded_not_asserted(recording: Recording) -> None:
    """Record how many citations were ungrounded. Do NOT fail on them (D-078).

    This repeats a mistake D-078 already fixed once, one level up: asserting
    `citation_violations == []` asserts that the *model behaved*, not that the code works.
    Grounding exists precisely because models cite from memory, so a violation is the checker
    succeeding -- turning the whole baseline sweep red for it would make the suite useless as
    a comparison tool and would punish a run for a fact about the model.

    Measured 2026-09-21 across ten questions: 1 ungrounded citation in 81 total, ~1.2%. That
    is a number D-079's open item asked for, and it only exists because this records rather
    than asserts.

    What *would* fail: a violation that names a paper the run actually retrieved, which would
    mean the checker itself is broken.
    """
    _record_score(recording, "ungrounded_citations", len(recording.citation_violations))
    _record_score(recording, "papers_in_context", len(recording.retrieval_context))

    retrieved = " ".join(recording.retrieval_context)
    for violation in recording.citation_violations:
        assert violation not in retrieved, (
            f"{recording.id}: {violation!r} was retrieved but recorded as ungrounded -- "
            "that is a checker bug, not a model one"
        )


@needs_recordings
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
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
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
def test_claims_are_supported_by_the_retrieved_papers(recording: Recording) -> None:
    """Faithfulness: is each claim in the review supported by the abstracts it cites?

    This is the gap `check_citations` cannot close. A review can cite only real, retrieved
    papers and still assert things those papers never said -- the citation is valid and the
    claim is invented.
    """
    metric = FaithfulnessMetric(threshold=FAITHFULNESS_FLOOR, model=JUDGE_MODEL)
    metric.measure(_case(recording))
    _record_score(recording, "faithfulness", metric.score or 0.0)

    assert metric.score is not None and metric.score >= FAITHFULNESS_FLOOR, (
        f"{recording.id}: faithfulness {metric.score:.2f} "
        f"(judge={JUDGE_MODEL}) -- {metric.reason}"
    )


@needs_recordings
@needs_key
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
def test_the_review_answers_the_question_asked(recording: Recording) -> None:
    """Answer relevancy: a faithful review of the wrong topic is still a bad review.

    Worth measuring separately because the failure modes are independent -- decomposition can
    drift into adjacent subtopics while every individual claim stays perfectly supported.
    """
    metric = AnswerRelevancyMetric(threshold=RELEVANCY_FLOOR, model=JUDGE_MODEL)
    metric.measure(_case(recording))
    _record_score(recording, "relevancy", metric.score or 0.0)

    assert metric.score is not None and metric.score >= RELEVANCY_FLOOR, (
        f"{recording.id}: relevancy {metric.score:.2f} "
        f"(judge={JUDGE_MODEL}) -- {metric.reason}"
    )


# ---- specificity: the metric O-13 actually depends on (D-092) ------------------------


@needs_recordings
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
def test_information_density_is_recorded(recording: Recording) -> None:
    """Deterministic specificity proxies: numeric and citation density (D-092).

    No judge, no cost, no variance -- crude, but they cannot drift and cost nothing to
    re-run, which is why they are recorded alongside the judged score rather than replaced
    by it (D-088).
    """
    _record_score(recording, "numeric_density", numeric_density(recording.review))
    _record_score(recording, "citation_density", citation_density(recording.review))


@needs_recordings
@needs_key
@pytest.mark.eval
@pytest.mark.parametrize("recording", RECORDINGS, ids=lambda r: f"{r.arm}/{r.id}")
def test_specificity_is_recorded(recording: Recording) -> None:
    """Does the review make concrete, checkable claims, or only describe topics?

    This is the metric O-13 rests on. Faithfulness is already ~0.97 and could not distinguish
    a 75% context cut (D-092), so full text will not show up there either. What full text
    should buy is reported numbers, named methods and stated limitations -- which is what this
    measures.

    Recorded, never asserted: a low score means the model wrote a vaguer review today, which
    is a fact about the run, not a regression in the code (D-078).
    """
    metric = specificity_metric(JUDGE_MODEL)
    metric.measure(_case(recording))
    _record_score(recording, "specificity", metric.score or 0.0)
