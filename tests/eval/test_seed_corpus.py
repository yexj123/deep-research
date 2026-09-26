"""The O-16 seed manifest is a frozen, dated artifact and must stay one.

Unlike the recordings, this artifact's value is almost entirely in its **date**: a corpus
seeded on 2026-09-25 is evidence about staleness only for as long as everyone agrees it was
seeded then, and only if it can be rebuilt with that date intact. These checks cost nothing
and run in the default suite, because the failure they guard against is silent and is
discovered months later when re-seeding is no longer possible.

They deliberately do **not** hit the network or rebuild the database.
"""

import json
from datetime import UTC, datetime

import pytest

from deep_research.agent.sources.models import Source
from tests.eval.recording import QUESTION_SETS, arm_name, arms, load_questions
from tests.eval.seed_corpus import MANIFEST_FILE, from_record, seed_queries, to_record

pytestmark = pytest.mark.skipif(
    not MANIFEST_FILE.exists(), reason="no seed manifest; run tests.eval.seed_corpus"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def test_the_manifest_is_dated_and_the_date_is_in_the_past(manifest: dict) -> None:
    """`seeded_at` is the whole point of the artifact (O-16).

    Without it the snapshot is just a corpus, and `--rebuild` would stamp today -- which is
    exactly the property the experiment cannot have.
    """
    seeded_at = datetime.fromisoformat(manifest["seeded_at"])
    assert seeded_at.tzinfo is not None, "a naive timestamp is ambiguous across machines"
    assert seeded_at <= datetime.now(UTC)


def test_the_manifest_covers_the_whole_staleness_set(manifest: dict) -> None:
    """A question seeded for but missing would silently drop out of the paired comparison."""
    seeded = {entry["id"] for entry in manifest["questions"]}
    declared = {case["id"] for case in load_questions("staleness")}
    assert seeded == declared, f"manifest and question set disagree: {seeded ^ declared}"


def test_both_arms_are_present_and_the_control_arm_is_not_empty(manifest: dict) -> None:
    """The slow arm is the negative control (O-16).

    Without it, a difference on the fast arm cannot be attributed to *age* rather than to
    corpus-vs-arXiv retrieval -- the confound D-104's missing negative population created.
    """
    arms = {entry["arm"] for entry in manifest["questions"]}
    assert arms == {"fast", "slow"}
    for arm in arms:
        assert sum(1 for e in manifest["questions"] if e["arm"] == arm) >= 3


def test_every_question_seeded_enough_papers_to_answer_from(manifest: dict) -> None:
    """A question with a near-empty corpus falls through to arXiv and measures nothing.

    The bound is deliberately loose: `tests.eval.seed_check` does the real retrieval check.
    This one only catches a seeding run that half-failed and was committed anyway.
    """
    for entry in manifest["questions"]:
        assert entry["paper_count"] >= 20, f"{entry['id']} seeded only {entry['paper_count']}"


def test_no_question_silently_failed_all_its_queries(manifest: dict) -> None:
    """Seeding reports per-query errors instead of raising, so nothing else would flag this."""
    for entry in manifest["questions"]:
        assert len(entry["errors"]) < len(entry["queries"]), (
            f"{entry['id']} had every seed query fail: {entry['errors']}"
        )


def test_newest_published_predates_the_seed(manifest: dict) -> None:
    """`newest_published` is what makes "the stale arm could not have seen this" a fact.

    Any paper a later run cites that was published after this could not have been in the
    snapshot -- no inference required. If it were *after* `seeded_at`, that reasoning breaks.
    """
    seeded_at = datetime.fromisoformat(manifest["seeded_at"])
    for entry in manifest["questions"]:
        newest = datetime.fromisoformat(entry["newest_published"])
        assert newest <= seeded_at, f"{entry['id']} holds a paper newer than the seed"


def test_a_paper_survives_the_json_round_trip(manifest: dict) -> None:
    """`--rebuild` reconstructs `Source`s from the manifest, so the encoding must be lossless.

    A dropped field would rebuild a corpus that is subtly not the one that was seeded, and
    nothing would fail -- the chunk text is title + summary, so a lost `published` or
    `version` stays invisible until someone tries to reason about dates.
    """
    record = manifest["questions"][0]["papers"][0]
    restored = from_record(record)
    assert isinstance(restored, Source)
    assert to_record(restored) == record


def test_the_staleness_set_is_registered_for_recording() -> None:
    """The set has to be reachable as EVAL_QUESTION_SET=staleness, or the protocol cannot run.

    Registering it is one line and easy to forget, and the failure would only appear months
    from now at the moment the experiment is finally runnable.
    """
    assert "staleness" in QUESTION_SETS


def test_the_corpus_age_reaches_the_arm_name() -> None:
    """Two runs of "the same arm" months apart are not the same configuration (O-16).

    `save()` writes to `recordings/<arm>/<id>.json` and overwrites by path, so without the age
    in the name November's run would land on top of the t=0 baseline and destroy the
    comparison it was recorded for -- silently, since the files would simply be newer. Corpus
    age is the experiment's independent variable, so leaving it out of the name is the D-094
    mislabelling along a new axis.
    """
    base = {"question_set": "staleness", "retrieval_unit": "abstract-stale", "max_depth": 2}
    assert arm_name({**base, "days_since_seed": 0}).endswith("-age0d")
    assert arm_name({**base, "days_since_seed": 62}).endswith("-age62d")
    assert arm_name({**base, "days_since_seed": 0}) != arm_name({**base, "days_since_seed": 62})


def test_the_age_suffix_is_absent_for_every_other_question_set() -> None:
    """The 140 committed arm names must not move (D-088).

    Renaming an existing arm would orphan every recording filed under it, and
    `test_recording_is_filed_under_the_arm_its_settings_describe` would fail across the board.
    """
    without = arm_name({"question_set": "broad", "retrieval_unit": "abstract", "max_depth": 2})
    assert "-age" not in without

    existing = [a for a in arms() if not a.startswith("staleness")]
    assert existing, "no pre-existing arms to protect; this check proves nothing"
    assert not any("-age" in a for a in existing)


def test_seed_queries_include_the_question_itself() -> None:
    """The question's own terms are the broadest seed; facets only add depth around it."""
    for case in load_questions("staleness"):
        assert seed_queries(case)[0] == case["question"]
        assert len(seed_queries(case)) > 1, f"{case['id']} has no facets"
