"""The question sets themselves must stay well-formed (D-088, O-14).

The sets are frozen and append-only, which makes them load-bearing in a way ordinary fixtures
are not: every recording ever made is keyed to a question id, so a duplicated or renamed id
silently re-points historical evidence at different text. These checks cost nothing and run in
the default suite, unlike anything that records or scores.
"""

import json

import pytest

from tests.eval.recording import DEFAULT_QUESTION_SET, QUESTION_SETS, load_questions


@pytest.mark.parametrize("name", sorted(QUESTION_SETS))
def test_every_declared_set_loads(name: str) -> None:
    """A set named in QUESTION_SETS must exist on disk, or a sweep fails only when paid for."""
    assert load_questions(name), f"{name} is declared but empty"


def test_unknown_set_fails_loudly() -> None:
    """A typo in EVAL_QUESTION_SET must crash, not silently record the default set (D-023).

    Falling back would produce recordings filed under the arm the typo named while containing
    answers to entirely different questions -- the D-094 mislabelling, one layer up.
    """
    with pytest.raises(ValueError, match="unknown question set"):
        load_questions("nonexistent")


@pytest.mark.parametrize("name", sorted(QUESTION_SETS))
def test_ids_are_unique_within_a_set(name: str) -> None:
    """Recordings are saved as `<id>.json`, so a duplicate id means one silently overwrites
    the other and the arm quietly holds nine questions while claiming ten."""
    ids = [case["id"] for case in load_questions(name)]
    assert len(ids) == len(set(ids)), f"{name} has duplicate ids"


def test_ids_are_unique_across_sets() -> None:
    """Two sets sharing an id would collide in `results.json`, which is keyed arm -> id.

    The arms differ by prefix so the files never collide on disk, but a shared id makes two
    different questions indistinguishable in any table built from the scores.
    """
    seen: dict[str, str] = {}
    for name in sorted(QUESTION_SETS):
        for case in load_questions(name):
            assert case["id"] not in seen, (
                f"id {case['id']!r} appears in both {seen[case['id']]} and {name}"
            )
            seen[case["id"]] = name


@pytest.mark.parametrize("name", sorted(QUESTION_SETS))
def test_questions_are_non_empty_strings(name: str) -> None:
    """A blank question would still produce a review -- from no sources (D-060) -- and would
    score as a legitimate data point rather than as the input error it is."""
    for case in load_questions(name):
        assert case["question"].strip(), f"{name}/{case['id']} has an empty question"


def test_the_narrow_set_matches_the_broad_set_in_size() -> None:
    """O-14's comparison is paired per question, so equal N keeps the two sets' results
    directly comparable in power rather than only in direction."""
    assert len(load_questions("narrow")) == len(load_questions(DEFAULT_QUESTION_SET))


def test_the_frozen_sets_declare_why_they_are_frozen() -> None:
    """The append-only rule lives in the file, not only in D-088.

    Someone reworking a question will open the JSON, not the decision log; if the warning is
    not there, the rule is not enforced by anything a person will actually see.
    """
    for name, path in sorted(QUESTION_SETS.items()):
        note = json.loads(path.read_text(encoding="utf-8")).get("note", "")
        assert "append-only" in note, f"{name} does not state the append-only rule"
