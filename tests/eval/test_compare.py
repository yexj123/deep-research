"""The comparison arithmetic is itself load-bearing evidence (D-094, O-14).

`compare.py` produces the tables that decisions.md quotes and the thesis will cite. An error
there is not a crashing bug -- it is a *plausible wrong number*, which is the hardest kind to
notice and the exact failure this project keeps rediscovering (D-062, D-069, D-084, and the
D-094 harness bug).

So the pairing is tested on synthetic arms with known answers, where the right result can be
computed by hand, rather than only being eyeballed on real data.
"""

import statistics as st

import pytest

from tests.eval import compare
from tests.eval.compare import Row, shared_questions
from tests.eval.recording import Recording, load_all


def _rows(arm_a: str, arm_b: str) -> dict[str, Row]:
    return {row.metric: row for row in compare.compare(arm_a, arm_b)}


# ---- pairing: the property that makes the comparison an experiment ---------------------


def test_only_shared_questions_are_compared(monkeypatch: pytest.MonkeyPatch) -> None:
    """An arm with an extra question must not drag the other arm's mean.

    If one arm has recorded a question the other has not, including it would compare a mean
    over 10 questions to a mean over 11 and call the difference an effect.
    """
    a = [_rec("x", papers=10), _rec("y", papers=20)]
    b = [_rec("x", papers=30), _rec("y", papers=40), _rec("z", papers=1000)]
    monkeypatch.setattr(compare, "load_all", lambda arm: a if arm == "A" else b)

    assert shared_questions("A", "B") == ["x", "y"]
    rows = _rows("A", "B")
    # z is excluded, so the delta is mean(30-10, 40-20) = 20, not anything involving 1000.
    assert rows["papers_retrieved"].delta == pytest.approx(20.0)
    assert rows["papers_retrieved"].n == 2


def test_comparing_arms_with_no_common_question_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two disjoint arms are not a paired comparison and must not silently produce a table.

    Returning an empty table would read as "no difference found", which is a conclusion, not
    an error message (D-023: crash early and loudly rather than fail silently).
    """
    monkeypatch.setattr(
        compare, "load_all", lambda arm: [_rec("x")] if arm == "A" else [_rec("y")]
    )
    with pytest.raises(ValueError, match="share no questions"):
        compare.compare("A", "B")


def test_the_delta_is_the_mean_of_differences_not_the_difference_of_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired, not unpaired -- the whole reason the arms share a question set.

    The two coincide for the mean itself, but not for its standard error, which is what
    decides whether a result gets reported as real. Here question `y` moves hugely and `x`
    barely: paired SE sees that disagreement, unpaired would not.
    """
    a = [_rec("x", papers=10), _rec("y", papers=10)]
    b = [_rec("x", papers=11), _rec("y", papers=99)]
    monkeypatch.setattr(compare, "load_all", lambda arm: a if arm == "A" else b)

    row = _rows("A", "B")["papers_retrieved"]
    diffs = [1.0, 89.0]
    assert row.delta == pytest.approx(st.mean(diffs))
    assert row.se_units == pytest.approx(
        st.mean(diffs) / (st.stdev(diffs) / 2**0.5)
    )


def test_a_metric_that_never_varies_reports_no_se_instead_of_infinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rounds` is forced by the setting, so its paired differences have zero spread.

    Dividing by that standard error yields infinity, which would print as the most significant
    result in the table. Undefined is the honest rendering.
    """
    a = [_rec("x", rounds=1), _rec("y", rounds=1)]
    b = [_rec("x", rounds=3), _rec("y", rounds=3)]
    monkeypatch.setattr(compare, "load_all", lambda arm: a if arm == "A" else b)

    row = _rows("A", "B")["rounds"]
    assert row.delta == pytest.approx(2.0)
    assert row.se_units is None
    assert "--" in row.format()


def test_metrics_are_paired_by_question_id_not_by_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recordings load in directory order, which need not match between arms.

    Pairing by position would silently compare `attention` in one arm against `diffusion` in
    the other and produce a table that looks entirely normal.
    """
    a = [_rec("x", papers=10), _rec("y", papers=20)]
    b = [_rec("y", papers=20), _rec("x", papers=10)]  # same data, opposite order
    monkeypatch.setattr(compare, "load_all", lambda arm: a if arm == "A" else b)

    assert _rows("A", "B")["papers_retrieved"].delta == pytest.approx(0.0)


# ---- the real data: the module must reproduce what the decision log claims -------------


def test_it_reproduces_the_depth_comparison_d094_reports() -> None:
    """D-094's headline numbers must fall out of the committed recordings (D-094).

    The decision log states 28.6 papers retrieved at one round against 81.3 at three, and
    specificity flat at 0.80 / 0.79. If the recordings no longer produce those, either the
    data was re-recorded or the arithmetic changed -- and the prose quoting it is now wrong.
    """
    rows = _rows("broad-abstract-top20-d0", "broad-abstract-top20-d2")

    assert rows["papers_retrieved"].mean_a == pytest.approx(28.6, abs=0.05)
    assert rows["papers_retrieved"].mean_b == pytest.approx(81.3, abs=0.05)
    assert rows["specificity"].mean_a == pytest.approx(0.80, abs=0.005)
    assert rows["specificity"].mean_b == pytest.approx(0.79, abs=0.005)
    assert abs(rows["specificity"].se_units) < 2, "D-094 claims this is inside the noise"


def test_the_prompt_size_is_the_mechanism_d094_identifies() -> None:
    """Ranking caps the synthesis prompt, so depth cannot enlarge it (D-091, D-094).

    This is the single row that explains the null result: extra rounds change *which* twenty
    papers reach the model, never how many. If it ever stops holding, D-094's explanation is
    void even if its numbers survive.
    """
    rows = _rows("broad-abstract-top20-d0", "broad-abstract-top20-d2")
    assert rows["papers_in_prompt"].delta == pytest.approx(0.0)
    assert rows["papers_in_prompt"].mean_a == pytest.approx(20.0)


def test_every_arm_pairs_against_every_other_without_error() -> None:
    """A smoke test over the real recordings: no arm combination crashes the analysis.

    Arms legitimately differ in which metrics they carry (an unscored arm has no specificity),
    and the comparison must degrade to the metrics they share rather than raising.
    """
    arms = sorted({r.arm for r in load_all()})
    paired = 0
    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1 :]:
            if shared_questions(arm_a, arm_b):
                assert compare.compare(arm_a, arm_b), f"{arm_a} vs {arm_b} produced no rows"
                paired += 1
    assert paired, "no two arms share questions; the comparison is untested on real data"


def _rec(qid: str, papers: int = 0, rounds: int = 1) -> Recording:
    """A minimal recording carrying only the fields a given assertion reads."""
    return Recording(
        id=qid,
        question=f"question {qid}",
        review="A review [arXiv:2411.18583].",
        retrieval_context=["abstract"] * 20,
        papers_retrieved=papers,
        coverage={"rounds": rounds, "explored": [], "empty": []},
        settings={"question_set": "broad", "max_depth": rounds - 1, "synthesis_top_n": 20},
    )
