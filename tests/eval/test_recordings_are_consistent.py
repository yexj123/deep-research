"""Recordings must describe themselves correctly (D-094).

Every other eval test judges what the *agent* produced. This one judges the *artifact*: a
recording is evidence, and evidence that misstates how it was produced is worse than no
evidence, because a comparison drawn from it looks sound.

The bug that motivated this file: `EVAL_MAX_DEPTH` patched `graph.MAX_DEPTH` but not
`coverage.MAX_DEPTH`, both of which bind the constant at import time. Routing obeyed the
override, so the d0 and d1 arms really did run 1 and 2 rounds -- but `_stop_reason` compared
against the *unpatched* ceiling, fell through its `depth > MAX_DEPTH` branch, and wrote "round
N found no papers that earlier rounds hadn't already seen" into all twenty recordings. That
sentence is the signature of the semantic exit firing, which is exactly the thing D-075 exists
to do and the thing D-090 measured as never happening. Read at face value it reverses the
finding.

Nothing here costs a model call or a network request; it reads the committed JSON only.
"""

import json
from pathlib import Path

import pytest

from tests.eval.recording import RECORDINGS_DIR, arm_name

CONVERGED = "found no papers that earlier rounds hadn't already seen"
CEILING = "the depth limit was reached"


def _recordings() -> list[tuple[str, dict]]:
    return [
        (f"{path.parent.name}/{path.stem}", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(RECORDINGS_DIR.glob("*/*.json"))
    ]


def test_there_are_recordings_to_check() -> None:
    """Guard against this whole file silently passing on an empty directory."""
    assert _recordings(), "no recordings found; the consistency checks below prove nothing"


@pytest.mark.parametrize("name,rec", _recordings(), ids=lambda v: v if isinstance(v, str) else "")
def test_recording_is_filed_under_the_arm_its_settings_describe(name: str, rec: dict) -> None:
    """The directory name must be derivable from `settings` (D-091's arm_name contract).

    A recording filed under the wrong arm is invisible: it reads as valid data for a
    configuration that never produced it, and every mean drawn from that arm is wrong.
    """
    assert name.split("/")[0] == arm_name(rec["settings"])


@pytest.mark.parametrize("name,rec", _recordings(), ids=lambda v: v if isinstance(v, str) else "")
def test_rounds_never_exceed_the_arms_ceiling(name: str, rec: dict) -> None:
    """A run may stop early, but it can never run more rounds than max_depth allows.

    `gap_check` increments depth after each round, so a run that exhausts its budget ends at
    `max_depth + 1` rounds (state.py:61). More than that means the override did not reach the
    routing decision, and the arm is not the configuration it claims to be.
    """
    assert 1 <= rec["coverage"]["rounds"] <= rec["settings"]["max_depth"] + 1, (
        f"{name}: {rec['coverage']['rounds']} rounds under max_depth="
        f"{rec['settings']['max_depth']}"
    )


@pytest.mark.parametrize("name,rec", _recordings(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_run_that_used_its_whole_budget_is_not_reported_as_converged(
    name: str, rec: dict
) -> None:
    """The stop reason must match the round count (the bug in this file's docstring).

    A run that used every round it was allowed hit the ceiling, whatever else was true of it.
    Claiming it "found no new papers" asserts the semantic exit fired -- the single most
    consequential claim a recording can make about D-075, and the one D-090 and D-094 both
    turn on. This is the D-084 pattern once more: reporting a finish the system did not earn.
    """
    coverage = rec["coverage"]
    exhausted = coverage["rounds"] == rec["settings"]["max_depth"] + 1
    if exhausted and coverage["explored"]:
        assert CONVERGED not in coverage["stopped_because"], (
            f"{name}: ran all {coverage['rounds']} rounds but reports convergence -- "
            f"{coverage['stopped_because']!r}. Check monkeypatch_depth() patches every module "
            "binding MAX_DEPTH."
        )
        assert CEILING in coverage["stopped_because"], (
            f"{name}: exhausted its depth budget but says {coverage['stopped_because']!r}"
        )
