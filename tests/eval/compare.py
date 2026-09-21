"""Paired comparison of two recorded arms (D-094, O-14).

Every number in D-092, D-093 and D-094 came out of an ad-hoc script typed into a shell and
then thrown away, which means none of them can be re-derived without rewriting the analysis
from the prose that quotes it. That is the wrong way round: the conclusion is load-bearing for
the thesis, so the arithmetic behind it has to be re-runnable on demand.

    uv run python -m tests.eval.compare broad-abstract-top20-d0 broad-abstract-top20-d2

**Why paired.** The two arms answer the *same* questions, so the questions themselves are a
controlled variable and the right statistic is the mean of the per-question differences, not
the difference of the means. Question-to-question variation in how hard a topic is swamps the
effect being measured otherwise -- `long-context` retrieves half what `attention` does under
any setting.

**What "SE units" means.** The mean paired difference divided by its standard error
(`stdev(diffs) / sqrt(n)`). Roughly: under 2 means the arms are indistinguishable on that
metric at this sample size. It is *not* a p-value and n=10 is small; a non-monotonic pattern
across three arms is better evidence of noise than any single number here (D-094).
"""

import json
import re
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tests.eval.recording import RECORDINGS_DIR, Recording, load_all

RESULTS_FILE = Path(__file__).parent / "results.json"
CITATION = re.compile(r"\[arXiv:[^\]]*\]")

# Metrics read straight off the recording, with no judge and no cost. These stay primary
# (D-088): they cannot drift between scoring runs and cost nothing to recompute.
FROM_RECORDING: dict[str, Callable[[Recording], float]] = {
    "papers_retrieved": lambda r: float(r.papers_retrieved),
    "papers_in_prompt": lambda r: float(len(r.retrieval_context)),
    "papers_cited": lambda r: float(len(set(CITATION.findall(r.review)))),
    "review_words": lambda r: float(len(r.review.split())),
    "rounds": lambda r: float(r.coverage.get("rounds", 0)),
    "searches": lambda r: float(
        len(r.coverage.get("explored", [])) + len(r.coverage.get("empty", []))
    ),
    "ungrounded_citations": lambda r: float(len(r.citation_violations)),
}

# Judged metrics, loaded from results.json. Absent until `pytest -m eval` has scored the arm.
FROM_SCORES = ("specificity", "faithfulness", "relevancy", "numeric_density", "citation_density")


@dataclass(frozen=True)
class Row:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    se_units: float | None
    n: int

    def format(self) -> str:
        se = "    --" if self.se_units is None else f"{self.se_units:+6.1f}"
        return (
            f"{self.metric:<22}{self.mean_a:>9.2f}{self.mean_b:>9.2f}"
            f"{self.delta:>+10.2f}{se}"
        )


def _scores(arm: str) -> dict[str, dict[str, float]]:
    if not RESULTS_FILE.exists():
        return {}
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8")).get("runs", {}).get(arm, {})


def _series(arm: str, metric: str, ids: list[str]) -> list[float] | None:
    """One value per question, in a fixed id order, or None if the arm lacks the metric."""
    if metric in FROM_RECORDING:
        by_id = {r.id: r for r in load_all(arm)}
        return [FROM_RECORDING[metric](by_id[qid]) for qid in ids]
    scored = _scores(arm)
    if not all(metric in scored.get(qid, {}) for qid in ids):
        return None
    return [float(scored[qid][metric]) for qid in ids]


def shared_questions(arm_a: str, arm_b: str) -> list[str]:
    """Question ids present in both arms.

    Comparing arms that do not share questions is the failure this function exists to prevent:
    the means would still compute, and would be a comparison of two different populations
    wearing the label of a controlled experiment.
    """
    a = {r.id for r in load_all(arm_a)}
    b = {r.id for r in load_all(arm_b)}
    return sorted(a & b)


def compare(arm_a: str, arm_b: str) -> list[Row]:
    """Paired differences (b - a) for every metric both arms carry."""
    ids = shared_questions(arm_a, arm_b)
    if not ids:
        raise ValueError(f"{arm_a} and {arm_b} share no questions; there is nothing to pair")

    rows: list[Row] = []
    for metric in list(FROM_RECORDING) + list(FROM_SCORES):
        series_a = _series(arm_a, metric, ids)
        series_b = _series(arm_b, metric, ids)
        if series_a is None or series_b is None:
            continue
        diffs = [y - x for x, y in zip(series_a, series_b)]
        mean = st.mean(diffs)
        # Zero variance means every question moved identically -- real for `rounds`, where the
        # setting forces the value. An SE of 0 would divide by zero and print a spurious
        # infinity, so it is reported as undefined instead.
        spread = st.stdev(diffs) if len(diffs) > 1 and len(set(diffs)) > 1 else 0.0
        se = spread / len(diffs) ** 0.5 if spread else 0.0
        rows.append(
            Row(metric, st.mean(series_a), st.mean(series_b), mean, mean / se if se else None, len(ids))
        )
    return rows


def render(arm_a: str, arm_b: str) -> str:
    rows = compare(arm_a, arm_b)
    ids = shared_questions(arm_a, arm_b)
    head = (
        f"{len(ids)} paired questions: {', '.join(ids)}\n\n"
        f"{'metric':<22}{arm_a.split('-')[-1]:>9}{arm_b.split('-')[-1]:>9}{'delta':>10}{'SE':>6}\n"
        + "-" * 56
    )
    tail = (
        "\nSE = mean paired difference / its standard error. Under 2 means indistinguishable\n"
        "at this sample size. Not a p-value; n is small (D-094)."
    )
    return "\n".join([head, *(r.format() for r in rows), tail])


def main() -> int:
    if len(sys.argv) != 3:
        available = sorted(d.name for d in RECORDINGS_DIR.iterdir() if d.is_dir())
        print(f"usage: python -m tests.eval.compare <arm-a> <arm-b>\n\narms:")
        print("\n".join(f"  {a}" for a in available))
        return 2
    print(render(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
