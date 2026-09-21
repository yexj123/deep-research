"""Specificity: does the review make concrete, checkable claims, or only describe topics?

**Why this metric exists.** Faithfulness and relevancy could not distinguish a **75% context
cut** (D-092), and the baseline sits at 0.970 / 0.990 (D-090). An instrument with no
resolution there will have none for full-text retrieval either, and O-13 would report "no
measurable effect" from a measurement incapable of detecting one.

The hypothesis O-13 actually rests on is not that full text makes reviews *more faithful* --
they are already ~0.97 -- but that it lets a review say things an abstract cannot: reported
numbers, experimental conditions, named methods, stated limitations. That is what this
measures.

Two halves, following D-088's rule that the free, exact measurement stays primary:

- **Deterministic proxies** (`numeric_density`, `citation_density`) -- no judge, no cost, no
  variance. Crude, but they cannot drift and they cost nothing to re-run.
- **A G-Eval criterion** -- semantic, judged, and therefore subject to everything D-079 says
  about judged numbers.
"""

import re

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

# Citations carry digits (`[arXiv:2411.18583]`), so they must be removed before counting
# numbers -- otherwise every review scores as highly numeric purely for citing things, and
# the metric would measure citation count twice under two names.
CITATION = re.compile(r"\[arXiv:[^\]]*\]")
# A numeric claim: an integer, decimal, percentage or multiplier. Deliberately simple; this is
# a proxy, and a clever regex would give false precision to something already crude.
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|x|×|-fold|B|M|k)?\b", re.IGNORECASE)


def _words(text: str) -> int:
    return max(len(text.split()), 1)


def numeric_density(review: str) -> float:
    """Numeric claims per 100 words, ignoring citation markers.

    A review saying "reduces memory by 4x at 7B parameters" is making checkable claims; one
    saying "several approaches have been proposed" is not. This cannot tell a *correct* number
    from a wrong one -- faithfulness is what does that -- it only measures whether the review
    commits to anything.
    """
    stripped = CITATION.sub(" ", review)
    return 100 * len(NUMBER.findall(stripped)) / _words(stripped)


def citation_density(review: str) -> float:
    """Distinct papers cited per 100 words.

    A review that cites twelve papers in 400 words is attributing claims densely; one citing
    three is mostly speaking in its own voice, which is harder to verify and easier to invent.
    """
    distinct = len(set(CITATION.findall(review)))
    return 100 * distinct / _words(CITATION.sub(" ", review))


def specificity_metric(model: str, threshold: float = 0.5) -> GEval:
    """A G-Eval criterion scoring concrete detail over topic-level description.

    The steps name the distinction explicitly rather than asking for "quality", because a
    vague criterion produces the same ~0.95 for everything -- which is the failure this metric
    exists to escape. `tests/eval/test_metrics.py` checks it actually separates a deliberately
    specific review from a deliberately vague one; a metric that cannot do that on constructed
    inputs will not separate two retrieval strategies either.
    """
    return GEval(
        name="Specificity",
        model=model,
        threshold=threshold,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Identify every claim in the output that is concrete and checkable: a reported "
            "number or measurement, a named method, model or dataset, an experimental "
            "condition, or an explicitly stated limitation.",
            "Identify every claim that is topic-level only -- for example 'several approaches "
            "have been proposed', 'this remains an active area of research', or naming a "
            "research direction without saying what was done or found.",
            "Judge how much of the output consists of the first kind rather than the second. "
            "A review that names methods and reports findings scores high; one that lists "
            "topics and says work exists on them scores low.",
            "Do not reward length, fluency, confident tone, or the number of citations. A "
            "long, well-written review that commits to nothing scores low.",
        ],
    )
