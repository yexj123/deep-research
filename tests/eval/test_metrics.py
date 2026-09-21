"""Test the measuring instrument before trusting it (O-11).

A metric that returns ~0.95 for everything is not a measurement, it is a number generator --
and that is exactly the failure this metric exists to escape: faithfulness and relevancy could
not distinguish a 75% context cut (D-092).

So before specificity is used to judge O-13, it gets judged itself, on two reviews constructed
to differ *only* in the property it claims to measure. If it cannot separate those, it will
not separate two retrieval strategies, and any O-13 conclusion drawn from it would be noise
wearing a decimal point.

The deterministic proxies are tested with no judge at all; the G-Eval criterion is marked
`eval` because it costs a model call.
"""

import pytest
from deepeval.test_case import LLMTestCase

from tests.eval.metrics import citation_density, numeric_density, specificity_metric
from tests.eval.test_review_quality import JUDGE_MODEL

QUESTION = "How does speculative decoding speed up language model inference?"

# Same question, same length, same confident tone, same citation count. The only difference is
# whether the sentences commit to anything.
SPECIFIC = (
    "Speculative decoding uses a small draft model to propose tokens that the target model "
    "verifies in parallel [arXiv:2211.17192]. Reported speedups are 2.0-2.5x on 70B models "
    "with a 7B drafter, at unchanged output distribution [arXiv:2302.01318]. Medusa instead "
    "adds 4 decoding heads to the target model, reaching 2.2x without a separate draft model "
    "[arXiv:2401.10774]. The gain falls below 1.3x when acceptance rates drop under 60%, which "
    "happens on out-of-distribution prompts [arXiv:2305.09781]."
)

VAGUE = (
    "Speculative decoding is an important technique for accelerating inference in large "
    "language models [arXiv:2211.17192]. Several approaches have been proposed in the "
    "literature, and researchers have explored a variety of strategies [arXiv:2302.01318]. "
    "This remains an active area of research, with ongoing work investigating different ways "
    "to improve performance [arXiv:2401.10774]. A number of challenges and open problems "
    "continue to be discussed by the community [arXiv:2305.09781]."
)


# ---- deterministic proxies: no judge, no cost ----------------------------------------


def test_numeric_density_separates_concrete_from_vague() -> None:
    """A review reporting measurements scores far above one that reports none."""
    assert numeric_density(SPECIFIC) > 3 * numeric_density(VAGUE)


def test_numeric_density_ignores_digits_inside_citations() -> None:
    """arXiv IDs are digits, and must not count as numeric claims.

    Without stripping them, every review would look highly numeric purely for citing things,
    and the proxy would be measuring citation count twice under two names.
    """
    only_citations = "As shown [arXiv:2211.17192] and [arXiv:2302.01318], this is known."
    assert numeric_density(only_citations) == 0.0


def test_citation_density_counts_distinct_papers() -> None:
    """Citing one paper three times is one paper, not three."""
    once = "Claim one [arXiv:2211.17192]."
    thrice = "Claim one [arXiv:2211.17192]. Two [arXiv:2211.17192]. Three [arXiv:2211.17192]."
    assert citation_density(thrice) < 3 * citation_density(once)


def test_the_proxies_do_not_reward_length() -> None:
    """Both are per-100-words, so padding a review cannot inflate them.

    A metric that rewarded length would make "the model wrote more" look like "the model said
    more", which is the confusion this whole file exists to avoid.
    """
    padded = SPECIFIC + " " + " ".join(["Additionally, this is discussed further."] * 20)
    assert numeric_density(padded) < numeric_density(SPECIFIC)


# ---- the judged criterion: does the instrument discriminate? -------------------------


@pytest.mark.eval
def test_specificity_separates_a_concrete_review_from_a_vague_one() -> None:
    """The instrument must resolve what it claims to measure (O-11, D-092).

    Both inputs answer the same question at the same length, in the same confident register,
    citing the same four papers. Only the concreteness differs. If G-Eval cannot separate
    these, it cannot separate abstracts from full text, and O-13's conclusion would be noise.

    A wide margin is demanded rather than a bare ordering: two arms of the same system will
    differ far less than these hand-built extremes, so an instrument that barely separates
    the extremes has no resolution left for the real comparison.
    """
    metric = specificity_metric(JUDGE_MODEL)

    metric.measure(LLMTestCase(input=QUESTION, actual_output=SPECIFIC))
    specific_score = metric.score or 0.0

    metric.measure(LLMTestCase(input=QUESTION, actual_output=VAGUE))
    vague_score = metric.score or 0.0

    print(f"\nspecificity: concrete={specific_score:.3f}  vague={vague_score:.3f}")
    assert specific_score - vague_score > 0.3, (
        f"the metric cannot resolve specificity: concrete={specific_score:.3f} vs "
        f"vague={vague_score:.3f}. Using it on O-13 would report noise as a result."
    )
