"""What the markdown-splitting bug cost, measured over the committed recordings (D-123).

Free and exact: every recorded review is real model output, and claim extraction is a pure
function of the review text plus the ids it may cite. No API key, no judge.

    uv run python -m tests.eval.claim_split

Prints, per arm and pooled: how many claims the old rule found, how many the fixed rule
finds, and how many of the old ones were merged blocks spanning more than one markdown block.
"""

import re
import statistics
from collections import Counter

from deep_research.agent.config import MAX_CLAIMS_CHECKED
from deep_research.agent.nodes.check_citations import CITATION_MARKER
from deep_research.agent.nodes.check_claims import BLOCK_MARKER, SENTENCE_SPLIT
from tests.eval.recording import arms, load_all

# The rule exactly as it stood before D-123: a boundary only before a capital *letter*.
OLD_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract(review: str, known_ids: set[str], pattern: re.Pattern[str]) -> list[str]:
    """`extract_claims`, parameterised by the splitter, returning just the claim text."""
    claims = []
    for sentence in pattern.split(review):
        if any(m.group(1) in known_ids for m in CITATION_MARKER.finditer(sentence)):
            claims.append(BLOCK_MARKER.sub("", sentence.strip()).strip())
    return claims


def ids_in(review: str) -> set[str]:
    """Every id the review cites.

    The recordings do not store `synthesized_from`, and the real `known_ids` only ever
    *narrows* the set. Taking every cited id therefore counts at least as many claims as the
    node would -- so the merge rate below is a lower bound, not an inflated one.
    """
    return {m.group(1) for m in CITATION_MARKER.finditer(review)}


def main() -> None:
    rows = []
    pooled: Counter[str] = Counter()
    merged_sizes: list[int] = []
    per_review: list[int] = []

    for arm in sorted(arms()):
        old_total = new_total = merged = 0
        for recording in load_all(arm):
            known = ids_in(recording.review)
            old = extract(recording.review, known, OLD_SPLIT)
            new = extract(recording.review, known, SENTENCE_SPLIT)
            old_total += len(old)
            new_total += len(new)
            per_review.append(len(new))
            for claim in old:
                # A claim that the fixed rule breaks up was a merged block.
                pieces = len(extract(claim, known, SENTENCE_SPLIT))
                if pieces > 1:
                    merged += 1
                    merged_sizes.append(pieces)
        rows.append((arm, old_total, new_total, merged))
        pooled["old"] += old_total
        pooled["new"] += new_total
        pooled["merged"] += merged

    print(f"{'arm':<28} {'old':>6} {'fixed':>6} {'merged':>7} {'merged %':>9}")
    print("-" * 60)
    for arm, old, new, merged in rows:
        share = f"{100 * merged / old:.1f}%" if old else "-"
        print(f"{arm:<28} {old:>6} {new:>6} {merged:>7} {share:>9}")
    print("-" * 60)
    old, new, merged = pooled["old"], pooled["new"], pooled["merged"]
    share = f"{100 * merged / old:.1f}%" if old else "-"
    print(f"{'POOLED':<28} {old:>6} {new:>6} {merged:>7} {share:>9}")

    if merged_sizes:
        print(
            f"\nA merged claim held {statistics.mean(merged_sizes):.1f} blocks on average, "
            f"up to {max(merged_sizes)}."
        )
        print(
            f"Claims recovered by the fix: {new - old} "
            f"({100 * (new - old) / old:.0f}% more claims checked per review)."
        )

    # More claims means more of them could hit the truncation cap, which would flip
    # `claims_checked` to False and report the review as only partly verified (D-113).
    over = sum(1 for count in per_review if count > MAX_CLAIMS_CHECKED)
    print(
        f"\nClaims per review after the fix: max {max(per_review)}, "
        f"median {statistics.median(per_review):.0f}. "
        f"Reviews over MAX_CLAIMS_CHECKED={MAX_CLAIMS_CHECKED}: {over} of {len(per_review)}."
    )


if __name__ == "__main__":
    main()
