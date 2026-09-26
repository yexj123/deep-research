"""Citation recency: O-16's primary metric, and it needs no judge.

Every other comparison in this repo leans on an LLM judge, which is noisy at n=10 and has
returned honest nulls on real retrieval differences twice (D-094, D-095). This one does not:
the publication date of a cited paper is a fact, and "the stale arm could not have cited this"
is decidable rather than arguable.

    uv run python -m tests.eval.recency <arm-a> <arm-b>

For each arm it reports, per question and pooled: how many cited papers postdate the corpus
seed. Those are exactly the papers a frozen corpus cannot reach, so the gap between the arms
**is** what staleness costs, in the units that matter.

Dates come from the arXiv id itself, which encodes YYMM of first submission. That is the only
source available for both arms: a paper the fresh arm found today is, by definition, in no
local record, so any metric reading stored metadata could measure only the stale arm.

**The resolution is one month, and that is a real limitation stated up front.** A paper
submitted on 2026-09-26 carries id `2609.*`, identical to one from 2026-09-01 -- so papers
published in the seed month but after the seed date cannot be distinguished from papers that
were already in the snapshot. This counts same-month papers as **not** post-seed, which
undercounts the fresh arm's advantage. That is the conservative direction: it can only make
staleness look smaller than it is, never larger. From October onwards the ambiguity disappears
entirely, so it costs nothing in the run this exists for.
"""

import json
import re
import statistics
import sys
from datetime import datetime

from deep_research.agent.nodes.check_citations import CITATION_MARKER
from tests.eval.recording import load_all
from tests.eval.seed_corpus import MANIFEST_FILE

# arXiv ids encode YYMM of first submission, which is all this needs and is available even
# for a paper no local record holds. "2509.04244" -> September 2025.
ARXIV_YYMM = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


def submitted(arxiv_id: str) -> tuple[int, int] | None:
    """(year, month) of first submission from the id itself, or None for the old scheme.

    Deliberately not the `published` field: the fresh arm cites papers that exist in no local
    record, so a metric depending on stored metadata could only ever measure the stale arm.
    The id is self-describing and always present.
    """
    match = ARXIV_YYMM.match(arxiv_id)
    if not match:
        return None  # pre-2007 ids carry no parseable date; counted separately
    year, month = int(match.group(1)), int(match.group(2))
    return 2000 + year, month


def after(arxiv_id: str, cutoff: datetime) -> bool | None:
    """Was this submitted after the seed? None when the id carries no date."""
    stamp = submitted(arxiv_id)
    if stamp is None:
        return None
    return stamp > (cutoff.year, cutoff.month)


def cited_ids(review: str) -> set[str]:
    return {m.group(1) for m in CITATION_MARKER.finditer(review)}


def report(arm: str, cutoff: datetime, seeded_ids: set[str]) -> dict[str, float]:
    recordings = sorted(load_all(arm), key=lambda r: r.id)
    if not recordings:
        raise SystemExit(f"no recordings for arm {arm!r}")

    print(f"\n{arm}")
    print(f"  {'question':<14} {'cited':>6} {'post-seed':>10} {'in snapshot':>12} {'undated':>8}")
    print("  " + "-" * 54)

    totals = {"cited": 0, "post": 0, "in_snap": 0, "undated": 0}
    per_question_post: list[int] = []
    for recording in recordings:
        ids = cited_ids(recording.review)
        post = sum(1 for i in ids if after(i, cutoff) is True)
        undated = sum(1 for i in ids if after(i, cutoff) is None)
        in_snap = len(ids & seeded_ids)
        totals["cited"] += len(ids)
        totals["post"] += post
        totals["in_snap"] += in_snap
        totals["undated"] += undated
        per_question_post.append(post)
        print(f"  {recording.id:<14} {len(ids):>6} {post:>10} {in_snap:>12} {undated:>8}")

    print("  " + "-" * 54)
    print(
        f"  {'TOTAL':<14} {totals['cited']:>6} {totals['post']:>10} "
        f"{totals['in_snap']:>12} {totals['undated']:>8}"
    )
    share = 100 * totals["post"] / totals["cited"] if totals["cited"] else 0.0
    print(f"  post-seed share: {share:.1f}%   mean per question: {statistics.mean(per_question_post):.1f}")
    return {"post": totals["post"], "cited": totals["cited"], "share": share}


def main(arm_a: str, arm_b: str) -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    cutoff = datetime.fromisoformat(manifest["seeded_at"])
    seeded_ids = {p["arxiv_id"] for q in manifest["questions"] for p in q["papers"]}

    print(f"corpus seeded {cutoff.date()} -- 'post-seed' means submitted after {cutoff.year}-{cutoff.month:02d}")
    a = report(arm_a, cutoff, seeded_ids)
    b = report(arm_b, cutoff, seeded_ids)

    print("\n" + "=" * 60)
    print(f"post-seed citations: {a['post']} ({arm_a})")
    print(f"                     {b['post']} ({arm_b})")
    print(f"share:               {a['share']:.1f}%  vs  {b['share']:.1f}%")
    print(
        "\nAt t=0 both arms should be near zero: nothing has been published since the seed\n"
        "yet, so there is nothing for a frozen corpus to miss. A gap opening here would mean\n"
        "the metric is measuring something other than age."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m tests.eval.recency <arm-a> <arm-b>")
    main(sys.argv[1], sys.argv[2])
