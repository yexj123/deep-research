"""Did the stale arm stay stale? (O-16)

`research_worker` falls through to live arXiv for any subtopic the corpus does not cover
(D-105). In the stale arm that fallback is **contamination**: it puts papers published after
the seed date into the arm whose defining property is that it cannot see them.

Nothing else would notice. The run succeeds, the review is good, and the comparison silently
measures a weaker effect than it claims to.

    uv run python -m tests.eval.stale_purity <arm>

Reports, per question: how many subtopics were answered locally, and whether any cited paper
postdates the seed -- which is the fact that would prove contamination rather than infer it.
"""

import json
import sys
from datetime import datetime

from deep_research.agent.nodes.check_citations import CITATION_MARKER
from tests.eval.recording import load_all
from tests.eval.seed_corpus import MANIFEST_FILE

STALE_ARM = "staleness-abstract-stale-top20-d2-adaptive"


def ids_in(text: str) -> set[str]:
    """Every arXiv id a block of text cites.

    Used on both the review and the retrieval context. A `Recording` stores neither as a list
    of ids -- an earlier version of this script read a `synthesized_from` key that does not
    exist on this dataclass, so every question reported zero cited papers and the arm was
    declared clean **vacuously**. Parsing the text is what the data actually supports.
    """
    return {m.group(1) for m in CITATION_MARKER.finditer(text)}


def main(arm: str = STALE_ARM) -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    seeded_at = datetime.fromisoformat(manifest["seeded_at"])
    # Every paper the snapshot holds, so a cited id outside this set came from somewhere else.
    seeded_ids = {p["arxiv_id"] for q in manifest["questions"] for p in q["papers"]}

    recordings = load_all(arm)
    if not recordings:
        raise SystemExit(f"no recordings for arm {arm!r}")

    print(f"arm: {arm}")
    print(f"seeded_at: {seeded_at.isoformat()}\n")
    print(f"{'question':<14} {'local/explored':>15} {'prompt':>7} {'cited':>6} {'off-snapshot':>13}")
    print("-" * 62)

    totals = {"prompt": 0, "cited": 0, "off_prompt": 0, "off_cited": 0, "fellthrough": 0}
    for recording in sorted(recordings, key=lambda r: r.id):
        coverage = recording.coverage
        local = len(coverage.get("local", []))
        explored = len(coverage.get("explored", []))
        # Two populations, both worth testing. The prompt set is what the writer could see;
        # the cited set is what reached the reader. A paper can contaminate the first without
        # reaching the second -- the distinction D-091 forced on `check_citations`.
        prompt_ids = ids_in("\n".join(recording.retrieval_context))
        cited_ids = ids_in(recording.review)
        off_prompt = prompt_ids - seeded_ids
        off_cited = cited_ids - seeded_ids

        totals["prompt"] += len(prompt_ids)
        totals["cited"] += len(cited_ids)
        totals["off_prompt"] += len(off_prompt)
        totals["off_cited"] += len(off_cited)
        totals["fellthrough"] += explored - local

        flag = "  <-- CONTAMINATED" if off_prompt or off_cited else ""
        print(
            f"{recording.id:<14} {f'{local}/{explored}':>15} {len(prompt_ids):>7} "
            f"{len(cited_ids):>6} {len(off_prompt):>13}{flag}"
        )

    print("-" * 62)
    print(
        f"{'TOTAL':<14} {'':>15} {totals['prompt']:>7} {totals['cited']:>6} "
        f"{totals['off_prompt']:>13}"
    )
    print(f"\nsubtopics that fell through to live arXiv: {totals['fellthrough']}")
    print(f"off-snapshot papers in a prompt:           {totals['off_prompt']}")
    print(f"off-snapshot papers actually cited:        {totals['off_cited']}")

    if totals["off_prompt"] or totals["fellthrough"]:
        print(
            "\nCONTAMINATED. A subtopic the snapshot did not cover fell through to live arXiv,\n"
            "so this arm can see papers published after the seed. Report it with the results:\n"
            "it bounds how much of any STALE-vs-FRESH null is really 'the stale arm was not\n"
            "stale'."
        )
    else:
        print("\nClean: every paper in every prompt came from the snapshot.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else STALE_ARM)
