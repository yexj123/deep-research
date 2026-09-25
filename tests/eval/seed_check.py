"""Does the seeded snapshot actually work as the stale arm? (O-16)

Seeding a corpus that cannot answer anything would be a silent failure of exactly this
project's recurring shape (D-062, D-069, O-5): the artifact exists, the manifest looks right,
and the experiment is discovered to be impossible a month from now when it is too late to
re-seed at the correct date.

So this asserts the two things the experiment depends on, today:

1. **Coverage** -- each question's facets clear `MIN_LOCAL_PAPERS` under the real retrieval
   function, so `research_worker` would answer locally rather than falling through to arXiv.
2. **Dating** -- every paper carries the manifest's `seeded_at`, so a max-age rule has
   something to bite on.
3. **Held-out coverage**, which is the only one of the three that is not circular. The facets
   in (1) are the queries the snapshot was *seeded with*, so of course they hit. What the
   experiment actually needs is that a subtopic the planner invents months from now -- which
   nobody seeded -- still finds papers. HELD_OUT below is that test: plausible subtopics,
   deliberately not in any facet list.

    uv run python -m tests.eval.seed_check
"""

import json

from deep_research.agent.config import LOCAL_SEARCH_TOP_K, MIN_LOCAL_PAPERS
from deep_research.persistence.corpus import ABSTRACT, connect, strongly_matching_papers
from tests.eval.seed_corpus import MANIFEST_FILE, SNAPSHOT_DB, load_questions, seed_queries

# Subtopics a planner could plausibly produce for each question, written **after** seeding and
# deliberately sharing no wording with any facet. If these miss, the stale arm falls straight
# through to arXiv and the experiment measures nothing -- so this is the check that decides
# whether the snapshot is usable, not the facet coverage above.
HELD_OUT: dict[str, list[str]] = {
    "rag-halluc": ["knowledge conflict parametric memory", "citation attribution in generated answers"],
    "agent-tools": ["multi step reasoning with external APIs", "failure recovery in autonomous agents"],
    "diffusion-lm": ["non autoregressive text generation", "denoising schedules for discrete data"],
    "moe-routing": ["sparse activation transformer efficiency", "capacity factor expert dropping"],
    "test-time": ["self consistency sampling", "process reward models verification"],
    "speculative": ["multi token prediction", "lossless acceleration of autoregressive sampling"],
    "pca-dim": ["singular value decomposition feature extraction", "whitening transformation"],
    "svm-kernel": ["maximum margin classification", "quadratic programming dual formulation"],
    "hmm-speech": ["acoustic modeling phoneme states", "forward backward parameter estimation"],
    "crf-tagging": ["named entity recognition sequence model", "part of speech tagging discriminative"],
}


def main() -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    db = connect(str(SNAPSHOT_DB))
    try:
        dates = {row[0] for row in db.execute("SELECT DISTINCT indexed_at FROM papers")}
        total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        manifest_total = sum(q["paper_count"] for q in manifest["questions"])
        distinct = len(
            {p["arxiv_id"] for q in manifest["questions"] for p in q["papers"]}
        )

        print(f"papers in snapshot : {total}")
        print(f"manifest rows      : {manifest_total} ({manifest_total - distinct} shared "
              f"across questions, deduplicated on the primary key)")
        print(f"distinct ids       : {distinct}")
        print(f"indexed_at values  : {dates}")
        assert total == distinct, "snapshot lost papers the manifest holds"
        assert dates == {manifest["seeded_at"]}, "back-stamp did not apply to every row"

        print(f"\ncoverage at MIN_LOCAL_PAPERS={MIN_LOCAL_PAPERS}, top_k={LOCAL_SEARCH_TOP_K}:")
        print(f"  {'question':<14} {'arm':<5} {'facets covered':>15}")
        print("  " + "-" * 38)
        uncovered = []
        for entry in load_questions():
            queries = seed_queries(entry)
            covered = 0
            for query_text in queries:
                ids = strongly_matching_papers(
                    db, query_text, LOCAL_SEARCH_TOP_K, ABSTRACT
                )
                if len(ids) >= MIN_LOCAL_PAPERS:
                    covered += 1
                else:
                    uncovered.append((entry["id"], query_text, len(ids)))
            print(f"  {entry['id']:<14} {entry['arm']:<5} {covered:>10}/{len(queries)}")

        if uncovered:
            print("\nNOT covered -- these would fall through to arXiv in the stale arm:")
            for question_id, query_text, found in uncovered:
                print(f"  {question_id}: {query_text!r} -> {found} paper(s)")
        else:
            print("\nEvery seeded query is answerable from the snapshot alone.")

        # The non-circular check. See HELD_OUT.
        print("\nheld-out subtopics (never seeded):")
        print(f"  {'question':<14} {'covered':>9}   {'papers per subtopic':>20}")
        print("  " + "-" * 48)
        hits = misses = 0
        for question_id, subtopics in HELD_OUT.items():
            counts = [
                len(strongly_matching_papers(db, s, LOCAL_SEARCH_TOP_K, ABSTRACT))
                for s in subtopics
            ]
            covered = sum(1 for c in counts if c >= MIN_LOCAL_PAPERS)
            hits += covered
            misses += len(counts) - covered
            print(f"  {question_id:<14} {covered:>6}/{len(counts)}   {counts}")
        print("  " + "-" * 48)
        print(f"  held-out coverage: {hits}/{hits + misses}")
        if misses:
            print(
                "\n  Some held-out subtopics miss. That is a finding, not necessarily a\n"
                "  failure: it bounds how far the snapshot generalises, and the stale arm\n"
                "  will search arXiv for those -- which must be reported in the results."
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
