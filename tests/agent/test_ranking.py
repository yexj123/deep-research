"""BM25 ranking of a run's own papers (D-091). No network, no model, no corpus.

Why this exists at all is a measurement, not a hunch: across ten baseline runs (D-090), 797
papers reached synthesis prompts and 81 were cited -- and the cited count was 6-12 whether 59
or 89 papers were supplied. The surplus was ~26k wasted tokens per run buying no coverage.

Two behaviours here invert silently if got wrong, so both are pinned: bm25()'s sign, and the
lowercasing that neutralizes FTS5's word operators.
"""

import pytest

from deep_research.agent.ranking import rank_sources
from tests.agent.fakes import make_source


def paper(arxiv_id: str, title: str, summary: str = "generic abstract text"):
    return make_source(arxiv_id=arxiv_id, title=title, summary=summary)


ATTENTION = paper("2401.00001", "Scaled Dot-Product Attention in Transformers",
                  "We analyse attention weights across transformer layers.")
MAMBA = paper("2401.00002", "Mamba: Selective State Space Models",
              "A selective state space model with linear-time sequence scaling.")
PROTEIN = paper("2401.00003", "Protein Folding with Graph Networks",
                "Predicting tertiary structure from amino acid sequences.")


# ---- ordering: the whole point ------------------------------------------------------


def test_the_most_relevant_paper_comes_first() -> None:
    """Ranking is by relevance to the question, best first (D-091).

    This is the assertion that catches a sign error: `bm25()` returns the NEGATIVE of the
    standard score, so `ORDER BY bm25(t) DESC` would return the *worst* matches with no error
    and entirely plausible-looking output.
    """
    ranked = rank_sources([PROTEIN, MAMBA, ATTENTION], "attention in transformers", top_n=1)
    assert [s.arxiv_id for s in ranked] == [ATTENTION.arxiv_id]


def test_exact_jargon_outranks_topically_similar_papers() -> None:
    """"Mamba" retrieves the Mamba paper (D-091).

    The argument for BM25 over embeddings: a 384-dim model blurs rare tokens like Mamba,
    FlashAttention or LoRA, and those are exactly what academic search runs on.
    """
    ranked = rank_sources([ATTENTION, PROTEIN, MAMBA], "mamba state space", top_n=1)
    assert ranked[0].arxiv_id == MAMBA.arxiv_id


# ---- the switchable baseline arm (D-090) --------------------------------------------


def test_top_n_none_returns_everything_unchanged() -> None:
    """None means "all", preserving order -- the pre-ranking behaviour (D-091).

    Kept switchable so both arms of the comparison come from one codebase rather than from git
    history, the same reason O-13 keeps the abstract tier as a baseline.
    """
    sources = [PROTEIN, MAMBA, ATTENTION]
    assert rank_sources(sources, "attention", top_n=None) == sources


def test_ranking_happens_even_when_nothing_is_cut() -> None:
    """With fewer papers than the cap, they are still ordered by relevance (D-091).

    A function called rank_sources should always rank: otherwise the prompt's ordering would
    depend on how many papers a run happened to retrieve, which is not a property anyone
    would expect or notice changing.
    """
    ranked = rank_sources([PROTEIN, ATTENTION], "attention in transformers", top_n=20)

    assert len(ranked) == 2
    assert ranked[0].arxiv_id == ATTENTION.arxiv_id


def test_the_cap_is_respected() -> None:
    """Exactly top_n papers survive."""
    assert len(rank_sources([ATTENTION, MAMBA, PROTEIN], "attention", top_n=2)) == 2


# ---- papers that match nothing ------------------------------------------------------


def test_papers_matching_nothing_are_kept_after_the_ranked_ones() -> None:
    """Non-matching papers fill remaining slots rather than being dropped (D-091).

    BM25 scores are corpus-relative: a term in more than half the papers gets degenerate IDF
    and scores collapse toward zero regardless of match quality (measured: 25 of 30 documents
    scored -0.000). Dropping non-matchers would make the cut depend on that artifact instead
    of on relevance.
    """
    ranked = rank_sources([PROTEIN, MAMBA, ATTENTION], "attention transformers", top_n=2)

    assert ranked[0].arxiv_id == ATTENTION.arxiv_id, "the match ranks first"
    assert len(ranked) == 2, "a non-matching paper fills the remaining slot"


def test_a_question_matching_nothing_still_returns_papers() -> None:
    """A run that already has results must not lose them to a failed ranking (D-091)."""
    ranked = rank_sources([ATTENTION, MAMBA, PROTEIN], "xyzzy plugh", top_n=2)
    assert len(ranked) == 2


def test_a_stopword_only_question_falls_back_instead_of_raising() -> None:
    """search_terms raises on an unsearchable question; ranking degrades rather than failing.

    decompose filters these before dispatch (D-073), so reaching here means ranking has
    nothing to go on -- not that the run is broken. Failing would throw away results the run
    already paid for.
    """
    sources = [ATTENTION, MAMBA, PROTEIN]
    ranked = rank_sources(sources, "What is it?", top_n=2)

    # top_n is still honoured: returning everything would quietly restore the ~90% surplus
    # this function exists to remove, on exactly the inputs nobody checks.
    assert len(ranked) == 2


# ---- FTS5 syntax safety (shared with the arXiv path) --------------------------------


@pytest.mark.parametrize(
    "question",
    [
        'attention" OR title:"x',   # quote + column filter
        "attention AND transformer",  # word operator, uppercase
        "attention NEAR/3 transformer",
        "attention* (transformer)",
        '"unterminated quote',
    ],
)
def test_fts5_syntax_in_a_question_cannot_break_ranking(question: str) -> None:
    """A question carrying FTS5 syntax must not raise, and must not act as an operator.

    Not hypothetical: an unstripped quote makes FTS5 raise
    `OperationalError: unterminated string`, killing a run that had already retrieved its
    papers. `search_terms` strips the symbolic operators (D-051) and lowercases the word ones
    -- and the lowercasing is a *security* property, since FTS5's operators are case-sensitive.
    Preserving the planner's capitalization to look tidier would reintroduce this silently.
    """
    ranked = rank_sources([ATTENTION, MAMBA, PROTEIN], question, top_n=2)
    assert len(ranked) == 2


def test_ranking_does_not_mutate_the_input() -> None:
    """The caller's list is left alone -- synthesize still holds state.sources."""
    sources = [PROTEIN, MAMBA, ATTENTION]
    rank_sources(sources, "attention", top_n=1)
    assert [s.arxiv_id for s in sources] == [PROTEIN.arxiv_id, MAMBA.arxiv_id, ATTENTION.arxiv_id]
