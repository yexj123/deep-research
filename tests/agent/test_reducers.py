"""State reducer tests (D-067, D-068). Pure functions: no graph, no network, no model.

These pin the merge semantics that parallel Send workers depend on. Graph-level tests that
actually run workers in parallel live in test_graph.py; these isolate the merge rules so a
failure points at the reducer rather than at the fan-out.
"""

import operator

from deep_research.agent.state import merge_sources, merge_subtopics, normalize_subtopic
from tests.agent.fakes import make_source


# ---- merge_sources -------------------------------------------------------------------


def test_merge_sources_appends_new_papers() -> None:
    """Papers from a second worker are appended to the first worker's (D-067)."""
    a = make_source(arxiv_id="2411.18583")
    b = make_source(arxiv_id="2502.00306")
    assert [s.arxiv_id for s in merge_sources([a], [b])] == ["2411.18583", "2502.00306"]


def test_merge_sources_drops_a_paper_already_present() -> None:
    """The same arxiv_id retrieved by two subtopics is kept once (D-067).

    Without this, format_papers lists the paper twice in the <papers> block, which wastes
    tokens and lets the model weight or cite it twice -- and check_citations cannot flag
    that, because the ID really was retrieved.
    """
    a = make_source(arxiv_id="2411.18583")
    duplicate = make_source(arxiv_id="2411.18583", title="Same paper, different worker")
    merged = merge_sources([a], [duplicate])
    assert len(merged) == 1
    assert merged[0].title == a.title, "the first-seen Source is the one kept"


def test_merge_sources_collapses_different_versions_of_one_paper() -> None:
    """v1 and v2 of a paper are one paper, because arxiv_id is canonical (D-044, D-067).

    This is why the reducer keys on arxiv_id and not on the Source object. Source is frozen
    and hashable, so a set[Source] would compile and silently keep both versions as two
    separate papers.
    """
    v1 = make_source(arxiv_id="2411.18583", version=1)
    v2 = make_source(arxiv_id="2411.18583", version=2)
    assert len(merge_sources([v1], [v2])) == 1


def test_merge_sources_dedups_within_a_single_update() -> None:
    """Two copies arriving in the same update are also collapsed (D-067)."""
    a = make_source(arxiv_id="2411.18583")
    again = make_source(arxiv_id="2411.18583")
    assert len(merge_sources([], [a, again])) == 1


def test_merge_sources_does_not_mutate_the_current_value() -> None:
    """The reducer returns a new list (D-067).

    LangGraph may still hold a reference to the current value, so a reducer that mutates it
    corrupts state in a way that only shows up under parallel writes.
    """
    current = [make_source(arxiv_id="2411.18583")]
    merge_sources(current, [make_source(arxiv_id="2502.00306")])
    assert len(current) == 1, "merge_sources mutated its left argument"


def test_merge_sources_with_an_empty_update_is_a_no_op() -> None:
    """A worker that found nothing (D-021) leaves sources unchanged (D-067)."""
    current = [make_source(arxiv_id="2411.18583")]
    assert [s.arxiv_id for s in merge_sources(current, [])] == ["2411.18583"]


# ---- merge_subtopics -----------------------------------------------------------------


def test_merge_subtopics_appends_a_new_subtopic() -> None:
    """A subtopic not seen before is added (D-017, D-067)."""
    assert merge_subtopics(["Attention"], ["Positional encoding"]) == [
        "Attention",
        "Positional encoding",
    ]


def test_merge_subtopics_dedups_on_case_and_whitespace() -> None:
    """"  ATTENTION " is the same subtopic as "Attention" (D-017).

    The planner is an LLM and rephrases, so exact string matching misses repeats. This is
    the cheap half of D-022; the paper-overlap check is the other half.
    """
    assert merge_subtopics(["Attention"], ["  ATTENTION "]) == ["Attention"]


def test_merge_subtopics_stores_the_original_text_not_the_normalized_form() -> None:
    """Normalization is a comparison concern only (D-067).

    Storing the normalized key would leak lowercased text into the planner prompt, the
    review's coverage section and the UI.
    """
    assert merge_subtopics([], ["Attention in BERT"]) == ["Attention in BERT"]


def test_merge_subtopics_dedups_within_a_single_update() -> None:
    """Two spellings of one subtopic in the same update collapse to the first (D-067)."""
    assert merge_subtopics([], ["Attention", "attention"]) == ["Attention"]


def test_merge_subtopics_does_not_mutate_the_current_value() -> None:
    """The reducer returns a new list (D-067). See merge_sources for why this matters."""
    current = ["Attention"]
    merge_subtopics(current, ["Positional encoding"])
    assert current == ["Attention"], "merge_subtopics mutated its left argument"


def test_normalize_subtopic_uses_casefold_not_lower() -> None:
    """casefold handles cases lower() misses, e.g. German 'ß' -> 'ss' (D-017).

    Relevant for the same reason as D-063: this project should not assume ASCII.
    """
    assert normalize_subtopic("  Straße ") == normalize_subtopic("STRASSE")


# ---- stdlib reducers used directly ---------------------------------------------------


def test_failed_subtopics_keeps_duplicates() -> None:
    """operator.add keeps repeats, because each entry is one failed attempt (D-020, D-067).

    decompose counts occurrences to apply the N=2 retry cap, so deduplicating here would
    silently disable it.
    """
    assert operator.add(["Attention"], ["Attention"]) == ["Attention", "Attention"]


def test_seen_paper_ids_union_does_not_mutate() -> None:
    """operator.or_ builds a new set rather than updating in place (D-067)."""
    current = {"2411.18583"}
    merged = operator.or_(current, {"2502.00306"})
    assert merged == {"2411.18583", "2502.00306"}
    assert current == {"2411.18583"}, "set union mutated its left argument"
