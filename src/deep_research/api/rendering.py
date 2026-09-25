"""Turn a model-authored review into HTML that is safe to put in the DOM (O-8).

The review is written by a model that has just read arXiv abstracts -- untrusted third-party
text (D-055). Rendering it to HTML is therefore an injection path, not a hypothetical one: an
abstract could carry markup the model copies into its answer.

Rendered on the server rather than in the browser so the escaping is in Python, where it is
testable, and so the page needs no client-side markdown or sanitizer library.
"""

from markdown_it import MarkdownIt

from deep_research.agent.coverage import Coverage
from deep_research.agent.exits import NOT_FINISHED

# html=False is the whole defense, and it is NOT the default: MarkdownIt() ships with
# html=True, which passes raw HTML straight through -- verified 2026-09-21, `<script>` tags
# in the source appeared verbatim in the output. With it off, raw HTML is escaped to text.
#
# linkify=False keeps bare URLs as text: the review's citations are [arXiv:<id>] markers
# (D-046), and auto-linking arbitrary strings the model produced is one more thing to reason
# about for no benefit.
_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": False})


def render_review(markdown: str) -> str:
    """Render review markdown to HTML with raw HTML escaped (O-8).

    Also confirmed safe by default in markdown-it-py: `javascript:` and `data:` URLs in
    markdown link syntax are not turned into links at all.
    """
    return _RENDERER.render(markdown)


def render_coverage(coverage: Coverage) -> str:
    """The "Coverage and limitations" section, as HTML (O-5).

    Built from state rather than asked of the model, for the same reason `decompose` filters
    in code rather than trusting the prompt (D-070, D-073): the code knows exactly what was
    lost, and a model asked to confess its own gaps may simply not.

    **Always rendered, including for a run that lost nothing (O-15).** It used to be hidden
    behind `is_complete`, which was right while the panel only listed losses. D-096 changed
    what it has to say: runs now stop after a single round, and a reader who watches three
    searches and then synthesis has no way to learn why it did not go deeper. "Enough papers
    were found to fill the synthesis context" reads as confidence; its absence invites "did it
    give up?" -- the question a judge asks about the project's headline feature.

    So the *stop reason* is unconditional and the *loss list* stays conditional, which is the
    distinction D-086 conflated. The heading follows suit: a clean run is headed "Coverage",
    not "Coverage and limitations", so the panel does not announce limitations that are not
    there.

    Every value goes through render_review, so a subtopic containing markup is escaped like
    anything else (D-085).
    """
    # An unfinished run has no stop reason, and "stopped because the run has not finished" is
    # nonsense that would now be on screen rather than hidden (D-084, D-096).
    finished = coverage.stopped_because != NOT_FINISHED
    searched = (
        f"Searched {len(coverage.explored)} subtopic(s) over {coverage.rounds} round(s), "
        f"finding {coverage.papers} paper(s)."
    )
    lines = [
        "## Coverage" if coverage.is_complete else "## Coverage and limitations",
        "",
        f"{searched} The run stopped because {coverage.stopped_because}."
        if finished
        else f"{searched} This run has not finished.",
        "",
    ]
    if coverage.local:
        # Stated in the always-visible part, not the loss list: answering from the corpus is
        # not a failure, but a reader deciding how current a review is needs to know that some
        # of it came from a cache rather than from today's arXiv (O-13, D-105).
        lines += [
            f"**Answered from the local corpus:** {', '.join(coverage.local)}. "
            "No new arXiv search was performed for those subtopics, so papers published since "
            "they were last indexed are not represented.",
            "",
        ]
    if coverage.unsupported_claims:
        # First, because it is the strongest caveat the panel carries: the other entries say
        # what the review did not cover, this one says part of what it *did* say may not be
        # backed by the paper it credits.
        lines += [
            f"**{len(coverage.unsupported_claims)} claim(s) may not be supported by the "
            "paper they cite.** A model checked each cited sentence against the text the "
            "review was written from; these are the ones it could not verify. It is a "
            "model judging a model, so treat it as a prompt to check rather than a verdict:",
            "",
        ]
        lines += [f"- {claim}" for claim in coverage.unsupported_claims]
        lines += [""]
    elif not coverage.claims_checked:
        # "Not checked" and "checked, found none" must not look identical (D-084).
        lines += [
            "**Claim support was not verified for this review.** The checker could not read "
            "its own reply, so the citations are ID-verified but the sentences around them "
            "are not.",
            "",
        ]
    if coverage.empty:
        lines += [
            "**No papers found for:** " + ", ".join(coverage.empty) + ".",
            "That is a finding in itself -- it may indicate a genuine gap in the literature, "
            "or a subtopic phrased in terms arXiv does not index.",
            "",
        ]
    if coverage.failed:
        attempts = ", ".join(f"{name} ({n} attempt(s))" for name, n in coverage.failed.items())
        lines += [
            "**Searches that failed:** " + attempts + ".",
            "These subtopics are absent from the review above, so its coverage is incomplete.",
            "",
        ]
    if coverage.skipped_entries:
        lines += [
            f"**{coverage.skipped_entries} arXiv entr(ies) were skipped** as malformed and "
            "are not represented in the review.",
            "",
        ]
    return render_review("\n".join(lines))
