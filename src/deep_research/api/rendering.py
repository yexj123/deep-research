"""Turn a model-authored review into HTML that is safe to put in the DOM (O-8).

The review is written by a model that has just read arXiv abstracts -- untrusted third-party
text (D-055). Rendering it to HTML is therefore an injection path, not a hypothetical one: an
abstract could carry markup the model copies into its answer.

Rendered on the server rather than in the browser so the escaping is in Python, where it is
testable, and so the page needs no client-side markdown or sanitizer library.
"""

from markdown_it import MarkdownIt

from deep_research.agent.coverage import Coverage

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

    Returns "" for a run that lost nothing, so a clean review isn't padded with a list of
    nothing. Every value goes through render_review, so a subtopic containing markup is
    escaped like anything else (D-085).
    """
    lines = [
        "## Coverage and limitations",
        "",
        f"Searched {len(coverage.explored)} subtopic(s) over {coverage.rounds} round(s), "
        f"finding {coverage.papers} paper(s). The run stopped because "
        f"{coverage.stopped_because}.",
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
