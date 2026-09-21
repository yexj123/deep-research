"""Turn a model-authored review into HTML that is safe to put in the DOM (O-8).

The review is written by a model that has just read arXiv abstracts -- untrusted third-party
text (D-055). Rendering it to HTML is therefore an injection path, not a hypothetical one: an
abstract could carry markup the model copies into its answer.

Rendered on the server rather than in the browser so the escaping is in Python, where it is
testable, and so the page needs no client-side markdown or sanitizer library.
"""

from markdown_it import MarkdownIt

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
