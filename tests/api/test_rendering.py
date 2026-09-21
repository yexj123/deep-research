"""Review rendering tests (O-8): model-authored markdown must not become live HTML.

The review is written by a model that has just read untrusted arXiv abstracts (D-055), so an
abstract carrying markup the model copies into its answer is a real injection path -- not a
hypothetical one. These pin that the rendered output is inert.
"""

import pytest

from deep_research.api.rendering import render_review


@pytest.mark.parametrize(
    ("label", "markdown"),
    [
        ("script tag", "Findings <script>alert('xss')</script> follow."),
        ("img onerror", '<img src=x onerror="alert(1)">'),
        ("svg onload", "<svg onload=alert(1)>"),
        ("tag breakout", "</p><script>alert(1)</script>"),
        ("iframe", '<iframe src="https://example.com"></iframe>'),
    ],
)
def test_raw_html_is_escaped_not_executed(label: str, markdown: str) -> None:
    """Raw HTML in the review is rendered as visible text, never as markup (O-8).

    This is NOT markdown-it-py's default: MarkdownIt() ships with html=True, which passes raw
    HTML straight through -- verified 2026-09-21, a <script> tag appeared verbatim in the
    output. The renderer is constructed with html=False for exactly this reason, and this test
    is what would catch that option being dropped.
    """
    html = render_review(markdown)

    assert "<script" not in html
    assert "<img" not in html
    assert "<svg" not in html
    assert "<iframe" not in html
    assert "&lt;" in html, f"{label}: the markup should survive as escaped, visible text"


@pytest.mark.parametrize(
    "scheme", ["javascript:alert('xss')", "data:text/html,<script>alert(1)</script>"]
)
def test_dangerous_link_schemes_do_not_become_links(scheme: str) -> None:
    """javascript: and data: URLs in markdown link syntax are not linkified (O-8).

    markdown-it-py blocks these by default, unlike html=True. Pinned anyway: it's a default
    being relied on, and a default that changes silently is exactly what bites later.
    """
    html = render_review(f"[click me]({scheme})")
    assert "<a " not in html


def test_ordinary_review_markdown_renders() -> None:
    """The formatting the synthesize prompt actually asks for still works (D-046).

    Escaping is worthless if it also breaks the headings and lists the prompt requires --
    this is the control that proves the renderer isn't just refusing everything.
    """
    html = render_review("## Overview\n\n**bold** text\n\n- one\n- two")

    assert "<h2>Overview</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<li>one</li>" in html


def test_a_citation_marker_survives_rendering() -> None:
    """[arXiv:<id>] passes through as text (D-046).

    Markdown treats [x] as a link only when followed by (...) or a reference definition, so
    the citation format is safe from being eaten -- but it is load-bearing enough to pin.
    """
    assert "[arXiv:2411.18583]" in render_review("As shown [arXiv:2411.18583].")


def test_an_empty_review_renders_to_nothing() -> None:
    """A run with no review yet (D-081's not-started state) renders empty, not an error."""
    assert render_review("") == ""
