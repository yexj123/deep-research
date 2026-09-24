"""Review rendering tests (O-8): model-authored markdown must not become live HTML.

The review is written by a model that has just read untrusted arXiv abstracts (D-055), so an
abstract carrying markup the model copies into its answer is a real injection path -- not a
hypothetical one. These pin that the rendered output is inert.
"""

import pytest

from deep_research.agent.coverage import Coverage
from deep_research.agent.exits import NOT_FINISHED, REASONS
from deep_research.api.rendering import render_coverage, render_review


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


# ---- O-15: the stop reason is always visible ----------------------------------------


def test_a_clean_run_is_headed_coverage_not_coverage_and_limitations() -> None:
    """The panel must not announce limitations a clean run does not have (O-15).

    Making the panel unconditional is only an improvement if it reads well when nothing is
    wrong. A heading that always says "and limitations" would teach a reader to expect bad
    news on every run, which is its own kind of dishonesty.
    """
    html = render_coverage(
        Coverage(explored=["a", "b"], papers=30, rounds=1, stopped_because="the prompt was full")
    )
    assert "<h2>Coverage</h2>" in html
    assert "limitations" not in html


def test_a_lossy_run_is_headed_coverage_and_limitations() -> None:
    """The other half of the same rule: when there ARE losses, say so in the heading (O-5)."""
    html = render_coverage(
        Coverage(explored=["a"], empty=["b"], papers=10, rounds=1, stopped_because="the shelf was bare")
    )
    assert "<h2>Coverage and limitations</h2>" in html
    assert "No papers found for" in html


def test_an_unfinished_run_is_not_described_as_having_stopped() -> None:
    """"The run stopped because the run has not finished" is nonsense (D-084, O-15).

    It was harmless while the panel was hidden for such runs; now that it is always rendered,
    the nonsense would be on screen. A resumable run says plainly that it has not finished.
    """
    html = render_coverage(
        Coverage(explored=["a"], papers=5, rounds=1, stopped_because=NOT_FINISHED)
    )
    assert "This run has not finished." in html
    assert "stopped because" not in html


def test_the_stop_reason_is_rendered_for_every_exit() -> None:
    """Whatever ended the run, the panel says so (O-15).

    Parameterised over the real sentences rather than a sample, so a new exit added to
    `exits.REASONS` without a thought for the UI shows up here.
    """
    for reason in REASONS.values():
        html = render_coverage(
            Coverage(explored=["a"], papers=30, rounds=1, stopped_because=reason)
        )
        assert reason in html, f"the panel dropped: {reason}"
