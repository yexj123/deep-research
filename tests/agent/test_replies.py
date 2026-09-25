"""Reading JSON out of a model reply (D-117).

Found by /demo-check, not by any test: asked for JSON, a real model returned its object
wrapped in a markdown fence, and `model_validate_json` rejected it on the first character.
The claim checker reported itself unable to read its own reply -- on a live run, after the
same code had worked in an isolated test that happened to get a bare object back.

The same vulnerability was in `decompose` from the start, and is the most likely explanation
for D-089's measured "1 run in 10 died with ValidationError on an unparseable plan".
"""

import pytest

from deep_research.agent.replies import strip_code_fence


def test_a_fenced_reply_is_unwrapped() -> None:
    """The exact shape /demo-check captured from a live model."""
    reply = '```json\n{"judgements": [{"claim": 1, "supported": true}]}\n```'
    assert strip_code_fence(reply) == '{"judgements": [{"claim": 1, "supported": true}]}'


def test_a_fence_without_a_language_tag_is_unwrapped() -> None:
    """Models emit both ```json and a bare ```."""
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_an_unfenced_reply_is_untouched() -> None:
    """Most replies are bare, so this must be safe to apply to every one of them."""
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_surrounding_whitespace_is_tolerated() -> None:
    """A trailing newline after the closing fence is common and must not defeat the match."""
    assert strip_code_fence('\n```json\n{"a": 1}\n```\n\n') == '{"a": 1}'


def test_a_fence_inside_the_json_is_not_treated_as_the_wrapper() -> None:
    """Only a fence wrapping the WHOLE reply is stripped.

    A string value containing backticks must survive, or the fix corrupts the very data it is
    meant to recover.
    """
    reply = '{"why": "the paper writes ```code``` here"}'
    assert strip_code_fence(reply) == reply


def test_prose_around_the_json_is_not_dug_out() -> None:
    """Deliberately NOT a general "find the JSON anywhere" helper (D-117).

    A model returning prose around its object is not following the prompt, and silently
    extracting the object would hide that -- the failure would move from loud to invisible,
    which is the direction this project always refuses.
    """
    reply = 'Sure! Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert strip_code_fence(reply) == reply
