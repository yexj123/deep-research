"""Reading JSON out of a model reply (D-117).

**Found by `/demo-check`, not by any test.** Asked for JSON, a real model returned:

    ```json
    {"judgements": [{"claim": 1, "supported": true}]}
    ```

`model_validate_json` rejects that on the first character, so the claim checker reported
itself unable to read its own reply -- on a live run, after the same code had worked in an
isolated test that happened to get a bare object back.

**The same vulnerability was in `decompose` from the beginning**, and it is the most likely
explanation for a measured failure this project has carried since D-089: *"Planner JSON
failure: 1 run in 10 died with `ValidationError` on an unparseable plan, and the same question
succeeded on retry."* A fence appearing sometimes and not others fits that exactly -- a retry
of the same prompt is a fresh sample, and most samples are unfenced.

That connection is **plausible, not proven**: D-089 recorded the failure rate, not the reply
text that caused it. The fix is harmless either way, and cheaper than the repair-retry D-070
left open.

Deliberately *not* a general "extract JSON from anywhere" helper. It strips a fence that
wraps the whole reply and nothing else: a model that returns prose around its JSON is not
following the prompt, and silently digging the object out would hide that.
"""

import re

# ```json\n{...}\n```  or  ```\n{...}\n``` -- the whole reply, optionally with a language tag.
CODE_FENCE: re.Pattern[str] = re.compile(
    r"\A\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.S
)


def strip_code_fence(text: str) -> str:
    """Return `text` with a surrounding markdown code fence removed, if there is one.

    Unfenced text is returned unchanged, so this is safe to apply to every reply.
    """
    match = CODE_FENCE.match(text)
    return match.group("body") if match else text
