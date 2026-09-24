"""Suite-wide setup. Currently one job: keep the tests off the network (D-102).

**The problem this solves.** A developer `.env` with `LANGSMITH_TRACING_V2=true` and a
LangSmith key turns tracing on for every LangChain call the suite makes. Nothing in `src/` or
`tests/` loads that file -- the LangSmith SDK finds it on its own -- so the suite silently
acquires a network dependency that no test asks for and no test mentions.

Measured 2026-09-24: with a key that no longer authorizes, the tracer retried and then blocked
for **~11 seconds** on a flush, attributed to whichever test happened to trigger it. The suite
ran anywhere between 14 and 97 seconds depending on how many flushes landed, and the "slow
test" moved around between runs. Three separate investigations went looking for a code
regression that did not exist.

Two reasons to disable it here rather than telling people to edit their `.env`:

- **A test suite that depends on an external service is not a test suite.** It fails, or
  crawls, for reasons unrelated to the code under test, and on a machine with no network it
  does both.
- **Traces carry the content.** Research questions, retrieved abstracts and generated reviews
  would leave the machine on every `pytest` run, including the paid eval sweeps. That sits
  badly with a project whose stated rule is that users bring their own keys and nothing of
  theirs is baked in.

These are set at import time, before any LangChain or LangSmith module is loaded, because the
tracer reads them once when it initializes. `python-dotenv` does not override variables that
already exist, so setting them here wins over `.env` without touching the file.

Real tracing while developing is unaffected: this only applies under pytest. To trace a
deliberate run, use the app or a script, not the suite.
"""

import os

# Every spelling LangSmith has used. `LANGSMITH_TRACING_V2` is the one this project's .env
# happened to set, and it is neither of the two documented names -- which is exactly why all
# of them are listed rather than the one that looked right.
for _flag in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ[_flag] = "false"

# Removed, not blanked: an empty key still produces a client that tries and fails, which is
# the 401-then-block behaviour above.
for _secret in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
    os.environ.pop(_secret, None)

# DeepEval's own telemetry, for the same two reasons.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
