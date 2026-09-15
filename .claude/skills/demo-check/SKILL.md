---
description: Pre-demo smoke test — confirms the server starts, a research run actually streams end to end, and nothing's broken. Run this before showing the app to a judge, an advisor, or anyone else.
disable-model-invocation: true
---

Run a smoke test of the whole app, not just the agent in isolation — a web
app has more failure surface than a script, and "the agent works" doesn't
mean "the demo works."

1. Start the backend and confirm it comes up clean (no import errors, no
   silently-failed startup).
2. Kick off one real research run through the actual HTTP route, not a
   direct Python call, and confirm the SSE stream delivers `updates`,
   `custom`, and `messages` events, in order, without stalling.
3. Confirm the run reaches `synthesize` and produces a report with actual
   citations — not a truncated or empty one from an earlier bug.
4. If there's a frontend, load it and watch for console errors, not just a
   visually-fine page.
5. Report anything flaky as flaky, not passing — a demo that works 8/10
   times is a liability, not a pass.

Keep this to what would actually embarrass me live: a broken import, a
stalled stream, a report with no sources, a console full of red. Don't turn
this into a full test-suite run — that's what `pytest` is for.
