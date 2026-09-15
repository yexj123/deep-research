---
description: Prep for the HackerRank Orchestrate AI Judge interview, or a thesis-advisor/GitHub-issue walkthrough — reviews recent work and drafts talking points. Run before either, or after a milestone to check you can explain what you just built.
disable-model-invocation: true
---

Review the current state of the project (`git log --oneline -20`, `git diff`
against the last commit, and the Architecture section of CLAUDE.md) and help
me prepare to explain it out loud — to the AI Judge if I'm still in the
24-hour window, my thesis advisor if I'm past it, or a stranger reading the
GitHub README either way.

1. Summarize what's implemented so far in plain terms, as if explaining it
   from scratch.
2. For each major design decision (recursion/depth control, fan-out via
   `Send`, citation grounding, state schema, the FastAPI/SSE streaming
   layer): state the decision and the *reason* — probing the "why," not
   just the "what," is the point of this exercise.
3. List the 5 toughest questions a skeptical reviewer would ask given what's
   actually in the code right now. Favor edge cases (a topic shaped to
   recurse forever, a source that 404s, a dropped SSE connection mid-stream)
   over generic ones.
4. Flag anything in the code I might struggle to explain live — a part that
   looks like it was written faster than it was understood.

Keep this conversational; don't write a report file unless I ask for one.
