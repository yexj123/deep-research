# Evaluation harness (O-11)

Two commands, deliberately separate:

```sh
uv run pytest -m record   # run the real agent over the question set, save the results
uv run pytest -m eval     # score those saved results with an LLM judge
```

Neither runs on `uv run pytest`. Both cost money.

## Why record and score are separate

Same reason `tests/agent/fixtures/arxiv/` holds saved arXiv responses: **capture real output
once, use it repeatedly.**

Recording runs the agent for real — about 35 seconds per question, several model calls, and
up to nine arXiv requests spaced by the rate limiter. Scoring only reads what was saved.

That split buys three things:

- **Changing a metric doesn't re-run the agent.** Adding a metric or adjusting a threshold is
  cents and seconds, not minutes and dollars.
- **Re-scoring with a different judge stays comparable**, because the thing being judged is
  byte-identical.
- **A recording is the baseline artifact.** When O-13 adds full text, its recordings carry
  `retrieval_unit: "full_text"` and get compared against today's `"abstract"` ones. Without
  fixed recordings there is nothing to compare *to*.

## What gets measured

**Deterministic, free, no judge** — and these are the stronger evidence:

| Check | What it means |
|---|---|
| `citation_violations == []` | Every cited ID is a paper the run actually retrieved (D-046, D-062) |
| `retrieval_context` non-empty | The run found papers, so a faithfulness score is meaningful |

**Judged** — the gap the deterministic checks cannot close:

| Metric | What it means |
|---|---|
| Faithfulness | Is each claim in the review supported by the abstracts it was written from? |
| Answer relevancy | Does the review answer the question that was asked? |

Faithfulness exists because `check_citations` verifies the *ID*, not the *claim*:

> *"Smith et al. showed X [arXiv:1234.5678]"* — valid ID, paper retrieved, and the paper never
> says X.

Relevancy is separate because the failure modes are independent: decomposition can drift into
adjacent subtopics while every individual claim stays perfectly supported.

## Rules that keep the numbers meaningful

**`questions.json` is append-only.** Add questions; never reword one. A reworded question
silently invalidates every recording that used it, and you will not notice.

**The judge model is pinned** (`gpt-4o-mini`) and printed in every assertion message. A
different judge gives different numbers, so an unrecorded judge makes two runs incomparable —
the same mistake as an unrecorded `max_depth` (D-079).

**Each recording stores its `settings`** — provider, model, `max_depth`, `max_subtopics`,
`retrieval_unit`. A recording that doesn't say how it was produced is not evidence.

**Thresholds are floors, not quality gates.** They're set low on purpose. A red test here
means "the model wrote a worse review today", which is not a code regression. Raise them only
once there's evidence of what this system actually scores.

## Cost

Recording: ten runs, each several model calls — the expensive half, run rarely.
Scoring: roughly $0.001–0.003 per test case, so a full sweep is cents.

Re-record a single question rather than the whole set:

```sh
uv run pytest -m record -k attention
```
