# What the measurements say

This project was built to answer a question, then measured to see whether it had.
**Three of its four headline ideas did not survive contact with the evaluation.**

That is the finding. Everything below is the evidence, the instrument that produced it, and
what it cost. Every table regenerates from the committed recordings with no API key; the
commands are at the bottom.

---

## The claim

> A system whose own evaluation overturns three of its four premises is a more defensible
> artifact than one that never asked.

| # | Premise, as designed | Verdict |
|---|---|---|
| 1 | **Recursive decomposition** finds literature a single search misses | **Not supported** |
| 2 | **Searching deeper** improves the review | **Not supported** |
| 3 | **Full text** lets a review say what abstracts cannot | **Not supported** |
| 4 | **A local corpus** makes repeat research cheaper | **Supported** — in cost, not quality |

---

## The instrument, built before the results

The order was deliberate: **the evaluation harness was built before the thing it would judge**
(D-088). Otherwise "full text improved the reviews" is unfalsifiable.

- **A frozen, append-only question set.** Twenty questions in two difficulty classes — ten
  broad survey prompts, ten *intersection* questions spanning two or three areas. Rewording
  one invalidates every recording made against it.
- **Record and score are separate.** Runs are captured once and judged repeatedly, so a new
  metric costs cents rather than a re-run and every arm stays byte-comparable.
- **Arms are derived from settings, never labelled by hand.** A recording physically cannot be
  filed under a configuration that did not produce it.
- **Paired analysis.** Arms answer the same questions, so the statistic is the mean of
  per-question *differences* — question difficulty varies far more than the effects measured.

### The metric was validated before it was trusted

Faithfulness and relevancy sit at **0.97 and 0.99** — at their ceiling, and unable to
distinguish a 75% context cut (D-092). A metric returning ~0.95 for everything is a number
generator, not a measurement.

So a **specificity** criterion was built, then tested on two reviews of the same question, the
same length, the same confident register, citing the same four papers, differing *only* in
whether the sentences commit to anything:

| | score |
|---|---|
| Deliberately concrete | **0.950** |
| Deliberately vague | **0.222** |

Real reviews score **~0.78**. That headroom is what makes the nulls below meaningful: the
instrument could have detected an improvement and did not.

---

## 1. Recursive decomposition — the exit could never fire

The agent re-plans while rounds keep finding new papers. Measured across ten questions:
**10 of 10 runs stopped on the depth ceiling; not one converged** (D-090).

The first diagnosis was that the planner paraphrases itself. Measuring before building the fix
showed otherwise:

- Within-run subtopic Jaccard: **median 0.14**, and every high-scoring pair inspected was two
  genuinely distinct topics.
- Rounds are near-disjoint in what they retrieve — `attention` round 2 found **30 papers, 27
  of them new**.

**No novelty-based rule can fire on that data**: not the round-level test, not subtopic
paper-overlap, not a lexical paraphrase filter. The exit was not too weak — **the premise that
a broad arXiv query runs out of new papers is false** (D-094).

A filter built to the original specification would have been dead code that passed its own
tests.

## 2. Searching deeper — 2.8× the retrieval, no gain

Three arms, same questions, one variable:

| | 1 round | 2 rounds | 3 rounds |
|---|---|---|---|
| arXiv searches | 3.0 | 6.0 | 9.1 |
| papers retrieved | 28.6 | 56.6 | **81.3** |
| **papers in the prompt** | **20.0** | **20.0** | **20.0** |
| papers cited | 8.0 | 8.7 | 8.5 |
| specificity | 0.80 | 0.79 | 0.79 |

Pooled over both question sets (20 paired questions), **specificity is −2.2 SE in favour of a
single round**, and citations are identical (+0.1 SE).

**The mechanism is one row.** Ranking truncates the prompt to twenty papers, so extra rounds
cannot enlarge what the model sees — only reshuffle which twenty win. BM25 over 28 candidates
picks as good a top-20 as BM25 over 81. Pruning and recursion turned out to be **substitutes,
not complements** (D-094).

Retested on intersection questions, where decomposition *should* win, it fails hardest:
**10.9% of deep searches return zero papers**, because the planner decomposes correctly into
literature that was never written (D-095).

> `spec-quant` round 2 asked for *"interactions between decoding and quantization in neural
> networks"* — a sensible subtopic, and arXiv has nothing at that intersection.

**What shipped instead of a lower ceiling:** adaptive exits that stop when another round
cannot change the review. **19 of 20 runs now stop after one round, −66% arXiv traffic, every
quality metric flat or better** (D-096).

## 3. Full text — 2.8× the prompt, nothing past 2 SE

The strongest remaining hypothesis, and the one specificity was built for: abstracts rarely
state measurements, and **five of ten reviews contained no numbers at all**.

365 papers fetched and chunked, 12,220 passages indexed. With retrieval and synthesis reading
the same text:

| metric | abstracts | full text | SE |
|---|---|---|---|
| prompt characters | 15,157 | **42,891** | 2.8× |
| specificity | 0.78 | 0.79 | +0.9 |
| numeric_density | 0.51 | 0.72 | +1.0 |
| faithfulness | 0.99 | 0.97 | **−1.9** |
| papers retrieved | 51.8 | 37.9 | **−8.9** |

**Nothing clears 2 SE in its favour.** Faithfulness is nominally down. Breadth falls 27%,
because one enriched paper occupies several top-k slots and crowds out distinct papers (D-111).

### The finding that outlasts the hypothesis

An earlier arm was built wrong: full text informed *retrieval* while the model still read
*abstracts*. It was caught only because `papers_retrieved` moved unexpectedly. Scored anyway,
it produced the sharpest result in the entire evaluation:

> **faithfulness 0.99 → 0.96, −3.1 SE** — the only metric to clear 2 SE in the *harmful*
> direction.

Papers were selected for body text the model never saw, handed over as abstracts that did not
support the topic, and cited anyway.

**Retrieval and synthesis must read the same text.** A retriever that selects on evidence the
writer never reads produces sources that look relevant to the machine and do not support the
claim for the reader. That is not specific to this project, and it was found only because an
accident produced a configuration nobody would have chosen to test (D-110).

It is now **unreachable in code** rather than documented as a caution: with excerpts disabled,
retrieval is restricted to the abstract tier.

## 4. The local corpus — the one that paid

Every search already returns abstracts the run pays for and discards. Indexing them costs
nothing and makes the corpus useful from the first run.

| | arXiv-only | local-first |
|---|---|---|
| **arXiv requests** | **65** | **1** |
| **wall clock** | 703 s | **248 s (−65%)** |
| specificity | 0.78 | 0.78 |
| faithfulness | 0.98 | 0.99 |
| papers cited | 7.95 | 7.90 |

**Every quality metric inside 2 SE, and the two that moved at all moved up** (D-107).

The mechanism is itself a measured result: **two runs of the same question retrieve ~89%
different papers** (mean Jaccard 11%) yet produce indistinguishable reviews. Paper *identity*
does not drive quality — topical relevance does, and the corpus has that (D-105).

Safe on a cold corpus by construction: an empty corpus covers nothing, so a first run falls
through to arXiv unchanged.

---

## What this cost

| | |
|---|---|
| Committed recordings | **140 runs across 14 arms** |
| Recording sweeps | 8 real-agent sweeps |
| Scoring sweeps | 6 judged sweeps |
| Full-text enrichment | 365 papers, ~40 minutes, ~1 GB of PDFs |

Regenerate any table with no API key and no network:

```sh
uv run python -m tests.eval.compare broad-abstract-top20-d0-fixed broad-abstract-top20-d2-fixed
uv run python -m tests.eval.compare \
    broad-abstract-local-top20-d2-adaptive broad-fulltext-local-top20-d2-adaptive \
    narrow-abstract-local-top20-d2-adaptive narrow-fulltext-local-top20-d2-adaptive
uv run python -m tests.eval.corpus_coverage
```

---

## What would overturn these results

A null result is only as strong as its stated limits:

- **n = 20 questions, one run per cell.** Effects smaller than ~2 SE are invisible here.
- **All questions are literature-review prompts.** A question requiring citation-graph
  traversal rather than topic survey is untested, and is the likeliest place recursion earns
  its keep.
- **The corpus experiment is the best case.** It was seeded from earlier runs of the same
  questions, so coverage was 100%. A corpus grown from adjacent research covers less.
- **Staleness is unmeasured.** Nothing re-fetches a subtopic the corpus already covers, so a
  topic can stay answered from old papers. Reported in the coverage panel; not yet bounded
  (O-16).
- **One judge model, one provider.** Judged numbers carry every caveat that implies.

---

## Why this is the defensible version

The alternative was a system that recursed three levels deep, fetched full text, and produced
reviews nobody had compared to anything. It would have demoed better.

What exists instead is **cheaper than its first design** — one round instead of three, one
arXiv request instead of 65 — **with no measured quality cost**, plus a written record of
every idea that did not survive being measured.

The negative results were not free. Each cost a recording sweep, a scoring sweep and an
afternoon. They are the part of this project that could not have been produced by building
faster.
