# The staleness experiment (O-16)

**Seeded 2026-09-25. Not runnable before roughly 2026-11-01.** This file exists so that
whoever runs it then — including me, having forgotten all of this — does not have to redesign
it, and cannot quietly redesign it into something easier.

---

## What O-16 asks

`LOCAL_FIRST` is on. A subtopic the corpus already covers is answered from the corpus and
arXiv is never called (D-105). Measured today that is a clear win: 65 arXiv requests became 1,
wall clock fell 65%, and every quality metric was unchanged (D-107).

But **nothing in the design ever re-fetches a subtopic the corpus already covers.** arXiv
grows by roughly 100 GB a month, so a topic researched in September can stay answered from
September's papers indefinitely — silently, which is this project's recurring failure shape
(D-062, D-069, O-5).

D-105 makes it *visible* — the coverage panel names locally-answered subtopics and warns that
papers published since are not represented. Visible is not the same as handled, and the open
question is whether it needs handling at all:

> **Does a corpus that stops refreshing actually make the review worse, and by how much?**

Until that has a number, any max-age rule is a guessed constant — the `recursion_limit = 150`
mistake that D-077 exists to prevent.

## Why the seed had to happen before the wait

The experiment compares a review written from months-old papers against one written from
current papers. **The old corpus has to exist before the months pass.** It cannot be
manufactured afterwards: `index_sources` stamps `indexed_at = now` and takes no override, so a
corpus built in November is a November corpus no matter which papers are in it.

That is the entire reason this was seeded on a date with nothing else to show for it.

## The artifact

| | |
|---|---|
| Question set | `tests/eval/questions-staleness.json` — 10 questions, frozen and append-only (D-088) |
| Manifest | `tests/eval/snapshots/corpus-seed.json` — **committed**, 1557 paper records, 2.6 MB |
| Snapshot DB | `tests/eval/snapshots/corpus-seed.sqlite` — **not committed** (`*.sqlite` is gitignored, D-010) |
| Seeded at | `2026-09-25T17:08:46Z`, recorded in the manifest |
| Contents | 1554 distinct papers (3 appear under two questions and dedupe on the primary key) |

Seeding called arXiv and **no model at all** — `search_arxiv` plus `index_sources`, both
already tested — so it cost only wall-clock under the rate limit.

**If the `.sqlite` is ever lost**, rebuild it from the committed manifest with no network:

```powershell
uv run python -m tests.eval.seed_corpus --rebuild
```

That restores the papers *and back-stamps the original `indexed_at`*. Be clear-eyed about what
that means: the date is **asserted by the manifest**, not observed by the database. That is
acceptable — the manifest is committed and dated in git, which is a stronger record than a
local file's mtime — but it is not the same claim, and a reader deserves the distinction.

Never run `seed_corpus` without `--rebuild` again. It would re-fetch from arXiv and stamp
today, destroying the only property that matters.

## The design: difference-in-differences, not before/after

Two arms differing in exactly one thing — whether retrieval reads the dated snapshot or
searches arXiv live — recorded **twice**: once at t=0 and once after the wait.

| Arm | Retrieval | Can cite papers from |
|---|---|---|
| **STALE** | the 2026-09-25 snapshot, `LOCAL_FIRST=1` | on or before 2026-09-25 |
| **FRESH** | live arXiv, `LOCAL_FIRST=0` | any date, including everything published during the wait |

**Why twice.** A single STALE-vs-FRESH comparison in November confounds two things: *age*
(September papers vs November papers) and *mechanism* (BM25 over a fixed corpus vs arXiv's
live relevance ranking). The slow arm controls for this indirectly. Recording the same pair at
t=0 — when the corpus is hours old and there is no age difference at all — measures the
mechanism gap **directly**, so November's number can have it subtracted:

```
staleness  =  (STALE − FRESH)ₙₒᵥ  −  (STALE − FRESH)ₜ₌₀
```

It also gives a noise floor. STALE reads the same frozen 1554 papers in November as it did at
t=0, so any STALE@t0-vs-STALE@Nov difference is pure run-to-run variance with retrieval held
fixed — which at n=10 could otherwise be mistaken for an effect.

The t=0 pair could only be recorded while the corpus was new, which is why it was done
immediately rather than planned for later.

### Arm names carry the corpus age

`arm_name` appends `-age{N}d`, derived from the committed manifest's `seeded_at` and the clock
— never hand-set. Two runs of "the same arm" months apart are **not** the same configuration
when corpus age is the independent variable, and `save()` overwrites by path, so without this
November's run would land on top of the baseline and destroy it silently. That is the D-094
mislabelling along a new axis, caught by the same consistency test.

The suffix appears only for the `staleness` question set, so the 140 pre-existing arm names
are untouched.

### The negative control is the point

A naive reading of a STALE-vs-FRESH difference is "age hurt the review". But the arms also
differ in *mechanism* — corpus retrieval versus arXiv search — so a difference could come from
either. D-105 measured the mechanism with a fresh corpus and found no quality difference,
which helps, but it is not the same run.

So the question set is **split by measured publication velocity**, and the slow half is the
control:

| Arm | Questions | arXiv submissions in the 30 days to seed |
|---|---|---|
| fast | `rag-halluc`, `agent-tools`, `diffusion-lm`, `moe-routing`, `test-time`, `stale-spec` | 51 – 280 |
| slow | `pca-dim`, `svm-kernel`, `hmm-speech`, `crf-tagging` | 0 – 30 |

`hmm-speech` and `crf-tagging` had **exactly zero** new submissions in that window. A stale
corpus can lose nothing on a topic that gained nothing.

- **Difference on fast, none on slow** → attributable to age. This is the result that would
  justify a max-age rule, and the fast/slow gap is the effect size to pick N from.
- **Difference on both** → the mechanism is contributing and the fast-arm number cannot be
  read as staleness. Do not report it as such.
- **Difference on neither** → staleness is not a real cost on this workload; O-16 closes as
  "report only" permanently rather than provisionally.

This is the negative population D-104's first sufficiency experiment lacked, built in from the
start rather than added after a surprising result.

## The t=0 baseline (recorded 2026-09-26, 15 h after the seed)

Arms `staleness-abstract-stale-top20-d2-adaptive-age0d` and
`staleness-abstract-top20-d2-adaptive-age0d`. **These are the numbers to subtract in
November.** They are not a result about staleness — at t=0 there is no staleness to measure.

**The stale arm is genuinely sealed.** `tests/eval/stale_purity.py`:

| | |
|---|---|
| subtopics answered locally | **30 / 30** — none fell through to live arXiv |
| papers reaching a prompt | 200, **0 off-snapshot** |
| papers cited | 88, **0 off-snapshot** |

This matters because `research_worker` falls back to arXiv for any subtopic the corpus does
not cover (D-105), and in this arm that fallback would be contamination: current papers
entering the arm whose defining property is that it cannot see them. Nothing else would have
noticed — the run succeeds and the review is fine.

**Citation recency — 0 post-seed on both sides**, which is the sanity check. Nothing had been
published since the seed, so a gap here would have meant the metric was measuring something
other than age:

| arm | cited | in snapshot | post-seed |
|---|---|---|---|
| STALE | 88 | 88 (100%) | **0** |
| FRESH | 78 | 35 (45%) | **0** |

**The interesting row is FRESH's middle column.** 43 of its 78 citations — **55%** — are
papers the snapshot does not hold, and *none of them are new*. They are older papers that
arXiv's live ranking surfaced and the snapshot's BM25 did not. Without this baseline, seeing
off-snapshot papers in November would be easy to read as staleness when more than half of it
is just the two retrieval methods disagreeing.

**Quality, paired over 10 questions** (`tests/eval/compare.py`, STALE minus FRESH):

| metric | STALE | FRESH | delta | SE |
|---|---|---|---|---|
| specificity | 0.80 | 0.77 | −0.03 | −1.3 |
| faithfulness | 0.98 | 0.97 | −0.01 | −0.6 |
| **relevancy** | 0.92 | 0.99 | **+0.08** | **+2.2** |
| citation_density | 2.12 | 1.93 | −0.19 | −1.3 |
| papers_retrieved | 47.9 | 28.1 | −19.8 | −6.6 |
| papers_in_prompt | 20.0 | 20.0 | 0.00 | — |

At zero age the two mechanisms are **indistinguishable on quality** — every metric under 2 SE
except one. That one matters: **relevancy favours the fresh arm by +0.08 at +2.2 SE**, so the
mechanism is not perfectly neutral, and a November relevancy gap of that size would mean
*nothing* about staleness. Subtracting it is the entire reason this pair exists.

`papers_retrieved` differing by −6.6 SE while `papers_in_prompt` is identical is expected, not
a finding: the corpus returns a full `LOCAL_SEARCH_TOP_K` per subtopic and arXiv returns fewer
after dedup, and ranking truncates both to `SYNTHESIS_TOP_N = 20` before the prompt (D-091).

## Running it

Record both arms. Each is a paid pass over 10 questions (~35 s per question).

```powershell
# STALE -- reads the dated snapshot
$env:EVAL_QUESTION_SET = "staleness"
$env:EVAL_LOCAL_FIRST  = "1"
$env:EVAL_STALE_CORPUS = "1"
uv run pytest -m record

# FRESH -- live arXiv, corpus off
$env:EVAL_STALE_CORPUS = "0"
$env:EVAL_LOCAL_FIRST  = "0"
uv run pytest -m record
```

These land as `staleness-abstract-stale-top20-d2-adaptive-age{N}d` and
`staleness-abstract-top20-d2-adaptive-age{N}d`, where N is days since the seed — so they
cannot collide with the `age0d` baseline. The arm name is *derived from the settings that
produced the run* (`arm_name`), so a run cannot be filed under a configuration it did not come
from, which is why `EVAL_STALE_CORPUS` reaches `retrieval_unit` and the age reaches
`days_since_seed` rather than only the corpus lookup.

Check the stale arm stayed stale **before** reading any result — a fall-through to live arXiv
invalidates the whole comparison:

```powershell
uv run python -m tests.eval.stale_purity staleness-abstract-stale-top20-d2-adaptive-age<N>d
```

Then the primary metric, then the judged one. Filter the eval to the new arms: unfiltered it
re-scores all 160 committed recordings, which is paid and pointless (results merge, so
filtering is safe).

```powershell
uv run python -m tests.eval.recency staleness-abstract-stale-top20-d2-adaptive-age<N>d staleness-abstract-top20-d2-adaptive-age<N>d
uv run pytest -m eval -k "age<N>d"
uv run python -m tests.eval.compare staleness-abstract-stale-top20-d2-adaptive-age<N>d staleness-abstract-top20-d2-adaptive-age<N>d
```

Finally, subtract the baseline. The difference-in-differences is the answer; the raw November
delta is not.

## What to measure

**Primary — citation recency. Free, deterministic, no judge** (`tests/eval/recency.py`).

> *any paper FRESH cites that was submitted after the seed could not have been in the
> snapshot.*

Count those. That number **is** what staleness costs, in the units that matter, and it needs
no model to believe. It read **0 for both arms at t=0**, as it must. Report it per arm (fast
vs slow) — on the slow arm it should stay near zero, and if it does not, the velocity
measurement was wrong.

Dates come from the arXiv id's YYMM, which is the only source available for *both* arms: a
paper the fresh arm found today exists in no local record, so anything reading stored metadata
could measure only the stale arm. **Resolution is one month**, so papers submitted in the seed
month but after the seed date are counted as not-post-seed — which undercounts the fresh arm's
advantage. That is the conservative direction, and from October onwards the ambiguity is gone.

**Secondary — quality, through the existing harness.** Faithfulness and specificity from
`tests/eval/metrics.py`, paired per question, reported in SE like every other comparison in
this repo (D-094). Treat this as secondary on purpose: it is judged, it is noisy at n=10, and
D-094/D-095 both showed this harness returning honest nulls on real differences in retrieval.
A null here alongside a large recency difference is itself a finding — *the review does not
get worse, it just gets older* — and that is a legitimate answer to O-16, not a failed
experiment.

## What would change the decision

O-16's current position is **report only**: the coverage panel says which subtopics were
answered locally and warns that newer papers are not represented, and nothing forces a
refresh. This experiment either keeps that or replaces it with **option B — a max age per
paper**, where coverage ignores chunks indexed more than N days ago.

N comes from the data or not at all. If the fast arm shows degradation appearing somewhere
between the seed date and the run date, N is bounded by that interval; if it does not, there
is no N to pick and inventing one would be D-077's mistake repeated.

## Threats to validity, stated in advance

- **n = 10, split 6/4.** Underpowered for a small effect, like every comparison in this repo.
  The recency metric is exact and does not care; the judged metrics do.
- **The corpus is dense** (1554 papers for 10 questions), and held-out subtopics returned a
  full top-20 almost every time. The STALE arm will comfortably answer locally, which is the
  condition the experiment wants — but it also means this is a *generous* corpus, so any
  degradation found is a lower bound on what a sparser one would show.
- **The planner is not frozen.** Each arm decomposes independently and will produce different
  subtopics, adding variance. Pairing is per question, not per subtopic, which is the same
  choice every other comparison here makes.
- **Topic velocity was measured once**, in the 30 days to the seed. If a "slow" topic has a
  burst during the wait, the control weakens — so re-run `tests.eval.velocity` on the run day
  and report both numbers rather than assuming the split still holds.
- **The date is asserted, not observed**, as described under *The artifact*.
- **The mechanism gap is not exactly zero.** Relevancy favoured the fresh arm by +0.08 at
  +2.2 SE at t=0. The DiD subtracts it, but subtracting an estimate measured at n=10 adds its
  own uncertainty — so a November effect of similar size should be treated as unresolved, not
  as a finding.
- **The baseline's `days_since_seed` was backfilled.** Both t=0 arms were recorded minutes
  before `arm_name` learned about corpus age, so the field was added afterwards, computed from
  each recording's own `recorded_at` against the manifest's `seeded_at` — derived from data
  already in the artifact, not asserted, and verified by
  `test_recording_is_filed_under_the_arm_its_settings_describe`. The runs themselves were not
  re-recorded and their reviews are untouched.

## Keeping this file current

| Change | What goes stale here |
|---|---|
| The seed is re-run without `--rebuild` | everything — the artifact table, the dates, and the experiment |
| `questions-staleness.json` gains a question | the arm table, and `n = 10` in the threats section |
| `MIN_LOCAL_PAPERS`, `LOCAL_SEARCH_TOP_K` or the retrieval function change | the "corpus is dense" threat, and `tests.eval.seed_check` must be re-run — the STALE arm may stop answering locally |
| `arm_name` or the `retrieval_unit` values change | the two arm names in *Running it* **and** the `-age{N}d` suffix that keeps November off the baseline |
| The t=0 arms are re-recorded | the whole baseline table — and the old ones must not be overwritten, since they are the subtrahend |
| The experiment is actually run | all of it — this becomes a decision entry, and O-16 closes |
