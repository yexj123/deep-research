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

## The design

Two arms, run **on the same day, months from now**, differing in exactly one thing: whether
retrieval reads the dated snapshot or searches arXiv live.

| Arm | Retrieval | Can cite papers from |
|---|---|---|
| **STALE** | the 2026-09-25 snapshot, `LOCAL_FIRST=1` | on or before 2026-09-25 |
| **FRESH** | live arXiv, `LOCAL_FIRST=0` | any date, including everything published during the wait |

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

These land as the arms `staleness-abstract-stale-top20-d2-adaptive` and
`staleness-abstract-top20-d2-adaptive`. The arm name is *derived from the settings that
produced the run* (`arm_name`), so a run cannot be filed under a configuration it did not come
from — which is why `EVAL_STALE_CORPUS` reaches `retrieval_unit` rather than only the corpus
lookup.

Then score and compare:

```powershell
uv run pytest -m eval
uv run python -m tests.eval.compare staleness-abstract-stale-top20-d2-adaptive staleness-abstract-top20-d2-adaptive
```

## What to measure

**Primary — citation recency. Free, deterministic, no judge.** Every recording stores the
papers that reached the prompt. For each arm, take the publication dates of the cited papers
and compare the distributions. The manifest's per-question `newest_published` makes one
statement exact rather than inferential:

> *any paper FRESH cites that was published after 2026-09-25 could not have been in the
> snapshot.*

Count those. That number **is** what staleness costs, in the units that matter, and it needs
no model to believe. Report it per arm (fast vs slow) — on the slow arm it should be near
zero, and if it is not, the velocity measurement was wrong.

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

## Keeping this file current

| Change | What goes stale here |
|---|---|
| The seed is re-run without `--rebuild` | everything — the artifact table, the dates, and the experiment |
| `questions-staleness.json` gains a question | the arm table, and `n = 10` in the threats section |
| `MIN_LOCAL_PAPERS`, `LOCAL_SEARCH_TOP_K` or the retrieval function change | the "corpus is dense" threat, and `tests.eval.seed_check` must be re-run — the STALE arm may stop answering locally |
| `arm_name` or the `retrieval_unit` values change | the two arm names in *Running it* |
| The experiment is actually run | all of it — this becomes a decision entry, and O-16 closes |
