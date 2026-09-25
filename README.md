# deep-research

A self-hosted deep research agent. Give it a research question: it plans subtopics, searches
arXiv for each one in parallel, recursively explores further while it keeps finding new
papers, and writes a review that **cites only papers it actually retrieved**.

Same idea as Open WebUI, but the chat is a LangGraph agent doing literature review. Python,
LangGraph, FastAPI, SQLite. No build step, no `node_modules`.

> **Status:** the agent and web layer both work end to end. Built one milestone at a time —
> see [`docs/progress.md`](docs/progress.md) for exactly what is done.

### It was measured, and three of its four ideas did not survive

Recursive decomposition, deeper search and full-text retrieval were each built, evaluated
against a frozen question set, and found **not** to improve review quality. The local corpus
did pay — in cost, not quality: **65 arXiv requests became 1, wall clock fell 65%**, with every
quality metric unchanged.

The agent you run today is therefore **cheaper than its first design** — one search round
instead of three — with no measured quality cost.

**[`docs/findings.md`](docs/findings.md) is the evidence**, including the metric validated
before it was trusted (0.950 concrete vs 0.222 vague), what each null cost to obtain, and what
would overturn it. Every table regenerates from the committed recordings with no API key.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An API key for OpenAI and/or DeepSeek. **None ship with this repo** — you bring your own.

## Setup

```sh
uv sync
```

Then set whichever provider you plan to use:

```sh
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
# bash / zsh
export OPENAI_API_KEY="sk-..."
```

`DEEPSEEK_API_KEY` if you want the DeepSeek option. Never commit a `.env`.

## Running it

```sh
uv run uvicorn deep_research.api.main:create_app --factory --reload
```

Open <http://127.0.0.1:8000>.

Type a question, pick a model, press **Research**. You'll see the node-by-node progress trail
and live status from each search, then the review streaming in token by token.

Past runs are listed in the sidebar; click one to reopen it. A run that never started or was
interrupted offers to start or resume — resuming continues from its last checkpoint rather
than starting over.

**A first run takes roughly 20 seconds and costs real money.** Most of the wall-clock time
is arXiv's rate limit (one request every three seconds, which this respects), not the model.

**Ask about a topic you have researched before and it takes ~7 seconds and makes no arXiv
request at all** — measured on a live run, 6.9s against 20.0s. Every search indexes its
results into a local corpus, and a later subtopic the corpus already covers is answered
from it. Quality is unchanged: see [`docs/findings.md`](docs/findings.md).

### What actually happens

```
intake → decompose → (one worker per subtopic, in parallel) → gap_check
             ↑                                                    │
             └───── another round could change the review ─────────┘
                                                                  │
                      synthesize → check_claims → check_citations → done

and inside each worker:

  subtopic → does the local corpus cover it?
               ├─ yes → answer from it, no network at all
               └─ no  → search arXiv, and index every result on the way through
```

| Step | What it does |
|---|---|
| `intake` | Validates the question and the run's provider |
| `decompose` | The model proposes subtopics; **code** filters out ones already explored, ones that failed twice, and ones arXiv can't search |
| `research_worker` | Checks the local corpus first; searches arXiv only if it isn't covered, queued for arXiv's rate limit. Either way, results are indexed for next time |
| `gap_check` | Another round only if one could change the review — not if the prompt is already full, nothing new was found, or the round came back empty. **19 of 20 runs stop after one round** |
| `synthesize` | Ranks the run's papers with BM25, keeps the top 20, writes the review from those abstracts |
| `check_claims` | Asks, in one call, whether each cited sentence is supported by the paper it cites — reading exactly what the writer read |
| `check_citations` | Verifies every `[arXiv:<id>]` marker against the papers actually retrieved |

## Configuration

Everything tunable lives in [`src/deep_research/agent/config.py`](src/deep_research/agent/config.py):

| Setting | Default | What it controls |
|---|---|---|
| `MODEL_NAMES` | `gpt-4o`, `deepseek-flash` | Which model each provider uses |
| `MAX_SUBTOPICS` | `3` | Subtopics proposed per round |
| `MAX_DEPTH` | `2` | A **ceiling, not a target** — the adaptive exits normally stop after one round |
| `SYNTHESIS_TOP_N` | `20` | Papers reaching the prompt, BM25-ranked. Measured: cutting to 20 saved 76% of prompt tokens with no quality change |
| `LOCAL_FIRST` | `True` | Answer a covered subtopic from the corpus instead of searching. Safe on a cold corpus: an empty one covers nothing |
| `MIN_LOCAL_PAPERS` | `3` | How many strongly-matching local papers count as coverage. Measured, not guessed |
| `EXCERPT_MAX_CHARS` | `0` | Full-text passages per paper. **0 = off**; 4000 reproduces the arm that measured it as not worth 2.8× the prompt |
| `ARXIV_MAX_RESULTS` | `10` | Papers per subtopic search |
| `ARXIV_MIN_INTERVAL_SECONDS` | `3.0` | arXiv's stated rate limit — don't lower this |
| `RECURSION_LIMIT` | `15` | Safety net only; the measured minimum is 14 |

Raising `MAX_SUBTOPICS` or `MAX_DEPTH` costs wall-clock time and money roughly linearly.
Most of these defaults come from a recorded experiment rather than intuition — each one's
`D-` number in [`docs/decisions.md`](docs/decisions.md) has the numbers.

Everything is stored in `deep_research.sqlite` in the working directory: checkpoints, run
history, **and the paper corpus** — one file (no second database until there is a reason).
Delete it to start fresh; it's gitignored. The corpus rebuilds itself from use; the history
does not.

**No PDFs are stored anywhere.** A PDF is fetched to memory, parsed and discarded; what
persists is extracted text. arXiv permits building indexes over content and prohibits
storing and serving e-prints, and that is the difference.

`SELECT title FROM papers` is a reading list of everything the agent has retrieved for you.

## Tests

```sh
uv run pytest
```

Over 850 tests, using fake chat models and saved arXiv responses. **No API keys, no network,
no cost**, and the suite is deliberately kept that way — a developer `.env` once turned
LangSmith tracing on and the suite silently acquired a network dependency that cost 11 seconds
per flush and sent run content off the machine (`D-102`).

Every test's docstring names the behaviour it pins down and the decision (`D-…`) it checks.
Several exist because a bug got through: a test that passes without exercising anything is
worse than no test, and this repo has shipped one (`D-115`).

Tests marked `integration` call real, paid APIs and are excluded by default:

```sh
uv run pytest -m integration
```

They're skipped when the provider's key isn't set.

## Honest limitations

This project treats "the system reporting success for less work than you assume" as its
primary failure mode, so these are stated rather than buried:

- **Ungrounded citations are caught, not prevented.** A model can invent a plausible arXiv ID;
  `check_citations` detects and reports it, but cannot stop it happening. **Measured across
  140 runs: 2 in 1103 citations, 0.18 per 100** (`D-112`) — rare enough that it is not the
  risk worth designing against.
- **Claim support is judged by a model, and labelled as such.** The real risk is a valid ID on
  a retrieved paper that never says what the review claims. `check_claims` flags those, but it
  is a model judging a model — the coverage panel says so, and tells you to treat it as a
  prompt to check rather than a verdict (`D-113`).
- **A repeat question may be answered from cache.** Subtopics the local corpus already covers
  skip arXiv entirely. The panel names them and warns that papers published since are not
  represented — but nothing yet forces a refresh, so a topic researched often can stay
  answered from old papers (`O-16`).
- **Every run reports what it didn't cover.** Failed searches, subtopics where nothing was
  published, malformed entries, and *why* the run stopped — because "the depth limit cut this
  off" and "the search converged" mean very different things.
- **arXiv only.** Semantic Scholar is designed for but not built.
- **English-first.** Non-ASCII terms work, but stopword removal is English-only.
- **Single-user, no auth.** Intended for localhost. If you expose it, put a password in front.
- **Abstracts, not full text — and that is a measured choice, not a missing feature.** Full
  text is built and switchable (`EXCERPT_MAX_CHARS`). Turned on it costs 2.8x the prompt and
  improves nothing past 2 SE, while faithfulness drifts *down*. See `D-111`.
- **One search round, not three.** Recursion exists and is bounded, but the adaptive exits
  stop 19 of 20 runs after a single round, because rounds two and three measurably added
  nothing (`D-094`, `D-096`).
- **Retrieval and synthesis always read the same text.** Selecting papers on full text while
  showing the model only abstracts measured *worse* than having no full text at all
  (faithfulness −3.1 SE, `D-110`). The configuration is unreachable rather than discouraged.

## Documentation

| File | What's in it |
|---|---|
| [`docs/findings.md`](docs/findings.md) | **What the measurements say** — the four premises, the evidence, and the three that failed |
| [`docs/decisions.md`](docs/decisions.md) | Every design choice, why, and what was rejected |
| [`docs/walkthrough.md`](docs/walkthrough.md) | One run end to end, with real captured values |
| [`docs/code-map.md`](docs/code-map.md) | What each file does and what imports what |
| [`docs/progress.md`](docs/progress.md) | What's done, what's next |

If you're wondering "why is it built this way?", `decisions.md` is the answer — including the
measurements behind each choice and the ones that turned out wrong.

## License

MIT. You bring your own API keys; arXiv metadata is used under their terms, and no paper PDFs
are cached or redistributed.
