# deep-research

A self-hosted deep research agent. Give it a research question: it plans subtopics, searches
arXiv for each one in parallel, recursively explores further while it keeps finding new
papers, and writes a review that **cites only papers it actually retrieved**.

Same idea as Open WebUI, but the chat is a LangGraph agent doing literature review. Python,
LangGraph, FastAPI, SQLite. No build step, no `node_modules`.

> **Status:** the agent and web layer both work end to end. Built one milestone at a time —
> see [`docs/progress.md`](docs/progress.md) for exactly what is done.

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

**A run takes roughly 30–60 seconds and costs real money.** Most of the wall-clock time is
arXiv's rate limit (one request every three seconds, which this respects), not the model.

### What actually happens

```
intake → decompose → (one worker per subtopic, in parallel) → gap_check
             ↑                                                    │
             └──────── depth left AND new papers found ───────────┘
                                                                  │
                                       synthesize → check_citations → done
```

| Step | What it does |
|---|---|
| `intake` | Validates the question and the run's provider |
| `decompose` | The model proposes subtopics; **code** filters out ones already explored, ones that failed twice, and ones arXiv can't search |
| `research_worker` | One arXiv search per subtopic, run in parallel but queued for arXiv's rate limit |
| `gap_check` | Another round only if depth remains **and** the last round found papers it hadn't seen |
| `synthesize` | Writes the review from the retrieved abstracts only |
| `check_citations` | Verifies every `[arXiv:<id>]` marker against the papers actually retrieved |

## Configuration

Everything tunable lives in [`src/deep_research/agent/config.py`](src/deep_research/agent/config.py):

| Setting | Default | What it controls |
|---|---|---|
| `MODEL_NAMES` | `gpt-4o`, `deepseek-flash` | Which model each provider uses |
| `MAX_SUBTOPICS` | `3` | Subtopics proposed per round |
| `MAX_DEPTH` | `2` | 0-indexed, so **3 search passes** at most |
| `ARXIV_MAX_RESULTS` | `10` | Papers per subtopic search |
| `ARXIV_MIN_INTERVAL_SECONDS` | `3.0` | arXiv's stated rate limit — don't lower this |
| `RECURSION_LIMIT` | `15` | Safety net only; the measured minimum is 13 |

Raising `MAX_SUBTOPICS` or `MAX_DEPTH` costs wall-clock time and money roughly linearly.

Runs are stored in `deep_research.sqlite` in the working directory — checkpoints and history
in one file. Delete it to start fresh; it's gitignored.

## Tests

```sh
uv run pytest
```

Over 200 tests, using fake chat models and saved arXiv responses. **No API keys, no network,
no cost.** Every test's docstring names the behaviour it pins down and the decision (`D-…`) it
checks.

Tests marked `integration` call real, paid APIs and are excluded by default:

```sh
uv run pytest -m integration
```

They're skipped when the provider's key isn't set.

## Honest limitations

This project treats "the system reporting success for less work than you assume" as its
primary failure mode, so these are stated rather than buried:

- **Ungrounded citations are caught, not prevented.** A model can invent a plausible arXiv ID;
  `check_citations` detects it and reports it, but cannot stop it happening. Observed in
  practice — see `D-079`.
- **Every run reports what it didn't cover.** Failed searches, subtopics where nothing was
  published, malformed entries, and *why* the run stopped — because "the depth limit cut this
  off" and "the search converged" mean very different things.
- **arXiv only.** Semantic Scholar is designed for but not built.
- **English-first.** Non-ASCII terms work, but stopword removal is English-only.
- **Single-user, no auth.** Intended for localhost. If you expose it, put a password in front.
- **Abstracts, not full text.** The review is written from titles and abstracts, which are
  treated as untrusted input.

## Documentation

| File | What's in it |
|---|---|
| [`docs/decisions.md`](docs/decisions.md) | Every design choice, why, and what was rejected |
| [`docs/walkthrough.md`](docs/walkthrough.md) | One run end to end, with real captured values |
| [`docs/code-map.md`](docs/code-map.md) | What each file does and what imports what |
| [`docs/progress.md`](docs/progress.md) | What's done, what's next |

If you're wondering "why is it built this way?", `decisions.md` is the answer — including the
measurements behind each choice and the ones that turned out wrong.

## License

MIT. You bring your own API keys; arXiv metadata is used under their terms, and no paper PDFs
are cached or redistributed.
