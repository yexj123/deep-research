# deep-research

A self-hosted deep research agent. Give it a research question: it searches
the literature, recursively explores subtopics where it finds gaps, and
writes a review with citations. Built with LangGraph and FastAPI.

> **Status:** early development. The agent is being built one milestone at a time.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Your own API key for OpenAI and/or DeepSeek. No keys come with the repo.

## Setup

```sh
uv sync
```

Set `OPENAI_API_KEY` and/or `DEEPSEEK_API_KEY` in your environment. Never
commit a `.env` file.

## Tests

```sh
uv run pytest
```

The default tests use fake chat models and need no API keys. Tests marked
`integration` call real APIs and are skipped when the provider's key isn't set.

## Design decisions

Every design choice, and the alternatives that were rejected, is recorded in
[`docs/decisions.md`](docs/decisions.md).

## License

MIT
