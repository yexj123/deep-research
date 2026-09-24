# Full-text fixtures

**Extracted text, not PDFs — and that is a licence decision (D-008, D-108).**

arXiv permits building indexes over full text and prohibits storing and serving e-prints. A
PDF checked into a public MIT repository is redistribution, whatever it is labelled. Text
extracted for indexing is the thing the terms permit, so that is what is committed here.

| File | Source | Why this one |
|---|---|---|
| `ieee-style.txt` | arXiv:2411.18583v1, first 12,000 chars | IEEE headings — Roman numerals, `I. INTRODUCTION` |
| `latex-style.txt` | arXiv:2510.22344v1, first 12,000 chars | LaTeX headings — Arabic, `1 Introduction` |

**Two conventions on purpose.** Measured 2026-09-24: each paper matches exactly one style and
scores **zero** on the other (Roman found 6/0/0 across three papers, Arabic 0/12/6). That is
what lets `split_sections` pick "whichever style matches more" without knowing the venue — and
a chunker handling only one convention would index the other as a single undifferentiated
blob, losing every section label with nothing failing.

**Trimmed to 12,000 characters** because the full papers run 26k–124k, and the first 12k
already covers frontmatter plus three or four sections — enough to exercise splitting,
labelling and the size bound.

**Regenerate** by downloading those two PDFs, extracting with
`deep_research.agent.sources.fulltext.extract_text`, and truncating. Respect the 3-second
arXiv interval (D-064) if fetching both.
