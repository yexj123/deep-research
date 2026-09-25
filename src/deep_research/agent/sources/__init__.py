"""Where papers come from. **This is a package, not an interface** (D-124).

There is no `Source` Protocol here and no base class, because there is one source and the
design is arXiv-specific on purpose:

- `models.Source` is "one arXiv paper" — its own docstring says so.
- `arxiv_id` is the **primary key** of `papers` in `persistence/corpus.py` and the foreign key
  of `chunks`; it is also `seen_paper_ids`, `synthesized_from`, and the thing
  `check_citations` validates against.
- `[arXiv:<id>]` is hard-coded in the synthesis prompt and in both regexes that verify it —
  the citation contract the whole grounding design rests on (D-046).

So adding a non-arXiv source is **not** "write a second client against the interface". It is a
primary-key migration plus a change to that contract. D-124 settled that this is not happening
until an experiment shows non-arXiv literature makes a review measurably better; Semantic
Scholar is separately closed to third-party apps, and 69% of OpenAlex results carry no arXiv
ID at all.

This docstring exists because the emptiness of this file was itself the bug: for weeks the
README and the task list both claimed a second source was "designed for but not built", and
nothing contradicted them.
"""
