# arXiv API fixtures

Saved responses for tests, served through `httpx.MockTransport` (D-049), so tests never hit the
network. arXiv metadata is CC0 (arXiv API terms of use), so storing it here is allowed. No PDFs.

| File | Origin | What it exercises |
|---|---|---|
| `search_ok.xml` | **Real**, captured 2026-09-16: `search_query=all:"retrieval augmented generation"`, `max_results=3`, HTTP 200 | normal parsing: 3 entries (2411.18583v1, 2502.00306v2, 2510.22344v1) |
| `search_empty.xml` | **Real**, captured 2026-09-16: `search_query=all:qzxwvkjhgfdsa`, HTTP 200 | zero results (`opensearch:totalResults` = 0) |
| `error_feed.xml` | **Real**, captured 2026-09-16: `id_list=not-a-valid-id`, **HTTP 400** | arXiv's error feed: one entry titled `Error` |
| `search_one_invalid_id.xml` | Derived from `search_ok.xml`: the second entry's `<id>` replaced with `http://arxiv.org/abs/not-a-real-id` | skip-and-count an invalid entry (D-045): 2 sources, 1 skipped |
| `truncated.xml` | Derived from `search_ok.xml`: cut in the middle of the second entry | `xml.etree.ElementTree.ParseError` |
| `with_doctype.xml` | Derived from `search_empty.xml`: a plain `<!DOCTYPE feed>` added | `DTDForbidden`, which proves `forbid_dtd=True` is set (defusedxml accepts a plain DOCTYPE by default) (D-047) |
| `with_entity.xml` | Derived from `search_empty.xml`: an `<!ENTITY>` declaration used in `<title>` | `DTDForbidden` with `forbid_dtd=True`, since the DOCTYPE holding the entity is rejected first. Without `forbid_dtd` it would be `EntitiesForbidden` (D-047) |
