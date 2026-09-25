"""Shape probe for the OpenAlex works endpoint (O-17).

Throwaway-ish: it prints one real response so the client can be written against observed JSON
rather than documentation. Same discipline as tests/agent/fixtures/arxiv/ -- capture the real
thing once, build against that.

    uv run python -m tests.eval.openalex_probe
"""

import json

import httpx

# OpenAlex asks for a contact address to be put in the "polite pool", which is faster and not
# rate-limited in practice. It is not a key and not a secret (D-002 is untouched by it).
MAILTO = "deep-research@example.invalid"
URL = "https://api.openalex.org/works"


def main() -> None:
    response = httpx.get(
        URL,
        params={
            "search": "speculative decoding language model inference",
            "per-page": 5,
            "mailto": MAILTO,
            "select": "id,ids,doi,title,publication_date,authorships,"
            "abstract_inverted_index,best_oa_location,primary_location,type",
        },
        timeout=30,
        follow_redirects=True,
    )
    print(f"status {response.status_code}")
    print("rate headers:", {k: v for k, v in response.headers.items() if "rate" in k.lower()})
    payload = response.json()
    print(f"total results: {payload['meta']['count']}")

    for work in payload["results"]:
        inverted = work.get("abstract_inverted_index")
        print("\n" + "-" * 70)
        print("title      :", work["title"])
        print("type       :", work.get("type"))
        print("date       :", work.get("publication_date"))
        print("doi        :", work.get("doi"))
        print("ids        :", json.dumps(work.get("ids", {})))
        print("authors    :", [a["author"]["display_name"] for a in work.get("authorships", [])][:4])
        print("abstract?  :", "yes" if inverted else "NO")
        if inverted:
            # Reconstruct: the index maps word -> [positions]. This is what a client must do,
            # and it is the main shape difference from arXiv's plain-text <summary>.
            positions = {p: word for word, ps in inverted.items() for p in ps}
            text = " ".join(positions[i] for i in sorted(positions))
            print("abstract   :", text[:220] + "...")
        loc = work.get("primary_location") or {}
        print("venue      :", (loc.get("source") or {}).get("display_name"))
        print("landing    :", loc.get("landing_page_url"))
        print("oa pdf     :", (work.get("best_oa_location") or {}).get("pdf_url"))


if __name__ == "__main__":
    main()
