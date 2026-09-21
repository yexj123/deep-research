"""gap_check node: counts the round that just finished (D-075, D-076).

It deliberately decides nothing. `route_after_gap_check` in `graph.py` reads the incremented
depth and does the routing, because which node runs next is ordering knowledge and only
`graph.py` holds that (code-map rule 2).

Note what this node is *not*: it does not look for gaps. `decompose` is the gap finder --
it receives `explored_subtopics` and is asked for something genuinely new (D-070). This
node only answers "is another round worth it?", which is why it calls no model.
"""

from typing import Any

from deep_research.agent.state import ResearchState


def gap_check(state: ResearchState) -> dict[str, Any]:
    """Increment `depth`. It counts rounds *completed*, so it goes up here (D-076).

    Incrementing before the routing decision means the final checkpoint records what
    actually happened -- three rounds completed reads as `depth == 3` -- rather than what
    was about to happen.
    """
    return {"depth": state.depth + 1}
