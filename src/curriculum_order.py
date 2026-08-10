"""Where a topic sits in the course, and which topic a question belongs to.

A question was being independently retrieved and approved for several topics: **16 of 149 distinct questions
(11%) sat in more than one topic**, five of them in three. Not shared content — **no session belongs to two
topics** (0 of 56) — just that every de-duplication mechanism is scoped to one topic (`_add_retained`,
`_drop_rejected` and the outcome balance are keyed by `topic_key`).

THE HOME IS THE EARLIEST TOPIC THAT COVERS THE QUESTION
-------------------------------------------------------
Not the best-fitting topic, and not the latest. The reviewer's rule: if the definition is taught in an
earlier topic and this topic elaborates on it, a question covering BOTH belongs here and must not be asked
earlier — a student cannot answer it yet. Conversely a bare definition belongs where it is first taught, not
in every later topic that happens to mention it.

Both rejected alternatives were measured on the real 16:

    "latest topic wins"  ->  sends "What is a prompt?" to No-Code AI Automation. Wrong: it pushes every
                             fundamental to the last topic that mentions it.
    "best fit wins"      ->  sends "How do you approach designing an effective prompt?" to AI Workflows
                             (0.742) over Prompt Engineering Fundamentals (0.729). Wrong: prompts are
                             TAUGHT at position 3, and a student there can already answer it.

`home_topic` takes the earliest topic in curriculum order whose fit is within `CROSS_TOPIC_COVER_RATIO` of
the best — "the first topic that covers it about as well as any". It is RELATIVE on purpose: this codebase
has repeatedly found that absolute similarity bars cannot separate overlapping populations
(`DEDUP_SEMANTIC_THRESHOLD`, `_outcome_coverage`'s proximity threshold, the dedup band), and the question
here is only which of a handful of topics covers one question best.

Measured on all 16, it moves questions in BOTH directions, which is the test of the rule:

    What is a prompt?                        -> pos 3  Prompt Eng   (0.665)  over No-Code (0.540)
    How do you approach designing a prompt?  -> pos 3  Prompt Eng   (0.729)  over AI Workflows (0.742) [!]
    How have you adopted AI in workflows?    -> pos 5  No-Code      (0.701)  over Prompt Eng (0.583)
    What is the HTTP Request node…?          -> pos 8  AI Workflows (0.703)  over No-Code (0.540)

THE ORDER COMES FROM THE GRAPH, NOT FROM DICT ORDER
---------------------------------------------------
`knowledge_graph.json` carries `session_order_edges` — a single-root DAG whose topological order covers
**52 of 52** sessions and ends at Fine-Tuning. `course_structure.json`'s key order happens to agree today,
which is the corroboration that the graph order is real, but that key order is an accident of insertion
that `AddCourse` can change, so it is not the source of truth.

A topic's position is the **max** over its sessions — the point by which a student has finished the topic —
and sessions absent from the graph are IGNORED rather than given a sentinel. A sentinel poisons the whole
topic through the max: the first version scored four topics at 999 because one of their sessions was
missing, which silently made every comparison against them wrong.
"""
from __future__ import annotations

import functools
import json


@functools.lru_cache(maxsize=1)
def session_positions() -> dict[str, int]:
    """Session name -> position in the course, from `session_order_edges`. {} when unavailable.

    Kahn's algorithm rather than a DFS so a cycle degrades to a partial order instead of recursing
    forever; a malformed graph must not take the pipeline down.
    """
    from src.config import DATA_DIR

    try:
        kg = json.loads((DATA_DIR / "knowledge_graph.json").read_text())
    except Exception:  # noqa: BLE001 — a missing graph must not break generation
        return {}
    edges = kg.get("session_order_edges") or []
    nxt: dict[str, list[str]] = {}
    indeg: dict[str, int] = {}
    nodes: set[str] = set()
    for e in edges:
        a, b = (e.get("from"), e.get("to")) if isinstance(e, dict) else (e[0], e[1])
        if not a or not b:
            continue
        nxt.setdefault(a, []).append(b)
        indeg[b] = indeg.get(b, 0) + 1
        nodes |= {a, b}
    order: list[str] = []
    seen: set[str] = set()
    queue = [n for n in nodes if not indeg.get(n)]
    while queue:
        n = queue.pop(0)
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        for m in nxt.get(n, []):
            indeg[m] = indeg.get(m, 0) - 1
            if indeg[m] <= 0:
                queue.append(m)
    # Any node left out by a cycle still gets a position, after everything ordered.
    for n in sorted(nodes - seen):
        order.append(n)
    return {s: i for i, s in enumerate(order)}


@functools.lru_cache(maxsize=1)
def _course_structure() -> dict:
    """topic -> session list. Read here rather than via `data_loader`, which only exposes the REVERSE
    index (`get_topic_for_session`) and has no forward accessor."""
    from src.config import DATA_DIR

    try:
        return json.loads((DATA_DIR / "course_structure.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


@functools.lru_cache(maxsize=256)
def topic_position(topic: str) -> int | None:
    """How far into the course a topic ENDS, or None when none of its sessions are in the graph.

    `max`, not `min`: the question is "by when has a student covered this topic", and sessions missing
    from the graph are skipped rather than defaulted — see the module docstring.

    None for a custom (user-added) topic, which lives in `memory` and not in `course_structure.json`.
    Callers sort those LAST rather than treating them as position 0.
    """
    sessions = _course_structure().get(topic) or []
    pos = session_positions()
    known = [pos[s["name"] if isinstance(s, dict) else s] for s in sessions
             if (s["name"] if isinstance(s, dict) else s) in pos]
    return max(known) if known else None


def home_topic(fits: dict[str, float], ratio: float | None = None) -> str | None:
    """The earliest topic in curriculum order that covers the question about as well as any.

    `fits` maps topic -> grounding score against that topic's reading material (build it with
    `filter_topic_sets._session_profile` + `_fits`, or `pipeline._session_profile`).

    Ties and unknown positions: a topic with no position sorts LAST, so a question is never homed to a
    topic whose place in the course we cannot establish while a known one is available.
    """
    from src.config import CROSS_TOPIC_COVER_RATIO

    if not fits:
        return None
    ratio = CROSS_TOPIC_COVER_RATIO if ratio is None else ratio
    best = max(fits.values())
    if best <= 0:
        # No grounding signal anywhere — fall back to curriculum order alone rather than pick arbitrarily.
        return sorted(fits, key=lambda t: (topic_position(t) is None, topic_position(t) or 0, t))[0]
    bar = ratio * best
    ordered = sorted(fits, key=lambda t: (topic_position(t) is None, topic_position(t) or 0, t))
    for topic in ordered:
        if fits[topic] >= bar:
            return topic
    return ordered[0]                      # unreachable: the best-fitting topic always clears its own bar
