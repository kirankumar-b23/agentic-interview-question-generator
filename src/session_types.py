"""Resolving a session's type (theory_heavy / code_heavy / mixed) from a session NAME.

There are four historical sources of `session_type` in this repo and they disagree:

  1. `scripts/build_knowledge_graph.py` — substring match on the session TITLE. `'mastering'`,
     `'kaggle'`, `'journey'` and `'in the real world'` are literal "theory signals", which is why
     `Building a Learning Path Generator` came out theory_heavy. Title-based, never reads content.
  2. `scripts/prepare_data.py` — hardcoded, including a blanket "gen_ai ⇒ theory_heavy".
  3. `src/session_understanding.py` — the LLM, from the session's actual READING MATERIAL. The live
     pipeline uses this one, and it is the only content-based source.
  4. `eval/eval_sets.json` — a stale 2026-06-17 copy of (1).

Live code should always prefer `SessionContext.session_type`, which is (3). This module exists for the
cases where there is no SessionContext to read — labelling a historical reviewer decision, or an eval
session with no reading material — and it resolves in order of trustworthiness:

    session_outcomes.review.json (LLM-derived, per session)  →  knowledge_graph.json (title heuristic)
    →  "mixed"

Reviewer labels are keyed on the COMBINED run name ("A + B + C"), so a name is split with
`memory._split_sessions` and folded with the same rule the pipeline uses: any code_heavy ⇒ code_heavy,
all theory_heavy ⇒ theory_heavy, else mixed.
"""
from __future__ import annotations

import json
from functools import lru_cache

from src.config import DATA_DIR, normalize_session_type

# LLM-derived per-session types, written by scripts/audit_outcomes.py. More trustworthy than the
# knowledge graph's title heuristic because it was produced from the reading material.
_REVIEW_JSON = DATA_DIR / "reading_materials" / "session_outcomes.review.json"
_KG_JSON = DATA_DIR / "knowledge_graph.json"


@lru_cache(maxsize=1)
def _type_index() -> dict[str, str]:
    """{session_name: session_type}, review.json winning over the knowledge graph."""
    index: dict[str, str] = {}

    # Lowest priority first so better sources overwrite.
    try:
        kg = json.loads(_KG_JSON.read_text(encoding="utf-8"))
        for name, info in (kg.get("sessions") or {}).items():
            if isinstance(info, dict) and info.get("session_type"):
                index[name] = normalize_session_type(info["session_type"])
    except Exception:  # noqa: BLE001 — type resolution must never break a run
        pass

    try:
        review = json.loads(_REVIEW_JSON.read_text(encoding="utf-8"))
        for name, info in (review or {}).items():
            if isinstance(info, dict) and info.get("session_type"):
                index[name] = normalize_session_type(info["session_type"])
    except Exception:  # noqa: BLE001
        pass

    return index


def _normalize_key(name: str) -> str:
    return " ".join((name or "").lower().split())


@lru_cache(maxsize=512)
def type_for_session(name: str) -> str | None:
    """Type for ONE session name (exact, then case/space-normalized). None if unknown."""
    if not name:
        return None
    index = _type_index()
    if name in index:
        return index[name]
    target = _normalize_key(name)
    for known, value in index.items():
        if _normalize_key(known) == target:
            return value
    return None


def fold_types(types: list[str]) -> str:
    """Combine per-session types the way the pipeline does (session_understanding.py:181-186)."""
    known = [normalize_session_type(t) for t in types if t]
    if not known:
        return "mixed"
    if "code_heavy" in known:
        return "code_heavy"
    if all(t == "theory_heavy" for t in known):
        return "theory_heavy"
    return "mixed"


def type_for_run(combined_name: str) -> str | None:
    """Type for a possibly-combined run name ("A + B + C"). None when no part is known.

    Returns None rather than "mixed" for an entirely unknown name, so callers can tell "this is a
    genuinely mixed session" from "we have no idea what this is" — the difference matters when the
    caller is deciding whether a per-type score is measurable at all.
    """
    from src.memory import _split_sessions

    parts = _split_sessions(combined_name or "")
    resolved = [t for t in (type_for_session(p) for p in parts) if t]
    return fold_types(resolved) if resolved else None
