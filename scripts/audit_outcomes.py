#!/usr/bin/env python3
"""Audit learning-outcome alignment for EVERY session, and seed an editable override store.

For each session in data/reading_materials/session_map.json it runs the normal understanding step
(reading-material → outcomes + interview_topics + session_type) and writes a human-reviewable report
to data/reading_materials/session_outcomes.review.json, flagging likely-weak resolutions (thin
reading material, too few outcomes, generic KG-fallback outcomes, missing interview topics).

It also seeds data/reading_materials/session_outcomes.json (the editable OVERRIDE store consumed by
session_understanding) with each session's derived outcomes/interview_topics — but ONLY if that file
does not already exist, so it never clobbers hand-curated edits. Correct the misaligned entries there
and they become locked-in and stable across runs.

Run:  python scripts/audit_outcomes.py         (makes one understanding LLM call per session)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SESSION_MAP_JSON, SESSION_OUTCOMES_JSON  # noqa: E402
from src.data_loader import get_data_store                       # noqa: E402
from src.session_understanding import understand_session          # noqa: E402

REVIEW_JSON = SESSION_OUTCOMES_JSON.parent / "session_outcomes.review.json"
_THIN_RM = 3000   # chars


def _outcome_flags(rm_chars: int, learning_outcomes: list, interview_topics: list) -> list[str]:
    f = []
    if rm_chars < _THIN_RM:
        f.append("thin_rm")
    if len(learning_outcomes) < 3:
        f.append("few_outcomes")
    # Real KG-fallback template is a BARE "Understand <kp label>" (terse, no elaboration). A normal
    # LLM outcome like "Understand the role of LLMs as the core engine…" is long → NOT flagged.
    if any(o.strip().lower().startswith("understand ") and len(o.split()) <= 6 for o in learning_outcomes):
        f.append("terse_kg_outcome")
    if not interview_topics:
        f.append("no_interview_topics")
    return f


def _flags(rm_chars: int, ctx) -> list[str]:
    return _outcome_flags(rm_chars, ctx.learning_outcomes, ctx.interview_topics)


def main() -> int:
    session_map = json.loads(SESSION_MAP_JSON.read_text(encoding="utf-8"))
    store = get_data_store()
    sessions = list(session_map.keys())
    print(f"auditing {len(sessions)} sessions …", flush=True)

    review: dict = {}
    seed: dict = {}
    flagged = 0
    for i, name in enumerate(sessions, 1):
        rm_chars = len(session_map.get(name) or "")
        try:
            ctx = understand_session([name], store)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(sessions)}] {name[:50]} — ERROR: {e}")
            review[name] = {"error": str(e)[:200], "rm_chars": rm_chars}
            continue
        fl = _flags(rm_chars, ctx)
        if fl:
            flagged += 1
        review[name] = {
            "rm_chars": rm_chars,
            "session_type": ctx.session_type,
            "learning_outcomes": ctx.learning_outcomes,
            "interview_topics": ctx.interview_topics,
            "flags": fl,
        }
        seed[name] = {
            "learning_outcomes": ctx.learning_outcomes,
            "interview_topics": ctx.interview_topics,
        }
        print(f"  [{i}/{len(sessions)}] {name[:50]:52} {('⚠ ' + ','.join(fl)) if fl else 'ok'}", flush=True)

    REVIEW_JSON.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote review → {REVIEW_JSON}  ({flagged} session(s) flagged for review)")

    if SESSION_OUTCOMES_JSON.exists():
        existing = json.loads(SESSION_OUTCOMES_JSON.read_text(encoding="utf-8"))
        print(f"override store already exists ({len(existing)} entries) — NOT overwriting your edits.")
    else:
        SESSION_OUTCOMES_JSON.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"seeded editable override → {SESSION_OUTCOMES_JSON}  ({len(seed)} sessions). Edit misaligned entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
