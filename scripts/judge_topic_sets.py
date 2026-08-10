#!/usr/bin/env python3
"""Run the two LLM-judged passes over each topic's ACCUMULATED set. Proposes; never deletes.

Both passes already exist and both have only ever run on the SELECTED set inside a single run, so the
accumulated per-topic sets — which is what Review and the Sheets export now read — have never been judged
by either:

  * `tools._same_thing_pass`  — the accurate duplicate test, judging pairs in [0.62, 0.82).
  * `tools._syllabus_audit`   — the accurate syllabus test, judging questions against the reading material.

WHAT EACH ONE CAN AND CANNOT SEE
--------------------------------
`_same_thing_pass` judges pairs `_near_duplicate_pairs` hands it, and that floor is `_SAME_THING_LOW =
0.62`. On the six hallucination questions in No-Code AI Automation five pairs sit in that band, but the two
most obviously identical — "What are hallucinations in LLMs" / "What is an AI hallucination" — score
**0.486** and are never shown to it. So this script is a COMPLEMENT to `filter_topic_sets.py --dupes`
(shared distinctive term, no similarity floor), not a replacement. Neither subsumes the other:

    --dupes             groups all six, cannot tell "what is it" from "how do you prevent it"
    _same_thing_pass    judges that distinction, never sees the 0.486 pair

`_syllabus_audit` is the only thing in the codebase that judges a question against the reading material
itself — the relevance judge sees `scope_in`/`scope_out` summaries. It is what answers "is
*Explain your methodology for designing and testing system prompts to prevent model hallucination*
supported?", where the material has `hallucinat*` 3 times and `system prompt` **0**.

PROPOSES ONLY
-------------
Both passes here are read-only by construction: `--apply` is deliberately absent. `_same_thing_pass`
mutates the state it is given, so the questions handed to it are throwaway `QuestionDetail` copies and the
verdicts are printed, not written. Collapsing a duplicate cluster is a judgement about your interview
(is "what is it" a separate ask from "how do you prevent it"?), and removing a reviewer-APPROVED question
on an LLM's say-so is exactly the kind of silent cut this project keeps getting bitten by. Act on the
output with `filter_topic_sets.py --collapse TERM` or `--quarantine`, which back up the database first.

COST: two calls per topic, ~18 for all nine. Needs a working `OPENROUTER_API_KEY`; a 401 is reported per
topic and the run continues, because a dead pass must not look like a clean set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import memory  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.models import GenerationConfig, QuestionDetail  # noqa: E402


def _context(sessions: list[str]) -> SimpleNamespace:
    """The topic's curated outcomes, as a run over these sessions would see them.

    `_syllabus_audit` scores against `coverage_targets(ctx)` — interview_topics, falling back to outcomes —
    so a context missing `interview_topics` makes it return empty and look like "nothing off-syllabus".
    """
    path = DATA_DIR / "reading_materials" / "session_outcomes.review.json"
    so = json.loads(path.read_text()) if path.exists() else {}
    oc, it, sc, kc = [], [], [], []
    for s in sessions:
        v = so.get(s) or {}
        oc += v.get("learning_outcomes") or []
        it += v.get("interview_topics") or []
        sc += v.get("scope_in") or []
        kc += v.get("key_concepts") or []
    return SimpleNamespace(session_name=" + ".join(sessions), session_type="mixed",
                           learning_outcomes=oc, interview_topics=it, key_concepts=kc,
                           scope_in=sc, scope_out=[], matched_kp_ids=[])


def _state(sessions: list[str], rows: list[dict]):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=list(sessions)), data_store=get_data_store())
    st.session_context = _context(sessions)
    qs = []
    for i, r in enumerate(rows):
        qs.append(QuestionDetail(question_id=f"tq{i}", category="GEN_AI", content=r["content"],
                                 topic="Gen AI", difficulty=r.get("difficulty") or "Medium",
                                 source=r.get("source") or "interview_db"))
    st.questions = {q.question_id: q for q in qs}
    return st, qs


def _band_size(qs) -> int:
    """How many pairs are ELIGIBLE to be judged, before `_SAME_THING_MAX_PAIRS` truncates.

    Worth reporting separately because the cap is 12 — sized for a ~10-question selected set — while an
    accumulated topic set is much larger. Measured on No-Code AI Automation (38 questions): **41 eligible
    pairs, 29 never judged**, and the one hallucination pair the judge does call redundant sits at rank 14,
    just past the cap. So "0 redundant" from this pass can mean "not looked at", which is precisely the
    silent-shortfall class this project keeps hitting.
    """
    from src import embeddings
    from src.config import DEDUP_SEMANTIC_THRESHOLD
    from src.tools import _SAME_THING_LOW

    if len(qs) < 2:
        return 0
    sim = embeddings.cosine_matrix([q.content for q in qs])
    if sim is None:
        return 0
    return sum(1 for i in range(len(qs)) for j in range(i + 1, len(qs))
               if _SAME_THING_LOW <= float(sim[i][j]) < DEDUP_SEMANTIC_THRESHOLD)


def _dupes(state, qs, rows) -> None:
    from src.tools import _SAME_THING_MAX_PAIRS, _near_duplicate_pairs, _same_thing_pass

    eligible = _band_size(qs)
    if eligible > _SAME_THING_MAX_PAIRS:
        print(f"   NOTE: {eligible} pairs are eligible but the cap is {_SAME_THING_MAX_PAIRS} — "
              f"{eligible - _SAME_THING_MAX_PAIRS} will NOT be judged")
    pairs = _near_duplicate_pairs(qs)
    if not pairs:
        print("   same-thing: no pair reaches the 0.62 judging floor "
              "(this does NOT mean no duplicates — see --dupes)")
        return
    # `_same_thing_pass` does `questions[:] = [...]`, so it MUTATES the list it is handed. A "what
    # disappeared" diff computed against `qs` afterwards can never find a removed question — the first
    # version of this reported "0 redundant" while the pass was removing them. Snapshot first.
    snapshot = {q.question_id: q.content for q in qs}
    res = _same_thing_pass(state, qs)
    judged = res.get("pairs_judged") or 0
    if not judged:
        print(f"   same-thing: PASS DID NOT RUN ({len(pairs)} pairs were ready) — check the API key")
        return
    gone = [c for qid, c in snapshot.items() if qid not in state.questions]
    flagged = [q for q in qs if getattr(q, "duplicate_of", None)]
    print(f"   same-thing: judged {judged} of {len(pairs)} ready pair(s) -> "
          f"{res.get('removed')} redundant, {res.get('flagged')} flagged")
    for c in gone:
        print(f"      REDUNDANT  {c[:88]}")
    for q in flagged:
        print(f"      flagged    {q.content[:70]}\n                 same as: {q.duplicate_of[:70]}")


def _syllabus(state, qs) -> None:
    from src.tools import _syllabus_audit

    res = _syllabus_audit(state, qs)
    off = res.get("off_syllabus") or []
    cov = res.get("coverage") or {}
    if not off and not cov:
        print("   syllabus  : PASS DID NOT RUN or returned nothing — check the API key")
        return
    frac = cov.get("fraction")
    print(f"   syllabus  : {len(off)} off-syllabus, judged coverage "
          f"{f'{frac:.2f}' if isinstance(frac, (int, float)) else 'unavailable'}")
    for item in off:
        q = state.questions.get(item.get("question_id")) if isinstance(item, dict) else None
        text = (q.content if q else str(item))[:80]
        why = item.get("concept") if isinstance(item, dict) else ""
        print(f"      OFF-SYLLABUS  {text}\n                    untaught: {why}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", help="substring of the topic key; default every topic")
    ap.add_argument("--skip-dupes", action="store_true")
    ap.add_argument("--skip-syllabus", action="store_true")
    args = ap.parse_args()

    structure = json.loads((DATA_DIR / "course_structure.json").read_text())
    conn = memory.get_connection()
    keys = [r[0] for r in conn.execute(
        "SELECT DISTINCT topic_key FROM topic_question_set ORDER BY topic_key")]
    conn.close()

    sessions_for = {}
    for topic, sess in structure.items():
        names = [s["name"] if isinstance(s, dict) else s for s in sess]
        for n in names:
            sessions_for.setdefault(memory.topic_key_for(n), names)

    for tk in keys:
        if args.topic and args.topic.lower() not in tk.lower():
            continue
        sessions = sessions_for.get(tk)
        rows = memory.get_topic_questions(tk)
        if not rows:
            continue
        if not sessions:
            print(f"{tk}  ({len(rows)} questions)\n   SKIPPED: no sessions resolved for this topic key")
            continue
        print(f"{tk}  ({len(rows)} questions, {len(sessions)} session(s))")
        state, qs = _state(sessions, rows)
        if not args.skip_dupes:
            _dupes(state, qs, rows)
        if not args.skip_syllabus:
            _syllabus(state, qs)
        print()

    print("Nothing was changed. Act on this with:")
    print("  scripts/filter_topic_sets.py --dupes --collapse TERM   (duplicates, backed up first)")
    print("  scripts/filter_topic_sets.py --evidence 'phrase'       (the material behind a question)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
