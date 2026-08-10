#!/usr/bin/env python3
"""Filter a topic's accumulated question set against its reading material.

Reports by default; only `--apply` writes, and cuts go to `quarantined_questions` (recoverable), never
deleted. `memory.db` is backed up before the first write.

    python scripts/filter_topic_sets.py                    # the proposal
    python scripts/filter_topic_sets.py --relative         # per-topic floor instead of absolute
    python scripts/filter_topic_sets.py --apply            # quarantine the proposed cuts

WHY THE FLOOR IS A JUDGEMENT CALL, NOT A CONSTANT
-------------------------------------------------
An ABSOLUTE floor is not equally fair across topics. Measured on the shipped sets, 0.45 sits at the
**16th percentile** of Gen AI Foundations but the **0th** of No-Code AI Automation and Image Generation —
so the same number cuts 16% of one topic and nothing in another, driven by how well each topic's material
happens to embed rather than by question quality. `--relative` uses `SESSION_FIT_RELATIVE * best_fit`,
the same shape `_score_session_fit` already uses at retrieval time, which self-calibrates per topic.

Both measures agree on WHERE the loosely-related questions are (Gen AI Foundations), which is the reason
to trust the concentration even though the exact boundary is arbitrary.

WHAT THIS MEASURE CANNOT DO
---------------------------
Proximity is not a verdict. CLAUDE.md records that on/off-syllabus similarity distributions OVERLAP
(legitimate 0.52-0.83, false 0.38-0.69), so no threshold separates them — "judgement against the material
is the only separator". Hence:

  * an `approved` question is NEVER cut on proximity alone — a reviewer blessed it;
  * a question naming one of the topic's `_tool_terms` is exempt;
  * every line of the report is labelled with the measure that produced it.

`--judge` (LLM reading the actual material via `tools._syllabus_audit`, one call per topic, its reply
verified in code by `_concept_is_absent`) is the accurate filter and is not yet wired here — it needs a
working OPENROUTER_API_KEY.

Cuts are quarantined but NOT recorded as rejections: `rejected_questions` is the reviewer's learning
signal that feeds `human_agreement` and the learned rules, and an automated grounding cut is not a human
judgement. So a later run may re-surface one of these — now scored by the fixed profile — and the reviewer
decides. That is deliberate.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics as st
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings, memory  # noqa: E402
from src.config import (DATA_DIR, MEMORY_DB, SESSION_FIT_RELATIVE,  # noqa: E402
                        SESSION_PROFILE_RM_WEIGHT)
from src.pipeline import _session_profile  # noqa: E402
from src.tools import _tool_terms  # noqa: E402


def _topic_context(sessions: list[str]) -> SimpleNamespace:
    """Merge the topic's curated outcomes, as a run over those sessions would see them."""
    path = DATA_DIR / "reading_materials" / "session_outcomes.review.json"
    so = json.loads(path.read_text()) if path.exists() else {}
    oc, it, sc, kc = [], [], [], []
    for s in sessions:
        v = so.get(s) or {}
        oc += v.get("learning_outcomes") or []
        it += v.get("interview_topics") or []
        sc += v.get("scope_in") or []
        kc += v.get("key_concepts") or []
    return SimpleNamespace(learning_outcomes=oc, interview_topics=it, key_concepts=kc, scope_in=sc,
                           scope_out=[], session_name=" + ".join(sessions))


def _fits(texts: list[str], curated: list[str], rm: list[str]) -> list[float]:
    cm = embeddings.cosine_matrix(texts, curated) if curated else None
    bm = embeddings.cosine_matrix(texts, rm) if rm else None
    out = []
    for i in range(len(texts)):
        a = float(max(cm[i])) if cm is not None else 0.0
        b = float(max(bm[i])) * SESSION_PROFILE_RM_WEIGHT if bm is not None else 0.0
        out.append(max(a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=float, default=0.45, help="absolute grounding floor (default 0.45)")
    ap.add_argument("--relative", action="store_true",
                    help=f"use {SESSION_FIT_RELATIVE} * best_fit per topic instead of --floor")
    ap.add_argument("--topic", default=None, help="substring match on one topic key")
    ap.add_argument("--apply", action="store_true", help="quarantine the proposed cuts")
    args = ap.parse_args()

    structure = json.loads((DATA_DIR / "course_structure.json").read_text())
    con = memory.get_connection()
    keys = [r["topic_key"] for r in con.execute(
        "SELECT topic_key, COUNT(*) n FROM topic_question_set GROUP BY topic_key ORDER BY n DESC")]
    con.close()

    plan, skipped = [], []
    for tk in keys:
        if args.topic and args.topic.lower() not in tk.lower():
            continue
        sessions = structure.get(tk) or []
        if not sessions:
            skipped.append(tk)                       # custom topic with no session list
            continue
        ctx = _topic_context(sessions)
        curated, rm = _session_profile(sessions, ctx)
        rows = memory.get_topic_questions(tk)
        if not rows or (not curated and not rm):
            continue
        fits = _fits([r["content"] for r in rows], curated, rm)
        if fits is None:
            continue
        floor = (SESSION_FIT_RELATIVE * max(fits)) if args.relative else args.floor
        terms = _tool_terms(ctx)
        cuts, exempt = [], []
        for row, fit in zip(rows, fits):
            if fit >= floor:
                continue
            tool = next((t for t in terms
                         if re.search(r"\b" + re.escape(t) + r"\b", row["content"], re.I)), None)
            reason = ("approved by a reviewer" if row["status"] == "approved"
                      else f"names a taught tool ({tool})" if tool else None)
            (exempt if reason else cuts).append((fit, row, reason))
        plan.append({"topic": tk, "floor": floor, "n": len(rows), "median": st.median(fits),
                     "cuts": cuts, "exempt": exempt})

    measure = "embedding-proximity (relative)" if args.relative else "embedding-proximity (absolute)"
    print(f"measure: {measure}\n")
    total_cut = total_exempt = 0
    for p in plan:
        if not p["cuts"] and not p["exempt"]:
            continue
        print(f"{p['topic']}  ({p['n']} questions, median {p['median']:.3f}, floor {p['floor']:.3f})")
        for fit, row, _ in p["cuts"]:
            print(f"   CUT  {fit:.3f} [{row['status'][:4]}] {row['content'][:76]}")
        for fit, row, reason in p["exempt"]:
            print(f"   KEEP {fit:.3f} [{row['status'][:4]}] {row['content'][:56]}  <- {reason}")
        print()
        total_cut += len(p["cuts"])
        total_exempt += len(p["exempt"])
    if skipped:
        print(f"  skipped {len(skipped)} topic(s) with no course_structure entry (custom runs)\n")
    print(f"=> {total_cut} to cut, {total_exempt} kept as exempt")

    if not args.apply:
        print("\n  [report only — pass --apply to quarantine the cuts]")
        return 0
    if not total_cut:
        print("\n  nothing to apply")
        return 0

    backup = Path(str(MEMORY_DB) + f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(MEMORY_DB, backup)
    print(f"\n  backed up {MEMORY_DB.name} -> {backup.name}")
    applied = 0
    for p in plan:
        for fit, row, _ in p["cuts"]:
            memory.quarantine_question(
                p["topic"], row["content"],
                f"grounding {fit:.3f} below the {p['floor']:.3f} floor ({measure})",
                row.get("first_run_id"))
            if memory.remove_topic_question(p["topic"], row["content"]):
                applied += 1
    # Re-render each affected canonical payload. Without this the cut exists in the database and NOT in
    # the product: /review/<id> and the Sheets export both read the payload, not the set.
    for p in plan:
        if p["cuts"]:
            n = memory.sync_canonical_payload(p["topic"])
            if n is not None:
                print(f"  synced {p['topic'][:44]}: payload now {n}")
    con = memory.get_connection()
    live = con.execute("SELECT COUNT(*) FROM topic_question_set").fetchone()[0]
    quar = con.execute("SELECT COUNT(*) FROM quarantined_questions").fetchone()[0]
    con.close()
    print(f"  quarantined {applied}   live now {live}   quarantined total {quar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
