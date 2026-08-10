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
from src.config import (DATA_DIR, MEMORY_DB, OUTCOME_CAP, SESSION_FIT_RELATIVE,  # noqa: E402
                        SESSION_PROFILE_RM_WEIGHT)
from src.llm_client import get_active_model  # noqa: E402
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



# ── Candidate near-duplicate clusters ────────────────────────────────────────────────────────────────
#
# Grouped by a SHARED DISTINCTIVE TERM, not by embedding similarity. That is not a stylistic choice:
# across the six hallucination questions in No-Code AI Automation, ZERO pairs reach the 0.82 dedup bar and
# the two most obviously identical — "What are hallucinations in LLMs" and "What is an AI hallucination" —
# score 0.486, the LOWEST of all fifteen pairs. Short definitional questions carry little signal, so
# similarity cannot group them at any threshold that is not also catastrophic elsewhere.
#
# Similarity is used only for COHESION (is this group about one thing?) and to order members.
#
# A cluster means "these ask about the same subject", NOT "these are duplicates". Measured on the shipped
# sets, 'hallucinat' (6) and 'large/language' (8) are genuine duplication, while 'workflow' (7) is seven
# genuinely different questions that happen to share a word. So collapsing is per-cluster and explicit
# (`--collapse TERM`); there is deliberately no blanket apply.
_CLUSTER_STOP = set("""what is are the a an of to in for and or how do you your with on can could would why
when which explain describe define does did it its this that these those be been being as at by from we they
not no if then than there their them use used using make makes made give given get gets between difference
differences vs versus example examples any some all more most other others improve approach through answer
aspect complex technique prevent structure design reason fail engineer engineering ensure handle manage build
create implement about should might where whose scenario situation project experience""".split())


def _terms(content: str) -> set:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", (content or "").lower())
             if w not in _CLUSTER_STOP and len(w) >= 5]
    return {re.sub(r"(ions?|s|ing|ed)$", "", w) for w in words}


def _clusters(rows: list, fits: list, min_cohesion: float = 0.45) -> list:
    """Candidate groups: shared distinctive term + mutually related members. Overlapping groups merged."""
    texts = [r["content"] for r in rows]
    sim = embeddings.cosine_matrix(texts)
    by_term: dict = {}
    for i, r in enumerate(rows):
        for t in _terms(r["content"]):
            by_term.setdefault(t, []).append(i)
    cap = max(2, int(0.30 * len(rows)))
    found = []
    for term, ids in sorted(by_term.items(), key=lambda kv: -len(kv[1])):
        if not (2 <= len(ids) <= cap):
            continue
        pairs = [float(sim[a][b]) for a in ids for b in ids if a < b] if sim is not None else []
        cohesion = st.mean(pairs) if pairs else 0.0
        if cohesion < min_cohesion:
            continue                      # a generic word shared by unrelated questions
        found.append({"term": term, "ids": set(ids), "cohesion": cohesion})
    # Merge groups that are substantially the SAME group ('large' and 'language' cover the same eight
    # questions). The bar is high and cohesion is re-checked afterwards: at 0.5 the merge chained through
    # partial overlaps and dragged "Have you used cloud services before…" into the LLM-definition cluster.
    merged: list = []
    for c in found:
        for m in merged:
            overlap = len(c["ids"] & m["ids"]) / min(len(c["ids"]), len(m["ids"]))
            if overlap < 0.8:
                continue
            union = m["ids"] | c["ids"]
            pairs = [float(sim[a][b]) for a in union for b in union if a < b] if sim is not None else []
            if pairs and st.mean(pairs) < min_cohesion:
                continue                  # merging would make the group incoherent — keep them apart
            m["ids"] = union
            m["term"] += "/" + c["term"]
            m["cohesion"] = st.mean(pairs) if pairs else m["cohesion"]
            break
        else:
            merged.append(dict(c))
    for m in merged:
        m["members"] = sorted(m["ids"], key=lambda i: (rows[i]["status"] != "approved", -fits[i]))
    return sorted(merged, key=lambda m: -len(m["ids"]))


def _evidence(structure: dict, keys: list, needle: str, only_topic: str = None) -> int:
    """How often does a question's own vocabulary appear in the material? Evidence, not a verdict.

    This is the check that answered "is this question supported?" in seconds: for
    "Explain your methodology for designing and testing system prompts to prevent model hallucination"
    the material has `hallucinat*` 3 times and **`system prompt` 0 times** — so it attaches two things the
    session never covers to one it does.

    A phrase can be absent while the concept is taught in other words, so this NEVER cuts anything.
    `_concept_is_absent` is deliberately not used here: it is a VERIFIER for an LLM's claim and returns
    False for everything (including "Split In Batches node", which has 0 occurrences), which is right for
    its job and useless as a detector.
    """
    from src.tools import _session_corpus

    for tk in keys:
        if only_topic and only_topic.lower() not in tk.lower():
            continue
        sessions = structure.get(tk) or []
        if not sessions:
            continue
        rows = [r for r in memory.get_topic_questions(tk) if needle.lower() in r["content"].lower()]
        if not rows:
            continue
        corpus = _session_corpus(sessions)
        print(f"{tk}   (material: {len(corpus)} chars across {len(sessions)} session(s))\n")
        for r in rows:
            print(f"  [{r['status'][:4]}] {r['content']}")
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", r["content"]) if len(w) >= 5]
            phrases = {w.lower() for w in words}
            toks = r["content"].split()
            phrases |= {" ".join(toks[i:i + 2]).strip(" ?.,\"'").lower() for i in range(len(toks) - 1)}
            scored = []
            for ph in phrases:
                if len(ph) < 5 or ph in _CLUSTER_STOP:
                    continue
                n = len(re.findall(re.escape(ph), corpus, re.I))
                scored.append((n, ph))
            absent = sorted(p for n, p in scored if n == 0)
            present = sorted(((n, p) for n, p in scored if n), reverse=True)[:6]
            print("      in the material : " + ", ".join(f"{p} x{n}" for n, p in present) or "      (nothing)")
            print("      NOT in it       : " + (", ".join(absent[:10]) or "(everything appears)"))
            print()
    return 0


def _dupes(structure: dict, keys: list, args) -> int:
    """Report candidate clusters, or collapse one by term."""
    total = 0
    for tk in keys:
        if args.topic and args.topic.lower() not in tk.lower():
            continue
        sessions = structure.get(tk) or []
        if not sessions:
            continue
        ctx = _topic_context(sessions)
        curated, rm = _session_profile(sessions, ctx)
        rows = memory.get_topic_questions(tk)
        if len(rows) < 2:
            continue
        fits = _fits([r["content"] for r in rows], curated, rm)
        groups = _clusters(rows, fits)
        if args.collapse:
            groups = [g for g in groups if args.collapse.lower() in g["term"].lower()]
            if not groups:
                continue
        if not groups:
            continue
        print(f"{tk}  ({len(rows)} questions)")
        for g in groups:
            print(f"   cluster '{g['term']}'  {len(g['ids'])} questions, cohesion {g['cohesion']:.3f}")
            for rank, i in enumerate(g["members"]):
                tag = "KEEP " if rank == 0 else "dup  "
                print(f"      {tag} fit {fits[i]:.3f} [{rows[i]['status'][:4]}] {rows[i]['content'][:66]}")
            total += len(g["ids"]) - 1
        print()
    if not args.collapse:
        print(f"=> {total} question(s) sit in candidate clusters beyond the best member.")
        print("   A cluster means 'same subject', NOT 'duplicate' — 'workflow' groups 7 genuinely")
        print("   different questions. Collapse one explicitly: --collapse hallucinat")
        return 0

    backup = Path(str(MEMORY_DB) + f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(MEMORY_DB, backup)
    print(f"  backed up {MEMORY_DB.name} -> {backup.name}")
    dropped = 0
    for tk in keys:
        if args.topic and args.topic.lower() not in tk.lower():
            continue
        sessions = structure.get(tk) or []
        if not sessions:
            continue
        ctx = _topic_context(sessions)
        curated, rm = _session_profile(sessions, ctx)
        rows = memory.get_topic_questions(tk)
        if len(rows) < 2:
            continue
        fits = _fits([r["content"] for r in rows], curated, rm)
        for g in _clusters(rows, fits):
            if args.collapse.lower() not in g["term"].lower():
                continue
            keeper = rows[g["members"][0]]
            for i in g["members"][1:]:
                memory.quarantine_question(
                    tk, rows[i]["content"],
                    f'duplicate of "{keeper["content"][:80]}" (cluster {g["term"]})',
                    rows[i].get("first_run_id"))
                if memory.remove_topic_question(tk, rows[i]["content"]):
                    dropped += 1
            n = memory.sync_canonical_payload(tk)
            print(f"  {tk[:46]}: kept 1, quarantined {len(g['members']) - 1}, payload now {n}")
    print(f"\n  quarantined {dropped}")
    return 0


def _cap(structure: dict, keys: list, args) -> int:
    """Balance each topic's set per interview topic. The destructive counterpart of `_cap_by_outcome`.

    Same function, different consequence — `tools._cap_by_outcome` trims only what a run SHIPS and leaves
    the accumulated set whole; this quarantines rows in the database. Both call
    `outcome_balance.balance_by_outcome` so the two cannot drift.

    Reports the uncovered outcomes either way: a topic supplying several questions is repetition the
    candidate hears, and a topic supplying none is a gap no question count reveals.
    """
    from src.outcome_balance import balance_by_outcome, make_llm_judge
    from src.pipeline import coverage_targets

    plan = []
    for tk in keys:
        if args.topic and args.topic.lower() not in tk.lower():
            continue
        sessions = structure.get(tk) or []
        if not sessions:
            continue
        ctx = _topic_context(sessions)
        outcomes = coverage_targets(ctx)
        rows = memory.get_topic_questions(tk)
        if not rows or not outcomes:
            continue
        curated, rm = _session_profile(sessions, ctx)
        texts = [r["content"] for r in rows]
        fits = _fits(texts, curated, rm)
        judge = None if args.no_judge else make_llm_judge(texts, model=get_active_model())
        res = balance_by_outcome(texts, outcomes, fits=fits,
                                 approved=[r["status"] == "approved" for r in rows],
                                 cap=args.cap, judge=judge, strict=args.strict)
        plan.append({"topic": tk, "rows": rows, "fits": fits, "outcomes": outcomes, "res": res})

        print(f"{tk}  ({len(rows)} questions, {len(outcomes)} interview topics) -> {len(res.keep)}")
        if res.judge_failed:
            print("   NOTE: the judge failed — fell back to cap-by-rank, nothing extra was dropped")
        dropped = set(res.drop)
        for t, members in sorted(res.assigned.items(), key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            print(f"   [{len(members)} -> {len(members) - sum(1 for i in members if i in dropped)}] "
                  f"{outcomes[t][:64]}")
            for i in sorted(members, key=lambda i: (rows[i]['status'] != 'approved', -fits[i])):
                print(f"      {'cut ' if i in dropped else 'KEEP'} fit {fits[i]:.3f} "
                      f"[{rows[i]['status'][:4]}] {texts[i][:64]}")
        if res.orphans:
            print(f"   {len(res.orphans)} orphan(s) KEPT — no interview topic describes them, so they are "
                  f"not redundant:")
            for i in res.orphans:
                print(f"      KEEP {texts[i][:74]}")
        if res.uncovered:
            print(f"   {len(res.uncovered)} outcome(s) with NO question:")
            for t in res.uncovered:
                print(f"      —— {outcomes[t][:74]}")
        print()

    total = sum(len(p["res"].drop) for p in plan)
    print(f"=> {total} question(s) to cut across {len(plan)} topic(s) at cap {args.cap}")
    if not args.apply:
        print("   [report only — pass --apply to quarantine the cuts]")
        return 0
    if not total:
        print("   nothing to apply")
        return 0

    backup = Path(str(MEMORY_DB) + f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(MEMORY_DB, backup)
    print(f"\n  backed up {MEMORY_DB.name} -> {backup.name}")
    applied = 0
    for p in plan:
        res, rows, outcomes = p["res"], p["rows"], p["outcomes"]
        for i, t in res.dropped_for.items():
            memory.quarantine_question(
                p["topic"], rows[i]["content"],
                f'interview topic already covered (cap {args.cap}): "{outcomes[t][:70]}"',
                rows[i].get("first_run_id"))
            if memory.remove_topic_question(p["topic"], rows[i]["content"]):
                applied += 1
        # Without this the cut exists in the database and NOT in the product: /review/<id> and the
        # Sheets export both read the canonical payload, not the set.
        if res.drop:
            n = memory.sync_canonical_payload(p["topic"])
            if n is not None:
                print(f"  synced {p['topic'][:44]}: payload now {n}")
    print(f"  quarantined {applied} (recoverable from quarantined_questions)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=float, default=0.45, help="absolute grounding floor (default 0.45)")
    ap.add_argument("--relative", action="store_true",
                    help=f"use {SESSION_FIT_RELATIVE} * best_fit per topic instead of --floor")
    ap.add_argument("--topic", default=None, help="substring match on one topic key")
    ap.add_argument("--apply", action="store_true", help="quarantine the proposed grounding cuts")
    ap.add_argument("--dupes", action="store_true",
                    help="report candidate near-duplicate clusters instead of grounding cuts")
    ap.add_argument("--collapse", metavar="TERM",
                    help="collapse ONE cluster by its term: keep the best member, quarantine the rest")
    ap.add_argument("--evidence", metavar="SUBSTRING",
                    help="show how often a question's distinctive phrases occur in the reading material")
    ap.add_argument("--cap", type=int, metavar="N", default=None,
                    help=f"balance per interview topic: keep at most N per outcome "
                         f"(default {OUTCOME_CAP}); a CEILING, the judge may keep fewer")
    ap.add_argument("--no-judge", action="store_true",
                    help="--cap without the LLM: pure cap by rank, fully reproducible, no API cost")
    ap.add_argument("--strict", action="store_true",
                    help="hard quota: drop everything past --cap even if the judge calls it distinct. "
                         "Blunt where interview_topics are coarse — see outcome_balance's docstring")
    args = ap.parse_args()

    structure = json.loads((DATA_DIR / "course_structure.json").read_text())
    con = memory.get_connection()
    keys = [r["topic_key"] for r in con.execute(
        "SELECT topic_key, COUNT(*) n FROM topic_question_set GROUP BY topic_key ORDER BY n DESC")]
    con.close()

    if args.evidence:
        return _evidence(structure, keys, args.evidence, args.topic)
    if args.dupes or args.collapse:
        return _dupes(structure, keys, args)
    if args.cap is not None:
        return _cap(structure, keys, args)

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
