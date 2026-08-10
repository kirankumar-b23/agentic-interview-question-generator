#!/usr/bin/env python3
"""Consolidate every historical run into ONE canonical run per topic.

WHY
---
The same topics were generated many times, so their questions are scattered across runs and History is a
list of near-duplicate versions. Measured on the shipped database: **53 runs across 9 topics holding 446
questions**, which reduce to 305 exact-unique, 278 that still pass today's gates, and **251** once
semantically deduplicated per topic — with **27** quarantined.

After this script each topic has one canonical run whose payload holds the merged set, so `/review/<id>`,
approve and the Sheets export keep working with no new UI. The other runs are FLAGGED, never deleted.

WHAT IT DOES NOT DO
-------------------
Nothing is destroyed. `run_results` payloads all stay and `run_history` rows are only annotated
(`superseded_by`), because those payloads are the replay data every threshold decision in this project was
calibrated against — the coverage gate, the relevance back-fill target and the de-stack verdict were all
measured from them. `memory.db` is copied to a timestamped backup before the first write regardless.

Questions today's gates reject are QUARANTINED, not dropped and not shipped: importing them would
re-admit exactly what the pipeline improvements removed (the 27 include the sentence fragment
"how you would iteratively improve prompts and guards to increase reliability." that CLAUDE.md records as
a shipped defect, and a title-cased heading).

Run:  python scripts/consolidate_topic_sets.py --dry-run    # report only, writes nothing
      python scripts/consolidate_topic_sets.py              # apply (backs up memory.db first)

No network, no LLM. Embeddings are local; it degrades to exact-dedup only if they are unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import memory  # noqa: E402
from src.assessment_items import is_assessment_item  # noqa: E402
from src.config import DEDUP_SEMANTIC_THRESHOLD, MEMORY_DB  # noqa: E402
from src.interview_format import is_hands_on_task  # noqa: E402
from src.quality import is_quality_question  # noqa: E402


def _gate_reason(content: str) -> str | None:
    """Why today's pipeline would refuse this question, or None if it still passes."""
    if not is_quality_question(content):
        return "fails the form gate (fragment, heading or expository prose)"
    if is_hands_on_task(content):
        return "hands-on task prompt — unanswerable in a conversational interview"
    if is_assessment_item(content):
        return "multiple-choice assessment item"
    return None


def _load_runs() -> tuple[list[dict], set]:
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.row_factory = sqlite3.Row
    approved = {r[0] for r in conn.execute("SELECT run_id FROM run_history WHERE approved=1")}
    # NEVER re-consume this script's own output. The canonical runs live in run_results too, so a second
    # invocation grouped them into their own topics and flagged them superseded — History then showed 2
    # rows instead of 9. Skip them explicitly rather than relying on the id prefix alone.
    try:
        canonical = {r[0] for r in conn.execute("SELECT run_id FROM topic_runs")}
    except sqlite3.OperationalError:
        canonical = set()
    rows = conn.execute(
        """SELECT rr.run_id, rr.payload_json, rr.created_at
           FROM run_results rr ORDER BY rr.created_at""").fetchall()
    conn.close()
    rows = [r for r in rows
            if r["run_id"] not in canonical and not str(r["run_id"]).startswith("topic-")]
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:  # noqa: BLE001 — a corrupt row must not stop the consolidation
            continue
        session = ((payload.get("context") or {}).get("session_name") or "").strip()
        if not session:
            continue
        out.append({"run_id": r["run_id"], "created_at": r["created_at"],
                    "session_name": session, "payload": payload})
    return out, approved


def _semantic_dedup(details: list[dict]) -> tuple[list[dict], int]:
    """Collapse rewordings within a topic. Keeps the FIRST of each cluster (approved sort first)."""
    if len(details) < 2:
        return details, 0
    from src import embeddings
    sim = embeddings.cosine_matrix([d["content"] for d in details])
    if sim is None:
        return details, 0                     # no embeddings → exact dedup only, reported as 0
    keep, dropped = [], set()
    for i in range(len(details)):
        if i in dropped:
            continue
        keep.append(details[i])
        for j in range(i + 1, len(details)):
            if j not in dropped and sim[i][j] >= DEDUP_SEMANTIC_THRESHOLD:
                dropped.add(j)
    return keep, len(dropped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    runs, approved_ids = _load_runs()
    if not runs:
        print("No persisted runs — nothing to consolidate.")
        return 1

    # Group every run's questions under its topic, keeping the FULL original detail dict so no field
    # (role, source_url, session_fit, company…) is lost by reconstruction.
    topics: dict[str, dict] = defaultdict(lambda: {"runs": [], "details": {}, "raw": 0})
    for run in runs:
        tk = memory.topic_key_for(run["session_name"])
        t = topics[tk]
        t["runs"].append(run)
        for q in ((run["payload"].get("output") or {}).get("question_details") or []):
            content = (q.get("content") or "").strip()
            if not content:
                continue
            t["raw"] += 1
            norm = memory.normalize_content(content)
            is_appr = run["run_id"] in approved_ids
            prev = t["details"].get(norm)
            # First occurrence wins, but an approved occurrence upgrades the status.
            if prev is None:
                t["details"][norm] = {"content": content, "detail": q, "first_run_id": run["run_id"],
                                      "session_name": run["session_name"],
                                      "status": "approved" if is_appr else "backfilled"}
            elif is_appr:
                prev["status"] = "approved"

    total = {"raw": 0, "exact": 0, "clean": 0, "live": 0, "quarantined": 0, "superseded": 0}
    plan = []
    for tk, t in sorted(topics.items(), key=lambda kv: -len(kv[1]["runs"])):
        exact = list(t["details"].values())
        exact.sort(key=lambda d: 0 if d["status"] == "approved" else 1)
        quarantine = [(d, r) for d in exact if (r := _gate_reason(d["content"]))]
        clean = [d for d in exact if not _gate_reason(d["content"])]
        live, sem_dropped = _semantic_dedup(clean)

        # Canonical run: newest APPROVED run for the topic, else the newest run.
        appr = [r for r in t["runs"] if r["run_id"] in approved_ids]
        canonical = (appr or t["runs"])[-1]

        plan.append({"topic_key": tk, "live": live, "quarantine": quarantine,
                     "canonical": canonical, "runs_all": list(t["runs"]),
                     "raw": t["raw"], "exact": len(exact), "clean": len(clean), "sem": sem_dropped})
        total["raw"] += t["raw"]; total["exact"] += len(exact); total["clean"] += len(clean)
        total["live"] += len(live); total["quarantined"] += len(quarantine)
        total["superseded"] += len(t["runs"])

    print(f"{'topic':<50}{'runs':>5}{'raw':>6}{'uniq':>6}{'clean':>6}{'LIVE':>6}{'quar':>6}")
    print("-" * 92)
    for p in plan:
        print(f"{p['topic_key'][:50]:<50}{len(p['runs_all']):>5}{p['raw']:>6}"
              f"{p['exact']:>6}{p['clean']:>6}{len(p['live']):>6}{len(p['quarantine']):>6}")
    print("-" * 92)
    print(f"{'TOTAL':<50}{len(runs):>5}{total['raw']:>6}{total['exact']:>6}"
          f"{total['clean']:>6}{total['live']:>6}{total['quarantined']:>6}")
    print(f"\n  {len(runs)} runs -> {len(plan)} NEW canonical runs "
          f"(all {total['superseded']} originals flagged superseded, 0 deleted, 0 payloads overwritten)")

    if args.dry_run:
        print("\n  [dry run — nothing written]")
        return 0

    backup = Path(str(MEMORY_DB) + f".{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(MEMORY_DB, backup)
    print(f"\n  backed up {MEMORY_DB.name} -> {backup.name}")

    memory.init_db()
    for p in plan:
        tk = p["topic_key"]
        memory.upsert_topic_questions(tk, [
            {"content": d["content"], "detail": d["detail"], "status": d["status"],
             "first_run_id": d["first_run_id"], "session_name": d["session_name"],
             "difficulty": d["detail"].get("difficulty"),
             "company": d["detail"].get("asked_in_company"),
             "source": d["detail"].get("source"), "kp_label": d["detail"].get("kp_label")}
            for d in p["live"]])
        for d, reason in p["quarantine"]:
            memory.quarantine_question(tk, d["content"], reason, d["first_run_id"])

        # The canonical run is a NEW synthetic run, not one of the originals.
        #
        # `save_run_result` is INSERT OR REPLACE, so writing the merged set onto an existing run would
        # OVERWRITE that run's own question list — destroying the very replay data this script promises to
        # keep (`yield_report` and every threshold calibration read `question_details` per run). A fresh
        # run_id leaves all 53 payloads pristine and is honest about being a merge.
        template = p["canonical"]
        # Reuse this topic's existing canonical run if there is one, else derive a STABLE id.
        # `hash()` is randomised per process (PYTHONHASHSEED), so deriving from it made the script
        # non-idempotent: a second run minted fresh ids and orphaned the previous canonical runs.
        canonical_id = memory.get_canonical_run(tk) or (
            "topic-" + hashlib.sha1(tk.encode("utf-8")).hexdigest()[:12])
        payload = json.loads(json.dumps(template["payload"]))     # deep copy; never mutate the original
        payload["run_id"] = canonical_id
        out = dict(payload.get("output") or {})
        out["question_details"] = [json.loads(r["detail_json"]) if r.get("detail_json")
                                   else {"content": r["content"], "question_id": r["content_norm"][:36],
                                         "category": "GEN_AI", "topic": "Gen AI",
                                         "difficulty": r.get("difficulty") or "Medium",
                                         "source": r.get("source") or "interview_db"}
                                   for r in memory.get_topic_questions(tk)]
        payload["output"] = out
        payload["consolidated_from"] = [r["run_id"] for r in p["runs_all"]]
        memory.save_run_result(canonical_id, payload)
        memory.save_run(run_id=canonical_id, session_name=tk,
                        question_count=len(out["question_details"]),
                        composite_score=0.0, loops_used=0, approved=False)
        memory.set_canonical_run(tk, canonical_id)
        for r in p["runs_all"]:
            memory.mark_superseded(r["run_id"], canonical_id)

    # Verify: nothing lost.
    conn = sqlite3.connect(str(MEMORY_DB))
    kept = conn.execute("SELECT COUNT(*) FROM run_results").fetchone()[0]
    hist = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    conn.close()
    print(f"  run_results rows: {kept} (was {len(runs)})   run_history rows: {hist}")
    if kept < len(runs):
        print("  ERROR: run payloads were lost — restore from the backup.")
        return 2
    print(f"  topic sets: {len(plan)}   live questions: {total['live']}   "
          f"quarantined: {total['quarantined']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
