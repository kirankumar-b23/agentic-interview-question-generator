"""Retrieval-yield harness — how many questions actually reach the reviewer.

WHY THIS EXISTS
---------------
A session's worth of changes each passed their own tests and the end-to-end yield still fell from 5
questions to 3. Every filter was tuned in isolation against its own fixtures; nothing measured the only
number that matters — how many good questions survive the whole funnel.

So: before changing any threshold, filter or gate, run this. It reads the runs persisted in
`memory.db` (`run_results.payload_json`) and reports the funnel plus the final set size, and it flags
runs where the funnel collapsed. Compare before/after a change.

    python3 scripts/yield_report.py                 # all persisted runs
    python3 scripts/yield_report.py --last 10
    python3 scripts/yield_report.py --topic n8n     # substring match on the session name

Reads only. Never writes, never calls an LLM or Tavily.

READING IT
----------
`raw -> pool` is what the form gate, topic-trim and per-source caps cost. `pool -> final` is the
semantic stack: session-fit, the off-topic pre-filter and the LLM relevance judge. It is NOT pure
subtraction: the last-resort open-web tier adds candidates after the pre-filter, which is what the
`+web` column accounts for — without it the stage counts look like they do not reconcile. When `final` sits at
or below MIN_QUESTIONS the set was supply-starved, and a threshold change that lowers `final` further is
a regression however good its own unit tests look.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MEMORY_DB, MIN_QUESTIONS  # noqa: E402


def _rows(limit: int | None, topic: str | None):
    con = sqlite3.connect(MEMORY_DB)
    approved = {r[0] for r in con.execute("select run_id from run_history where approved=1")}
    q = "select run_id, payload_json, created_at from run_results order by created_at"
    out = []
    for run_id, payload, created in con.execute(q):
        try:
            d = json.loads(payload)
        except Exception:  # noqa: BLE001 — a corrupt row must not stop the report
            continue
        name = ((d.get("context") or {}).get("session_name") or "")
        if topic and topic.lower() not in name.lower():
            continue
        out.append((run_id, d, created, run_id in approved))
    return out[-limit:] if limit else out


def _funnel(d: dict) -> dict:
    out = d.get("output") or {}
    md = out.get("metadata") or {}
    raw = md.get("raw_fetched") or {}
    stages = Counter(r.get("stage") for r in (d.get("removed") or []))
    return {
        "raw": sum(v for v in raw.values() if isinstance(v, int)),
        "pool": md.get("pool_size") or 0,
        "final": len(out.get("question_details") or []),
        "session_fit": stages.get("session_fit", 0),
        "prefilter": stages.get("off_topic_prefilter", 0),
        "relevance": stages.get("relevance", 0),
        "suppressed": stages.get("suppressed", 0),
        # Counted explicitly: a filter that silently shrinks the pool reads as "nothing was dropped".
        "hands_on": stages.get("hands_on", 0),
        # The open-web tier ADDS candidates after the prefilter, so pool -> final is not pure
        # subtraction. Without this column a topped-up run looks like the stage counts do not add up.
        "web_added": (raw.get("open_web") or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--last", type=int, default=None, help="only the N most recent runs")
    ap.add_argument("--topic", default=None, help="substring match on the session name")
    args = ap.parse_args()

    rows = _rows(args.last, args.topic)
    if not rows:
        print("No persisted runs matched. (run_results is written by main.py on each completed run.)")
        return 1

    print(f"{'run':<10}{'raw':>7}{'pool':>7}{'hand-':>7}{'fit-':>6}{'pre-':>6}{'+web':>6}{'rel-':>6}{'FINAL':>7}"
          f"{'verdict':>9}  topic")
    print("-" * 104)
    finals, starved, approved_finals = [], 0, []
    for run_id, d, _created, was_approved in rows:
        f = _funnel(d)
        rep = d.get("report") or {}
        name = ((d.get("context") or {}).get("session_name") or "")[:34]
        if f["final"] == 0:
            continue                      # dead run (API outage) — not a yield signal
        finals.append(f["final"])
        if was_approved:
            approved_finals.append(f["final"])
        if f["final"] <= MIN_QUESTIONS:
            starved += 1
        print(f"{run_id[:8]:<10}{f['raw']:>7}{f['pool']:>7}{f['hands_on']:>7}{f['session_fit']:>6}{f['prefilter']:>6}"
              f"{f['web_added']:>6}{f['relevance']:>6}{f['final']:>7}{(rep.get('pass_fail') or '?'):>9}"
              f"  {name}{'  [APPROVED]' if was_approved else ''}")

    n = len(finals)
    print()
    print(f"  runs with output      : {n}")
    print(f"  median final set size : {sorted(finals)[n // 2]}")
    print(f"  mean                  : {sum(finals) / n:.1f}")
    print(f"  supply-starved (<= {MIN_QUESTIONS}) : {starved}/{n} ({100 * starved / n:.0f}%)")
    if approved_finals:
        print(f"  reviewer-APPROVED runs: n={len(approved_finals)}, "
              f"median size {sorted(approved_finals)[len(approved_finals) // 2]}")
    print()
    print("  THE NUMBER TO WATCH is median final set size. A change that improves any single metric")
    print("  while lowering this is a regression — that is how a 5-question topic became a 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
