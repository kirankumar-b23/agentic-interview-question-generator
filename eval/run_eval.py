#!/usr/bin/env python3
"""Scored regression harness for the question-generation pipeline.

Runs the full pipeline over a sample of sessions from eval/eval_sets.json and reports
the quality metrics that matter (relevance, outcome coverage, source diversity,
difficulty balance, set size), with PASS/FAIL thresholds. Exits non-zero if any run
fails — so it can gate changes / catch regressions.

Usage:
  python eval/run_eval.py                         # 3 random eval sessions
  python eval/run_eval.py --n 5 --seed 1          # 5 random (reproducible)
  python eval/run_eval.py --session "Generative AI Foundations"
  python eval/run_eval.py --all                   # every eval session (slow / costs)
Requires OPENROUTER_API_KEY (+ TAVILY_API_KEY for web questions).
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import GenerationConfig                # noqa: E402
from src.pipeline import AgentPipeline                 # noqa: E402
from src.config import MIN_QUESTIONS, MAX_QUESTIONS    # noqa: E402
from src import memory                                 # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_sets.json"
# A run PASSES only if it is genuinely relevant AND covers the session AND is the right size.
THRESHOLDS = {"relevance": 0.6, "coverage": 0.6}


def _quiet(*_a, **_k):
    return None


def run_one(session_name: str, max_q: int) -> dict:
    cfg = GenerationConfig(session_names=[session_name], max_questions=max_q,
                           min_questions=MIN_QUESTIONS, category="GEN_AI")
    res = AgentPipeline().run(cfg, str(uuid.uuid4()), _quiet)
    m = res.quality_report.metric_scores
    out = res.curated_output
    return {
        "session": session_name,
        "count": len(out.question_details),
        "relevance": m.get("relevance", 0.0),
        "coverage": m.get("outcome_coverage", 0.0),
        "diversity": m.get("source_diversity", 0.0),
        "difficulty": m.get("difficulty_balance", 0.0),
        "composite": res.quality_report.composite_score,
        "sources": dict(out.metadata.source_counts),
        "contents": [q.content for q in out.question_details],
    }


def _bad_by_session() -> dict:
    """Reviewer-rejected ('bad') question texts (normalized) grouped by session, from feedback_examples.json."""
    out: dict[str, set] = {}
    for ex in memory.get_feedback_examples():
        if ex.get("decision") == "bad" and ex.get("question"):
            out.setdefault(ex.get("session", ""), set()).add(memory.normalize_content(ex["question"]))
    return out


def _passed(r: dict) -> bool:
    return (r["relevance"] >= THRESHOLDS["relevance"]
            and r["coverage"] >= THRESHOLDS["coverage"]
            and MIN_QUESTIONS <= r["count"] <= MAX_QUESTIONS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of random eval sessions")
    ap.add_argument("--session", action="append", help="specific session name(s)")
    ap.add_argument("--all", action="store_true", help="run every eval session")
    ap.add_argument("--max-questions", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(EVAL_PATH.read_text())
    all_sessions = [s["session_name"] for s in data.get("eval_sessions", [])]
    if args.session:
        targets = args.session
    elif args.all:
        targets = all_sessions
    else:
        targets = random.Random(args.seed).sample(all_sessions, min(args.n, len(all_sessions)))

    bad_by_session = _bad_by_session()
    fb_examples = memory.get_feedback_examples()

    rows, errors = [], []
    for name in targets:
        try:
            r = run_one(name, args.max_questions)
            # Feedback alignment: did any reviewer-rejected ('bad') question reappear in this run?
            bad = bad_by_session.get(name, set())
            r["fb_violations"] = sum(1 for c in r["contents"]
                                     if memory.normalize_content(c) in bad) if bad else 0
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            errors.append((name, str(e)[:100]))

    hdr = f"{'session':45s} {'n':>3} {'rel':>5} {'cov':>5} {'div':>5} {'dif':>5} {'comp':>5}  verdict"
    print(hdr)
    print("-" * len(hdr))
    fails = 0
    for r in rows:
        ok = _passed(r)
        fails += 0 if ok else 1
        viol = f"  ⚠{r['fb_violations']} rejected-reappeared" if r.get("fb_violations") else ""
        print(f"{r['session'][:45]:45s} {r['count']:>3} {r['relevance']:>5.2f} {r['coverage']:>5.2f} "
              f"{r['diversity']:>5.2f} {r['difficulty']:>5.2f} {r['composite']:>5.2f}  {'PASS' if ok else 'FAIL'}{viol}")
    for name, err in errors:
        fails += 1
        print(f"{name[:45]:45s}  ERROR: {err}")

    if rows:
        print(f"\nAGGREGATE  mean_relevance={statistics.mean(r['relevance'] for r in rows):.2f}  "
              f"mean_coverage={statistics.mean(r['coverage'] for r in rows):.2f}  "
              f"mean_count={statistics.mean(r['count'] for r in rows):.1f}")
    # Feedback loop summary (reviewer decisions fed back into eval).
    if fb_examples:
        good = sum(1 for e in fb_examples if e.get("decision") == "good")
        bad = sum(1 for e in fb_examples if e.get("decision") == "bad")
        total_viol = sum(r.get("fb_violations", 0) for r in rows)
        print(f"FEEDBACK   {len(fb_examples)} reviewer decisions ({good} good / {bad} bad) across "
              f"{len({e.get('session') for e in fb_examples})} session(s); "
              f"rejected-reappeared this run: {total_viol}")
    total = len(rows) + len(errors)
    print(f"RESULT  {total - fails}/{total} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
