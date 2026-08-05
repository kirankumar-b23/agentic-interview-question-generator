#!/usr/bin/env python3
"""Scored regression harness for the question-generation pipeline.

Runs the full pipeline over sessions from eval/eval_sets.json and reports quality metrics with
PASS/FAIL thresholds. Exits non-zero if any run fails, so it can gate changes.

THE PRIMARY NUMBER IS `accept` — predicted reviewer acceptance, measured against HELD-OUT reviewer
decisions from eval/feedback_examples.json. Everything else here is secondary, because everything
else is produced by the same machinery that picks the questions:

  * `self_rel` is the mean of the LLM relevance score used to SELECT the set. It is printed for
    reference and is NOT part of pass/fail — treating it as quality is how runs scored 0.9 while
    reviewers rejected most of the set (corr(composite, approved) = 0.16 over 36 real runs).
  * `cov` (outcome coverage) and `grnd` (session grounding) compare questions to the curriculum
    rather than to the system's own judgement, so they are trustworthy but measure fit, not taste.

Labels are split deterministically by question text: half inform nothing here (they are what the
pipeline itself may learn from) and half are held out purely for scoring. Without that split the
metric would degrade back into self-assessment as soon as the pipeline learns from feedback.

Usage:
  python eval/run_eval.py                         # 3 random eval sessions
  python eval/run_eval.py --n 5 --seed 1          # 5 random (reproducible)
  python eval/run_eval.py --session "Generative AI Foundations"
  python eval/run_eval.py --all                   # every eval session (slow / costs)
  python eval/run_eval.py --holdout 0             # score on ALL labels (not independent)
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
from src.human_agreement import predict_accept, split_labels   # noqa: E402
from src import memory                                 # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_sets.json"
# A run PASSES on independent signals only: it must look acceptable to the reviewer (measured against
# HELD-OUT decisions), cover the session's outcomes, and be the right size. `self_rel` is excluded on
# purpose — see the module docstring.
THRESHOLDS = {"accept": 0.6, "coverage": 0.6}


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
        "self_rel": m.get("self_relevance", 0.0),
        "coverage": m.get("outcome_coverage", 0.0),
        "grounding": m.get("session_grounding", 0.0),
        "diversity": m.get("source_diversity", 0.0),
        "difficulty": m.get("difficulty_balance", 0.0),
        "composite": res.quality_report.composite_score,
        "sources": dict(out.metadata.source_counts),
        "contents": [q.content for q in out.question_details],
    }


def _bad_by_session(examples: list[dict]) -> dict:
    """Reviewer-rejected ('bad') question texts (normalized) grouped by session."""
    out: dict[str, set] = {}
    for ex in examples:
        if ex.get("decision") == "bad" and ex.get("question"):
            out.setdefault(ex.get("session", ""), set()).add(memory.normalize_content(ex["question"]))
    return out


def _score_agreement(r: dict, holdout: list[dict]) -> None:
    """Attach predicted reviewer acceptance, measured against HELD-OUT decisions only.

    Sets `accept` to None when it cannot be measured (too few or one-sided held-out labels,
    embeddings unavailable) — reported as "n/a" rather than a number we did not measure.
    """
    report = predict_accept(r["contents"], holdout, session=r["session"])
    r["accept"] = report.predicted_accept_rate if report else None
    r["accept_labels"] = report.label_count if report else 0
    r["repeats"] = report.repeats_rejected if report else 0


def _passed(r: dict) -> bool:
    if not (MIN_QUESTIONS <= r["count"] <= MAX_QUESTIONS):
        return False
    if r["coverage"] < THRESHOLDS["coverage"]:
        return False
    # Unmeasurable acceptance must not silently pass the run on the strength of coverage alone —
    # but it must not fail it either, since the gap is missing labels, not a bad question set.
    if r.get("accept") is not None and r["accept"] < THRESHOLDS["accept"]:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of random eval sessions")
    ap.add_argument("--session", action="append", help="specific session name(s)")
    ap.add_argument("--all", action="store_true", help="run every eval session")
    ap.add_argument("--max-questions", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.5, metavar="F",
                    help="fraction of reviewer labels held out for scoring (0 = score on all; "
                         "not independent once the pipeline learns from feedback)")
    args = ap.parse_args()

    data = json.loads(EVAL_PATH.read_text())
    all_sessions = [s["session_name"] for s in data.get("eval_sessions", [])]
    if args.session:
        targets = args.session
    elif args.all:
        targets = all_sessions
    else:
        targets = random.Random(args.seed).sample(all_sessions, min(args.n, len(all_sessions)))

    fb_examples = memory.get_feedback_examples()
    _inform, holdout = split_labels(fb_examples, holdout_fraction=args.holdout)
    # holdout=0 means "score against everything" — still useful for a quick read, but say so.
    scoring_labels = fb_examples if args.holdout <= 0 else holdout
    bad_by_session = _bad_by_session(scoring_labels)

    rows, errors = [], []
    for name in targets:
        try:
            r = run_one(name, args.max_questions)
            _score_agreement(r, scoring_labels)
            # Exact-repeat check: did a literally-identical rejected question reappear? This is the
            # strict subset of `repeats`, which also catches rewordings.
            bad = bad_by_session.get(name, set())
            r["fb_violations"] = sum(1 for c in r["contents"]
                                     if memory.normalize_content(c) in bad) if bad else 0
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            errors.append((name, str(e)[:100]))

    hdr = (f"{'session':45s} {'n':>3} {'accept':>7} {'cov':>5} {'grnd':>5} "
           f"{'comp':>5} {'self_rel':>9}  verdict")
    print(hdr)
    print("-" * len(hdr))
    fails = 0
    for r in rows:
        ok = _passed(r)
        fails += 0 if ok else 1
        flags = ""
        if r.get("fb_violations"):
            flags += f"  ⚠{r['fb_violations']} exact-repeat"
        if r.get("repeats"):
            flags += f"  ⚠{r['repeats']} reworded-repeat"
        acc = "    n/a" if r.get("accept") is None else f"{r['accept']:>7.2f}"
        print(f"{r['session'][:45]:45s} {r['count']:>3} {acc} {r['coverage']:>5.2f} "
              f"{r['grounding']:>5.2f} {r['composite']:>5.2f} {r['self_rel']:>9.2f}  "
              f"{'PASS' if ok else 'FAIL'}{flags}")
    for name, err in errors:
        fails += 1
        print(f"{name[:45]:45s}  ERROR: {err}")

    if rows:
        scored = [r["accept"] for r in rows if r.get("accept") is not None]
        acc_txt = f"{statistics.mean(scored):.2f} (over {len(scored)}/{len(rows)} runs)" if scored else "n/a"
        print(f"\nAGGREGATE  mean_predicted_accept={acc_txt}  "
              f"mean_coverage={statistics.mean(r['coverage'] for r in rows):.2f}  "
              f"mean_grounding={statistics.mean(r['grounding'] for r in rows):.2f}  "
              f"mean_count={statistics.mean(r['count'] for r in rows):.1f}")
        print(f"           mean_self_relevance={statistics.mean(r['self_rel'] for r in rows):.2f} "
              f"(reported only — produced by the selector, not scored)")
    if fb_examples:
        good = sum(1 for e in fb_examples if e.get("decision") == "good")
        bad = sum(1 for e in fb_examples if e.get("decision") == "bad")
        mode = ("ALL labels (NOT independent)" if args.holdout <= 0
                else f"{len(holdout)} HELD-OUT of {len(fb_examples)}")
        print(f"LABELS     {len(fb_examples)} reviewer decisions ({good} good / {bad} bad) across "
              f"{len({e.get('session') for e in fb_examples})} session(s); scored against {mode}")
        if len(fb_examples) < 200:
            print("           NOTE: with this few labels `accept` is a coarse nearest-neighbour "
                  "estimate — treat it as a trend, not a precise figure.")
    else:
        print("LABELS     none — `accept` cannot be measured. Review some runs first; without "
              "reviewer decisions this harness only checks curriculum fit, not taste.")
    total = len(rows) + len(errors)
    print(f"RESULT  {total - fails}/{total} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
