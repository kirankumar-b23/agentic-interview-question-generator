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
from src.config import (MAX_QUESTIONS, MIN_QUESTIONS, SESSION_TYPES,  # noqa: E402
                        eval_thresholds, normalize_session_type)
from src.data_loader import get_data_store                   # noqa: E402
from src.session_types import type_for_run                   # noqa: E402
from src.human_agreement import predict_accept, split_labels   # noqa: E402
from src import memory                                 # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_sets.json"
# A run PASSES on independent signals only: it must look acceptable to the reviewer (measured against
# HELD-OUT decisions of the SAME session type), cover the session's outcomes, and be the right size.
# `self_rel` is excluded on purpose — see the module docstring.
#
# The bars are PER TYPE (config.EVAL_THRESHOLDS_BY_TYPE). A code-heavy session is scored against banks
# holding almost no implementation questions, so it scores lower on coverage and grounding for reasons
# about SOURCE COVERAGE rather than question quality. One global bar either failed every code session
# or set theory's bar too low.


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
        # The type the PIPELINE resolved for this run, from the reading material. eval_sets.json also
        # carries a session_type, but it is a stale copy of a title-substring heuristic and disagrees
        # with this on 9 of the 22 sessions where both exist. Reading it from the run means the eval
        # measures the type the pipeline actually acted on, and the two cannot drift apart.
        "session_type": normalize_session_type(res.context.session_type if res.context else None),
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
    """Attach predicted reviewer acceptance, from HELD-OUT decisions of the SAME session type.

    `allow_pooled=False` is the point: a type with too few labels gets `None`, not a number borrowed
    from another type's taste. Scoring an implementation question against theory-session decisions
    reliably marks it "too specific, not conceptual" — the pooled number would look measured and be
    wrong. Reported as "n/a" with the reason instead.
    """
    report = predict_accept(r["contents"], holdout, session=r["session"],
                            session_type=r["session_type"], allow_pooled=False)
    r["accept"] = report.predicted_accept_rate if report else None
    r["accept_labels"] = report.label_count if report else 0
    r["accept_pool"] = report.label_pool if report else None
    r["repeats"] = report.repeats_rejected if report else 0


def _passed(r: dict) -> bool:
    """Pass/fail against the bars for THIS session's type."""
    t = eval_thresholds(r["session_type"])
    if not (MIN_QUESTIONS <= r["count"] <= MAX_QUESTIONS):
        return False
    if r["coverage"] < t["coverage"]:
        return False
    if r["grounding"] < t["grounding"]:
        return False
    # Unmeasurable acceptance must not silently pass the run on the strength of coverage alone —
    # but it must not fail it either, since the gap is missing labels, not a bad question set.
    if r.get("accept") is not None and r["accept"] < t["accept"]:
        return False
    return True


def _stratified_sample(pool: list[str], types: dict[str, str], n: int, seed: int) -> list[str]:
    """Sample across session types instead of uniformly over a type-skewed name list.

    The eval name list is 85% code-heavy (an artefact of a course whose reading material isn't loaded),
    so uniform sampling gave `--n 3` two or three code-heavy sessions and almost never exercised the
    theory path. Round-robin across the types present gives every type a turn first.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for name in pool:
        by_type.setdefault(types.get(name, "mixed"), []).append(name)
    for names in by_type.values():
        rng.shuffle(names)

    out: list[str] = []
    order = sorted(by_type)                      # deterministic type order for a given seed
    while len(out) < min(n, len(pool)):
        progressed = False
        for t in order:
            if len(out) >= n:
                break
            if by_type[t]:
                out.append(by_type[t].pop())
                progressed = True
        if not progressed:
            break
    return out


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
    ap.add_argument("--type", dest="session_type", choices=SESSION_TYPES,
                    help="only run sessions of this type (resolved from reading material)")
    ap.add_argument("--include-unrunnable", action="store_true",
                    help="also run sessions with no reading material (they exercise the fallback "
                         "resolution path, not the real curriculum path)")
    args = ap.parse_args()

    data = json.loads(EVAL_PATH.read_text())
    # NOTE: only the session NAME is taken from eval_sets.json. Its `session_type` is a stale copy of a
    # title-substring heuristic; the authoritative type comes from the run (see run_one).
    all_sessions = [s["session_name"] for s in data.get("eval_sessions", [])]

    # Quarantine sessions with no reading material. Running them scores the knowledge-graph/LLM
    # fallback path and reports it as a curriculum result — 24 of the 46 eval names are in this state,
    # and 21 of those are the code-heavy ones, so silently including them skewed everything.
    store = get_data_store()
    runnable = [n for n in all_sessions if store.get_session_content(n)]
    skipped = [n for n in all_sessions if n not in set(runnable)]
    pool = all_sessions if args.include_unrunnable else runnable

    # Pre-resolve types so `--type` and stratified sampling work before anything is run. This uses the
    # name-based resolver (LLM-derived review file, then the knowledge graph); the RUN then reports the
    # type the pipeline actually resolved, and a disagreement is printed rather than hidden.
    predicted_type = {n: (type_for_run(n) or "mixed") for n in pool}
    if args.session_type:
        pool = [n for n in pool if predicted_type[n] == args.session_type]

    if args.session:
        targets = args.session
    elif args.all:
        targets = pool
    else:
        targets = _stratified_sample(pool, predicted_type, args.n, args.seed)

    if skipped and not args.include_unrunnable:
        print(f"SKIPPED    {len(skipped)} session(s) with no reading material "
              f"(use --include-unrunnable to run them anyway)")
    if not targets:
        print("No runnable sessions match those filters.")
        return 1

    fb_examples = memory.get_feedback_examples()
    _inform, holdout = split_labels(fb_examples, holdout_fraction=args.holdout)
    # holdout=0 means "score against everything" — still useful for a quick read, but say so.
    scoring_labels = fb_examples if args.holdout <= 0 else holdout
    bad_by_session = _bad_by_session(scoring_labels)

    rows, errors = [], []
    for name in targets:
        try:
            r = run_one(name, args.max_questions)
            # The pipeline's own resolution is authoritative. If the name-based prediction disagreed,
            # flag it — that disagreement is exactly how the stale eval labels went unnoticed.
            listed = predicted_type.get(name)
            if listed and listed != r["session_type"]:
                r["type_mismatch"] = listed
            _score_agreement(r, scoring_labels)
            # Exact-repeat check: did a literally-identical rejected question reappear? This is the
            # strict subset of `repeats`, which also catches rewordings.
            bad = bad_by_session.get(name, set())
            r["fb_violations"] = sum(1 for c in r["contents"]
                                     if memory.normalize_content(c) in bad) if bad else 0
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            errors.append((name, str(e)[:100]))

    # ── Per-type tables. Each states its own n, because with the unrunnable sessions quarantined
    # there are only a handful of sessions per type and a two-session mean is not a trend.
    fails = 0
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["session_type"], []).append(r)

    for stype in sorted(by_type):
        group = by_type[stype]
        t = eval_thresholds(stype)
        print(f"\n{stype.upper()}  (n={len(group)})   bars: coverage ≥{t['coverage']:.2f} · "
              f"grounding ≥{t['grounding']:.2f} · accept ≥{t['accept']:.2f}")
        hdr = (f"  {'session':43s} {'n':>3} {'accept':>7} {'cov':>5} {'grnd':>5} "
               f"{'comp':>5} {'self_rel':>9}  verdict")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in group:
            ok = _passed(r)
            fails += 0 if ok else 1
            flags = ""
            if r.get("fb_violations"):
                flags += f"  ⚠{r['fb_violations']} exact-repeat"
            if r.get("repeats"):
                flags += f"  ⚠{r['repeats']} reworded-repeat"
            # A mismatch between the pre-resolved type and the run's own is worth seeing, not hiding.
            if r.get("type_mismatch"):
                flags += f"  ⚠listed as {r['type_mismatch']}"
            acc = "    n/a" if r.get("accept") is None else f"{r['accept']:>7.2f}"
            print(f"  {r['session'][:43]:43s} {r['count']:>3} {acc} {r['coverage']:>5.2f} "
                  f"{r['grounding']:>5.2f} {r['composite']:>5.2f} {r['self_rel']:>9.2f}  "
                  f"{'PASS' if ok else 'FAIL'}{flags}")

        scored = [r["accept"] for r in group if r.get("accept") is not None]
        if scored:
            pools = {r.get("accept_pool") for r in group if r.get("accept_pool")}
            print(f"    mean_accept={statistics.mean(scored):.2f} over {len(scored)}/{len(group)} "
                  f"(labels from {', '.join(sorted(pools))})")
        else:
            print(f"    mean_accept=n/a — no held-out reviewer decisions for {stype} sessions yet, "
                  f"so acceptance is unmeasured for this type (not assumed good, and NOT scored "
                  f"against another type's decisions)")
        print(f"    mean_coverage={statistics.mean(r['coverage'] for r in group):.2f}  "
              f"mean_grounding={statistics.mean(r['grounding'] for r in group):.2f}  "
              f"mean_count={statistics.mean(r['count'] for r in group):.1f}")

    for name, err in errors:
        fails += 1
        print(f"  {name[:43]:43s}  ERROR: {err}")

    if rows:
        print(f"\nALL TYPES  n={len(rows)}  "
              f"mean_coverage={statistics.mean(r['coverage'] for r in rows):.2f}  "
              f"mean_grounding={statistics.mean(r['grounding'] for r in rows):.2f}  "
              f"mean_count={statistics.mean(r['count'] for r in rows):.1f}")
        print(f"           mean_self_relevance={statistics.mean(r['self_rel'] for r in rows):.2f} "
              f"(reported only — produced by the selector, not scored)")
        print("           NOTE: cross-type means are for orientation only. The per-type tables above "
              "are the verdict, since the two types are scored against different bars.")
    if fb_examples:
        good = sum(1 for e in fb_examples if e.get("decision") == "good")
        bad = sum(1 for e in fb_examples if e.get("decision") == "bad")
        mode = ("ALL labels (NOT independent)" if args.holdout <= 0
                else f"{len(holdout)} HELD-OUT of {len(fb_examples)}")
        print(f"LABELS     {len(fb_examples)} reviewer decisions ({good} good / {bad} bad) across "
              f"{len({e.get('session') for e in fb_examples})} session(s); scored against {mode}")
        # Per-type label inventory. This is the number to watch: a type with a one-sided or empty pool
        # cannot have its acceptance measured at all, and that fact should be visible every run rather
        # than inferred from an "n/a" in a table.
        per_type: dict[str, dict[str, int]] = {}
        for e in fb_examples:
            if not (e.get("question") or "").strip():
                continue
            t = type_for_run(e.get("session") or "") or "unknown"
            per_type.setdefault(t, {"good": 0, "bad": 0})[
                "good" if e.get("decision") == "good" else "bad"] += 1
        for t in sorted(per_type):
            g, b = per_type[t]["good"], per_type[t]["bad"]
            usable = "measurable" if g and b else "NOT measurable (one-sided)"
            print(f"           {t:14s} {g:>3} good / {b:>3} bad — {usable}")
        for t in SESSION_TYPES:
            if t not in per_type:
                print(f"           {t:14s}   0 good /   0 bad — NOT measurable (no decisions yet); "
                      f"review one {t} run to start closing this")
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
