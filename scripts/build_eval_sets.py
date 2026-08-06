"""Build comprehensive eval sets for all 45 sessions.

Preserves existing hand-written evals, generates lightweight evals
for remaining sessions from knowledge graph KP labels.
Adds coding question evals (good + bad) + format rules.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
EVAL_PATH = PROJECT_ROOT / "eval" / "eval_sets.json"


def main():
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_map = {s["session_name"]: s for s in existing["eval_sessions"]}
    kps = kg["knowledge_points"]

    all_sessions = []

    for name, info in sorted(kg["sessions"].items()):
        course = info.get("course", "unknown")
        stype = info.get("session_type", "mixed")
        kp_ids = info.get("kp_ids", [])
        outcomes = info.get("learning_outcomes", [])

        kp_labels = [kps[k]["label"] for k in kp_ids if k in kps]
        if not outcomes and kp_labels:
            outcomes = [f"Understand {l}" for l in kp_labels[:4]]

        session = {
            "session_name": name,
            "session_type": stype,
            "course": course,
            "kp_ids": kp_ids,
            "expected_outcomes": outcomes,
        }

        # ── Preserve existing hand-written evals ──
        if name in existing_map:
            ex = existing_map[name]
            old_good = ex.get("good_questions", [])
            old_bad = ex.get("bad_questions", [])

            if isinstance(old_good, dict):
                session["good_questions"] = old_good
            else:
                session["good_questions"] = {"theory": old_good, "coding": []}

            if isinstance(old_bad, dict):
                session["bad_questions"] = old_bad
            else:
                session["bad_questions"] = {"theory": old_bad, "coding": []}
        else:
            # ── Generate theory evals from KP labels ──
            good_theory = []
            diffs = ["Easy", "Medium", "Hard"]
            for i, label in enumerate(kp_labels[:3]):
                good_theory.append({
                    "content": f"Explain {label} and its practical applications.",
                    "difficulty": diffs[i % 3],
                    "why_good": f"Directly tests KP: {label}",
                })

            bad_theory = [
                {
                    "content": "What is programming?",
                    "bad_type": "too_generic",
                    "why_bad": "Too generic, not session-specific",
                },
            ]

            other_sessions = [n for n in kg["sessions"] if n != name]
            if other_sessions:
                other = other_sessions[len(name) % len(other_sessions)]
                bad_theory.append({
                    "content": f"Explain concepts from {other}.",
                    "bad_type": "off_topic",
                    "why_bad": f"About {other}, not {name}",
                })

            session["good_questions"] = {"theory": good_theory, "coding": []}
            session["bad_questions"] = {"theory": bad_theory, "coding": []}

        # ── Coding evals based on session type ──
        if stype in ("code_heavy", "mixed"):
            if not session["good_questions"].get("coding"):
                good_coding = []
                for label in kp_labels[:2]:
                    fn_name = label.lower().replace(" ", "_").replace("-", "_")[:30]
                    fn_name = "".join(c for c in fn_name if c.isalnum() or c == "_")
                    good_coding.append({
                        "title": label[:40] + " Implementation",
                        "content": f"Write a function that demonstrates {label}. The function should accept appropriate parameters and return a meaningful result.",
                        "starter_code": f"def {fn_name}():\n    # Write your code here\n    pass",
                        "difficulty": "Medium",
                        "why_good": f"Tests {label}, concise content, proper starter code",
                    })
                session["good_questions"]["coding"] = good_coding

            if not session["bad_questions"].get("coding"):
                session["bad_questions"]["coding"] = [
                    {
                        "title": "Generic Sort",
                        "content": "## Task\nImplement a sorting algorithm.\n## Input\n- A list of integers",
                        "bad_type": "wrong_format",
                        "why_bad": "Uses markdown headers instead of plain text. Generic DSA, not session-specific",
                    }
                ]

        elif stype == "theory_heavy":
            if not session["good_questions"].get("coding"):
                session["good_questions"]["coding"] = []
            if not session["bad_questions"].get("coding"):
                session["bad_questions"]["coding"] = [
                    {
                        "title": "Code Implementation",
                        "content": "Write code to implement concepts from this session.",
                        "bad_type": "wrong_session",
                        "why_bad": f"Theory-heavy session should not have coding questions",
                    }
                ]

        all_sessions.append(session)

    # ── Add any existing eval sessions NOT in knowledge graph ──
    kg_names = set(kg["sessions"].keys())
    for ex_name, ex_data in existing_map.items():
        if ex_name not in kg_names:
            old_good = ex_data.get("good_questions", [])
            old_bad = ex_data.get("bad_questions", [])

            session = {
                "session_name": ex_name,
                "session_type": ex_data.get("session_type", "mixed"),
                "course": ex_data.get("course", "unknown"),
                "kp_ids": [],
                "expected_outcomes": ex_data.get("expected_outcomes", []),
            }

            if isinstance(old_good, dict):
                session["good_questions"] = old_good
            else:
                session["good_questions"] = {"theory": old_good, "coding": []}

            if isinstance(old_bad, dict):
                session["bad_questions"] = old_bad
            else:
                session["bad_questions"] = {"theory": old_bad, "coding": []}

            all_sessions.append(session)
            print(f"  Preserved orphan eval: {ex_name}")

    # ── Emit ONLY what is consumed ──
    #
    # This script used to write 342 `good_questions`/`bad_questions` exemplars plus a `format_rules`
    # block. Nothing ever read any of it, and the exemplars were machine-templated:
    #   "Explain {KP_LABEL} and its practical applications."   with   why_good: "Directly tests KP: …"
    #   "What is programming?"  (verbatim, 41 times)
    #   an off-topic distractor chosen by  other_sessions[len(name) % len(other_sessions)]
    # 342 authoritative-looking fake questions invite someone to calibrate on them — which is exactly
    # the mistake per-type calibration is meant to avoid. The real reviewer decisions live in
    # eval/feedback_examples.json and are the only labels any code should use.
    #
    # `session_type` is also dropped. It was a pass-through of the knowledge graph's title-substring
    # heuristic, and it disagreed with the pipeline's own (reading-material) resolution on 9 of the 22
    # sessions where both existed. eval/run_eval.py now resolves the type per run instead, so the two
    # cannot drift apart. See src/session_types.py.
    for session in all_sessions:
        for dead in ("good_questions", "bad_questions", "session_type"):
            session.pop(dead, None)

    output = {
        "metadata": {
            "version": "4.0",
            "built_at": datetime.now().strftime("%Y-%m-%d"),
            "description": ("Eval session list: names + KPs + expected outcomes. Question exemplars "
                            "and session_type were removed in v4 — see the comment in "
                            "scripts/build_eval_sets.py. Reviewer labels: eval/feedback_examples.json."),
            "total_sessions": len(all_sessions),
        },
        "eval_sessions": all_sessions,
    }

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Eval sets for {len(all_sessions)} sessions:")
    print(f"  Good theory: {tgt}")
    print(f"  Bad theory:  {tbt}")
    print(f"  Good coding: {tgc}")
    print(f"  Bad coding:  {tbc}")
    print(f"  Total:       {tgt + tbt + tgc + tbc}")


if __name__ == "__main__":
    main()
