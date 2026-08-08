#!/usr/bin/env python3
"""Remove multiple-choice items from the retrieval corpora, and repair glued answers.

WHY
---
CLAUDE.md states that assessment items must never enter the retrieval corpus, enforced by
`tests/test_data_integrity.py`. That assertion had no tell for LETTERED options, so
`data/interview_questions.json` was carrying 52 MCQs with the test green — aptitude items and
C/Java syntax questions, none company-attributed:

    "What is the right way to initialize an array? A) int num[6] = {2,4,12,5,45,5}; B) int n[] = …"

The same sweep found 6 rows that are NOT MCQs but real questions with the ANSWER concatenated on
("What's RLHF, and why does it matter?A. RLHF (Reinforcement Learning from Human Feedback) trains…").
Those are repaired, not deleted — see `src/assessment_items` for why the two shapes need different
remedies and why the delete rule needs TWO option letters.

Run:  python scripts/strip_assessment_items.py --dry-run   # report only, writes nothing
      python scripts/strip_assessment_items.py             # rewrite both banks (keeps a .bak each)

Read/rewrite only — no network, no LLM. Clear `.cache/` afterwards: the corpus matrix is cached by
content digest, so a stale cache keeps serving the removed rows.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assessment_items import is_assessment_item, strip_glued_answer  # noqa: E402
from src.config import GENAI_BANK_JSON, INTERVIEW_QUESTIONS_JSON  # noqa: E402
from src.quality import is_quality_question  # noqa: E402

# (path, key holding the row list — None when the file IS the list)
BANKS = [(INTERVIEW_QUESTIONS_JSON, "questions"), (GENAI_BANK_JSON, None)]


def _sweep(rows: list) -> tuple[list, list, list]:
    """Return (kept, deleted, repaired) where `repaired` holds (before, after) pairs."""
    kept, deleted, repaired = [], [], []
    for row in rows:
        text = (row.get("content") or "").strip()
        if is_assessment_item(text):
            deleted.append(text)
            continue
        fixed = strip_glued_answer(text)
        if fixed != text:
            # A repair that yields something the form gate rejects is not a repair — keep the row out
            # rather than shipping a stem like "And?" into the bank.
            if not is_quality_question(fixed):
                deleted.append(text)
                continue
            repaired.append((text, fixed))
            row = {**row, "content": fixed}
        kept.append(row)
    return kept, deleted, repaired


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--show", type=int, default=3, help="sample rows to print per bucket")
    args = ap.parse_args()

    total_del = total_rep = 0
    for path, key in BANKS:
        if not path.exists():
            print(f"{path.name}: not present, skipped")
            continue
        data = json.loads(path.read_text())
        rows = data[key] if key else data
        kept, deleted, repaired = _sweep(rows)
        total_del += len(deleted)
        total_rep += len(repaired)

        print(f"\n{path.name}: {len(rows)} rows → {len(kept)} kept "
              f"({len(deleted)} MCQ deleted, {len(repaired)} glued answer repaired)")
        for t in deleted[: args.show]:
            print(f"   DELETE  {t[:96]}")
        for before, after in repaired[: args.show]:
            print(f"   REPAIR  {before[:60]}…\n        → {after[:96]}")

        if args.dry_run or (not deleted and not repaired):
            continue
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        if key:
            data[key] = kept
            if isinstance(data.get("metadata"), dict) and "total" in data["metadata"]:
                data["metadata"]["total"] = len(kept)
        else:
            data = kept
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"   wrote {path.name} (backup: {backup.name})")

    print(f"\nTotal: {total_del} deleted, {total_rep} repaired."
          + ("  [dry run — nothing written]" if args.dry_run else "  Now clear .cache/."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
