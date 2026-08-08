#!/usr/bin/env python3
"""One-pass credibility cleaner for data/genai_question_bank.json.

Applies the same rules now baked into the harvest/runtime so the EXISTING bank is trustworthy:
  1. Drop rows from LOW-TRUST domains (reddit / quora / medium / dev.to).
  2. Strip scrape artifacts ("Q:" prefixes, stray backslashes) from the question text.
  3. Drop rows failing the FORM-quality gate (boilerplate / logistics / fragments / headings).
  4. Re-blank JUNK company attribution on web rows ("Fine-Tuning", "REST API", tech terms) so they
     fall back to their source-site label; leaves curated xlsx/seed companies untouched.
  5. De-duplicate by normalized text, preferring the company-bearing variant.

Writes the cleaned bank back (after a .bak backup) and prints a report. Read/rewrite only — no network.

Run:  python scripts/clean_bank.py --dry-run     # report only, writes nothing
      python scripts/clean_bank.py               # clean + rewrite (keeps a .bak)
      python scripts/clean_bank.py --show 20     # also print sample dropped rows per reason
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENAI_BANK_JSON, EDU_PLATFORM_DOMAINS  # noqa: E402
from src.sources.base import domain, split_into_clauses  # noqa: E402
from src.sources.tavily_search import _valid_company  # noqa: E402
from src.quality import is_quality_question, strip_artifacts  # noqa: E402

_LOW_TRUST = {"reddit.com", "quora.com", "medium.com", "dev.to"}


def _presplit(bank: list) -> tuple[list, int]:
    """Expand a compound question row ("X and difference between Y") into one ATOMIC row per clause,
    preserving all fields (company/difficulty/role/source/attribution). Only splits when EVERY resulting
    clause is a well-formed standalone question — otherwise the row is left intact (no fragment pollution).
    Returns (expanded_rows, num_rows_split)."""
    out, split_n = [], 0
    for q in bank:
        clauses = split_into_clauses(strip_artifacts(q.get("content", "")))
        if len(clauses) > 1 and all(is_quality_question(c) for c in clauses):
            split_n += 1
            for c in clauses:
                r = dict(q)
                r["content"] = c
                r["id"] = str(uuid.uuid4())
                out.append(r)
        else:
            out.append(q)
    return out, split_n


def _dom(url: str) -> str:
    d = domain(url or "")
    return "medium.com" if d.endswith(".medium.com") else d


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print up to N sample dropped rows per reason")
    args = ap.parse_args()

    if not GENAI_BANK_JSON.exists():
        print(f"ERROR: {GENAI_BANK_JSON} not found.")
        return 1
    bank = json.loads(GENAI_BANK_JSON.read_text(encoding="utf-8"))
    start = len(bank)

    # Pre-split compound questions into atomic rows BEFORE cleaning/dedup.
    bank, split_n = _presplit(bank)

    dropped_domain = dropped_form = reblanked = dropped_edu = 0
    kept: dict[str, dict] = {}   # norm-key → record (company-bearing preferred)
    dup_removed = 0
    # Reject report: reason → sample texts, so a filter change can be reviewed before it is trusted.
    samples: dict[str, list[str]] = defaultdict(list)

    def _note(reason: str, text: str):
        if len(samples[reason]) < max(args.show, 5):
            samples[reason].append((text or "")[:110])

    for q in bank:
        url = q.get("source_url") or ""
        d = _dom(url)
        # 1. low-trust domain
        if d in _LOW_TRUST:
            dropped_domain += 1
            _note("low-trust domain", q.get("content", ""))
            continue
        # 1b. education-platform CLASS content — keep only interview-question pages (URL has "interview").
        if d in EDU_PLATFORM_DOMAINS and "interview" not in url.lower():
            dropped_edu += 1
            _note("edu class-content", q.get("content", ""))
            continue
        # 2. strip artifacts (also drops "… | <Site> Interview Questions" SEO tails)
        content = strip_artifacts(q.get("content", ""))
        if content != q.get("content"):
            q["content"] = content
        # 3. form-quality gate
        if not is_quality_question(content):
            dropped_form += 1
            _note("form-quality", content)
            continue
        # 4. re-blank junk company on web rows only (curated xlsx/seed names are trusted as-is)
        if q.get("source") == "web" and q.get("company"):
            cleaned = _valid_company(q["company"])
            if cleaned != q["company"]:
                reblanked += 1
                _note("company re-blanked", f"{q['company']!r} → {cleaned!r}")
                q["company"] = cleaned  # None (→ source-site label) or a cleaned name
        # 5. dedup, preferring a company-bearing variant
        key = _norm(content)
        if not key:
            dropped_form += 1
            continue
        if key in kept:
            dup_removed += 1
            if not kept[key].get("company") and q.get("company"):
                kept[key] = q     # upgrade to the company-bearing variant
            continue
        kept[key] = q

    cleaned_bank = list(kept.values())

    if args.dry_run:
        print("DRY RUN — no files written.")
    else:
        # Keep the previous bank recoverable; this script is destructive by design.
        shutil.copy2(GENAI_BANK_JSON, GENAI_BANK_JSON.with_suffix(".json.bak"))
        GENAI_BANK_JSON.write_text(json.dumps(cleaned_bank, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"wrote {GENAI_BANK_JSON.name} (backup: {GENAI_BANK_JSON.name}.bak)")

    withco = sum(1 for q in cleaned_bank if q.get("company"))
    print(f"cleaned bank: {start} → {len(cleaned_bank)}")
    print(f"  compound rows pre-split:    {split_n}")
    print(f"  dropped (low-trust domain): {dropped_domain}")
    print(f"  dropped (edu class-content): {dropped_edu}")
    print(f"  dropped (form-quality):     {dropped_form}")
    print(f"  dropped (duplicates):       {dup_removed}")
    print(f"  companies re-blanked:       {reblanked}")
    # "unattributed", not "source-site": attribution no longer borrows the site name for a row with
    # no company (see models.attribution_label) — those rows surface as NIAT.
    print(f"  with real company: {withco}  | unattributed (NIAT): {len(cleaned_bank) - withco}")
    print(f"  sources: {dict(Counter(q.get('source') for q in cleaned_bank))}")
    print(f"  difficulty: {dict(Counter(q.get('difficulty') for q in cleaned_bank))}")

    if args.show:
        print("\nreject samples")
        for reason, rows in samples.items():
            print(f"  [{reason}]")
            for r in rows[:args.show]:
                print(f"    - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
