#!/usr/bin/env python3
"""One-pass credibility cleaner for data/genai_question_bank.json.

Applies the same rules now baked into the harvest/runtime so the EXISTING bank is trustworthy:
  1. Drop rows from LOW-TRUST domains (reddit / quora / medium / dev.to).
  2. Strip scrape artifacts ("Q:" prefixes, stray backslashes) from the question text.
  3. Drop rows failing the FORM-quality gate (boilerplate / logistics / fragments / headings).
  4. Re-blank JUNK company attribution on web rows ("Fine-Tuning", "REST API", tech terms) so they
     fall back to their source-site label; leaves curated xlsx/seed companies untouched.
  5. De-duplicate by normalized text, preferring the company-bearing variant.

Writes the cleaned bank back and prints a report. Read/rewrite only — no network.
Run:  python scripts/clean_bank.py
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENAI_BANK_JSON              # noqa: E402
from src.sources.base import domain                 # noqa: E402
from src.sources.tavily_search import _valid_company  # noqa: E402
from src.quality import is_quality_question, strip_artifacts  # noqa: E402

_LOW_TRUST = {"reddit.com", "quora.com", "medium.com", "dev.to"}


def _dom(url: str) -> str:
    d = domain(url or "")
    return "medium.com" if d.endswith(".medium.com") else d


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def main() -> int:
    if not GENAI_BANK_JSON.exists():
        print(f"ERROR: {GENAI_BANK_JSON} not found.")
        return 1
    bank = json.loads(GENAI_BANK_JSON.read_text(encoding="utf-8"))
    start = len(bank)

    dropped_domain = dropped_form = reblanked = 0
    kept: dict[str, dict] = {}   # norm-key → record (company-bearing preferred)
    dup_removed = 0

    for q in bank:
        # 1. low-trust domain
        if _dom(q.get("source_url")) in _LOW_TRUST:
            dropped_domain += 1
            continue
        # 2. strip artifacts
        content = strip_artifacts(q.get("content", ""))
        if content != q.get("content"):
            q["content"] = content
        # 3. form-quality gate
        if not is_quality_question(content):
            dropped_form += 1
            continue
        # 4. re-blank junk company on web rows only (curated xlsx/seed names are trusted as-is)
        if q.get("source") == "web" and q.get("company"):
            cleaned = _valid_company(q["company"])
            if cleaned != q["company"]:
                reblanked += 1
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
    GENAI_BANK_JSON.write_text(json.dumps(cleaned_bank, indent=2, ensure_ascii=False), encoding="utf-8")

    withco = sum(1 for q in cleaned_bank if q.get("company"))
    print(f"cleaned bank: {start} → {len(cleaned_bank)}")
    print(f"  dropped (low-trust domain): {dropped_domain}")
    print(f"  dropped (form-quality):     {dropped_form}")
    print(f"  dropped (duplicates):       {dup_removed}")
    print(f"  companies re-blanked:       {reblanked}")
    print(f"  with real company: {withco}  | source-site/NIAT: {len(cleaned_bank) - withco}")
    print(f"  sources: {dict(Counter(q.get('source') for q in cleaned_bank))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
