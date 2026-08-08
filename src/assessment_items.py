"""Assessment-item detection for the retrieval corpora.

WHY THIS IS SEPARATE FROM THE FORM GATE
---------------------------------------
`quality.is_quality_question` deliberately does NOT reject MCQs — 61% of the 1,772 course-MCQ rows in
`data/curriculum/*.json` pass it ("Which of the following images represents the node used to send HTTP
requests…"), because they ARE well-formed questions. They are simply the wrong KIND of item: nobody is
asked to pick option C in an interview.

That was documented as "a data assertion, not a gate", enforced only by
`tests/test_data_integrity.py`. The assertion had a hole: `MCQ_SHAPES` carried tells for
"which of the following" / "all of the above" / "options:" but **none for lettered options**, so
`data/interview_questions.json` — the main retrieval bank — was carrying **52** of them with the test
green:

    "What is the right way to initialize an array? A) int num[6] = {2,4,12,5,45,5}; B) int n[] = …"
    "Identify the pure virtual function. A) virtual void func() = 1; B) void virtual func() = 0; …"

TWO SHAPES, TWO DIFFERENT REMEDIES
----------------------------------
Measured across both shipped banks (2,887 rows), a lettered marker means one of two things, and they
must not be treated alike:

* **≥2 distinct option letters ⇒ a real MCQ.** Delete it; the question is inseparable from its option
  list. 52 rows in `interview_questions.json`, 0 in the GenAI bank.
* **exactly one marker ⇒ an ANSWER glued onto the question**, not an option list. 6 rows, e.g.
  `"What is a Large Language Model (LLM)?A. Think of LLMs as massive neural networks trained on…"`.
  These are genuine, good interview questions with scrape residue; deleting them throws away real
  content. **Repair by truncating at the marker** — the 6 become clean 7-10 word questions.

Requiring TWO letters for the delete rule is what protects a legitimate parenthetical: a coding
problem reading "the fewest number of digits (d) and base number (m)" matches one marker and must
survive. One letter is never enough evidence to delete.

The word ceiling in `quality.py` is NOT a substitute for the repair pass: of those 6 glued rows only 2
exceed 40 words, so the ceiling would have missed 3 and *deleted* the other 2 rather than fixing them.
"""
from __future__ import annotations

import re

# A lettered option/answer marker: "A) ", "b. ", "(C) " — at a word start, never mid-token.
_LETTER_MARKER = re.compile(r"(?:^|[\s(])([A-Da-d])[\).]\s")

# A marker sitting immediately after the question mark is an option/answer list opening:
# "…in OOP? A. Overriding a parent class method". Distinct from a mid-sentence "(d)".
_GLUED_ANSWER = re.compile(r"\?\s*(?:[A-Da-d][\).]|Answers?\s*[:\-]|Ans\s*[:\-])\s", re.IGNORECASE)


def _option_letters(text: str) -> set:
    return {m.group(1).lower() for m in _LETTER_MARKER.finditer(text or "")}


def is_assessment_item(text: str) -> bool:
    """True for a multiple-choice item, which must never enter the retrieval corpus.

    Requires ≥2 DISTINCT option letters. One marker alone is a glued answer or a legitimate
    parenthetical — see `strip_glued_answer`.
    """
    return len(_option_letters(text)) >= 2


def strip_glued_answer(text: str) -> str:
    """Return just the question when an answer has been concatenated onto it, else the text unchanged.

    `"What's RLHF, and why does it matter?A. RLHF (Reinforcement Learning from Human Feedback) trains…"`
    → `"What's RLHF, and why does it matter?"`

    Only fires on a single-marker row: with ≥2 option letters the row is an MCQ and truncating it would
    manufacture a bare stem out of an assessment item, hiding it from `is_assessment_item` instead of
    removing it.
    """
    t = (text or "").strip()
    if not t or is_assessment_item(t):
        return t
    m = _GLUED_ANSWER.search(t)
    if not m:
        return t
    # Keep the question mark itself; the marker starts right after it.
    return t[: t.index("?", m.start()) + 1].strip()
