"""Is this question answerable in a CONVERSATIONAL interview?

Students answer these out loud in a mock interview — no keyboard, no IDE, no whiteboard. A question that
demands a produced artifact is unanswerable by construction, so it burns a slot and makes the set look
worse than it is. Real runs shipped exactly that:

    "Implement an input box to interact with the Gemini API. Send user input to the specified API…"
    "Build and integrate LLM applications."

WHY THIS IS NOT IN THE FORM GATE
--------------------------------
`quality.is_quality_question` answers "is this a usable, well-formed question?" — and a hands-on task IS
one. This module answers a different question: "does it suit our interview FORMAT". Two concrete reasons
they must stay apart:

* Conflating shape with suitability is how `MCQ_SHAPES` ended up as the only thing standing between
  course multiple-choice items and the retrieval bank (see `assessment_items`).
* `scripts/clean_bank.py` deletes every row that fails the form gate. Putting this rule there would
  permanently destroy **217 real, company-attributed coding questions** — and the LMS unit-import format
  the export targets has coding tabs. This is reversible POLICY (`config.CONVERSATIONAL_ONLY`), not a
  data defect, so it filters at runtime and leaves the corpus intact.

"DESIGN" IS DELIBERATELY NOT HANDS-ON
------------------------------------
It is the one do-verb that is answerable in conversation: *"Design a news aggregator system"* means "talk
me through the architecture", which is exactly what a conversational interview is for. Measured, treating
it as hands-on is actively harmful — across the last 6 runs the broad rule pushed **2 of 6 under the
5-question minimum** (6→4 and 5→4) and it dropped *"Design an RSS News Feed Service"*, one of only three
tool-specific questions the n8n retrieval work recovered.

The compound `"design and implement"` IS caught: it reads like discussion and demands an artifact.

THE DISCRIMINATOR — three parts, all load-bearing
-------------------------------------------------
1. A **PRODUCE verb**. Producing an artifact, not discussing one.
2. In a **DIRECTIVE frame** — a bare imperative, or an explicit request to do it now ("Can you write…",
   "Your task is to build…"). The frame matters because the same verb is fine inside a question ABOUT
   doing the thing.
3. A **wh-opener exemption**, which is what separates the two classes that share a verb:

       "How did you implement JWT authentication in your project?"   KEEP — about past work
       "Implement JWT authentication"                                SKIP — do it now
       "How do you implement authentication in a web application?"   KEEP — general practice

Measured on the 2,828 shipped rows: rejects **217 (7.7%)**. A 14-row random sample of the rejects was
audited and every one was genuinely keyboard work. No run in the last 6 falls under the minimum because
of this rule; the worst single loss was 9 → 7. A shortfall it does create is handled by
`pipeline._top_up_from_open_web`, which already triggers below 60% of the requested count.
"""
from __future__ import annotations

import re

# Verbs that demand a PRODUCED artifact. "design" is absent on purpose — see the module docstring.
_PRODUCE = (r"(?:write|code|implement|program|refactor|debug|build|create|integrate|deploy)")

# A directive: "do this now". Bare imperative, or an explicit request. The final alternative catches
# "design and implement", which no bare-verb rule would.
_DIRECTIVE = re.compile(
    r"^(?:now[,\s]+|please\s+|next[,\s]+|first[,\s]+)?" + _PRODUCE + r"\b"
    r"|^(?:can|could|would)\s+you\s+(?:please\s+)?" + _PRODUCE + r"\b"
    r"|\byour\s+task\s+is\s+to\s+" + _PRODUCE + r"\b"
    r"|\bi\s+(?:want|need)\s+you\s+to\s+" + _PRODUCE + r"\b"
    r"|\bdesign\s+and\s+(?:then\s+)?implement\b",
    re.IGNORECASE,
)

# An interrogative opener means the question is ABOUT the work, not a demand to perform it. This single
# exemption is what keeps "How did you implement JWT…?" — a strong conversational question about a
# candidate's own project — out of the reject pile.
_ABOUT = re.compile(r"^(?:how|why|what|when|which|where)\b", re.IGNORECASE)


def is_hands_on_task(text: str) -> bool:
    """True when answering would require producing code or another artifact.

    Fail-open on empty input: nothing to judge is not a hands-on task.
    """
    t = (text or "").strip()
    if not t or _ABOUT.match(t):
        return False
    return bool(_DIRECTIVE.search(t))
