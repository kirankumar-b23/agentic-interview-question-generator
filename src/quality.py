"""Form-quality gate for harvested/retrieved questions.

The relevance gate judges TOPIC ("is this about the session?"). It does NOT judge FORM — so
boilerplate ("What to expect from your interview"), page headings ("How MCP Works Internally"),
comment fragments ("Also, how diverse is the sample?"), interview logistics ("How long is the
interview process?") and mid-sentence scraps can be topically on-topic yet are not real, usable
interview questions. `is_quality_question` rejects those; `strip_artifacts` cleans scrape residue
(leading "Q:"/"Q." markers, stray backslashes/markdown). Reused by the harvest, the Tavily
extractor, the runtime pre-filters, and the one-pass bank cleaner so noise cannot enter or survive.
"""
from __future__ import annotations

import re

from src.sources.base import looks_like_question, _Q_STARTS

# Leading scrape markers: "Q:", "Q.", "Q1:", "Ans:", stray backslashes, bullet/hash noise.
_LEAD_JUNK = re.compile(r"^\s*(?:[\\>#•\-*]+\s*)?(?:q(?:uestion)?\s*\d*\s*[.:)-]\s*|ans(?:wer)?\s*[.:)-]\s*)?",
                        re.IGNORECASE)
_BACKSLASH = re.compile(r"\\+")
_WS = re.compile(r"\s+")

# Mid-thought openers → the line is a fragment lifted out of context, not a standalone question.
_FRAGMENT_STARTS = {
    "also", "and", "but", "so", "or", "then", "plus", "additionally", "moreover",
    "however", "thus", "hence", "therefore", "besides", "yet", "because", "although",
}

# Boilerplate / meta / logistics / CTA — never a real interview question even if it ends in "?".
_REJECT_PATTERNS = [
    r"what to expect from",
    r"\bhow to (?:answer|prepare|ace|crack|approach|nail|pass)\b",
    r"in this (?:article|post|blog|guide|tutorial|section|video)",
    r"\b(?:contact us|let us know|reach out|email us|get in touch)\b",
    r"[\w.+-]+@[\w-]+\.[\w.]+",                     # e-mail address
    r"\b(?:click here|read more|learn more|subscribe|sign up|newsletter|follow us)\b",
    r"\bhow long (?:is|does|will|are)\b.*\b(?:interview|hiring|process|take|last)\b",
    r"\btable of contents\b",
    r"\b(?:we are|we're|now) hiring\b|\bapply now\b|\bjob description\b",
    r"\b(?:comment below|share this|upvote|please like|please share)\b",
    r"^\s*(?:overview|introduction|conclusion|summary|references?|resources?)\s*$",
]
_REJECT_RE = [re.compile(p, re.IGNORECASE) for p in _REJECT_PATTERNS]

_WH_STARTS = ("what", "how", "why", "when", "where", "which", "who", "whose", "whom")
# A short line is OK only if it opens like a real question/task ("What are embeddings?"); a short
# NOUN-PHRASE ending in "?" ("The code quality?") is a scrap. Reuse the harvest's question starters.
_SHORT_OK_STARTS = tuple(set(_Q_STARTS) | set(_WH_STARTS) | {"is", "are", "do", "does", "can", "should"})
_MIN_WORDS = 3          # hard floor
_SHORT_WORDS = 5        # below this, require a question/task opener


def strip_artifacts(text: str) -> str:
    """Remove scrape residue: leading Q:/Q./Ans: markers, backslashes, markdown noise, extra spaces."""
    if not text:
        return text
    s = _BACKSLASH.sub("", text)
    s = _LEAD_JUNK.sub("", s)
    s = s.strip(" \t`*_[]#>-")
    return _WS.sub(" ", s).strip()


def _looks_like_heading(t: str) -> bool:
    """A wh-word phrase with NO question mark that is Title-Cased reads as a page heading
    ("How MCP Works Internally", "What Are Embeddings" without a '?')."""
    if t.endswith("?"):
        return False
    if not t.lower().startswith(_WH_STARTS):
        return False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t)]
    if len(words) < 2:
        return False
    capped = sum(1 for w in words if w[:1].isupper())
    return capped / len(words) >= 0.6


def is_quality_question(text: str) -> bool:
    """True if `text` is a well-formed, standalone interview question/task prompt (form only —
    relevance is judged separately). Assumes/So callers may pass raw text; it strips artifacts first."""
    t = strip_artifacts(text)
    if not t:
        return False
    if not looks_like_question(t):            # length + ends-with-? or imperative/wh starter
        return False
    words = t.split()
    if len(words) < _MIN_WORDS:
        return False
    if words[0].strip(",.").lower() in _FRAGMENT_STARTS:   # "Also, how diverse…" → mid-thought scrap
        return False
    # Short lines must open like a question/task; a short noun phrase ("The code quality?") is a scrap.
    if len(words) < _SHORT_WORDS and not t.lower().startswith(_SHORT_OK_STARTS):
        return False
    if _looks_like_heading(t):                # "How MCP Works Internally" → heading, not a question
        return False
    if any(rx.search(t) for rx in _REJECT_RE):             # boilerplate / logistics / CTA
        return False
    return True
