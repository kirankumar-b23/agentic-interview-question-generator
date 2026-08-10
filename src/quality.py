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

# Leading scrape markers: "Q:", "Q.", "Q1:", "Q1)", "Ans:", stray backslashes, bullet/hash noise.
# The marker's separator must be followed by whitespace so a real term like "Q-learning" / "Q-value"
# (hyphen followed by a letter, no space) is NOT mistaken for a "Q." scrape marker and truncated.
_LEAD_JUNK = re.compile(r"^\s*(?:[\\>#•\-*]+\s*)?(?:q(?:uestion)?\s*\d*\s*[.:)-]\s+|ans(?:wer)?\s*[.:)-]\s+)?",
                        re.IGNORECASE)
_BACKSLASH = re.compile(r"\\+")
_WS = re.compile(r"\s+")
_PARENS = re.compile(r"\([^)]*\)")

# Trailing site/brand/SEO tail on a scraped page title:
#   "Build an Enterprise RAG Workflow | Dataford Interview Questions"
#   "What is RAG? - GeeksforGeeks"
# Only stripped when the tail after the separator is short AND reads like site furniture, so a real
# question containing a dash or pipe ("Fine-Tuning vs RAG - which to use?") is left alone.
_SITE_TAIL = re.compile(
    r"\s*[|»·–—-]\s*"
    r"(?=[^|»·–—]{0,60}$)"                                  # tail must be short
    r"[^|»·–—]*\b(?:interview|questions?|answers?|guide|tutorials?|blog|examples?|"
    r"cheat\s*sheet|faqs?|geeksforgeeks|javatpoint|simplilearn|interviewbit|"
    r"dataford|scaler|edureka|intellipaat)\b[^|»·–—]*$",
    re.IGNORECASE,
)

# Mid-thought openers → the line is a fragment lifted out of context, not a standalone question.
# Conservative: only openers that rarely begin a real standalone question (NOT when/if/given/while — those
# begin many legitimate questions like "When should you fine-tune?").
_FRAGMENT_STARTS = {
    "also", "and", "but", "so", "or", "then", "plus", "additionally", "moreover",
    "however", "thus", "hence", "therefore", "besides", "yet",
}

# Listicle/heading openers → a document heading ("List of Key LLM Parameters", "Types of ..."), not a
# question. Applied ONLY when the text does NOT end with "?" (so "Key differences …?" is exempt).
_HEADING_STARTS = ("list of ", "types of ", "examples of ", "overview of ", "introduction to ",
                   "top ", "key ", "benefits of ", "advantages of ", "pros and cons")

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
    # Editorial/clickbait comparison headline, not an interview question:
    #   "Is GPT Image 2 the Best Image Generation Model?"
    r"^\s*(?:is|are|was|were)\b.{0,60}\bthe\s+(?:best|worst|top|greatest|future|right|only)\b",
    # Interviewer rating an individual's own project — not reusable across candidates:
    #   "On a scale of 1 to 10, how accurate is your AI …"
    r"\bon a scale of\s*\d+\s*(?:to|-|–)\s*\d+",
    # Pronoun-subject fragment lifted out of context — the antecedent is in the page, not the line:
    #   "how does it affect generation?", "why is it important?", "how can they be reduced?"
    r"^\s*(?:how|what|why|when|where)\s+(?:do|does|did|is|are|was|were|will|would|can|could)\s+"
    r"(?:it|they|this|that|these|those)\b",
    # Truncated scrape of a lesson heading: "What happens during generation" — a bare
    # "what happens <prep> <one noun>" with nothing else is not an answerable question.
    r"^\s*what happens\s+(?:during|in|at|on|with|after|before)\s+[\w-]+\s*\??\s*$",
    # POSSESSIVE-reference fragment: "What are its key components?", "What are their advantages?"
    # The thing being described is named earlier on the page, not in the question. Anchored to the
    # opening so a mid-sentence possessive with a real antecedent survives
    # ("What is its purpose in the Adam optimizer?").
    r"^\s*(?:what|how|why|when|where)\s+(?:is|are|was|were|do|does|did)\s+(?:its|their|his|her)\b",
]

# DEMONSTRATIVE-reference fragment: the question defers to context the page had and the question does
# not — "…in these systems", "…in such architectures", "…as shown above". Matched anywhere in the
# text (unlike the possessive rule) because the dangling reference usually lands at the end, and
# there is no legitimate standalone phrasing of "these systems" without an antecedent.
_DANGLING_REFERENCE = re.compile(
    r"\b(?:in|for|with|of|across|within|to)\s+(?:these|those|such)\s+"
    r"(?:systems?|models?|cases?|scenarios?|architectures?|approaches?|methods?|techniques?|"
    r"frameworks?|tools?|pipelines?|settings?|situations?|contexts?)\b"
    r"|\bas\s+(?:shown|described|discussed|mentioned|explained)\s+(?:above|below|earlier|previously)\b"
    r"|^\s*explain\s+how\s+(?:this|that|it)\s+is\s+used\b",
    re.IGNORECASE,
)
_REJECT_RE = [re.compile(p, re.IGNORECASE) for p in _REJECT_PATTERNS]

# A question that ENDS on a bare demonstrative/pronoun refers to something only the source page knew.
# Found while adding the open-web tier: "Are you willing to work on this?" and "Any resources to support
# this?" are real forum lines that passed every other gate, including on an allowed page.
_TRAILING_DEMONSTRATIVE = re.compile(r"\b(?:this|that|these|those|it|them|one)\s*\?\s*$", re.IGNORECASE)
# A capitalised term after the first word, or a letter-digit token (n8n, GPT-4, Automatic1111): the
# question names something concrete, so a trailing pronoun has an antecedent inside the question.
_NAMED_SUBJECT = re.compile(r"(?<!^)\b(?:[A-Z][A-Za-z0-9-]{1,}|[a-z]+\d+[a-z\d]*)\b")


def _names_a_subject(text: str) -> bool:
    words = text.split()
    return bool(words) and bool(_NAMED_SUBJECT.search(" ".join(words[1:])))

# CANDIDATE-voice question: the candidate asking the interviewer about their own career, harvested
# from forum threads ("If I get a chance, can I move to the AI agents stack?"). Requires BOTH a
# first-person subject AND a career-move term, so a technical "How can I reduce latency?" survives.
_SELF_SUBJECT = re.compile(r"\b(?:can|could|will|would|should|may|do|did|am)\s+i\b|\bif i\b|\bmy\s+(?:career|profile|package|salary)\b",
                           re.IGNORECASE)
_CAREER_TERMS = re.compile(
    r"\b(?:move to|switch(?:ing)? to|join|joining|chance|opportunit(?:y|ies)|eligible|apply|applying|"
    r"salary|package|hike|ctc|offer|referral|fresher|placement|stipend|notice period)\b",
    re.IGNORECASE,
)

_WH_STARTS = ("what", "how", "why", "when", "where", "which", "who", "whose", "whom")
# A short line is OK only if it opens like a real question/task ("What are embeddings?"); a short
# NOUN-PHRASE ending in "?" ("The code quality?") is a scrap. Reuse the harvest's question starters.
_SHORT_OK_STARTS = tuple(set(_Q_STARTS) | set(_WH_STARTS) | {"is", "are", "do", "does", "can", "should"})
_MIN_WORDS = 3          # hard floor
_SHORT_WORDS = 5        # below this, require a question/task opener

# ── Long expository prose ────────────────────────────────────────────────────────────────────────
# There was no UPPER bound at all, and a live run shipped 56 words of interviewer rubric as a
# question: "When your prompt produces the wrong output, the question is how quickly you can narrow
# down why. The failure could be in the instruction itself…". It passes every gate above — it opens
# with "When" (a legitimate `_Q_STARTS` word) and is normal-cased.
#
# A BARE WORD CEILING IS THE WRONG FIX, and this was measured before writing the rule. Across the
# 2,835 shipped rows, 26 exceed 40 words and the class is genuinely mixed: 7 are prose blobs like the
# one above, but the rest are real long asks — "Build an API for a leave request system in an HR
# management system using Flask, FastAPI…", "Can you explain the request flow when a user creates a
# blog and hits publish…", plus HackerRank-style coding specs. Rejecting on length alone would have
# killed 7 blobs and 9 genuine questions with them, the same overlapping-distributions trap that
# `_outcome_coverage` and the dedup threshold both document.
#
# What separates them is whether the text ASKS the candidate anything:
#   * its own question mark — one inside a QUOTED example does not count, which is what makes
#     "A recruiter might ask, “Tell me about a project where you applied LLMs”…" prose and not a
#     question; or
#   * a sentence opening with an imperative task verb ("Build …", "Design and implement …").
# Plus a tell for text that talks ABOUT the hiring process rather than asking anything, applied only
# above the length bar so "What do recruiters look for in a GenAI candidate?" is untouched.
#
# Measured: rejects 8 of 2,835 rows (0.28%) — all 7 prose blobs, and one coding word problem the
# dormant coding path cannot use anyway. Two known misses remain (an answer-explanation shipped as a
# question); perfect separation is not available here and a tighter rule costs genuine questions.
_LONG_WORDS = 40
# Imperative task openers, derived FROM `_Q_STARTS` so the two cannot drift — minus the wh-words and
# the discourse-ambiguous ones. "Given" is excluded deliberately: "Given the above, I wanted to
# build…" and "Given these, the best way to prepare is…" are both blog prose, while the coding specs
# that legitimately open that way carry a later "find"/"determine" and survive on those.
_NOT_IMPERATIVE = {"what", "why", "how", "when", "where", "given", "suppose", "tell me", "can you",
                   "difference between", "walk me through"}
_TASK_VERBS = frozenset(
    {s.strip().lower() for s in _Q_STARTS if s.strip().lower() not in _NOT_IMPERATIVE}
    | {"make", "complete", "develop", "refactor", "optimize", "optimise", "debug", "draw",
       "suggest", "propose", "calculate", "determine", "identify", "assume", "traverse"}
)
_QUOTED_SPAN = re.compile(r"[\"“][^\"“”]{8,}?[\"”]|[‘'][^’']{8,}?[’']")
_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?])\s+|(?<=[:;])\s+|(?<=,)\s+)([A-Za-z][a-z]+)")
_INTERVIEW_META = re.compile(
    r"\b(?:a recruiter|recruiters|interviewers|hiring managers?|mock interview|the best answers?|"
    r"you can expect|most candidates|interview prep|in the interview|for the job)\b", re.IGNORECASE)


def _asks_something(t: str) -> bool:
    """Does this text put a question or a task TO the candidate (vs. narrating about interviews)?"""
    if "?" in _QUOTED_SPAN.sub(" ", t):          # its OWN question mark, not a quoted example
        return True
    return any(m.group(1).lower() in _TASK_VERBS for m in _SENTENCE_START.finditer(t))


def _is_expository_prose(t: str) -> bool:
    """A long body that never actually asks anything — see `_LONG_WORDS`."""
    if len(t.split()) <= _LONG_WORDS:
        return False
    return not _asks_something(t) or bool(_INTERVIEW_META.search(t))


def strip_artifacts(text: str) -> str:
    """Remove scrape residue: leading Q:/Q./Ans: markers, backslashes, markdown noise, extra spaces."""
    if not text:
        return text
    s = _BACKSLASH.sub("", text)
    s = _LEAD_JUNK.sub("", s)
    s = _SITE_TAIL.sub("", s)          # drop "… | Dataford Interview Questions" SEO tails
    s = s.strip(" \t`*_[]#>-")
    # A quote mark left dangling AFTER the terminal punctuation is scrape residue, not content:
    # a shipped set carried `How do you prevent hallucinations?"`. Only stripped when it follows ? . or !
    # — a quote elsewhere may be a real quotation, and measured across both banks (2,828 rows) exactly one
    # row matches and zero have the mark anywhere but after a terminal, so this is safe and narrow.
    s = re.sub(r'(?<=[?.!])\s*["”“\'’]+\s*$', "", s)
    s = _WS.sub(" ", s).strip()
    # A Title-cased wh-question often harvested as a page heading ("What Is Gradient Descent",
    # "How Does Backpropagation Work") is a real interview question missing its mark — restore the
    # "?" so the heading filter keeps it instead of discarding a genuine question.
    if s and not s.endswith(("?", ".", "!", ":")) and s.lower().startswith(_WH_STARTS) and len(s.split()) >= 3:
        s = s + "?"
    return s


def _title_case_ratio(t: str) -> float:
    """Fraction of alphabetic words that start with a capital — Title Case detector."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", t)
    if not words:
        return 0.0
    return sum(1 for w in words if w[:1].isupper()) / len(words)


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
    return _title_case_ratio(t) >= 0.6


def _looks_like_article_title(t: str) -> bool:
    """A Title-Cased line that is a blog/tutorial/video title rather than a question:
    "Build Human-Like AI Voice App with Gemini 3.1 Flash TTS", "Closed Source Model — Locked In?".

    Requires ≥5 words, so short real asks ("Explain Stable diffusion") are never caught.

    The Title-Case bar depends on whether the line is SHAPED like a question at all:

      * Ends with "?" or opens with a question/instruction verb → needs ≈total capitalization (0.85).
        Real questions capitalize their content words and proper nouns freely:
        "What are Prompt Engineering Techniques?" (80%) and "Explain Mode Collapse in GANs." (80%)
        are genuine, while "Can Voice Agents Handle Bilingual Customers?" (100%) and
        "Build Human-Like AI Voice App with Gemini 3.1 Flash TTS" (89%) are titles.
      * Neither → a bare Title-Cased phrase ("Top 5 Diffusion Models Explained") is a heading, so
        the lower 0.70 bar applies.
    """
    # Parenthesised glosses are acronym expansions ("Explain GANs (Generative Adversarial
    # Networks)") and are capitalised by nature — they say nothing about the line's casing, so
    # exclude them from both the word count and the ratio.
    core = _PARENS.sub(" ", t).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", core)
    if len(words) < 5:
        return False
    question_shaped = core.endswith("?") or core.lower().startswith(_Q_STARTS)
    return _title_case_ratio(core) >= (0.85 if question_shaped else 0.70)


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
    # Long body that never asks the candidate anything → interviewer rubric / blog prose, not a
    # question. Deliberately not a bare word ceiling — see `_LONG_WORDS` for the measurement.
    if _is_expository_prose(t):
        return False
    if words[0].strip(",.").lower() in _FRAGMENT_STARTS:   # "Also, how diverse…" / "And what about…" → scrap
        return False
    # Listicle/heading opener without a question mark → doc heading, not a question.
    if not t.endswith("?") and t.lower().startswith(_HEADING_STARTS):
        return False
    # Short lines must open like a question/task; a short noun phrase ("The code quality?") is a scrap.
    if len(words) < _SHORT_WORDS and not t.lower().startswith(_SHORT_OK_STARTS):
        return False
    # SENTENCE CONTINUATION: opens lowercase and never asks anything — the clause was split off a
    # larger sentence, so its subject is in the half that got left behind ("how you would apply them
    # in a Generative AI solution.", "why it matters.", "when you would set it to 0."). A live run
    # shipped "how you would iteratively improve prompts and guards to increase reliability." this way.
    # Requiring the missing "?" is what keeps it safe: 37 bank rows open lowercase and DO end in "?"
    # ("what is the attention mechanism?") — those are real questions that merely lost their capital,
    # and they all survive. Of the 29 this rejects, none was a genuine question.
    if t[:1].islower() and not t.endswith("?"):
        return False
    if _looks_like_heading(t):                # "How MCP Works Internally" → heading, not a question
        return False
    if _looks_like_article_title(t):          # "Build Human-Like AI Voice App with Gemini …" → title
        return False
    if any(rx.search(t) for rx in _REJECT_RE):             # boilerplate / logistics / CTA / clickbait
        return False
    # Candidate asking about their own career, not an interview question put TO a candidate.
    if _SELF_SUBJECT.search(t) and _CAREER_TERMS.search(t):
        return False
    # Ends on a bare demonstrative — the thing being asked about is only on the page it came from:
    # "Are you willing to work on this?", "Any resources to support this?" (both real forum lines that
    # otherwise passed every gate). Distinct from the `_DANGLING_REFERENCE` rule above, which needs a
    # noun after the demonstrative. A question ending in a real word is unaffected: "…what merge modes
    # does it support?" ends on "support", not on "it".
    # …but ONLY when the question names nothing for the pronoun to refer to. "What is the HTTP Request
    # node and when do you use it?" is self-contained — "it" is the node, named in the same sentence —
    # and a bare trailing-pronoun rule wrongly rejected it. A named subject (a capitalised term after
    # the first word, or a letter-digit token like n8n/GPT-4) is what separates the two.
    if _TRAILING_DEMONSTRATIVE.search(t) and not _names_a_subject(t):
        return False
    if _DANGLING_REFERENCE.search(t):         # "…in these systems" → antecedent left on the page
        return False
    return True
