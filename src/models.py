from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator, computed_field
import re
import uuid


def new_uuid() -> str:
    return str(uuid.uuid4())


NIAT = "NIAT"  # last-resort label when neither a real company nor a source site is known

_QNUM_PREFIX = re.compile(r"^\s*(?:[•\-*]\s*)?(?:q(?:uestion)?\s*)?\d+\s*[.):]\s*", re.IGNORECASE)

# Friendly display names for known question sources (used when a question has no verified company).
# Team decision: show the SOURCE SITE for no-company questions (blogs like GeeksforGeeks/DataCamp are
# valuable for prep even without a company tag); fall back to NIAT only when neither is known.
_SOURCE_SITE_NAMES = {
    "geeksforgeeks.org": "GeeksforGeeks", "datacamp.com": "DataCamp",
    "interviewbit.com": "InterviewBit", "analyticsvidhya.com": "Analytics Vidhya",
    "kdnuggets.com": "KDnuggets", "machinelearningmastery.com": "Machine Learning Mastery",
    "towardsdatascience.com": "Towards Data Science", "towardsai.net": "Towards AI",
    "simplilearn.com": "Simplilearn", "edureka.co": "Edureka", "intellipaat.com": "Intellipaat",
    "projectpro.io": "ProjectPro", "turing.com": "Turing", "365datascience.com": "365 Data Science",
    "mlstack.cafe": "MLStack.cafe", "tryexponent.com": "Exponent",
    "interviewquery.com": "Interview Query", "hellointerview.com": "Hello Interview",
    "glassdoor.com": "Glassdoor", "glassdoor.co.in": "Glassdoor", "ambitionbox.com": "AmbitionBox",
    "prepfully.com": "Prepfully", "igotanoffer.com": "IGotAnOffer", "leetcode.com": "LeetCode",
    "kaggle.com": "Kaggle", "huggingface.co": "Hugging Face", "medium.com": "Medium", "dev.to": "DEV",
    "stackoverflow.com": "Stack Overflow", "ai.stackexchange.com": "AI Stack Exchange",
    "prachub.com": "PracHub", "dataford.io": "Dataford", "prepinsta.com": "PrepInsta",
    "scaler.com": "Scaler", "educative.io": "Educative", "naukri.com": "Naukri",
    "techinterviewhandbook.org": "Tech Interview Handbook", "neetcode.io": "NeetCode",
    "builtin.com": "Built In", "365datascience.com": "365 Data Science",
}


def strip_question_prefix(text: str) -> str:
    """Remove a leading question number/label like 'Q1.', 'Q2)', 'Question 3:', '1.'."""
    if not text:
        return text
    return _QNUM_PREFIX.sub("", text).strip()


def _site_name_from_url(source_url: str | None) -> str | None:
    """Friendly source-site name from a URL, or None. E.g. https://www.geeksforgeeks.org/x → GeeksforGeeks."""
    if not source_url:
        return None
    m = re.search(r"https?://([^/]+)", source_url) or re.match(r"([^/]+)", source_url)
    host = (m.group(1) if m else "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    if host in _SOURCE_SITE_NAMES:
        return _SOURCE_SITE_NAMES[host]
    # Unknown allowlisted-ish host → derive a readable label from the registrable name.
    label = host.split(":")[0].split(".")
    base = label[-2] if len(label) >= 2 else label[0]
    return base[:1].upper() + base[1:] if base else None


def attribution_label(asked_in_company: str | None, source: str | None = None,
                      source_url: str | None = None) -> str:
    """WHO ASKED this question: the real company in UPPERCASE if known, else the placeholder "NIAT".

    This deliberately does NOT fall back to the source site. It used to, and a real run shipped a set
    attributed to "Analytics Vidhya", "Indeed", "Edureka" and "DataCamp" — content sites, sitting in
    the same column and the same UI tag as a genuine "ANTHROPIC". A reviewer reads that as "Indeed
    asked this in an interview", when the question was in fact a bullet scraped from an Indeed
    job-description page. One of those two claims is a company attribution and the other is
    provenance; merging them makes the honest case indistinguishable from the misleading one.

    Provenance is not lost — it moves to `source_site` (below), which the review UI shows as a
    separate "via <site>" tag. Fabricated/garbage company values are filtered upstream by tavily's
    `_valid_company`.
    """
    if asked_in_company and asked_in_company.strip():
        return asked_in_company.strip().upper()
    return NIAT


# --- Generation Config (user input) ---

class GenerationConfig(BaseModel):
    session_names: list[str]          # One or more sessions to combine
    max_questions: int = 15
    min_questions: int = 5
    model: str | None = None          # runtime-selected LLM (OpenRouter id); None → configured default
    preview: bool = False             # TESTING: pause after Validation to inspect picked questions
    category: str = "GEN_AI"          # course category → drives sheet branding (Tags/framework)
    # Declared type of the COURSE (theory_heavy | code_heavy | mixed), carried for provenance only.
    # Nothing reads it, and per-type behaviour must not key off it: the authoritative value is the
    # per-session `SessionContext.session_type`, resolved from that session's reading material. A
    # course-level label would flatten a mixed course to one type.
    course_type: str | None = None
    difficulty_bias: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    dry_run: bool = False

    @property
    def session_name(self) -> str:
        """Combined session name for display."""
        return " + ".join(self.session_names)


# --- Topic Resolution Models ---

class KPMatch(BaseModel):
    kp_id: str                    # "KP_GLOBAL_0777"
    kp_label: str                 # "Zero-shot prompting techniques"
    relevance: float              # 0.0 - 1.0
    source_file: str              # "gen_ai_final.json"


class TopicMatch(BaseModel):
    topic: str                    # "AI_ML"
    sub_topic: str | None = None  # "PROMPT_ENGINEERING"
    confidence: float             # 0.0 - 1.0


class SessionContext(BaseModel):
    session_name: str
    learning_outcomes: list[str]
    key_concepts: list[str]
    # Transferable, interview-relevant GenAI concepts a candidate would actually be asked about — derived
    # from the session so hands-on tool/build sessions (e.g. an n8n workflow) map to real interview topics
    # (LLM summarization, prompt engineering, API integration) rather than tool UI/nodes. Used for
    # retrieval queries + relevance grounding. Defaults empty for backward-compat with cached resolutions.
    interview_topics: list[str] = Field(default_factory=list)
    scope_in: list[str]
    scope_out: list[str]
    session_type: Literal["theory_heavy", "code_heavy", "mixed"]
    matched_kp_ids: list[KPMatch]
    matched_csv_topics: list[TopicMatch]
    prerequisite_kp_chain: list[str]  # ordered KP IDs
    difficulty_distribution: dict[str, float]


# --- Question Models (output rows) ---

class QuestionDetail(BaseModel):
    question_id: str = Field(default_factory=new_uuid)
    category: str                           # GEN_AI, LLM, PYTHON, SQL, DSA, RESUME_DEEP_DIVE
    content: str                            # Full question text
    topic: str
    sub_topic: str | None = None
    difficulty: str | None = None           # Easy / Medium / Hard
    language: str | None = None
    framework: str | None = None
    tool: str | None = None
    asked_in_company: str | None = None
    role: str | None = None
    source_url: str | None = None
    source: Literal["curriculum", "interview_db", "web", "github", "generated"]
    kp_label: str | None = None
    expected_answer: str | None = None
    relevance_score: float | None = None    # 0.0–1.0 relevance to session (set by validate_relevance)
    session: str | None = None               # which selected session this question best matches
    # Hybrid semantic+lexical score from the bank retriever (set by QuestionBankRetriever.search).
    # Recall-stage signal only — NOT a relevance judgement.
    retrieval_score: float | None = None
    # Cosine similarity to THIS session's own profile (learning outcomes + interview topics +
    # reading material), set by the session-grounded pre-gate. Used to rank candidates before the
    # LLM relevance pass and to tier the review UI. None when embeddings are unavailable.
    session_fit: float | None = None
    # The question EXACTLY as sourced, kept only when the post-relevance scope trim shortened it
    # (`tools._scope_trim` dropped an off-syllabus sub-clause — "…improve prompts and guards" →
    # "…improve prompts"). Storing the original rather than a bare `adapted` boolean is deliberate:
    # the edit is auditable, so a reviewer can see exactly what was removed from a question that
    # still carries a company's name.
    original_content: str | None = None
    # A concept this question requires that appears NOWHERE in its session's reading material, set by
    # `tools._syllabus_audit`. On-domain and on-syllabus are different things: a real run shipped four
    # questions the gate called "on-domain" that tested guardrails, production debugging, hallucination
    # and ambiguous-intent fallbacks — none of which either session teaches. Flagged, not rejected: the
    # reviewer decides whether to keep a question that goes beyond the material.
    off_syllabus_concept: str | None = None

    @field_validator("expected_answer", mode="before")
    @classmethod
    def coerce_answer(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v

    @computed_field
    @property
    def adapted(self) -> bool:
        """True when `content` is no longer verbatim from the source (scope-trimmed)."""
        return bool(self.original_content and self.original_content.strip() != self.content.strip())

    @computed_field
    @property
    def attribution(self) -> str:
        """WHO ASKED it — real company in UPPERCASE, or NIAT. Never a website (see attribution_label)."""
        return attribution_label(self.asked_in_company, self.source, self.source_url)

    @computed_field
    @property
    def source_site(self) -> str | None:
        """WHERE IT WAS FOUND (e.g. "GeeksforGeeks") — provenance, kept strictly separate from
        `attribution` so a content site is never presented as the asking company. Populated whenever
        there is a URL, including for company-attributed questions (the site is where the claim came
        from, which is worth keeping); None when there is no URL."""
        return _site_name_from_url(self.source_url)


class CodingQuestion(BaseModel):
    id: str = Field(default_factory=new_uuid)
    category: str                           # SQL_CODING_DEEP_DIVE, PYTHON_CODING, LLM_APP_CODING
    title: str
    content: str                            # Markdown problem statement
    code_id: str | None = None              # Links to CodeSnippet
    topic: str
    sub_topic: str | None = None
    difficulty: str | None = None
    language: str
    framework: str | None = None
    tool: str | None = None
    asked_in_company: str | None = None
    source: Literal["curriculum", "interview_db", "web", "github", "generated"]
    expected_answer: str | None = None

    @computed_field
    @property
    def attribution(self) -> str:
        """Real company if known, else honest source label (never fabricated)."""
        return attribution_label(self.asked_in_company, self.source, None)


class CodeSnippet(BaseModel):
    code_id: str
    code_content: str
    language: str                           # PYTHON, CPP, JAVA, JAVASCRIPT, SQL


class CodeAnalysisQuestion(BaseModel):
    """Code analysis / MCQ question — from Excel Sheet 8 format.

    The student reads a code snippet (linked via code_id) and answers the question.
    """
    question_id: str = Field(default_factory=new_uuid)
    tag_name: str                            # e.g. "python_code_analysis"
    content: str                             # Problem description / question text
    code_id: str                             # Links to CodeSnippet with the code to analyze
    title: str                               # Short title e.g. "Validate_subsequence"
    correct_answer: str | None = None        # Expected output / correct option
    difficulty: str | None = None
    topic: str | None = None
    source: Literal["curriculum", "interview_db", "web", "github", "generated"] = "curriculum"


# --- Pipeline Output Models ---

# `LocalPool` was removed here. It declared a `curriculum_questions` pool and was instantiated
# nowhere — a model implying the pipeline has a curriculum question source, which it does not and
# must not (see the note in config.py).


class WebPool(BaseModel):
    web_questions: list[QuestionDetail] = Field(default_factory=list)
    web_coding_questions: list[CodingQuestion] = Field(default_factory=list)


class CurationMetadata(BaseModel):
    total_candidates: int = 0
    dedup_removed: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    questions_from_web: int = 0
    # Retrieval funnel (observability): raw fetched → pooled → removed → final
    raw_fetched: dict[str, int] = Field(default_factory=dict)   # per-source raw hit counts
    pool_size: int = 0                                          # candidates gathered before ranking
    removed_by_relevance: int = 0
    removed_by_dedup: int = 0


class CuratedOutput(BaseModel):
    question_details: list[QuestionDetail] = Field(default_factory=list)
    coding_questions: list[CodingQuestion] = Field(default_factory=list)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    metadata: CurationMetadata = Field(default_factory=CurationMetadata)


class FlaggedQuestion(BaseModel):
    """An unresolved quality-gate objection, surfaced to the reviewer.

    `question_id` is None for set-level issues (too few/too many questions, an unscored set, a
    session with no representation) — those aren't attributable to one question.
    """
    question_id: str | None = None
    issue: str                      # off-domain | duplicate | malformed | too-few | too-many | …
    suggestion: str = ""

    @classmethod
    def from_gate(cls, raw: dict) -> "FlaggedQuestion":
        return cls(question_id=raw.get("id"),
                   issue=str(raw.get("issue") or "unspecified"),
                   suggestion=str(raw.get("suggestion") or ""))


class QualityReport(BaseModel):
    composite_score: float = 0.0
    metric_scores: dict[str, float] = Field(default_factory=dict)
    pass_fail: Literal["pass", "fail"] = "fail"
    flagged_questions: list[FlaggedQuestion] = Field(default_factory=list)
    critique: list[str] = Field(default_factory=list)
    loops_used: int = 0
    api_usage: dict = Field(default_factory=dict)
    # Web-search (Tavily) health for this run, so a bank-only fallback is visible in the UI:
    # ok | empty | no_key | quota | auth | rate | full | error | not_run
    web_status: str = "not_run"
    web_error: str | None = None
