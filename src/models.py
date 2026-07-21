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
    """Attribution for output: the real company in UPPERCASE if known; otherwise the SOURCE SITE
    (e.g. "GeeksforGeeks") derived from the source URL; else the placeholder "NIAT".
    Fabricated/garbage company values are filtered upstream (tavily `_valid_company`)."""
    if asked_in_company and asked_in_company.strip():
        return asked_in_company.strip().upper()
    site = _site_name_from_url(source_url)
    if site:
        return site
    return NIAT


# --- Generation Config (user input) ---

class GenerationConfig(BaseModel):
    session_names: list[str]          # One or more sessions to combine
    max_questions: int = 15
    min_questions: int = 5
    model: str | None = None          # runtime-selected LLM (OpenRouter id); None → configured default
    preview: bool = False             # TESTING: pause after Validation to inspect picked questions
    category: str = "GEN_AI"          # course category → drives sheet branding (Tags/framework)
    course_type: str | None = None    # theory_heavy | code_heavy | mixed (from the selected course)
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
    source: Literal["curriculum", "interview_db", "web", "generated"]
    kp_label: str | None = None
    expected_answer: str | None = None
    relevance_score: float | None = None    # 0.0–1.0 relevance to session (set by validate_relevance)
    session: str | None = None               # which selected session this question best matches

    @field_validator("expected_answer", mode="before")
    @classmethod
    def coerce_answer(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v

    @computed_field
    @property
    def attribution(self) -> str:
        """Real company if known, else honest source label (never fabricated)."""
        return attribution_label(self.asked_in_company, self.source, self.source_url)


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
    source: Literal["curriculum", "interview_db", "web", "generated"]
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
    source: Literal["curriculum", "interview_db", "web", "generated"] = "curriculum"


# --- Pipeline Output Models ---

class LocalPool(BaseModel):
    curriculum_questions: list[QuestionDetail] = Field(default_factory=list)
    interview_questions: list[QuestionDetail] = Field(default_factory=list)
    coding_questions: list[CodingQuestion] = Field(default_factory=list)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    local_count: int = 0


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
    question_id: str
    reason: str
    score: float


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
