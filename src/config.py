import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Prepared data files (from scripts/prepare_data.py)
INTERVIEW_QUESTIONS_JSON = DATA_DIR / "interview_questions.json"
# Curated GenAI question bank (built by scripts/build_genai_bank.py) — used for GEN_AI sessions
# because interview_questions.json is a Python/SWE set with almost no GenAI content.
GENAI_BANK_JSON = DATA_DIR / "genai_question_bank.json"
KNOWLEDGE_GRAPH_JSON = DATA_DIR / "knowledge_graph.json"

# Curriculum context (KP supplements — runtime, loaded by data_loader)
GEN_AI_JSON = DATA_DIR / "curriculum/gen_ai_final.json"
LLM_APPS_JSON = DATA_DIR / "curriculum/llm_applications_kp_links_final_fixed.json"
FLASK_JSON = DATA_DIR / "curriculum/flask_kp_links_final.json"
# Reading materials (runtime, used by session_understanding)
GEN_AI_RM = DATA_DIR / "reading_materials/gen_ai_reading_material.md"
LLM_APPS_RM = DATA_DIR / "reading_materials/llm_applications_reading_material.md"
# Precise per-session reading-material map (built by scripts/build_session_reading_material.py)
SESSION_MAP_JSON = DATA_DIR / "reading_materials/session_map.json"
# Human-curated learning-outcome/interview-topic OVERRIDES (editable). When a session appears here,
# its curated outcomes/interview_topics win over the LLM-derived ones. Built/seeded by
# scripts/audit_outcomes.py; safe to hand-edit. Missing file → pure LLM derivation (unchanged).
SESSION_OUTCOMES_JSON = DATA_DIR / "reading_materials/session_outcomes.json"
# Raw source files (used by prepare_data.py only)
INTERVIEW_CSV = DATA_DIR / "raw/Interview Intelligence Master_ 2026 - Master Sheet.csv"
MEMORY_DB = PROJECT_ROOT / "memory.db"

# Model configuration
ENV = os.getenv("ENV", "development")  # development | staging | production

MODEL_CONFIG = {
    "development": "anthropic/claude-haiku-4-5",            # Cheap for testing
    "staging": "anthropic/claude-sonnet-4-6",
    "production": "anthropic/claude-sonnet-4-6",
}

LLM_MODEL = os.getenv("LLM_MODEL", MODEL_CONFIG.get(ENV, MODEL_CONFIG["development"]))

# Models selectable from the UI at runtime (OpenRouter ids). Extend as needed.
MODEL_OPTIONS = [
    {"id": "anthropic/claude-haiku-4-5",  "label": "Claude Haiku 4.5 · fast & cheap"},
    {"id": "anthropic/claude-sonnet-4-6", "label": "Claude Sonnet 4.6 · balanced"},
    {"id": "anthropic/claude-opus-4.1",   "label": "Claude Opus 4.1 · highest quality"},
    {"id": "openai/gpt-4o-mini",          "label": "GPT-4o mini · fast & cheap"},
    {"id": "openai/gpt-4o",               "label": "GPT-4o · balanced"},
]

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Question constraints
MIN_QUESTIONS = 5
# Upper bound only for the coding-slot math + UI slider — the FINAL set is NOT trimmed to this
# (see tool_submit_question_set: every outcome-relevant question is kept). Raised so the legacy
# app cap doesn't clip the now-uncapped deliverable.
MAX_QUESTIONS = 60
# Candidates are gathered into a WIDE pool, scored for relevance across the whole pool, ranked,
# and (now) ALL relevant ones are kept — no trim-to-max. Per-source caps stop any single source
# monopolising the pool; they are raised so the relevance judge sees far more candidates.
BANK_POOL_CAP = 150       # curated interview data
WEB_POOL_CAP = 120        # fresh company-attributed questions (Tavily)
GITHUB_POOL_CAP = 30      # curated GitHub repos (disabled by default)
# Overall cost guard — total candidates that reach the (LLM-scored) relevance stage. Every candidate
# costs an LLM call, so this caps spend even though per-source caps sum higher.
CANDIDATE_POOL_TARGET = 300
RELEVANCE_BATCH_SIZE = 25  # candidates scored per LLM call in validate_relevance (smaller → no JSON truncation)
# Keep candidates scoring at/above this relevance; below → dropped (min-floor still applies).
# 0.5 (not 0.6) because the scorer rates good foundational questions ~0.55–0.75; a stricter
# bar under-fills sets and starves coverage/per-session representation.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
# Absolute lower bound for the min-questions backfill. When too few candidates clear RELEVANCE_THRESHOLD,
# we top up toward min_questions ONLY from candidates at/above this floor — never below it. So a session
# with a genuinely thin on-topic pool returns FEWER questions rather than padding with off-topic ones.
RELEVANCE_FLOOR = float(os.getenv("RELEVANCE_FLOOR", "0.35"))
# SEMANTIC topic pre-gate before the (LLM) relevance scoring. An ABSOLUTE threshold doesn't work inside a
# single domain (every GenAI question is ~0.3–0.45 similar to any GenAI profile), so we use a COMPARATIVE
# gate: assign each candidate to its nearest COURSE-TOPIC profile and drop it only if it belongs to a
# DIFFERENT topic than this run's — i.e. its best other-topic similarity beats this topic's by a margin.
# This reliably removes cross-topic questions (prompt-engineering Qs in an image-gen session) without
# dropping genuinely on-topic or shared ones. TF-IDF/embeddings-unavailable or unknown topic → skip (no-op).
SEMANTIC_TOPIC_MARGIN = float(os.getenv("SEMANTIC_TOPIC_MARGIN", "0.06"))
SEMANTIC_PREFILTER_FLOOR = float(os.getenv("SEMANTIC_PREFILTER_FLOOR", "0.12"))  # also drop totally-unrelated
# Final selection: greedy MMR with coverage + difficulty bonuses.
#   score = λ·relevance − (1−λ)·redundancy + COVERAGE_BONUS·(covers a new outcome)
#                                           + DIFFICULTY_BONUS·(fills an under-filled difficulty bucket)
MMR_LAMBDA = 0.7
SELECT_COVERAGE_BONUS = 0.15      # nudge toward covering every learning outcome
SELECT_DIFFICULTY_BONUS = 0.10    # nudge toward the Easy/Medium/Hard target mix
SELECT_SESSION_BONUS = 0.12       # nudge so each session (multi-session topic) is represented
SELECT_ATTRIBUTION_BONUS = 0.12   # nudge so the set mixes real-company + source-labeled questions
SELECT_ROLE_BONUS = 0.12          # nudge questions for a TARGET job role above generic ("General") ones

# The final set is NOT trimmed to a product cap (keep every outcome-relevant question); this is only a
# high safety guard against a pathological pool so review/export never explode.
FINAL_SET_CAP = 200

# Target job roles per course category. `query_titles` seed role-framed Tavily searches; `bonus_tags`
# are the stored _infer_role categories that earn SELECT_ROLE_BONUS in final ordering. Role is a
# RANKING signal only — nothing is dropped for role (outcome relevance is the sole hard gate).
TARGET_ROLES = {
    "GEN_AI": {
        "query_titles": ["GenAI Engineer", "LLM Application Developer", "Machine Learning Engineer",
                         "Prompt Engineer", "Data Scientist"],
        "bonus_tags": {"ML/AI Engineer", "Prompt Engineer", "Data Scientist", "GenAI Product Manager"},
    },
}
DEFAULT_TARGET_ROLES = TARGET_ROLES["GEN_AI"]


def target_roles(category: str | None) -> dict:
    return TARGET_ROLES.get((category or "").upper(), DEFAULT_TARGET_ROLES)

# Semantic embeddings (free, local sentence-transformers) for redundancy/coverage/attribution.
# Falls back to TF-IDF automatically if the model/library is unavailable.
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "1") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_COVERAGE_THRESHOLD = float(os.getenv("EMBED_COVERAGE_THRESHOLD", "0.30"))  # cosine ≥ ⇒ outcome covered
DEFAULT_DIFFICULTY_DISTRIBUTION = {"easy": 0.3, "medium": 0.5, "hard": 0.2}


def pool_target(max_questions: int) -> int:
    """Total candidate-pool ceiling before relevance ranking — never below max_questions."""
    return max(CANDIDATE_POOL_TARGET, max_questions or 0)
DEDUP_THRESHOLD = 0.85            # TF-IDF cosine (fallback path / cross-run dedup)
# Semantic (embeddings) near-duplicate threshold for within-run dedup. TF-IDF misses REWORDED dupes
# ("What are LLMs?" vs "What are Large Language Models in AI?"); embeddings catch them. 0.82 collapses
# clear rewordings while keeping genuinely distinct angles ("how do LLMs work?" stays separate).
DEDUP_SEMANTIC_THRESHOLD = float(os.getenv("DEDUP_SEMANTIC_THRESHOLD", "0.82"))
QUALITY_PASS_THRESHOLD = 0.75
MAX_EVAL_RETRIES = 2
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "20"))

# Live question harvesting (tools 12 & 13 — search_github_questions / search_web_questions)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")   # optional; raises GitHub API rate limit from 60→5000/hr
# GitHub repos are general ML/Data-Science (Python/stats noise, no company attribution) — disabled.
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "0") == "1"
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
TAVILY_MAX_OUTCOMES = int(os.getenv("TAVILY_MAX_OUTCOMES", "14"))
TAVILY_MAX_RECORDS = int(os.getenv("TAVILY_MAX_RECORDS", "800"))

# Approximate USD price per 1M tokens (input/output) for cost ESTIMATES only.
# Update as provider pricing changes — figures are representative, not billed amounts.
MODEL_PRICING = {
    "anthropic/claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "anthropic/claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "anthropic/claude-opus-4.1":   {"in": 15.00, "out": 75.00},
    "openai/gpt-4o-mini":          {"in": 0.15, "out": 0.60},
    "openai/gpt-4o":               {"in": 2.50, "out": 10.00},
}


def estimate_cost(usage: dict | None) -> float | None:
    """Estimated USD cost for a run's token usage, using MODEL_PRICING.
    Returns None if the model is unknown/unpriced (e.g. pre-tracking runs)."""
    if not usage:
        return None
    price = MODEL_PRICING.get(usage.get("model") or "")
    if not price:
        return None
    return round(
        (usage.get("prompt_tokens", 0) / 1_000_000) * price["in"]
        + (usage.get("completion_tokens", 0) / 1_000_000) * price["out"],
        4,
    )


INTERVIEW_GITHUB_REPOS = [
    "llmgenai/LLMInterviewQuestions",
    "amitshekhariitbhu/ai-engineering-interview-questions",
    "amitshekhariitbhu/machine-learning-interview-questions",
    "Devinterview-io/llms-interview-questions",
    "shafaypro/CrackingMachineLearningInterview",
    "khangich/machine-learning-interview",
    "alirezadir/Machine-Learning-Interviews",
    "andrewekhalel/MLQuestions",
    "kojino/120-Data-Science-Interview-Questions",
    "youssefHosni/Data-Science-Interview-Questions-Answers",
    "rbhatia46/Data-Science-Interview-Resources",
    "Sroy20/machine-learning-interview-questions",
]

INTERVIEW_SOURCE_ALLOWLIST = {
    "tryexponent.com", "datalemur.com", "stratascratch.com", "prachub.com",
    "interviewquery.com", "prepfully.com", "igotanoffer.com", "glassdoor.com",
    "teamblind.com", "leetcode.com", "indeed.com", "interviewing.io",
    "hellointerview.com", "ambitionbox.com", "geeksforgeeks.org", "interviewbit.com",
    "prepinsta.com", "indiabix.com", "naukri.com",
    # NOTE: reddit.com, quora.com, medium.com, dev.to are DELIBERATELY EXCLUDED — low-trust
    # (comment fragments, personal blogs, headings-as-questions). Credibility over volume.
    "datascience.stackexchange.com", "stats.stackexchange.com",
    "stackoverflow.com", "datacamp.com", "analyticsvidhya.com", "kdnuggets.com",
    "towardsai.net", "towardsdatascience.com", "tredence.com", "igmguru.com",
    "vinsys.com", "novelvista.com", "generativeaimasters.in", "blockchain-council.org",
    "amquesteducation.com", "simplilearn.com", "edureka.co", "intellipaat.com",
    "projectpro.io", "turing.com", "springboard.com", "mlstack.cafe",
    "365datascience.com", "builtin.com",
    # Additional real-company interview-question sources
    "glassdoor.co.in", "careercup.com", "comparably.com", "fishbowlapp.com",
    "educative.io", "scaler.com", "levels.fyi", "ambitionbox.in",
    "tealhq.com", "interviewkickstart.com",
    # High-signal additions (review): first-hand experiences + curated prep
    "1point3acres.com", "workat.tech", "hackerearth.com", "hackerrank.com", "freecodecamp.org",
    # GenAI/ML Q&A + forums, SWE interview-experience, extra attribution
    "ai.stackexchange.com", "huggingface.co", "kaggle.com", "machinelearningmastery.com",
    "techinterviewhandbook.org", "neetcode.io", "taro.co", "bigtechinterviews.com",
    # High-signal company-tagged GenAI/ML interview source
    "dataford.io",
}
