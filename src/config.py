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

# NOT declared here on purpose: data/curriculum/*.json.
# Those are course assessment items (1,610 of 1,822 are multiple-choice), and they are BUILD-TIME
# inputs only — `scripts/build_knowledge_graph.py` reads them by literal path to regenerate
# knowledge_graph.json. They are never loaded at runtime and must never reach the retrieval corpus.
# Constants for them used to live here and `data_loader` used them to build a question list nothing
# read, while printing "Loaded 1819 curriculum questions into bank" — which is exactly the confusion
# an unused path invites, so the path is gone rather than merely unused.
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
# Per-request timeout. The OpenAI SDK's default is 600s, long enough for one wedged call to hold a
# run — and its SSE stream — open for ten minutes.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

# Question constraints
MIN_QUESTIONS = 5
# Upper bound of the UI's "target count" slider. tool_submit_question_set DOES trim the final set to
# the run's own `config.max_questions` (clamped by FINAL_SET_CAP) — an earlier comment here claimed
# the set was never trimmed, which was true for one release and has not been since.
MAX_QUESTIONS = 60
# Candidates are gathered into a WIDE pool and scored for relevance across the whole pool; selection
# then ranks and keeps the best up to the requested count. Per-source caps stop any single source
# monopolising the pool, and are generous so the relevance judge sees plenty of candidates.
BANK_POOL_CAP = 150       # curated interview data
WEB_POOL_CAP = 120        # fresh company-attributed questions (Tavily)
GITHUB_POOL_CAP = 30      # curated GitHub repos (disabled by default)
# NOTE: a CANDIDATE_POOL_TARGET / pool_target() pair used to sit here as an "overall pool cost guard".
# Nothing ever read it, and its only apparent consumer (AgentState.remaining_capacity) was itself dead
# and fed a phantom key into the progress summary. The pool is bounded by the per-source caps above.
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
# A candidate that STRONGLY matches THIS session's topic (cur similarity ≥ this) is never dropped by the
# topic pre-gate, even if some other topic's pooled max is higher. Guards against the max-over-many-topics
# inflation bias that could otherwise over-drop genuinely on-topic questions.
SEMANTIC_CUR_KEEP = float(os.getenv("SEMANTIC_CUR_KEEP", "0.60"))
# Final selection: greedy MMR with coverage + difficulty bonuses.
#   score = λ·relevance − (1−λ)·redundancy + COVERAGE_BONUS·(covers a new outcome)
#                                           + DIFFICULTY_BONUS·(fills an under-filled difficulty bucket)
MMR_LAMBDA = 0.7
SELECT_COVERAGE_BONUS = 0.15      # nudge toward covering every learning outcome
SELECT_DIFFICULTY_BONUS = 0.10    # nudge toward the Easy/Medium/Hard target mix
SELECT_SESSION_BONUS = 0.12       # nudge so each session (multi-session topic) is represented
SELECT_ATTRIBUTION_BONUS = 0.12   # nudge so the set mixes real-company + source-labeled questions
SELECT_ROLE_BONUS = 0.12          # nudge questions for a TARGET job role above generic ("General") ones
# Demote candidates that look more like a REJECTED question than an ACCEPTED one. Exact repeats are
# dropped outright upstream (normalized-string match); this catches REWORDINGS, which otherwise come
# back every run and get rejected again. Scaled by the RELATIVE margin (see tools._feedback_penalty),
# which is typically 0.0–0.3, so the effective demotion is up to ~0.15 — enough to reorder near-ties
# without overriding relevance. A ranking penalty, not a hard filter: a thin pool still fills.
SELECT_REJECTED_PENALTY = float(os.getenv("SELECT_REJECTED_PENALTY", "0.5"))

# Hard ceiling on the delivered set, whatever the UI asked for — a safety guard against a
# pathological pool so review and export never explode.
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


# ── Per-session-type tuning ──────────────────────────────────────────────────
# `session_type` (theory_heavy | code_heavy | mixed) was computed, stored, printed into two prompt
# headers, and acted on NOWHERE. These two tables are what make it mean something.

SESSION_TYPES = ("theory_heavy", "code_heavy", "mixed")


def normalize_session_type(value: str | None) -> str:
    """Coerce anything to a known session type. Unknown/None → 'mixed' (the neutral default)."""
    v = (value or "").strip().lower()
    return v if v in SESSION_TYPES else "mixed"


# Difficulty mix per type. A code-heavy session's questions are implementation and design work, which
# sits higher on the scale than recall; a theory session legitimately has more Easy definitional
# questions. One global 30/50/20 penalised whichever type it didn't describe.
#
# IMPORTANT: read via `difficulty_targets()` at the two places that actually decide the mix —
# `tools.tool_check_difficulty_balance` and `tools._select_final`. Do NOT route this through
# `SessionContext.difficulty_distribution`: that field is hardcoded at every construction site and read
# by nothing, so setting it looks like a fix and changes no behaviour.
DIFFICULTY_BY_TYPE = {
    "theory_heavy": {"Easy": 0.35, "Medium": 0.50, "Hard": 0.15},
    "code_heavy":   {"Easy": 0.20, "Medium": 0.50, "Hard": 0.30},
    "mixed":        {"Easy": 0.30, "Medium": 0.50, "Hard": 0.20},
}


def difficulty_targets(session_type: str | None) -> dict[str, float]:
    """Target Easy/Medium/Hard proportions for this session type."""
    return DIFFICULTY_BY_TYPE[normalize_session_type(session_type)]


# Eval pass/fail bars per type. A code-heavy session is scored against banks that hold almost no
# implementation questions (`interview_questions.json` is project/auth/Python; the GenAI bank is
# conceptual), so it scores systematically lower on coverage and grounding for reasons that are about
# SOURCE COVERAGE, not question quality. A single bar therefore either fails every code session or
# sets theory's bar too low — neither tells you anything.
#
# Raise the code-heavy bars as implementation questions actually reach the banks; that rise is the
# signal that the source gap is closing.
EVAL_THRESHOLDS_BY_TYPE = {
    "theory_heavy": {"accept": 0.60, "coverage": 0.60, "grounding": 0.45},
    "code_heavy":   {"accept": 0.55, "coverage": 0.45, "grounding": 0.35},
    "mixed":        {"accept": 0.60, "coverage": 0.55, "grounding": 0.40},
}


def eval_thresholds(session_type: str | None) -> dict[str, float]:
    return EVAL_THRESHOLDS_BY_TYPE[normalize_session_type(session_type)]

# Semantic embeddings (free, local sentence-transformers) for redundancy/coverage/attribution.
# Falls back to TF-IDF automatically if the model/library is unavailable.
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "1") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Disk cache for corpus embedding matrices (keyed by corpus digest + model), so the one-off encode
# of ~1.5k bank questions isn't repaid on every process start.
EMBED_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"
# HYBRID bank retrieval: score = w·embedding_cosine + (1−w)·tfidf_cosine.
# Semantic-dominant because pure TF-IDF matched words not meaning (an F5-TTS query returned RAG
# questions); TF-IDF keeps a real minority weight because exact tool names ("n8n", "LoRA", "F5-TTS")
# are exactly where lexical matching beats a small embedding model.
HYBRID_EMBED_WEIGHT = float(os.getenv("HYBRID_EMBED_WEIGHT", "0.6"))
# Floor on the hybrid score. Unlike TF-IDF (0 for no shared words), embedding cosine is non-trivial
# even for unrelated text, so a real floor is needed to stop the long tail entering the pool.
HYBRID_MIN_SCORE = float(os.getenv("HYBRID_MIN_SCORE", "0.12"))
# Session-grounded pre-gate (replaces relying on pooled COURSE-TOPIC profiles alone). A candidate is
# scored against THIS session's own profile — curated learning outcomes + interview topics + reading
# material. Below the floor it is dropped before the (expensive) LLM relevance pass.
SESSION_FIT_FLOOR = float(os.getenv("SESSION_FIT_FLOOR", "0.18"))
# …and a RELATIVE floor, as a fraction of the best fit found for THIS session. An absolute floor alone
# is mis-calibrated because the achievable fit depends on how well the banks happen to cover a session:
# measured over the GenAI bank, "Introduction to AI Agents" peaks at 0.86 while "Mastering Image
# Generation with Stable Diffusion" peaks at 0.66, so a floor of 0.18 keeps 85% of candidates for the
# first and 68% for the second — i.e. it barely filters either. Scaling to the session's own ceiling
# keeps a comparable proportion of the best candidates whatever the coverage. The absolute floor above
# still applies, so a session where nothing fits is not rescued by having a low ceiling.
SESSION_FIT_RELATIVE = float(os.getenv("SESSION_FIT_RELATIVE", "0.5"))
# Review tiering: candidates at/above this session_fit are surfaced as "high confidence" in the UI.
SESSION_FIT_HIGH = float(os.getenv("SESSION_FIT_HIGH", "0.35"))
# How much of the session profile is reading material (vs curated outcomes). The reading material is
# long and dilutes short outcome statements, so outcomes are embedded separately and we take the MAX
# similarity over profile texts; this cap just bounds how many RM chunks join the profile.
# Reading-material chunking for the grounding profile (`pipeline._session_profile`).
#
# SESSION_PROFILE_RM_CHUNKS is a RUNAWAY GUARD, not a sampling budget. It used to be 12 AND the code
# stride-sampled down to it, so only ~36% of a session's material reached the profile and the bullet
# defining the HTTP Request node (85 chars) was dropped entirely — a question about something the
# session literally teaches scored 0.275. Treating this cap as a budget is what caused that; raise it
# rather than sampling.
SESSION_PROFILE_RM_CHUNKS = int(os.getenv("SESSION_PROFILE_RM_CHUNKS", "400"))
# Keep short paragraphs: in this curriculum the bullets ARE the tool/node definitions, which is exactly
# what grounding needs to match. Only true noise (a bare heading, a stray token) is below this.
RM_CHUNK_MIN_CHARS = int(os.getenv("RM_CHUNK_MIN_CHARS", "40"))
# Long paragraphs are sub-split at a word boundary instead of truncated, so their tail stays searchable.
RM_CHUNK_MAX_CHARS = int(os.getenv("RM_CHUNK_MAX_CHARS", "600"))
# Reading-material chunks are instructional prose, not statements of interview intent — a setup
# walkthrough about copying an auth token matches generic auth questions. Discount them so an
# RM-only match has to be distinctly stronger than a curated-outcome match to keep a candidate.
SESSION_PROFILE_RM_WEIGHT = float(os.getenv("SESSION_PROFILE_RM_WEIGHT", "0.85"))
EMBED_COVERAGE_THRESHOLD = float(os.getenv("EMBED_COVERAGE_THRESHOLD", "0.30"))  # cosine ≥ ⇒ outcome covered


DEDUP_THRESHOLD = 0.85            # TF-IDF cosine (fallback path / cross-run dedup)
# Semantic (embeddings) near-duplicate threshold for within-run dedup. TF-IDF misses REWORDED dupes
# ("What are LLMs?" vs "What are Large Language Models in AI?"); embeddings catch them. 0.82 collapses
# clear rewordings while keeping genuinely distinct angles ("how do LLMs work?" stays separate).
DEDUP_SEMANTIC_THRESHOLD = float(os.getenv("DEDUP_SEMANTIC_THRESHOLD", "0.82"))
# Removed: QUALITY_PASS_THRESHOLD, MAX_EVAL_RETRIES, DEFAULT_DIFFICULTY_DISTRIBUTION — defined here
# and read by nothing, not even inside this file. QUALITY_PASS_THRESHOLD was the actively
# misleading one: the gate decides with the explicit per-condition bars in
# `pipeline._build_quality_report` (`gate_checks`), so a stale "pass threshold" constant read
# like the thing being enforced. Same hazard that got the curriculum path constants deleted —
# an unused path is what invites the confusion.
# Per-agent tool-call budgets live on each agent class (BaseAgent.max_tool_calls), which is what
# actually applies. A module-level MAX_TOOL_CALLS used to sit here reading an env var that nothing
# consumed, so setting it appeared to work and did nothing.

# The mock interview is CONVERSATIONAL — answered out loud, with no keyboard, IDE or whiteboard. So a
# question demanding a produced artifact ("Write a Python program to…", "Implement an input box…") cannot
# be answered at all and only burns a slot. `interview_format.is_hands_on_task` decides; the pool filter
# lives in `pipeline._drop_hands_on`.
#
# This is POLICY, not a data defect, which is why it is a runtime flag and not a `quality.py` rule: the
# form gate feeds `scripts/clean_bank.py`, so putting it there would permanently delete 217 real
# company-attributed coding questions the LMS coding tabs exist for. Set to 0 to ship them again.
CONVERSATIONAL_ONLY = os.getenv("CONVERSATIONAL_ONLY", "1") == "1"

# Questions per interview topic in `outcome_balance.balance_by_outcome` — used ONLY by its opt-in
# `strict=True` quota mode, NOT by the default path. Kept because the quota is occasionally the right
# blunt instrument, but it is not what balances a set: a question is normally dropped only when the LLM
# judge says another question already covers that outcome.
#
# Why the quota is not the default, measured: a cap of 2 is about right on No-Code AI Automation (22
# interview topics for 38 questions) and destroys Gen AI Foundations (15 topics for 54), where one coarse
# topic holds 14 questions and the quota deletes 12 — including three genuinely distinct "what is the
# difference between…" questions. The number was measuring how finely the curriculum enumerates
# `interview_topics`, not whether the questions repeat.
#
# The problem the balance exists for is real regardless: `coverage_efficiency` asks "did each question
# earn its place against a DISTINCT topic" but only within a single run's selected set, and
# `tool_submit_question_set` runs `_same_thing_pass` BEFORE `_add_retained`, so the ACCUMULATED set was
# never judged against itself. That is how hallucination came to be asked six times in 38 questions.
OUTCOME_CAP = int(os.getenv("OUTCOME_CAP", "2"))
# Below this, a question's best-matching interview topic is not a real match, so the question is an ORPHAN
# and is KEPT rather than counted against any cap. Not a tuning knob — it is what stops the cap deleting
# on-topic questions the outcome list fails to describe: "What is the Split In Batches node used for?"
# matches its best outcome at 0.173 and "What are nodes in N8N" at 0.132, because `interview_topics`
# under-describes n8n. Those are the n8n gap showing up again, not duplicates.
OUTCOME_ORPHAN_FLOOR = float(os.getenv("OUTCOME_ORPHAN_FLOOR", "0.35"))

# Stop a run outright when the Tavily pre-flight fails, instead of continuing bank-only. ON by default.
#
# Retrieval is the run. With the web tier dead, every downstream stage — the Evaluation agent, the
# relevance judge, the syllabus audit, the same-thing pass, the outcome balance, up to three gate
# critiques — still executes against a pool the failure already decided, and buys nothing.
#
# THE COST OF THIS IS MEASURED AND ACCEPTED, not assumed. Across 62 persisted runs the bank supplies
# **75%** of all shipped questions (459 of 615), and on a 17-run sample **12 would have shipped >= 5
# questions bank-only** while 5 would genuinely have been too thin. So this guard refuses runs that would
# have worked. That is the chosen policy — guaranteed no wasted spend over best-effort output.
#
# Set to 0 for exactly the previous behaviour: a failed pre-flight disables web search and the run
# continues bank-only, with the failure surfaced in the report banner.
REQUIRE_WEB_SEARCH = os.getenv("REQUIRE_WEB_SEARCH", "1") == "1"
# Tavily statuses worth a second probe. `no_key`, `auth` and `quota` are terminal — nothing will work
# today, so re-probing is pure latency. A rate limit or a network blip is not the same thing, and one 429
# should not be able to kill an 8-topic batch.
WEB_PREFLIGHT_RETRY_STATUSES = ("rate", "error")

# Live question harvesting (tools 12 & 13 — search_github_questions / search_web_questions)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")   # optional; raises GitHub API rate limit from 60→5000/hr
# GitHub repos are general ML/Data-Science (Python/stats noise, no company attribution) — disabled.
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "0") == "1"
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
TAVILY_MAX_OUTCOMES = int(os.getenv("TAVILY_MAX_OUTCOMES", "14"))
TAVILY_MAX_RECORDS = int(os.getenv("TAVILY_MAX_RECORDS", "800"))

# ── Open-web tier: LAST RESORT only ──────────────────────────────────────────
# Searches WITHOUT `include_domains`, so it can reach content the 67-domain allowlist cannot — the
# banks and the allowlist hold nothing on n8n nodes, Gemini configuration or Automatic1111, which is
# exactly what several sessions teach. Measured: "n8n workflow automation interview questions" returns
# 0 allowlisted records and 24 open-web candidates, including "What is the Merge node and what merge
# modes does it support?", which matches a session outcome verbatim.
#
# It runs ONLY when the trusted tiers under-deliver (see tools.tool_search_web_questions), because open
# results are much noisier: of 71 measured candidates ALL 71 passed the form gate while only 29 came
# from a genuine interview-question page. Hence the two extra gates below.
OPEN_WEB_ENABLED = os.getenv("OPEN_WEB_ENABLED", "1") == "1"
OPEN_WEB_MAX_RECORDS = int(os.getenv("OPEN_WEB_MAX_RECORDS", "60"))
OPEN_WEB_MAX_TERMS = int(os.getenv("OPEN_WEB_MAX_TERMS", "4"))

# How short a set has to be before the open web is worth the noise. Fraction of the REQUESTED count,
# floored at MIN_QUESTIONS: at the default request of 15 the tier engages below 9.
#
# It used to engage below MIN_QUESTIONS alone, and that is why this tier — written specifically for the
# n8n gap — had never once run on a live run. Run 8fb9fcb3 asked for 15, survived with EXACTLY 5, and
# `surviving >= MIN_QUESTIONS` read that as satisfied. Landing precisely on the floor is the most
# starved a run can be while still producing output. See `pipeline._open_web_shortfall`, which keeps a
# separate `<= MIN_QUESTIONS` clause so a request of 5 cannot reintroduce the same off-by-one.
OPEN_WEB_TRIGGER_RATIO = float(os.getenv("OPEN_WEB_TRIGGER_RATIO", "0.6"))

# The page must LOOK like an interview-question page. This one test kept the 29 clean candidates and
# dropped all 42 noisy ones (forum chatter, vendor docs, tutorial headings) in the measurement above.
# It is the EDU_PLATFORM_DOMAINS "interview must be in the URL" rule, generalised to any domain.
OPEN_WEB_PAGE_TELLS = (
    "interview question", "interview-question", "interview questions", "interviewquestions",
    "/interview", "interview-prep", "questions and answers", "questions-and-answers", "q&a",
)

# Domains that produced the junk. Forums and social are conversation, not interview questions
# ("Are you willing to work on this?"); code hosts return template prose; vendor docs return
# documentation headings ("What it does?" from ai.google.dev).
OPEN_WEB_BLOCKED_DOMAINS = {
    "facebook.com", "linkedin.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "github.com", "gitlab.com", "bitbucket.org", "youtube.com",
    "pinterest.com", "slideshare.net", "scribd.com", "coursera.org", "udemy.com",
}
# Subdomain prefixes that mark a forum or a documentation site whatever the registrable domain.
OPEN_WEB_BLOCKED_PREFIXES = ("community.", "forum.", "forums.", "docs.", "developer.", "support.",
                             "help.", "status.", "blog.")

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

# Education / tutorial platforms: their non-interview pages (/topics/, /blog/, /article/, /tutorials/) are
# COURSE content, not interview questions — the Tavily extractor otherwise pulls lesson section-headings
# ("What is a Vector Database?") as questions. For these domains we accept ONLY pages whose URL path
# contains "interview".
EDU_PLATFORM_DOMAINS = {
    "scaler.com", "analyticsvidhya.com", "projectpro.io", "edureka.co", "datacamp.com",
    "towardsdatascience.com", "towardsai.net", "pub.towardsai.net", "igmguru.com", "educative.io",
    "kdnuggets.com", "simplilearn.com", "intellipaat.com", "365datascience.com",
    "machinelearningmastery.com", "mlstack.cafe", "novelvista.com", "tredence.com",
    "generativeaimasters.in", "vinsys.com",
    "geeksforgeeks.org",   # now gated too (was exempt) — only its interview-question pages, not tutorials
}

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
