"""Agentic workflow — single agent with tools that autonomously curates interview questions.

The agent drives the workflow via OpenRouter tool_use. Python code only
executes tools and manages state — it doesn't make workflow decisions.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Callable

from src.models import (
    GenerationConfig, SessionContext, QuestionDetail, CodingQuestion, CodeSnippet,
    CuratedOutput, CurationMetadata, QualityReport, FlaggedQuestion,
)
from src.data_loader import DataStore, get_data_store
from src.llm_client import get_client, chat_completion_json
from src.config import LLM_MODEL, MAX_TOOL_CALLS, MIN_QUESTIONS, MAX_QUESTIONS
from src.tools import TOOL_SCHEMAS, TOOL_DISPATCH

MAX_REVISION_ROUNDS = 2


# ── Agent State ─────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state accumulated across all agents in the pipeline."""
    config: GenerationConfig
    data_store: DataStore
    questions: dict[str, QuestionDetail] = field(default_factory=dict)
    coding_questions: dict[str, CodingQuestion] = field(default_factory=dict)
    code_snippets: dict[str, CodeSnippet] = field(default_factory=dict)
    learning_outcomes: list[str] = field(default_factory=list)
    session_context: SessionContext | None = None
    has_bank_questions: bool = True
    submitted: bool = False
    dedup_removed: int = 0
    removed: list[dict] = field(default_factory=list)  # rejected questions {content, reason, stage, ...}
    removed_by_relevance: int = 0
    # Validated-but-not-selected candidates, kept so the quality-gate revision can BACKFILL
    # replacements (instead of only shrinking). excluded = ids never to re-add (flagged/rejected).
    reserve: dict = field(default_factory=dict)      # q_id -> QuestionDetail
    excluded: set = field(default_factory=set)       # q_ids removed/rejected, never re-added
    tool_log: list[dict] = field(default_factory=list)
    # Retrieval funnel stats (observability): {source: raw_hit_count} + max pool reached
    raw_fetched: dict = field(default_factory=dict)
    pool_size: int = 0
    # Candidates actually ADDED to the pool per source ("bank"/"web"/"github") — used to
    # enforce per-source pool caps so no one source monopolises the pool.
    added_by_source: dict = field(default_factory=dict)
    # Per-session reading-material text (multi-session topics) — used to attribute each
    # question to the session it best matches, so final selection represents every session.
    session_profiles: dict = field(default_factory=dict)   # session_name -> profile text
    # Web-search health for this run (surfaced in the report/UI so a bank-only fallback is visible):
    # not_run | ok | empty | no_key | quota | auth | rate | full | error
    web_status: str = "not_run"
    web_error: str | None = None   # human-readable Tavily failure detail, if any
    web_search_disabled: bool = False   # set when the pre-flight Tavily health check fails (skip web calls)
    # Pipeline-level state (set by UnderstandingAgent, read by RetrievalAgent)
    suggested_queries: list[str] = field(default_factory=list)
    # Quality gate revision instructions (set by pipeline, read by EvaluationAgent)
    revision_notes: list[dict] = field(default_factory=list)
    # API usage tracking across all agents and tool calls in this run
    api_usage: dict = field(default_factory=lambda: {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tavily_calls": 0,
    })

    @property
    def total_questions(self) -> int:
        return len(self.questions) + len(self.coding_questions)

    @property
    def remaining_capacity(self) -> int:
        from src.config import pool_target
        return pool_target(self.config.max_questions) - self.total_questions

    @property
    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions.values():
            counts[q.source] = counts.get(q.source, 0) + 1
        for q in self.coding_questions.values():
            counts[q.source] = counts.get(q.source, 0) + 1
        return counts

    @property
    def difficulty_counts(self) -> dict[str, int]:
        counts = {"Easy": 0, "Medium": 0, "Hard": 0}
        for q in self.questions.values():
            d = q.difficulty or "Medium"
            counts[d] = counts.get(d, 0) + 1
        for q in self.coding_questions.values():
            d = q.difficulty or "Medium"
            counts[d] = counts.get(d, 0) + 1
        return counts

    def to_curated_output(self) -> CuratedOutput:
        return CuratedOutput(
            question_details=list(self.questions.values()),
            coding_questions=list(self.coding_questions.values()),
            code_snippets=list(self.code_snippets.values()),
            metadata=CurationMetadata(
                total_candidates=self.total_questions,
                dedup_removed=self.dedup_removed,
                source_counts=self.source_counts,
                questions_from_web=sum(1 for q in self.questions.values() if q.source == "web"),
                raw_fetched=dict(self.raw_fetched),
                pool_size=self.pool_size,
                removed_by_relevance=self.removed_by_relevance,
                removed_by_dedup=self.dedup_removed,
            ),
        )


# ── Pipeline Result ─────────────────────────────────────────────────────────

class PipelineResult:
    def __init__(self):
        self.run_id: str = str(uuid.uuid4())
        self.context: SessionContext | None = None
        self.local_pool = None
        self.web_pool = None
        self.curated_output: CuratedOutput | None = None
        self.quality_report: QualityReport | None = None
        self.approved: bool = False
        self.error: str | None = None
        self.awaiting_gate: bool = False   # TESTING: preview mode — picked, not yet gated
        self.removed: list = []             # rejected questions + reasons (for Review transparency)
        self.category: str = "GEN_AI"       # course category → sheet branding


EmitFn = Callable[[str, str, str, str], None]


# ── Critique Gate ───────────────────────────────────────────────────────────

def _critique_question_set(state: AgentState) -> dict:
    """LLM critiques the final set. Returns pass/fail + must_fix list."""
    if not state.session_context:
        return {"pass": True, "must_fix": []}

    outcomes = "\n".join(f"- {o}" for o in state.session_context.learning_outcomes)
    topics = ", ".join(getattr(state.session_context, "interview_topics", None) or []) or "(same as outcomes)"
    q_list = [{"id": q.question_id, "content": q.content[:200], "difficulty": q.difficulty}
              for q in state.questions.values()]
    cq_list = [{"id": q.id, "title": q.title, "difficulty": q.difficulty}
               for q in state.coding_questions.values()]

    result = chat_completion_json(
        system_prompt=f"""You are a quality gate for interview question sets.

Session: {state.session_context.session_name}
Learning Outcomes:
{outcomes}
Interview Topics (transferable concepts this session prepares for — questions testing these are ON-topic
even if they don't name the specific tool/product used in the session): {topics}

Check ONLY these hard failures — only flag something if it is CLEARLY wrong:
1. A question is from a completely different domain (e.g. a SQL question in an AI agents session).
   A question about one of the Interview Topics is NOT a different domain — do not flag it.
2. Two questions are near-identical duplicates (same wording, not just same topic)
3. Total set has fewer than {MIN_QUESTIONS} questions

DO NOT flag questions as off-topic just because they don't match a specific outcome word-for-word.
Questions that are topically related to the session's subject area should PASS.
If the set looks reasonable, return pass=true with an empty must_fix list.

Respond in JSON:
{{
    "pass": true/false,
    "must_fix": [
        {{"id": "...", "issue": "off-topic / duplicate / too-few", "suggestion": "..."}}
    ],
    "summary": "One-line overall verdict"
}}""",
        user_prompt=f"Theory questions:\n{json.dumps(q_list)}\n\nCoding questions:\n{json.dumps(cq_list)}",
        max_tokens=1500,
        temperature=0.0,
        on_usage=lambda u: (
            state.api_usage.__setitem__("llm_calls", state.api_usage["llm_calls"] + 1),
            state.api_usage.__setitem__("prompt_tokens", state.api_usage["prompt_tokens"] + (u.prompt_tokens or 0)),
            state.api_usage.__setitem__("completion_tokens", state.api_usage["completion_tokens"] + (u.completion_tokens or 0)),
        ),
    )

    return result


# ── Context Trimming ────────────────────────────────────────────────────────

def _compact_tool_content(content: str, max_len: int = 1500) -> str:
    """Trim tool result content to keep conversation context small."""
    if len(content) <= max_len:
        return content
    return content[:max_len] + '..."}'



# ── Helpers ─────────────────────────────────────────────────────────────────

def _msg_to_dict(msg) -> dict:
    """Convert an OpenAI message object to a serializable dict."""
    d = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _summarize_result(tool_name: str, result: dict) -> str:
    """Short summary of a tool result for the progress UI."""
    if "error" in result:
        return f"Error: {result['error'][:80]}"

    summaries = {
        "understand_session": lambda r: f"{r.get('session_type','')} session — {len(r.get('learning_outcomes',[]))} outcomes, {len(r.get('matched_kps',[]))} KPs",
        "search_question_bank": lambda r: f"Found {r.get('found', 0)} relevant (total: {r.get('total_accumulated', 0)}, remaining: {r.get('remaining_capacity', '?')})",
        "validate_relevance": lambda r: f"Kept {r.get('kept', 0)}, removed {r.get('removed', 0)} irrelevant",
        "deduplicate_questions": lambda r: f"Kept {r.get('kept', 0)}, removed {r.get('removed', 0)} duplicates",
        "check_difficulty_balance": lambda r: f"E:{r.get('counts',{}).get('Easy',0)} M:{r.get('counts',{}).get('Medium',0)} H:{r.get('counts',{}).get('Hard',0)} {'OK' if r.get('balanced') else 'Fix'}",
        "check_outcome_coverage": lambda r: f"{r.get('covered',0)}/{r.get('total_outcomes',0)} outcomes covered",
        "generate_expected_answers": lambda r: f"Generated {r.get('generated', 0)} answers",
        "generate_interview_questions": lambda r: f"Generated {r.get('generated', 0)} interview questions (total: {r.get('total_accumulated', '?')})",
        "generate_coding_questions": lambda r: f"Generated {r.get('generated', 0)} coding questions",
        "remove_question": lambda r: f"Removed — {r.get('remaining', '?')} left",
        "submit_question_set": lambda r: f"Submitted {r.get('total_questions',0)} ({r.get('theory',0)}T + {r.get('coding',0)}C)",
    }

    fn = summaries.get(tool_name)
    if fn:
        try:
            return fn(result)
        except Exception:
            pass
    return json.dumps(result)[:100]
