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
from src.config import FINAL_SET_CAP, MIN_QUESTIONS
from src.tools import TOOL_SCHEMAS, TOOL_DISPATCH

# NOTE: the live revision-round limit is pipeline.MAX_REVISION_ROUNDS. A duplicate constant used to
# sit here, unused, where it could silently diverge from the one that actually applies.


# ── Agent State ─────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state accumulated across all agents in the pipeline."""
    config: GenerationConfig
    data_store: DataStore
    questions: dict[str, QuestionDetail] = field(default_factory=dict)
    # DORMANT: nothing assigns these today. Coding-question GENERATION is permanently blocked, and no
    # retrieval path produces them, so they are always empty — which makes the CodingQuestion /
    # CodeSnippet sheet tabs and the coding branches in selection and review unreachable. Kept because
    # the exported sheet's tab structure is part of the LMS unit import format, and because retrieved
    # (not generated) coding questions remain the intended way to fill them.
    coding_questions: dict[str, CodingQuestion] = field(default_factory=dict)
    code_snippets: dict[str, CodeSnippet] = field(default_factory=dict)
    learning_outcomes: list[str] = field(default_factory=list)
    session_context: SessionContext | None = None
    has_bank_questions: bool = True
    # True once the final set has been SELECTED (ranked + trimmed) by tool_submit_question_set.
    # Read by pipeline._enforce_submission — if the Evaluation agent never called submit, the
    # pipeline calls it directly rather than serializing the raw candidate pool.
    submitted: bool = False
    submit_forced: bool = False    # the pipeline had to submit on the agent's behalf
    # Quality-gate verdict, carried into QualityReport so it reaches the reviewer instead of
    # existing only in the SSE log.
    gate_forced: bool = False          # shipped despite an unresolved gate failure
    gate_issues: list[dict] = field(default_factory=list)   # unresolved must_fix entries
    gate_summary: str = ""             # the gate's one-line verdict
    relevance_scored: bool = True      # False when the relevance judge failed for EVERY batch
    # Phases that died on an API error. A phase used to fail silently and let the run continue with a
    # partial pool, so a retrieval outage looked like "this session just has few questions".
    phase_errors: list[str] = field(default_factory=list)
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

def _deterministic_gate_issues(state: AgentState) -> list[dict]:
    """Gate checks that don't need an LLM — run first, because they are exact and free.

    The previous gate asked the model for all of its checks, and the only three it asked about
    (off-domain, near-identical wording, too-few) were each already enforced deterministically
    upstream, so it contributed no signal. These are the checks nothing else makes.
    """
    from src.quality import is_quality_question

    issues: list[dict] = []
    questions = list(state.questions.values())
    total = len(questions) + len(state.coding_questions)
    min_q = getattr(state.config, "min_questions", MIN_QUESTIONS) or MIN_QUESTIONS
    max_q = min(getattr(state.config, "max_questions", None) or FINAL_SET_CAP, FINAL_SET_CAP)

    if total < min_q:
        issues.append({"id": None, "issue": "too-few",
                       "suggestion": f"Only {total} question(s); this run asked for at least {min_q}. "
                                     f"The on-topic pool was too thin — do NOT remove more."})
    # Nothing else checks the upper bound. If selection didn't run, this is what catches it.
    if total > max_q:
        issues.append({"id": None, "issue": "too-many",
                       "suggestion": f"{total} questions exceeds the requested {max_q} — the set was "
                                     f"never trimmed. Re-run submit_question_set."})
    # A total relevance-judge failure leaves every candidate at the neutral default score, which
    # otherwise looks like a clean pass.
    if not state.relevance_scored:
        issues.append({"id": None, "issue": "unscored",
                       "suggestion": "The relevance judge failed for every batch, so nothing in this "
                                     "set was actually scored for topical fit."})
    # Form garbage — page headings, fragments, blog titles. The bank is gated at build time, but
    # web-harvested candidates reach here too.
    for q in questions:
        if not is_quality_question(q.content):
            issues.append({"id": q.question_id, "issue": "malformed",
                           "suggestion": "Not a well-formed standalone interview question "
                                         "(heading, title or fragment) — remove it."})
    # Per-session representation for a combined run: a session contributing nothing means the
    # reviewer gets a set that doesn't cover what they selected.
    selected = [s for s in (getattr(state.config, "session_names", None) or []) if s]
    if len(selected) > 1:
        covered = {q.session for q in questions if q.session}
        missing = [s for s in selected if s not in covered]
        # Only flag when attribution actually ran; an all-None `session` field means it didn't.
        if covered and missing:
            issues.append({"id": None, "issue": "session-gap",
                           "suggestion": f"No questions represent: {', '.join(missing)}."})
    return issues


def _critique_question_set(state: AgentState) -> dict:
    """Quality gate over the final set. Returns {pass, must_fix, summary}.

    Deterministic checks run first (`_deterministic_gate_issues`); the LLM is asked only about the
    things that genuinely need judgement — off-domain drift and semantic duplicates.

    Fails CLOSED: an unparseable or errored critique returns `pass=False`. It used to be read via
    `.get("pass", True)` against a `{}` returned on any failure, so every LLM hiccup became a
    silent approval.
    """
    if not state.questions and not state.coding_questions:
        return {"pass": False, "must_fix": [{"id": None, "issue": "empty",
                                             "suggestion": "The set is empty."}],
                "summary": "No questions were produced."}

    issues = _deterministic_gate_issues(state)

    if not state.session_context:
        # Without a resolved session there is nothing to judge topical fit against; report what the
        # deterministic checks found rather than claiming a pass.
        return {"pass": not issues, "must_fix": issues,
                "summary": "No session context — only structural checks ran."}

    outcomes = "\n".join(f"- {o}" for o in state.session_context.learning_outcomes)
    topics = ", ".join(getattr(state.session_context, "interview_topics", None) or []) or "(same as outcomes)"
    q_list = [{"id": q.question_id, "content": q.content[:400], "difficulty": q.difficulty}
              for q in state.questions.values()]
    cq_list = [{"id": q.id, "title": q.title, "difficulty": q.difficulty}
               for q in state.coding_questions.values()]

    try:
        result = chat_completion_json(
            system_prompt=f"""You are a quality gate for interview question sets. Judge ONLY the two
things below — structural checks (set size, form, coverage) already ran separately.

Session: {state.session_context.session_name}
Learning Outcomes:
{outcomes}
Interview Topics (transferable concepts this session prepares for — a question testing one of these
is ON-topic even if it doesn't name the specific tool or product used in the session): {topics}

1. OFF-DOMAIN — the question tests a different technology or domain than this session teaches.
   Judge against the outcomes and interview topics above. A question that is broad but still about
   this session's subject matter is NOT off-domain. Flag a question that would make a reviewer ask
   "why is this here?" — for example a SQL-joins question in an AI-agents session, or a generic
   "describe a project you built" question in any session.
2. DUPLICATE — two questions test the same thing. Include REWORDINGS, not just identical wording:
   "What are the parts of an agent?" and "What are an AI agent's core components?" are duplicates.
   Flag the weaker of the pair.

Flag what is genuinely wrong. Do not invent problems to look thorough, and do not pass a set you
would not hand to an interviewer.

Respond in JSON:
{{
    "pass": true/false,
    "must_fix": [{{"id": "...", "issue": "off-domain | duplicate", "suggestion": "..."}}],
    "summary": "One-line overall verdict"
}}""",
            user_prompt=f"Theory questions:\n{json.dumps(q_list)}\n\nCoding questions:\n{json.dumps(cq_list)}",
            max_tokens=2000,
            temperature=0.0,
            on_usage=lambda u: _record_usage(state, u),
        )
    except Exception as exc:  # noqa: BLE001 — includes JSONResponseError
        issues.append({"id": None, "issue": "gate-error",
                       "suggestion": f"The quality gate could not run ({type(exc).__name__}). The "
                                     f"set was NOT checked for off-domain questions or duplicates."})
        return {"pass": False, "must_fix": issues,
                "summary": f"Quality gate failed to run: {exc}"}

    llm_issues = [i for i in (result.get("must_fix") or []) if isinstance(i, dict)]
    all_issues = issues + llm_issues
    return {
        "pass": not all_issues,
        "must_fix": all_issues,
        "summary": result.get("summary", "") or f"{len(all_issues)} issue(s) found.",
    }


def _record_usage(state: AgentState, u) -> None:
    """Accumulate one LLM call's token usage onto the run."""
    state.api_usage["llm_calls"] = state.api_usage.get("llm_calls", 0) + 1
    state.api_usage["prompt_tokens"] = state.api_usage.get("prompt_tokens", 0) + (u.prompt_tokens or 0)
    state.api_usage["completion_tokens"] = (state.api_usage.get("completion_tokens", 0)
                                            + (u.completion_tokens or 0))


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
