"""RetrievalAgent — searches question bank + GitHub + web for real interview questions."""

from __future__ import annotations
from typing import TYPE_CHECKING

from src.agents.base_agent import BaseAgent
from src.tools import TOOL_DISPATCH, TOOL_SCHEMAS

if TYPE_CHECKING:
    from src.agent import AgentState


from src.config import GITHUB_ENABLED

# GitHub disabled by default (general ML/DS noise, no company attribution).
_TOOL_NAMES = {"search_question_bank", "search_web_questions"}
if GITHUB_ENABLED:
    _TOOL_NAMES.add("search_github_questions")
_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _TOOL_NAMES]
_DISPATCH = {k: TOOL_DISPATCH[k] for k in _TOOL_NAMES}


class RetrievalAgent(BaseAgent):
    name = "retrieval"
    display_name = "Retrieving Questions"
    max_tool_calls = 10

    def get_tool_schemas(self) -> list[dict]:
        return _SCHEMAS

    def get_tool_dispatch(self) -> dict:
        return _DISPATCH

    def run(self, state: "AgentState", emit) -> None:
        """LLM loop (bank first, then web, then GitHub per the prompt) fills a wide pool;
        a later stage scores relevance and trims to max. The curated bank goes first so it
        is never starved by web results."""
        from src.tools import tool_search_web_questions
        from src.agent import _summarize_result

        # LLM loop: bank → web → github, gathering up to the candidate pool.
        super().run(state, emit)

        # Safety net: guarantee company-attributed web questions were fetched even if the
        # LLM skipped that step (bank has already had first pick of the pool).
        outcomes = (state.session_context.learning_outcomes
                    if state.session_context else [])
        if outcomes and "web" not in state.raw_fetched:
            emit("search_web_questions", "running",
                 "Retrieving Questions: fetching company-attributed questions from Tavily...")
            result = tool_search_web_questions(state, outcomes)
            state.tool_log.append({"agent": "retrieval", "tool": "search_web_questions",
                                   "args_keys": ["outcomes"], "has_error": "error" in result})
            emit("search_web_questions", "done", _summarize_result("search_web_questions", result))

    def get_system_prompt(self, state: AgentState) -> str:
        if not state.session_context:
            return "No session context available — do not call any tools."

        from src.config import BANK_POOL_CAP, WEB_POOL_CAP, GITHUB_POOL_CAP
        ctx = state.session_context
        min_q = state.config.min_questions
        max_q = state.config.max_questions

        queries = "\n".join(f"  - {q}" for q in (state.suggested_queries or [ctx.session_name]))

        bank_hint = (
            "The question bank has matches for this session — start with search_question_bank."
            if state.has_bank_questions
            else "Bank may be thin for this topic — use all three sources."
        )

        return f"""You are a question retrieval specialist. Find REAL interview questions from verified sources.

## Session: {ctx.session_name}  |  Type: {ctx.session_type}
## Final target: {min_q}–{max_q} questions — but first gather a WIDE candidate pool from BOTH sources.
A later stage scores every candidate for relevance to the reading material and trims to the
final {max_q}. So your job here is RECALL: collect many plausible candidates from every source.
Do NOT stop early at {max_q}. Each source has its own quota, so use BOTH — a source being
"full" does NOT mean you are done; move to the next source.

## Retrieval Strategy — use BOTH sources (each has an independent quota)
1. `search_question_bank` (up to {BANK_POOL_CAP}) — the curated interview data; call it 3–5 times, once per query below.
2. `search_web_questions` (up to {WEB_POOL_CAP}) — ALWAYS call it for real company-attributed questions (Glassdoor, AmbitionBox, Exponent…). This is the freshest source; never skip it.
   Pass SHORT topic keywords (2–4 words each), NOT full outcome sentences.
   Good: ["LangChain RAG", "AI agents memory", "RAG retrieval augmented generation"]
   Bad:  ["Implement LangChain RecursiveCharacterTextSplitter", "Build RAG pipelines using LangChain"]
3. Stop only when both sources are exhausted or at their quota.

## Queries to Use
{queries}

## Rules
- DO NOT generate questions — only retrieve from real sources (generation is disabled)
- Use the exact queries listed above; do not invent generic terms
- Do NOT repeat the same query twice
- {bank_hint}
- web search brings real company interview questions with attribution — always run it"""

    def get_user_prompt(self, state: AgentState) -> str:
        if not state.session_context:
            return "No session context — skip all searches."
        ctx = state.session_context
        queries = (state.suggested_queries or [ctx.session_name])[:3]
        q_hint = ", ".join(f'"{q}"' for q in queries)
        return (
            f"Retrieve questions for: {ctx.session_name}.\n"
            f"Queries: {q_hint}.\n"
            f"Start with search_question_bank."
        )
