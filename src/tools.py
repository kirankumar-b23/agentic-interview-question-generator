"""Tool definitions + implementations for the agentic workflow.

13 tools the agent calls autonomously via OpenRouter tool_use:
1. understand_session — extract outcomes, KPs, session type (MUST be first)
2. search_question_bank — TF-IDF search with session-aware relevance filtering
3. search_github_questions — fetch questions from curated GitHub interview repos
4. search_web_questions — Tavily web search (real, company-attributed questions)
5. validate_relevance — LLM checks each question against session outcomes
6. deduplicate_questions — TF-IDF dedup
7. check_difficulty_balance — Easy/Medium/Hard distribution
8. check_outcome_coverage — which outcomes are covered
9. generate_expected_answers — LLM generates answer outlines
10. generate_interview_questions — BLOCKED (real questions only; always returns blocked)
11. generate_coding_questions — LLM generates coding problems (code-heavy sessions only)
12. remove_question — drop a specific question
13. submit_question_set — finalize and end the run
"""

from __future__ import annotations
import json
import uuid
from typing import TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import QuestionDetail, CodingQuestion
from src.llm_client import chat_completion_json


def _usage_cb(state: "AgentState"):
    """Return an on_usage callback that accumulates LLM token stats into state.api_usage."""
    def _cb(usage):
        state.api_usage["llm_calls"] += 1
        state.api_usage["prompt_tokens"] += usage.prompt_tokens or 0
        state.api_usage["completion_tokens"] += usage.completion_tokens or 0
    return _cb


from src.config import DEDUP_THRESHOLD, pool_target
from src.question_bank import get_retriever
from src.sources.base import split_into_clauses, looks_like_question

if TYPE_CHECKING:
    from src.agent import AgentState


# ── Tool Schemas ────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "understand_session",
            "description": "Analyze session reading material to extract learning outcomes, Knowledge Points (KPs), key concepts, scope boundaries, and session type. Returns structured context + suggested search queries. MUST be called FIRST before any search.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_question_bank",
            "description": "Search the pre-indexed question bank (~2000+ questions). Uses TF-IDF similarity + session-aware relevance filtering. Use the suggested_search_queries from understand_session as queries. Do NOT invent generic terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A KP label or learning outcome from understand_session (e.g., 'Configure Gemini API authentication and key management'). Be specific."
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"],
                        "description": "Optional difficulty filter"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 8)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_github_questions",
            "description": "Search verified GitHub repositories for real interview questions matching the given learning outcomes. Use when the question bank is thin for a topic — returns questions as actually written in open-source interview prep repos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Learning outcome phrases to search for (e.g. ['rag evaluation metrics', 'prompt injection defense']). Use plain phrases, not snake_case."
                    }
                },
                "required": ["outcomes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_questions",
            "description": "Search 65+ verified domains (Glassdoor, AmbitionBox, Exponent, GeeksforGeeks, LeetCode, etc.) for real interview questions with company attribution via Tavily. Requires TAVILY_API_KEY. Use when you need questions that came from actual company interviews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Learning outcome phrases to search for (e.g. ['RAG evaluation', 'LLM fine-tuning'])."
                    }
                },
                "required": ["outcomes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_relevance",
            "description": "LLM evaluates ALL accumulated questions against the session's learning outcomes. Removes questions that don't match any outcome. Call AFTER all searches, BEFORE submitting.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deduplicate_questions",
            "description": "Remove near-duplicate questions (cosine similarity > 0.85).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_difficulty_balance",
            "description": "Check current Easy/Medium/Hard distribution against target (30/50/20).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_outcome_coverage",
            "description": "Check which learning outcomes are covered by current questions and which are missing.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_expected_answers",
            "description": "Generate 2-3 bullet answer outlines for questions that lack them.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_coding_questions",
            "description": "Generate coding interview problems. Only for code-heavy or mixed sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of coding questions (1-4)"},
                    "topics": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Specific topics from the session"
                    },
                    "language": {"type": "string", "description": "Programming language (default: Python)"}
                },
                "required": ["count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "LLM generates theory/conceptual interview questions based on session learning outcomes. Use when the question bank doesn't have enough relevant questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of questions to generate (1-10)"},
                    "outcomes": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Learning outcomes to generate questions for"
                    },
                    "difficulty_mix": {
                        "type": "string",
                        "description": "Difficulty distribution hint, e.g., '3 Easy, 4 Medium, 3 Hard'"
                    }
                },
                "required": ["count", "outcomes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_question",
            "description": "Remove a specific question from the set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["question_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_question_set",
            "description": "Finalize and submit the question set for human review. THIS ENDS THE RUN. Only call when you have 5-15 relevant questions.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ── Tool Implementations ────────────────────────────────────────────────────

def tool_understand_session(state: AgentState) -> dict:
    """Step 1: LLM reads session RM, extracts outcomes, maps to KPs."""
    from src.session_understanding import understand_session

    context = understand_session(state.config.session_names, state.data_store)
    state.session_context = context
    state.learning_outcomes = context.learning_outcomes

    # Per-session profile text (multi-session topics) for attributing each question to the
    # session it best matches — enables per-session representation in final selection.
    if len(state.config.session_names) > 1:
        for name in state.config.session_names:
            state.session_profiles[name] = (state.data_store.get_session_content(name) or name)[:4000]

    # Ground retrieval queries in what the READING MATERIAL actually teaches — the
    # key_concepts / scope_in extracted from the RM — not the loosely-mapped global KP
    # labels (which pulled off-topic questions). KP labels are only a last-resort fallback.
    concept_queries = list(dict.fromkeys(
        [c for c in (context.key_concepts + context.scope_in) if c and c.strip()]
    ))
    if not concept_queries:
        concept_queries = [kp.kp_label for kp in context.matched_kp_ids if kp.relevance >= 0.5]
    # Interview-phrased variants improve matching against company question content
    interview_queries = [f"{q} interview" for q in concept_queries[:3]]
    all_queries = (concept_queries + interview_queries)[:8]

    # Probe the bank with the top query to estimate coverage for this session
    has_bank_questions = False
    estimated_bank_count = 0
    if all_queries:
        from src.question_bank import get_retriever_for
        retriever = get_retriever_for(getattr(state.config, "category", None))
        probe = retriever.search(all_queries[0], limit=5)
        estimated_bank_count = len(probe) * min(len(all_queries), 5)
        has_bank_questions = len(probe) > 0

    state.has_bank_questions = has_bank_questions

    bank_hint = (
        f"Bank has ~{estimated_bank_count}+ potential matches for this session."
        if has_bank_questions
        else "Bank may have few/no questions for this topic — prioritise search_github_questions and search_web_questions after bank searches."
    )

    return {
        "session_name": context.session_name,
        "session_type": context.session_type,
        "learning_outcomes": context.learning_outcomes,
        "key_concepts": context.key_concepts,
        "scope_in": context.scope_in,
        "scope_out": context.scope_out,
        "matched_kps": [
            {"kp_id": kp.kp_id, "label": kp.kp_label, "relevance": kp.relevance}
            for kp in context.matched_kp_ids
        ],
        "suggested_search_queries": all_queries,
        "has_bank_questions": has_bank_questions,
        "estimated_bank_question_count": estimated_bank_count,
        "instruction": f"Use suggested_search_queries for search_question_bank. {bank_hint}",
    }


def tool_search_question_bank(state: AgentState, query: str, difficulty: str = None, limit: int = 8) -> dict:
    """Search question bank with TF-IDF + session-aware relevance filtering.
    GEN_AI sessions use the curated GenAI bank; others use the Python/SWE bank."""
    from src.question_bank import get_retriever_for
    retriever = get_retriever_for(getattr(state.config, "category", None))

    from src.config import BANK_POOL_CAP
    bank_have = state.added_by_source.get("bank", 0)
    max_to_add = BANK_POOL_CAP - bank_have
    if max_to_add <= 0:
        return {"found": 0, "warning": f"Bank quota full ({BANK_POOL_CAP}). Use web/github or submit."}

    actual_limit = min(limit, max_to_add, 8)
    exclude_ids = set(state.questions.keys())

    results = retriever.search(
        query=query, difficulty=difficulty,
        limit=actual_limit + 5,  # fetch extra for filtering
        exclude_ids=exclude_ids,
    )

    # Session-aware post-retrieval relevance filter
    scope_keywords: set[str] = set()
    if state.session_context:
        for term in (state.session_context.learning_outcomes +
                     state.session_context.key_concepts +
                     state.session_context.scope_in):
            scope_keywords.update(
                w.lower() for w in term.split()
                if len(w) >= 4 and w.lower() not in {
                    "with", "that", "this", "from", "what", "have", "will",
                    "about", "between", "their", "using", "should", "could",
                    "would", "does", "they", "been", "more", "than", "also",
                    "each", "when", "which", "into", "some", "other",
                }
            )

        filtered = []
        for qd in results:
            content_lower = qd.content.lower()
            matches = sum(1 for kw in scope_keywords if kw in content_lower)
            # Require at least 1 keyword match for short Qs, 2 for longer ones
            min_required = 1 if len(qd.content) < 100 else 2
            if matches >= min_required:
                filtered.append(qd)
        results = filtered[:actual_limit]

    added = []
    for qd in results:
        # Split a compound bank question down to its on-topic clause (same as web).
        if scope_keywords:
            trimmed = _trim_to_topic(qd.content, scope_keywords)
            if trimmed:
                qd.content = trimmed
        state.questions[qd.question_id] = qd
        added.append({
            "id": qd.question_id,
            "content": qd.content[:150],
            "difficulty": qd.difficulty,
            "topic": qd.topic,
            "source": qd.source,
        })

    state.raw_fetched["bank"] = state.raw_fetched.get("bank", 0) + len(added)
    state.added_by_source["bank"] = state.added_by_source.get("bank", 0) + len(added)
    total = len(state.questions) + len(state.coding_questions)
    state.pool_size = max(state.pool_size, total)
    return {
        "found": len(added),
        "questions": added,
        "total_accumulated": total,
        "bank_remaining": BANK_POOL_CAP - state.added_by_source.get("bank", 0),
    }


def tool_validate_relevance(state: AgentState) -> dict:
    """LLM evaluates each question's relevance to session outcomes."""
    if not state.session_context or not state.questions:
        return {"error": "Call understand_session first and gather questions"}

    from src import memory as _memory
    learned_rules = _memory.get_learned_rules()
    rules_block = ""
    if learned_rules:
        rules_block = (
            "## Learned rejection rules (apply these first):\n"
            + "\n".join(f"- {r}" for r in learned_rules[:20])
            + "\n\n"
        )

    outcomes_str = "\n".join(f"- {o}" for o in state.session_context.learning_outcomes)
    concepts_str = ", ".join(state.session_context.key_concepts)
    scope_out_str = ", ".join(state.session_context.scope_out) if state.session_context.scope_out else "none"

    # Ground the judge in the ACTUAL reading material (the outcomes list alone is too thin —
    # it lets keyword overlap masquerade as relevance). Include a bounded excerpt of each
    # selected session's RM, split evenly to keep total cost in check.
    names = state.config.session_names or [state.session_context.session_name]
    per_cap = max(1200, 4000 // max(1, len(names)))
    rm_parts = []
    for name in names:
        content = state.data_store.get_session_content(name)
        if content:
            rm_parts.append(f"### {name}\n{content[:per_cap]}")
    rm_block = "\n\n".join(rm_parts) if rm_parts else "(no reading material found — use the outcomes/concepts above)"

    system_prompt = f"""{rules_block}You score interview questions by how well they test what a SPECIFIC session actually teaches.

Session: {state.session_context.session_name}

Learning Outcomes:
{outcomes_str}

Key Concepts: {concepts_str}
Out of Scope (score these ≤0.2): {scope_out_str}

## What this session actually teaches (reading material — judge against THIS, not just keywords)
{rm_block}

Give EACH question a relevance score from 0.0 to 1.0 for how well it tests a concept the session teaches:
- 0.8–1.0 — directly tests a technical concept explained in the reading material above.
- 0.4–0.7 — genuinely about the session's subject matter but broad or only partially on-topic.
- 0.0–0.2 — NOT testing this session's subject matter. Score LOW even if a keyword overlaps and even
  if it came from a real company. This INCLUDES:
    * behavioral / HR / teamwork / leadership / "tell me about yourself",
    * interview-logistics or process questions — e.g. "can you share your screen and open the app",
      "walk me through your project/portal", "can you use AI tools during the interview",
      "experience contributing to open source",
    * product-management / roadmap / prioritization,
    * anything in the Out-of-Scope list, or a different technology/domain than this session.

CRITICAL: keyword overlap is NOT relevance. "Share your screen and open the payment portal" is an
interview-logistics question and scores ~0.1 even though the session mentions AI screen-sharing.
The question must test the session's actual subject matter.

Also tag each question's difficulty for THIS session's level:
- "Easy" — recall/definition of a single concept.
- "Medium" — apply/compare concepts or explain how something works.
- "Hard" — multi-step reasoning, trade-offs, design, or deep internals.
Score AND tag every item by its number.

Respond in JSON only:
{{"scores": [{{"n": 1, "score": 0.0, "difficulty": "Easy|Medium|Hard"}}]}}"""

    # The pool can be large (bank+web+github) — score it in batches so no single call
    # truncates. Each batch numbers items 1..N locally and we map back to the q_id.
    from src.config import RELEVANCE_BATCH_SIZE
    items = list(state.questions.items())   # [(q_id, QuestionDetail), ...]
    score_by_qid: dict[str, float] = {}
    diff_by_qid: dict[str, str] = {}
    _valid_diff = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
    for start in range(0, len(items), RELEVANCE_BATCH_SIZE):
        batch = items[start:start + RELEVANCE_BATCH_SIZE]
        numbered = [{"n": i + 1, "q": q.content[:220]} for i, (_, q) in enumerate(batch)]
        result = chat_completion_json(
            system_prompt=system_prompt,
            user_prompt=f"Score these {len(numbered)} questions:\n{json.dumps(numbered)}",
            max_tokens=2560,
            on_usage=_usage_cb(state),
        )
        for s in (result.get("scores") or []):
            try:
                n = int(s["n"]); score = max(0.0, min(1.0, float(s["score"])))
            except (TypeError, ValueError, KeyError):
                continue
            if 1 <= n <= len(batch):
                q_id = batch[n - 1][0]
                score_by_qid[q_id] = score
                d = _valid_diff.get(str(s.get("difficulty", "")).strip().lower())
                if d:
                    diff_by_qid[q_id] = d

    from src.config import RELEVANCE_THRESHOLD
    THRESHOLD = RELEVANCE_THRESHOLD
    min_keep = max(1, state.config.min_questions)

    # Apply scores + LLM difficulty tags (difficulty was hard-coded "Medium" for web/github;
    # the content-based tag makes difficulty real across all sources).
    scored = []  # (q_id, score)
    for q_id, q in items:
        score = score_by_qid.get(q_id, 0.5)
        q.relevance_score = score
        if q_id in diff_by_qid:
            q.difficulty = diff_by_qid[q_id]
        scored.append((q_id, score))

    # Drop below-threshold candidates, but keep at least `min_keep` (backfill with the
    # highest-scored below-threshold ones so a wide-but-weak pool never underfills).
    scored.sort(key=lambda t: t[1], reverse=True)
    keep_ids = {q_id for q_id, s in scored if s >= THRESHOLD}
    if len(keep_ids) < min_keep:
        for q_id, s in scored:
            if len(keep_ids) >= min_keep:
                break
            keep_ids.add(q_id)

    to_remove = [q_id for q_id, s in scored if q_id not in keep_ids]
    for q_id in to_remove:
        q = state.questions.pop(q_id, None)
        if q is not None:
            state.removed.append({
                "content": q.content,
                "reason": f"Below relevance threshold ({q.relevance_score:.2f}) for this session",
                "stage": "relevance", "difficulty": q.difficulty, "company": q.attribution,
                "relevance_score": q.relevance_score,
            })

    # Attribute each surviving question to the session it best matches (per-session
    # representation in final selection). Profiles exist only for multi-session topics.
    if state.session_profiles:
        _attribute_sessions(list(state.questions.values()), state.session_profiles)

    state.removed_by_relevance += len(to_remove)
    return {
        "kept": len(state.questions),
        "removed": len(to_remove),
        "removed_ids": to_remove,
        "remaining_total": len(state.questions) + len(state.coding_questions),
    }


def tool_deduplicate_questions(state: AgentState) -> dict:
    if len(state.questions) <= 1:
        return {"kept": len(state.questions), "removed": 0}

    questions = list(state.questions.values())
    texts = [q.content for q in questions]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    source_priority = {"interview_db": 0, "web": 1, "generated": 2}

    def _keep_rank(q):
        # Higher relevance wins; tie-break by source priority (lower = better).
        return (-(q.relevance_score if q.relevance_score is not None else 0.0),
                source_priority.get(q.source, 9))

    to_remove = set()
    for i in range(len(questions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(questions)):
            if j in to_remove:
                continue
            if sim_matrix[i][j] > DEDUP_THRESHOLD:
                # Drop the weaker of the pair (keep higher relevance / better source).
                worse = j if _keep_rank(questions[i]) <= _keep_rank(questions[j]) else i
                to_remove.add(worse)

    for idx in to_remove:
        q = state.questions.pop(questions[idx].question_id, None)
        if q is not None:
            state.removed.append({
                "content": q.content, "reason": "Near-duplicate of another question",
                "stage": "duplicate", "difficulty": q.difficulty, "company": q.attribution,
            })

    within_run_removed = len(to_remove)

    # Cross-run dedup: remove questions too similar to previously approved ones
    cross_run_removed = 0
    if state.session_context:
        try:
            from src import memory as _memory
            bank_qs = _memory.get_bank_questions(state.session_context.session_name)
            if bank_qs and state.questions:
                bank_texts = [b["content"] for b in bank_qs]
                current_qs = list(state.questions.values())
                current_texts = [q.content for q in current_qs]
                all_texts = bank_texts + current_texts
                vec2 = TfidfVectorizer(stop_words="english", max_features=5000)
                tfidf2 = vec2.fit_transform(all_texts)
                bank_mat = tfidf2[:len(bank_texts)]
                curr_mat = tfidf2[len(bank_texts):]
                cross_sim = cosine_similarity(curr_mat, bank_mat)
                for i, row in enumerate(cross_sim):
                    if row.max() > DEDUP_THRESHOLD:
                        q = state.questions.pop(current_qs[i].question_id, None)
                        if q is not None:
                            state.removed.append({
                                "content": q.content, "reason": "Duplicate of a previously approved question",
                                "stage": "duplicate", "difficulty": q.difficulty, "company": q.attribution,
                            })
                        cross_run_removed += 1
        except Exception:
            pass

    return {
        "kept": len(state.questions),
        "removed": within_run_removed,
        "cross_run_removed": cross_run_removed,
    }


def tool_check_difficulty_balance(state: AgentState) -> dict:
    counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    for q in state.questions.values():
        counts[q.difficulty or "Medium"] = counts.get(q.difficulty or "Medium", 0) + 1
    for q in state.coding_questions.values():
        counts[q.difficulty or "Medium"] = counts.get(q.difficulty or "Medium", 0) + 1
    total = sum(counts.values())
    actual = {k: round(v / max(total, 1), 2) for k, v in counts.items()}
    balanced = total > 0 and all(abs(actual.get(k, 0) - v) < 0.25 for k, v in {"Easy": 0.30, "Medium": 0.50, "Hard": 0.20}.items())
    return {"total": total, "counts": counts, "actual_pct": actual, "balanced": balanced}


def tool_check_outcome_coverage(state: AgentState) -> dict:
    if not state.session_context:
        return {"error": "Call understand_session first"}
    outcomes = state.session_context.learning_outcomes
    all_content = " ".join(q.content.lower() for q in state.questions.values())
    all_content += " " + " ".join(q.content.lower() for q in state.coding_questions.values())

    covered, missing = [], []
    for outcome in outcomes:
        words = [w.lower() for w in outcome.split() if len(w) > 3]
        if any(w in all_content for w in words):
            covered.append(outcome)
        else:
            missing.append(outcome)
    return {
        "total_outcomes": len(outcomes), "covered": len(covered),
        "missing_count": len(missing), "missing": missing,
        "coverage_pct": round(len(covered) / max(len(outcomes), 1), 2),
    }


def tool_generate_expected_answers(state: AgentState) -> dict:
    """Disabled — expected answers are no longer produced."""
    return {
        "generated": 0,
        "blocked": True,
        "reason": "Expected-answer generation is disabled. Proceed to submit_question_set.",
    }


def tool_generate_interview_questions(state: AgentState, count: int, outcomes: list[str], difficulty_mix: str = None) -> dict:
    """Blocked — only real company interview questions from the bank and live sources are used."""
    return {
        "generated": 0,
        "blocked": True,
        "reason": (
            "Question generation is disabled. Only real company interview questions are used. "
            "Try more searches with search_question_bank, search_github_questions, or search_web_questions. "
            "If all sources are exhausted, call submit_question_set with what you have."
        ),
    }


def tool_generate_coding_questions(state: AgentState, count: int, topics: list[str] = None, language: str = "Python") -> dict:
    """Generate coding questions in Nxtmock portal format: concise content + separate starter code."""
    from src.models import CodeSnippet

    count = min(count, 4)
    max_to_add = state.config.max_questions - len(state.questions) - len(state.coding_questions)
    count = min(count, max(0, max_to_add))
    if count <= 0:
        return {"generated": 0, "warning": "At max capacity"}

    topics_str = ", ".join(topics) if topics else state.config.session_name
    lang_upper = language.upper()

    result = chat_completion_json(
        system_prompt=f"""Generate {count} coding interview questions about: {topics_str}.

IMPORTANT FORMAT — match this exact style:

For each question provide:
- title: Short name (e.g., "Build a Gemini API Chat Function")
- content: Plain text problem description. Concise, 1-4 sentences describing what to build. Include sample input/output if applicable. Do NOT use markdown headers (no ## or **).
- difficulty: "Easy", "Medium", or "Hard" (vary them)
- topic: Clean topic name
- starter_code: A {language} code template with the function signature and a "# Write your code here" comment. Just the skeleton, no solution.

Example output format:
{{
    "title": "Build a Gemini API Chat Function",
    "content": "Write a function that connects to the Google Gemini API using the google-genai package, sends a user prompt, and returns the model's text response. The function should load the API key from environment variables.\\n\\nSample Input: prompt = 'What is machine learning?'\\nSample Output: 'Machine learning is a subset of AI...'",
    "difficulty": "Medium",
    "topic": "Gemini API",
    "starter_code": "import os\\nfrom google import genai\\n\\ndef chat_with_gemini(prompt: str) -> str:\\n    # Write your code here\\n    pass"
}}

Respond in JSON: {{"coding_questions": [...]}}""",
        user_prompt=f"Generate {count} coding questions in {language}.", max_tokens=4000,
        on_usage=_usage_cb(state),
    )

    added = []
    for cq_data in result.get("coding_questions", []):
        q_id = str(uuid.uuid4())
        code_id = str(uuid.uuid4())
        starter = cq_data.get("starter_code", f"# Write your {language} code here\n")

        cq = CodingQuestion(
            id=q_id,
            category=f"{lang_upper}_CODING",
            title=cq_data.get("title", "Coding Question"),
            content=cq_data.get("content", ""),
            code_id=code_id,
            topic=cq_data.get("topic", topics_str[:50]),
            difficulty=cq_data.get("difficulty", "Medium"),
            language=language,
            source="generated",
        )
        state.coding_questions[q_id] = cq

        # Store starter code as separate CodeSnippet
        snippet = CodeSnippet(
            code_id=code_id,
            code_content=starter,
            language=lang_upper,
        )
        state.code_snippets[code_id] = snippet

        added.append({"id": q_id, "title": cq.title, "difficulty": cq.difficulty})
    return {"generated": len(added), "questions": added}


def tool_search_github_questions(state: AgentState, outcomes: list) -> dict:
    """Harvest real interview questions from verified GitHub repos via the GitHub REST API."""
    from src.sources.github_repo import GithubRepoConnector
    from src.config import GITHUB_POOL_CAP
    records = GithubRepoConnector().fetch(outcomes)

    capacity = GITHUB_POOL_CAP - state.added_by_source.get("github", 0)
    if capacity <= 0:
        return {"found": 0, "warning": f"GitHub quota full ({GITHUB_POOL_CAP})."}

    topic_keywords = _topic_keywords(state.session_context)
    added = []
    for rec in records[:capacity]:
        content = rec.question_text
        if topic_keywords:                       # split compound Qs to the on-topic clause
            content = _trim_to_topic(content, topic_keywords)
            if not content:
                continue
        q_id = str(uuid.uuid4())
        qd = QuestionDetail(
            question_id=q_id,
            category="THEORY",
            content=content,
            topic=rec.source_type.split(":")[-1] if ":" in rec.source_type else "Interview",
            difficulty="Medium",
            source="web",
            source_url=rec.source_url,
        )
        state.questions[q_id] = qd
        added.append({
            "id": q_id,
            "content": content[:150],
            "source_url": rec.source_url,
            "source_type": rec.source_type,
        })

    state.raw_fetched["github"] = state.raw_fetched.get("github", 0) + len(records)
    state.added_by_source["github"] = state.added_by_source.get("github", 0) + len(added)
    total = len(state.questions) + len(state.coding_questions)
    state.pool_size = max(state.pool_size, total)
    return {
        "found": len(records),
        "added": len(added),
        "total_accumulated": total,
        "github_remaining": GITHUB_POOL_CAP - state.added_by_source.get("github", 0),
    }


_QUERY_STOPWORDS = {
    "with", "that", "this", "from", "what", "have", "will", "about", "using",
    "build", "create", "learn", "your", "into", "some", "each", "when", "which",
    "should", "between", "their", "could", "would", "does", "they", "been", "more",
    "than", "also", "other",
}


def _topic_keywords(ctx) -> set:
    """Salient (>=4-char) keywords from the session's RM-derived scope — used to trim
    compound questions to their on-topic clause. Shared by web + github + bank paths."""
    kws: set = set()
    if not ctx:
        return kws
    for term in (ctx.scope_in + ctx.key_concepts + ctx.learning_outcomes):
        for w in term.lower().split():
            if len(w) >= 4 and w not in _QUERY_STOPWORDS:
                kws.add(w)
    return kws


def _trim_to_topic(text: str, topic_keywords: set) -> str:
    """Keep clauses from the start through the LAST on-topic clause; drop an
    unrelated trailing clause. Return "" if no clause is on-topic (caller skips)."""
    clauses = split_into_clauses(text)
    if len(clauses) <= 1:                        # single ask → today's behaviour
        low = text.lower()
        return text if any(k in low for k in topic_keywords) else ""
    last_hit = -1
    for i, c in enumerate(clauses):
        cl = c.lower()
        if any(k in cl for k in topic_keywords):
            last_hit = i
    if last_hit < 0:
        return ""
    if last_hit == len(clauses) - 1:             # every clause on-topic → keep whole
        return text
    trimmed = ". ".join(clauses[:last_hit + 1]).strip()
    return trimmed if looks_like_question(trimmed) else text


def tool_search_web_questions(state: AgentState, outcomes: list) -> dict:
    """Harvest real interview questions with company attribution from 65+ verified domains via Tavily."""
    from src import config as cfg
    if not cfg.TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not set — skipping web search", "status": "skipped"}

    from src.sources.tavily_search import TavilyConnector
    from src.config import WEB_POOL_CAP
    ctx = state.session_context

    # Check the web quota BEFORE calling Tavily — don't burn API calls when web is full.
    capacity = WEB_POOL_CAP - state.added_by_source.get("web", 0)
    if capacity <= 0:
        return {"found": 0, "warning": f"Web quota full ({WEB_POOL_CAP})."}

    # Query the session's substantive CONCEPTS (key_concepts + scope_in), not just short
    # scope labels — so deep conceptual questions (CoT, prompting, RAG) surface, not only
    # generic company-review noise. Supplement with whatever the agent passed.
    seen_terms: set[str] = set()
    search_terms: list[str] = []
    for t in ((ctx.key_concepts[:4] if ctx and ctx.key_concepts else [])
              + (ctx.scope_in[:4] if ctx and ctx.scope_in else [])
              + list(outcomes)):
        tl = t.lower().strip()
        if tl and tl not in seen_terms:
            seen_terms.add(tl)
            search_terms.append(t)
    # Cap terms for latency — each term costs several Tavily calls; recall is already
    # high (hundreds of records) with a handful of terms.
    search_terms = search_terms[:4]
    records, tavily_calls, tavily_error = TavilyConnector().fetch(search_terms or outcomes)
    state.api_usage["tavily_calls"] += tavily_calls

    # Surface a real failure (quota/auth/rate limit) instead of silently looking
    # like "no web questions found".
    if tavily_error and not records:
        return {"found": 0, "added": 0, "status": "error", "error": tavily_error}

    # Prioritise attributed company questions and premium sources over Reddit noise
    _PRIORITY_SOURCES = {"tryexponent.com", "ambitionbox.com", "glassdoor.com",
                         "interviewquery.com", "datalemur.com", "prepfully.com",
                         "igotanoffer.com", "leetcode.com"}
    records.sort(key=lambda r: (
        0 if r.company else 1,
        0 if any(s in r.source_type for s in _PRIORITY_SOURCES) else 1,
    ))

    # Use session topic for proper metadata
    session_topic = (state.session_context.key_concepts[0]
                     if state.session_context and state.session_context.key_concepts
                     else "Interview")

    # Topic keywords for the on-topic question-splitting pre-filter (shared with github/bank).
    topic_keywords = _topic_keywords(ctx)

    # Fill the web quota (relevance ranking + trim-to-max happen later); scan a
    # generous slice of the sorted records since the keyword pre-filter drops many.
    take = min(len(records), capacity + 40)
    added = []
    for rec in records[:take]:
        # Trim a two-part question down to its on-topic clause (drops an unrelated
        # trailing ask); "" means no clause is on-topic → skip the whole record.
        content = rec.question_text
        if topic_keywords:
            content = _trim_to_topic(content, topic_keywords)
            if not content:
                continue

        if len(added) >= capacity:
            break

        q_id = str(uuid.uuid4())
        qd = QuestionDetail(
            question_id=q_id,
            category=getattr(state.config, "category", "GEN_AI"),
            content=content,
            topic=session_topic,
            difficulty="Medium",
            source="web",
            asked_in_company=rec.company,
            source_url=rec.source_url,
        )
        state.questions[q_id] = qd
        added.append({
            "id": q_id,
            "content": content[:150],
            "company": rec.company,
            "source_url": rec.source_url,
            "source_type": rec.source_type,
        })

    state.raw_fetched["web"] = state.raw_fetched.get("web", 0) + len(records)
    state.added_by_source["web"] = state.added_by_source.get("web", 0) + len(added)
    total = len(state.questions) + len(state.coding_questions)
    state.pool_size = max(state.pool_size, total)
    return {
        "found": len(records),
        "added": len(added),
        "total_accumulated": total,
        "web_remaining": WEB_POOL_CAP - state.added_by_source.get("web", 0),
    }


def tool_remove_question(state: AgentState, question_id: str, reason: str = "") -> dict:
    # Guarantee the minimum: refuse a theory removal that would drop below min_questions
    # UNLESS the reserve can backfill it. This stops the quality-gate revision from
    # collapsing a thin set (e.g. down to 1) when there's nothing to refill from.
    total = len(state.questions) + len(state.coding_questions)
    min_q = max(1, state.config.min_questions)
    if (question_id in state.questions and total <= min_q
            and not [qid for qid in state.reserve if qid not in state.excluded]):
        return {"removed": False,
                "warning": f"At minimum ({min_q}) with no backfill available — keeping this question.",
                "remaining": total}
    q = state.questions.pop(question_id, None) or state.coding_questions.pop(question_id, None)
    if q is not None:
        state.excluded.add(question_id)          # never re-add a rejected question
        state.reserve.pop(question_id, None)
        state.removed.append({
            "content": getattr(q, "content", getattr(q, "title", "")),
            "reason": reason or "Removed at quality gate", "stage": "curation",
            "difficulty": getattr(q, "difficulty", None), "company": getattr(q, "attribution", None),
        })
        return {"removed": True, "remaining": len(state.questions) + len(state.coding_questions)}
    return {"removed": False, "error": f"Question {question_id} not found"}


def _select_final(questions: list, k: int, outcomes: list) -> list:
    """Pick k questions balancing relevance, diversity, outcome coverage, and difficulty mix.

    Greedy per-step score for candidate c given the already-selected set S:
        λ·relevance(c) − (1−λ)·max_{s∈S} sim(c,s)
        + COVERAGE_BONUS   if c covers a learning outcome not yet covered by S
        + DIFFICULTY_BONUS if c's difficulty bucket is still under its target quota
    relevance = LLM relevance_score; sim/coverage = TF-IDF cosine (same vectorizer as dedup).
    Seeds with the most relevant question. This is the MMR core (relevance + diversity) with
    coverage and difficulty nudges so the final set spans the session's outcomes and levels.
    """
    from src.config import (MMR_LAMBDA, SELECT_COVERAGE_BONUS, SELECT_DIFFICULTY_BONUS,
                            SELECT_SESSION_BONUS, SELECT_ATTRIBUTION_BONUS)

    def _rel(q):
        return q.relevance_score if q.relevance_score is not None else 0.0

    def _attr(q):  # True if backed by a credible real company (not NIAT/source-labeled)
        c = q.asked_in_company
        return bool(c) and str(c).strip().upper() != "NIAT"

    if k <= 0:
        return []
    if len(questions) <= k:
        return sorted(questions, key=lambda q: -_rel(q))

    texts = [q.content for q in questions]
    rel = [_rel(q) for q in questions]

    # Similarity for redundancy + coverage: semantic embeddings when available, else TF-IDF.
    from src import embeddings
    from src.config import EMBED_COVERAGE_THRESHOLD
    sim = embeddings.cosine_matrix(texts)
    if sim is not None:
        qo = embeddings.cosine_matrix(texts, outcomes) if outcomes else None
        cov_thresh = EMBED_COVERAGE_THRESHOLD
    else:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        qmat = vec.fit_transform(texts)
        sim = cosine_similarity(qmat)
        qo = cosine_similarity(qmat, vec.transform(outcomes)) if outcomes else None
        cov_thresh = 0.10

    # Which outcomes each question plausibly covers.
    covers = [set() for _ in questions]
    if outcomes and qo is not None:
        for i in range(len(questions)):
            covers[i] = {j for j in range(len(outcomes)) if qo[i][j] >= cov_thresh}

    # Difficulty target counts for k (Easy 30% / Medium 50% / Hard 20%).
    diff_target = {"Easy": round(k * 0.3), "Medium": round(k * 0.5), "Hard": round(k * 0.2)}

    # Per-session target (multi-session topics): aim for balanced representation.
    sessions = [q.session for q in questions if q.session]
    distinct_sessions = list(dict.fromkeys(sessions))
    sess_target = (k / len(distinct_sessions)) if len(distinct_sessions) > 1 else 0

    # Attribution balance: only nudge when the pool actually has BOTH kinds; aim ~half
    # company-attributed so the set mixes real-company examples with substantive ones.
    has_attr = any(_attr(q) for q in questions)
    has_unattr = any(not _attr(q) for q in questions)
    attr_target = (k / 2) if (has_attr and has_unattr) else 0

    seed = max(range(len(questions)), key=lambda i: rel[i])
    selected = [seed]
    covered = set(covers[seed])
    diff_have = {"Easy": 0, "Medium": 0, "Hard": 0}
    diff_have[questions[seed].difficulty or "Medium"] = 1
    sess_have = {questions[seed].session: 1} if questions[seed].session else {}
    attr_have = 1 if _attr(questions[seed]) else 0
    unattr_have = 0 if _attr(questions[seed]) else 1

    while len(selected) < k:
        best, best_score = None, float("-inf")
        for c in range(len(questions)):
            if c in selected:
                continue
            div = max(sim[c][s] for s in selected)
            cov_new = SELECT_COVERAGE_BONUS if (covers[c] - covered) else 0.0
            bucket = questions[c].difficulty or "Medium"
            need = SELECT_DIFFICULTY_BONUS if diff_have.get(bucket, 0) < diff_target.get(bucket, 0) else 0.0
            sess = questions[c].session
            # Deficit-scaled: an under-represented (esp. zero-coverage) session gets a
            # bigger push than one already near its fair share.
            sess_need = (SELECT_SESSION_BONUS * max(0.0, sess_target - sess_have.get(sess, 0))
                         if (sess_target and sess) else 0.0)
            # Attribution-balance nudge toward the under-filled bucket (company vs source-labeled).
            if attr_target and _attr(questions[c]):
                attr_need = SELECT_ATTRIBUTION_BONUS if attr_have < attr_target else 0.0
            elif attr_target:
                attr_need = SELECT_ATTRIBUTION_BONUS if unattr_have < (k - attr_target) else 0.0
            else:
                attr_need = 0.0
            score = MMR_LAMBDA * rel[c] - (1 - MMR_LAMBDA) * div + cov_new + need + sess_need + attr_need
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        covered |= covers[best]
        diff_have[questions[best].difficulty or "Medium"] = diff_have.get(questions[best].difficulty or "Medium", 0) + 1
        if questions[best].session:
            sess_have[questions[best].session] = sess_have.get(questions[best].session, 0) + 1
        if _attr(questions[best]):
            attr_have += 1
        else:
            unattr_have += 1
    return [questions[i] for i in selected]


def _attribute_sessions(questions: list, profiles: dict) -> None:
    """Tag each question with the session profile it most resembles (TF-IDF cosine).
    Single-session topics tag all with that session; enables per-session representation."""
    names = list(profiles.keys())
    if not names:
        return
    if len(names) == 1:
        for q in questions:
            q.session = names[0]
        return
    if not questions:
        return
    prof_texts = [profiles[n] for n in names]
    qtexts = [q.content for q in questions]
    from src import embeddings
    sims = embeddings.cosine_matrix(qtexts, prof_texts)   # [nq x n_sessions], semantic
    if sims is None:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        mat = vec.fit_transform(prof_texts + qtexts)
        sims = cosine_similarity(mat[len(names):], mat[:len(names)])
    for i, q in enumerate(questions):
        q.session = names[int(sims[i].argmax())]


def tool_submit_question_set(state: AgentState) -> dict:
    # Selection pool = current theory questions + reserve (validated-but-not-selected),
    # minus anything explicitly excluded (rejected in a prior revision). This lets a
    # revision BACKFILL from the reserve instead of the set only ever shrinking.
    pool = {**state.reserve, **state.questions}
    for qid in state.excluded:
        pool.pop(qid, None)
    outcomes = state.session_context.learning_outcomes if state.session_context else []
    keep_theory = max(0, state.config.max_questions - len(state.coding_questions))

    selected = _select_final(list(pool.values()), keep_theory, outcomes)
    selected_ids = {q.question_id for q in selected}

    # New selected set; everything else in the pool goes to the reserve (recoverable),
    # NOT to `removed` (it wasn't rejected, just not chosen this round).
    state.questions = {q.question_id: q for q in selected}
    state.reserve = {qid: q for qid, q in pool.items() if qid not in selected_ids}

    total = len(state.questions) + len(state.coding_questions)
    state.submitted = True
    return {"submitted": True, "total_questions": total,
            "theory": len(state.questions), "coding": len(state.coding_questions),
            "reserve": len(state.reserve)}


# ── Dispatch ────────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "understand_session": tool_understand_session,
    "search_question_bank": tool_search_question_bank,
    "search_github_questions": tool_search_github_questions,
    "search_web_questions": tool_search_web_questions,
    "validate_relevance": tool_validate_relevance,
    "deduplicate_questions": tool_deduplicate_questions,
    "check_difficulty_balance": tool_check_difficulty_balance,
    "check_outcome_coverage": tool_check_outcome_coverage,
    "generate_expected_answers": tool_generate_expected_answers,
    "generate_interview_questions": tool_generate_interview_questions,
    "generate_coding_questions": tool_generate_coding_questions,
    "remove_question": tool_remove_question,
    "submit_question_set": tool_submit_question_set,
}
