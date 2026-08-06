"""Tool definitions + implementations, split across the four agents.

Each agent gets a focused subset (see src/agents/*.get_tool_schemas):
  Understanding — understand_session (must run first; everything downstream needs its outcomes)
  Retrieval     — search_question_bank, search_web_questions, and search_github_questions
                  only when GITHUB_ENABLED=1 (off by default)
  Validation    — validate_relevance, deduplicate_questions
  Evaluation    — check_difficulty_balance, check_outcome_coverage, remove_question,
                  submit_question_set

The generate_* tools are permanently blocked (real, sourced questions only) and are in no agent's
subset, so they are unreachable; they remain in TOOL_DISPATCH only so a stray call gets a clear
refusal rather than an "unknown tool" error.
"""

from __future__ import annotations
import json
import re
import uuid
from typing import TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import QuestionDetail, CodingQuestion
from src.llm_client import chat_completion_json, run_model


def _usage_cb(state: "AgentState"):
    """Return an on_usage callback that accumulates LLM token stats into state.api_usage."""
    def _cb(usage):
        state.api_usage["llm_calls"] += 1
        state.api_usage["prompt_tokens"] += usage.prompt_tokens or 0
        state.api_usage["completion_tokens"] += usage.completion_tokens or 0
    return _cb


from src.config import DEDUP_THRESHOLD, normalize_session_type
from src.quality import is_quality_question, strip_artifacts
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
            "description": "Search the pre-indexed question bank of real interview questions. Ranks by a hybrid of semantic similarity and keyword match, then applies session-aware filtering. Use the suggested_search_queries from understand_session verbatim — do NOT invent generic terms.",
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
            "description": "Search an allowlist of interview-question domains (Glassdoor, AmbitionBox, Exponent, GeeksforGeeks, LeetCode, and similar) for real questions with company attribution, via Tavily. Requires TAVILY_API_KEY. Use for questions that came from actual company interviews.",
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
            "description": "Remove near-duplicate questions, including rewordings (semantic similarity when embeddings are available, otherwise keyword similarity).",
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
            "description": "DISABLED — coding questions are not generated; only real retrieved ones are used. Do not call.",
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
            "description": "Finalize and submit the question set for human review. THIS ENDS THE RUN. It ranks the validated pool and keeps the best ones up to the run's requested count, so submit once you have gathered and validated candidates — you do not need to prune to the target yourself. Call this even if the pool is smaller than requested.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ── Tool Implementations ────────────────────────────────────────────────────

def tool_understand_session(state: AgentState) -> dict:
    """Step 1: LLM reads session RM, extracts outcomes, maps to KPs."""
    from src.session_understanding import understand_session

    context = understand_session(state.config.session_names, state.data_store,
                                 model=run_model(state))
    state.session_context = context
    state.learning_outcomes = context.learning_outcomes

    # Per-session profile (multi-session topics) for attributing each question to the session it best
    # matches — which is what enables per-session representation in final selection.
    #
    # This used to store ONE reading-material blob per session truncated to 4000 chars. Two problems,
    # both measured on a real run of "Introduction to AI Agents + Building a Learning Path Generator":
    #   * truncation — the second session's material is 8,806 chars, so 55% of it was discarded;
    #   * a single blob DILUTES any specific match, which is exactly what `_session_profile`'s own
    #     docstring warns about ("LISTS of short texts, not one blob … so long reading material cannot
    #     dilute a short, precise outcome statement").
    # The result: every LearnPath score collapsed to 0.17–0.41 and all 12 questions in that run were
    # labelled with the first session. Reproduced: blob → 12/0, chunked → 6/6. Reusing
    # `_session_profile` means attribution and the session-fit gate now agree by construction.
    if len(state.config.session_names) > 1:
        from src.pipeline import _session_profile   # local: pipeline imports this module
        for name in state.config.session_names:
            curated, rm_chunks = _session_profile([name], None)
            state.session_profiles[name] = {"curated": curated, "rm": rm_chunks}

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

    # Probe the bank to see whether it covers this session at all.
    #
    # This reports a MEASURED count, not an extrapolation. It used to return
    # `len(probe) * min(len(all_queries), 5)` — a 5-result probe multiplied by the query count — and
    # present that invented number to the model as `"Bank has ~N+ potential matches"`. The model has no
    # way to know it was fabricated, so it planned its retrieval against a figure nobody measured.
    has_bank_questions = False
    probe_hits = 0
    probe_limit = 25
    if all_queries:
        from src.question_bank import get_retriever_for
        retriever = get_retriever_for(getattr(state.config, "category", None))
        probe = retriever.search(all_queries[0], limit=probe_limit)
        probe_hits = len(probe)
        has_bank_questions = probe_hits > 0

    state.has_bank_questions = has_bank_questions

    bank_hint = (
        f"The bank returned {probe_hits} match(es) for the FIRST query alone"
        + (f" (probe capped at {probe_limit}, so there may be more)." if probe_hits >= probe_limit
           else ". Other queries may match different questions.")
        if has_bank_questions
        else "The bank returned nothing for the first query — it is likely thin for this topic, so lean on search_web_questions."
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
        # Measured hits for the first query only — deliberately NOT an estimate of total coverage.
        "bank_probe_hits": probe_hits,
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
        # Return the SAME keys as the success path. Omitting them made the progress line read
        # "Found 0 relevant (total: 0, bank quota left: ?)" — reporting an empty pool at the moment
        # the pool was actually full.
        return {"found": 0, "total_accumulated": len(state.questions) + len(state.coding_questions),
                "bank_remaining": 0,
                "warning": f"Bank quota full ({BANK_POOL_CAP}) — move to web search or submit."}

    # Per-call cap raised 8→25 so a single search pulls a much wider slice into the pool
    # (the whole pool is relevance-scored later; we want the judge to see more candidates).
    actual_limit = min(limit, max_to_add, 25)
    exclude_ids = set(state.questions.keys())

    results = retriever.search(
        query=query, difficulty=difficulty,
        limit=actual_limit + 10,  # fetch extra for filtering
        exclude_ids=exclude_ids,
    )
    raw_hits = len(results)   # before any filtering — this is what the funnel reports

    # Session-aware post-retrieval relevance filter
    scope_keywords: set[str] = set()
    if state.session_context:
        _it = getattr(state.session_context, "interview_topics", None) or []
        for term in (state.session_context.learning_outcomes +
                     state.session_context.key_concepts +
                     state.session_context.scope_in + list(_it)):
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
        qd.content = strip_artifacts(qd.content)
        # Split a compound bank question down to its on-topic clause (same as web).
        if scope_keywords:
            trimmed = _trim_to_topic(qd.content, scope_keywords)
            if trimmed:
                qd.content = trimmed
        # Form-quality gate — drop boilerplate/logistics/fragments/headings that slipped into the bank.
        if not is_quality_question(qd.content):
            continue
        state.questions[qd.question_id] = qd
        added.append({
            "id": qd.question_id,
            "content": qd.content[:150],
            "difficulty": qd.difficulty,
            "topic": qd.topic,
            "source": qd.source,
        })

    # raw_fetched is documented as RAW hit counts per source, so count what the retriever returned —
    # not what survived filtering. Counting `added` here made the bank column non-comparable with the
    # web/github columns and broke the funnel it exists to show.
    state.raw_fetched["bank"] = state.raw_fetched.get("bank", 0) + raw_hits
    state.added_by_source["bank"] = state.added_by_source.get("bank", 0) + len(added)
    total = len(state.questions) + len(state.coding_questions)
    state.pool_size = max(state.pool_size, total)
    return {
        "found": len(added),
        "questions": added,
        "total_accumulated": total,
        "bank_remaining": BANK_POOL_CAP - state.added_by_source.get("bank", 0),
    }


# What a strong question looks like depends on the KIND of session. Hand-written domain guidance, kept
# separate from the reviewer's examples: it applies even for a session type with no labels yet.
_TYPE_GUIDANCE = {
    "code_heavy": """## This is a CODE-HEAVY session — weight questions accordingly
Candidates score HIGH when they ask the candidate to build, debug, choose between implementations, or
explain a design decision: writing the code, wiring an API, handling a failure mode, picking a data
structure or a chunking strategy, reasoning about latency or cost, reading code and saying what it does.
Score a purely definitional question ("What is X?") LOWER here even when X is on-topic — this session
taught the candidate to *do* the thing, so recall alone under-tests it.
Still reject: questions about the candidate's own past projects, and anything from another domain.""",
    "theory_heavy": """## This is a THEORY-HEAVY session — weight questions accordingly
Candidates score HIGH when they test understanding of a concept, a comparison, a trade-off, or a
mechanism: what something is, how it works, why one approach beats another, what breaks and why.
Score deep implementation minutiae LOWER here — API signatures, config flags, framework-specific
plumbing — because this session did not teach them; a concept question is the on-target form.
Still reject: questions about the candidate's own past projects, and anything from another domain.""",
    "mixed": """## This is a MIXED session — both forms are on-target
Concept questions and implementation questions both score well provided the subject is taught in the
reading material. Judge on subject matter, not on form.
Still reject: questions about the candidate's own past projects, and anything from another domain.""",
}


def _type_guidance(session_type: str | None) -> str:
    return _TYPE_GUIDANCE[normalize_session_type(session_type)]


def _feedback_examples_block(state: AgentState, per_side: int = 12) -> str:
    """Prompt block of the reviewer's own past accept/reject decisions, or "" if there are none.

    Prefers decisions made on THIS session (taste is session-specific) and tops up with decisions
    from other sessions, so a brand-new session still gets calibration. Capped per side to bound
    prompt cost, and balanced so the judge doesn't infer "reject everything" from a lopsided history
    (the real log is 68 rejections to 24 acceptances).
    """
    try:
        from src import memory as _memory
        examples = _memory.get_feedback_examples()
    except Exception:  # noqa: BLE001 — feedback is an enhancement, never a hard dependency
        return ""
    if not examples:
        return ""

    from src.session_types import type_for_run

    session = state.session_context.session_name if state.session_context else None
    session_type = normalize_session_type(
        state.session_context.session_type if state.session_context else None)

    def tier(decision: str) -> tuple[list[str], str]:
        """Narrowest useful pool of decisions: this session → this session TYPE → everything.

        The type tier is the one that matters. Pooling types mis-calibrates both: an implementation
        question resembles the "too specific, not conceptual" pattern the reviewer established on
        THEORY material, so a code-heavy session judged on theory decisions is judged wrongly.
        """
        rows = [e for e in examples
                if e.get("decision") == decision and (e.get("question") or "").strip()]
        same_session = [e["question"].strip() for e in rows
                        if (e.get("session") or "").strip() == (session or "").strip()]
        if len(same_session) >= 3:
            return same_session[:per_side], "this session"
        same_type = [e["question"].strip() for e in rows
                     if type_for_run(e.get("session") or "") == session_type]
        if len(same_type) >= 3:
            return same_type[:per_side], f"{session_type} sessions"
        return [e["question"].strip() for e in rows][:per_side], "all session types"

    accepted, acc_pool = tier("good")
    rejected, rej_pool = tier("bad")
    if not accepted and not rejected:
        return ""
    # Balance the two sides so neither dominates the judge's impression of the reviewer's bar.
    n = min(per_side, max(len(accepted), len(rejected)))
    accepted, rejected = accepted[:n], rejected[:n]

    pool = acc_pool if acc_pool == rej_pool else f"{acc_pool} / {rej_pool}"
    block = ("## Reviewer's past decisions — calibrate to this taste\n"
             f"Real accept/reject decisions by the human who will review your output, drawn from "
             f"**{pool}**. This session is **{session_type}**.\n")
    if accepted:
        block += ("\nACCEPTED (score these kinds generously):\n"
                  + "\n".join(f"- {q[:200]}" for q in accepted) + "\n")
    if rejected:
        block += ("\nREJECTED (score anything of this kind ≤0.3, including rewordings):\n"
                  + "\n".join(f"- {q[:200]}" for q in rejected) + "\n")
    if pool == "all session types":
        block += ("\nNOTE: no decisions exist yet for this session type, so the examples above mix "
                  "types. Weigh the session-type guidance below more heavily than these examples.\n")
    return block + "\n"


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
    # Show the judge what THIS reviewer actually accepted and rejected. Distilled rules only exist
    # when a reviewer typed a free-text reason (they rarely do — 68 rejections produced zero rules),
    # so the accept/reject decisions themselves are the feedback signal that is actually available.
    # Concrete examples calibrate the judge's taste in a way an abstract rule cannot.
    rules_block += _feedback_examples_block(state)

    outcomes_str = "\n".join(f"- {o}" for o in state.session_context.learning_outcomes)
    concepts_str = ", ".join(state.session_context.key_concepts)
    topics_str = ", ".join(getattr(state.session_context, "interview_topics", None) or []) or "(same as key concepts)"
    scope_out_str = ", ".join(state.session_context.scope_out) if state.session_context.scope_out else "none"
    # What counts as a strong question differs by session kind — see _TYPE_GUIDANCE.
    type_guidance = _type_guidance(state.session_context.session_type)

    # Ground the judge in the ACTUAL reading material (the outcomes list alone is too thin —
    # it lets keyword overlap masquerade as relevance). Include a bounded excerpt of each
    # selected session's RM, split evenly to keep total cost in check.
    names = state.config.session_names or [state.session_context.session_name]
    # Feed the judge (near-)full reading material: 1–2 sessions → the whole RM each; 3+ → ~10k each.
    # The old 4000//n cut showed the judge ~11% of a 12k RM for a 3-session run, so it never saw the
    # concept a genuinely on-topic question tested and scored it low. The RM is the source of truth.
    per_cap = min(12000, max(6000, 30000 // max(1, len(names))))
    rm_parts = []
    for name in names:
        content = state.data_store.get_session_content(name)
        if content:
            rm_parts.append(f"### {name}\n{content[:per_cap]}")
    rm_block = "\n\n".join(rm_parts) if rm_parts else "(no reading material found — use the outcomes/concepts above)"

    system_prompt = f"""{rules_block}You score interview questions by how well they test what a SPECIFIC session actually teaches.

Session: {state.session_context.session_name}

## What this session actually teaches — THE READING MATERIAL BELOW IS THE SOURCE OF TRUTH
{rm_block}

The Learning Outcomes / Key Concepts below are only a SHORT SUMMARY of the reading material — they do
NOT list every sub-topic. A question is ON-TOPIC if its subject is taught ANYWHERE in the reading
material above (or is one of the Interview Topics), EVEN IF no learning outcome names it word-for-word.
Do NOT down-score a question just because it isn't in the outcomes list — judge against the reading
material itself.

Learning Outcomes (summary only):
{outcomes_str}

Key Concepts: {concepts_str}
Interview Topics (transferable concepts this session prepares a candidate for — a question testing one of
these IS on-topic, even if it doesn't name the specific tool/product used in the session): {topics_str}
Out of Scope (score these ≤0.2): {scope_out_str}

{type_guidance}

Give EACH question a relevance score from 0.0 to 1.0 for how well it tests a concept the session teaches
(any concept present in the reading material above OR one of the Interview Topics):
- 0.8–1.0 — directly tests a technical concept explained anywhere in the reading material / interview topics.
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
    failed_qids: set[str] = set()   # items in a batch the model failed to score at all (API/parse failure)
    _valid_diff = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
    batches_total = batches_ok = 0
    for start in range(0, len(items), RELEVANCE_BATCH_SIZE):
        batch = items[start:start + RELEVANCE_BATCH_SIZE]
        batches_total += 1
        numbered = [{"n": i + 1, "q": q.content[:600]} for i, (_, q) in enumerate(batch)]
        try:
            result = chat_completion_json(
                model=run_model(state),      # the run's model, not the UI global
                system_prompt=system_prompt,
                user_prompt=f"Score these {len(numbered)} questions:\n{json.dumps(numbered)}",
                max_tokens=4096,   # roomy vs a 25-item batch (~30 tokens/item) so JSON never truncates
                on_usage=_usage_cb(state),
            )
        except Exception as exc:  # noqa: BLE001 — one bad batch must not lose the other batches
            print(f"[relevance] batch {batches_total} failed ({type(exc).__name__}: {exc})")
            failed_qids.update(q_id for q_id, _ in batch)
            continue
        batch_scored = 0
        for s in (result.get("scores") or []):
            try:
                n = int(s["n"]); score = max(0.0, min(1.0, float(s["score"])))
            except (TypeError, ValueError, KeyError):
                continue
            if 1 <= n <= len(batch):
                q_id = batch[n - 1][0]
                score_by_qid[q_id] = score
                batch_scored += 1
                d = _valid_diff.get(str(s.get("difficulty", "")).strip().lower())
                if d:
                    diff_by_qid[q_id] = d
        # If the whole batch produced ZERO scores, that's an API/parse failure, not a real
        # "all irrelevant" signal — don't nuke those questions to a low default; mark them so
        # they keep a neutral (threshold) score below.
        if batch_scored == 0:
            failed_qids.update(q_id for q_id, _ in batch)
        else:
            batches_ok += 1

    # A partial failure is survivable — the neutral default below keeps those candidates in play.
    # A TOTAL failure is not: every candidate would be kept at the threshold score, the relevance
    # filter would remove nothing, and the run would report a clean pass having never checked topical
    # fit at all. Record it so the quality gate fails and the report says so.
    state.relevance_scored = batches_ok > 0 or batches_total == 0
    if not state.relevance_scored:
        print(f"[relevance] ALL {batches_total} batch(es) failed — the set was never scored")

    from src.config import RELEVANCE_THRESHOLD, RELEVANCE_FLOOR
    THRESHOLD = RELEVANCE_THRESHOLD
    min_keep = max(1, state.config.min_questions)

    # Apply scores + LLM difficulty tags. A question the model OMITTED — whether from a fully-FAILED
    # batch or a partially-scored one — was NEVER actually judged, so we must not treat "no score" as
    # "irrelevant". Both cases get a neutral THRESHOLD score (kept / backfill-eligible) rather than
    # being silently dropped below the floor; dedup and final selection trim any genuine excess later.
    default_unscored = THRESHOLD
    scored = []  # (q_id, score)
    for q_id, q in items:
        if q_id in score_by_qid:
            score = score_by_qid[q_id]
        elif q_id in failed_qids:
            score = THRESHOLD
        else:
            score = default_unscored
        q.relevance_score = score
        if q_id in diff_by_qid:
            q.difficulty = diff_by_qid[q_id]
        scored.append((q_id, score))

    # Keep everything at/above THRESHOLD. If that's fewer than `min_keep`, top up toward min_keep
    # ONLY from candidates at/above RELEVANCE_FLOOR (never pad below the floor). A genuinely thin
    # on-topic pool therefore returns FEWER questions instead of loosely-related filler.
    scored.sort(key=lambda t: t[1], reverse=True)
    keep_ids = {q_id for q_id, s in scored if s >= THRESHOLD}
    if len(keep_ids) < min_keep:
        for q_id, s in scored:
            if len(keep_ids) >= min_keep:
                break
            if s >= RELEVANCE_FLOOR:
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

    # SEMANTIC dedup (embeddings) catches REWORDED near-duplicates that TF-IDF misses
    # ("What are LLMs?" ≈ "What are Large Language Models in AI?"). Fall back to TF-IDF if unavailable.
    from src import embeddings
    from src.config import DEDUP_SEMANTIC_THRESHOLD
    sim_matrix = embeddings.cosine_matrix(texts)
    if sim_matrix is not None:
        dup_threshold = DEDUP_SEMANTIC_THRESHOLD
    else:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        sim_matrix = cosine_similarity(vectorizer.fit_transform(texts))
        dup_threshold = DEDUP_THRESHOLD

    # Keep the BEST representative of a duplicate cluster: highest relevance, then a REAL company
    # (over an unattributed NIAT row), then a sensible source order (curated/seed > web > generated).
    source_priority = {"seed": 0, "nxtmock": 0, "interview_db": 0, "curriculum": 1, "web": 2, "generated": 3}

    def _real_company(q):
        c = q.asked_in_company
        return bool(c) and str(c).strip().upper() != "NIAT"

    def _keep_rank(q):
        return (-(q.relevance_score if q.relevance_score is not None else 0.0),
                0 if _real_company(q) else 1,
                source_priority.get(q.source, 9))

    to_remove = set()
    for i in range(len(questions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(questions)):
            if j in to_remove:
                continue
            if sim_matrix[i][j] > dup_threshold:
                # Drop the weaker of the pair (keep higher relevance / real company / better source).
                worse = j if _keep_rank(questions[i]) <= _keep_rank(questions[j]) else i
                to_remove.add(worse)
                # If the OUTER pivot i is the one being removed, stop comparing against it — a
                # question that is itself being deleted must not keep acting as the dedup pivot
                # (that would remove later j's as "duplicates of a deleted question").
                if worse == i:
                    break

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


# How far a level's share may sit from its target before the mix counts as unbalanced. Shared by the
# `balanced` verdict and the achievability check so the two can never disagree about what "balanced"
# means — they did, and the stricter one reported satisfiable mixes as impossible.
DIFFICULTY_TOLERANCE = 0.25


def _difficulty_feasible(total: int, target: dict[str, float],
                         available: dict[str, int]) -> tuple[bool, dict]:
    """Can a set of `total` questions drawn from `available` land inside every tolerance band?

    Returns (feasible, shortfall). Each band allows a count in
    [ceil((share − tol)·total), floor((share + tol)·total)]. The mix is reachable when every band's
    minimum can be supplied AND the per-level maxima, capped by what exists, still total `total`.
    The second half is what catches an EXCESS: the live run had 4 Medium in a 3-Medium band with no
    Easy or spare Hard to swap in, so the set was stuck despite no level being individually short.
    """
    import math

    if total <= 0:
        return True, {}
    lower, upper, shortfall = {}, {}, {}
    for level, share in target.items():
        lower[level] = max(0, math.ceil((share - DIFFICULTY_TOLERANCE) * total))
        upper[level] = max(0, math.floor((share + DIFFICULTY_TOLERANCE) * total))
        have = available.get(level, 0)
        if have < lower[level]:
            shortfall[level] = {"need": lower[level], "available": have}

    if sum(lower.values()) > total:          # bands cannot coexist at this set size
        return False, shortfall
    capacity = sum(min(upper[lvl], available.get(lvl, 0)) for lvl in target)
    if capacity < total:
        # No single level is necessarily short — the set simply cannot be re-mixed into the bands.
        for level in target:
            have = available.get(level, 0)
            if have < upper[level] and level not in shortfall:
                shortfall[level] = {"need": upper[level], "available": have}
        return False, shortfall
    return not shortfall, shortfall


def tool_check_difficulty_balance(state: AgentState) -> dict:
    """Difficulty mix against the target for THIS session's type.

    The target used to be one global 30/50/20 literal. A code-heavy session's questions are
    implementation and design work, which sits higher on the scale than recall, so a global target
    reported a correctly-weighted code set as unbalanced (and vice versa for theory).
    """
    from src.config import difficulty_targets, normalize_session_type

    session_type = normalize_session_type(
        state.session_context.session_type if state.session_context else None)
    target = difficulty_targets(session_type)

    counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    for q in state.questions.values():
        counts[q.difficulty or "Medium"] = counts.get(q.difficulty or "Medium", 0) + 1
    for q in state.coding_questions.values():
        counts[q.difficulty or "Medium"] = counts.get(q.difficulty or "Medium", 0) + 1
    total = sum(counts.values())
    actual = {k: round(v / max(total, 1), 2) for k, v in counts.items()}
    balanced = total > 0 and all(
        abs(actual.get(k, 0) - v) < DIFFICULTY_TOLERANCE for k, v in target.items())

    # ACHIEVABILITY. "Unbalanced" alone tells the agent to fix something it may have no material to
    # fix with. A real code_heavy run reported E:0 M:4 H:1 against a 20/50/30 target and returned
    # "Fix" on all three revision rounds, because the pool held zero Easy candidates — the agent
    # burned both revisions on an impossible instruction. Say what is actually available so the agent
    # can stop, and so the report can record the shortfall as a corpus gap rather than a set defect.
    available = dict(counts)
    for qid, q in state.reserve.items():
        if qid in state.excluded:
            continue
        d = q.difficulty or "Medium"
        available[d] = available.get(d, 0) + 1

    # Feasibility is judged against the SAME tolerance band that defines `balanced` — not against the
    # exact target share. Demanding the exact proportion called a satisfiable set unachievable
    # (code_heavy wants 30% Hard, which at 5 questions rounds to 2, yet 1 Hard is already inside the
    # band). A set of `total` questions can be balanced iff every band's minimum can be met and the
    # bands' maxima, capped by what actually exists, can still add up to `total`.
    achievable, shortfall = _difficulty_feasible(total, target, available)

    out = {"total": total, "counts": counts, "actual_pct": actual, "balanced": balanced,
           "session_type": session_type, "target_pct": target,
           "available_by_difficulty": available, "achievable": achievable}
    if not balanced and not achievable:
        out["note"] = (
            "Target NOT achievable from the available pool — "
            + "; ".join(f"{lvl}: need {v['need']}, only {v['available']} exist"
                        for lvl, v in sorted(shortfall.items()))
            + ". Do not spend revision rounds on this; the shortfall is a source-coverage limit."
        )
    return out


def tool_check_outcome_coverage(state: AgentState) -> dict:
    """Which learning outcomes the current set covers.

    Uses the SAME measure as the quality report (`pipeline._outcome_coverage`). It used to test whether
    any word longer than three characters from an outcome appeared anywhere in the CONCATENATED text of
    every question — which reports near-100% coverage for almost any set, since one question mentioning
    "model" satisfied every outcome containing that word. The agent was therefore told coverage was
    fine while the report scored it low, and had no way to act on the difference.
    """
    if not state.session_context:
        return {"error": "Call understand_session first"}
    outcomes = state.session_context.learning_outcomes
    if not outcomes:
        return {"total_outcomes": 0, "covered": 0, "missing_count": 0, "missing": [],
                "coverage_pct": 1.0, "note": "This session has no resolved learning outcomes."}

    # Prefer the syllabus audit's judged coverage, exactly as the report does, so the agent and the
    # report can never disagree. It is only available from the second round onward (the audit runs
    # inside submit), so the measure in use is always stated.
    judged = getattr(state, "judged_coverage", None) or {}
    if judged.get("method") and (judged.get("covered") or judged.get("missing")):
        covered, missing = list(judged["covered"]), list(judged["missing"])
        measure = ("judged against the reading material — an outcome counts only when a question "
                   "actually examines it (same as the quality report)")
    else:
        from src.pipeline import _outcome_coverage_detail
        covered, missing = _outcome_coverage_detail(state)
        measure = ("semantic similarity to each outcome (same as the quality report); credits NEARBY "
                   "questions, so treat it as an upper bound")
    total = len(covered) + len(missing) or len(outcomes)
    return {
        "total_outcomes": total, "covered": len(covered),
        "missing_count": len(missing), "missing": missing,
        "coverage_pct": round(len(covered) / max(total, 1), 2),
        "measure": measure,
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


def tool_generate_coding_questions(state: AgentState, count: int = 0, topics: list[str] = None, language: str = "Python") -> dict:
    """Blocked — coding questions are NOT generated. Only real coding questions found in retrieval are
    used (if any); otherwise the coding set is left empty."""
    return {
        "generated": 0,
        "blocked": True,
        "reason": (
            "Coding-question generation is disabled. No questions are created — only real, retrieved "
            "questions are used. Proceed to submit_question_set with the questions you have."
        ),
    }


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
    dropped_form = 0
    for rec in records[:capacity]:
        # FORM gate. This was the only harvest path that skipped it — the bank applies it at build
        # time and the Tavily path applies it inline — so enabling GITHUB_ENABLED would have let page
        # headings and fragments straight into the pool.
        content = strip_artifacts(rec.question_text)
        if topic_keywords:                       # split compound Qs to the on-topic clause
            content = _trim_to_topic(content, topic_keywords)
            if not content:
                continue
        if not is_quality_question(content):
            dropped_form += 1
            continue
        q_id = str(uuid.uuid4())
        qd = QuestionDetail(
            question_id=q_id,
            category="THEORY",
            content=content,
            topic=rec.source_type.split(":")[-1] if ":" in rec.source_type else "Interview",
            difficulty="Medium",
            # Tagged "github", not "web": mislabelling it inflated the web source count and the
            # diversity metric while hiding that GitHub contributed anything.
            source="github",
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
        "dropped_malformed": dropped_form,
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
    interview_topics = getattr(ctx, "interview_topics", None) or []
    for term in (ctx.scope_in + ctx.key_concepts + ctx.learning_outcomes + list(interview_topics)):
        for w in term.lower().split():
            if len(w) >= 4 and w not in _QUERY_STOPWORDS:
                kws.add(w)
    return kws


def _trim_to_topic(text: str, topic_keywords: set) -> str:
    """Drop an unrelated TRAILING clause from a compound question. "" if nothing is on-topic.

    This rewrites a real question's text, so it is deliberately conservative: it only ever removes
    whole trailing clauses, never reorders or rephrases, and it bails out (returning the original)
    if the result stops looking like a question. The re-join uses the ORIGINAL separator rather than
    forcing ". ", which used to turn "X, and how does Y?" into "X. and how does Y?" — a silent
    mutation of sourced content that then got exported as what the company asked.
    """
    clauses = split_into_clauses(text)
    if len(clauses) <= 1:                        # single ask → keep or drop, never rewrite
        low = text.lower()
        return text if any(k in low for k in topic_keywords) else ""
    last_hit = -1
    for i, c in enumerate(clauses):
        if any(k in c.lower() for k in topic_keywords):
            last_hit = i
    if last_hit < 0:
        return ""
    if last_hit == len(clauses) - 1:             # every clause on-topic → keep whole, unmodified
        return text
    # Cut the original string at the end of the last on-topic clause, so the surviving prefix is
    # byte-identical to the source (punctuation and spacing included).
    tail = clauses[last_hit]
    cut = text.find(tail)
    trimmed = text[:cut + len(tail)].strip() if cut >= 0 else ". ".join(clauses[:last_hit + 1]).strip()
    if trimmed and not trimmed.endswith(("?", ".", "!")):
        trimmed += "?" if "?" in text else "."
    return trimmed if looks_like_question(trimmed) else text


def tool_search_web_questions(state: AgentState, outcomes: list) -> dict:
    """Harvest real interview questions with company attribution from 65+ verified domains via Tavily."""
    from src import config as cfg
    if not cfg.TAVILY_API_KEY:
        state.web_status = "no_key"
        return {"error": "TAVILY_API_KEY not set — skipping web search", "status": "skipped"}
    # Pre-flight health check (in RetrievalAgent) failed with a terminal error → don't burn calls.
    if getattr(state, "web_search_disabled", False):
        return {"found": 0, "status": "skipped",
                "warning": f"Web search skipped — Tavily API unavailable ({state.web_status})."}

    from src.sources.tavily_search import TavilyConnector
    from src.config import WEB_POOL_CAP
    ctx = state.session_context

    # Check the web quota BEFORE calling Tavily — don't burn API calls when web is full.
    capacity = WEB_POOL_CAP - state.added_by_source.get("web", 0)
    if capacity <= 0:
        if state.web_status == "not_run":
            state.web_status = "full"
        return {"found": 0, "warning": f"Web quota full ({WEB_POOL_CAP})."}

    # Query the session's substantive CONCEPTS. interview_topics come FIRST — these are the
    # transferable, interview-relevant concepts (so hands-on tool/build sessions retrieve real
    # questions about the underlying skills, not the tool UI), and the first term also carries the
    # role/job-title queries in TavilyConnector.fetch. Then key_concepts + scope_in + agent-passed.
    interview_topics = (ctx.interview_topics if ctx and getattr(ctx, "interview_topics", None) else [])
    seen_terms: set[str] = set()
    search_terms: list[str] = []
    for t in (list(interview_topics[:6])
              + (ctx.key_concepts[:4] if ctx and ctx.key_concepts else [])
              + (ctx.scope_in[:4] if ctx and ctx.scope_in else [])
              + list(outcomes)):
        tl = t.lower().strip()
        if tl and tl not in seen_terms:
            seen_terms.add(tl)
            search_terms.append(t)
    # Cap terms for latency — each term costs a couple of Tavily calls (plus role queries on the
    # first term). Raised from 4 → 8 so multi-outcome sessions get broader recall.
    search_terms = search_terms[:8]
    records, tavily_calls, tavily_error = TavilyConnector().fetch(search_terms or outcomes)
    state.api_usage["tavily_calls"] += tavily_calls

    # Surface a real failure (quota/auth/rate limit) instead of silently looking
    # like "no web questions found".
    if tavily_error and not records:
        el = tavily_error.lower()
        state.web_status = ("quota" if ("quota" in el or "usage limit" in el or "exceeds your plan" in el)
                            else "auth" if ("unauthorized" in el or "invalid api key" in el or "401" in el)
                            else "rate" if "rate limit" in el
                            else "error")
        state.web_error = tavily_error
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
    from src.quality import is_quality_question, strip_artifacts
    take = min(len(records), capacity + 40)
    added = []
    for rec in records[:take]:
        # Trim a two-part question down to its on-topic clause (drops an unrelated
        # trailing ask); "" means no clause is on-topic → skip the whole record.
        content = strip_artifacts(rec.question_text)
        if topic_keywords:
            content = _trim_to_topic(content, topic_keywords)
            if not content:
                continue
        # Form-quality gate — reject boilerplate/logistics/fragments/headings.
        if not is_quality_question(content):
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
    # Web search succeeded this run (records returned) — mark ok once, never downgrade a prior ok.
    if records:
        state.web_status = "ok"
    elif state.web_status in ("not_run", "full"):
        state.web_status = "empty"
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


# Minimum cosine similarity to a rejected question before the reword penalty applies at all.
# Measured on the real label set: a genuine reword scores ≈1.0, while same-domain-but-different
# questions (including reviewer-APPROVED ones) top out around 0.73 — so 0.75 separates them.
REWORD_FLOOR = 0.75


def _feedback_penalty(texts: list[str]) -> list[float]:
    """How much more each text resembles a REJECTED question than an ACCEPTED one (0.0 = not at all).

    This is a NARROW anti-reword mechanism, not a general model of the reviewer's taste, and the two
    guards below are what keep it that way:

      1. An absolute floor. A candidate must be at least `REWORD_FLOOR` similar to some rejected
         question before any penalty applies. Without it the score punishes topics the reviewer
         simply hasn't labelled yet: with a label set about agents and prompting, an unrelated
         "How does a diffusion model add noise?" scored a LARGER penalty than an actual reworded
         rejection, purely because nothing in the accepted set resembled it either.
      2. A relative margin. Within one domain everything is broadly similar to everything — an
         approved "What are the core components of an AI Agent?" sits at 0.73 cosine to a rejected
         agent-architecture question. Subtracting the accepted side cancels that shared baseline:

             penalty = max(0, sim_to_nearest_rejected − sim_to_nearest_accepted)

    Broad taste calibration is handled elsewhere and more honestly — by showing the judge real
    examples (`_feedback_examples_block`) and by reporting predicted acceptance in the quality report.

    All of the reviewer's decisions count, not just this session's: rejections are overwhelmingly
    about the KIND of question ("describe your experience building X") rather than the session.
    Returns all-zeros when either side is missing or embeddings are unavailable.
    """
    zeros = [0.0] * len(texts)
    if not texts:
        return zeros
    try:
        from src import embeddings, memory as _memory
        examples = _memory.get_feedback_examples()
        rejected = [e["question"] for e in examples
                    if e.get("decision") == "bad" and (e.get("question") or "").strip()]
        accepted = [e["question"] for e in examples
                    if e.get("decision") == "good" and (e.get("question") or "").strip()]
        # Both sides are needed for the baseline to cancel; with only rejections every candidate
        # would be penalised roughly equally, which is the same as no signal but costs quality.
        if not rejected or not accepted:
            return zeros
        bad_sim = embeddings.cosine_matrix(texts, rejected)
        good_sim = embeddings.cosine_matrix(texts, accepted)
        if bad_sim is None or good_sim is None:
            return zeros
        out = []
        for i in range(len(texts)):
            nearest_bad = float(max(bad_sim[i]))
            if nearest_bad < REWORD_FLOOR:      # not a reword of anything rejected → leave it alone
                out.append(0.0)
                continue
            out.append(max(0.0, nearest_bad - float(max(good_sim[i]))))
        return out
    except Exception:  # noqa: BLE001 — selection must never fail on the feedback layer
        return zeros


def _select_final(questions: list, k: int, outcomes: list, role_tags: set | None = None,
                  session_type: str | None = None) -> list:
    """Order/pick k questions balancing relevance, diversity, outcome coverage, difficulty, and role.

    Greedy per-step score for candidate c given the already-selected set S:
        λ·relevance(c) − (1−λ)·max_{s∈S} sim(c,s)
        + COVERAGE_BONUS   if c covers a learning outcome not yet covered by S
        + DIFFICULTY_BONUS if c's difficulty bucket is still under its target quota
    relevance = LLM relevance_score; sim/coverage = TF-IDF cosine (same vectorizer as dedup).
    Seeds with the most relevant question. This is the MMR core (relevance + diversity) with
    coverage and difficulty nudges so the final set spans the session's outcomes and levels.
    """
    from src.config import (MMR_LAMBDA, SELECT_COVERAGE_BONUS, SELECT_DIFFICULTY_BONUS,
                            SELECT_SESSION_BONUS, SELECT_ATTRIBUTION_BONUS, SELECT_ROLE_BONUS,
                            SELECT_REJECTED_PENALTY)
    role_tags = role_tags or set()

    def _rel(q):
        return q.relevance_score if q.relevance_score is not None else 0.0

    def _attr(q):  # True if backed by a credible real company (an unattributed row reads NIAT)
        c = q.asked_in_company
        return bool(c) and str(c).strip().upper() != "NIAT"

    if not questions or k <= 0:
        return []
    # Always run the MMR ordering (even when keeping ALL — k == len) so the role/coverage/difficulty
    # bonuses shape the ORDER, not just a flat relevance sort. k is clamped to the pool size.
    k = min(k, len(questions))

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

    # Penalty for looking more like a rejected question than an accepted one. Exact repeats are
    # removed earlier by `_drop_rejected`, but that is a normalized-string match, so a REWORDING of a
    # rejected question sails through and gets rejected all over again. A ranking penalty, not a hard
    # drop, so a session with a thin pool still fills.
    reject_penalty = _feedback_penalty(texts)

    # Difficulty target counts for k, weighted for THIS session's type — code-heavy sessions skew
    # harder because implementation and design questions sit above recall on the scale.
    from src.config import difficulty_targets
    _mix = difficulty_targets(session_type)
    diff_target = {level: round(k * share) for level, share in _mix.items()}

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
            # Role ranking bonus: a question tagged for a TARGET role ranks above generic ("General").
            role_need = SELECT_ROLE_BONUS if (role_tags and questions[c].role in role_tags) else 0.0
            score = (MMR_LAMBDA * rel[c] - (1 - MMR_LAMBDA) * div
                     + cov_new + need + sess_need + attr_need + role_need
                     - SELECT_REJECTED_PENALTY * reject_penalty[c])
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
    """Tag each question with the session it most resembles. Single-session topics tag all with it.

    Scores against CHUNKS and takes the max per session — never one concatenated blob. A blob makes
    the comparison a function of how long each session's reading material is rather than how well it
    matches: on a real two-session run every question landed on the first session (12/0) because the
    second session's 8.8k-char blob diluted every specific match to 0.17–0.41. The same questions
    split 6/6 once chunked. `_session_profile` produces the chunks, so this and `_score_session_fit`
    now use the same grounding.

    Reading-material prose is discounted the same way the fit gate discounts it: a setup walkthrough
    about pasting an API key otherwise out-matches a precise outcome statement.
    """
    names = list(profiles.keys())
    if not names:
        return
    if len(names) == 1:
        for q in questions:
            q.session = names[0]
        return
    if not questions:
        return
    from src import embeddings
    from src.config import SESSION_PROFILE_RM_WEIGHT

    qtexts = [q.content for q in questions]

    def _texts(name) -> tuple[list[str], list[str]]:
        prof = profiles[name]
        if isinstance(prof, dict):
            return list(prof.get("curated") or []), list(prof.get("rm") or [])
        return ([prof] if isinstance(prof, str) else list(prof or [])), []   # legacy/cached shape

    # [n_questions x n_sessions] best weighted similarity.
    scores = [[0.0] * len(names) for _ in qtexts]
    embeddings_ok = False
    for s_idx, name in enumerate(names):
        curated, rm = _texts(name)
        for texts, weight in ((curated, 1.0), (rm, SESSION_PROFILE_RM_WEIGHT)):
            if not texts:
                continue
            sim = embeddings.cosine_matrix(qtexts, texts)
            if sim is None:
                continue
            embeddings_ok = True
            for q_idx in range(len(qtexts)):
                scores[q_idx][s_idx] = max(scores[q_idx][s_idx], float(max(sim[q_idx])) * weight)

    if not embeddings_ok:
        # TF-IDF fallback. Still per-session, but a bag of chunks joined per session is all TF-IDF can
        # compare, so length bias returns here — acceptable only because embeddings are unavailable.
        joined = [" ".join(sum(_texts(n), [])) or n for n in names]
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        mat = vec.fit_transform(joined + qtexts)
        sims = cosine_similarity(mat[len(names):], mat[:len(names)])
        for i, q in enumerate(questions):
            q.session = names[int(sims[i].argmax())]
        return

    for i, q in enumerate(questions):
        q.session = names[max(range(len(names)), key=lambda s: scores[i][s])]


# ── Post-relevance scope trim ────────────────────────────────────────────────
#
# Drops an off-syllabus sub-clause from an otherwise on-topic question:
#   "how you would iteratively improve prompts and guards to increase reliability."
#     → "how you would iteratively improve prompts"          (guards are not taught in the session)
#
# WHY THIS RUNS AFTER THE RELEVANCE GATE, on the selected set only.
#
# The tempting version — "at retrieval time, cut any trailing conjunct not grounded in the session" —
# was measured across the 1400-row GenAI bank and fires on **155 rows**, destroying the entire
# comparison class:
#   "…the difference between supervised and unsupervised learning?"  → cuts "unsupervised"
#   "What is the difference between top-k and top-p sampling?"       → cuts "top-p"
#   "What are the trade-offs between RAG and fine-tuning?"           → cuts "fine-tuning"
#   "Explain self-attention and multi-head attention."               → cuts "multi-head"
#   "how do you detect and reduce them?" / "Design and conduct …"    → cuts a VERB, not a topic
# The same rule applied to the FINAL SELECTED set fires on 1 of 5 questions — exactly the intended
# one. The reason is simple: a supervised-vs-unsupervised question is rejected as off-topic for this
# session long before any trim, so it never reaches the trimmer. An off-topic question must be
# REJECTED, never surgically edited into a different question.
#
# This is NOT `_trim_to_topic`. That one splits COMPOUND questions at clause boundaries between
# separate asks and deliberately never rewrites a single ask (`and guards` is a noun, so
# `split_into_clauses` correctly returns one clause). Both are kept; they solve different problems.

# Frames where both sides of the "and" are constitutive — the largest hazard measured. Never trim.
_COMPARISON_FRAME = re.compile(
    r"\b(?:difference|differences|distinction|trade-?offs?|compare|comparison|versus|vs\.?|"
    r"pros and cons|advantages and disadvantages|similarities)\b", re.IGNORECASE)

# Fixed pairs that are one idea, not two. Splitting these produces nonsense ("What are the pros").
_IDIOM_PAIR = re.compile(
    r"\b(?:pros and cons|advantages and disadvantages|benefits and (?:drawbacks|limitations|risks)|"
    r"strengths and weaknesses|inputs? and outputs?|read and write|trial and error|"
    r"question and answer|terms and conditions|risks and (?:benefits|mitigations))\b", re.IGNORECASE)

# A backstop against a "trim" that guts the question. Kept low (0.4) because the real discriminator is
# the shared-head-noun rule below, not length: "Explain chain-of-thought prompting and vector store
# indexing." is a legitimate trim that keeps only 3 of 7 words (0.43).
_TRIM_MIN_WORD_RATIO = 0.4

_CONJUNCTIONS = {"and", "or"}
# Verb stems that appear as the second half of a verb pair SHARING one object ("detect and reduce
# them"). Cutting one leaves the object dangling from a single verb. Narrow on purpose — see
# `_shares_object` — because "design a prompt and evaluate BLEU scores" is a genuinely separate ask
# where the second verb has its own object, and blocking that was over-broad.
_VERB_CONJUNCT_HINTS = {
    "reduce", "mitigate", "detect", "prevent", "conduct", "optimize", "optimise", "evaluate",
    "deploy", "monitor", "measure", "validate", "test", "debug", "maintain", "scale", "improve",
    "handle", "manage", "explain", "describe", "implement", "design", "build", "create", "compare",
}
# A cut ending in one of these has no object of its own — it borrowed the head's.
_OBJECT_PRONOUNS = {"them", "it", "this", "that", "these", "those", "one", "both", "each"}

# A trim must not stop on a function word. Without this, a model that cut one word too late produced
# "How would you iteratively improve prompts and?" — a prefix, above the word ratio, and accepted by
# the form gate because it starts with "How" and ends with "?". Grammatically it is broken.
_DANGLING_TAIL = {
    "and", "or", "but", "nor", "the", "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "into", "about", "between", "across", "over", "under", "than", "then",
    "when", "while", "if", "is", "are", "was", "were", "do", "does", "did", "how", "what", "why",
    "using", "via", "per", "vs", "versus",
}


def _shares_object(cut: list[str]) -> bool:
    """True when the cut span is a verb sharing the head's object ("… and reduce them?").

    Distinguishes that from a second ask carrying its own object ("… and evaluate BLEU scores?"),
    which is exactly the off-syllabus tail this pass is meant to remove.
    """
    return (len(cut) >= 2 and cut[0] in _CONJUNCTIONS and cut[1] in _VERB_CONJUNCT_HINTS
            and len(cut) <= 3 and cut[-1] in _OBJECT_PRONOUNS)


def _shares_head_noun(kept: list[str], cut: list[str]) -> bool:
    """True when the cut repeats a noun from the kept part — an ELIDED comparison, never a trim.

    "Explain self-attention | and multi-head attention." names two kinds of the same thing, so the
    second half is constitutive even though no "difference between" frame is present. Measured as the
    one hazard that survived every other guard. Contrast "…prompting | and vector store indexing",
    which shares nothing and is a real off-syllabus tail.
    """
    kept_long = [w for w in kept if len(w) >= 4]
    for c in cut:
        if len(c) < 4 or c in _CONJUNCTIONS:
            continue
        for k in kept_long:
            if c in k or k in c:
                return True
    return False

_TRIM_SYSTEM = """You remove OFF-SYLLABUS sub-clauses from real interview questions.

You are given a session's scope and a list of questions already judged relevant to it. For each
question decide whether it asks about an extra thing that this session does NOT teach, tacked on to
something it does teach.

Rules you MUST follow:
- Return the question TRIMMED BY DELETING A TRAILING PORTION ONLY. Never reword, reorder, add or
  substitute a single word. The result must be a prefix of the original.
- If both sides of an "and" are part of the same comparison or contrast, or the question is about the
  relationship between them, return it UNCHANGED. Comparisons need both halves.
- If the conjunction joins two VERBS applied to the same object ("detect and reduce them"), return it
  UNCHANGED.
- If everything in the question is within scope, return it UNCHANGED.
- The trimmed result must still read as a complete, answerable question or task.
- Prefer UNCHANGED whenever you are unsure. A missed trim is far cheaper than a mangled question.

Respond with JSON: {"results": [{"n": 1, "text": "<trimmed or unchanged>"}, ...]} — one entry per
question, in the order given."""


def _norm_words(text: str) -> list[str]:
    """Lowercased word list, punctuation stripped — the unit the prefix check compares."""
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'&/-]*", (text or "").lower()) if w]


def _accept_trim(original: str, candidate: str) -> str | None:
    """The trimmed text if it is a legitimate trim of `original`, else None (caller keeps original).

    Every check here is mechanical and runs in OUR code, not in the model's. The model is asked for a
    prefix; whether it returned one is verified, because a reworded question attributed to a real
    company is precisely the failure this whole path has to avoid.
    """
    cand = (candidate or "").strip()
    orig = (original or "").strip()
    if not cand or cand == orig:
        return None
    if _COMPARISON_FRAME.search(orig) or _IDIOM_PAIR.search(orig):
        return None                       # both sides constitutive → never trim
    if len(cand) >= len(orig):
        return None                       # a "trim" that grew is a rewrite

    ow, cw = _norm_words(orig), _norm_words(cand)
    if not cw or len(cw) < 3:
        return None
    if cw != ow[:len(cw)]:
        return None                       # not a contiguous prefix → reworded or reordered
    if len(cw) / len(ow) < _TRIM_MIN_WORD_RATIO:
        return None                       # gutted rather than trimmed
    if cw[-1] in _DANGLING_TAIL:
        return None                       # cut one word too late — "…improve prompts and?"
    if len(cw) >= 2 and cw[-2] == "to":
        return None                       # cut inside an infinitive — "…prompts and guards to increase?"

    # What was cut must be an off-syllabus TARGET, not the verb the question hangs on, and not the
    # second half of an elided comparison.
    cut = ow[len(cw):]
    if _shares_object(cut) or _shares_head_noun(cw, cut):
        return None

    # Re-punctuate from the original: a trimmed question keeps the original's mark, not a bare cut.
    # Strip any separator the cut left dangling FIRST. A live Haiku run cut at clause commas and this
    # appended the mark straight onto them, shipping "Are you aware of agentic workflows,?" and
    # "…for a fictional client,." — both of which sail through the form gate because they do end in a
    # valid mark. The trailing comma is in the string, not in the word list `cw`, so the dangling-tail
    # check above cannot see it.
    cand = cand.rstrip().rstrip(",;:-–—").rstrip()
    if not cand:
        return None
    if not cand.endswith(("?", ".", "!")):
        cand += "?" if orig.rstrip().endswith("?") else "."
    return cand if is_quality_question(cand) else None


def _scope_trim(state: AgentState, questions: list) -> list[dict]:
    """Trim off-syllabus trailing clauses from the SELECTED questions. Returns a log of what changed.

    Fail-open: any error, unparseable reply or rejected candidate leaves the set untouched. One LLM
    call per run, over ~5–60 questions.
    """
    ctx = state.session_context
    if not ctx or not questions:
        return []

    scope_in = list(getattr(ctx, "scope_in", None) or [])[:24]
    outcomes = list(getattr(ctx, "learning_outcomes", None) or [])[:24]
    scope_out = list(getattr(ctx, "scope_out", None) or [])[:24]
    if not scope_in and not outcomes:
        return []                         # nothing to judge against

    numbered = [{"n": i + 1, "q": q.content[:600]} for i, q in enumerate(questions)]
    user = (
        f"SESSION TEACHES (in scope):\n- " + "\n- ".join(scope_in or outcomes)
        + f"\n\nLEARNING OUTCOMES:\n- " + "\n- ".join(outcomes)
        + (f"\n\nEXPLICITLY NOT IN THIS SESSION:\n- " + "\n- ".join(scope_out) if scope_out else "")
        + f"\n\nQUESTIONS:\n{json.dumps(numbered)}"
    )
    try:
        result = chat_completion_json(
            model=run_model(state),        # the run's model, never the UI global
            system_prompt=_TRIM_SYSTEM,
            user_prompt=user,
            max_tokens=4096,
            on_usage=_usage_cb(state),
        )
    except Exception as exc:  # noqa: BLE001 — a failed trim must not lose the question set
        print(f"[scope_trim] skipped ({type(exc).__name__}: {exc})")
        return []

    trims: list[dict] = []
    for row in (result.get("results") or []):
        try:
            idx = int(row["n"]) - 1
            proposed = str(row.get("text") or "")
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(questions)):
            continue
        q = questions[idx]
        accepted = _accept_trim(q.content, proposed)
        if not accepted:
            continue
        trims.append({"question_id": q.question_id, "before": q.content, "after": accepted})
        q.original_content = q.content     # keep the verbatim source for audit
        q.content = accepted
    if trims:
        print(f"[scope_trim] trimmed {len(trims)} question(s) to the session's scope")
    return trims


# Minimum similarity to a session Knowledge Point before a question is tagged with it. Deliberately
# permissive: the questions reaching this point have already cleared the session-fit floor, so the job
# is picking WHICH KP, not re-deciding relevance. Below this the question is left untagged rather than
# given the least-wrong label.
KP_LABEL_MIN_SIM = 0.30


def _assign_kp_labels(questions: list, context) -> int:
    """Tag each selected question with the session Knowledge Point it best matches.

    `QuestionDetail.kp_label` existed and was written by nothing: retrieved questions carry the bank's
    own `topic`, never a KP, so every question in every exported set had `kp_label = None`. The
    session already resolves its KPs (`SessionContext.matched_kp_ids`), so this is a local
    embedding match, not another LLM call.

    Fail-open and honest: returns 0 and leaves labels as None when embeddings are unavailable or the
    session resolved no KPs — an untagged question is better than a confidently wrong tag.
    """
    kps = [kp for kp in (getattr(context, "matched_kp_ids", None) or [])
           if (getattr(kp, "kp_label", "") or "").strip()]
    if not questions or not kps:
        return 0
    from src import embeddings

    labels = list(dict.fromkeys(kp.kp_label.strip() for kp in kps))
    sim = embeddings.cosine_matrix([q.content for q in questions], labels)
    if sim is None:
        return 0
    tagged = 0
    for i, q in enumerate(questions):
        best = int(sim[i].argmax())
        if float(sim[i][best]) >= KP_LABEL_MIN_SIM:
            q.kp_label = labels[best]
            tagged += 1
    return tagged


# ── Syllabus audit: is the concept taught here, and which outcome does the question really test? ──
#
# Two problems this replaces, both measured on the "Introduction to AI Agents + Building a Learning
# Path Generator" run:
#
# 1. ON-DOMAIN IS NOT ON-SYLLABUS. The quality gate said "All 12 theory questions are on-domain … no
#    off-domain items detected" and it was right — every question was about AI agents. But four tested
#    concepts that appear NOWHERE in either session's reading material: `fallback`/`ambiguous intent`,
#    `guardrails`/`infinite loops`, `production`/`diagnose`, `hallucination`. All four were the Hard
#    questions, so the set earned `difficulty_balance` 0.87 precisely by going off-syllabus. Nothing in
#    the pipeline read the reading material to check.
#
# 2. PROXIMITY WAS BEING SCORED AS COVERAGE. `_outcome_coverage_detail` credits an outcome when any
#    question is within 0.30 cosine, and that is 35% of the composite. Measured false credits:
#      "Integrate multiple Google APIs (Docs, Calendar, Drive)"  ← the HALLUCINATION question (0.38)
#      "Design system prompts that orchestrate agent reasoning"  ← "when does agent reasoning go off
#                                                                  the rails?" (0.69, shared word)
#      "Deploy a no-code automation using n8n"                   ← "deployed into production…" (0.47)
#    Raising the threshold cannot fix this: across the 11 credited pairs, legitimate credits span
#    0.52–0.83 and false ones 0.38–0.69 — they OVERLAP, and so does shared-distinctive-term count
#    (0–2 vs 0–1). Neither signal separates them, so the measure needs judgement against the material,
#    not a different cut-off.
#
# One LLM call over the selected set, and the reply is verified mechanically: a claimed untaught
# concept must genuinely be absent from the session corpus, and outcome indices must be in range.

_SYLLABUS_SYSTEM = """You audit interview questions against the exact material a session teaches.

You get the session reading material, its numbered learning outcomes, and numbered questions already
judged relevant to the domain. For each question report two things:

1. "untaught": the single core concept the question requires that is NOT present in the reading
   material, or null if everything it needs is there. Judge the CONCEPT, not the wording — if the
   material teaches an idea in different words, that counts as taught. Name the concept using words
   from the QUESTION.
2. "covers": the numbers of the learning outcomes this question genuinely EXAMINES. Being about a
   similar area is not enough — a question about hallucination does not examine an outcome about
   integrating Google Docs and Calendar, even though both concern agents. Use [] when it examines
   none of them. Most questions examine zero or one.

Be strict on "covers" and conservative on "untaught": prefer null when the material arguably covers it.

Respond with JSON: {"questions": [{"n": 1, "untaught": null, "covers": [2]}, ...]}"""


def _session_corpus(session_names: list) -> str:
    """Everything a session teaches, lowercased: its reading material plus curated outcomes.

    Used to VERIFY an off-syllabus claim — a concept the model says is untaught must actually be
    absent from here, otherwise the claim is discarded.
    """
    import json as _json
    from src.config import DATA_DIR

    parts: list[str] = []
    try:
        from src.data_loader import get_data_store
        store = get_data_store()
        for name in (session_names or []):
            parts.append(store.get_session_content(name) or "")
    except Exception:  # noqa: BLE001
        pass
    try:
        so = _json.loads((DATA_DIR / "reading_materials" / "session_outcomes.json")
                         .read_text(encoding="utf-8"))
        for name in (session_names or []):
            ov = so.get(name) or {}
            parts += list(ov.get("learning_outcomes", [])) + list(ov.get("interview_topics", []))
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parts).lower()


def _concept_is_absent(concept: str, corpus: str) -> bool:
    """True when none of the concept's DISTINCTIVE words appear in the session corpus.

    The guard against a fabricated off-syllabus claim: the model may only flag a concept the material
    genuinely never mentions.

    Ubiquitous domain words are filtered out first, and that filtering is load-bearing. Requiring every
    word to be missing marked "ambiguous user intent" as taught, because "user" appears throughout any
    agent lesson while "ambiguous" and "intent" appear nowhere. Conversely a plain majority rule would
    let "agent guardrails" fail (1 of 2 missing). Judging only the distinctive words gets both right,
    and a concept made entirely of ubiquitous words yields no claim at all — the conservative outcome.
    """
    words = [w for w in re.findall(r"[a-z][a-z-]{3,}", (concept or "").lower())
             if w not in _SYLLABUS_STOP and w not in _UBIQUITOUS_DOMAIN]
    if not words:
        return False
    return all(w not in corpus and w.rstrip("s") not in corpus for w in words)


_SYLLABUS_STOP = {
    "what", "how", "why", "when", "where", "which", "that", "this", "with", "from", "your", "would",
    "could", "should", "into", "about", "their", "them", "they", "does", "have", "been", "being",
    "such", "than", "then", "there", "these", "those", "when", "while", "using", "used", "make",
    "made", "more", "most", "some", "each", "very", "also", "well", "like", "just", "only",
}
# Words so common in any GenAI lesson that their presence says nothing about whether a CONCEPT is
# taught. Excluded from the absence check so a phrase is judged on its distinctive terms.
_UBIQUITOUS_DOMAIN = {
    "agent", "agents", "agentic", "llms", "model", "models", "tool", "tools", "user", "users",
    "workflow", "workflows", "session", "sessions", "data", "system", "systems", "task", "tasks",
    "prompt", "prompts", "input", "output", "outputs", "response", "responses", "step", "steps",
    "example", "examples", "information", "context", "process",
}


def _syllabus_audit(state: AgentState, questions: list) -> dict:
    """Flag off-syllabus questions and compute JUDGED outcome coverage for the selected set.

    Returns {'off_syllabus': [...], 'coverage': {...}} — `coverage` empty when unavailable, so the
    caller falls back to the embedding measure and reports which method produced the number.
    Fail-open: any error leaves the set untouched and unflagged.
    """
    ctx = state.session_context
    if not ctx or not questions:
        return {"off_syllabus": [], "coverage": {}}
    outcomes = list(getattr(ctx, "learning_outcomes", None) or [])
    names = list(getattr(state.config, "session_names", None) or [])
    if not outcomes:
        return {"off_syllabus": [], "coverage": {}}

    material = []
    try:
        from src.data_loader import get_data_store
        store = get_data_store()
        for name in names:
            content = store.get_session_content(name)
            if content:
                material.append(f"### SESSION: {name}\n{content[:9000]}")
    except Exception:  # noqa: BLE001
        pass
    if not material:
        return {"off_syllabus": [], "coverage": {}}

    numbered_o = "\n".join(f"{i+1}. {o}" for i, o in enumerate(outcomes))
    numbered_q = json.dumps([{"n": i + 1, "q": q.content[:400]} for i, q in enumerate(questions)])
    user = (f"READING MATERIAL:\n{chr(10).join(material)}\n\n"
            f"LEARNING OUTCOMES:\n{numbered_o}\n\nQUESTIONS:\n{numbered_q}")
    try:
        result = chat_completion_json(
            model=run_model(state),          # the run's model, never the UI global
            system_prompt=_SYLLABUS_SYSTEM,
            user_prompt=user,
            max_tokens=4096,
            on_usage=_usage_cb(state),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[syllabus] audit skipped ({type(exc).__name__}: {exc})")
        return {"off_syllabus": [], "coverage": {}}

    corpus = _session_corpus(names)
    off: list[dict] = []
    covered_idx: set[int] = set()
    pairs: list[dict] = []
    for row in (result.get("questions") or []):
        try:
            idx = int(row["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(questions)):
            continue
        q = questions[idx]

        concept = (row.get("untaught") or "").strip() if isinstance(row.get("untaught"), str) else ""
        # Verified, not trusted: the concept must really be missing from the material.
        if concept and _concept_is_absent(concept, corpus):
            q.off_syllabus_concept = concept
            off.append({"question_id": q.question_id, "content": q.content, "concept": concept})

        for raw in (row.get("covers") or []):
            try:
                o_idx = int(raw) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= o_idx < len(outcomes):
                covered_idx.add(o_idx)
                pairs.append({"outcome": outcomes[o_idx], "question": q.content[:120]})

    coverage = {
        "covered": [outcomes[i] for i in sorted(covered_idx)],
        "missing": [o for i, o in enumerate(outcomes) if i not in covered_idx],
        "pairs": pairs,
        "method": "llm-judged",
    }
    if off:
        print(f"[syllabus] {len(off)} question(s) test a concept absent from the session material")
    return {"off_syllabus": off, "coverage": coverage}


def _rank_key(q):
    """Selection value of a candidate: judged relevance first, then session fit."""
    return ((q.relevance_score if q.relevance_score is not None else 0.0),
            (q.session_fit if q.session_fit is not None else 0.0))


def _ensure_session_representation(selected: list, pool: list, session_names: list, k: int) -> dict:
    """Guarantee every configured session that HAS candidates holds at least one slot.

    `_select_final` only *nudges* toward per-session balance with a bonus, and that nudge was dead
    whenever attribution collapsed onto one session (`sess_target` is 0 unless the pool shows >1
    distinct session). A real run shipped 12 questions with an entire session unrepresented; the gate
    flagged `session-gap` and the set shipped anyway. This makes representation a guarantee rather
    than a preference — while staying honest about the case it CANNOT fix: a session with no
    candidates at all is reported, never padded.

    Mutates `selected` in place. Returns {'per_session', 'no_candidates', 'swapped'}.
    """
    names = [n for n in (session_names or []) if n]
    if len(names) < 2 or not selected:
        return {"per_session": {}, "no_candidates": [], "swapped": 0}

    by_session: dict[str, list] = {n: [] for n in names}
    for q in pool:
        if q.session in by_session:
            by_session[q.session].append(q)

    chosen = {q.question_id for q in selected}
    swapped = 0
    for name in names:
        cands = [q for q in by_session[name] if q.question_id not in chosen]
        have = sum(1 for q in selected if q.session == name)
        if have or not cands:
            continue                      # already represented, or nothing to represent it with
        # Displace the weakest question from whichever session is most over-represented, so the swap
        # costs the set as little as possible and never drops another session to zero.
        counts: dict[str, int] = {}
        for q in selected:
            counts[q.session] = counts.get(q.session, 0) + 1
        donors = [q for q in selected if counts.get(q.session, 0) > 1]
        if not donors:
            break                         # every slot is the last one for its session — leave it
        drop = min(donors, key=_rank_key)
        add = max(cands, key=_rank_key)
        selected[selected.index(drop)] = add
        chosen.discard(drop.question_id)
        chosen.add(add.question_id)
        swapped += 1

    final_counts: dict[str, int] = {n: 0 for n in names}
    for q in selected:
        if q.session in final_counts:
            final_counts[q.session] += 1
    return {
        "per_session": final_counts,
        # Configured sessions the RETRIEVAL never found a question for. This is a source-coverage
        # fact, not a selection defect, and the report must say so rather than imply padding.
        "no_candidates": [n for n in names if not by_session[n]],
        "swapped": swapped,
    }


def tool_submit_question_set(state: AgentState) -> dict:
    # Selection pool = current theory questions + reserve (validated-but-not-selected),
    # minus anything explicitly excluded (rejected in a prior revision). This lets a
    # revision BACKFILL from the reserve instead of the set only ever shrinking.
    pool = {**state.reserve, **state.questions}
    for qid in state.excluded:
        pool.pop(qid, None)
    outcomes = state.session_context.learning_outcomes if state.session_context else []
    # SELECTION LEVEL: after the wide pool is relevance-filtered + de-duplicated, trim to the REQUESTED
    # count (the UI 'Target count' slider → config.max_questions) as a CEILING. _select_final ranks by
    # MMR (relevance − redundancy) + coverage/difficulty/session/attribution/role bonuses and keeps the
    # best N. A thin session with fewer than N survivors returns all of them (never padded) since
    # _select_final clamps k to the pool size.
    from src.config import FINAL_SET_CAP, target_roles
    target = min(getattr(state.config, "max_questions", 12) or 12, FINAL_SET_CAP)
    keep_theory = max(0, target - len(state.coding_questions))
    role_tags = target_roles(getattr(state.config, "category", None)).get("bonus_tags", set())

    selected = _select_final(list(pool.values()), keep_theory, outcomes, role_tags=role_tags,
                             session_type=(state.session_context.session_type
                                           if state.session_context else None))
    # Representation is enforced AFTER ranking, so a session that has candidates cannot be shut out by
    # a bonus that only nudges. Sessions with no candidates at all are reported, never padded.
    session_rep = _ensure_session_representation(
        selected, list(pool.values()), list(getattr(state.config, "session_names", None) or []),
        keep_theory)
    state.session_representation = session_rep
    selected_ids = {q.question_id for q in selected}

    # New selected set; everything else in the pool goes to the reserve (recoverable),
    # NOT to `removed` (it wasn't rejected, just not chosen this round).
    state.questions = {q.question_id: q for q in selected}
    state.reserve = {qid: q for qid, q in pool.items() if qid not in selected_ids}

    # Both passes act on the SELECTED set only. The reserve is re-selected each revision round, so
    # working on it would spend embeddings and an LLM call on questions that may never ship — and for
    # the scope trim, running before the relevance gate is actively harmful (see `_scope_trim`).
    kp_tagged = _assign_kp_labels(selected, state.session_context) if state.session_context else 0
    trims = _scope_trim(state, selected)
    if trims:
        state.scope_trims.extend(trims)
    # Reset before re-auditing: a revision round re-selects, so last round's flags must not persist for
    # questions that are no longer in the set.
    for q in selected:
        q.off_syllabus_concept = None
    audit = _syllabus_audit(state, selected)
    state.off_syllabus = audit["off_syllabus"]
    state.judged_coverage = audit["coverage"]

    total = len(state.questions) + len(state.coding_questions)
    state.submitted = True
    return {"submitted": True, "total_questions": total,
            "theory": len(state.questions), "coding": len(state.coding_questions),
            "reserve": len(state.reserve), "kp_tagged": kp_tagged,
            "scope_trimmed": len(trims), "off_syllabus": len(state.off_syllabus),
            "per_session": session_rep.get("per_session") or {},
            "sessions_without_candidates": session_rep.get("no_candidates") or []}


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
