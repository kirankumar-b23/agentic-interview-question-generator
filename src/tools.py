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
    # (over a source-site/NIAT label), then a sensible source order (curated/seed > web > generated).
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
    balanced = total > 0 and all(abs(actual.get(k, 0) - v) < 0.25 for k, v in target.items())
    return {"total": total, "counts": counts, "actual_pct": actual, "balanced": balanced,
            "session_type": session_type, "target_pct": target}


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

    from src.pipeline import _outcome_coverage_detail
    covered, missing = _outcome_coverage_detail(state)
    return {
        "total_outcomes": len(outcomes), "covered": len(covered),
        "missing_count": len(missing), "missing": missing,
        "coverage_pct": round(len(covered) / len(outcomes), 2),
        "measure": "semantic similarity to each outcome (same as the quality report)",
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

    def _attr(q):  # True if backed by a credible real company (not NIAT/source-labeled)
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
