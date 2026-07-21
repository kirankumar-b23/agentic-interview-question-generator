"""Understand the Session — extracts learning outcomes, maps to KPs.

Uses knowledge_graph.json first (instant, no LLM call needed).
Falls back to LLM only when session is not in the knowledge graph.
"""

import json

from src.models import SessionContext, KPMatch, TopicMatch
from src.data_loader import DataStore
from src.llm_client import chat_completion_json
from src import memory
from src.config import SESSION_OUTCOMES_JSON


def _load_outcome_overrides() -> dict:
    """Human-curated per-session outcome/interview_topic overrides (editable JSON). {} if absent."""
    try:
        if SESSION_OUTCOMES_JSON.exists():
            data = json.loads(SESSION_OUTCOMES_JSON.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 — a malformed override file must never break resolution
        print(f"[outcomes] override file unreadable ({e}); ignoring")
    return {}


def _overrides_signature() -> str:
    """Short hash of the override file so edits self-invalidate the resolution cache."""
    import hashlib
    try:
        raw = SESSION_OUTCOMES_JSON.read_bytes() if SESSION_OUTCOMES_JSON.exists() else b""
    except Exception:  # noqa: BLE001
        raw = b""
    return hashlib.md5(raw).hexdigest()[:8]


def _apply_override(name: str, context: SessionContext, overrides: dict) -> SessionContext:
    """If `name` has a curated override, replace its learning_outcomes / interview_topics
    (and scope_out if given) with the curated values. Other fields keep their derived values."""
    ov = overrides.get(name)
    if not isinstance(ov, dict):
        return context
    lo = [o for o in ov.get("learning_outcomes", []) if isinstance(o, str) and o.strip()]
    it = [o for o in ov.get("interview_topics", []) if isinstance(o, str) and o.strip()]
    so = [o for o in ov.get("scope_out", []) if isinstance(o, str) and o.strip()]
    if lo:
        context.learning_outcomes = lo
    if it:
        context.interview_topics = it
    elif lo:
        context.interview_topics = context.interview_topics or lo
    if so:
        context.scope_out = so
    return context


LLM_SYSTEM_PROMPT = """You are an expert curriculum analyst. Given a session's reading material, you must:

1. Extract 3-7 core learning outcomes (what students should know/do after this session)
2. Identify key concepts with clear scope boundaries
3. Identify "interview_topics": the transferable, interview-relevant GenAI concepts a candidate who
   completed this session would actually be asked about in a real interview
4. Determine session type: "theory_heavy", "code_heavy", or "mixed"
5. Map the session to the most relevant Knowledge Point IDs from the provided catalog
6. Map the session to the most relevant interview CSV topic/sub_topic values

Respond in JSON with this exact structure:
{
    "learning_outcomes": ["outcome1", "outcome2", ...],
    "key_concepts": ["concept1", "concept2", ...],
    "interview_topics": ["transferable interview concept", ...],
    "scope_in": ["topic in scope", ...],
    "scope_out": ["topic NOT in scope", ...],
    "session_type": "theory_heavy" | "code_heavy" | "mixed",
    "matched_kp_ids": [
        {"kp_id": "KP_GLOBAL_XXXX", "kp_label": "...", "relevance": 0.9}
    ],
    "matched_csv_topics": [
        {"topic": "TOPIC_NAME", "sub_topic": "SUB_TOPIC_OR_NULL", "confidence": 0.85}
    ]
}

Rules:
- Only match KP IDs that are DIRECTLY relevant to this session's content
- If the session has coding projects/exercises, mark as "code_heavy"
- If it's mostly conceptual/explanatory, mark as "theory_heavy"
- scope_out should list related topics NOT covered in this specific session
- interview_topics MUST be the transferable GenAI skills/concepts, NOT the specific tool/product/UI. For a
  hands-on build session (e.g. building a news summarizer in the n8n tool with RSS/Aggregate/Gmail nodes),
  interview_topics are things like "LLM text summarization", "prompt engineering", "LLM API integration",
  "workflow automation with LLMs", "chaining LLM calls" — NOT "n8n Aggregate Node" or "RSS Feed Read Node".
  For a pure-theory session, interview_topics will closely mirror the key_concepts. Give 3-8 items.
"""


def understand_session(session_names: list[str], data_store: DataStore) -> SessionContext:
    """Extract structured understanding of session(s).

    Each session is resolved from ITS OWN reading material (or the knowledge
    graph) and the results are merged. This keeps every selected unit's content
    fairly represented instead of concatenating them into one truncated blob —
    the key to relevant, on-topic questions for multi-session topics.

    Per-session strategy:
    1. Reading material available → LLM extracts KPs from that session's content
    2. No reading material → knowledge_graph.json (pre-computed KPs)
    3. Not in graph either → pure LLM fallback
    """
    combined_name = " + ".join(session_names)

    # Cache key includes a hash of the sessions' reading material, so an edit to the RM
    # (or a session split/rename) self-invalidates the cache instead of serving stale
    # outcomes forever (the old key was the bare combined name with no invalidation).
    import hashlib
    rm_sig = hashlib.md5(
        " ".join((data_store.get_session_content(n) or "") for n in session_names).encode("utf-8")
    ).hexdigest()[:10]
    # Include the override-file signature so hand-edits to curated outcomes self-invalidate the cache.
    cache_key = f"{combined_name}::{rm_sig}::ov{_overrides_signature()}"

    cached = memory.get_cached_resolution(cache_key)
    if cached:
        return SessionContext(**cached)

    if len(session_names) == 1:
        context = _resolve_single(session_names[0], data_store)
    else:
        per_session = [_resolve_single(name, data_store) for name in session_names]
        context = _merge_contexts(per_session, combined_name)

    memory.cache_resolution(cache_key, context.model_dump())
    return context


def _resolve_single(name: str, data_store: DataStore) -> SessionContext:
    """Resolve one session to a SessionContext using its own content, then apply any curated override."""
    if data_store.get_session_content(name):
        # Reading material is primary: extract only KPs that are directly taught.
        context = _from_llm([name], name, data_store)
    else:
        context = _from_knowledge_graph([name], name, data_store)
        if not context:
            context = _from_llm([name], name, data_store)
    return _apply_override(name, context, _load_outcome_overrides())


def _merge_contexts(contexts: list[SessionContext], combined_name: str) -> SessionContext:
    """Union the per-session contexts into one (dedup, preserve order)."""
    def _dedup(items):
        return list(dict.fromkeys(items))

    outcomes, concepts, interview_topics, scope_in, scope_out = [], [], [], [], []
    kp_by_id: dict[str, KPMatch] = {}
    csv_topics: list[TopicMatch] = []
    types = []
    for c in contexts:
        outcomes += c.learning_outcomes
        concepts += c.key_concepts
        interview_topics += c.interview_topics
        scope_in += c.scope_in
        scope_out += c.scope_out
        csv_topics += c.matched_csv_topics
        types.append(c.session_type)
        for kp in c.matched_kp_ids:
            kp_by_id.setdefault(kp.kp_id, kp)

    if "code_heavy" in types:
        session_type = "code_heavy"
    elif types and all(t == "theory_heavy" for t in types):
        session_type = "theory_heavy"
    else:
        session_type = "mixed"

    matched_kps = list(kp_by_id.values())
    kp_ids = [kp.kp_id for kp in matched_kps]
    # merge each session's prerequisite chain plus the unioned KP set
    chain_union = _dedup([k for c in contexts for k in c.prerequisite_kp_chain] + kp_ids)

    return SessionContext(
        session_name=combined_name,
        learning_outcomes=_dedup(outcomes),
        key_concepts=_dedup(concepts),
        interview_topics=_dedup(interview_topics) or _dedup(concepts),
        scope_in=_dedup(scope_in),
        scope_out=_dedup(scope_out),
        session_type=session_type,
        matched_kp_ids=matched_kps,
        matched_csv_topics=csv_topics,
        prerequisite_kp_chain=chain_union,
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )


def _from_knowledge_graph(
    session_names: list[str],
    combined_name: str,
    data_store: DataStore,
) -> SessionContext | None:
    """Try to build SessionContext from knowledge_graph.json data."""
    all_kp_ids = []
    all_outcomes = []
    all_concepts = []
    session_types = []

    for name in session_names:
        info = data_store.get_session_info(name)
        if not info:
            return None  # Session not in graph — need LLM

        all_kp_ids.extend(info.get("kp_ids", []))
        all_outcomes.extend(info.get("learning_outcomes", []))
        all_concepts.extend(info.get("key_concepts", []))
        session_types.append(info.get("session_type", "mixed"))

    if not all_kp_ids:
        return None  # No KP mapping — need LLM

    # Build KPMatch objects
    matched_kps = []
    for kp_id in list(dict.fromkeys(all_kp_ids)):  # deduplicate preserving order
        label = data_store.kp_catalog.get(kp_id, kp_id)
        source = data_store.kp_source_map.get(kp_id, "unknown")
        matched_kps.append(KPMatch(
            kp_id=kp_id, kp_label=label,
            relevance=0.9, source_file=source,
        ))

    # Determine session type
    if "code_heavy" in session_types:
        session_type = "code_heavy"
    elif all(t == "theory_heavy" for t in session_types):
        session_type = "theory_heavy"
    else:
        session_type = "mixed"

    # Build prerequisite chain
    prereq_chain = data_store.get_kp_ancestors(all_kp_ids)

    # If outcomes are empty, generate basic ones from KP labels
    if not all_outcomes:
        all_outcomes = [f"Understand {kp.kp_label}" for kp in matched_kps[:5]]

    # Build scope_in from KP labels
    scope_in = list(dict.fromkeys([kp.kp_label for kp in matched_kps]))

    key_concepts = all_concepts if all_concepts else [kp.kp_label for kp in matched_kps[:5]]
    return SessionContext(
        session_name=combined_name,
        learning_outcomes=all_outcomes,
        key_concepts=key_concepts,
        # KG path has no LLM to infer transferable topics — mirror the concepts/KP labels.
        interview_topics=list(key_concepts),
        scope_in=scope_in,
        scope_out=[],
        session_type=session_type,
        matched_kp_ids=matched_kps,
        matched_csv_topics=[],
        prerequisite_kp_chain=prereq_chain,
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )


def _safe_topic_matches(raw: list) -> list[TopicMatch]:
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        try:
            out.append(TopicMatch(**t))
        except Exception:
            pass
    return out


def _from_llm(
    session_names: list[str],
    combined_name: str,
    data_store: DataStore,
) -> SessionContext:
    """Fall back to LLM-based session analysis."""
    rm_parts = []
    for name in session_names:
        content = data_store.get_session_content(name)
        if content:
            rm_parts.append(f"### {name}\n{content}")

    rm_content = "\n\n---\n\n".join(rm_parts) if rm_parts else f"Session: {combined_name}"

    kp_list = "\n".join(
        f"- {kp_id}: {kp_label}"
        for kp_id, kp_label in sorted(data_store.kp_catalog.items())
    )

    result = chat_completion_json(
        system_prompt=LLM_SYSTEM_PROMPT,
        # Feed the FULL per-session reading material (RM is pre-capped at 12k/session upstream).
        # The old 8000-char cut truncated the lesson's tail, so deep sub-topics taught later
        # (e.g. VAE/GAN/sampling for image-gen) never reached the outcome/interview_topic extractor.
        user_prompt=f"## Session: {combined_name}\n\n## Reading Material\n{rm_content[:14000]}\n\n## KP Catalog\n{kp_list}",
        max_tokens=3000,
    )

    matched_kps = []
    for kp_data in result.get("matched_kp_ids", []):
        if not isinstance(kp_data, dict):
            continue
        try:
            kp_id = kp_data.get("kp_id", "")
            matched_kps.append(KPMatch(
                kp_id=kp_id,
                kp_label=kp_data.get("kp_label", data_store.kp_catalog.get(kp_id, "")),
                relevance=float(kp_data.get("relevance", 0.5)),
                source_file=data_store.kp_source_map.get(kp_id, "unknown"),
            ))
        except Exception:
            pass

    kp_ids = [kp.kp_id for kp in matched_kps]
    prereq_chain = data_store.get_kp_ancestors(kp_ids)

    key_concepts = result.get("key_concepts", [])
    interview_topics = result.get("interview_topics") or []
    if not isinstance(interview_topics, list):
        interview_topics = []
    # Fallback: if the model omitted interview_topics, mirror the key_concepts (fine for theory sessions).
    interview_topics = [t for t in interview_topics if isinstance(t, str) and t.strip()] or list(key_concepts)

    return SessionContext(
        session_name=combined_name,
        learning_outcomes=result.get("learning_outcomes", []),
        key_concepts=key_concepts,
        interview_topics=interview_topics,
        scope_in=result.get("scope_in", []),
        scope_out=result.get("scope_out", []),
        session_type=result.get("session_type", "mixed"),
        matched_kp_ids=matched_kps,
        matched_csv_topics=_safe_topic_matches(result.get("matched_csv_topics", [])),
        prerequisite_kp_chain=prereq_chain,
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )
