"""AgentPipeline — coordinates four specialized agents in sequence.

Flow:
  UnderstandingAgent → RetrievalAgent → ValidationAgent → EvaluationAgent
                                                               ↓
                                               quality_gate (up to 2 revisions)
                                                               ↓
                                                       PipelineResult
"""

from __future__ import annotations
import math
import re
import uuid
from dataclasses import dataclass
from typing import Callable

from src.agent import AgentState, PipelineResult, _critique_question_set
from src.data_loader import get_data_store
from src.models import GenerationConfig, SessionContext, CurationMetadata, CuratedOutput, QualityReport
from src.config import MIN_QUESTIONS, MAX_QUESTIONS, OUTCOME_CAP
from src.agents import UnderstandingAgent, RetrievalAgent, ValidationAgent, EvaluationAgent

MAX_REVISION_ROUNDS = 2
# Floor on the candidate pool the session-fit gate may leave behind. The LLM relevance pass drops
# more candidates still, so the gate must never hand it a starved pool even if SESSION_FIT_FLOOR is
# mis-tuned for an unusual session. Only the highest-fit rejects are restored.
_MIN_POOL_AFTER_FIT = 30

# (run_id, step_id, status, detail, **structured_fields)
EmitFn = Callable[..., None]

_TOPIC_PROFILE_CACHE = None   # {topic: [profile texts]} — built once from course_structure + session_outcomes


def _load_topic_profiles() -> dict:
    """Per-course-topic semantic profile texts (interview_topics + learning_outcomes of the topic's
    sessions, from the curated session_outcomes.json; falls back to the session name). Cached; {} on error."""
    global _TOPIC_PROFILE_CACHE
    if _TOPIC_PROFILE_CACHE is not None:
        return _TOPIC_PROFILE_CACHE
    import json
    from src.config import DATA_DIR
    prof: dict[str, list[str]] = {}
    try:
        cs = json.loads((DATA_DIR / "course_structure.json").read_text(encoding="utf-8"))
        so_path = DATA_DIR / "reading_materials" / "session_outcomes.json"
        so = json.loads(so_path.read_text(encoding="utf-8")) if so_path.exists() else {}
        for topic, sessions in cs.items():
            texts: list[str] = []
            for s in sessions:
                ov = so.get(s) or {}
                texts += [t for t in (ov.get("interview_topics", []) + ov.get("learning_outcomes", []))
                          if isinstance(t, str) and t.strip()]
                if not ov:
                    texts.append(s)   # fallback: the session name itself
            if texts:
                prof[topic] = texts
    except Exception:  # noqa: BLE001 — pre-gate must never break the pipeline
        prof = {}
    _TOPIC_PROFILE_CACHE = prof
    return prof


def _session_profile(session_names, ctx) -> tuple[list[str], list[str]]:
    """Semantic profile for THIS run's session(s) — the grounding the pre-gate was missing.

    Returns `(curated_texts, reading_material_chunks)` so the caller can weight them differently:

      * curated_texts — per-session `learning_outcomes` + `interview_topics` from
        session_outcomes.json (hand-checked, covers all 53 sessions), plus whatever the
        Understanding agent resolved for this run. This is the session's *intent*.
      * reading_material_chunks — slices of the session's OWN reading material, so wording the
        outcomes never mention (tools, APIs, model names taught in the lesson) still matches. This
        is instructional prose, so it is noisier: a setup walkthrough about copying an auth token
        will match generic auth questions. Hence the caller discounts it.

    Both are LISTS of short texts, not one blob: the caller takes the MAX similarity, so long
    reading material cannot dilute a short, precise outcome statement.
    """
    import json
    from src.config import (DATA_DIR, RM_CHUNK_MAX_CHARS, RM_CHUNK_MIN_CHARS,
                            SESSION_PROFILE_RM_CHUNKS)

    texts: list[str] = []
    rm_texts: list[str] = []

    # 1. Curated per-session outcomes (NOT the pooled course-topic profile).
    try:
        so_path = DATA_DIR / "reading_materials" / "session_outcomes.json"
        so = json.loads(so_path.read_text(encoding="utf-8")) if so_path.exists() else {}
    except Exception:  # noqa: BLE001 — profile building must never break the pipeline
        so = {}
    for name in (session_names or []):
        ov = so.get(name) or {}
        texts += [t for t in (list(ov.get("learning_outcomes", []))
                              + list(ov.get("interview_topics", [])))
                  if isinstance(t, str) and t.strip()]

    # 2. Whatever the Understanding agent resolved for this run.
    if ctx is not None:
        texts += [t for t in (list(getattr(ctx, "interview_topics", None) or [])
                              + list(getattr(ctx, "learning_outcomes", None) or [])
                              + list(getattr(ctx, "key_concepts", None) or []))
                  if isinstance(t, str) and t.strip()]

    # 3. Reading-material chunks — EVERY substantive paragraph of THIS session's content.
    #
    # This used to keep only ~36% of the material, and it discarded exactly the passages that identify
    # what a session teaches. Three lossy steps compounded:
    #
    #   chunks = [c for c in content.split("\n\n") if len(c.strip()) > 120]   # drops short paragraphs
    #   step   = max(1, len(chunks) // 12)                                     # then STRIDE-SAMPLES
    #   rm_texts += [c[:800] for c in chunks[::step][:12]]                     # then truncates
    #
    # Measured on the No-Code sessions: 24,521 chars of material -> 8,728 in the profile. The bullet that
    # defines the node most precisely is 85 characters —
    #     '- **HTTP Request Node**: Allows n8n to talk to almost any web service that has an API.'
    # — so the >120 filter threw it away, the phrase appeared in NONE of the 12 chunks, and
    # "What is the HTTP Request node and when do you use it?" scored 0.275 against a session that
    # literally teaches it. After this change it scores 0.540 (0.703 on the other topic that teaches it),
    # while the topic's median moves 0.615 -> 0.622: the fix reaches the questions naming something
    # specific and leaves the rest alone.
    #
    # This is not a display bug. `_score_session_fit` DROPS candidates below a floor derived from these
    # scores (107-169 per run, the largest cut in the funnel), `session_grounding` is 20% of the
    # composite, `_rank_key` uses it as the tiebreak, and `_attribute_sessions` reuses this profile.
    # Same class as the 4,000-char truncation already documented for `_attribute_sessions`.
    #
    # Bullets are the most information-dense lines in this material, so the floor is low on purpose.
    try:
        from src.data_loader import get_data_store
        store = get_data_store()
        for name in (session_names or []):
            content = store.get_session_content(name)
            if not content:
                continue
            for para in content.split("\n\n"):
                p = para.strip()
                if len(p) < RM_CHUNK_MIN_CHARS:
                    continue                      # true noise only — a bare heading or stray token
                # Sub-split a long paragraph at a word boundary rather than truncating it, so its tail
                # is still searchable instead of silently discarded.
                while len(p) > RM_CHUNK_MAX_CHARS:
                    cut = p.rfind(" ", 0, RM_CHUNK_MAX_CHARS)
                    if cut <= 0:
                        cut = RM_CHUNK_MAX_CHARS
                    rm_texts.append(p[:cut].strip())
                    p = p[cut:].strip()
                if p:
                    rm_texts.append(p)
            if len(rm_texts) >= SESSION_PROFILE_RM_CHUNKS:
                # Runaway guard only. Reaching it means the material is unexpectedly large, not that a
                # sample was wanted — treating this cap as a sampling budget is what caused the bug.
                rm_texts = rm_texts[:SESSION_PROFILE_RM_CHUNKS]
                break
    except Exception:  # noqa: BLE001
        pass

    def _dedup(seq: list[str]) -> list[str]:
        """Preserve order, drop repeats (identical outcomes recur across combined sessions)."""
        seen, out = set(), []
        for t in seq:
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(t.strip())
        return out

    return _dedup(texts), _dedup(rm_texts)


def _topic_profiles(session_names) -> tuple[set, list]:
    """(current topics for this run, profile texts of all OTHER topics). Empty → caller skips the gate."""
    from src.data_loader import get_topic_for_session
    prof = _load_topic_profiles()
    if not prof:
        return set(), []
    cur = {t for t in (get_topic_for_session(s) for s in (session_names or [])) if t and t in prof}
    if not cur:
        return set(), []
    other = [t for topic, texts in prof.items() if topic not in cur for t in texts]
    return cur, other


def _current_model(state) -> str:
    """The model THIS run is using, for per-run cost estimation.

    Read from the run's own config, not from a process-wide global: two concurrent runs would
    otherwise both report whichever model started last, and price their tokens at its rate.
    """
    from src.llm_client import run_model
    return run_model(state)


def _unrepresented_terms(terms, questions) -> list:
    """Which of this session's tool names NO surviving question mentions.

    Word-boundary matched, so "n8n" does not match inside another token and "RSS" does not match
    "across". Cheap on purpose — a substring pass over ≤300 candidates, no embeddings — because it runs
    on every run whether or not the tier fires.
    """
    if not terms:
        return []
    blob = " ".join((q.content or "") for q in questions)
    missing = []
    for term in terms:
        t = (term or "").strip()
        if not t:
            continue
        if not re.search(r"\b" + re.escape(t) + r"\b", blob, re.IGNORECASE):
            missing.append(t)
    return missing


def _open_web_shortfall(surviving: int, requested: int, missing_tools=None):
    """Why the open web is warranted, or None to skip it. See `config.OPEN_WEB_TRIGGER_RATIO`.

    Two independent triggers, because a count and a coverage gap are different failures:

    * **materially short** of the requested count — not merely under `MIN_QUESTIONS`. The old guard was
      `surviving >= MIN_QUESTIONS`, and run 8fb9fcb3 asked for 15, survived with EXACTLY 5 and skipped
      the tier entirely. The `<= MIN_QUESTIONS` clause is kept SEPARATE from the ratio: with
      `requested == MIN_QUESTIONS` the ratio floors to 5 and `surviving < 5` would reintroduce the very
      off-by-one this exists to fix.
    * **a tool this session teaches that nothing asks about** — fires at ANY count. A full set of 15 on
      an n8n session that never says "n8n" is still the wrong set, and no count test can see it.
    """
    from src.config import MIN_QUESTIONS, OPEN_WEB_TRIGGER_RATIO

    threshold = max(MIN_QUESTIONS, math.ceil(OPEN_WEB_TRIGGER_RATIO * max(0, requested)))
    if surviving < threshold or surviving <= MIN_QUESTIONS:
        return (f"Only {surviving} question(s) from trusted sources, against {requested} requested "
                f"(top-up threshold {threshold})")
    if missing_tools:
        return (f"{surviving} question(s), but nothing asks about "
                f"{', '.join(missing_tools[:3])} — which this session teaches")
    return None


class AgentPipeline:

    # ── Reusable stages ───────────────────────────────────────────────────
    def _pick_questions(self, state, emit):
        """Stages 1–3: Understanding → Retrieval → Validation (the 'picked' set)."""
        UnderstandingAgent().run(state, emit)
        RetrievalAgent().run(state, emit)
        # Suppress previously-rejected questions (per session) BEFORE validation/selection — so a
        # rejected question never resurfaces on re-generation and doesn't take a slot. Matched by
        # normalized content (question_ids regenerate each run).
        self._drop_rejected(state, emit)
        # Drop questions that cannot be ANSWERED in a conversational interview ("Write a Python program
        # to…"). Here, before session-fit embeddings and long before the LLM relevance judge, so no spend
        # goes on candidates that can never ship. Any shortfall this creates reaches
        # `_top_up_from_open_web` at the end of this method, which already backfills below 60% of the
        # requested count — so the filter needs no wiring of its own.
        self._drop_hands_on(state, emit)
        # SESSION-grounded scoring first: score every candidate against THIS session's own outcomes +
        # reading material, drop the unrelated tail, and rank the pool best-first. This is the check
        # that was missing — the cross-topic gate below never filtered WITHIN a course topic.
        self._score_session_fit(state, emit)
        # Cheap SEMANTIC pre-gate: drop clearly cross-topic candidates (embedding-distant from the session
        # profile) BEFORE the expensive LLM relevance scoring — cuts noise + LLM cost. Skips if embeddings
        # unavailable.
        self._prefilter_semantic(state, emit)
        ValidationAgent().run(state, emit)
        # Guarantee relevance filtering ran — the validation agent is prompt-advised, not code-forced,
        # so if it never scored anything (no question carries a relevance_score) we run it directly.
        # Without this, an agent that skips validate_relevance lets the whole raw pool through unfiltered.
        # Run if ANY question is still unscored (not all-or-nothing) — a partial agent scoring must not
        # let the remaining unscored candidates leak through unfiltered. tool_validate_relevance scores
        # the whole set, so a full re-score here is correct and idempotent.
        if state.questions and any(q.relevance_score is None for q in state.questions.values()):
            from src.tools import tool_validate_relevance
            emit("validate_relevance", "running", "Enforcing relevance validation (unscored candidates present)...")
            tool_validate_relevance(state)
            emit("validate_relevance", "done", f"{len(state.questions)} question(s) after relevance filter.")
        # Always dedup (semantic) — the agent is only prompt-advised to, and with retrieval uncapped the
        # pool is large and redundant. Idempotent, so running it again is harmless.
        if len(state.questions) > 1:
            from src.tools import tool_deduplicate_questions
            before = len(state.questions)
            tool_deduplicate_questions(state)
            emit("deduplicate_questions", "done",
                 f"Deduplicated: {before} → {len(state.questions)} distinct question(s).")
        # LAST RESORT, and it has to be HERE. The first attempt put this inside
        # `tool_search_web_questions`, where `state.questions` is the raw CANDIDATE POOL (~270) — so the
        # "are we short?" test was never true and the tier was dead code that never once ran. Only after
        # relevance filtering and dedup is the shortfall real.
        self._top_up_from_open_web(state, emit)

    def _top_up_from_open_web(self, state, emit):
        """Search the open web when the trusted sources left this session short OR off its own tools.

        The banks and the 67-domain allowlist hold nothing on n8n nodes, Gemini configuration or
        Automatic1111 — which is what several sessions actually teach. Measured: an unrestricted search
        for "n8n workflow automation interview questions" returns 24 usable candidates where the
        allowlisted search returns 0, including "What is the Merge node and what merge modes does it
        support?", which matches a session outcome verbatim.

        Everything it adds is flagged `unvetted_source` and carries no company attribution, and it still
        faces the relevance judge and the session-fit gate — both re-run below, because a candidate that
        skipped scoring would bypass the only stages that read the question against the session.
        """
        from src.config import OPEN_WEB_ENABLED

        if not OPEN_WEB_ENABLED or not state.session_context:
            return
        from src.sources.tavily_search import fetch_open_web
        from src.tools import _qualify_tool_terms, _tool_terms, add_open_web_records

        surviving = len(state.questions) + len(state.coding_questions)
        requested = getattr(state.config, "max_questions", None) or 12
        tool_terms = _tool_terms(state.session_context)
        # Tools this session teaches that NO surviving question mentions. Run 8fb9fcb3's topic named
        # n8n, RSS Feed Read Node, Schedule Trigger and Gmail Send Node across 10 of 22 outcomes and
        # shipped 5 questions, none of which mentioned any of them. Count alone cannot see that.
        missing_tools = _unrepresented_terms(tool_terms, state.questions.values())
        reason = _open_web_shortfall(surviving, requested, missing_tools)
        if not reason:
            return

        # Ask for what is actually MISSING first; fall back to the session's tools, then its topics.
        terms = missing_tools or tool_terms or (
            list(getattr(state.session_context, "interview_topics", None) or [])[:2])
        if not terms:
            return
        # Qualify with the platform before querying the OPEN web, where a bare "Merge" or "RSS" means
        # something else entirely. See `_qualify_tool_terms` for the measured noise this removes.
        terms = _qualify_tool_terms(terms)
        emit("open_web", "running",
             f"{reason} — searching the open web for {', '.join(terms[:3])}…")
        records, calls, err = fetch_open_web(terms)
        state.api_usage["tavily_calls"] += calls
        state.open_web_used = True
        before_ids = set(state.questions)
        added = add_open_web_records(state, records)
        state.open_web_added = added
        emit("open_web", "done",
             f"Added {added} unvetted question(s) from {len({r.source_type for r in records})} "
             f"domain(s)" + (f" — {err}" if err else ""))
        if not added:
            return
        # New candidates must be judged like every other one. Session-fit FIRST and scoped to just the
        # new ids: without it they keep `session_fit = None`, which drops them out of the
        # `session_grounding` mean (it averages non-None only) and sinks them in Review's fit ranking
        # (`_rank_key` reads None as 0.0) — a metric that silently measured only the vetted subset.
        new_ids = set(state.questions) - before_ids
        self._score_session_fit(state, emit, only_ids=new_ids)
        from src.tools import tool_deduplicate_questions, tool_validate_relevance
        tool_validate_relevance(state)
        if len(state.questions) > 1:
            tool_deduplicate_questions(state)
        emit("open_web", "done",
             f"{len(state.questions)} question(s) after judging the open-web additions.")

    def _drop_hands_on(self, state, emit):
        """Remove questions that cannot be ANSWERED out loud — see `interview_format`.

        The mock interview is conversational, so "Write a Python program to generate the Fibonacci
        series" is not a hard question, it is an impossible one. Two real runs shipped such prompts
        ("Implement an input box to interact with the Gemini API…", "Build and integrate LLM
        applications.") and each wasted a slot in a set of 9.

        Policy, not a data defect: gated on `config.CONVERSATIONAL_ONLY` and applied to the POOL, so the
        corpus keeps the 217 real coding questions it holds. "Design a news aggregator system" is
        deliberately NOT caught — it is answerable by talking through the architecture.
        """
        from src.config import CONVERSATIONAL_ONLY
        if not CONVERSATIONAL_ONLY or not state.questions:
            return
        from src.interview_format import is_hands_on_task

        drop = [qid for qid, q in state.questions.items() if is_hands_on_task(q.content)]
        for qid in drop:
            q = state.questions.pop(qid, None)
            if q is None:
                continue
            state.excluded.add(qid)
            state.removed.append({
                "content": q.content,
                "reason": "Requires writing code or building something — not answerable in a "
                          "conversational interview",
                "stage": "hands_on", "difficulty": q.difficulty, "company": q.attribution,
            })
        if drop:
            emit("hands_on", "done",
                 f"Dropped {len(drop)} hands-on task prompt(s) that cannot be answered out loud; "
                 f"{len(state.questions)} remain.",
                 dropped=len(drop), kept=len(state.questions))

    def _drop_rejected(self, state, emit):
        """Remove questions whose normalized content was previously rejected for this session."""
        ctx = state.session_context
        if not ctx or not state.questions:
            return
        from src import memory
        rejected = memory.get_rejected_norms(ctx.session_name)
        if not rejected:
            return
        drop = [qid for qid, q in state.questions.items()
                if memory.normalize_content(q.content) in rejected]
        for qid in drop:
            q = state.questions.pop(qid, None)
            if q is not None:
                state.excluded.add(qid)
                state.removed.append({
                    "content": q.content, "reason": "Previously rejected for this session",
                    "stage": "suppressed", "difficulty": q.difficulty, "company": q.attribution,
                })
        if drop:
            emit("suppress_rejected", "done", f"Suppressed {len(drop)} previously-rejected question(s).")

    def _score_session_fit(self, state, emit, only_ids=None):
        """SESSION-grounded scoring + gate.

        The cross-topic pre-gate below only removes candidates belonging to a DIFFERENT course
        topic — inside a topic it removes nothing, so an image-generation session happily kept any
        GenAI question. This stage scores every candidate against THIS session's own profile
        (curated learning outcomes + interview topics + its reading material), stores the score on
        the question as `session_fit`, drops the clearly-unrelated tail, and re-orders the pool
        best-first so the batched LLM relevance pass sees the strongest candidates first.

        `only_ids` scores and gates JUST those candidates, for the open-web top-up. It is deliberately
        not a plain re-run: the floor is relative to the pool's best fit, so re-running over a pool that
        just grew by up to 60 open-web candidates could move the bar and drop VETTED questions whose
        place was already decided correctly. With `only_ids` the floor is still computed across the whole
        pool — the session's real bar, not the new batch's own — but only the named ids can be dropped.

        Fail-open: no-op when embeddings are unavailable or no profile can be built.
        """
        ctx = state.session_context
        if not ctx or not state.questions:
            return
        from src import embeddings
        from src.config import (SESSION_FIT_FLOOR, SESSION_FIT_RELATIVE,
                                SESSION_PROFILE_RM_WEIGHT)

        curated, rm_chunks = _session_profile(state.config.session_names, ctx)
        if not curated and not rm_chunks:
            return

        items = [(qid, q) for qid, q in state.questions.items()
                 if only_ids is None or qid in only_ids]
        if not items:
            return
        contents = [q.content for _, q in items]
        cur_sim = embeddings.cosine_matrix(contents, curated) if curated else None
        rm_sim = embeddings.cosine_matrix(contents, rm_chunks) if rm_chunks else None
        if cur_sim is None and rm_sim is None:   # embeddings unavailable → leave the pool untouched
            return

        for i, (qid, q) in enumerate(items):
            # Curated intent counts at full weight; reading-material prose is discounted, so an
            # RM-only match must be distinctly stronger to keep a candidate.
            best_curated = float(max(cur_sim[i])) if cur_sim is not None else 0.0
            best_rm = float(max(rm_sim[i])) * SESSION_PROFILE_RM_WEIGHT if rm_sim is not None else 0.0
            fit = max(best_curated, best_rm)
            q.session_fit = round(fit, 4)

        # Rank across EVERY scored candidate, not just the ones scored on this call, so a subset run
        # measures the new batch against the session's real bar.
        scored = [(q.session_fit, qid, q) for qid, q in state.questions.items()
                  if q.session_fit is not None]
        scored.sort(key=lambda t: t[0], reverse=True)

        # Floor is the stricter of an absolute bar and a fraction of THIS session's best fit, so the
        # gate filters comparably whether the banks cover the session well or barely at all.
        best_fit = scored[0][0] if scored else 0.0
        floor = max(SESSION_FIT_FLOOR, SESSION_FIT_RELATIVE * best_fit)
        droppable = {qid for _, qid, _ in scored} if only_ids is None else set(only_ids)
        keep = [t for t in scored if t[0] >= floor or t[1] not in droppable]
        # Guard against a mis-tuned floor starving the pool: the LLM relevance pass drops more
        # candidates still, so always leave it a workable pool. Restores the highest-fit rejects only.
        floor_applied = True
        if len(keep) < _MIN_POOL_AFTER_FIT:
            kept_ids = {t[1] for t in keep}
            spare = [t for t in scored if t[1] not in kept_ids]
            keep = keep + spare[:_MIN_POOL_AFTER_FIT - len(keep)]
            floor_applied = False
        keep.sort(key=lambda t: t[0], reverse=True)
        keep_ids = {qid for _, qid, _ in keep}

        for fit, qid, q in scored:
            if qid in keep_ids:
                continue
            state.questions.pop(qid, None)
            state.removed.append({
                "content": q.content,
                "reason": f"Not grounded in this session's outcomes/reading material (fit {fit:.2f})",
                "stage": "session_fit", "difficulty": q.difficulty, "company": q.attribution,
            })

        # Re-insert best-first so relevance batching and any downstream truncation favour good fits.
        # Anything the scoring could not reach (no `session_fit`) is carried over rather than dropped —
        # clearing the dict and refilling it from `scored` alone would silently delete it.
        carried = [(qid, q) for qid, q in state.questions.items() if qid not in keep_ids]
        state.questions.clear()
        for _, qid, q in keep:
            state.questions[qid] = q
        for qid, q in carried:
            state.questions[qid] = q

        # Report the rule that was ACTUALLY applied — saying "below fit X" when the pool guard
        # overrode X described a filter that never ran.
        basis = (f"below fit {floor:.2f}" if floor_applied
                 else f"outside the top {_MIN_POOL_AFTER_FIT} (fit floor {floor:.2f} would have left "
                      f"too few to score)")
        scope = "" if only_ids is None else f" (open-web batch of {len(items)} only)"
        emit("session_fit", "done",
             f"Session-fit scored {len(items)} candidate(s){scope} against this session's outcomes + "
             f"reading material; dropped {len(scored) - len(keep)} {basis}, "
             f"{len(state.questions)} remain (best {best_fit:.2f}).",
             kept=len(keep), dropped=len(scored) - len(keep), floor=round(floor, 3),
             best_fit=round(best_fit, 3))

    def _prefilter_semantic(self, state, emit):
        """COMPARATIVE topic pre-gate: drop a candidate that belongs to a DIFFERENT course topic than this
        run's (its nearest other-topic profile beats this run's by a margin), or is totally unrelated.
        No-op if embeddings unavailable, the run's topic can't be resolved, or no other-topic profiles."""
        ctx = state.session_context
        if not ctx or len(state.questions) <= 1:
            return
        from src import embeddings
        from src.config import SEMANTIC_TOPIC_MARGIN, SEMANTIC_PREFILTER_FLOOR, SEMANTIC_CUR_KEEP

        cur_profile = [p for p in (list(getattr(ctx, "interview_topics", None) or [])
                                   + list(ctx.key_concepts or []) + list(ctx.learning_outcomes or []))
                       if p and p.strip()]
        cur_topics, other_texts = _topic_profiles(state.config.session_names)
        if not cur_profile or not cur_topics or not other_texts:
            return  # can't resolve topics reliably → let the LLM judge filter

        items = list(state.questions.items())
        contents = [q.content for _, q in items]
        cur_sim = embeddings.cosine_matrix(contents, cur_profile)
        oth_sim = embeddings.cosine_matrix(contents, other_texts)
        if cur_sim is None or oth_sim is None:            # embeddings unavailable → skip
            return

        # A question that NAMES a tool this session teaches is on-syllabus by construction, and this
        # comparative gate is structurally wrong for it. Run 8fb9fcb3 retrieved exactly one real n8n
        # question — "What kind of workflows have you built with n8n before…" — and this stage dropped
        # it, because pooled across the whole GenAI course an n8n question resembles "the course" less
        # than a prompt-engineering question does. The exemption is narrow: `_tool_terms` returns
        # concrete product names and `[]` for a theory-only session, `_score_session_fit` already ran
        # BEFORE this stage (so the "interviewing at RSS Security" false positives stay dropped and
        # cannot be resurrected here), and the LLM relevance judge and syllabus audit still follow.
        from src.tools import _tool_terms
        tool_res = [re.compile(r"\b" + re.escape(t.strip()) + r"\b", re.IGNORECASE)
                    for t in (_tool_terms(ctx) or []) if t and t.strip()]

        drop = []
        exempt = 0
        for i, (qid, q) in enumerate(items):
            cur = max(cur_sim[i])
            oth = max(oth_sim[i])
            # Never drop a candidate that STRONGLY matches this session's topic — guards against the
            # max-over-many-other-topics inflation bias (oth is a max over far more texts than cur).
            if cur >= SEMANTIC_CUR_KEEP:
                continue
            if tool_res and any(rx.search(q.content or "") for rx in tool_res):
                exempt += 1
                continue
            # Otherwise: drop if it clearly belongs to another topic (beats this one by a margin),
            # or is unrelated to everything.
            if oth > cur + SEMANTIC_TOPIC_MARGIN or max(cur, oth) < SEMANTIC_PREFILTER_FLOOR:
                drop.append(qid)
        for qid in drop:
            q = state.questions.pop(qid, None)
            if q is not None:
                state.removed.append({
                    "content": q.content, "reason": "Belongs to a different topic (semantic pre-filter)",
                    "stage": "off_topic_prefilter", "difficulty": q.difficulty, "company": q.attribution,
                })
        if drop or exempt:
            kept_note = f" ({exempt} kept for naming a tool this session teaches)" if exempt else ""
            emit("prefilter", "done",
                 f"Pre-filtered {len(drop)} off-topic candidate(s){kept_note}; "
                 f"{len(state.questions)} remain for scoring.")

    def _enforce_submission(self, state, emit) -> bool:
        """Guarantee the final set has actually been SELECTED. Returns True if we had to force it.

        Ranking, the coverage/difficulty/session/attribution/role bonuses and the trim to the
        requested count all live inside `tool_submit_question_set`, and the Evaluation agent is only
        prompt-advised to call it. When it doesn't — a text-only reply, an API error that ends the
        phase, or a budget spent on `check_*`/`remove_question` calls — nothing selects anything and
        `state.questions` is still the raw candidate pool. Serializing that ships up to ~270 unranked,
        untrimmed questions to the reviewer while the run reports success. So call it ourselves.
        """
        if state.submitted:
            return False
        from src.tools import tool_submit_question_set
        before = len(state.questions)
        tool_submit_question_set(state)
        emit("submit_question_set", "warning",
             f"Evaluation agent never submitted — selected {len(state.questions)} of {before} "
             f"candidate(s) directly so the set is ranked and trimmed.")
        return True

    def _score_unscored_fits(self, state, emit):
        """Give every shipped question a `session_fit`, dropping nothing.

        Retained questions come from the topic's accumulated set, not from this run's pool, so they have
        no fit. Left unscored they would repeat the exact defect the open-web tier hit: `grounding_score`
        averages only non-None fits, so `session_grounding` would silently describe just the freshly-found
        subset, and `_rank_key` reads None as 0.0, sinking settled questions to the bottom of Review.

        This deliberately does NOT reuse `_score_session_fit(only_ids=…)`: that applies the relative floor
        and DROPS what falls below it. A retained question is already settled — it gets flagged
        (`stale_reason`), never removed, so the reviewer decides.
        """
        unscored = [q for q in state.questions.values() if q.session_fit is None]
        if not unscored or not state.session_context:
            return
        from src import embeddings
        from src.config import SESSION_PROFILE_RM_WEIGHT

        curated, rm_chunks = _session_profile(state.config.session_names, state.session_context)
        if not curated and not rm_chunks:
            return
        contents = [q.content for q in unscored]
        cur_sim = embeddings.cosine_matrix(contents, curated) if curated else None
        rm_sim = embeddings.cosine_matrix(contents, rm_chunks) if rm_chunks else None
        if cur_sim is None and rm_sim is None:
            return
        for i, q in enumerate(unscored):
            best_curated = float(max(cur_sim[i])) if cur_sim is not None else 0.0
            best_rm = float(max(rm_sim[i])) * SESSION_PROFILE_RM_WEIGHT if rm_sim is not None else 0.0
            q.session_fit = round(max(best_curated, best_rm), 4)
        emit("session_fit", "done",
             f"Scored {len(unscored)} carried-over question(s) so grounding covers the whole set.",
             kept=len(unscored), dropped=0)

    def _evaluate_and_gate(self, state, emit) -> int:
        """Stage 4 + quality-gate loop. Returns revision rounds used."""
        eval_agent = EvaluationAgent()
        revision_round = 0
        while True:
            state.submitted = False
            eval_agent.run(state, emit)
            # Do this BEFORE the critique so the gate judges the set the reviewer will actually see.
            if self._enforce_submission(state, emit):
                state.submit_forced = True
            # Retained questions arrive from the topic set with no `session_fit`. Score them here, and
            # drop NOTHING — see `_score_unscored_fits`.
            self._score_unscored_fits(state, emit)

            emit("critique", "running", "Quality gate — critiquing final set...")
            critique = _critique_question_set(state)
            # An unparseable/failed critique must NOT be read as approval — `.get("pass", True)`
            # turned every LLM hiccup into a silent pass. `_critique_question_set` returns
            # `pass=False` with an explicit note when it could not judge.
            critique_pass = bool(critique.get("pass", False))
            must_fix = critique.get("must_fix", []) or []
            critique_summary = critique.get("summary", "")

            if critique_pass or revision_round >= MAX_REVISION_ROUNDS:
                forced = not critique_pass
                state.gate_forced = forced
                state.gate_issues = list(must_fix) if forced else []
                state.gate_summary = critique_summary
                label = "Passed" if critique_pass else f"Force-passed after {revision_round} revision(s)"
                emit("critique", "done", f"{label}: {critique_summary}")
                break
            revision_round += 1
            state.revision_notes = must_fix
            emit("critique", "retry",
                 f"Quality gate failed — {len(must_fix)} issue(s), revision {revision_round}/{MAX_REVISION_ROUNDS}")
        return revision_round

    def _build_result(self, state, config, run_id, revision_round) -> PipelineResult:
        result = PipelineResult()
        result.run_id = run_id
        result.curated_output = state.to_curated_output()
        result.quality_report = _build_quality_report(state, revision_round)
        result.quality_report.api_usage = dict(state.api_usage)
        result.context = state.session_context or _fallback_context(config, state)
        result.removed = list(state.removed)
        result.category = getattr(config, "category", "GEN_AI")
        return result

    # ── Full run (default path — behavior unchanged) ──────────────────────
    def run(self, config: GenerationConfig, run_id: str, emit_fn: EmitFn) -> PipelineResult:
        result = PipelineResult()
        result.run_id = run_id
        state = AgentState(config=config, data_store=get_data_store())

        def emit(step_id: str, status: str, detail: str = "", **fields):
            # **fields carries structured data (agent, duration_ms, tokens, counts) to the UI so it
            # renders numbers instead of regex-scraping them out of `detail`.
            emit_fn(run_id, step_id, status, detail, **fields)

        emit("agent", "running", "Pipeline starting — 4-agent workflow...")
        state.api_usage["model"] = _current_model(state)
        try:
            self._pick_questions(state, emit)
            revision_round = self._evaluate_and_gate(state, emit)
            result = self._build_result(state, config, run_id, revision_round)
            # Structured totals so the client doesn't have to parse this sentence to learn them.
            emit("complete", "done",
                 f"Done! {state.total_questions} questions, {len(state.tool_log)} total tool calls, "
                 f"{revision_round} revision(s)",
                 questions=state.total_questions, tool_calls=len(state.tool_log),
                 revisions=revision_round, usage=dict(state.api_usage),
                 score=result.quality_report.composite_score if result.quality_report else None,
                 verdict=result.quality_report.pass_fail if result.quality_report else None)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.error = str(exc).split('\n')[0][:200]
            emit("error", "error", f"Pipeline failed: {result.error}")
        return result

    # ── TESTING: preview mode — pause after Validation, resume into the gate ──
    def run_preview(self, config: GenerationConfig, run_id: str, emit_fn: EmitFn):
        """Stages 1–3 only; returns (partial_result, state) with awaiting_gate=True."""
        result = PipelineResult()
        result.run_id = run_id
        state = AgentState(config=config, data_store=get_data_store())

        def emit(step_id: str, status: str, detail: str = "", **fields):
            # **fields carries structured data (agent, duration_ms, tokens, counts) to the UI so it
            # renders numbers instead of regex-scraping them out of `detail`.
            emit_fn(run_id, step_id, status, detail, **fields)

        emit("agent", "running", "Preview mode — picking questions (quality gate deferred)...")
        state.api_usage["model"] = _current_model(state)
        try:
            self._pick_questions(state, emit)
            result.curated_output = state.to_curated_output()
            result.context = state.session_context or _fallback_context(config, state)
            result.removed = list(state.removed)
            result.category = getattr(config, "category", "GEN_AI")
            result.awaiting_gate = True
            emit("complete", "done",
                 f"Preview ready — {state.total_questions} picked question(s) to verify.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.error = str(exc).split('\n')[0][:200]
            emit("error", "error", f"Pipeline failed: {result.error}")
        return result, state

    def finalize(self, state: AgentState, run_id: str, emit_fn: EmitFn) -> PipelineResult:
        """Resume from a preview: run Evaluation + quality gate on the retained state."""
        result = PipelineResult()
        result.run_id = run_id

        def emit(step_id: str, status: str, detail: str = "", **fields):
            # **fields carries structured data (agent, duration_ms, tokens, counts) to the UI so it
            # renders numbers instead of regex-scraping them out of `detail`.
            emit_fn(run_id, step_id, status, detail, **fields)

        emit("agent", "running", "Resuming — quality evaluation & gate...")
        state.api_usage.setdefault("model", _current_model(state))
        try:
            revision_round = self._evaluate_and_gate(state, emit)
            result = self._build_result(state, state.config, run_id, revision_round)
            emit("complete", "done",
                 f"Done! {state.total_questions} questions, {revision_round} revision(s)",
                 questions=state.total_questions, revisions=revision_round,
                 usage=dict(state.api_usage),
                 score=result.quality_report.composite_score if result.quality_report else None,
                 verdict=result.quality_report.pass_fail if result.quality_report else None)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.error = str(exc).split('\n')[0][:200]
            emit("error", "error", f"Pipeline failed: {result.error}")
        return result


def coverage_targets(ctx) -> list[str]:
    """WHAT a question set is expected to cover: the session's interview topics.

    Not its learning outcomes. Outcomes describe the LESSON, setup steps included, and a large
    fraction of them cannot be examined in an interview at all. Measured on the Image-Generation
    topic, 8 of 18 outcomes (44%) were environment mechanics — "Set up a Kaggle account with phone
    verification", "Manage Kaggle session duration and GPU quota", "Use Ngrok to create secure
    tunnels", "Install and configure Automatic1111 WebUI". No real interview question can cover those,
    so the maximum coverage an interview set could reach was 0.56, already under the 0.60 pass bar,
    with unlimited questions. Every set failed and the sets were not the problem.

    `interview_topics` is the field built from the same reading material for exactly this purpose —
    its extraction prompt requires "the transferable GenAI skills/concepts, NOT the specific
    tool/product/UI". For the very same Kaggle session it yields "GPU resource allocation and VRAM
    requirements", "Cost-performance tradeoffs in cloud computing", "API authentication and secure
    token management". It was already feeding retrieval, the session-fit profile and the relevance
    judge; coverage was the one place that ignored it.

    Falls back to learning outcomes when a session has no interview topics, so a session that predates
    the curated field still gets measured rather than silently scoring 1.0.
    """
    if ctx is None:
        return []
    topics = [t for t in (getattr(ctx, "interview_topics", None) or []) if t and t.strip()]
    return topics or [o for o in (getattr(ctx, "learning_outcomes", None) or []) if o and o.strip()]


def _outcome_coverage_detail(state: AgentState) -> tuple[list[str], list[str]]:
    """(covered, missing) coverage targets for the current set — see `coverage_targets`.

    Semantic similarity (embeddings) when available, else TF-IDF. Shared with
    `tools.tool_check_outcome_coverage` so the agent and the report cannot disagree about coverage.
    """
    ctx = state.session_context
    outcomes = coverage_targets(ctx)
    questions = [q.content for q in state.questions.values()]
    questions += [q.content for q in state.coding_questions.values()]
    if not outcomes or not questions:
        return [], outcomes
    try:
        from src import embeddings
        from src.config import EMBED_COVERAGE_THRESHOLD
        qo = embeddings.cosine_matrix(outcomes, questions)   # [n_outcomes x n_questions]
        thresh = EMBED_COVERAGE_THRESHOLD
        if qo is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            mat = vec.fit_transform(questions + outcomes)
            qo = cosine_similarity(mat[len(questions):], mat[:len(questions)])
            thresh = 0.10
        covered, missing = [], []
        for i, outcome in enumerate(outcomes):
            (covered if qo[i].max() >= thresh else missing).append(outcome)
        return covered, missing
    except Exception:  # noqa: BLE001 — scoring must never break a completed run
        return [], outcomes


@dataclass
class CoverageResult:
    """Two coverage numbers, because one cannot answer both questions honestly.

    * `topic_coverage` — covered / ALL the session's interview topics. The truthful statement about how
      much of the session the set examines. REPORTED, never gated: it is bounded by supply, so on a
      topic with 22 interview topics and only 5 real questions available it can never exceed 0.23 no
      matter how good the questions are. Gating on it failed every set, including one that scored
      0.227 — its exact arithmetic maximum.
    * `coverage_efficiency` — covered / min(n_topics, n_questions). "Did each question earn its place
      against a DISTINCT topic?" SCORED and gated, because it is achievable at any set size while still
      punishing a set whose questions pile onto the same topic (5 questions hitting 3 topics = 0.60).

    `method` says which measure produced the covered set, so a proximity-derived number is never
    mistaken for a judged one.
    """
    topic_coverage: float
    coverage_efficiency: float
    covered: int
    total: int
    n_questions: int
    method: str

    @property
    def supply_capped(self) -> bool:
        """True when there are fewer questions than topics, so `topic_coverage` cannot reach 1.0."""
        return self.n_questions < self.total

    # NOTE: there is deliberately no `ceiling` property. A first version computed
    # `n_questions / total` as "the best coverage this many questions could reach", and a live run
    # disproved it — 6 questions examined 11 of 22 topics, because one question can legitimately
    # examine several. The note that quoted that ceiling was stating a falsehood to the reviewer.


def _outcome_coverage(state: AgentState) -> CoverageResult:
    """Coverage of the session's INTERVIEW TOPICS by the current set.

    Prefers `state.judged_coverage` — the syllabus audit's read of which topics each question actually
    EXAMINES — over embedding proximity, which was demonstrably crediting things it should not: a
    hallucination question was credited with "Integrate multiple Google APIs (Docs, Calendar, Drive)"
    at 0.38. A higher threshold cannot separate those (legitimate credits measured 0.52–0.83, false
    ones 0.38–0.69 — overlapping), which is why the judged measure exists.
    """
    ctx = state.session_context
    targets = coverage_targets(ctx)
    n_q = len(state.questions) + len(state.coding_questions)
    if not targets:
        # Nothing to cover is vacuously covered — but only if there IS a set. A run that lost its
        # session context and produced nothing used to report perfect coverage on zero questions.
        frac = 1.0 if n_q else 0.0
        return CoverageResult(frac, frac, 0, 0, n_q, "no-targets")

    judged = getattr(state, "judged_coverage", None) or {}
    if judged.get("method") and (judged.get("covered") or judged.get("missing")):
        n_cov = len(judged.get("covered") or [])
        total = n_cov + len(judged.get("missing") or [])
        method = judged["method"]
    else:
        covered, missing = _outcome_coverage_detail(state)
        n_cov, total, method = len(covered), len(covered) + len(missing), "embedding-proximity"
    total = total or len(targets)

    achievable = max(1, min(total, n_q)) if n_q else 1
    return CoverageResult(
        topic_coverage=round(n_cov / total, 3) if total else 0.0,
        coverage_efficiency=round(min(1.0, n_cov / achievable), 3),
        covered=n_cov, total=total, n_questions=n_q, method=method,
    )


def _human_agreement(state: AgentState):
    """Agreement between this question set and the reviewer's own past decisions, or None.

    Uses ALL recorded labels: the reviewer looking at this run wants the best available estimate.
    (The eval harness deliberately holds labels out instead — see eval/run_eval.py — because there
    the point is to measure the system rather than to inform the person reviewing it.)
    """
    try:
        from src import memory
        from src.human_agreement import predict_accept
        contents = [q.content for q in state.questions.values()]
        session = state.session_context.session_name if state.session_context else None
        return predict_accept(contents, memory.get_feedback_examples(), session=session)
    except Exception:  # noqa: BLE001 — reporting must never break a completed run
        return None


def _build_quality_report(state: AgentState, revision_round: int) -> QualityReport:
    total_q = state.total_questions
    diff = state.difficulty_counts
    sources = state.source_counts

    # SELF-reported relevance: the mean of the LLM score that SELECTED these questions. Reported for
    # transparency but deliberately NOT part of the composite or the pass decision — a selector
    # cannot be its own judge, and treating it as quality is why runs scored 0.9 while reviewers
    # rejected most of the set (corr(composite, approved) was 0.16 over 36 runs).
    rels = [q.relevance_score for q in state.questions.values() if q.relevance_score is not None]
    self_relevance = round(sum(rels) / len(rels), 3) if rels else None
    cov = _outcome_coverage(state)
    coverage_method = cov.method
    # SCORED: efficiency. REPORTED: honest topic coverage. See `CoverageResult` for why one number
    # cannot do both jobs without failing every supply-capped set.
    coverage_score = cov.coverage_efficiency
    topic_coverage = cov.topic_coverage
    # Mean grounding in THIS session's outcomes + reading material (set by the session-fit gate).
    # Independent of any LLM judgement — it compares questions to the curriculum, not to itself.
    fits = [q.session_fit for q in state.questions.values() if q.session_fit is not None]
    grounding_score = round(sum(fits) / len(fits), 3) if fits else None
    # Agreement with the reviewer's OWN past accept/reject decisions — the only signal here that does
    # not originate from the selector. None when there aren't enough labels or embeddings are off.
    agreement = _human_agreement(state)

    # Secondary hygiene metrics.
    # size_score rewards reaching the REQUESTED target (was: flat 1.0 for any count in [MIN,MAX], which
    # gave the gate zero incentive to exceed MIN — the "always 5" symptom). A full set scores 1.0; a
    # smaller set gets proportional credit. This never pushes toward off-topic padding: the relevance
    # floor upstream caps supply, and the gate/eval agent are told not to pad below the floor.
    # Target = the SAME requested-count ceiling submit uses (min(max_questions or 12, FINAL_SET_CAP)),
    # so the scored size-target matches the delivered count.
    from src.config import FINAL_SET_CAP
    target = max(MIN_QUESTIONS, min(getattr(state.config, "max_questions", None) or 12, FINAL_SET_CAP))
    size_score = 1.0 if total_q >= target else round(max(0.0, total_q / target), 3)
    diversity_score = min(1.0, len(sources) / 2)
    diff_target = {"Easy": 0.3, "Medium": 0.5, "Hard": 0.2}
    diff_score = (
        1.0 - sum(abs(diff.get(k, 0) / max(total_q, 1) - v) for k, v in diff_target.items()) / 2
        if total_q > 0 else 0
    )

    # Composite is built ONLY from signals independent of the selector:
    #   coverage    — questions vs the session's learning outcomes (embedding cosine)
    #   grounding   — questions vs the session's outcomes + reading material (session_fit)
    #   agreement   — questions vs the reviewer's own past accept/reject decisions
    #   size        — did we reach the requested count
    # `self_relevance` and `difficulty_balance` are reported but excluded: the first is the
    # selector grading itself; the second scores against difficulty labels that are ~95% "Medium"
    # in the GenAI bank, so it measures label noise rather than question difficulty.
    # Weights are renormalised over whatever is actually available, so a missing signal neither
    # inflates nor deflates the score.
    parts = [(0.35, coverage_score), (0.20, grounding_score), (0.15, size_score)]
    if agreement is not None:
        parts.append((0.30, agreement.predicted_accept_rate))
    live = [(w, v) for w, v in parts if v is not None]
    total_w = sum(w for w, _ in live)
    composite = round(sum(w * v for w, v in live) / total_w, 3) if total_w else 0.0
    if total_q < MIN_QUESTIONS:
        composite = min(composite, 0.4)

    # Honest pass/fail: every question earns its place against a distinct interview topic, the set
    # meets the size floor, and the composite clears the bar.
    #
    # `predicted_accept` deliberately has NO veto here, though it keeps its 30% composite weight. It is
    # a 1-NN estimate over ~15 reviewer labels that `human_agreement` itself documents as an optimistic
    # upper bound at 65% accuracy; it read 0.2 on two consecutive runs purely because those topics had
    # few matching labels. A signal that weak must not outvote coverage, grounding and size combined.
    GATE_BAR = 0.6
    # The LLM critique is listed alongside the scored conditions because it is ALSO a gate: a
    # force-passed set is failed further down regardless of the numbers. Omitting it showed three green
    # checks beside a FAIL verdict on a real run — worse than no breakdown at all.
    _critique_ok = not getattr(state, "gate_forced", False)
    _n_issues = len(getattr(state, "gate_issues", None) or [])
    gate_checks = [
        {"name": "coverage efficiency", "value": coverage_score, "bar": GATE_BAR,
         "ok": coverage_score >= GATE_BAR},
        {"name": "question count", "value": total_q, "bar": MIN_QUESTIONS,
         "ok": total_q >= MIN_QUESTIONS},
        {"name": "composite", "value": composite, "bar": GATE_BAR, "ok": composite >= GATE_BAR},
        {"name": "reviewer critique", "value": ("no objections" if _critique_ok
                                                else f"{_n_issues} unresolved"),
         "bar": "no objections", "ok": _critique_ok},
    ]
    passed = all(c["ok"] for c in gate_checks)

    # Honest, non-fabricated notes so a reviewer sees WHY a set is weak.
    notes: list[str] = []
    if total_q < MIN_QUESTIONS:
        notes.append(f"Only {total_q} question(s) — few real interview questions were available "
                     f"for this session (minimum is {MIN_QUESTIONS}).")
    elif total_q < target:
        notes.append(f"{total_q} question(s) — fewer than the requested {target}; the on-topic real "
                     f"questions available for this session were limited (kept on-topic over padding).")
    # Name the failing conditions with their values and bars. The report used to carry `pass_fail` and
    # nothing else, so a failed set gave a reviewer no way to tell a thin corpus from a bad set.
    _failed = [c for c in gate_checks if not c["ok"]]
    if _failed:
        notes.append("Gate not passed — "
                     + "; ".join(f"{c['name']} {c['value']} (needs {c['bar']})" for c in _failed) + ".")
    if coverage_score < 0.6:
        notes.append(f"{cov.covered} of {cov.total} interview topics examined, and the questions do not "
                     f"spread across distinct topics (efficiency {coverage_score}) — several are "
                     f"testing the same thing.")
    # A supply cap is a corpus fact, not a quality problem, and must not read like one.
    if cov.supply_capped and cov.total:
        notes.append(
            f"Topic coverage {topic_coverage} ({cov.covered} of {cov.total} interview topics) is "
            f"limited by supply, not quality — only {cov.n_questions} on-topic real question(s) were "
            f"available. The gate scores coverage EFFICIENCY instead ({coverage_score}: distinct topics "
            f"per question), so a thin corpus is reported here rather than counted against the set."
        )
    # Which method produced the coverage number, and — when judged — the outcome↔question pairing, so
    # a spurious credit is visible instead of silently scored at 35% of the composite.
    _judged = getattr(state, "judged_coverage", None) or {}
    if coverage_method == "embedding-proximity":
        notes.append("Outcome coverage measured by embedding proximity (the syllabus audit did not "
                     "run) — it credits an outcome for a NEARBY question, not necessarily one that "
                     "tests it. Treat it as an upper bound.")
    elif _judged.get("pairs"):
        notes.append("Outcome coverage judged against the reading material; each covered outcome is "
                     "listed with the question credited for it.")
    # The last-resort tier reached outside the vetted allowlist — say so, with the count.
    _unvetted = sum(1 for q in state.questions.values() if getattr(q, "unvetted_source", False))
    if _unvetted:
        notes.append(
            f"{_unvetted} of {total_q} question(s) came from OUTSIDE the trusted source list — the "
            f"open-web tier ran because the bank and the allowlisted sites could not fill this session. "
            f"They carry no company attribution and are tagged 'unvetted source' in review; check them "
            f"more closely than the rest."
        )
    elif getattr(state, "open_web_used", False):
        notes.append("The open-web tier ran but contributed nothing that passed the filters.")

    # On-domain is not on-syllabus. The gate judges domain; this judges the session's own material.
    if state.off_syllabus:
        _c = "; ".join(f"“{o['concept']}”" for o in state.off_syllabus[:4])
        notes.append(
            f"{len(state.off_syllabus)} question(s) test a concept that appears nowhere in this "
            f"session's reading material ({_c}). They are on-domain but beyond the syllabus — keep "
            f"them only if you want candidates stretched past what was taught."
        )
    # Per-session representation, and the honest version of a session with nothing to offer.
    _rep = getattr(state, "session_representation", None) or {}
    if _rep.get("no_candidates"):
        notes.append("No questions could be found at all for: "
                     + ", ".join(_rep["no_candidates"])
                     + " — a source-coverage gap, not a selection choice.")
    elif _rep.get("per_session") and len(_rep["per_session"]) > 1:
        notes.append("Per-session split: "
                     + ", ".join(f"{k} = {v}" for k, v in _rep["per_session"].items()))
    # An edit to a sourced question must never be silent — these still carry a company's name.
    if state.scope_trims:
        notes.append(
            f"{len(state.scope_trims)} question(s) were trimmed to this session's scope (an "
            f"off-syllabus clause was removed); they are marked 'adapted' in review and keep their "
            f"original text for comparison."
        )
    # Name the retained/new split. With the topic's accumulated set carried in, a re-run that finds
    # NOTHING new would otherwise read as a healthy 40-question run. Reported, never gated — gating on
    # new-questions-found would fail every mature topic that is simply finished, the same reason
    # `topic_coverage` is reported and not gated.
    _retained = [q for q in state.questions.values() if getattr(q, "retained", False)]
    if _retained:
        _fresh = len(state.questions) - len(_retained)
        _stale = [q for q in _retained if getattr(q, "stale_reason", None)]
        _unrev = [q for q in _retained if getattr(q, "retained_status", None) != "approved"]
        notes.append(
            f"{len(state.questions)} question(s): {len(_retained)} carried over from this topic's "
            f"existing set, {_fresh} newly found this run."
            + (f" {len(_unrev)} of the carried-over ones were never reviewer-approved (imported from run "
               f"history)." if _unrev else "")
            + (f" {len(_stale)} would be rejected by today's gates and are flagged for you rather than "
               f"removed." if _stale else ""))
        if _fresh == 0:
            notes.append("No NEW questions were found for this topic — the set is unchanged. That is a "
                         "supply result, not a quality failure.")
    # Per-outcome balance. Both halves are reported because they are different problems with the same
    # cause: a topic that supplies several questions is repetition the candidate hears, and a topic that
    # supplies none is a gap no count of questions reveals. On No-Code AI Automation three topics held 47%
    # of a 38-question set while 9 of 22 had nothing.
    _capped = sum(1 for r in (state.removed or []) if r.get("stage") == "outcome_cap")
    if _capped:
        notes.append(
            f"{_capped} question(s) were dropped because their interview topic was already covered "
            f"(cap {OUTCOME_CAP} per topic). Questions matching no topic are kept, not capped.")
    if state.uncovered_outcomes:
        _u = "; ".join(f"“{o}”" for o in state.uncovered_outcomes[:6])
        notes.append(
            f"{len(state.uncovered_outcomes)} interview topic(s) have NO question in this set: {_u}"
            + ("; …" if len(state.uncovered_outcomes) > 6 else "")
            + " — reported, not gated; retrieval for these is a separate decision.")
    # Say what the conversational filter cost. A pool filter that silently shrinks the supply reads as
    # "this session has few questions", which is the misdiagnosis the yield harness exists to prevent.
    _hands_on = sum(1 for r in (state.removed or []) if r.get("stage") == "hands_on")
    if _hands_on:
        notes.append(
            f"{_hands_on} hands-on task prompt(s) were skipped as unanswerable in a conversational "
            f"interview (they asked the candidate to write or build something). Set "
            f"CONVERSATIONAL_ONLY=0 to include them.")
    # A duplicate the set could not drop must be said out loud. Run 8fb9fcb3 shipped one silently: the
    # critique named it, the set held exactly MIN_QUESTIONS, and `remove_question` had to refuse.
    _dupes = [q for q in state.questions.values() if getattr(q, "duplicate_of", None)]
    if _dupes:
        notes.append(
            f"{len(_dupes)} question(s) test the same thing as another question in this set and could "
            f"NOT be dropped without falling under the {MIN_QUESTIONS}-question minimum — they are "
            f"flagged for the reviewer to choose between.")
    if grounding_score is not None and grounding_score < 0.30:
        notes.append(f"Weak grounding (mean session fit {grounding_score:.2f}) — these questions are "
                     f"only loosely tied to this session's outcomes and reading material.")
    if agreement is not None:
        notes.append(f"Predicted reviewer acceptance {agreement.predicted_accept_rate:.0%}, estimated "
                     f"from {agreement.label_count} past accept/reject decision(s)"
                     + (f"; {agreement.repeats_rejected} question(s) closely repeat something already "
                        f"rejected." if agreement.repeats_rejected else "."))
    else:
        notes.append("Predicted reviewer acceptance unavailable — not enough past review decisions "
                     "(needs both accepted and rejected examples) to estimate it.")
    # Web-search health: if web retrieval was unavailable/failed, this set is BANK-ONLY — say so loudly.
    _web_note = {
        "quota": "⚠ Web search hit its usage limit — this set is bank-only (no fresh web questions).",
        "auth": "⚠ Web search unauthorized (bad/expired Tavily key) — this set is bank-only.",
        "rate": "⚠ Web search was rate-limited — this set is bank-only.",
        "no_key": "⚠ No Tavily key configured — web search skipped; this set is bank-only.",
        "error": "⚠ Web search failed — this set is bank-only.",
    }.get(getattr(state, "web_status", "not_run"))
    if _web_note:
        notes.insert(0, _web_note)

    # Quality-gate outcome. These used to exist only in the SSE log, so a force-passed set could
    # display a clean "Pass" — the reviewer had no way to know the gate had objected.
    gate_issues = list(getattr(state, "gate_issues", None) or [])
    if getattr(state, "gate_forced", False):
        notes.insert(0, f"⚠ Quality gate did NOT pass this set — shipped after "
                        f"{revision_round} revision attempt(s) with {len(gate_issues)} unresolved "
                        f"issue(s). {getattr(state, 'gate_summary', '') or ''}".strip())
    if getattr(state, "submit_forced", False):
        notes.insert(0, "⚠ The evaluation agent never submitted a final set — the pipeline selected "
                        "and trimmed it directly. Ranking is applied, but the agent's own "
                        "coverage/difficulty checks did not run.")
    if not getattr(state, "relevance_scored", True):
        notes.insert(0, "⚠ The relevance judge failed for every batch — NOTHING in this set was "
                        "actually scored for topical fit. Treat it as unvalidated.")
    # A phase that died on an API error used to leave no trace in the report, so a retrieval outage
    # was indistinguishable from "this session genuinely has few questions".
    for err in (getattr(state, "phase_errors", None) or []):
        notes.insert(0, f"⚠ A pipeline stage failed and was skipped — {err}. This set is incomplete.")

    # A gate that objected, an unscored set, or a set the agent never submitted cannot be a "pass",
    # however good the independent metrics look.
    if (getattr(state, "gate_forced", False) or not getattr(state, "relevance_scored", True)
            or getattr(state, "phase_errors", None)):
        passed = False

    from src.models import FlaggedQuestion
    return QualityReport(
        composite_score=composite,
        flagged_questions=[FlaggedQuestion.from_gate(i) for i in gate_issues],
        metric_scores={
            # Scored into the composite (all independent of the selector):
            "coverage_efficiency": coverage_score,
            "session_grounding": grounding_score if grounding_score is not None else 0.0,
            "predicted_accept": (agreement.predicted_accept_rate if agreement else 0.0),
            "set_size": round(size_score, 2),
            # Reported for transparency, NOT scored — see the composite comment above.
            # `topic_coverage` is honest but supply-bounded: with fewer questions than topics it cannot
            # reach 1.0, so scoring it failed every thin set (one scored its exact maximum, 0.227).
            "topic_coverage": topic_coverage,
            "self_relevance": round(self_relevance, 2) if self_relevance is not None else 0.0,
            "source_diversity": round(diversity_score, 2),
            "difficulty_balance": round(diff_score, 2),
        },
        gate_checks=gate_checks,
        pass_fail="pass" if passed else "fail",
        critique=notes,
        loops_used=revision_round,
        web_status=getattr(state, "web_status", "not_run"),
        web_error=getattr(state, "web_error", None),
    )


def _fallback_context(config: GenerationConfig, state: AgentState) -> SessionContext:
    return SessionContext(
        session_name=config.session_name,
        learning_outcomes=state.learning_outcomes,
        key_concepts=[], scope_in=[], scope_out=[],
        session_type="mixed", matched_kp_ids=[],
        matched_csv_topics=[], prerequisite_kp_chain=[],
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )
