"""AgentPipeline — coordinates four specialized agents in sequence.

Flow:
  UnderstandingAgent → RetrievalAgent → ValidationAgent → EvaluationAgent
                                                               ↓
                                               quality_gate (up to 2 revisions)
                                                               ↓
                                                       PipelineResult
"""

from __future__ import annotations
import uuid
from typing import Callable

from src.agent import AgentState, PipelineResult, _critique_question_set
from src.data_loader import get_data_store
from src.models import GenerationConfig, SessionContext, CurationMetadata, CuratedOutput, QualityReport
from src.config import MIN_QUESTIONS, MAX_QUESTIONS
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
    from src.config import DATA_DIR, SESSION_PROFILE_RM_CHUNKS

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

    # 3. Reading-material chunks — paragraph-ish slices of THIS session's content only.
    try:
        from src.data_loader import get_data_store
        store = get_data_store()
        for name in (session_names or []):
            content = store.get_session_content(name)
            if not content:
                continue
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 120]
            step = max(1, len(chunks) // SESSION_PROFILE_RM_CHUNKS) if chunks else 1
            rm_texts += [c[:800] for c in chunks[::step][:SESSION_PROFILE_RM_CHUNKS]]
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

    def _score_session_fit(self, state, emit):
        """SESSION-grounded scoring + gate.

        The cross-topic pre-gate below only removes candidates belonging to a DIFFERENT course
        topic — inside a topic it removes nothing, so an image-generation session happily kept any
        GenAI question. This stage scores every candidate against THIS session's own profile
        (curated learning outcomes + interview topics + its reading material), stores the score on
        the question as `session_fit`, drops the clearly-unrelated tail, and re-orders the pool
        best-first so the batched LLM relevance pass sees the strongest candidates first.

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

        items = list(state.questions.items())
        contents = [q.content for _, q in items]
        cur_sim = embeddings.cosine_matrix(contents, curated) if curated else None
        rm_sim = embeddings.cosine_matrix(contents, rm_chunks) if rm_chunks else None
        if cur_sim is None and rm_sim is None:   # embeddings unavailable → leave the pool untouched
            return

        scored = []
        for i, (qid, q) in enumerate(items):
            # Curated intent counts at full weight; reading-material prose is discounted, so an
            # RM-only match must be distinctly stronger to keep a candidate.
            best_curated = float(max(cur_sim[i])) if cur_sim is not None else 0.0
            best_rm = float(max(rm_sim[i])) * SESSION_PROFILE_RM_WEIGHT if rm_sim is not None else 0.0
            fit = max(best_curated, best_rm)
            q.session_fit = round(fit, 4)
            scored.append((fit, qid, q))
        scored.sort(key=lambda t: t[0], reverse=True)

        # Floor is the stricter of an absolute bar and a fraction of THIS session's best fit, so the
        # gate filters comparably whether the banks cover the session well or barely at all.
        best_fit = scored[0][0] if scored else 0.0
        floor = max(SESSION_FIT_FLOOR, SESSION_FIT_RELATIVE * best_fit)
        keep = [t for t in scored if t[0] >= floor]
        # Guard against a mis-tuned floor starving the pool: the LLM relevance pass drops more
        # candidates still, so always leave it a workable pool. Restores the highest-fit rejects only.
        floor_applied = True
        if len(keep) < _MIN_POOL_AFTER_FIT:
            keep = scored[:_MIN_POOL_AFTER_FIT]
            floor_applied = False
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
        state.questions.clear()
        for _, qid, q in keep:
            state.questions[qid] = q

        # Report the rule that was ACTUALLY applied — saying "below fit X" when the pool guard
        # overrode X described a filter that never ran.
        basis = (f"below fit {floor:.2f}" if floor_applied
                 else f"outside the top {_MIN_POOL_AFTER_FIT} (fit floor {floor:.2f} would have left "
                      f"too few to score)")
        emit("session_fit", "done",
             f"Session-fit scored {len(scored)} candidate(s) against this session's outcomes + "
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

        drop = []
        for i, (qid, _) in enumerate(items):
            cur = max(cur_sim[i])
            oth = max(oth_sim[i])
            # Never drop a candidate that STRONGLY matches this session's topic — guards against the
            # max-over-many-other-topics inflation bias (oth is a max over far more texts than cur).
            if cur >= SEMANTIC_CUR_KEEP:
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
        if drop:
            emit("prefilter", "done",
                 f"Pre-filtered {len(drop)} off-topic candidate(s); {len(state.questions)} remain for scoring.")

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


def _outcome_coverage_detail(state: AgentState) -> tuple[list[str], list[str]]:
    """(covered, missing) learning outcomes for the current set.

    Semantic similarity (embeddings) when available, else TF-IDF. Shared with
    `tools.tool_check_outcome_coverage` so the agent and the report cannot disagree about coverage.
    """
    ctx = state.session_context
    outcomes = list(ctx.learning_outcomes) if ctx else []
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


def _outcome_coverage(state: AgentState) -> tuple[float, str]:
    """(fraction of outcomes genuinely examined, method used).

    Prefers `state.judged_coverage` — the syllabus audit's read of which outcomes each question
    actually EXAMINES — over the embedding proximity measure, which was demonstrably crediting
    outcomes it should not: a hallucination question was credited with covering "Integrate multiple
    Google APIs (Docs, Calendar, Drive)" at 0.38, and "when does agent reasoning go off the rails?"
    with "Design system prompts that orchestrate agent reasoning" at 0.69. A higher threshold cannot
    separate those: legitimate credits measured 0.52–0.83 and false ones 0.38–0.69, overlapping, and
    shared-distinctive-term counts overlap too (0–2 vs 0–1). Since this is 35% of the composite, the
    method that produced it is reported alongside the number.
    """
    ctx = state.session_context
    outcomes = ctx.learning_outcomes if ctx else []
    if not outcomes:
        # Nothing to cover is vacuously covered — but only if there IS a set. A run that lost its
        # session context and produced nothing used to report perfect coverage on zero questions.
        return (1.0 if (state.questions or state.coding_questions) else 0.0), "no-outcomes"
    judged = getattr(state, "judged_coverage", None) or {}
    if judged.get("method") and (judged.get("covered") or judged.get("missing")):
        n_cov = len(judged.get("covered") or [])
        total = n_cov + len(judged.get("missing") or [])
        if total:
            return n_cov / total, judged["method"]
    covered, _ = _outcome_coverage_detail(state)
    return len(covered) / len(outcomes), "embedding-proximity"


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
    _cov_raw, coverage_method = _outcome_coverage(state)
    coverage_score = round(_cov_raw, 3)
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

    # Honest pass/fail: covers the session, is grounded in it, meets the size floor, and — when we
    # have enough reviewer labels to judge — would mostly survive review.
    passed = (coverage_score >= 0.6 and total_q >= MIN_QUESTIONS and composite >= 0.6
              and (agreement is None or agreement.predicted_accept_rate >= 0.6))

    # Honest, non-fabricated notes so a reviewer sees WHY a set is weak.
    notes: list[str] = []
    if total_q < MIN_QUESTIONS:
        notes.append(f"Only {total_q} question(s) — few real interview questions were available "
                     f"for this session (minimum is {MIN_QUESTIONS}).")
    elif total_q < target:
        notes.append(f"{total_q} question(s) — fewer than the requested {target}; the on-topic real "
                     f"questions available for this session were limited (kept on-topic over padding).")
    if coverage_score < 0.6:
        notes.append("Some learning outcomes are not covered by the available questions.")
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
            "outcome_coverage": coverage_score,
            "session_grounding": grounding_score if grounding_score is not None else 0.0,
            "predicted_accept": (agreement.predicted_accept_rate if agreement else 0.0),
            "set_size": round(size_score, 2),
            # Reported for transparency, NOT scored — see the composite comment above.
            "self_relevance": round(self_relevance, 2) if self_relevance is not None else 0.0,
            "source_diversity": round(diversity_score, 2),
            "difficulty_balance": round(diff_score, 2),
        },
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
