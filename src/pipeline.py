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

EmitFn = Callable[[str, str, str, str], None]   # (run_id, step_id, status, detail)

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


def _current_model() -> str:
    """The model this run is using (for per-run cost estimation)."""
    from src.llm_client import get_active_model
    return get_active_model()


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

    def _evaluate_and_gate(self, state, emit) -> int:
        """Stage 4 + quality-gate loop. Returns revision rounds used."""
        eval_agent = EvaluationAgent()
        revision_round = 0
        while True:
            state.submitted = False
            eval_agent.run(state, emit)

            emit("critique", "running", "Quality gate — critiquing final set...")
            critique = _critique_question_set(state)
            critique_pass = critique.get("pass", True)
            must_fix = critique.get("must_fix", [])
            critique_summary = critique.get("summary", "")

            if critique_pass or revision_round >= MAX_REVISION_ROUNDS:
                state.submitted = True
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

        def emit(step_id: str, status: str, detail: str = ""):
            emit_fn(run_id, step_id, status, detail)

        emit("agent", "running", "Pipeline starting — 4-agent workflow...")
        state.api_usage["model"] = _current_model()
        try:
            self._pick_questions(state, emit)
            revision_round = self._evaluate_and_gate(state, emit)
            result = self._build_result(state, config, run_id, revision_round)
            emit("complete", "done",
                 f"Done! {state.total_questions} questions, {len(state.tool_log)} total tool calls, {revision_round} revision(s)")
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

        def emit(step_id: str, status: str, detail: str = ""):
            emit_fn(run_id, step_id, status, detail)

        emit("agent", "running", "Preview mode — picking questions (quality gate deferred)...")
        state.api_usage["model"] = _current_model()
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

        def emit(step_id: str, status: str, detail: str = ""):
            emit_fn(run_id, step_id, status, detail)

        emit("agent", "running", "Resuming — quality evaluation & gate...")
        state.api_usage.setdefault("model", _current_model())
        try:
            revision_round = self._evaluate_and_gate(state, emit)
            result = self._build_result(state, state.config, run_id, revision_round)
            emit("complete", "done",
                 f"Done! {state.total_questions} questions, {revision_round} revision(s)")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.error = str(exc).split('\n')[0][:200]
            emit("error", "error", f"Pipeline failed: {result.error}")
        return result


def _outcome_coverage(state: AgentState) -> float:
    """Fraction of learning outcomes that at least one final question addresses
    (TF-IDF cosine ≥ 0.10 between the question text and the outcome)."""
    ctx = state.session_context
    outcomes = ctx.learning_outcomes if ctx else []
    questions = [q.content for q in state.questions.values()]
    if not outcomes:
        return 1.0
    if not questions:
        return 0.0
    try:
        # Semantic coverage (embeddings) when available, else TF-IDF.
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
        covered = sum(1 for row in qo if row.max() >= thresh)
        return covered / len(outcomes)
    except Exception:
        return 0.0


def _build_quality_report(state: AgentState, revision_round: int) -> QualityReport:
    total_q = state.total_questions
    diff = state.difficulty_counts
    sources = state.source_counts

    # The two metrics that actually matter for an interview set:
    rels = [q.relevance_score for q in state.questions.values() if q.relevance_score is not None]
    relevance_score = round(sum(rels) / len(rels), 3) if rels else 0.5
    coverage_score = round(_outcome_coverage(state), 3)

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

    # Relevance + coverage dominate; size/diversity/difficulty are hygiene.
    composite = round(0.40 * relevance_score + 0.25 * coverage_score
                      + 0.15 * size_score + 0.10 * diversity_score + 0.10 * diff_score, 3)
    if total_q < MIN_QUESTIONS:
        composite = min(composite, 0.4)

    # Honest pass/fail: a set only "passes" if it is genuinely relevant AND covers the
    # session AND meets the size floor — not merely diverse/balanced.
    passed = (relevance_score >= 0.6 and coverage_score >= 0.6
              and total_q >= MIN_QUESTIONS and composite >= 0.6)

    # Honest, non-fabricated notes so a reviewer sees WHY a set is weak.
    notes: list[str] = []
    if total_q < MIN_QUESTIONS:
        notes.append(f"Only {total_q} question(s) — few real interview questions were available "
                     f"for this session (minimum is {MIN_QUESTIONS}).")
    elif total_q < target:
        notes.append(f"{total_q} question(s) — fewer than the requested {target}; the on-topic real "
                     f"questions available for this session were limited (kept on-topic over padding).")
    if relevance_score < 0.6:
        notes.append("Low mean relevance — the available real questions are only loosely on-topic "
                     "for this session.")
    if coverage_score < 0.6:
        notes.append("Some learning outcomes are not covered by the available questions.")
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

    return QualityReport(
        composite_score=composite,
        metric_scores={
            "relevance": round(relevance_score, 2),
            "outcome_coverage": coverage_score,
            "set_size": round(size_score, 2),
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
