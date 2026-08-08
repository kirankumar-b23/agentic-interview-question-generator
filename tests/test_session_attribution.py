"""Per-session attribution, representation, and the syllabus audit.

All four behaviours here were found by evaluating a real run — `b5d94fee`, topic "Introduction to AI
Agents + Building a Learning Path Generator", 12 questions — against the two sessions' reading
material. That evaluation found:

  * all 12 questions labelled with the FIRST session, though 6 fit the second better;
  * the second session unrepresented in the final set, flagged by the gate and shipped anyway;
  * four questions (all the Hard ones) testing concepts absent from both sessions' material;
  * outcome coverage of 0.846 inflated by proximity — a hallucination question was credited with
    covering "Integrate multiple Google APIs (Docs, Calendar, Drive)".

Each test asserts the observable consequence, using the real session names and the real questions.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail

AGENTS = "Introduction to AI Agents"
LEARNPATH = "Building a Learning Path Generator"

# Verbatim from the run that prompted this work.
RUN_QUESTIONS = [
    "What are the core components of an AI Agent?",
    "How would you implement a fallback mechanism when an AI agent encounters an ambiguous user intent?",
    "Why is memory critical for the performance of AI agents?",
    "How would you implement guardrails to prevent an autonomous agent from executing harmful actions?",
    "How do agents decide when to stop a task?",
    "Describe the working mechanism of AI agents and their interaction with environments or tasks.",
    "An AI agent is deployed into production but is not performing accurately. How would you diagnose it?",
    "What strategies would you use to evaluate and mitigate hallucination in an agentic workflow?",
    "When should AI agents be considered for solving problems?",
    "Describe the Main Architectural Patterns for Building AI Agents",
    "When does agent reasoning go off the rails?",
    "What types of tools can be integrated with AI workflows?",
]


def _q(content, session=None, rel=0.7, fit=0.6, qid=None):
    q = QuestionDetail(category="GEN_AI", content=content, topic="Gen AI", source="interview_db",
                       difficulty="Medium", relevance_score=rel, session_fit=fit)
    if qid:
        q.question_id = qid
    q.session = session
    return q


# ── Fix 1: attribution scores chunks, not one truncated blob ─────────────────

class TestSessionAttributionUsesChunks:
    def test_two_sessions_split_instead_of_collapsing_onto_one(self):
        """The measured defect: a single 4000-char blob per session sent all 12 to the first one.

        `Building a Learning Path Generator`'s material is 8,806 chars, so 55% was discarded and what
        remained diluted every specific match into 0.17–0.41. Chunk-and-max splits the same questions.
        """
        from src.pipeline import _session_profile
        from src.tools import _attribute_sessions

        profiles = {}
        for name in (AGENTS, LEARNPATH):
            curated, rm = _session_profile([name], None)
            profiles[name] = {"curated": curated, "rm": rm}
        qs = [_q(c) for c in RUN_QUESTIONS]
        _attribute_sessions(qs, profiles)

        labels = {q.session for q in qs}
        if len(labels) == 1 and not profiles[AGENTS]["curated"]:
            pytest.skip("session material unavailable in this environment")
        assert labels == {AGENTS, LEARNPATH}, \
            "both sessions must be represented in the labels, not just the first"
        # And specifically: the build-session questions must not all be filed under the theory session.
        assert sum(1 for q in qs if q.session == LEARNPATH) >= 3

    def test_profile_is_chunks_not_a_truncated_blob(self):
        """Guards the shape. A str value here is the old blob and reintroduces the length bias."""
        from src.pipeline import _session_profile

        curated, rm = _session_profile([LEARNPATH], None)
        assert isinstance(curated, list) and isinstance(rm, list)
        assert all(isinstance(t, str) for t in curated + rm)
        # No chunk may be the whole document — that is the dilution the blob caused.
        assert all(len(t) <= 800 for t in rm)

    def test_single_session_still_tags_everything(self):
        from src.tools import _attribute_sessions

        qs = [_q(c) for c in RUN_QUESTIONS[:3]]
        _attribute_sessions(qs, {AGENTS: {"curated": ["agents"], "rm": []}})
        assert all(q.session == AGENTS for q in qs)

    def test_legacy_string_profile_still_works(self):
        """Cached resolutions may still hold the old string shape; it must not crash."""
        from src.tools import _attribute_sessions

        qs = [_q("What are the core components of an AI Agent?")]
        _attribute_sessions(qs, {AGENTS: "agents memory tools", LEARNPATH: "google docs calendar"})
        assert qs[0].session in (AGENTS, LEARNPATH)


# ── Fix 2: representation is enforced, and a bare cupboard is reported ──────

class TestSessionRepresentationIsEnforced:
    def test_a_session_with_candidates_gets_a_slot(self):
        """`_select_final` only NUDGES via a bonus, and the nudge is dead when attribution collapses."""
        from src.tools import _ensure_session_representation

        pool = [_q(f"agents q{i}", session=AGENTS, rel=0.9) for i in range(5)]
        pool.append(_q("learning path q", session=LEARNPATH, rel=0.4))
        selected = [q for q in pool if q.session == AGENTS][:5]      # what MMR would pick on score

        rep = _ensure_session_representation(selected, pool, [AGENTS, LEARNPATH], 5)
        assert rep["swapped"] == 1
        assert any(q.session == LEARNPATH for q in selected)
        assert len(selected) == 5                                    # size preserved, not grown
        assert rep["per_session"][LEARNPATH] == 1

    def test_it_displaces_the_weakest_question_of_an_overrepresented_session(self):
        from src.tools import _ensure_session_representation

        weak = _q("weakest agents q", session=AGENTS, rel=0.31, qid="weak")
        pool = [weak] + [_q(f"agents q{i}", session=AGENTS, rel=0.9) for i in range(3)]
        pool.append(_q("learning path q", session=LEARNPATH, rel=0.5, qid="lp"))
        selected = [q for q in pool if q.session == AGENTS]

        _ensure_session_representation(selected, pool, [AGENTS, LEARNPATH], 4)
        assert "weak" not in {q.question_id for q in selected}, "must drop the weakest, not an arbitrary one"
        assert "lp" in {q.question_id for q in selected}

    def test_a_session_with_no_candidates_is_reported_never_padded(self):
        """The honest case, and the real situation in the run: retrieval found nothing for it."""
        from src.tools import _ensure_session_representation

        pool = [_q(f"agents q{i}", session=AGENTS) for i in range(4)]
        selected = list(pool)

        rep = _ensure_session_representation(selected, pool, [AGENTS, LEARNPATH], 4)
        assert rep["no_candidates"] == [LEARNPATH]
        assert rep["swapped"] == 0
        assert len(selected) == 4                       # nothing invented to fill the gap
        assert all(q.session == AGENTS for q in selected)

    def test_it_never_empties_another_session_to_fill_one(self):
        """With one slot per session there is no donor, so the set is left alone rather than churned."""
        from src.tools import _ensure_session_representation

        a, b = _q("agents q", session=AGENTS), _q("learnpath q", session=LEARNPATH)
        third = "Third Session"
        pool = [a, b, _q("third q", session=third, rel=0.9)]
        selected = [a, b]

        rep = _ensure_session_representation(selected, pool, [AGENTS, LEARNPATH, third], 2)
        assert len(selected) == 2
        assert {q.session for q in selected} == {AGENTS, LEARNPATH}
        assert rep["swapped"] == 0

    def test_single_session_topics_are_untouched(self):
        from src.tools import _ensure_session_representation

        pool = [_q(f"q{i}", session=AGENTS) for i in range(3)]
        selected = list(pool)
        rep = _ensure_session_representation(selected, pool, [AGENTS], 3)
        assert rep == {"per_session": {}, "no_candidates": [], "swapped": 0}
        assert len(selected) == 3


# ── Fix 3: on-domain is not on-syllabus ─────────────────────────────────────

class TestOffSyllabusClaimsAreVerified:
    def test_a_concept_absent_from_the_material_is_absent(self):
        from src.tools import _concept_is_absent, _session_corpus

        corpus = _session_corpus([AGENTS, LEARNPATH])
        if not corpus.strip():
            pytest.skip("session material unavailable in this environment")
        # The four the evaluation found: none of these words occur in either session's material.
        for concept in ("hallucination", "guardrails", "ambiguous user intent", "production diagnosis"):
            assert _concept_is_absent(concept, corpus), concept

    def test_a_taught_concept_is_not_reported_untaught(self):
        """The guard against a fabricated claim: these ARE in the material, so a claim is discarded."""
        from src.tools import _concept_is_absent, _session_corpus

        corpus = _session_corpus([AGENTS, LEARNPATH])
        if not corpus.strip():
            pytest.skip("session material unavailable in this environment")
        for concept in ("memory", "tools", "ReAct", "core components"):
            assert not _concept_is_absent(concept, corpus), concept

    def test_empty_or_stopword_only_concept_is_never_a_claim(self):
        from src.tools import _concept_is_absent

        for concept in ("", "   ", "the with that"):
            assert not _concept_is_absent(concept, "anything")


def _audit_state(questions, outcomes):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=[AGENTS, LEARNPATH]),
                    data_store=get_data_store())
    st.session_context = SimpleNamespace(learning_outcomes=list(outcomes), scope_in=[], scope_out=[],
                                         key_concepts=[], interview_topics=[], matched_kp_ids=[],
                                         session_type="code_heavy")
    st.questions = {q.question_id: q for q in questions}
    return st


class TestSyllabusAudit:
    OUTCOMES = ["Identify the three core components of AI agents",
                "Integrate multiple Google APIs (Docs, Calendar, Drive) with an LLM-powered agent"]

    def test_flags_the_untaught_concept_and_keeps_the_question(self, monkeypatch):
        import src.tools as tools

        qs = [_q("What strategies would you use to evaluate and mitigate hallucination?")]
        st = _audit_state(qs, self.OUTCOMES)
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "questions": [{"n": 1, "untaught": "hallucination", "covers": []}]})

        out = tools._syllabus_audit(st, qs)
        assert len(out["off_syllabus"]) == 1
        assert qs[0].off_syllabus_concept == "hallucination"
        assert qs[0].content.startswith("What strategies")     # flagged, never rejected or edited

    def test_a_claim_about_a_TAUGHT_concept_is_discarded(self, monkeypatch):
        """Verified, not trusted. "memory" is in the material, so the model cannot flag it."""
        import src.tools as tools

        qs = [_q("Why is memory critical for the performance of AI agents?")]
        st = _audit_state(qs, self.OUTCOMES)
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "questions": [{"n": 1, "untaught": "memory", "covers": [1]}]})

        out = tools._syllabus_audit(st, qs)
        assert out["off_syllabus"] == []
        assert qs[0].off_syllabus_concept is None

    def test_coverage_is_judged_not_proximity(self, monkeypatch):
        """The hallucination question must NOT be credited with the Google APIs outcome."""
        import src.tools as tools

        qs = [_q("What strategies would you use to evaluate and mitigate hallucination?")]
        st = _audit_state(qs, self.OUTCOMES)
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "questions": [{"n": 1, "untaught": "hallucination", "covers": []}]})

        cov = tools._syllabus_audit(st, qs)["coverage"]
        assert cov["method"] == "llm-judged"
        assert cov["covered"] == []
        assert len(cov["missing"]) == 2

    def test_out_of_range_outcome_indices_are_dropped(self, monkeypatch):
        import src.tools as tools

        qs = [_q("What are the core components of an AI Agent?")]
        st = _audit_state(qs, self.OUTCOMES)
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "questions": [{"n": 1, "untaught": None, "covers": [1, 99, 0, -3, "x"]}]})

        cov = tools._syllabus_audit(st, qs)["coverage"]
        assert cov["covered"] == [self.OUTCOMES[0]]      # only index 1 (1-based) survives

    def test_llm_failure_leaves_the_set_unflagged_and_falls_back(self, monkeypatch):
        import src.tools as tools

        qs = [_q("What strategies would you use to evaluate and mitigate hallucination?")]
        st = _audit_state(qs, self.OUTCOMES)

        def boom(**kw):
            raise RuntimeError("api down")
        monkeypatch.setattr(tools, "chat_completion_json", boom)

        out = tools._syllabus_audit(st, qs)
        assert out == {"off_syllabus": [], "coverage": {}}
        assert qs[0].off_syllabus_concept is None

    def test_uses_the_runs_model(self, monkeypatch):
        import src.tools as tools

        qs = [_q("What are the core components of an AI Agent?")]
        st = _audit_state(qs, self.OUTCOMES)
        st.config = GenerationConfig(session_names=[AGENTS, LEARNPATH],
                                     model="anthropic/claude-sonnet-4-6")
        seen = {}
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: (seen.update(kw), {"questions": []})[1])

        tools._syllabus_audit(st, qs)
        assert seen.get("model") == "anthropic/claude-sonnet-4-6"


# ── Fix 4: coverage reports WHICH method produced it ────────────────────────

class TestCoverageMethodIsReported:
    OUTCOMES = ["Identify the three core components of AI agents",
                "Integrate multiple Google APIs with an LLM-powered agent"]

    def test_judged_coverage_wins_over_proximity(self):
        from src.pipeline import _outcome_coverage

        st = _audit_state([_q("What are the core components of an AI Agent?")], self.OUTCOMES)
        st.judged_coverage = {"covered": [self.OUTCOMES[0]], "missing": [self.OUTCOMES[1]],
                              "pairs": [], "method": "llm-judged"}
        cov = _outcome_coverage(st); frac, method = cov.topic_coverage, cov.method
        assert method == "llm-judged"
        assert frac == pytest.approx(0.5)

    def test_falls_back_to_proximity_and_says_so(self):
        """The number must never be presented without the method that produced it."""
        from src.pipeline import _outcome_coverage

        st = _audit_state([_q("What are the core components of an AI Agent?")], self.OUTCOMES)
        st.judged_coverage = {}
        cov = _outcome_coverage(st); frac, method = cov.topic_coverage, cov.method
        assert method == "embedding-proximity"
        assert 0.0 <= frac <= 1.0

    def test_the_agent_tool_and_the_report_agree_on_the_judged_number(self):
        """CLAUDE.md requires these two never disagree about coverage."""
        from src.pipeline import _outcome_coverage
        from src.tools import tool_check_outcome_coverage

        st = _audit_state([_q("What are the core components of an AI Agent?")], self.OUTCOMES)
        st.judged_coverage = {"covered": [self.OUTCOMES[0]], "missing": [self.OUTCOMES[1]],
                              "pairs": [], "method": "llm-judged"}
        assert tool_check_outcome_coverage(st)["coverage_pct"] == pytest.approx(
            round(_outcome_coverage(st).topic_coverage, 2), abs=0.01)
