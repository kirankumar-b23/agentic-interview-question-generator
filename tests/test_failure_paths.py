"""Regression tests for the failure paths that used to ship wrong output while reporting success.

Each class here corresponds to a defect where the happy path worked and the failure path was silent.
That is the whole point: these are not edge cases nobody hits, they are what happens on a 429, a
truncated response, or an agent that answers with prose instead of a tool call.

No LLM or network is used — the model boundary is stubbed.
"""
import pytest

from src.agent import AgentState, _critique_question_set, _deterministic_gate_issues
from src.data_loader import get_data_store
from src.models import GenerationConfig, QuestionDetail, SessionContext
from src.pipeline import AgentPipeline, _build_quality_report


def _context(name="Introduction to AI Agents", **kw) -> SessionContext:
    base = dict(
        session_name=name,
        learning_outcomes=["Understand the core components of an AI agent"],
        key_concepts=["agents"], interview_topics=["AI agent architecture"],
        scope_in=["agents"], scope_out=["fine-tuning"], session_type="mixed",
        matched_kp_ids=[], matched_csv_topics=[], prerequisite_kp_chain=[],
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )
    base.update(kw)
    return SessionContext(**base)


def _state(n_questions=10, *, max_questions=5, min_questions=5, sessions=None) -> AgentState:
    cfg = GenerationConfig(session_names=sessions or ["Introduction to AI Agents"],
                           max_questions=max_questions, min_questions=min_questions)
    st = AgentState(config=cfg, data_store=get_data_store())
    st.session_context = _context()
    for i in range(n_questions):
        st.questions[f"q{i}"] = QuestionDetail(
            question_id=f"q{i}", category="GEN_AI", topic="Gen AI", source="interview_db",
            content=f"What are the core components of an AI agent, part {i}?",
            difficulty="Medium", relevance_score=0.8, session_fit=0.6,
        )
    return st


class TestSelectionIsGuaranteed:
    """A1 — selection, ranking and the trim live inside `tool_submit_question_set`, and the agent is
    only prompt-advised to call it. When it doesn't, the raw candidate pool used to be serialized as
    the final set: up to ~270 unranked questions, reported as a successful run."""

    @pytest.fixture(autouse=True)
    def _no_llm(self, monkeypatch):
        """`_enforce_submission` calls `tool_submit_question_set`, which makes four OpenRouter calls
        (`_scope_trim`, `_syllabus_audit`, `_same_thing_pass`, `_cap_by_outcome`). These tests are about
        the trim, not the LLM, and every one of those paths is fail-open — so the calls were being
        attempted, swallowed, and spending credit invisibly. Caught by the conftest network ledger."""
        import src.tools as tools_mod
        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        """Submit reads the topic's ACCUMULATED set (`tools._add_retained`), so without this the test
        depends on live `memory.db` contents: it was silently pulling in 32 real questions, and the
        assertion below was really measuring production data. Any edit to the shipped sets then moved
        the numbers here for reasons unrelated to the trim.
        """
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "failure_paths.db")
        memory.init_db()

    def test_pool_is_trimmed_when_the_agent_never_submits(self, monkeypatch):
        st = _state(n_questions=40, max_questions=5)
        # Simulate an Evaluation agent that does nothing at all (API error, or a text-only reply).
        monkeypatch.setattr(AgentPipeline, "_evaluate_and_gate",
                            lambda self, state, emit: 0, raising=True)
        pipeline = AgentPipeline()
        forced = pipeline._enforce_submission(st, lambda *a, **k: None)
        assert forced is True
        # Count only what THIS run selected: submit also carries the topic's accumulated set in
        # (`tools._add_retained`), so the shipped total legitimately exceeds the requested count. The
        # invariant here is that the raw 40-question pool was trimmed, not that the set is small.
        fresh = [q for q in st.questions.values() if not q.retained]
        assert len(fresh) == 5, "the pool must be trimmed to the requested count"

    def test_no_op_when_the_agent_did_submit(self):
        st = _state(n_questions=40, max_questions=5)
        st.submitted = True
        pipeline = AgentPipeline()
        assert pipeline._enforce_submission(st, lambda *a, **k: None) is False
        assert len(st.questions) == 40, "an already-submitted set must not be re-selected"

    def test_forcing_is_recorded_in_the_report(self):
        st = _state(n_questions=6, max_questions=6)
        st.submit_forced = True
        report = _build_quality_report(st, 0)
        assert any("never submitted" in n for n in report.critique)

    def test_emits_a_warning_the_user_can_see(self):
        st = _state(n_questions=40, max_questions=5)
        seen = []
        AgentPipeline()._enforce_submission(st, lambda step, status, detail="", **k: seen.append((step, status)))
        assert ("submit_question_set", "warning") in seen


class TestUnscoredSetFailsLoudly:
    """A3 — unscored candidates default to the keep-threshold, which is right for a partial batch
    failure and wrong for a total one: the filter removes nothing and the run looks clean."""

    def test_total_scoring_failure_flags_the_report(self):
        st = _state()
        st.relevance_scored = False
        report = _build_quality_report(st, 0)
        assert report.pass_fail == "fail"
        assert any("relevance judge failed" in n and "scored" in n for n in report.critique)

    def test_total_scoring_failure_is_a_gate_issue(self):
        st = _state()
        st.relevance_scored = False
        issues = _deterministic_gate_issues(st)
        assert any(i["issue"] == "unscored" for i in issues)

    def test_a_scored_set_is_not_flagged(self):
        assert not any(i["issue"] == "unscored" for i in _deterministic_gate_issues(_state()))


class TestGateFailsClosed:
    """A4 — the gate was read as `.get("pass", True)` against a `{}` returned on any LLM failure, so
    every hiccup became a silent approval."""

    def test_unparseable_critique_is_a_failure(self, monkeypatch):
        from src.llm_client import JSONResponseError
        import src.agent as agent_mod

        def _boom(**_kw):
            raise JSONResponseError("truncated")
        monkeypatch.setattr(agent_mod, "chat_completion_json", _boom)

        verdict = _critique_question_set(_state())
        assert verdict["pass"] is False
        assert any(i["issue"] == "gate-error" for i in verdict["must_fix"])

    def test_an_empty_set_never_passes(self):
        st = _state(n_questions=0)
        assert _critique_question_set(st)["pass"] is False

    def test_force_pass_is_recorded_in_the_report(self):
        st = _state()
        st.gate_forced = True
        st.gate_issues = [{"id": "q1", "issue": "duplicate", "suggestion": "drop it"}]
        st.gate_summary = "one duplicate remains"
        report = _build_quality_report(st, 2)
        assert report.pass_fail == "fail", "a force-passed set must not display as a pass"
        assert any("did NOT pass" in n for n in report.critique)
        assert [f.issue for f in report.flagged_questions] == ["duplicate"]


class TestDeterministicGateChecks:
    """B — checks nothing else in the pipeline makes."""

    def test_oversized_set_is_caught(self):
        st = _state(n_questions=40, max_questions=8)
        assert any(i["issue"] == "too-many" for i in _deterministic_gate_issues(st))

    def test_undersized_set_uses_the_per_run_minimum(self):
        """The gate interpolated the GLOBAL minimum, ignoring the run's own."""
        st = _state(n_questions=6, min_questions=10, max_questions=12)
        issues = _deterministic_gate_issues(st)
        too_few = [i for i in issues if i["issue"] == "too-few"]
        assert too_few and "10" in too_few[0]["suggestion"]

    def test_too_few_does_not_advise_removing_more(self):
        st = _state(n_questions=2, min_questions=5, max_questions=12)
        too_few = [i for i in _deterministic_gate_issues(st) if i["issue"] == "too-few"]
        assert "do NOT remove" in too_few[0]["suggestion"]

    def test_malformed_question_is_caught(self):
        st = _state(n_questions=1, max_questions=8)
        st.questions["bad"] = QuestionDetail(
            question_id="bad", category="GEN_AI", topic="Gen AI", source="web",
            content="What are its key components?",     # possessive-reference fragment
        )
        assert any(i["issue"] == "malformed" for i in _deterministic_gate_issues(st))

    def test_unrepresented_session_is_caught(self):
        st = _state(n_questions=4, max_questions=8, sessions=["Session A", "Session B"])
        for q in st.questions.values():
            q.session = "Session A"
        assert any(i["issue"] == "session-gap" for i in _deterministic_gate_issues(st))

    def test_no_session_gap_when_attribution_did_not_run(self):
        """All-None `session` means attribution never ran — not that a session is unrepresented."""
        st = _state(n_questions=4, max_questions=8, sessions=["Session A", "Session B"])
        assert not any(i["issue"] == "session-gap" for i in _deterministic_gate_issues(st))

    def test_a_good_set_produces_no_issues(self):
        st = _state(n_questions=5, max_questions=5)
        for q in st.questions.values():
            q.session = "Introduction to AI Agents"
        assert _deterministic_gate_issues(st) == []


class TestPhaseErrorsAreVisible:
    """A phase that died on an API error used to leave no trace, so an outage looked like a thin
    session."""

    def test_phase_error_fails_the_report_and_says_why(self):
        st = _state()
        st.phase_errors = ["Retrieval Agent: 503 upstream unavailable"]
        report = _build_quality_report(st, 0)
        assert report.pass_fail == "fail"
        assert any("stage failed" in n for n in report.critique)


class TestPerRunModel:
    """A5/A6 — the model came from a module global, so the UI picker did nothing for agent loops and
    two concurrent runs retargeted each other."""

    def test_run_uses_its_own_configured_model(self):
        from src.llm_client import run_model
        st = _state()
        st.config.model = "openai/gpt-4o"
        assert run_model(st) == "openai/gpt-4o"

    def test_two_runs_keep_separate_models(self):
        from src.llm_client import run_model, set_active_model
        a, b = _state(), _state()
        a.config.model = "anthropic/claude-sonnet-4-6"
        b.config.model = "openai/gpt-4o-mini"
        set_active_model("anthropic/claude-opus-4.1")   # a third tab changes the picker
        assert run_model(a) == "anthropic/claude-sonnet-4-6"
        assert run_model(b) == "openai/gpt-4o-mini"

    def test_falls_back_to_the_configured_default(self):
        from src.config import LLM_MODEL
        from src.llm_client import run_model
        st = _state()
        st.config.model = None
        assert run_model(st) == LLM_MODEL

    def test_blank_model_is_not_used(self):
        from src.config import LLM_MODEL
        from src.llm_client import run_model
        st = _state()
        st.config.model = "   "
        assert run_model(st) == LLM_MODEL


class TestJSONExtraction:
    """A4 — `_extract_json` returned `{}` for both "no JSON" and "empty JSON"."""

    @pytest.mark.parametrize("text", ["", "   ", "I could not answer that.",
                                      '{"pass": true', "```json\n{\"a\":\n```"])
    def test_unparseable_returns_none(self, text):
        from src.llm_client import _extract_json
        assert _extract_json(text) is None

    def test_empty_object_is_a_valid_result(self):
        from src.llm_client import _extract_json
        assert _extract_json("{}") == {}

    @pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```',
                                      'Sure! {"a": 1} hope that helps'])
    def test_parses_fenced_and_wrapped_json(self, text):
        from src.llm_client import _extract_json
        assert _extract_json(text) == {"a": 1}


class TestJsonModeRejectionDetection:
    """Only an unsupported-parameter error justifies a second paid call; any exception used to."""

    @pytest.mark.parametrize("msg", [
        "Unsupported parameter: response_format",
        "unrecognized request argument supplied: response_format",
        "json_object is not supported by this model",
    ])
    def test_recognises_a_json_mode_rejection(self, msg):
        from src.llm_client import _is_json_mode_rejection
        assert _is_json_mode_rejection(Exception(msg))

    @pytest.mark.parametrize("msg", ["401 Unauthorized", "429 rate limit exceeded",
                                     "insufficient credits"])
    def test_other_errors_are_not_retried_without_json_mode(self, msg):
        from src.llm_client import _is_json_mode_rejection
        assert not _is_json_mode_rejection(Exception(msg))
