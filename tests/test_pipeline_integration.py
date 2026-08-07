"""End-to-end pipeline tests with the LLM boundary stubbed.

These drive the REAL `AgentPipeline.run` — all four agents, the tool dispatch, the session-fit gate,
selection and the quality gate — with only `client.chat.completions.create` and
`chat_completion_json` replaced. That's the layer worth stubbing: everything above it is the logic
these tests exist to check, and stubbing it makes the failure paths reproducible instead of requiring
a live API outage to observe.

The scenarios mirror the defects in tests/test_failure_paths.py, but through the whole pipeline
rather than a single function.
"""
import json
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig
from src.pipeline import AgentPipeline

SESSION = "Introduction to AI Agents"


# ── LLM stubs ────────────────────────────────────────────────────────────────

def _usage(prompt=100, completion=50):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _tool_call(tool_id, name, args):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        type="function",
    )


def _response(tool_calls=None, content=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=_usage())


class FakeAgentLLM:
    """Serves canned tool-call responses per agent phase, keyed by which tools the agent was given."""

    def __init__(self, *, evaluation_submits=True, evaluation_text_only=False):
        self.evaluation_submits = evaluation_submits
        self.evaluation_text_only = evaluation_text_only
        self.calls = []
        self.models_seen = []

    def create(self, *, model=None, messages=None, tools=None, **_kw):
        self.models_seen.append(model)
        names = {t["function"]["name"] for t in (tools or [])}
        self.calls.append(names)

        if "understand_session" in names:
            return _response([_tool_call("c1", "understand_session", {})])
        if "search_question_bank" in names:
            return _response([_tool_call("c2", "search_question_bank",
                                         {"query": "AI agent components", "limit": 25})])
        if "validate_relevance" in names:
            return _response([_tool_call("c3", "validate_relevance", {}),
                              _tool_call("c4", "deduplicate_questions", {})])
        if "submit_question_set" in names:
            if self.evaluation_text_only:
                # The failure this exists to reproduce: the agent replies with prose and no tool call,
                # so nothing ever selects or trims the candidate pool.
                return _response(content="The set looks good to me.")
            if self.evaluation_submits:
                return _response([_tool_call("c5", "submit_question_set", {})])
            return _response([_tool_call("c6", "check_outcome_coverage", {})])
        return _response(content="done")


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub every network boundary: the agent loop, the JSON helpers, and Tavily.

    Tavily has to be stubbed too — the RetrievalAgent runs a live health check and a web search
    before anything else, which makes these tests slow, non-deterministic and dependent on an API
    key. The bank retrieval that remains is local and is what these tests care about.
    """
    fake = FakeAgentLLM()

    def _install():
        import src.agents.base_agent as base
        monkeypatch.setattr(base, "get_client",
                            lambda: SimpleNamespace(chat=SimpleNamespace(completions=fake)))
        # The gate and the relevance judge go through chat_completion_json.
        import src.agent as agent_mod
        import src.tools as tools_mod
        monkeypatch.setattr(agent_mod, "chat_completion_json",
                            lambda **kw: {"pass": True, "must_fix": [], "summary": "ok"})
        monkeypatch.setattr(tools_mod, "chat_completion_json", _fake_relevance)
        import src.session_understanding as su
        monkeypatch.setattr(su, "chat_completion_json", lambda **kw: _fake_session_context())

        # No web calls: report Tavily as unavailable so the run goes bank-only.
        import src.sources.tavily_search as tav
        monkeypatch.setattr(tav.TavilyConnector, "health_check",
                            lambda self: (False, "no_key", "stubbed out for tests"))
        monkeypatch.setattr(tools_mod, "tool_search_web_questions",
                            lambda state, *a, **kw: {"found": 0, "added": 0, "total_accumulated":
                                                     len(state.questions), "web_remaining": 0})
        return fake

    return _install


def _fake_relevance(*, user_prompt="", **_kw):
    """Score every question in the batch as clearly relevant."""
    payload = user_prompt[user_prompt.find("["):] if "[" in user_prompt else "[]"
    try:
        items = json.loads(payload)
    except json.JSONDecodeError:
        items = []
    return {"scores": [{"n": it["n"], "score": 0.85, "difficulty": "Medium"} for it in items]}


def _fake_session_context():
    return {
        "learning_outcomes": ["Understand the core components of an AI agent",
                              "Explain how agent memory and planning work"],
        "key_concepts": ["agents", "memory", "planning"],
        "interview_topics": ["AI agent architecture", "agent memory"],
        "scope_in": ["agents"], "scope_out": ["image generation"],
        "session_type": "mixed",
        "matched_kp_ids": [], "matched_csv_topics": [],
        "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    }


def _run(max_questions=8, model="anthropic/claude-haiku-4-5"):
    cfg = GenerationConfig(session_names=[SESSION], max_questions=max_questions,
                           category="GEN_AI", model=model)
    events = []

    def emit(run_id, step, status, detail="", **fields):
        events.append({"step": step, "status": status, "detail": detail, **fields})

    result = AgentPipeline().run(cfg, "test-run", emit)
    return result, events


class TestHappyPath:
    def test_run_completes_and_selects_a_set(self, stub_llm):
        stub_llm()
        result, events = _run(max_questions=8)
        assert result.error is None
        assert result.curated_output is not None
        assert 0 < len(result.curated_output.question_details) <= 8

    def test_completion_event_carries_structured_totals(self, stub_llm):
        """The UI reads these fields; it used to regex them out of the prose `detail`."""
        stub_llm()
        _, events = _run()
        complete = next(e for e in events if e["step"] == "complete")
        assert isinstance(complete["questions"], int)
        assert "usage" in complete and "revisions" in complete
        assert complete["verdict"] in ("pass", "fail")

    def test_events_carry_agent_attribution_and_timing(self, stub_llm):
        stub_llm()
        _, events = _run()
        tool_events = [e for e in events if e["status"] == "done" and not e["step"].startswith("phase:")]
        assert any("agent" in e for e in tool_events), "events must say which agent acted"
        assert any("duration_ms" in e for e in tool_events), "events must carry timing"

    def test_run_uses_the_model_from_its_own_config(self, stub_llm):
        fake = stub_llm()
        result, _ = _run(model="openai/gpt-4o-mini")
        assert set(m for m in fake.models_seen if m) == {"openai/gpt-4o-mini"}
        assert result.quality_report.api_usage.get("model") == "openai/gpt-4o-mini"


class TestEvaluationAgentNeverSubmits:
    """A1 through the real pipeline: the pool must still be selected and trimmed."""

    def test_text_only_reply_still_yields_a_trimmed_set(self, stub_llm, monkeypatch):
        fake = stub_llm()
        fake.evaluation_text_only = True
        result, events = _run(max_questions=6)
        details = result.curated_output.question_details
        assert len(details) <= 6, f"pool was shipped untrimmed: {len(details)} questions"
        assert any(e["step"] == "submit_question_set" and e["status"] == "warning" for e in events)

    def test_the_report_says_the_agent_did_not_submit(self, stub_llm):
        fake = stub_llm()
        fake.evaluation_text_only = True
        result, _ = _run(max_questions=6)
        assert any("never submitted" in n for n in result.quality_report.critique)

    def test_budget_spent_elsewhere_still_yields_a_trimmed_set(self, stub_llm):
        """The agent calls only check_* tools and never submits."""
        fake = stub_llm()
        fake.evaluation_submits = False
        result, _ = _run(max_questions=4)
        assert len(result.curated_output.question_details) <= 4


class TestGateFailureSurfaces:
    def test_force_pass_is_reported_and_fails_the_verdict(self, stub_llm, monkeypatch):
        stub_llm()
        import src.agent as agent_mod
        monkeypatch.setattr(agent_mod, "chat_completion_json", lambda **kw: {
            "pass": False,
            "must_fix": [{"id": "x", "issue": "duplicate", "suggestion": "drop one"}],
            "summary": "a duplicate remains",
        })
        result, _ = _run()
        report = result.quality_report
        assert report.pass_fail == "fail"
        assert any("did NOT pass" in n for n in report.critique)
        assert report.flagged_questions

    def test_gate_exception_does_not_pass_the_set(self, stub_llm, monkeypatch):
        stub_llm()
        import src.agent as agent_mod
        from src.llm_client import JSONResponseError

        def _boom(**_kw):
            raise JSONResponseError("truncated reply")
        monkeypatch.setattr(agent_mod, "chat_completion_json", _boom)
        result, _ = _run()
        assert result.quality_report.pass_fail == "fail"


class TestApiOutage:
    """A phase dying on an API error must be visible, not look like a thin session."""

    def test_agent_api_error_is_reported(self, stub_llm, monkeypatch):
        stub_llm()
        import src.agents.base_agent as base

        class Dead:
            def create(self, **_kw):
                raise RuntimeError("Error code: 401 - Missing Authentication header")
        monkeypatch.setattr(base, "get_client",
                            lambda: SimpleNamespace(chat=SimpleNamespace(completions=Dead())))
        result, _ = _run()
        assert result.error is None, "an outage should degrade, not crash the run"
        assert result.quality_report.pass_fail == "fail"
        assert any("stage failed" in n for n in result.quality_report.critique)
        assert any("401" in n for n in result.quality_report.critique), "the cause must be named"


class TestRelevanceJudgeOutage:
    def test_total_relevance_failure_is_reported(self, stub_llm, monkeypatch):
        stub_llm()
        import src.tools as tools_mod
        from src.llm_client import JSONResponseError

        def _boom(**_kw):
            raise JSONResponseError("no JSON")
        monkeypatch.setattr(tools_mod, "chat_completion_json", _boom)
        result, _ = _run()
        assert result.quality_report.pass_fail == "fail"
        assert any("relevance judge failed" in n for n in result.quality_report.critique)


class TestConcurrentRuns:
    """Two runs in flight must not contaminate each other.

    The model used to come from a module-level global that each run set on start, so a second browser
    tab retargeted the first run's in-flight calls and the first run's `api_usage["model"]` then
    mispriced its own tokens.
    """

    def test_each_run_uses_its_own_model(self, stub_llm):
        fake = stub_llm()
        per_run_models = {}

        # Record which model each create() call used, keyed by the run that was active.
        original_create = fake.create

        def _tracking_create(*, model=None, **kw):
            per_run_models.setdefault(model, 0)
            per_run_models[model] += 1
            return original_create(model=model, **kw)
        fake.create = _tracking_create

        a, _ = _run(model="anthropic/claude-haiku-4-5")
        b, _ = _run(model="openai/gpt-4o-mini")

        assert a.quality_report.api_usage["model"] == "anthropic/claude-haiku-4-5"
        assert b.quality_report.api_usage["model"] == "openai/gpt-4o-mini"
        # Both models were actually used — neither run silently rode the other's choice.
        assert per_run_models.get("anthropic/claude-haiku-4-5", 0) > 0
        assert per_run_models.get("openai/gpt-4o-mini", 0) > 0

    def test_ui_default_does_not_override_a_run(self, stub_llm):
        """Changing the picker mid-run must not retarget the run."""
        from src.llm_client import set_active_model
        stub_llm()
        set_active_model("anthropic/claude-opus-4.1")     # a third tab changes the picker
        result, _ = _run(model="openai/gpt-4o-mini")
        assert result.quality_report.api_usage["model"] == "openai/gpt-4o-mini"

    def test_runs_keep_independent_state(self, stub_llm):
        """Each run must honour its OWN config and carry its own report.

        Note: the two runs legitimately return overlapping question_ids — bank questions keep their
        corpus id, so the same question retrieved twice is the same id by design. What must not leak is
        per-run STATE: the requested count, the usage totals and the report.
        """
        stub_llm()
        a, _ = _run(max_questions=4)
        b, _ = _run(max_questions=8)

        assert len(a.curated_output.question_details) <= 4
        assert len(b.curated_output.question_details) <= 8
        assert a.quality_report is not b.quality_report
        # Usage is accumulated per run, so neither total includes the other's calls.
        assert a.quality_report.api_usage["llm_calls"] > 0
        assert b.quality_report.api_usage["llm_calls"] > 0
        # The question objects are distinct instances even when ids coincide, so a decision or a
        # score written on one run's set cannot mutate the other's.
        a_by_id = {q.question_id: q for q in a.curated_output.question_details}
        for q in b.curated_output.question_details:
            if q.question_id in a_by_id:
                assert q is not a_by_id[q.question_id], "runs must not share question INSTANCES"


class TestConversationalOnlyRunsInTheRealPipeline:
    """The hands-on filter has to be WIRED, not merely correct.

    `tests/test_interview_format.py` proves `_drop_hands_on` behaves; it cannot prove
    `_pick_questions` calls it. A live run would have shown that, and the run intended for it died on a
    401 (the OpenRouter key in .env lost its `sk-` prefix), so this closes the gap with the LLM boundary
    stubbed instead — the same trick the rest of this file uses.
    """

    # ON-TOPIC for the stubbed session (agents / memory / planning) ON PURPOSE. The first version of
    # this test injected "Write a Python program to reverse a linked list", which the session-fit gate
    # dropped on its own — so the test passed even with the filter unwired and proved nothing. It has to
    # be a candidate that ONLY the hands-on filter has a reason to remove.
    HANDS_ON = ("Write a Python function that gives an AI agent short-term memory using a list, and "
                "explain how the agent's planning loop reads from it.")

    def _inject(self, monkeypatch, text):
        """Force one hands-on candidate into the pool the bank returns."""
        import src.tools as tools_mod
        real = tools_mod.tool_search_question_bank

        def _with_hands_on(state, *a, **kw):
            out = real(state, *a, **kw)
            from src.models import QuestionDetail
            q = QuestionDetail(question_id="injected-hands-on", category="GEN_AI", content=text,
                               topic="Gen AI", difficulty="Medium", source="interview_db")
            state.questions[q.question_id] = q
            return out

        monkeypatch.setattr(tools_mod, "tool_search_question_bank", _with_hands_on)

    def test_a_hands_on_candidate_is_removed_from_the_pool_and_never_ships(self, stub_llm, monkeypatch):
        """Asserts the POOL removal, not just absence from the final set.

        "It isn't in the shipped set" is a vacuous claim here and it took a mutation check to notice:
        `_select_final` trims ~150 candidates to 8, so an injected question misses the cut whether or not
        the filter ran, and the assertion passed with the filter unwired. The removal record is the part
        only this filter can produce.
        """
        stub_llm()
        self._inject(monkeypatch, self.HANDS_ON)

        result, events = _run(max_questions=8)
        assert result.error is None
        dropped = [r for r in (result.removed or [])
                   if r.get("stage") == "hands_on" and r.get("content") == self.HANDS_ON]
        assert dropped, "the injected hands-on candidate must be removed AT the hands_on stage"
        shipped = result.curated_output.question_details
        assert "injected-hands-on" not in {q.question_id for q in shipped}

    def test_the_drop_is_reported_on_its_own_step(self, stub_llm, monkeypatch):
        """A pool filter that shrinks supply silently gets misread as 'this session has few
        questions' — the exact misdiagnosis scripts/yield_report.py exists to prevent."""
        stub_llm()
        self._inject(monkeypatch, self.HANDS_ON)

        result, events = _run(max_questions=8)
        steps = [e for e in events if e["step"] == "hands_on"]
        assert steps, "the filter must emit its own step so a run is debuggable"
        assert steps[0].get("dropped", 0) >= 1
        assert any(r.get("stage") == "hands_on" for r in (result.removed or []))

    def test_with_the_flag_off_the_same_candidate_survives_the_filter(self, stub_llm, monkeypatch):
        """Proves the flag governs real pipeline behaviour, not just the helper."""
        import src.config as cfg
        monkeypatch.setattr(cfg, "CONVERSATIONAL_ONLY", False)
        stub_llm()
        self._inject(monkeypatch, self.HANDS_ON)

        result, events = _run(max_questions=8)
        assert not any(r.get("stage") == "hands_on" for r in (result.removed or []))
