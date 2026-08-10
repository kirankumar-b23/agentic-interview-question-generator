"""The quality gate's deterministic objections, and the one that failed every mature topic.

Two real runs came back `fail` with EVERY numeric condition passing:

    coverage efficiency  0.857 / 0.733   bar 0.6   ok
    question count          39 / 46      bar 5     ok
    composite            0.659 / 0.698   bar 0.6   ok
    reviewer critique    12 / 15 unresolved        FAIL   <-- decided the verdict alone

The first unresolved objection on both was `too-many`: *"39 questions exceeds the requested 15 — the set was
never trimmed."* That is false. `tools._add_retained` deliberately carries the topic's accumulated set into
every run and the final set is NOT capped, so the check was measuring the wrong number — it fired against
36 + 3 and 43 + 3, could never pass a mature topic, and spent both revision rounds telling the Evaluation
agent to trim a set it must not trim.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """`_critique_question_set` makes one LLM call for the judged part. These tests are about the
    DETERMINISTIC checks around it, so stub the boundary — the network guard would fail us otherwise."""
    import src.agent as agent_mod
    monkeypatch.setattr(agent_mod, "chat_completion_json",
                        lambda **kw: {"pass": True, "must_fix": [], "summary": "stubbed"})


def _state(n_retained: int, n_fresh: int, max_questions: int):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=max_questions),
                    data_store=get_data_store())
    st.session_context = SimpleNamespace(
        session_name="S", session_type="mixed", learning_outcomes=[], interview_topics=["T"],
        key_concepts=[], scope_in=[], scope_out=[], matched_kp_ids=[])
    st.relevance_scored = True
    for i in range(n_retained + n_fresh):
        q = QuestionDetail(question_id=f"q{i}", category="GEN_AI", topic="t", difficulty="Medium",
                           source="interview_db",
                           content=f"What is generative AI concept number {i} used for?")
        q.retained = i < n_retained
        st.questions[q.question_id] = q
    return st


def _issues(st, kind):
    from src.agent import _critique_question_set
    return [i for i in (_critique_question_set(st).get("must_fix") or []) if i.get("issue") == kind]


class TestTooManyCountsNewlyFound:
    """The numbers are the two real runs. Asserting on a synthetic 20-vs-15 would pass either way."""

    @pytest.mark.parametrize("retained,fresh", [(36, 3), (43, 3)])
    def test_a_mature_topic_no_longer_objects(self, retained, fresh):
        assert _issues(_state(retained, fresh, 15), "too-many") == [], (
            f"{retained} carried over + {fresh} newly found is a healthy accumulated set, not an "
            f"untrimmed one")

    def test_the_raw_pool_failure_mode_still_objects(self):
        """The check's actual purpose: `_enforce_submission` failed and ~263 unranked candidates shipped.
        All newly found, so it must still fire — this is what a blanket removal would have lost."""
        found = _issues(_state(0, 263, 15), "too-many")
        assert found, "an untrimmed raw candidate pool must still be caught"
        assert "263 newly-found" in found[0]["suggestion"]

    def test_the_suggestion_does_not_send_a_reviewer_to_re_trim(self):
        found = _issues(_state(0, 263, 15), "too-many")
        assert "Carried-over questions are excluded" in found[0]["suggestion"], (
            "the old wording blamed the whole set, so a reviewer could not tell which part was wrong")

    def test_a_normal_run_within_the_slider_is_silent(self):
        assert _issues(_state(0, 12, 15), "too-many") == []

    def test_only_newly_found_counts_not_the_total(self):
        """The mutation this guards: reverting to `len(state.questions)`. 4 fresh is under the bar of 5
        while the 40-question total is far over it, so the two rules disagree here by construction."""
        st = _state(36, 4, 5)
        assert len(st.questions) == 40
        assert _issues(st, "too-many") == [], "counted the total instead of the newly-found questions"


class TestTooFewStillCountsTheTotal:
    """`too-few` is the opposite case: a 39-question set carried over from the topic is not thin, and a
    3-question set is — regardless of where the questions came from."""

    def test_a_thin_set_objects_even_when_carried_over(self):
        found = _issues(_state(3, 0, 15), "too-few")
        assert found, "3 questions is thin whether they are new or retained"

    def test_a_large_carried_over_set_does_not(self):
        assert _issues(_state(39, 0, 15), "too-few") == []


class TestTheVerdictIsDecidedByRealObjections:
    """`passed = all(c["ok"] …)` in `_build_quality_report`, so one false objection failed the run despite
    three green numeric checks. With the false one gone, a clean set reaches `pass`."""

    def test_a_healthy_accumulated_set_can_pass_the_gate(self, monkeypatch):
        from src.pipeline import _build_quality_report

        st = _state(36, 3, 15)
        st.relevance_scored = True
        st.gate_forced = False
        st.gate_issues = []
        report = _build_quality_report(st, 0)
        critique_check = [c for c in report.gate_checks if c["name"] == "reviewer critique"][0]
        assert critique_check["ok"] is True
        assert critique_check["value"] == "no objections"

    def test_an_unresolved_objection_still_fails_it(self):
        from src.pipeline import _build_quality_report

        st = _state(36, 3, 15)
        st.gate_forced = True
        st.gate_issues = [{"id": None, "issue": "duplicate", "suggestion": "x"}]
        report = _build_quality_report(st, 2)
        assert report.pass_fail == "fail"
        assert [c for c in report.gate_checks if c["name"] == "reviewer critique"][0]["ok"] is False
