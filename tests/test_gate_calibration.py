"""The quality gate: what it scores coverage against, and how it normalises for supply.

Three consecutive runs failed the gate and the sets were not the problem. Two causes, both pinned here:

1. **Coverage was scored against `learning_outcomes`** — the LESSON, setup steps included. Measured
   across the 53 curated sessions, 36 of 322 outcomes are environment mechanics ("Set up a Kaggle
   account with phone verification", "Use Ngrok to create secure tunnels"), concentrated so badly that
   one session has 5 of 6 uncoverable. On the Image-Generation topic that capped the best achievable
   coverage at 0.56 — under the 0.60 pass bar with unlimited questions. `interview_topics`, built from
   the same reading material for exactly this purpose, has **0 of 385** such items.

2. **Coverage was scored as a fraction of ALL targets**, so it was bounded by supply. Judged coverage
   credits few targets per question, so with 5 questions and 22 topics it lands around 0.23. One run
   scored 0.227 and was failed for it.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail
from src.pipeline import _outcome_coverage, coverage_targets


def _q(content):
    return QuestionDetail(category="GEN_AI", content=content, topic="Gen AI",
                          source="interview_db", difficulty="Medium")


def _state(questions=(), outcomes=(), topics=(), judged=None):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=15),
                    data_store=get_data_store())
    st.session_context = SimpleNamespace(
        learning_outcomes=list(outcomes), interview_topics=list(topics),
        key_concepts=[], scope_in=[], scope_out=[], matched_kp_ids=[], session_type="mixed",
    )
    st.questions = {q.question_id: q for q in questions}
    if judged is not None:
        st.judged_coverage = judged
    return st


# ── What coverage is measured against ───────────────────────────────────────

class TestCoverageTargetsAreInterviewTopics:
    LESSON = ["Set up a Kaggle account with phone verification to enable GPU access",
              "Use Ngrok to create secure tunnels for accessing remote instances"]
    TOPICS = ["GPU resource allocation and VRAM requirements",
              "API authentication and secure token management"]

    def test_interview_topics_win_over_learning_outcomes(self):
        ctx = SimpleNamespace(learning_outcomes=self.LESSON, interview_topics=self.TOPICS)
        assert coverage_targets(ctx) == self.TOPICS

    def test_falls_back_to_outcomes_when_a_session_has_no_topics(self):
        """A session predating the curated field must still be measured, not silently scored 1.0."""
        ctx = SimpleNamespace(learning_outcomes=self.LESSON, interview_topics=[])
        assert coverage_targets(ctx) == self.LESSON

    def test_blank_entries_are_ignored(self):
        ctx = SimpleNamespace(learning_outcomes=["real outcome"], interview_topics=["  ", ""])
        assert coverage_targets(ctx) == ["real outcome"]

    def test_no_context_is_empty_not_an_error(self):
        assert coverage_targets(None) == []

    def test_the_shipped_curated_data_has_no_setup_steps_in_its_topics(self):
        """The claim this whole change rests on, asserted against the real data file."""
        import json
        import re

        from src.config import DATA_DIR
        so = json.loads((DATA_DIR / "reading_materials" / "session_outcomes.json")
                        .read_text(encoding="utf-8"))
        setup = re.compile(r"phone verification|gpu quota|free tier|notebook settings|ngrok|webui|"
                           r"session duration|troubleshoot common setup", re.I)
        offenders = [t for d in so.values() for t in (d.get("interview_topics") or []) if setup.search(t)]
        assert not offenders, f"interview_topics must stay interviewable, found: {offenders[:3]}"


# ── Supply normalisation ────────────────────────────────────────────────────

class TestCoverageIsNormalisedForSupply:
    TOPICS = [f"topic {i}" for i in range(22)]

    def _judged(self, n_covered, total):
        return {"covered": [f"topic {i}" for i in range(n_covered)],
                "missing": [f"topic {i}" for i in range(n_covered, total)],
                "pairs": [], "method": "llm-judged"}

    def test_five_questions_covering_five_of_22_is_full_efficiency(self):
        """The exact run that failed at its own maximum: 5 questions, 22 topics, 5 covered."""
        st = _state([_q(f"q{i}?") for i in range(5)], topics=self.TOPICS,
                    judged=self._judged(5, 22))
        cov = _outcome_coverage(st)
        assert cov.coverage_efficiency == 1.0, "every question earned its place"
        assert cov.topic_coverage == pytest.approx(0.227, abs=0.001), "and the honest number is still reported"
        assert cov.supply_capped is True

    def test_questions_piling_onto_the_same_topic_still_score_low(self):
        """The normalisation must not become a free pass — this is the discriminating case."""
        st = _state([_q(f"q{i}?") for i in range(5)], topics=self.TOPICS,
                    judged=self._judged(3, 22))
        cov = _outcome_coverage(st)
        assert cov.coverage_efficiency == pytest.approx(0.6, abs=0.001)

    def test_two_of_five_questions_earning_their_place_fails(self):
        st = _state([_q(f"q{i}?") for i in range(5)], topics=self.TOPICS,
                    judged=self._judged(2, 22))
        assert _outcome_coverage(st).coverage_efficiency < 0.6

    def test_a_set_larger_than_the_topic_list_is_not_capped(self):
        """With more questions than topics, efficiency and coverage converge — no free ride."""
        st = _state([_q(f"q{i}?") for i in range(10)], topics=[f"topic {i}" for i in range(4)],
                    judged=self._judged(2, 4))
        cov = _outcome_coverage(st)
        assert cov.supply_capped is False
        assert cov.coverage_efficiency == pytest.approx(0.5, abs=0.001)
        assert cov.topic_coverage == pytest.approx(0.5, abs=0.001)

    def test_no_targets_with_no_questions_is_zero_not_perfect(self):
        cov = _outcome_coverage(_state([], topics=[]))
        assert cov.topic_coverage == 0.0 and cov.coverage_efficiency == 0.0


# ── The gate verdict ────────────────────────────────────────────────────────

def _report(state, **kw):
    from src.pipeline import _build_quality_report
    return _build_quality_report(state, kw.get("revision_round", 0))


class TestGateVerdictAndItsExplanation:
    TOPICS = [f"topic {i}" for i in range(22)]

    def _state_for(self, n_q, n_covered, fit=0.7):
        qs = [_q(f"question number {i}?") for i in range(n_q)]
        for q in qs:
            q.session_fit = fit
            q.relevance_score = 0.8
        st = _state(qs, topics=self.TOPICS,
                    judged={"covered": [f"topic {i}" for i in range(n_covered)],
                            "missing": [f"topic {i}" for i in range(n_covered, 22)],
                            "pairs": [], "method": "llm-judged"})
        return st

    def test_gate_checks_name_every_condition_with_its_value_and_bar(self):
        """The report used to carry pass_fail and nothing else, which is why a failure was unreadable."""
        rep = _report(self._state_for(5, 2))
        names = {c["name"] for c in rep.gate_checks}
        assert names == {"coverage efficiency", "question count", "composite", "reviewer critique"}
        for c in rep.gate_checks:
            assert "value" in c and "bar" in c and isinstance(c["ok"], bool)
        failed = [c["name"] for c in rep.gate_checks if not c["ok"]]
        assert "coverage efficiency" in failed
        assert any("Gate not passed" in n for n in rep.critique)

    def test_a_supply_capped_set_is_told_it_is_supply_capped(self):
        """A corpus fact must not read as a quality problem."""
        rep = _report(self._state_for(5, 5))
        note = " ".join(rep.critique)
        assert "limited by supply, not quality" in note
        assert "of 22 interview topics" in note

    def test_predicted_accept_cannot_veto_on_its_own(self):
        """It keeps its 30% composite weight but no longer fails a set single-handedly: it read 0.2 on
        two consecutive runs purely because those topics had few reviewer labels."""
        st = self._state_for(12, 12)
        rep = _report(st)
        assert not any(c["name"] == "predicted_accept" for c in rep.gate_checks)

    def test_both_coverage_numbers_are_reported(self):
        rep = _report(self._state_for(5, 5))
        assert rep.metric_scores["coverage_efficiency"] == 1.0
        assert rep.metric_scores["topic_coverage"] == pytest.approx(0.227, abs=0.001)

    def test_the_weakest_of_the_three_real_runs_still_fails(self):
        """7efceef9: 5 questions examining only 3 distinct topics. The fix must not pass this."""
        rep = _report(self._state_for(5, 3, fit=0.646))
        assert rep.pass_fail == "fail"


class TestTheVerdictBreakdownIsComplete:
    """A live run showed three green checks beside a FAIL verdict, because the LLM critique is a
    SEPARATE gate (a force-passed set is failed regardless of the numbers) and was not listed."""

    TOPICS = [f"topic {i}" for i in range(22)]

    def _state(self, forced):
        qs = [_q(f"question number {i}?") for i in range(8)]
        for q in qs:
            q.session_fit = 0.7
            q.relevance_score = 0.8
        st = _state(qs, topics=self.TOPICS,
                    judged={"covered": [f"topic {i}" for i in range(8)],
                            "missing": [f"topic {i}" for i in range(8, 22)],
                            "pairs": [], "method": "llm-judged"})
        st.gate_forced = forced
        st.gate_issues = [{"issue": "duplicate"}] if forced else []
        return st

    def test_a_force_passed_set_shows_the_critique_as_the_failing_check(self):
        from src.pipeline import _build_quality_report

        rep = _build_quality_report(self._state(True), 2)
        crit = [c for c in rep.gate_checks if c["name"] == "reviewer critique"]
        assert crit and crit[0]["ok"] is False, "the critique must appear as a check, and as failing"
        assert rep.pass_fail == "fail"
        assert not all(c["ok"] for c in rep.gate_checks), "no all-green breakdown beside a FAIL verdict"

    def test_a_clean_run_shows_every_check_green_and_passes(self):
        from src.pipeline import _build_quality_report

        rep = _build_quality_report(self._state(False), 0)
        assert all(c["ok"] for c in rep.gate_checks)
        assert rep.pass_fail == "pass"

    def test_the_supply_note_states_no_ceiling_it_cannot_prove(self):
        """The first version claimed "cannot cover more than N of T" — and a live run covered 11 of 22
        with 6 questions, because one question can examine several topics."""
        from src.pipeline import _build_quality_report

        rep = _build_quality_report(self._state(False), 0)
        note = " ".join(rep.critique)
        assert "cannot cover more than" not in note
        assert "ceiling" not in note
