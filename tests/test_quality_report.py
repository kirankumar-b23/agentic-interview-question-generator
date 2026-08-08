"""Tests for the scoring in `_build_quality_report` and `_outcome_coverage`.

These pin the central correction made to this project's evaluation: the composite score must be
built ONLY from signals independent of the selector. Historically 40% of it was the mean LLM
relevance score used to *pick* the questions, so the number rose whenever the selector got more
confident rather than when the set got better — measured correlation with reviewer approval across
36 real runs was r = 0.16. If a future change puts `self_relevance` back into the composite or the
pass decision, these tests fail.
"""
import pytest

from src import embeddings
from src.agent import AgentState
from src.data_loader import get_data_store
from src.models import GenerationConfig, QuestionDetail, SessionContext
from src.pipeline import _build_quality_report, _outcome_coverage

OUTCOMES = ["Understand the core components of an AI agent",
            "Explain how agent memory and planning work"]

ON_TOPIC = [("What are the core components of an AI Agent?", "Easy", 0.86),
            ("Why is memory critical for the performance of AI agents?", "Medium", 0.71),
            ("How do agents plan a multi-step task?", "Medium", 0.62),
            ("How do agents decide when to stop a task?", "Hard", 0.55),
            ("What is the role of an orchestrator in AI agents?", "Easy", 0.51)]

OFF_TOPIC = [("Explain the complete workflow and architecture of your project.", "Hard", 0.31),
             ("Tell me about yourself", "Easy", 0.10),
             ("What is the difference between a list and a tuple in Python?", "Easy", 0.14),
             ("How do you handle authentication in a REST API?", "Medium", 0.11),
             ("On a scale of 1 to 10, how good is your code?", "Easy", 0.09)]


def _context() -> SessionContext:
    return SessionContext(
        session_name="Introduction to AI Agents",
        learning_outcomes=OUTCOMES,
        key_concepts=["agents", "memory", "planning"],
        interview_topics=["AI agent architecture", "agent memory"],
        scope_in=["agents"], scope_out=["fine-tuning"],
        session_type="mixed",
        matched_kp_ids=[], matched_csv_topics=[], prerequisite_kp_chain=[],
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )


def _state(rows, *, max_questions: int = 5, self_relevance: float = 0.9) -> AgentState:
    cfg = GenerationConfig(session_names=["Introduction to AI Agents"], max_questions=max_questions)
    st = AgentState(config=cfg, data_store=get_data_store())
    st.session_context = _context()
    for i, (content, difficulty, fit) in enumerate(rows):
        st.questions[f"q{i}"] = QuestionDetail(
            question_id=f"q{i}", category="GEN_AI", content=content, topic="Gen AI",
            difficulty=difficulty, source="interview_db",
            relevance_score=self_relevance, session_fit=fit,
        )
    return st


needs_embeddings = pytest.mark.skipif(not embeddings.available(),
                                      reason="sentence-transformers unavailable")


class TestCompositeIsSelectorIndependent:
    def test_self_relevance_is_reported_but_not_scored(self):
        """Raising the selector's own confidence from 0.5 to 1.0 must not move the composite."""
        low = _build_quality_report(_state(ON_TOPIC, self_relevance=0.5), 0)
        high = _build_quality_report(_state(ON_TOPIC, self_relevance=1.0), 0)
        assert low.composite_score == high.composite_score
        assert low.metric_scores["self_relevance"] == 0.5
        assert high.metric_scores["self_relevance"] == 1.0

    def test_perfect_self_relevance_cannot_pass_a_badly_grounded_set(self):
        """The old scoring let a confident selector carry an off-topic set to a passing score."""
        report = _build_quality_report(_state(OFF_TOPIC, self_relevance=1.0), 0)
        assert report.pass_fail == "fail"

    def test_difficulty_balance_reported_but_not_scored(self):
        """GenAI-bank difficulty labels are ~95% "Medium", so scoring them measures label noise.
        An all-Medium set and a balanced set must score the same composite."""
        balanced = [(c, d, f) for (c, d, f) in ON_TOPIC]
        all_medium = [(c, "Medium", f) for (c, _, f) in ON_TOPIC]
        assert (_build_quality_report(_state(balanced), 0).composite_score
                == _build_quality_report(_state(all_medium), 0).composite_score)

    def test_grounding_is_averaged_from_session_fit(self):
        report = _build_quality_report(_state(ON_TOPIC), 0)
        expected = round(sum(f for *_, f in ON_TOPIC) / len(ON_TOPIC), 3)
        assert report.metric_scores["session_grounding"] == pytest.approx(expected, abs=0.01)


@needs_embeddings
class TestScoresTrackQuality:
    def test_on_topic_set_outscores_off_topic_set(self):
        on = _build_quality_report(_state(ON_TOPIC), 0)
        off = _build_quality_report(_state(OFF_TOPIC), 0)
        assert on.composite_score > off.composite_score

    def test_on_topic_set_has_higher_coverage(self):
        on = _build_quality_report(_state(ON_TOPIC), 0)
        off = _build_quality_report(_state(OFF_TOPIC), 0)
        assert on.metric_scores["topic_coverage"] > off.metric_scores["topic_coverage"]


class TestSizeFloor:
    def test_below_minimum_is_capped_and_fails(self):
        report = _build_quality_report(_state(ON_TOPIC[:2], max_questions=8), 0)
        assert report.composite_score <= 0.4
        assert report.pass_fail == "fail"

    def test_short_set_says_why(self):
        report = _build_quality_report(_state(ON_TOPIC[:2], max_questions=8), 0)
        assert any("question(s)" in n for n in report.critique)

    def test_empty_set_does_not_crash(self):
        report = _build_quality_report(_state([], max_questions=8), 0)
        assert report.pass_fail == "fail"


class TestReportHonesty:
    def test_reports_when_acceptance_is_unmeasurable(self):
        """Either a measured acceptance figure or an explicit statement that it is unavailable —
        never silence, and never an unmeasured number presented as measured."""
        notes = " ".join(_build_quality_report(_state(ON_TOPIC), 0).critique).lower()
        assert "predicted reviewer acceptance" in notes

    def test_all_metrics_present(self):
        scores = _build_quality_report(_state(ON_TOPIC), 0).metric_scores
        for key in ("coverage_efficiency", "topic_coverage", "session_grounding", "predicted_accept", "set_size",
                    "self_relevance", "source_diversity", "difficulty_balance"):
            assert key in scores, f"missing metric: {key}"

    def test_revision_rounds_are_recorded(self):
        assert _build_quality_report(_state(ON_TOPIC), 2).loops_used == 2


class TestOutcomeCoverage:
    def test_no_outcomes_is_full_coverage(self):
        """Nothing to cover → vacuously covered, not a zero that reads as failure."""
        st = _state(ON_TOPIC)
        st.session_context.learning_outcomes = []
        assert _outcome_coverage(st).topic_coverage == 1.0

    def test_no_questions_is_zero_coverage(self):
        assert _outcome_coverage(_state([])).topic_coverage == 0.0

    @needs_embeddings
    def test_relevant_questions_cover_outcomes(self):
        assert _outcome_coverage(_state(ON_TOPIC)).topic_coverage > 0.0

    @needs_embeddings
    def test_coverage_is_a_fraction(self):
        assert 0.0 <= _outcome_coverage(_state(OFF_TOPIC)).topic_coverage <= 1.0
