"""Tests for the human-agreement metric — the eval's only selector-independent signal.

The behaviour that matters most is what it does when it CANNOT measure: it must return None so the
caller reports "unknown", never a number nobody measured. A silent 0.0 would look like a real
finding and would be averaged into aggregates as if it were one.
"""
import pytest

from src import embeddings
from src.human_agreement import (DECISION_MARGIN, NEAR_DUPLICATE_REJECT, predict_accept,
                                 split_labels)

GOOD = [{"question": "What are the core components of an AI Agent?", "decision": "good",
         "session": "S1"},
        {"question": "Why is memory critical for AI agents?", "decision": "good", "session": "S1"}]
BAD = [{"question": "Explain the complete workflow and architecture of your project.",
        "decision": "bad", "session": "S1"},
       {"question": "How have you adopted AI in your workflows?", "decision": "bad",
        "session": "S1"}]

needs_embeddings = pytest.mark.skipif(not embeddings.available(),
                                      reason="sentence-transformers unavailable")


class TestUnmeasurableCasesReturnNone:
    """None means "not measured". These must never come back as 0.0."""

    def test_no_questions(self):
        assert predict_accept([], GOOD + BAD) is None

    def test_blank_questions(self):
        assert predict_accept(["", "   "], GOOD + BAD) is None

    def test_no_labels(self):
        assert predict_accept(["What is RAG?"], []) is None

    def test_only_accepted_labels(self):
        """With no rejected examples everything looks acceptable — the number would be meaningless."""
        assert predict_accept(["What is RAG?"], GOOD) is None

    def test_only_rejected_labels(self):
        assert predict_accept(["What is RAG?"], BAD) is None

    def test_labels_without_question_text(self):
        junk = [{"question": "", "decision": "good"}, {"question": None, "decision": "bad"}]
        assert predict_accept(["What is RAG?"], junk) is None


class TestSplitLabels:
    def test_split_is_deterministic(self):
        a1, b1 = split_labels(GOOD + BAD)
        a2, b2 = split_labels(GOOD + BAD)
        assert [e["question"] for e in a1] == [e["question"] for e in a2]
        assert [e["question"] for e in b1] == [e["question"] for e in b2]

    def test_no_question_appears_on_both_sides(self):
        inform, holdout = split_labels(GOOD + BAD)
        assert not ({e["question"] for e in inform} & {e["question"] for e in holdout})

    def test_split_loses_nothing(self):
        inform, holdout = split_labels(GOOD + BAD)
        assert len(inform) + len(holdout) == 4

    def test_holdout_zero_keeps_everything_for_informing(self):
        inform, holdout = split_labels(GOOD + BAD, holdout_fraction=0.0)
        assert len(holdout) == 0 and len(inform) == 4

    def test_holdout_one_holds_everything_out(self):
        inform, holdout = split_labels(GOOD + BAD, holdout_fraction=1.0)
        assert len(inform) == 0 and len(holdout) == 4

    def test_drops_labels_with_no_question_text(self):
        inform, holdout = split_labels([{"question": "", "decision": "good"}])
        assert not inform and not holdout


@needs_embeddings
class TestPredictions:
    def test_reports_label_counts_it_used(self):
        r = predict_accept(["What is RAG?"], GOOD + BAD)
        assert r.n_good_labels == 2 and r.n_bad_labels == 2 and r.label_count == 4

    def test_scores_every_question(self):
        qs = ["What is RAG?", "What are embeddings?"]
        r = predict_accept(qs, GOOD + BAD)
        assert r.n_scored == 2 and set(r.per_question) == set(qs)

    def test_rate_is_a_fraction(self):
        r = predict_accept(["What is RAG?", "Tell me about yourself"], GOOD + BAD)
        assert 0.0 <= r.predicted_accept_rate <= 1.0

    def test_exact_repeat_of_a_rejected_question_is_predicted_reject(self):
        """An identical rejected question must never be predicted acceptable — this is the case the
        old exact-string check covered, and it must keep working."""
        repeat = BAD[0]["question"]
        r = predict_accept([repeat], GOOD + BAD)
        assert r.per_question[repeat] is False
        assert r.repeats_rejected == 1

    def test_question_like_an_approved_one_is_predicted_accept(self):
        r = predict_accept(["What are the main components of an AI agent?"], GOOD + BAD)
        assert r.per_question["What are the main components of an AI agent?"] is True

    def test_session_scoped_labels_ignored_when_too_sparse(self):
        """Fewer than 3 of either side for the session → fall back to all labels rather than
        producing a confident number from two examples."""
        r = predict_accept(["What is RAG?"], GOOD + BAD, session="S1")
        assert r is not None and r.label_count == 4

    def test_summary_is_readable(self):
        assert "predicted_accept=" in predict_accept(["What is RAG?"], GOOD + BAD).summary()


def test_thresholds_are_sane():
    assert 0.0 < DECISION_MARGIN < 0.2
    assert 0.5 < NEAR_DUPLICATE_REJECT <= 1.0
