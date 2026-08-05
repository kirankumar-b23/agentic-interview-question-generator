"""Tests for the reword penalty used in final selection (`tools._feedback_penalty`).

This function went through two wrong versions before the current one, and both failures are pinned
here because they are easy to reintroduce:

  1. Absolute similarity to the rejected set. Within one domain everything is similar to everything,
     so a reviewer-APPROVED question scored a 0.73 penalty and was demoted almost as hard as an
     actual reworded rejection.
  2. Relative margin with no floor. That fixed approved questions but broke UNLABELLED topics: with
     a label set about agents and prompting, a diffusion question scored a LARGER penalty than a real
     reworded rejection, because nothing in the accepted set resembled it either.

The penalty must fire on rewordings and on nothing else.
"""
import pytest

from src import embeddings
from src.tools import REWORD_FLOOR, _feedback_penalty

REJECTED = "Describe the architecture of an AI agent workflow you have built."
APPROVED = "What are the core components of an AI Agent?"

LABELS = [
    {"question": APPROVED, "decision": "good", "session": "S"},
    {"question": "Why is memory critical for AI agents?", "decision": "good", "session": "S"},
    {"question": "What is prompt engineering?", "decision": "good", "session": "S"},
    {"question": REJECTED, "decision": "bad", "session": "S"},
    {"question": "How have you adopted AI in your workflows? Walk through examples.",
     "decision": "bad", "session": "S"},
]

needs_embeddings = pytest.mark.skipif(not embeddings.available(),
                                      reason="sentence-transformers unavailable")


@pytest.fixture
def labels(monkeypatch):
    """Use a fixed label set rather than whatever happens to be in eval/feedback_examples.json."""
    from src import memory
    monkeypatch.setattr(memory, "get_feedback_examples", lambda: LABELS)


class TestDegradesSafely:
    def test_empty_input(self):
        assert _feedback_penalty([]) == []

    def test_no_labels_means_no_penalty(self, monkeypatch):
        from src import memory
        monkeypatch.setattr(memory, "get_feedback_examples", lambda: [])
        assert _feedback_penalty([REJECTED]) == [0.0]

    def test_one_sided_labels_mean_no_penalty(self, monkeypatch):
        """Without accepted examples the shared-baseline subtraction is impossible, so the score
        would punish every candidate roughly equally — worse than no signal."""
        from src import memory
        monkeypatch.setattr(memory, "get_feedback_examples",
                            lambda: [e for e in LABELS if e["decision"] == "bad"])
        assert _feedback_penalty([REJECTED]) == [0.0]

    def test_returns_one_score_per_input(self, labels):
        assert len(_feedback_penalty(["a", "b", "c"])) == 3

    def test_scores_are_never_negative(self, labels):
        assert all(p >= 0.0 for p in _feedback_penalty([APPROVED, REJECTED, "What is RAG?"]))


@needs_embeddings
class TestPenaltyTargetsOnlyRewordings:
    def test_exact_rejected_question_is_penalised(self, labels):
        assert _feedback_penalty([REJECTED])[0] > 0.0

    def test_reworded_rejection_is_penalised(self, labels):
        """The whole point: `_drop_rejected` catches exact strings, this catches rewordings."""
        reworded = "Can you describe an AI agent workflow architecture that you built yourself?"
        assert _feedback_penalty([reworded])[0] > 0.0

    def test_approved_question_is_not_penalised(self, labels):
        """Regression guard for failure mode 1."""
        assert _feedback_penalty([APPROVED])[0] == 0.0

    def test_other_approved_question_is_not_penalised(self, labels):
        assert _feedback_penalty(["What is prompt engineering?"])[0] == 0.0

    @pytest.mark.parametrize("question", [
        "How does a diffusion model add noise during training?",
        "What is a negative prompt in Stable Diffusion?",
        "How do you configure an n8n webhook trigger?",
    ])
    def test_unlabelled_topics_are_not_penalised(self, labels, question):
        """Regression guard for failure mode 2 — a topic the reviewer has never judged must not be
        demoted just because the accepted set does not happen to cover it."""
        assert _feedback_penalty([question])[0] == 0.0

    def test_rewording_outscores_an_approved_question(self, labels):
        reworded, approved = _feedback_penalty([REJECTED, APPROVED])
        assert reworded > approved


def test_reword_floor_separates_the_two_cases():
    """The floor sits between same-domain similarity (~0.73 measured) and a true reword (~1.0)."""
    assert 0.7 < REWORD_FLOOR < 0.95
