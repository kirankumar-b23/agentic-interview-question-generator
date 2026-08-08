"""The relevance back-fill targets the REQUESTED count, not the absolute minimum.

`tool_validate_relevance` keeps everything at/above `RELEVANCE_THRESHOLD` (0.50), then tops up from
`RELEVANCE_FLOOR` (0.35) upward. That top-up used to stop at `min_questions` (5), so a run asking for 15
was filled to 5 and stopped — the same off-by-one as the old open-web trigger, in the stage that cuts a
mean 88% of everything reaching it.

Replayed across 13 persisted runs using their stored scores: median shipped-equivalent 6 -> 14, 11 of 13
gaining. The admitted [0.35, 0.50) band is grounded as well as what already ships (mean `session_fit`
0.610 vs 0.625) and clearly better than sub-floor material (0.561).

The FLOOR is the invariant these tests protect: a thin on-topic pool must return FEWER questions, never
loosely-related filler.
"""
import json
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


def _q(i):
    return QuestionDetail(question_id=f"q{i}", category="GEN_AI", content=f"What is concept {i}?",
                          topic="Gen AI", difficulty="Medium", source="interview_db")


def _state(n, max_questions=15, min_questions=5):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=max_questions,
                                            min_questions=min_questions),
                    data_store=get_data_store())
    st.session_context = SimpleNamespace(
        session_name="S", session_type="theory_heavy", learning_outcomes=["Understand concepts"],
        interview_topics=[], key_concepts=["concepts"], scope_in=[], scope_out=[], matched_kp_ids=[])
    st.questions = {q.question_id: q for q in (_q(i) for i in range(n))}
    return st


def _stub_scores(monkeypatch, score_for):
    """Score each question via `score_for(question_index)`, mirroring the real reply shape.

    Keyed on the question TEXT, not on `n`: candidates are sent in batches of `RELEVANCE_BATCH_SIZE`
    and `n` restarts at 1 in every batch, so scoring by `n` silently gave the first few of EVERY batch
    the same score and the assertions measured the wrong thing.
    """
    import re

    import src.tools as tools_mod

    def fake(**kw):
        payload = kw.get("user_prompt", "")
        items = json.loads(payload[payload.find("["):]) if "[" in payload else []
        out = []
        for it in items:
            m = re.search(r"concept (\d+)", it.get("q", ""))
            idx = int(m.group(1)) if m else 0
            out.append({"n": it["n"], "score": score_for(idx), "difficulty": "Medium"})
        return {"scores": out}

    monkeypatch.setattr(tools_mod, "chat_completion_json", fake)


class TestBackfillTargetsTheAsk:
    def test_it_fills_toward_the_requested_count_not_the_minimum(self, monkeypatch):
        """3 strong + plenty in the band. Old behaviour stopped at 5; the ask is 15."""
        from src.tools import tool_validate_relevance

        st = _state(30, max_questions=15, min_questions=5)
        _stub_scores(monkeypatch, lambda i: 0.90 if i < 3 else 0.42)   # 0.42 is inside [0.35, 0.50)

        tool_validate_relevance(st)
        assert len(st.questions) == 15, "should reach the requested count, not stop at min_questions"

    def test_nothing_below_the_floor_is_ever_admitted(self, monkeypatch):
        """The invariant. Supply is short of the ask, and the rest is sub-floor filler."""
        from src.config import RELEVANCE_FLOOR
        from src.tools import tool_validate_relevance

        st = _state(30, max_questions=15, min_questions=5)
        _stub_scores(monkeypatch, lambda i: 0.90 if i < 4 else 0.20)   # everything else below 0.35

        tool_validate_relevance(st)
        assert len(st.questions) == 4, "a thin pool must return fewer, not loosely-related filler"
        assert all(q.relevance_score >= RELEVANCE_FLOOR for q in st.questions.values())

    def test_a_strong_pool_is_not_trimmed_here(self, monkeypatch):
        """This stage filters; it must not also cap. Selection does the trim to the requested count."""
        from src.tools import tool_validate_relevance

        st = _state(30, max_questions=15, min_questions=5)
        _stub_scores(monkeypatch, lambda i: 0.85)

        tool_validate_relevance(st)
        assert len(st.questions) == 30, "everything above THRESHOLD survives; trimming happens later"

    def test_the_floor_still_wins_when_the_ask_is_huge(self, monkeypatch):
        from src.tools import tool_validate_relevance

        st = _state(20, max_questions=60, min_questions=5)
        _stub_scores(monkeypatch, lambda i: 0.55 if i < 2 else (0.40 if i < 7 else 0.10))

        tool_validate_relevance(st)
        assert len(st.questions) == 7, "2 above threshold + 5 in the band, and nothing below the floor"

    @pytest.mark.parametrize("ask,expected", [(5, 5), (8, 8), (15, 15)])
    def test_the_target_tracks_the_ask(self, monkeypatch, ask, expected):
        from src.tools import tool_validate_relevance

        st = _state(30, max_questions=ask, min_questions=5)
        _stub_scores(monkeypatch, lambda i: 0.90 if i < 2 else 0.45)

        tool_validate_relevance(st)
        assert len(st.questions) == expected
