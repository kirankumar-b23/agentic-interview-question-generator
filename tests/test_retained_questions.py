"""A re-run ADDS to a topic's set instead of shipping another version.

The accumulation already existed in the database — approved questions bank on approve, and
`scripts/consolidate_topic_sets.py` merged the historical runs in — but nothing put the set back into a
run's output, so a re-run shipped only the delta and looked like a fresh, smaller set. `_add_retained`
closes that.

What these pin:

* the split (carried over vs newly found) and that duplicates are not doubled;
* that a carried-over question today's gates reject is FLAGGED, not dropped — a reviewer approved it;
* that retained questions get a `session_fit`, which is the trap the open-web tier already hit once:
  `grounding_score` averages only non-None fits, so leaving them unscored makes `session_grounding`
  silently describe just the freshly-found subset, and `_rank_key` reads None as 0.0.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src import memory
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
    memory.init_db()
    return memory


def _q(content, qid=None):
    return QuestionDetail(question_id=qid or content[:20], category="GEN_AI", content=content,
                          topic="t", difficulty="Medium", source="interview_db")


def _state(questions, session="Introduction to AI Agents"):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=[session]), data_store=get_data_store())
    st.session_context = SimpleNamespace(
        session_name=session, session_type="mixed", learning_outcomes=["Understand agents"],
        interview_topics=["AI agent architecture"], key_concepts=["agents"], scope_in=[], scope_out=[],
        matched_kp_ids=[])
    st.questions = {q.question_id: q for q in questions}
    return st


class TestRetainedAreCarriedIn:
    def test_the_existing_set_is_added_to_this_runs_picks(self, db):
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        # Real question shapes: "Kept question one" is not a question, so the gate would rightly flag
        # it stale and this test would be measuring my fixture rather than the merge.
        db.upsert_topic_questions(tk, [{"content": "What is agent memory?", "status": "approved"},
                                       {"content": "How does agent planning work?", "status": "backfilled"}])
        fresh = [_q("What is the ReAct pattern?")]
        st = _state(fresh)

        out = _add_retained(st, fresh)
        assert out == {"retained": 2, "newly_found": 1, "stale": 0}
        assert len(st.questions) == 3
        contents = {q.content for q in st.questions.values()}
        assert contents == {"What is the ReAct pattern?", "What is agent memory?",
                            "How does agent planning work?"}

    def test_a_question_this_run_also_found_is_not_doubled(self, db):
        """Cross-run dedup already removes candidates duplicating the set; this must not undo that."""
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{"content": "What is an AI agent?"}])
        fresh = [_q("what is an ai agent")]          # same content, different punctuation/case
        st = _state(fresh)

        out = _add_retained(st, fresh)
        assert out["retained"] == 0
        assert len(st.questions) == 1, "identity is normalized content, so this is the same question"

    def test_carried_over_questions_are_marked_with_their_status(self, db):
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{"content": "What is agent memory?", "status": "approved"},
                                       {"content": "How does agent planning work?", "status": "backfilled"}])
        fresh = []
        st = _state(fresh)
        _add_retained(st, fresh)

        by = {q.content: q for q in st.questions.values()}
        assert by["What is agent memory?"].retained is True
        assert by["What is agent memory?"].retained_status == "approved"
        assert by["How does agent planning work?"].retained_status == "backfilled", (
            "175 of the consolidated 251 were never approved — that must stay visible")

    def test_an_empty_set_leaves_the_run_untouched(self, db):
        from src.tools import _add_retained

        fresh = [_q("What is an AI agent?")]
        st = _state(fresh)
        assert _add_retained(st, fresh) == {"retained": 0, "newly_found": 1, "stale": 0}
        assert len(st.questions) == 1

    def test_the_full_original_detail_is_preserved(self, db):
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{
            "content": "How do you give an agent memory?", "company": "ACME", "difficulty": "Hard",
            "detail": {"content": "How do you give an agent memory?", "role": "Prompt Engineer",
                       "asked_in_company": "ACME", "difficulty": "Hard"}}])
        fresh = []
        st = _state(fresh)
        _add_retained(st, fresh)
        q = next(iter(st.questions.values()))
        assert q.role == "Prompt Engineer" and q.asked_in_company == "ACME" and q.difficulty == "Hard"


class TestStaleIsFlaggedNotDropped:
    @pytest.mark.parametrize("content,why", [
        ("how you would iteratively improve prompts and guards to increase reliability.",
         "sentence fragment — the exact defect CLAUDE.md records as shipped"),
        ("Write a Python program to reverse a linked list.", "hands-on task"),
    ])
    def test_it_still_ships_but_carries_a_reason(self, db, content, why):
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{"content": content, "status": "approved"}])
        fresh = []
        st = _state(fresh)

        out = _add_retained(st, fresh)
        assert out["retained"] == 1 and out["stale"] == 1
        q = next(iter(st.questions.values()))
        assert q.stale_reason, f"should be flagged ({why})"
        assert q.content == content, "a previously approved question must not be silently removed"

    def test_a_still_good_question_is_not_flagged(self, db):
        from src.tools import _add_retained

        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{"content": "What is the ReAct pattern in AI agents?"}])
        fresh = []
        st = _state(fresh)
        _add_retained(st, fresh)
        assert next(iter(st.questions.values())).stale_reason is None


class TestRetainedGetAFit:
    def test_every_shipped_question_ends_up_scored(self, db, monkeypatch):
        """The open-web trap, again: unscored questions vanish from `session_grounding` (it averages
        non-None only) and sink in Review because `_rank_key` reads None as 0.0."""
        import src.pipeline as pl
        from src import embeddings

        monkeypatch.setattr(pl, "_session_profile", lambda names, ctx: (["AI agents and memory"], []))
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, profile=None: [[0.55] for _ in texts])

        scored = _q("What is agent planning?")
        scored.session_fit = 0.7
        unscored = _q("What is agent memory?")
        unscored.retained = True
        st = _state([scored, unscored])

        pl.AgentPipeline()._score_unscored_fits(st, lambda *a, **k: None)
        assert st.questions[unscored.question_id].session_fit == 0.55
        assert st.questions[scored.question_id].session_fit == 0.7, "must not rescore what was scored"

    def test_it_drops_nothing(self, db, monkeypatch):
        """Unlike `_score_session_fit(only_ids=…)`, which applies the relative floor. A retained question
        is settled: it gets flagged, never removed."""
        import src.pipeline as pl
        from src import embeddings

        monkeypatch.setattr(pl, "_session_profile", lambda names, ctx: (["AI agents"], []))
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, profile=None: [[0.01] for _ in texts])   # far below any floor
        weak = _q("What is an unrelated topic entirely?")
        weak.retained = True
        st = _state([weak])

        pl.AgentPipeline()._score_unscored_fits(st, lambda *a, **k: None)
        assert len(st.questions) == 1, "a below-floor retained question must survive"
        assert st.questions[weak.question_id].session_fit == 0.01


class TestTheReportNamesTheSplit:
    def test_a_zero_new_rerun_is_visibly_zero_new(self, db):
        """Otherwise a re-run that found nothing reads as a healthy 40-question run."""
        from src.pipeline import _build_quality_report

        kept = _q("What is agent memory?")
        kept.retained = True
        kept.retained_status = "approved"
        kept.session_fit = 0.6
        st = _state([kept])
        st.relevance_scored = True

        report = _build_quality_report(st, 0)
        joined = " ".join(report.critique)
        assert "carried over" in joined
        assert "No NEW questions" in joined

    def test_the_split_is_reported_when_both_are_present(self, db):
        from src.pipeline import _build_quality_report

        kept = _q("What is agent memory?")
        kept.retained = True
        kept.retained_status = "backfilled"
        kept.session_fit = 0.6
        new = _q("What is the ReAct pattern?")
        new.session_fit = 0.6
        st = _state([kept, new])
        st.relevance_scored = True

        joined = " ".join(_build_quality_report(st, 0).critique)
        assert "1 carried over" in joined and "1 newly found" in joined
        assert "never reviewer-approved" in joined
        assert "No NEW questions" not in joined


class TestItIsActuallyWiredIn:
    """The tests above call `_add_retained` directly, so they prove it WORKS, not that it RUNS.

    A mutation check caught that: unwiring the call from `tool_submit_question_set` left all of them
    green. Same class of vacuous test as the conversational filter's first attempt. These go through
    submit, which is the only thing that proves the feature is reachable.
    """

    def test_submit_ships_the_topic_set_alongside_this_runs_picks(self, db, monkeypatch):
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})
        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [{"content": "What is agent memory?", "status": "approved"}])

        fresh = [_q(f"What is agent concept number {i}?", qid=f"q{i}") for i in range(6)]
        st = _state(fresh)
        st.config.max_questions = 5

        out = tool_submit_question_set(st)
        assert out["retained"] == 1, "submit must carry the topic's set in"
        assert out["newly_found"] >= 1
        assert any(q.retained for q in st.questions.values())
        assert "What is agent memory?" in {q.content for q in st.questions.values()}

    def test_submit_reports_the_split_in_its_result(self, db, monkeypatch):
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})
        tk = db.topic_key_for("Introduction to AI Agents")
        db.upsert_topic_questions(tk, [
            {"content": "What is agent memory?"}, {"content": "How does planning work?"}])
        fresh = [_q("What is the ReAct pattern?")]
        st = _state(fresh)

        out = tool_submit_question_set(st)
        assert out["retained"] == 2 and out["newly_found"] == 1
        assert "retained_stale" in out
