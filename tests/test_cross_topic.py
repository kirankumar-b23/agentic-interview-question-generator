"""One question, one topic — homed by curriculum order.

**16 of 149 distinct questions (11%) sat in more than one topic**, five in three. Not shared content: no
session belongs to two topics (0 of 56). Every de-duplication mechanism was scoped to a single topic, so a
student sitting several of these mock interviews would be asked *"What is a prompt?"* three times.

THE RULE IS NOT "BEST FIT", AND NOT "LATEST TOPIC"
--------------------------------------------------
A question belongs to the EARLIEST topic whose material actually covers it: if the definition is taught
earlier and this topic elaborates, a question covering BOTH belongs here and cannot be asked earlier; a bare
definition belongs where it is first taught, not in every later topic that mentions it.

Both alternatives were measured on the real 16 and both are wrong:

    latest wins  ->  "What is a prompt?" goes to No-Code AI Automation
    best fit     ->  "How do you approach designing an effective prompt?" goes to AI Workflows (0.742)
                     instead of Prompt Engineering Fundamentals (0.729), where prompts are TAUGHT

`TestTheRuleMovesQuestionsBothWays` is the load-bearing test: a rule that only ever picks the earliest, or
only the best fit, fails one of its two cases.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


@pytest.fixture
def clear_caches():
    """`session_positions`, `_course_structure` and `topic_position` are lru_cached, so a test that
    monkeypatches the data underneath them sees a stale value — and worse, LEAKS it into later tests.

    Resolved by NAME on each pass and skipped when the attribute has no `cache_clear`: fixture teardown
    runs before `monkeypatch` undoes its patches, so at teardown one of these may still be a plain stub.
    """
    import src.curriculum_order as co

    def _clear():
        for name in ("session_positions", "_course_structure", "topic_position"):
            fn = getattr(co, name, None)
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()

    _clear()
    yield
    _clear()

# Verbatim from the shipped sets, with the real per-topic grounding fits.
PROMPT_Q = "What is a prompt?"
DESIGN_Q = "How do you approach designing an effective prompt?"
HTTP_Q = "What is the HTTP Request node and when do you use it?"
ADOPTED_Q = "How have you adopted AI in your workflows? Walk through examples."

PROMPT_ENG = "Productivity Power-Up with AI Tools & Prompt Engineering Fundamentals"
NO_CODE = "No-Code AI Automation"
AI_WORKFLOWS = "AI Workflows for Enhanced Productivity"
IMAGE_GEN = "Mastering Image Generation"


class TestTheOrderComesFromTheGraph:
    """`course_structure.json`'s key order happens to agree today, which is the corroboration — but it is
    an accident of insertion that `AddCourse` can change, so `session_order_edges` is the source."""

    def test_every_shipped_topic_resolves(self):
        import json

        from src.config import DATA_DIR
        from src.curriculum_order import topic_position

        structure = json.loads((DATA_DIR / "course_structure.json").read_text())
        missing = [t for t in structure if topic_position(t) is None]
        assert missing == [], f"no position for {missing}"

    def test_the_sequence_increases_along_a_known_chain(self):
        from src.curriculum_order import topic_position

        chain = ["Course Overview", "Gen AI Foundations & Capabilities", PROMPT_ENG, NO_CODE,
                 AI_WORKFLOWS, IMAGE_GEN]
        positions = [topic_position(t) for t in chain]
        assert positions == sorted(positions), f"curriculum order is not monotonic: {positions}"
        assert len(set(positions)) == len(positions), "two topics share a position"

    def test_the_order_covers_every_session_in_the_graph(self):
        import json

        from src.config import DATA_DIR
        from src.curriculum_order import session_positions

        kg = json.loads((DATA_DIR / "knowledge_graph.json").read_text())
        nodes = {e["from"] for e in kg["session_order_edges"]} | {e["to"] for e in kg["session_order_edges"]}
        assert set(session_positions()) == nodes, "a session in the graph got no position"

    def test_a_topic_survives_a_session_missing_from_the_graph(self, monkeypatch, clear_caches):
        """A sentinel position poisons the whole topic through the `max`. The first version scored four
        topics at 999 because one of their sessions was absent, silently breaking every comparison."""
        import src.curriculum_order as co

        monkeypatch.setattr(co, "_course_structure",
                            lambda: {"T": ["Advanced Prompt Engineering", "Not In The Graph At All"]})
        pos = co.topic_position("T")
        assert pos is not None and pos < 100, f"an absent session poisoned the position: {pos}"

    def test_no_session_in_the_graph_gives_None_not_zero(self, monkeypatch, clear_caches):
        """None, so callers can sort it LAST. Position 0 would make a custom topic look like the very
        first thing in the course and win every home decision."""
        import src.curriculum_order as co

        monkeypatch.setattr(co, "_course_structure", lambda: {"T": ["Nowhere At All"]})
        assert co.topic_position("T") is None

    def test_a_cycle_degrades_instead_of_hanging(self, tmp_path, monkeypatch, clear_caches):
        """Kahn's algorithm, not a DFS: a malformed graph must not recurse forever or lose nodes."""
        import json

        import src.config as cfg
        import src.curriculum_order as co

        (tmp_path / "knowledge_graph.json").write_text(json.dumps(
            {"session_order_edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "A"},
                                     {"from": "root", "to": "A"}]}))
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
        pos = co.session_positions()
        assert set(pos) == {"root", "A", "B"}, f"a cycle lost nodes: {pos}"
        assert pos["root"] == 0

    def test_a_missing_graph_is_not_an_error(self, tmp_path, monkeypatch, clear_caches):
        import src.config as cfg
        import src.curriculum_order as co

        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)      # no knowledge_graph.json here
        assert co.session_positions() == {}


class TestTheRuleMovesQuestionsBothWays:
    """The load-bearing test. Real strings, real fits."""

    def test_a_fundamental_stays_where_it_is_taught(self):
        """Prompt Engineering (pos 3) fits 0.665; No-Code (5) 0.540; Image Generation (11) 0.491."""
        from src.curriculum_order import home_topic
        assert home_topic({NO_CODE: 0.540, PROMPT_ENG: 0.665, IMAGE_GEN: 0.491}) == PROMPT_ENG

    def test_it_beats_best_fit_when_the_earlier_topic_covers_it(self):
        """AI Workflows fits HIGHER (0.742) than Prompt Engineering (0.729), and is still wrong: prompts
        are taught at pos 3, so a student there can already answer it. `best fit` fails this case."""
        from src.curriculum_order import home_topic
        assert home_topic({PROMPT_ENG: 0.729, AI_WORKFLOWS: 0.742}) == PROMPT_ENG

    def test_it_moves_LATER_when_the_earlier_topic_does_not_cover_it(self):
        """The reviewer's actual rule. No-Code (0.540) does not cover the HTTP Request node; AI Workflows
        (0.703) does. `earliest always wins` fails this case."""
        from src.curriculum_order import home_topic
        assert home_topic({NO_CODE: 0.540, AI_WORKFLOWS: 0.703}) == AI_WORKFLOWS

    def test_another_later_case_from_the_real_data(self):
        from src.curriculum_order import home_topic
        assert home_topic({PROMPT_ENG: 0.583, NO_CODE: 0.701}) == NO_CODE

    def test_the_ratio_is_what_decides_the_direction(self):
        """Self-demonstrating: at ratio 1.0 the rule degenerates to best-fit and gets the design case
        wrong. That is the mutation this guards."""
        from src.curriculum_order import home_topic
        assert home_topic({PROMPT_ENG: 0.729, AI_WORKFLOWS: 0.742}, ratio=1.0) == AI_WORKFLOWS

    def test_a_topic_with_no_position_never_wins_over_one_with(self):
        from src.curriculum_order import home_topic
        assert home_topic({"A Totally Custom Topic": 0.9, PROMPT_ENG: 0.85}) == PROMPT_ENG

    def test_no_grounding_signal_falls_back_to_curriculum_order(self):
        from src.curriculum_order import home_topic
        assert home_topic({AI_WORKFLOWS: 0.0, PROMPT_ENG: 0.0}) == PROMPT_ENG

    def test_empty_is_none(self):
        from src.curriculum_order import home_topic
        assert home_topic({}) is None


def _q(content, qid=None):
    return QuestionDetail(question_id=qid or content[:20], category="GEN_AI", content=content,
                          topic="t", difficulty="Medium", source="interview_db")


class TestSuppressionIsWiredIntoThePipeline:
    """`_drop_hands_on`, `_add_retained` and `_cap_by_outcome` each shipped with a vacuous direct-call test
    first: `_select_final` trims anyway, so "it isn't in the shipped set" passes with the call unwired.
    These go through `_pick_questions` and assert the removal RECORD."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
        memory.init_db()
        return memory

    @staticmethod
    def _state(questions, session="Advanced Prompt Engineering"):
        from src.agent import AgentState
        from src.data_loader import get_data_store

        st = AgentState(config=GenerationConfig(session_names=[session]), data_store=get_data_store())
        st.session_context = SimpleNamespace(
            session_name=session, session_type="mixed", learning_outcomes=["Understand prompting"],
            interview_topics=["Prompt engineering"], key_concepts=["prompt"], scope_in=[], scope_out=[],
            matched_kp_ids=[])
        st.questions = {q.question_id: q for q in questions}
        return st

    def test_a_question_owned_by_another_topic_is_dropped_and_the_owner_named(self, db):
        from src.pipeline import AgentPipeline

        db.upsert_topic_questions(IMAGE_GEN, [{"content": PROMPT_Q, "status": "approved"}])
        qs = [_q(PROMPT_Q, qid="dup"), _q("What is a system prompt used for?", qid="keep")]
        st = self._state(qs)
        AgentPipeline()._drop_used_by_other_topics(st, lambda *a, **k: None)

        assert set(st.questions) == {"keep"}
        rec = [r for r in st.removed if r["stage"] == "other_topic"]
        assert len(rec) == 1
        assert IMAGE_GEN in rec[0]["reason"], (
            "the owner must be NAMED, or a reviewer cannot tell a correct suppression from a mis-home")

    def test_this_topics_own_questions_are_never_suppressed(self, db):
        """It only sees freshly-retrieved candidates; `_add_retained` runs later, inside submit."""
        from src.pipeline import AgentPipeline

        tk = db.topic_key_for("Advanced Prompt Engineering")
        db.upsert_topic_questions(tk, [{"content": PROMPT_Q, "status": "approved"}])
        qs = [_q(PROMPT_Q, qid="own")]
        st = self._state(qs)
        AgentPipeline()._drop_used_by_other_topics(st, lambda *a, **k: None)
        assert set(st.questions) == {"own"}, "a question this topic owns must survive"

    def test_it_runs_from_pick_questions(self, db, monkeypatch):
        """Mutation target: unwiring the call from `_pick_questions`."""
        import src.pipeline as pl

        db.upsert_topic_questions(IMAGE_GEN, [{"content": PROMPT_Q, "status": "approved"}])
        seen = {}
        real = pl.AgentPipeline._drop_used_by_other_topics

        def spy(self, state, emit):
            seen["called"] = True
            return real(self, state, emit)

        monkeypatch.setattr(pl.AgentPipeline, "_drop_used_by_other_topics", spy)
        # Stop right after the filter — everything downstream needs an LLM.
        monkeypatch.setattr(pl.UnderstandingAgent, "run", lambda self, s, e: None)
        monkeypatch.setattr(pl.RetrievalAgent, "run", lambda self, s, e: None)
        monkeypatch.setattr(pl.AgentPipeline, "_score_session_fit",
                            lambda self, s, e, only_ids=None: (_ for _ in ()).throw(_Stop()))
        monkeypatch.setattr(pl.AgentPipeline, "_tavily_preflight", lambda self, s, e: None)

        st = self._state([_q(PROMPT_Q, qid="dup")])
        with pytest.raises(_Stop):
            pl.AgentPipeline()._pick_questions(st, lambda *a, **k: None)
        assert seen.get("called"), "the filter is not wired into _pick_questions"
        assert [r for r in st.removed if r["stage"] == "other_topic"]

    def test_it_runs_before_the_relevance_judge(self, db, monkeypatch):
        """Absence at the end proves nothing — `_select_final` trims anyway. What matters is that no
        embedding or LLM stage is reached for a candidate that cannot ship."""
        import src.pipeline as pl

        db.upsert_topic_questions(IMAGE_GEN, [{"content": PROMPT_Q, "status": "approved"}])
        order = []
        monkeypatch.setattr(pl.AgentPipeline, "_tavily_preflight", lambda self, s, e: None)
        monkeypatch.setattr(pl.UnderstandingAgent, "run", lambda self, s, e: None)
        monkeypatch.setattr(pl.RetrievalAgent, "run", lambda self, s, e: None)
        real = pl.AgentPipeline._drop_used_by_other_topics
        monkeypatch.setattr(pl.AgentPipeline, "_drop_used_by_other_topics",
                            lambda self, s, e: (order.append("suppress"), real(self, s, e))[1])
        monkeypatch.setattr(pl.AgentPipeline, "_score_session_fit",
                            lambda self, s, e, only_ids=None: (order.append("fit"),
                                                               (_ for _ in ()).throw(_Stop()))[0])

        st = self._state([_q(PROMPT_Q, qid="dup")])
        with pytest.raises(_Stop):
            pl.AgentPipeline()._pick_questions(st, lambda *a, **k: None)
        assert order == ["suppress", "fit"], f"wrong order: {order}"

    def test_no_other_topics_is_not_an_error(self, db):
        from src.pipeline import AgentPipeline
        st = self._state([_q(PROMPT_Q, qid="x")])
        AgentPipeline()._drop_used_by_other_topics(st, lambda *a, **k: None)
        assert set(st.questions) == {"x"}


class _Stop(Exception):
    """Sentinel to halt `_pick_questions` after the stage under test."""


class TestTheReportSaysWhatWasWithheld:
    def test_the_note_names_the_owner_and_the_way_to_re_home(self):
        """A question silently missing because another topic owns it is indistinguishable from one that
        was never found — and the accepted risk here is a shared fundamental locked to one topic."""
        from src.agent import AgentState
        from src.data_loader import get_data_store
        from src.pipeline import _build_quality_report

        st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
        st.session_context = SimpleNamespace(
            session_name="S", session_type="mixed", learning_outcomes=[], interview_topics=["T"],
            key_concepts=[], scope_in=[], scope_out=[], matched_kp_ids=[])
        st.questions = {q.question_id: q for q in [_q("What is X?"), _q("What is Y?")]}
        st.removed = [{"content": PROMPT_Q, "stage": "other_topic",
                       "reason": f"Already used in another topic: {IMAGE_GEN}"}]
        notes = " ".join(_build_quality_report(st, 0).critique)
        assert "withheld" in notes and IMAGE_GEN in notes
        assert "--cross-topic" in notes, "the reviewer needs to know how to re-home it"


class TestTheBankIsKeyedOnTheTopic:
    """Every stored `question_bank` key is a JOINED session name, and two are different groupings of the
    same sessions — so an exact-string match meant a differently-grouped run saw none of its own topic's
    banked questions and cross-run dedup silently did nothing."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
        memory.init_db()
        return memory

    def test_a_single_session_query_finds_a_joined_banked_question(self, db):
        joined = ("Building Social Media Content Automation Workflow | Part 1 + "
                  "Advanced Prompt Engineering")
        db.save_question_to_bank("q1", joined, PROMPT_Q, "interview_db")
        found = [b["content"] for b in db.get_bank_questions("Advanced Prompt Engineering")]
        assert PROMPT_Q in found, "the exact-string match is what this fixes"

    def test_a_different_grouping_of_the_same_topic_also_finds_it(self, db):
        db.save_question_to_bank("q1", "Advanced Prompt Engineering", PROMPT_Q, "interview_db")
        other = "Building Social Media Content Automation Workflow | Part 2"
        assert PROMPT_Q in [b["content"] for b in db.get_bank_questions(other)]

    def test_another_topic_does_not_see_it(self, db):
        db.save_question_to_bank("q1", "Advanced Prompt Engineering", PROMPT_Q, "interview_db")
        assert db.get_bank_questions("Generative AI Foundations") == [], (
            "the fix must widen the match to the TOPIC, not to everything")
