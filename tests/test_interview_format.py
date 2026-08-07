"""Hands-on task prompts are unanswerable in a conversational interview.

Students answer out loud — no keyboard, no IDE, no whiteboard. Real runs shipped prompts that cannot be
answered at all: "Implement an input box to interact with the Gemini API…" and "Build and integrate LLM
applications." each took a slot in a set of 9.

The tests here pin the DISCRIMINATOR, not a verb list, because the two failure modes are opposite and a
naive rule hits one or the other:

  * too broad — treating "design" as hands-on. Measured, that pushed 2 of the last 6 runs UNDER the
    5-question minimum and dropped "Design an RSS News Feed Service", one of only three tool-specific
    questions the n8n retrieval work recovered.
  * too narrow — matching only a bare sentence-initial verb, which misses "Can you create a DataFrame…?"
    and "Design and implement an API".

The three pairs in KEEP/SKIP below are the ones that separate a real rule from a keyword match.
"""
from types import SimpleNamespace

import pytest

from src.interview_format import is_hands_on_task
from src.models import GenerationConfig, QuestionDetail


class TestTheDiscriminator:
    # Every string here is verbatim from data/interview_questions.json or a shipped question set.
    SKIP = [
        ("Write a Python program to generate the Fibonacci series up to a given number.",
         "the archetype — needs a keyboard"),
        ("Implement an input box to interact with the Gemini API. Send user input to the specified API.",
         "shipped in run 5807bb2b"),
        ("Build and integrate LLM applications.", "shipped in run 5807bb2b"),
        ("Design and implement an API for a given task.",
         "reads like discussion, demands an artifact — no bare-verb rule catches this"),
        ("Create an input box in a frontend application that allows users to send text messages.",
         "'create' + a UI artifact"),
        ("Can you create a DataFrame in Python with employee details?",
         "the polite frame does not make it answerable out loud"),
        ("Write a function to check if a given string is a palindrome.", "classic coding exercise"),
        ("Write a simple endpoint using Flask/FastAPI that updates a resource in MongoDB.",
         "backend deliverable"),
    ]

    KEEP = [
        ("Design a news aggregator system", "verbal architecture discussion — the whole point of the ask"),
        ("Design a system prompt for a Gemini-powered assistant that must summarize long financial "
         "documents into a precise three-bullet format.", "prompt design is spoken, and it is syllabus"),
        ("Design an RSS News Feed Service",
         "one of three tool-specific questions the n8n work recovered"),
        ("How would you design a scalable backend system for a high-traffic application?",
         "wh-opener — asks ABOUT designing"),
        ("How do you implement and handle authentication in a web application?",
         "general practice, not a demand to do it"),
        ("How did you implement JWT (JSON Web Token) authentication in your project?",
         "about the candidate's own past work — a strong conversational question"),
        ("Why did you implement JWT authentication in your project?", "reasoning about a past choice"),
        ("What is the Merge node and what merge modes does it support?", "plain knowledge question"),
        ("What are some of the design patterns you frequently use, and why?",
         "'design' as a noun, nothing to produce"),
    ]

    @pytest.mark.parametrize("text,why", SKIP)
    def test_hands_on_prompts_are_caught(self, text, why):
        assert is_hands_on_task(text), f"should skip ({why})"

    @pytest.mark.parametrize("text,why", KEEP)
    def test_conversational_questions_survive(self, text, why):
        assert not is_hands_on_task(text), f"should keep ({why})"

    def test_design_versus_design_and_implement(self):
        """The pair that a verb list gets wrong in both directions."""
        assert not is_hands_on_task("Design a data pipeline for streaming events")
        assert is_hands_on_task("Design and implement a data pipeline for streaming events")

    def test_asking_about_implementing_versus_being_told_to_implement(self):
        """Same verb, opposite verdicts — the wh-opener exemption is what separates them."""
        assert not is_hands_on_task("How do you implement retry logic in a webhook consumer?")
        assert is_hands_on_task("Implement retry logic in a webhook consumer.")

    def test_empty_input_is_not_a_task(self):
        for value in ("", "   ", None):
            assert not is_hands_on_task(value)


def _q(content, qid=None):
    return QuestionDetail(question_id=qid or content[:24], category="GEN_AI", content=content,
                          topic="t", difficulty="Medium", source="web")


def _state(questions):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
    st.session_context = SimpleNamespace(
        session_name="S", session_type="code_heavy", learning_outcomes=["Build an n8n workflow"],
        interview_topics=[], key_concepts=["Workflow automation"], scope_in=[], scope_out=[],
        matched_kp_ids=[])
    st.questions = {q.question_id: q for q in questions}
    return st


class TestThePoolFilter:
    QS = [_q("Write a Python program to reverse a string.", qid="code"),
          _q("Build and integrate LLM applications.", qid="build"),
          _q("What is the Merge node and what merge modes does it support?", qid="keep1"),
          _q("Design a news aggregator system", qid="keep2"),
          _q("How did you implement JWT authentication in your project?", qid="keep3")]

    def _run(self, monkeypatch, enabled=True):
        import src.config as cfg
        import src.pipeline as pl

        # `_drop_hands_on` imports the flag inside the function, so patching the module attribute is
        # what a real env override does.
        monkeypatch.setattr(cfg, "CONVERSATIONAL_ONLY", enabled)
        st = _state([_q(q.content, qid=q.question_id) for q in self.QS])
        pl.AgentPipeline()._drop_hands_on(st, lambda *a, **k: None)
        return st

    def test_it_drops_the_hands_on_prompts_and_keeps_the_rest(self, monkeypatch):
        st = self._run(monkeypatch)
        assert set(st.questions) == {"keep1", "keep2", "keep3"}

    def test_each_drop_is_recorded_with_its_own_stage(self, monkeypatch):
        """`scripts/yield_report.py` counts this stage — a filter that shrinks the pool silently reads
        as 'this session has few questions', which is the misdiagnosis the harness exists to prevent."""
        st = self._run(monkeypatch)
        stages = [r["stage"] for r in st.removed]
        assert stages == ["hands_on", "hands_on"]
        assert all("conversational" in r["reason"] for r in st.removed)

    def test_the_flag_really_turns_it_off(self, monkeypatch):
        """CONVERSATIONAL_ONLY=0 must restore the previous behaviour exactly — this is policy, and the
        corpus keeps the 217 coding questions, so the escape hatch has to work."""
        st = self._run(monkeypatch, enabled=False)
        assert len(st.questions) == 5
        assert st.removed == []

    def test_an_empty_pool_is_not_an_error(self, monkeypatch):
        import src.pipeline as pl
        st = _state([])
        pl.AgentPipeline()._drop_hands_on(st, lambda *a, **k: None)
        assert st.questions == {}


class TestTheBackfillCannotReintroduceThem:
    """`_top_up_from_open_web` fires precisely when the set is short, and the open web is full of
    "write a function to…" content. Without the check in `add_open_web_records` the backfill would hand
    straight back what the pool filter just removed, and the filter would look broken."""

    @staticmethod
    def _records():
        return [SimpleNamespace(question_text="Write a function to flatten a nested list.",
                                source_url="https://ex.com/interview-questions", source_type="open_web:ex.com"),
                SimpleNamespace(question_text="What is the Merge node in n8n and when do you use it?",
                                source_url="https://ex.com/interview-questions", source_type="open_web:ex.com")]

    def test_a_hands_on_open_web_record_is_refused(self, monkeypatch):
        import src.tools as tools
        st = _state([])
        monkeypatch.setattr(tools, "_topic_keywords", lambda ctx: set())

        added = tools.add_open_web_records(st, self._records())
        kept = [q.content for q in st.questions.values()]
        assert added == 1, "only the conversational record should land"
        assert not any("Write a function" in c for c in kept)

    def test_with_the_flag_off_it_lands_like_before(self, monkeypatch):
        import src.config as cfg
        import src.tools as tools
        monkeypatch.setattr(cfg, "CONVERSATIONAL_ONLY", False)
        st = _state([])
        monkeypatch.setattr(tools, "_topic_keywords", lambda ctx: set())

        assert tools.add_open_web_records(st, self._records()) == 2
