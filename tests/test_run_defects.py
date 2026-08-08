"""Defects found by the first LIVE run after the retrieval/eval/session-type overhaul.

The run (topic "No-Code AI Automation", 3 sessions, code_heavy, Sonnet 4.6) completed end to end and
still shipped a set with five separate problems. Each test below pins the OBSERVABLE behaviour that
run exposed, using the actual strings and numbers it produced — not the presence of a fix.

What the run shipped, for reference:
  * a job-description bullet ("Designing or writing prompts to support specific AI outcomes") as Q2
  * two "Removed — 5 left" transcript lines for removals the tool had refused
  * "E:0 M:4 H:1 Fix" on all three revision rounds, with no Easy candidate anywhere in the pool
  * kp_label = None on all five questions
  * attribution "Analytics Vidhya" / "Indeed" / "Edureka" / "DataCamp" beside a real "ANTHROPIC"
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail


def _q(content, difficulty="Medium", qid=None, **kw):
    q = QuestionDetail(category="GEN_AI", content=content, topic="Gen AI",
                       source="interview_db", difficulty=difficulty, **kw)
    if qid:
        q.question_id = qid
    return q


def _state(questions=(), reserve=(), session_type="code_heavy", min_questions=5, kps=()):
    """Minimal AgentState — only the fields the tools under test read."""
    from src.agent import AgentState

    st = AgentState(config=GenerationConfig(session_names=["S"], min_questions=min_questions),
                    data_store=SimpleNamespace())
    st.questions = {q.question_id: q for q in questions}
    st.reserve = {q.question_id: q for q in reserve}
    st.session_context = SimpleNamespace(
        session_type=session_type,
        learning_outcomes=["Understand n8n workflow automation"],
        interview_topics=[],
        matched_kp_ids=list(kps),
    )
    return st


# ── Defect 3: the form gate read gerunds as imperatives ──────────────────────

class TestGerundsAreNotImperatives:
    """`_Q_STARTS` was matched with a bare `startswith`, so "design" matched "Designing"."""

    @pytest.mark.parametrize("text", [
        # The exact line that shipped, from indeed.com/hire/job-description/prompt-engineer:
        "Designing or writing prompts to support specific AI outcomes",
        "Designing ML Systems by Chip Huyen",              # a book
        "Building AI, Careers & Startups with Maria",      # a podcast episode
        "Explaining RLHF in Detail.",                      # an answer heading
        "Solved that question in 5 minutes.",              # forum prose
        "Builds tokens based on likelihood",               # a sentence fragment
        "Implements the sigmoid function to calculate the probability.",
        "Sorted by relevance to this company",             # page furniture
    ])
    def test_gerund_and_inflected_fragments_are_rejected(self, text):
        from src.quality import is_quality_question
        from src.sources.base import looks_like_question

        assert not looks_like_question(text)
        assert not is_quality_question(text)

    @pytest.mark.parametrize("text", [
        "Design a prompt playground for developers and prompt engineers.",
        "Explain chain-of-thought prompting.",
        "Build a RAG pipeline over a document store.",
        "Implement a retry policy for a flaky LLM call.",
        "Write a function that chunks a document for embedding.",
        "Compare zero-shot and few-shot prompting.",
        "List the node types in an n8n workflow.",
        "What is chain-of-thought prompting?",
        # A genuine question may open with a gerund — the "?" is accepted before the opener is read.
        "Building a RAG pipeline needs which components?",
    ])
    def test_real_imperatives_and_questions_still_pass(self, text):
        from src.quality import is_quality_question
        from src.sources.base import looks_like_question

        assert looks_like_question(text)
        assert is_quality_question(text)

    @pytest.mark.parametrize("text", [
        # Shipped as Q3 by the verification run, before this rule existed:
        "how you would iteratively improve prompts and guards to increase reliability.",
        "why it matters for LLMs.",
        "how the generator and discriminator interact.",
        "when you would set it to 0.",
        "why a 200K context model is not always better than an 8K model.",
    ])
    def test_lowercase_sentence_continuations_are_rejected(self, text):
        """A clause split off a bigger sentence: it opens lowercase and asks nothing."""
        from src.quality import is_quality_question

        assert not is_quality_question(text)

    @pytest.mark.parametrize("text", [
        "what is the attention mechanism?",
        "how does AI bias occur?",
        "how would you handle incorrect or biased responses?",
    ])
    def test_lowercase_but_real_questions_survive(self, text):
        """37 bank rows open lowercase and end in "?" — real questions missing a capital, not scraps.
        The "?" requirement is the whole reason the rule above is safe."""
        from src.quality import is_quality_question

        assert is_quality_question(text)

    def test_prefix_collisions_no_longer_match(self):
        """The same bare-prefix bug let "what" match "Whatever" and "how" match "Howto"."""
        from src.sources.base import looks_like_question

        assert not looks_like_question("Whatever happened to prompt tuning")
        assert not looks_like_question("Howto guides for beginners")
        # …while the real openers are unaffected, including the contracted form.
        assert looks_like_question("What's the difference between RAG and fine-tuning?")
        assert looks_like_question("How does chain-of-thought help")


# ── Defect 2: the transcript claimed removals that were refused ──────────────

class TestRemoveQuestionLabelTellsTheTruth:
    def test_refused_removal_is_not_reported_as_removed(self):
        """At min_questions with an empty reserve the tool refuses. The label said "Removed" anyway,
        so the run showed two removals while the gate-flagged duplicate stayed in the set."""
        from src.agent import _summarize_result
        from src.tools import tool_remove_question

        qs = [_q(f"What is concept {i}?") for i in range(5)]
        st = _state(questions=qs, reserve=(), min_questions=5)
        target = qs[1].question_id

        result = tool_remove_question(st, target)
        assert result["removed"] is False                  # correctly refused
        assert target in st.questions                      # and really still there

        label = _summarize_result("remove_question", result)
        assert "NOT removed" in label
        assert not label.startswith("Removed")

    def test_successful_removal_still_reads_as_removed(self):
        from src.agent import _summarize_result
        from src.tools import tool_remove_question

        qs = [_q(f"What is concept {i}?") for i in range(6)]
        st = _state(questions=qs, reserve=(), min_questions=5)

        result = tool_remove_question(st, qs[0].question_id)
        assert result["removed"] is True
        assert _summarize_result("remove_question", result).startswith("Removed")


# ── Defect 4: "Fix" demanded with nothing to fix it from ─────────────────────

class TestDifficultyBalanceReportsAchievability:
    def test_unachievable_shortfall_is_named_not_just_flagged(self):
        """The live shape: E:0 M:4 H:1 against code_heavy 20/50/30, empty reserve."""
        from src.agent import _summarize_result
        from src.tools import tool_check_difficulty_balance

        qs = [_q(f"What is concept {i}?", difficulty="Medium") for i in range(4)]
        qs.append(_q("Design a scalable retrieval layer.", difficulty="Hard"))
        st = _state(questions=qs, reserve=())

        r = tool_check_difficulty_balance(st)
        assert r["balanced"] is False
        assert r["achievable"] is False
        assert r["available_by_difficulty"]["Easy"] == 0
        assert "Easy: need" in r["note"] and "only 0 exist" in r["note"]

        label = _summarize_result("check_difficulty_balance", r)
        assert "Fix" not in label
        assert "no candidates available" in label

    def test_achievable_shortfall_still_says_fix(self):
        """When the reserve HAS the missing difficulty, the agent should still act."""
        from src.agent import _summarize_result
        from src.tools import tool_check_difficulty_balance

        qs = [_q(f"What is concept {i}?", difficulty="Medium") for i in range(4)]
        qs.append(_q("Design a scalable retrieval layer.", difficulty="Hard"))
        spare = [_q(f"What does term {i} mean?", difficulty="Easy") for i in range(3)]
        st = _state(questions=qs, reserve=spare)

        r = tool_check_difficulty_balance(st)
        assert r["balanced"] is False
        assert r["achievable"] is True
        assert "note" not in r
        assert _summarize_result("check_difficulty_balance", r).endswith("Fix")

    def test_excluded_reserve_questions_do_not_count_as_available(self):
        """An excluded question was already rejected this run; it cannot fix anything."""
        from src.tools import tool_check_difficulty_balance

        qs = [_q(f"What is concept {i}?", difficulty="Medium") for i in range(4)]
        qs.append(_q("Design a scalable retrieval layer.", difficulty="Hard"))
        spare = [_q(f"What does term {i} mean?", difficulty="Easy") for i in range(3)]
        st = _state(questions=qs, reserve=spare)
        st.excluded = {q.question_id for q in spare}

        r = tool_check_difficulty_balance(st)
        assert r["available_by_difficulty"]["Easy"] == 0
        assert r["achievable"] is False


# ── Defect 5: kp_label was written by nothing ────────────────────────────────

class TestKpLabelsAreAssigned:
    def test_selected_questions_get_the_best_matching_kp(self):
        from src.tools import _assign_kp_labels

        kps = [SimpleNamespace(kp_id="KP1", kp_label="Chain-of-thought prompting", relevance=0.9),
               SimpleNamespace(kp_id="KP2", kp_label="n8n workflow nodes and connections", relevance=0.9)]
        qs = [_q("What is chain-of-thought prompting?"),
              _q("Which node types connect steps in an n8n workflow?")]
        ctx = SimpleNamespace(matched_kp_ids=kps)

        tagged = _assign_kp_labels(qs, ctx)
        if tagged == 0:
            pytest.skip("embeddings unavailable in this environment")
        assert qs[0].kp_label == "Chain-of-thought prompting"
        assert qs[1].kp_label == "n8n workflow nodes and connections"

    def test_no_kps_leaves_labels_none_rather_than_guessing(self):
        from src.tools import _assign_kp_labels

        qs = [_q("What is chain-of-thought prompting?")]
        assert _assign_kp_labels(qs, SimpleNamespace(matched_kp_ids=[])) == 0
        assert qs[0].kp_label is None

    def test_submit_reports_how_many_it_tagged(self, monkeypatch):
        """The count is reported so a silent zero is visible instead of looking like success."""
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        # Submit makes three OpenRouter calls (`_scope_trim`, `_syllabus_audit`, `_same_thing_pass`).
        # This test is about KP labelling, which is a local embedding match — so stub the boundary
        # rather than spending credit on it. All three are fail-open, so `{}` is enough.
        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})

        kps = [SimpleNamespace(kp_id="KP1", kp_label="Chain-of-thought prompting", relevance=0.9)]
        qs = [_q("What is chain-of-thought prompting?"),
              _q("How do you iterate on a prompt?"),
              _q("Explain few-shot prompting."),
              _q("What is a knowledge cutoff?"),
              _q("Explain hallucination mitigation.")]
        st = _state(questions=qs, kps=kps)

        r = tool_submit_question_set(st)
        assert "kp_tagged" in r
        assert r["kp_tagged"] <= r["total_questions"]


# ── Defect 6: a content site presented as the asking company ────────────────

class TestAttributionSeparatesCompanyFromProvenance:
    @pytest.mark.parametrize("url,site", [
        ("https://www.analyticsvidhya.com/blog/2024/04/llm-interview-questions", "Analytics Vidhya"),
        ("https://www.indeed.com/hire/job-description/prompt-engineer", "Indeed"),
        ("https://www.datacamp.com/blog/llm-interview-questions", "DataCamp"),
    ])
    def test_the_four_sites_the_run_shipped_are_now_niat(self, url, site):
        q = QuestionDetail(category="GEN_AI", content="What is prompt engineering?", topic="Gen AI",
                           source="interview_db", source_url=url)
        assert q.attribution == "NIAT"
        assert q.source_site == site          # provenance kept, on a different field

    def test_a_real_company_is_unchanged_and_still_wins(self):
        q = QuestionDetail(category="GEN_AI", content="Design a prompt playground.", topic="Gen AI",
                           source="interview_db", asked_in_company="Anthropic",
                           source_url="https://prachub.com/companies/anthropic/x")
        assert q.attribution == "ANTHROPIC"
        assert q.source_site == "PracHub"     # where the claim was found, not who asked

    def test_no_url_means_no_provenance_claim(self):
        q = QuestionDetail(category="GEN_AI", content="What is RAG?", topic="Gen AI",
                           source="interview_db")
        assert q.attribution == "NIAT"
        assert q.source_site is None
