"""Post-relevance scope trim: removing an off-syllabus clause from an on-topic question.

Prompted by a real shipped question:

    how you would iteratively improve prompts and guards to increase reliability.

`guard`/`guardrail` appears ZERO times in the curated scope and zero times in the reading material for
the three "No-Code AI Automation" sessions, so "and guards" should have been dropped.

The tempting fix — trim ungrounded trailing conjuncts at retrieval time — was measured across the
1400-row GenAI bank and fires on 155 rows, destroying the comparison class ("difference between
supervised and unsupervised learning?" → cuts "unsupervised"). Applied to the FINAL SELECTED set it
fires on 1 of 5. Hence the trim runs after the relevance gate, and every hazard below is pinned here.

These tests assert the OBSERVABLE outcome of `_accept_trim` — the verifier that decides whether the
model's proposed text is a legitimate trim — because the guard is what keeps a company-attributed
question honest, not the prompt.
"""
import re
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail
from src.tools import _accept_trim, _scope_trim


# ── Trims that must happen ───────────────────────────────────────────────────

class TestOffSyllabusTailIsTrimmed:
    @pytest.mark.parametrize("original,proposed,expected", [
        # The case that prompted this work, in its well-formed form (see the fragment test below).
        ("How would you iteratively improve prompts and guards to increase reliability?",
         "How would you iteratively improve prompts",
         "How would you iteratively improve prompts?"),
        # Vector stores are in this session's scope_out.
        ("Explain chain-of-thought prompting and vector store indexing.",
         "Explain chain-of-thought prompting",
         "Explain chain-of-thought prompting."),
        # A second ask with its OWN object — evaluation metrics are out of scope.
        ("How do you design a prompt and evaluate BLEU scores?",
         "How do you design a prompt",
         "How do you design a prompt?"),
        ("What node types in n8n would you use, and how do you fine-tune a LoRA adapter?",
         "What node types in n8n would you use",
         "What node types in n8n would you use?"),
    ])
    def test_trailing_off_scope_clause_removed(self, original, proposed, expected):
        assert _accept_trim(original, proposed) == expected

    @pytest.mark.parametrize("original,proposed,expected", [
        # All three shipped MANGLED from a live Haiku run before the trailing separator was stripped:
        # the mark was appended straight onto the clause comma, and "…workflows,?" passes the form gate
        # because it does end in "?". The comma lives in the string, not in the word list, so the
        # dangling-tail check cannot see it.
        ("Are you aware of agentic workflows, and what frameworks have you used in that context?",
         "Are you aware of agentic workflows,", "Are you aware of agentic workflows?"),
        ("Build a customer-support AI agent for a fictional client, then choose which features to implement.",
         "Build a customer-support AI agent for a fictional client,",
         "Build a customer-support AI agent for a fictional client."),
        ("Can you explain how Agentic AI functions, particularly its architecture, using an example?",
         "Can you explain how Agentic AI functions, particularly its architecture,",
         "Can you explain how Agentic AI functions, particularly its architecture?"),
    ])
    def test_a_cut_at_a_comma_is_repunctuated_not_appended_to(self, original, proposed, expected):
        assert _accept_trim(original, proposed) == expected

    def test_no_trim_ever_ends_in_a_separator_before_its_mark(self):
        for orig, prop in [
            ("Are you aware of agentic workflows, and what frameworks have you used?",
             "Are you aware of agentic workflows,"),
            ("Explain chain-of-thought prompting; and vector store indexing.",
             "Explain chain-of-thought prompting;"),
        ]:
            out = _accept_trim(orig, prop)
            if out:
                assert not re.search(r"[,;:]\s*[.?!]$", out), out

    def test_punctuation_comes_from_the_original(self):
        """A question keeps its '?', a task keeps its '.' — the cut must not change the mood."""
        assert _accept_trim("How do you design a prompt and evaluate BLEU scores?",
                            "How do you design a prompt").endswith("?")
        assert _accept_trim("Explain chain-of-thought prompting and vector store indexing.",
                            "Explain chain-of-thought prompting").endswith(".")


# ── Trims that must NOT happen ──────────────────────────────────────────────

class TestConstitutiveConjunctsAreNeverCut:
    """Each of these was measured in the bank as a false positive of the naive rule."""

    @pytest.mark.parametrize("original,proposed,why", [
        ("Can you explain the difference between supervised and unsupervised learning?",
         "Can you explain the difference between supervised", "comparison frame"),
        ("What is the difference between top-k and top-p (nucleus) sampling?",
         "What is the difference between top-k", "comparison frame"),
        ("What are the trade-offs between RAG and fine-tuning?",
         "What are the trade-offs between RAG", "comparison frame"),
        ("Compare zero-shot and few-shot prompting.",
         "Compare zero-shot", "comparison frame"),
        # No comparison keyword at all — caught only by the shared-head-noun rule.
        ("Explain self-attention and multi-head attention.",
         "Explain self-attention", "elided comparison, shared head noun"),
        # The second verb borrows the first's object.
        ("how do you detect and reduce them?", "how do you detect", "shared object"),
        ("What are the pros and cons of few-shot prompting?",
         "What are the pros", "fixed idiom"),
        ("What are the advantages and disadvantages of few-shot prompting?",
         "What are the advantages", "fixed idiom"),
    ])
    def test_left_byte_identical(self, original, proposed, why):
        assert _accept_trim(original, proposed) is None, why


class TestModelRepliesAreVerifiedNotTrusted:
    """The model is asked for a prefix; whether it returned one is checked in our code."""

    ORIGINAL = "How would you iteratively improve prompts and guards to increase reliability?"

    def test_reworded_reply_is_rejected(self):
        assert _accept_trim(self.ORIGINAL, "How do you improve prompts iteratively?") is None

    def test_reordered_reply_is_rejected(self):
        assert _accept_trim(self.ORIGINAL, "How would you improve iteratively prompts") is None

    def test_added_words_are_rejected(self):
        assert _accept_trim(self.ORIGINAL, self.ORIGINAL + " and safety?") is None

    def test_gutted_reply_is_rejected(self):
        assert _accept_trim(self.ORIGINAL, "How would you") is None

    def test_unchanged_reply_is_not_a_trim(self):
        assert _accept_trim(self.ORIGINAL, self.ORIGINAL) is None

    @pytest.mark.parametrize("proposed", [
        # A model that cut one word too late leaves a grammatically broken question that is still a
        # valid prefix, still above the word ratio, and still passes the form gate because it opens
        # with "How" and ends with "?". Only an explicit dangling-tail check catches these.
        "How would you iteratively improve prompts and",
        "How would you iteratively improve prompts and guards to",
        "How would you iteratively improve prompts and guards to increase",
    ])
    def test_cut_leaving_a_dangling_word_is_rejected(self, proposed):
        assert _accept_trim(self.ORIGINAL, proposed) is None

    def test_a_trim_ending_on_a_real_object_is_still_accepted(self):
        """The dangling guard must not swallow legitimate trims that end in an article + noun."""
        assert _accept_trim("How do you design a prompt and evaluate BLEU scores?",
                            "How do you design a prompt") == "How do you design a prompt?"

    def test_result_must_still_be_a_well_formed_question(self):
        """The cited string is a lowercase FRAGMENT, so no trim of it can be well-formed — and the
        form gate must refuse it rather than ship a shorter fragment. This is why the exact question
        the reviewer flagged is unsalvageable, and why it was removed from the bank instead."""
        frag = "how you would iteratively improve prompts and guards to increase reliability."
        assert _accept_trim(frag, "how you would iteratively improve prompts") is None


# ── The pass as wired into a run ────────────────────────────────────────────

def _q(content, **kw):
    return QuestionDetail(category="GEN_AI", content=content, topic="Gen AI",
                          source="interview_db", **kw)


def _state(questions):
    from src.agent import AgentState

    st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=SimpleNamespace())
    st.questions = {q.question_id: q for q in questions}
    st.session_context = SimpleNamespace(
        scope_in=["Prompt engineering techniques", "Chain-of-thought prompting"],
        learning_outcomes=["Master zero-shot, one-shot, few-shot and chain-of-thought prompting"],
        scope_out=["LLM guardrails and safety", "Vector stores"],
        key_concepts=[], interview_topics=[], matched_kp_ids=[],
        session_type="theory_heavy",     # read by _select_final via tool_submit_question_set
    )
    return st


class TestScopeTrimPass:
    def test_trim_records_the_original_and_keeps_the_company(self, monkeypatch):
        """Reviewer's choice: a trimmed question keeps its company attribution and is marked adapted."""
        import src.tools as tools

        q = _q("How would you iteratively improve prompts and guards to increase reliability?",
               asked_in_company="Intuit")
        st = _state([q])
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "results": [{"n": 1, "text": "How would you iteratively improve prompts"}]})

        trims = _scope_trim(st, [q])
        assert len(trims) == 1
        assert q.content == "How would you iteratively improve prompts?"
        assert q.original_content.endswith("increase reliability?")   # source kept for audit
        assert q.adapted is True
        assert q.attribution == "INTUIT"                              # attribution unchanged

    def test_untouched_question_has_no_original_and_is_not_adapted(self, monkeypatch):
        import src.tools as tools

        q = _q("What is chain-of-thought prompting?")
        st = _state([q])
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "results": [{"n": 1, "text": "What is chain-of-thought prompting?"}]})

        assert _scope_trim(st, [q]) == []
        assert q.original_content is None and q.adapted is False

    def test_llm_failure_leaves_every_question_untouched(self, monkeypatch):
        """Fail-open: a trim outage must never cost the set."""
        import src.tools as tools

        original = "How would you improve prompts and guards to increase reliability?"
        q = _q(original)
        st = _state([q])

        def boom(**kw):
            raise RuntimeError("api down")
        monkeypatch.setattr(tools, "chat_completion_json", boom)

        assert _scope_trim(st, [q]) == []
        assert q.content == original and q.original_content is None

    def test_uses_the_runs_model_not_the_ui_global(self, monkeypatch):
        """The documented trap: omitting `model=` silently rides `get_active_model()`."""
        import src.tools as tools

        q = _q("How would you improve prompts and guards to increase reliability?")
        st = _state([q])
        st.config = GenerationConfig(session_names=["S"], model="anthropic/claude-sonnet-4-6")
        seen = {}

        def capture(**kw):
            seen.update(kw)
            return {"results": []}
        monkeypatch.setattr(tools, "chat_completion_json", capture)

        _scope_trim(st, [q])
        assert seen.get("model") == "anthropic/claude-sonnet-4-6"

    def test_out_of_range_index_is_ignored(self, monkeypatch):
        import src.tools as tools

        q = _q("What is chain-of-thought prompting?")
        st = _state([q])
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "results": [{"n": 99, "text": "anything"}, {"n": 0, "text": "anything"}]})

        assert _scope_trim(st, [q]) == []
        assert q.content == "What is chain-of-thought prompting?"

    def test_submit_runs_the_trim_and_reports_it(self, monkeypatch):
        """Pins the WIRING. A live run reported 0 trims, which is correct for a set with no off-scope
        clauses but indistinguishable from the pass never being called — so assert it is called from
        `tool_submit_question_set` and that the count reaches both the result and `state.scope_trims`.
        """
        import src.tools as tools

        qs = [_q("How would you iteratively improve prompts and guards to increase reliability?",
                 asked_in_company="Intuit")]
        qs += [_q(f"What is prompting technique {i}?") for i in range(4)]
        st = _state(qs)
        monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: {
            "results": [{"n": 1, "text": "How would you iteratively improve prompts"}]})
        # Selection needs the same session profile the real run has; keep it simple and deterministic.
        monkeypatch.setattr(tools, "_select_final", lambda cands, k, *a, **kw: list(cands)[:k])
        monkeypatch.setattr(tools, "_assign_kp_labels", lambda *a, **kw: 0)

        out = tools.tool_submit_question_set(st)
        assert out["scope_trimmed"] == 1
        assert len(st.scope_trims) == 1
        assert st.scope_trims[0]["after"] == "How would you iteratively improve prompts?"

    def test_no_scope_means_no_trimming(self, monkeypatch):
        """Without a resolved scope there is nothing to judge against — do not guess."""
        import src.tools as tools

        q = _q("How would you improve prompts and guards to increase reliability?")
        st = _state([q])
        st.session_context = SimpleNamespace(scope_in=[], learning_outcomes=[], scope_out=[],
                                             key_concepts=[], interview_topics=[], matched_kp_ids=[])
        called = []
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: called.append(1) or {"results": []})

        assert _scope_trim(st, [q]) == []
        assert not called, "should not spend an LLM call with no scope to judge against"
