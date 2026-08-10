"""The same-thing pass over the SELECTED set.

Run 8fb9fcb3 shipped both of these as separate questions:

    "How would you modify a system prompt to ensure the model always responds in structured JSON
     format?"
    "How do you write effective prompts for consistent JSON output?"

The quality gate's critique named them a duplicate. Semantic dedup did not: they measure **0.767**
against its **0.82** bar. And the bar cannot be lowered — across 900 GenAI-bank rows, 209 pairs sit in
[0.74, 0.82) and the band mixes real duplicates with legitimately distinct definition-vs-practice
pairs, so no cutoff separates them.

Two things these tests exist to stop:
  * a future change that "fixes" this by lowering `DEDUP_SEMANTIC_THRESHOLD` (kills the comparison and
    definition-vs-practice classes corpus-wide), and
  * a pass that pushes a set under `MIN_QUESTIONS` — which is the real reason the duplicate shipped:
    the set held exactly 5, so `remove_question` had to refuse.
"""
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail

Q4 = ("How would you modify a system prompt to ensure the model always responds in structured "
      "JSON format?")
Q5 = "How do you write effective prompts for consistent JSON output?"


def _q(content, relevance=0.8, qid=None):
    return QuestionDetail(question_id=qid or content[:20], category="GEN_AI", content=content,
                          topic="t", difficulty="Medium", source="web", relevance_score=relevance)


def _state(questions):
    from src.agent import AgentState
    from src.data_loader import get_data_store

    st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
    st.session_context = SimpleNamespace(learning_outcomes=[], scope_in=[], scope_out=[],
                                         key_concepts=[], interview_topics=[], matched_kp_ids=[],
                                         session_type="theory_heavy")
    st.questions = {q.question_id: q for q in questions}
    return st


def _pair_at(monkeypatch, questions, sim=0.767, verdict=True):
    """Force exactly one candidate pair (the first two) at `sim`, and a model reply for it."""
    import src.tools as tools
    from src import embeddings

    n = len(questions)

    def matrix(texts, other=None):
        m = [[1.0 if i == j else 0.1 for j in range(n)] for i in range(n)]
        m[0][1] = m[1][0] = sim
        return m

    monkeypatch.setattr(embeddings, "cosine_matrix", matrix)
    monkeypatch.setattr(tools, "chat_completion_json",
                        lambda **kw: {"pairs": [{"n": 1, "same": verdict}]})


class TestNearDuplicatePairSelection:
    def test_a_pair_under_the_dedup_bar_is_offered_for_judgement(self, monkeypatch):
        import src.tools as tools

        qs = [_q(Q4), _q(Q5)]
        _pair_at(monkeypatch, qs, sim=0.767)
        assert len(tools._near_duplicate_pairs(qs)) == 1

    def test_a_pair_at_or_above_the_bar_is_not_re_judged(self, monkeypatch):
        """`tool_deduplicate_questions` already removed those; spending a call re-decides nothing."""
        import src.tools as tools
        from src.config import DEDUP_SEMANTIC_THRESHOLD

        qs = [_q(Q4), _q(Q5)]
        _pair_at(monkeypatch, qs, sim=DEDUP_SEMANTIC_THRESHOLD + 0.01)
        assert tools._near_duplicate_pairs(qs) == []

    def test_a_distant_pair_costs_nothing(self, monkeypatch):
        import src.tools as tools

        qs = [_q("What is RAG?"), _q("How do you deploy a model on Kubernetes?")]
        _pair_at(monkeypatch, qs, sim=0.30)
        assert tools._near_duplicate_pairs(qs) == []


class TestRemovalRespectsTheFloor:
    def test_above_the_floor_the_weaker_question_is_removed(self, monkeypatch):
        import src.tools as tools
        from src.config import MIN_QUESTIONS

        qs = [_q(Q4, relevance=0.9, qid="strong"), _q(Q5, relevance=0.4, qid="weak")]
        qs += [_q(f"Distinct question {i}?", qid=f"d{i}") for i in range(MIN_QUESTIONS)]
        st = _state(qs)
        _pair_at(monkeypatch, qs)

        out = tools._same_thing_pass(st, qs)
        assert out["removed"] == 1 and out["flagged"] == 0
        assert "weak" not in st.questions, "the lower-relevance question should go"
        assert "strong" in st.questions
        assert any(r["stage"] == "duplicate" for r in st.removed)

    def test_at_the_floor_it_flags_instead_of_dropping(self, monkeypatch):
        """THE 8fb9fcb3 case. A set of exactly MIN_QUESTIONS cannot afford a removal, and shipping the
        duplicate silently is what happened last time — so it is flagged for the reviewer."""
        import src.tools as tools
        from src.config import MIN_QUESTIONS

        qs = [_q(Q4, relevance=0.9, qid="strong"), _q(Q5, relevance=0.4, qid="weak")]
        qs += [_q(f"Distinct question {i}?", qid=f"d{i}") for i in range(MIN_QUESTIONS - 2)]
        assert len(qs) == MIN_QUESTIONS
        st = _state(qs)
        _pair_at(monkeypatch, qs)

        out = tools._same_thing_pass(st, qs)
        assert out["removed"] == 0 and out["flagged"] == 1
        assert len(st.questions) == MIN_QUESTIONS, "must never push the set under the minimum"
        assert st.questions["weak"].duplicate_of == Q4

    def test_it_never_takes_the_set_below_the_floor_across_many_pairs(self, monkeypatch):
        """A cluster of duplicates must stop removing at the floor, not keep going."""
        import src.tools as tools
        from src import embeddings
        from src.config import MIN_QUESTIONS

        qs = [_q(f"How do you get consistent JSON output variant {i}?", relevance=0.5 + i / 100,
                 qid=f"j{i}") for i in range(MIN_QUESTIONS + 2)]
        n = len(qs)
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0 if i == j else 0.75 for j in range(n)]
                                                       for i in range(n)])
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: {"pairs": [{"n": k + 1, "same": True} for k in range(12)]})
        st = _state(qs)

        tools._same_thing_pass(st, qs)
        assert len(st.questions) >= MIN_QUESTIONS


class TestTheReplyIsVerifiedInCode:
    def test_a_pair_we_never_asked_about_is_ignored(self, monkeypatch):
        """Same discipline as `_accept_trim`: the model's reply is checked, not trusted."""
        import src.tools as tools
        from src import embeddings

        qs = [_q(Q4, qid="a"), _q(Q5, qid="b"), _q("What is RAG?", qid="c")]
        n = len(qs)
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0 if i == j else 0.1 for j in range(n)]
                                                       for i in range(n)])
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: {"pairs": [{"n": 99, "same": True}]})
        st = _state(qs)

        out = tools._same_thing_pass(st, qs)
        assert out["removed"] == 0 and out["flagged"] == 0
        assert len(st.questions) == 3

    def test_a_false_verdict_changes_nothing(self, monkeypatch):
        import src.tools as tools
        from src.config import MIN_QUESTIONS

        qs = [_q(Q4, qid="a"), _q(Q5, qid="b")]
        qs += [_q(f"Distinct {i}?", qid=f"d{i}") for i in range(MIN_QUESTIONS)]
        st = _state(qs)
        _pair_at(monkeypatch, qs, verdict=False)

        out = tools._same_thing_pass(st, qs)
        assert out["removed"] == 0 and out["flagged"] == 0

    def test_an_llm_failure_leaves_the_set_untouched(self, monkeypatch):
        import src.tools as tools
        from src import embeddings

        qs = [_q(Q4, qid="a"), _q(Q5, qid="b")]
        n = len(qs)
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0, 0.767], [0.767, 1.0]])

        def boom(**kw):
            raise RuntimeError("provider down")

        monkeypatch.setattr(tools, "chat_completion_json", boom)
        st = _state(qs)

        out = tools._same_thing_pass(st, qs)
        assert out == {"pairs_judged": 0, "removed": 0, "flagged": 0}
        assert len(st.questions) == 2

    def test_no_embeddings_means_no_call_at_all(self, monkeypatch):
        import src.tools as tools
        from src import embeddings

        called = []
        monkeypatch.setattr(embeddings, "cosine_matrix", lambda *a, **k: None)
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: called.append(1) or {"pairs": []})
        st = _state([_q(Q4), _q(Q5)])

        tools._same_thing_pass(st, list(st.questions.values()))
        assert not called, "fail-open: no embeddings → no LLM spend"


class TestTheDedupThresholdIsNotTheFix:
    def test_the_threshold_stays_where_the_measurement_put_it(self):
        """A future 'fix' that lowers this to catch the 0.767 pair would merge 209 pairs in 900 bank
        rows, destroying the definition-vs-practice and comparison classes. The judged pass above is
        the fix; this pins the constant so the cheap wrong fix fails loudly."""
        from src.config import DEDUP_SEMANTIC_THRESHOLD

        assert DEDUP_SEMANTIC_THRESHOLD >= 0.80, (
            "Lowering the corpus-wide dedup bar is not the way to catch same-thing pairs — "
            "see tools._same_thing_pass and the band measurement in its comment.")


class TestTheJudgeIsBatchedHereToo:
    """`_same_thing_pass` used to send every pair in ONE call at `max_tokens=1024`, with a 12-pair cap.

    Both halves were wrong, and both were measured elsewhere in this codebase before being fixed here:

    * a 12-pair cap sized for a ~10-question selected set left **29 of 41 eligible pairs unjudged** on a
      38-question accumulated set, and the one hallucination pair the judge calls redundant ranked 14th —
      so "0 redundant" meant "not looked at";
    * verdict quality degrades with batch size — a pair called SAME 3 times out of 3 on its own was called
      "different" inside a batch of 52 (see `outcome_balance.JUDGE_BATCH`).
    """

    def test_the_cap_is_a_runaway_guard_not_a_sampling_budget(self):
        from src.tools import _SAME_THING_MAX_PAIRS
        assert _SAME_THING_MAX_PAIRS >= 100, (
            "12 silently dropped 29 of 41 eligible pairs on a real set while reporting 0 redundant")

    def test_pairs_go_out_in_small_batches(self, monkeypatch):
        import src.tools as tools
        from src import embeddings
        from src.outcome_balance import JUDGE_BATCH

        n = 12
        qs = [_q(f"What is generative AI idea number {i} about?", qid=f"q{i}") for i in range(n)]
        # Every pair inside the judging band, so all 66 are eligible.
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0 if i == j else 0.7 for j in range(n)]
                                                       for i in range(n)])
        sizes = []

        def fake_chat(**kw):
            sizes.append(kw["user_prompt"].count('"n":'))
            return {"pairs": []}

        monkeypatch.setattr(tools, "chat_completion_json", fake_chat)
        out = tools._same_thing_pass(_state(qs), qs)

        assert len(sizes) > 1, "a single call is the defect this fixes"
        assert max(sizes) <= JUDGE_BATCH, f"batch of {max(sizes)} exceeds {JUDGE_BATCH}"
        assert sum(sizes) == out["pairs_judged"], "every eligible pair must be judged, not sampled"

    def test_a_total_outage_reports_nothing_judged(self, monkeypatch):
        """Per-batch fail-open returns [] instead of raising, so reporting `len(pairs)` here would claim
        the set was checked when every call died — the silent-success class this project keeps hitting."""
        import src.tools as tools
        from src import embeddings

        qs = [_q(Q4, qid="a"), _q(Q5, qid="b")]
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0, 0.767], [0.767, 1.0]])
        monkeypatch.setattr(tools, "chat_completion_json",
                            lambda **kw: (_ for _ in ()).throw(RuntimeError("provider down")))
        assert tools._same_thing_pass(_state(qs), qs) == {"pairs_judged": 0, "removed": 0, "flagged": 0}

    def test_one_failed_batch_still_acts_on_the_rest(self, monkeypatch):
        import src.tools as tools
        from src import embeddings

        n = 12
        qs = [_q(f"What is generative AI idea number {i} about?", qid=f"q{i}") for i in range(n)]
        monkeypatch.setattr(embeddings, "cosine_matrix",
                            lambda texts, other=None: [[1.0 if i == j else 0.7 for j in range(n)]
                                                       for i in range(n)])
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("rate limited")
            return {"pairs": [{"n": 1, "same": True}]}

        monkeypatch.setattr(tools, "chat_completion_json", flaky)
        out = tools._same_thing_pass(_state(qs), qs)
        assert out["pairs_judged"] > 0, "one bad batch must not report the whole pass as unjudged"
        assert out["removed"] + out["flagged"] > 0, "verdicts from the surviving batches must be acted on"
