"""One interview topic must not supply half the set.

Written after a review of No-Code AI Automation found hallucination asked SIX times in 38 questions. The
framing that made it fixable was not "these are duplicates" — no similarity threshold can see that class
(`tests/test_topic_dupes.py` pins the 0.486 measurement) — it is that three interview topics held 47% of
the set while 9 of 22 held nothing.

Every test here corresponds to a way this could have shipped as a data-loss bug instead of a fix:

* `TestAnOrphanIsKeptNeverCut` — a question no outcome describes is not redundant. Getting this wrong
  deletes the very n8n questions this project spent a round recovering.
* `TestTheQuotaIsOptIn` — the first design dropped everything past a per-outcome cap. It removed 12 of 14
  genuinely distinct questions on Gen AI Foundations, because the cap measures how finely the curriculum
  enumerates `interview_topics`, not whether questions repeat.
* `TestSmallBatchesAreTheWholePoint` — the duplicates that appeared to justify a quota were a batching
  artefact. This is the most consequential finding in the module.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models import GenerationConfig, QuestionDetail
from src.outcome_balance import balance_by_outcome

# Verbatim from the shipped No-Code AI Automation set.
HALLUCINATION = [
    "What strategies do you employ to mitigate hallucinations in generative AI models?",   # applied
    'How do you prevent hallucinations?"',                                                 # applied, dup
    "What are hallucinations in LLMs",                                                     # definition
    "What is an AI hallucination",                                                         # definition, dup
    "What is hallucination, and how can it be controlled using prompt engineering?",
    "Explain your methodology for designing and testing system prompts to prevent model hallucination.",
]
OUTCOMES = [
    "Hallucination detection and mitigation strategies",
    "Workflow automation with LLMs",
    "Human-in-the-loop AI systems",
]


def _judge(*pairs):
    """A stub judge. Returning fixed pairs keeps this offline — the network guard stays armed."""
    return lambda _asked: list(pairs)


class TestTheHallucinationPile:
    """The regression case. Six questions on one outcome must come out as two."""

    def test_nothing_is_dropped_without_a_verdict(self):
        """The default criterion is JUDGEMENT, not counting. With a judge that finds nothing redundant,
        six questions on one outcome all survive — deliberately. A quota here deleted 12 of 14 distinct
        questions on another topic (`TestTheQuotaIsOptIn`)."""
        res = balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6] * 6,
                                 approved=[True] * 6, cap=2, judge=_judge(), min_keep=1)
        assert res.drop == [], "a silent count-based cut is what this design rejects"

    def test_six_become_two_when_the_judge_says_so(self):
        """The real verdicts a 10-pair-batched judge returns on these six. 6 -> 2 by judgement alone,
        which is what the quota was mistakenly credited with achieving."""
        res = balance_by_outcome(
            HALLUCINATION, OUTCOMES, fits=[0.622, 0.609, 0.516, 0.641, 0.719, 0.513],
            approved=[True] * 4 + [False] * 2, cap=2, min_keep=1,
            judge=_judge((0, 1), (0, 5), (2, 3), (3, 4)))
        assert len(res.keep) == 2, f"expected 6 -> 2, got {[HALLUCINATION[i] for i in res.keep]}"

    def test_the_two_kept_are_a_definition_and_an_applied_question(self):
        """Asserted by STRING, not count. Keeping two definitional questions would satisfy a count
        assertion and still be the bug the reviewer complained about: `--no-judge` on the real data keeps
        'mitigate hallucinations' AND 'How do you prevent hallucinations?', which are one ask.
        """
        fits = [0.622, 0.609, 0.516, 0.641, 0.719, 0.513]
        # The judge sees every within-outcome pair; these are the two it calls redundant.
        same = _judge((0, 1), (2, 3))
        res = balance_by_outcome(HALLUCINATION, OUTCOMES, fits=fits, approved=[True] * 4 + [False] * 2,
                                 cap=2, judge=same, min_keep=1)
        kept = [HALLUCINATION[i] for i in res.keep]
        assert HALLUCINATION[0] in kept, "the applied question must survive"
        assert HALLUCINATION[1] not in kept, "'prevent hallucinations' duplicates 'mitigate hallucinations'"
        # Exactly one of the two definitional questions, never both. WHICH one is decided by fit, so
        # naming it would pin this fixture's numbers instead of the rule.
        definitional = {HALLUCINATION[2], HALLUCINATION[3]}
        assert len(definitional & set(kept)) == 1, (
            f"expected one definitional question alongside the applied one, got {kept}")

    def test_every_within_outcome_pair_is_offered_to_the_judge(self):
        """15 pairs for 6 members. The first version only generated pairs for OVER-served outcomes, so an
        outcome holding exactly `cap` members was never judged at all."""
        seen = {}

        def judge(pairs):
            seen["n"] = len(pairs)
            return []

        balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6] * 6, approved=[True] * 6,
                           cap=2, judge=judge, min_keep=1)
        assert seen["n"] == 15, f"expected all 6-choose-2 pairs, got {seen['n']}"


class TestSmallBatchesAreTheWholePoint:
    """The judge's accuracy collapses with batch size, and that was the actual bug.

    Measured against the live model on *"What are hallucinations in LLMs"* / *"What is an AI
    hallucination"* — the pair a reviewer flagged by hand:

        batch of 1   -> SAME, 3 of 3 trials
        batch of 52  -> "different"

    The first diagnosis blamed the per-outcome CAP for the surviving duplicates and would have "fixed" it
    with a hard quota that deleted 12 distinct questions on another topic. The cap was innocent.

    These tests pin the batching itself, offline, because `JUDGE_BATCH` is the kind of constant someone
    raises to "save calls" — which silently degrades every verdict rather than failing.
    """

    def test_pairs_are_sent_in_small_batches(self):
        from src.outcome_balance import JUDGE_BATCH, make_llm_judge

        sizes = []

        def fake_chat(**kw):
            sizes.append(kw["user_prompt"].count('"n":'))
            return {"pairs": []}

        import itertools

        texts = [f"question number {i}?" for i in range(20)]
        pairs = list(itertools.combinations(range(20), 2))[:25]
        make_llm_judge(texts, model="m", complete=fake_chat)(pairs)

        assert len(sizes) == 3, f"25 pairs at batch {JUDGE_BATCH} should be 3 calls, got {len(sizes)}"
        assert max(sizes) <= JUDGE_BATCH
        assert sum(sizes) == 25, "every pair must be judged — batching is not sampling"

    def test_the_batch_size_stays_small(self):
        from src.outcome_balance import JUDGE_BATCH
        assert JUDGE_BATCH <= 15, (
            "raising this degrades verdict accuracy; measured wrong at 52 pairs in one call")

    def test_one_failed_batch_does_not_discard_the_others(self):
        """Per-batch fail-open. Letting one bad chunk raise would surface as `judge_failed` and disable
        the whole pass — turning a transient error into 'nothing was redundant'."""
        from src.outcome_balance import make_llm_judge

        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("rate limited")
            return {"pairs": [{"n": 1, "same": True}]}

        import itertools

        texts = [f"question number {i}?" for i in range(30)]
        pairs = list(itertools.combinations(range(30), 2))[:20]
        out = make_llm_judge(texts, model="m", complete=flaky)(pairs)

        assert calls["n"] == 2, "both batches must be attempted"
        assert out, "verdicts from the surviving batch must be kept"


class TestOneJudgeCallPerOutcome:
    """A verdict must depend only on its own outcome's members.

    Found by re-running `--apply`: it cut 8 MORE questions, including *"What is the difference between a
    base model and an instruction-tuned model?"* which the first pass had deliberately kept. The cause was
    not a flaky model — the same 28 pairs of one outcome gave 3 "same" verdicts when batched alongside
    other outcomes' pairs and **0 across 3 trials** when batched alone. Flat batching made the whole pass
    non-idempotent.
    """

    def test_pairs_from_different_outcomes_are_never_batched_together(self):
        batches = []

        def judge(pairs):
            batches.append(list(pairs))
            return []

        texts = HALLUCINATION + WORKFLOW + HUMAN_LOOP
        balance_by_outcome(texts, OUTCOMES, fits=[0.6] * len(texts),
                           approved=[True] * len(texts), judge=judge, min_keep=1)

        assert len(batches) == 3, f"expected one call per outcome with >=2 members, got {len(batches)}"
        # Each call's pairs must all come from a single outcome — i.e. index ranges must not mix.
        groups = [set(range(0, 6)), set(range(6, 9)), set(range(9, 12))]
        for b in batches:
            touched = {i for p in b for i in p}
            assert any(touched <= g for g in groups), f"a batch mixed outcomes: {touched}"

    def test_one_outcome_failing_still_balances_the_others(self):
        """Per-outcome fail-open. A flat call meant one error discarded every verdict."""
        seen = {"n": 0}

        def flaky(pairs):
            seen["n"] += 1
            if seen["n"] == 1:
                raise RuntimeError("rate limited")
            return list(pairs)

        texts = HALLUCINATION + WORKFLOW + HUMAN_LOOP
        res = balance_by_outcome(texts, OUTCOMES, fits=[0.6] * len(texts),
                                 approved=[True] * len(texts), judge=flaky, min_keep=1)
        assert res.judge_failed is True
        assert res.drop, "the outcomes that DID judge must still be balanced"
        assert not (set(res.drop) & set(range(6))), "the failed outcome keeps all its questions"


class TestMajorityVoteProtectsTheDestructivePath:
    """`--apply` writes to the database, so one flapping verdict is permanent. 2 of 3 required."""

    def test_a_single_dissenting_trial_does_not_cut(self):
        from src.outcome_balance import majority

        calls = {"n": 0}

        def flappy(pairs):
            calls["n"] += 1
            return [pairs[0]] if calls["n"] == 1 else []        # "same" once, "different" twice

        assert majority(flappy)([(0, 1)]) == [], "1 of 3 must not be enough to delete a real question"

    def test_two_of_three_is_enough(self):
        from src.outcome_balance import majority

        calls = {"n": 0}

        def mostly(pairs):
            calls["n"] += 1
            return [] if calls["n"] == 2 else [pairs[0]]

        assert majority(mostly)([(0, 1)]) == [(0, 1)]

    def test_the_script_uses_it_only_when_applying(self):
        """Report mode stays single-pass so previewing is cheap; the write path pays 3x."""
        src = (Path(__file__).resolve().parent.parent / "scripts" / "filter_topic_sets.py").read_text()
        assert "if judge and args.apply:" in src and "majority(judge)" in src

    def test_the_pipeline_does_not_pay_for_it(self):
        """`_cap_by_outcome` only trims what a run SHIPS and re-derives it every run, so a wrong verdict
        costs nothing permanent — paying 3x per run to stabilise a reversible decision is not worth it."""
        src = (Path(__file__).resolve().parent.parent / "src" / "tools.py").read_text()
        cap = src[src.index("def _cap_by_outcome"):src.index("def _add_retained")]
        assert "majority" not in cap


class TestTheQuotaIsOptIn:
    """`strict=True` restores the original hard quota. It exists, it is not the default, and the reason
    is measured: on Gen AI Foundations the topic "Pre-trained vs fine-tuned models" holds 14 questions and
    the quota deletes 12, including three distinct "what is the difference between…" asks.
    """

    FINE_TUNING = ["What is the difference between pre-training and fine-tuning?",
                   "What are the different Fine-tuning methods?",
                   "What is the difference between fine-tuning and retrieval augmented generation?",
                   "What is the difference between a base model and an instruction-tuned model?"]

    def test_the_default_keeps_distinct_questions_sharing_a_coarse_outcome(self):
        res = balance_by_outcome(self.FINE_TUNING, ["Pre-trained vs fine-tuned models"],
                                 fits=[0.746, 0.535, 0.468, 0.463], approved=[False] * 4,
                                 cap=2, judge=_judge(), min_keep=1)
        assert res.drop == [], "these are different asks; only a quota would remove them"

    def test_strict_removes_them_which_is_why_it_is_not_the_default(self):
        res = balance_by_outcome(self.FINE_TUNING, ["Pre-trained vs fine-tuned models"],
                                 fits=[0.746, 0.535, 0.468, 0.463], approved=[False] * 4,
                                 cap=2, judge=_judge(), min_keep=1, strict=True)
        assert len(res.keep) == 2 and len(res.drop) == 2

    def test_a_judged_pair_is_removed_even_under_the_cap(self):
        """Two members, cap 2, so no quota applies — the verdict alone must still act. The first version
        only generated pairs for OVER-served outcomes, so 'Explain prompting techniques' and 'How do you
        approach designing an effective prompt?' both shipped."""
        two = ["Explain prompting techniques", "How do you approach designing an effective prompt?"]
        res = balance_by_outcome(two, ["Selecting appropriate prompting techniques"],
                                 fits=[0.714, 0.699], approved=[True, True], cap=2,
                                 judge=_judge((0, 1)), min_keep=1)
        assert len(res.keep) == 1

    def test_an_outcome_under_the_cap_is_untouched(self):
        res = balance_by_outcome(["What is an AI hallucination"], OUTCOMES, fits=[0.6],
                                 approved=[True], cap=2, judge=_judge(), min_keep=1)
        assert res.drop == []

    def test_an_outcome_with_no_questions_is_reported_never_padded(self):
        res = balance_by_outcome(["What is an AI hallucination"], OUTCOMES, fits=[0.6],
                                 approved=[True], cap=2, min_keep=1)
        assert len(res.uncovered) == 2, "the other two outcomes have nothing"
        assert len(res.keep) == 1, "and nothing is invented to fill them"


class TestAnOrphanIsKeptNeverCut:
    """The case that separates a fix from silent data loss.

    These three are verbatim from the No-Code set and match their best interview topic at 0.132-0.173,
    because `interview_topics` under-describes n8n — the known n8n gap in a new place. They are not
    redundant, and a cap that counts them against a ceiling deletes real questions.
    """

    N8N = ["What are nodes in N8N and how are they categorized?",
           "What is the HTTP Request node and when do you use it?",
           "What is the Split In Batches node used for?"]

    def test_questions_matching_no_outcome_survive(self):
        texts = self.N8N + HALLUCINATION
        res = balance_by_outcome(texts, OUTCOMES, fits=[0.3] * 3 + [0.6] * 6,
                                 approved=[False] * 9, cap=2, judge=_judge(), min_keep=1)
        kept = {texts[i] for i in res.keep}
        for q in self.N8N:
            assert q in kept, f"orphan deleted as a duplicate: {q!r}"
        assert len(res.orphans) == 3

    def test_orphans_are_not_counted_against_any_cap(self):
        """Even under the opt-in quota, which is the mode that could delete them."""
        res = balance_by_outcome(self.N8N, OUTCOMES, fits=[0.3] * 3, approved=[False] * 3,
                                 cap=1, judge=_judge(), min_keep=1, strict=True)
        assert len(res.keep) == 3, "three orphans, quota 1, and all three must survive"
        assert res.drop == []

    def test_the_floor_is_what_makes_them_orphans(self):
        """Mutation-guard: with the floor at 0.0 nothing is an orphan and the cap eats them. If this
        stops failing, the orphan protection has been removed."""
        res = balance_by_outcome(self.N8N, OUTCOMES, fits=[0.3] * 3, approved=[False] * 3,
                                 cap=1, orphan_floor=0.0, judge=_judge(), min_keep=1, strict=True)
        assert len(res.orphans) == 0 and len(res.keep) < 3


class TestRankingWithinAnOutcome:
    def test_approved_beats_a_higher_fit_backfilled_question(self):
        """Real fits from the shipped set. 'What are hallucinations in LLMs' is approved at 0.516 and
        must outrank the backfilled 0.719 — a reviewer's decision is newer information than a score."""
        texts = ["What are hallucinations in LLMs",
                 "What is hallucination, and how can it be controlled using prompt engineering?"]
        res = balance_by_outcome(texts, OUTCOMES[:1], fits=[0.516, 0.719],
                                 approved=[True, False], cap=1, min_keep=1, strict=True)
        assert res.keep == [0], "reviewer-approved must win despite the lower fit"

    def test_among_equals_the_higher_fit_wins(self):
        texts = ["What are hallucinations in LLMs", "What is an AI hallucination"]
        res = balance_by_outcome(texts, OUTCOMES[:1], fits=[0.4, 0.8],
                                 approved=[False, False], cap=1, min_keep=1, strict=True)
        assert res.keep == [1]


class TestFailOpen:
    def test_a_judge_that_raises_falls_back_to_cap_by_rank(self):
        def boom(_pairs):
            raise RuntimeError("openrouter 401")

        res = balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6] * 6, approved=[True] * 6,
                                 cap=2, judge=boom, min_keep=1)
        plain = balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6] * 6, approved=[True] * 6,
                                   cap=2, min_keep=1)
        assert res.judge_failed is True
        assert res.keep == plain.keep, "a dead judge must equal no judge, never lose extra questions"

    def test_a_verdict_about_a_pair_we_never_asked_about_is_ignored(self):
        """Same verify-in-code discipline as `_accept_trim` and `_concept_is_absent`."""
        res = balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6] * 6, approved=[True] * 6,
                                 cap=6, judge=_judge((0, 99), (3, 4)), min_keep=1)
        assert len(res.keep) == 5, "only the real pair (3,4) may act; (0,99) is discarded"
        assert res.drop == [4]

    def test_no_outcomes_means_no_capping(self):
        res = balance_by_outcome(HALLUCINATION, [], fits=[0.6] * 6, approved=[True] * 6, cap=2)
        assert res.keep == list(range(6)) and res.drop == []

    def test_empty_input_is_not_an_error(self):
        res = balance_by_outcome([], OUTCOMES, cap=2)
        assert res.keep == [] and res.drop == []

    def test_mismatched_lengths_are_rejected_loudly(self):
        with pytest.raises(ValueError):
            balance_by_outcome(HALLUCINATION, OUTCOMES, fits=[0.6], approved=[True] * 6)


class TestTheMinimumQuestionFloor:
    def test_the_cap_never_takes_a_set_below_min_questions(self):
        """A balanced set too small to run is worse than a slightly repetitive one, which is why
        `remove_question` already refuses at this floor."""
        res = balance_by_outcome(HALLUCINATION, OUTCOMES[:1], fits=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                 approved=[True] * 6, cap=1, min_keep=5, strict=True)
        assert len(res.keep) == 5, f"floor not respected, kept {len(res.keep)}"

    def test_restored_questions_are_the_best_ranked_ones(self):
        res = balance_by_outcome(HALLUCINATION, OUTCOMES[:1], fits=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                 approved=[True] * 6, cap=1, min_keep=3, strict=True)
        assert res.keep == [0, 1, 2], "the floor must restore by rank, not arbitrarily"


# ── Wiring ────────────────────────────────────────────────────────────────────────────────────────────
#
# Asserting "the question isn't in the shipped set" is VACUOUS here, for exactly the reason it was for
# `_drop_hands_on`: `_select_final` trims the pool anyway, so such an assertion passes with the call
# unwired. Assert the REMOVAL RECORD, through the real caller.

# A realistically-spread fixture. With only the 6 hallucination questions the MIN_QUESTIONS floor
# correctly restores 3 and the cap looks broken — the same "fixtures must be realistically sized" trap
# that `tests/test_topic_dupes.py` hit with the 30% cluster cap.
WORKFLOW = ["How do you automate a workflow using an LLM?",
            "What does workflow automation with LLMs involve?",
            "How would you automate a multi-step workflow with an LLM?"]
HUMAN_LOOP = ["Why is a human in the loop needed for an AI system?",
              "How do you design human oversight into an AI system?",
              "When should a human review AI output?"]
SPREAD = HALLUCINATION + WORKFLOW + HUMAN_LOOP      # 12 questions over the 3 OUTCOMES


def _q(content, qid=None, fit=0.6, approved=False):
    q = QuestionDetail(question_id=qid or content[:24], category="GEN_AI", content=content,
                       topic="Gen AI", difficulty="Medium", source="interview_db")
    q.session_fit = fit
    if approved:
        q.retained_status = "approved"
    return q


class TestItRunsFromSubmit:
    """`TestTheCapWorks` below calls `_cap_by_outcome` directly, so it proves the function WORKS, not
    that it RUNS. A mutation check caught that: replacing the call in `tool_submit_question_set` with a
    stub left every one of those tests green.

    Fourth time this class of vacuous test has appeared in this codebase (`_drop_hands_on`,
    `_add_retained`, `_score_unscored_fits`). Only a test through the caller proves reachability.
    """

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
        memory.init_db()
        return memory

    @staticmethod
    def _submit_state(questions):
        from src.agent import AgentState
        from src.data_loader import get_data_store

        st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
        st.session_context = SimpleNamespace(
            session_name="S", session_type="theory_heavy", learning_outcomes=[],
            interview_topics=list(OUTCOMES), key_concepts=[], scope_in=[], scope_out=[],
            matched_kp_ids=[])
        st.questions = {q.question_id: q for q in questions}
        st.config.max_questions = 60          # do not let the trim mask the cap
        return st

    @staticmethod
    def _all_same(monkeypatch):
        """A judge calling every offered pair redundant, so each outcome collapses to one question.
        Stubbing the FACTORY is required: `_cap_by_outcome` builds the judge itself, and stubbing only
        `chat_completion_json` yields an empty reply — i.e. no verdicts and no cuts, which cannot
        distinguish "wired and judged nothing" from "not wired at all"."""
        monkeypatch.setattr("src.outcome_balance.make_llm_judge",
                            lambda *a, **k: (lambda pairs: list(pairs)))

    def test_submit_caps_and_reports_it(self, db, monkeypatch):
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        # Stub the LLM boundary. `_scope_trim`, `_same_thing_pass` and `_syllabus_audit` all read
        # `chat_completion_json`; an empty reply makes each a no-op, leaving the balance as the only actor.
        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})
        self._all_same(monkeypatch)

        qs = [_q(c, qid=f"s{i}", approved=True) for i, c in enumerate(SPREAD)]
        out = tool_submit_question_set(self._submit_state(qs))

        # 12 questions over 3 outcomes -> 1 each = 3, then the MIN_QUESTIONS floor restores 2.
        assert out["outcome_capped"] == 7, (
            f"the balance must run FROM submit, not merely be callable — got {out.get('outcome_capped')}")
        assert out["theory"] == 5
        assert out["outcomes_uncovered"] == [], "all three outcomes are served by this fixture"

    def test_submit_records_the_stage_and_names_the_outcome(self, db, monkeypatch):
        """`scripts/yield_report.py` counts stages and the quality report names this one. A filter that
        shrinks the set silently reads as 'this topic has few questions'."""
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})
        self._all_same(monkeypatch)
        qs = [_q(c, qid=f"s{i}", approved=True) for i, c in enumerate(SPREAD)]
        st = self._submit_state(qs)
        tool_submit_question_set(st)

        capped = [r for r in st.removed if r.get("stage") == "outcome_cap"]
        assert len(capped) == 7
        assert any("Hallucination detection" in r["reason"] for r in capped), (
            "the reason must name the outcome that was already covered")

    def test_submit_reports_uncovered_outcomes(self, db, monkeypatch):
        import src.tools as tools_mod
        from src.tools import tool_submit_question_set

        monkeypatch.setattr(tools_mod, "chat_completion_json", lambda **kw: {})
        qs = [_q(c, qid=f"h{i}", approved=True) for i, c in enumerate(HALLUCINATION)]
        st = self._submit_state(qs)
        out = tool_submit_question_set(st)

        assert out["outcomes_uncovered"] == ["Workflow automation with LLMs",
                                             "Human-in-the-loop AI systems"]
        assert st.uncovered_outcomes == out["outcomes_uncovered"]


class TestTheCapWorks:
    @staticmethod
    def _state(questions):
        from src.agent import AgentState
        from src.data_loader import get_data_store

        st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
        st.session_context = SimpleNamespace(
            session_name="S", session_type="theory_heavy", learning_outcomes=[],
            interview_topics=list(OUTCOMES), key_concepts=[], scope_in=[], scope_out=[],
            matched_kp_ids=[])
        st.questions = {q.question_id: q for q in questions}
        return st

    def test_a_capped_question_is_recorded_with_its_own_stage(self, monkeypatch):
        """`scripts/yield_report.py` counts stages, and the quality report names this one. A filter that
        shrinks the set silently reads as 'this topic has few questions'."""
        import src.tools as tools
        monkeypatch.setattr(tools, "_cap_by_outcome", tools._cap_by_outcome)   # no-op; kept explicit
        monkeypatch.setattr(tools, "make_llm_judge", None, raising=False)

        qs = [_q(c, qid=f"h{i}", approved=True) for i, c in enumerate(SPREAD)]
        st = self._state(qs)
        monkeypatch.setattr("src.outcome_balance.make_llm_judge",
                            lambda *a, **k: (lambda pairs: list(pairs)))
        out = tools._cap_by_outcome(st, qs)

        assert out["removed"] == 7, f"expected the balance to bind, got {out}"
        stages = [r["stage"] for r in st.removed]
        assert stages == ["outcome_cap"] * 7
        assert all("already covered" in r["reason"] for r in st.removed)
        assert len(qs) == 5, "the list handed in must be mutated, as the caller relies on it"

    def test_the_uncovered_outcomes_land_on_the_state(self, monkeypatch):
        monkeypatch.setattr("src.outcome_balance.make_llm_judge",
                            lambda *a, **k: (lambda pairs: []))
        qs = [_q(c, qid=f"h{i}", approved=True) for i, c in enumerate(HALLUCINATION)]
        st = self._state(qs)
        import src.tools as tools
        tools._cap_by_outcome(st, qs)   # judge finds nothing; uncovered must still be reported
        assert st.uncovered_outcomes == ["Workflow automation with LLMs", "Human-in-the-loop AI systems"]

    def test_no_session_context_is_not_an_error(self, monkeypatch):
        import src.tools as tools
        qs = [_q(c, qid=f"h{i}") for i, c in enumerate(HALLUCINATION)]
        st = self._state(qs)
        st.session_context = None
        assert tools._cap_by_outcome(st, qs)["removed"] == 0
        assert len(qs) == 6

    def test_a_context_with_no_interview_topics_caps_nothing(self, monkeypatch):
        import src.tools as tools
        qs = [_q(c, qid=f"h{i}") for i, c in enumerate(HALLUCINATION)]
        st = self._state(qs)
        st.session_context.interview_topics = []
        st.session_context.learning_outcomes = []
        assert tools._cap_by_outcome(st, qs)["removed"] == 0
        assert len(qs) == 6


class TestTheQualityReportNamesIt:
    def test_both_halves_are_reported(self):
        """A capped question and an uncovered outcome are different problems with one cause; the report
        must name both, and neither is gated."""
        from src.agent import AgentState
        from src.data_loader import get_data_store
        from src.pipeline import _build_quality_report

        st = AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())
        st.session_context = SimpleNamespace(
            session_name="S", session_type="theory_heavy", learning_outcomes=[],
            interview_topics=list(OUTCOMES), key_concepts=[], scope_in=[], scope_out=[],
            matched_kp_ids=[])
        st.questions = {q.question_id: q for q in [_q(c, qid=f"h{i}")
                                                   for i, c in enumerate(HALLUCINATION[:2])]}
        st.removed = [{"content": "x", "reason": "Interview topic already covered", "stage": "outcome_cap"}]
        st.uncovered_outcomes = ["Human-in-the-loop AI systems"]

        report = _build_quality_report(st, 0)
        notes = " ".join(report.critique)
        assert "already covers the same" in notes
        assert "judged, not counted" in notes, (
            "the note must not credit a CAP — the pipeline runs strict=False, so counting plays no part")
        assert "Human-in-the-loop AI systems" in notes
        assert "not gated" in notes


class TestTheScriptSharesTheSameFunction:
    """Two call sites with deliberately different consequences must not drift."""

    def test_the_script_imports_the_same_balancer(self):
        src = (Path(__file__).resolve().parent.parent / "scripts" / "filter_topic_sets.py").read_text()
        assert "from src.outcome_balance import balance_by_outcome" in src
        assert "sync_canonical_payload" in src, (
            "a cut that skips this exists in the database and not in Review — that gap has bitten twice")


class TestApplyReachesTheProduct:
    """`--apply` must change what Review renders, not only the table.

    Asserting the table count alone passes in both the broken and the fixed world: `/review/<id>` and the
    Sheets export read the canonical PAYLOAD. Quarantining 15 questions once left four payloads showing
    their old counts, and it has bitten twice.
    """

    @pytest.fixture
    def script(self):
        spec = importlib.util.spec_from_file_location(
            "filter_topic_sets", Path(__file__).resolve().parent.parent / "scripts" / "filter_topic_sets.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["filter_topic_sets"] = m
        spec.loader.exec_module(m)
        return m

    @pytest.fixture
    def db(self, tmp_path, monkeypatch, script):
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
        monkeypatch.setattr(script, "MEMORY_DB", tmp_path / "t.db")
        memory.init_db()
        tk = "topic:cap"
        memory.upsert_topic_questions(
            tk, [{"content": c, "status": "approved", "difficulty": "Medium", "source": "interview_db"}
                 for c in SPREAD])
        memory.save_run_result("canon-1", {"output": {"question_details": [
            {"content": c, "question_id": c[:24]} for c in SPREAD]}})
        memory.set_canonical_run(tk, "canon-1")
        return memory, tk

    def _run(self, script, monkeypatch, apply):
        monkeypatch.setattr(script, "_topic_context", lambda s: SimpleNamespace(
            learning_outcomes=[], interview_topics=list(OUTCOMES), key_concepts=[],
            scope_in=[], scope_out=[]))
        monkeypatch.setattr(script, "_session_profile", lambda s, c: ([], []))
        monkeypatch.setattr(script, "_fits", lambda texts, cur, rm: [0.6] * len(texts))
        # strict + no_judge keeps this offline and deterministic: the plumbing (backup, quarantine,
        # payload re-render) is what is under test here, not the cut criterion.
        return script._cap({"topic:cap": ["S1"]}, ["topic:cap"],
                           SimpleNamespace(topic=None, cap=2, no_judge=True, apply=apply,
                                           strict=True))

    def test_the_payload_shrinks_not_just_the_table(self, db, script, monkeypatch):
        memory, tk = db
        assert len(memory.get_run_result("canon-1")["output"]["question_details"]) == 12

        self._run(script, monkeypatch, apply=True)

        assert len(memory.get_topic_questions(tk)) == 6, "two per outcome kept in the table"
        after = memory.get_run_result("canon-1")["output"]["question_details"]
        assert len(after) == 6, "and the payload Review reads was re-rendered — this is the step that ships"

    def test_report_mode_changes_nothing(self, db, script, monkeypatch):
        memory, tk = db
        self._run(script, monkeypatch, apply=False)
        assert len(memory.get_topic_questions(tk)) == 12
        assert memory.get_quarantined(tk) == []

    def test_the_cuts_are_recoverable(self, db, script, monkeypatch):
        memory, tk = db
        self._run(script, monkeypatch, apply=True)
        q = memory.get_quarantined(tk)
        assert len(q) == 6
        assert all("already covered" in r["reason"] for r in q), (
            "the reason must name WHY, so a recovery decision can be made without re-deriving it")
