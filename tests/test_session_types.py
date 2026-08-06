"""Per-session-type behaviour: resolution, thresholds, difficulty mix, and label pools.

The premise of all of this is that `session_type` means something. It previously did not — it was
computed, stored, printed into two prompt headers, and acted on nowhere. These tests assert the
*observable* consequences of a type, so a future change that quietly restores the decorative version
fails here rather than passing silently.

They also pin the trap that made the original labels wrong: there were four disagreeing sources of
`session_type`, and the one the eval used was a stale copy of a title-substring heuristic.
"""
import json

import pytest

from src.agent import AgentState
from src.config import (DIFFICULTY_BY_TYPE, EVAL_THRESHOLDS_BY_TYPE, SESSION_TYPES,
                        difficulty_targets, eval_thresholds, normalize_session_type)
from src.data_loader import get_data_store
from src.models import GenerationConfig, QuestionDetail, SessionContext
from src.session_types import fold_types, type_for_run, type_for_session


def _state(session_type="mixed", difficulties=()) -> AgentState:
    st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=10),
                    data_store=get_data_store())
    st.session_context = SessionContext(
        session_name="S", learning_outcomes=["Build a RAG pipeline"], key_concepts=["rag"],
        interview_topics=["RAG"], scope_in=[], scope_out=[], session_type=session_type,
        matched_kp_ids=[], matched_csv_topics=[], prerequisite_kp_chain=[],
        difficulty_distribution={})
    for i, d in enumerate(difficulties):
        st.questions[f"q{i}"] = QuestionDetail(
            question_id=f"q{i}", category="GEN_AI", content=f"Question {i} about RAG chunking",
            topic="Gen AI", source="interview_db", difficulty=d, relevance_score=0.8)
    return st


class TestNormalization:
    @pytest.mark.parametrize("value", SESSION_TYPES)
    def test_known_types_pass_through(self, value):
        assert normalize_session_type(value) == value

    @pytest.mark.parametrize("value", [None, "", "  ", "garbage", "CODE_HEAVY?"])
    def test_unknown_becomes_mixed(self, value):
        """Never raise on an unexpected value — 'mixed' is the neutral default."""
        assert normalize_session_type(value) == "mixed"

    def test_case_and_space_tolerant(self):
        assert normalize_session_type("  Code_Heavy ") == "code_heavy"


class TestTypeResolution:
    def test_known_session_resolves(self):
        assert type_for_session("Building Rest APIs using Flask") == "code_heavy"

    def test_unknown_session_is_none_not_mixed(self):
        """None means 'we don't know'; 'mixed' means 'genuinely both'. Callers need the difference to
        decide whether a per-type score is measurable at all."""
        assert type_for_session("Totally Fabricated Session") is None

    def test_combined_run_name_is_folded(self):
        combined = "Introduction to AI Agents + Building a Learning Path Generator"
        assert type_for_run(combined) in SESSION_TYPES

    def test_fold_is_order_independent(self):
        """Reviewer labels are keyed on ' + '-joined names, and the old exact-string match dropped to
        the global pool on any ordering difference."""
        a = "Introduction to AI Agents + Building a Learning Path Generator"
        b = "Building a Learning Path Generator + Introduction to AI Agents"
        assert type_for_run(a) == type_for_run(b)

    def test_entirely_unknown_run_is_none(self):
        assert type_for_run("Nonexistent A + Nonexistent B") is None

    @pytest.mark.parametrize("types,expected", [
        (["code_heavy", "theory_heavy"], "code_heavy"),      # any code wins
        (["theory_heavy", "theory_heavy"], "theory_heavy"),  # all theory
        (["theory_heavy", "mixed"], "mixed"),
        ([], "mixed"),
    ])
    def test_fold_matches_the_pipeline_rule(self, types, expected):
        """Must match session_understanding's fold, or eval and pipeline disagree about the same run."""
        assert fold_types(types) == expected


class TestDifficultyTargetsReachTheCode:
    """`SessionContext.difficulty_distribution` is the trap here: hardcoded at every construction site
    and read by nothing. These assert the OBSERVABLE target, not that a constant exists."""

    def test_code_heavy_targets_harder_than_theory(self):
        assert difficulty_targets("code_heavy")["Hard"] > difficulty_targets("theory_heavy")["Hard"]
        assert difficulty_targets("theory_heavy")["Easy"] > difficulty_targets("code_heavy")["Easy"]

    @pytest.mark.parametrize("stype", SESSION_TYPES)
    def test_every_target_sums_to_one(self, stype):
        assert sum(DIFFICULTY_BY_TYPE[stype].values()) == pytest.approx(1.0)

    def test_check_difficulty_balance_uses_the_type(self):
        from src.tools import tool_check_difficulty_balance
        mix = ["Easy"] * 5 + ["Medium"] * 3 + ["Hard"] * 2
        theory = tool_check_difficulty_balance(_state("theory_heavy", mix))
        code = tool_check_difficulty_balance(_state("code_heavy", mix))
        assert theory["target_pct"] != code["target_pct"], "the target must differ by type"
        assert theory["session_type"] == "theory_heavy" and code["session_type"] == "code_heavy"

    def test_selection_mix_differs_by_type(self):
        """The end-to-end consequence: the same candidate pool yields a different Easy/Hard split."""
        from src.tools import _select_final
        pool = [QuestionDetail(question_id=f"s{i}", category="GEN_AI", topic="Gen AI",
                               source="interview_db", relevance_score=0.8, difficulty=d,
                               content=f"Question {i} about agent memory and planning")
                for i, d in enumerate(["Easy"] * 6 + ["Medium"] * 6 + ["Hard"] * 6)]
        outcomes = ["Understand agent memory"]
        theory = _select_final(pool, 10, outcomes, session_type="theory_heavy")
        code = _select_final(pool, 10, outcomes, session_type="code_heavy")
        hard = lambda sel: sum(1 for q in sel if q.difficulty == "Hard")   # noqa: E731
        assert hard(code) > hard(theory)


class TestEvalThresholds:
    def test_code_heavy_bars_are_lower_and_documented_as_source_coverage(self):
        """Lower bars reflect that the banks hold almost no implementation questions — not that a
        code session's questions are allowed to be worse."""
        assert eval_thresholds("code_heavy")["coverage"] < eval_thresholds("theory_heavy")["coverage"]
        assert eval_thresholds("code_heavy")["grounding"] < eval_thresholds("theory_heavy")["grounding"]

    @pytest.mark.parametrize("stype", SESSION_TYPES)
    def test_every_type_has_all_three_bars(self, stype):
        assert set(EVAL_THRESHOLDS_BY_TYPE[stype]) == {"accept", "coverage", "grounding"}

    def test_unknown_type_gets_the_neutral_bars(self):
        assert eval_thresholds("nonsense") == eval_thresholds("mixed")


class TestLabelPoolsAreTypeScoped:
    """The quality lever: an implementation question resembles the 'too specific, not conceptual'
    pattern the reviewer established on THEORY material, so pooling the types mis-judges both."""

    LABELS = [
        {"session": "Introduction to Flask", "question": "Write a Flask route that streams a response",
         "decision": "good"},
        {"session": "Introduction to Flask", "question": "Implement request validation for this endpoint",
         "decision": "good"},
        {"session": "Building Rest APIs using Flask", "question": "Debug why this POST returns 415",
         "decision": "good"},
        {"session": "Introduction to Flask", "question": "Describe a REST API you have built",
         "decision": "bad"},
        {"session": "Building Rest APIs using Flask", "question": "Walk me through your best project",
         "decision": "bad"},
        {"session": "Introduction to Flask", "question": "Tell me about yourself", "decision": "bad"},
    ]

    def test_type_scoped_pool_is_reported(self):
        from src.human_agreement import _label_texts
        good, bad, pool = _label_texts(self.LABELS, session=None, session_type="code_heavy")
        assert good and bad
        assert "code_heavy" in pool, "the caller must be told which pool produced the score"

    def test_missing_type_returns_insufficient_rather_than_borrowing(self):
        from src.human_agreement import _label_texts
        good, bad, pool = _label_texts(self.LABELS, session=None, session_type="theory_heavy",
                                       allow_pooled=False)
        assert not (good and bad), "must not borrow code-heavy labels for a theory session"
        assert "insufficient" in pool

    def test_pooled_fallback_is_labelled_when_allowed(self):
        from src.human_agreement import _label_texts
        _good, _bad, pool = _label_texts(self.LABELS, session=None, session_type="theory_heavy",
                                         allow_pooled=True)
        assert pool == "all sessions"

    def test_predict_accept_refuses_rather_than_guessing(self):
        """allow_pooled=False on a type with no labels must yield None, not a number."""
        from src.human_agreement import predict_accept
        assert predict_accept(["What is a REST API?"], self.LABELS,
                              session_type="theory_heavy", allow_pooled=False) is None

    def test_the_shipped_labels_split_by_type(self):
        """Guards the finding that made this work necessary: with the title heuristic these all looked
        theory_heavy; with the LLM-derived source they split. If this collapses to one type again, the
        resolver has regressed to the title-based source."""
        from src import memory
        labels = memory.get_feedback_examples()
        if not labels:
            pytest.skip("no reviewer decisions recorded yet")
        types = {type_for_run(e.get("session") or "") for e in labels if e.get("question")}
        assert len(types - {None}) >= 2, f"expected labels across types, got {types}"


class TestJudgeGuidanceIsTypeSpecific:
    def test_code_and_theory_guidance_differ(self):
        from src.tools import _type_guidance
        code, theory = _type_guidance("code_heavy"), _type_guidance("theory_heavy")
        assert code != theory
        assert "CODE-HEAVY" in code and "THEORY-HEAVY" in theory

    def test_code_guidance_rewards_implementation(self):
        from src.tools import _type_guidance
        assert "debug" in _type_guidance("code_heavy").lower()

    def test_theory_guidance_does_not_reward_implementation_minutiae(self):
        from src.tools import _type_guidance
        assert "minutiae" in _type_guidance("theory_heavy").lower()

    @pytest.mark.parametrize("stype", SESSION_TYPES)
    def test_every_type_still_rejects_experience_questions(self, stype):
        """The reviewer's clearest rule must survive in all three variants."""
        from src.tools import _type_guidance
        assert "past projects" in _type_guidance(stype).lower()

    def test_unknown_type_gets_the_mixed_guidance(self):
        from src.tools import _type_guidance
        assert _type_guidance("nonsense") == _type_guidance("mixed")


class TestEvalSetNoLongerCarriesStaleLabels:
    def test_session_type_and_exemplars_are_gone(self):
        """They were a stale title-heuristic copy and 342 machine-templated fake questions that
        nothing read. Re-adding either invites calibrating on synthetic data."""
        import pathlib
        path = pathlib.Path(__file__).resolve().parent.parent / "eval" / "eval_sets.json"
        data = json.loads(path.read_text())
        assert "format_rules" not in data
        for entry in data["eval_sessions"]:
            assert "session_type" not in entry
            assert "good_questions" not in entry
            assert "bad_questions" not in entry

    def test_eval_sessions_still_have_what_the_harness_uses(self):
        import pathlib
        path = pathlib.Path(__file__).resolve().parent.parent / "eval" / "eval_sets.json"
        data = json.loads(path.read_text())
        assert data["eval_sessions"]
        for entry in data["eval_sessions"]:
            assert entry.get("session_name")


class TestStratifiedSampling:
    def test_draws_across_types_instead_of_the_skewed_majority(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from run_eval import _stratified_sample

        pool = [f"code{i}" for i in range(20)] + ["theory1", "theory2"]
        types = {n: ("code_heavy" if n.startswith("code") else "theory_heavy") for n in pool}
        picked = _stratified_sample(pool, types, 2, seed=0)
        assert {types[p] for p in picked} == {"code_heavy", "theory_heavy"}, \
            "a 20:2 pool must still surface the minority type"

    def test_is_deterministic_for_a_seed(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from run_eval import _stratified_sample

        pool = [f"s{i}" for i in range(10)]
        types = {n: ("code_heavy" if int(n[1:]) % 2 else "theory_heavy") for n in pool}
        assert _stratified_sample(pool, types, 4, 7) == _stratified_sample(pool, types, 4, 7)

    def test_never_returns_more_than_asked_or_available(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
        from run_eval import _stratified_sample

        pool = ["a", "b", "c"]
        types = dict.fromkeys(pool, "mixed")
        assert len(_stratified_sample(pool, types, 10, 0)) == 3
        assert len(_stratified_sample(pool, types, 2, 0)) == 2
