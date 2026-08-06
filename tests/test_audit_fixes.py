"""Regression tests for the audit sweep.

Most of these guard against SILENT wrong behaviour — a near-miss session name adopting another
session's outcomes, a coverage tool that always says "fine", an export column asserting a company that
never asked the question. None of them would have raised an error; they'd just quietly degrade output.
"""
import pytest

from src.agent import AgentState
from src.data_loader import get_data_store
from src.models import GenerationConfig, QuestionDetail, SessionContext
from src.tools import _trim_to_topic


def _state(contents=(), outcomes=("Understand the core components of an AI agent",)) -> AgentState:
    cfg = GenerationConfig(session_names=["Introduction to AI Agents"], max_questions=8)
    st = AgentState(config=cfg, data_store=get_data_store())
    st.session_context = SessionContext(
        session_name="Introduction to AI Agents", learning_outcomes=list(outcomes),
        key_concepts=["agents"], interview_topics=[], scope_in=[], scope_out=[],
        session_type="mixed", matched_kp_ids=[], matched_csv_topics=[],
        prerequisite_kp_chain=[], difficulty_distribution={},
    )
    for i, c in enumerate(contents):
        st.questions[f"q{i}"] = QuestionDetail(
            question_id=f"q{i}", category="GEN_AI", content=c, topic="Gen AI",
            source="interview_db", difficulty="Medium",
        )
    return st


class TestSessionLookupIsNotFuzzy:
    """A substring match let "… | Part 1" inherit "… | Part 2"'s knowledge points, and any short
    custom name that happened to be a substring of a real session adopted its outcomes."""

    def test_exact_name_resolves(self):
        assert get_data_store().get_session_info("Introduction to AI Agents")

    @pytest.mark.parametrize("near_miss", ["Intro", "AI Agents", "Introduction", "agents"])
    def test_substring_does_not_resolve(self, near_miss):
        assert get_data_store().get_session_info(near_miss) is None, \
            f"{near_miss!r} must not silently adopt another session's outcomes"

    def test_case_and_spacing_still_normalize(self):
        """Normalized matching is intentional — only the loose substring match was the bug."""
        ds = get_data_store()
        assert ds.get_session_info("introduction to ai agents") is not None

    def test_unknown_session_is_none(self):
        assert get_data_store().get_session_info("Totally Fabricated Session Name") is None


class TestCoverageAgreesWithTheReport:
    """The tool used substring overlap over the CONCATENATED text of every question, so one question
    mentioning "model" satisfied every outcome containing that word. The agent was told coverage was
    fine while the report scored it low."""

    def test_off_topic_set_is_not_reported_as_covered(self):
        from src.tools import tool_check_outcome_coverage
        st = _state(contents=["What is the difference between a list and a tuple in Python?"],
                    outcomes=("Understand the core components of an AI agent",))
        assert tool_check_outcome_coverage(st)["coverage_pct"] < 1.0

    def test_on_topic_set_is_covered(self):
        from src.tools import tool_check_outcome_coverage
        st = _state(contents=["What are the core components of an AI agent?"])
        assert tool_check_outcome_coverage(st)["coverage_pct"] > 0.0

    def test_tool_and_report_agree(self):
        from src.pipeline import _outcome_coverage
        from src.tools import tool_check_outcome_coverage
        st = _state(contents=["What are the core components of an AI agent?",
                              "Tell me about yourself"])
        assert tool_check_outcome_coverage(st)["coverage_pct"] == pytest.approx(
            round(_outcome_coverage(st), 2), abs=0.01)

    def test_no_outcomes_is_reported_honestly(self):
        from src.tools import tool_check_outcome_coverage
        result = tool_check_outcome_coverage(_state(contents=["x"], outcomes=()))
        assert result["total_outcomes"] == 0 and "note" in result

    def test_empty_set_with_outcomes_is_zero_not_perfect(self):
        """A run that lost its context and produced nothing used to report perfect coverage."""
        from src.pipeline import _outcome_coverage
        assert _outcome_coverage(_state(contents=())) == 0.0


class TestTrimDoesNotMutateSourcedText:
    """This rewrites a real question's text, so it must only ever cut whole trailing clauses."""

    def test_drops_the_off_topic_tail(self):
        out = _trim_to_topic("What is prompt engineering, and how do you deploy a Flask app?",
                             {"prompt"})
        assert out == "What is prompt engineering?"

    def test_does_not_replace_internal_punctuation(self):
        """Re-joining with ". " turned "X, and how does Y?" into "X. and how does Y?" — a silent
        edit to sourced content that then got exported as what the company asked."""
        out = _trim_to_topic("Explain RAG. Then describe your favourite IDE.", {"rag"})
        assert ". and" not in out and out.startswith("Explain RAG")

    def test_fully_on_topic_text_is_returned_verbatim(self):
        text = "What are embeddings and how do embeddings get indexed?"
        assert _trim_to_topic(text, {"embedding"}) == text

    def test_nothing_on_topic_returns_empty(self):
        assert _trim_to_topic("How do you configure Kubernetes autoscaling?", {"prompt"}) == ""

    def test_single_clause_is_never_rewritten(self):
        text = "What is prompt engineering?"
        assert _trim_to_topic(text, {"prompt"}) == text


class TestGithubHarvestIsGated:
    """It was the only harvest path skipping the form gate, and it mislabelled its source as "web"."""

    def test_github_is_a_valid_source_value(self):
        q = QuestionDetail(category="THEORY", content="What is RAG?", topic="Gen AI", source="github")
        assert q.source == "github"

    def test_form_gate_functions_are_module_level(self):
        """They're used by two harvest paths now; a function-local import would silently diverge."""
        import src.tools as tools
        assert callable(tools.is_quality_question) and callable(tools.strip_artifacts)


class TestSheetsExportHonesty:
    def test_source_site_is_not_exported_as_a_company(self):
        """`attribution` falls back to the source site for in-app provenance ("GeeksforGeeks").
        Writing that into asked_in_company asserted a company that never asked the question."""
        q = QuestionDetail(category="GEN_AI", content="What is RAG?", topic="Gen AI", source="web",
                           source_url="https://www.geeksforgeeks.org/llm-interview")
        assert q.attribution == "GeeksforGeeks"          # fine in the app
        assert not (q.asked_in_company or "")           # nothing to claim in the sheet column

    def test_verified_company_is_still_exported(self):
        q = QuestionDetail(category="GEN_AI", content="What is RAG?", topic="Gen AI",
                           source="interview_db", asked_in_company="Anthropic")
        assert (q.asked_in_company or "").upper() == "ANTHROPIC"

    def test_interactive_oauth_is_off_by_default(self):
        """flow.run_local_server blocks on a browser on the SERVER machine — never in a request."""
        import src.sheets_writer as sw
        assert sw.ALLOW_INTERACTIVE_OAUTH is False


class TestRetrievalPromptMatchesAvailableTools:
    def test_prompt_does_not_advertise_disabled_github(self):
        from src.agents import RetrievalAgent
        from src.config import GITHUB_ENABLED
        prompt = RetrievalAgent().get_system_prompt(_state())
        if not GITHUB_ENABLED:
            assert "github" not in prompt.lower(), \
                "the prompt described a tool the agent was not given"

    def test_prompt_source_count_matches_tool_count(self):
        from src.agents import RetrievalAgent
        from src.agents.retrieval_agent import _TOOL_NAMES
        prompt = RetrievalAgent().get_system_prompt(_state())
        expected = "all three sources" if len(_TOOL_NAMES) >= 3 else "both sources"
        assert expected in prompt
