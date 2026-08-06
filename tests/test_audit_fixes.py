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


class TestFixesThatSilentlyDidNotLand:
    """Guards for changes that were REPORTED as done but never applied.

    A string-replace that matches nothing fails silently, and a commit message is not evidence. Each
    of these asserts the observable behaviour rather than the presence of a line of code.
    """

    def test_bank_search_summary_shows_a_real_number(self):
        """It read `remaining_capacity`; the tool returns `bank_remaining`, so it always printed "?"."""
        from src.agent import _summarize_result
        out = _summarize_result("search_question_bank",
                                {"found": 8, "total_accumulated": 64, "bank_remaining": 86})
        assert "86" in out and "?" not in out

    def test_dead_capacity_property_is_gone(self):
        from src.agent import AgentState
        assert not hasattr(AgentState, "remaining_capacity")

    def test_pool_target_knob_is_retired(self):
        """It looked like a live cost guard and nothing read it."""
        import src.config as cfg
        assert not hasattr(cfg, "pool_target")
        assert not hasattr(cfg, "CANDIDATE_POOL_TARGET")

    def test_understand_session_reports_measured_hits_not_an_estimate(self):
        """It multiplied a 5-result probe by the query count and presented that as bank coverage."""
        import inspect
        import src.tools as tools
        src = inspect.getsource(tools.tool_understand_session)
        assert "estimated_bank_question_count" not in src
        assert "bank_probe_hits" in src

    def test_too_few_verdict_does_not_tell_the_agent_to_remove_more(self):
        """The revision prompt said "use remove_question" for every issue, including too-few."""
        from src.agents import EvaluationAgent
        from src.agent import AgentState
        from src.data_loader import get_data_store
        from src.models import GenerationConfig

        st = AgentState(config=GenerationConfig(session_names=["S"], max_questions=8),
                        data_store=get_data_store())
        st.session_context = _state().session_context
        st.revision_notes = [{"id": None, "issue": "too-few", "suggestion": "only 2 questions"}]
        prompt = EvaluationAgent().get_system_prompt(st)
        assert "Do NOT remove anything" in prompt
        assert "reserve" in prompt.lower(), "the agent must be told submit backfills from the reserve"


class TestSeqIsMonotonic:
    """`seq = len(history)` collapsed to a constant once the history was trimmed, and both consumers
    de-duplicate on seq — so a long run would stop delivering events entirely."""

    def test_seq_is_unique_and_increasing_past_the_cap(self, monkeypatch):
        import src.orchestrator as orch
        monkeypatch.setattr(orch, "MAX_HISTORY_EVENTS", 5)
        rid = "seq-monotonic-test"
        try:
            for i in range(12):
                orch._emit(rid, f"s{i}", "done", "")
            seqs = [e["seq"] for e in orch.get_history(rid)]
            assert len(seqs) == len(set(seqs)), f"duplicate seq values: {seqs}"
            assert seqs == sorted(seqs)
        finally:
            orch._emit(rid, "complete", "done", "")
            orch.cleanup_progress(rid)

    def test_replay_after_a_seq_returns_only_newer_events(self, monkeypatch):
        import src.orchestrator as orch
        rid = "seq-replay-test"
        try:
            for i in range(5):
                orch._emit(rid, f"s{i}", "done", "")
            newer = [e["step"] for e in orch.get_history(rid, after_seq=2)]
            assert newer == ["s3", "s4"]
        finally:
            orch._emit(rid, "complete", "done", "")
            orch.cleanup_progress(rid)


class TestGateNeverAbortsTheRun:
    def test_structural_check_failure_becomes_a_gate_issue(self, monkeypatch):
        """An exception in the structural checks used to escape and abort the whole run."""
        import src.agent as agent_mod
        monkeypatch.setattr(agent_mod, "_deterministic_gate_issues",
                            lambda state: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(agent_mod, "chat_completion_json",
                            lambda **kw: {"pass": True, "must_fix": [], "summary": "ok"})
        st = _state(contents=["What are the core components of an AI agent?"])
        verdict = agent_mod._critique_question_set(st)
        assert verdict["pass"] is False
        assert any(i["issue"] == "gate-error" for i in verdict["must_fix"])


class TestFinishedRunsStayReadableForAWhile:
    """The stream's `finally` used to call cleanup_progress(run_id), which deleted the retention window
    the instant a run finished — so reloading the progress page after completion found an empty,
    not-finished-looking run and heartbeated until the 900s stall bound. Reclamation is time-based."""

    def test_a_just_finished_run_survives_prune(self):
        import src.orchestrator as orch
        rid = "retain-now"
        try:
            orch._emit(rid, "complete", "done", "done", questions=8)
            orch.prune_finished()                    # what the stream's finally now calls
            assert orch.is_finished(rid) is True
            assert len(orch.get_history(rid)) == 1, "a reload right after completion must see history"
        finally:
            orch.cleanup_progress(rid)

    def test_an_old_run_is_reclaimed(self):
        import time
        import src.orchestrator as orch
        rid = "retain-old"
        orch._emit(rid, "complete", "done", "done")
        orch.prune_finished(now=time.time() + orch.RETAIN_FINISHED_SECONDS + 1)
        assert orch.is_finished(rid) is False
        assert orch.get_history(rid) == []

    def test_the_stream_no_longer_deletes_one_run_on_close(self):
        """Guard the specific regression: main's stream teardown must not target a single run.

        Checks for a CALL, not the identifier — the explanatory comment mentions it by name.
        """
        import ast
        import inspect
        import textwrap
        import main
        tree = ast.parse(textwrap.dedent(inspect.getsource(main.api_stream)))
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "cleanup_progress" not in called
        assert "prune_finished" in called, "time-based reclamation must still run"


class TestBankQuotaFullStillReportsNumbers:
    """The early return omitted the keys the transcript renders, so a FULL pool was reported as
    "Found 0 relevant (total: 0, bank quota left: ?)"."""

    def test_quota_full_result_has_the_same_keys_as_success(self, monkeypatch):
        from src.agent import _summarize_result
        from src.config import BANK_POOL_CAP
        from src.tools import tool_search_question_bank

        st = _state(contents=["What are the core components of an AI agent?"])
        st.added_by_source["bank"] = BANK_POOL_CAP        # quota exhausted
        result = tool_search_question_bank(st, query="agents")

        assert result["found"] == 0
        assert result["bank_remaining"] == 0
        assert result["total_accumulated"] == 1, "the pool is not empty just because the quota is"
        rendered = _summarize_result("search_question_bank", result)
        assert "?" not in rendered


class TestLightFallbackIsComplete:
    """A system-light visitor with no stored theme got light surfaces but a dark accent gradient, so
    primary buttons stayed dark-themed while the page turned light."""

    def test_prefers_color_scheme_block_sets_every_var_the_explicit_theme_does(self):
        import pathlib
        import re
        css = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "src" / "index.css"
        text = css.read_text()

        def vars_in(block_start: str) -> set:
            i = text.index(block_start)
            body = text[i:text.index("\n}", i)]
            return set(re.findall(r"(--[a-z0-9-]+)\s*:", body))

        explicit = vars_in(':root[data-theme="light"] {')
        fallback = vars_in(":root:not([data-theme]) {")
        # The fallback may legitimately omit shadows; colour tokens must all be present.
        missing = {v for v in explicit - fallback if not v.startswith("--shadow")}
        assert not missing, f"prefers-color-scheme fallback is missing: {sorted(missing)}"
