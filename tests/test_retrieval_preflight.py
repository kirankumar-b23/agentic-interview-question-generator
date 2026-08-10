"""When the web tier is down, the run stops before spending anything.

Retrieval is not a preliminary to the "real" work — it IS the work. A failed Tavily probe used to only set
`web_search_disabled` and the run carried on bank-only through the relevance judge, the Evaluation agent,
the syllabus audit, the same-thing pass, the outcome balance and up to three quality-gate critiques, all
judging a pool the failure had already decided.

THE COST IS ACCEPTED, NOT ASSUMED. Across 62 persisted runs the bank supplies 75% of shipped questions
(459 of 615), and on a 17-run sample 12 would have cleared MIN_QUESTIONS bank-only. So this guard refuses
runs that would have worked; `REQUIRE_WEB_SEARCH=0` is the way back. `TestTheEscapeHatch` guards that,
because it is the only route to the previous behaviour.

The load-bearing test is `TestNothingIsSpent`. Everything else here would also pass with the probe left in
its old position inside `RetrievalAgent.run` — which runs AFTER `UnderstandingAgent` and its per-session
`chat_completion_json` calls, making "stopped before spending" false.
"""
import pathlib
import tempfile

import pytest

from src.agent import RetrievalUnavailable
from src.models import GenerationConfig


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Submit reads the accumulated set; never touch the real `memory.db`."""
    from src import memory
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "preflight.db")
    memory.init_db()


@pytest.fixture
def probe(monkeypatch):
    """Stub `health_check` with a scripted sequence and count the calls."""
    import src.sources.tavily_search as tav

    calls = []

    def install(*sequence):
        it = iter(sequence)
        last = {"v": sequence[-1]}

        def fake(self):
            calls.append(1)
            try:
                return next(it)
            except StopIteration:
                return last["v"]

        monkeypatch.setattr(tav.TavilyConnector, "health_check", fake, raising=True)
        return calls

    return install


@pytest.fixture
def llm_spy(monkeypatch):
    """Record every LLM call, at BOTH boundaries. Returns the recording list.

    There are two, and covering only one makes the "nothing was spent" assertion a lie that passes:

    * `chat_completion_json` — the direct JSON calls (relevance judge, gate, session understanding,
      scope trim, syllabus audit, same-thing, outcome balance). Every module that does
      `from src.llm_client import chat_completion_json` binds it at import time and needs its own patch.
    * `base_agent.get_client` — the AGENT TOOL LOOPS call `client.chat.completions.create` directly.
      This is the one a mutation check caught: with the pre-flight moved back after the Understanding
      agent, the assertion still passed and only `tests/netguard.py` noticed the 3 real connections.
    """
    import src.agents.base_agent as ba
    import src.llm_client as lc
    import src.session_understanding as su
    import src.tools as tools

    spent: list = []
    monkeypatch.setattr(lc, "chat_completion_json", lambda **kw: (spent.append("json"), {})[1])
    monkeypatch.setattr(tools, "chat_completion_json", lambda **kw: (spent.append("json"), {})[1])
    monkeypatch.setattr(su, "chat_completion_json", lambda **kw: (spent.append("json"), {})[1])

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    spent.append("tool_loop")
                    raise AssertionError("an agent tool loop reached the LLM")

    monkeypatch.setattr(ba, "get_client", lambda: _Fake())
    return spent


def _state():
    from src.agent import AgentState
    from src.data_loader import get_data_store
    return AgentState(config=GenerationConfig(session_names=["S"]), data_store=get_data_store())


def _preflight(st, emit=None):
    from src.pipeline import AgentPipeline
    return AgentPipeline()._tavily_preflight(st, emit or (lambda *a, **k: None))


class TestNothingIsSpent:
    """The whole point of the change, and the only test that proves it.

    A test asserting merely "the run errored" passes with the probe in its old position after the
    Understanding agent — i.e. with the per-session LLM calls already paid for.
    """

    def test_a_failed_preflight_costs_zero_llm_calls(self, monkeypatch, probe, llm_spy):
        import src.pipeline as pl

        spent = llm_spy
        probe((False, "quota", "exceeded your plan's usage limit"))

        result = pl.AgentPipeline().run(
            GenerationConfig(session_names=["Advanced Prompt Engineering"]),
            "run-abort", lambda *a, **k: None)

        assert spent == [], f"{len(spent)} LLM call(s) were spent before stopping"
        assert result.error and "Stopped before spending" in result.error

    def test_the_understanding_agent_never_runs(self, monkeypatch, probe):
        """Positional proof: if the probe were still inside RetrievalAgent, this would have run."""
        import src.pipeline as pl

        ran = []
        monkeypatch.setattr(pl.UnderstandingAgent, "run",
                            lambda self, state, emit: ran.append(1), raising=True)
        probe((False, "auth", "invalid api key"))

        pl.AgentPipeline().run(GenerationConfig(session_names=["S"]), "r", lambda *a, **k: None)
        assert ran == [], "the Understanding agent ran before the web tier was checked"


@pytest.fixture
def no_tool_loop(monkeypatch):
    """Neuter the agent's LLM tool loop. These tests are about the PROBE, and running the real loop
    reaches OpenRouter — `tests/netguard.py` recorded 3 real connections per test, swallowed by the
    fail-open handler so the assertions still passed."""
    from src.agents.base_agent import BaseAgent
    monkeypatch.setattr(BaseAgent, "run", lambda self, state, emit: None, raising=True)


class TestOneProbePerRun:
    def test_the_retrieval_agent_does_not_probe_again(self, monkeypatch, probe, no_tool_loop):
        """A second probe doubles the latency and the API call for information already on the state —
        a silent cost regression that no behavioural assertion would catch."""
        from src.agents import RetrievalAgent

        calls = probe((True, "ok", "1 result"))
        st = _state()
        _preflight(st)
        assert len(calls) == 1 and st.web_probes == 1

        RetrievalAgent().run(st, lambda *a, **k: None)
        assert len(calls) == 1, f"health_check ran {len(calls)} times — it must run once per run"

    def test_an_agent_driven_directly_still_probes(self, probe, no_tool_loop):
        """A unit test or script using RetrievalAgent alone has no pre-flight, so it must self-check
        rather than silently run with `web_status = 'not_run'`."""
        from src.agents import RetrievalAgent

        calls = probe((False, "quota", "limit"))
        st = _state()
        assert st.web_probes == 0
        RetrievalAgent().run(st, lambda *a, **k: None)
        assert len(calls) == 1 and st.web_search_disabled is True


class TestTerminalStatusesAbortWithoutRetry:
    @pytest.mark.parametrize("status,detail", [
        ("no_key", "TAVILY_API_KEY not set"),
        ("auth", "invalid api key"),
        ("quota", "exceeds your plan's usage limit"),
    ])
    def test_no_second_probe(self, probe, status, detail):
        """Nothing will work today, so re-probing is pure latency."""
        calls = probe((False, status, detail))
        st = _state()
        with pytest.raises(RetrievalUnavailable) as exc:
            _preflight(st)
        assert len(calls) == 1, f"{status} must not be retried"
        assert exc.value.status == status
        assert st.web_status == status and st.web_error == detail

    def test_a_healthy_probe_proceeds_untouched(self, probe):
        calls = probe((True, "ok", "3 result(s)"))
        st = _state()
        _preflight(st)                                    # must not raise
        assert len(calls) == 1
        assert st.web_status == "ok" and st.web_search_disabled is False


class TestTransientIsRetriedOnce:
    """One 429 must not be able to kill an 8-topic batch."""

    @pytest.mark.parametrize("status", ["rate", "error"])
    def test_recovering_on_the_retry_enables_web_search(self, probe, status):
        """Asserted as 'web search is ENABLED', not merely 'it did not raise' — a run that continues
        with `web_search_disabled` True is the bank-only outcome this guard exists to prevent."""
        calls = probe((False, status, "429 rate limited"), (True, "ok", "1 result"))
        st = _state()
        _preflight(st)
        assert len(calls) == 2
        assert st.web_status == "ok"
        assert st.web_search_disabled is False, "recovered, so web search must be usable"

    def test_still_failing_on_the_retry_aborts(self, probe):
        calls = probe((False, "rate", "429"), (False, "rate", "429"))
        with pytest.raises(RetrievalUnavailable):
            _preflight(_state())
        assert len(calls) == 2, "exactly one retry, not a loop"

    def test_the_retry_set_is_configured_not_hardcoded(self):
        from src.config import WEB_PREFLIGHT_RETRY_STATUSES
        assert set(WEB_PREFLIGHT_RETRY_STATUSES) == {"rate", "error"}
        assert "quota" not in WEB_PREFLIGHT_RETRY_STATUSES
        assert "auth" not in WEB_PREFLIGHT_RETRY_STATUSES


class TestTheEscapeHatch:
    """`REQUIRE_WEB_SEARCH=0` is the only route back to the previous behaviour, and the guard refuses
    runs that measurably would have worked (12 of 17 on the persisted sample), so it has to work."""

    def test_zero_means_run_bank_only(self, monkeypatch, probe):
        import src.config as cfg
        monkeypatch.setattr(cfg, "REQUIRE_WEB_SEARCH", False)
        probe((False, "quota", "limit"))

        st = _state()
        _preflight(st)                                    # must NOT raise
        assert st.web_search_disabled is True, "web calls must still be skipped"
        assert st.web_status == "quota" and st.web_error == "limit", (
            "the report banner reads these — the failure must stay visible")

    def test_the_default_is_the_guard_on(self):
        from src.config import REQUIRE_WEB_SEARCH
        assert REQUIRE_WEB_SEARCH is True


class TestHowItIsReported:
    def test_the_message_names_the_cause_and_the_way_out(self, probe):
        import src.pipeline as pl

        probe((False, "quota", "exceeded your plan's usage limit"))
        result = pl.AgentPipeline().run(GenerationConfig(session_names=["S"]), "r",
                                        lambda *a, **k: None)
        msg = result.error
        assert "quota" in msg, "the status must be named so this is not mistaken for a bad topic"
        assert "exceeded your plan" in msg
        assert "REQUIRE_WEB_SEARCH=0" in msg, (
            "without the escape hatch this reads as a bug rather than configured policy")

    def test_the_stream_emits_the_cause_then_terminates(self, probe):
        """`main.py` treats `complete`/`error` as terminal; Progress and the batch poller both depend
        on one of them arriving, or the run appears to hang."""
        import src.pipeline as pl

        probe((False, "auth", "invalid api key"))
        events = []
        pl.AgentPipeline().run(GenerationConfig(session_names=["S"]), "r",
                               lambda rid, step, status, detail="", **f: events.append((step, status)))
        steps = [s for s, _ in events]
        assert "tavily_health" in steps, "Progress must show WHY it stopped"
        assert steps[-1] == "error", "the stream must close on a terminal event"

    def test_no_traceback_for_an_expected_stop(self, monkeypatch, probe):
        """It is a policy decision, not a crash — printing a traceback would make an operator hunt a bug."""
        import traceback as tb_mod

        import src.pipeline as pl
        printed = []
        monkeypatch.setattr(tb_mod, "print_exc", lambda *a, **k: printed.append(1))
        probe((False, "quota", "limit"))
        pl.AgentPipeline().run(GenerationConfig(session_names=["S"]), "r", lambda *a, **k: None)
        assert printed == []

    def test_preview_mode_stops_the_same_way(self, probe):
        import src.pipeline as pl

        probe((False, "quota", "limit"))
        result, state = pl.AgentPipeline().run_preview(
            GenerationConfig(session_names=["S"]), "r", lambda *a, **k: None)
        assert result.error and "Stopped before spending" in result.error
        assert result.awaiting_gate is False, "a stopped run must not offer a review screen"


class TestAFailedRunReachesHistory:
    """`main._persist_result` used to return early on ANY error, so a failed run was persisted nowhere.

    A Tavily outage was then indistinguishable in History from never having pressed Generate, and a
    reviewer had no way to tell a retrieval problem from a bad topic.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    def test_the_row_is_recorded_with_the_cause_and_no_payload(self, client, probe, monkeypatch):
        import time

        from src import memory

        probe((False, "quota", "exceeded your plan's usage limit"))
        r = client.post("/api/generate", json={
            "session_names": ["Building Social Media Content Automation Workflow | Part 1"],
            "max_questions": 5})
        assert r.status_code == 200
        rid = r.json()["run_id"]
        # Wait for PERSISTENCE, not for /api/result. `main._run` sets `_results[run_id]` on line 457 and
        # persists on 458, so the endpoint reports the failure before the row exists — polling the
        # endpoint passed this test alone and failed it under full-suite load.
        rows = []
        for _ in range(200):
            rows = [x for x in memory.get_run_history() if x["run_id"] == rid]
            if rows:
                break
            time.sleep(0.05)
        assert rows, "a failed run must still appear in History"
        row = rows[0]
        assert row["question_count"] == 0
        assert "quota" in (row["error"] or ""), "the row must name the cause"
        assert row["session_name"] != "Unknown", (
            "the abort precedes UnderstandingAgent, so the result needs a fallback context or every "
            "failed row reads 'Unknown'")
        assert memory.get_run_result(rid) is None, (
            "no question set exists, so Review must have nothing to open")

    def test_the_api_returns_the_reason_not_a_bare_500(self, client, probe):
        import time

        probe((False, "auth", "invalid api key"))
        rid = client.post("/api/generate", json={
            "session_names": ["Advanced Prompt Engineering"], "max_questions": 5}).json()["run_id"]
        for _ in range(80):
            res = client.get(f"/api/result/{rid}")
            if res.status_code != 409:
                break
            time.sleep(0.05)
        # The React client reads `body.error`, never FastAPI's `detail`.
        assert "auth" in (res.json().get("error") or "")


class TestABatchKeepsGoing:
    """A dead web tier fails every topic, and the worker must still ATTEMPT each one.

    Asserting only "the rows are failed" is not enough: a worker that died on the first topic also leaves
    the rest failed (or queued). The assertion is that every topic was attempted — the same reason
    `test_batch_generate.py::TestRunsAreSequential` asserts non-overlap rather than a call count.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app)

    @pytest.fixture
    def topics(self, monkeypatch):
        import main
        fake = {"Topic Alpha": ["Alpha | Part 1"], "Topic Beta": ["Beta | Part 1"],
                "Topic Gamma": ["Gamma | Part 1"]}
        monkeypatch.setattr(main, "_gen_ai_topics", lambda: fake)
        return fake

    @staticmethod
    def _await(client, batch_id, timeout=20):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = client.get(f"/api/batch/{batch_id}").json()
            if body.get("finished"):
                return body
            time.sleep(0.02)
        raise AssertionError("batch did not finish in time")

    def test_every_topic_is_attempted_and_each_row_says_why(self, client, topics, monkeypatch, probe):
        import main
        import src.pipeline as pl

        monkeypatch.setattr(main, "_persist_result", lambda *a, **k: None)
        attempted = []
        real_run = pl.AgentPipeline.run

        def spy(self, config, run_id=None, emit_fn=None):
            attempted.append(list(config.session_names))
            return real_run(self, config, run_id, emit_fn or (lambda *a, **k: None))

        monkeypatch.setattr(pl.AgentPipeline, "run", spy, raising=True)
        probe((False, "quota", "exceeded your plan's usage limit"))

        r = client.post("/api/generate/batch",
                        json={"course": "gen_ai", "topics": list(topics), "count": 5})
        assert r.status_code == 200
        body = self._await(client, r.json()["batch_id"])

        assert len(attempted) == 3, f"the worker stopped early — only attempted {attempted}"
        assert all(row["status"] == "failed" for row in body["runs"])
        assert all("quota" in (row["error"] or "") for row in body["runs"]), (
            "each row must name the cause, or a dead web tier reads as three bad topics")
