"""Multi-topic batches: one run per topic, executed sequentially, continuing past a failure.

Picking N topics queues N independent pipeline runs rather than one merged run, because review,
approve, rejection feedback and learned rules are all keyed by run_id and each run gets its own
spreadsheet. Merging would give one gate verdict and one all-or-nothing approve for every topic.

`run_pipeline` is stubbed throughout — these tests must cost no API credit, which the conftest network
guard enforces anyway.

These POST **valid** bodies. `tests/test_api.py` documents why that matters: every earlier endpoint test
here posted a body that failed Pydantic validation, so execution never reached the handler, and a
NameError in `api_generate` shipped while the suite stayed green.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def topics(monkeypatch):
    """Three known topics, so the test does not depend on the shipped course structure."""
    fake = {
        "Topic Alpha": ["Alpha | Part 1", "Alpha | Part 2"],
        "Topic Beta": ["Beta | Part 1"],
        "Topic Gamma": ["Gamma | Part 1", "Gamma | Part 2", "Gamma | Part 3"],
    }
    monkeypatch.setattr(main, "_gen_ai_topics", lambda: fake)
    return fake


def _result(n_questions=6, verdict="pass", error=None):
    """Minimal stand-in for a PipelineResult, shaped only as far as the batch worker reads it."""
    from types import SimpleNamespace
    if error:
        return SimpleNamespace(error=error, curated_output=None, quality_report=None, context=None)
    return SimpleNamespace(
        error=None,
        curated_output=SimpleNamespace(question_details=[object()] * n_questions, coding_questions=[]),
        quality_report=SimpleNamespace(pass_fail=verdict, composite_score=0.7, loops_used=1,
                                       api_usage={}),
        context=SimpleNamespace(session_name="S"))


def _stub_pipeline(monkeypatch, behaviour):
    """Record every call in order and return whatever `behaviour(session_names)` decides."""
    calls = []
    monkeypatch.setattr(main, "_persist_result", lambda *a, **k: None)

    def fake(config, run_id=None):
        calls.append(list(config.session_names))
        return behaviour(list(config.session_names))

    monkeypatch.setattr(main, "run_pipeline", fake)
    return calls


def _await_batch(client, batch_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/batch/{batch_id}").json()
        if body.get("finished"):
            return body
        time.sleep(0.02)
    raise AssertionError("batch did not finish in time")


class TestOneRunPerTopic:
    def test_three_topics_queue_three_runs(self, client, topics, monkeypatch):
        _stub_pipeline(monkeypatch, lambda s: _result())
        r = client.post("/api/generate/batch",
                        json={"topics": list(topics), "max_questions": 8, "course": "gen_ai"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["runs"]) == 3
        assert len({x["run_id"] for x in body["runs"]}) == 3, "run_ids must be distinct"
        assert [x["topic"] for x in body["runs"]] == list(topics), "order must be preserved"
        # Each run carries THAT topic's sessions — not the union.
        assert body["runs"][0]["sessions"] == topics["Topic Alpha"]
        _await_batch(client, body["batch_id"])

    def test_each_run_gets_only_its_own_topics_sessions(self, client, topics, monkeypatch):
        """The whole point of not merging: one SessionContext per topic."""
        calls = _stub_pipeline(monkeypatch, lambda s: _result())
        r = client.post("/api/generate/batch", json={"topics": ["Topic Alpha", "Topic Beta"]})
        _await_batch(client, r.json()["batch_id"])
        assert calls == [topics["Topic Alpha"], topics["Topic Beta"]]

    def test_a_repeated_topic_runs_once(self, client, topics, monkeypatch):
        calls = _stub_pipeline(monkeypatch, lambda s: _result())
        r = client.post("/api/generate/batch",
                        json={"topics": ["Topic Beta", "Topic Beta", " Topic Beta "]})
        _await_batch(client, r.json()["batch_id"])
        assert len(calls) == 1


class TestRunsAreSequential:
    def test_no_two_runs_overlap(self, client, topics, monkeypatch):
        """Sequential, not parallel: a batch is N full pipelines, and memory.db is SQLite.

        Asserts OVERLAP, not call count — a concurrent worker would still produce 3 calls.
        """
        active = {"now": 0, "max": 0}
        lock = threading.Lock()

        def behaviour(_sessions):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)                      # long enough for an overlap to be observable
            with lock:
                active["now"] -= 1
            return _result()

        _stub_pipeline(monkeypatch, behaviour)
        r = client.post("/api/generate/batch", json={"topics": list(topics)})
        _await_batch(client, r.json()["batch_id"])
        assert active["max"] == 1, f"runs overlapped (max concurrent = {active['max']})"


class TestOneFailureDoesNotStopTheBatch:
    def test_a_middle_topic_failing_still_leaves_the_rest_running(self, client, topics, monkeypatch):
        def behaviour(sessions):
            if sessions == topics["Topic Beta"]:
                raise RuntimeError("provider exploded")
            return _result()

        calls = _stub_pipeline(monkeypatch, behaviour)
        r = client.post("/api/generate/batch", json={"topics": list(topics)})
        body = _await_batch(client, r.json()["batch_id"])

        assert len(calls) == 3, "the batch must attempt every topic"
        by_topic = {x["topic"]: x for x in body["runs"]}
        assert by_topic["Topic Beta"]["status"] == "failed"
        assert "provider exploded" in by_topic["Topic Beta"]["error"]
        assert by_topic["Topic Alpha"]["status"] == "done"
        assert by_topic["Topic Gamma"]["status"] == "done", "a later topic must still produce a set"

    def test_a_pipeline_level_error_is_reported_not_raised(self, client, topics, monkeypatch):
        """`result.error` is the pipeline's own 'this run died' channel, distinct from an exception."""
        _stub_pipeline(monkeypatch, lambda s: _result(error="API key invalid"))
        r = client.post("/api/generate/batch", json={"topics": ["Topic Alpha"]})
        body = _await_batch(client, r.json()["batch_id"])
        assert body["runs"][0]["status"] == "failed"
        assert "API key invalid" in body["runs"][0]["error"]

    def test_the_verdict_and_count_are_reported_per_topic(self, client, topics, monkeypatch):
        _stub_pipeline(monkeypatch, lambda s: _result(n_questions=9, verdict="fail"))
        r = client.post("/api/generate/batch", json={"topics": ["Topic Beta"]})
        body = _await_batch(client, r.json()["batch_id"])
        assert body["runs"][0]["question_count"] == 9
        assert body["runs"][0]["verdict"] == "fail"


class TestErrorContract:
    def test_no_topics_is_a_400_with_the_error_key(self, client, topics):
        r = client.post("/api/generate/batch", json={"topics": []})
        assert r.status_code == 400
        assert "error" in r.json(), "the React client reads body.error, not body.detail"

    def test_an_unknown_topic_is_refused_and_named(self, client, topics):
        r = client.post("/api/generate/batch", json={"topics": ["Topic Alpha", "Nope"]})
        assert r.status_code == 400
        assert "Nope" in r.json()["error"]

    def test_nothing_is_queued_when_one_topic_is_unknown(self, client, topics, monkeypatch):
        """All-or-nothing validation: a half-started batch would silently spend credit."""
        calls = _stub_pipeline(monkeypatch, lambda s: _result())
        client.post("/api/generate/batch", json={"topics": ["Topic Alpha", "Nope"]})
        time.sleep(0.1)
        assert calls == []

    def test_an_unknown_batch_id_is_404(self, client):
        assert client.get("/api/batch/does-not-exist").status_code == 404


class TestStatusNeverBlocks:
    def test_polling_returns_immediately_while_runs_are_in_flight(self, client, topics, monkeypatch):
        """/api/result already 409s in flight because polling tabs starved the threadpool."""
        _stub_pipeline(monkeypatch, lambda s: (time.sleep(0.3), _result())[1])
        r = client.post("/api/generate/batch", json={"topics": list(topics)})
        batch_id = r.json()["batch_id"]

        start = time.time()
        body = client.get(f"/api/batch/{batch_id}").json()
        assert time.time() - start < 0.2, "status must not join the worker"
        assert body["finished"] is False
        assert {x["status"] for x in body["runs"]} <= {"queued", "running", "done", "failed"}
        _await_batch(client, batch_id, timeout=15)


class TestBatchIdIsPersisted:
    def test_a_run_records_the_batch_it_belonged_to(self, tmp_path, monkeypatch):
        """The registry is bounded and pruned, so the batch view needs a durable copy."""
        from src import memory

        db = tmp_path / "m.db"
        monkeypatch.setattr(memory, "MEMORY_DB", db)
        memory.init_db()
        memory.save_run(run_id="r1", session_name="Alpha | Part 1", question_count=6,
                        composite_score=0.7, loops_used=1, batch_id="batch-xyz")
        memory.save_run(run_id="r2", session_name="Beta | Part 1", question_count=4,
                        composite_score=0.6, loops_used=1, batch_id="batch-xyz")
        memory.save_run(run_id="solo", session_name="Other", question_count=5,
                        composite_score=0.5, loops_used=1)

        rows = memory.get_batch_runs("batch-xyz")
        assert [r["run_id"] for r in rows] == ["r1", "r2"]
        assert "solo" not in {r["run_id"] for r in rows}, "a single run must not join a batch"
        assert memory.get_batch_runs("nope") == []

    def test_history_exposes_batch_id_so_the_ui_can_group(self, tmp_path, monkeypatch):
        from src import memory

        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m2.db")
        memory.init_db()
        memory.save_run(run_id="r1", session_name="A", question_count=1, composite_score=0.1,
                        loops_used=0, batch_id="b1")
        assert memory.get_run_history()[0]["batch_id"] == "b1"
