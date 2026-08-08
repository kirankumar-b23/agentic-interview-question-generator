"""HTTP-level tests for the FastAPI app.

The app was ported from Flask, and the contract the React client depends on is easy to break in a
port. These tests pin the parts that would fail silently in the browser:

  * errors must be `{"error": "..."}`, not FastAPI's default `{"detail": ...}`
    (`frontend/src/lib/api.js` reads `body.error`)
  * unknown non-API paths must serve the SPA so react-router can handle client-side routes
  * unknown /api paths must 404 as JSON rather than returning index.html

MOST IMPORTANTLY: `TestGenerateReachesTheHandler` and `TestRejectAll` post VALID bodies. Every earlier
test here posted a body that failed Pydantic validation, so execution never reached a handler body —
which is how a `NameError` in `api_generate` (a missing module-level import) shipped and made every
POST /api/generate a 500 with a green test suite. A validation test is not an endpoint test.

The pipeline itself is stubbed, so no LLM or network is involved.
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture
def no_pipeline(monkeypatch):
    """Let a request reach and complete a handler without running the real pipeline."""
    started = []

    def _fake_start(run_id, target):
        started.append(run_id)          # never actually run the pipeline thread

    monkeypatch.setattr(main, "_start", _fake_start)
    return started


class TestGenerateReachesTheHandler:
    """The regression guard for the P0: a valid body must get all the way through api_generate.

    `raise_server_exceptions=False` so a handler exception surfaces as a 500 we can assert on, rather
    than being re-raised and masking which endpoint broke.
    """

    def test_valid_body_starts_a_run(self, no_pipeline):
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/generate",
                   json={"session_names": ["Introduction to AI Agents"], "max_questions": 6})
        assert r.status_code == 200, f"api_generate raised: {r.text[:300]}"
        assert r.json().get("run_id")
        assert no_pipeline, "the handler must have reached _start"

    def test_custom_topic_only_is_accepted(self, no_pipeline):
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/generate", json={"session_names": [], "custom_topic": "Vector databases"})
        assert r.status_code == 200, r.text[:300]

    def test_model_choice_is_carried_into_the_run(self, no_pipeline):
        """The picked model must reach the run. It is also recorded as the UI default."""
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/generate", json={"session_names": ["Introduction to AI Agents"],
                                          "model": "openai/gpt-4o-mini"})
        assert r.status_code == 200, r.text[:300]
        from src.llm_client import get_active_model
        assert get_active_model() == "openai/gpt-4o-mini"

    def test_preview_mode_is_accepted(self, no_pipeline):
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/generate", json={"session_names": ["Introduction to AI Agents"],
                                          "preview": True})
        assert r.status_code == 200, r.text[:300]


class TestRejectAll:
    """A2 — the plan named this and it had no test. Rejecting everything must export nothing AND bank
    nothing: a banked reject suppresses future candidates as "already approved"."""

    @pytest.fixture
    def staged_run(self, tmp_path, monkeypatch):
        """A finished run in memory, with all persistence pointed at a throwaway database."""
        from src import memory
        from src.models import (CurationMetadata, CuratedOutput, QualityReport, QuestionDetail,
                                SessionContext)

        db = tmp_path / "review.db"
        monkeypatch.setattr(memory, "MEMORY_DB", db)
        monkeypatch.setattr(memory, "_FEEDBACK_EXAMPLES", tmp_path / "feedback.json")
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "rules.md")
        memory.init_db()

        result = main.PipelineResult()
        result.run_id = "reject-all"
        qs = [QuestionDetail(question_id=f"ra{i}", category="GEN_AI", topic="Gen AI",
                             source="interview_db", content=f"What is agent memory {i}?")
              for i in range(3)]
        result.curated_output = CuratedOutput(session_name="S", question_details=qs,
                                              coding_questions=[], code_snippets=[],
                                              metadata=CurationMetadata())
        result.quality_report = QualityReport()
        result.context = SessionContext(
            session_name="S", learning_outcomes=["x"], key_concepts=["y"], scope_in=[], scope_out=[],
            session_type="mixed", matched_kp_ids=[], matched_csv_topics=[],
            prerequisite_kp_chain=[], difficulty_distribution={})
        main._results["reject-all"] = result
        yield db, qs
        main._results.pop("reject-all", None)

    def test_export_is_refused(self, staged_run):
        _db, qs = staged_run
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/approve/reject-all", json={
            "action": "approve", "accepted_ids": [],
            "rejected_feedback": {q.question_id: "off_topic" for q in qs},
            "decisions_sent": True})
        assert r.status_code == 400
        assert "nothing to export" in r.json()["error"].lower()

    def test_rejected_questions_are_not_banked(self, staged_run):
        """The dangerous half: banking them would suppress them from FUTURE runs as approved."""
        db, qs = staged_run
        c = TestClient(main.app, raise_server_exceptions=False)
        c.post("/api/approve/reject-all", json={
            "action": "approve", "accepted_ids": [],
            "rejected_feedback": {q.question_id: "off_topic" for q in qs},
            "decisions_sent": True})
        banked = sqlite3.connect(db).execute("SELECT COUNT(*) FROM question_bank").fetchone()[0]
        assert banked == 0

    def test_accepting_some_still_exports_only_those(self, staged_run, monkeypatch):
        _db, qs = staged_run
        # Stub the export: a real one would try Google OAuth from inside the test.
        import src.sheets_writer as sw
        monkeypatch.setattr(sw, "write_to_sheets", lambda **kw: "https://example.test/sheet")
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/approve/reject-all", json={
            "action": "approve", "accepted_ids": [qs[0].question_id],
            "rejected_feedback": {q.question_id: "off_topic" for q in qs[1:]},
            "decisions_sent": True})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["saved"] == 1

    def test_reject_action_reaches_the_handler(self, staged_run, no_pipeline):
        """Covers the second missing import — the reject branch called get_active_model().
        `no_pipeline` keeps it from spawning a real regeneration thread."""
        _db, qs = staged_run
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/approve/reject-all", json={
            "action": "reject", "accepted_ids": [qs[0].question_id],
            "rejected_feedback": {qs[1].question_id: "off_topic"},
            "decisions_sent": True})
        assert r.status_code == 200, f"reject branch raised: {r.text[:300]}"
        assert r.json()["status"] == "rejected" and r.json().get("run_id")


class TestErrorShape:
    def test_validation_error_uses_the_error_key(self, client):
        """A bad body must not surface as FastAPI's `detail` — the client would show "undefined"."""
        r = client.post("/api/generate", json={"max_questions": "not-a-number"})
        assert r.status_code == 400
        assert "error" in r.json() and "detail" not in r.json()

    def test_validation_error_names_the_offending_field(self, client):
        r = client.post("/api/generate", json={"max_questions": 9999})
        assert "max_questions" in r.json()["error"]

    def test_http_error_uses_the_error_key(self, client):
        r = client.get("/api/result/does-not-exist")
        assert r.status_code == 404
        assert "error" in r.json() and "detail" not in r.json()

    def test_missing_sessions_is_rejected(self, client):
        r = client.post("/api/generate", json={"session_names": []})
        assert r.status_code == 400
        assert "session" in r.json()["error"].lower()

    def test_blank_session_names_are_rejected(self, client):
        r = client.post("/api/generate", json={"session_names": ["", "  "]})
        assert r.status_code == 400

    def test_unknown_review_action_is_rejected(self, client):
        r = client.post("/api/approve/nope", json={"action": "explode"})
        assert r.status_code == 404          # run not found is checked first
        assert "error" in r.json()


class TestReadEndpoints:
    def test_sessions(self, client):
        body = client.get("/api/sessions").json()
        assert isinstance(body["sessions"], list) and body["sessions"]

    def test_topics_defaults_to_gen_ai(self, client):
        assert isinstance(client.get("/api/topics").json()["topics"], dict)

    def test_courses_always_includes_the_builtin(self, client):
        ids = [c["id"] for c in client.get("/api/courses").json()["courses"]]
        assert "gen_ai" in ids

    def test_history_shape(self, client):
        assert isinstance(client.get("/api/history").json()["runs"], list)

    def test_usage_shape(self, client, monkeypatch):
        # /api/usage reads the live OpenRouter balance (main.py: `get_credit_balance() or {}`), so this
        # shape assertion was making a real request on every suite run. None is the supported path.
        import main as main_mod
        monkeypatch.setattr(main_mod, "get_credit_balance", lambda: None)

        totals = client.get("/api/usage").json()["totals"]
        for key in ("runs", "llm_calls", "prompt_tokens", "tokens", "est_cost"):
            assert key in totals


class TestCourseAuthoringValidation:
    def test_missing_required_field_is_a_400_not_a_500(self, client):
        r = client.post("/api/courses/session", json={"topic": "T"})
        assert r.status_code == 400

    def test_blank_reading_material_is_rejected(self, client):
        r = client.post("/api/courses/session", json={
            "course_name": "X", "topic": "T", "session_name": "S", "reading_material": "   "})
        assert r.status_code == 400

    def test_import_requires_a_course_name(self, client):
        assert client.post("/api/courses/import", json={"course_name": ""}).status_code == 400

    def test_import_rejects_markdown_with_no_headings(self, client):
        r = client.post("/api/courses/import",
                        json={"course_name": "Test", "markdown": "no headings here"})
        assert r.status_code == 400
        assert "sessions" in r.json()["error"].lower()


class TestMarkdownParsing:
    def test_hash_is_a_session_when_there_are_no_subheadings(self):
        parsed = main._parse_course_markdown("# Session A\nbody a\n# Session B\nbody b", "Course")
        assert [(t, s) for t, s, _ in parsed] == [("Course", "Session A"), ("Course", "Session B")]

    def test_hash_is_a_topic_when_subheadings_exist(self):
        parsed = main._parse_course_markdown("# Topic\n## Session\nbody", "Course")
        assert parsed == [("Topic", "Session", "body")]

    def test_reading_material_is_captured(self):
        parsed = main._parse_course_markdown("# T\n## S\nline one\nline two", "C")
        assert parsed[0][2] == "line one\nline two"

    def test_empty_markdown_yields_nothing(self):
        assert main._parse_course_markdown("", "C") == []


class TestSpaRouting:
    def test_unknown_api_path_is_json_404(self, client):
        r = client.get("/api/not-a-real-endpoint")
        assert r.status_code == 404
        assert r.json()["error"]

    @pytest.mark.skipif(not main._has_react, reason="frontend/dist not built")
    def test_client_side_route_serves_the_spa(self, client):
        """react-router owns /review/<id>; the server must return index.html, not a 404."""
        r = client.get("/review/some-run-id")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]

    @pytest.mark.skipif(not main._has_react, reason="frontend/dist not built")
    def test_index_is_not_cached(self, client):
        """A rebuilt bundle has new hashed filenames, so a cached index.html would load dead assets."""
        assert "no-store" in client.get("/").headers.get("cache-control", "")

    @pytest.mark.skipif(not main._has_react, reason="frontend/dist not built")
    def test_path_traversal_is_refused(self, client):
        r = client.get("/../../etc/passwd")
        assert "root:" not in r.text


class TestBulkActionsRespectTheExportRule:
    """Tier 3's bulk accept/reset must not be able to re-open the reject-all hole.

    The client always sends `decisions_sent: true`, so a bulk-cleared set (nothing accepted) is still
    an explicit decision and must be refused rather than exporting everything. These assert the
    SERVER contract that guarantees it, whatever the UI does.
    """

    def test_explicit_empty_acceptance_is_still_refused(self):
        """What a bulk "Reset" followed by Export would send."""
        body = main.ReviewRequest(action="approve", accepted_ids=[], rejected_feedback={},
                                  decisions_sent=True)
        assert body.has_explicit_decisions is True

    def test_a_client_that_sends_nothing_is_treated_as_no_decisions(self):
        """Backwards compatibility: an old client with no flag and no ids made no decisions."""
        body = main.ReviewRequest(action="approve")
        assert body.has_explicit_decisions is False

    def test_rejections_alone_count_as_explicit_decisions(self):
        body = main.ReviewRequest(action="approve", rejected_feedback={"q1": "off_topic"})
        assert body.has_explicit_decisions is True

    def test_bulk_accepted_ids_are_honoured(self, staged_bulk_run, monkeypatch):
        """A bulk accept-above-fit sends many ids at once; all of them must export."""
        import src.sheets_writer as sw
        monkeypatch.setattr(sw, "write_to_sheets", lambda **kw: "https://example.test/s")
        qs = staged_bulk_run
        c = TestClient(main.app, raise_server_exceptions=False)
        r = c.post("/api/approve/bulk-run", json={
            "action": "approve",
            "accepted_ids": [q.question_id for q in qs[:2]],
            "rejected_feedback": {qs[2].question_id: "too_generic"},
            "decisions_sent": True})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["saved"] == 2

    @pytest.fixture
    def staged_bulk_run(self, tmp_path, monkeypatch):
        from src import memory
        from src.models import (CurationMetadata, CuratedOutput, QualityReport, QuestionDetail,
                                SessionContext)
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "bulk.db")
        monkeypatch.setattr(memory, "_FEEDBACK_EXAMPLES", tmp_path / "fb.json")
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "rules.md")
        memory.init_db()

        result = main.PipelineResult()
        result.run_id = "bulk-run"
        qs = [QuestionDetail(question_id=f"bk{i}", category="GEN_AI", topic="Gen AI",
                             source="interview_db", content=f"What is agent planning {i}?",
                             session_fit=0.8 - i * 0.2)
              for i in range(3)]
        result.curated_output = CuratedOutput(session_name="S", question_details=qs,
                                              coding_questions=[], code_snippets=[],
                                              metadata=CurationMetadata())
        result.quality_report = QualityReport()
        result.context = SessionContext(
            session_name="S", learning_outcomes=["x"], key_concepts=["y"], scope_in=[], scope_out=[],
            session_type="mixed", matched_kp_ids=[], matched_csv_topics=[],
            prerequisite_kp_chain=[], difficulty_distribution={})
        main._results["bulk-run"] = result
        yield qs
        main._results.pop("bulk-run", None)
