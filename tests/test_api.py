"""HTTP-level tests for the FastAPI app.

The app was ported from Flask, and the contract the React client depends on is easy to break in a
port. These tests pin the parts that would fail silently in the browser:

  * errors must be `{"error": "..."}`, not FastAPI's default `{"detail": ...}`
    (`frontend/src/lib/api.js` reads `body.error`)
  * unknown non-API paths must serve the SPA so react-router can handle client-side routes
  * unknown /api paths must 404 as JSON rather than returning index.html

No LLM, network, or pipeline run is involved — only routing, validation and response shape.
"""
import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


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

    def test_usage_shape(self, client):
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
