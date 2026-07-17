"""Flask web app — JSON API serving the Questor React SPA (frontend/dist/)."""

import json
import os
import uuid
import threading
from flask import Flask, request, Response, send_from_directory, jsonify
from src.orchestrator import (
    run_pipeline, run_preview_pipeline, finalize_pipeline,
    get_progress_queue, cleanup_progress,
)
from src.agent import PipelineResult
from src.models import GenerationConfig
from src.data_loader import get_data_store
from src import memory

# Serve React build from frontend/dist/ when it exists
REACT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
_has_react = os.path.isdir(REACT_DIST)

app = Flask(__name__,
            static_folder=REACT_DIST if _has_react else "static",
            static_url_path="")
app.secret_key = "iqg-dev-secret-key-change-in-production"

# In-memory store for pipeline results (keyed by run_id)
_results: dict[str, PipelineResult] = {}
# Track running pipelines
_running: dict[str, threading.Thread] = {}
# TESTING: preview mode — retained AgentState between the pick and gate phases
_preview_states: dict = {}


def _payload(result: "PipelineResult", run_id: str) -> dict:
    """Build the /api/result response shape from a PipelineResult."""
    ctx = result.context
    return {
        "run_id": run_id,
        "context": ctx.model_dump() if ctx else None,
        "output": result.curated_output.model_dump() if result.curated_output else None,
        "report": result.quality_report.model_dump() if result.quality_report else None,
        "awaiting_gate": getattr(result, "awaiting_gate", False),  # TESTING: preview mode
        "removed": getattr(result, "removed", []),  # rejected questions + reasons
    }


def _persist_result(run_id: str, result: "PipelineResult"):
    """Persist a completed run so Review + re-export survive server restarts,
    and surface it in History immediately (approved=0)."""
    try:
        if not result or result.error or not result.curated_output:
            return
        memory.save_run_result(run_id, _payload(result, run_id))
        total_q = (len(result.curated_output.question_details)
                   + len(result.curated_output.coding_questions))
        memory.save_run(
            run_id=run_id,
            session_name=result.context.session_name if result.context else "Unknown",
            question_count=total_q,
            composite_score=result.quality_report.composite_score if result.quality_report else 0,
            loops_used=result.quality_report.loops_used if result.quality_report else 0,
            approved=False,
            api_usage=dict(result.quality_report.api_usage) if result.quality_report else None,
        )
    except Exception:
        pass


def _load_result(run_id: str) -> "PipelineResult | None":
    """In-memory result, or reconstruct one from the persisted payload."""
    result = _results.get(run_id)
    if result:
        return result
    payload = memory.get_run_result(run_id)
    if not payload:
        return None
    from src.models import CuratedOutput, SessionContext, QualityReport
    r = PipelineResult()
    r.run_id = run_id
    r.context = SessionContext.model_validate(payload["context"]) if payload.get("context") else None
    r.curated_output = CuratedOutput.model_validate(payload["output"]) if payload.get("output") else None
    r.quality_report = QualityReport.model_validate(payload["report"]) if payload.get("report") else None
    return r


# ── JSON API ─────────────────────────────────────────────────────────────────

@app.route("/api/meta")
def api_meta():
    """Runtime info for the UI: active model, selectable models, credit balance, bank size."""
    from src.config import MODEL_OPTIONS
    from src.llm_client import get_credit_balance, get_active_model
    try:
        from src.question_bank import get_retriever
        bank_count = get_retriever().get_stats().get("total", 0)
    except Exception:
        bank_count = 0
    return jsonify({
        "model": get_active_model(),
        "models": MODEL_OPTIONS,
        "credits": get_credit_balance(),
        "bank_count": bank_count,
    })


@app.route("/api/sessions")
def api_sessions():
    data_store = get_data_store()
    return jsonify({"sessions": data_store.get_session_names()})


def _gen_ai_topics():
    """Built-in Gen AI course topics (the existing flat course_structure.json)."""
    path = os.path.join(os.path.dirname(__file__), "data", "course_structure.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@app.route("/api/topics")
def api_topics():
    course = request.args.get("course", "gen_ai")
    if course and course != "gen_ai":
        return jsonify({"topics": memory.get_course_topics(course)})
    return jsonify({"topics": _gen_ai_topics()})


@app.route("/api/courses")
def api_courses():
    """List selectable courses: the built-in Gen AI course + any user-added ones."""
    courses = [{"id": "gen_ai", "name": "Gen AI", "category": "GEN_AI",
                "course_type": "mixed", "builtin": True}]
    for c in memory.get_courses():
        courses.append({"id": c["course_id"], "name": c["name"], "category": c["category"],
                        "course_type": c.get("course_type", "mixed"), "builtin": False})
    return jsonify({"courses": courses})


def _slugify(name: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_") or "course"


@app.route("/api/courses/session", methods=["POST"])
def api_add_course_session():
    """Add a single session (create the course if new)."""
    b = request.get_json(force=True) or {}
    name = (b.get("course_name") or "").strip()
    course_id = (b.get("course_id") or "").strip() or _slugify(name)
    topic = (b.get("topic") or "").strip()
    session_name = (b.get("session_name") or "").strip()
    reading = (b.get("reading_material") or "").strip()
    if not (course_id and topic and session_name and reading):
        return jsonify({"error": "course, topic, session name and reading material are required"}), 400
    category = (b.get("category") or _slugify(name).upper() or "COURSE").upper()
    course_type = b.get("course_type") or "mixed"
    if not memory.get_course(course_id):
        memory.add_course(course_id, name or course_id, category, course_type)
    kps = b.get("kps") or None
    memory.add_course_session(course_id, topic, session_name, reading,
                              b.get("session_type"), kps)
    return jsonify({"course_id": course_id, "status": "added"})


@app.route("/api/courses/import", methods=["POST"])
def api_import_course():
    """Bulk-add a course from a Markdown blob: '#'=topic, '##'=session (body=reading material)."""
    b = request.get_json(force=True) or {}
    name = (b.get("course_name") or "").strip()
    if not name:
        return jsonify({"error": "course name is required"}), 400
    markdown = b.get("markdown") or ""
    parsed = _parse_course_markdown(markdown, default_topic=name)
    if not parsed:
        return jsonify({"error": "No sessions found — use '# Topic' and '## Session' headings"}), 400
    course_id = _slugify(name)
    category = (b.get("category") or _slugify(name).upper()).upper()
    course_type = b.get("course_type") or "mixed"
    memory.add_course(course_id, name, category, course_type)
    for topic, session, reading in parsed:
        memory.add_course_session(course_id, topic, session, reading, None, None)
    return jsonify({"course_id": course_id, "sessions": len(parsed),
                    "topics": len({t for t, _, _ in parsed}), "status": "imported"})


@app.route("/api/courses/<course_id>", methods=["DELETE"])
def api_delete_course(course_id):
    memory.delete_course(course_id)
    return jsonify({"status": "deleted", "course_id": course_id})


def _parse_course_markdown(md: str, default_topic: str):
    """Return [(topic, session, reading_material)]. '#'=topic, '##'=session.
    If only '#' headings exist, each '#' is a session under default_topic."""
    import re as _re
    lines = md.splitlines()
    # Detect whether any '##' session headings exist.
    has_sub = any(_re.match(r"^##\s+\S", ln) for ln in lines)
    out = []
    cur_topic = default_topic
    cur_session = None
    buf = []

    def _flush():
        if cur_session is not None:
            out.append((cur_topic, cur_session, "\n".join(buf).strip()))

    for ln in lines:
        m2 = _re.match(r"^##\s+(.*)$", ln)
        m1 = _re.match(r"^#\s+(.*)$", ln)
        if has_sub and m1 and not m2:
            _flush(); cur_session = None; buf = []
            cur_topic = m1.group(1).strip() or default_topic
        elif has_sub and m2:
            _flush(); buf = []
            cur_session = m2.group(1).strip()
        elif not has_sub and m1:
            _flush(); buf = []
            cur_session = m1.group(1).strip()
        else:
            buf.append(ln)
    _flush()
    return [(t, s, r) for (t, s, r) in out if s]


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True) or {}
    session_names = body.get("session_names", [])
    max_questions = int(body.get("max_questions", 12))
    custom_topic = body.get("custom_topic", "").strip()
    model = (body.get("model") or "").strip() or None
    preview = bool(body.get("preview", False))  # TESTING: preview mode
    category = (body.get("category") or "GEN_AI").strip().upper().replace(" ", "_")
    course_type = body.get("course_type") or None

    if custom_topic:
        session_names.append(custom_topic)
    if not session_names:
        return jsonify({"error": "No sessions provided"}), 400

    config = GenerationConfig(
        session_names=session_names,
        max_questions=min(max_questions, 15),
        model=model,
        preview=preview,
        category=category,
        course_type=course_type,
    )
    run_id = str(uuid.uuid4())
    get_progress_queue(run_id)

    def _run():
        from src.llm_client import set_active_model
        set_active_model(config.model)
        if config.preview:  # TESTING: pause after picking, before the gate
            result, state = run_preview_pipeline(config, run_id=run_id)
            _results[run_id] = result
            if not result.error:
                _preview_states[run_id] = state
            return
        result = run_pipeline(config, run_id=run_id)
        _results[run_id] = result
        _persist_result(run_id, result)

    t = threading.Thread(target=_run, daemon=True)
    _running[run_id] = t
    t.start()

    return jsonify({"run_id": run_id})


@app.route("/api/stream/<run_id>")
def api_stream(run_id: str):
    q = get_progress_queue(run_id)

    def generate_events():
        while True:
            try:
                event = q.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("step") in ("complete", "error"):
                    break
            except Exception:
                yield f"data: {json.dumps({'step': 'timeout', 'status': 'error', 'detail': 'Timeout'})}\n\n"
                break
        cleanup_progress(run_id)

    return Response(generate_events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/result/<run_id>")
def api_result(run_id: str):
    thread = _running.get(run_id)
    if thread and thread.is_alive():
        thread.join(timeout=300)

    from src.data_loader import get_topic_for_session

    def _with_topic(p):
        sess = (p.get("context") or {}).get("session_name", "") or ""
        p["topic"] = get_topic_for_session(sess)
        return p

    result = _results.get(run_id)
    if not result:
        # Fall back to the persisted run (survives server restarts / history).
        payload = memory.get_run_result(run_id)
        if payload:
            return jsonify(_with_topic(payload))
        return jsonify({"error": "Run not found or still processing"}), 404
    if result.error:
        return jsonify({"error": result.error}), 500

    return jsonify(_with_topic(_payload(result, run_id)))


# ── TESTING: preview mode — resume a paused run through the quality gate ──────
@app.route("/api/proceed/<run_id>", methods=["POST"])
def api_proceed(run_id: str):
    state = _preview_states.pop(run_id, None)
    if state is None:
        return jsonify({"error": "No preview run to proceed (expired or already finalized)"}), 404
    get_progress_queue(run_id)  # fresh queue for the finalize phase

    def _finalize():
        from src.llm_client import set_active_model
        set_active_model(state.config.model)
        result = finalize_pipeline(state, run_id)
        _results[run_id] = result
        _persist_result(run_id, result)

    t = threading.Thread(target=_finalize, daemon=True)
    _running[run_id] = t
    t.start()
    return jsonify({"run_id": run_id})


@app.route("/api/approve/<run_id>", methods=["POST"])
def api_approve(run_id: str):
    result = _load_result(run_id)
    if not result:
        return jsonify({"error": "Run not found"}), 404

    body = request.get_json(force=True) or {}
    action = body.get("action", "approve")
    accepted_ids = body.get("accepted_ids", [])
    rejected_feedback = body.get("rejected_feedback", {})

    if action == "approve":
        if accepted_ids:
            result.curated_output.question_details = [
                q for q in result.curated_output.question_details
                if q.question_id in accepted_ids
            ]
            result.curated_output.coding_questions = [
                q for q in result.curated_output.coding_questions
                if q.id in accepted_ids
            ]
        result.approved = True
        total_q = len(result.curated_output.question_details) + len(result.curated_output.coding_questions)
        # Save accepted questions to cross-run bank for future dedup
        try:
            session_name = result.context.session_name if result.context else "Unknown"
            accepted_set = set(accepted_ids) if accepted_ids else None
            for q in result.curated_output.question_details:
                if accepted_set is None or q.question_id in accepted_set:
                    memory.save_question_to_bank(q.question_id, session_name, q.content, q.source)
        except Exception:
            pass
        try:
            memory.save_run(
                run_id=run_id,
                session_name=result.context.session_name,
                question_count=total_q,
                composite_score=result.quality_report.composite_score if result.quality_report else 0,
                loops_used=result.quality_report.loops_used if result.quality_report else 0,
                approved=True,
                api_usage=dict(result.quality_report.api_usage) if result.quality_report else None,
            )
        except Exception:
            pass
        # Export to Google Sheets
        sheet_url = None
        sheet_error = None
        try:
            from src.sheets_writer import write_to_sheets
            session_name = result.context.session_name if result.context else "Unknown"
            sheet_url = write_to_sheets(
                output=result.curated_output,
                report=result.quality_report,
                session_name=session_name,
                run_id=run_id,
                category=getattr(result, "category", "GEN_AI"),
            )
        except Exception as e:
            sheet_error = str(e)
        resp = {"status": "approved", "saved": total_q}
        if sheet_url:
            resp["sheet_url"] = sheet_url
        if sheet_error:
            resp["sheet_error"] = sheet_error
        return jsonify(resp)

    elif action == "reject":
        # Distil rejection reasons into learned rules (max 5 per run)
        reasons = [v for v in rejected_feedback.values() if isinstance(v, str) and v.strip()]
        for reason in reasons[:5]:
            try:
                rule = memory.distill_rule(
                    result.context.session_name if result.context else "Unknown", reason)
                if rule:
                    memory.append_learned_rule(rule)
            except Exception:
                pass

        session_names = result.context.session_name.split(" + ")
        from src.llm_client import get_active_model
        config = GenerationConfig(session_names=session_names, max_questions=15,
                                  model=get_active_model())
        try:
            conn = memory.get_connection()
            conn.execute("DELETE FROM session_resolutions WHERE session_name = ?",
                        (result.context.session_name,))
            conn.commit()
            conn.close()
        except Exception:
            pass

        new_run_id = str(uuid.uuid4())
        get_progress_queue(new_run_id)

        def _rerun():
            from src.llm_client import set_active_model
            set_active_model(config.model)
            new_result = run_pipeline(config, run_id=new_run_id)
            _results[new_run_id] = new_result
            _persist_result(new_run_id, new_result)

        t = threading.Thread(target=_rerun, daemon=True)
        _running[new_run_id] = t
        t.start()
        return jsonify({"status": "rejected", "run_id": new_run_id})

    return jsonify({"error": "Unknown action"}), 400


@app.route("/api/history")
def api_history():
    # Start from persisted SQLite runs (survive server restarts)
    db_runs = memory.get_run_history(limit=100)
    db_ids = {r["run_id"] for r in db_runs}
    # Prepend any in-memory runs not yet approved/persisted.
    # Skip incomplete runs so History = one row per completed run:
    #   - preview runs awaiting the quality gate (not finalized)
    #   - errored runs, or runs with no curated output
    for run_id, result in list(_results.items()):
        if run_id in db_ids:
            continue
        if getattr(result, "awaiting_gate", False) or getattr(result, "error", None) or not getattr(result, "curated_output", None):
            continue
        if True:
            session_name = "Unknown"
            q_count = 0
            try:
                if result.context:
                    session_name = result.context.session_name
                if result.curated_output:
                    q_count = (len(result.curated_output.question_details) +
                               len(result.curated_output.coding_questions))
            except Exception:
                pass
            db_runs.insert(0, {"run_id": run_id, "session_name": session_name,
                                "question_count": q_count, "composite_score": None,
                                "approved": 0, "created_at": None, "api_usage": {}})
    # Annotate each run with its derived course topic + estimated cost for display
    from src.data_loader import get_topic_for_session
    from src.config import estimate_cost
    for r in db_runs:
        r["topic"] = get_topic_for_session(r.get("session_name", "") or "")
        r["cost"] = estimate_cost(r.get("api_usage") or {})
    return jsonify({"runs": db_runs})


@app.route("/api/usage")
def api_usage():
    """Overall workflow usage aggregated across all persisted runs (+ real OpenRouter spend)."""
    from src.config import estimate_cost
    from src.llm_client import get_credit_balance
    totals = {"runs": 0, "llm_calls": 0, "prompt_tokens": 0,
              "completion_tokens": 0, "tavily_calls": 0, "est_cost": 0.0}
    for r in memory.get_run_history(limit=1000):
        u = r.get("api_usage") or {}
        if not u:
            continue
        totals["runs"] += 1
        totals["llm_calls"] += u.get("llm_calls", 0) or 0
        totals["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
        totals["completion_tokens"] += u.get("completion_tokens", 0) or 0
        totals["tavily_calls"] += u.get("tavily_calls", 0) or 0
        c = estimate_cost(u)
        if c:
            totals["est_cost"] += c
    totals["est_cost"] = round(totals["est_cost"], 4)
    totals["tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]

    # Real OpenRouter key spend/remaining
    credits = get_credit_balance() or {}
    key_remaining = credits.get("key_remaining")
    key_limit = credits.get("key_limit")
    openrouter = {
        "remaining": key_remaining if key_remaining is not None else credits.get("account_remaining"),
        "used": round(key_limit - key_remaining, 2) if (key_limit is not None and key_remaining is not None) else None,
        "scope": credits.get("scope"),
    }
    return jsonify({"totals": totals, "openrouter": openrouter})


# ── React SPA catch-all ───────────────────────────────────────────────────────
# Flask's static_url_path="" intercepts unknown paths before the /<path> route,
# so we use a 404 handler to serve the SPA for all non-API client-side routes.

if _has_react:
    def _spa_index():
        # Never cache index.html so a rebuilt (re-hashed) bundle always loads.
        resp = send_from_directory(REACT_DIST, "index.html")
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path):
        if path and os.path.exists(os.path.join(REACT_DIST, path)):
            return send_from_directory(REACT_DIST, path)
        return _spa_index()

    @app.errorhandler(404)
    def spa_fallback(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return _spa_index()


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
