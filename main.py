"""FastAPI app — JSON API serving the Questor React SPA (frontend/dist/).

Ported from the previous Flask app. Three things drove the move:

  * A real ASGI server (uvicorn) instead of the Werkzeug development server, which was running with
    `debug=True` and is not meant to serve anyone but the developer.
  * Pydantic request models. The project already uses Pydantic v2 for its data models, so request
    validation now comes from the same place instead of hand-written `.get()` chains and `int()` casts
    that raised 500s on bad input.
  * Generated API docs at /docs, which makes the endpoint surface inspectable.

Two deliberate compatibility choices:

  * Error responses keep the `{"error": "..."}` shape the React client parses (`frontend/src/lib/api.js`
    reads `body.error`), NOT FastAPI's default `{"detail": ...}`. The exception handlers below do that
    translation, so route code can raise `HTTPException` idiomatically.
  * Route handlers that block — waiting on a pipeline thread, reading a progress queue — are plain
    `def`, not `async def`. Starlette runs those in a threadpool, so blocking work never stalls the
    event loop. Making them `async def` would be actively wrong here.

Run:  uvicorn main:app --port 5000            (add --reload while developing)
      python main.py                          (equivalent, for convenience)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import memory
from src.agent import PipelineResult
from src.data_loader import get_data_store
# Imported at module level on purpose. These used to be function-local imports inside api_meta and
# api_usage, while api_generate called set_active_model() with no import in scope — so every
# POST /api/generate raised NameError and returned 500. Keep them here so a caller cannot be added
# without the name being bound.
from src.llm_client import get_active_model, get_credit_balance, set_active_model
from src.models import GenerationConfig
from src.orchestrator import (finalize_pipeline, get_history, get_progress_queue, is_finished,
                              prune_finished, run_pipeline, run_preview_pipeline)
from src.rejection_rules import rule_for

log = logging.getLogger("questor")

# How often a quiet stream emits a heartbeat, and how long total silence is tolerated before the run
# is declared stalled. The gap between two progress events is legitimately minutes for a tool that
# makes a dozen sequential LLM calls, so the stall bound is generous.
SSE_HEARTBEAT_SECONDS = 15.0
SSE_STALL_SECONDS = float(os.getenv("SSE_STALL_SECONDS", "900"))

# Caps on the in-memory run maps. Finished runs live in SQLite and are reloaded on demand, so these
# only bound memory — they don't lose anything. Insertion order makes the oldest the first evicted.
MAX_CACHED_RESULTS = int(os.getenv("MAX_CACHED_RESULTS", "50"))
MAX_CACHED_PREVIEWS = int(os.getenv("MAX_CACHED_PREVIEWS", "10"))
MAX_CACHED_BATCHES = int(os.getenv("MAX_CACHED_BATCHES", "20"))

REACT_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
_has_react = os.path.isdir(REACT_DIST)

app = FastAPI(
    title="Questor — Agentic Interview Question Generator",
    description="Curates real, sourced interview questions for course sessions.",
    version="2.0.0",
)

# Auto-import exported runs on startup so a fresh clone gets the shared question sets.
memory.init_db()
_export_path = os.path.join(os.path.dirname(__file__), "data", "exported_runs.json")
if os.path.exists(_export_path):
    conn = memory.get_connection()
    count = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    conn.close()
    if count == 0:
        from scripts.import_runs import import_runs
        import_runs()

# In-memory store for pipeline results (keyed by run_id). Completed runs are also persisted to
# SQLite by _persist_result, so a restart loses only in-flight runs.
_results: dict[str, PipelineResult] = {}
_running: dict[str, threading.Thread] = {}
# TESTING: preview mode — retained AgentState between the pick and gate phases
_preview_states: dict = {}
# Multi-topic batches, keyed by batch_id: one row per topic with its own run_id. Runs execute
# SEQUENTIALLY on a single worker thread — see `_run_batch` for why parallel is the wrong choice.
# Bounded by `_prune_run_state` like the stores above; the durable copy is `run_history.batch_id`.
_batches: dict[str, dict] = {}


# ── Error shape ──────────────────────────────────────────────────────────────
# The React client reads `body.error`; FastAPI defaults to `body.detail`. Translate both
# HTTPException and request-validation failures so the frontend needs no changes.

@app.exception_handler(HTTPException)
async def _http_error(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError):
    """Report the first validation problem in the client's expected shape."""
    first = (exc.errors() or [{}])[0]
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "request"
    return JSONResponse(status_code=400,
                        content={"error": f"{field}: {first.get('msg', 'invalid value')}"})


# ── Request models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    session_names: list[str] = Field(default_factory=list)
    # Legacy input only — the FINAL set is not trimmed to this (see config.MAX_QUESTIONS).
    max_questions: int = Field(default=12, ge=1, le=60)
    custom_topic: str = ""
    model: str | None = None
    preview: bool = False           # TESTING: pause after picking, before the quality gate
    category: str = "GEN_AI"
    course_type: str | None = None

    def resolved_sessions(self) -> list[str]:
        names = [s for s in self.session_names if (s or "").strip()]
        if self.custom_topic.strip():
            names.append(self.custom_topic.strip())
        return names

    def resolved_category(self) -> str:
        return (self.category or "GEN_AI").strip().upper().replace(" ", "_")


class BatchGenerateRequest(BaseModel):
    """Generate for several topics from one click — ONE RUN PER TOPIC, not one merged run.

    Each run keeps its own question set, quality gate, review screen and spreadsheet
    (`sheets_writer` titles it "<Topic> - <sessions> (NxtMock)"), because review, approve, rejection
    feedback and learned rules are all keyed by run_id. Merging topics would give one verdict and one
    all-or-nothing approve for everything, and would push 6-12 sessions into a single SessionContext —
    the per-session attribution collapse documented in CLAUDE.md showed up at TWO.
    """
    topics: list[str] = Field(default_factory=list)
    course: str = "gen_ai"
    max_questions: int = Field(default=12, ge=1, le=60)
    model: str | None = None
    category: str = "GEN_AI"
    course_type: str | None = None

    def resolved_topics(self) -> list[str]:
        """De-duplicated, order-preserving. A repeated topic would otherwise run twice."""
        seen, out = set(), []
        for t in self.topics:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def resolved_category(self) -> str:
        return (self.category or "GEN_AI").strip().upper().replace(" ", "_")


class AddSessionRequest(BaseModel):
    course_name: str = ""
    course_id: str = ""
    topic: str
    session_name: str
    reading_material: str
    category: str | None = None
    course_type: str = "mixed"
    session_type: str | None = None
    kps: list | None = None


class ImportCourseRequest(BaseModel):
    course_name: str
    markdown: str = ""
    category: str | None = None
    course_type: str = "mixed"


class ReviewRequest(BaseModel):
    """Reviewer decisions. `rejected_feedback` maps question_id → reason (a taxonomy key from
    REJECT_REASONS in Review.jsx, or free text).

    `decisions_sent` distinguishes "the client made explicit per-question decisions" from "the client
    sent nothing" — WITHOUT it, an empty `accepted_ids` was ambiguous, and the export path read it as
    "no filter requested" rather than "the reviewer accepted nothing". Rejecting every question and
    clicking Export therefore exported and banked all of them.
    """
    action: str = "approve"
    accepted_ids: list[str] = Field(default_factory=list)
    rejected_feedback: dict[str, str] = Field(default_factory=dict)
    decisions_sent: bool = False

    @property
    def has_explicit_decisions(self) -> bool:
        return self.decisions_sent or bool(self.accepted_ids) or bool(self.rejected_feedback)


# ── Result helpers ───────────────────────────────────────────────────────────

def _payload(result: PipelineResult, run_id: str) -> dict:
    """Build the /api/result response shape from a PipelineResult."""
    from src.config import SESSION_FIT_HIGH
    ctx = result.context
    return {
        "run_id": run_id,
        "context": ctx.model_dump() if ctx else None,
        "output": result.curated_output.model_dump() if result.curated_output else None,
        "report": result.quality_report.model_dump() if result.quality_report else None,
        "awaiting_gate": getattr(result, "awaiting_gate", False),  # TESTING: preview mode
        "removed": getattr(result, "removed", []),  # rejected questions + reasons
        # Review tiering: questions at/above this session_fit are shown as "high confidence", so a
        # large uncapped set stays reviewable — the reviewer works top-down and stops where fit falls.
        "thresholds": {"session_fit_high": SESSION_FIT_HIGH},
    }


def _persist_result(run_id: str, result: PipelineResult, batch_id: str | None = None) -> None:
    """Persist a completed run so Review + re-export survive restarts, and surface it in History."""
    try:
        if not result:
            return
        if result.error or not result.curated_output:
            # Record WHY it produced nothing. Previously this returned early, so a Tavily outage was
            # indistinguishable in History from never having pressed Generate. No `run_results` payload
            # — there is no question set — so Review correctly has nothing to open.
            memory.save_run(
                run_id=run_id,
                session_name=(result.context.session_name if result.context else "Unknown"),
                question_count=0, composite_score=0, loops_used=0, approved=False,
                batch_id=batch_id, error=result.error or "Run produced no question set")
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
            batch_id=batch_id,
        )
    except Exception as e:  # noqa: BLE001 — a persistence failure must not lose the in-memory run
        log.error("failed to persist run %s: %s", run_id, e)


def _load_result(run_id: str) -> PipelineResult | None:
    """In-memory result, or reconstruct one from the persisted payload."""
    result = _results.get(run_id)
    if result:
        return result
    payload = memory.get_run_result(run_id)
    if not payload:
        return None
    from src.models import CuratedOutput, QualityReport, SessionContext
    r = PipelineResult()
    r.run_id = run_id
    r.context = SessionContext.model_validate(payload["context"]) if payload.get("context") else None
    r.curated_output = (CuratedOutput.model_validate(payload["output"])
                        if payload.get("output") else None)
    r.quality_report = (QualityReport.model_validate(payload["report"])
                        if payload.get("report") else None)
    return r


def _start(run_id: str, target) -> None:
    """Run a pipeline phase on a daemon thread, tracking it so /api/status can report on it."""
    _prune_run_state()
    t = threading.Thread(target=target, daemon=True)
    _running[run_id] = t
    t.start()


def _prune_run_state() -> None:
    """Bound the in-memory run maps.

    These are module-level dicts that were never evicted, so a long-lived server accumulated every
    PipelineResult (each holding a full question set) and every preview AgentState until restart.
    Completed runs are persisted to SQLite and reloaded on demand by `_load_result`, so dropping the
    oldest in-memory copies costs nothing but a database read.
    """
    for name, store, cap in (("results", _results, MAX_CACHED_RESULTS),
                             ("previews", _preview_states, MAX_CACHED_PREVIEWS),
                             ("batches", _batches, MAX_CACHED_BATCHES)):
        while len(store) > cap:
            store.pop(next(iter(store)), None)
    for run_id, thread in [(r, t) for r, t in _running.items() if not t.is_alive()]:
        if len(_running) <= MAX_CACHED_RESULTS:
            break
        _running.pop(run_id, None)


# ── Metadata ─────────────────────────────────────────────────────────────────

@app.get("/api/meta")
def api_meta():
    """Runtime info for the UI: active model, selectable models, credit balance, bank size."""
    from src.config import MODEL_OPTIONS
    try:
        from src.question_bank import get_retriever
        stats = get_retriever().get_stats()
    except Exception as e:  # noqa: BLE001 — the UI must still load if the bank is missing
        log.warning("question bank unavailable: %s", e)
        stats = {}
    return {
        "model": get_active_model(),
        "models": MODEL_OPTIONS,
        "credits": get_credit_balance(),
        "bank_count": stats.get("total", 0),
        # Which retrieval ranking is active (hybrid semantic vs TF-IDF only) — surfaced so a missing
        # embedding model is visible in the UI rather than silently degrading question quality.
        "bank_index": stats.get("index"),
    }


@app.get("/api/sessions")
def api_sessions():
    return {"sessions": get_data_store().get_session_names()}


def _gen_ai_topics() -> dict:
    """Built-in Gen AI course topics (the flat course_structure.json)."""
    path = os.path.join(os.path.dirname(__file__), "data", "course_structure.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@app.get("/api/topics")
def api_topics(course: str = "gen_ai"):
    if course and course != "gen_ai":
        return {"topics": memory.get_course_topics(course)}
    return {"topics": _gen_ai_topics()}


@app.get("/api/courses")
def api_courses():
    """Selectable courses: the built-in Gen AI course plus any user-added ones."""
    courses = [{"id": "gen_ai", "name": "Gen AI", "category": "GEN_AI",
                "course_type": "mixed", "builtin": True}]
    for c in memory.get_courses():
        courses.append({"id": c["course_id"], "name": c["name"], "category": c["category"],
                        "course_type": c.get("course_type", "mixed"), "builtin": False})
    return {"courses": courses}


# ── Course authoring ─────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_") or "course"


@app.post("/api/courses/session")
def api_add_course_session(body: AddSessionRequest):
    """Add a single session, creating the course if it is new."""
    course_id = body.course_id.strip() or _slugify(body.course_name)
    topic, session_name = body.topic.strip(), body.session_name.strip()
    reading = body.reading_material.strip()
    if not (course_id and topic and session_name and reading):
        raise HTTPException(400, "course, topic, session name and reading material are required")
    category = (body.category or _slugify(body.course_name).upper() or "COURSE").upper()
    if not memory.get_course(course_id):
        memory.add_course(course_id, body.course_name.strip() or course_id, category,
                          body.course_type)
    memory.add_course_session(course_id, topic, session_name, reading,
                              body.session_type, body.kps or None)
    return {"course_id": course_id, "status": "added"}


def _parse_course_markdown(md: str, default_topic: str) -> list[tuple[str, str, str]]:
    """Return [(topic, session, reading_material)]. '#'=topic, '##'=session.
    If only '#' headings exist, each '#' is a session under default_topic."""
    lines = md.splitlines()
    has_sub = any(re.match(r"^##\s+\S", ln) for ln in lines)
    out: list[tuple[str, str, str]] = []
    cur_topic, cur_session, buf = default_topic, None, []

    def _flush():
        if cur_session is not None:
            out.append((cur_topic, cur_session, "\n".join(buf).strip()))

    for ln in lines:
        m2 = re.match(r"^##\s+(.*)$", ln)
        m1 = re.match(r"^#\s+(.*)$", ln)
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


@app.post("/api/courses/import")
def api_import_course(body: ImportCourseRequest):
    """Bulk-add a course from a Markdown blob: '#'=topic, '##'=session (body=reading material)."""
    name = body.course_name.strip()
    if not name:
        raise HTTPException(400, "course name is required")
    parsed = _parse_course_markdown(body.markdown, default_topic=name)
    if not parsed:
        raise HTTPException(400, "No sessions found — use '# Topic' and '## Session' headings")
    course_id = _slugify(name)
    category = (body.category or _slugify(name).upper()).upper()
    memory.add_course(course_id, name, category, body.course_type)
    for topic, session, reading in parsed:
        memory.add_course_session(course_id, topic, session, reading, None, None)
    return {"course_id": course_id, "sessions": len(parsed),
            "topics": len({t for t, _, _ in parsed}), "status": "imported"}


@app.delete("/api/courses/{course_id}")
def api_delete_course(course_id: str):
    memory.delete_course(course_id)
    return {"status": "deleted", "course_id": course_id}


# ── Generation ───────────────────────────────────────────────────────────────

@app.post("/api/generate")
def api_generate(body: GenerateRequest):
    session_names = body.resolved_sessions()
    if not session_names:
        raise HTTPException(400, "No sessions provided")

    config = GenerationConfig(
        session_names=session_names,
        max_questions=body.max_questions,
        model=(body.model or "").strip() or None,
        preview=body.preview,
        category=body.resolved_category(),
        course_type=body.course_type,
    )
    run_id = str(uuid.uuid4())
    get_progress_queue(run_id)
    # Remember the choice for the picker's default only. The run itself takes its model from
    # `config` (threaded through AgentState), so a second tab starting a different model cannot
    # retarget this run's in-flight calls.
    set_active_model(config.model)

    def _run():
        if config.preview:  # TESTING: pause after picking, before the gate
            result, state = run_preview_pipeline(config, run_id=run_id)
            _results[run_id] = result
            if not result.error:
                _preview_states[run_id] = state
            return
        result = run_pipeline(config, run_id=run_id)
        _results[run_id] = result
        _persist_result(run_id, result)

    _start(run_id, _run)
    return {"run_id": run_id}


def _topics_for_course(course: str) -> dict:
    """Same topic→sessions map the picker reads, so the batch cannot disagree with the UI."""
    if course and course != "gen_ai":
        return memory.get_course_topics(course) or {}
    return _gen_ai_topics() or {}


def _run_batch(batch_id: str) -> None:
    """Run each topic's pipeline one at a time, and keep going when one fails.

    SEQUENTIAL on purpose, not for caution. `_start` has no global lock so N runs *could* go at once,
    but a batch of 3 topics is 3 full pipelines: firing them together multiplies the LLM/Tavily burst
    (this project has already exhausted both a Tavily plan and an OpenRouter key's headroom), and
    `memory.db` is SQLite — concurrent pipelines writing run history and feedback is lock contention
    for no user benefit.

    CONTINUE ON FAILURE: each topic is an independent deliverable, so a dead phase or a bad gate on one
    is recorded against that row and the worker moves on — the same discipline as `state.phase_errors`.
    A systemic failure (bad key) will simply mark every row failed, which is the honest outcome.
    """
    batch = _batches.get(batch_id)
    if not batch:
        return
    for row in batch["runs"]:
        run_id = row["run_id"]
        row["status"] = "running"
        try:
            config = GenerationConfig(
                session_names=list(row["sessions"]),
                max_questions=batch["max_questions"],
                model=batch["model"],
                category=batch["category"],
                course_type=batch["course_type"],
            )
            result = run_pipeline(config, run_id=run_id)
            _results[run_id] = result
            _persist_result(run_id, result, batch_id=batch_id)
            if result.error:
                row["status"], row["error"] = "failed", str(result.error)
            else:
                row["status"] = "done"
                out = result.curated_output
                row["question_count"] = (len(out.question_details) + len(out.coding_questions)) if out else 0
                row["verdict"] = (result.quality_report.pass_fail if result.quality_report else None)
        except Exception as exc:  # noqa: BLE001 — one topic must not take the batch down
            log.error("batch %s: topic %r failed: %s", batch_id[:8], row["topic"], exc)
            row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
    batch["finished"] = True


@app.post("/api/generate/batch")
def api_generate_batch(body: BatchGenerateRequest):
    """Queue one pipeline run per selected topic. Returns immediately with every run_id."""
    topics = body.resolved_topics()
    if not topics:
        raise HTTPException(400, "No topics provided")

    available = _topics_for_course(body.course)
    unknown = [t for t in topics if not (available.get(t) or [])]
    if unknown:
        raise HTTPException(400, f"Unknown or empty topic(s): {', '.join(unknown)}")

    batch_id = str(uuid.uuid4())
    runs = []
    for topic in topics:
        run_id = str(uuid.uuid4())
        # Create the SSE queue up front so /api/stream/{run_id} and the Progress page work per topic
        # with no changes at all — a queued run streams as soon as the worker reaches it.
        get_progress_queue(run_id)
        runs.append({"run_id": run_id, "topic": topic, "sessions": list(available[topic]),
                     "status": "queued", "question_count": None, "verdict": None, "error": None})

    _batches[batch_id] = {
        "batch_id": batch_id, "runs": runs, "finished": False,
        "max_questions": body.max_questions, "model": (body.model or "").strip() or None,
        "category": body.resolved_category(), "course_type": body.course_type,
    }
    # Display default for the picker only; each run takes its model from its own GenerationConfig.
    set_active_model((body.model or "").strip() or None)
    _start(batch_id, lambda: _run_batch(batch_id))
    return {"batch_id": batch_id,
            "runs": [{k: r[k] for k in ("run_id", "topic", "sessions")} for r in runs]}


@app.get("/api/batch/{batch_id}")
def api_batch_status(batch_id: str):
    """Per-topic batch status. NON-BLOCKING by design — never join the worker here.

    `/api/result` already returns 409 while a run is in flight because polling tabs blocking on a
    thread starved Starlette's bounded threadpool and stalled every other endpoint.
    """
    batch = _batches.get(batch_id)
    if batch:
        thread = _running.get(batch_id)
        return {
            "batch_id": batch_id,
            "running": bool(thread and thread.is_alive()),
            "finished": batch["finished"],
            "runs": [{k: r[k] for k in
                      ("run_id", "topic", "sessions", "status", "question_count", "verdict", "error")}
                     for r in batch["runs"]],
        }
    # Fall back to the durable copy so a restart (or an evicted registry entry) still renders.
    persisted = memory.get_batch_runs(batch_id)
    if not persisted:
        raise HTTPException(404, "Batch not found")
    return {
        "batch_id": batch_id, "running": False, "finished": True,
        "runs": [{"run_id": r["run_id"], "topic": r["session_name"], "sessions": [],
                  "status": "done", "question_count": r["question_count"],
                  "verdict": None, "error": None} for r in persisted],
    }


@app.get("/api/stream/{run_id}")
def api_stream(run_id: str, after: int = -1):
    """Server-sent events for live pipeline progress.

    A plain (sync) generator on purpose: `queue.get(timeout=…)` blocks, and Starlette iterates sync
    generators in a threadpool, so the event loop stays free.

    Two things this has to get right, because the old version got both wrong:

    * **Replay.** Everything already emitted is sent first (`after` skips what the client has), so a
      reload or a dropped connection resumes the transcript instead of starting blank.
    * **Silence is not failure.** Progress is emitted per tool call, and one tool can legitimately run
      for minutes (`validate_relevance` makes a dozen sequential LLM calls). The old 120s queue
      timeout turned that silence into `{"step": "timeout", "status": "error"}` and closed the
      stream — the UI showed "Pipeline Error" for runs that went on to succeed. Now a quiet interval
      emits a heartbeat, and the stream only ends when the run actually ends or genuinely stalls.
    """
    q = get_progress_queue(run_id)

    def events():
        try:
            last_seq = after
            for event in get_history(run_id, after_seq=after):
                last_seq = event.get("seq", last_seq)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("step") in ("complete", "error"):
                    return          # already finished — replay and close

            idle = 0.0
            while True:
                try:
                    event = q.get(timeout=SSE_HEARTBEAT_SECONDS)
                except Exception:
                    idle += SSE_HEARTBEAT_SECONDS
                    if idle >= SSE_STALL_SECONDS:
                        yield ("data: " + json.dumps({
                            "step": "error", "status": "error",
                            "detail": f"No progress for {int(idle)}s — the run appears to have stalled.",
                        }) + "\n\n")
                        return
                    yield ("data: " + json.dumps({"step": "heartbeat", "status": "running",
                                                  "detail": "", "idle_s": int(idle)}) + "\n\n")
                    continue
                idle = 0.0
                if event.get("seq", 0) <= last_seq:
                    continue        # already replayed
                last_seq = event.get("seq", last_seq)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("step") in ("complete", "error"):
                    return
        finally:
            # Reclaim only runs that finished LONG ago. Calling cleanup_progress(run_id) here was
            # self-defeating: when a run completes, this generator returns, the `finally` fires, the
            # run is already in `_finished`, and the retention window is deleted on the spot. Reloading
            # /progress/<id> after completion then found an empty, not-finished-looking run and sat
            # emitting heartbeats until the 900s stall bound — so RETAIN_FINISHED_SECONDS could never
            # be what actually reclaimed anything.
            prune_finished()

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # stop nginx buffering the stream
        "Connection": "keep-alive",
    })


@app.get("/api/status/{run_id}")
def api_status(run_id: str):
    """Whether a run is still going. Cheap and non-blocking — poll this, don't block on the result."""
    thread = _running.get(run_id)
    running = bool(thread and thread.is_alive())
    return {
        "run_id": run_id,
        "running": running,
        "finished": is_finished(run_id),
        "has_result": run_id in _results or memory.get_run_result(run_id) is not None,
    }


@app.get("/api/result/{run_id}")
def api_result(run_id: str):
    """The finished run's payload, or 409 while it is still in flight.

    This used to `thread.join(timeout=300)`. Starlette runs sync handlers in a bounded threadpool
    (40 workers), so a handful of tabs polling long runs consumed every worker and stalled *all*
    other endpoints — including the SSE streams those tabs were waiting on. Clients follow the
    progress stream and fetch the result once it reports completion.
    """
    thread = _running.get(run_id)
    if thread and thread.is_alive() and run_id not in _results:
        raise HTTPException(409, "Run still in progress")

    from src.data_loader import get_topic_for_session

    def _with_topic(p: dict) -> dict:
        sess = (p.get("context") or {}).get("session_name", "") or ""
        p["topic"] = get_topic_for_session(sess)
        return p

    result = _results.get(run_id)
    if not result:
        # Fall back to the persisted run (survives restarts / history).
        payload = memory.get_run_result(run_id)
        if payload:
            return _with_topic(payload)
        raise HTTPException(404, "Run not found or still processing")
    if result.error:
        raise HTTPException(500, result.error)
    return _with_topic(_payload(result, run_id))


# ── TESTING: preview mode — resume a paused run through the quality gate ──────

@app.post("/api/proceed/{run_id}")
def api_proceed(run_id: str):
    state = _preview_states.pop(run_id, None)
    if state is None:
        raise HTTPException(404, "No preview run to proceed (expired or already finalized)")
    get_progress_queue(run_id)  # fresh queue for the finalize phase

    def _finalize():
        # The model comes from state.config, carried over from the preview phase.
        result = finalize_pipeline(state, run_id)
        _results[run_id] = result
        _persist_result(run_id, result)

    _start(run_id, _finalize)
    return {"run_id": run_id}


# ── Review: approve → Sheets, or reject → learn + regenerate ─────────────────

def _learn_from_reasons(session_name: str, rejected_feedback: dict[str, str]) -> None:
    """Turn rejection reasons into learned rules the relevance judge will read next run.

    A taxonomy key maps straight to a canonical rule — deterministic, no LLM call, and no
    near-duplicate rules accumulating. Anything else is free text and gets distilled.
    """
    reasons = [v for v in rejected_feedback.values() if isinstance(v, str) and v.strip()]
    for reason in dict.fromkeys(reasons):       # de-duplicate, preserve order
        try:
            rule = rule_for(reason) or memory.distill_rule(session_name, reason)
            if rule:
                memory.append_learned_rule(rule)
        except Exception as e:  # noqa: BLE001
            log.warning("could not learn from rejection reason %r: %s", reason, e)


@app.post("/api/approve/{run_id}")
def api_approve(run_id: str, body: ReviewRequest):
    result = _load_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")

    accepted_ids = body.accepted_ids
    rejected_feedback = body.rejected_feedback
    session_name = result.context.session_name if result.context else "Unknown"

    if body.action == "approve":
        # Record reviewer decisions (feedback loop) BEFORE filtering to accepted-only. These writes
        # are the entire learning signal, so a failure is logged loudly rather than swallowed.
        try:
            acc, rej = set(accepted_ids), set(rejected_feedback)
            for q in list(result.curated_output.question_details):
                if q.question_id in rej:
                    memory.record_feedback(run_id, session_name, q.question_id, q.content, "bad")
                elif not accepted_ids or q.question_id in acc:
                    memory.record_feedback(run_id, session_name, q.question_id, q.content, "good")
            # A question dropped from an otherwise-approved set is still a rejection: suppress it and
            # learn from it exactly as the reject path does.
            dropped = [q.content for q in result.curated_output.question_details
                       if q.question_id in rej]
            if dropped:
                memory.record_rejections(session_name, dropped)
            _learn_from_reasons(session_name, rejected_feedback)
        except Exception as e:  # noqa: BLE001
            log.error("FAILED to persist reviewer feedback for run %s: %s", run_id, e)

        # Filter to what the reviewer accepted. Keyed on `has_explicit_decisions`, not on
        # `accepted_ids` being non-empty: accepting nothing is a real decision that must export
        # nothing, and the old `if accepted_ids:` read it as "export everything".
        if body.has_explicit_decisions:
            accepted = set(accepted_ids)
            result.curated_output.question_details = [
                q for q in result.curated_output.question_details if q.question_id in accepted
            ]
            result.curated_output.coding_questions = [
                q for q in result.curated_output.coding_questions if q.id in accepted
            ]

        total_q = (len(result.curated_output.question_details)
                   + len(result.curated_output.coding_questions))
        if total_q == 0:
            # Nothing to export. Refuse rather than creating an empty spreadsheet and marking the run
            # approved. The reviewer's rejections were already recorded above, so nothing is lost.
            raise HTTPException(400, "No questions accepted — nothing to export. "
                                     "Accept at least one question, or use Regenerate.")
        memory.save_run(
            run_id=run_id, session_name=session_name, question_count=total_q,
            composite_score=result.quality_report.composite_score if result.quality_report else 0,
            loops_used=result.quality_report.loops_used if result.quality_report else 0,
            approved=True,
            api_usage=dict(result.quality_report.api_usage) if result.quality_report else None,
        )
        memory.save_run_result(run_id, _payload(result, run_id))
        for q in result.curated_output.question_details:
            memory.save_question_to_bank(q.question_id, session_name, q.content, q.source)

        # Fold the approved set into this TOPIC's accumulating set, and make this run the topic's
        # canonical holder. That is what stops the next re-run producing another version: the following
        # run carries these questions in (`tools._add_retained`) and adds only what is new, while this
        # run's predecessor is flagged so History keeps one row per topic.
        try:
            topic_key = memory.topic_key_for(session_name)
            previous = memory.get_canonical_run(topic_key)
            memory.upsert_topic_questions(topic_key, [
                {"content": q.content, "detail": q.model_dump(mode="json"), "status": "approved",
                 "first_run_id": run_id, "session_name": session_name,
                 "difficulty": q.difficulty, "company": q.asked_in_company,
                 "source": q.source, "kp_label": q.kp_label}
                for q in result.curated_output.question_details])
            memory.set_canonical_run(topic_key, run_id)
            if previous and previous != run_id:
                memory.mark_superseded(previous, run_id)
        except Exception as e:  # noqa: BLE001 — the approval itself must not be lost to this
            log.error("failed to update the topic set for run %s: %s", run_id, e)

        sheet_url = sheet_error = None
        try:
            from src.sheets_writer import write_to_sheets
            sheet_url = write_to_sheets(
                output=result.curated_output,
                report=result.quality_report,
                session_name=session_name,
                run_id=run_id,
                category=getattr(result, "category", "GEN_AI"),
            )
        except Exception as e:  # noqa: BLE001 — export failure must not lose the approval
            sheet_error = str(e)
            log.error("Sheets export failed for run %s: %s", run_id, e)

        resp = {"status": "approved", "saved": total_q}
        if sheet_url:
            resp["sheet_url"] = sheet_url
        if sheet_error:
            resp["sheet_error"] = sheet_error
        return resp

    if body.action == "reject":
        all_q = list(result.curated_output.question_details)
        original_count = len(all_q) or 12

        # If the client sent no ids at all (legacy reject-all), treat everything as rejected.
        rejected_ids = set(rejected_feedback)
        if not rejected_ids and not accepted_ids:
            rejected_ids = {q.question_id for q in all_q}
        rejected_qs = [q for q in all_q if q.question_id in rejected_ids]
        accepted_qs = ([q for q in all_q if q.question_id in accepted_ids] if accepted_ids
                       else [q for q in all_q if q.question_id not in rejected_ids])

        # Suppress rejected content (per individual session, so it transfers to other combinations).
        try:
            memory.record_rejections(session_name, [q.content for q in rejected_qs])
        except Exception as e:  # noqa: BLE001
            log.error("FAILED to persist rejections for %r: %s", session_name, e)
        try:
            for q in rejected_qs:
                memory.record_feedback(run_id, session_name, q.question_id, q.content, "bad")
            for q in accepted_qs:
                memory.record_feedback(run_id, session_name, q.question_id, q.content, "good")
        except Exception as e:  # noqa: BLE001
            log.error("FAILED to persist reviewer feedback for run %s: %s", run_id, e)
        _learn_from_reasons(session_name, rejected_feedback)

        # Preserve course identity on regeneration: `category` drives the sheet branding, so without
        # it a non-GenAI course re-exports as GenAI.
        #
        # `course_type` is carried for provenance only. It is NOT what steers per-type behaviour — an
        # earlier comment here claimed it did, which was never true: nothing reads
        # GenerationConfig.course_type. The authoritative type is the one the pipeline resolves from the
        # reading material (SessionContext.session_type), and that is what
        # config.difficulty_targets / eval_thresholds and the relevance judge's per-type guidance key
        # off. Carrying the course's declared type here would let a stale course-level label override a
        # per-session resolution.
        # Reuse the ORIGINAL run's model rather than whatever the picker currently shows, so a
        # regeneration reproduces the run being corrected instead of inheriting another tab's choice.
        prior_usage = (result.quality_report.api_usage if result.quality_report else {}) or {}
        config = GenerationConfig(
            session_names=memory._split_sessions(session_name),
            max_questions=original_count,
            model=prior_usage.get("model") or get_active_model(),
            category=getattr(result, "category", "GEN_AI") or "GEN_AI",
            course_type=getattr(result.context, "session_type", None) if result.context else None,
        )
        # Drop the cached session resolution so the re-run re-derives outcomes.
        cleared = memory.clear_session_resolution(session_name)
        log.info("cleared %d cached resolution(s) for %r", cleared, session_name)

        new_run_id = str(uuid.uuid4())
        get_progress_queue(new_run_id)

        def _rerun():
            new_result = run_pipeline(config, run_id=new_run_id)
            # A failed regeneration has no curated_output. Touching it raised an AttributeError that
            # died unlogged in this daemon thread, leaving _results unset and the client polling a 404.
            if new_result.error or not new_result.curated_output:
                _results[new_run_id] = new_result
                log.error("regeneration %s failed: %s", new_run_id, new_result.error)
                return
            # PIN the accepted questions; fill freed slots with NEW distinct ones (rejected content is
            # already suppressed inside the pipeline). Keep the original set size.
            accepted_norms = {memory.normalize_content(q.content) for q in accepted_qs}
            fresh = [q for q in new_result.curated_output.question_details
                     if memory.normalize_content(q.content) not in accepted_norms]
            new_result.curated_output.question_details = (accepted_qs + fresh)[:original_count]
            _results[new_run_id] = new_result
            _persist_result(new_run_id, new_result)

        _start(new_run_id, _rerun)
        return {"status": "rejected", "run_id": new_run_id}

    raise HTTPException(400, f"Unknown action: {body.action!r}")


# ── History & usage ──────────────────────────────────────────────────────────

@app.get("/api/history")
def api_history():
    """One row per completed run: persisted SQLite runs plus any in-memory runs not yet saved."""
    db_runs = memory.get_run_history(limit=100)
    db_ids = {r["run_id"] for r in db_runs}
    for run_id, result in list(_results.items()):
        if run_id in db_ids:
            continue
        # Skip incomplete runs so History stays one row per completed run: preview runs awaiting the
        # quality gate, errored runs, and runs with no curated output.
        if (getattr(result, "awaiting_gate", False) or getattr(result, "error", None)
                or not getattr(result, "curated_output", None)):
            continue
        session_name, q_count = "Unknown", 0
        try:
            if result.context:
                session_name = result.context.session_name
            q_count = (len(result.curated_output.question_details)
                       + len(result.curated_output.coding_questions))
        except Exception as e:  # noqa: BLE001
            log.warning("could not summarise in-memory run %s: %s", run_id, e)
        db_runs.insert(0, {"run_id": run_id, "session_name": session_name,
                           "question_count": q_count, "composite_score": None,
                           "approved": 0, "created_at": None, "api_usage": {}})

    from src.config import estimate_cost
    from src.data_loader import get_topic_for_session
    for r in db_runs:
        r["topic"] = get_topic_for_session(r.get("session_name", "") or "")
        r["cost"] = estimate_cost(r.get("api_usage") or {})
    return {"runs": db_runs}


@app.get("/api/usage")
def api_usage():
    """Workflow usage aggregated across persisted runs, plus real OpenRouter key spend."""
    from src.config import estimate_cost
    totals = {"runs": 0, "llm_calls": 0, "prompt_tokens": 0,
              "completion_tokens": 0, "tavily_calls": 0, "est_cost": 0.0}
    for r in memory.get_run_history(limit=1000):
        u = r.get("api_usage") or {}
        if not u:
            continue
        totals["runs"] += 1
        for key in ("llm_calls", "prompt_tokens", "completion_tokens", "tavily_calls"):
            totals[key] += u.get(key, 0) or 0
        totals["est_cost"] += estimate_cost(u) or 0.0
    totals["est_cost"] = round(totals["est_cost"], 4)
    totals["tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]

    credits = get_credit_balance() or {}
    key_remaining, key_limit = credits.get("key_remaining"), credits.get("key_limit")
    openrouter = {
        "remaining": key_remaining if key_remaining is not None else credits.get("account_remaining"),
        "used": (round(key_limit - key_remaining, 2)
                 if (key_limit is not None and key_remaining is not None) else None),
        "scope": credits.get("scope"),
    }
    return {"totals": totals, "openrouter": openrouter}


# ── React SPA ────────────────────────────────────────────────────────────────
# Mounted last so every /api route above wins. Unknown non-API paths return index.html, because
# react-router owns client-side routing.

if _has_react:
    app.mount("/assets", StaticFiles(directory=os.path.join(REACT_DIST, "assets")), name="assets")

    def _spa_index() -> FileResponse:
        # Never cache index.html, so a rebuilt (re-hashed) bundle always loads.
        return FileResponse(os.path.join(REACT_DIST, "index.html"),
                            headers={"Cache-Control": "no-store, must-revalidate"})

    @app.get("/{path:path}", include_in_schema=False)
    def serve_react(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = os.path.join(REACT_DIST, path)
        # `commonpath` blocks path traversal ("../../etc/passwd") from escaping the build dir.
        if (path and os.path.isfile(candidate)
                and os.path.commonpath([os.path.realpath(candidate),
                                        os.path.realpath(REACT_DIST)]) == os.path.realpath(REACT_DIST)):
            return FileResponse(candidate)
        return _spa_index()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)
