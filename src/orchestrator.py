"""Orchestrator — progress fan-out and a thin wrapper around AgentPipeline.

Progress events are both QUEUED (for the currently-attached SSE stream) and RETAINED (so a client
that reloads, sleeps, or loses its connection can replay what it missed). Retention is what makes the
transcript survivable: previously the stream's `finally` block deleted the queue on client disconnect,
so a browser reload during a run reattached to a brand-new empty queue and never saw `complete` —
the UI sat on "Generating…" until the 120s gap timeout declared a false error for a run that
succeeded.

Events carry structure, not just prose. The UI used to regex the question count out of a
human-readable sentence (`/(\\d+) questions/`), which broke the moment the wording changed.
"""

import queue
import threading
import time
import uuid

from src.agent import PipelineResult
from src.pipeline import AgentPipeline

# Per-run event fan-out. `_queues` feeds the attached stream; `_history` lets a reconnect catch up.
_progress_queues: dict[str, queue.Queue] = {}
_history: dict[str, list[dict]] = {}
_finished: set[str] = set()
_next_seq: dict[str, int] = {}     # monotonic per-run event counter (NOT len(history) — see _emit)
_lock = threading.Lock()

# Keep runs addressable long enough for a reviewer to reload the page, then reclaim them.
MAX_HISTORY_EVENTS = 400          # per run; a long run emits ~30-60
RETAIN_FINISHED_SECONDS = 1800    # 30 min
_finished_at: dict[str, float] = {}


def _emit(run_id: str, step_id: str, status: str, detail: str = "", **fields):
    """Record one progress event and hand it to the attached stream, if any.

    `fields` carries structured data (agent, duration_ms, tokens, counts) so the UI can render
    timing and funnel numbers instead of parsing sentences.
    """
    event = {"step": step_id, "status": status, "detail": detail, "ts": time.time()}
    event.update({k: v for k, v in fields.items() if v is not None})

    with _lock:
        hist = _history.setdefault(run_id, [])
        # A MONOTONIC counter, not len(hist). Using the list length meant that once the trim below
        # capped the list, every subsequent event got the same seq — with a cap of 5, nine emits
        # produced [0, 5, 5, 5, 5]. Both consumers de-duplicate on seq (main.py's stream loop and
        # useAgentRun), so past the cap nothing new was delivered: a long run would never emit
        # `complete` and the UI would sit on "Generating…" until the stall timeout.
        _next_seq[run_id] = seq = _next_seq.get(run_id, 0)
        _next_seq[run_id] += 1
        event["seq"] = seq
        hist.append(event)
        # Keep the head (which agent ran first) and drop the middle; the tail matters most.
        if len(hist) > MAX_HISTORY_EVENTS:
            del hist[1:len(hist) - MAX_HISTORY_EVENTS + 1]
        if step_id in ("complete", "error"):
            _finished.add(run_id)
            _finished_at[run_id] = time.time()
        q = _progress_queues.get(run_id)
    if q:
        q.put(event)


def get_progress_queue(run_id: str) -> queue.Queue:
    with _lock:
        if run_id not in _progress_queues:
            _progress_queues[run_id] = queue.Queue()
        return _progress_queues[run_id]


def get_history(run_id: str, after_seq: int = -1) -> list[dict]:
    """Events already emitted for this run, for replay on (re)connect."""
    with _lock:
        return [e for e in _history.get(run_id, []) if e.get("seq", 0) > after_seq]


def is_finished(run_id: str) -> bool:
    with _lock:
        return run_id in _finished


def cleanup_progress(run_id: str):
    """Drop a finished run's buffers. A no-op while the run is still going.

    Called when a stream ends. It must NOT discard state for a live run just because the browser
    went away — the pipeline keeps running on its own thread and its remaining events (including
    `complete`) still need somewhere to land.
    """
    with _lock:
        if run_id not in _finished:
            return
        _progress_queues.pop(run_id, None)
        _history.pop(run_id, None)
        _finished.discard(run_id)
        _finished_at.pop(run_id, None)
        _next_seq.pop(run_id, None)


def prune_finished(now: float | None = None) -> int:
    """Reclaim buffers for runs that finished a while ago. Returns how many were dropped."""
    now = now if now is not None else time.time()
    with _lock:
        stale = [rid for rid, t in _finished_at.items() if now - t > RETAIN_FINISHED_SECONDS]
        for rid in stale:
            _progress_queues.pop(rid, None)
            _history.pop(rid, None)
            _finished.discard(rid)
            _finished_at.pop(rid, None)
            _next_seq.pop(rid, None)
    return len(stale)


def run_pipeline(config, run_id: str | None = None) -> PipelineResult:
    """Run the 4-agent interview question generation pipeline."""
    actual_run_id = run_id or str(uuid.uuid4())
    get_progress_queue(actual_run_id)
    return AgentPipeline().run(config, actual_run_id, _emit)


# ── TESTING: preview mode wrappers ────────────────────────────────────────────
def run_preview_pipeline(config, run_id: str | None = None):
    """Stages 1–3 only; returns (partial_result, state) to resume later."""
    actual_run_id = run_id or str(uuid.uuid4())
    get_progress_queue(actual_run_id)
    return AgentPipeline().run_preview(config, actual_run_id, _emit)


def finalize_pipeline(state, run_id: str) -> PipelineResult:
    """Resume a preview run through Evaluation + quality gate."""
    get_progress_queue(run_id)
    return AgentPipeline().finalize(state, run_id, _emit)
