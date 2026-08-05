"""SQLite memory layer — caching, run history, RLHF feedback, cross-run question bank."""

import sqlite3
import json
import pathlib
import re
from src.config import MEMORY_DB

_RULES_FILE = pathlib.Path(__file__).parent.parent / "data" / "learned_rules.md"
_MAX_RULE_CHARS = 200


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session_resolutions (
            session_name    TEXT PRIMARY KEY,
            resolution_json TEXT NOT NULL,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS run_history (
            run_id          TEXT PRIMARY KEY,
            session_name    TEXT NOT NULL,
            question_count  INTEGER,
            composite_score REAL,
            loops_used      INTEGER,
            approved        INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS run_results (
            run_id       TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS courses (
            course_id    TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            category     TEXT NOT NULL,
            course_type  TEXT DEFAULT 'mixed',
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS course_sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id        TEXT NOT NULL,
            topic            TEXT NOT NULL,
            session_name     TEXT NOT NULL,
            reading_material TEXT,
            session_type     TEXT,
            kps_json         TEXT,
            created_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS question_feedback (
            question_id  TEXT NOT NULL,
            run_id       TEXT NOT NULL,
            feedback     TEXT NOT NULL,
            session_name TEXT,
            content      TEXT,
            created_at   TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (question_id, run_id)
        );

        CREATE TABLE IF NOT EXISTS question_bank (
            question_id  TEXT PRIMARY KEY,
            session_name TEXT NOT NULL,
            content      TEXT NOT NULL,
            source       TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );

        -- Human-rejected questions, keyed by NORMALIZED content (question_ids regenerate per run, so
        -- identity must be content). Session-scoped: a rejection sticks for that session on future runs.
        CREATE TABLE IF NOT EXISTS rejected_questions (
            session_name TEXT NOT NULL,
            content_norm TEXT NOT NULL,
            content      TEXT,
            created_at   TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (session_name, content_norm)
        );
    """)
    conn.commit()
    # Migrate: add columns to existing databases
    for migration in [
        "ALTER TABLE run_history ADD COLUMN api_usage_json TEXT",
        "ALTER TABLE question_feedback ADD COLUMN session_name TEXT",
        "ALTER TABLE question_feedback ADD COLUMN content TEXT",
    ]:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.close()


# --- Session Resolution Cache ---

def get_cached_resolution(session_name: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT resolution_json FROM session_resolutions WHERE session_name = ?",
        (session_name,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row["resolution_json"])
    return None


def cache_resolution(session_name: str, resolution: dict):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO session_resolutions (session_name, resolution_json) VALUES (?, ?)",
        (session_name, json.dumps(resolution))
    )
    conn.commit()
    conn.close()


def clear_session_resolution(session_name: str) -> int:
    """Delete cached resolutions for `session_name`. Returns the number of rows removed.

    Resolutions are keyed by a COMPOSITE string — `"{name}::{reading_material_hash}::ov{overrides}"`
    (see session_understanding.cache_resolution callers) — so an exact-match DELETE on the bare name
    matches nothing. The regenerate-after-reject path did exactly that and silently cleared 0 rows,
    which is why rejecting a set never actually re-derived its outcomes.
    """
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM session_resolutions WHERE session_name = ? OR session_name LIKE ?",
        (session_name, f"{session_name}::%"),
    )
    removed = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    conn.close()
    return removed


# --- Run History ---

def save_run(run_id: str, session_name: str, question_count: int,
             composite_score: float, loops_used: int, approved: bool = False,
             api_usage: dict | None = None):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO run_history
           (run_id, session_name, question_count, composite_score, loops_used, approved, api_usage_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, session_name, question_count, composite_score, loops_used, int(approved),
         json.dumps(api_usage) if api_usage else None)
    )
    conn.commit()
    conn.close()


def save_run_result(run_id: str, payload: dict):
    """Persist the full run payload ({context, output, report}) so Review and
    re-export survive server restarts."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO run_results (run_id, payload_json) VALUES (?, ?)",
        (run_id, json.dumps(payload))
    )
    conn.commit()
    conn.close()


def get_run_result(run_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT payload_json FROM run_results WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    return json.loads(row["payload_json"]) if row else None


# --- Custom Courses (user-added, multi-course support) ---

def add_course(course_id: str, name: str, category: str, course_type: str = "mixed"):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO courses (course_id, name, category, course_type) VALUES (?, ?, ?, ?)",
        (course_id, name, category, course_type),
    )
    conn.commit()
    conn.close()


def add_course_session(course_id: str, topic: str, session_name: str,
                       reading_material: str = "", session_type: str | None = None,
                       kps: list | None = None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO course_sessions
           (course_id, topic, session_name, reading_material, session_type, kps_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (course_id, topic, session_name, reading_material, session_type,
         json.dumps(kps) if kps else None),
    )
    conn.commit()
    conn.close()


def get_courses() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT course_id, name, category, course_type, created_at FROM courses ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_course(course_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT course_id, name, category, course_type FROM courses WHERE course_id = ?",
        (course_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_course_topics(course_id: str) -> dict:
    """Return {topic: [session_names]} for a custom course (insertion order)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT topic, session_name FROM course_sessions WHERE course_id = ? ORDER BY id",
        (course_id,),
    ).fetchall()
    conn.close()
    topics: dict = {}
    for r in rows:
        topics.setdefault(r["topic"], []).append(r["session_name"])
    return topics


def get_custom_session(session_name: str) -> dict | None:
    """Reading material + course metadata for a user-added session (exact, then normalized)."""
    conn = get_connection()
    row = conn.execute(
        """SELECT cs.session_name, cs.reading_material, cs.session_type, cs.topic,
                  c.course_id, c.category, c.course_type
           FROM course_sessions cs JOIN courses c ON c.course_id = cs.course_id
           WHERE cs.session_name = ? ORDER BY cs.id DESC LIMIT 1""",
        (session_name,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_course(course_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM course_sessions WHERE course_id = ?", (course_id,))
    conn.execute("DELETE FROM courses WHERE course_id = ?", (course_id,))
    conn.commit()
    conn.close()


def get_run_history(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT run_id, session_name, question_count, composite_score, loops_used, approved, "
        "created_at, api_usage_json FROM run_history ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["api_usage"] = json.loads(d.pop("api_usage_json") or "{}") or {}
        result.append(d)
    return result


# NOTE: a `suppress_boost` table with get_suppress_list()/get_boost_list() readers used to live here.
# It had no writer and no caller anywhere in the codebase, and the table was empty after 36 runs, so
# it was removed rather than left looking like a working feature. Suppression of rejected questions is
# handled by `rejected_questions` (exact, below) plus the semantic penalty in tools._select_final.
# An existing memory.db may still contain the unused table; it is harmless and simply ignored.


# --- Learned Rules (distilled from human rejections) ---

def get_learned_rules() -> list[str]:
    """Read learned validation rules from data/learned_rules.md."""
    try:
        text = _RULES_FILE.read_text()
        section = text.split("## Rules")[-1]
        return [line.lstrip("- ").strip() for line in section.splitlines()
                if line.strip().startswith("- ")]
    except Exception:
        return []


def append_learned_rule(rule: str) -> bool:
    """Append a new rule to learned_rules.md. Returns True if added, False if duplicate/invalid."""
    rule = rule.strip()[:_MAX_RULE_CHARS]
    if not rule or "##" in rule or "<!--" in rule:
        return False
    if rule in get_learned_rules():
        return False
    _RULES_FILE.parent.mkdir(exist_ok=True)
    if not _RULES_FILE.exists():
        _RULES_FILE.write_text(
            "# Learned Validation Rules\n"
            "<!-- Auto-generated from reviewer rejections. Do not edit manually. -->\n\n"
            "## Rules\n"
        )
    with _RULES_FILE.open("a") as f:
        f.write(f"- {rule}\n")
    return True


def distill_rule(session_name: str, reason: str) -> str:
    """Use LLM to distil a rejection reason into a reusable validation rule."""
    from src.llm_client import chat_completion_json
    try:
        result = chat_completion_json(
            system_prompt="You convert interview question rejection reasons into reusable validation rules.",
            user_prompt=(
                f'A reviewer rejected an interview question for a session on "{session_name}" '
                f'with this reason:\n"{reason}"\n\n'
                f"Write a ≤200-char rule starting with \"Reject if\" or \"Skip if\" that generalises "
                f"this rejection so future questions with the same problem are caught automatically. "
                f"Be specific and actionable.\n"
                f'Return JSON: {{"rule": "..."}}'
            ),
            max_tokens=200,
        )
        return (result.get("rule") or "").strip()[:_MAX_RULE_CHARS]
    except Exception:
        return ""


# --- Cross-Run Question Bank ---

def save_question_to_bank(question_id: str, session_name: str, content: str, source: str = None):
    """Save an approved question to the persistent cross-run bank."""
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO question_bank (question_id, session_name, content, source) VALUES (?, ?, ?, ?)",
        (question_id, session_name, content, source)
    )
    conn.commit()
    conn.close()


def get_bank_questions(session_name: str) -> list[dict]:
    """Return all banked questions for the given session (for cross-run dedup)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT question_id, content FROM question_bank WHERE session_name = ? ORDER BY created_at DESC",
        (session_name,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Rejected-question suppression (per session, keyed by normalized content) ---

def normalize_content(text: str) -> str:
    """Normalization used for rejected-question identity (lowercase, punctuation-stripped, collapsed)."""
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _split_sessions(session_name: str) -> list[str]:
    """Explode a combined run name ("A + B + C") into its individual session names.

    Rejections must be stored per INDIVIDUAL session, not against the joined string. Keyed on the
    combination, a rejection learned while running "A + B" would not suppress anything when "A" is
    later run with "C" — the same bad question would come back and be rejected again.
    """
    parts = [p.strip() for p in (session_name or "").split(" + ")]
    return [p for p in parts if p] or ([session_name] if session_name else [])


def record_rejections(session_name: str, contents: list[str]) -> int:
    """Persist rejected question texts (by normalized content) so they never resurface.

    Recorded once per individual session in `session_name`, so suppression follows the session
    into any future combination. Returns the number of NEW (session, content) rows stored —
    re-rejecting the same question does not inflate the count.
    """
    conn = get_connection()
    n = 0
    sessions = _split_sessions(session_name)
    for c in contents:
        norm = normalize_content(c)
        if not norm:
            continue
        for sess in sessions:
            cur = conn.execute(
                "INSERT OR IGNORE INTO rejected_questions (session_name, content_norm, content) "
                "VALUES (?, ?, ?)",
                (sess, norm, c),
            )
            n += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    conn.close()
    return n


def get_rejected_norms(session_name: str) -> set[str]:
    """Normalized contents previously rejected for any of these session(s).

    Accepts a combined name and unions the rejections of each individual session, so a question
    rejected in one combination stays suppressed in every other combination containing that session.
    """
    sessions = _split_sessions(session_name)
    if not sessions:
        return set()
    # Include the combined name too: rows written before rejections were keyed per-session are
    # stored under the joined string, and they must keep suppressing.
    if session_name and session_name not in sessions:
        sessions = sessions + [session_name]
    conn = get_connection()
    placeholders = ",".join("?" for _ in sessions)
    rows = conn.execute(
        f"SELECT content_norm FROM rejected_questions WHERE session_name IN ({placeholders})",
        sessions,
    ).fetchall()
    conn.close()
    return {r["content_norm"] for r in rows}


# --- Reviewer-decision feedback → question_feedback table + eval/feedback_examples.json ---

_FEEDBACK_EXAMPLES = pathlib.Path(__file__).parent.parent / "eval" / "feedback_examples.json"


def record_feedback(run_id: str, session_name: str, question_id: str, content: str, decision: str) -> None:
    """Persist one reviewer decision ('good' accepted / 'bad' rejected): fills the question_feedback table
    (provenance) AND appends to eval/feedback_examples.json so the eval harness reflects real choices."""
    if not content or decision not in ("good", "bad"):
        return
    try:
        conn = get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO question_feedback (question_id, run_id, feedback, session_name, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (question_id or "", run_id or "", decision, session_name or "", content or ""),
        )
        conn.commit()
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        data = []
        if _FEEDBACK_EXAMPLES.exists():
            data = json.loads(_FEEDBACK_EXAMPLES.read_text(encoding="utf-8")) or []
        data.append({"session": session_name, "question": content, "decision": decision})
        _FEEDBACK_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
        _FEEDBACK_EXAMPLES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def get_feedback_examples() -> list[dict]:
    """Reviewer-decision examples ({session, question, decision}) for the eval harness. [] if none."""
    try:
        if _FEEDBACK_EXAMPLES.exists():
            return json.loads(_FEEDBACK_EXAMPLES.read_text(encoding="utf-8")) or []
    except Exception:  # noqa: BLE001
        pass
    return []


def _backfill_rejections_per_session() -> int:
    """One-time migration: re-key rejections stored under a COMBINED run name ("A + B + C") so each
    individual session also has its own row.

    Rejections used to be keyed on the joined name, so a question rejected while running "A + B" did
    not suppress when "A" was later run with "C" — it came back and was rejected again. Splitting the
    existing rows makes those already-recorded rejections useful in every future combination.

    Idempotent (INSERT OR IGNORE) and cheap, so it is safe to run on every import. Returns the number
    of rows added.
    """
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT session_name, content_norm, content FROM rejected_questions "
            "WHERE session_name LIKE '% + %'"
        ).fetchall()
        added = 0
        for row in rows:
            for sess in _split_sessions(row["session_name"]):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO rejected_questions (session_name, content_norm, content) "
                    "VALUES (?, ?, ?)",
                    (sess, row["content_norm"], row["content"]),
                )
                added += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
        conn.close()
        return added
    except Exception:  # noqa: BLE001 — a failed migration must not stop the app from starting
        return 0


# Initialize on import
init_db()
_backfill_rejections_per_session()
