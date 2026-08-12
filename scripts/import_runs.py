"""Import run results from data/exported_runs.json into memory.db (skips duplicates)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import memory

EXPORT_PATH = Path(__file__).parent.parent / "data" / "exported_runs.json"


def import_runs(path: Path = EXPORT_PATH) -> int:
    """Import runs from the export file. Returns the number of new runs imported."""
    if not path.exists():
        print(f"No export file at {path}")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        print("Export file contains no runs.")
        return 0

    memory.init_db()
    conn = memory.get_connection()

    # Get existing run_ids to skip duplicates
    existing = {row["run_id"] for row in
                conn.execute("SELECT run_id FROM run_history").fetchall()}

    imported = 0
    for r in runs:
        if r["run_id"] in existing:
            continue

        # Insert into run_history
        conn.execute(
            """INSERT INTO run_history
               (run_id, session_name, question_count, composite_score, loops_used,
                approved, created_at, api_usage_json, batch_id, superseded_by, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["run_id"], r["session_name"], r["question_count"],
             r["composite_score"], r["loops_used"], r["approved"],
             r["created_at"], json.dumps(r.get("api_usage") or {}),
             r.get("batch_id"), r.get("superseded_by"), r.get("error"))
        )

        # Insert into run_results if payload exists
        if r.get("payload"):
            conn.execute(
                "INSERT OR IGNORE INTO run_results (run_id, payload_json, created_at) VALUES (?, ?, ?)",
                (r["run_id"], json.dumps(r["payload"]), r["created_at"])
            )

        imported += 1

    conn.commit()
    conn.close()
    print(f"Imported {imported} new runs ({len(existing)} already existed).")
    return imported


if __name__ == "__main__":
    import_runs()
