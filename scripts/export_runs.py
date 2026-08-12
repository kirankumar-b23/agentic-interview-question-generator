"""Export run results from memory.db to data/exported_runs.json for cross-machine sharing."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import memory

EXPORT_PATH = Path(__file__).parent.parent / "data" / "exported_runs.json"


def export_runs():
    memory.init_db()
    runs = memory.get_run_history(limit=1000, include_superseded=True)
    if not runs:
        print("No runs to export.")
        return

    exported = []
    for r in runs:
        payload = memory.get_run_result(r["run_id"])
        exported.append({
            "run_id": r["run_id"],
            "session_name": r["session_name"],
            "question_count": r["question_count"],
            "composite_score": r["composite_score"],
            "loops_used": r["loops_used"],
            "approved": r["approved"],
            "created_at": r["created_at"],
            "api_usage": r["api_usage"],
            "batch_id": r.get("batch_id"),
            "superseded_by": r.get("superseded_by"),
            "error": r.get("error"),
            "payload": payload,
        })

    data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(exported),
        "runs": exported,
    }

    EXPORT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(exported)} runs to {EXPORT_PATH}")


if __name__ == "__main__":
    export_runs()
