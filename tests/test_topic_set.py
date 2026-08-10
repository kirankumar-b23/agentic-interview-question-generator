"""The accumulating per-topic question set, and the consolidation of historical runs into it.

Re-running a topic used to produce a separate version — new run, new review, new spreadsheet. Now each
topic has ONE canonical run holding a set that grows, and the 53 historical runs were merged into 9.

What these tests protect, in order of how badly it would hurt to break:

1. **Nothing is deleted.** `run_results` payloads are the replay data every threshold decision in this
   project was calibrated from (the coverage gate, the relevance back-fill target, the de-stack verdict).
   The consolidation flags rows; it must never overwrite or remove a payload.
2. **The canonical run is synthetic.** `save_run_result` is INSERT OR REPLACE, so writing the merged set
   onto an existing run would destroy that run's own question list. A fresh id keeps every original.
3. **It is idempotent and does not eat its own output.** Running it twice minted new ids (via randomised
   `hash()`) and then grouped the canonical runs into their own topics, flagging them superseded — after
   which History showed 2 rows instead of 9.
4. **Identity is normalized content, not question_id**, which regenerates every run.
"""
import json

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src import memory
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
    memory.init_db()
    return memory


class TestTopicKey:
    def test_a_joined_run_name_resolves_to_its_topic(self, db):
        """Differently-grouped runs of one topic must share a set, so the key cannot be the run name."""
        a = db.topic_key_for("Build Your Own AI News Summarizer | Part 1")
        b = db.topic_key_for("Build Your Own AI News Summarizer | Part 1 + "
                             "Build Your Own AI News Summarizer | Part 2")
        assert a == b, "one session and a grouping containing it must land on the same topic"

    def test_an_unknown_session_never_keys_on_none(self, db):
        """A None key would pool every custom run into one bucket, merging unrelated sets."""
        k1 = db.topic_key_for("Some Bespoke Session")
        k2 = db.topic_key_for("A Totally Different Bespoke Session")
        assert k1 and k2 and k1 != k2
        assert k1.startswith("session:")


class TestAccumulation:
    def test_identity_is_normalized_content_not_question_id(self, db):
        """`question_bank` keys on question_id, which regenerates per run, so the same content banked
        repeatedly. Punctuation and case must not create a second row."""
        assert db.upsert_topic_questions("T", [{"content": "What is RAG?"}]) == 1
        assert db.upsert_topic_questions("T", [{"content": "what is rag"}]) == 0
        assert len(db.get_topic_questions("T")) == 1

    def test_a_second_run_adds_only_what_is_new(self, db):
        db.upsert_topic_questions("T", [{"content": "Q one"}, {"content": "Q two"}])
        added = db.upsert_topic_questions("T", [{"content": "Q two"}, {"content": "Q three"}])
        assert added == 1
        assert {q["content"] for q in db.get_topic_questions("T")} == {"Q one", "Q two", "Q three"}

    def test_approval_upgrades_status_but_never_downgrades(self, db):
        """A reviewer accepting a backfilled question is new information; the reverse is not."""
        db.upsert_topic_questions("T", [{"content": "Q", "status": "backfilled"}])
        db.upsert_topic_questions("T", [{"content": "Q", "status": "approved"}])
        assert db.get_topic_questions("T")[0]["status"] == "approved"
        db.upsert_topic_questions("T", [{"content": "Q", "status": "backfilled"}])
        assert db.get_topic_questions("T")[0]["status"] == "approved"

    def test_approved_questions_ship_first(self, db):
        db.upsert_topic_questions("T", [{"content": "later", "status": "backfilled"},
                                        {"content": "blessed", "status": "approved"}])
        assert db.get_topic_questions("T")[0]["content"] == "blessed"

    def test_the_full_question_detail_survives(self, db):
        """Reconstructing from columns would drop role, source_url, session_fit and the rest."""
        detail = {"content": "Q", "role": "Prompt Engineer", "session_fit": 0.61,
                  "asked_in_company": "ACME", "source_url": "https://x"}
        db.upsert_topic_questions("T", [{"content": "Q", "detail": detail}])
        got = json.loads(db.get_topic_questions("T")[0]["detail_json"])
        assert got["role"] == "Prompt Engineer" and got["session_fit"] == 0.61

    def test_removing_a_question_drops_it_from_the_set(self, db):
        db.upsert_topic_questions("T", [{"content": "Q one"}, {"content": "Q two"}])
        assert db.remove_topic_question("T", "Q one") is True
        assert {q["content"] for q in db.get_topic_questions("T")} == {"Q two"}
        assert db.remove_topic_question("T", "never there") is False


class TestQuarantine:
    def test_a_quarantined_question_is_kept_and_out_of_the_set(self, db):
        """"Without losing the data" without re-admitting what the improvements removed."""
        db.quarantine_question("T", "how you would iteratively improve prompts and guards.",
                               "fails the form gate", "run-1")
        assert db.get_topic_questions("T") == []
        q = db.get_quarantined("T")
        assert len(q) == 1 and "form gate" in q[0]["reason"]


class TestCanonicalRunAndSheet:
    def test_the_canonical_run_is_recorded_per_topic(self, db):
        db.set_canonical_run("T", "run-a")
        assert db.get_canonical_run("T") == "run-a"
        db.set_canonical_run("T", "run-b")
        assert db.get_canonical_run("T") == "run-b"
        assert db.get_canonical_run("other") is None

    def test_the_sheet_keeps_its_lms_identifiers_across_exports(self, db):
        """org_id/interview_id are fresh uuid4()s per call in sheets_writer, so a re-export with new ones
        looks like a brand-new interview to the LMS import."""
        db.save_topic_sheet("T", "sid", "https://s", org_id="org-1", interview_id="int-1")
        db.save_topic_sheet("T", "sid", "https://s")          # a later export passes none
        s = db.get_topic_sheet("T")
        assert s["org_id"] == "org-1" and s["interview_id"] == "int-1"

    def test_marking_superseded_deletes_nothing(self, db):
        db.save_run(run_id="r1", session_name="S", question_count=5, composite_score=0.5, loops_used=1)
        db.save_run_result("r1", {"output": {"question_details": [{"content": "Q"}]}})
        db.mark_superseded("r1", "canonical-1")
        payload = db.get_run_result("r1")
        assert payload, "the payload must survive being superseded — it is the replay data"
        assert payload["output"]["question_details"][0]["content"] == "Q"


class TestHistoryHidesSupersededRuns:
    def test_superseded_runs_are_hidden_by_default_but_still_readable(self, db):
        """After consolidating, showing the originals alongside the canonical run made History LONGER
        (62 rows) than before, which is the opposite of the ask."""
        db.save_run(run_id="canon", session_name="Topic", question_count=40,
                    composite_score=0.7, loops_used=1)
        for i in range(3):
            db.save_run(run_id=f"old{i}", session_name="Topic", question_count=5,
                        composite_score=0.5, loops_used=1)
            db.mark_superseded(f"old{i}", "canon")

        default = db.get_run_history()
        assert [r["run_id"] for r in default] == ["canon"]
        assert len(db.get_run_history(include_superseded=True)) == 4
        assert default[0]["question_count"] == 40

    def test_a_run_that_was_never_superseded_still_shows(self, db):
        db.save_run(run_id="solo", session_name="S", question_count=6, composite_score=0.6, loops_used=1)
        assert [r["run_id"] for r in db.get_run_history()] == ["solo"]
