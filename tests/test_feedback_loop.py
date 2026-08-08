"""Tests for the reviewer-feedback loop.

Context for why these exist: after 36 runs and 68 rejections the system had learned exactly nothing —
`data/learned_rules.md` held zero rules, the `suppress_boost` table was empty with no writer, and
rejections were keyed on the COMBINED run name so they never transferred between session
combinations. These tests pin each of those repairs.

DB-touching tests are redirected to a temporary database so they never write to the real memory.db.
"""
import json
import re
from pathlib import Path

import pytest

from src import memory
from src.rejection_rules import REJECTION_RULES, rule_for


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point memory at a throwaway SQLite file and create the schema in it."""
    db = tmp_path / "test_memory.db"
    monkeypatch.setattr(memory, "MEMORY_DB", db)
    memory.init_db()
    return db


class TestSessionKeySplitting:
    def test_splits_a_combined_run_name(self):
        assert memory._split_sessions("A + B + C") == ["A", "B", "C"]

    def test_single_session_is_unchanged(self):
        assert memory._split_sessions("Introduction to AI Agents") == ["Introduction to AI Agents"]

    def test_preserves_pipes_inside_a_session_name(self):
        """"| Part 1" is part of the name — only " + " separates sessions."""
        assert memory._split_sessions(
            "Build Your Own AI News Summarizer | Part 1 + Advanced Prompt Engineering"
        ) == ["Build Your Own AI News Summarizer | Part 1", "Advanced Prompt Engineering"]

    def test_trims_whitespace(self):
        assert memory._split_sessions("A  +  B") == ["A", "B"]

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input(self, value):
        assert memory._split_sessions(value) == []


class TestRejectionSuppressionTransfers:
    """The core bug: a rejection learned in one combination must apply in every other combination
    containing that session."""

    def test_rejection_transfers_to_a_different_combination(self, temp_db):
        memory.record_rejections("Intro to Agents + Learning Path Generator", ["Tell me about yourself"])
        norms = memory.get_rejected_norms("Intro to Agents + Deploying LLM Applications")
        assert memory.normalize_content("Tell me about yourself") in norms

    def test_rejection_applies_to_the_session_run_alone(self, temp_db):
        memory.record_rejections("Intro to Agents + Learning Path Generator", ["Tell me about yourself"])
        assert memory.get_rejected_norms("Intro to Agents")

    def test_unrelated_session_is_not_suppressed(self, temp_db):
        memory.record_rejections("Intro to Agents", ["Tell me about yourself"])
        assert not memory.get_rejected_norms("Mastering Image Generation")

    def test_recording_is_idempotent(self, temp_db):
        first = memory.record_rejections("A + B", ["What is RAG?"])
        second = memory.record_rejections("A + B", ["What is RAG?"])
        assert first > 0 and second == 0, "re-rejecting the same question must not add rows"

    def test_blank_content_is_ignored(self, temp_db):
        assert memory.record_rejections("A", ["", "   "]) == 0

    def test_normalization_ignores_case_and_punctuation(self, temp_db):
        memory.record_rejections("A", ["What is RAG?"])
        assert memory.normalize_content("what is rag") in memory.get_rejected_norms("A")

    def test_empty_session_name_returns_nothing(self, temp_db):
        assert memory.get_rejected_norms("") == set()

    def test_backfill_is_idempotent(self, temp_db):
        memory.record_rejections("A + B", ["What is RAG?"])
        memory._backfill_rejections_per_session()
        assert memory._backfill_rejections_per_session() == 0


class TestLearnedRules:
    def test_rule_is_appended_and_readable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "learned_rules.md")
        assert memory.append_learned_rule("Reject if the question is off-topic.")
        assert "Reject if the question is off-topic." in memory.get_learned_rules()

    def test_duplicate_rule_is_not_appended_twice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "learned_rules.md")
        memory.append_learned_rule("Reject if X.")
        assert memory.append_learned_rule("Reject if X.") is False
        assert memory.get_learned_rules().count("Reject if X.") == 1

    def test_rejects_markup_that_would_corrupt_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "learned_rules.md")
        assert memory.append_learned_rule("## Rules") is False
        assert memory.append_learned_rule("<!-- comment -->") is False
        assert memory.append_learned_rule("") is False

    def test_missing_file_yields_no_rules_rather_than_raising(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "_RULES_FILE", tmp_path / "does_not_exist.md")
        assert memory.get_learned_rules() == []


class TestRejectionTaxonomyStaysInSyncWithTheUI:
    """The review UI sends a reason KEY; the server maps it to a canonical rule. If the two lists
    drift, a reviewer's click silently teaches nothing — exactly the failure this replaced."""

    REVIEW_JSX = Path(__file__).resolve().parent.parent / "frontend" / "src" / "pages" / "Review.jsx"

    def _ui_keys(self) -> set[str]:
        block = re.search(r"const REJECT_REASONS = \[(.*?)\]", self.REVIEW_JSX.read_text(), re.S)
        assert block, "REJECT_REASONS not found in Review.jsx"
        return set(re.findall(r"key:\s*'([^']+)'", block.group(1)))

    def test_every_ui_reason_has_a_server_rule(self):
        missing = self._ui_keys() - set(REJECTION_RULES)
        assert not missing, f"UI sends reason keys with no rule in REJECTION_RULES: {missing}"

    def test_no_orphan_server_rules(self):
        orphans = set(REJECTION_RULES) - self._ui_keys()
        assert not orphans, f"REJECTION_RULES entries no reason chip can produce: {orphans}"

    def test_rules_are_actionable_reject_statements(self):
        for key, rule in REJECTION_RULES.items():
            assert rule.lower().startswith(("reject if", "skip if")), f"{key}: {rule!r}"
            assert len(rule) <= 200, f"{key}: rule too long to store ({len(rule)} chars)"

    def test_taxonomy_key_resolves_without_an_llm_call(self):
        """A chip click must map straight to a rule — the whole point is determinism."""
        assert rule_for("off_topic") == REJECTION_RULES["off_topic"]

    def test_free_text_reason_is_not_treated_as_a_taxonomy_key(self):
        """Unknown reasons must return None so the caller falls back to LLM distillation."""
        assert rule_for("this one felt too easy for the session") is None
        assert rule_for("") is None


class TestFeedbackExamplesFile:
    def test_shipped_file_has_the_expected_shape(self):
        """run_eval and the relevance judge both read this file; a shape change breaks both."""
        rows = memory.get_feedback_examples()
        if not rows:
            pytest.skip("no reviewer decisions recorded yet")
        for row in rows:
            assert set(row) >= {"session", "question", "decision"}
            assert row["decision"] in ("good", "bad")

    def test_file_is_valid_json_list(self):
        path = memory._FEEDBACK_EXAMPLES
        if not path.exists():
            pytest.skip("no feedback file yet")
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
