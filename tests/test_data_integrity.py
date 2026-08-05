"""Data-integrity tests over the shipped data/ files.

These exist because the failures they catch are SILENT at runtime: a session missing from
session_map.json makes `understand_session` fall back to the knowledge graph without erroring, so
question quality quietly degrades for that session and nobody finds out until a reviewer rejects the
whole set. Three such sessions were found this way.

They read the real data files rather than fixtures — the point is to verify the shipped data.
"""
import json

import pytest

from src.config import DATA_DIR, GENAI_BANK_JSON, INTERVIEW_QUESTIONS_JSON

SESSION_MAP = DATA_DIR / "reading_materials" / "session_map.json"
SESSION_OUTCOMES = DATA_DIR / "reading_materials" / "session_outcomes.json"
COURSE_STRUCTURE = DATA_DIR / "course_structure.json"

# Sessions present in course_structure.json that have no reading material yet. Listed explicitly so
# the test reports genuinely NEW gaps instead of failing on the known backlog. Shrink this list as
# the material is authored; never grow it without a reason.
KNOWN_MISSING_READING_MATERIAL = {
    "Your Learning Journey",
    "Enhancing Productivity with AI",
    "Fine-Tuning LLMs | Part 2",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _course_sessions() -> list[str]:
    """Flatten course_structure.json (topic → [session, …]) into one session list."""
    out = []
    for sessions in _load(COURSE_STRUCTURE).values():
        if isinstance(sessions, list):
            out.extend(s for s in sessions if isinstance(s, str))
    return out


class TestSessionCoverage:
    def test_no_new_sessions_missing_reading_material(self):
        missing = set(_course_sessions()) - set(_load(SESSION_MAP)) - KNOWN_MISSING_READING_MATERIAL
        assert not missing, (
            f"session(s) with no reading material (silent knowledge-graph fallback): {sorted(missing)}. "
            f"Either build their material or add them to KNOWN_MISSING_READING_MATERIAL."
        )

    def test_session_map_has_no_orphans(self):
        """Reading material for a session nobody can select is dead weight and usually a typo."""
        orphans = set(_load(SESSION_MAP)) - set(_course_sessions())
        assert not orphans, f"session_map entries not in course_structure: {sorted(orphans)}"

    def test_reading_material_is_substantial(self):
        """A near-empty entry passes the exact-key lookup but teaches the pipeline nothing."""
        thin = {k: len(v) for k, v in _load(SESSION_MAP).items()
                if not isinstance(v, str) or len(v) < 500}
        assert not thin, f"reading material too short to be useful: {thin}"

    def test_curated_outcomes_exist_for_mapped_sessions(self):
        """The session-fit gate leans on curated outcomes; missing ones silently weaken it."""
        outcomes = _load(SESSION_OUTCOMES)
        missing = [s for s in _load(SESSION_MAP) if s not in outcomes]
        assert not missing, f"sessions with reading material but no curated outcomes: {sorted(missing)}"


class TestQuestionBanks:
    def test_interview_bank_is_fully_attributed(self):
        """Placement data is the one source with verified employers — keep it that way."""
        rows = _load(INTERVIEW_QUESTIONS_JSON)["questions"]
        unattributed = [r["id"] for r in rows if not (r.get("company") or "").strip()]
        assert not unattributed, f"{len(unattributed)} interview-bank rows lost their company"

    @pytest.mark.skipif(not GENAI_BANK_JSON.exists(), reason="GenAI bank not built")
    def test_genai_bank_has_no_form_garbage(self):
        """The bank is gated at build time (build_genai_bank) and swept by scripts/clean_bank.py;
        this fails if either regressed or if rows were added by hand."""
        from src.quality import is_quality_question

        bad = [r["content"] for r in _load(GENAI_BANK_JSON) if not is_quality_question(r["content"])]
        assert not bad, (f"{len(bad)} malformed question(s) in the GenAI bank — run "
                         f"`python scripts/clean_bank.py`. First few: {bad[:5]}")

    @pytest.mark.skipif(not GENAI_BANK_JSON.exists(), reason="GenAI bank not built")
    def test_genai_bank_companies_are_plausible(self):
        """No page-topic word or source-site brand should be presented as an employer."""
        from src.sources.tavily_search import _valid_company

        bogus = sorted({c for r in _load(GENAI_BANK_JSON)
                        if (c := (r.get("company") or "").strip()) and _valid_company(c) is None})
        assert not bogus, f"implausible company attributions in the GenAI bank: {bogus}"

    @pytest.mark.skipif(not GENAI_BANK_JSON.exists(), reason="GenAI bank not built")
    def test_genai_bank_ids_are_unique(self):
        ids = [r["id"] for r in _load(GENAI_BANK_JSON)]
        assert len(ids) == len(set(ids)), "duplicate question ids in the GenAI bank"
