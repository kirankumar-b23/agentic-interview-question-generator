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


# MCQ tells. These are assessment items, not interview questions — nobody is asked to "match the
# following" in an interview. The form gate does NOT catch them: 61% of the 1,772 rows in
# data/curriculum/*.json pass `is_quality_question` ("Which of the following images represents the node
# used to send HTTP requests…"), which is why this is asserted against the shipped corpora directly.
MCQ_SHAPES = (
    "which of the following", "match the following", "statement i:", "statement ii:",
    "all of the above", "none of the above", "choose the correct option", "select the correct",
    "options:", "\\_\\_\\_", "________",
)

# The tells above are all PROSE tells, and that was the hole: they carried nothing for LETTERED
# options, so `interview_questions.json` sat on 52 MCQs with this file green — C/Java syntax items and
# aptitude problems ("Two oranges, 3 bananas and 4 apples cost Rs.15…"), none company-attributed.
# `src.assessment_items` owns the shape rules so the sweeper script and this assertion cannot drift.


class TestNoAssessmentItemsInTheRetrievalCorpus:
    """The curriculum MCQs must never reach retrieval.

    A startup line read "Loaded 1819 curriculum questions into bank" immediately above "Question bank
    ready: 1509 questions indexed", so it looked as though course MCQs were being retrieved. They were
    not — the list was read by nothing — but the dead loader has been removed and these tests pin the
    OUTCOME (nothing MCQ-shaped is in the corpora) rather than the absence of a function.
    """

    @pytest.mark.parametrize("path,key", [(INTERVIEW_QUESTIONS_JSON, "questions"), (GENAI_BANK_JSON, None)])
    def test_no_mcq_shaped_questions_in_the_banks(self, path, key):
        data = _load(path)
        rows = data[key] if key else data
        bad = [r["content"] for r in rows
               if any(tell in (r.get("content") or "").lower() for tell in MCQ_SHAPES)]
        assert not bad, (f"{len(bad)} assessment-item(s) in {path.name} — these are MCQs, not "
                         f"interview questions. First few: {bad[:3]}")

    @pytest.mark.parametrize("path,key", [(INTERVIEW_QUESTIONS_JSON, "questions"), (GENAI_BANK_JSON, None)])
    def test_no_lettered_option_items_in_the_banks(self, path, key):
        """The gap the prose tells left: an option LIST rather than an option phrase.

        Fix with `python scripts/strip_assessment_items.py`, then clear `.cache/`.
        """
        from src.assessment_items import is_assessment_item

        data = _load(path)
        rows = data[key] if key else data
        bad = [r["content"] for r in rows if is_assessment_item(r.get("content") or "")]
        assert not bad, (f"{len(bad)} lettered-option MCQ(s) in {path.name}. First few: {bad[:3]}")

    @pytest.mark.parametrize("path,key", [(INTERVIEW_QUESTIONS_JSON, "questions"), (GENAI_BANK_JSON, None)])
    def test_no_answer_is_glued_onto_a_question(self, path, key):
        """A real question carrying its own answer is a scrape artifact, not an assessment item.

        `"What's RLHF, and why does it matter?A. RLHF (Reinforcement Learning from Human Feedback)…"`
        would ship to a reviewer as a 30-word blob that states its own answer. Repaired, not deleted —
        the word ceiling in `quality.py` is not a substitute: only 2 of the 6 exceeded 40 words.
        """
        from src.assessment_items import strip_glued_answer

        data = _load(path)
        rows = data[key] if key else data
        bad = [r["content"] for r in rows
               if strip_glued_answer(r.get("content") or "") != (r.get("content") or "").strip()]
        assert not bad, (f"{len(bad)} row(s) in {path.name} carry a glued answer. First few: {bad[:2]}")

    def test_the_runtime_loader_exposes_no_curriculum_questions(self):
        """The pool that fed the misleading log line is gone, not merely unread."""
        from src.data_loader import get_data_store

        assert not hasattr(get_data_store(), "curriculum_questions")

    def test_dropping_the_loader_cost_the_kp_catalog_nothing(self):
        """The removed function also merged KPs — but all 106 it referenced were already in the graph."""
        from src.data_loader import get_data_store

        assert len(get_data_store().kp_catalog) >= 113

    def test_curriculum_files_remain_on_disk_as_build_inputs(self):
        """`scripts/build_knowledge_graph.py` regenerates knowledge_graph.json from these by literal
        path, so ignoring them at runtime must not become deleting them."""
        curriculum = DATA_DIR / "curriculum"
        assert (curriculum / "gen_ai_final.json").exists()
        assert (curriculum / "llm_applications_kp_links_final_fixed.json").exists()

    def test_config_declares_no_runtime_path_to_them(self):
        """An unused constant is what invited the dead loader in the first place."""
        import src.config as cfg

        for name in ("GEN_AI_JSON", "LLM_APPS_JSON", "FLASK_JSON"):
            assert not hasattr(cfg, name), f"{name} is back — the curriculum files are build-time only"
