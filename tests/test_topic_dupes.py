"""Near-duplicate clustering, and reading-material evidence per question.

Both exist because of one review finding: No-Code AI Automation asked about hallucination SIX times in a
38-question set, and one of those — *"Explain your methodology for designing and testing system prompts to
prevent model hallucination"* — turned out to be unsupported by the material.

THE MEASUREMENT THAT DICTATES THE DESIGN
----------------------------------------
Across those six questions, **zero pairs reach the 0.82 dedup bar**, and the two most obviously identical
— "What are hallucinations in LLMs" and "What is an AI hallucination" — score **0.486**, the LOWEST of all
fifteen pairs. Short definitional questions carry little signal, and "LLMs" vs "AI" pushes them apart. So
clustering groups by a SHARED DISTINCTIVE TERM and uses similarity only for cohesion and ordering. A test
that merely checked "the six are grouped" would pass under a similarity-only implementation on a luckier
example, so the 0.486 pair is asserted directly.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "filter_topic_sets", Path(__file__).resolve().parent.parent / "scripts" / "filter_topic_sets.py")


@pytest.fixture(scope="module")
def mod():
    m = importlib.util.module_from_spec(_SPEC)
    sys.modules["filter_topic_sets"] = m
    _SPEC.loader.exec_module(m)
    return m


HALLUCINATION_SIX = [
    "Explain your methodology for designing and testing system prompts to prevent model hallucination.",
    'How do you prevent hallucinations?"',
    "What are hallucinations in LLMs",
    "What strategies do you employ to mitigate hallucinations in generative AI models?",
    "What is an AI hallucination",
    "What is hallucination, and how can it be controlled using prompt engineering?",
]


def _rows(contents, status="backfilled"):
    return [{"content": c, "status": status, "content_norm": c.lower(), "first_run_id": None}
            for c in contents]


# A cluster term must appear in at most ~30% of the set, so a cluster is never just "the whole topic".
# That means a realistic set size is part of the condition: six hallucination questions are 16% of the
# real 38-question topic, but 100% of a six-item fixture, which the cap correctly refuses. Padding with
# unrelated questions is what makes the fixture represent the finding instead of the cap.
_FILLER = [f"What is unrelated concept number {i} in cloud infrastructure?" for i in range(32)]


def _realistic(contents, status="backfilled"):
    return _rows(list(contents) + _FILLER, status)


class TestSimilarityCannotGroupThem:
    def test_the_two_most_identical_questions_score_lowest(self):
        """The finding this whole design rests on. If this ever stops being true, revisit the approach —
        but do not 'just lower the threshold': 0.486 would collapse most distinct questions too."""
        from src import embeddings
        from src.config import DEDUP_SEMANTIC_THRESHOLD

        a, b = "What are hallucinations in LLMs", "What is an AI hallucination"
        sim = embeddings.cosine_matrix([a, b])
        if sim is None:
            pytest.skip("embeddings unavailable")
        score = float(sim[0][1])
        assert score < 0.55, f"{score:.3f} — expected these to score LOW despite being the same question"
        assert score < DEDUP_SEMANTIC_THRESHOLD, "and far below the dedup bar, which is why dedup missed it"

    def test_no_pair_of_the_six_reaches_the_dedup_bar(self):
        from src import embeddings
        from src.config import DEDUP_SEMANTIC_THRESHOLD

        sim = embeddings.cosine_matrix(HALLUCINATION_SIX)
        if sim is None:
            pytest.skip("embeddings unavailable")
        over = [(i, j) for i in range(6) for j in range(i + 1, 6)
                if float(sim[i][j]) >= DEDUP_SEMANTIC_THRESHOLD]
        assert not over, "if any pair now clears the bar, embedding dedup would have caught it"


class TestClusteringGroupsThem:
    def test_the_six_land_in_one_cluster(self, mod):
        rows = _realistic(HALLUCINATION_SIX)
        groups = mod._clusters(rows, [0.6] * len(rows))
        hal = [g for g in groups if "hallucinat" in g["term"]]
        assert hal, f"no hallucination cluster; found {[g['term'] for g in groups]}"
        assert len(hal[0]["ids"]) == 6, f"expected all six grouped, got {len(hal[0]['ids'])}"
        assert hal[0]["ids"] == set(range(6)), "and exactly the six, with no filler dragged in"

    def test_the_keeper_is_the_approved_best_fit_member(self, mod):
        rows = _realistic(HALLUCINATION_SIX)
        rows[3]["status"] = "approved"                # the mitigation question
        fits = [0.50, 0.51, 0.52, 0.62, 0.53, 0.71] + [0.3] * len(_FILLER)
        groups = mod._clusters(rows, fits)
        hal = [g for g in groups if "hallucinat" in g["term"]][0]
        keeper = hal["members"][0]
        assert rows[keeper]["status"] == "approved", "reviewer-blessed must outrank a better-scoring import"

    def test_unrelated_questions_sharing_a_word_do_not_cluster(self, mod):
        """Cohesion is the ONLY thing separating these — asserted by turning it off in the same test.

        The first version of this test used hallucination vs Kubernetes, which share no distinctive term
        at all, so it passed with the cohesion bar set to 0.0 and proved nothing. These two are verbatim
        from the live sets and share the stem `strategy` at cohesion **0.124**; across all nine topics the
        bar rejects 55 such term-groups.
        """
        rows = _rows([
            "How do you handle error propagation in autonomous agent systems, and what is your strategy "
            "for recovery?",
            "How would you design a prompt strategy to extract highly specific, structured JSON data from "
            "a document?",
        ])
        assert mod._clusters(rows, [0.6, 0.6]) == [], "a shared word is not a shared subject"
        assert mod._clusters(rows, [0.6, 0.6], min_cohesion=0.0), (
            "without the cohesion bar they DO group — which is what this test exists to prevent")

    def test_a_cluster_needs_at_least_two_members(self, mod):
        assert mod._clusters(_rows(["What is an AI hallucination"]), [0.6]) == []


class TestApplyingAClusterReachesTheProduct:
    """A cut that lands in `topic_question_set` but not in the canonical payload is invisible.

    This has bitten twice: quarantining 15 questions left four payloads showing their old counts, because
    `/review/<id>` and the Sheets export read the payload, not the table. So this asserts the PAYLOAD
    shrinks — the table count alone would pass in both the broken and the fixed world.
    """

    @pytest.fixture
    def db(self, tmp_path, monkeypatch, mod):
        from src import memory
        monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "t.db")
        monkeypatch.setattr(mod, "MEMORY_DB", tmp_path / "t.db")   # the script's own backup target
        memory.init_db()

        tk = "topic:dupes"
        memory.upsert_topic_questions(
            tk, [{"content": c, "status": "approved", "difficulty": "Medium", "source": "interview_db"}
                 for c in HALLUCINATION_SIX + _FILLER])
        memory.save_run_result("canon-1", {"output": {"question_details": [
            {"content": c, "question_id": c[:24]} for c in HALLUCINATION_SIX + _FILLER]}})
        memory.set_canonical_run(tk, "canon-1")
        return memory, tk

    def _apply(self, mod, monkeypatch, term):
        from types import SimpleNamespace
        monkeypatch.setattr(mod, "_topic_context", lambda s: SimpleNamespace(
            learning_outcomes=[], interview_topics=[], key_concepts=[], scope_in=[], scope_out=[]))
        monkeypatch.setattr(mod, "_session_profile", lambda s, c: ([], []))
        monkeypatch.setattr(mod, "_fits", lambda texts, cur, rm: [0.6] * len(texts))
        return mod._dupes({"topic:dupes": ["S1"]}, ["topic:dupes"],
                          SimpleNamespace(topic=None, collapse=term))

    def test_the_payload_shrinks_not_just_the_table(self, db, mod, monkeypatch):
        memory, tk = db
        before = len(memory.get_run_result("canon-1")["output"]["question_details"])
        assert before == 38

        self._apply(mod, monkeypatch, "hallucinat")

        assert len(memory.get_topic_questions(tk)) == 33, "five of the six are gone from the table"
        after = memory.get_run_result("canon-1")["output"]["question_details"]
        assert len(after) == 33, "and the payload Review reads was re-rendered — this is the step that ships"
        assert sum("hallucinat" in q["content"].lower() for q in after) == 1, "exactly one keeper"

    def test_the_losers_are_recoverable_never_deleted(self, db, mod, monkeypatch):
        """A judgement call about your interview must be reversible."""
        memory, tk = db
        self._apply(mod, monkeypatch, "hallucinat")
        q = memory.get_quarantined(tk)
        assert len(q) == 5
        assert all("duplicate of" in r["reason"] for r in q)
        assert all("hallucinat" in r["content"].lower() for r in q)

    def test_an_unmatched_term_changes_nothing(self, db, mod, monkeypatch):
        memory, tk = db
        self._apply(mod, monkeypatch, "kubernetes")
        assert len(memory.get_topic_questions(tk)) == 38
        assert memory.get_quarantined(tk) == []


class TestMaterialEvidence:
    def test_the_words_are_present_but_the_compound_concept_is_not(self):
        """The precise answer to "is this question supported?": every word appears, the CONCEPT does not.

        `system prompt` occurs 0 times while `hallucinat*` occurs 3 — the asymmetry is the finding, so
        both halves are asserted. A test on the zero alone would pass on material that mentions nothing.
        """
        import re

        from src.tools import _session_corpus

        corpus = _session_corpus(["Building Social Media Content Automation Workflow | Part 1",
                                  "Building Social Media Content Automation Workflow | Part 2",
                                  "Advanced Prompt Engineering"])
        if not corpus:
            pytest.skip("shipped reading material unavailable")
        assert len(re.findall(r"hallucinat", corpus, re.I)) >= 3, "the material DOES teach hallucination"
        assert len(re.findall(r"system prompts?", corpus, re.I)) == 0, (
            "it does NOT teach system prompts — which is why that question is off-syllabus")
        assert len(re.findall(r"methodolog", corpus, re.I)) == 0

    def test_concept_is_absent_must_not_be_used_as_a_detector(self):
        """It is a VERIFIER for an LLM claim and stays silent when unsure, so it finds nothing on its own.

        Pinned because wiring it up as a detector looks obviously right and yields zero detections.
        """
        from src.tools import _concept_is_absent, _session_corpus

        corpus = _session_corpus(["Building Social Media Content Automation Workflow | Part 1"])
        if not corpus:
            pytest.skip("shipped reading material unavailable")
        assert _concept_is_absent("Split In Batches node", corpus) is False, (
            "it reports False even for a phrase with 0 occurrences — do not use it to detect")


class TestTrailingQuoteArtifact:
    def test_a_quote_after_the_question_mark_is_scrape_residue(self):
        from src.quality import strip_artifacts
        assert strip_artifacts('How do you prevent hallucinations?"') == \
            "How do you prevent hallucinations?"

    def test_a_genuine_quotation_is_left_alone(self):
        """Narrow on purpose: measured across both banks, exactly one row matches and none has a quote
        anywhere but after a terminal mark."""
        from src.quality import strip_artifacts
        for text in ['He asked "what is RAG?" in the interview',
                     'Explain the term "hallucination"',
                     "What is RAG?"]:
            assert strip_artifacts(text) == text
