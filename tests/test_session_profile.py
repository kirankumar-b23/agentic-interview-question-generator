"""The grounding profile must contain the whole reading material, not a sample of it.

`_session_profile` used to keep ~36% of a session's material. Three lossy steps compounded: a
`len(c) > 120` filter that dropped short paragraphs, a `chunks[::step]` STRIDE SAMPLE down to 12, and a
`[:800]` truncation. Measured on the No-Code sessions, 24,521 chars became 8,728.

The consequence was not cosmetic. The line that defines the node most precisely is 85 characters:

    '- **HTTP Request Node**: Allows n8n to talk to almost any web service that has an API.'

so the filter discarded it, the phrase appeared in NONE of the 12 surviving chunks, and
"What is the HTTP Request node and when do you use it?" scored **0.275** against a session that
literally teaches it. `_score_session_fit` DROPS candidates on a floor derived from that score, and it is
the largest cut in the funnel (107-169 per run) — so tool-specific questions were being discarded at
retrieval time. `session_grounding` (20% of the composite), `_rank_key` and `_attribute_sessions` all read
the same profile.

These tests use the SHIPPED reading material rather than a fixture, because the defect was invisible to
crafted inputs: any short synthetic paragraph would have been dropped by the same filter and the test
would have "passed" while describing nothing real.
"""
import re

import pytest

SESSIONS = ["Building Social Media Content Automation Workflow | Part 1",
            "Building Social Media Content Automation Workflow | Part 2",
            "Advanced Prompt Engineering"]
QUESTION = "What is the HTTP Request node and when do you use it?"


@pytest.fixture(scope="module")
def profile():
    from types import SimpleNamespace
    from src.pipeline import _session_profile
    ctx = SimpleNamespace(learning_outcomes=[], interview_topics=[], key_concepts=[])
    curated, rm = _session_profile(SESSIONS, ctx)
    if not rm:
        pytest.skip("shipped reading material not available")
    return curated, rm


class TestTheMaterialActuallyReachesTheProfile:
    def test_the_short_bullet_that_defines_the_node_survives(self, profile):
        """85 characters, and the single most useful line about the node. The old >120 filter cut it."""
        _curated, rm = profile
        assert any(re.search(r"HTTP Request", c, re.I) for c in rm), (
            "the phrase is in the reading material; if it is missing here the profile is sampling again")

    def test_the_profile_is_not_a_stride_sample(self, profile):
        """12 chunks was the old cap AND the sampling target. A real session yields far more."""
        _curated, rm = profile
        assert len(rm) > 40, f"only {len(rm)} chunks — the profile looks sampled, not complete"

    def test_no_chunk_is_a_hard_truncation_at_the_old_limit(self, profile):
        """Long paragraphs are sub-split at a word boundary now, not cut at 800 chars mid-word."""
        _curated, rm = profile
        assert not any(len(c) == 800 for c in rm)


class TestTheQuestionThisBugHid:
    def test_a_question_about_a_taught_node_scores_as_grounded(self, profile):
        """The regression this file exists for: 0.275 before, ~0.54 after.

        A future "tidy the chunker" change that reintroduces sampling fails here.
        """
        from src import embeddings
        from src.config import SESSION_PROFILE_RM_WEIGHT

        curated, rm = profile
        cm = embeddings.cosine_matrix([QUESTION], curated) if curated else None
        bm = embeddings.cosine_matrix([QUESTION], rm)
        if bm is None:
            pytest.skip("embeddings unavailable")
        curated_best = float(max(cm[0])) if cm is not None else 0.0
        rm_best = float(max(bm[0])) * SESSION_PROFILE_RM_WEIGHT
        fit = max(curated_best, rm_best)
        assert fit > 0.45, (
            f"fit {fit:.3f} — a question about a node this session teaches must not read as ungrounded; "
            f"`_score_session_fit` drops candidates on this number")

    def test_the_reading_material_carries_it_not_the_curated_outcomes(self, profile):
        """Why the RM half matters: `interview_topics` are deliberately tool-agnostic, so the curated
        texts alone cannot ground a question that names a specific node."""
        from src import embeddings

        curated, rm = profile
        cm = embeddings.cosine_matrix([QUESTION], curated) if curated else None
        bm = embeddings.cosine_matrix([QUESTION], rm)
        if cm is None or bm is None:
            pytest.skip("embeddings unavailable")
        assert float(max(bm[0])) > float(max(cm[0])), (
            "the material should match a node question better than the tool-agnostic outcomes do")


class TestTheCapIsAGuardNotABudget:
    def test_the_cap_is_far_above_a_real_session(self):
        """Conflating a runaway guard with a sampling budget is what caused the bug."""
        from src.config import SESSION_PROFILE_RM_CHUNKS
        assert SESSION_PROFILE_RM_CHUNKS >= 200, (
            "this is a runaway guard; setting it near a real session's chunk count resumes sampling")

    def test_short_paragraphs_are_kept(self):
        from src.config import RM_CHUNK_MIN_CHARS
        assert RM_CHUNK_MIN_CHARS <= 60, (
            "the bullets ARE the tool definitions in this curriculum — an 85-char line must survive")
