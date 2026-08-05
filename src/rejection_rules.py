"""Canonical validation rule per rejection-reason key sent by the review UI.

Lives here rather than in the web layer because it is domain knowledge, not HTTP concerns: the
pipeline consumes these rules (via `memory.append_learned_rule` →
`tools.tool_validate_relevance`), and the eval harness needs them without importing a server.

The keys must match `REJECT_REASONS` in `frontend/src/pages/Review.jsx` — `tests/test_feedback_loop.py`
asserts that, because a drift there means a reviewer's click silently teaches nothing.

These are pre-written rather than LLM-distilled: a taxonomy click already states exactly what was
wrong, so a round-trip through the model would only risk paraphrasing it into a near-duplicate of a
rule that already exists. Free-text reasons still go through `memory.distill_rule`.
"""

REJECTION_RULES: dict[str, str] = {
    "off_topic": (
        "Reject if the question is about a different technology or domain than the session teaches, "
        "even if a keyword overlaps."
    ),
    "too_generic": (
        "Reject if the question is so broad it could be asked for any AI course and does not test a "
        "specific concept from this session."
    ),
    "not_grounded": (
        "Reject if the concept the question tests is not taught anywhere in this session's reading "
        "material."
    ),
    "experience": (
        "Reject if the question asks the candidate to describe their own past experience, projects or "
        "workflows instead of testing a concept."
    ),
    "not_question": (
        "Reject if the text is a page heading, article title or sentence fragment rather than a "
        "well-formed standalone interview question."
    ),
    "duplicate": (
        "Reject if the question restates another question already in the set, even in different words."
    ),
    "wrong_level": (
        "Reject if the question's difficulty is far above or below what this session teaches."
    ),
}


def rule_for(reason: str) -> str | None:
    """Canonical rule for a taxonomy key, or None when `reason` is free text to be distilled."""
    return REJECTION_RULES.get((reason or "").strip())
