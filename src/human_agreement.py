"""Human-agreement scoring — the one quality signal that does NOT come from the selector.

Every other metric in this project is produced by the same machinery that picks the questions.
`relevance` in particular is the mean of the LLM relevance score used to *select* the set, so a run
can score 0.9 while a reviewer rejects 12 of its 15 questions (this actually happened: measured
correlation between composite score and reviewer approval across 36 runs was r = 0.16).

This module scores a question set against the reviewer's OWN past decisions
(`eval/feedback_examples.json`, written by `memory.record_feedback`). It is a nearest-neighbour
estimate, not a trained model: embed the known-accepted and known-rejected question texts, then for
each candidate compare its similarity to each side.

MEASURED BEHAVIOUR (92 labels, half held out, all-MiniLM-L6-v2): 65% accuracy against held-out
human decisions versus a 23% always-accept baseline. Recall on known-GOOD questions was 1.00 — it
never predicted a reviewer-approved question would be rejected — while 45% of known-BAD questions
were still predicted acceptable. So it is CONSERVATIVE: `predicted_accept_rate` is an optimistic
upper bound, and a low value is far more meaningful than a high one. Re-measure after label growth.

Deliberately kept dependency-light and honest about its limits:
  * It reports how many labels informed the estimate — with few labels the number means little.
  * `predict_accept` takes the label list as an argument so a caller can hold labels OUT
    (see eval/run_eval.py), which is what keeps the eval independent once the pipeline starts
    learning from the same feedback.
  * Returns None (not a fake 0.0) when it cannot score, so callers can say "unknown" instead of
    reporting a number they did not measure.
"""
from __future__ import annotations

from dataclasses import dataclass

# A candidate is called "rejected-like" when it is closer to the rejected set than the accepted set
# by at least this margin. A margin (rather than a bare >) keeps near-ties out of the reject bucket,
# since neighbouring good/bad examples are often only a few hundredths apart.
DECISION_MARGIN = 0.02
# Similarity at/above this to a known-REJECTED question means "effectively the same question the
# reviewer already turned down", regardless of how close the accepted set is.
NEAR_DUPLICATE_REJECT = 0.90


@dataclass
class AgreementReport:
    """Outcome of scoring a question set against reviewer decisions."""
    predicted_accept_rate: float          # 0.0–1.0 fraction predicted acceptable
    n_scored: int                         # questions that received a prediction
    n_good_labels: int                    # accepted examples that informed the estimate
    n_bad_labels: int                     # rejected examples that informed the estimate
    repeats_rejected: int                 # candidates near-identical to an already-rejected question
    per_question: dict[str, bool]         # question text → predicted acceptable
    # WHICH labels produced this number: "this session", "code_heavy sessions", "all sessions".
    # Reported because a score from the wrong type must not be indistinguishable from a right one.
    label_pool: str = "all sessions"

    @property
    def label_count(self) -> int:
        return self.n_good_labels + self.n_bad_labels

    def summary(self) -> str:
        return (f"predicted_accept={self.predicted_accept_rate:.2f} "
                f"({self.n_scored} scored vs {self.n_good_labels} good / {self.n_bad_labels} bad "
                f"labels from {self.label_pool}; {self.repeats_rejected} repeat-rejections)")


def split_labels(examples: list[dict], holdout_fraction: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Deterministically split reviewer labels into (inform, holdout).

    Deterministic on the question text — NOT random — so repeated eval runs are comparable and the
    same question never lands on both sides. Use `inform` for anything that feeds generation and
    `holdout` for scoring, so the eval stays honest once the pipeline learns from feedback.
    """
    import hashlib

    inform, holdout = [], []
    cut = int(max(0.0, min(1.0, holdout_fraction)) * 1000)
    for ex in examples:
        q = (ex.get("question") or "").strip()
        if not q:
            continue
        bucket = int(hashlib.sha256(q.encode("utf-8")).hexdigest()[:6], 16) % 1000
        (holdout if bucket < cut else inform).append(ex)
    return inform, holdout


MIN_LABELS_PER_SIDE = 3   # below this a pool is too thin to prefer over a broader one


def _split(rows: list[dict]) -> tuple[list[str], list[str]]:
    good = [e["question"] for e in rows
            if e.get("decision") == "good" and (e.get("question") or "").strip()]
    bad = [e["question"] for e in rows
           if e.get("decision") == "bad" and (e.get("question") or "").strip()]
    return good, bad


def _label_texts(examples: list[dict], session: str | None,
                 session_type: str | None = None,
                 allow_pooled: bool = True) -> tuple[list[str], list[str], str]:
    """(accepted, rejected, pool_name) — the narrowest label pool that is thick enough to be useful.

    Three tiers, narrowest first, because reviewer taste is specific before it is general:

      1. **this session** — the same session name, folded so ordering differences don't miss
         (`session_types.type_for_run` handles the " + "-joined names labels are keyed on);
      2. **this session type** — a code-heavy session judged by code-heavy decisions. This is the tier
         that matters: an implementation question resembles the "too specific, not conceptual" pattern
         the reviewer established on THEORY material, so pooling the two mis-calibrates both;
      3. **everything** — only when `allow_pooled`, and the caller is told which tier was used.

    Returning the pool name is the point: a score computed against the wrong type must not look
    identical to one computed against the right type.
    """
    from src.session_types import type_for_run

    rows = [e for e in (examples or []) if (e.get("question") or "").strip()]

    if session:
        target = type_for_run(session)
        same_session = [e for e in rows if type_for_run(e.get("session") or "") is not None
                        and (e.get("session") or "").strip() == (session or "").strip()]
        s_good, s_bad = _split(same_session)
        if len(s_good) >= MIN_LABELS_PER_SIDE and len(s_bad) >= MIN_LABELS_PER_SIDE:
            return s_good, s_bad, "this session"
        # Fall through to the type tier, resolving the session's type from the name if not given.
        session_type = session_type or target

    if session_type:
        same_type = [e for e in rows if type_for_run(e.get("session") or "") == session_type]
        t_good, t_bad = _split(same_type)
        if len(t_good) >= MIN_LABELS_PER_SIDE and len(t_bad) >= MIN_LABELS_PER_SIDE:
            return t_good, t_bad, f"{session_type} sessions"
        if not allow_pooled:
            # The caller asked for a type-specific score and there isn't one. Say so with empty
            # sides; predict_accept turns that into None rather than a number from the wrong pool.
            return t_good, t_bad, f"{session_type} sessions (insufficient)"

    good, bad = _split(rows)
    return good, bad, "all sessions"


def predict_accept(questions: list[str], examples: list[dict],
                   session: str | None = None, session_type: str | None = None,
                   allow_pooled: bool = True) -> AgreementReport | None:
    """Predict which `questions` the reviewer would accept, from their past decisions.

    `session_type` narrows the label pool to decisions made on sessions of the same type. With
    `allow_pooled=False` a type that lacks labels returns None instead of borrowing another type's
    taste — which is what an honest per-type eval needs.

    Returns None when the estimate cannot be made (no questions, one-sided or empty label set, or
    embeddings unavailable) — callers must report "unknown" rather than substitute a number.
    """
    from src import embeddings

    questions = [q for q in questions if (q or "").strip()]
    if not questions:
        return None

    good, bad, pool = _label_texts(examples or [], session, session_type, allow_pooled)
    # Both sides are required: with only accepted examples every candidate looks acceptable, and
    # with only rejected ones every candidate looks rejectable. Either way the number is meaningless.
    if not good or not bad:
        return None

    good_sim = embeddings.cosine_matrix(questions, good)
    bad_sim = embeddings.cosine_matrix(questions, bad)
    if good_sim is None or bad_sim is None:
        return None

    # Nearest neighbour on each side — a plain max, deliberately.
    #
    # The label sets are lopsided (68 rejections to 24 acceptances), so a max over the larger side
    # arguably wins on sample count alone. Averaging the top-3 on each side with equal k was tried to
    # correct for that and measured WORSE on held-out labels: accuracy 0.60 vs 0.65, and recall on
    # known-good questions fell from 1.00 to 0.89. With label sets this small a single close match is
    # the signal, and averaging dilutes it. Revisit if the label count grows past a few hundred.
    per_question: dict[str, bool] = {}
    repeats = 0
    for i, q in enumerate(questions):
        best_good = float(max(good_sim[i]))
        best_bad = float(max(bad_sim[i]))
        if best_bad >= NEAR_DUPLICATE_REJECT:
            repeats += 1
            per_question[q] = False
            continue
        per_question[q] = not (best_bad - best_good > DECISION_MARGIN)

    accepted = sum(1 for ok in per_question.values() if ok)
    return AgreementReport(
        predicted_accept_rate=round(accepted / len(questions), 3),
        n_scored=len(questions),
        n_good_labels=len(good),
        n_bad_labels=len(bad),
        repeats_rejected=repeats,
        per_question=per_question,
        label_pool=pool,
    )
