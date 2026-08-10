"""One interview topic must not supply half the set.

A review of No-Code AI Automation found hallucination asked **six times in 38 questions**. The framing that
made it tractable was not "these are duplicates" — it is that the set was unbalanced against the syllabus:

    Prompt engineering for multi-step content generation ....  7 questions
    Hallucination detection and mitigation strategies .......  6
    Improving LLM reliability through prompt engineering ....  5
    API integration for workflow automation ................  4
    ... and 8 of the 22 interview topics had NONE

Three topics held 47% of the set. `coverage_efficiency` already measures exactly this — "did each question
earn its place against a DISTINCT topic" — but only inside a single run's selected set. Nothing applied it
to the ACCUMULATED per-topic set, and `tools.tool_submit_question_set` runs `_same_thing_pass` BEFORE
`_add_retained`, so the retained set is never judged against itself or against the run's new picks.

WHY NOT JUST DEDUPLICATE
------------------------
Because no similarity threshold can see this class. Across the six hallucination questions ZERO pairs reach
the 0.82 `DEDUP_SEMANTIC_THRESHOLD`, and the two most obviously identical — *"What are hallucinations in
LLMs"* / *"What is an AI hallucination"* — score **0.486**, the LOWEST of all fifteen pairs. The judged
`_same_thing_pass` is accurate but its floor is 0.62, so it never sees that pair either. Grouping by
OUTCOME is what actually finds the pile, because "same outcome" is the property that matters in an
interview: a candidate should not be asked twice about one thing the session teaches.

THE RULES, EACH LOAD-BEARING
----------------------------
1. **Assign** each question to its best-matching interview topic (local embeddings, free). The outcome
   grouping is what makes pairing tractable AND meaningful — it is the axis a candidate actually
   experiences as repetition.
2. **An ORPHAN is KEPT, never cut.** A best match below `OUTCOME_ORPHAN_FLOOR` means no outcome describes
   the question. *"What is the Split In Batches node used for?"* matches at **0.173** and *"What are nodes
   in N8N"* at **0.132** — genuine n8n questions the outcome list fails to describe (the n8n gap in a new
   place). Counting them as redundant would turn this from a fix into silent data loss.
3. **Rank within an outcome: reviewer-`approved` first, then `session_fit` descending.** A lower-fit
   approved question deliberately beats a higher-fit backfilled one — the reviewer's decision is newer
   information than a similarity score, the same convention `filter_topic_sets._clusters` uses.
4. **Judge EVERY within-outcome pair, in SMALL BATCHES, and drop only what the judge calls redundant.**
   Both halves of that sentence were learned the hard way; see `JUDGE_BATCH` and the note below.

A HARD QUOTA WAS THE FIRST DESIGN, AND IT WAS WRONG
---------------------------------------------------
The first version kept at most `cap` questions per outcome and dropped the rest. It fixed the
hallucination pile and, on Gen AI Foundations, deleted **12 of 14** questions under one coarse topic —
including *"What is the difference between pre-training and fine-tuning?"*, *"What are the different
Fine-tuning methods?"* and *"What is the difference between a base model and an instruction-tuned
model?"*. Those are distinct asks that merely share a coarse outcome.

The tell was that the SAME quota was right on No-Code (22 topics for 38 questions) and destructive on Gen
AI Foundations (15 topics for 54). It was measuring how finely the curriculum happens to enumerate
`interview_topics`, not whether questions repeat. Same overlapping-distributions trap as
`_outcome_coverage`'s proximity threshold and `DEDUP_SEMANTIC_THRESHOLD`: no count separates the two
populations, so a judgement has to.

**And the duplicates the quota appeared to be needed for were a batching artefact** — see `JUDGE_BATCH`.
With the judge asked 10 pairs at a time instead of 52, judgement alone gives hallucination 6 -> 2 (what
the quota achieved) while keeping the fine-tuning distinctions (what it destroyed). `strict=True` keeps
the quota available; it is not the default and should stay that way.

An outcome with NO questions is reported, never padded — the same discipline as
`pipeline._ensure_session_representation`'s `no_candidates`.

FAIL-OPEN, AND PURE ON PURPOSE
------------------------------
`judge` is injected rather than imported so this module needs no LLM and no `pipeline`/`tools` import —
it is unit-testable with a stub and cannot create an import cycle. A judge that raises, returns nothing, or
returns indices we never asked about degrades to KEEPING EVERYTHING: nothing is dropped without a verdict. Only a pair we actually asked about, by its own index, can be acted on — the same verify-in-code
discipline as `_scope_trim`'s `_accept_trim` and `_syllabus_audit`'s `_concept_is_absent`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BalanceResult:
    """`keep`/`drop` are indices into the input list, so the caller decides what removal means.

    That split is why one function serves both call sites: the pipeline caps only what it ships (leaving
    `topic_question_set` whole, so the cap is idempotent and reversible), while
    `scripts/filter_topic_sets.py --apply` quarantines the dropped rows in the database.
    """
    keep: list[int] = field(default_factory=list)
    drop: list[int] = field(default_factory=list)
    orphans: list[int] = field(default_factory=list)
    # outcome index -> indices assigned to it (orphans excluded)
    assigned: dict[int, list[int]] = field(default_factory=dict)
    uncovered: list[int] = field(default_factory=list)
    pairs_judged: int = 0
    same_verdicts: int = 0
    judge_failed: bool = False

    @property
    def dropped_for(self) -> dict[int, int]:
        """dropped index -> the outcome it was capped against. For a truthful removal reason."""
        return {i: t for t, members in self.assigned.items() for i in members if i in set(self.drop)}


def _assign(texts: list[str], outcomes: list[str], orphan_floor: float):
    """Best-matching outcome per question; below the floor it is an orphan.

    Returns (assigned, orphans). Embeddings unavailable => everything is an orphan, i.e. nothing is
    capped. That is the right fail-open: without a grounding signal we cannot say what is redundant.
    """
    from src import embeddings

    sim = embeddings.cosine_matrix(texts, outcomes) if (texts and outcomes) else None
    if sim is None:
        return {}, list(range(len(texts)))
    assigned: dict[int, list[int]] = {}
    orphans: list[int] = []
    for i in range(len(texts)):
        best = max(range(len(outcomes)), key=lambda t: float(sim[i][t]))
        if float(sim[i][best]) < orphan_floor:
            orphans.append(i)
        else:
            assigned.setdefault(best, []).append(i)
    return assigned, orphans


def balance_by_outcome(texts: list[str], outcomes: list[str], *, fits: list[float] | None = None,
                       approved: list[bool] | None = None, cap: int | None = None,
                       orphan_floor: float | None = None, min_keep: int | None = None,
                       judge=None, strict: bool = False) -> BalanceResult:
    """Remove questions an outcome already covers, letting `judge` decide what "already" means.

    `judge(pairs)` receives `[(a, b), ...]` index pairs and returns the subset it considers the same ask
    (any iterable of pairs, order-insensitive). It is only ever asked about within-outcome pairs.

    `strict` PICKS THE CUT CRITERION, and the default is the conservative one — measured, not assumed:

    * `strict=False` (default): a question is dropped only when the judge says it tests the same thing as
      a question already kept for that outcome. `cap` then only decides which outcomes are crowded enough
      to examine.
    * `strict=True`: a hard quota — anything past `cap` goes, judged or not.

    **Why the quota is not the default.** It behaves completely differently depending on how FINE the
    session's `interview_topics` are, which is a property of the curriculum, not of the questions. On
    No-Code AI Automation (22 topics, 38 questions) a quota of 2 is about right. On Gen AI Foundations
    (15 topics, 54 questions) one coarse topic — "Pre-trained vs fine-tuned models and when to use each" —
    absorbs **14 questions**, and the quota deletes 12 of them including *"What is the difference between
    pre-training and fine-tuning?"*, *"What are the different Fine-tuning methods?"* and *"What is the
    difference between a base model and an instruction-tuned model?"*. Those are distinct asks that happen
    to share a coarse outcome, and losing them is not what balancing was for. Same overlapping-
    distributions trap as `_outcome_coverage`'s proximity threshold and the dedup bar: the number cannot
    separate the two populations, so a judgement has to.

    `min_keep` is a floor on the whole result (`MIN_QUESTIONS`): when trimming would take a set below it,
    the highest-ranked dropped questions are restored. A balanced set too small to run is worse than a
    slightly repetitive one, and `remove_question` already refuses at this floor for the same reason.
    """
    from src.config import MIN_QUESTIONS, OUTCOME_CAP, OUTCOME_ORPHAN_FLOOR

    cap = OUTCOME_CAP if cap is None else cap
    orphan_floor = OUTCOME_ORPHAN_FLOOR if orphan_floor is None else orphan_floor
    min_keep = MIN_QUESTIONS if min_keep is None else min_keep
    n = len(texts)
    fits = list(fits or [0.0] * n)
    approved = list(approved or [False] * n)
    if len(fits) != n or len(approved) != n:
        raise ValueError("fits and approved must be the same length as texts")

    res = BalanceResult()
    if n == 0 or not outcomes or cap < 1:
        res.keep = list(range(n))
        return res

    res.assigned, res.orphans = _assign(texts, outcomes, orphan_floor)
    res.uncovered = [t for t in range(len(outcomes)) if t not in res.assigned]

    # approved first, then higher fit, then original order for a stable result
    rank = lambda i: (not approved[i], -fits[i], i)  # noqa: E731
    ordered = {t: sorted(members, key=rank) for t, members in res.assigned.items()}

    # EVERY outcome with >=2 members — restricting this to over-served outcomes is the bug described
    # in the module docstring.
    #
    # ONE judge call PER OUTCOME, not one flat call for the whole set. A verdict depends on which OTHER
    # pairs share its batch, which was found the hard way: the same 28 pairs of "Pre-trained vs
    # fine-tuned models" yielded 3 "same" verdicts when batched alongside other outcomes' pairs and
    # **0 across 3 trials** when batched alone. Flat batching therefore made the whole pass
    # non-idempotent — a second `--apply` cut 8 more questions, including
    # *"What is the difference between a base model and an instruction-tuned model?"*, which the first
    # pass had deliberately kept. Judging one outcome at a time makes a verdict a function of that
    # outcome's own members and nothing else.
    same: set[tuple[int, int]] = set()
    total_pairs = 0
    for t, members in ordered.items():
        group = [(a, b) for x, a in enumerate(members) for b in members[x + 1:]]
        if not group or judge is None:
            continue
        total_pairs += len(group)
        asked = {(min(a, b), max(a, b)) for a, b in group}
        try:
            for a, b in (judge(group) or []):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                if key in asked:                      # verified: only a pair we actually asked about
                    same.add(key)
        except Exception as exc:  # noqa: BLE001 — a dead judge must not lose the set
            # Per-outcome fail-open: one bad outcome keeps its questions, the rest are still balanced.
            print(f"[outcome_balance] outcome {t} skipped ({type(exc).__name__}: {exc})")
            res.judge_failed = True
    res.pairs_judged = total_pairs
    res.same_verdicts = len(same)
    is_same = lambda a, b: (min(a, b), max(a, b)) in same  # noqa: E731

    keep: list[int] = list(res.orphans)
    dropped: list[int] = []
    for t, members in ordered.items():
        sel: list[int] = []
        for i in members:
            if any(is_same(i, k) for k in sel):
                dropped.append(i)                     # judged redundant against something kept
            elif strict and len(sel) >= cap:
                dropped.append(i)                     # hard quota — opt-in, see the docstring
            else:
                sel.append(i)
        keep += sel

    # Floor: restore the best-ranked drops rather than ship an unrunnable set.
    if len(keep) < min_keep and dropped:
        for i in sorted(dropped, key=rank)[:min_keep - len(keep)]:
            keep.append(i)
        dropped = [i for i in dropped if i not in set(keep)]

    res.keep = sorted(keep)
    res.drop = sorted(dropped)
    return res


JUDGE_BATCH = 10
"""Pairs per LLM call. Small ON PURPOSE, and the size is measured rather than chosen.

The judge's accuracy collapses with batch size, which is the single most consequential thing found while
building this. On *"What are hallucinations in LLMs"* / *"What is an AI hallucination"* — the pair a
reviewer flagged by hand and the most obviously identical of the six:

    batch of 1   -> SAME, 3 trials out of 3
    batch of 52  -> "different"

And in a 4-pair batch it correctly called the fine-tuning pairs DISTINCT while calling
*"How do you prevent hallucinations?"* / *"What strategies do you employ to mitigate hallucinations"* the
same. So small batches are right in BOTH directions; one large batch was wrong in both.

This is why the balance rule does not need a hard quota. The first version blamed the CAP for leaving
duplicates and would have "fixed" it by deleting everything past two per outcome — which on Gen AI
Foundations destroyed 12 distinct questions under one coarse topic. The cap was never the problem.

Cost: 52 pairs is 6 calls instead of 1, on Haiku. Cheap, and the alternative is verdicts that are wrong.
"""


def majority(judge, trials: int = 3):
    """Wrap a judge so a pair counts as redundant only if MOST trials say so.

    For the DESTRUCTIVE path (`filter_topic_sets.py --cap --apply`), where a wrong verdict removes a real
    question from the database. The pipeline path deliberately does NOT use this: it only trims what a run
    ships, re-derives the result every run, and paying 3x on every run to stabilise a reversible decision
    is not worth it.

    Why it is needed at all: verdicts are not perfectly reproducible, and a single flap is permanent once
    applied. A second `--apply` removed 8 further questions — one of them a question the first pass had
    explicitly kept as distinct. Requiring 2 of 3 makes an erosion like that need two independent
    misjudgements instead of one.
    """
    from collections import Counter

    def voted(pairs):
        votes: Counter = Counter()
        for _ in range(trials):
            for a, b in (judge(pairs) or []):
                votes[(min(a, b), max(a, b))] += 1
        need = trials // 2 + 1
        return [p for p, n in votes.items() if n >= need]

    return voted


def make_llm_judge(texts: list[str], *, model: str, complete=None, on_usage=None,
                   max_pairs: int = 400, batch: int = JUDGE_BATCH):
    """A `judge` backed by one JSON completion, reusing `tools._SAME_THING_SYSTEM`.

    Deliberately NOT the default: callers pass this in, so tests never reach the network.

    **`complete` must be the caller's own `chat_completion_json` reference, and that is not a style
    preference.** Every LLM pass in `tools.py` goes through that module's module-level symbol, which is
    what the whole test suite stubs (`monkeypatch.setattr(tools, "chat_completion_json", …)`). The first
    version imported it from `src.llm_client` inside this function, so the stub missed it and
    `tests/netguard.py` recorded **3 real network connections per test** — swallowed by the fail-open
    handler, so the assertions passed while a working key would have spent credit. That is exactly the
    leak class the guard was built for.

    `max_pairs` is a runaway guard on the prompt, not a sampling budget — conflating those is what left
    `_same_thing_pass` judging 12 of 41 eligible pairs on a 38-question set while reporting "0 redundant".
    Pairs beyond it are reported as unjudged rather than silently treated as distinct.
    """
    import json

    def judge(pairs):
        from src.tools import _SAME_THING_SYSTEM

        chat = complete
        if chat is None:                      # explicit fallback, never the path a caller in-tree takes
            from src.llm_client import chat_completion_json as chat

        use = pairs[:max_pairs]
        if len(pairs) > max_pairs:
            print(f"[outcome_balance] {len(pairs) - max_pairs} pair(s) beyond the {max_pairs} guard "
                  f"were NOT judged")
        out = []
        failures = 0
        for start in range(0, len(use), batch):
            chunk = use[start:start + batch]
            numbered = [{"n": k + 1, "a": texts[a][:400], "b": texts[b][:400]}
                        for k, (a, b) in enumerate(chunk)]
            try:
                result = chat(
                    model=model,
                    system_prompt=_SAME_THING_SYSTEM,
                    user_prompt=f"PAIRS:\n{json.dumps(numbered)}",
                    max_tokens=1024,
                    on_usage=on_usage,
                )
            except Exception as exc:  # noqa: BLE001
                # Per-batch fail-open. One bad chunk must not discard every verdict already collected —
                # raising here would surface as `judge_failed` and silently disable the whole pass.
                failures += 1
                print(f"[outcome_balance] batch {start // batch + 1} failed "
                      f"({type(exc).__name__}: {exc})")
                continue
            for row in (result.get("pairs") or []):
                try:
                    k = int(row["n"]) - 1
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= k < len(chunk) and bool(row.get("same")):
                    out.append(chunk[k])
        if failures:
            print(f"[outcome_balance] {failures} batch(es) failed — those pairs were NOT judged, so "
                  f"their questions are KEPT")
        return out

    return judge
