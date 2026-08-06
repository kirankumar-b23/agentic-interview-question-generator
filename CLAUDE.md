# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Interview Question Generator — a multi-agent pipeline that curates **real** interview questions for course sessions. It reads each session's reading material, resolves it to learning outcomes + Knowledge Points, retrieves real questions from a pre-indexed bank, GitHub interview repos, and Tavily web search, then validates, deduplicates, and assembles a question set. A human reviews before Google Sheets export.

**Architecture**: A 4-agent pipeline (Understanding → Retrieval → Validation → Evaluation) followed by an LLM quality gate, driven via OpenRouter tool_use. A React SPA (served by FastAPI/uvicorn) shows live progress and a per-question review screen. **Question generation is disabled — only real, sourced questions are used.**

## How It Works

```
User selects a topic → all its sessions/units are passed in →
  Understanding agent: understand_session (per-session reading material → outcomes, KPs) →
  Retrieval agent: search_question_bank (interview data first) → search_web_questions
    (Tavily, real company-attributed) → search_github_questions (supplemental) →
  Validation agent: validate_relevance (lenient) + deduplicate_questions →
  Evaluation agent: difficulty/coverage checks → generate_expected_answers →
    (coding Qs only if code-heavy) → submit_question_set →
  Quality gate (LLM critique; up to 2 revision rounds) →
Human reviews (React UI, per-question accept/reject) →
  Approve → Google Sheets | Reject → distil feedback into learned rules + re-generate
```

## Project Structure

```
main.py                             # FastAPI: JSON API + serves React SPA (frontend/dist). Run: uvicorn main:app --port 5000
frontend/                           # React SPA (Vite). Pages: SessionSelector, AddCourse, Progress, Review, History
  src/components/Icon.jsx           # Inline SVG icon set (replaces emoji) + step/agent icon maps
  src/components/AgentTranscript.jsx# Grouped, timed, collapsible trace of what the agents did
  src/components/QualityPanel.jsx   # Verdict + critique notes; scored vs reported-only metrics
  src/components/TopBar.jsx         # The one page header (was copy-pasted per page)
  src/components/ErrorBoundary.jsx  # Stops a bad payload white-screening the app
  src/components/Sidebar.jsx        # Course/topic/model/count controls + credit balance
  src/components/PipelineStepper.jsx# Live pipeline stage indicator
  src/pages/Review.jsx              # Per-question review, keyboard-first triage
  src/pages/Progress.jsx            # Live agent transcript
  src/pages/AddCourse.jsx           # Add/import a course (new sessions/topics)
scripts/
  prepare_data.py                   # One-time: CSV→JSON, build knowledge graph, eval sets
  build_session_reading_material.py # Build data/reading_materials/session_map.json (per-session content)
  build_knowledge_graph.py          # Build data/knowledge_graph.json (KPs + prerequisite edges)
  build_eval_sets.py                # Build eval/eval_sets.json (good/bad validation examples)
  build_genai_bank.py               # Harvest the GenAI bank (Tavily); applies the FORM gate before writing
  clean_bank.py                     # Sweep an existing bank: --dry-run to report, --show N for reject samples
  ingest_xlsx_questions.py          # Ingest real questions from data/raw/*.xlsx
  audit_outcomes.py                 # Seed/edit data/reading_materials/session_outcomes.json (curated outcomes)
  auth_sheets.py                    # One-time Google Sheets OAuth → token.json
data/
  interview_questions.json          # 1,499 company-attributed questions — general SWE/Python, little GenAI
  genai_question_bank.json          # 1,400 curated GenAI questions (build_genai_bank.py; swept by clean_bank.py)
  knowledge_graph.json              # KPs + sessions + prerequisite edges
  course_structure.json             # Topic → list of sessions (units); drives UI selection
  reading_materials/session_map.json# Canonical session name → that session's reading material
  reading_materials/*.md            # Source course reading materials (gen_ai, llm_applications)
  curriculum/*.json                 # BUILD-TIME KP source for build_knowledge_graph.py ONLY.
                                    # Course MCQs, not interview questions — never loaded at runtime.
src/
  pipeline.py                       # AgentPipeline.run — orchestrates the 4 agents + quality gate
  agents/                           # UnderstandingAgent, RetrievalAgent, ValidationAgent, EvaluationAgent
  agent.py                          # AgentState, PipelineResult, _critique_question_set (quality gate)
  tools.py                          # Tool schemas + implementations (generation tool is blocked)
  session_understanding.py          # Per-session resolution + merge → SessionContext (RM-first, KG fallback)
  question_bank.py                  # Hybrid (embedding + TF-IDF) retriever; corpus vectors disk-cached in .cache/
  embeddings.py                     # Local sentence-transformers (MiniLM); degrades to TF-IDF if absent
  quality.py                        # FORM gate — is this a well-formed standalone question?
  human_agreement.py                # Predicted reviewer acceptance from past accept/reject decisions
  rejection_rules.py                # Rejection-reason key → canonical learned rule (matches Review.jsx)
  orchestrator.py                   # Per-run SSE fan-out WITH replay history + heartbeats
  sources/tavily_search.py          # Tavily web search + URL-based company extraction
  sources/github_repo.py            # GitHub interview-repo fetch (REST API)
  llm_client.py                     # OpenRouter client (chat + JSON extraction)
  models.py                         # Pydantic models; attribution (who asked) vs source_site (where found)
  data_loader.py                    # Loads prepared JSON + per-session reading material (exact lookup)
  memory.py                         # SQLite: session cache, run history, RLHF learned rules
  config.py                         # Model selection, paths, constraints, Tavily/GitHub source lists
  sheets_writer.py                  # Google Sheets export (OAuth)
eval/
  eval_sets.json                    # 46 eval sessions + format rules
  feedback_examples.json            # Reviewer accept/reject decisions (runtime data; gitignored)
  run_eval.py                       # Scored harness. Primary metric = predicted reviewer acceptance
tests/                              # pytest — no LLM or network required
```

## Question Sources (real only — no generation)

| Source | Tool | Company attribution |
|--------|------|---------------------|
| Pre-indexed bank (1,499 Qs) | `search_question_bank` (hybrid embedding+TF-IDF) | Yes (verified) |
| Tavily web (69-domain allowlist) | `search_web_questions` | Best-effort from URL |
| GitHub interview repos | `search_github_questions` | No |
| `generate_interview_questions` | **BLOCKED** | — |

Attribution for output is computed by `attribution_label` (`src/models.py`) and surfaced via `QuestionDetail.attribution`: the **real company in UPPERCASE** when known, otherwise the placeholder **`NIAT`** — never a website name, fragment, or fabricated company. Garbage candidates are filtered upstream by Tavily's `_valid_company`.

**`attribution` (who asked) and `source_site` (where it was found) are two different fields, deliberately.** `attribution` once fell back to the source site, and a live run shipped questions labelled "Analytics Vidhya", "Indeed", "Edureka" and "DataCamp" in the same field, tag and column as a genuine "ANTHROPIC" — one of those asked a candidate a question and the rest are content sites. Provenance now lives on `QuestionDetail.source_site` and renders as a separate, quieter `via <site>` tag in Review. The Sheets export reads `asked_in_company` only and is unaffected by either.

## Reading Material → Relevance

`data/reading_materials/session_map.json` (built by `scripts/build_session_reading_material.py`) maps each canonical session name to ONLY that session's reading material. `data_loader.get_session_content` does exact + normalized lookup (no fuzzy substring fallback). `understand_session` resolves each selected session from its own content and merges, so multi-session topics get accurate, on-topic outcomes → sharper retrieval queries.

## Tech Stack

- **LLM**: Claude Haiku 4.5 (dev) / Sonnet (prod) via OpenRouter (`openai` SDK with tool_use)
- **Search**: scikit-learn TF-IDF (bank) + Tavily API (web) + GitHub REST API
- **Knowledge graph**: networkx DAG for prerequisite ordering
- **Data models**: Pydantic v2 (field validators + computed fields)
- **Cache/History**: SQLite (session resolutions, run history, RLHF learned rules)
- **Web**: FastAPI JSON API (uvicorn, Pydantic request models) + React SPA (Vite, react-router); SSE for live progress. Errors use `{"error": ...}`, not FastAPI's default `{"detail": ...}`, because the React client reads `body.error`.
- **Output**: gspread + google-auth-oauthlib (Google Sheets via OAuth)
- **Data prep**: pandas (CSV)

## Key Constraints

- Minimum 5 questions; the final set is **NOT** capped (`MAX_QUESTIONS=60` bounds only the UI slider
  and coding-slot maths; `FINAL_SET_CAP=200` is a safety guard). Every outcome-relevant question is
  kept, and the review UI ranks by `session_fit` and tiers the list so a large set stays reviewable.
- Max tool calls bounded per agent (cost control)
- Question generation is DISABLED — only real, sourced questions
- No live scraping (offline scraping removed entirely)
- Retrieval order: interview bank first, then Tavily web. GitHub is **off by default**
  (`GITHUB_ENABLED=0` — the repos are general ML/DS with no company attribution)
- Coding questions only for code-heavy sessions
- Human approval required before Google Sheets write
- Reject → distils feedback into learned rules + re-generates with same sessions
- Session understanding uses the per-session reading material first, knowledge graph as fallback
- Quality gate critiques the final set; agent revises up to 2 rounds

## Scoring & evaluation (read before touching metrics)

The composite score is built **only** from signals independent of the selector:
`outcome_coverage`, `session_grounding`, `predicted_accept`, `set_size`.

`self_relevance` (the mean LLM relevance score used to *pick* the questions) and
`difficulty_balance` (scored against GenAI-bank labels that are ~95% "Medium") are **reported but not
scored**. They were 50% of the old composite, which is why runs scored 0.9 while reviewers rejected
most of the set — measured `corr(composite, approved) = 0.16` across 36 real runs.

Two retrieval/grounding layers matter and are easy to confuse:
- `question_bank.py` ranks with a **hybrid** score (0.6·embedding + 0.4·TF-IDF). Pure TF-IDF returned
  RAG questions for an F5-TTS query; pure embeddings miss exact tool names ("n8n", "LoRA").
- `pipeline._score_session_fit` scores every candidate against **this session's own** outcomes +
  reading material. The older `_prefilter_semantic` compares pooled *course-topic* profiles and so
  never filtered anything *within* a topic.

`src/human_agreement.py` estimates reviewer acceptance from `eval/feedback_examples.json`
(1-NN over embeddings). It returns **None** when it cannot measure — callers must report "unknown",
never substitute 0.0. `eval/run_eval.py` scores against **held-out** labels so the metric stays
independent as the pipeline learns from the same feedback.

## Failure paths (read before touching the pipeline)

The happy path is not the risky part. These are load-bearing:

- **Selection is enforced, not requested.** Ranking and the trim live in `tool_submit_question_set`,
  and the Evaluation agent is only *prompt-advised* to call it. `pipeline._enforce_submission` calls
  it directly when the agent didn't, and says so in the report. Without that, a text-only reply or an
  API error ships the raw candidate pool (~270 unranked questions) as the final set.
- **`chat_completion_json` raises `JSONResponseError`** rather than returning `{}`. The quality gate
  reads its verdict with `pass=False` as the default, so an unparseable critique fails closed.
- **A total relevance-judge failure sets `state.relevance_scored = False`**, which fails the gate and
  the report. Unscored candidates default to the keep-threshold (right for one bad batch, unsafe as a
  silent whole-run default).
- **A dead phase is recorded in `state.phase_errors`** and named in the report. Agent tool-loops are
  retried on transient errors; a swallowed outage used to look like "this session has few questions".
- **Model comes from the run** (`llm_client.run_model(state)` ← `GenerationConfig.model`), for the
  agent tool-loops AND every `chat_completion_json` call (gate, relevance judge, session
  understanding). `get_active_model()` is a UI display default ONLY. Passing no `model=` to
  `chat_completion_json` silently falls back to that global — which is how the two most expensive
  stages kept riding it after the first fix.
- **`main.py` imports `llm_client` at MODULE level.** Function-local imports there once left
  `set_active_model` unbound in `api_generate`, so every POST /api/generate was a 500 while the test
  suite stayed green (every test posted a body that failed validation before reaching the handler).
  Endpoint tests must post VALID bodies.
- **Event `seq` is a monotonic counter, not `len(history)`.** Using the length meant every event past
  `MAX_HISTORY_EVENTS` shared a seq; both consumers dedupe on it, so a long run stopped delivering
  events and never emitted `complete`.
- **`/api/result` returns 409 while a run is in flight.** Do not reintroduce a blocking `thread.join`:
  Starlette's threadpool is bounded and a few polling tabs starved every other endpoint.
- **SSE events are retained per run** (`orchestrator.get_history`) so a reload replays the transcript,
  and a quiet stream heartbeats. A 120s silence between events is normal for `validate_relevance`.
- **`_Q_STARTS` is matched on a WORD BOUNDARY** (`sources/base.py:_Q_START_RE`), not with
  `startswith`. The bare prefix made every gerund an imperative — `"design"` matched `"Designing"` —
  so job-description bullets, book titles and mid-sentence prose passed the form gate. A live run
  shipped *"Designing or writing prompts to support specific AI outcomes"* (an indeed.com/hire bullet)
  as a question, and the fix removed 27 such rows from the GenAI bank. A question ending in `?` is
  accepted before the opener is consulted, so real questions that open with a gerund are unaffected.
- **A tool's progress label must read the tool's result flags.** `remove_question` legitimately
  refuses at `min_questions` with an empty reserve (`{"removed": False, "warning": …}`), and the label
  printed `"Removed — N left"` regardless, so a run showed two successful removals while the
  gate-flagged duplicate stayed in the shipped set. Same class as the silent-failure paths above.
- **`check_difficulty_balance` reports `achievable`, not just `balanced`.** "Unbalanced" alone told
  the agent to fix something it had no material to fix with: a code_heavy run returned `E:0 M:4 H:1`
  against a 20/50/30 target on all three revision rounds with zero Easy candidates in the pool, and
  burned both revisions. Feasibility is judged against the same `DIFFICULTY_TOLERANCE` band that
  defines `balanced` — an exact-share check reports satisfiable mixes as impossible.
- **`kp_label` is assigned at selection** (`tools._assign_kp_labels`, called from
  `tool_submit_question_set`). It is a local embedding match against `SessionContext.matched_kp_ids`,
  not an LLM call, and it tags only the SELECTED set. Before this it was written by nothing, so every
  exported question carried `kp_label = None`.

## On-domain is not on-syllabus (read before touching coverage or attribution)

Found by evaluating run `b5d94fee` ("Introduction to AI Agents + Building a Learning Path Generator",
12 questions) against the two sessions' reading material. The gate said *"All 12 theory questions are
on-domain … no off-domain items detected"* **and** *"No questions represent: Building a Learning Path
Generator."* Both were true. Only 5 of 12 tested something either session actually teaches.

- **`_attribute_sessions` scores CHUNKS per session, never one blob.** It used to compare each question
  against the session's reading material truncated to 4,000 chars — so the second session lost 55% of
  its material and every specific match diluted to 0.17–0.41. Result: all 12 questions labelled with
  the first session. Reproduced: blob → **12/0**, chunk-and-max → **6/6**. It now reuses
  `pipeline._session_profile`, so attribution and the session-fit gate share one grounding.
- **`_ensure_session_representation` makes per-session representation a guarantee.** `_select_final`
  only *nudges* with a bonus, and that nudge is dead whenever `sess_target` sees one distinct session —
  which is exactly what the attribution bug caused. A session with candidates now gets a slot by
  displacing the weakest question of an over-represented session; a session with **no** candidates is
  reported (`no_candidates`) and never padded.
- **`_syllabus_audit` judges questions against the reading material itself.** Nothing else does: the
  relevance judge sees `scope_in`/`scope_out` summaries, never the material. It sets
  `QuestionDetail.off_syllabus_concept` (flag, don't reject — the reviewer decides) and returns
  LLM-judged outcome coverage. Its reply is verified in code: a claimed untaught concept must genuinely
  be absent (`_concept_is_absent`), and outcome indices must be in range.
- **`_concept_is_absent` filters ubiquitous domain words before judging.** Requiring every word to be
  missing marked "ambiguous user intent" as taught, because "user" appears throughout any agent lesson.
  A majority rule instead lets "agent guardrails" slip (1 of 2 missing). Judge only the distinctive
  words; a concept made entirely of ubiquitous words yields no claim.
- **`_outcome_coverage` returns `(fraction, method)` and the report always names the method.**
  Proximity was being scored as coverage at **35% of the composite**: a hallucination question was
  credited with *"Integrate multiple Google APIs (Docs, Calendar, Drive)"* (0.38), and *"when does agent
  reasoning go off the rails?"* with *"Design system prompts that orchestrate agent reasoning"* (0.69).
  **Raising the threshold cannot fix this** — legitimate credits measured 0.52–0.83 and false ones
  0.38–0.69, overlapping, and shared-distinctive-term counts overlap too (0–2 vs 0–1). Judgement
  against the material is the only separator; the embedding measure remains the fail-open fallback and
  is labelled `embedding-proximity` so nobody reads it as verified.
- **`tool_check_outcome_coverage` prefers the same judged coverage as the report**, so the agent and the
  report cannot disagree. It is only available from the second round on (the audit runs inside submit),
  so the measure in use is always stated in the tool result.
- **Difficulty can be earned off-syllabus.** In that run all four Hard questions were the ungrounded
  ones and all three Easy questions were taught — `difficulty_balance` scored 0.87 *because* the set
  went beyond the material. Read `difficulty_balance` alongside `off_syllabus`, never alone.

## Assessment items must never enter the retrieval corpus

- **`data/curriculum/*.json` are course MCQs, not interview questions** (1,610 of 1,822 are
  multiple-choice) and are **build-time inputs only** — `scripts/build_knowledge_graph.py` reads them by
  literal path. `data_loader` used to parse them into `self.curriculum_questions` and print
  *"Loaded 1819 curriculum questions into bank"* right above *"Question bank ready: 1509 questions
  indexed"*, which reads as though they were indexed. **They never were** — nothing read that list. The
  loader, the list and the `GEN_AI_JSON`/`LLM_APPS_JSON`/`FLASK_JSON` constants are all gone; an unused
  path is what invited the confusion. Do not re-add them, and do not delete the files.
- **The form gate does NOT stop MCQs**: 61% of those 1,772 rows pass `is_quality_question`
  (*"Which of the following images represents the node used to send HTTP requests…"*). So the guard is a
  data assertion, not a gate — `tests/test_data_integrity.py::TestNoAssessmentItemsInTheRetrievalCorpus`
  checks the shipped corpora for MCQ shapes (lettered options, "match the following", `___` blanks).
- That test immediately found **10 genuine MCQs already in `interview_questions.json`** — the main
  retrieval bank — which the dead-loader story had nothing to do with: aptitude/logical-reasoning items
  and C/Java syntax MCQs, none company-attributed. Removed (1,509 → **1,499**, `.bak` kept). Clear
  `.cache/` after any bank edit; the corpus matrix is cached by digest.
- The **7 rows tagged `source: "curriculum"` in the GenAI bank are NOT MCQs** — they are real
  company-attributed questions (BluePond.AI, Blackcoat AI, Medoc Health). The `source` value is a
  misnomer from the bank build. Leave them.

## Two different "slicing" mechanisms — do not merge them

- **`_trim_to_topic` + `split_into_clauses`** cut a COMPOUND question at boundaries between separate
  asks (sentence ends, `and then`/`as well as`/`;`, and a bare `and` only when a question stem
  follows). With a single clause it is **keep-or-drop and never rewrites**. That conservatism is
  deliberate: it is what stops `Compare precision and recall` being torn in half.
- **`tools._scope_trim`** removes an off-syllabus clause from a single on-topic ask
  (`…improve prompts and guards` → `…improve prompts`). It runs **after the relevance gate**, from
  `tool_submit_question_set`, on the selected set only — one LLM call per run.
- **Never move the scope trim to retrieval time.** Measured across the 1400-row bank, a
  trim-ungrounded-conjuncts rule fires on **155 rows** and destroys the comparison class
  (`difference between supervised and unsupervised learning?` → cuts `unsupervised`;
  `trade-offs between RAG and fine-tuning?` → cuts `fine-tuning`). On the final selected set the same
  rule fires on 1 of 5. An off-topic question must be REJECTED by relevance, never edited into a
  different question.
- **The model's reply is verified in code** (`_accept_trim`), never trusted: contiguous word prefix,
  still passes `is_quality_question`, no comparison frame (`difference|trade-off|compare|versus`), no
  fixed idiom (`pros and cons`), not a shared-object verb pair (`detect and reduce them`), not an
  elided comparison (`_shares_head_noun`: `self-attention and multi-head attention` repeats
  "attention"), and ≥40% of words survive. Any violation keeps the original.
- `_TRIM_MIN_WORD_RATIO = 0.4` is measured, not chosen. A legitimate trim keeps 3 of 7 words (0.43);
  0.6 rejected the very case the pass exists for.
- A trimmed question **keeps its company attribution** and is marked `adapted`
  (`QuestionDetail.original_content` holds the verbatim source; Review shows an `adapted` tag with the
  original on hover; the quality report names the count). Sheets **Tab 1 has exactly 10 LMS columns** —
  do not add one for this.
- `scope_out` is LLM-derived per run and curated for **0 of 53** sessions, so the trim must not depend
  on it. It is passed to the trim prompt as a hint only; the run that raised this did not list
  guardrails in its 26-item `scope_out`.

## Web/UI notes

- `main.py` errors use `{"error": ...}`, not FastAPI's `{"detail": ...}` — the React client reads
  `body.error`. Blocking handlers stay sync so Starlette runs them in a threadpool.
- Emoji are not the icon system; `frontend/src/components/Icon.jsx` is. Progress renders
  `AgentTranscript` (grouped per agent, timed, collapsible) from structured SSE fields — not a flat log.
- `QualityPanel` separates SCORED metrics from reported-only ones and renders `report.critique`. Adding
  `self_relevance` back alongside the scored metrics undoes the point of the scoring work.
- Review is keyboard-first (`j/k`, `a`, `r`, `1-7`, `u`, `shift+A` accept-above-fit, `esc` clear
  filters, `?` help), with filters by fit band / difficulty / source / attribution. The cursor indexes
  the FILTERED list and resets when filters change.
- Bulk actions write explicit per-question decisions so `decisions_sent` stays truthful — an
  all-rejected or bulk-cleared set must still be refused at export, not read as "no filter requested".
- Every page uses `<TopBar>` (it renders the `<h1>`). Inline `<header className="topbar">` should stay
  at zero; there were seven after the component was first written.
- **There is a `button` reset (`index.css`), and it is load-bearing.** Without it, any button that did
  not declare its own `background`/`border`/`font`/`width` rendered as native UA chrome — `#efefef`
  fill, `2px outset` border, Arial 13.3px, shrink-wrapped. That was the whole reason the UI "looked
  ugly": measured on History, `.sidebar-run-item` was Arial on rgb(239,239,239) at **five different
  widths**, and `.sidebar-logo` kept a UA border on three sides because it set only `border-bottom`.
- **`.shell-main` is THE scroller; `.page-content` is a layout band.** Not every page has a
  `.page-content` (Review renders `.action-banner` / `.review-gutter` directly), so scoping the scroll
  to it left Review clipped inside a 100vh shell with nothing to scroll. Two traps to keep straight:
  `overflow-x: hidden` on an ancestor silently makes it a scroll container and **kills
  `position: sticky`** in the child (that is why the masthead and the table's column labels used to
  scroll away); and `overflow: hidden` on a flex child sets its automatic minimum size to **zero**, so
  a height-bound flex column crushes it and it clips its own content with no scrollbar — a 2626px
  table rendered into a 564px box, rows 9–38 unreachable, indistinguishable from "the content fits".
  Hence `.page-content > * { flex-shrink: 0 }`.
- The masthead's height is the token `--topbar-h`, and the sticky `<thead>` offsets by exactly it.
  An elastic masthead lets the two overlap.
- `.hist-table` is `table-layout: fixed` **on purpose**. With auto layout the table's minimum width
  came from the longest topic string, so clipping appeared at scattered viewport widths (640, 700,
  999, 1100) and each breakpoint that fixed one broke another. Verified clean 320→1920px.
- UI changes here are verified with Playwright + the system Chrome (`/usr/bin/google-chrome` — the
  bundled playwright build does not match the downloaded browsers). The harness asserts computed
  styles and reachability, not just appearance, because every failure above looked fine in a
  screenshot.

## Session type is load-bearing now

`session_type` (theory_heavy | code_heavy | mixed) used to be computed, stored, printed into two prompt
headers, and acted on nowhere. It now drives real behaviour, so keep these straight:

- **The authoritative value is `SessionContext.session_type`** — the LLM's read of the session's own
  reading material. There are three other, worse sources; do not use them for behaviour:
  the knowledge graph's **title-substring** guess (`build_knowledge_graph.py:infer_session_type` —
  `'mastering'`/`'kaggle'`/`'journey'` count as theory signals), `prepare_data.py`'s hardcodes, and the
  stale copy that used to live in `eval_sets.json`. They disagreed with the pipeline on 9 of 22 sessions.
- **`src/session_types.py`** resolves a type from a session NAME for the cases with no SessionContext
  (labelling a historical reviewer decision, an eval session with no reading material). It prefers the
  LLM-derived `session_outcomes.review.json` over the knowledge graph, and folds `" + "`-joined run
  names with the pipeline's own rule. `type_for_run` returns **None** for a wholly unknown name —
  "unknown" is not "mixed", and callers need that difference to know whether a score is measurable.
- **What varies by type:** `config.difficulty_targets()` (read at `tools.tool_check_difficulty_balance`
  and `tools._select_final`), `config.eval_thresholds()` (read by `eval/run_eval.py`), the relevance
  judge's `_TYPE_GUIDANCE` block, and which reviewer labels calibrate the judge
  (`tools._feedback_examples_block` → this session → this session TYPE → all, and it states which).
- **Do NOT route per-type behaviour through `SessionContext.difficulty_distribution` or
  `GenerationConfig.course_type`.** Both are hardcoded/written and read by nothing — changing them
  looks like a fix and does nothing. `course_type` is course-level anyway, so it would flatten a mixed
  course to one type.
- **`predict_accept(..., session_type=..., allow_pooled=False)`** returns `None` rather than scoring a
  code-heavy set against theory decisions. That substitution is not conservative: an implementation
  question matches the "too specific, not conceptual" pattern the reviewer established on theory
  material, so the pooled number looks measured and is wrong.

## Dormant: coding questions

Nothing assigns `state.coding_questions`. Generation is blocked and no retrieval path produces them,
so the CodingQuestion/CodeSnippet sheet tabs and every coding branch are unreachable. The tabs stay
because they're part of the LMS unit import format.

## Tests

`pytest tests/ -q` — 432 tests, no LLM or network required. Beyond unit coverage:
- `tests/test_pipeline_integration.py` drives the REAL pipeline with only the LLM boundary stubbed,
  including each outage above. This is the cheapest way to check a pipeline change.
- `tests/test_failure_paths.py` and `tests/test_audit_fixes.py` pin the silent-failure classes.
- `tests/test_data_integrity.py` checks the shipped `data/` files (missing reading material, bank
  form-garbage, implausible company attribution).
- A guard that `REJECT_REASONS` in `Review.jsx` matches `src/rejection_rules.py`.
- `tests/test_session_types.py` asserts the *observable* consequences of a session type (the selected
  difficulty mix actually differs, the same metrics pass one type's bars and fail another's), not that
  a constant exists — because the decorative version of this passed every structural check.
- `tests/test_audit_fixes.py::TestFixesThatSilentlyDidNotLand` asserts *observable behaviour* for
  fixes that were once reported as done but never applied (a no-op string replace fails silently, and
  a commit message is not evidence). Prefer that style over asserting a line of code exists.
