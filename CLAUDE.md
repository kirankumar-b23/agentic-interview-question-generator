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
  src/pages/Batch.jsx               # Multi-topic batch: one row per topic, links into each review
scripts/
  prepare_data.py                   # One-time: CSV→JSON, build knowledge graph, eval sets
  build_session_reading_material.py # Build data/reading_materials/session_map.json (per-session content)
  build_knowledge_graph.py          # Build data/knowledge_graph.json (KPs + prerequisite edges)
  build_eval_sets.py                # Build eval/eval_sets.json (good/bad validation examples)
  build_genai_bank.py               # Harvest the GenAI bank (Tavily); applies the FORM gate before writing
  clean_bank.py                     # Sweep an existing bank: --dry-run to report, --show N for reject samples
  strip_assessment_items.py         # Remove lettered-option MCQs + repair glued answers (both banks)
  ingest_xlsx_questions.py          # Ingest real questions from data/raw/*.xlsx
  audit_outcomes.py                 # Seed/edit data/reading_materials/session_outcomes.json (curated outcomes)
  auth_sheets.py                    # One-time Google Sheets OAuth → token.json
data/
  interview_questions.json          # 1,447 company-attributed questions — general SWE/Python, little GenAI
  genai_question_bank.json          # 1,381 curated GenAI questions (build_genai_bank.py; swept by clean_bank.py)
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
  interview_format.py               # Is it answerable OUT LOUD? (hands-on task prompts; CONVERSATIONAL_ONLY)
  assessment_items.py               # MCQ / glued-answer shapes — a data assertion, NOT a form gate
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
tests/                              # pytest — no network, ENFORCED by conftest.py + netguard.py
```

## Question Sources (real only — no generation)

| Source | Tool | Company attribution |
|--------|------|---------------------|
| Pre-indexed bank (1,447 Qs) | `search_question_bank` (hybrid embedding+TF-IDF) | Yes (verified) |
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
`coverage_efficiency`, `session_grounding`, `predicted_accept`, `set_size`.

`self_relevance` (the mean LLM relevance score used to *pick* the questions) and
`difficulty_balance` (scored against GenAI-bank labels that are ~95% "Medium") are **reported but not
scored**. They were 50% of the old composite, which is why runs scored 0.9 while reviewers rejected
most of the set — measured `corr(composite, approved) = 0.16` across 36 real runs.

### Coverage: what it measures against, and why there are two numbers

Three consecutive runs failed the gate and the sets were not the problem. Both causes are structural.

- **Coverage scores `interview_topics`, NOT `learning_outcomes`** (`pipeline.coverage_targets`).
  Outcomes describe the LESSON, setup steps included: across the 53 curated sessions **36 of 322
  outcomes** are environment mechanics ("Set up a Kaggle account with phone verification", "Use Ngrok
  to create secure tunnels"), and one session has **5 of 6** uncoverable. No interview question can
  cover those, so on the Image-Generation topic the best achievable coverage was **0.56 — under the
  0.60 pass bar with unlimited questions**. `interview_topics` is built from the same reading material
  for exactly this purpose and has **0 of 385** such items. It already fed retrieval, the session-fit
  profile and the relevance judge; coverage was the one place that ignored it.
- **Two numbers, because one cannot do both jobs** (`pipeline.CoverageResult`):
  `topic_coverage` = covered / ALL topics is honest but **bounded by supply** — with 5 questions and 22
  topics it cannot exceed 0.23, and one run scored **0.227, its exact maximum, and was failed**. So it
  is **reported, never gated**. `coverage_efficiency` = covered / min(topics, questions) — "did each
  question earn its place against a *distinct* topic" — is **scored and gated**, achievable at any set
  size, and still fails a set whose questions pile onto one topic (5 questions → 3 topics = 0.60).
- **`predicted_accept` has no veto.** It keeps its 30% composite weight but cannot fail a set alone: it
  is a 1-NN estimate over ~15 labels that `human_agreement` documents as an optimistic upper bound at
  65% accuracy, and it read 0.2 on two consecutive runs purely because those topics had few labels.
- **`QualityReport.gate_checks` names every condition with its value and bar**, and a supply cap is
  stated as a corpus fact. The report used to carry `pass_fail` alone, so three failures in a row gave
  no way to tell a thin corpus from a bad set.
- **Calibration is replayed, not guessed.** `run_results.payload_json` persists every run, so a
  threshold change is checked against history. On the 9 runs with the full metric set the new rule
  passes 6 (the reviewer-approved one included) and still fails the three weakest; 0 runs regress from
  pass to fail. Runs missing `session_grounding`/`predicted_accept` must be EXCLUDED from any replay —
  their composite renormalises to ~1.0 and inflated a first pass at this from 67% to 88%.

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
- **That assertion had no tell for LETTERED options, and 52 MCQs were sitting in the main bank with the
  test green.** `MCQ_SHAPES` held only PROSE tells ("which of the following", "all of the above",
  "options:"), so `"What is the right way to initialize an array? A) int num[6] = {2,4,12,5,45,5}; B) …"`
  passed. `src/assessment_items.py` now owns the shape rules and both the test and
  `scripts/strip_assessment_items.py` import them, so the sweeper and the assertion cannot drift.
  (1,499 → **1,447**.)
- **Two shapes, two remedies — do not treat a lettered marker as one thing.** `≥2 distinct option
  letters` ⇒ a real MCQ, delete it. **Exactly one marker ⇒ an ANSWER glued onto a real question**
  (`"What's RLHF, and why does it matter?A. RLHF (Reinforcement Learning from Human Feedback) trains…"`,
  6 rows) — **repair by truncating at the marker**, never delete: those are genuine questions with
  scrape residue and became clean 7-10 word questions. Requiring TWO letters to delete is what protects
  a legitimate parenthetical (a coding problem reading `"the fewest number of digits (d) and base
  number (m)"` matches one marker and must survive). The word rule in `quality.py` is **not** a
  substitute for the repair: only 2 of those 6 exceeded 40 words, so it would have missed 3 and deleted
  the other 2.
- `interview_questions.json` still has **127 rows failing the form gate** (embedded code blocks,
  HackerRank specs). That is pre-existing, no test guards it, and it is NOT the MCQ class — do not
  conflate the two when cleaning.

## The form gate has no upper bound — but length is not the discriminator

A live run shipped **56 words of interviewer rubric** as a question: *"When your prompt produces the
wrong output, the question is how quickly you can narrow down why. The failure could be in the
instruction itself…"*. It passed everything — it opens with "When" (a legitimate `_Q_STARTS` word) and
is normal-cased — because `quality.py` had `_MIN_WORDS = 3` and **no maximum at all**.

- **A bare word ceiling is the wrong fix, and this was measured before the rule was written.** Of the 26
  shipped rows over 40 words, 7 are prose blobs and the rest are *genuine* long asks — *"Build an API
  for a leave request system in an HR management system using Flask, FastAPI…"*, *"Can you explain the
  request flow when a user creates a blog and hits publish…"* — plus HackerRank coding specs. Rejecting
  on length would have killed 7 blobs and **9 genuine questions** with them. Same overlapping-
  distributions trap as `_outcome_coverage`'s proximity threshold and the dedup bar.
- **What separates them is whether the text ASKS the candidate anything** (`quality._asks_something`):
  its OWN question mark — one inside a *quoted example* does not count, which is what makes *"A
  recruiter might ask, “Tell me about a project where you applied LLMs”…"* prose — or a sentence opening
  with an imperative task verb.
- `_TASK_VERBS` is **derived from `_Q_STARTS`** so the two cannot drift, minus wh-words and minus
  **`given`**: *"Given the above, I wanted to build…"* and *"Given these, the best way to prepare is…"*
  are blog prose, while the coding specs that open that way carry a later `find`/`determine` and survive
  on those.
- `_INTERVIEW_META` (talks ABOUT recruiters/hiring rather than asking) applies **only above the length
  bar**, so *"What do recruiters look for in a GenAI candidate?"* is untouched.
- Measured: rejects **8 of 2,835 rows (0.28%)** — all 7 blobs, plus one coding word problem the dormant
  coding path cannot use. Two known misses remain; a tighter rule costs genuine questions. Reviving
  coding retrieval will need a code-aware exemption.

## The open-web tier, and why it had never once fired

Written specifically for the n8n gap, and dead on arrival for two rounds. Run `8fb9fcb3` shipped 5
questions for a topic whose sessions name n8n, RSS Feed Read Node, Schedule Trigger and Gmail Send Node
across **10 of 22 outcomes**, and **0 of the 5 questions touched any of it**.

- **`surviving >= MIN_QUESTIONS` read the floor as success.** Surviving was **exactly 5** and
  `MIN_QUESTIONS` is **5**, against a request of 15 (`set_size` scored 0.33). Landing precisely on the
  floor is the most starved a run can be while still producing output. `_open_web_shortfall` now fires
  below `max(MIN_QUESTIONS, ceil(OPEN_WEB_TRIGGER_RATIO × requested))` — 9 of 15 at the 0.6 default.
- **The `<= MIN_QUESTIONS` clause is SEPARATE from the ratio and must stay.** With
  `requested == MIN_QUESTIONS` the ratio floors to 5 and `surviving < 5` reintroduces the very
  off-by-one this exists to fix. `tests/…::TestTheTriggerThatNeverFired` asserts the whole table rather
  than the constant, because the table is the defect.
- **A second trigger fires on zero tool representation, at ANY count.** A full set of 15 on an n8n
  session that never says "n8n" is still the wrong set and no count test can see it
  (`_unrepresented_terms`, word-boundary matched, `[]` for theory-only sessions). Those missing terms
  **become the query**, ahead of the old generic `_tool_terms(ctx)[:4]`.
- **The cross-topic pre-filter killed the one real n8n question retrieved** — *"What kind of workflows
  have you built with n8n before…"* — because `_prefilter_semantic` compares POOLED course-topic
  profiles, and pooled across the GenAI course an n8n question resembles "the course" less than a prompt
  question does. A candidate naming a `_tool_terms` term is now exempt from **that stage only**;
  `_score_session_fit` runs BEFORE it (so the "interviewing at RSS Security" false positives stay
  dropped and cannot be resurrected), and the relevance judge and syllabus audit still follow.
- **`_tool_terms` must return PRODUCTS, not capitalised concepts — it reaches a live search.** On the real
  curated outcomes it returned `['Acting', 'Reasoning', 'Observation', 'ReAct']` for an AI-agents session,
  so the zero-representation trigger fanned out to Tavily for *"Observation interview questions"* and
  **exhausted the Tavily plan** (21 pipeline runs in the integration file × up to 4 searches each). The
  docstring's claim that theory sessions yield `[]` was false against real data, and the test asserting it
  passed only on a crafted input. Fixed by extending `_NOT_A_TOOL` with the abstract/process vocabulary.
  Two rules measured and **rejected**: mention count *inverts* the answer (`n8n` is mentioned once,
  `Observation`/`Reasoning`/`Acting` twice each), and requiring a product identifier (letter+digit or
  ALL-CAPS) silences theory sessions but also loses `Kaggle`, `ChatGPT`, `Gamma`, `Otter`, `Bolt`,
  `Hugging Face`, `Google Sheets`. Anything added to that extraction reaches a paid search.
- **Open-web queries are PLATFORM-QUALIFIED, and skipping that made the whole tier useless.** The first
  run where the tier actually fired searched *"RSS, n8n, Merge"* and every candidate came back about
  **Merge the company** (*"What was your interview with Merge like?"*), `pandas.merge()`, merging sorted
  lists, and *"the ServiceNow RSS web service"*. The relevance judge correctly rejected all of them, so
  the tier fired, spent its calls and added nothing on-topic — a failure that looks identical to "the
  web has no n8n content". The cause is upstream of both: the reading material writes *"Merge and
  Aggregate **nodes**"* and *"**RSS** Feed Read Node"* with a lowercase head, so `_PROPER_RUN` can only
  ever capture the bare `Merge` / `RSS`. `_qualify_tool_terms` prefixes them with the session's platform
  (the highest-ranked letter-digit token — `n8n`) → *"n8n Merge"*, *"n8n RSS"*. With no such token the
  terms are returned unchanged rather than qualified by a guess.
- **`_score_session_fit` takes `only_ids` and it is not a convenience.** Open-web additions used to land
  with `session_fit = None`, which drops them out of `session_grounding` (it averages non-None only, so
  the metric silently measured just the vetted subset) and sinks them in Review's ranking (`_rank_key`
  reads None as 0.0). A blanket re-run is the wrong fix: the floor is **relative to the pool's best
  fit**, so re-scoring after adding up to 60 candidates moves the bar and evicts vetted questions whose
  place was already decided. With `only_ids` the floor is still computed across the whole pool — the
  session's real bar — but only the named ids can be dropped.

## Same-thing duplicates: judged on the selected set, never by threshold

Run `8fb9fcb3` shipped *"How would you modify a system prompt to ensure the model always responds in
structured JSON format?"* **and** *"How do you write effective prompts for consistent JSON output?"*.
The gate's critique named them; semantic dedup measured them at **0.767** against its **0.82** bar.

- **Do NOT lower `DEDUP_SEMANTIC_THRESHOLD`.** Measured: 209 pairs in 900 GenAI-bank rows sit in
  [0.74, 0.82), and the band holds both real duplicates (*"What is a neural network?"* / *"What is a
  Neural Network and ANN?"*, 0.815) and legitimately distinct pairs (*"What is fine-tuning in LLMs?"* /
  *"best practices for LLM fine-tuning?"*, 0.819). Overlapping distributions — no cutoff separates them.
  `tests/test_same_thing.py::TestTheDedupThresholdIsNotTheFix` pins the constant so the cheap wrong fix
  fails loudly.
- **`tools._same_thing_pass` judges instead**, and it is affordable only where it runs: one LLM call over
  the **5-60 SELECTED** questions, after `_scope_trim` (trimming can pull two questions onto the same
  core ask). Never at retrieval time — that is the mistake `_scope_trim` documents at length.
- **The reply is verified in code**: only a pair index we actually asked about can be acted on, and only
  pairs in `[_SAME_THING_LOW, DEDUP_SEMANTIC_THRESHOLD)` are ever offered — anything above the bar was
  already removed by `tool_deduplicate_questions`.
- **At `MIN_QUESTIONS` it FLAGS instead of removing** (`QuestionDetail.duplicate_of`, a `same as another`
  tag in Review, and the report names the count). That floor case is exactly why the duplicate shipped
  last time: the critique flagged it, the set held exactly 5, and `remove_question` correctly refused.

## The interview is CONVERSATIONAL — hands-on prompts are unanswerable

Students answer out loud: no keyboard, no IDE, no whiteboard. So *"Write a Python program to generate the
Fibonacci series"* is not a hard question, it is an impossible one. Two real runs shipped such prompts
(*"Implement an input box to interact with the Gemini API…"*, *"Build and integrate LLM applications."*),
each burning a slot in a set of 9. `interview_format.is_hands_on_task` decides; `pipeline._drop_hands_on`
filters the pool; `config.CONVERSATIONAL_ONLY=0` turns it off.

- **"Design"/"draw" ARE hands-on — this was REVERSED, and the reversal is the interesting part.** They
  were excluded for one round on the argument that *"Design a news aggregator system"* means "talk me
  through the architecture". Reversed on the reviewer's call: an imperative *"Design an end-to-end image
  generation system. Cover the following: …"* is a whiteboard exercise, and a candidate with no board
  cannot do it any more than they can type code.
  **The wh-opener exemption is the only reason this is safe** and not a keyword ban on "design" —
  measured on the 236 live questions, **21 are skipped** (*"Design an LLM API pipeline"*) while
  **17 are kept** (*"How would you design an agentic workflow?"*, *"How do you approach designing an
  effective prompt?"*). Adding the verbs without the exemption removes both halves.
  When design was first excluded, the measured cost of including it was that **2 of 6 runs fell under the
  5-question minimum**; that risk is now absorbed because a topic's accumulated set is carried into every
  run (`tools._add_retained`). Cost at ingest: 75 of 2,828 bank rows (2.7%). The compound
  **`"design and implement"` IS caught** — it reads like discussion and demands an artifact.
- **A wh-opener is never hands-on**, and that single exemption carries the whole distinction:
  *"How did you implement JWT authentication in your project?"* is a strong question about a candidate's
  own work; *"Implement JWT authentication"* is a task. Same verb, opposite verdicts.
- **Verb alone is not enough — the FRAME matters.** *"Can you create a DataFrame in Python…?"* is
  keyboard work despite the polite frame, so the rule also matches `can/could/would you`,
  `your task is to`, `I want you to`.
- **This is NOT in `quality.py`, and that is deliberate.** A hands-on task IS a well-formed question; the
  form gate answers shape, this answers format-suitability. Concretely, `scripts/clean_bank.py` deletes
  every row failing the form gate, so putting it there would **permanently destroy 217 real
  company-attributed coding questions** that the LMS coding tabs exist for. Same separation-of-concerns
  reason `src/assessment_items.py` is its own module.
- **It filters the POOL, right after `_drop_rejected`** — before session-fit embeddings and long before
  the LLM relevance judge, so nothing is spent on candidates that cannot ship. A shortfall it creates
  needs no new wiring: `_top_up_from_open_web` runs at the end of `_pick_questions` and already fires
  below 60% of the requested count.
- **The same predicate runs in `add_open_web_records`.** The tier fires precisely when the set is short,
  and the open web is full of "write a function to…" — without it the backfill hands straight back what
  the pool filter just removed, and the filter looks broken.
- Measured: rejects **217 of 2,828 rows (7.7%)**; a 14-row random audit of the rejects found no false
  positives; **no run in the last 6 falls under the minimum** because of it (worst single loss 9→7).
- **`removed[].stage == "hands_on"` is counted by `scripts/yield_report.py` and named in the report.** A
  pool filter that shrinks supply silently gets misread as "this session has few questions" — the exact
  misdiagnosis the yield harness exists to prevent.
- **A test asserting only "it isn't in the shipped set" is VACUOUS here**, and a mutation check is what
  caught it: `_select_final` trims ~150 candidates to 8, so an injected question misses the cut whether or
  not the filter ran, and the assertion passed with the filter unwired. Assert the **removal record**.

## The suite must cost nothing — and it did not

CLAUDE.md claimed "no LLM or network required" for months while every `pytest tests/` run spent real
money. Nothing enforced it, and a live call that succeeds is indistinguishable from a stub that works.

- **`tests/conftest.py` blocks outbound sockets** (loopback stays open for Starlette's in-process
  `TestClient`) and sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`. **If embeddings break under the guard,
  set those — do not weaken the guard.** Opt out with `@pytest.mark.allow_network`, used by nothing.
- **RAISING IS NOT ENOUGH, and this is the subtle part.** Every leaking call site — `_scope_trim`,
  `_syllabus_audit`, `_same_thing_pass`, `tool_validate_relevance`, `fetch_open_web` — is deliberately
  FAIL-OPEN. The first version of the guard was armed and correct, the whole suite passed, and it reported
  **zero leaks while calls were being attempted and swallowed**. So attempts are RECORDED in
  `tests/netguard.py` and the offending test is failed in teardown.
- `netguard.py` is a separate module on purpose: pytest loads `conftest.py` through its own plugin
  machinery, so `from tests.conftest import …` in a test yields a SECOND module object with a different
  exception class and a different ledger.
- **The measured leaks were four, not the three a static audit predicted** — file-level "does it stub
  `chat_completion_json`?" gave a false negative on `test_failure_paths.py::TestSelectionIsGuaranteed`,
  whose two tests reach `tool_submit_question_set` via `_enforce_submission`. The predicted Tavily leak
  was already gone, fixed by the `_NOT_A_TOOL` change. **Measure; do not audit by grep.**
- Corroboration: the suite got **16s faster** (108s → 92s) once the round-trips went.

## The de-stack verdict: the semantic filters are NOT the problem

Recorded so the "three overlapping filters" theory is not re-litigated. Replayed both embedding stages
offline against 13 persisted runs (free — they are local embeddings, and the pool is reconstructible from
`removed[]` + `question_details`):

- `_score_session_fit` and `_prefilter_semantic` overlap **65%** — not duplicates.
- **The prefilter's unique contribution is 644 candidates averaging 0.501 session-fit against 0.625 for
  shipped questions; NONE reach the median shipped fit, and it has never dropped a question that
  shipped.** Removing it would only push weaker material into the expensive LLM stage.
- The LLM judge is **not** anti-correlated with grounding: rejected 0.564 vs shipped 0.627, inverted in
  only 1 of 13 runs. (Do not build a thesis on a single run — that one was the outlier.)

### What actually bound the funnel: the relevance back-fill stopped at the floor

`tool_validate_relevance` cuts a mean **88%** of everything reaching it. It keeps ≥ `RELEVANCE_THRESHOLD`
(0.50) then tops up from ≥ `RELEVANCE_FLOOR` (0.35) — but the target was `min_questions` (5), **not the
requested count**. The same off-by-one as the old open-web trigger: a supply-aware net calibrated to the
absolute minimum, so a run asking for 15 filled to 5 and stopped.

Replayed on the persisted `relevance_score`s: median shipped-equivalent **6 → 14**, 11 of 13 runs gaining.
The admitted [0.35, 0.50) band is grounded **as well as what ships** — mean `session_fit` 0.610 vs 0.625,
against 0.561 below the floor — so the 0.50 cutoff was discarding questions indistinguishable in grounding
from the ones it kept. `coverage_efficiency` was replayed too and holds at 1.0 across all 13, so bigger
sets do not cost the gate. **The FLOOR stays absolute**: a thin on-topic pool returns FEWER questions, never
loosely-related filler.

## A re-run ADDS to a topic's set — it does not make another version

`tools._add_retained` unions the topic's accumulated set into the shipped output, so re-running after a
pipeline improvement keeps what exists and adds only what is new.

- **The accumulation already existed and pointed the wrong way.** Approved questions have always banked on
  approve, and cross-run dedup already removed candidates duplicating them — so a re-run computed the
  delta correctly and then shipped ONLY the delta, looking like a fresh, smaller set. The missing piece
  was putting the set back into the output.
- **`_add_retained` runs after `_same_thing_pass` and before `_syllabus_audit`**: the same-thing pass
  should judge only this run's own picks against each other, not re-litigate settled questions, and the
  audit should flag retained questions like everything else.
- **A retained question today's gates reject is FLAGGED (`stale_reason`), never dropped.** A reviewer
  approved it and the improvements post-date parts of the set; a silent removal is a surprise.
- **`_score_unscored_fits` gives retained questions a `session_fit`, and drops NOTHING.** It deliberately
  does not reuse `_score_session_fit(only_ids=…)`, which applies the relative floor and removes what falls
  below. Leaving them unscored is the open-web trap: `grounding_score` averages non-None fits only, so
  `session_grounding` would describe just the freshly-found subset, and `_rank_key` reads None as 0.0.
- **The report names the retained/new split, reported never gated** — a re-run that finds nothing new
  would otherwise read as a healthy 40-question run. Gating on new-questions-found would fail every mature
  topic that is simply finished, the same reason `topic_coverage` is not gated.
- **`retained_status` keeps 'approved' distinct from 'backfilled'** (76 vs 175 after consolidation), so a
  one-off import of unreviewed questions does not silently acquire reviewer blessing.
- **Tests that call these helpers directly prove nothing about wiring.** A mutation check showed that
  unwiring both `_add_retained` and `_score_unscored_fits` left every unit test green — they call the
  functions directly. `TestItIsActuallyWiredIn` goes through `tool_submit_question_set` and
  `TestRetainedQuestionsThroughTheWholePipeline` through `AgentPipeline.run`. Third time this class of
  vacuous test appeared in this codebase; assert through the caller.
- **Trim assertions must count NEWLY-FOUND questions.** Four tests asserted the shipped total was <= the
  requested count; retained questions legitimately break that, and the invariant they protect is that the
  raw pool was trimmed.

## One spreadsheet per topic

`write_to_sheets` reuses the topic's sheet instead of creating one per approve.

- It used to `client.create(...)` unconditionally and never persist the URL, so every approve minted
  another spreadsheet and no topic had a single reference. `topic_sheets` now stores the id.
- **`org_id`/`interview_id` are persisted and reused.** They were fresh `uuid4()`s per call, so a
  re-export would look like a brand-new interview to the LMS import rather than an update.
- **Title is `"<Topic> - NxtMock"`** — no session names. Including them was misleading once the sheet
  became per-topic, since a differently-grouped re-run updates that same sheet; real titles also ran past
  100 characters.
- **Tab 1 is cleared before writing.** The write starts at A1 on a REUSED sheet, so a previous, longer
  export would leave its tail behind and silently mix removed questions into the current set.
- A deleted or inaccessible sheet falls back to create-and-re-save: an approve must not fail because a
  spreadsheet went away. Accepted knowingly: rewriting in place overwrites manual edits.

## The grounding profile must be the WHOLE reading material, not a sample

`_session_profile` used to keep **~36%** of a session's material. Three lossy steps compounded: a
`len(c) > 120` filter, a `chunks[::step]` **stride sample** down to 12, and a `[:800]` truncation.
Measured on the No-Code sessions: 24,521 chars → 8,728.

- **The loss was biased toward exactly what identifies a session.** The line defining the node most
  precisely is 85 characters — `- **HTTP Request Node**: Allows n8n to talk to almost any web service…` —
  so the `>120` filter discarded it, the phrase appeared in NONE of the 12 surviving chunks, and
  *"What is the HTTP Request node and when do you use it?"* scored **0.275** against a session that
  literally teaches it. Now **0.540** (0.703 on the other topic that teaches it).
- **`SESSION_PROFILE_RM_CHUNKS` is a RUNAWAY GUARD, not a sampling budget** — now 400. Conflating the two
  is what caused the bug. Raise it rather than sampling. `RM_CHUNK_MIN_CHARS = 40` because in this
  curriculum the bullets ARE the tool definitions; long paragraphs are sub-split at a word boundary
  rather than truncated.
- **Do not overclaim the effect, which was measured both ways.** The dramatic gain is on questions naming
  something specific: a topic's median fit moves only 0.615 → 0.622, ~5 of 40 questions change materially.
  Replayed over 7 persisted runs, the fit gate would drop **936 → 924** candidates — only 12 fewer,
  because `SESSION_FIT_RELATIVE` makes the floor 0.5 × the best fit, so lifting every score lifts the
  floor too. `session_grounding` gains **+0.013** mean, worth **+0.003** on the composite. So this fixes
  per-question scoring, ranking and any filtering built on it; it is **not** a retrieval supply win, and
  the earlier hypothesis that it explained the n8n shortage is **not supported**.
- Cost: **4.5×** more chunks per profile (168 → 750 across six topics), local MiniLM. The suite went from
  ~105s to ~140s because the integration tests run real pipelines.
- `tests/test_session_profile.py` asserts against the SHIPPED material, not a fixture: any short synthetic
  paragraph would have been dropped by the same filter, so a crafted test would have passed while
  describing nothing. Mutation-checked — restoring either the `>120` filter or the stride fails it.

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

## Multi-topic batches: ONE RUN PER TOPIC, never a merged run

`POST /api/generate/batch` queues one full pipeline per selected topic; `src/pipeline.py`, the agents and
the gate are untouched by it.

- **Do not "simplify" this into one merged run.** Review, approve, rejection feedback and learned rules
  are all keyed by `run_id`, and `sheets_writer` titles one spreadsheet per run
  `"<Topic> - <sessions> (NxtMock)"`. Merging would give one gate verdict and one all-or-nothing approve
  for every topic, and would push 6-12 sessions into a single `SessionContext` — the per-session
  attribution collapse documented above showed up at **two**. (22 topics ship, 1-4 sessions each.)
- **Runs are SEQUENTIAL, and `_start` has no lock to stop you making them parallel.** Two reasons it
  must stay sequential: a 3-topic batch is 3 full pipelines, so parallel multiplies the LLM/Tavily burst
  (this project has already exhausted a Tavily plan and an OpenRouter key's headroom); and `memory.db` is
  SQLite, where concurrent pipeline writes are lock contention for no user benefit.
  `tests/test_batch_generate.py::TestRunsAreSequential` asserts **no two runs OVERLAP**, not the call
  count — a parallel worker still produces N calls, so counting proves nothing.
- **A failing topic does not stop the batch** — it is recorded on that row and the worker moves on, the
  same discipline as `state.phase_errors`. Validation is all-or-nothing though: one unknown topic rejects
  the whole request before anything is queued, so a half-started batch cannot silently spend credit.
- **`GET /api/batch/{id}` must never join the worker.** `/api/result` already 409s in flight because
  polling tabs blocking on a thread starved Starlette's bounded threadpool.
- Every `run_id` is minted and given its SSE queue up front, so `/api/stream/{run_id}` and the Progress
  page work per topic with **no SSE changes**. `run_history.batch_id` is the durable copy, since the
  in-memory `_batches` registry is bounded and pruned.
- Preview mode is deliberately unavailable for a batch: it pauses mid-run for a human, which would stall
  every queued topic behind it.

## Repeated questions in an accumulated set, and what each duplicate check can actually see

A review of No-Code AI Automation found hallucination asked **six times in 38 questions**, and prompt
engineering about as often. Three separate mechanisms exist to catch that and each missed for its own
reason, so do not "fix" one by pointing at another.

- **Embedding similarity cannot group them at ANY usable threshold.** Across the six, **zero pairs reach
  the 0.82 `DEDUP_SEMANTIC_THRESHOLD`**, and the two most obviously identical — *"What are hallucinations
  in LLMs"* / *"What is an AI hallucination"* — score **0.486, the LOWEST of all fifteen pairs**. Short
  definitional questions carry little signal and "LLMs" vs "AI" pushes them apart. That is why
  `filter_topic_sets._clusters` groups by a **shared distinctive term** and uses similarity only for
  cohesion and member order. `tests/test_topic_dupes.py` asserts the 0.486 pair directly so a future
  "just lower the threshold" cannot look reasonable.
- **A cluster means "same subject", not "duplicate".** On the shipped sets `hallucinat` (6) and
  `large/language` (8) are real duplication while `workflow` (7) is seven different questions sharing a
  word. So collapsing is per-cluster and explicit (`--collapse TERM`), never blanket. Cohesion ≥ 0.45 is
  what keeps a merely-shared word from forming a cluster; it rejects **55 term-groups** across the nine
  topics, and a test that does not exercise a real rejected group is vacuous (the first version used
  hallucination-vs-Kubernetes, which share no distinctive term, and passed with the bar set to 0.0).
- **The cluster cap is a share of the set** (`max(2, 30%)`), so a cluster is never just "the whole topic".
  A consequence for tests: six questions are 16% of the real 38-question set but 100% of a six-item
  fixture, which the cap correctly refuses — fixtures must be realistically sized.
- **`_same_thing_pass` is the accurate test and never saw most of the set.** `_SAME_THING_MAX_PAIRS = 12`
  was sized for a ~10-question selected set; the 38-question accumulated set has **41 eligible pairs, so
  29 are never judged**, and the one hallucination pair the judge does call redundant ranks **14th** —
  just past the cap. With the cap lifted the judge calls **17 of 41** pairs the same thing. So "0
  redundant" from this pass can mean "not looked at". Its floor is `_SAME_THING_LOW = 0.62`, which also
  means the 0.486 pair is never shown to it — `--dupes` and this pass are complements, neither subsumes
  the other.
- **`_same_thing_pass` MUTATES the list it is handed** (`questions[:] = [...]`). A caller that diffs
  against that same list afterwards finds nothing removed; `scripts/judge_topic_sets.py` reported
  "0 redundant" while the pass was removing five. Snapshot before the call, or read the returned counts.

### Why the off-syllabus flag stayed silent on a question that IS off-syllabus

*"Explain your methodology for designing and testing system prompts to prevent model hallucination"* —
reviewer-approved. The material teaches hallucination (`hallucinat*` **3–5** occurrences) but has
**`system prompt` 0 times** and `methodolog` **0 times**.

`_syllabus_audit`'s LLM **did** claim it, as `system prompt design methodology`. `_concept_is_absent`
**dropped the claim** — and correctly, by its own rule. `system`/`prompt` are in `_UBIQUITOUS_DOMAIN`, so
the distinctive words are `design` and `methodology`; `methodology` is absent but **`design` appears 4
times** in unrelated senses ("design prompts that improve consistency"), and the rule is `all(missing)`.
One present word vetoes the claim. Across that topic the LLM proposed 9 untaught concepts and **1
survived**.

That is the documented conservative trade-off, not a bug: a majority rule instead lets "agent guardrails"
through on 1-of-2. The consequence to remember is that **a compound concept whose words are individually
common can never be auto-flagged**, so `--evidence` (phrase-occurrence counts, no LLM) is the tool for
that class — it reports the asymmetry (`hallucinat* x3`, `system prompt x0`) and never cuts anything.

## The judge degrades with batch size — that was the bug, not the cap

The most consequential finding of the balance work, recorded because the wrong diagnosis was very
convincing. A reviewer flagged that hallucination was still asked six times in No-Code AI Automation.
Measured on *"What are hallucinations in LLMs"* / *"What is an AI hallucination"* — the most obviously
identical of the six:

```
batch of  1 pair   -> SAME, 3 of 3 trials
batch of 52 pairs  -> "different"
```

In a 4-pair batch the same judge correctly called the fine-tuning pairs DISTINCT *and* the
prevent/mitigate hallucination pair the same. **Small batches are right in both directions; one large
batch was wrong in both.** So `outcome_balance.JUDGE_BATCH = 10`, and 52 pairs costs 6 Haiku calls
instead of 1. Do not raise it to "save calls" — it degrades every verdict silently rather than failing.

- **`_same_thing_pass` sends up to `_SAME_THING_MAX_PAIRS = 12` in ONE call and is at the edge of this.**
  Not changed yet; it is the likeliest explanation for its "0 redundant" verdicts on large sets, on top of
  the cap documented above.
- **The first diagnosis blamed a per-outcome CAP and would have shipped real data loss.** With a hard
  quota of 2, Gen AI Foundations lost **12 of the 14** questions under one coarse topic
  ("Pre-trained vs fine-tuned models"), including *"What is the difference between pre-training and
  fine-tuning?"*, *"What are the different Fine-tuning methods?"* and *"What is the difference between a
  base model and an instruction-tuned model?"* — distinct asks that merely share a coarse outcome.
- **The tell was that the same quota was right on one topic and destructive on another.** No-Code has 22
  interview topics for 38 questions, Gen AI Foundations 15 for 54. The quota was measuring how finely the
  curriculum enumerates `interview_topics`, not whether the questions repeat. Same
  overlapping-distributions trap as `_outcome_coverage`'s proximity threshold and
  `DEDUP_SEMANTIC_THRESHOLD`: no count separates the populations, so a judgement has to.

## Per-outcome balance: `src/outcome_balance.py`

`coverage_efficiency` asks "did each question earn its place against a DISTINCT interview topic" — but
only inside a single run's selected set. Nothing applied it to the ACCUMULATED per-topic set, and
`tool_submit_question_set` runs `_same_thing_pass` **before** `_add_retained`, so the accumulated set was
never judged against itself or against the run's new picks. Six hallucination questions accumulated one
run at a time, each below the 0.82 embedding bar.

- **Grouping is by interview topic, and cutting is by JUDGEMENT alone** (`strict=False` default). `cap` /
  `OUTCOME_CAP` only feeds the opt-in `strict=True` quota; nothing is dropped without a verdict.
- **An ORPHAN is KEPT, never cut.** Best-outcome match below `OUTCOME_ORPHAN_FLOOR = 0.35` means no
  outcome describes the question: *"What is the Split In Batches node used for?"* matches at **0.173**,
  *"What are nodes in N8N"* at **0.132**. Those are the n8n gap showing up in the outcome list, not
  duplicates. Counting them as redundant turns the feature into silent data loss.
- **Ranking is approved-first, then `session_fit`** — a reviewer's decision is newer information than a
  score, so *"What are hallucinations in LLMs"* (0.516, approved) is kept over a 0.719 backfilled one.
- **Two call sites, deliberately different in what they destroy.** `tools._cap_by_outcome` trims only what
  a run SHIPS and leaves `topic_question_set` whole, so it is idempotent and reversible;
  `scripts/filter_topic_sets.py --cap 2 --apply` quarantines rows recoverably and calls
  `memory.sync_canonical_payload`. Both call `balance_by_outcome`, so they cannot drift.
- **`make_llm_judge(complete=…)` must receive the CALLER's `chat_completion_json`.** The first version
  imported it from `src.llm_client` inside the judge, so every existing
  `monkeypatch.setattr(tools, "chat_completion_json", …)` missed it and `tests/netguard.py` recorded
  **3 real network calls per test** — swallowed by the fail-open handler, so assertions passed while a
  working key would have spent credit.
- **`tests/test_failure_paths.py` had NO database isolation** and was reading the live `memory.db`:
  `_add_retained` silently pulled in 32 real questions, so its "the pool was trimmed to 5" assertion was
  really measuring production data and moved whenever the shipped sets changed.
- **ONE judge call PER OUTCOME, and this is what makes the pass idempotent.** A verdict depends on which
  OTHER pairs share its batch: the same 28 pairs of "Pre-trained vs fine-tuned models" gave 3 "same"
  verdicts batched alongside other outcomes' pairs and **0 across 3 trials** batched alone. With one flat
  call, a second `--apply` cut **8 more** questions — including *"What is the difference between a base
  model and an instruction-tuned model?"*, which the first pass had deliberately KEPT as distinct. Per
  outcome, the pass converges in one step (41 → 38 → 38 → 38) and a re-run proposes **0** cuts.
- **`majority()` (2 of 3) guards the DESTRUCTIVE path only.** `--apply` writes to the database, so one flap
  is permanent; `_cap_by_outcome` only trims what a run ships and re-derives it every run, so paying 3×
  there to stabilise a reversible decision is not worth it. Report mode is single-pass, so the preview can
  differ slightly from `--apply` — in the conservative direction.
- Applied to the 8 accumulated sets: **213 → 169**, hallucination on No-Code **6 → 2** (a definition and
  an applied question), `memory.db` backed up, all 46 cuts in `quarantined_questions`.
- **The quality-report note must not name `OUTCOME_CAP`.** The pipeline runs `strict=False`, so counting
  plays no part in what was dropped; an earlier note credited "cap 2 per topic" and described the wrong
  mechanism entirely.
- The uncovered half is reported and never gated: **9 of 22** No-Code outcomes have no question at all
  (*Human-in-the-loop*, *Testing and validation*, *Scheduling and triggering*…). Retrieval for those is a
  separate decision — the same reason `topic_coverage` is reported and not scored.

## Retrieval is the run — a dead web tier stops it before anything is spent

`REQUIRE_WEB_SEARCH=1` (default). A failed Tavily pre-flight raises `agent.RetrievalUnavailable` and the run
ends; it used to only set `web_search_disabled` and carry on bank-only through the relevance judge, the
Evaluation agent, the syllabus audit, the same-thing pass, the outcome balance and up to three gate
critiques — all judging a pool the failure had already decided.

- **THE POSITION OF THE PROBE IS THE FEATURE.** It moved from `RetrievalAgent.run` to
  `pipeline._tavily_preflight`, called FIRST in `_pick_questions`, because `RetrievalAgent` runs *after*
  `UnderstandingAgent` and understanding costs one `chat_completion_json` per session. Checking in the old
  place and then stopping would still have spent those calls. `RetrievalAgent` now reads the pre-probed
  status (`state.web_probes`) instead of probing again — a second probe is a silent cost regression no
  behavioural assertion catches, so a test pins the count at 1.
- **The cost of this guard is measured and accepted, not assumed.** The bank supplies **75%** of all shipped
  questions (459 of 615 across 62 runs), and on a 17-run sample **12 would have shipped ≥5 questions
  bank-only**. So the guard refuses runs that would have worked — the chosen trade-off is guaranteed-no-waste
  over best-effort output. `REQUIRE_WEB_SEARCH=0` restores the previous behaviour exactly and is the only
  route back, so `TestTheEscapeHatch` guards it.
- **Terminal vs transient is a real distinction.** `no_key`/`auth`/`quota` fail immediately — nothing will
  work today. `rate`/`error` get ONE retry (`WEB_PREFLIGHT_RETRY_STATUSES`): a single 429 must not be able to
  kill an 8-topic batch. The retry lives in the caller, not in `health_check`, which stays one honest probe.
- **"Nothing is spent" needs a spy on BOTH LLM boundaries.** `chat_completion_json` (bound at import time by
  every module that uses it, so each needs its own patch) **and** `base_agent.get_client`, because the agent
  tool loops call `client.chat.completions.create` directly. A mutation check exposed this: with the probe
  moved back after Understanding, the assertion still passed and only `tests/netguard.py` saw the 3 real
  connections.
- **A failed run now reaches History.** `main._persist_result` returned early on any error, so a Tavily
  outage was indistinguishable from never having pressed Generate. It now writes a `run_history` row with
  `error` and `question_count = 0` and NO `run_results` payload, so Review has nothing to open — History
  renders it as `Failed`, non-clickable. The aborted result gets `_fallback_context` or every failed row
  would read "Unknown".
- **`/api/result` reports the failure before the row is persisted** (`main.py:457` sets `_results`, 458
  persists). A test that polls the endpoint and then reads History races — wait for the row.
- **`tests/test_pipeline_integration.py` reports Tavily HEALTHY and stubs the web calls instead.** It used to
  stub `health_check` as `no_key` to force bank-only, which now aborted all 18 pipeline runs in that file
  before they reached the behaviour under test.

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

`pytest tests/ -q` — 711 tests. **No LLM or network required, and that is now ENFORCED** by an
autouse guard in `tests/conftest.py` (see "The suite must cost nothing" below) rather than being a
hopeful claim. Beyond unit coverage:
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
