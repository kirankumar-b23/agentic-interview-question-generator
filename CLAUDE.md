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
  interview_questions.json          # 1,509 company-attributed questions — general SWE/Python, little GenAI
  genai_question_bank.json          # 1,465 curated GenAI questions (build_genai_bank.py; swept by clean_bank.py)
  knowledge_graph.json              # KPs + sessions + prerequisite edges
  course_structure.json             # Topic → list of sessions (units); drives UI selection
  reading_materials/session_map.json# Canonical session name → that session's reading material
  reading_materials/*.md            # Source course reading materials (gen_ai, llm_applications)
  curriculum/*.json                 # KP-mapped curriculum questions (catalog source)
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
  models.py                         # Pydantic models; QuestionDetail.attribution (honest company-or-source)
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
| Pre-indexed bank (1,509 Qs) | `search_question_bank` (TF-IDF) | Yes (verified) |
| Tavily web (69-domain allowlist) | `search_web_questions` | Best-effort from URL |
| GitHub interview repos | `search_github_questions` | No |
| `generate_interview_questions` | **BLOCKED** | — |

Attribution for output is computed by `attribution_label` (`src/models.py`) and surfaced via `QuestionDetail.attribution`: the **real company in UPPERCASE** when known, otherwise the placeholder **`NIAT`** — never a website name, fragment, or fabricated company. Garbage candidates are filtered upstream by Tavily's `_valid_company`.

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
- **Model comes from the run** (`llm_client.run_model(state)` ← `GenerationConfig.model`). The
  module-level model is a UI display default ONLY — using it for work made the picker a no-op for
  agent loops and let concurrent tabs retarget each other.
- **`/api/result` returns 409 while a run is in flight.** Do not reintroduce a blocking `thread.join`:
  Starlette's threadpool is bounded and a few polling tabs starved every other endpoint.
- **SSE events are retained per run** (`orchestrator.get_history`) so a reload replays the transcript,
  and a quiet stream heartbeats. A 120s silence between events is normal for `validate_relevance`.

## Web/UI notes

- `main.py` errors use `{"error": ...}`, not FastAPI's `{"detail": ...}` — the React client reads
  `body.error`. Blocking handlers stay sync so Starlette runs them in a threadpool.
- Emoji are not the icon system; `frontend/src/components/Icon.jsx` is. Progress renders
  `AgentTranscript` (grouped per agent, timed, collapsible) from structured SSE fields — not a flat log.
- `QualityPanel` separates SCORED metrics from reported-only ones and renders `report.critique`. Adding
  `self_relevance` back alongside the scored metrics undoes the point of the scoring work.
- Review is keyboard-first (`j/k`, `a`, `r`, `1-7`, `u`, `?`).

## Dormant: coding questions

Nothing assigns `state.coding_questions`. Generation is blocked and no retrieval path produces them,
so the CodingQuestion/CodeSnippet sheet tabs and every coding branch are unreachable. The tabs stay
because they're part of the LMS unit import format.

## Tests

`pytest tests/ -q` — 257 tests, no LLM or network required. Beyond unit coverage:
- `tests/test_pipeline_integration.py` drives the REAL pipeline with only the LLM boundary stubbed,
  including each outage above. This is the cheapest way to check a pipeline change.
- `tests/test_failure_paths.py` and `tests/test_audit_fixes.py` pin the silent-failure classes.
- `tests/test_data_integrity.py` checks the shipped `data/` files (missing reading material, bank
  form-garbage, implausible company attribution).
- A guard that `REJECT_REASONS` in `Review.jsx` matches `src/rejection_rules.py`.
