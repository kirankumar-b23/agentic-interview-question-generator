# Questor

**Questor** is an agentic workflow that autonomously *quests* for real, company-attributed
interview questions for a course topic, validates them against what the session actually
teaches, and hands a curated set to a human for one-click export into the **NxtMock**
interview portal (Google Sheets).

You pick a topic; four cooperating agents read the session material, retrieve real
questions from a curated bank + the web, filter them for relevance, and assemble a
review-ready set. A human approves; Questor exports.

---

## What it does

- **Understands the session** from its own reading material → learning outcomes + Knowledge Points.
- **Retrieves real questions** (no LLM-invented questions) from two sources, in order:
  1. a pre-indexed **question bank**, ranked by a **hybrid** of semantic similarity and keyword match,
  2. **Tavily web search** over an allowlist of interview-question domains (Glassdoor, AmbitionBox,
     Exponent, …) with best-effort company attribution.

  A GitHub interview-repo source exists but is **off by default** (`GITHUB_ENABLED=0`): those repos are
  general ML/DS content with no company attribution.
- **Scores every candidate against the session** — its curated outcomes, interview topics and its own
  reading material — then filters for relevance and de-duplicates (including rewordings).
- **Evaluates** difficulty balance & outcome coverage, then runs a **quality gate** (structural checks
  in code, plus an LLM pass for off-domain drift and duplicates; up to 2 revision rounds).
- **Human review** in a React UI → **export to Google Sheets** in NxtMock portal format, or **reject → regenerate**.

Attribution is honest rather than flattering: a verified company is shown in UPPERCASE, otherwise the
**source site** it was found on (e.g. `GeeksforGeeks`), and only `NIAT` when neither is known. The
Sheets export writes a company **only** when one is verified — a source site is provenance, not an
employer.

---

## How it works

```
Topic (→ its sessions)  +  model  +  question count
        │
        ▼
┌──────────────── Agent pipeline (src/pipeline.py) ────────────────┐
│  1 Understanding  understand_session → outcomes + KPs             │
│  2 Retrieval      question_bank → Tavily web  (GitHub: opt-in)   │
│  3 Validation     session-fit → relevance → deduplicate          │
│  4 Evaluation     difficulty/coverage checks → select final set  │
│        │                                                          │
│        ▼  Quality Gate (LLM critique, ≤2 revisions)              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
Human review (React)  ──approve──▶  Google Sheets (NxtMock)
        │
        └──reject──▶  regenerate (same topic, learned rules updated)
```

Runs are **persisted to SQLite**, so Review and re-export survive server restarts and appear
in History. The **model** and **light/dark theme** are selectable at runtime from the sidebar.

**Multiple courses** are supported: the built-in Gen AI course ships by default, and you can
**add or import** more courses (new topics/sessions) from the **Add Course** page — each course
drives its own topic list in the sidebar. A **preview gate** lets you sanity-check the resolved
session/outcomes before committing a full run, and per-run **usage** (LLM/Tavily calls, tokens,
cost estimate) is tracked and shown in History.

See **`docs/pipeline-walkthrough.html`** for a step-by-step walkthrough of a run — one step at a
time, each with its live SSE step id, the code that runs it, what goes wrong there and how to check
it. Open it directly in a browser; no build step. **`CLAUDE.md`** has the load-bearing failure paths
and the scoring rationale.

---

## Project structure

```
main.py                    # FastAPI JSON API + serves the React build (uvicorn)
docs/pipeline-walkthrough.html  # step-by-step walkthrough of a run (open in a browser)
docs/deep-research-report.md    # source strategy behind the Tavily allowlist
src/
  pipeline.py              # AgentPipeline — orchestrates the 4 agents + quality gate
  agents/                  # Understanding / Retrieval / Validation / Evaluation agents
  agent.py                 # AgentState, PipelineResult, quality-gate critique
  tools.py                 # Tool schemas + implementations (generation disabled)
  session_understanding.py # Per-session reading material → SessionContext (+ KG fallback)
  question_bank.py         # Hybrid (embedding + keyword) retriever over the question banks
  human_agreement.py       # Predicted reviewer acceptance from past accept/reject decisions
  embeddings.py            # Local sentence-transformers (MiniLM); falls back to TF-IDF
  quality.py               # Form gate — is this a well-formed standalone question?
  interview_format.py      # Is it answerable OUT LOUD? (skips hands-on "write a program" prompts)
  assessment_items.py      # MCQ / glued-answer shapes — a data assertion, not a form gate
  session_types.py         # theory_heavy | code_heavy | mixed from a session NAME
  rejection_rules.py       # Rejection-reason key → the rule the next run's judge reads
  sources/                 # tavily_search.py (web) + github_repo.py
  llm_client.py            # OpenRouter client + runtime model + credit balance
  sheets_writer.py         # Google Sheets export (OAuth)
  data_loader.py           # Loads data files; topic↔session mapping
  memory.py                # SQLite: run persistence, history, learned rules
  models.py                # Pydantic models
  config.py                # Models list, paths, constraints
  orchestrator.py          # Per-run SSE fan-out with replay history + heartbeats
frontend/                  # React SPA (Vite). Pages: SessionSelector, AddCourse, Progress, Review, History
data/
  interview_questions.json           # 1,447 company-attributed questions (general SWE/Python)
  genai_question_bank.json           # 1,381 curated GenAI questions
  knowledge_graph.json               # KPs + sessions + prerequisites
  course_structure.json              # Topic → sessions (drives the UI + topic labels)
  reading_materials/session_map.json # Per-session reading material (built)
  curriculum/*.json                  # KP-mapped curriculum (data-prep input)
scripts/
  prepare_data.py                    # One-time: CSV → JSON, knowledge graph, eval sets
  build_session_reading_material.py  # Build reading_materials/session_map.json
  build_knowledge_graph.py, build_eval_sets.py, auth_sheets.py
```

---

## Setup

### Prerequisites
- Python 3.12+, Node 18+
- **OpenRouter** API key (LLM calls) and **Tavily** API key (web search)
- Google OAuth client (for Sheets export); GitHub token optional (raises API rate limit)

### 1. Environment (`.env` in repo root)
```dotenv
OPENROUTER_API_KEY=sk-or-...
TAVILY_API_KEY=tvly-...
GITHUB_TOKEN=github_pat_...        # optional
Client_ID=...apps.googleusercontent.com   # Google OAuth (Sheets)
Client_Secret=...
COMPOSIO_API_KEY=...               # optional; referenced by .mcp.json as ${COMPOSIO_API_KEY}
# optional overrides:
ENV=development                    # development | staging | production
LLM_MODEL=anthropic/claude-haiku-4-5
```

**The OpenRouter key must keep its `sk-` prefix.** A truncated `or-...` value (73 chars → 70) is the
single most common way a run dies here: every stage returns `401 Missing Authentication header`, and
because each phase is fail-open the run completes and reports "few questions were available" rather than
an auth error. It has happened twice. If a run yields nothing, check the key length first.

`COMPOSIO_API_KEY` lives in `.env` so it is not inline in `.mcp.json`, but **Claude Code expands
`${VAR}` from its own process environment and does not read `.env`** — so it also needs exporting from
your shell profile for the MCP server to authenticate.

### 2. Install & build
```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```
The React build lands in `frontend/dist/`, which FastAPI serves. **Rebuild after any frontend change.**

### 3. (Optional) Rebuild data
Prebuilt data files are committed, so you can skip this. To regenerate from source:
```bash
python3 scripts/prepare_data.py                 # needs data/raw/*.csv
python3 scripts/build_session_reading_material.py
```

### 4. Run
```bash
uvicorn main:app --port 5000              # → http://127.0.0.1:5000
uvicorn main:app --port 5000 --reload     # development (auto-restart on edit)
python3 main.py                           # equivalent to the first form
```
Interactive API docs are served at `/docs`.
**Authorize Google Sheets once before your first export:**
```bash
python3 scripts/auth_sheets.py     # opens a browser, caches token.json
```
The server will not open a browser itself — inside a request that would block a worker thread
indefinitely waiting for consent on the *server* machine.

---

## Reviewing a generated set

The review screen is keyboard-first, because an uncapped set can be 40+ questions:

| Key | Action |
|---|---|
| `j` / `k` | move the cursor |
| `a` / `r` | accept / reject |
| `1`–`7` | give a rejection reason (after `r`) — this is what teaches the pipeline |
| `u` | undo the decision on this question |
| `shift`+`A` | accept everything at or above the high-confidence fit threshold |
| `esc` | clear filters |
| `?` | show the shortcut list |

Questions are ranked by **session fit** — similarity to that session's learning outcomes and reading
material — so working top-down and stopping where fit falls off is the intended flow. Filters isolate a
slice (fit band, difficulty, source, real-company vs source-labelled attribution).

Rejection reasons matter: each one maps to a rule the relevance judge reads on the next run
(`src/rejection_rules.py`). Rejecting without a reason still suppresses the question, but teaches
nothing.

## Tests, and the yield harness

```bash
pytest tests/ -q                 # 573 tests. Stop the server first — it holds the memory.db lock
python scripts/yield_report.py   # how many questions actually reach the reviewer
```

**The suite costs nothing to run, and that is enforced, not hoped for.** `tests/conftest.py` blocks
outbound sockets and *records* every attempt, failing the test that made one. Recording matters because
the LLM/Tavily call sites are deliberately fail-open: a version of this guard that only raised let the
whole suite pass while reporting zero leaks. Real leaks had been spending OpenRouter credit and exhausted
a Tavily plan. If a test genuinely needs the network, mark it `@pytest.mark.allow_network`.

Before changing any threshold, filter or gate, run `scripts/yield_report.py` and compare. **Median final
set size is the number to watch** — a change that improves a single metric while lowering it is a
regression, which is how a 5-question topic once became a 3.

## Evaluating a change

`eval/run_eval.py` runs the pipeline over sessions from `eval/eval_sets.json` and scores them
**per session type**, because the two types are not comparable:

```bash
python eval/run_eval.py --n 3                    # stratified across types
python eval/run_eval.py --type code_heavy --n 3  # one segment
python eval/run_eval.py --all
```

A code-heavy session is scored against banks that hold almost no implementation questions, so it
scores lower on coverage and grounding for reasons about **source coverage**, not question quality.
It therefore has its own bars (`config.EVAL_THRESHOLDS_BY_TYPE`) — one global bar either failed every
code session or set theory's too low. As implementation questions reach the banks, raise the
code-heavy bars; that rise is the signal the gap is closing.

Predicted reviewer acceptance is scored only against decisions from the **same** session type, and
reports `n/a` when that type has none rather than borrowing another type's taste. Each run prints a
per-type label inventory so you can see which types are measurable.

Sessions with no reading material are skipped and counted (they'd otherwise exercise the fallback
resolution path and be reported as a curriculum result).

## Usage

1. In the sidebar pick a **Course**, a **Topic**, a **Model**, and the **max questions** (default 7).
   (Use **Add Course** to add or import more courses.)
2. Click **Generate** and watch the live pipeline.
3. On the **Review** screen, accept/reject questions.
4. **Export to Sheets** (creates a NxtMock-formatted spreadsheet) or **Reject & Regenerate**.

**Runtime controls:** model dropdown + light/dark toggle (both remembered in the browser);
OpenRouter credit balance is shown in the sidebar footer.

---

## Configuration

| Where | Setting |
|-------|---------|
| `src/config.py` → `MODEL_OPTIONS` | Models offered in the UI dropdown |
| `src/config.py` → `MODEL_CONFIG` / `ENV` | Default model per environment |
| `src/config.py` → `MIN_QUESTIONS` / `MAX_QUESTIONS` | Set size bounds (5–60; the UI slider's range) |
| `src/config.py` → `INTERVIEW_SOURCE_ALLOWLIST` | Tavily web domains |
| `src/config.py` → `INTERVIEW_GITHUB_REPOS` | GitHub source repos |
| `CONVERSATIONAL_ONLY` (env, default on) | Skip hands-on prompts ("Write a Python program to…") that a candidate cannot answer out loud. Set `0` to include them |
| `OPEN_WEB_ENABLED` / `OPEN_WEB_TRIGGER_RATIO` (env) | The last-resort unrestricted web search, and how short a set must be before it engages (default 0.6 × requested) |
| `RELEVANCE_THRESHOLD` / `RELEVANCE_FLOOR` (env) | Keep-bar and the absolute floor for the back-fill. The floor is never crossed: a thin on-topic pool returns fewer questions, not filler |

---

## Tech stack

- **Backend:** FastAPI + uvicorn (JSON API + SSE), SQLite persistence
- **Agents/LLM:** OpenRouter (Anthropic / OpenAI models) via the `openai` SDK, tool-use pipeline
- **Retrieval:** local sentence-transformers embeddings + scikit-learn TF-IDF (hybrid bank ranking), Tavily API (web); GitHub REST API when enabled
- **Knowledge graph:** networkx (KP prerequisites), Pydantic v2 models
- **Frontend:** React + Vite (react-router), tokenized light/dark design system
- **Export:** gspread + google-auth-oauthlib (Google Sheets, OAuth)

---

## Notes

- **Real questions only** — LLM question *generation* is intentionally disabled; niche topics with little web coverage may return fewer questions.
- **Google Sheets** requires OAuth on first use; `token.json`, `.env`, and `memory.db` are gitignored.
- A fresh clone must run `npm run build` before FastAPI can serve the UI (`frontend/dist/` is gitignored).
