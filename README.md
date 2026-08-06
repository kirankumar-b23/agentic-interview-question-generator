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

See **`docs/architecture.html`** (open in a browser) for an interactive diagram.

---

## Project structure

```
main.py                    # FastAPI JSON API + serves the React build (uvicorn)
docs/architecture.html     # Standalone architecture visualization
src/
  pipeline.py              # AgentPipeline — orchestrates the 4 agents + quality gate
  agents/                  # Understanding / Retrieval / Validation / Evaluation agents
  agent.py                 # AgentState, PipelineResult, quality-gate critique
  tools.py                 # Tool schemas + implementations (generation disabled)
  session_understanding.py # Per-session reading material → SessionContext (+ KG fallback)
  question_bank.py         # Hybrid (embedding + keyword) retriever over the question banks
  human_agreement.py       # Predicted reviewer acceptance from past accept/reject decisions
  quality.py               # Form gate — is this a well-formed standalone question?
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
  interview_questions.json           # ~1,500 company-attributed questions (TF-IDF bank)
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
# optional overrides:
ENV=development                    # development | staging | production
LLM_MODEL=anthropic/claude-haiku-4-5
```

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
| `src/config.py` → `MIN_QUESTIONS` / `MAX_QUESTIONS` | Set size bounds (5–15) |
| `src/config.py` → `INTERVIEW_SOURCE_ALLOWLIST` | Tavily web domains |
| `src/config.py` → `INTERVIEW_GITHUB_REPOS` | GitHub source repos |

---

## Tech stack

- **Backend:** Flask (JSON API + SSE), SQLite persistence
- **Agents/LLM:** OpenRouter (Anthropic / OpenAI models) via the `openai` SDK, tool-use pipeline
- **Retrieval:** local sentence-transformers embeddings + scikit-learn TF-IDF (hybrid bank ranking), Tavily API (web); GitHub REST API when enabled
- **Knowledge graph:** networkx (KP prerequisites), Pydantic v2 models
- **Frontend:** React + Vite (react-router), tokenized light/dark design system
- **Export:** gspread + google-auth-oauthlib (Google Sheets, OAuth)

---

## Notes

- **Real questions only** — LLM question *generation* is intentionally disabled; niche topics with little web coverage may return fewer questions.
- **Google Sheets** requires OAuth on first use; `token.json`, `.env`, and `memory.db` are gitignored.
- A fresh clone must run `npm run build` before Flask can serve the UI.
