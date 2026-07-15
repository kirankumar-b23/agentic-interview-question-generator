# ⚠️ These specs describe an earlier/broader design — not the current code

The documents in this folder (`00`–`10`) were written for an **earlier, broader NxtMock
design**: a LangGraph + Neon-Postgres **MCQ-quiz-generation** app with a `backend/` package
layout, a 35-checkpoint rubric, and quiz product types (Classroom Quiz / MCQ Practice /
Module Quiz).

**They do NOT reflect the current implementation.** This repo is **Questor** — a 4-agent
(Understanding → Retrieval → Validation → Evaluation) **interview-question harvester** with a
`src/` layout, **SQLite** persistence (no LangGraph/Postgres), real-questions-only retrieval
(bank + Tavily + GitHub), and Google Sheets export.

For the accurate, current architecture see:

- [`/README.md`](../../README.md)
- [`/CLAUDE.md`](../../CLAUDE.md)
- [`/docs/architecture.html`](../../docs/architecture.html)

Kept for historical reference only.
