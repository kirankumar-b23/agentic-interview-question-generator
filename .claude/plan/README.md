# ⚠️ These plans describe an earlier/broader design — not the current code

The feature plans in this folder (`feature-01`…`feature-11`, the explainers, and the
`*.html` flow diagrams) target an **earlier, broader NxtMock design**: a LangGraph +
Neon-Postgres **MCQ-quiz-generation** app (`backend/` layout) with Classroom Quiz / MCQ
Practice / Module Quiz product types and a 35-checkpoint rubric.

**They do NOT reflect the current implementation.** This repo is **Questor** — a 4-agent
(Understanding → Retrieval → Validation → Evaluation) **interview-question harvester** with a
`src/` layout, **SQLite** persistence (no LangGraph/Postgres), real-questions-only retrieval
(bank + Tavily + GitHub), and Google Sheets export.

For the accurate, current architecture see:

- [`/README.md`](../../README.md)
- [`/CLAUDE.md`](../../CLAUDE.md)
- [`/docs/architecture.html`](../../docs/architecture.html)

Kept for historical reference only.
