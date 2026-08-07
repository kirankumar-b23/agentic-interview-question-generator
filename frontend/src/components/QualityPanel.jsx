import { useState } from 'react'
import Icon from './Icon.jsx'

/**
 * The run's quality verdict, honestly presented.
 *
 * What this replaces mattered: the old bar showed a score, then every entry of `metric_scores` as an
 * identical chip. That put `self_relevance` — the mean of the LLM score used to *pick* the questions —
 * next to `predicted_accept` as an equal, which is exactly the impression the scoring work removed
 * from the composite. And `report.critique`, the list of plain-English warnings the pipeline computes
 * ("Predicted reviewer acceptance 62%", "Weak grounding", "the gate did not pass this set"), was never
 * rendered at all.
 *
 * So: the scored metrics are shown as the verdict's basis; the reported-only ones are separated and
 * labelled as not counting; and the notes are the most prominent thing after the score.
 */

// Metrics that make up the composite, in the order they're weighted.
const SCORED = [
  ['coverage_efficiency', 'Coverage efficiency', 'Did each question earn its place against a distinct interview topic'],
  ['predicted_accept', 'Predicted acceptance', 'Estimated from your own past accept/reject decisions'],
  ['session_grounding', 'Session grounding', 'Similarity to this session’s outcomes and reading material'],
  ['set_size', 'Set size', 'How close the set is to the requested count'],
]

// Reported for transparency but deliberately excluded from the score.
// `topic_coverage` sits here rather than in SCORED on purpose: it is bounded by SUPPLY, so with fewer
// real questions available than the session has interview topics it can never reach 1.0. Scoring it
// failed every thin set — one scored 0.227, its exact arithmetic maximum, and was marked a failure.
const REPORTED = [
  ['topic_coverage', 'Topic coverage', 'Share of the session’s interview topics examined — bounded by how many real questions exist'],
  ['self_relevance', 'Self-relevance', 'The selector’s own confidence — not evidence of quality'],
  ['difficulty_balance', 'Difficulty mix', 'Scored against bank labels that are ~95% "Medium"'],
  ['source_diversity', 'Source diversity', 'How many distinct sources contributed'],
]

const WEB_STATUS_WARN = {
  quota: 'Web search hit its usage limit — this set is bank-only (no fresh web questions).',
  auth: 'Web search unauthorized (bad or expired Tavily key) — this set is bank-only.',
  rate: 'Web search was rate-limited — this set is bank-only.',
  no_key: 'No Tavily key configured — web search skipped; this set is bank-only.',
  error: 'Web search failed — this set is bank-only.',
}

function band(value) {
  if (value >= 0.75) return 'good'
  if (value >= 0.5) return 'mid'
  return 'poor'
}

function Metric({ id, label, hint, value, muted }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className={`qm${muted ? ' qm-muted' : ''}`} title={hint}>
      <span className="qm-label">{label}</span>
      <span className={`qm-value qm-${muted ? 'muted-v' : band(value ?? 0)}`}>{pct}</span>
      <span className="qm-track" aria-hidden="true">
        <span className={`qm-fill qm-fill-${muted ? 'muted-v' : band(value ?? 0)}`}
              style={{ width: `${pct}%` }} />
      </span>
    </div>
  )
}

export default function QualityPanel({ report }) {
  const [showReported, setShowReported] = useState(false)
  if (!report) return null

  const score = Math.round((report.composite_score ?? 0) * 100)
  const passing = report.pass_fail === 'pass'
  const metrics = report.metric_scores || {}
  const notes = report.critique || []
  const flagged = report.flagged_questions || []
  const webWarn = WEB_STATUS_WARN[report.web_status]

  return (
    <section className="quality" aria-label="Quality report">
      <header className={`q-head ${passing ? 'q-head-pass' : 'q-head-fail'}`}>
        <span className="q-verdict">
          <Icon name={passing ? 'check' : 'alert'} size={15} />
          {passing ? 'Passed quality checks' : 'Did not pass quality checks'}
        </span>
        <span className="q-score" title="Composite of the scored metrics below">
          {score}<em>/100</em>
        </span>
        {report.loops_used > 0 && (
          <span className="q-loops">
            <Icon name="refresh" size={12} /> {report.loops_used} revision
            {report.loops_used > 1 ? 's' : ''}
          </span>
        )}
      </header>

      {/* Which condition decided the verdict. Without this the panel showed pass/fail and left the
          reviewer to guess whether a failure meant a bad set or a thin corpus. */}
      {(report.gate_checks || []).length > 0 && (
        <ul className="q-gate" aria-label="Gate conditions">
          {report.gate_checks.map((c) => (
            <li key={c.name} className={c.ok ? 'q-gate-ok' : 'q-gate-bad'}>
              <Icon name={c.ok ? 'check' : 'x'} size={11} />
              <span className="q-gate-name">{c.name}</span>
              <span className="q-gate-val">{c.value}</span>
              <span className="q-gate-bar">needs {c.bar}</span>
            </li>
          ))}
        </ul>
      )}

      {webWarn && (
        <p className="q-note q-note-warn">
          <Icon name="alert" size={14} />
          <span>{webWarn}{report.web_error ? ` (${report.web_error})` : ''}</span>
        </p>
      )}

      {/* The notes the pipeline computes and used to discard. Warnings first. */}
      {notes.length > 0 && (
        <ul className="q-notes">
          {notes.map((n, i) => {
            const warn = n.trimStart().startsWith('⚠')
            return (
              <li key={i} className={`q-note${warn ? ' q-note-warn' : ''}`}>
                <Icon name={warn ? 'alert' : 'info'} size={14} />
                <span>{n.replace(/^\s*⚠\s*/, '')}</span>
              </li>
            )
          })}
        </ul>
      )}

      {flagged.length > 0 && (
        <div className="q-flagged">
          <h4><Icon name="alert" size={13} /> Unresolved gate objections ({flagged.length})</h4>
          <ul>
            {flagged.map((f, i) => (
              <li key={i}>
                <span className="q-flag-kind">{f.issue}</span>
                {f.suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="q-metrics">
        {SCORED.map(([id, label, hint]) => (
          <Metric key={id} id={id} label={label} hint={hint} value={metrics[id]} />
        ))}
      </div>

      <button className="q-toggle" onClick={() => setShowReported((v) => !v)} aria-expanded={showReported}>
        <Icon name={showReported ? 'chevronDown' : 'chevronRight'} size={12} />
        Not counted toward the score ({REPORTED.length})
      </button>
      {showReported && (
        <div className="q-metrics q-metrics-muted">
          <p className="q-muted-why">
            These are reported for transparency only. <strong>Self-relevance</strong> is the selector
            grading its own picks, and <strong>difficulty mix</strong> is measured against bank labels
            that are almost entirely “Medium” — neither is evidence the set is good.
          </p>
          {REPORTED.map(([id, label, hint]) => (
            <Metric key={id} id={id} label={label} hint={hint} value={metrics[id]} muted />
          ))}
        </div>
      )}
    </section>
  )
}
