import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import Icon from '../components/Icon.jsx'
import TopBar from '../components/TopBar.jsx'

/** Date over time on two lines — the date column was the widest thing in the table on one line. */
function DateCell({ ts }) {
  if (!ts) return <span className="muted">—</span>
  try {
    const d = new Date(ts.includes('T') ? ts : ts + 'Z')
    return (
      <>
        {d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
        <span className="hist-date-time">
          {d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
        </span>
      </>
    )
  } catch {
    return ts
  }
}

/**
 * The composite score as a meter plus the number.
 *
 * A coloured pill reads as a badge you were awarded. This is a measurement, and a low one is the
 * more meaningful reading (see the scoring notes: composite correlated 0.16 with reviewer approval
 * before the selector-independent rework), so it is drawn as a bar you can compare down the column.
 */
function ScoreBadge({ score }) {
  if (score == null) return <span className="muted">—</span>
  const pct = Math.round(score * 100)
  const cls = pct >= 70 ? 'score-pass' : pct >= 50 ? 'score-mid' : 'score-fail'
  return (
    <span className={`hist-score ${cls}`} title={`Composite score ${pct}%`}>
      <span className="hist-meter">
        <span className="hist-meter-fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </span>
      <span className="hist-score-num">{pct}%</span>
    </span>
  )
}

/**
 * Per-run cost of work, under the score.
 *
 * The Tavily count used to be appended here and did not fit the column, so the line clipped to
 * "20 LLM · 63.1K tok ·" with a dangling separator. Search count is a totals-strip number, not a
 * per-row triage signal, so it moves to the `title` rather than being truncated in place.
 */
function UsageCell({ usage }) {
  if (!usage || !usage.llm_calls) return null
  const tokens = ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0))
  const tokenStr = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}K` : tokens
  const full = `${usage.llm_calls} LLM calls · ${tokenStr} tokens`
    + (usage.tavily_calls > 0 ? ` · ${usage.tavily_calls} Tavily searches` : '')
  return (
    <span className="hist-usage" title={full}>
      {usage.llm_calls} LLM · {tokenStr} tok
    </span>
  )
}

function fmtTokens(n) {
  if (!n) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return String(n)
}

export default function History() {
  const navigate = useNavigate()
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [usage, setUsage] = useState(null)

  useEffect(() => {
    api.getHistory()
      .then(d => setRuns(d.runs || []))
      .catch(() => {})
      .finally(() => setLoading(false))
    api.getUsage().then(setUsage).catch(() => {})
  }, [])

  const approvedCount = runs.filter(r => r.approved).length

  return (
    <>
      <TopBar
        title="Run history"
        sub={`${runs.length} run${runs.length !== 1 ? 's' : ''}${approvedCount > 0 ? ` · ${approvedCount} approved` : ''}`}
        actions={
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>
            <Icon name="arrowLeft" size={13} /> Home
          </button>
        }
      />

      <div className="page-content">
        {/* Overall usage summary */}
        {usage?.totals && (
          <div className="usage-summary">
            <div className="usage-tile"><span className="usage-num">{usage.totals.runs}</span><span className="usage-lbl">Runs</span></div>
            <div className="usage-tile"><span className="usage-num">{usage.totals.llm_calls}</span><span className="usage-lbl">LLM calls</span></div>
            <div className="usage-tile"><span className="usage-num">{fmtTokens(usage.totals.tokens)}</span><span className="usage-lbl">Tokens</span></div>
            {/* Short labels on purpose: at 1280px each tile is ~155px wide, so "Tavily searches" and
                "OpenRouter left · $147.76 used" wrapped to two and three lines and left the strip
                ragged. The spend moves to its own sub-line rather than being crammed in. */}
            <div className="usage-tile"><span className="usage-num">{usage.totals.tavily_calls}</span><span className="usage-lbl">Tavily</span></div>
            <div className="usage-tile"><span className="usage-num">${usage.totals.est_cost.toFixed(2)}</span><span className="usage-lbl">Est. cost</span></div>
            <div className="usage-tile usage-tile-or">
              <span className="usage-num">{usage.openrouter?.remaining != null ? `$${usage.openrouter.remaining.toFixed(2)}` : '—'}</span>
              <span className="usage-lbl">Credit left</span>
              {usage.openrouter?.used != null && (
                <span className="usage-sub">${usage.openrouter.used.toFixed(2)} used</span>
              )}
            </div>
          </div>
        )}

        <section className="card card-table">
          {loading ? (
            <div className="hist-loading" aria-busy="true" aria-label="Loading history">
              {Array.from({ length: 5 }).map((_, i) => <div key={i} className="skeleton skeleton-row" />)}
            </div>
          ) : runs.length === 0 ? (
            <div className="hist-empty">
              <Icon name="history" size={22} />
              <span>No runs yet — generate your first question set from the sidebar.</span>
            </div>
          ) : (
            <div className="hist-table-wrap">
              <table className="hist-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Run</th>
                    <th className="num-col">Qs</th>
                    <th>Score</th>
                    <th className="num-col">Cost</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => (
                    // The row is the control. It carried `cursor: pointer` with no handler, so it
                    // advertised itself as clickable and wasn't — while the real action lived in a
                    // 9th column that sat off-screen at 1280px. The keyboard path is explicit
                    // below, since a <tr> is not focusable.
                    <tr
                      key={run.run_id}
                      className="hist-row"
                      onClick={() => navigate(`/review/${run.run_id}`)}
                      title={`${run.topic || run.session_name} — ${run.created_at || ''}`}
                    >
                      <td className="hist-date"><DateCell ts={run.created_at} /></td>
                      <td className="hist-run">
                        <span className="hist-run-topic" title={run.topic || ''}>{run.topic || '—'}</span>
                        <span className="hist-run-sessions" title={run.session_name}>{run.session_name}</span>
                      </td>
                      <td className="hist-count">{run.question_count ?? '—'}</td>
                      <td className="hist-score-cell">
                        <ScoreBadge score={run.composite_score} />
                        <UsageCell usage={run.api_usage} />
                      </td>
                      <td className="hist-count hist-cost">
                        {run.cost != null ? `$${run.cost.toFixed(4)}` : '—'}
                      </td>
                      <td className="hist-go">
                        <span className={`hist-status ${run.approved ? 'hist-approved' : 'hist-pending'}`}>
                          {run.approved ? 'Approved' : 'In memory'}
                        </span>
                        {/* Keeps the run reachable by keyboard: a <tr> takes no focus, so the
                            visible affordance has to be a real button. */}
                        <button
                          className="hist-open"
                          onClick={(e) => { e.stopPropagation(); navigate(`/review/${run.run_id}`) }}
                          aria-label={`Open run for ${run.session_name}`}
                        >
                          <Icon name="arrowRight" size={13} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
