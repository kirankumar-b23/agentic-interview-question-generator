import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import Icon from '../components/Icon.jsx'
import TopBar from '../components/TopBar.jsx'

function formatDate(ts) {
  if (!ts) return '—'
  try {
    const d = new Date(ts.includes('T') ? ts : ts + 'Z')
    return d.toLocaleDateString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return ts
  }
}

function ScoreBadge({ score }) {
  if (score == null) return <span className="muted">—</span>
  const pct = Math.round(score * 100)
  const cls = pct >= 70 ? 'score-pass' : pct >= 50 ? 'score-mid' : 'score-fail'
  return <span className={`hist-score ${cls}`}>{pct}%</span>
}

function UsageCell({ usage }) {
  if (!usage || !usage.llm_calls) return <span className="muted">—</span>
  const tokens = ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0))
  const tokenStr = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}K` : tokens
  return (
    <span className="hist-usage">
      {usage.llm_calls} LLM · {tokenStr} tok
      {usage.tavily_calls > 0 && ` · ${usage.tavily_calls} Tavily`}
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
            <div className="usage-tile"><span className="usage-num">{usage.totals.tavily_calls}</span><span className="usage-lbl">Tavily searches</span></div>
            <div className="usage-tile"><span className="usage-num">${usage.totals.est_cost.toFixed(2)}</span><span className="usage-lbl">Est. cost</span></div>
            <div className="usage-tile usage-tile-or">
              <span className="usage-num">{usage.openrouter?.remaining != null ? `$${usage.openrouter.remaining.toFixed(2)}` : '—'}</span>
              <span className="usage-lbl">OpenRouter left{usage.openrouter?.used != null ? ` · $${usage.openrouter.used.toFixed(2)} used` : ''}</span>
            </div>
          </div>
        )}

        <section className="card">
          {loading ? (
            <div aria-busy="true" aria-label="Loading history">
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
                    <th>Topic</th>
                    <th>Sessions</th>
                    <th className="num-col">Questions</th>
                    <th>Score</th>
                    <th>API Usage</th>
                    <th className="num-col">Cost</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => (
                    <tr key={run.run_id} className="hist-row">
                      <td className="hist-date">{formatDate(run.created_at)}</td>
                      <td className="hist-session" title={run.topic || ''}>{run.topic || '—'}</td>
                      <td className="hist-session" title={run.session_name}>{run.session_name}</td>
                      <td className="hist-count">{run.question_count ?? '—'}</td>
                      <td><ScoreBadge score={run.composite_score} /></td>
                      <td><UsageCell usage={run.api_usage} /></td>
                      <td className="hist-count hist-cost">
                        {run.cost != null ? `$${run.cost.toFixed(4)}` : '—'}
                      </td>
                      <td>
                        <span className={`hist-status ${run.approved ? 'hist-approved' : 'hist-pending'}`}>
                          {run.approved ? 'Approved' : 'In Memory'}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => navigate(`/review/${run.run_id}`)}
                          aria-label={`View run for ${run.session_name}`}
                        >
                          View <Icon name="arrowRight" size={12} />
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
