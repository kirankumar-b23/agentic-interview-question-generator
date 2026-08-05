import { Fragment, useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import PipelineStepper from '../components/PipelineStepper.jsx'

const WEB_STATUS_WARN = {
  quota: 'Web search hit its usage limit — this set is bank-only (no fresh web questions).',
  auth: 'Web search unauthorized (bad/expired Tavily key) — this set is bank-only.',
  rate: 'Web search was rate-limited — this set is bank-only.',
  no_key: 'No Tavily key configured — web search skipped; this set is bank-only.',
  error: 'Web search failed — this set is bank-only.',
}

function QualityBar({ report }) {
  if (!report) return null
  const score = Math.round(report.composite_score * 100)
  const isPassing = report.pass_fail === 'pass'
  const webWarn = WEB_STATUS_WARN[report.web_status]
  return (
    <div>
      {webWarn && (
        <div className="web-warning-banner" style={{
          background: '#fff4e5', border: '1px solid #ffb74d', color: '#8a5200',
          padding: '8px 12px', borderRadius: 6, marginBottom: 8, fontSize: 13,
        }}>
          ⚠ {webWarn}{report.web_error ? ` (${report.web_error})` : ''}
        </div>
      )}
      <div className={`quality-bar ${isPassing ? 'qb-pass' : 'qb-fail'}`}>
        <span className="qb-badge">{isPassing ? '✅ Pass' : '⚠️ Below threshold'}</span>
        <span className="qb-score">Score: {score}/100</span>
        <div className="qb-metrics">
          {Object.entries(report.metric_scores || {}).map(([k, v]) => (
            <span key={k} className="qb-metric">
              {k.replace(/_/g, ' ')}: {Math.round(v * 100)}
            </span>
          ))}
        </div>
        {report.loops_used > 0 && (
          <span className="qb-loops">{report.loops_used} revision round(s)</span>
        )}
      </div>
    </div>
  )
}

// One-click rejection reasons. These exist because the free-text reason box was effectively never
// filled in — 68 rejections produced zero learned rules — so the system never learned anything from
// being told "no". Each `key` maps to a canonical rule server-side (see app.py REJECTION_RULES), so
// picking a chip is enough to teach the pipeline; no typing and no LLM call required.
const REJECT_REASONS = [
  { key: 'off_topic',    label: 'Off-topic' },
  { key: 'too_generic',  label: 'Too generic' },
  { key: 'not_grounded', label: 'Not in this session' },
  { key: 'experience',   label: 'Asks about experience' },
  { key: 'not_question', label: 'Not a real question' },
  { key: 'duplicate',    label: 'Duplicate' },
  { key: 'wrong_level',  label: 'Wrong difficulty' },
]

function CompactQuestion({
  id, content, title, difficulty,
  company, role, topic, subTopic, language, source, sourceUrl,
  snippet, decision, onDecide, index, fit, reason,
}) {
  const [open, setOpen] = useState(false)
  const isCoding = !!title
  const diff = difficulty || 'Medium'
  const diffClass = diff === 'Easy' ? 'd-easy' : diff === 'Hard' ? 'd-hard' : 'd-medium'

  return (
    <div className={`cq-row${decision === 'accepted' ? ' cq-row-accepted' : decision === 'rejected' ? ' cq-row-rejected' : ''}`}>
      <div className="cq-main" onClick={() => setOpen(o => !o)}>
        <span className="cq-num">Q{index + 1}</span>
        <span className="cq-text">{isCoding ? title : content}</span>
        <div className="cq-tags">
          {company && <span className="cq-company" title={company}>{company}</span>}
          {typeof fit === 'number' && (
            <span
              className="cq-fit"
              title="Session fit — cosine similarity to this session's learning outcomes and reading material"
            >{fit.toFixed(2)}</span>
          )}
          <span className={`cq-diff ${diffClass}`}>{diff}</span>
        </div>
        <div className="cq-btns" onClick={e => e.stopPropagation()}>
          <button
            className={`cq-btn cq-accept${decision === 'accepted' ? ' active' : ''}`}
            onClick={() => onDecide(id, 'accepted')}
          >✓</button>
          <button
            className={`cq-btn cq-reject${decision === 'rejected' ? ' active' : ''}`}
            onClick={() => onDecide(id, 'rejected')}
          >✕</button>
        </div>
      </div>

      {/* Why it was rejected. One click, and the pipeline learns a rule from it. */}
      {decision === 'rejected' && (
        <div className="cq-reasons" onClick={e => e.stopPropagation()}>
          <span className="cq-reasons-label">Why?</span>
          {REJECT_REASONS.map(r => (
            <button
              key={r.key}
              className={`cq-reason-chip${reason === r.key ? ' active' : ''}`}
              onClick={() => onDecide(id, 'rejected', r.key)}
            >{r.label}</button>
          ))}
        </div>
      )}

      {open && (
        <div className="cq-detail">
          {isCoding && content && (
            <p style={{ fontSize: '0.82rem', color: '#c9d1d9', marginBottom: '0.5rem' }}>{content}</p>
          )}
          {snippet?.code_content && (
            <pre className="q-code-pre">{snippet.code_content}</pre>
          )}
          {sourceUrl && (
            <a
              href={sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="cq-resource-link"
              onClick={e => e.stopPropagation()}
            >
              ↗ Verify source
            </a>
          )}
          <div className="cq-meta-tags">
            {company  && <span className="cq-tag cq-tag-company">{company}</span>}
            {role     && <span className="cq-tag cq-tag-role">{role}</span>}
            {topic    && <span className="cq-tag cq-tag-topic">{topic}</span>}
            {subTopic && <span className="cq-tag cq-tag-subtopic">{subTopic}</span>}
            {language && <span className="cq-tag cq-tag-lang">{language}</span>}
            {source   && <span className="cq-tag cq-tag-source">{source}</span>}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Review() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [decisions, setDecisions] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)
  const [sheetUrl, setSheetUrl] = useState(null)
  const [sheetError, setSheetError] = useState(null)

  useEffect(() => {
    api.getResult(runId)
      .then(d => { setResult(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [runId])

  function onDecide(qid, status, reason) {
    setDecisions(cur => {
      // Re-clicking the same reason clears it, and switching to accepted drops any stale reason.
      if (status !== 'rejected') return { ...cur, [qid]: { status } }
      const prev = cur[qid]?.reason
      return { ...cur, [qid]: { status, reason: reason === prev ? undefined : (reason ?? prev) } }
    })
  }

  async function handleApprove() {
    const acceptedIds = []
    const rejectedFeedback = {}
    const allIds = [
      ...(result.output?.question_details || []).map(q => q.question_id),
      ...(result.output?.coding_questions || []).map(q => q.id),
    ]
    for (const id of allIds) {
      const d = decisions[id]
      if (!d || d.status === 'accepted') acceptedIds.push(id)
      // Send the reason on approve too — a question dropped from an otherwise-approved set is just
      // as informative as one from a fully rejected set, and used to be discarded here.
      else rejectedFeedback[id] = d.reason || ''
    }
    setSubmitting(true)
    try {
      const resp = await api.approve(runId, acceptedIds, rejectedFeedback, 'approve')
      if (resp.sheet_url) setSheetUrl(resp.sheet_url)
      if (resp.sheet_error) setSheetError(resp.sheet_error)
      setDone(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  async function handleReject() {
    // Send the per-question decisions so the backend keeps ACCEPTED questions, suppresses the
    // REJECTED ones (persisted per session), and refills the freed slots with new distinct questions.
    const acceptedIds = []
    const rejectedFeedback = {}
    const allIds = [
      ...(result.output?.question_details || []).map(q => q.question_id),
      ...(result.output?.coding_questions || []).map(q => q.id),
    ]
    for (const id of allIds) {
      if (decisions[id]?.status === 'rejected') rejectedFeedback[id] = decisions[id]?.reason || ''
      else acceptedIds.push(id)
    }
    setSubmitting(true)
    try {
      const { run_id } = await api.approve(runId, acceptedIds, rejectedFeedback, 'reject')
      navigate(`/progress/${run_id}`)
    } catch (e) {
      setError(e.message)
      setSubmitting(false)
    }
  }

  // TESTING: preview mode — send the picked set through the quality gate
  async function handleProceed() {
    setSubmitting(true)
    try {
      await api.proceed(runId)
      navigate(`/progress/${runId}`)
    } catch (e) {
      setError(e.message)
      setSubmitting(false)
    }
  }

  if (loading) return (
    <>
      <header className="topbar">
        <div className="topbar-title-group">
          <span className="topbar-title">Review Questions</span>
        </div>
        <PipelineStepper completedUntil="gate" activeStage="review" />
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>← History</button>
      </header>
      <div className="page-content"><p className="muted loading">Loading results…</p></div>
    </>
  )

  if (error && !result) return (
    <>
      <header className="topbar">
        <div className="topbar-title-group"><span className="topbar-title">Review Questions</span></div>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>← History</button>
      </header>
      <div className="page-content"><div className="alert alert-error">{error}</div></div>
    </>
  )

  if (!result) return null

  // Rank by session fit (highest first) so the reviewer works top-down and can stop where fit
  // falls off — the set size is deliberately uncapped, so ordering is what keeps review tractable.
  // Questions without a fit score (embeddings unavailable) keep their original order at the end.
  const fitOf = q => (typeof q.session_fit === 'number' ? q.session_fit : -1)
  const questions = [...(result.output?.question_details || [])]
    .sort((a, b) => fitOf(b) - fitOf(a))
  const fitHigh = result.thresholds?.session_fit_high ?? 0.35
  // Index of the first question below the high-confidence bar → where the divider goes.
  const firstLowFit = questions.findIndex(q => fitOf(q) >= 0 && fitOf(q) < fitHigh)
  const codingQs = result.output?.coding_questions || []
  const snippets = result.output?.code_snippets || []
  const snippetMap = Object.fromEntries(snippets.map(s => [s.code_id, s]))
  const total = questions.length + codingQs.length
  const rejectedCount = Object.values(decisions).filter(d => d.status === 'rejected').length
  const approvedCount = total - rejectedCount

  if (done) {
    return (
      <>
        <header className="topbar">
          <div className="topbar-title-group"><span className="topbar-title">Export Complete</span></div>
          <PipelineStepper completedUntil="export" />
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>← History</button>
        </header>
        <div className="page-content">
          <div className="done-banner">
            <h1>Approved!</h1>
            {sheetUrl ? (
              <>
                <p>{approvedCount} question{approvedCount !== 1 ? 's' : ''} exported to Google Sheets.</p>
                <a href={sheetUrl} target="_blank" rel="noopener noreferrer" className="btn btn-primary" style={{ marginBottom: '0.75rem', display: 'inline-block' }}>
                  Open Google Sheet
                </a>
              </>
            ) : sheetError ? (
              <>
                <p>{approvedCount} question{approvedCount !== 1 ? 's' : ''} approved (saved locally).</p>
                <p style={{ color: '#f85149', fontSize: '0.85rem' }}>Sheets export failed: {sheetError}</p>
              </>
            ) : (
              <p>{approvedCount} question{approvedCount !== 1 ? 's' : ''} approved and saved.</p>
            )}
            <button className="btn btn-primary" onClick={() => navigate('/')}>Generate more</button>
          </div>
        </div>
      </>
    )
  }

  const sessionName = result.context?.session_name || 'Session'
  const sessionType = result.context?.session_type
  const displayName = result.topic || sessionName
  const awaitingGate = !!result.awaiting_gate   // TESTING: preview mode

  return (
    <>
      <header className="topbar">
        <div className="topbar-title-group">
          <span className="topbar-title">{awaitingGate ? 'Preview — Picked Questions' : 'Review Questions'}</span>
          <span className="topbar-sub" title={sessionName}>{displayName}{sessionType ? ` · ${sessionType}` : ''}</span>
        </div>
        <PipelineStepper completedUntil="gate" activeStage="review" />
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>← History</button>
      </header>

      {/* Action banner */}
      <div className="action-banner" style={awaitingGate ? { borderLeftColor: 'var(--warn)' } : undefined}>
        <div className="ab-text">
          <span className="ab-title">
            {awaitingGate
              ? '🧪 Testing preview — verify the picked questions (not yet quality-checked)'
              : 'Action needed — Review before publishing to portal'}
          </span>
          <span className="ab-sub">
            {total} questions for <strong style={{ color: 'var(--text)' }} title={sessionName}>{displayName}</strong>
            {' · '}{questions.length} theory · {codingQs.length} coding
            {!awaitingGate && result.report && ` · Quality ${Math.round(result.report.composite_score * 100)}/100`}
            {!awaitingGate && result.report?.loops_used > 0 && ` · ${result.report.loops_used} revision(s)`}
          </span>
        </div>
        <div className="ab-btns">
          {awaitingGate ? (
            <button className="btn btn-primary" disabled={submitting} onClick={handleProceed}>
              {submitting ? 'Submitting…' : 'Proceed to Quality Gate ▸'}
            </button>
          ) : (
            <>
              <button className="btn btn-reject-all"
                      disabled={submitting || rejectedCount === 0}
                      title={rejectedCount === 0 ? 'Mark one or more questions as rejected first' : ''}
                      onClick={handleReject}>
                {rejectedCount > 0
                  ? `↺ Replace ${rejectedCount} rejected & regenerate`
                  : '↺ Replace rejected & regenerate'}
              </button>
              <button className="btn btn-primary" disabled={submitting} onClick={handleApprove}>
                {submitting ? 'Exporting…' : `↑ Export to Sheets (${approvedCount})`}
              </button>
            </>
          )}
        </div>
      </div>

      {error && <div className="alert alert-error" style={{ margin: '0 1.25rem' }}>{error}</div>}

      {/* Quality bar */}
      <div style={{ padding: '0.75rem 1.25rem 0' }}>
        <QualityBar report={result.report} />
      </div>

      {/* Learning outcomes */}
      {result.context?.learning_outcomes?.length > 0 && (
        <details className="card outcomes-card" style={{ margin: '0.75rem 1.25rem 0' }}>
          <summary>
            <strong>Learning Outcomes</strong> ({result.context.learning_outcomes.length})
          </summary>
          <ul className="outcome-list">
            {result.context.learning_outcomes.map((o, i) => <li key={i}>{o}</li>)}
          </ul>
        </details>
      )}

      {/* Compact question sets — theory + coding. Coding panel is hidden when empty
          so theory uses the full width. */}
      <div className={`q-sets-grid${codingQs.length === 0 ? ' single' : ''}`}>
        <div className="q-set-panel">
          <div className="q-set-head">
            <span className="q-set-title">Theory Questions</span>
            <span className="q-set-badge">{questions.length}</span>
          </div>
          <div className="q-set-body">
            {questions.length === 0 ? (
              <p className="muted" style={{ padding: '1rem' }}>No theory questions.</p>
            ) : (
              questions.map((q, i) => (
                <Fragment key={q.question_id}>
                  {i === firstLowFit && firstLowFit > 0 && (
                    <div className="cq-tier-divider">
                      Below fit {fitHigh.toFixed(2)} — weaker match to this session, review closely
                    </div>
                  )}
                  <CompactQuestion
                    id={q.question_id}
                    content={q.question || q.content}
                    difficulty={q.difficulty_level || q.difficulty}
                    company={q.attribution || q.asked_in_company}
                    role={q.role}
                    topic={q.topic}
                    subTopic={q.sub_topic}
                    source={q.source}
                    sourceUrl={q.source_url}
                    fit={typeof q.session_fit === 'number' ? q.session_fit : undefined}
                    decision={decisions[q.question_id]?.status}
                    reason={decisions[q.question_id]?.reason}
                    onDecide={onDecide}
                    index={i}
                  />
                </Fragment>
              ))
            )}
          </div>
        </div>

        {codingQs.length > 0 && (
        <div className="q-set-panel">
          <div className="q-set-head">
            <span className="q-set-title">Coding Questions</span>
            <span className="q-set-badge">{codingQs.length}</span>
          </div>
          <div className="q-set-body">
            {(
              codingQs.map((q, i) => (
                <CompactQuestion
                  key={q.id}
                  id={q.id}
                  title={q.title}
                  content={q.problem_statement || q.content}
                  difficulty={q.difficulty}
                  company={q.attribution || q.asked_in_company}
                  topic={q.topic}
                  subTopic={q.sub_topic}
                  language={q.language}
                  source={q.source}
                  sourceUrl={q.source_url}
                  snippet={snippetMap[q.code_id]}
                  decision={decisions[q.id]?.status}
                  reason={decisions[q.id]?.reason}
                  onDecide={onDecide}
                  index={i}
                />
              ))
            )}
          </div>
        </div>
        )}
      </div>

      {/* Rejected questions + reasons */}
      {result.removed?.length > 0 && (
        <details className="card rejected-card" style={{ margin: '0 1.35rem 1.6rem' }}>
          <summary className="rejected-summary">
            🗑️ Rejected questions ({result.removed.length}) — why they were dropped
          </summary>
          <div className="rejected-list">
            {result.removed.map((r, i) => (
              <div key={i} className="rej-row">
                <div className="rej-head">
                  <span className={`rej-stage rej-${r.stage || 'other'}`}>{r.stage || 'removed'}</span>
                  {r.difficulty && <span className="rej-diff">{r.difficulty}</span>}
                </div>
                <div className="rej-q">{r.content}</div>
                {r.reason && <div className="rej-reason">↳ {r.reason}</div>}
              </div>
            ))}
          </div>
        </details>
      )}
    </>
  )
}
