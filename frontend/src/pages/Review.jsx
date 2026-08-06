import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Icon from '../components/Icon.jsx'
import PipelineStepper from '../components/PipelineStepper.jsx'
import QualityPanel from '../components/QualityPanel.jsx'
import { api } from '../lib/api.js'
import TopBar from '../components/TopBar.jsx'

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
  // `company` is WHO ASKED (a real company, or the NIAT placeholder); `site` is WHERE IT WAS FOUND.
  // They render as separate tags on purpose — a content site used to be shown in the company tag,
  // so "Indeed" looked like an employer that had asked the question.
  // `original` is set only when the scope trim shortened the question: it still carries the company's
  // name, so the edit has to be visible and the source text inspectable.
  company, site, original, offSyllabus, role, topic, subTopic, language, source, sourceUrl,
  snippet, decision, onDecide, index, fit, reason, focused, onFocus,
}) {
  const [open, setOpen] = useState(false)
  const rowRef = useRef(null)
  const isCoding = !!title
  const diff = difficulty || 'Medium'
  const diffClass = diff === 'Easy' ? 'd-easy' : diff === 'Hard' ? 'd-hard' : 'd-medium'

  // Keep the keyboard cursor in view as it moves through a long set.
  useEffect(() => {
    if (focused) rowRef.current?.scrollIntoView({ block: 'nearest' })
  }, [focused])

  const cls = ['cq-row',
    decision === 'accepted' ? 'cq-row-accepted' : decision === 'rejected' ? 'cq-row-rejected' : '',
    focused ? 'cq-row-focused' : ''].filter(Boolean).join(' ')

  return (
    <div className={cls} ref={rowRef} onMouseEnter={onFocus}>
      {/* The disclosure is its own button rather than a click handler on the row: the row also holds
          the accept/reject buttons, and nesting interactive elements is invalid and untabbable. */}
      <div className="cq-main">
        <button
          type="button"
          className="cq-disclose"
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          title={open ? 'Hide details' : 'Show details'}
        >
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="cq-caret" />
          <span className="cq-num">Q{index + 1}</span>
          <span className={`cq-text${open ? ' cq-text-full' : ''}`}>{isCoding ? title : content}</span>
        </button>
        <div className="cq-tags">
          {company && <span className="cq-company" title={`Asked at ${company}`}>{company}</span>}
          {site && <span className="cq-via" title={`Found on ${site} — provenance, not the asking company`}>via {site}</span>}
          {original && (
            <span className="cq-adapted"
                  title={`Trimmed to this session's scope. As sourced: "${original}"`}>adapted</span>
          )}
          {offSyllabus && (
            <span className="cq-offsyllabus"
                  title={`Tests "${offSyllabus}", which does not appear in this session's reading material. On-domain, but beyond the syllabus.`}>off-syllabus</span>
          )}
          {typeof fit === 'number' && (
            <span
              className="cq-fit"
              title="Session fit — similarity to this session's learning outcomes and reading material"
            >{fit.toFixed(2)}</span>
          )}
          <span className={`cq-diff ${diffClass}`}>{diff}</span>
        </div>
        <div className="cq-btns">
          <button
            className={`cq-btn cq-accept${decision === 'accepted' ? ' active' : ''}`}
            onClick={() => onDecide(id, 'accepted')}
            aria-pressed={decision === 'accepted'}
            aria-label={`Accept question ${index + 1}`}
            title="Accept (a)"
          ><Icon name="check" size={13} /></button>
          <button
            className={`cq-btn cq-reject${decision === 'rejected' ? ' active' : ''}`}
            onClick={() => onDecide(id, 'rejected')}
            aria-pressed={decision === 'rejected'}
            aria-label={`Reject question ${index + 1}`}
            title="Reject (r)"
          ><Icon name="x" size={13} /></button>
        </div>
      </div>

      {/* Why it was rejected. One click, and the pipeline learns a rule from it. */}
      {decision === 'rejected' && (
        <div className="cq-reasons">
          <span className="cq-reasons-label">Why?</span>
          {REJECT_REASONS.map((r, i) => (
            <button
              key={r.key}
              className={`cq-reason-chip${reason === r.key ? ' active' : ''}`}
              onClick={() => onDecide(id, 'rejected', r.key)}
              aria-pressed={reason === r.key}
              title={`${r.label} (${i + 1})`}
            >{r.label}</button>
          ))}
        </div>
      )}

      {open && (
        <div className="cq-detail">
          {/* The row line-clamps to two lines, so expanding must show the full text. This used to be
              gated on `isCoding`, which meant a theory question longer than two lines could never be
              read in full anywhere in the UI. */}
          {content && <p className="cq-detail-text">{content}</p>}
          {snippet?.code_content && <pre className="q-code-pre">{snippet.code_content}</pre>}
          {sourceUrl && (
            <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className="cq-resource-link">
              <Icon name="external" size={12} /> Verify source
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

  const onDecide = useCallback((qid, status, reason) => {
    setDecisions(cur => {
      // Re-clicking the same reason clears it, and switching to accepted drops any stale reason.
      if (status !== 'rejected') return { ...cur, [qid]: { status } }
      const prev = cur[qid]?.reason
      return { ...cur, [qid]: { status, reason: reason === prev ? undefined : (reason ?? prev) } }
    })
  }, [])

  // Rank by session fit (highest first) so the reviewer works top-down and can stop where fit falls
  // off — the set size is deliberately uncapped, so ordering is what keeps review tractable.
  // Questions with no fit score (embeddings unavailable) keep their original order, at the end.
  const fitOf = (q) => (typeof q.session_fit === 'number' ? q.session_fit : -1)
  const ranked = useMemo(
    () => [...(result?.output?.question_details || [])].sort((a, b) => fitOf(b) - fitOf(a)),
    [result],
  )

  // ── Filters ──
  // Ranking alone is not enough for an uncapped set: to judge "are the low-fit ones all bad?" or
  // "did the web source contribute anything usable?" the reviewer has to be able to isolate a slice.
  const fitHighThreshold = result?.thresholds?.session_fit_high ?? 0.35
  const [filters, setFilters] = useState({ fit: 'all', difficulty: 'all', source: 'all', attribution: 'all' })
  const setFilter = (key, value) => setFilters((f) => ({ ...f, [key]: value }))
  const filtersActive = Object.values(filters).some((v) => v !== 'all')

  const bandOf = (q) => {
    const f = fitOf(q)
    if (f < 0) return 'unscored'
    if (f >= fitHighThreshold) return 'high'
    if (f >= fitHighThreshold * 0.6) return 'review'
    return 'low'
  }
  // A real company vs the NIAT placeholder. Keyed on `asked_in_company`, not on `attribution`, so it
  // stays true regardless of how attribution is rendered (see models.attribution_label — the site
  // name is provenance and lives in `source_site`, never here).
  const hasCompany = (q) => !!(q.asked_in_company || '').trim()

  const visible = useMemo(() => ranked.filter((q) => (
    (filters.fit === 'all' || bandOf(q) === filters.fit)
    && (filters.difficulty === 'all' || (q.difficulty || 'Medium') === filters.difficulty)
    && (filters.source === 'all' || (q.source || '') === filters.source)
    && (filters.attribution === 'all'
        || (filters.attribution === 'company' ? hasCompany(q) : !hasCompany(q)))
  )), [ranked, filters, fitHighThreshold])

  const sourcesPresent = useMemo(
    () => [...new Set(ranked.map((q) => q.source).filter(Boolean))], [ranked])

  // ── Keyboard triage ──
  // An uncapped set can be 50+ questions; accept/reject was mouse-only, one click per row, with the
  // buttons ~1000px apart from the text on a wide screen. j/k to move, a/r to decide, 1-7 to give a
  // reason. This is the difference between a reviewable set and a chore.
  const [cursor, setCursor] = useState(0)
  const [showHelp, setShowHelp] = useState(false)

  // The cursor indexes the VISIBLE list. Clamp on ANY change to that list, not just on a filter
  // change — a fresh result with fewer questions would otherwise leave the cursor past the end and
  // `visible[cursor]` undefined.
  useEffect(() => { setCursor(0) }, [filters])
  useEffect(() => {
    setCursor((c) => Math.min(c, Math.max(0, visible.length - 1)))
  }, [visible.length])

  // ── Bulk actions ──
  // Accepting the high-confidence band in one keystroke is the point of ranking: the reviewer then
  // only hand-judges the tail. These write explicit decisions, so `decisions_sent` stays truthful and
  // an all-rejected set still cannot export.
  const acceptAboveFit = useCallback(() => {
    setDecisions((cur) => {
      const next = { ...cur }
      for (const q of ranked) {
        if (fitOf(q) >= fitHighThreshold) next[q.question_id] = { status: 'accepted' }
      }
      return next
    })
  }, [ranked, fitHighThreshold])

  const clearDecisions = useCallback(() => setDecisions({}), [])
  const aboveFitCount = useMemo(
    () => ranked.filter((q) => fitOf(q) >= fitHighThreshold).length, [ranked, fitHighThreshold])

  useEffect(() => {
    function onKey(e) {
      // Never hijack typing, and let real shortcuts through.
      const tag = e.target?.tagName
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target?.isContentEditable) return
      if (!visible.length) return

      const current = visible[cursor]
      const key = e.key

      if (key === 'j' || key === 'ArrowDown') { setCursor(c => Math.min(c + 1, visible.length - 1)); e.preventDefault() }
      else if (key === 'k' || key === 'ArrowUp') { setCursor(c => Math.max(c - 1, 0)); e.preventDefault() }
      else if (key === 'a' && current) { onDecide(current.question_id, 'accepted'); setCursor(c => Math.min(c + 1, visible.length - 1)) }
      else if (key === 'r' && current) { onDecide(current.question_id, 'rejected') }
      else if (key === 'u' && current) { setDecisions(cur => { const n = { ...cur }; delete n[current.question_id]; return n }) }
      else if (key === 'A') { acceptAboveFit(); e.preventDefault() }        // shift+a
      else if (key === '?') setShowHelp(v => !v)
      else if (key === 'Escape') { setShowHelp(false); setFilters({ fit: 'all', difficulty: 'all', source: 'all', attribution: 'all' }) }
      else if (/^[1-7]$/.test(key) && current && decisions[current.question_id]?.status === 'rejected') {
        onDecide(current.question_id, 'rejected', REJECT_REASONS[Number(key) - 1].key)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [visible, cursor, decisions, onDecide, acceptAboveFit])

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
      <TopBar title="Review questions" actions={
        <>
          <PipelineStepper completedUntil="gate" activeStage="review" />
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>
            <Icon name="arrowLeft" size={13} /> History
          </button>
        </>
      } />
      <div className="page-content" aria-busy="true" aria-label="Loading results">
        <div className="skeleton skeleton-panel" />
        {Array.from({ length: 6 }).map((_, i) => <div key={i} className="skeleton skeleton-row" />)}
      </div>
    </>
  )

  if (error && !result) return (
    <>
      <TopBar title="Review questions" actions={<button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>
            <Icon name="arrowLeft" size={13} /> History
          </button>} />
      <div className="page-content"><div className="alert alert-error">{error}</div></div>
    </>
  )

  if (!result) return null

  const questions = ranked
  // Index of the first question below the high-confidence bar → where the divider goes. Computed
  // against the VISIBLE list, since that is what is rendered; filtering otherwise put the divider in
  // an arbitrary place.
  const visibleFirstLowFit = visible.findIndex(q => fitOf(q) >= 0 && fitOf(q) < fitHighThreshold)
  const codingQs = result.output?.coding_questions || []
  const snippets = result.output?.code_snippets || []
  const snippetMap = Object.fromEntries(snippets.map(s => [s.code_id, s]))
  const total = questions.length + codingQs.length
  const rejectedCount = Object.values(decisions).filter(d => d.status === 'rejected').length
  const approvedCount = total - rejectedCount

  if (done) {
    return (
      <>
        <TopBar title="Export complete" actions={
          <>
            <PipelineStepper completedUntil="export" />
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>
            <Icon name="arrowLeft" size={13} /> History
          </button>
          </>
        } />
        <div className="page-content">
          <div className="done-banner">
            <h2>Approved</h2>
            {sheetUrl ? (
              <>
                <p>{approvedCount} question{approvedCount !== 1 ? 's' : ''} exported to Google Sheets.</p>
                <a href={sheetUrl} target="_blank" rel="noopener noreferrer" className="btn btn-primary done-sheet-link">
                  Open Google Sheet
                </a>
              </>
            ) : sheetError ? (
              <>
                <p>{approvedCount} question{approvedCount !== 1 ? 's' : ''} approved (saved locally).</p>
                <p className="alert alert-error">Sheets export failed: {sheetError}</p>
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
      <TopBar
        title={awaitingGate ? 'Preview — picked questions' : 'Review questions'}
        sub={`${displayName}${sessionType ? ` · ${sessionType}` : ''}`}
        actions={
          <>
            <PipelineStepper completedUntil="gate" activeStage="review" />
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>
            <Icon name="arrowLeft" size={13} /> History
          </button>
          </>
        }
      />

      {/* Action banner */}
      <div className="action-banner" style={awaitingGate ? { borderLeftColor: 'var(--warn)' } : undefined}>
        <div className="ab-text">
          <span className="ab-title">
            {awaitingGate
              ? 'Testing preview — verify the picked questions (not yet quality-checked)'
              : 'Action needed — Review before publishing to portal'}
          </span>
          <span className="ab-sub">
            {total} questions for <strong title={sessionName}>{displayName}</strong>
            {' · '}{questions.length} theory · {codingQs.length} coding
            {!awaitingGate && result.report && ` · Quality ${Math.round(result.report.composite_score * 100)}/100`}
            {!awaitingGate && result.report?.loops_used > 0 && ` · ${result.report.loops_used} revision(s)`}
          </span>
        </div>
        <div className="ab-btns">
          {awaitingGate ? (
            <button className="btn btn-primary" disabled={submitting} onClick={handleProceed}>
              {submitting ? 'Submitting…' : 'Proceed to quality gate'}
              <Icon name="arrowRight" size={14} />
            </button>
          ) : (
            <>
              <button className="btn btn-reject-all"
                      disabled={submitting || rejectedCount === 0}
                      title={rejectedCount === 0 ? 'Mark one or more questions as rejected first' : ''}
                      onClick={handleReject}>
                <Icon name="refresh" size={14} />
                {rejectedCount > 0
                  ? `Replace ${rejectedCount} rejected & regenerate`
                  : 'Replace rejected & regenerate'}
              </button>
              {/* Exporting with nothing accepted used to send an empty accepted_ids, which the
                  server read as "no filter" and exported + banked every rejected question. */}
              <button className="btn btn-primary"
                      disabled={submitting || approvedCount === 0}
                      title={approvedCount === 0
                        ? 'Every question is rejected — accept at least one, or regenerate'
                        : ''}
                      onClick={handleApprove}>
                <Icon name="export" size={14} />
                {submitting ? 'Exporting…' : `Export to Sheets (${approvedCount})`}
              </button>
            </>
          )}
        </div>
      </div>

      {error && <div className="review-gutter"><div className="alert alert-error">{error}</div></div>}

      <div className="review-gutter">
        <QualityPanel report={result.report} />
      </div>

      {/* Learning outcomes */}
      {result.context?.learning_outcomes?.length > 0 && (
        <details className="card outcomes-card">
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
            <span className="q-set-title">Theory questions</span>
            <span className="q-set-badge">
              {filtersActive ? `${visible.length} / ${questions.length}` : questions.length}
            </span>
            <button className="kbd-hint" onClick={() => setShowHelp(v => !v)} aria-expanded={showHelp}>
              <Icon name="keyboard" size={13} /> shortcuts
            </button>
          </div>
          {showHelp && (
            <div className="kbd-help">
              <div><kbd>j</kbd><kbd>k</kbd> move</div>
              <div><kbd>a</kbd> accept</div>
              <div><kbd>r</kbd> reject</div>
              <div><kbd>1</kbd>–<kbd>7</kbd> reason (after reject)</div>
              <div><kbd>u</kbd> undo</div>
              <div><kbd>shift</kbd>+<kbd>a</kbd> accept all above fit</div>
              <div><kbd>esc</kbd> clear filters</div>
              <div><kbd>?</kbd> toggle this</div>
            </div>
          )}

          {/* Filters + bulk actions. Ranking gets the best questions to the top; these let the
              reviewer isolate a slice ("are the low-fit ones all bad?") and clear the confident
              majority in one action so only the tail needs hand-judging. */}
          <div className="q-toolbar">
            <Icon name="filter" size={13} className="q-toolbar-ico" />
            <label className="sr-only" htmlFor="flt-fit">Filter by session fit</label>
            <select id="flt-fit" className="q-filter" value={filters.fit}
                    onChange={(e) => setFilter('fit', e.target.value)}>
              <option value="all">any fit</option>
              <option value="high">high fit (≥ {fitHighThreshold.toFixed(2)})</option>
              <option value="review">needs a look</option>
              <option value="low">low fit</option>
              <option value="unscored">unscored</option>
            </select>

            <label className="sr-only" htmlFor="flt-diff">Filter by difficulty</label>
            <select id="flt-diff" className="q-filter" value={filters.difficulty}
                    onChange={(e) => setFilter('difficulty', e.target.value)}>
              <option value="all">any difficulty</option>
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
            </select>

            {sourcesPresent.length > 1 && (
              <>
                <label className="sr-only" htmlFor="flt-src">Filter by source</label>
                <select id="flt-src" className="q-filter" value={filters.source}
                        onChange={(e) => setFilter('source', e.target.value)}>
                  <option value="all">any source</option>
                  {sourcesPresent.map((src) => <option key={src} value={src}>{src}</option>)}
                </select>
              </>
            )}

            <label className="sr-only" htmlFor="flt-attr">Filter by attribution</label>
            <select id="flt-attr" className="q-filter" value={filters.attribution}
                    onChange={(e) => setFilter('attribution', e.target.value)}>
              <option value="all">any attribution</option>
              <option value="company">real company</option>
              <option value="source">unattributed (NIAT)</option>
            </select>

            {filtersActive && (
              <button className="q-filter-clear"
                      onClick={() => setFilters({ fit: 'all', difficulty: 'all', source: 'all', attribution: 'all' })}>
                <Icon name="x" size={11} /> clear
              </button>
            )}

            <span className="q-toolbar-spacer" />
            <button className="btn btn-ghost btn-sm" onClick={acceptAboveFit}
                    disabled={aboveFitCount === 0}
                    title={`Accept the ${aboveFitCount} question(s) at or above fit ${fitHighThreshold.toFixed(2)}`}>
              <Icon name="check" size={13} /> Accept {aboveFitCount} above fit
            </button>
            <button className="btn btn-ghost btn-sm" onClick={clearDecisions}
                    disabled={Object.keys(decisions).length === 0}>
              <Icon name="refresh" size={13} /> Reset
            </button>
          </div>
          <div className="q-set-body">
            {questions.length === 0 ? (
              <div className="empty-state">
                <Icon name="info" size={22} />
                <p>No theory questions survived validation for this session. The report above says why.</p>
              </div>
            ) : visible.length === 0 ? (
              <div className="empty-state">
                <Icon name="filter" size={22} />
                <p>No questions match these filters — {questions.length} are hidden.</p>
                <button className="btn btn-ghost btn-sm"
                        onClick={() => setFilters({ fit: 'all', difficulty: 'all', source: 'all', attribution: 'all' })}>
                  Clear filters
                </button>
              </div>
            ) : (
              visible.map((q, i) => (
                <Fragment key={q.question_id}>
                  {i === visibleFirstLowFit && visibleFirstLowFit > 0 && (
                    <div className="cq-tier-divider">
                      Below fit {fitHighThreshold.toFixed(2)} — weaker match to this session, review closely
                    </div>
                  )}
                  <CompactQuestion
                    id={q.question_id}
                    content={q.question || q.content}
                    difficulty={q.difficulty_level || q.difficulty}
                    company={q.attribution || q.asked_in_company}
                    site={q.source_site}
                    original={q.original_content}
                    offSyllabus={q.off_syllabus_concept}
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
                    focused={i === cursor}
                    onFocus={() => setCursor(i)}
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
        <details className="card rejected-card">
          <summary className="rejected-summary">
            Rejected questions ({result.removed.length}) — why they were dropped
          </summary>
          <div className="rejected-list">
            {result.removed.map((r, i) => (
              <div key={i} className="rej-row">
                <div className="rej-head">
                  <span className={`rej-stage rej-${r.stage || 'other'}`}>{r.stage || 'removed'}</span>
                  {r.difficulty && <span className="rej-diff">{r.difficulty}</span>}
                </div>
                <div className="rej-q">{r.content}</div>
                {r.reason && <div className="rej-reason">{r.reason}</div>}
              </div>
            ))}
          </div>
        </details>
      )}
    </>
  )
}
