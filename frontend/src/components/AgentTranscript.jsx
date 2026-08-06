import { useEffect, useMemo, useRef, useState } from 'react'
import Icon, { AGENT_ICONS, STEP_ICONS } from './Icon.jsx'

/**
 * A trace of what the agents actually did, replacing the flat tool log.
 *
 * The log it replaces printed one row per SSE event with an emoji, the raw snake_case tool name
 * (`search_question_bank`) and a backend-composed sentence. It had no grouping, no timing, no way to
 * see a step's numbers, and a tool called five times produced five identical-looking rows.
 *
 * This reads as an instrument trace: agents are groups on a vertical rail, each tool is a timed step
 * beneath its agent, and the numbers the pipeline already computes (pool sizes, drop counts, fit
 * floors) are pulled out of the prose into a readable row. Expanding a step shows the full detail.
 */

const GROUP_ORDER = ['understanding', 'retrieval', 'validation', 'evaluation', 'gate', 'other']

const GROUP_LABELS = {
  understanding: 'Understanding',
  retrieval: 'Retrieval',
  validation: 'Validation',
  evaluation: 'Evaluation',
  gate: 'Quality gate',
  other: 'Pipeline',
}

// Human labels for the raw step identifiers. Anything unmapped falls back to a de-underscored form,
// so a new tool degrades to "search related repos" rather than breaking.
const STEP_LABELS = {
  understand_session: 'Resolve session',
  search_question_bank: 'Search question bank',
  search_web_questions: 'Search the web',
  search_github_questions: 'Search GitHub repos',
  tavily_health: 'Check web search',
  validate_relevance: 'Score relevance',
  deduplicate_questions: 'Remove duplicates',
  suppress_rejected: 'Suppress past rejections',
  session_fit: 'Score session fit',
  prefilter: 'Drop cross-topic',
  check_difficulty_balance: 'Check difficulty mix',
  check_outcome_coverage: 'Check outcome coverage',
  remove_question: 'Remove question',
  submit_question_set: 'Select final set',
  critique: 'Critique the set',
}

const humanStep = (step) =>
  STEP_LABELS[step] || step.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())

function formatDuration(ms) {
  if (ms == null) return null
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60000)
  return `${m}m ${Math.round((ms % 60000) / 1000)}s`
}

/**
 * Pull the numbers out of a backend detail string so they can be shown as data.
 * Purely additive — the full sentence is still available when the step is expanded — so a wording
 * change degrades to "no chips" rather than to a wrong number.
 */
function extractMetrics(detail = '') {
  const chips = []
  const add = (label, value) => value && chips.push({ label, value })

  const funnel = detail.match(/(\d+)\s*(?:→|->)\s*(\d+)/)
  if (funnel) add('', `${funnel[1]} → ${funnel[2]}`)

  const found = detail.match(/(?:Found|found)\s+(\d+)/)
  if (found && !funnel) add('found', found[1])

  const dropped = detail.match(/dropped\s+(\d+)/i)
  if (dropped) add('dropped', dropped[1])

  const removed = detail.match(/removed\s+(\d+)/i)
  if (removed) add('removed', removed[1])

  const kept = detail.match(/[Kk]ept\s+(\d+)/)
  if (kept) add('kept', kept[1])

  const suppressed = detail.match(/Suppressed\s+(\d+)/i)
  if (suppressed) add('suppressed', suppressed[1])

  const fit = detail.match(/best (?:fit )?([\d.]+)/i)
  if (fit) add('best fit', fit[1])

  const floor = detail.match(/below fit ([\d.]+)/i)
  if (floor) add('floor', floor[1])

  return chips.slice(0, 4)
}

function StatusMark({ status }) {
  if (status === 'running') {
    return <span className="tr-mark tr-mark-run"><Icon name="spinner" size={13} className="spin" /></span>
  }
  if (status === 'error') {
    return <span className="tr-mark tr-mark-err"><Icon name="x" size={13} /></span>
  }
  if (status === 'warning') {
    return <span className="tr-mark tr-mark-warn"><Icon name="alert" size={13} /></span>
  }
  return <span className="tr-mark tr-mark-ok"><Icon name="check" size={13} /></span>
}

function Step({ step }) {
  const [open, setOpen] = useState(false)
  const metrics = useMemo(() => extractMetrics(step.detail), [step.detail])
  const duration = formatDuration(step.durationMs)
  const label = humanStep(step.tool)
  // Only offer expansion when there is more to read than the row already shows.
  const expandable = !!step.detail && (step.detail.length > 68 || metrics.length > 0)

  return (
    <div className={`tr-step${step.status === 'error' ? ' tr-step-err' : ''}`}>
      <button
        type="button"
        className="tr-step-row"
        onClick={() => expandable && setOpen((o) => !o)}
        aria-expanded={expandable ? open : undefined}
        disabled={!expandable}
      >
        {expandable ? (
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="tr-caret" />
        ) : (
          <span className="tr-caret-spacer" />
        )}
        <Icon name={STEP_ICONS[step.tool] || 'cpu'} size={14} className="tr-step-ico" />
        <span className="tr-step-name">{label}</span>
        <span className="tr-step-metrics">
          {metrics.map((m, i) => (
            <span key={i} className="tr-chip">
              {m.label && <em>{m.label}</em>}{m.value}
            </span>
          ))}
        </span>
        {step.tokens > 0 && (
          <span className="tr-step-tok" title="tokens this step spent">
            {step.tokens >= 1000 ? `${(step.tokens / 1000).toFixed(1)}K` : step.tokens}
          </span>
        )}
        {duration && <span className="tr-step-time">{duration}</span>}
        <StatusMark status={step.status} />
      </button>
      {open && step.detail && <div className="tr-step-detail">{step.detail}</div>}
    </div>
  )
}

function Group({ id, steps, phase, timing }) {
  const durationMs = timing?.durationMs
  const failed = phase === 'error' || steps.some((s) => s.status === 'error')
  const running = phase === 'running' || steps.some((s) => s.status === 'running')
  const state = failed ? 'error' : running ? 'running' : phase === 'done' ? 'done' : 'pending'

  return (
    <section className={`tr-group tr-group-${state}`} aria-label={GROUP_LABELS[id] || id}>
      <header className="tr-group-head">
        <span className={`tr-node tr-node-${state}`}>
          <Icon name={AGENT_ICONS[id] || 'cpu'} size={15} />
        </span>
        <h3 className="tr-group-name">{GROUP_LABELS[id] || id}</h3>
        {timing?.tokens > 0 && (
          <span className="tr-group-tok" title="tokens used by this stage">
            {(timing.tokens / 1000).toFixed(1)}K
          </span>
        )}
        {durationMs != null && <span className="tr-group-time">{formatDuration(durationMs)}</span>}
        <StatusMark status={state === 'pending' ? 'running' : state === 'done' ? 'done' : state} />
      </header>
      <div className="tr-steps">
        {steps.length === 0
          ? <p className="tr-empty">waiting…</p>
          : steps.map((s, i) => <Step key={`${s.tool}-${i}`} step={s} />)}
      </div>
    </section>
  )
}

export default function AgentTranscript({
  steps = [], phaseStatus = {}, phaseTiming = {}, critiqueStatus,
  revisionCount = 0, forcePassed = false, elapsedMs = 0, apiUsage, status,
}) {
  const scrollRef = useRef(null)
  const pinned = useRef(true)

  // Follow the trace while the user is at the bottom; stop the moment they scroll up to read.
  useEffect(() => {
    const el = scrollRef.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [steps.length])

  function onScroll(e) {
    const el = e.currentTarget
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  }

  const grouped = useMemo(() => {
    const byGroup = new Map()
    for (const s of steps) {
      if (!byGroup.has(s.group)) byGroup.set(s.group, [])
      byGroup.get(s.group).push(s)
    }
    // Show every stage that has run or is running, in pipeline order.
    return GROUP_ORDER
      .filter((g) => byGroup.has(g) || phaseStatus[g] || (g === 'gate' && critiqueStatus))
      .map((g) => ({
        id: g,
        steps: byGroup.get(g) || [],
        phase: g === 'gate' ? critiqueStatus : phaseStatus[g],
        timing: phaseTiming[g],
      }))
  }, [steps, phaseStatus, phaseTiming, critiqueStatus])

  const tokens = apiUsage ? (apiUsage.prompt_tokens || 0) + (apiUsage.completion_tokens || 0) : 0

  return (
    <section className="transcript" aria-label="Agent transcript">
      <header className="tr-head">
        <span className="tr-head-title">
          <Icon name="layers" size={14} /> Agent transcript
        </span>
        <span className="tr-head-meta">
          {elapsedMs > 0 && <span><Icon name="clock" size={12} /> {formatDuration(elapsedMs)}</span>}
          {steps.length > 0 && <span>{steps.length} steps</span>}
          {tokens > 0 && <span><Icon name="zap" size={12} /> {(tokens / 1000).toFixed(1)}K tokens</span>}
          {apiUsage?.model && <span className="tr-head-model">{apiUsage.model}</span>}
        </span>
      </header>

      {(revisionCount > 0 || forcePassed) && (
        <div className={`tr-banner${forcePassed ? ' tr-banner-warn' : ''}`}>
          <Icon name={forcePassed ? 'alert' : 'refresh'} size={14} />
          {forcePassed
            ? `Quality gate did not pass this set — shipped after ${revisionCount} revision attempt(s). Review closely.`
            : `Quality gate requested ${revisionCount} revision${revisionCount > 1 ? 's' : ''}.`}
        </div>
      )}

      <div className="tr-body" ref={scrollRef} onScroll={onScroll}>
        {grouped.length === 0 ? (
          <p className="tr-waiting">
            <Icon name="spinner" size={14} className="spin" /> Starting the pipeline…
          </p>
        ) : (
          grouped.map((g) => <Group key={g.id} {...g} />)
        )}
        {/* Screen readers get stage transitions announced; the per-step firehose would be unusable. */}
        <p className="sr-only" aria-live="polite">
          {status === 'done' ? 'Pipeline complete.'
            : status === 'error' ? 'Pipeline failed.'
            : grouped.length ? `${GROUP_LABELS[grouped[grouped.length - 1].id]} in progress.` : ''}
        </p>
      </div>
    </section>
  )
}
