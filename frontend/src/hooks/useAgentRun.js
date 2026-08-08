import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api.js'

const PHASES = ['understanding', 'retrieval', 'validation', 'evaluation']

// Which agent owns each tool, so the transcript can group tool calls under the agent that made them.
// The backend also sends an `agent` field; this is the fallback for events that predate it.
const TOOL_OWNER = {
  understand_session: 'understanding',
  search_question_bank: 'retrieval',
  search_web_questions: 'retrieval',
  search_github_questions: 'retrieval',
  tavily_health: 'retrieval',
  validate_relevance: 'validation',
  deduplicate_questions: 'validation',
  suppress_rejected: 'validation',
  session_fit: 'validation',
  prefilter: 'validation',
  check_difficulty_balance: 'evaluation',
  check_outcome_coverage: 'evaluation',
  remove_question: 'evaluation',
  submit_question_set: 'evaluation',
  critique: 'gate',
}

/**
 * Subscribe to a run and shape its event stream into something renderable.
 *
 * Everything here is derived from STRUCTURED event fields. The previous version read the question
 * count out of a human-readable sentence with `/(\d+) questions/`, which broke whenever the wording
 * changed, and captured no timestamps at all — so no duration could ever be displayed.
 */
export function useAgentRun(runId) {
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState('running')   // running | reconnecting | done | error
  const [questionCount, setQuestionCount] = useState(null)
  const [errorDetail, setErrorDetail] = useState('')
  const [apiUsage, setApiUsage] = useState(null)
  const seen = useRef(new Set())

  useEffect(() => {
    if (!runId) return
    setEvents([])
    setStatus('running')
    setQuestionCount(null)
    setErrorDetail('')
    setApiUsage(null)
    seen.current = new Set()

    const sub = api.stream(runId, (ev) => {
      if (ev.step === 'reconnecting') {
        setStatus((cur) => (cur === 'done' || cur === 'error' ? cur : 'reconnecting'))
        return
      }
      // The server replays history on (re)connect, so drop anything already applied.
      if (ev.seq != null) {
        if (seen.current.has(ev.seq)) return
        seen.current.add(ev.seq)
      }
      setEvents((cur) => [...cur, ev])

      if (ev.step === 'complete') {
        setStatus('done')
        if (typeof ev.questions === 'number') setQuestionCount(ev.questions)
        if (ev.usage) setApiUsage(ev.usage)
      } else if (ev.step === 'error') {
        setStatus((cur) => (cur === 'done' ? 'done' : 'error'))
        if (ev.detail) setErrorDetail(ev.detail)
      } else {
        setStatus((cur) => (cur === 'reconnecting' ? 'running' : cur))
      }
    })
    return () => sub.close()
  }, [runId])

  // Usage totals come with the completion event; fall back to the result payload for older runs.
  useEffect(() => {
    if (status !== 'done' || !runId || apiUsage) return
    api.getResult(runId)
      .then((d) => { if (d?.report?.api_usage) setApiUsage(d.report.api_usage) })
      .catch(() => {})
  }, [status, runId, apiUsage])

  const derived = useMemo(() => {
    const phaseStatus = {}
    const phaseTiming = {}
    for (const ev of events) {
      if (!ev.step?.startsWith('phase:')) continue
      const phase = ev.step.slice(6)
      phaseStatus[phase] = ev.status
      const t = phaseTiming[phase] || (phaseTiming[phase] = {})
      if (ev.status === 'running') {
        t.startedAt = ev.ts
      } else {
        t.endedAt = ev.ts
        if (ev.duration_ms != null) t.durationMs = ev.duration_ms
        // The group header renders this; without capturing it the phase token count was always
        // undefined and silently never displayed.
        if (ev.tokens != null) t.tokens = ev.tokens
      }
    }
    // Prefer the server's measured duration; fall back to the timestamp delta.
    for (const t of Object.values(phaseTiming)) {
      if (t.durationMs == null && t.startedAt && t.endedAt) {
        t.durationMs = Math.round((t.endedAt - t.startedAt) * 1000)
      }
    }

    // Tool events grouped under their owning agent, collapsing each tool's running→done pair into
    // one entry so a tool called five times reads as five steps, not ten rows.
    const toolEvents = events.filter((ev) => ev.step && !ev.step.startsWith('phase:'))
    const steps = []
    const openByKey = new Map()
    for (const ev of toolEvents) {
      const group = ev.agent || TOOL_OWNER[ev.step] || 'other'
      const key = `${group}:${ev.step}`
      if (ev.status === 'running') {
        const step = { group, tool: ev.step, startedAt: ev.ts, status: 'running', detail: ev.detail, data: ev, tokens: undefined }
        openByKey.set(key, step)
        steps.push(step)
        continue
      }
      const open = openByKey.get(key)
      if (open && open.status === 'running') {
        openByKey.delete(key)
        Object.assign(open, {
          status: ev.status,
          detail: ev.detail || open.detail,
          endedAt: ev.ts,
          data: ev,
          tokens: ev.tokens,
          durationMs: ev.duration_ms ?? (open.startedAt && ev.ts
            ? Math.round((ev.ts - open.startedAt) * 1000) : undefined),
        })
      } else {
        steps.push({ group, tool: ev.step, status: ev.status, detail: ev.detail, startedAt: ev.ts, data: ev })
      }
    }

    const critiques = events.filter((e) => e.step === 'critique')
    const lastCritique = critiques[critiques.length - 1]
    // Revisions are the `retry` events. Counting `done` gave 1 for a run with zero revisions
    // (exactly one `done` fires, at pass or force-pass) and never showed 2.
    const revisionCount = critiques.filter((e) => e.status === 'retry').length
    const forcePassed = !!lastCritique?.detail?.startsWith('Force-passed')

    const first = events.find((e) => e.ts)
    const last = [...events].reverse().find((e) => e.ts)
    const elapsedMs = first && last ? Math.round((last.ts - first.ts) * 1000) : 0

    return {
      phaseStatus,
      phaseTiming,
      toolEvents,
      steps,
      critiqueStatus: lastCritique ? (lastCritique.status === 'done' ? 'done' : 'running') : null,
      revisionCount,
      forcePassed,
      elapsedMs,
      currentPhase: PHASES.slice().reverse().find((p) => phaseStatus[p] === 'running') || null,
    }
  }, [events])

  return { events, status, questionCount, errorDetail, apiUsage, ...derived }
}
