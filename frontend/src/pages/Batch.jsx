import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Icon from '../components/Icon.jsx'
import TopBar from '../components/TopBar.jsx'
import { api } from '../lib/api.js'

/**
 * Multi-topic batch progress: one row per topic, each its own independent run.
 *
 * A batch is N pipelines run back-to-back, NOT one merged run — each topic keeps its own question set,
 * quality gate, review screen and spreadsheet. So this page is a launcher into N reviews, not a
 * combined result view, and a failure on one topic is reported against that row while the rest carry on.
 */

const LABEL = {
  queued: 'Queued',
  running: 'Generating…',
  done: 'Ready to review',
  failed: 'Failed',
}

function StatusPill({ status }) {
  return (
    <span className={`batch-pill batch-pill-${status}`}>
      {status === 'running' && <span className="batch-spinner" aria-hidden="true" />}
      {LABEL[status] || status}
    </span>
  )
}

export default function Batch() {
  const { batchId } = useParams()
  const navigate = useNavigate()
  const [batch, setBatch] = useState(null)
  const [error, setError] = useState('')
  const timer = useRef(null)

  const poll = useCallback(async () => {
    try {
      const d = await api.getBatch(batchId)
      setBatch(d)
      return d.finished
    } catch (e) {
      setError(e?.message || 'Could not load this batch')
      return true                     // stop polling a batch we cannot read
    }
  }, [batchId])

  useEffect(() => {
    let stopped = false
    async function tick() {
      const done = await poll()
      if (stopped || done) return
      // Status is a cheap non-blocking read (it never joins the worker), but a batch runs for minutes,
      // so poll politely rather than hammering it.
      timer.current = setTimeout(tick, 2000)
    }
    tick()
    return () => { stopped = true; if (timer.current) clearTimeout(timer.current) }
  }, [poll])

  const runs = batch?.runs || []
  const done = runs.filter(r => r.status === 'done').length
  const failed = runs.filter(r => r.status === 'failed').length

  return (
    <>
      <TopBar title="Batch run" sub={`${runs.length} topic${runs.length === 1 ? '' : 's'} · one question set each`} />

      {error && <div className="batch-error">{error}</div>}

      <div className="page-content">
        <div className="batch-summary">
          <span><strong>{done}</strong> ready</span>
          {failed > 0 && <span className="batch-summary-failed"><strong>{failed}</strong> failed</span>}
          <span className="muted">
            {batch?.finished
              ? 'Batch complete.'
              : 'Topics run one at a time — you can review each as it lands.'}
          </span>
        </div>

        <div className="batch-list">
          {runs.map((r, i) => (
            <div key={r.run_id} className={`batch-row batch-row-${r.status}`}>
              <span className="batch-index">{i + 1}</span>
              <div className="batch-main">
                <div className="batch-topic" title={r.topic}>{r.topic}</div>
                <div className="batch-meta">
                  {(r.sessions || []).length > 0 && (
                    <span>{r.sessions.length} session{r.sessions.length === 1 ? '' : 's'}</span>
                  )}
                  {r.question_count != null && <span>{r.question_count} questions</span>}
                  {r.verdict && (
                    <span className={r.verdict === 'pass' ? 'batch-verdict-pass' : 'batch-verdict-fail'}>
                      gate {r.verdict}
                    </span>
                  )}
                </div>
                {r.error && <div className="batch-row-error" title={r.error}>{r.error}</div>}
              </div>
              <StatusPill status={r.status} />
              <div className="batch-actions">
                {r.status === 'running' && (
                  <button className="batch-btn" onClick={() => navigate(`/progress/${r.run_id}`)}>
                    Watch <Icon name="chevronRight" size={12} />
                  </button>
                )}
                {r.status === 'done' && (
                  <button className="batch-btn batch-btn-primary"
                          onClick={() => navigate(`/review/${r.run_id}`)}>
                    Review <Icon name="chevronRight" size={12} />
                  </button>
                )}
              </div>
            </div>
          ))}
          {!batch && !error && <div className="batch-row muted">Loading…</div>}
        </div>
      </div>
    </>
  )
}
