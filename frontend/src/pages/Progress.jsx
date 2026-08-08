import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import AgentTranscript from '../components/AgentTranscript.jsx'
import Icon from '../components/Icon.jsx'
import PipelineStepper from '../components/PipelineStepper.jsx'
import TopBar from '../components/TopBar.jsx'
import { useAgentRun } from '../hooks/useAgentRun.js'

/**
 * Live view of a run.
 *
 * The previous version showed a static left-to-right architecture diagram whose tool chips were
 * hardcoded strings that never matched the real tool names and never lit up, then a flat log of raw
 * identifiers. It also auto-navigated to Review 800ms after completion, so nothing on the page could
 * actually be read. Now the page shows what the agents did, and the reviewer decides when to move on.
 */
export default function Progress() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const run = useAgentRun(runId)
  const { status, questionCount, errorDetail, apiUsage, phaseStatus, critiqueStatus } = run
  const [autoAdvance, setAutoAdvance] = useState(() => localStorage.getItem('iqg-autoadvance') !== '0')

  const isDone = status === 'done'
  const isError = status === 'error'

  // Opt-in, and long enough to read the outcome first. The old 800ms jump meant the transcript was
  // never legible, which is most of why the pipeline felt like a black box.
  useEffect(() => {
    if (!isDone || !autoAdvance) return
    const t = setTimeout(() => navigate(`/review/${runId}`), 4000)
    return () => clearTimeout(t)
  }, [isDone, autoAdvance, runId, navigate])

  useEffect(() => {
    localStorage.setItem('iqg-autoadvance', autoAdvance ? '1' : '0')
  }, [autoAdvance])

  const title = isDone ? 'Pipeline complete' : isError ? 'Pipeline failed' : 'Generating questions'

  return (
    <>
      <TopBar
        title={title}
        sub={`Run ${runId?.slice(0, 8)}`}
        actions={<PipelineStepper phaseStatus={phaseStatus} critiqueStatus={critiqueStatus} />}
      />

      <div className="page-content">
        {status === 'reconnecting' && (
          <div className="alert alert-warn">
            <Icon name="refresh" size={15} className="spin" />
            <span>Connection lost — reconnecting. The run continues on the server.</span>
          </div>
        )}

        {isDone && (
          <div className="alert alert-success">
            <Icon name="check" size={15} />
            <span>
              {questionCount ? `${questionCount} questions ready for review.` : 'Run finished.'}
            </span>
            <label className="alert-toggle">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
              />
              Go to review automatically
            </label>
            <button className="btn btn-primary btn-sm" onClick={() => navigate(`/review/${runId}`)}>
              Review questions <Icon name="arrowRight" size={13} />
            </button>
          </div>
        )}

        {isError && (
          <div className="alert alert-error">
            <div className="alert-line">
              <Icon name="alert" size={15} />
              <span>The pipeline could not finish.</span>
              <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>
                <Icon name="arrowLeft" size={13} /> Start over
              </button>
            </div>
            {errorDetail && <p className="alert-detail">{errorDetail}</p>}
          </div>
        )}

        <AgentTranscript {...run} />
      </div>
    </>
  )
}
