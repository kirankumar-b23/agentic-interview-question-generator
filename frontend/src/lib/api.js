async function j(res) {
  if (res.ok) return res.json()
  let body = {}
  try { body = await res.json() } catch {}
  throw Object.assign(new Error(body.error || res.statusText), { status: res.status })
}

export const api = {
  getSessions: () => fetch('/api/sessions').then(j),
  getTopics: (courseId) => fetch(`/api/topics${courseId ? `?course=${encodeURIComponent(courseId)}` : ''}`).then(j),
  getCourses: () => fetch('/api/courses').then(j),
  getHistory: () => fetch('/api/history').then(j),
  getUsage: () => fetch('/api/usage').then(j),
  getMeta: () => fetch('/api/meta').then(j),

  addCourseSession: (payload) =>
    fetch('/api/courses/session', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(j),
  importCourse: (payload) =>
    fetch('/api/courses/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(j),
  deleteCourse: (courseId) =>
    fetch(`/api/courses/${encodeURIComponent(courseId)}`, { method: 'DELETE' }).then(j),

  generate: (sessionNames, maxQuestions, model, preview, course) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_names: sessionNames, max_questions: maxQuestions, model, preview,
        category: course?.category, course_type: course?.course_type,
      }),
    }).then(j),

  // TESTING: preview mode — resume a paused run through the quality gate
  proceed: (runId) =>
    fetch(`/api/proceed/${runId}`, { method: 'POST' }).then(j),

  getResult: (runId) => fetch(`/api/result/${runId}`).then(j),

  // `decisionsSent` tells the server the reviewer made explicit per-question decisions. Without it
  // an empty acceptedIds is ambiguous, and the server used to read "accepted nothing" as
  // "no filter requested" — exporting and banking every rejected question.
  approve: (runId, acceptedIds, rejectedFeedback, action, decisionsSent = true) =>
    fetch(`/api/approve/${runId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        accepted_ids: acceptedIds,
        rejected_feedback: rejectedFeedback,
        action,
        decisions_sent: decisionsSent,
      }),
    }).then(j),

  /**
   * Subscribe to a run's progress stream.
   *
   * EventSource reconnects on its own after a network drop, but it cannot tell a dropped connection
   * from a server that finished and closed — so the old code reported "Stream disconnected" on every
   * successful run, and reconnected forever into an empty queue. Here:
   *   - terminal events (`complete` / `error`) close the stream deliberately, so no retry storm;
   *   - anything else that closes it is a real drop, and the server replays missed events on
   *     reconnect (see orchestrator.get_history), so the transcript survives a tab reload;
   *   - `heartbeat` frames are swallowed — they exist only to keep the connection warm.
   */
  stream: (runId, onEvent) => {
    let closed = false
    let lastSeq = -1
    let es = null

    const finish = () => {
      closed = true
      if (es) es.close()
    }

    // Reopen with `after=lastSeq` so a reconnect resumes from where we left off. EventSource's own
    // reconnect reopens the original URL, which replayed the entire transcript every time; the server
    // supports the offset (orchestrator.get_history) and this is what actually uses it.
    const open = () => {
      if (closed) return
      es = new EventSource(`/api/stream/${runId}?after=${lastSeq}`)
      es.onmessage = (e) => {
        let ev
        try { ev = JSON.parse(e.data) } catch { return }
        if (typeof ev.seq === 'number') lastSeq = Math.max(lastSeq, ev.seq)
        if (ev.step === 'heartbeat') return
        onEvent(ev)
        if (ev.step === 'complete' || ev.step === 'error') finish()
      }
      es.onerror = () => {
        if (closed || es.readyState !== EventSource.CLOSED) return
        // The browser closed it for good; reconnect ourselves so `after` is honoured.
        onEvent({ step: 'reconnecting', status: 'running', detail: 'Connection lost — reconnecting…' })
        setTimeout(open, 1500)
      }
    }
    open()
    return { close: finish }
  },
}
