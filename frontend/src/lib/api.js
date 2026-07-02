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

  approve: (runId, acceptedIds, rejectedFeedback, action) =>
    fetch(`/api/approve/${runId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accepted_ids: acceptedIds, rejected_feedback: rejectedFeedback, action }),
    }).then(j),

  stream: (runId, onEvent) => {
    const es = new EventSource(`/api/stream/${runId}`)
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)) } catch {}
    }
    es.onerror = () => onEvent({ step: 'stream_error', status: 'error', detail: 'Stream disconnected' })
    return es
  },
}
