export const BASE = import.meta.env.VITE_API_URL || ''

async function j(res) {
  if (res.ok) return res.json()
  let body = {}
  try { body = await res.json() } catch {}
  throw Object.assign(new Error(body.error || res.statusText), { status: res.status })
}

export const api = {
  getSessions: () => fetch(`${BASE}/api/sessions`).then(j),
  getTopics: (courseId) => fetch(`${BASE}/api/topics${courseId ? `?course=${encodeURIComponent(courseId)}` : ''}`).then(j),
  getCourses: () => fetch(`${BASE}/api/courses`).then(j),
  getHistory: () => fetch(`${BASE}/api/history`).then(j),
  getUsage: () => fetch(`${BASE}/api/usage`).then(j),
  getMeta: () => fetch(`${BASE}/api/meta`).then(j),

  addCourseSession: (payload) =>
    fetch(`${BASE}/api/courses/session`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(j),
  importCourse: (payload) =>
    fetch(`${BASE}/api/courses/import`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(j),
  deleteCourse: (courseId) =>
    fetch(`${BASE}/api/courses/${encodeURIComponent(courseId)}`, { method: 'DELETE' }).then(j),

  generate: (sessionNames, maxQuestions, model, preview, course) =>
    fetch(`${BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_names: sessionNames, max_questions: maxQuestions, model, preview,
        category: course?.category, course_type: course?.course_type,
      }),
    }).then(j),

  // TESTING: preview mode — resume a paused run through the quality gate
  proceed: (runId) =>
    fetch(`${BASE}/api/proceed/${runId}`, { method: 'POST' }).then(j),

  getResult: (runId) => fetch(`${BASE}/api/result/${runId}`).then(j),

  approve: (runId, acceptedIds, rejectedFeedback, action) =>
    fetch(`${BASE}/api/approve/${runId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accepted_ids: acceptedIds, rejected_feedback: rejectedFeedback, action }),
    }).then(j),

  stream: (runId, onEvent) => {
    const es = new EventSource(`${BASE}/api/stream/${runId}`)
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)) } catch {}
    }
    es.onerror = () => onEvent({ step: 'stream_error', status: 'error', detail: 'Stream disconnected' })
    return es
  },
}
