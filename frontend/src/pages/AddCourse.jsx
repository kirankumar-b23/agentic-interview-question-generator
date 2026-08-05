import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import Icon from '../components/Icon.jsx'

const TYPES = [
  { v: 'mixed', l: 'Mixed' },
  { v: 'code_heavy', l: 'Code-heavy' },
  { v: 'theory_heavy', l: 'Theory-heavy' },
]

export default function AddCourse() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('session')   // 'session' | 'course'
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [err, setErr] = useState(null)

  // shared
  const [courseName, setCourseName] = useState('')
  const [category, setCategory] = useState('')
  const [courseType, setCourseType] = useState('mixed')

  // single session
  const [topic, setTopic] = useState('')
  const [sessionName, setSessionName] = useState('')
  const [reading, setReading] = useState('')

  // full course
  const [markdown, setMarkdown] = useState('')

  function reset() { setMsg(null); setErr(null) }

  async function submitSession() {
    reset()
    if (!courseName.trim() || !topic.trim() || !sessionName.trim() || !reading.trim()) {
      setErr('Course, topic, session name and reading material are all required.'); return
    }
    setBusy(true)
    try {
      const r = await api.addCourseSession({
        course_name: courseName, category: category || undefined, course_type: courseType,
        topic, session_name: sessionName, reading_material: reading,
      })
      setMsg(`Added “${sessionName}” to ${courseName}. Select it from the sidebar Course dropdown.`)
      setTopic(''); setSessionName(''); setReading('')
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  async function submitCourse() {
    reset()
    if (!courseName.trim() || !markdown.trim()) {
      setErr('Course name and Markdown content are required.'); return
    }
    setBusy(true)
    try {
      const r = await api.importCourse({
        course_name: courseName, category: category || undefined, course_type: courseType, markdown,
      })
      setMsg(`Imported ${r.sessions} session(s) across ${r.topics} topic(s) into ${courseName}.`)
      setMarkdown('')
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-title-group">
          <span className="topbar-title">Add a Course</span>
          <span className="topbar-sub">Provide reading material — questions are retrieved &amp; validated against it</span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}><Icon name="arrowLeft" size={13} /> Home</button>
      </header>

      <div className="page-content">
        {/* Mode toggle */}
        <div className="addc-modes">
          <button className={`addc-mode ${mode === 'session' ? 'active' : ''}`} onClick={() => { setMode('session'); reset() }}>
            ＋ Single session
          </button>
          <button className={`addc-mode ${mode === 'course' ? 'active' : ''}`} onClick={() => { setMode('course'); reset() }}>
            Full course (paste Markdown)
          </button>
        </div>

        <section className="card addc-form">
          {/* Shared course fields */}
          <label className="field-label" htmlFor="ac-name">Course name</label>
          <input id="ac-name" className="text-input" placeholder="e.g. Full-Stack Development" value={courseName} onChange={e => setCourseName(e.target.value)} />

          <div className="addc-row">
            <div className="addc-grow">
              <label className="field-label" htmlFor="ac-cat">Category tag (for the sheet)</label>
              <input id="ac-cat" className="text-input" placeholder="auto from name (e.g. FULL_STACK)" value={category} onChange={e => setCategory(e.target.value)} />
            </div>
            <div className="addc-type-field">
              <label className="field-label" htmlFor="ac-type">Type</label>
              <select id="ac-type" className="text-input" value={courseType} onChange={e => setCourseType(e.target.value)}>
                {TYPES.map(t => <option key={t.v} value={t.v}>{t.l}</option>)}
              </select>
            </div>
          </div>

          {mode === 'session' ? (
            <>
              <label className="field-label" htmlFor="ac-topic">Topic</label>
              <input id="ac-topic" className="text-input" placeholder="e.g. Databases" value={topic} onChange={e => setTopic(e.target.value)} />
              <label className="field-label" htmlFor="ac-session">Session / unit name</label>
              <input id="ac-session" className="text-input" placeholder="e.g. SQL Basics" value={sessionName} onChange={e => setSessionName(e.target.value)} />
              <label className="field-label" htmlFor="ac-reading">Reading material</label>
              <textarea id="ac-reading" className="addc-textarea" placeholder="Paste this session's reading material / notes…" value={reading} onChange={e => setReading(e.target.value)} />
              <button className="btn btn-primary btn-full" disabled={busy} onClick={submitSession}>
                {busy ? 'Adding…' : 'Add session'}
              </button>
            </>
          ) : (
            <>
              <label className="field-label">
                Course Markdown — use <code>#</code> for a Topic and <code>##</code> for a Session (body = reading material)
              </label>
              <textarea
                className="addc-textarea addc-textarea-lg"
                placeholder={"# Databases\n## SQL Basics\nSELECT, JOINs, indexes…\n## Transactions\nACID, isolation levels…\n\n# Web\n## HTTP & REST\nMethods, status codes…"}
                value={markdown}
                onChange={e => setMarkdown(e.target.value)}
              />
              <button className="btn btn-primary btn-full" disabled={busy} onClick={submitCourse}>
                {busy ? 'Importing…' : 'Import course'}
              </button>
            </>
          )}

          {err && <div className="alert alert-error" style={{ marginTop: '0.8rem' }}>{err}</div>}
          {msg && (
            <div className="alert alert-success" style={{ marginTop: '0.8rem' }}>
              {msg} <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>Go generate ▸</button>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
