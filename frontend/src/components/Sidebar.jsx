import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { api } from '../lib/api.js'

export default function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()

  const [courses, setCourses] = useState([{ id: 'gen_ai', name: 'Gen AI', category: 'GEN_AI', course_type: 'mixed', builtin: true }])
  const [selectedCourse, setSelectedCourse] = useState('gen_ai')
  const [topics, setTopics] = useState({})
  const [selectedTopic, setSelectedTopic] = useState('')
  const [maxQuestions, setMaxQuestions] = useState(7)
  const [preview, setPreview] = useState(false)   // TESTING: preview before quality gate
  const [starting, setStarting] = useState(false)
  const [history, setHistory] = useState([])
  const [meta, setMeta] = useState(null)
  const [selectedModel, setSelectedModel] = useState(() => {
    try { return localStorage.getItem('iqg-model') || '' } catch { return '' }
  })
  const [theme, setTheme] = useState(
    () => (typeof document !== 'undefined' && document.documentElement.dataset.theme) || 'dark'
  )

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.dataset.theme = next
    try { localStorage.setItem('iqg-theme', next) } catch {}
  }

  function pickModel(id) {
    setSelectedModel(id)
    try { localStorage.setItem('iqg-model', id) } catch {}
  }

  useEffect(() => {
    api.getCourses().then(d => { if (d.courses?.length) setCourses(d.courses) }).catch(() => {})
    api.getHistory().then(d => setHistory(d.runs || [])).catch(() => {})
    api.getMeta().then(m => {
      setMeta(m)
      setSelectedModel(prev => prev || m.model || '')  // default to active model
    }).catch(() => {})
  }, [])

  // Load topics whenever the selected course changes (also refetch on navigation
  // so a newly-added course/session shows up).
  useEffect(() => {
    api.getTopics(selectedCourse).then(d => setTopics(d.topics || {})).catch(() => {})
    setSelectedTopic('')
  }, [selectedCourse])

  useEffect(() => {
    api.getHistory().then(d => setHistory(d.runs || [])).catch(() => {})
    api.getCourses().then(d => { if (d.courses?.length) setCourses(d.courses) }).catch(() => {})
  }, [location.pathname])

  const courseObj = courses.find(c => c.id === selectedCourse)

  async function handleGenerate() {
    if (!selectedTopic || starting) return
    setStarting(true)
    try {
      const sessionNames = topics[selectedTopic] || []
      const { run_id } = await api.generate(sessionNames, maxQuestions, selectedModel || undefined, preview, courseObj)
      navigate(`/progress/${run_id}`)
    } catch {
      // stay on page
    } finally {
      setStarting(false)
    }
  }

  const sessions = selectedTopic ? (topics[selectedTopic] || []) : []

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo" onClick={() => navigate('/')}>
        <span className="sidebar-logo-main">NxtMock</span>
        <span className="sidebar-logo-sub">Interview Generator</span>
      </div>

      {/* Course + Topic pickers */}
      <div className="sidebar-picker">
        <div className="sidebar-picker-head">
          <span className="sidebar-section-label" style={{ padding: 0 }}>Course</span>
          <span className="sidebar-view-all" onClick={() => navigate('/add')}>＋ Add</span>
        </div>
        <div className="sidebar-select-wrap">
          <select
            className="sidebar-topic-select"
            value={selectedCourse}
            onChange={e => setSelectedCourse(e.target.value)}
          >
            {courses.map(c => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>

        <span className="sidebar-section-label">Topic</span>
        <div className="sidebar-select-wrap">
          <select
            className="sidebar-topic-select"
            value={selectedTopic}
            onChange={e => setSelectedTopic(e.target.value)}
          >
            <option value="">{Object.keys(topics).length ? 'Select a topic…' : 'No topics in this course'}</option>
            {Object.keys(topics).map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Sessions included in this topic — info only */}
        {sessions.length > 0 && (
          <>
            <span className="sidebar-section-label" style={{ marginTop: '0.5rem' }}>
              Sessions included ({sessions.length})
            </span>
            <div className="sidebar-session-list">
              {sessions.map(s => (
                <div key={s} className="sidebar-session-info" title={s}>
                  · {s}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Generate area */}
      <div className="sidebar-generate-area">
        <div className="sidebar-maxq-label">
          Max questions: <strong>{maxQuestions}</strong>
        </div>
        <input
          type="range" min={5} max={15} value={maxQuestions}
          onChange={e => setMaxQuestions(+e.target.value)}
          className="sidebar-range"
        />
        {/* TESTING: preview before quality gate */}
        <label className="sidebar-preview-toggle">
          <input type="checkbox" checked={preview} onChange={e => setPreview(e.target.checked)} />
          Preview picks before quality gate
        </label>
        <button
          className="sidebar-gen-btn"
          disabled={!selectedTopic || starting}
          onClick={handleGenerate}
        >
          {starting ? 'Starting…' : selectedTopic ? 'Generate Questions ▸' : 'Select a topic first'}
        </button>
      </div>

      {/* Recent Runs — successful only (History page shows all) */}
      <div className="sidebar-history">
        <div className="sidebar-history-header">
          <span className="sidebar-history-label">Recent Runs</span>
          {history.length > 0 && (
            <span className="sidebar-view-all" onClick={() => navigate('/history')}>View all</span>
          )}
        </div>
        {(() => {
          const successful = history.filter(r => (r.question_count || 0) > 0)
          if (successful.length === 0) return <div className="sidebar-empty-hist">No successful runs yet</div>
          return successful.slice(0, 6).map(run => (
            <div
              key={run.run_id}
              className="sidebar-run-item"
              onClick={() => navigate(`/review/${run.run_id}`)}
            >
              <span className="sidebar-run-name" title={run.session_name}>{run.topic || run.session_name}</span>
              <span className="sidebar-run-meta">{run.question_count}q · {run.run_id.slice(0, 7)}</span>
            </div>
          ))
        })()}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-model-label">Model</div>
        {meta?.models?.length > 0 ? (
          <select
            className="sidebar-topic-select"
            value={selectedModel}
            onChange={e => pickModel(e.target.value)}
          >
            {meta.models.map(m => (
              <option key={m.id} value={m.id}>{m.label || m.id}</option>
            ))}
          </select>
        ) : (
          <span className="sidebar-model-chip">
            {(selectedModel || meta?.model || '').replace(/^.*\//, '') || '—'}
          </span>
        )}
        <div className="sidebar-model-via" style={{ marginTop: '0.3rem' }}>via OpenRouter</div>
        <div className="sidebar-model-via" style={{ marginTop: '0.35rem' }}>
          {meta?.credits?.remaining != null
            ? `Credits: $${meta.credits.remaining.toFixed(2)} left${meta.credits.scope === 'key' ? ' (key)' : ''}`
            : 'Credits: —'}
        </div>
        <button className="theme-toggle" onClick={toggleTheme} title="Toggle light / dark theme">
          <span className="tt-ico">{theme === 'dark' ? '☀️' : '🌙'}</span>
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
      </div>
    </aside>
  )
}
