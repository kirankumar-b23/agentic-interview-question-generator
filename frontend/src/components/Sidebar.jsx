import { useState, useEffect, useMemo } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { api } from '../lib/api.js'
import Icon from './Icon.jsx'

export default function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()

  const [courses, setCourses] = useState([{ id: 'gen_ai', name: 'Gen AI', category: 'GEN_AI', course_type: 'mixed', builtin: true }])
  const [selectedCourse, setSelectedCourse] = useState('gen_ai')
  const [topics, setTopics] = useState({})
  const [selectedTopic, setSelectedTopic] = useState('')
  const [selectMode, setSelectMode] = useState('topic')   // 'topic' | 'session'
  const [selectedUnit, setSelectedUnit] = useState('')
  const [maxQuestions, setMaxQuestions] = useState(12)
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
    setSelectedUnit('')
  }, [selectedCourse])

  // Flat, de-duplicated list of every unit/session in the current course.
  const units = useMemo(() => [...new Set(Object.values(topics).flat())], [topics])

  useEffect(() => {
    api.getHistory().then(d => setHistory(d.runs || [])).catch(() => {})
    api.getCourses().then(d => { if (d.courses?.length) setCourses(d.courses) }).catch(() => {})
  }, [location.pathname])

  const courseObj = courses.find(c => c.id === selectedCourse)

  const canGen = selectMode === 'topic' ? !!selectedTopic : !!selectedUnit

  async function handleGenerate() {
    if (!canGen || starting) return
    setStarting(true)
    try {
      const sessionNames = selectMode === 'topic' ? (topics[selectedTopic] || []) : [selectedUnit]
      if (!sessionNames.length || !sessionNames[0]) return
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
      <button className="sidebar-logo" onClick={() => navigate('/')} aria-label="NxtMock home">
        <span className="sidebar-logo-main">NxtMock</span>
        <span className="sidebar-logo-sub">Interview Generator</span>
      </button>

      {/* Course + Topic pickers */}
      <div className="sidebar-picker">
        <div className="sidebar-picker-head">
          <label className="sidebar-section-label sidebar-label-flush" htmlFor="sb-course">Course</label>
          <button className="sidebar-view-all" onClick={() => navigate('/add')}>
            <Icon name="plus" size={12} /> Add
          </button>
        </div>
        <div className="sidebar-select-wrap">
          <select
            id="sb-course"
            className="sidebar-topic-select"
            value={selectedCourse}
            onChange={e => setSelectedCourse(e.target.value)}
          >
            {courses.map(c => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>

        {/* Choose whether to generate for a whole topic or a single unit */}
        <span className="sidebar-section-label">Select by</span>
        <div className="sidebar-mode-radios">
          <label>
            <input
              type="radio" name="selmode" checked={selectMode === 'topic'}
              onChange={() => { setSelectMode('topic'); setSelectedUnit('') }}
            />
            Topic-wise
          </label>
          <label>
            <input
              type="radio" name="selmode" checked={selectMode === 'session'}
              onChange={() => { setSelectMode('session'); setSelectedTopic('') }}
            />
            Session-wise
          </label>
        </div>

        {selectMode === 'topic' ? (
          <>
            <label className="sidebar-section-label" htmlFor="sb-topic">Topic</label>
            <div className="sidebar-select-wrap">
              <select
                id="sb-topic"
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
                <span className="sidebar-section-label sidebar-label-spaced">
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
          </>
        ) : (
          <>
            <label className="sidebar-section-label" htmlFor="sb-unit">Unit</label>
            <div className="sidebar-select-wrap">
              <select
                id="sb-unit"
                className="sidebar-topic-select"
                value={selectedUnit}
                onChange={e => setSelectedUnit(e.target.value)}
              >
                <option value="">{units.length ? 'Select a unit…' : 'No units in this course'}</option>
                {units.map(u => (
                  <option key={u} value={u}>{u}</option>
                ))}
              </select>
            </div>
          </>
        )}
      </div>

      {/* Generate area */}
      <div className="sidebar-generate-area">
        <div className="sidebar-maxq-label">
          Target count: <strong>{maxQuestions}</strong>
          <span className="sidebar-maxq-hint"> (all relevant questions are kept)</span>
        </div>
        <input
          id="sb-count" type="range" min={5} max={40} value={maxQuestions}
          aria-valuetext={`${maxQuestions} questions`}
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
          disabled={!canGen || starting}
          onClick={handleGenerate}
        >
          {starting
            ? 'Starting…'
            : canGen
              ? 'Generate questions'
              : selectMode === 'topic' ? 'Select a topic first' : 'Select a unit first'}
        </button>
      </div>

      {/* Recent Runs — successful only (History page shows all) */}
      <div className="sidebar-history">
        <div className="sidebar-history-header">
          <span className="sidebar-history-label">Recent Runs</span>
          {history.length > 0 && (
            <button className="sidebar-view-all" onClick={() => navigate('/history')}>View all</button>
          )}
        </div>
        {(() => {
          const successful = history.filter(r => (r.question_count || 0) > 0)
          if (successful.length === 0) return <div className="sidebar-empty-hist">No successful runs yet</div>
          return successful.slice(0, 6).map(run => (
            <button
              key={run.run_id}
              className="sidebar-run-item"
              onClick={() => navigate(`/review/${run.run_id}`)}
            >
              <span className="sidebar-run-name" title={run.session_name}>{run.topic || run.session_name}</span>
              <span className="sidebar-run-meta">{run.question_count}q · {run.run_id.slice(0, 7)}</span>
            </button>
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
        <div className="sidebar-model-via">via OpenRouter</div>
        <div className="sidebar-model-via">
          {meta?.credits?.remaining != null
            ? `Credits: $${meta.credits.remaining.toFixed(2)} left${meta.credits.scope === 'key' ? ' (key)' : ''}`
            : 'Credits: —'}
        </div>
        <button className="theme-toggle" onClick={toggleTheme} title="Toggle light / dark theme">
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={13} />
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
      </div>
    </aside>
  )
}
