import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import Icon from '../components/Icon.jsx'

const PIPELINE = [
  { icon: 'understand', label: 'Understand' },
  { icon: 'retrieve',   label: 'Retrieve' },
  { icon: 'validate',   label: 'Validate' },
  { icon: 'evaluate',   label: 'Evaluate' },
  { icon: 'gate',     label: 'Quality gate' },
  { icon: 'review',     label: 'Review' },
  { icon: 'export',     label: 'Export' },
]

export default function WelcomePage() {
  const navigate = useNavigate()
  const [topicCount, setTopicCount] = useState(null)
  const [bankCount, setBankCount] = useState(null)
  const [courses, setCourses] = useState([])

  useEffect(() => {
    api.getTopics().then(d => setTopicCount(Object.keys(d.topics || {}).length)).catch(() => {})
    api.getMeta().then(m => setBankCount(m.bank_count)).catch(() => {})
    api.getCourses().then(d => setCourses(d.courses || [])).catch(() => {})
  }, [])

  return (
    <>
      <header className="topbar">
        <div className="topbar-title-group">
          <span className="topbar-title">Questor</span>
          <span className="topbar-sub">Agentic interview-question workflow</span>
        </div>
      </header>

      <div className="page-content">
        {/* Hero */}
        <section className="home-hero">
          <span className="home-eyebrow">AGENTIC · REAL QUESTIONS · EXPORT-READY</span>
          <h1 className="home-hero-title">Questor</h1>
          <p className="home-hero-tagline">
            Curate real, company-attributed interview questions for any course topic — retrieved from a
            verified bank and the web, validated against what the session actually teaches, and handed to
            you review-ready for the NxtMock portal.
          </p>
        </section>

        {/* Stats */}
        <div className="home-stats">
          <div className="home-stat-card">
            <span className="home-stat-num">{topicCount ?? '—'}</span>
            <span className="home-stat-label">Topics available</span>
          </div>
          <div className="home-stat-card">
            <span className="home-stat-num">{bankCount != null ? bankCount.toLocaleString() : '—'}</span>
            <span className="home-stat-label">Questions indexed</span>
          </div>
          <div className="home-stat-card">
            <span className="home-stat-num">4</span>
            <span className="home-stat-label">AI agents in pipeline</span>
          </div>
        </div>

        {/* Mini pipeline */}
        <section className="home-pipeline">
          {PIPELINE.map((s, i) => (
            <div key={s.label} className="hp-wrap">
              <div className="hp-node">
                <span className="hp-ico"><Icon name={s.icon} size={15} /></span>
                <span className="hp-label">{s.label}</span>
              </div>
              {i < PIPELINE.length - 1 && <Icon name="chevronRight" size={12} className="hp-arrow" />}
            </div>
          ))}
        </section>

        {/* Courses */}
        <section className="card">
          <div className="card-title-row">
            <h2 className="card-title" style={{ marginBottom: 0 }}>Courses</h2>
            <button className="btn btn-primary btn-sm" onClick={() => navigate('/add')}>＋ Add course</button>
          </div>
          <div className="course-grid">
            {courses.map(c => (
              <div key={c.id} className="course-card">
                <span className="course-name">{c.name}</span>
                <span className="course-cat">{c.category}</span>
                {c.builtin && <span className="course-badge">built-in</span>}
              </div>
            ))}
            {courses.length === 0 && (
              <div className="empty-state">
                <Icon name="book" size={22} />
                <p>No courses yet. Add one to start generating question sets.</p>
              </div>
            )}
          </div>
          <p className="muted" style={{ fontSize: '0.76rem', marginTop: '0.6rem' }}>
            Pick a course &amp; topic from the sidebar to generate. Add your own via “Add course”.
          </p>
        </section>

        {/* Getting started */}
        <div className="home-hint">
          <span className="home-hint-step">1</span> Pick a topic in the sidebar
          <span className="home-hint-sep">·</span>
          <span className="home-hint-step">2</span> Choose model &amp; question count
          <span className="home-hint-sep">·</span>
          <span className="home-hint-step">3</span> Generate &amp; review
        </div>
      </div>
    </>
  )
}
