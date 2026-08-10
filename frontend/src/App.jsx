import { useEffect, useState } from 'react'
import { BrowserRouter, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import Icon from './components/Icon.jsx'
import Sidebar from './components/Sidebar.jsx'
import AddCourse from './pages/AddCourse.jsx'
import History from './pages/History.jsx'
import Progress from './pages/Progress.jsx'
import Review from './pages/Review.jsx'
import Batch from './pages/Batch.jsx'
import SessionSelector from './pages/SessionSelector.jsx'

/**
 * App shell.
 *
 * The sidebar is a real off-canvas drawer below 900px. Previously it just became `position: static`
 * while sitting BEFORE the main content in the DOM, so on a phone every route opened with ~800px of
 * control panel and pushed the actual page a full viewport below the fold, with no way to collapse it.
 */
function Layout() {
  const [navOpen, setNavOpen] = useState(false)
  const location = useLocation()

  // Close the drawer on navigation; leaving it open over the new page is disorienting.
  useEffect(() => { setNavOpen(false) }, [location.pathname])

  // Escape closes it, matching every other overlay convention.
  useEffect(() => {
    if (!navOpen) return
    const onKey = (e) => { if (e.key === 'Escape') setNavOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen])

  return (
    <div className={`shell${navOpen ? ' shell-nav-open' : ''}`}>
      <button
        className="nav-toggle"
        onClick={() => setNavOpen((o) => !o)}
        aria-label={navOpen ? 'Close navigation' : 'Open navigation'}
        aria-expanded={navOpen}
      >
        <Icon name={navOpen ? 'x' : 'layers'} size={17} />
      </button>
      {navOpen && <div className="nav-scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />}
      <Sidebar />
      <div className="shell-main">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<SessionSelector />} />
          <Route path="/progress/:runId" element={<Progress />} />
          <Route path="/review/:runId" element={<Review />} />
          <Route path="/batch/:batchId" element={<Batch />} />
          <Route path="/history" element={<History />} />
          <Route path="/add" element={<AddCourse />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
