import { Component } from 'react'
import Icon from './Icon.jsx'

/**
 * Catches render-time errors so one bad payload doesn't white-screen the app.
 *
 * There was no boundary at all, and Review is the most data-dependent screen in the app — it reads
 * nested optional fields off a run payload and does arithmetic on scores. Any missing shape there took
 * the whole SPA down with a blank page and no way back.
 */
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('[ui] render error', error, info?.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="page-content">
        <div className="empty-state">
          <Icon name="alert" size={28} />
          <h2>Something broke while rendering this page</h2>
          <p>The run itself is unaffected — it is saved on the server.</p>
          <pre className="empty-state-detail">{String(error?.message || error)}</pre>
          <div className="empty-state-actions">
            <button className="btn btn-ghost btn-sm" onClick={() => this.setState({ error: null })}>
              <Icon name="refresh" size={13} /> Try again
            </button>
            <a className="btn btn-primary btn-sm" href="/">Back to start</a>
          </div>
        </div>
      </div>
    )
  }
}
