/**
 * The page header, in one place.
 *
 * Every page rendered its own copy of this markup — six near-identical blocks that had already
 * drifted apart (one branch dropped the stepper, the title/sub structure varied). It also emitted the
 * page title as a <span>, so four of five pages had no <h1> and the heading order started at <h2>.
 */
export default function TopBar({ title, sub, actions, children }) {
  return (
    <header className="topbar">
      <div className="topbar-title-group">
        <h1 className="topbar-title">{title}</h1>
        {sub && <span className="topbar-sub">{sub}</span>}
      </div>
      {children}
      {actions && <div className="topbar-actions">{actions}</div>}
    </header>
  )
}
