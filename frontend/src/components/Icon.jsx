/**
 * Inline SVG icon set.
 *
 * Replaces the emoji glyphs this app used as its icon system. Emoji is the single strongest
 * "prototype" signal in a product UI: it renders differently per platform, can't take the theme's
 * colour, can't be sized against the type scale, and reads as decoration rather than instrumentation.
 *
 * The set is deliberately narrow and consistent — one 24px grid, 1.5 stroke, round caps, no fills —
 * so icons read as engraved marks on an instrument panel rather than illustrations. Everything uses
 * `currentColor`, so an icon inherits its meaning from the text colour around it and both themes work
 * with no per-icon overrides.
 */

const PATHS = {
  // ── Pipeline stages ──
  understand: 'M12 3a4 4 0 0 0-4 4v1a3 3 0 0 0 0 6v1a4 4 0 0 0 8 0v-1a3 3 0 0 0 0-6V7a4 4 0 0 0-4-4Z M12 3v18',
  retrieve: 'M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z M4 7v5c0 1.7 3.6 3 8 3s8-1.3 8-3V7 M4 12v5c0 1.7 3.6 3 8 3s8-1.3 8-3v-5',
  validate: 'M3 5h18 M6 12h12 M10 19h4',
  evaluate: 'M12 3v3 M5 6h14 M7 6l-3 7h6L7 6Z M17 6l-3 7h6l-3-7Z M8 21h8 M12 6v15',
  gate: 'M12 3 4 6v6c0 4.4 3.2 8.2 8 9 4.8-.8 8-4.6 8-9V6l-8-3Z M9 12l2 2 4-4',
  review: 'M16 20v-1a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v1 M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z M16 11l2 2 4-4',
  export: 'M12 15V3 M8 7l4-4 4 4 M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4',

  // ── Tools ──
  bank: 'M4 5.5A2.5 2.5 0 0 1 6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5v-13Z M4 18.5A2.5 2.5 0 0 1 6.5 16H20 M9 7h7',
  web: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M3 12h18 M12 3c2.5 2.4 4 5.6 4 9s-1.5 6.6-4 9c-2.5-2.4-4-5.6-4-9s1.5-6.6 4-9Z',
  github: 'M9 20c-4.5 1.4-4.5-2.3-6-3m12 5v-3.6c0-1 .1-1.4-.5-2 2.7-.3 4.5-1.4 4.5-5a4 4 0 0 0-1.1-2.8 3.7 3.7 0 0 0-.1-2.8s-1.1-.3-3.5 1.3a13 13 0 0 0-6.6 0C5.3 3.5 4.2 3.8 4.2 3.8a3.7 3.7 0 0 0-.1 2.8A4 4 0 0 0 3 9.4c0 3.6 1.8 4.7 4.5 5-.6.6-.6 1.2-.5 2V20',
  target: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z M12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z',
  dedupe: 'M8 8h11a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z M16 5V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h1',
  balance: 'M12 3v18 M5 8h14 M5 8 2 15h6L5 8Z M19 8l-3 7h6l-3-7Z',
  coverage: 'M3 20h18 M6 20V10 M11 20V4 M16 20v-7 M21 20v-11',
  remove: 'M4 7h16 M10 11v6 M14 11v6 M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12 M9 7V4h6v3',
  submit: 'M4 12l5 5L20 6',
  filter: 'M3 5h18l-7 8v6l-4 2v-8L3 5Z',
  suppress: 'M4 4l16 16 M12 20a8 8 0 1 1 8-8 8 8 0 0 1-8 8Z',

  // ── Status ──
  check: 'M4 12l5 5L20 6',
  x: 'M6 6l12 12 M18 6 6 18',
  alert: 'M12 4 2.5 20h19L12 4Z M12 10v4 M12 17.5v.5',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 11v6 M12 7.5v.5',
  clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 7v5l3.5 2',
  spinner: 'M12 3v4 M12 17v4 M3 12h4 M17 12h4 M5.6 5.6l2.9 2.9 M15.5 15.5l2.9 2.9 M5.6 18.4l2.9-2.9 M15.5 8.5l2.9-2.9',
  zap: 'M13 2 4 14h7l-1 8 9-12h-7l1-8Z',

  // ── Navigation & chrome ──
  chevronRight: 'M9 6l6 6-6 6',
  chevronDown: 'M6 9l6 6 6-6',
  arrowLeft: 'M19 12H5 M11 6l-6 6 6 6',
  arrowRight: 'M5 12h14 M13 6l6 6-6 6',
  external: 'M14 4h6v6 M20 4l-9 9 M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5',
  plus: 'M12 5v14 M5 12h14',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7 M20 5v6h-6',
  history: 'M3 12a9 9 0 1 0 3-6.7 M3 4v4h4 M12 8v4l3 2',
  sun: 'M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z M12 2v2 M12 20v2 M2 12h2 M20 12h2 M4.9 4.9l1.5 1.5 M17.6 17.6l1.5 1.5 M4.9 19.1l1.5-1.5 M17.6 6.4l1.5-1.5',
  moon: 'M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z',
  keyboard: 'M3 7h18a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z M7 11h.01 M11 11h.01 M15 11h.01 M8 14h8',
  layers: 'M12 3 3 8l9 5 9-5-9-5Z M3 13l9 5 9-5 M3 17.5l9 5 9-5',
  book: 'M4 5.5A2.5 2.5 0 0 1 6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5v-13Z M4 18.5A2.5 2.5 0 0 1 6.5 16H20',
  cpu: 'M6 6h12v12H6z M9 9h6v6H9z M12 2v4 M12 18v4 M2 12h4 M18 12h4',
}

// Tool/step name → icon, so the transcript never has to guess. Names match the SSE `step` field.
export const STEP_ICONS = {
  understand_session: 'understand',
  search_question_bank: 'bank',
  search_web_questions: 'web',
  search_github_questions: 'github',
  tavily_health: 'web',
  validate_relevance: 'target',
  deduplicate_questions: 'dedupe',
  suppress_rejected: 'suppress',
  session_fit: 'filter',
  prefilter: 'filter',
  check_difficulty_balance: 'balance',
  check_outcome_coverage: 'coverage',
  remove_question: 'remove',
  submit_question_set: 'submit',
  critique: 'gate',
  complete: 'check',
  error: 'alert',
  heartbeat: 'clock',
}

// Agent/phase → icon, for the transcript's group headers and the stepper.
export const AGENT_ICONS = {
  understanding: 'understand',
  retrieval: 'retrieve',
  validation: 'validate',
  evaluation: 'evaluate',
  gate: 'gate',
  review: 'review',
  export: 'export',
  other: 'cpu',
}

/**
 * @param {string} name  key from PATHS
 * @param {number} size  px (matches the surrounding type size; 24-grid scales cleanly)
 * @param {string} title accessible name — omit for purely decorative icons beside visible text
 */
export default function Icon({ name, size = 16, title, className = '', strokeWidth = 1.5, ...rest }) {
  const d = PATHS[name]
  if (!d) return null
  return (
    <svg
      className={`icon${className ? ` ${className}` : ''}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
      {...rest}
    >
      {title && <title>{title}</title>}
      {d.split(' M').map((seg, i) => (
        <path key={i} d={i === 0 ? seg : `M${seg}`} />
      ))}
    </svg>
  )
}
