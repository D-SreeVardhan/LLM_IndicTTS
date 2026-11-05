import React, { useState } from 'react'
import type { StageTrace } from '../api'

interface Props {
  trace: StageTrace[]
  totalMs: number
}

const STAGE_ICONS: Record<string, string> = {
  translate_to_en:   '🔤',
  tool_router:       '🔧',
  retrieve:          '🔍',
  generate:          '✍',
  guard:             '🛡',
  translate_from_en: '🔤',
}

export function PipelineTrace({ trace, totalMs }: Props) {
  const [open, setOpen] = useState(false)

  if (!trace.length) return null

  return (
    <div style={styles.wrapper}>
      <button style={styles.toggle} onClick={() => setOpen((o) => !o)}>
        <span style={styles.toggleLabel}>Pipeline trace</span>
        <span style={styles.toggleRight}>
          <span style={styles.totalMs}>{totalMs.toFixed(0)} ms total</span>
          <span style={styles.chevron}>{open ? '▲' : '▼'}</span>
        </span>
      </button>

      {open && (
        <div style={styles.body}>
          {trace.map((s, i) => (
            <div key={i} style={styles.row}>
              <span style={styles.icon}>{STAGE_ICONS[s.stage] ?? '◈'}</span>
              <span style={styles.stageName}>{s.stage.replace(/_/g, ' ')}</span>
              {s.detail && <span style={styles.detail}>{s.detail}</span>}
              <span style={styles.ms}>{s.latency_ms.toFixed(1)} ms</span>
              <div
                style={{
                  ...styles.bar,
                  width: `${Math.min(100, (s.latency_ms / totalMs) * 100)}%`,
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    marginTop: '12px',
    border: '1px solid var(--rule)',
    borderRadius: 'var(--radius)',
    overflow: 'hidden',
  },
  toggle: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    width: '100%', padding: '8px 14px',
    background: 'var(--paper-dark)', border: 'none',
    cursor: 'pointer', fontFamily: 'var(--font-display)',
  },
  toggleLabel: { fontSize: '11px', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)' },
  toggleRight: { display: 'flex', alignItems: 'center', gap: '10px' },
  totalMs: { fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--turmeric)' },
  chevron: { fontSize: '10px', color: 'var(--ink-muted)' },
  body: { padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: '6px', background: '#faf8f3' },
  row: {
    display: 'grid',
    gridTemplateColumns: '20px 1fr auto auto',
    alignItems: 'center', gap: '8px',
    fontSize: '12px',
    position: 'relative',
  },
  icon: { fontSize: '13px', lineHeight: 1 },
  stageName: { color: 'var(--ink)', fontWeight: 500, textTransform: 'capitalize' },
  detail: { fontSize: '11px', color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px' },
  ms: { fontFamily: 'var(--font-mono)', color: 'var(--turmeric)', fontSize: '11px', whiteSpace: 'nowrap' },
  bar: {
    position: 'absolute', bottom: '-2px', left: '28px',
    height: '2px', background: 'var(--oxblood)', opacity: 0.35,
    borderRadius: '1px', transition: 'width 0.3s',
  },
}
