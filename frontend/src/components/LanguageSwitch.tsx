import React from 'react'
import type { LangCode } from '../api'

interface Props {
  value: LangCode
  onChange: (l: LangCode) => void
}

const LANGS: { code: LangCode; label: string; script: string }[] = [
  { code: 'hi', label: 'Hindi',   script: 'हिं' },
  { code: 'kn', label: 'Kannada', script: 'ಕನ್ನಡ' },
  { code: 'ta', label: 'Tamil',   script: 'தமிழ்' },
  { code: 'en', label: 'English', script: 'EN' },
]

const styles: Record<string, React.CSSProperties> = {
  row: {
    display: 'flex',
    gap: '6px',
  },
  btn: {
    padding: '6px 14px',
    border: '1.5px solid var(--rule-dark)',
    borderRadius: 'var(--radius)',
    background: 'transparent',
    cursor: 'pointer',
    fontFamily: 'inherit',
    fontSize: '14px',
    color: 'var(--ink-muted)',
    transition: 'all 0.15s',
    lineHeight: 1.4,
  },
  active: {
    borderColor: 'var(--oxblood)',
    color: 'var(--oxblood)',
    fontWeight: 600,
    background: 'rgba(122,26,26,0.06)',
  },
}

export function LanguageSwitch({ value, onChange }: Props) {
  return (
    <div style={styles.row} role="group" aria-label="Select language">
      {LANGS.map((l) => (
        <button
          key={l.code}
          onClick={() => onChange(l.code)}
          title={l.label}
          aria-pressed={value === l.code}
          style={{
            ...styles.btn,
            ...(value === l.code ? styles.active : {}),
          }}
          className={`lang-${l.code}`}
        >
          {l.script}
        </button>
      ))}
    </div>
  )
}
