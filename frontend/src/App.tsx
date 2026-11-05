import React, { useCallback, useRef, useState } from 'react'
import { askQuestion, transcribeAudio } from './api'
import type { AskResponse, LangCode } from './api'
import { AnswerCard } from './components/AnswerCard'
import { LanguageSwitch } from './components/LanguageSwitch'
import { PipelineTrace } from './components/PipelineTrace'
import { SuggestedQuestions } from './components/SuggestedQuestions'
import { VoiceRecorder } from './components/VoiceRecorder'

type Status = 'idle' | 'transcribing' | 'answering' | 'error'

const PLACEHOLDER: Record<LangCode, string> = {
  en: 'Type your campus question here…',
  hi: 'अपना प्रश्न यहाँ टाइप करें…',
  kn: 'ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ…',
  ta: 'உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்…',
}

export default function App() {
  const [lang, setLang]             = useState<LangCode>('hi')
  const [query, setQuery]           = useState('')
  const [status, setStatus]         = useState<Status>('idle')
  const [response, setResponse]     = useState<AskResponse | null>(null)
  const [errorMsg, setErrorMsg]     = useState('')
  const [showSuggested, setShowSuggested] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleRecorded = useCallback(async (blob: Blob) => {
    setStatus('transcribing')
    setErrorMsg('')
    try {
      const result = await transcribeAudio(blob, lang)
      setQuery(result.text)
      // Auto-submit after transcription
      await submitQuery(result.text, lang)
    } catch (err: unknown) {
      setErrorMsg((err as Error).message ?? 'Transcription failed')
      setStatus('error')
    }
  }, [lang])

  const submitQuery = useCallback(async (text: string, activeLang: LangCode) => {
    if (!text.trim()) return
    setStatus('answering')
    setShowSuggested(false)
    setErrorMsg('')
    try {
      // Hindi uses both routes; we default to direct here (pivot tested in eval)
      const route = activeLang === 'hi' ? 'direct' : 'pivot'
      const res = await askQuestion(text.trim(), activeLang, route)
      setResponse(res)
      setStatus('idle')
    } catch (err: unknown) {
      setErrorMsg((err as Error).message ?? 'Request failed')
      setStatus('error')
    }
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitQuery(query, lang)
  }

  const handleSuggested = (q: string) => {
    setQuery(q)
    submitQuery(q, lang)
  }

  const handleLangChange = (l: LangCode) => {
    setLang(l)
    setQuery('')
    setResponse(null)
    setShowSuggested(true)
  }

  const busy = status === 'transcribing' || status === 'answering'

  return (
    <div style={layout.page}>
      {/* ── Header ── */}
      <header style={layout.header}>
        <div style={layout.headerInner}>
          <div style={layout.brand}>
            <RVULogo />
            <div>
              <h1 style={layout.title}>RVU Campus Assistant</h1>
              <p style={layout.subtitle}>Ask about fees, exams, admissions & more</p>
            </div>
          </div>
          <LanguageSwitch value={lang} onChange={handleLangChange} />
        </div>
      </header>

      {/* ── Main ── */}
      <main style={layout.main}>
        <div style={layout.container}>

          {/* Voice recorder */}
          <section style={layout.voiceSection}>
            <VoiceRecorder onRecorded={handleRecorded} disabled={busy} />
          </section>

          {/* Divider */}
          <div style={layout.divider}>
            <span style={layout.dividerText}>or type your question</span>
          </div>

          {/* Text input */}
          <form onSubmit={handleSubmit} style={layout.form}>
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={PLACEHOLDER[lang]}
              rows={3}
              className={`lang-${lang}`}
              style={layout.textarea}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e as unknown as React.FormEvent)
                }
              }}
              disabled={busy}
            />
            <button
              type="submit"
              disabled={busy || !query.trim()}
              style={{
                ...layout.submitBtn,
                ...(busy || !query.trim() ? layout.submitBtnDisabled : {}),
              }}
            >
              {status === 'transcribing' ? 'Transcribing…'
               : status === 'answering'  ? 'Thinking…'
               : 'Ask →'}
            </button>
          </form>

          {/* Status */}
          {busy && (
            <p style={layout.statusMsg}>
              {status === 'transcribing' ? '🎙 Converting speech to text…' : '📚 Searching knowledge base…'}
            </p>
          )}

          {/* Error */}
          {status === 'error' && errorMsg && (
            <p style={layout.error}>⚠ {errorMsg}</p>
          )}

          {/* Answer */}
          {response && (
            <>
              <AnswerCard response={response} lang={lang} />
              <PipelineTrace trace={response.trace} totalMs={response.total_latency_ms} />
            </>
          )}

          {/* Suggested questions */}
          {showSuggested && !busy && (
            <section style={layout.suggestedSection}>
              <p style={layout.suggestedLabel}>Suggested questions</p>
              <SuggestedQuestions lang={lang} onSelect={handleSuggested} />
            </section>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer style={layout.footer}>
        <p>
          CS3235 Project · AI4Bharat IndicConformer + IndicTrans2 + BharatGen Param1 + Indic Parler-TTS ·{' '}
          <a href="https://rvu.edu.in" target="_blank" rel="noopener noreferrer">rvu.edu.in</a>
        </p>
      </footer>
    </div>
  )
}

function RVULogo() {
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none" aria-hidden="true">
      <rect width="44" height="44" rx="6" fill="var(--oxblood)" />
      <text x="22" y="30" textAnchor="middle" fontFamily="Georgia, serif" fontWeight="700" fontSize="20" fill="#f5f0e8">RV</text>
    </svg>
  )
}

const layout: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    display: 'flex', flexDirection: 'column',
    background: 'var(--paper)',
  },
  header: {
    borderBottom: '2px solid var(--rule)',
    background: 'var(--paper)',
    position: 'sticky', top: 0, zIndex: 100,
  },
  headerInner: {
    maxWidth: '760px', margin: '0 auto',
    padding: '14px 24px',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    flexWrap: 'wrap', gap: '12px',
  },
  brand: { display: 'flex', alignItems: 'center', gap: '14px' },
  title: {
    fontFamily: 'var(--font-display)',
    fontSize: '22px', fontWeight: 700,
    color: 'var(--oxblood)', lineHeight: 1.2,
  },
  subtitle: {
    fontSize: '13px', color: 'var(--ink-muted)', marginTop: '2px',
  },
  main: { flex: 1, padding: '32px 24px 48px' },
  container: { maxWidth: '720px', margin: '0 auto' },
  voiceSection: {
    display: 'flex', justifyContent: 'center',
    padding: '28px 0 20px',
  },
  divider: {
    display: 'flex', alignItems: 'center', gap: '12px',
    margin: '4px 0 20px',
  },
  dividerText: {
    fontSize: '12px', color: 'var(--rule-dark)',
    textTransform: 'uppercase', letterSpacing: '0.08em',
    whiteSpace: 'nowrap',
    flex: '0 0 auto',
    padding: '0 4px',
  },
  form: { display: 'flex', flexDirection: 'column', gap: '10px' },
  textarea: {
    width: '100%', padding: '14px 16px',
    border: '1.5px solid var(--rule)',
    borderRadius: 'var(--radius-lg)',
    background: '#fff',
    fontFamily: 'inherit', fontSize: '16px', color: 'var(--ink)',
    resize: 'vertical', lineHeight: 1.6,
    outline: 'none',
    transition: 'border-color 0.15s',
  },
  submitBtn: {
    alignSelf: 'flex-end',
    padding: '11px 28px',
    background: 'var(--oxblood)',
    color: '#fff',
    border: 'none', borderRadius: 'var(--radius)',
    cursor: 'pointer',
    fontFamily: 'var(--font-display)', fontSize: '15px', fontWeight: 600,
    letterSpacing: '0.03em',
    transition: 'background 0.15s',
  },
  submitBtnDisabled: { background: 'var(--rule-dark)', cursor: 'not-allowed' },
  statusMsg: {
    textAlign: 'center', color: 'var(--ink-muted)', fontSize: '14px', marginTop: '16px',
  },
  error: {
    marginTop: '16px', padding: '12px 16px',
    background: '#fdf0f0', border: '1px solid #d9a0a0',
    borderRadius: 'var(--radius)', color: 'var(--error)', fontSize: '14px',
  },
  suggestedSection: {
    marginTop: '32px', paddingTop: '24px',
    borderTop: '1px solid var(--rule)',
  },
  suggestedLabel: {
    fontSize: '11px', letterSpacing: '0.1em',
    textTransform: 'uppercase', color: 'var(--ink-muted)',
    fontWeight: 700, marginBottom: '14px',
  },
  footer: {
    borderTop: '1px solid var(--rule)',
    padding: '12px 24px',
    textAlign: 'center',
    fontSize: '12px', color: 'var(--ink-muted)',
  },
}
