import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getSource, speakText } from '../api'
import type { AskResponse, Citation, LangCode } from '../api'

interface Props {
  response: AskResponse
  lang: LangCode
}

export function AnswerCard({ response, lang }: Props) {
  const [audioUrl, setAudioUrl]   = useState<string | null>(null)
  const [loadingAudio, setLoadingAudio] = useState(false)
  const [audioError, setAudioError]     = useState(false)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  const handlePlay = async () => {
    if (audioUrl) {
      audioRef.current?.play()
      return
    }
    setLoadingAudio(true)
    setAudioError(false)
    try {
      const url = await speakText(response.answer, lang)
      setAudioUrl(url)
      setLoadingAudio(false)
      setTimeout(() => {
        audioRef.current?.play()
      }, 100)
    } catch {
      setAudioError(true)
      setLoadingAudio(false)
    }
  }

  const langClass = `lang-${lang}`

  return (
    <div style={styles.card} data-refused={response.refused}>
      {/* Header bar */}
      <div style={styles.header}>
        <span style={styles.headerLabel}>
          {response.refused ? '⚠ Could not find a grounded answer' : 'Answer'}
        </span>
        <div style={styles.headerRight}>
          <span style={styles.latency}>{response.total_latency_ms.toFixed(0)} ms</span>

          {/* Play button */}
          <button
            style={{ ...styles.iconBtn, ...(loadingAudio ? styles.iconBtnLoading : {}) }}
            onClick={handlePlay}
            disabled={loadingAudio}
            title="Listen to answer"
            aria-label="Play audio"
          >
            {loadingAudio ? '…' : '▶'}
          </button>
        </div>
      </div>

      {/* Answer text — render as markdown so tables, bold, bullets all display correctly */}
      <div className={`${langClass} md-answer`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {response.answer}
        </ReactMarkdown>
      </div>

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
      onPlay={() => {}}
      onEnded={() => {}}
          controls
          style={styles.audio}
        />
      )}
      {loadingAudio && <p style={styles.audioNote}>Preparing audio — this takes about a minute…</p>}
      {audioError && <p style={styles.audioErr}>Audio generation failed.</p>}

      {/* Citations */}
      {response.citations.length > 0 && (
        <div style={styles.citations}>
          <span style={styles.citLabel}>Sources</span>
          <div style={styles.citRow}>
            {response.citations.map((c) => (
              <CitationChip key={c.chunk_id} citation={c} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}


function CitationChip({ citation }: { citation: Citation }) {
  const [open, setOpen]     = useState(false)
  const [full, setFull]     = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const toggle = async () => {
    if (!open && !full && !citation.chunk_id.startsWith('tool_')) {
      setLoading(true)
      try {
        const src = await getSource(citation.chunk_id)
        setFull(src.text)
      } catch {
        setFull(citation.excerpt)
      }
      setLoading(false)
    }
    setOpen((o) => !o)
  }

  return (
    <div style={styles.chip}>
      <button style={styles.chipBtn} onClick={toggle}>
        {citation.title.length > 48 ? citation.title.slice(0, 45) + '…' : citation.title}
        <span style={styles.chipArrow}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={styles.chipBody}>
          {loading ? (
            <em>Loading…</em>
          ) : (
            <>
              <p style={styles.chipText}>{full ?? citation.excerpt}</p>
              {citation.source_url && (
                <a
                  href={citation.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={styles.chipLink}
                >
                  View source ↗
                </a>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    background: '#fff',
    border: '1.5px solid var(--rule)',
    borderRadius: 'var(--radius-lg)',
    padding: '20px 24px',
    marginTop: '20px',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: '12px',
  },
  headerLabel: {
    fontWeight: 700, fontSize: '11px', letterSpacing: '0.12em',
    textTransform: 'uppercase', color: 'var(--ink-muted)',
  },
  headerRight: { display: 'flex', alignItems: 'center', gap: '10px' },
  latency: {
    fontSize: '11px', color: 'var(--rule-dark)', fontFamily: 'var(--font-mono)',
  },
  iconBtn: {
    background: 'var(--paper-dark)', border: '1.5px solid var(--rule)',
    borderRadius: '50%', width: '32px', height: '32px',
    cursor: 'pointer', fontSize: '12px', color: 'var(--oxblood)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  iconBtnLoading: { opacity: 0.6, cursor: 'wait' },
  answerText: {
    fontSize: '17px', lineHeight: 1.7, color: 'var(--ink)',
    whiteSpace: 'pre-wrap',
  },
  audio: { marginTop: '12px', width: '100%', accentColor: 'var(--oxblood)' },
  audioErr: { marginTop: '8px', color: 'var(--error)', fontSize: '13px' },
  audioNote: { marginTop: '8px', color: 'var(--ink-muted)', fontSize: '13px', fontStyle: 'italic' },
  citations: { marginTop: '20px', borderTop: '1px solid var(--rule)', paddingTop: '14px' },
  citLabel: {
    fontSize: '10px', letterSpacing: '0.12em', textTransform: 'uppercase',
    color: 'var(--ink-muted)', fontWeight: 700, display: 'block', marginBottom: '8px',
  },
  citRow: { display: 'flex', flexDirection: 'column', gap: '6px' },
  chip: { border: '1px solid var(--rule)', borderRadius: 'var(--radius)' },
  chipBtn: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    width: '100%', padding: '8px 12px',
    background: 'var(--paper)',
    border: 'none', cursor: 'pointer',
    fontFamily: 'var(--font-display)', fontSize: '13px', color: 'var(--ink)',
    textAlign: 'left',
  },
  chipArrow: { marginLeft: '8px', color: 'var(--ink-muted)', fontSize: '10px' },
  chipBody: { padding: '12px', background: '#fff', borderTop: '1px solid var(--rule)' },
  chipText: { fontSize: '13px', lineHeight: 1.6, color: 'var(--ink)', whiteSpace: 'pre-wrap' },
  chipLink: {
    display: 'inline-block', marginTop: '8px',
    fontSize: '12px', color: 'var(--oxblood)',
  },
}
