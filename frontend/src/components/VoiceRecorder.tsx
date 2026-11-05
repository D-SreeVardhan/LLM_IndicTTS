import React, { useCallback, useEffect, useRef, useState } from 'react'

interface Props {
  onRecorded: (blob: Blob) => void
  disabled?: boolean
}

const SAMPLE_RATE = 16000
const WAVEFORM_BARS = 30

export function VoiceRecorder({ onRecorded, disabled }: Props) {
  const [recording, setRecording] = useState(false)
  const [bars, setBars] = useState<number[]>(Array(WAVEFORM_BARS).fill(2))
  const mediaRef    = useRef<MediaRecorder | null>(null)
  const chunksRef   = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animRef     = useRef<number>(0)
  const streamRef   = useRef<MediaStream | null>(null)

  const stopAnim = () => {
    cancelAnimationFrame(animRef.current)
    setBars(Array(WAVEFORM_BARS).fill(2))
  }

  const startAnim = (analyser: AnalyserNode) => {
    const data = new Uint8Array(analyser.frequencyBinCount)
    const tick = () => {
      analyser.getByteFrequencyData(data)
      const step = Math.floor(data.length / WAVEFORM_BARS)
      const newBars = Array.from({ length: WAVEFORM_BARS }, (_, i) => {
        const avg = data.slice(i * step, (i + 1) * step).reduce((a, b) => a + b, 0) / step
        return Math.max(2, Math.round((avg / 255) * 48))
      })
      setBars(newBars)
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
  }

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
      const src = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      src.connect(analyser)
      analyserRef.current = analyser

      chunksRef.current = []
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        onRecorded(blob)
        stream.getTracks().forEach((t) => t.stop())
        ctx.close()
      }
      mr.start(100)
      mediaRef.current = mr
      setRecording(true)
      startAnim(analyser)
    } catch (err) {
      console.error('Mic error', err)
    }
  }, [onRecorded])

  const stop = useCallback(() => {
    mediaRef.current?.stop()
    setRecording(false)
    stopAnim()
  }, [])

  useEffect(() => () => { stopAnim(); streamRef.current?.getTracks().forEach((t) => t.stop()) }, [])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault()
      recording ? stop() : start()
    }
  }

  return (
    <div style={styles.wrapper}>
      {/* Waveform */}
      <div
        style={styles.waveform}
        aria-hidden="true"
        data-recording={recording}
      >
        {bars.map((h, i) => (
          <span
            key={i}
            style={{ ...styles.bar, height: `${h}px`, opacity: recording ? 1 : 0.25 }}
          />
        ))}
      </div>

      {/* Button */}
      <button
        style={{
          ...styles.btn,
          ...(recording ? styles.btnActive : {}),
          ...(disabled ? styles.btnDisabled : {}),
        }}
        onClick={recording ? stop : start}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
        title={recording ? 'Release to send' : 'Hold to speak'}
      >
        <MicIcon recording={recording} />
        <span style={styles.label}>{recording ? 'Recording…' : 'Hold & Speak'}</span>
      </button>
    </div>
  )
}

function MicIcon({ recording }: { recording: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="2" width="6" height="11" rx="3" fill={recording ? 'currentColor' : 'none'} />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8"  y1="23" x2="16" y2="23" />
    </svg>
  )
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' },
  waveform: {
    display: 'flex', alignItems: 'center', gap: '3px',
    height: '52px', padding: '0 4px',
  },
  bar: {
    width: '3px', borderRadius: '2px',
    background: 'var(--oxblood)',
    transition: 'height 0.06s ease',
    display: 'inline-block',
  },
  btn: {
    display: 'flex', alignItems: 'center', gap: '10px',
    padding: '12px 28px',
    background: 'var(--paper-dark)',
    border: '2px solid var(--rule-dark)',
    borderRadius: '40px',
    cursor: 'pointer',
    fontFamily: 'var(--font-display)',
    fontSize: '15px',
    color: 'var(--ink)',
    transition: 'all 0.15s',
    userSelect: 'none',
  },
  btnActive: {
    background: 'var(--oxblood)',
    borderColor: 'var(--oxblood)',
    color: '#fff',
    boxShadow: '0 0 0 4px rgba(122,26,26,0.15)',
  },
  btnDisabled: { opacity: 0.45, cursor: 'not-allowed' },
  label: { fontWeight: 600, letterSpacing: '0.02em' },
}
