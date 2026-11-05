/**
 * api.ts — typed client for the RVU backend
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export type LangCode = 'hi' | 'kn' | 'ta' | 'en'
export type Route    = 'direct' | 'pivot'

export interface Citation {
  chunk_id: string
  title: string
  source_url?: string
  excerpt: string
}

export interface StageTrace {
  stage: string
  latency_ms: number
  detail?: string
}

export interface AskResponse {
  answer: string
  answer_lang: LangCode
  citations: Citation[]
  refused: boolean
  refusal_reason?: string
  trace: StageTrace[]
  total_latency_ms: number
}

export interface TranscribeResponse {
  text: string
  lang_detected: LangCode
  decode_mode: string
  latency_ms: number
}

export interface SourceResponse {
  chunk_id: string
  title: string
  source_url?: string
  category?: string
  text: string
}

// ── /api/transcribe ────────────────────────────────────────────────────────
export async function transcribeAudio(
  audioBlob: Blob,
  lang: LangCode,
): Promise<TranscribeResponse> {
  const form = new FormData()
  form.append('audio', audioBlob, 'recording.webm')
  form.append('lang', lang)
  form.append('decode_mode', 'rnnt')

  const res = await fetch(`${BASE}/api/transcribe`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Transcribe failed: ${res.statusText}`)
  return res.json()
}

// ── /api/ask ───────────────────────────────────────────────────────────────
export async function askQuestion(
  text: string,
  lang: LangCode,
  route: Route = 'direct',
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang, route }),
  })
  if (!res.ok) throw new Error(`Ask failed: ${res.statusText}`)
  return res.json()
}

// ── /api/speak ─────────────────────────────────────────────────────────────
export async function speakText(text: string, lang: LangCode): Promise<string> {
  const res = await fetch(`${BASE}/api/speak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
  })
  if (!res.ok) throw new Error(`Speak failed: ${res.statusText}`)
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

// ── /api/sources/{id} ─────────────────────────────────────────────────────
export async function getSource(chunkId: string): Promise<SourceResponse> {
  const res = await fetch(`${BASE}/api/sources/${encodeURIComponent(chunkId)}`)
  if (!res.ok) throw new Error(`Source fetch failed: ${res.statusText}`)
  return res.json()
}
