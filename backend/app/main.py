"""
main.py
───────
FastAPI application for the RVU Multilingual Voice Assistant.

Endpoints:
  POST /api/transcribe  – audio bytes → transcript
  POST /api/ask         – {text, lang, route} → answer + citations + trace
  POST /api/speak       – {text, lang} → WAV audio
  GET  /api/sources/{id} – chunk text for citation popover
  GET  /api/health       – readiness probe
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger

from backend.app.config import settings
from backend.app.pipeline import codemix, generate, retrieve
from backend.app.pipeline.orchestrator import run as pipeline_run
from backend.app.pipeline.tts import synthesize, _load as _load_tts
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    LangCode,
    SourceResponse,
    SpeakRequest,
    TranscribeResponse,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# Single-threaded executor — all heavy inference runs serially in one background
# thread, keeping the async event loop free and avoiding MPS thread contention.
_inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rvu-inference")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RVU Voice Assistant …")

    # 1. Corpus + BM25 (always – very fast, <100 ms)
    retrieve._load_corpus()

    # 2. Dense embedder (optional – ~4 s; prevents cold-start lag on first query)
    if settings.keep_embed_resident:
        logger.info("Pre-loading bge-m3 embedder …")
        try:
            retrieve._load_embedder()
        except Exception as exc:
            logger.warning(f"bge-m3 pre-load failed (will load lazily on first query): {exc}")

    # 3. Param-1 LLM (optional – ~9 s on MPS; saves ~9 s on the first LLM query)
    if settings.keep_llm_resident:
        logger.info("Pre-loading Param-1 LLM …")
        generate.warm_up()  # warm_up() already handles its own exceptions gracefully

    # 4. Indic Parler-TTS (optional – ~5 s; saves cold-start on first /api/speak call)
    if settings.keep_tts_resident:
        logger.info("Pre-loading Indic Parler-TTS …")
        try:
            _load_tts()
        except Exception as exc:
            logger.warning(f"TTS pre-load failed (will load lazily on first /api/speak call): {exc}")

    logger.success("RVU Voice Assistant ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RVU Multilingual Voice Assistant",
    description="Campus query assistant for RV University – Hindi, Kannada, Tamil, English",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /api/health ───────────────────────────────────────────────────────────
@app.get("/api/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── /api/transcribe ───────────────────────────────────────────────────────
@app.post("/api/transcribe", response_model=TranscribeResponse, tags=["voice"])
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (WAV, WebM, OGG, MP3)"),
    lang: LangCode = Form("hi"),
    decode_mode: str = Form("rnnt"),
):
    """Convert spoken audio to text using IndicConformer."""
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _inference_pool, partial(codemix.transcribe_in_script, audio_bytes, lang, decode_mode)
    )
    return TranscribeResponse(
        text=result["text"],
        lang_detected=lang,
        decode_mode=result["decode_mode"],
        latency_ms=result["latency_ms"],
    )


# ── /api/ask ──────────────────────────────────────────────────────────────
@app.post("/api/ask", response_model=AskResponse, tags=["qa"])
async def ask(request: AskRequest) -> AskResponse:
    """Answer a campus query in the requested language."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty query text")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_inference_pool, pipeline_run, request)


# ── /api/speak ────────────────────────────────────────────────────────────
@app.post("/api/speak", tags=["voice"])
async def speak(request: SpeakRequest):
    """Synthesise text to speech using Indic Parler-TTS."""
    loop = asyncio.get_event_loop()
    audio_bytes, _ = await loop.run_in_executor(
        _inference_pool, partial(synthesize, request.text, request.lang)
    )
    if not audio_bytes:
        raise HTTPException(status_code=503, detail="TTS synthesis failed")
    return Response(content=audio_bytes, media_type="audio/wav")


# ── /api/sources/{id} ────────────────────────────────────────────────────
@app.get("/api/sources/{chunk_id}", response_model=SourceResponse, tags=["qa"])
async def get_source(chunk_id: str):
    """Return the full text of a chunk by its ID (for citation popover)."""
    # chunks.jsonl is written by scripts/ingest.py (one JSON object per line)
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise HTTPException(status_code=503, detail="Knowledge base not yet built")

    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("chunk_id") == chunk_id:
                return SourceResponse(
                    chunk_id=c["chunk_id"],
                    title=c.get("title", ""),
                    source_url=c.get("source_url"),
                    category=c.get("category"),
                    text=c.get("text", c.get("content", "")),
                )

    raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
