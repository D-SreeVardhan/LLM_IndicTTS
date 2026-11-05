"""Pydantic request/response models shared across the API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Language codes we support ──────────────────────────────────────────────
LangCode = Literal["hi", "kn", "ta", "en"]


# ── /api/transcribe ────────────────────────────────────────────────────────
class TranscribeResponse(BaseModel):
    text: str
    lang_detected: Optional[LangCode] = None
    decode_mode: str = "rnnt"
    latency_ms: float = 0.0


# ── /api/ask ───────────────────────────────────────────────────────────────
class Citation(BaseModel):
    chunk_id: str
    title: str
    source_url: Optional[str] = None
    excerpt: str = ""


class StageTrace(BaseModel):
    stage: str
    latency_ms: float
    detail: Optional[str] = None


class AskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    lang: LangCode = "hi"
    route: Literal["direct", "pivot"] = "direct"


class AskResponse(BaseModel):
    answer: str
    answer_lang: LangCode
    citations: list[Citation] = []
    refused: bool = False
    refusal_reason: Optional[str] = None
    trace: list[StageTrace] = []
    total_latency_ms: float = 0.0


# ── /api/speak ─────────────────────────────────────────────────────────────
class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    lang: LangCode = "hi"


# ── /api/sources/{id} ──────────────────────────────────────────────────────
class SourceResponse(BaseModel):
    chunk_id: str
    title: str
    source_url: Optional[str] = None
    category: Optional[str] = None
    text: str
