"""
orchestrator.py
───────────────
Ties the full pipeline together:

  1. [Optional] Translate input → English  (kn/ta always; hi if route='pivot')
  2. Tool router  → exact fee/calendar/placement answers
  3. Hybrid retrieval
  4. Param1 answer generation
  5. Grounding guard
  6. [Optional] Translate answer → user's language
  7. Return structured AskResponse
"""

import re
import time
from typing import Literal

from loguru import logger

from backend.app.config import settings
from backend.app.pipeline import generate, guard, retrieve, tools, translate
from backend.app.schemas import AskRequest, AskResponse, Citation, StageTrace, LangCode

_CITATION_RE = re.compile(r"\[S(\d+)\]")


def _extract_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    """Map [S1] … tags back to the corresponding chunk."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1)) - 1  # zero-based
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            cid = c["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                citations.append(
                    Citation(
                        chunk_id=cid,
                        title=c["title"],
                        source_url=c.get("source_url"),
                        excerpt=c["text"][:200],
                    )
                )
    return citations


def run(
    request: AskRequest,
    *,
    retrieval_mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
    use_tools: bool = True,
) -> AskResponse:
    overall_start = time.perf_counter()
    trace: list[StageTrace] = []
    lang: LangCode = request.lang

    # For Hindi, apply the HINDI_ROUTE env setting:
    #   "direct" – send Hindi text as-is (no translation before routing)
    #   "pivot"  – always translate Hi→En before tool routing + retrieval
    #   "both"   – use pivot (translate first) for best accuracy
    effective_route: Literal["direct", "pivot"] = request.route
    if lang == "hi" and settings.hindi_route in ("pivot", "both"):
        effective_route = "pivot"

    # ── 1. Translate to English ─────────────────────────────────────────
    query_en = request.text
    if lang != "en" and (lang in ("kn", "ta") or effective_route == "pivot"):
        t0 = time.perf_counter()
        query_en, _ = translate.to_english(request.text, lang)
        stage_ms = (time.perf_counter() - t0) * 1000
        trace.append(StageTrace(stage="translate_to_en", latency_ms=round(stage_ms, 1),
                                detail=f"{lang}→en"))
        logger.debug(f"Translated: {request.text!r} → {query_en!r}")

    # ── 2. Tool router ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    tool_result = tools.tool_route(query_en) if use_tools else None
    tool_ms = (time.perf_counter() - t0) * 1000

    if tool_result:
        trace.append(StageTrace(stage="tool_router", latency_ms=round(tool_ms, 1),
                                detail=f"matched:{tool_result['tool']}"))
        answer_en = tool_result["answer"]
        chunks: list[dict] = []
        top_score = 1.0
        tool_citation = Citation(
            chunk_id=tool_result["chunk_id"],
            title=f"RVU {tool_result['tool'].capitalize()} data (structured)",
            source_url="https://rvu.edu.in",
            excerpt=answer_en[:200],
        )
        citations = [tool_citation]
        refused = False
        refusal_reason = None

        # Jump straight to translation
        goto_translate = True

    else:
        trace.append(StageTrace(stage="tool_router", latency_ms=round(tool_ms, 1),
                                detail="no match"))
        goto_translate = False

    if not goto_translate:
        # ── 3. Retrieval ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = retrieve.retrieve(query_en, mode=retrieval_mode)
        ret_ms = (time.perf_counter() - t0) * 1000
        top_score = chunks[0]["score"] if chunks else 0.0
        trace.append(StageTrace(stage="retrieve", latency_ms=round(ret_ms, 1),
                                detail=f"{len(chunks)} chunks, top_score={top_score:.3f}"))

        # ── 4. Param1 generation ─────────────────────────────────────────
        t0 = time.perf_counter()
        if chunks:
            try:
                gen_result = generate.generate(query_en, chunks)
                answer_en = gen_result["answer"]
                gen_ms = gen_result["latency_ms"]
            except RuntimeError as exc:
                # Model not downloaded yet — produce a graceful refusal
                logger.warning(f"LLM unavailable: {exc}")
                answer_en = (
                    "I don't have that information. "
                    "Please contact the relevant RVU office."
                )
                gen_ms = 0.0
        else:
            answer_en = (
                "I don't have sufficient information to answer your question. "
                "Please contact the relevant RVU office."
            )
            gen_ms = 0.0
        trace.append(StageTrace(stage="generate", latency_ms=gen_ms))

        # ── 5. Grounding guard ────────────────────────────────────────────
        guard_result = guard.check(answer_en, chunks, top_score)
        refused = not guard_result["passed"] or guard_result["should_refuse"]
        refusal_reason = guard_result["reason"]

        if not guard_result["passed"]:
            top_cat = chunks[0].get("category") if chunks else None
            answer_en = guard.refusal_message(refusal_reason, top_cat)
            citations = []
        elif guard_result["should_refuse"]:
            top_cat = chunks[0].get("category") if chunks else None
            answer_en = guard.refusal_message(None, top_cat)
            citations = []
        else:
            citations = _extract_citations(answer_en, chunks)
            # Sources are shown as chips; inline [S1] tags only hurt translation and TTS.
            answer_en = re.sub(r"\s*\[S\d+\]", "", answer_en).strip()

        trace.append(StageTrace(stage="guard",
                                detail=f"passed={guard_result['passed']}, refuse={guard_result['should_refuse']}",
                                latency_ms=0.1))

    # ── 6. Translate answer back ─────────────────────────────────────────
    answer_final = answer_en
    if lang != "en" and not refused:
        t0 = time.perf_counter()
        answer_final, _ = translate.from_english(answer_en, lang)
        back_ms = (time.perf_counter() - t0) * 1000
        trace.append(StageTrace(stage="translate_from_en", latency_ms=round(back_ms, 1),
                                detail=f"en→{lang}"))

    total_ms = (time.perf_counter() - overall_start) * 1000

    return AskResponse(
        answer=answer_final,
        answer_lang=lang,
        citations=citations,
        refused=refused,
        refusal_reason=refusal_reason,
        trace=trace,
        total_latency_ms=round(total_ms, 1),
    )
