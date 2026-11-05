"""
generate.py
───────────
Grounded answer generation using BharatGen Param-1-2.9B-Instruct.

The model is given retrieved context chunks and must cite each used
chunk as [S1], [S2], etc.  Low temperature for factual consistency.
"""

import re
import time
from pathlib import Path

import torch
from loguru import logger

from backend.app.config import settings

_tokenizer = None
_model = None


_load_error: str | None = None  # cache load failure reason


def _load():
    global _tokenizer, _model, _load_error
    if _model is not None:
        return _tokenizer, _model
    if _load_error is not None:
        raise RuntimeError(_load_error)

    # NOTE: transformers and huggingface_hub are imported lazily here to avoid
    # loading the 500 MB+ library at startup when the LLM is not pre-warmed.
    # This is an intentional performance optimisation, not a circular-dependency workaround.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    logger.info(f"Loading Param1 from {settings.llm_model_id} …")
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        snap_path = snapshot_download(settings.llm_model_id, local_files_only=True)
        _tokenizer = AutoTokenizer.from_pretrained(snap_path, trust_remote_code=True)

        # Param-1 stores its chat template in a separate jinja file – inject it.
        if not _tokenizer.chat_template:
            jinja_path = Path(snap_path) / "chat_template.jinja"
            if jinja_path.exists():
                _tokenizer.chat_template = jinja_path.read_text()
                logger.debug("Injected chat_template.jinja into tokenizer.")

        # Pick best available device: MPS (Apple Silicon) > CPU
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
            load_kwargs: dict = {"torch_dtype": torch.float16}
        else:
            _device = torch.device("cpu")
            load_kwargs = {"torch_dtype": torch.float32}  # fp32 for CPU stability

        _model = AutoModelForCausalLM.from_pretrained(
            snap_path,
            trust_remote_code=True,
            **load_kwargs,
        )
        _model = _model.to(_device)
        _model.eval()
        logger.success("Param1 loaded.")
    except Exception as exc:
        _load_error = str(exc)
        logger.warning(
            f"Param1 not available ({exc}). "
            "Run scripts/download_models.py to download the model."
        )
        raise RuntimeError(_load_error) from exc
    return _tokenizer, _model


def warm_up() -> None:
    """Pre-load Param-1 into memory (call at server startup if keep_llm_resident=True)."""
    try:
        _load()
        logger.success("Param1 warm-up complete.")
    except RuntimeError as exc:
        logger.warning(f"Param1 warm-up skipped: {exc}")


_MAX_CONTEXT_CHUNKS = 5
_WINDOW_CHARS = 360
_STOPWORDS = {
    "the", "a", "an", "is", "are", "be", "what", "when", "how", "much", "should", "for", "of",
    "to", "in", "at", "on", "and", "or", "i", "my", "do", "does", "can", "rvu", "university",
}


def _best_window(text: str, query: str, size: int) -> str:
    """Return the `size`-char window of `text` (sentence-aligned) with the most query-term hits."""
    if len(text) <= size:
        return text
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS and len(w) > 2}
    starts = [0] + [m.end() for m in re.finditer(r"[.\n●•]\s*", text)]
    lower = text.lower()

    def hits(start: int) -> int:
        window = lower[start:start + size]
        return sum(window.count(t) for t in terms)

    best = max(starts, key=hits) if terms else 0
    return text[best:best + size].strip()


def _build_system_prompt(context_blocks: list[str]) -> str:
    ctx_text = "\n\n".join(
        f"[S{i+1}] {block}" for i, block in enumerate(context_blocks)
    )
    return (
        "You are an authoritative assistant for RV University (RVU), Bengaluru.\n"
        "Answer ONLY using the provided sources below. "
        "For every factual claim, cite the source tag like [S1] inline.\n"
        "If the answer cannot be found in the sources, reply: "
        '"I don\'t have that information. Please contact the relevant RVU office."\n'
        "Keep the answer concise (3–5 sentences max) and factual.\n"
        "The answer will be translated and read aloud, so write short, complete, plain sentences: "
        "no markdown, bullet points, tables, headings or symbols like ₹ or %; "
        "write 'rupees' and 'percent' instead.\n\n"
        "=== SOURCES ===\n"
        f"{ctx_text}\n"
        "=== END OF SOURCES ==="
    )


def generate(
    query_en: str,
    context_chunks: list[dict],
    max_new_tokens: int = 150,  # 150 is plenty for 3-5 sentence factual answers
) -> dict:
    """
    Generate a grounded English answer using Param1.

    Args:
        query_en      – English question text
        context_chunks – list of retrieved chunk dicts (must have 'text' key)
        max_new_tokens – generation cap

    Returns dict with keys:
        answer       – generated answer string
        latency_ms   – generation time
    """
    tok, mdl = _load()

    # Keep the context short (~1,800 chars): pass the most query-relevant
    # window of each of the top chunks rather than each chunk's opening lines,
    # so facts buried mid-chunk (e.g. "80% attendance") still reach the model.
    context_texts = [
        _best_window(c["text"], query_en, _WINDOW_CHARS) for c in context_chunks[:_MAX_CONTEXT_CHUNKS]
    ]
    system_prompt = _build_system_prompt(context_texts)

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_en},
    ]

    t0 = time.perf_counter()

    inputs = tok.apply_chat_template(
        conversation,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(mdl.device)

    with torch.no_grad():
        output = mdl.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.3,  # low for factual answers
            eos_token_id=tok.eos_token_id,
            use_cache=True,   # KV-cache is critical for speed
        )

    generated_tokens = output[0][inputs.shape[-1]:]
    answer = tok.decode(generated_tokens, skip_special_tokens=True).strip()
    if mdl.device.type == "mps":
        torch.mps.empty_cache()  # release cached buffers so other models don't push the LLM into swap

    latency = (time.perf_counter() - t0) * 1000
    logger.debug(f"Param1 generation: {latency:.0f}ms – {len(answer)} chars")

    return {
        "answer": answer,
        "latency_ms": round(latency, 1),
    }
