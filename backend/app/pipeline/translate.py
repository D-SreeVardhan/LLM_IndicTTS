"""
translate.py
────────────
Bidirectional translation between English and Indic languages using
IndicTrans2-distilled-200M models via the IndicTransToolkit IndicProcessor.
"""

import re
import time
from typing import Literal

import torch
from loguru import logger

from backend.app.config import settings

Direction = Literal["indic_to_en", "en_to_indic"]

# Flores-200 language codes expected by IndicTrans2
_FLORES: dict[str, str] = {
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "en": "eng_Latn",
}

_models: dict[Direction, tuple] = {}  # (tokenizer, model, processor)


def _load(direction: Direction):
    if direction in _models:
        return _models[direction]

    # NOTE: transformers is imported lazily here to avoid loading the heavy library
    # at startup when translation is not pre-loaded. This is an intentional
    # performance optimisation, not a circular-dependency workaround.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    model_id = (
        settings.trans_indic_en_model_id
        if direction == "indic_to_en"
        else settings.trans_en_indic_model_id
    )
    logger.info(f"Loading IndicTrans2 [{direction}] from {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    mdl.eval()

    # NOTE: IndicTransToolkit is an optional dependency imported lazily.
    try:
        from IndicTransToolkit import IndicProcessor  # noqa: PLC0415
        proc = IndicProcessor(inference=True)
    except ImportError:
        logger.warning("IndicTransToolkit not found – using fallback (no pre/post processing)")
        proc = None

    _models[direction] = (tok, mdl, proc)
    logger.success(f"IndicTrans2 [{direction}] loaded.")
    return tok, mdl, proc


def translate(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
) -> list[str]:
    """
    Translate a list of texts.

    src_lang / tgt_lang should be ISO 639-1 codes: hi, kn, ta, en.
    Returns a list of translated strings.
    """
    if src_lang == tgt_lang:
        return texts

    if src_lang == "en":
        direction: Direction = "en_to_indic"
    else:
        direction = "indic_to_en"

    src_flores = _FLORES.get(src_lang, src_lang)
    tgt_flores = _FLORES.get(tgt_lang, tgt_lang)

    t0 = time.perf_counter()
    tok, mdl, proc = _load(direction)

    if proc is not None:
        # Full IndicProcessor path
        batch = proc.preprocess_batch(texts, src_lang=src_flores, tgt_lang=tgt_flores)
        inputs = tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(mdl.device)
        with torch.no_grad():
            generated = mdl.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=512,
            )
        decoded = tok.batch_decode(generated, skip_special_tokens=True)
        output = proc.postprocess_batch(decoded, lang=tgt_flores)
    else:
        # Fallback without IndicProcessor
        src_prefix = f">>{src_flores}<< "
        prefixed = [src_prefix + t for t in texts]
        inputs = tok(prefixed, return_tensors="pt", padding=True, truncation=True).to(mdl.device)
        with torch.no_grad():
            generated = mdl.generate(**inputs, max_new_tokens=512)
        output = tok.batch_decode(generated, skip_special_tokens=True)

    latency = (time.perf_counter() - t0) * 1000
    logger.debug(f"Translate [{src_lang}→{tgt_lang}] {latency:.0f}ms, {len(texts)} segment(s)")
    return output


def to_english(text: str, src_lang: str) -> tuple[str, float]:
    """Convenience wrapper. Returns (translated_text, latency_ms)."""
    t0 = time.perf_counter()
    result = translate([text], src_lang=src_lang, tgt_lang="en")
    ms = (time.perf_counter() - t0) * 1000
    return result[0], round(ms, 1)


# Split after . ! ? only when the previous char is lower-case/digit/bracket and the next
# sentence starts with a capital, so abbreviations like "B. Tech" stay intact.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[a-z0-9)][.!?])\s+(?=[A-Z0-9])|\n+")

_MARKUP_RE = re.compile(r"\*\*|__|`|^#{1,6}\s*|^\s*[-*•]\s+|\|", re.M)

# IndicTrans2 can emit runs of danda / pipes when fed tables or very long input.
_PUNCT_RUN_RE = re.compile(r"\s*([।|.])(?:\s*[।|.])+")


# IndicTrans2 masks numbers like "2,50,000" with an <ID1> placeholder, and for
# Kannada the placeholder itself gets translated, so the amount is lost. Spelling
# amounts as "2 lakh 50 thousand" survives translation and reads naturally aloud.
_INDIAN_AMOUNT_RE = re.compile(r"\b\d{1,2}(?:,\d{2})*,\d{3}\b")


def _amount_in_words(match: re.Match) -> str:
    value = int(match.group().replace(",", ""))
    parts = []
    for unit, name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if value >= unit:
            parts.append(f"{value // unit} {name}")
            value %= unit
    if value:
        parts.append(str(value))
    return " ".join(parts)


def _split_sentences(text: str) -> list[str]:
    text = _MARKUP_RE.sub(" ", text)
    text = _INDIAN_AMOUNT_RE.sub(_amount_in_words, text)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip(" .")]


def _tidy_output(text: str) -> str:
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    text = text.replace("|", " ")
    return re.sub(r"\s{2,}", " ", text).strip()


def from_english(text: str, tgt_lang: str) -> tuple[str, float]:
    """
    Translate an English answer sentence-by-sentence (IndicTrans2 is a
    sentence-level model; long multi-sentence input degrades into repetition).
    Returns (translated_text, latency_ms).
    """
    t0 = time.perf_counter()
    sentences = _split_sentences(text) or [text]
    translated = translate(sentences, src_lang="en", tgt_lang=tgt_lang)
    result = " ".join(_tidy_output(s) for s in translated if s.strip())
    ms = (time.perf_counter() - t0) * 1000
    return result, round(ms, 1)
