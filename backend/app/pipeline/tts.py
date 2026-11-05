"""
tts.py
──────
Text-to-Speech using AI4Bharat Indic Parler-TTS.

Returns 44.1-kHz mono WAV bytes for each language.

Key details:
  - Description tokenizer must come from the model's TEXT ENCODER (flan-t5-large),
    not from the TTS model itself.  Using the wrong tokenizer produces garbled audio.
  - Sentences are synthesised as one padded batch; each is trimmed to its own
    audio length, so long answers are not cut off at a single token budget.
  - Speaker names in the descriptions are the model card's recommended voices.
  - Markdown must be stripped before synthesis (asterisks, bullets → garbled speech).
"""

import io
import re
import time

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from backend.app.config import settings
from backend.app.schemas import LangCode

_tts_model = None
_tts_tokenizer = None       # tokenizer for the TEXT TO BE SPOKEN
_desc_tokenizer = None      # tokenizer for the SPEAKER DESCRIPTION (T5/flan-t5-large)


# ── Markdown → plain text ─────────────────────────────────────────────────
# Patterns are applied IN ORDER — source-footer and citation removal must
# happen before italic stripping so asterisks don't leave orphaned text.

_MD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Remove source footers like *Source: ...* before italic stripping
    (re.compile(r"\*Source:[^\n]*", re.I),          ""),
    (re.compile(r"Source:[^\n]*AY \d{4}", re.I),    ""),
    # Citation tags [S1] [S2]
    (re.compile(r"\[S\d+\]"),                        ""),
    # AY year noise "(AY 2026-27)"
    (re.compile(r"\(AY \d{4}-\d{2,4}\)"),           ""),
    # Bold / italic / underline
    (re.compile(r"\*\*(.*?)\*\*"),                   r"\1"),
    (re.compile(r"\*(.*?)\*"),                        r"\1"),
    (re.compile(r"__(.*?)__"),                        r"\1"),
    # Headings
    (re.compile(r"#{1,6}\s*"),                        ""),
    # Code blocks / inline code
    (re.compile(r"`{1,3}(.*?)`{1,3}", re.S),         r"\1"),
    # Markdown table dividers and pipes
    (re.compile(r"\|\s*[-:]+\s*\|"),                  ""),
    (re.compile(r"\|"),                               " "),
    # Bullet points — replace dash/bullet with nothing (sentence follows)
    (re.compile(r"^\s*[-*•]\s+", re.M),               ""),
    # Special characters → spoken equivalents
    (re.compile(r"₹"),                               "rupees "),
    (re.compile(r"→"),                               "to"),
    (re.compile(r"≥"),                               "at least"),
    (re.compile(r"≤"),                               "at most"),
    (re.compile(r"–|—"),                             ", "),
    (re.compile(r"·"),                               ", "),
    # Runs of danda / full stops (translation artefacts) → one sentence break
    (re.compile(r"\s*([।.])(?:\s*[।.])+"),           r"\1"),
    # Collapse all newlines and excess whitespace to single space
    (re.compile(r"\n+"),                             " "),
    (re.compile(r"\s{2,}"),                          " "),
]

# Safety limit only: answers are 3–5 sentences, so the whole answer is spoken.
# Longer text is cut at a sentence end, or failing that a word boundary, never mid-word.
_MAX_TTS_CHARS = 700

# Hindi ends sentences with the danda (।), not a full stop
_SENTENCE_ENDS = ".!?।"


def _clean_for_tts(text: str) -> str:
    """Strip markdown and normalise text for TTS synthesis."""
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    text = text.strip()
    if len(text) > _MAX_TTS_CHARS:
        truncated = text[:_MAX_TTS_CHARS]
        last_break = max(truncated.rfind(f"{p} ") for p in _SENTENCE_ENDS)
        if last_break > _MAX_TTS_CHARS // 2:
            text = truncated[: last_break + 1]
        else:
            text = truncated[: truncated.rfind(" ")].rstrip(" ,")
    # Ensure text ends with punctuation so the model stops cleanly
    if text and text[-1] not in _SENTENCE_ENDS:
        text += "."
    return text


# ── Model loading ─────────────────────────────────────────────────────────

def _load():
    global _tts_model, _tts_tokenizer, _desc_tokenizer
    if _tts_model is not None:
        return

    logger.info(f"Loading Indic Parler-TTS from {settings.tts_model_id} …")

    # NOTE: parler_tts and the transformers tokenizers are imported lazily here
    # to avoid loading these libraries at server startup when TTS is not pre-warmed.
    # This is an intentional performance optimisation, not a circular-dependency workaround.
    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: PLC0415
        from transformers import AutoTokenizer as HFAutoTokenizer  # noqa: PLC0415

        # CPU, not MPS: on MPS the backend's memory grew past 30 GB over a few
        # requests alongside Param-1 on a 16 GB Mac.
        _tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            settings.tts_model_id,
            token=settings.hf_token or None,
        )
        _tts_model = _tts_model.to("cpu")
        _tts_model.eval()

        # Text tokenizer — tokenises the TEXT TO BE SPOKEN
        _tts_tokenizer = HFAutoTokenizer.from_pretrained(settings.tts_model_id)

        # Description tokenizer — MUST come from the text-encoder (flan-t5-large),
        # NOT from the TTS model. Using the wrong tokenizer produces garbled audio.
        desc_encoder_name = _tts_model.config.text_encoder._name_or_path
        _desc_tokenizer = HFAutoTokenizer.from_pretrained(
            desc_encoder_name, local_files_only=True
        )
        logger.success(
            f"Indic Parler-TTS loaded. "
            f"desc_tokenizer={desc_encoder_name}, "
            f"sample_rate={_tts_model.config.sampling_rate}"
        )

    except (ImportError, Exception) as exc:
        logger.warning(f"Parler-TTS not available ({exc}) – /api/speak will return 503")


# ── Speaker descriptions ──────────────────────────────────────────────────
# Descriptions must name the language/accent explicitly for Indic Parler-TTS
# to select the right speaker voice. A brisk pace matters on CPU: every second
# of audio costs ~86 decoder steps.

_SPEAKER_DESC: dict[LangCode, str] = {
    "hi": (
        "Divya speaks Hindi at a moderately fast pace with a clear, natural accent. "
        "The recording is very clear and close-sounding, with no background noise."
    ),
    "kn": (
        "Anu speaks Kannada at a moderately fast pace with a clear, natural accent. "
        "The recording is very clear and close-sounding, with no background noise."
    ),
    "ta": (
        "Jaya speaks Tamil at a moderately fast pace with a clear, natural accent. "
        "The recording is very clear and close-sounding, with no background noise."
    ),
    "en": (
        "Mary speaks English at a moderately fast pace with a clear Indian accent. "
        "The recording is very clear and close-sounding, with no background noise."
    ),
}


# ── Synthesis ─────────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")
_MIN_SEGMENT_CHARS = 30    # merge fragments shorter than this into the previous segment
_MAX_SEGMENT_CHARS = 90    # split longer sentences at a comma, else at a space
_TOKENS_PER_CHAR = 15      # audio-token budget per character; generation stops at EOS
_MAX_NEW_TOKENS = 2600     # ~30 s of audio, well above any segment
_PAUSE_S = 0.3
# Speech measured at ~13.5 chars/s; a take faster than this ended early and dropped words.
_MAX_CHARS_PER_S = 20


def _split_point(sentence: str) -> int:
    """Index to cut a too-long sentence at: last comma, else last space, within the limit."""
    window = sentence[: _MAX_SEGMENT_CHARS]
    comma = window.rfind(", ")
    if comma >= _MIN_SEGMENT_CHARS:
        return comma + 1
    space = window.rfind(" ")
    return space if space >= _MIN_SEGMENT_CHARS else _MAX_SEGMENT_CHARS


def _segments(text: str) -> list[str]:
    """Split into short segments so they can be synthesised as one parallel batch."""
    segments: list[str] = []
    carry = ""  # a short opening fragment such as "ಬಿ." (B.) is spoken with what follows
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = f"{carry} {sentence}".strip()
        carry = ""
        if not segments and len(sentence) < _MIN_SEGMENT_CHARS:
            carry = sentence
            continue
        while len(sentence) > _MAX_SEGMENT_CHARS:
            cut = _split_point(sentence)
            segments.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if segments and len(sentence) < _MIN_SEGMENT_CHARS:
            segments[-1] = f"{segments[-1]} {sentence}"
        else:
            segments.append(sentence)
    if carry:
        segments.append(carry)
    return segments


def _generate(segments: list[str], description: str) -> list[np.ndarray]:
    """Synthesise segments as one padded batch; each result is trimmed to its own length."""
    device = _tts_model.device
    desc_inputs = _desc_tokenizer([description] * len(segments), return_tensors="pt", padding=True).to(device)
    text_inputs = _tts_tokenizer(segments, return_tensors="pt", padding=True).to(device)
    max_tokens = min(_MAX_NEW_TOKENS, max(len(s) for s in segments) * _TOKENS_PER_CHAR)

    with torch.no_grad():
        output = _tts_model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=text_inputs.input_ids,
            prompt_attention_mask=text_inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            return_dict_in_generate=True,
        )
    return [
        audio[: int(length)].astype(np.float32)
        for audio, length in zip(output.sequences.float().cpu().numpy(), output.audios_length)
    ]


def synthesize(text: str, lang: LangCode) -> tuple[bytes, float]:
    """
    Synthesise text to speech with Indic Parler-TTS.

    Segments are generated as one batch (parallel decoding) and joined with
    short pauses; segments that ended early are regenerated once.
    Returns (wav_bytes, latency_ms); wav_bytes is empty if the model is unavailable.
    """
    t0 = time.perf_counter()

    clean_text = _clean_for_tts(text)
    segments = _segments(clean_text) or [clean_text]
    logger.debug(f"TTS [{lang}] {len(clean_text)} chars in {len(segments)} segment(s): {clean_text[:80]!r}")

    _load()
    if _tts_model is None:
        return b"", round((time.perf_counter() - t0) * 1000, 1)

    description = _SPEAKER_DESC.get(lang, _SPEAKER_DESC["en"])
    sample_rate: int = getattr(_tts_model.config, "sampling_rate", None) or 44100
    takes = _generate(segments, description)

    # Sampling sometimes stops a segment early; regenerate those once and keep the longer take.
    short = [i for i, (seg, audio) in enumerate(zip(segments, takes))
             if len(audio) / sample_rate < len(seg) / _MAX_CHARS_PER_S]
    if short:
        logger.debug(f"TTS [{lang}] retrying {len(short)} segment(s) that ended early")
        for i, audio in zip(short, _generate([segments[i] for i in short], description)):
            if len(audio) > len(takes[i]):
                takes[i] = audio

    pause = np.zeros(int(sample_rate * _PAUSE_S), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for audio in takes:
        pieces += [audio, pause]
    audio_arr = np.concatenate(pieces[:-1]) if pieces else pause

    buf = io.BytesIO()
    sf.write(buf, audio_arr, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        f"TTS [{lang}] {ms:.0f}ms – {len(audio_arr) / sample_rate:.1f}s audio, "
        f"{len(segments)} segment(s), {len(short)} retried"
    )
    return wav_bytes, round(ms, 1)
