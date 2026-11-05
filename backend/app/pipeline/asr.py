"""
asr.py
──────
Automatic Speech Recognition using IndicConformer-600M-Multilingual.

Supports CTC and RNNT decoding.  Resamples input to 16 kHz mono.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger

from backend.app.config import settings

_model = None  # lazy-loaded singleton


def _load_model():
    global _model
    if _model is not None:
        return _model
    # NOTE: transformers AutoModel is imported lazily here to avoid loading
    # the library at startup when the ASR model is not pre-warmed.
    # This is an intentional performance optimisation, not a circular-dependency workaround.
    from transformers import AutoModel  # noqa: PLC0415

    logger.info(f"Loading IndicConformer from {settings.asr_model_id} …")
    _model = AutoModel.from_pretrained(settings.asr_model_id, trust_remote_code=True)
    _model.eval()
    logger.success("IndicConformer loaded.")
    return _model


def _convert_to_wav(audio_bytes: bytes) -> Path:
    """Convert any audio format to 16-kHz mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_path = tmp_in_path.with_suffix(".wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_in_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(tmp_out_path),
        ],
        capture_output=True,
    )
    tmp_in_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return tmp_out_path


# IndicConformer supports 22 Indian languages. English is NOT supported by the model;
# main.py recognises English speech with the Hindi model and translates it (see codemix.py).
_LANG_CODES: dict[str, str] = {
    "hi": "hi",
    "kn": "kn",
    "ta": "ta",
    # "en" intentionally omitted: IndicConformer-600M does not support English ASR.
}


def transcribe(
    audio_bytes: bytes,
    lang: str = "hi",
    decode_mode: Literal["ctc", "rnnt"] | None = None,
) -> dict:
    """
    Transcribe audio bytes (WebM, WAV, OGG, …) to text.

    Returns a dict with keys:
        text         – transcribed string
        latency_ms   – inference time
        decode_mode  – which decoding was used
    """
    if decode_mode is None:
        decode_mode = settings.asr_decode_mode

    t0 = time.perf_counter()

    wav_path = _convert_to_wav(audio_bytes)
    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    finally:
        wav_path.unlink(missing_ok=True)

    # data shape: (samples, channels) → (channels, samples)
    data = data.T  # → (channels, samples)

    # Mono: average channels
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)

    wav = torch.from_numpy(data)  # (1, samples)

    # Ensure 16 kHz via linear resample if needed
    if sr != 16000:
        orig_len = wav.shape[-1]
        new_len = int(orig_len * 16000 / sr)
        wav = F.interpolate(wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)

    model = _load_model()
    if lang not in _LANG_CODES:
        raise ValueError(
            f"ASR language '{lang}' is not supported by IndicConformer. "
            f"Supported languages: {list(_LANG_CODES.keys())}. "
            "For English, use the text input field directly."
        )
    lang_code = _LANG_CODES[lang]

    with torch.no_grad():
        text = model(wav, lang_code, decode_mode)

    latency = (time.perf_counter() - t0) * 1000
    logger.debug(f"ASR [{lang_code}/{decode_mode}] {latency:.0f}ms → {text!r}")

    return {
        "text": text if isinstance(text, str) else str(text),
        "latency_ms": round(latency, 1),
        "decode_mode": decode_mode,
    }
