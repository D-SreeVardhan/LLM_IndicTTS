#!/usr/bin/env python3
"""
download_models.py
──────────────────
Downloads and smoke-tests every model used by the RVU voice assistant.
Shows a live per-model progress bar with download speed and ETA.

    python scripts/download_models.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import threading
import torch
from loguru import logger
from huggingface_hub import login, snapshot_download

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# ── Auth ──────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    logger.info("Logged in to Hugging Face.")
else:
    logger.warning("HF_TOKEN not set – gated models will fail.")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Device: {DEVICE}  |  Python {sys.version.split()[0]}")

# ── Model registry ────────────────────────────────────────────────────────
MODELS = [
    ("IndicConformer ASR",        "ai4bharat/indic-conformer-600m-multilingual"),
    ("IndicTrans2  indic→en",     "ai4bharat/indictrans2-indic-en-dist-200M"),
    ("IndicTrans2  en→indic",     "ai4bharat/indictrans2-en-indic-dist-200M"),
    ("Param-1-2.9B-Instruct LLM", "bharatgenai/Param-1-2.9B-Instruct"),
    ("Indic Parler-TTS",          "ai4bharat/indic-parler-tts"),
    ("bge-m3 Embedder",           "BAAI/bge-m3"),
]

IGNORE_PATTERNS = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"]

# ── Helpers ───────────────────────────────────────────────────────────────

def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:5.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _get_cache_size(model_id: str) -> int:
    """Return bytes already cached locally for this model."""
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
    if not cache_dir.exists():
        return 0
    return sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())


def download_with_progress(label: str, model_id: str) -> bool:
    """Download a model and show a progress bar via a polling thread."""
    before = _get_cache_size(model_id)

    print()
    print(f"  ┌─ {label}")
    print(f"  │  {model_id}")
    print(f"  │  Cached so far: {_sizeof_fmt(before)}")

    done_flag = threading.Event()
    result_box = [False]

    def _do_download():
        try:
            snapshot_download(
                repo_id=model_id,
                token=HF_TOKEN or None,
                ignore_patterns=IGNORE_PATTERNS,
            )
            result_box[0] = True
        except Exception as exc:
            result_box[0] = exc
        finally:
            done_flag.set()

    t = threading.Thread(target=_do_download, daemon=True)
    t.start()

    # Poll cache size and render a progress bar
    t0 = time.time()
    last_size = before
    speed_samples: list[float] = []

    while not done_flag.is_set():
        time.sleep(1.5)
        now_size = _get_cache_size(model_id)
        delta = max(0, now_size - last_size)
        speed_samples.append(delta / 1.5)
        if len(speed_samples) > 6:
            speed_samples.pop(0)
        avg_speed = sum(speed_samples) / max(len(speed_samples), 1)
        elapsed = time.time() - t0
        total_new = now_size - before
        last_size = now_size

        speed_str = f"{_sizeof_fmt(avg_speed)}/s" if avg_speed > 0 else "  …  "
        size_str  = _sizeof_fmt(total_new)
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        # Spinner frames
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = spinner[int(elapsed * 4) % len(spinner)]

        line = f"\r  │  {frame}  +{size_str:>10}   {speed_str:>12}   {elapsed_str}"
        sys.stdout.write(line)
        sys.stdout.flush()

    t.join()
    elapsed = time.time() - t0
    after = _get_cache_size(model_id)
    total_dl = after - before

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    if result_box[0] is True:
        print(f"  └─ ✅  Done  |  +{_sizeof_fmt(total_dl)}  |  {elapsed:.0f}s")
        return True
    else:
        err = result_box[0]
        print(f"  └─ ❌  Failed: {err}")
        if "401" in str(err) or "403" in str(err) or "gated" in str(err).lower():
            print(f"       → Visit https://huggingface.co/{model_id} and click")
            print("         \"Agree and access repository\", then re-run this script.")
        return False


# ── Smoke tests ───────────────────────────────────────────────────────────

def smoke_asr() -> bool:
    print("\n  🔊 Smoke-testing IndicConformer …", end=" ", flush=True)
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
        )
        wav = torch.zeros(1, 16000)
        out = model(wav, "hi", "ctc")
        print(f"✅  CTC output: {repr(str(out)[:40])}")
        del model
        return True
    except Exception as exc:
        print(f"❌  {exc}")
        return False


def smoke_translate() -> bool:
    print("  🔤 Smoke-testing IndicTrans2 indic→en …", end=" ", flush=True)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True
        )
        print("✅  Loaded")
        del tok, mdl
        return True
    except Exception as exc:
        print(f"❌  {exc}")
        return False


def smoke_llm() -> bool:
    print("  🧠 Smoke-testing Param1 (generates 5 tokens) …", end=" ", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bharatgenai/Param-1-2.9B-Instruct")
        mdl = AutoModelForCausalLM.from_pretrained(
            "bharatgenai/Param-1-2.9B-Instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        inp = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(inp, max_new_tokens=5, use_cache=False)
        txt = tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True)
        print(f"✅  Output: {repr(txt[:40])}")
        del tok, mdl
        return True
    except Exception as exc:
        print(f"❌  {exc}")
        return False


def smoke_embed() -> bool:
    print("  🔍 Smoke-testing bge-m3 …", end=" ", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-m3")
        v = m.encode(["test"])
        print(f"✅  Shape: {v.shape}")
        del m
        return True
    except Exception as exc:
        print(f"❌  {exc}")
        return False


def smoke_tts() -> bool:
    print("  🗣  Smoke-testing Indic Parler-TTS …", end=" ", flush=True)
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts", token=HF_TOKEN or None
        )
        print("✅  Loaded")
        del model
        return True
    except Exception as exc:
        print(f"❌  {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   RVU Voice Assistant — Model Downloader             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    dl_results: dict[str, bool] = {}
    for label, model_id in MODELS:
        ok = download_with_progress(label, model_id)
        dl_results[label] = ok

    print()
    print("─" * 56)
    print("  Running smoke tests …")
    print("─" * 56)

    smoke_map = {
        "IndicConformer ASR":         smoke_asr,
        "IndicTrans2  indic→en":      smoke_translate,
        "Param-1-2.9B-Instruct LLM":  smoke_llm,
        "bge-m3 Embedder":            smoke_embed,
        "Indic Parler-TTS":           smoke_tts,
    }

    smoke_results: dict[str, bool] = {}
    for label, fn in smoke_map.items():
        if dl_results.get(label, False):
            smoke_results[label] = fn()
        else:
            smoke_results[label] = False

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Final Summary                                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    all_ok = True
    for label, model_id in MODELS:
        dl  = "✅" if dl_results.get(label)     else "❌"
        smk = "✅" if smoke_results.get(label)  else "⚠ "
        if not dl_results.get(label): all_ok = False
        print(f"║  {dl} DL  {smk} Smoke   {label:<30} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  {'ALL MODELS READY  🚀' if all_ok else 'Some downloads failed — see ❌ above':^52} ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
