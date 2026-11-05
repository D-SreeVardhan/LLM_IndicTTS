#!/usr/bin/env python3
"""
wer_eval.py
───────────
Evaluate IndicConformer ASR on:
  - A Kathbath / IndicVoices subset  (files in eval/datasets/utterances/)
  - Optional team-recorded RVU questions

Reports WER and CER per language for CTC vs RNNT decoding.
Writes results to eval/results/wer_results.csv.

Usage:
    python eval/wer_eval.py [--decode ctc|rnnt|both] [--lang hi|kn|ta|all]
"""

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
from jiwer import cer, wer
from loguru import logger

RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UTT_DIR = ROOT / "eval" / "datasets" / "utterances"
ASR_LANGS = ("hi", "kn", "ta")

# ── Normaliser ────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, drop punctuation/symbols and zero-width joiners for WER.

    Filters by Unicode category rather than regex \\w, which does not match
    Indic vowel signs and would split every word apart.
    """
    text = unicodedata.normalize("NFKC", text).replace("\u200c", "").replace("\u200d", "")
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.lower().split())


# ── Load ASR model ────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModel
    from backend.app.config import settings

    logger.info(f"Loading IndicConformer ({settings.asr_model_id}) …")
    model = AutoModel.from_pretrained(settings.asr_model_id, trust_remote_code=True)
    model.eval()
    return model


def transcribe_file(model, wav_path: Path, lang: str, decode_mode: str) -> str:
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    with torch.no_grad():
        out = model(wav, lang, decode_mode)
    return out if isinstance(out, str) else str(out)


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate(
    model,
    manifest: list[dict],   # list of {audio_path, reference, lang}
    decode_mode: str,
) -> dict[str, list]:
    per_lang: dict[str, dict[str, list]] = {}

    for item in manifest:
        lang = item["lang"]
        if lang not in per_lang:
            per_lang[lang] = {"hyp": [], "ref": []}
        wav_path = Path(item["audio_path"])
        if not wav_path.is_absolute():
            wav_path = ROOT / wav_path
        if not wav_path.exists():
            logger.warning(f"  Missing: {wav_path}")
            continue
        try:
            hyp = transcribe_file(model, wav_path, lang, decode_mode)
        except Exception as exc:
            logger.error(f"  Error on {wav_path}: {exc}")
            continue
        per_lang[lang]["hyp"].append(normalise(hyp))
        per_lang[lang]["ref"].append(normalise(item["reference"]))

    results = {}
    for lang, data in per_lang.items():
        if not data["hyp"]:
            continue
        w = wer(data["ref"], data["hyp"])
        c = cer(data["ref"], data["hyp"])
        results[lang] = {"wer": round(w * 100, 2), "cer": round(c * 100, 2), "n": len(data["hyp"])}
        logger.info(f"  [{lang}] WER={w*100:.1f}%  CER={c*100:.1f}%  (n={len(data['hyp'])})")
    return results


# ── Manifest builder ─────────────────────────────────────────────────────

def build_manifest(lang_filter: str | None) -> list[dict]:
    """
    Scan eval/datasets/utterances/ for *.wav + matching *.txt transcripts.
    Directory structure expected:
        utterances/<lang>/<id>.wav
        utterances/<lang>/<id>.txt
    """
    manifest = []
    if not UTT_DIR.exists():
        logger.warning(f"Utterances directory not found: {UTT_DIR}")
        logger.warning("Creating example structure. Add real recordings to proceed.")
        _create_example_structure()
        return manifest

    for lang_dir in sorted(UTT_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if lang not in ASR_LANGS:
            continue
        if lang_filter and lang_filter != "all" and lang != lang_filter:
            continue
        for wav_file in sorted(lang_dir.glob("*.wav")):
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                logger.warning(f"  No reference for {wav_file.name}")
                continue
            reference = txt_file.read_text(encoding="utf-8").strip()
            manifest.append({
                "audio_path": str(wav_file),
                "reference": reference,
                "lang": lang,
            })
    logger.info(f"Manifest: {len(manifest)} utterances")
    return manifest


def _create_example_structure():
    """Create placeholder directories so users know where to put recordings."""
    for lang in ["hi", "kn", "ta"]:
        (UTT_DIR / lang).mkdir(parents=True, exist_ok=True)
    readme = UTT_DIR / "README.md"
    readme.write_text(
        "# Utterances dataset\n\n"
        "Place recordings here:\n"
        "  utterances/<lang_code>/<id>.wav  (16 kHz mono)\n"
        "  utterances/<lang_code>/<id>.txt  (reference transcript)\n\n"
        "Supported lang codes: hi, kn, ta\n\n"
        "Sources:\n"
        "- Kathbath: https://github.com/AI4Bharat/Kathbath\n"
        "- IndicVoices: https://huggingface.co/datasets/ai4bharat/IndicVoices\n"
        "- Team-recorded RVU questions (campus vocabulary)\n",
        encoding="utf-8",
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", choices=["ctc", "rnnt", "both"], default="both")
    parser.add_argument("--lang", default="all")
    args = parser.parse_args()

    manifest = build_manifest(args.lang)
    if not manifest:
        logger.error("No utterances found. See eval/datasets/utterances/README.md")
        sys.exit(1)

    model = load_model()

    modes = ["ctc", "rnnt"] if args.decode == "both" else [args.decode]
    all_rows = []

    for mode in modes:
        logger.info(f"\n── Decoding: {mode} ──")
        results = evaluate(model, manifest, mode)
        for lang, metrics in results.items():
            all_rows.append({
                "decode_mode": mode,
                "lang": lang,
                "wer_pct": metrics["wer"],
                "cer_pct": metrics["cer"],
                "n_utterances": metrics["n"],
            })

    # Save CSV
    out_csv = RESULTS_DIR / "wer_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decode_mode", "lang", "wer_pct", "cer_pct", "n_utterances"])
        writer.writeheader()
        writer.writerows(all_rows)
    logger.success(f"WER results saved to {out_csv}")

    # Print summary table
    logger.info("\n── Summary ──")
    logger.info(f"{'Mode':<8} {'Lang':<6} {'WER %':<10} {'CER %':<10} {'N'}")
    for r in all_rows:
        logger.info(f"{r['decode_mode']:<8} {r['lang']:<6} {r['wer_pct']:<10} {r['cer_pct']:<10} {r['n_utterances']}")


if __name__ == "__main__":
    main()
