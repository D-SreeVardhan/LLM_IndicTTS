#!/usr/bin/env python3
"""
multilingual_eval.py
────────────────────
Voice-in evaluation across English, Hindi, Kannada and Tamil.

Every question in eval/datasets/multilingual_qa.json is asked in all four
languages.  Indic questions are spoken by the macOS system voices (Lekha,
Soumya, Vani; Rishi for English) and saved under eval/datasets/utterances/,
so the same audio can be re-scored later with wer_eval.py.

For each (question, language) it reports:
  - ASR word / character error rate of the spoken question
  - which component answered (tool router or Param-1 + retrieval)
  - cross-lingual consistency: same component as the typed English question
  - whether the answer is written in the requested language's script
  - end-to-end latency for the typed and the spoken question

Run (macOS only, because of `say`):
    python eval/multilingual_eval.py
"""

import argparse
import csv
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jiwer import cer, wer
from loguru import logger

from backend.app.pipeline import asr, codemix
from backend.app.pipeline.orchestrator import run as run_pipeline
from backend.app.schemas import AskRequest
from eval.wer_eval import normalise

DATASET = ROOT / "eval" / "datasets" / "multilingual_qa.json"
UTT_DIR = ROOT / "eval" / "datasets" / "utterances"
RESULTS_DIR = ROOT / "eval" / "results"
OUT_CSV = RESULTS_DIR / "multilingual_results.csv"

LANGS = ("en", "hi", "kn", "ta")
VOICES = {"en": "Rishi", "hi": "Lekha", "kn": "Soumya", "ta": "Vani"}
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "kn": (0x0C80, 0x0CFF),
    "ta": (0x0B80, 0x0BFF),
}


def synthesise(text: str, lang: str, wav_path: Path) -> None:
    """Speak *text* with the macOS voice for *lang* into a 16 kHz mono WAV."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "q.aiff"
        subprocess.run(["say", "-v", VOICES[lang], "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
        )
    wav_path.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")


def script_share(text: str, lang: str) -> float:
    """Fraction of letters in *text* that belong to the target language's script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    if lang == "en":
        return sum(ch.isascii() for ch in letters) / len(letters)
    lo, hi = SCRIPT_RANGES[lang]
    return sum(lo <= ord(ch) <= hi for ch in letters) / len(letters)


def answered_by(trace) -> str:
    for stage in trace:
        if stage.stage == "tool_router" and (stage.detail or "").startswith("matched:"):
            return "tool:" + stage.detail.split(":", 1)[1]
    return "llm"


def ask(text: str, lang: str) -> dict:
    resp = run_pipeline(AskRequest(text=text, lang=lang, route="direct"))
    return {
        "answer": resp.answer,
        "answered_by": answered_by(resp.trace),
        "refused": resp.refused,
        "latency_ms": resp.total_latency_ms,
    }


def main() -> None:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 – speech recognition (ASR model is freed before the LLM loads)
    transcripts: dict[tuple[str, str], dict] = {}
    for item in items:
        for lang in LANGS:
            wav = UTT_DIR / lang / f"{item['id']}.wav"
            if not wav.exists():
                synthesise(item[lang], lang, wav)
            result = codemix.transcribe_in_script(wav.read_bytes(), lang, "rnnt")
            transcripts[(item["id"], lang)] = result
            logger.info(f"[{lang}] {item[lang]!r} → {result['text']!r}")
    asr._model = None
    gc.collect()

    # Phase 2 – answer the typed and the spoken question
    rows = []
    for item in items:
        reference_route = None
        for lang in LANGS:
            heard = transcripts[(item["id"], lang)]
            typed = ask(item[lang], lang)
            spoken = ask(heard["text"], lang)
            if lang == "en":
                reference_route = typed["answered_by"]
            rows.append({
                "id": item["id"],
                "lang": lang,
                "question": item[lang],
                "transcript": heard["text"],
                "wer_pct": round(wer(normalise(item[lang]), normalise(heard["text"])) * 100, 1),
                "cer_pct": round(cer(normalise(item[lang]), normalise(heard["text"])) * 100, 1),
                "asr_ms": round(heard["latency_ms"]),
                "typed_route": typed["answered_by"],
                "spoken_route": spoken["answered_by"],
                "typed_consistent": typed["answered_by"] == reference_route,
                "spoken_consistent": spoken["answered_by"] == reference_route,
                "typed_refused": typed["refused"],
                "script_share": round(script_share(typed["answer"], lang), 3),
                "typed_ms": round(typed["latency_ms"]),
                "spoken_ms": round(spoken["latency_ms"]),
                "typed_answer": typed["answer"],
                "spoken_answer": spoken["answer"],
            })
            logger.info(f"[{item['id']}/{lang}] typed={typed['answered_by']} spoken={spoken['answered_by']} "
                        f"wer={rows[-1]['wer_pct']}%")

    save(rows)


def save(rows: list[dict]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for lang in LANGS:
        lr = [r for r in rows if r["lang"] == lang]
        n = len(lr)
        refs = [normalise(r["question"]) for r in lr]
        hyps = [normalise(r["transcript"]) for r in lr]
        summary[lang] = {
            "n": n,
            "wer_pct": round(wer(refs, hyps) * 100, 1),
            "cer_pct": round(cer(refs, hyps) * 100, 1),
            "typed_consistency_pct": round(100 * sum(r["typed_consistent"] for r in lr) / n, 1),
            "spoken_consistency_pct": round(100 * sum(r["spoken_consistent"] for r in lr) / n, 1),
            "answer_script_share_pct": round(100 * sum(r["script_share"] for r in lr) / n, 1),
            "tool_answered": sum(r["typed_route"].startswith("tool") for r in lr),
            "median_typed_ms": sorted(r["typed_ms"] for r in lr)[n // 2],
            "median_spoken_ms": sorted(r["spoken_ms"] for r in lr)[n // 2],
            "median_asr_ms": sorted(r["asr_ms"] for r in lr)[n // 2],
        }
    (RESULTS_DIR / "summary_multilingual.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.success(f"Results saved to {OUT_CSV}")
    for lang, s in summary.items():
        logger.info(f"{lang}: {s}")


def rescore() -> None:
    """Recompute error rates and the summary from a saved CSV (no models needed)."""
    with OUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        ref, hyp = normalise(r["question"]), normalise(r["transcript"])
        r["wer_pct"] = round(wer(ref, hyp) * 100, 1)
        r["cer_pct"] = round(cer(ref, hyp) * 100, 1)
        for key in ("typed_consistent", "spoken_consistent", "typed_refused"):
            r[key] = r[key] == "True"
        for key in ("asr_ms", "typed_ms", "spoken_ms"):
            r[key] = int(r[key])
        r["script_share"] = float(r["script_share"])
    save(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true",
                        help="re-score the saved results CSV without re-running the models")
    if parser.parse_args().rescore:
        rescore()
    else:
        main()
