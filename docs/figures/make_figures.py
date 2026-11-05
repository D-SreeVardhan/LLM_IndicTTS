#!/usr/bin/env python3
"""
Figures for the research paper.

    python docs/figures/make_figures.py

Writes architecture.png, and results.png when the eval summaries exist.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "eval" / "results"

INK = "#1a1a1a"
OXBLOOD = "#7b0a0a"
TURMERIC = "#c88e00"
PAPER = "#f5f0e8"
PAPER_DARK = "#e8e0d0"
MUTED = "#6b6257"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})


def _box(ax, x, y, w, h, title, sub, face=PAPER_DARK, edge=INK, title_colour=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=face, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=title_colour)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color=MUTED)


def _arrow(ax, x1, y1, x2, y2, label=None, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=colour, linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom",
                fontsize=8, color=colour, style="italic")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    w, h = 1.55, 0.85
    top = 3.35
    # Input lane
    _box(ax, 0.1, top, w, h, "Speech / Text", "hi · kn · ta · en", face=PAPER, edge=OXBLOOD,
         title_colour=OXBLOOD)
    _box(ax, 2.05, top, w, h, "IndicConformer", "600M · RNNT (ASR)")
    _box(ax, 4.0, top, w, h, "Script normaliser", "code-mix → native")
    _box(ax, 5.95, top, w, h, "IndicTrans2", "indic→en · 200M")
    _arrow(ax, 0.1 + w, top + h / 2, 2.05, top + h / 2)
    _arrow(ax, 2.05 + w, top + h / 2, 4.0, top + h / 2)
    _arrow(ax, 4.0 + w, top + h / 2, 5.95, top + h / 2)

    # Router
    mid = 1.85
    _box(ax, 7.9, mid + 0.75, 1.95, h, "Intent router", "regex + keywords", face=PAPER, edge=TURMERIC)
    _arrow(ax, 5.95 + w, top + h / 2, 8.85, mid + 0.75 + h)

    # Tool path
    _box(ax, 7.9, mid - 0.55, 1.95, h, "Structured tools", "fees · calendar · placement")
    _arrow(ax, 8.85, mid + 0.75, 8.85, mid - 0.55 + h, label=None, colour=TURMERIC)
    ax.text(8.95, mid + 0.42, "match", fontsize=8, color=TURMERIC, style="italic")

    # RAG path
    _box(ax, 5.95, mid + 0.1, w, h, "Hybrid retrieval", "BM25 + bge-m3 · RRF")
    _box(ax, 4.0, mid + 0.1, w, h, "Param-1 2.9B", "grounded, cited answer")
    _box(ax, 2.05, mid + 0.1, w, h, "Grounding guard", "citation + score check")
    _arrow(ax, 7.9, mid + 0.75 + h / 2, 5.95 + w, mid + 0.1 + h / 2, colour=MUTED)
    ax.text(7.05, mid + 1.05, "no match", fontsize=8, color=MUTED, style="italic")
    _arrow(ax, 5.95, mid + 0.1 + h / 2, 4.0 + w, mid + 0.1 + h / 2)
    _arrow(ax, 4.0, mid + 0.1 + h / 2, 2.05 + w, mid + 0.1 + h / 2)

    # Output lane
    low = 0.15
    _box(ax, 2.05, low, w, h, "IndicTrans2", "en→indic · 200M")
    _box(ax, 4.0, low, w, h, "Indic Parler-TTS", "938M · 44.1 kHz")
    _box(ax, 5.95, low, w, h, "Answer + audio", "sources shown as chips", face=PAPER, edge=OXBLOOD,
         title_colour=OXBLOOD)
    _arrow(ax, 2.05 + w / 2, mid + 0.1, 2.05 + w / 2, low + h)
    _arrow(ax, 7.9 + 0.2, mid - 0.55 + h / 2, 2.05 + w, low + h * 0.75, colour=TURMERIC, dashed=True)
    _arrow(ax, 2.05 + w, low + h / 2, 4.0, low + h / 2)
    _arrow(ax, 4.0 + w, low + h / 2, 5.95, low + h / 2)

    fig.savefig(HERE / "architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results() -> None:
    ml = RESULTS / "summary_multilingual.json"
    if not ml.exists():
        return
    summary = json.loads(ml.read_text())
    langs = ["en", "hi", "kn", "ta"]
    names = {"en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = range(len(langs))
    wer_vals = [summary[lg]["wer_pct"] for lg in langs]
    cer_vals = [summary[lg]["cer_pct"] for lg in langs]
    left.bar([i - 0.18 for i in x], wer_vals, 0.36, color=OXBLOOD, label="WER")
    left.bar([i + 0.18 for i in x], cer_vals, 0.36, color=TURMERIC, label="CER")
    left.set_xticks(list(x), [names[lg] for lg in langs])
    left.set_ylabel("Error rate (%)")
    left.set_title("(a) Speech recognition of spoken questions")
    left.legend(frameon=False)

    typed = [summary[lg]["typed_consistency_pct"] for lg in langs]
    spoken = [summary[lg]["spoken_consistency_pct"] for lg in langs]
    right.bar([i - 0.18 for i in x], typed, 0.36, color=INK, label="Typed question")
    right.bar([i + 0.18 for i in x], spoken, 0.36, color=MUTED, label="Spoken question")
    right.set_xticks(list(x), [names[lg] for lg in langs])
    right.set_ylim(0, 105)
    right.set_ylabel("Same route as English (%)")
    right.set_title("(b) Cross-lingual routing consistency")
    right.legend(frameon=False, loc="lower left")

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "results.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results()
