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


def _connector(ax, points, colour=INK, dashed=False):
    """Orthogonal connector through `points` (each turn is 90 degrees), arrowhead on the last leg."""
    style = "--" if dashed else "-"
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=colour, lw=1.1, linestyle=style, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=colour, linestyle=style, shrinkA=0, shrinkB=0))


def _label(ax, x, y, text, colour, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=8, color=colour, style="italic")


def architecture() -> None:
    """Three rows that snake: input → router (left to right), retrieval → guard (right to left),
    then the spoken answer (right to left). The tool path runs down a lane right of the grid."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    w, h, step = 1.6, 0.85, 1.95
    col = [0.1 + i * step for i in range(5)]
    top, mid, low = 3.4, 1.8, 0.2

    def left(c, y):
        return col[c], y + h / 2

    def right(c, y):
        return col[c] + w, y + h / 2

    def above(c, y):
        return col[c] + w / 2, y + h

    def below(c, y):
        return col[c] + w / 2, y

    # Row 1: question in, translated to English, routed
    _box(ax, col[0], top, w, h, "Speech / Text", "hi · kn · ta · en", face=PAPER, edge=OXBLOOD,
         title_colour=OXBLOOD)
    _box(ax, col[1], top, w, h, "IndicConformer", "600M · RNNT (ASR)")
    _box(ax, col[2], top, w, h, "Script normaliser", "code-mix → native")
    _box(ax, col[3], top, w, h, "IndicTrans2", "indic→en · 200M")
    _box(ax, col[4], top, w, h, "Intent router", "regex + keywords", face=PAPER, edge=TURMERIC)
    for c in range(4):
        _connector(ax, [right(c, top), left(c + 1, top)])

    # Row 2: retrieval path, right to left
    _box(ax, col[4], mid, w, h, "Hybrid retrieval", "BM25 + bge-m3 · RRF")
    _box(ax, col[3], mid, w, h, "Param-1 2.9B", "grounded, cited answer")
    _box(ax, col[2], mid, w, h, "Grounding guard", "citation + grounding")
    _connector(ax, [below(4, top), above(4, mid)], colour=MUTED)
    _label(ax, col[4] + w / 2 + 0.08, (top + mid + h) / 2, "no match", MUTED)
    _connector(ax, [left(4, mid), right(3, mid)])
    _connector(ax, [left(3, mid), right(2, mid)])

    # Row 3: answer out, right to left; the tool answer joins IndicTrans2 from the right
    _box(ax, col[4], low, w, h, "Structured tools", "fees · dates · placement")
    _box(ax, col[2], low, w, h, "IndicTrans2", "en→indic · 200M")
    _box(ax, col[1], low, w, h, "Indic Parler-TTS", "938M · 44.1 kHz")
    _box(ax, col[0], low, w, h, "Answer + audio", "with source chips", face=PAPER, edge=OXBLOOD,
         title_colour=OXBLOOD)
    _connector(ax, [below(2, mid), above(2, low)])
    _connector(ax, [left(2, low), right(1, low)])
    _connector(ax, [left(1, low), right(0, low)])

    lane = col[4] + w + 0.35
    _connector(ax, [right(4, top), (lane, top + h / 2), (lane, low + h / 2), right(4, low)],
               colour=TURMERIC, dashed=True)
    _label(ax, lane + 0.08, (mid + low + h) / 2 + 0.4, "match", TURMERIC)
    _connector(ax, [left(4, low), right(2, low)], colour=TURMERIC, dashed=True)

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

    exact = [summary[lg]["exact"] for lg in langs]
    rest = [summary[lg]["n"] - e for lg, e in zip(langs, exact)]
    right.bar(list(x), exact, 0.5, color=INK, label="Transcribed exactly")
    right.bar(list(x), rest, 0.5, bottom=exact, color=PAPER_DARK, edgecolor=MUTED, label="With errors")
    right.set_xticks(list(x), [names[lg] for lg in langs])
    right.set_ylim(0, 9)
    right.set_ylabel("Spoken questions (of 8)")
    right.set_title("(b) Exact transcripts per language")
    right.legend(frameon=False, loc="upper right", ncols=2)

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "results.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results()
