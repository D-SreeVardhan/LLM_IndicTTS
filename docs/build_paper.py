#!/usr/bin/env python3
"""
Builds the CS3235 research paper (DOCX) on top of the course template.

    python docs/build_paper.py

Needs the course template "CS3235 Research Paper Template – RVU_v2.0.docx" in the
project root (a course hand-out, not committed), the figures in docs/figures/
and the evaluation outputs in eval/results/.  Result tables are filled from
those files.
"""

import csv
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
RESULTS = ROOT / "eval" / "results"
TEMPLATE = ROOT / "CS3235 Research Paper Template – RVU_v2.0.docx"
OUT = ROOT / "docs" / "CS3235_TeamNo_Multilingual_Voice_Assistant.docx"

REPO_URL = "https://github.com/D-SreeVardhan/LLM_IndicTTS"
LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil"}
BODY_LINE_SPACING = 1.15


# ── Document helpers ─────────────────────────────────────────────────────

class Paper:
    def __init__(self) -> None:
        self.doc = Document(str(TEMPLATE))
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self._style()

    def _style(self) -> None:
        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.size = Pt(12)
        normal.paragraph_format.line_spacing = BODY_LINE_SPACING
        normal.paragraph_format.space_after = Pt(4)
        for name, size in (("Heading 1", 13), ("Heading 2", 12)):
            st = styles[name]
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.color.rgb = RGBColor(0, 0, 0)
            st.paragraph_format.space_before = Pt(8 if name == "Heading 1" else 4)
            st.paragraph_format.space_after = Pt(2)
            st.paragraph_format.keep_with_next = True
        styles["List Bullet"].font.size = Pt(12)
        for name in ("Normal", "Heading 1", "Heading 2", "List Bullet"):
            # Theme-font attributes in the template override explicit font names. The
            # complex-script slot keeps the theme font so Indic text falls back to a font
            # that has the glyphs.
            rfonts = styles[name].element.get_or_add_rPr().get_or_add_rFonts()
            for attr in ("w:asciiTheme", "w:hAnsiTheme"):
                rfonts.attrib.pop(qn(attr), None)
            for attr in ("w:ascii", "w:hAnsi"):
                rfonts.set(qn(attr), "Times New Roman")

    # **bold**, *italic* and _{subscript} inline markup
    def _runs(self, para, text: str, size: float | None = None, italic: bool = False) -> None:
        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\})", text):
            if not part:
                continue
            if part.startswith("**"):
                run = para.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("*"):
                run = para.add_run(part[1:-1])
                run.italic = True
            elif part.startswith("_{"):
                run = para.add_run(part[2:-1])
                run.font.subscript = True
            else:
                run = para.add_run(part)
                run.italic = italic or None
            if size:
                run.font.size = Pt(size)

    def centered(self, text: str, size: float, bold: bool = False, after: float = 2) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(after)
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold

    def h1(self, text: str) -> None:
        self.doc.add_heading(text, level=1)

    def h2(self, text: str) -> None:
        self.doc.add_heading(text, level=2)

    def para(self, text: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._runs(p, text)

    def lead(self, label: str, text: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p.add_run(label)
        run.bold = True
        run.italic = True
        self._runs(p, text)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = BODY_LINE_SPACING
            self._runs(p, item)

    def equation(self, text: str, number: int) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._runs(p, text, italic=True)
        p.add_run(f"     ({number})")

    def caption(self, text: str, keep_with_next: bool) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = keep_with_next
        self._runs(p, text, size=10)

    def figure(self, paths: list[Path], caption: str, width: float) -> None:
        """One image, or several side by side in a borderless row."""
        if len(paths) == 1:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0
            p.add_run().add_picture(str(paths[0]), width=Inches(width))
        else:
            t = self.doc.add_table(rows=1, cols=len(paths))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for cell, path in zip(t.rows[0].cells, paths):
                cp = cell.paragraphs[0]
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.line_spacing = 1.0
                cp.paragraph_format.keep_with_next = True
                cp.add_run().add_picture(str(path), width=Inches(width))
        self.caption(caption, keep_with_next=False)

    def table(self, header: list[str], rows: list[list[str]], caption: str,
              widths: list[float], size: float = 10) -> None:
        self.caption(caption, keep_with_next=True)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, text in zip(t.rows[0].cells, header):
            self._cell(cell, f"**{text}**", size)
            self._shade(cell)
        for row in rows:
            for cell, text in zip(t.add_row().cells, row):
                self._cell(cell, text, size)
        t.autofit = False
        for col, w in zip(t.columns, widths):
            col.width = Inches(w)
        for i, row in enumerate(t.rows):
            for cell, w in zip(row.cells, widths):
                cell.width = Inches(w)
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                tr_pr.append(OxmlElement("w:tblHeader"))
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing = 1.0

    def _cell(self, cell, text: str, size: float) -> None:
        p = cell.paragraphs[0]
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(1)
        self._runs(p, text, size=size)

    @staticmethod
    def _shade(cell) -> None:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "E8E0D0")
        cell._tc.get_or_add_tcPr().append(shd)

    def reference(self, number: int, text: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.45)
        p.paragraph_format.space_after = Pt(3)
        self._runs(p, f"[{number}]\t{text}", size=11)

    def save(self) -> None:
        self.doc.save(str(OUT))


# ── Data ─────────────────────────────────────────────────────────────────

def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Content ──────────────────────────────────────────────────────────────

def build() -> None:
    ml = load_json("summary_multilingual.json")
    ret = load_json("summary_retrieval.json")
    v1 = load_json("summary_hybrid.json")
    v1_rows = load_csv("rag_results_hybrid.csv")
    tool_rows = {r["id"]: r for r in load_csv("tool_router_results.csv")}

    tool_ids = [i for i, r in tool_rows.items() if r["tool"] and r["in_scope"] == "True"]
    v1_tool = [float(r["fact_score"]) for r in v1_rows if r["id"] in tool_ids]
    v1_tool_acc = 100 * sum(v1_tool) / len(v1_tool)
    # Tool answers took a few milliseconds; anything slower went through Param-1.
    v1_llm = [r for r in v1_rows if float(r["latency_ms"]) > 1000]
    v1_llm_rows = [r for r in v1_llm if r["in_scope"] == "True"]
    v1_llm_lat = sorted(float(r["latency_ms"]) / 1000 for r in v1_llm)
    indic = ("hi", "kn", "ta")
    indic_cer = sum(ml[lg]["cer_pct"] for lg in indic) / 3
    routed = sum(ml[lg]["routed_n"] for lg in ml)
    routed_same = sum(ml[lg]["spoken_same"] for lg in ml)

    p = Paper()

    # Title block
    p.centered("CS3235 – Course Project Research Paper", 11)
    p.centered("RV University, Bengaluru", 11, after=10)
    p.centered("Speak, Retrieve, Answer: A Voice-First Multilingual Campus Assistant "
               "Built on Indian Open Models", 16, bold=True, after=8)
    p.centered("Desu Sree Vardhan (1RUA24CSE7004)", 12)
    p.centered("School of Computer Science and Engineering, RV University, Bengaluru, India", 12)
    p.centered("[email address]   |   Guide: [Faculty Name]", 11, after=12)

    p.lead("Abstract— ",
           "Administrative information at Indian universities, such as fees, examination dates and placement "
           "rules, is usually published only in English. This paper presents a voice-in, voice-out assistant "
           "for RV University that answers such questions in Hindi, Kannada, Tamil and English using only "
           "Indian open models: IndicConformer for speech recognition, IndicTrans2 for translation, BharatGen "
           "Param-1 (2.9B) for grounded generation and Indic Parler-TTS for speech synthesis. Fee, calendar and "
           "placement questions are answered by deterministic tools over structured records; other questions go "
           "through hybrid BM25 and bge-m3 retrieval, cited generation and a grounding guard. On a 14-question "
           f"gold set, hybrid retrieval reaches {ret['recall_at_k_pct']['hybrid']:.1f}% recall at five passages "
           f"and the tool router reaches {ret['tool_key_fact_accuracy_pct']:.1f}% key-fact accuracy without the "
           "language model. On spoken questions in four languages, recognition averages a "
           f"{indic_cer:.1f}% character error rate for the Indian languages, and all {routed} fully processed "
           "questions reached the same answering component as their typed English version.")
    p.lead("Keywords— ",
           "Retrieval-Augmented Generation, Indic Speech Recognition, Machine Translation, Text-to-Speech, "
           "Tool Routing, Multilingual Question Answering")

    # 1 Introduction
    p.h1("1. Introduction")
    p.para("RV University (RVU), Bengaluru publishes its fees, calendars and examination, admission and "
           "placement policies as English web pages and PDFs, while its students and their parents speak "
           "Kannada, Hindi, Tamil and other languages and must search long documents for a single date or "
           "amount. Large language models (LLMs) could answer conversationally, but they invent plausible "
           "numbers when they lack the facts [16], most open LLMs handle Kannada and Tamil poorly, and a voice "
           "interface needs speech models that cope with Indic scripts, acronyms such as \"BCA\" and code-mixing.")
    p.para("**Problem statement.** Given a spoken or typed question in Hindi, Kannada, Tamil or English, return a "
           "short, correct, spoken answer in the same language, grounded in RVU's documents, or refuse when they "
           "lack the answer, using only AI4Bharat and BharatGen models as the brief requires. **RQ1:** Can such a "
           "pipeline answer campus questions usefully? **RQ2:** Do structured tools answer factual questions more "
           "reliably than retrieval-augmented generation? **RQ3:** How consistent is it across languages and "
           "between typed and spoken input?")
    p.para("**Contributions.** (i) A voice assistant built only from Indian open models that runs on a 16 GB "
           "laptop; (ii) a two-tier design of deterministic tools over structured records with grounded hybrid "
           "retrieval-augmented generation as the fallback; (iii) a script normaliser for code-mixed and English "
           "speech; (iv) an RVU knowledge base and two evaluation sets released with the code.")

    # 2 Related work
    p.h1("2. Related Work / Literature Review")
    p.para("**Speech and translation.** The Conformer [1] combines convolution and self-attention and is trained "
           "with CTC [2] or the RNN transducer (RNNT) [3]; IndicConformer [19], trained on IndicVoices [4], covers "
           "all 22 scheduled languages, while Whisper [5] is robust for English but weaker on Kannada and Tamil. "
           "IndicTrans2 [6] translates all 22, Aksharantar [7] transliterates Latin and Indic scripts, and "
           "IndicNLPSuite [18] released Indian-language corpora and models.")
    p.para("**Retrieval, generation and speech synthesis.** Retrieval-augmented generation (RAG) [8] conditions a "
           "generator on retrieved passages, which reduces but does not remove hallucination [9], [16]. BM25 [11] "
           "suits exact terms, dense retrievers such as bge-m3 [10] capture paraphrases, and reciprocal rank fusion "
           "(RRF) [12] combines them; Toolformer [17] showed that tool calls improve factual accuracy. Param-1 [13] "
           "is BharatGen's 2.9B English–Hindi model, and Indic Parler-TTS [15] extends Parler-TTS [14] to 21 "
           "Indian languages. Table 1 summarises these works.")
    p.table(
        ["Ref.", "Model", "Data", "Strength", "Limitation here"],
        [
            ["[4], [19]", "IndicConformer", "IndicVoices", "Native ASR, 22 languages", "Spells out acronyms"],
            ["[5]", "Whisper", "680k h web audio", "Robust English ASR", "Weak Kannada, Tamil"],
            ["[6]", "IndicTrans2", "BPCC", "22 scheduled languages", "Sentence-level"],
            ["[8]", "RAG", "Wikipedia", "Grounded QA", "Can invent numbers"],
            ["[10]", "bge-m3", "100+ languages", "Multilingual retrieval", "568M in memory"],
            ["[17]", "Toolformer", "API-call data", "Exact facts via tools", "Needs fine-tuning"],
            ["[14], [15]", "Indic Parler-TTS", "Described speech", "Natural voices", "Slow decoding"],
        ],
        "Table 1: Summary of related work",
        widths=[0.75, 1.25, 1.25, 1.65, 1.6],
        size=9.5,
    )

    # 3 Problem formulation
    p.h1("3. Problem Formulation")
    p.para("Let *L* = {hi, kn, ta, en} be the supported languages and let the knowledge base be *K* = (*S*, *D*), "
           "where *S* holds structured records (fees, calendar events and placement rules) and *D* = {d₁, …, d_{N}} "
           "is a set of text passages. The input is an audio clip *a*, or a text *x*, in a language ℓ ∈ *L*. "
           "The system computes")
    p.equation("q = T_{ℓ→en}(ASR(a, ℓ))", 1)
    p.equation("y_{en} = f_{t}(q, S) if r(q) = t;  else y_{en} = G(q, R_{k}(q, D)) if g(y_{en}, R_{k}) holds", 2)
    p.para("and returns *y*_{ℓ} = *T*_{en→ℓ}(*y*_{en}) spoken as *s* = TTS(*y*_{ℓ}, ℓ), where *r* is the intent router, *f*_{t} a deterministic tool, *R*_{k} the top-*k* retriever, *G* the LLM, "
           "*g* the grounding guard and *T* translation; a rejected answer becomes a refusal naming the "
           "responsible office. We assume short single-turn questions, a 2026–27 knowledge base and one speaker "
           "in a quiet room.")

    # 4 Dataset
    p.h1("4. Dataset")
    p.h2("4.1 Data Source and Description")
    p.para("No public dataset covers RVU, so a knowledge base was built from eight official English sources, "
           "five rvu.edu.in pages and three PDFs (Table A1). It holds 54 text passages (admissions 20, exams 20, "
           "policy 10, fees 4), 32 structured fee records and 53 undergraduate (UG) and postgraduate (PG) "
           "calendar events, all from public documents used for non-commercial academic work.")
    p.h2("4.2 Data Preprocessing")
    p.para("Pages were scraped to Markdown and PDFs converted to text, without menus and repeated headers. Text "
           "was split at headings and into 400-token windows with 80-token overlap; the fee table and calendars "
           "were also parsed into JSON records. Passages were embedded with bge-m3 and indexed with BM25.")
    p.h2("4.3 Data Split")
    p.para("No model is fine-tuned, so there is no training split. The **gold set** has 14 English questions: 11 "
           "answerable (3 fees, 4 exams, 2 admissions, 2 placement) with key facts and source categories, and 3 "
           "out-of-scope. The **multilingual set** has 8 intents written natively in all four languages (32 "
           "questions), spoken by a macOS voice per language. Both were written by the author.")

    # 5 Methodology
    p.h1("5. Proposed Methodology")
    p.h2("5.1 System Architecture")
    p.para("In Fig. 1, IndicConformer transcribes the audio, a script normaliser handles English written in "
           "Indic script, and IndicTrans2 translates the question to English. The router tries five tools (fees, "
           "calendar, placement, vision and mission, programmes); otherwise hybrid retrieval selects passages, "
           "Param-1 writes a cited answer and the guard accepts or refuses it. The answer is translated back and "
           "spoken by Indic Parler-TTS.")
    p.figure([FIG / "architecture.png"], "Fig. 1: Proposed system architecture. Solid arrows show the "
             "retrieval path; the dashed arrow shows the tool path, which skips the LLM.", width=4.9)
    p.h2("5.2 Model Description")
    p.para("Table 2 lists the models, used as released. IndicConformer has CTC and RNNT heads; RNNT is used "
           "because it conditions on earlier tokens. Param-1, a decoder-only Transformer, is the BharatGen model "
           "that fits in memory at 16-bit precision and follows English instructions reliably. Indic Parler-TTS "
           "decodes audio-codec tokens conditioned on the text and a voice description.")
    p.table(
        ["Role", "Model (Hugging Face ID)", "Params", "Precision / device"],
        [
            ["Speech recognition", "ai4bharat/indic-conformer-600m-multilingual", "600M", "fp32 / CPU"],
            ["Translation to English", "ai4bharat/indictrans2-indic-en-dist-200M", "200M", "fp32 / CPU"],
            ["Translation from English", "ai4bharat/indictrans2-en-indic-dist-200M", "200M", "fp32 / CPU"],
            ["Answer generation", "bharatgenai/Param-1-2.9B-Instruct", "2.9B", "fp16 / Apple GPU"],
            ["Dense retrieval", "BAAI/bge-m3", "568M", "fp16"],
            ["Speech synthesis", "ai4bharat/indic-parler-tts", "0.9B", "fp32 / CPU"],
        ],
        "Table 2: Models used in the pipeline",
        widths=[1.55, 3.05, 0.6, 1.3],
        size=9.5,
    )
    p.h2("5.3 LLM / GenAI Design")
    p.para("**Prompting and retrieval.** Param-1 is prompted zero-shot (Appendix C) for three to five plain "
           "sentences that can be translated and spoken, a citation such as [S1] per claim, and \"I don't have "
           "that information\" when the sources are insufficient. BM25 and bge-m3 rankings are fused with RRF, "
           "score(d) = Σ 1/(60 + rank(d)), and the top *k* = 5 passages form the context; decoding uses "
           "temperature 0.3 and at most 150 new tokens.")
    p.para("**Tools, guard and speech.** The router applies regular expressions and keywords to the English "
           "question. An LLM answer is rejected if it is short and uncited or weakly supported, and refusals name "
           "the responsible office. Amounts such as 2,50,000 become \"2 lakh 50 thousand\" before translation, "
           "because IndicTrans2's digit placeholders sometimes leaked. A transcript of English words in Indic "
           "script (\"वेन इज द नेक्स्ट एग्जाम\") is rewritten through English. Speech is synthesised in short "
           "segments in one batch and joined with pauses.")
    p.h2("5.4 Training Details")
    p.para("No model was trained: there is no labelled RVU data, a 2.9B model cannot be fine-tuned on a 16 GB "
           "laptop, and retrieval and tools supply the institutional knowledge. Only inference settings were "
           "tuned, on development questions outside both evaluation sets.")

    # 6 Prototype
    p.h1("6. Prototype and Deployment")
    p.h2("6.1 Prototype Description (UI/UX)")
    p.para("The web interface (React 19, TypeScript, Vite) is a single page styled like a campus notice board, "
           "in off-white, ink, oxblood and turmeric rather than the usual blue of AI products (Fig. A1). Users "
           "pick a language, then speak (press \"Hold & Speak\" to start and again to stop), type or choose a "
           "suggested question; each answer card shows the answer, its sources, a play button and a per-stage "
           "timing trace (Figs. A2, A3).")
    p.h2("6.2 Deployment Details")
    p.para("A FastAPI backend (endpoints for health, transcribe, ask, speak and sources) runs inference on one "
           "worker thread behind the Vite interface, and one script, run.sh, starts both servers and opens the "
           "browser. **Public URL:** not hosted, because the models need about 10 GB of memory and two are "
           "gated; the demonstration runs locally.")
    p.h2("6.3 GitHub Repository")
    p.para(f"{REPO_URL} holds the code, knowledge base, evaluation sets, results and this paper (Table A3).")

    # 7 Experimental setup
    p.h1("7. Experimental Setup")
    p.h2("7.1 Implementation Environment")
    p.para("An Apple MacBook Pro (M1 Pro, 16 GB) with Python 3.11.9, PyTorch 2.10.0 and Transformers 4.46.1. "
           "Param-1 ran on the Apple GPU and the other models on the CPU, since speech synthesis on the GPU "
           "beside Param-1 grew the backend past 30 GB.")
    p.h2("7.2 Evaluation Metrics")
    p.para("For a reference of *N* words with *S* substitutions, *D* deletions and *I* insertions,")
    p.equation("WER = (S + D + I) / N", 3)
    p.para("and CER is the same over characters. **Recall@k** is the share of answerable gold questions with a "
           "top-*k* passage from the gold category; **key-fact accuracy** is the mean share of key facts (amounts, "
           "dates, percentages) found verbatim in the answer; **routing consistency** is the share of questions "
           "answered by the same component as the typed English version.")
    p.h2("7.3 Baselines")
    p.para("BM25-only and dense-only retrieval are the baselines for fusion. For RQ2 the tool router is compared "
           "with an earlier build (5 November 2026) whose tools returned Markdown tables and matched more loosely; "
           "for RQ3 the typed English question is the reference for each intent.")

    # 8 Results
    p.h1("8. Results and Analysis")
    p.h2("8.1 Retrieval and Answer Accuracy")
    rr = ret["recall_at_k_pct"]
    p.para(f"In Table 3, hybrid retrieval and BM25 find a passage from the right category for {rr['hybrid']:.1f}% of "
           f"answerable questions, against {rr['dense']:.1f}% for dense retrieval alone, because campus questions "
           "contain exact terms such as \"MBA\" or \"hall ticket\". For RQ2, the tool router answers "
           f"{ret['tool_answered']} of the 11 answerable questions with {ret['tool_key_fact_accuracy_pct']:.1f}% "
           f"key-fact accuracy, up from {v1_tool_acc:.1f}% for the earlier build, and never captures an "
           "out-of-scope question; its two errors are a B.Tech fee by JEE-Main category, which the records do not "
           "distinguish, and a BCA answer without the skill fee. In the earlier build, the answerable "
           "questions that reached Param-1 scored "
           f"{100 * sum(float(r['fact_score']) for r in v1_llm_rows) / len(v1_llm_rows):.1f}% and took "
           f"{v1_llm_lat[0]:.0f}–{v1_llm_lat[-1]:.0f} s because the machine was swapping; in the final build the "
           "LLM path took 8.5–12.9 s and tool answers 3.4–6.6 s, mostly translation (Figs. A2, A3). Only one "
           "of three out-of-scope questions was refused (Section 9).")
    p.table(
        ["Configuration", "Questions", "Metric", "Result"],
        [
            ["BM25 only", "11 answerable", "Recall@5", f"{rr['bm25']:.1f}%"],
            ["Dense (bge-m3) only", "11 answerable", "Recall@5", f"{rr['dense']:.1f}%"],
            ["**Hybrid (BM25 + bge-m3, RRF)**", "11 answerable", "Recall@5", f"**{rr['hybrid']:.1f}%**"],
            ["Earlier build, same questions", f"{len(tool_ids)} tool-routed", "Key-fact accuracy", f"{v1_tool_acc:.1f}%"],
            ["**Final build, tool path**", f"{ret['tool_answered']} tool-routed", "Key-fact accuracy",
             f"**{ret['tool_key_fact_accuracy_pct']:.1f}%**"],
            ["Final build, tool path", "3 out-of-scope", "Wrong tool matches",
             str(ret["tool_false_matches_out_of_scope"])],
            ["Earlier build, full pipeline", "11 answerable", "Key-fact accuracy", f"{v1['key_fact_accuracy_pct']:.1f}%"],
            ["Earlier build, full pipeline", "3 out-of-scope", "Refusal accuracy", f"{v1['refusal_accuracy_pct']:.1f}%"],
        ],
        "Table 3: Performance comparison on the 14-question gold set",
        widths=[2.55, 1.35, 1.45, 1.0],
    )
    p.h2("8.2 Multilingual Speech Input")
    p.para("For RQ3 (Table 4, Fig. 2), native recognition is accurate, with character error rates of "
           f"{ml['hi']['cer_pct']:.1f}% (Hindi), {ml['kn']['cer_pct']:.1f}% (Kannada) and {ml['ta']['cer_pct']:.1f}% "
           "(Tamil). Word error rates are higher mainly because acronyms are written as separate letters "
           "(\"ಬಿ ಸಿ ಎ\" for \"ಬಿಸಿಎ\") without any change in meaning. English through the Hindi model has the "
           f"highest CER ({ml['en']['cer_pct']:.1f}%). The run was stopped after {routed} of 32 questions when it "
           "and a second copy of the application exhausted memory; all of these reached the same component as "
           f"the typed English question, whether typed or spoken ({routed_same} of {routed}).")
    p.figure([FIG / "results.png"], "Fig. 2: (a) Word and character error rates of the spoken questions; "
             "(b) number of the eight spoken questions per language that were transcribed exactly.", width=4.2)
    p.table(
        ["Language", "WER (%)", "CER (%)", "Exact transcripts", "Same route as English"],
        [[LANG_NAMES[lg], f"{s['wer_pct']:.1f}", f"{s['cer_pct']:.1f}", f"{s['exact']} / {s['n']}",
          f"{s['spoken_same']} / {s['routed_n']}"] for lg, s in ml.items()],
        "Table 4: Speech recognition and routing on the multilingual spoken set",
        widths=[1.1, 0.9, 0.9, 1.5, 1.8],
    )
    p.h2("8.3 Sample Outputs")
    p.para("Figs. A2 and A3 show typical outputs; Table A2 lists notable transcripts and answers.")

    # 9 Discussion
    p.h1("9. Discussion")
    p.para("**Error analysis.** The tool router never asks the LLM for a number and retrieval is strong, so "
           "errors come from three places. *(i) Recognition:* similar words are confused (Tamil சேர்க்கை, admission, heard "
           "as செயற்கை, artificial), and English through Hindi is weakest (\"sports\" became \"spousal\"). "
           "*(ii) Refusal:* RRF scores are normalised so the top passage always scores 1, so the minimum-score "
           "check never fires and refusal depends on Param-1 declining in the expected words. The same path "
           "causes false refusals: Param-1 declined the Tamil attendance question (Fig. A3) although the corpus "
           "states the 80 percent rule. *(iii) Translation:* IndicTrans2 sometimes changes a key word (\"next "
           "exam\" became \"next election\" for Tamil).")
    p.para("**Comparison and limitations.** The low character error rates agree with results reported for "
           "IndicConformer [4], [19] and the benefit of tools agrees with Toolformer [17]; unlike English-only "
           "RAG [8], [9], translate-route-translate behaves consistently across Indian languages. The evaluation "
           "sets are small and synthetic-voiced, the multilingual run did not reach the LLM intents, LLM gold "
           "results come from the earlier build, and speech synthesis is slower than real time.")

    # 10 Ethics
    p.h1("10. Ethical Considerations and Responsible AI")
    p.para("**Privacy:** audio is processed locally and never stored. **Bias:** with one synthetic voice per "
           "language, accented or code-mixed speech may be served worse until real students are tested. "
           "**Misuse:** numbers come from records, answers show sources and unsupported answers point to the "
           "responsible office. **Licensing:** model and document terms were respected.")

    # 11 Conclusion
    p.h1("11. Conclusion and Future Work")
    p.para("A four-language voice assistant built only from Indian open models reached "
           f"{ret['tool_key_fact_accuracy_pct']:.1f}% key-fact accuracy with tools, {rr['hybrid']:.1f}% retrieval "
           f"recall and a {indic_cer:.1f}% native character error rate, with consistent routing across languages. "
           "Future work: calibrated refusal, transliterated English speech, streamed synthesis and real users.")

    p.h1("Acknowledgment")
    p.para("The author thanks [Faculty Name], School of Computer Science and Engineering, RV University, for "
           "guidance, and AI4Bharat (IIT Madras) and the BharatGen consortium for releasing the open models on "
           "which this work is built.")

    p.h1("References")
    for i, ref in enumerate(REFERENCES, start=1):
        p.reference(i, ref)

    p.h1("Appendix")
    p.para("**A. Supplementary figures and tables.**")
    p.figure([FIG / "ui_home.png"], "Fig. A1: Home screen with the language switch, the Hold & Speak button and "
             "suggested questions in Hindi", width=3.6)
    p.figure([FIG / "ui_hindi_answer.png", FIG / "ui_kannada_answer.png"],
             "Fig. A2: Tool answers. Left: Hindi \"अगली परीक्षा कब है\" (when is the next exam), answered by the "
             "calendar tool. Right: Kannada \"ಬಿಸಿಎ ಶುಲ್ಕ ಎಷ್ಟು\" (what is the BCA fee), answered by the fee tool "
             "with the amount spelled out as \"2 ಲಕ್ಷ 50 ಸಾವಿರ\".", width=2.75)
    p.figure([FIG / "ui_tamil_answer.png", FIG / "ui_tamil_refusal.png"],
             "Fig. A3: Tamil. Left: \"அடுத்த தேர்வு எப்போது\" (when is the next exam), answered by the calendar "
             "tool. Right: \"எவ்வளவு வருகை கட்டாயம்\" (how much attendance is mandatory), which went through "
             "retrieval and Param-1 and was refused (Section 9).", width=2.75)
    p.table(
        ["Source", "Type", "Category", "Used by"],
        [
            ["Fee Structure AY 2026-27", "Web page", "fees", "Fee tool, passages"],
            ["Admissions and Refund Policy", "Web page", "admissions", "Passages"],
            ["UG Academic Calendar 2026-27", "Web page", "exams", "Calendar tool, passages"],
            ["PG Academic Calendar 2026-27", "Web page", "exams", "Calendar tool, passages"],
            ["Sports Scholarships 2026-27", "Web page", "admissions", "Passages"],
            ["Examination Policy", "PDF", "exams", "Passages"],
            ["Placement Policy Quick Reference 2027", "PDF", "policy", "Placement tool, passages"],
            ["Vision and Mission Statements", "PDF", "policy", "Vision tool, passages"],
        ],
        "Table A1: Knowledge-base sources",
        widths=[2.7, 0.9, 1.0, 1.9],
        size=9.5,
    )
    p.table(
        ["Input", "Output"],
        [
            ["Spoken Hindi: अगली परीक्षा कब है", "Transcript identical to the question; calendar tool; Hindi "
             "answer listing UG practical and semester-end examination dates (Fig. A2)"],
            ["Spoken Tamil: சேர்க்கையை ரத்து செய்தால் …", "Transcript \"செயற்கையை …\": admission (சேர்க்கை) "
             "misheard as artificial (செயற்கை)"],
            ["Spoken English: How do sports scholarships work?", "Transcript \"How Do Spousal Scholarships "
             "Work\" after recognition through the Hindi model"],
            ["Spoken English while Hindi is selected: When is the next exam?",
             "Hindi model output \"वैन इज़ द नेक्स्ट एग्जाम\", normalised to \"When is the next exam?\""],
            ["Typed English (earlier build): What attendance percentage is mandatory at RVU?",
             "Param-1: \"The attendance requirement at RVU is 80%\", with citation [S2]"],
            ["Typed English: Who won the IPL in 2025?", "Param-1 declined in its own words but was not marked "
             "as a refusal by the guard"],
        ],
        "Table A2: Sample outputs from the evaluation runs",
        widths=[2.6, 3.9],
        size=9.5,
    )
    p.table(
        ["Path", "Contents"],
        [
            ["backend/app/pipeline/", "Speech recognition, script normaliser, translation, tools, retrieval, "
                                      "generation, guard, speech synthesis, orchestrator"],
            ["backend/app/main.py", "FastAPI application and endpoints"],
            ["frontend/", "React and TypeScript user interface"],
            ["data/", "Source documents; passages, fee and calendar records, retrieval index"],
            ["scripts/", "Knowledge-base ingestion and model download"],
            ["eval/", "Evaluation scripts, gold and multilingual sets with audio, results"],
            ["docs/", "This paper, its figures and the scripts that build them"],
            ["run.sh", "One-command launcher for the demonstration"],
        ],
        "Table A3: Repository structure",
        widths=[1.9, 4.6],
        size=9.5,
    )
    p.para(f"**B. Reproducibility.** Code, data and results: {REPO_URL}. Retrieval and tool results: "
           "python eval/rag_eval.py --retrieval-only. Full-pipeline gold set: python eval/rag_eval.py "
           "--ablation hybrid (or bm25_only, dense_only, no_tools). Multilingual set: python "
           "eval/multilingual_eval.py. Figures: python docs/figures/make_figures.py and, with the application "
           "running, python docs/figures/take_screenshots.py.")
    p.para("**C. Generation prompt.** *You are an authoritative assistant for RV University (RVU), Bengaluru. "
           "Answer ONLY using the provided sources below. For every factual claim, cite the source tag like [S1] "
           "inline. If the answer cannot be found in the sources, reply: \"I don't have that information. Please "
           "contact the relevant RVU office.\" Keep the answer concise (3–5 sentences max) and factual. The "
           "answer will be translated and read aloud, so write short, complete, plain sentences: no markdown, "
           "bullet points, tables, headings or symbols like ₹ or %; write 'rupees' and 'percent' instead.* "
           "The numbered sources follow, between \"=== SOURCES ===\" and \"=== END OF SOURCES ===\" markers.")
    p.para("**D. Use of AI tools.** An AI coding assistant (Cursor) was used to help write and debug parts of "
           "the code and to draft and edit this paper. The author designed the system, chose the models and ran "
           "the experiments; all reported numbers are taken from the result files in the repository.")

    p.save()
    print(f"Wrote {OUT}")


REFERENCES = [
    "A. Gulati et al., \"Conformer: Convolution-augmented transformer for speech recognition,\" in *Proc. "
    "Interspeech*, 2020, pp. 5036–5040.",
    "A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, \"Connectionist temporal classification: Labelling "
    "unsegmented sequence data with recurrent neural networks,\" in *Proc. 23rd Int. Conf. Machine Learning "
    "(ICML)*, 2006, pp. 369–376.",
    "A. Graves, \"Sequence transduction with recurrent neural networks,\" arXiv:1211.3711, 2012.",
    "T. Javed et al., \"IndicVoices: Towards building an inclusive multilingual speech dataset for Indian "
    "languages,\" in *Findings of the Association for Computational Linguistics: ACL 2024*, 2024.",
    "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, \"Robust speech recognition via "
    "large-scale weak supervision,\" in *Proc. 40th Int. Conf. Machine Learning (ICML)*, 2023.",
    "J. Gala et al., \"IndicTrans2: Towards high-quality and accessible machine translation models for all 22 "
    "scheduled Indian languages,\" *Transactions on Machine Learning Research*, 2023.",
    "Y. Madhani et al., \"Aksharantar: Open Indic-language transliteration datasets and models for the next "
    "billion users,\" in *Findings of the Association for Computational Linguistics: EMNLP 2023*, 2023.",
    "P. Lewis et al., \"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" in *Proc. Advances "
    "in Neural Information Processing Systems (NeurIPS)*, 2020.",
    "Y. Gao et al., \"Retrieval-augmented generation for large language models: A survey,\" arXiv:2312.10997, "
    "2023.",
    "J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, \"M3-Embedding: Multi-linguality, "
    "multi-functionality, multi-granularity text embeddings through self-knowledge distillation,\" in *Findings "
    "of the Association for Computational Linguistics: ACL 2024*, 2024.",
    "S. Robertson and H. Zaragoza, \"The probabilistic relevance framework: BM25 and beyond,\" *Foundations and "
    "Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.",
    "G. V. Cormack, C. L. A. Clarke, and S. Büttcher, \"Reciprocal rank fusion outperforms Condorcet and "
    "individual rank learning methods,\" in *Proc. 32nd Int. ACM SIGIR Conf. Research and Development in "
    "Information Retrieval*, 2009, pp. 758–759.",
    "BharatGen, \"Param-1-2.9B-Instruct,\" Hugging Face model card, 2025. [Online]. Available: "
    "https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct",
    "D. Lyth and S. King, \"Natural language guidance of high-fidelity text-to-speech with synthetic "
    "annotations,\" arXiv:2402.01912, 2024.",
    "AI4Bharat, \"Indic Parler-TTS,\" Hugging Face model card, 2024. [Online]. Available: "
    "https://huggingface.co/ai4bharat/indic-parler-tts",
    "Z. Ji et al., \"Survey of hallucination in natural language generation,\" *ACM Computing Surveys*, vol. 55, "
    "no. 12, pp. 1–38, 2023.",
    "T. Schick et al., \"Toolformer: Language models can teach themselves to use tools,\" in *Proc. Advances in "
    "Neural Information Processing Systems (NeurIPS)*, 2023.",
    "D. Kakwani et al., \"IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained "
    "multilingual language models for Indian languages,\" in *Findings of the Association for Computational "
    "Linguistics: EMNLP 2020*, 2020, pp. 4948–4961.",
    "AI4Bharat, \"IndicConformer-600M multilingual,\" Hugging Face model card, 2024. [Online]. Available: "
    "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual",
]


if __name__ == "__main__":
    build()
