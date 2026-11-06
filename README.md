# RVU Multilingual Voice Assistant

**CS3235 course project · RV University, Bengaluru**
Desu Sree Vardhan (1RUA24CSE7004)

Ask a question about RV University out loud in **Hindi, Kannada, Tamil or English**, and get a short,
sourced answer back in the same language, as text and as speech. It covers fees, exam dates, the academic
calendar, admissions and refunds, scholarships and placement rules.

Every model in the pipeline comes from Indian open-model efforts: **AI4Bharat** (IndicConformer,
IndicTrans2, Indic Parler-TTS) and **BharatGen** (Param-1). The whole system runs locally on a 16 GB
Apple Silicon laptop, with no fine-tuning.

**Research paper:** [`docs/CS3235_Multilingual_Voice_Assistant.pdf`](docs/CS3235_Multilingual_Voice_Assistant.pdf)
(Word version alongside it; rebuilt with `python docs/build_paper.py`).

**Study guide:** [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md) walks through every stage, from the
microphone to the spoken answer, with the models, how they work, a real traced example and self-test
questions.

---

## How it works

```
 Speech / text (hi · kn · ta · en)
        │
        ▼
 IndicConformer-600M ──► script normaliser ──► IndicTrans2 (indic → en)
   (speech → text)        (English spoken in                │
                           Indic script → native)           ▼
                                                     Intent router
                                       match ┌───────────┴───────────┐ no match
                                             ▼                       ▼
                                  Structured tools          Hybrid retrieval (BM25 + bge-m3, RRF)
                                  fees · calendar ·                  ▼
                                  placement · vision ·      Param-1 2.9B (cited answer)
                                  programmes                         ▼
                                             │              Grounding guard (refuse if unsupported)
                                             └───────────┬───────────┘
                                                         ▼
                                   IndicTrans2 (en → indic) ──► Indic Parler-TTS ──► answer + audio
```

- **Tools before the LLM.** Fee amounts, calendar dates and placement rules are read from structured
  records, so the numbers students get are never invented by the model.
- **Grounded generation.** Everything else goes through hybrid retrieval and Param-1, which must cite its
  passages. A guard refuses unsupported answers and points to the right office.
- **Code-mixed speech.** If you speak English while Hindi, Kannada or Tamil is selected, the transcript
  comes back as English in Indic script ("वेन इज द नेक्स्ट एग्जाम"). The normaliser detects this and
  rewrites it in the native language ("अगली परीक्षा कब होगी?").
- **English voice input.** IndicConformer has no English model, so English speech is recognised with the
  Hindi model and translated to English with IndicTrans2.
- **Speakable answers.** Answers are plain sentences. Amounts are spelled out ("2 lakh 50 thousand"), and
  markup and symbols are removed so translation and speech synthesis read them naturally.

## Models

| Role | Model | Params |
|------|-------|--------|
| Speech recognition | [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M |
| Translation | [`ai4bharat/indictrans2-indic-en-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M), [`indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) | 2 × 200M |
| Answer generation | [`bharatgenai/Param-1-2.9B-Instruct`](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct) | 2.9B |
| Speech synthesis | [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) | 0.9B |
| Dense retrieval | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | 568M |

---

## Quick start

### Requirements

- macOS on Apple Silicon with 16 GB RAM (or Linux with a CUDA GPU)
- Python 3.11, Node.js 20, ffmpeg
- A Hugging Face account that has accepted the licences of the gated models (Param-1 and Indic Parler-TTS)

### 1. Install

```bash
pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit
pip install git+https://github.com/huggingface/parler-tts.git
```

### 2. Configure

```bash
cp .env.example .env      # then set HF_TOKEN
```

### 3. Download the models (about 20 GB of disk, one time)

```bash
python scripts/download_models.py
```

The processed knowledge base and retrieval index are already in `data/processed/`. To rebuild them from
the source pages and PDFs, run `python scripts/ingest.py`.

### 4. Run

```bash
./run.sh            # loads every model up front, then opens http://localhost:5173
./run.sh --fast     # loads models on first use: quick start and lower memory (recommended on 16 GB)
./run.sh --no-open  # don't open the browser
```

`run.sh` checks the prerequisites, installs frontend packages if needed, frees ports 8000 and 5173, starts
the FastAPI backend and the Vite frontend, waits until both are ready, and stops both on **Ctrl+C**.

**Demo tips:** close other heavy apps first. Tool-answered questions (fees, exam dates, placement rules)
come back in a few seconds. Open-ended questions go through Param-1 and take longer. Spoken audio takes
about a minute on a CPU and covers the first couple of sentences.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Readiness probe |
| `/api/transcribe` | POST | Audio (`multipart`, `lang`) → transcript in the selected language's script |
| `/api/ask` | POST | `{text, lang, route}` → answer, citations, refusal flag, per-stage trace |
| `/api/speak` | POST | `{text, lang}` → WAV audio |
| `/api/sources/{id}` | GET | Passage text behind a citation |

## Evaluation

```bash
# Retrieval recall and tool-router accuracy only (no LLM, runs in under a minute)
python eval/rag_eval.py --retrieval-only

# Answer quality on the 14-question English gold set, with ablations
python eval/rag_eval.py --ablation hybrid       # full system
python eval/rag_eval.py --ablation bm25_only
python eval/rag_eval.py --ablation dense_only
python eval/rag_eval.py --ablation no_tools     # plain RAG baseline

# 8 intents × 4 languages, spoken by the macOS system voices (macOS only)
python eval/multilingual_eval.py                # ASR error rates, routing consistency, latency
python eval/multilingual_eval.py --rescore      # re-score saved results without the models

# CTC vs RNNT decoding on the saved Indic utterances
python eval/wer_eval.py --decode both
```

Results are written to `eval/results/`. Headline numbers (details and caveats in the paper):

| Measurement | Result |
|-------------|--------|
| Recall@5 on 11 answerable gold questions (hybrid / BM25 / dense) | 90.9% / 90.9% / 81.8% |
| Key-fact accuracy of the tool router (8 tool-answered questions) | 81.2% (earlier build: 68.8%) |
| Out-of-scope questions wrongly captured by a tool | 0 of 3 |
| Speech recognition character error rate (Hindi / Kannada / Tamil / English) | 3.0% / 2.4% / 3.5% / 7.6% |
| Spoken questions routed like their typed English version | 17 of 17 processed |

The multilingual run was stopped after 17 of its 32 questions when the laptop ran out of memory, so it
does not yet cover the LLM-answered intents.

## Knowledge base

| Source | Category |
|--------|----------|
| [Fee Structure AY 2026-27](https://rvu.edu.in/fee-structure/fee-structure-ay-2026-27/) | fees |
| [Admissions and Refund Policy](https://rvu.edu.in/admissions/) | admissions |
| [UG Academic Calendar 2026-27](https://rvu.edu.in/academic-calendar/academic-calendar-for-ay-2026-27/) | exams |
| [PG Academic Calendar 2026-27](https://rvu.edu.in/academic-calendar/pg-academic-calendar-2026-27/) | exams |
| [Sports Scholarships 2026-27](https://rvu.edu.in/scholarships/sports-scholarships/) | admissions |
| Examination Policy (PDF) | exams |
| Placement Policy Quick Reference 2027 (PDF) | policy |
| Vision and Mission Statements (PDF) | policy |

| RVU Campus Guide (synthetic demo data) | campus |

54 official passages, 32 structured fee records and 53 calendar events, plus 43 synthetic campus-guide
passages.

The campus guide (`data/raw/synthetic/rvu_campus_guide.md`) is **made-up demo data** so the assistant can
answer everyday student questions the official pages don't cover: hostel charges, rules and mess timings,
buses, library and gym hours, Wi-Fi, clubs, certificates, anti-ragging, counselling, safety and emergency
numbers, labs and the maker space, startups, research, online courses, laptops, plagiarism, NSS and NCC,
accessibility, printing and the ATM, parents' updates and key contacts. Answers drawn
from it cite "RVU Campus Guide (synthetic demo data)". Its figures are not real RVU figures. It was added
after the paper's evaluation, so the paper's numbers describe the official corpus only. On the expanded
corpus, hybrid Recall@5 is 81.8% (`eval/results/summary_retrieval_with_campus_guide.json`). The two
misses are a fee question that the fee tool answers anyway and an attendance question whose correct
passage is retrieved under a different gold label.

After editing the guide, rebuild only its passages and the index (the official passages are kept):

```bash
python scripts/ingest.py --synthetic-only
```

Questions it can now answer: "What is the hostel fee?", "Is there a bus service?", "When is the library
open?", "How do I get a bonafide certificate?", "Who do I contact about ragging?", "What are the emergency
numbers?", "Do I need a laptop?", "Can I get credits for NPTEL courses?", "Is there an ATM on campus?",
"How do I get support for my startup idea?"

## Project structure

```
LLM_Prj/
├── backend/app/
│   ├── main.py               FastAPI app and endpoints
│   ├── config.py             Settings (.env)
│   ├── schemas.py            Request / response models
│   └── pipeline/
│       ├── asr.py            IndicConformer speech recognition
│       ├── codemix.py        Code-mixed transcripts, English-via-Hindi speech path
│       ├── translate.py      IndicTrans2, speakable amounts
│       ├── tools.py          Fee / calendar / placement / programme tools
│       ├── retrieve.py       BM25 + bge-m3 hybrid retrieval
│       ├── generate.py       Param-1 grounded generation
│       ├── guard.py          Grounding guard and refusals
│       ├── tts.py            Indic Parler-TTS
│       └── orchestrator.py   End-to-end pipeline
├── frontend/                 React + TypeScript + Vite interface
├── data/
│   ├── raw/                  Source PDFs, scraped pages, synthetic/ campus guide
│   └── processed/            Passages, fee and calendar records, retrieval index
├── eval/
│   ├── rag_eval.py           Gold-set evaluation and ablations
│   ├── multilingual_eval.py  Four-language spoken evaluation
│   ├── wer_eval.py           ASR word / character error rate
│   ├── datasets/             gold_qa.json, multilingual_qa.json, utterances/
│   └── results/              CSV and JSON outputs
├── scripts/
│   ├── ingest.py             Builds the knowledge base and index
│   └── download_models.py    Downloads and smoke-tests the models
├── docs/                     Research paper, figures and study guide
├── run.sh                    One-command launcher
├── requirements.txt
└── .env.example
```

## Design notes

- **Param-1 is English/Hindi only**, so Kannada and Tamil questions always go through English. Hindi uses
  the English pivot by default (`HINDI_ROUTE` in `.env`).
- **Speech synthesis runs on the CPU.** On the Apple GPU it was about twice as fast, but next to Param-1 it
  pushed the backend past 30 GB of memory on a 16 GB machine.
- **Full answers are spoken.** The answer is split into short segments at sentence and comma boundaries
  and synthesised as a batch. Parler-TTS sometimes stops a segment early, so any segment whose audio is
  too short for its text is generated once more and the longer take is kept.
- **No generic AI look.** The interface uses a campus notice-board palette (off-white paper, ink black,
  oxblood and turmeric), not the usual blue gradients.
