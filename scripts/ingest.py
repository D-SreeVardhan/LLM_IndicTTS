#!/usr/bin/env python3
"""
ingest.py
─────────
Full ingestion pipeline:
  1. Scrape official RVU web pages  → data/raw/web/
  2. Download RVU policy PDFs       → data/raw/pdfs/
  3. OCR scanned PDFs via Apple Vision (ocrmac + PyMuPDF)
  4. Parse fee table  → data/processed/fees.json
  5. Parse calendars  → data/processed/calendar.json
  6. Chunk all text   → data/processed/chunks.jsonl
  7. Embed chunks     → data/processed/index/

Run:
    python scripts/ingest.py [--rebuild-index]
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger

RAW_PDF_DIR = ROOT / "data" / "raw" / "pdfs"
RAW_WEB_DIR = ROOT / "data" / "raw" / "web"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = PROCESSED_DIR / "index"

for d in [RAW_PDF_DIR, RAW_WEB_DIR, PROCESSED_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Source definitions ────────────────────────────────────────────────────

WEB_SOURCES = [
    {
        "title": "RVU Fee Structure AY 2026-27",
        "url": "https://rvu.edu.in/fee-structure/fee-structure-ay-2026-27/",
        "category": "fees",
    },
    {
        "title": "RVU Admissions and Refund Policy",
        "url": "https://rvu.edu.in/admissions/",
        "category": "admissions",
    },
    {
        "title": "RVU UG Academic Calendar 2026-27",
        "url": "https://rvu.edu.in/academic-calendar/academic-calendar-for-ay-2026-27/",
        "category": "exams",
    },
    {
        "title": "RVU PG Academic Calendar 2026-27",
        "url": "https://rvu.edu.in/academic-calendar/pg-academic-calendar-2026-27/",
        "category": "exams",
    },
    {
        "title": "RVU Sports Scholarships 2026-27",
        "url": "https://rvu.edu.in/scholarships/sports-scholarships/",
        "category": "admissions",
    },
]

PDF_REMOTE = [
    {
        "title": "RVU Examination Policy",
        "url": "https://d2b6aloc836m7p.cloudfront.net/wp-content/uploads/2026/04/RVU-Examination-Policy.pdf",
        "category": "exams",
        "filename": "RVU-Examination-Policy.pdf",
    },
    {
        "title": "RVU UGC Proforma",
        "url": "https://rvu.edu.in/wp-content/uploads/2026/01/UGC-Proforma-01.12.2025-1.pdf",
        "category": "policy",
        "filename": "UGC-Proforma.pdf",
    },
]

LOCAL_PDFS = [
    {
        "title": "RVU Placement Policy Quick Reference 2027",
        "filename": "PLACEMENT POLICY – QUICK REFERENCE (2027).pdf",
        "category": "policy",
        "needs_ocr": False,
    },
    {
        "title": "RVU Vision and Mission Statements",
        "filename": "RVU - Vision & Mission Statements.pdf",
        "category": "policy",
        "needs_ocr": False,
    },
    {
        "title": "RVU Programme Outcomes 2025",
        "filename": "Programme Outcomes 2025.pdf",
        "category": "policy",
        "needs_ocr": True,  # Scanned – needs Apple Vision OCR
    },
]

CHUNK_SIZE = 400      # tokens (approx – we use word count × 1.3)
CHUNK_OVERLAP = 80


# ── Helpers ───────────────────────────────────────────────────────────────

def _chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _word_tokens(text: str) -> int:
    """Rough token estimate: words × 1.3."""
    return int(len(text.split()) * 1.3)


def _split_into_chunks(text: str, source_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Split text on heading boundaries then enforce size limit."""
    # Split on markdown headings or ALL-CAPS lines first
    sections = re.split(r"\n(?=#{1,3} |\n[A-Z][A-Z ]{5,}\n)", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()
        if _word_tokens(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            # Sliding window within the section
            step = max(1, int(CHUNK_SIZE / 1.3) - int(CHUNK_OVERLAP / 1.3))
            size = int(CHUNK_SIZE / 1.3)
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + size])
                if chunk:
                    chunks.append(chunk)

    result = []
    for chunk in chunks:
        cid = _chunk_id(chunk)
        result.append(
            {
                "chunk_id": cid,
                "title": source_meta["title"],
                "source_url": source_meta.get("url"),
                "category": source_meta.get("category", "policy"),
                "text": chunk,
            }
        )
    return result


# ── Step 1: Scrape web pages ──────────────────────────────────────────────

def scrape_web() -> list[dict[str, Any]]:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    all_chunks: list[dict[str, Any]] = []

    for src in WEB_SOURCES:
        slug = re.sub(r"[^a-z0-9]+", "_", src["url"].lower().split("rvu.edu.in")[-1])[:60]
        cached = RAW_WEB_DIR / f"{slug}.md"

        if cached.exists():
            md = cached.read_text(encoding="utf-8")
            logger.info(f"  [cache] {src['title']}")
        else:
            logger.info(f"  [fetch] {src['title']}")
            try:
                resp = requests.get(src["url"], timeout=30, headers={"User-Agent": "RVU-Bot/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                # Remove nav/footer/script noise
                for tag in soup.select("header, footer, nav, script, style, .menu, .sidebar"):
                    tag.decompose()
                main = soup.select_one("main, article, .entry-content, .page-content, #content")
                html = str(main) if main else str(soup.body)
                md = markdownify(html, heading_style="ATX", strip=["img", "a"])
                cached.write_text(md, encoding="utf-8")
                time.sleep(1)  # polite crawl delay
            except Exception as exc:
                logger.warning(f"    Could not fetch {src['url']}: {exc}")
                continue

        chunks = _split_into_chunks(md, src)
        all_chunks.extend(chunks)
        logger.success(f"    → {len(chunks)} chunks from {src['title']}")

    return all_chunks


# ── Step 2: Download remote PDFs ──────────────────────────────────────────

def download_remote_pdfs() -> None:
    import requests

    for src in PDF_REMOTE:
        dest = RAW_PDF_DIR / src["filename"]
        if dest.exists():
            logger.info(f"  [exists] {src['filename']}")
            continue
        logger.info(f"  [download] {src['filename']}")
        try:
            r = requests.get(src["url"], timeout=20, headers={"User-Agent": "RVU-Bot/1.0"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            logger.success(f"    ✓ saved {src['filename']} ({len(r.content)//1024} KB)")
        except Exception as exc:
            logger.warning(f"    ✗ {src['filename']} (skipped – will use web-scraped version): {exc}")


# ── Step 3 & 4: Extract text from PDFs ───────────────────────────────────

def _extract_pdf_text(pdf_path: Path, needs_ocr: bool) -> str:
    import fitz  # PyMuPDF

    if needs_ocr:
        return _ocr_pdf(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning(f"  Cannot open {pdf_path.name} ({exc}). Checking for cached .txt …")
        # Try to read the pre-cached .txt version instead
        txt_cache = RAW_WEB_DIR / (pdf_path.stem.lower().replace(" ", "_") + ".txt")
        if txt_cache.exists():
            return txt_cache.read_text(encoding="utf-8")
        # Also try exact stem match
        for candidate in RAW_WEB_DIR.glob("*.txt"):
            if pdf_path.stem.lower().replace("-", "_") in candidate.stem.lower():
                return candidate.read_text(encoding="utf-8")
        logger.warning(f"  No cached text found for {pdf_path.name}")
        return ""

    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _ocr_pdf(pdf_path: Path) -> str:
    """Use Apple Vision via ocrmac to OCR a scanned PDF."""
    try:
        import fitz
        from ocrmac import OCR

        doc = fitz.open(str(pdf_path))
        texts = []
        for i, page in enumerate(doc):
            # Render page to PNG in memory
            pix = page.get_pixmap(dpi=200)
            png_bytes = pix.tobytes("png")
            # Write temp PNG for ocrmac
            tmp = Path(f"/tmp/rvu_ocr_page_{i}.png")
            tmp.write_bytes(png_bytes)
            result = OCR(str(tmp), language_preference=["en-IN", "en-US"]).recognize()
            page_text = "\n".join(r[0] for r in result) if result else ""
            texts.append(page_text)
            tmp.unlink(missing_ok=True)
        doc.close()
        return "\n\n".join(texts)
    except ImportError:
        logger.warning("ocrmac not installed. Falling back to PyMuPDF text extraction.")
        import fitz
        doc = fitz.open(str(pdf_path))
        texts = [p.get_text() for p in doc]
        doc.close()
        return "\n\n".join(texts)


def extract_pdfs() -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []

    all_pdf_sources = LOCAL_PDFS + [
        {"title": s["title"], "filename": s["filename"], "category": s["category"], "needs_ocr": False, "url": s["url"]}
        for s in PDF_REMOTE
    ]

    for src in all_pdf_sources:
        pdf_path = RAW_PDF_DIR / src["filename"]
        if not pdf_path.exists():
            logger.warning(f"  [missing] {src['filename']} – skipping")
            continue

        cache_txt = RAW_WEB_DIR / (pdf_path.stem + ".txt")
        if cache_txt.exists():
            text = cache_txt.read_text(encoding="utf-8")
            logger.info(f"  [cache] {src['title']}")
        else:
            logger.info(f"  [extract] {src['title']}")
            text = _extract_pdf_text(pdf_path, src.get("needs_ocr", False))
            cache_txt.write_text(text, encoding="utf-8")

        meta = {
            "title": src["title"],
            "url": src.get("url"),
            "category": src["category"],
        }
        chunks = _split_into_chunks(text, meta)
        all_chunks.extend(chunks)
        logger.success(f"    → {len(chunks)} chunks from {src['title']}")

    return all_chunks


# ── Step 5: Build fees.json ────────────────────────────────────────────────

def build_fees_table() -> None:
    """
    Parse fee info from already-scraped fee page markdown.
    Produces data/processed/fees.json with a flat list of programme records.
    """
    matches = list(RAW_WEB_DIR.glob("*fee*structure*"))
    if not matches:
        logger.warning("Fee page not scraped yet – fees.json will be empty")
        (PROCESSED_DIR / "fees.json").write_text("[]")
        return

    text = matches[0].read_text(encoding="utf-8")

    # Parse the markdown fee table — rows may have 4–6 columns due to merged cells
    records = []
    current_school = ""
    current_programme = ""

    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Sl"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [c for c in cells if c != ""]
        if not cells:
            continue

        # Full row: Sl.No | School | Programme | Specialisation | Intake | Tuition | Fees
        if len(cells) >= 6:
            sl = cells[0]
            if sl.startswith("---") or sl.lower() in ("sl. no.", "sl no", "s.no"):
                continue
            school = cells[1] if cells[1] else current_school
            if school:
                current_school = school
            programme = cells[2] if cells[2] else current_programme
            if programme:
                current_programme = programme
            specialisation = cells[3]
            intake = cells[4]
            tuition = cells[5]

        # Continuation row with fewer columns (merged cells): Programme | Spec | Intake | Tuition | Fees
        elif len(cells) == 5:
            programme = cells[0] if cells[0] else current_programme
            if programme:
                current_programme = programme
            specialisation = cells[1]
            intake = cells[2]
            tuition = cells[3]
            school = current_school

        # Even shorter: Spec | Intake | Tuition | Fees
        elif len(cells) == 4:
            programme = current_programme
            specialisation = cells[0]
            intake = cells[1]
            tuition = cells[2]
            school = current_school
        else:
            continue

        if tuition and "---" not in tuition and tuition not in ("Tuition Fees In (₹)", "Tuition"):
            records.append({
                "school": school,
                "programme": programme,
                "specialisation": specialisation,
                "intake": intake,
                "tuition_fees": tuition,
            })

    (PROCESSED_DIR / "fees.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.success(f"  fees.json → {len(records)} records")


# ── Step 6: Build calendar.json ───────────────────────────────────────────

def build_calendar() -> None:
    """Extract key academic calendar dates into a structured JSON."""
    cal_events = []

    for pattern, cal_type in [("*ug*academic*calendar*", "UG"), ("*pg*academic*calendar*", "PG")]:
        matches = list(RAW_WEB_DIR.glob(pattern))
        if not matches:
            continue
        text = matches[0].read_text(encoding="utf-8")

        # Extract markdown table rows
        rows = re.findall(
            r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
            text,
        )
        for activity, start, end in rows:
            activity = activity.strip()
            start = start.strip()
            end = end.strip()
            if not activity or activity.startswith("---") or activity.lower() == "activity":
                continue
            cal_events.append({
                "type": cal_type,
                "activity": activity,
                "start_date": start,
                "end_date": end,
            })

    (PROCESSED_DIR / "calendar.json").write_text(
        json.dumps(cal_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.success(f"  calendar.json → {len(cal_events)} events")


# ── Step 7: Embed index ───────────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> np.ndarray:
    """Try FlagEmbedding first, fall back to sentence-transformers."""
    # Attempt 1: FlagEmbedding BGEM3
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        BATCH = 32
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            vecs = model.encode(batch, batch_size=len(batch))["dense_vecs"]
            all_vecs.append(vecs)
        return np.vstack(all_vecs).astype("float32")
    except Exception as exc:
        logger.warning(f"FlagEmbedding failed ({exc}), trying sentence-transformers …")

    # Attempt 2: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.astype("float32")
    except Exception as exc:
        logger.warning(f"sentence-transformers also failed ({exc}), using TF-IDF fallback …")

    # Attempt 3: sklearn TF-IDF as last resort (no semantic search but still indexes)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=768)
        tfidf = vectorizer.fit_transform(texts).toarray()
        return tfidf.astype("float32")
    except Exception as exc:
        logger.error(f"All embedding backends failed: {exc}")
        return np.zeros((len(texts), 1), dtype="float32")


def build_index(chunks: list[dict[str, Any]]) -> None:
    texts = [c["text"] for c in chunks]
    logger.info(f"  Embedding {len(texts)} chunks …")

    embeddings = _get_embeddings(texts)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    # Save chunk metadata (without the raw text – load from chunks.jsonl)
    meta = [
        {"chunk_id": c["chunk_id"], "title": c["title"], "category": c["category"]}
        for c in chunks
    ]
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.success(f"  Dense index saved: {embeddings.shape}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Re-embed even if index already exists")
    args = parser.parse_args()

    logger.info("── Step 1: Scraping web pages ──")
    web_chunks = scrape_web()

    logger.info("── Step 2: Downloading remote PDFs ──")
    download_remote_pdfs()

    logger.info("── Step 3-4: Extracting PDF text ──")
    pdf_chunks = extract_pdfs()

    all_chunks = web_chunks + pdf_chunks
    logger.info(f"  Total chunks: {len(all_chunks)}")

    # Deduplicate by chunk_id
    seen: set[str] = set()
    unique = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)
    logger.info(f"  Unique chunks: {len(unique)}")

    # Write chunks.jsonl
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.success(f"  chunks.jsonl → {len(unique)} chunks")

    logger.info("── Step 5: Building fees.json ──")
    build_fees_table()

    logger.info("── Step 6: Building calendar.json ──")
    build_calendar()

    index_exists = (INDEX_DIR / "embeddings.npy").exists()
    if not index_exists or args.rebuild_index:
        logger.info("── Step 7: Building dense index ──")
        build_index(unique)
    else:
        logger.info("── Step 7: Dense index exists – skipping (use --rebuild-index to force) ──")

    logger.success("Ingestion complete.")


if __name__ == "__main__":
    main()
