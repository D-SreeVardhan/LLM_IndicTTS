#!/usr/bin/env python3
"""
rag_eval.py
───────────
Evaluate the full retrieval-augmented generation pipeline:

  - Retrieval Recall@k
  - Key-fact accuracy (exact match for numbers, dates)
  - Citation precision
  - Refusal accuracy on out-of-scope questions
  - Latency per stage

Requires:
  - data/processed/chunks.jsonl to exist (run scripts/ingest.py first)
  - eval/datasets/gold_qa.json         (hand-written gold set)

Run:
    python eval/rag_eval.py [--ablation retrieve_only|no_tools|pivot|direct]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from backend.app.pipeline import tools
from backend.app.pipeline.orchestrator import run as run_pipeline
from backend.app.pipeline.retrieve import retrieve
from backend.app.schemas import AskRequest

RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GOLD_PATH   = ROOT / "eval" / "datasets" / "gold_qa.json"

# ── Key-fact extractor ────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")
_DATE_RE   = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{4})\b",
    re.I,
)


def extract_key_facts(text: str) -> set[str]:
    """Extract numbers and dates from text for exact-match scoring."""
    facts: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group().replace(",", ""))
    for m in _DATE_RE.finditer(text):
        facts.add(m.group().lower().strip())
    return facts


def fact_overlap(predicted: str, gold_facts: list[str]) -> float:
    """Fraction of gold key facts found in the predicted answer."""
    if not gold_facts:
        return 1.0
    pred_facts = extract_key_facts(predicted)
    hit = sum(1 for f in gold_facts if f in pred_facts or f in predicted.lower())
    return hit / len(gold_facts)


# ── Build gold dataset if missing ────────────────────────────────────────

GOLD_TEMPLATE = [
    # Fees
    {"id": "f01", "lang": "en", "in_scope": True,
     "question": "What is the annual tuition fee for B.Tech (AIML) at RVU under the JEE-Mains seat category for AY 2026-27?",
     "gold_answer": "₹5,00,000 per annum",
     "key_facts": ["5,00,000", "500000"],
     "source_category": "fees"},
    {"id": "f02", "lang": "en", "in_scope": True,
     "question": "How much is the tuition fee for MBA at RVU?",
     "gold_answer": "MBA tuition fees are ₹5,50,000 (Regular) or ₹6,25,000 (AI & Data Science / Global Financial Markets).",
     "key_facts": ["5,50,000", "6,25,000"],
     "source_category": "fees"},
    {"id": "f03", "lang": "en", "in_scope": True,
     "question": "What is the BCA fee at RVU?",
     "gold_answer": "₹2,50,000 tuition fee plus ₹21,000 skill/exam fees per year.",
     "key_facts": ["2,50,000", "21,000"],
     "source_category": "fees"},
    # Exams
    {"id": "e01", "lang": "en", "in_scope": True,
     "question": "When are the UG Semester End Examinations for the odd semester AY 2026-27?",
     "gold_answer": "7th December 2026 to 31st December 2026",
     "key_facts": ["7th december 2026", "31st december 2026"],
     "source_category": "exams"},
    {"id": "e02", "lang": "en", "in_scope": True,
     "question": "What is the minimum CIE score required to appear for SEE at RVU?",
     "gold_answer": "Students must score at least 40% in CIE to appear for SEE.",
     "key_facts": ["40"],
     "source_category": "exams"},
    {"id": "e03", "lang": "en", "in_scope": True,
     "question": "When are hall tickets issued for the even semester 2027?",
     "gold_answer": "3rd May 2027 to 4th May 2027",
     "key_facts": ["3rd may 2027", "4th may 2027"],
     "source_category": "exams"},
    {"id": "e04", "lang": "en", "in_scope": True,
     "question": "What attendance percentage is mandatory at RVU?",
     "gold_answer": "Students must maintain 80% attendance in every course.",
     "key_facts": ["80"],
     "source_category": "exams"},
    # Admissions
    {"id": "a01", "lang": "en", "in_scope": True,
     "question": "If a student cancels admission between 16 August 2026 and 30 August 2026, what refund do they get?",
     "gold_answer": "90% refund of first year full fees.",
     "key_facts": ["90"],
     "source_category": "admissions"},
    {"id": "a02", "lang": "en", "in_scope": True,
     "question": "When does the UG induction start for AY 2026-27?",
     "gold_answer": "27th July 2026",
     "key_facts": ["27th july 2026"],
     "source_category": "exams"},
    # Placement
    {"id": "p01", "lang": "en", "in_scope": True,
     "question": "How many company drives can a student attend at RVU placements?",
     "gold_answer": "A student may attend a maximum of 5 notified placement drives per cycle.",
     "key_facts": ["5"],
     "source_category": "policy"},
    {"id": "p02", "lang": "en", "in_scope": True,
     "question": "What CTC threshold applies for the 1.5x offer progression rule?",
     "gold_answer": "The student's existing offer must be below ₹8 LPA.",
     "key_facts": ["8"],
     "source_category": "policy"},
    # Out-of-scope
    {"id": "oos01", "lang": "en", "in_scope": False,
     "question": "What is the best restaurant near RVU?",
     "gold_answer": None, "key_facts": [], "source_category": None},
    {"id": "oos02", "lang": "en", "in_scope": False,
     "question": "Who won the IPL in 2025?",
     "gold_answer": None, "key_facts": [], "source_category": None},
    {"id": "oos03", "lang": "en", "in_scope": False,
     "question": "Give me the personal phone number of the Vice Chancellor.",
     "gold_answer": None, "key_facts": [], "source_category": None},
]


def ensure_gold() -> list[dict]:
    if GOLD_PATH.exists():
        return json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    logger.info("Creating gold_qa.json from template …")
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(json.dumps(GOLD_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Gold dataset saved to {GOLD_PATH} ({len(GOLD_TEMPLATE)} questions)")
    return GOLD_TEMPLATE


# ── Evaluation helpers ────────────────────────────────────────────────────

# ablation → (retrieval mode, tool router enabled)
ABLATIONS = {
    "hybrid":     ("hybrid", True),
    "bm25_only":  ("bm25", True),
    "dense_only": ("dense", True),
    "no_tools":   ("hybrid", False),
    "pivot":      ("hybrid", True),
    "direct":     ("hybrid", True),
}


def run_retrieval_only(question: str, top_k: int, mode: str) -> list[dict]:
    return retrieve(question, top_k=top_k, mode=mode)


def run_full_pipeline(question: str, lang: str, route: str, mode: str, use_tools: bool) -> dict:
    req = AskRequest(text=question, lang=lang, route=route)
    resp = run_pipeline(req, retrieval_mode=mode, use_tools=use_tools)
    return {
        "answer": resp.answer,
        "refused": resp.refused,
        "citations": [c.model_dump() for c in resp.citations],
        "trace": [t.model_dump() for t in resp.trace],
        "total_latency_ms": resp.total_latency_ms,
    }


def _answered_by(trace: list[dict]) -> str:
    """'tool:<name>' if the tool router answered, else 'llm'."""
    for stage in trace:
        detail = stage.get("detail") or ""
        if stage["stage"] == "tool_router" and detail.startswith("matched:"):
            return "tool:" + detail.split(":", 1)[1]
    return "llm"


def retrieval_only(gold: list[dict], top_k: int) -> None:
    """Recall@k of each retrieval mode, and key-fact accuracy of the tool router alone."""
    in_scope = [item for item in gold if item.get("source_category")]
    summary: dict = {"top_k": top_k, "n_in_scope": len(in_scope), "recall_at_k_pct": {}}
    for mode in ("bm25", "dense", "hybrid"):
        hits = sum(
            any(c.get("category") == item["source_category"] for c in retrieve(item["question"], top_k=top_k, mode=mode))
            for item in in_scope
        )
        summary["recall_at_k_pct"][mode] = round(100 * hits / len(in_scope), 1)

    rows = []
    for item in gold:
        result = tools.tool_route(item["question"])
        rows.append({
            "id": item["id"],
            "in_scope": item["in_scope"],
            "tool": result["tool"] if result else "",
            "fact_score": fact_overlap(result["answer"], item["key_facts"]) if result else "",
            "answer": result["answer"] if result else "",
        })
    answered = [r for r in rows if r["tool"] and r["in_scope"]]
    summary["tool_answered"] = len(answered)
    summary["tool_key_fact_accuracy_pct"] = round(100 * sum(r["fact_score"] for r in answered) / len(answered), 1)
    summary["tool_false_matches_out_of_scope"] = sum(1 for r in rows if r["tool"] and not r["in_scope"])

    with (RESULTS_DIR / "tool_router_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "summary_retrieval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.success(f"Retrieval-only summary: {summary}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=list(ABLATIONS),
        default="hybrid",
        help="Which retrieval / routing mode to evaluate",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval recall for every mode and the tool router, without the LLM")
    args = parser.parse_args()

    if args.retrieval_only:
        retrieval_only(ensure_gold(), args.top_k)
        return

    mode, use_tools = ABLATIONS[args.ablation]
    gold = ensure_gold()
    logger.info(f"Loaded {len(gold)} gold items (ablation={args.ablation})")

    rows = []
    recall_total, recall_hit = 0, 0
    fact_scores = []
    cite_precision_scores = []
    refusal_correct = 0
    refusal_total = 0
    latencies = []

    for item in gold:
        q = item["question"]
        in_scope = item["in_scope"]

        # Retrieval
        chunks = run_retrieval_only(q, args.top_k, mode)
        source_cat = item.get("source_category")
        if source_cat and chunks:
            hit = any(c.get("category") == source_cat for c in chunks)
            recall_hit += int(hit)
        recall_total += 1 if source_cat else 0

        # Full pipeline
        lang = item.get("lang", "en")
        route = "pivot" if args.ablation == "pivot" else "direct"
        result = run_full_pipeline(q, lang, route, mode, use_tools)
        latencies.append(result["total_latency_ms"])

        # Refusal accuracy
        if not in_scope:
            refusal_total += 1
            if result["refused"]:
                refusal_correct += 1

        # Fact accuracy
        if in_scope and item.get("key_facts"):
            score = fact_overlap(result["answer"], item["key_facts"])
            fact_scores.append(score)

        # Citation precision: share of cited passages from the gold source category.
        # Tool answers cite their structured record, so only LLM answers are scored.
        answered_by = _answered_by(result["trace"])
        if in_scope and answered_by == "llm" and result["citations"] and source_cat:
            category_of = {c["chunk_id"]: c.get("category") for c in chunks}
            cited = [c["chunk_id"] for c in result["citations"]]
            correct = sum(1 for cid in cited if category_of.get(cid) == source_cat)
            cite_precision_scores.append(correct / len(cited))

        rows.append({
            "id": item["id"],
            "in_scope": in_scope,
            "question": q,
            "gold_answer": item.get("gold_answer", ""),
            "predicted": result["answer"],
            "answered_by": answered_by,
            "refused": result["refused"],
            "fact_score": fact_overlap(result["answer"], item.get("key_facts", [])),
            "latency_ms": result["total_latency_ms"],
        })

    # Aggregate metrics
    recall_k = recall_hit / recall_total if recall_total else float("nan")
    avg_fact = sum(fact_scores) / len(fact_scores) if fact_scores else float("nan")
    avg_cite_prec = sum(cite_precision_scores) / len(cite_precision_scores) if cite_precision_scores else float("nan")
    refusal_acc = refusal_correct / refusal_total if refusal_total else float("nan")
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")

    logger.info("\n── RAG Evaluation Summary ──")
    logger.info(f"  Ablation mode:        {args.ablation}")
    logger.info(f"  Retrieval Recall@{args.top_k}:  {recall_k*100:.1f}%")
    logger.info(f"  Key-fact accuracy:    {avg_fact*100:.1f}%")
    logger.info(f"  Citation precision:   {avg_cite_prec*100:.1f}%")
    logger.info(f"  Refusal accuracy:     {refusal_acc*100:.1f}%  ({refusal_correct}/{refusal_total})")
    logger.info(f"  Avg latency:          {avg_latency:.0f} ms")

    # Save CSV
    out_csv = RESULTS_DIR / f"rag_results_{args.ablation}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save summary JSON
    summary = {
        "ablation": args.ablation,
        "top_k": args.top_k,
        "n_questions": len(gold),
        "recall_at_k": round(recall_k * 100, 1),
        "key_fact_accuracy_pct": round(avg_fact * 100, 1),
        "citation_precision_pct": round(avg_cite_prec * 100, 1),
        "refusal_accuracy_pct": round(refusal_acc * 100, 1),
        "avg_latency_ms": round(avg_latency, 0),
    }
    (RESULTS_DIR / f"summary_{args.ablation}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.success(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
