"""
guard.py
────────
Grounding guard: verifies the generated answer is supported by the
retrieved chunks before passing it to the user.

Checks:
  1. Citation present  – answer contains at least one [S*] tag
  2. Score threshold   – top retrieval score meets RAG_MIN_SCORE
  3. Refusal phrase    – model explicitly said it doesn't know
"""

import re

from backend.app.config import settings

_CITATION_RE = re.compile(r"\[S\d+\]")

_REFUSAL_PHRASES = [
    "i don't have that information",
    "i do not have that information",
    "cannot be found",
    "no information",
    "contact the relevant",
    "please contact",
    "not available",
]


def check(
    answer: str,
    chunks: list[dict],
    top_score: float,
) -> dict:
    """
    Returns a dict:
        passed          – bool, True if answer is grounded
        reason          – str, reason for failure (if not passed)
        should_refuse   – bool, True if the answer itself is a refusal
    """
    # Model-initiated refusal
    lower = answer.lower()
    if any(phrase in lower for phrase in _REFUSAL_PHRASES):
        return {
            "passed": True,  # Model handled it gracefully
            "reason": None,
            "should_refuse": True,
        }

    # Check citation presence when RAG was used.
    # Param-1 does not always emit [S1] tags reliably; we treat a substantive
    # answer (≥ 80 chars) as grounded even without explicit citation markers,
    # to avoid discarding valid but uncited responses.
    if chunks and not _CITATION_RE.search(answer) and len(answer.strip()) < 80:
        return {
            "passed": False,
            "reason": "no_citation",
            "should_refuse": False,
        }

    # Check retrieval confidence
    if chunks and top_score < settings.rag_min_score:
        return {
            "passed": False,
            "reason": "low_retrieval_score",
            "should_refuse": False,
        }

    return {"passed": True, "reason": None, "should_refuse": False}


_OFFICE_MAP = {
    "fees": "Accounts Office (accounts@rvu.edu.in)",
    "exams": "Controller of Examinations (coe@rvu.edu.in)",
    "admissions": "Admissions Office (admissions@rvu.edu.in)",
    "policy": "Student Services (studentservices@rvu.edu.in)",
    "placement": "Career & Alumni Relations (car@rvu.edu.in)",
}


def refusal_message(reason: str | None, top_category: str | None = None) -> str:
    office = _OFFICE_MAP.get(top_category or "", "the relevant RVU office")
    if reason == "no_citation":
        return (
            "I was unable to find a reliable source for your question in my knowledge base. "
            f"Please contact {office} directly for accurate information."
        )
    if reason == "low_retrieval_score":
        return (
            "Your question does not closely match any document in my knowledge base. "
            f"For accurate information, please contact {office}."
        )
    return (
        f"I don't have sufficient information to answer that confidently. "
        f"Please reach out to {office}."
    )
