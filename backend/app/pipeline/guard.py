"""
guard.py
────────
Grounding guard: verifies the generated answer is supported by the
retrieved chunks before passing it to the user.

Checks:
  1. Refusal phrase    – model explicitly said it doesn't know
  2. Grounded numbers  – an answer without [S*] tags may only state numbers found in the sources
  3. Score threshold   – top retrieval score meets RAG_MIN_SCORE
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
# Param-1 often declines in its own words: "I'm sorry, but I don't have specific information about …",
# "as an AI, I don't have the ability to provide real-time information …"
_REFUSAL_RE = re.compile(
    r"\bi (?:don't|do not) have (?:any |specific |enough |sufficient |the )?(?:that )?(?:specific )?information\b"
    r"|\bas an ai\b|\bi (?:don't|do not) have the ability\b"
)
_NUMBER_RE = re.compile(r"\d+")
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def _ungrounded_numbers(answer: str, chunks: list[dict]) -> set[str]:
    """Numbers in the answer that appear in none of the retrieved passages ("2,50,000" counts as 250000).

    Numbered-list markers ("1. Go to …") are ignored.
    """
    source_numbers = {n for c in chunks for n in _NUMBER_RE.findall(c.get("text", "").replace(",", ""))}
    body = _LIST_MARKER_RE.sub("", answer).replace(",", "")
    return set(_NUMBER_RE.findall(body)) - source_numbers


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
    if any(phrase in lower for phrase in _REFUSAL_PHRASES) or _REFUSAL_RE.search(lower.replace("’", "'")):
        return {
            "passed": True,  # Model handled it gracefully
            "reason": None,
            "should_refuse": True,
        }

    # Param-1 often omits [S1] tags even when it copied the facts, so an uncited answer is
    # accepted unless it states a number (fee, date, percentage) that no source contains.
    if chunks and not _CITATION_RE.search(answer) and _ungrounded_numbers(answer, chunks):
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
    "campus": "Student Services (studentservices@rvu.edu.in)",
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
