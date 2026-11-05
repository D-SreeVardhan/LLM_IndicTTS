"""
tools.py
────────
Tool router: handles structured queries about fees, calendar dates,
placement policy, and programme catalogue directly from parsed JSON files,
bypassing the LLM for high-precision factual lookups.

Every answer is written as short, plain spoken sentences (no markdown,
tables, bullets or symbols) because it is translated sentence-by-sentence
by IndicTrans2 and then read aloud by Parler-TTS.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

_fees: list[dict] = []
_calendar: list[dict] = []

# ── Programme duration lookup ─────────────────────────────────────────────
# Durations in years by degree prefix (lower-cased match against programme name)
_PROG_DURATION: dict[str, str] = {
    "b. tech": "4 years",
    "b.tech": "4 years",
    "b. design": "4 years",
    "b.des": "4 years",
    "b.a. llb": "5 years",
    "b.b.a llb": "5 years",
    "bba llb": "5 years",
    "ba llb": "5 years",
    "b.sc": "3 years",
    "b. sc": "3 years",
    "bsc": "3 years",
    "b.a": "3 years",
    "b. a": "3 years",
    "b.com": "3 years",
    "b. com": "3 years",
    "bca": "3 years",
    "bba": "3 years",
    "b.b.a": "3 years",
    "mba": "2 years",
    "emba": "18 months",
    "m. tech": "2 years",
    "m.tech": "2 years",
    "m. design": "2 years",
    "m.des": "2 years",
    "m.sc": "2 years",
    "m. sc": "2 years",
    "m.a": "2 years",
    "m. a": "2 years",
    "l.l.m": "1 year",
    "llm": "1 year",
}


def _load():
    global _fees, _calendar
    fees_path = PROCESSED_DIR / "fees.json"
    cal_path = PROCESSED_DIR / "calendar.json"

    if fees_path.exists() and not _fees:
        _fees = json.loads(fees_path.read_text(encoding="utf-8"))
    if cal_path.exists() and not _calendar:
        _calendar = json.loads(cal_path.read_text(encoding="utf-8"))


# ── Speech-friendly text helpers ─────────────────────────────────────────

def _speakable(text: str) -> str:
    """Turn raw catalogue text into something a TTS voice can read cleanly."""
    text = text.replace("&", " and ").replace("·", ",").replace("#", "")
    text = re.sub(r"\s+[–—-]\s+", ", ", text)
    text = re.sub(r"\(\s*Hons\s*\)", "Honours", text, flags=re.I)
    text = re.sub(r"\s*\(([^)]*)\)", r", \1,", text)
    text = re.sub(r"(?<![\d.])\b\d\.\s+(?=[A-Za-z])", ", ", text)  # "1. Data 2. Policy" lists
    text = re.sub(r"\s*,[\s,]*", ", ", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _normalise_degrees(q_lower: str) -> str:
    """'b. tech' / 'b tech' / 'b.tech' → 'btech' so keyword matching is stable."""
    return re.sub(r"\b([bm])\.?\s?(tech|com|sc|des)\b", r"\1\2", q_lower)


_UNINFORMATIVE_SPEC = re.compile(r"^(regular|general|as approved\b.*)$", re.I)


def _programme_label(rec: dict) -> str:
    """Best human label for a fee/programme record (the source PDF columns are shifted for some rows)."""
    programme = rec["programme"]
    extras: list[str] = []
    for field in ("specialisation", "intake"):
        value = _speakable(rec.get(field, ""))
        if (
            not value
            or value.isdigit()
            or _UNINFORMATIVE_SPEC.match(value)
            or _compact(value) in _compact(programme)
        ):
            continue
        extras.append(value)
    label = programme
    if extras:
        first_word = _compact(programme.split()[0])
        if _compact(extras[0]).startswith(first_word):
            label = extras.pop(0)
        if extras:
            label += " in " + ", ".join(extras)
    return _speakable(label)


def _indian_amount(value: int) -> str:
    """Format 360000 as '3,60,000' (Indian digit grouping)."""
    s = str(value)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
    return f"{head},{tail}"


def _fee_phrase(raw_fee: str) -> str:
    amounts = re.findall(r"\d{1,2}(?:,\d{2})*,\d{3}", raw_fee)
    if "GOK" in raw_fee.upper():
        if amounts:
            return (
                f"{amounts[-1]} rupees for university seats, and as per Government of "
                "Karnataka norms for government quota seats"
            )
        return "as per Government of Karnataka norms"
    if amounts:
        return f"{amounts[0]} rupees"
    return _speakable(raw_fee.replace("/-", ""))


# ── Fee tool ─────────────────────────────────────────────────────────────

# "how much" alone is not a fee question ("how much attendance is needed?")
_FEE_KEYWORDS = re.compile(
    r"\b(fee|fees|tuition|costs?|price|charges?|₹|rupees?)\b", re.I
)

_PROGRAMME_MAP = {
    "btech": ["B. Tech", "B.Tech"],
    "mba": ["MBA"],
    "bca": ["BCA"],
    "bsc": ["B. Sc", "B.Sc"],
    "ba": ["B.A", "B.A."],
    "bba": ["B.B.A", "BBA"],
    "bcom": ["B. Com", "B.Com"],
    "mdes": ["M. Design", "M.Des"],
    "bdes": ["B. Design", "B.Des"],
    "law": ["LLB", "B.A. LLB", "B.B.A LLB"],
    "msc": ["M.Sc"],
}

_FEE_EXAMPLE_KEYS = ("bca", "bcom", "mba")


def _records_for(prog_key: str, records: list[dict]) -> list[dict]:
    patterns = [re.compile(rf"\b{re.escape(lbl.lower())}") for lbl in _PROGRAMME_MAP[prog_key]]
    return [rec for rec in records if any(p.search(rec["programme"].lower()) for p in patterns)]


def _fee_answer(query: str) -> str | None:
    _load()
    if not _fees:
        return None
    if not _FEE_KEYWORDS.search(query):
        return None

    q_norm = _normalise_degrees(query.lower())
    matches: list[dict] = []
    for prog_key in _PROGRAMME_MAP:
        if re.search(rf"\b{re.escape(prog_key)}\b", q_norm):
            matches.extend(rec for rec in _records_for(prog_key, _fees) if rec not in matches)

    if not matches:
        numeric = [
            int(m.replace(",", ""))
            for rec in _fees
            for m in re.findall(r"\d{1,2}(?:,\d{2})*,\d{3}", rec["tuition_fees"])
        ]
        examples = []
        for key in _FEE_EXAMPLE_KEYS:
            recs = _records_for(key, _fees)
            if recs:
                examples.append(f"{_programme_label(recs[0])} is {_fee_phrase(recs[0]['tuition_fees'])}")
        sentences = []
        if numeric:
            low, high = min(numeric), max(numeric)
            sentences.append(
                f"Tuition fees for the 2026-27 academic year range from {_indian_amount(low)} rupees "
                f"to {_indian_amount(high)} rupees, depending on the programme."
            )
        if examples:
            sentences.append("For example, " + ", ".join(examples[:-1]) + f", and {examples[-1]}.")
        sentences.append("Ask about a specific programme to hear its exact fee.")
        return " ".join(sentences)

    sentences = [
        f"The tuition fee for {_programme_label(rec)} at the {_speakable(rec['school'])} "
        f"is {_fee_phrase(rec['tuition_fees'])}."
        for rec in matches
    ]
    return " ".join(sentences)


# ── Calendar tool ─────────────────────────────────────────────────────────

_CAL_KEYWORDS = re.compile(
    r"\b(exam|examination|see|cie|calendar|schedule|date|when|induction|result|hall.?ticket|semester|break|convocation|start|begin|registration|instruction)\b",
    re.I,
)


# Exam words alone ("passing criteria for SEE") are not a date question.
_DATE_INTENT = re.compile(
    r"\b(when|date|dates|schedule|calendar|start|starts|begin|begins|timetable|time table|day)\b", re.I
)
_NON_DATE_INTENT = re.compile(
    r"\b(score|marks|minimum|criteria|pass|passing|eligib\w*|percent\w*|attendance|rule|policy)\b", re.I
)

_UPCOMING_WORDS = re.compile(r"\b(next|upcoming|coming|soon|nearest|current)\b", re.I)

_EXAM_TERMS = ["sem-end examination", "see practical", "practical & jury", "summer semester see"]

_DATE_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})")

# Raw calendar activity (lower-case substring) → spoken description
_SPOKEN_ACTIVITY: list[tuple[str, str]] = [
    ("summer semester see", "summer semester end examinations and practicals"),
    ("sem-end examination (theory)", "semester end theory examinations"),
    ("sem-end examination", "semester end examinations"),
    ("practical", "practical examinations and jury submissions"),
    ("cie finalization", "internal marks finalisation and exam eligibility"),
    ("preparation leave", "preparation leave"),
    ("hall ticket", "hall ticket distribution"),
    ("summer semester instruction", "summer semester classes"),
    ("instructions", "regular classes"),
    ("paper viewing", "answer paper viewing"),
    ("declaration of summer", "summer semester results declaration"),
    ("summer semester result", "summer semester results review"),
    ("review of results", "results review"),
    ("declaration of results", "results declaration"),
    ("internship/semester break", "internship and semester break"),
    ("semester break", "semester break"),
    ("inauguration", "inauguration and induction"),
    ("registration", "registration for the 2027-28 academic year"),
    ("convocation", "convocation"),
]


def _parse_date(raw: str) -> date | None:
    m = _DATE_RE.search(raw or "")
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").date()
    except ValueError:
        return None


def _spoken_date(raw: str) -> str:
    """'08th January 2027(Tentative)' → '8th January 2027, tentatively'."""
    text = re.sub(r"\b0(\d)", r"\1", raw.strip())
    tentative = "tentative" in text.lower()
    text = re.sub(r"\(?\s*tentative\s*\)?", "", text, flags=re.I).strip()
    return f"{text}, tentatively" if tentative else text


def _spoken_activity(activity: str) -> str:
    lower = activity.lower()
    for key, spoken in _SPOKEN_ACTIVITY:
        if key in lower:
            return spoken
    return _speakable(activity).lower()


def _event_window(ev: dict) -> tuple[date | None, date | None]:
    start, end = _parse_date(ev["start_date"]), _parse_date(ev["end_date"])
    return start or end, end or start


def _event_sentence(ev: dict, first_of_type: bool) -> str:
    start_raw = ev["start_date"] if _parse_date(ev["start_date"]) else ""
    end_raw = ev["end_date"] if _parse_date(ev["end_date"]) else ""
    activity = _spoken_activity(ev["activity"])
    verb = "are" if activity.endswith("s") else "is"
    level = "undergraduate" if ev["type"] == "UG" else "postgraduate"
    subject = f"For {level} students, the {activity}" if first_of_type else f"The {activity}"
    if start_raw and end_raw:
        return f"{subject} {verb} from {_spoken_date(start_raw)} to {_spoken_date(end_raw)}."
    return f"{subject} {verb} on {_spoken_date(start_raw or end_raw)}."


def _calendar_answer(query: str) -> str | None:
    _load()
    if not _calendar:
        return None
    if not _CAL_KEYWORDS.search(query):
        return None
    if _NON_DATE_INTENT.search(query) and not _DATE_INTENT.search(query):
        return None

    q_lower = query.lower()
    wants_upcoming = bool(_UPCOMING_WORDS.search(q_lower))

    # Try to match a specific activity
    search_terms: list[str] = []
    if "induction" in q_lower:
        search_terms = ["induction", "inauguration"]
    elif "hall ticket" in q_lower or "hall-ticket" in q_lower:
        search_terms = ["hall ticket"]
    elif "result" in q_lower:
        search_terms = ["result", "declaration"]
    elif (
        # IndicTrans2 renders Tamil "தேர்வு" (exam) as "election"
        re.search(r"\b(see|exams?|examinations?|tests?|elections?)\b", q_lower)
        or "semester end" in q_lower
    ):
        search_terms = _EXAM_TERMS
    elif "cie" in q_lower or "internal" in q_lower:
        search_terms = ["cie finalization"]
    elif "break" in q_lower:
        search_terms = ["semester break", "break"]
    elif "convocation" in q_lower:
        search_terms = ["convocation"]
    elif "start" in q_lower or "begin" in q_lower or "instruction" in q_lower:
        search_terms = ["instruction", "induction", "inauguration"]
    elif "registration" in q_lower:
        search_terms = ["registration"]

    # UG vs PG filter
    cal_type_filter = None
    if "pg" in q_lower or "postgraduate" in q_lower or "master" in q_lower:
        cal_type_filter = "PG"
    elif "ug" in q_lower or "undergraduate" in q_lower or "bachelor" in q_lower:
        cal_type_filter = "UG"

    def _matches_type(ev):
        if cal_type_filter:
            return ev["type"] == cal_type_filter
        return True

    matched = [
        ev for ev in _calendar
        if _matches_type(ev)
        and (not search_terms or any(t in ev["activity"].lower() for t in search_terms))
        and _event_window(ev)[0] is not None
    ]
    matched.sort(key=lambda ev: (ev["type"] != "UG", _event_window(ev)[0]))

    if wants_upcoming or not search_terms:
        today = date.today()
        matched = [ev for ev in matched if _event_window(ev)[1] >= today]
        # Nearest two events per UG/PG track keeps the spoken answer short.
        per_type: dict[str, int] = {}
        nearest = []
        for ev in matched:
            if per_type.get(ev["type"], 0) < 2:
                per_type[ev["type"]] = per_type.get(ev["type"], 0) + 1
                nearest.append(ev)
        matched = nearest

    if not matched:
        return None

    return " ".join(
        _event_sentence(ev, first_of_type=i == 0 or matched[i - 1]["type"] != ev["type"])
        for i, ev in enumerate(matched)
    )


# ── Placement tool ────────────────────────────────────────────────────────

_PLACEMENT_KEYWORDS = re.compile(
    r"\b(placements?|job offer|dream offer|one offer|offer rule|offer progression|ctc|lpa|placement drives?|company drives?|stdc|placed|recruiters?|campus recruit)\b",
    re.I,
)

_PLACEMENT_SUMMARY = (
    "Placements at RV University are coordinated by the Career and Alumni Relations office. "
    "To be eligible, you must have no academic backlogs and at least 80 percent attendance in pre-placement training. "
    "Under the one offer rule, once your name is on a recruiter's final list, you are considered placed. "
    "You can attend up to 5 company drives in a placement cycle. "
    "Offer progression starts once 40 percent of eligible students in your school are placed. "
    "You may then try for a better offer if your current salary is below 8 lakh per annum "
    "and the new offer is at least one and a half times your current one. "
    "A dream offer is one above your school's highest past salary, and it does not count towards the 5 drive limit."
)


def _placement_answer(query: str) -> str | None:
    if _PLACEMENT_KEYWORDS.search(query):
        return _PLACEMENT_SUMMARY
    return None


# ── Vision / Mission tool ─────────────────────────────────────────────────

_VISION_KEYWORDS = re.compile(
    r"\b(vision|mission|goal|purpose|aim|objective|about\s+rvu|about\s+rv\s+university|what\s+is\s+rvu|who\s+is\s+rvu)\b",
    re.I,
)

_VISION_SUMMARY = (
    "RV University's vision is to be a globally recognised, technology-driven institution "
    "that nurtures critical thinkers, creative problem solvers and compassionate leaders. "
    "Its mission is to pursue academic excellence through innovative teaching, research and hands-on learning. "
    "It aims to prepare students to adapt to a changing world and to build a diverse academic community "
    "through national and international collaborations. "
    "It also encourages empathy, civic responsibility and entrepreneurship, "
    "and works closely with industry so that graduates are career ready. "
    "The university has nine schools, covering liberal arts and sciences, design, business, economics and public policy, "
    "law, computer science and engineering, film and media, allied health, and continuing education."
)


def _vision_answer(query: str) -> str | None:
    if _VISION_KEYWORDS.search(query):
        return _VISION_SUMMARY
    return None


# ── Programmes / Courses tool ─────────────────────────────────────────────

_PROG_KEYWORDS = re.compile(
    r"\b(course|courses|programme|programs?|degree|degrees|offered|offer|duration|how long|years?|study|studying|btech|b\.tech|mba|bca|bsc|mtech|m\.tech|llb|llm|bba|bcom|bdes|mdes)\b",
    re.I,
)


def _get_duration(programme: str) -> str:
    """Return duration string for a given programme name."""
    prog_lower = programme.lower()
    for prefix, duration in _PROG_DURATION.items():
        if prog_lower.startswith(prefix):
            return duration
    return "varies"


def _programmes_answer(query: str) -> str | None:
    _load()
    if not _fees:
        return None
    if not _PROG_KEYWORDS.search(query):
        return None

    q_norm = _normalise_degrees(query.lower())

    # Try to filter by a specific programme type first
    _PROG_FILTER: dict[str, list[str]] = {
        "btech": ["b. tech", "b.tech"],
        "mba": ["mba"],
        "emba": ["emba"],
        "bca": ["bca"],
        "bsc": ["b. sc", "b.sc", "bsc"],
        "mtech": ["m. tech", "m.tech"],
        "llb": ["llb"],
        "llm": ["l.l.m", "llm"],
        "bba": ["b.b.a", "bba"],
        "bcom": ["b. com", "b.com"],
        "bdes": ["b. design", "b.des"],
        "mdes": ["m. design", "m.des"],
        "msc": ["m.sc", "m. sc"],
    }

    matched_records: list[dict] = []
    for filter_key, prog_labels in _PROG_FILTER.items():
        if re.search(rf"\b{filter_key}\b", q_norm):
            for rec in _fees:
                if any(lbl in rec["programme"].lower() for lbl in prog_labels):
                    matched_records.append(rec)
            break

    if not matched_records:
        return _catalogue_overview()

    # De-duplicate by (school, programme) pair
    seen: set[tuple[str, str]] = set()
    sentences: list[str] = []
    for rec in matched_records:
        key = (rec["school"], rec["programme"])
        if key in seen:
            continue
        seen.add(key)
        duration = _get_duration(rec["programme"])
        length = f"a {duration.removesuffix('s')} programme" if duration != "varies" else "a programme"
        sentences.append(
            f"{_programme_label(rec)} is {length} offered by the {_speakable(rec['school'])}."
        )
    return " ".join(sentences)


def _degree_name(programme: str) -> str:
    """'MBA – Artificial Intelligence…' → 'MBA', 'B.Sc. (Hons)' → 'B.Sc'."""
    return _speakable(re.split(r"\(|\s[–—-]\s", programme)[0]).rstrip(".")


def _catalogue_overview() -> str:
    ug: dict[str, str] = {}
    pg: dict[str, str] = {}
    for rec in _fees:
        name = _degree_name(rec["programme"])
        bucket = ug if name.upper().startswith("B") else pg
        bucket.setdefault(_compact(name), name)
    schools = {_compact(_speakable(rec["school"])) for rec in _fees}

    def _join(names: list[str]) -> str:
        return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]

    sentences = [f"RV University offers programmes across {len(schools)} schools."]
    if ug:
        sentences.append(f"Undergraduate programmes include {_join(list(ug.values()))}.")
    if pg:
        sentences.append(f"Postgraduate programmes include {_join(list(pg.values()))}.")
    sentences.append("Ask about a specific programme to hear its duration and fees.")
    return " ".join(sentences)


# ── Main router ───────────────────────────────────────────────────────────

# Topics answered by the campus guide passages. The tools' keywords are broad ("fee",
# "when", "offered"), so "hostel fee" or "clubs offered" would otherwise get a wrong tool answer.
_CAMPUS_GUIDE_TOPICS = re.compile(
    r"\b(hostels?|mess|warden|bus|buses|transport|shuttle|metro|library|canteen|cafeteria|food|"
    r"wi-?fi|internet|it helpdesk|medical cent(?:re|er)|doctor|insurance|counsell?(?:ing|ors?)|"
    r"wellbeing|mental health|ragging|grievances?|complaints?|harassment|clubs?|fest|festival|"
    r"gym|sports facilit\w*|bonafide|transcripts?|migration|certificates?|id card|identity card|"
    r"portal|lms|mentors?|internships?|exchange|study abroad|parking|dress code|contacts?|"
    r"office hours|address|located|location|reach)\b",
    re.I,
)


def tool_route(query: str) -> dict | None:
    """
    Try structured tools before falling back to RAG.

    Returns a dict with keys:
        answer   – plain spoken-style answer string
        tool     – name of the tool that answered
        chunk_id – synthetic chunk id for citation
    Or None if no tool matched.
    """
    if _CAMPUS_GUIDE_TOPICS.search(query):
        return None
    for tool_name, fn in [
        ("fee", _fee_answer),
        ("calendar", _calendar_answer),
        ("placement", _placement_answer),
        ("vision_mission", _vision_answer),
        ("programmes", _programmes_answer),
    ]:
        answer = fn(query)
        if answer:
            logger.debug(f"Tool router: matched '{tool_name}'")
            return {
                "answer": answer,
                "tool": tool_name,
                "chunk_id": f"tool_{tool_name}",
            }
    return None
