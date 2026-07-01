"""Deterministic, reviewable extraction of explicit AND implicit constraints
from a JD. Goes beyond simple regex extraction by:

  1. Detecting "trap" sentences (explicit warnings about what NOT to do)
  2. Identifying soft vs hard constraints (e.g. "this range is not a requirement")
  3. Flagging contradictions and tensions (e.g. remote + onsite language)
  4. Extracting behavioral/availability requirements
  5. Building a priority-ordered constraint list for downstream scoring

The output is always marked `requires_human_review: True` because no
deterministic parser can fully capture JD nuance — but it gets close
enough to be useful as a starting point for new JDs.
"""

from __future__ import annotations
import copy
import re

CITY_NAMES = (
    "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Gurugram", "Gurgaon",
    "Bengaluru", "Bangalore", "Chennai", "Kolkata", "Ahmedabad", "Jaipur",
    "Chandigarh", "Lucknow", "Indore", "Bhopal", "Nagpur", "Visakhapatnam",
    "Vizag", "Kochi", "Cochin", "Thiruvananthapuram", "Trivandrum",
    "Coimbatore", "Bhubaneswar",
)

# Phrases that indicate a constraint is SOFT (aspirational, not hard)
SOFT_CONSTRAINT_PHRASES = (
    "not a requirement", "not a hard requirement", "not a hard constraint",
    "nice to have", "preferred", "we'd like", "ideal but not required",
    "soft", "guideline", "flexible", "case-by-case",
)

# Phrases that indicate JD is aware of traps / reading-between-the-lines
NUANCE_SIGNAL_PHRASES = (
    "let's be honest", "read between the lines", "we're going to do this differently",
    "explicit trap", "intentional trap", "we actually need", "the right answer is not",
    "this is not the kind of role",
)


def compile_job_profile(jd_text: str, base: dict) -> dict:
    """Compile a raw JD text into a structured job profile.

    Args:
        jd_text: Raw job description text.
        base: Base profile to extend/override.

    Returns:
        Enriched job profile dictionary with extracted metadata.
    """
    if not jd_text.strip():
        raise ValueError("JD text is empty")

    p = copy.deepcopy(base)
    text = " ".join(jd_text.split())

    p["source"] = {
        "type": "automatically_compiled",
        "parser": "v8-deterministic",
        "requires_human_review": True,
    }
    p["jd_text_for_embedding"] = text

    # --- Role title extraction ---
    # Use MULTILINE so $ matches at end of each line; . does not match
    # newline without DOTALL, so (.+)$ captures the full line content.
    title = re.search(
        r"(?:Job Description|Role|Position|Title)\s*:\s*(.+)$",
        jd_text, re.I | re.MULTILINE,
    )
    if title:
        p["role_title"] = title.group(1).strip()

    # --- Company extraction ---
    company = re.search(
        r"Company\s*:\s*(.+)$",
        jd_text, re.I | re.MULTILINE,
    )
    if company:
        p["company"] = company.group(1).strip()

    # --- Experience band ---
    years = re.search(r"(\d{1,2})\s*[–—\-]\s*(\d{1,2})\s*years", text, re.I)
    if years:
        p["experience_band"].update(
            min_years=int(years.group(1)),
            max_years=int(years.group(2)),
        )

    # --- Location extraction ---
    cities = [c for c in CITY_NAMES if re.search(rf"\b{re.escape(c)}\b", text, re.I)]
    if cities:
        p["location"]["preferred_cities"] = list(dict.fromkeys(cities))

    work_mode = "remote" if re.search(r"\bremote\b", text, re.I) else None
    if work_mode:
        p["location"]["work_mode"] = work_mode
    elif re.search(r"\bhybrid\b", text, re.I):
        p["location"]["work_mode"] = "hybrid"
    elif re.search(r"\boffice\b|\bonsite\b|\bin[- ]office\b", text, re.I):
        p["location"]["work_mode"] = "onsite"

    # --- Notice period ---
    notice = re.search(
        r"(?:sub-|within\s+|up to\s+)?(\d{1,3})[- ]day notice",
        text, re.I,
    )
    if notice:
        p["notice_period"]["ideal_max_days"] = int(notice.group(1))

    # --- Visa sponsorship ---
    if re.search(r"(?:no|don't|cannot)\s*(?:sponsor|provide).*visa", text, re.I):
        p["location"]["visa_sponsorship"] = False

    # --- Soft constraints detection ---
    lower = text.lower()
    p["experience_band"]["soft"] = any(p in lower for p in SOFT_CONSTRAINT_PHRASES)

    # --- Classification of statements ---
    explicit_must = []
    explicit_optional = []
    explicit_negative = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.strip()
        if re.search(r"\b(must|required|absolutely need|need to have|disqualif|will not move forward)\b", s, re.I):
            explicit_must.append(s)
        if re.search(r"\b(nice to have|preferred|we'd like|optional|would like you to have)\b", s, re.I):
            explicit_optional.append(s)
        if re.search(r"\b(do not want|must not|explicitly do not|not a fit|won't work)\b", s, re.I):
            explicit_negative.append(s)

    # --- Contradictions / tensions ---
    contradictions = []
    if "remote" in lower and re.search(r"in[- ]office|onsite|on-site", lower):
        contradictions.append("remote_and_onsite_language")
    if years and re.search(r"range(?: is|,)? not a requirement|outside the band|flexible.*years", lower):
        contradictions.append("experience_band_is_soft")
    if cities and "visa" in lower and "sponsor" in lower:
        contradictions.append("location_may_require_visa")

    # --- Trap detection (JD self-awareness) ---
    trap_detected = any(p in lower for p in NUANCE_SIGNAL_PHRASES)
    if trap_detected:
        contradictions.append("jd_contains_trap_warnings")

    p["jd_intelligence"] = {
        "explicit_must_statements": explicit_must[:20],
        "explicit_optional_statements": explicit_optional[:20],
        "explicit_negative_statements": explicit_negative[:20],
        "ambiguities_or_tensions": contradictions,
        "trap_warnings_detected": trap_detected,
        "has_soft_constraints": p["experience_band"]["soft"],
        "priority_order": [
            "hard_disqualifiers",
            "must_have_capabilities",
            "availability",
            "nice_to_have_capabilities",
        ],
        "review_required": True,  # always requires review
    }

    return p
