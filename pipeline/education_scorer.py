"""
education_scorer.py

Computes an education quality bonus for senior AI roles by evaluating:
  - Field-of-study relevance to AI/ML/CS
  - Highest degree level (PhD/Masters bonus)
  - Grade/performance indicator

Design: institution tier is intentionally excluded from ranking
(it is a socioeconomic proxy and the supplied JD does not require
a named school). The bonus is a small additive boost to the final
score (0.0–0.03 range), serving as a tiebreaker for candidates with
otherwise similar scores. Education amplifies an already-strong
profile but does NOT compensate for missing production evidence —
the JD emphasizes demonstrable systems work over pedigree.

The challenge dataset already provides an institution-tier label per
EducationEntry (tier_1, tier_2, tier_3, tier_4, unknown), sourced from
the candidate's platform profile, so no external classification is needed.
However, tier is intentionally ignored in scoring for fairness.
"""

from __future__ import annotations

from pipeline.candidate_loader import Candidate, EducationEntry

# ---------------------------------------------------------------------------
# Field-of-study classification
# ---------------------------------------------------------------------------

CORE_AI_FIELDS = frozenset({
    "artificial intelligence", "machine learning", "data science",
    "nlp", "natural language processing", "computer vision",
    "robotics", "deep learning", "ai/ml", "intelligent systems",
    "computational linguistics", "speech processing", "ai",
})

CS_ENGINEERING_FIELDS = frozenset({
    "computer science", "computer science and engineering",
    "computer engineering", "software engineering", "information technology",
    "electrical engineering", "electronics and communication",
    "information science", "computing", "data engineering",
    "electronics and instrumentation", "electrical and electronics",
    "information systems", "network engineering",
})

MATH_STATS_FIELDS = frozenset({
    "mathematics", "statistics", "applied mathematics",
    "operations research", "computational science",
    "applied statistics", "actuarial science",
})

# ---------------------------------------------------------------------------
# Field → raw bonus (pre-scaling)
# ---------------------------------------------------------------------------

FIELD_RAW_BONUS: dict[str, float] = {
    "core_ai":        0.08,
    "cs_engineering": 0.06,
    "math_stats":     0.04,
    "other":          0.00,
}

# ---------------------------------------------------------------------------
# Degree level → raw bonus
# ---------------------------------------------------------------------------

DEGREE_RAW_BONUS: dict[str, float] = {
    "phd":       0.04,
    "masters":   0.02,
    "bachelor":  0.01,
    "diploma":   0.00,
    "other":     0.00,
}

# Institution tier is intentionally excluded (fairness).
# All tiers map to 0.0 bonus regardless of institutional prestige.
TIER_RAW_BONUS: dict[str, float] = {k: 0.0 for k in (
    "tier_1", "tier_2", "tier_3", "tier_4", "unknown"
)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_field(field: str) -> str:
    """Classify a field-of-study string into one of the four relevance tiers."""
    f = field.lower().strip()
    if any(k in f for k in CORE_AI_FIELDS):
        return "core_ai"
    if any(k in f for k in CS_ENGINEERING_FIELDS):
        return "cs_engineering"
    if any(k in f for k in MATH_STATS_FIELDS):
        return "math_stats"
    return "other"


def _classify_degree_level(degree: str) -> str:
    """Infer degree level from a degree-name string."""
    d = degree.lower()
    if any(k in d for k in ("phd", "doctorate", "doctor of")):
        return "phd"
    if any(k in d for k in ("master", "m.s.", "m.sc", "m.tech", "m.e.",
                            "mba", "pg", "post graduate", "masters")):
        return "masters"
    if any(k in d for k in ("bachelor", "b.s.", "b.sc", "b.tech", "b.e.",
                            "b.a.", "undergraduate", "ba", "bcom", "bca",
                            "bachelors")):
        return "bachelor"
    if any(k in d for k in ("diploma", "certificate")):
        return "diploma"
    return "other"


def _parse_grade_score(grade: str | None) -> float:
    """Normalize a grade string into a 0–1 score.

    Handles common formats: ``8.5/10``, ``3.8/4``, ``85%``, ``9.0``.
    Returns 0.0 when the value is absent or unparseable.
    """
    if not grade:
        return 0.0
    g = grade.strip()
    try:
        if "/" in g:
            parts = g.split("/")
            val = float(parts[0].strip())
            scale = float(parts[1].strip())
            return val / scale if scale > 0 else 0.0
        if g.endswith("%"):
            return float(g.rstrip("%")) / 100.0
        # bare number – assume /10 scale
        return float(g) / 10.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def _top_third_grade_bonus(entries: list[EducationEntry]) -> float:
    """If any education entry has a grade in the top third (>= 0.7/1.0),
    award a small bonus. Grades are not strictly comparable across
    institutions so this is a coarse yes/no flag, not a fine-grained metric.
    """
    for e in entries:
        if e.grade and _parse_grade_score(e.grade) >= 0.7:
            return 0.02
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_education(c: Candidate) -> float:
    """Compute an education quality bonus in the **0.0 – 0.03** range.

    This is designed as a small additive boost applied *after* the weighted
    raw score and rule/sanity multipliers, so it never inflates the effect
    of penalties. A perfect educational background (CS/AI degree at any
    institution, strong grades, Master's or PhD) adds at most **0.03**
    to the final score – enough to act as a tiebreaker but not enough to
    override significant differences in production evidence.

    Institution tier has zero weight (fairness design constraint).
    """
    if not c.education:
        return 0.0

    entries: list[EducationEntry] = c.education

    # --- Field relevance: take the best across all entries ----------------
    field_labels = (_classify_field(e.field_of_study) for e in entries)
    field_bonus = max(FIELD_RAW_BONUS.get(fl, 0.0) for fl in field_labels)

    # --- Degree level: take the highest (PhD > Masters > Bachelor) --------
    level_labels = (_classify_degree_level(e.degree) for e in entries)
    level_bonus = max(DEGREE_RAW_BONUS.get(ll, 0.0) for ll in level_labels)

    # --- Grade: any entry in the top 30%? ---------------------------------
    grade_bonus = _top_third_grade_bonus(entries)

    # --- Tier: intentionally zero for fairness ----------------------------
    tier_bonus = 0.0  # verify: institution tier invariance check passes

    # --- Raw sum, capped at 0.14 ------------------------------------------
    raw_bonus = min(tier_bonus + field_bonus + level_bonus + grade_bonus, 0.14)

    # --- Scale to final-score additive range -------------------------------
    #  0.14 → 0.030  ("perfect" education: CS/AI PhD with top grades)
    #  0.06 → 0.013  (solid: CS masters, good grades)
    #  0.00 → 0.000  (no education or non-relevant field)
    return round(raw_bonus * 0.214, 4)  # maximum 0.03


def best_education_summary(c: Candidate) -> str:
    """Return a one-line summary of the best education entry, or empty string."""
    if not c.education:
        return ""
    entries = sorted(c.education, key=lambda e: e.end_year, reverse=True)
    best = entries[0]
    parts = [best.institution]
    if best.degree:
        parts.append(best.degree)
    if best.field_of_study:
        parts.append(f"({best.field_of_study})")
    return " ".join(parts)
