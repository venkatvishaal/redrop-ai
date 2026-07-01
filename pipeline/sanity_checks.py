"""
sanity_checks.py

Validates internal consistency of a candidate profile - the "lie
detector" layer. Distinct from honeypot_detector.py: sanity_checks flags
soft inconsistencies that lower confidence in a profile (and feed a
penalty), while honeypot_detector looks for the dataset's specific,
deliberately-planted impossible-profile patterns that warrant hard
exclusion. There's some overlap by design - sanity_checks is the general
mechanism, honeypot_detector encodes the specific known traps.

Checks implemented:
  1. years_of_experience vs sum(career_history.duration_months) - large
     mismatches suggest either a fabricated headline number or missing
     career history.
  2. signup_date vs last_active_date - last_active before signup is
     impossible.
  3. "expert" proficiency skills with 0 duration_months - claiming
     mastery of something used for zero time.
  4. years_of_experience implausibly exceeds time elapsed since the
     candidate's EARLIEST career_history start_date (ground-truth anchor;
     deliberately not education-based, see inline comment for why).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List

from pipeline.candidate_loader import Candidate

REFERENCE_YEAR = 2026


@dataclass
class SanitySignals:
    issues: List[str] = field(default_factory=list)
    tenure_mismatch_years: float = 0.0
    has_expert_zero_duration: bool = False
    signup_after_active: bool = False
    education_timeline_implausible: bool = False
    sanity_penalty_multiplier: float = 1.0


def score_sanity(c: Candidate) -> SanitySignals:
    issues = []
    penalty = 1.0

    # 1. Tenure mismatch
    total_months = sum(ch.duration_months for ch in c.career_history)
    total_years_from_history = total_months / 12.0
    mismatch = abs(total_years_from_history - c.years_of_experience)
    if c.career_history and mismatch > 3.0:
        issues.append(
            f"years_of_experience ({c.years_of_experience}) vs sum of career_history "
            f"durations ({round(total_years_from_history, 1)}) differ by {round(mismatch, 1)} years."
        )
        penalty *= 0.7

    # 2. signup vs last_active date ordering
    signup_after_active = False
    try:
        signup = date.fromisoformat(c.redrob_signals.get("signup_date", "1970-01-01"))
        last_active = date.fromisoformat(c.redrob_signals.get("last_active_date", "1970-01-01"))
        if last_active < signup:
            signup_after_active = True
            issues.append("last_active_date precedes signup_date.")
            penalty *= 0.5
    except (ValueError, TypeError):
        pass

    # 3. Expert proficiency with zero duration
    has_expert_zero = any(
        s.proficiency == "expert" and s.duration_months == 0 for s in c.skills
    )
    if has_expert_zero:
        expert_zero_skills = [s.name for s in c.skills if s.proficiency == "expert" and s.duration_months == 0]
        issues.append(f"'Expert' proficiency claimed with 0 duration_months for: {expert_zero_skills}.")
        penalty *= 0.6

    # 4. Career-start plausibility: a candidate cannot have more years of
    # experience than the time elapsed since their EARLIEST career_history
    # start_date (the ground-truth anchor). We deliberately do NOT use
    # education end_year for this check - many real candidates work while
    # finishing a degree, take a second/later degree mid-career, or list a
    # non-chronologically-first qualification, so "years since latest
    # education" is not a reliable plausibility anchor and produced a
    # ~22% false-positive rate across the full candidate pool when tried.
    # Earliest career_history start_date is the actual ground-truth field
    # for "when did this person start working."
    education_implausible = False  # kept in the dataclass name for compatibility
    if c.career_history:
        start_years = [ch.start_date.year for ch in c.career_history if ch.start_date]
        if start_years:
            earliest_start_year = min(start_years)
            years_since_start = REFERENCE_YEAR - earliest_start_year
            if c.years_of_experience > years_since_start + 1.5:
                education_implausible = True
                issues.append(
                    f"years_of_experience ({c.years_of_experience}) exceeds plausible working "
                    f"years since earliest career_history start ({earliest_start_year})."
                )
                penalty *= 0.5

    return SanitySignals(
        issues=issues,
        tenure_mismatch_years=round(mismatch, 2),
        has_expert_zero_duration=has_expert_zero,
        signup_after_active=signup_after_active,
        education_timeline_implausible=education_implausible,
        sanity_penalty_multiplier=round(penalty, 3),
    )