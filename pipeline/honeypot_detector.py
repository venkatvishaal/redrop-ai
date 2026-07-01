"""
honeypot_detector.py

Identifies the ~80 honeypot candidates the submission_spec.md describes:
"subtly impossible profiles (e.g., 8 years of experience at a company
founded 3 years ago; 'expert' proficiency in 10 skills with 0 years
used)." These are forced to relevance tier 0 in the hidden ground truth,
and the spec explicitly disqualifies submissions with honeypot rate >10%
in the top 100 at Stage 3.

We don't have ground-truth company founding dates in this dataset, so the
"founded N years ago" example is approximated via the tenure-mismatch and
education-timeline checks (a candidate can't have spent longer at a
company, or longer working overall, than is chronologically possible).
The "expert in many skills with 0 duration" example is checked directly.

This module is intentionally stricter/more binary than sanity_checks.py:
sanity_checks produces a soft multiplier for borderline inconsistencies;
honeypot_detector flags a candidate as an outright honeypot (hard
exclude) only when multiple independent impossible-profile signals stack,
to keep the false-positive rate low (we don't want to accidentally
exclude a real candidate with one sloppy data-entry field).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pipeline.candidate_loader import Candidate
from pipeline.sanity_checks import SanitySignals

# Threshold: how many independent "impossible profile" flags must fire
# before we call it a honeypot rather than just a sloppy/noisy profile.
HONEYPOT_FLAG_THRESHOLD = 2

# "expert in N+ skills with 0 duration" pattern from the spec's own example
EXPERT_ZERO_DURATION_COUNT_THRESHOLD = 3


@dataclass
class HoneypotSignals:
    is_honeypot: bool
    flags: List[str] = field(default_factory=list)
    flag_count: int = 0


def detect_honeypot(c: Candidate, sanity: SanitySignals) -> HoneypotSignals:
    flags = []

    # Flag A: severe tenure mismatch (years_of_experience vs actual history)
    if sanity.tenure_mismatch_years > 5.0:
        flags.append(f"severe_tenure_mismatch({sanity.tenure_mismatch_years}y)")

    # Flag B: education timeline says they couldn't have this much experience
    if sanity.education_timeline_implausible:
        flags.append("education_timeline_implausible")

    # Flag C: signup after last_active (logically impossible)
    if sanity.signup_after_active:
        flags.append("signup_after_last_active")

    # Flag D: many "expert" skills all claimed with 0 duration_months -
    # the spec's literal example ("expert proficiency in 10 skills with 0
    # years used")
    expert_zero = [s for s in c.skills if s.proficiency == "expert" and s.duration_months == 0]
    if len(expert_zero) >= EXPERT_ZERO_DURATION_COUNT_THRESHOLD:
        flags.append(f"mass_expert_zero_duration({len(expert_zero)}_skills)")
    elif len(expert_zero) >= 1 and sanity.has_expert_zero_duration:
        # single instance still counts toward the threshold, just weaker
        flags.append(f"expert_zero_duration({len(expert_zero)}_skills)")

    # Flag E: current role duration_months implausibly exceeds time since
    # the candidate's most recent education end_year (a structural version
    # of the "company founded after candidate claims to have worked there"
    # example, applied to the candidate's own timeline since we lack
    # company founding dates in this dataset).
    current_roles = [ch for ch in c.career_history if ch.is_current]
    if current_roles and c.education:
        latest_grad = max((ed.end_year for ed in c.education if ed.end_year), default=None)
        if latest_grad:
            role = current_roles[0]
            if role.start_date and role.start_date.year < latest_grad - 1:
                flags.append("current_role_predates_graduation")

    is_honeypot = len(flags) >= HONEYPOT_FLAG_THRESHOLD

    return HoneypotSignals(
        is_honeypot=is_honeypot,
        flags=flags,
        flag_count=len(flags),
    )