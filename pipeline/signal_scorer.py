"""
signal_scorer.py

Converts the 23 redrob_signals fields into a single behavioral/availability
score in [0,1]. Per the JD's explicit instruction: "a perfect-on-paper
candidate who hasn't logged in for 6 months and has a 5% recruiter
response rate is, for hiring purposes, not actually available. Down-weight
them appropriately." This module is that down-weighting mechanism.

Log-scaling is used for count-like signals (profile_views, search
appearances, connections, endorsements) because the marginal value of the
100th profile view is much less than the 1st - we want "clearly visible
and engaged" to separate from "totally invisible," but not let a candidate
with 5000 profile views dominate one with 200 just because of raw scale.

Recency (last_active_date) is treated as the single highest-leverage
signal here, per the JD's own framing - an otherwise-great candidate who's
gone quiet for 6 months is explicitly called out as a problem case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict

REFERENCE_DATE = date.today()
DATE_FMT = "%Y-%m-%d"


def _parse(s: str) -> date:
    return datetime.strptime(s, DATE_FMT).date()


def _log_scale(value: float, soft_cap: float) -> float:
    """Maps value in [0, inf) to roughly [0,1], saturating gently past
    soft_cap. log1p keeps small values meaningfully differentiated."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(soft_cap), 1.0)


def _recency_score(last_active: date, reference: date) -> float:
    days = (reference - last_active).days
    if days < 0:
        days = 0
    # Full credit within 14 days, decaying to ~0 by 180 days (6 months),
    # matching the JD's explicit "hasn't logged in for 6 months" example.
    if days <= 14:
        return 1.0
    if days >= 180:
        return 0.0
    # smooth decay between 14 and 180 days
    return round(1.0 - ((days - 14) / (180 - 14)), 3)


@dataclass
class BehavioralSignals:
    behavioral_score: float
    recency_score: float
    response_score: float
    engagement_score: float
    reliability_score: float
    is_stale: bool  # inactive 180+ days
    notice_period_days: int
    willing_to_relocate: bool
    preferred_work_mode: str
    expected_salary_min: float
    expected_salary_max: float
    github_activity_score: float


def score_signals(redrob_signals: Dict[str, Any], reference_date: date = REFERENCE_DATE) -> BehavioralSignals:
    rs = redrob_signals

    last_active = _parse(rs.get("last_active_date", reference_date.isoformat()))
    recency = _recency_score(last_active, reference_date)
    is_stale = (reference_date - last_active).days >= 180

    response_rate = float(rs.get("recruiter_response_rate", 0) or 0)
    avg_resp_hours = float(rs.get("avg_response_time_hours", 999) or 999)
    # fast responders get a bonus; scale so 24h = great, 200h+ = poor
    speed_factor = max(0.0, 1.0 - min(avg_resp_hours / 200.0, 1.0))
    response_score = round(0.7 * response_rate + 0.3 * speed_factor, 3)

    profile_views = _log_scale(float(rs.get("profile_views_received_30d", 0) or 0), soft_cap=200)
    search_appearance = _log_scale(float(rs.get("search_appearance_30d", 0) or 0), soft_cap=1000)
    saved_by_recruiters = _log_scale(float(rs.get("saved_by_recruiters_30d", 0) or 0), soft_cap=50)
    endorsements = _log_scale(float(rs.get("endorsements_received", 0) or 0), soft_cap=200)
    connections = _log_scale(float(rs.get("connection_count", 0) or 0), soft_cap=500)
    engagement_score = round(
        0.30 * profile_views + 0.30 * search_appearance + 0.25 * saved_by_recruiters
        + 0.10 * endorsements + 0.05 * connections,
        3,
    )

    interview_completion = float(rs.get("interview_completion_rate", 0) or 0)
    offer_acceptance = float(rs.get("offer_acceptance_rate", -1) or -1)
    offer_acceptance_norm = max(offer_acceptance, 0.0)  # -1 (no history) treated neutrally as 0, not penalized
    verified_bonus = (
        0.4 * bool(rs.get("verified_email", False))
        + 0.3 * bool(rs.get("verified_phone", False))
        + 0.3 * bool(rs.get("linkedin_connected", False))
    )
    profile_complete = float(rs.get("profile_completeness_score", 0) or 0)
    # Dataset versions use either a 0-1 fraction or a 0-100 percentage.
    if profile_complete > 1:
        profile_complete /= 100.0
    profile_complete = min(max(profile_complete, 0.0), 1.0)
    assessment_values = list((rs.get("skill_assessment_scores") or {}).values())
    assessment = (sum(float(v) for v in assessment_values) / len(assessment_values)
                  if assessment_values else 0.5)
    if assessment > 1:
        assessment /= 100.0
    assessment = min(max(assessment, 0.0), 1.0)
    github_raw = float(rs.get("github_activity_score", -1) or -1)
    github = 0.5 if github_raw < 0 else min(github_raw / 100.0, 1.0)
    reliability_score = round(0.35 * interview_completion + 0.20 * offer_acceptance_norm
                              + 0.15 * verified_bonus + 0.15 * profile_complete
                              + 0.10 * assessment + 0.05 * github, 3)

    applications = _log_scale(float(rs.get("applications_submitted_30d", 0) or 0), 20)
    market_intent = max(float(bool(rs.get("open_to_work_flag", False))), applications)

    behavioral_score = round(
        0.35 * recency + 0.25 * response_score + 0.15 * engagement_score
        + 0.15 * reliability_score + 0.10 * market_intent,
        4,
    )

    return BehavioralSignals(
        behavioral_score=behavioral_score,
        recency_score=recency,
        response_score=response_score,
        engagement_score=engagement_score,
        reliability_score=reliability_score,
        is_stale=is_stale,
        notice_period_days=int(rs.get("notice_period_days", 0) or 0),
        willing_to_relocate=bool(rs.get("willing_to_relocate", False)),
        preferred_work_mode=rs.get("preferred_work_mode", "unknown"),
        expected_salary_min=float((rs.get("expected_salary_range_inr_lpa") or {}).get("min", 0) or 0),
        expected_salary_max=float((rs.get("expected_salary_range_inr_lpa") or {}).get("max", 0) or 0),
        github_activity_score=float(rs.get("github_activity_score", -1) or -1),
    )
