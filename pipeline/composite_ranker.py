"""
composite_ranker.py

The central brain. Combines semantic_fit, skill core-fit, experience fit,
behavioral fit, and location/logistics fit into a single final_score,
applies rule_engine multipliers, derives core_fit_status, and writes the
per-candidate reasoning string the submission spec scores at Stage 4.

Reasoning generation deliberately pulls concrete facts (years, current
title, named matched/missing capabilities, specific signal values) rather
than templated praise - the spec's Stage 4 checklist explicitly penalizes
"templated reasoning that just inserts the candidate's name" and rewards
"specific facts ... honest concerns ... rank consistency."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.candidate_loader import Candidate
from pipeline.experience_scorer import ExperienceSignals
from pipeline.evidence_scorer import EvidenceSignals
from pipeline.signal_scorer import BehavioralSignals
from pipeline.location_scorer import LocationSignals
from pipeline.sanity_checks import SanitySignals
from pipeline.honeypot_detector import HoneypotSignals
from pipeline.rule_engine import RuleEngineResult
from pipeline.education_scorer import score_education, best_education_summary


@dataclass
class RankedCandidate:
    candidate: Candidate
    final_score: float
    core_fit_status: str  # strong_pass | conditional_pass | weak_fit
    reasoning: str
    semantic_score: float
    skill_score: float
    experience_score: float
    behavioral_score: float
    logistics_score: float
    rule_multiplier: float
    sanity_multiplier: float
    hard_excluded: bool
    education_bonus: float = 0.0
    hard_exclude_reason: str = ""
    debug: dict = field(default_factory=dict)


def _experience_subscore(experience: ExperienceSignals, evidence: EvidenceSignals, job_profile: dict) -> float:
    band = job_profile["experience_band"]
    years = experience.total_years
    if band["min_years"] <= years <= band["max_years"]:
        band_score = 1.0
    elif years < band["min_years"]:
        # Steeper-than-proportional penalty below the minimum: this is a
        # "Senior" / founding-team role, so meaningfully under-experienced
        # candidates should be penalized harder than the gap alone would
        # suggest. A multiplier >1 on the shortfall fraction does this
        # (e.g. missing 40% of the minimum costs more than 40% of the
        # score). Previously a shallow linear ramp let a 3.0y candidate
        # score 0.667 on this dimension - only ~33% worse than a perfect
        # match - which let several 3-4y candidates reach the top 100 of a
        # role the JD describes as "Senior."
        shortfall_fraction = (band["min_years"] - years) / band["min_years"]
        band_score = max(0.0, 1.0 - 2.2 * shortfall_fraction)
    else:
        # Mild falloff above the max - excess seniority is a much smaller
        # concern than insufficient seniority for this role.
        excess = years - band["max_years"]
        band_score = max(0.0, 1.0 - excess / 10.0)

    evidence_component = (
        0.5 * evidence.retrieval_evidence_score
        + 0.3 * evidence.eval_framework_evidence_score
        + 0.2 * evidence.production_ml_evidence_score
    )

    return round(0.4 * band_score + 0.3 * experience.seniority_trajectory_score + 0.3 * evidence_component, 4)


def _core_fit_status(final_score: float, must_have_fraction: float) -> str:
    if final_score >= 0.65 and must_have_fraction >= 0.55:
        return "strong_pass"
    if final_score >= 0.40:
        return "conditional_pass"
    return "weak_fit"


def _build_reasoning(
    c: Candidate,
    skill_result: dict,
    experience: ExperienceSignals,
    behavioral: BehavioralSignals,
    location: LocationSignals,
    sanity: SanitySignals,
    rule_result: RuleEngineResult,
    job_profile: dict,
) -> str:
    parts: List[str] = []

    parts.append(f"{c.current_title} at {c.current_company} ({c.years_of_experience}y exp, {c.location}).")

    must_have = job_profile["must_have_capabilities"]
    matched = [d.capability_id for d in skill_result["must_have_details"] if d.matched]
    missing = [d.capability_id for d in skill_result["must_have_details"] if not d.matched]
    if matched:
        parts.append(f"Evidence for: {', '.join(matched)}.")
    if missing:
        parts.append(f"Weak/no evidence for: {', '.join(missing)}.")

    if rule_result.applied_rules:
        parts.append(f"Concerns: {', '.join(rule_result.applied_rules)}.")

    if behavioral.is_stale:
        parts.append(f"Inactive {(_days_since(c.redrob_signals.get('last_active_date')))}d - availability risk.")
    else:
        parts.append(f"Response rate {c.redrob_signals.get('recruiter_response_rate', 0)}, recently active.")

    if location.is_preferred_city:
        parts.append("Based in a preferred city.")
    elif not location.is_india:
        parts.append(f"Located outside India ({c.country}) - no visa sponsorship per JD.")

    if sanity.issues:
        parts.append("Profile flags: " + "; ".join(sanity.issues[:1]) + ".")

    return " ".join(parts)


def _days_since(date_str: Optional[str]) -> int:
    if not date_str:
        return 999
    from datetime import date, datetime
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (date.today() - d).days)
    except (ValueError, TypeError):
        return 999


def _case_by_case_exception_met(
    semantic_score: float, skill_score: float, experience_score: float, behavioral_score: float
) -> bool:
    """The JD's exact wording for outside-India candidates is 'case-by-case,'
    not a hard exclude - so this pipeline does not implement a blanket
    `if country != 'India': exclude`. But two independent audits both
    converged on the same finding: a blended-average penalty (location at
    13% weight) wasn't steep enough on its own, since a non-India
    candidate could still reach the top 20 by being merely strong on
    everything else while average elsewhere. "Case-by-case" should mean a
    genuinely exceptional candidate, not "scored well on a weighted
    average that happens to include a location penalty."

    This gate requires EVERY one of the four non-logistics dimensions to
    individually clear a high bar - not an average - before a non-India
    candidate is allowed to survive the location penalty's effect. A
    candidate who is excellent on three dimensions and merely good on the
    fourth does not qualify; this is deliberately closer to a hard
    exclude in practice while still being textually faithful to "case-by-
    case" rather than a blanket country check.
    """
    THRESHOLD = 0.70
    return (
        semantic_score >= THRESHOLD
        and skill_score >= THRESHOLD
        and experience_score >= THRESHOLD
        and behavioral_score >= THRESHOLD
    )


def build_ranked_candidate(
    c: Candidate,
    semantic_score: float,
    skill_result: dict,
    experience: ExperienceSignals,
    evidence: EvidenceSignals,
    behavioral: BehavioralSignals,
    location: LocationSignals,
    sanity: SanitySignals,
    honeypot: HoneypotSignals,
    rule_result: RuleEngineResult,
    job_profile: dict,
) -> RankedCandidate:
    weights = job_profile["weights"]

    experience_score = _experience_subscore(experience, evidence, job_profile)
    education_bonus = score_education(c)

    raw_score = (
        weights["semantic_fit"] * semantic_score
        + weights["core_fit"] * skill_result["skill_score"]
        + weights["experience_fit"] * experience_score
        + weights["behavioral_fit"] * behavioral.behavioral_score
        + weights["location_logistics_fit"] * location.logistics_score
    )

    final_score = raw_score * rule_result.multiplier * sanity.sanity_penalty_multiplier
    final_score += education_bonus
    final_score = min(max(final_score, 0.0), 1.0)

    non_india_exception_failed = False
    if not location.is_india:
        if not _case_by_case_exception_met(
            semantic_score, skill_result["skill_score"], experience_score, behavioral.behavioral_score
        ):
            non_india_exception_failed = True
            final_score = 0.0

    if rule_result.hard_exclude or honeypot.is_honeypot:
        final_score = 0.0

    core_fit_status = _core_fit_status(final_score, skill_result["must_have_fraction"])

    reasoning = _build_reasoning(
        c, skill_result, experience, behavioral, location, sanity, rule_result, job_profile
    )

    # Append education summary to reasoning when it adds signal
    edu_summary = best_education_summary(c)
    if edu_summary and education_bonus > 0.0:
        reasoning += f" Education: {edu_summary}."

    return RankedCandidate(
        candidate=c,
        final_score=round(final_score, 6),
        core_fit_status=core_fit_status,
        reasoning=reasoning,
        semantic_score=round(semantic_score, 4),
        skill_score=skill_result["skill_score"],
        experience_score=experience_score,
        behavioral_score=behavioral.behavioral_score,
        logistics_score=location.logistics_score,
        education_bonus=education_bonus,
        rule_multiplier=rule_result.multiplier,
        sanity_multiplier=sanity.sanity_penalty_multiplier,
        hard_excluded=bool(rule_result.hard_exclude or honeypot.is_honeypot or non_india_exception_failed),
        hard_exclude_reason=(
            rule_result.hard_exclude_reason
            or ("honeypot_detected" if honeypot.is_honeypot else "")
            or ("non_india_case_by_case_exception_not_met" if non_india_exception_failed else "")
        ),
        debug={
            "must_have_fraction": skill_result["must_have_fraction"],
            "nice_to_have_fraction": skill_result["nice_to_have_fraction"],
            "framework_enthusiast_flag": skill_result["framework_enthusiast_flag"],
            "title_chaser_flag": experience.title_chaser_flag,
            "consulting_only_flag": experience.consulting_only_flag,
            "stale_architect_flag": experience.stale_architect_flag,
            "honeypot_flags": honeypot.flags,
            "sanity_issues": sanity.issues,
        },
    )
