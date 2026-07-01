"""
rule_engine.py

Applies the declarative penalty rules defined in job_profile.latest.json's
hard_disqualifiers / penalty_caps sections, using the boolean flags
produced by experience_scorer, skill_scorer, and honeypot_detector.

Kept separate from those scorers so the *consequence* of a detection
(how much to multiply final_score by) is configuration, not buried logic -
this is what the architecture doc calls "applies score caps or
multipliers based on identified patterns (e.g. pen_frequent_switcher or
pen_framework_enthusiast_no_systems)."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pipeline.experience_scorer import ExperienceSignals
from pipeline.skill_scorer import score_skills


@dataclass
class RuleEngineResult:
    multiplier: float
    applied_rules: List[str] = field(default_factory=list)
    hard_exclude: bool = False
    hard_exclude_reason: str = ""


def apply_rules(
    experience: ExperienceSignals,
    skill_result: dict,
    honeypot_is_honeypot: bool,
    job_profile: dict,
) -> RuleEngineResult:
    caps = job_profile["penalty_caps"]
    multiplier = 1.0
    applied = []

    if honeypot_is_honeypot:
        return RuleEngineResult(
            multiplier=0.0,
            applied_rules=["honeypot_detected"],
            hard_exclude=True,
            hard_exclude_reason="honeypot_detected",
        )

    if experience.consulting_only_flag:
        m = _extract_multiplier(caps["consulting_only_no_product"])
        multiplier *= m
        applied.append(f"pen_consulting_only_no_product(x{m})")

    if experience.title_chaser_flag:
        m = _extract_multiplier(caps["title_chaser"])
        multiplier *= m
        applied.append(f"pen_frequent_switcher(x{m})")

    if skill_result.get("framework_enthusiast_flag"):
        m = _extract_multiplier(caps["framework_enthusiast_no_systems"])
        multiplier *= m
        applied.append(f"pen_framework_enthusiast_no_systems(x{m})")

    if experience.stale_architect_flag:
        m = _extract_multiplier(caps["stale_coder_architect"])
        multiplier *= m
        applied.append(f"pen_stale_coder_architect(x{m})")

    if experience.cv_speech_robotics_no_nlp_flag:
        m = _extract_multiplier(caps["cv_speech_robotics_no_nlp_ir"])
        multiplier *= m
        applied.append(f"pen_cv_speech_robotics_no_nlp_ir(x{m})")

    return RuleEngineResult(
        multiplier=round(multiplier, 4),
        applied_rules=applied,
        hard_exclude=False,
    )


def _extract_multiplier(rule_text: str) -> float:
    """Parses 'multiply final_score by 0.35' -> 0.35. Kept defensive since
    this string lives in a config file a human might edit."""
    try:
        return float(rule_text.split("by")[-1].strip().rstrip("."))
    except (ValueError, IndexError):
        return 1.0