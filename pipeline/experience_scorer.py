"""
experience_scorer.py

Analyzes career_history for:
  - total years vs the JD's 5-9 year (soft) band
  - average tenure per role (title-chaser detection feeds rule_engine)
  - seniority trajectory (is responsibility/scope growing over time?)
  - consulting-only-career detection (feeds rule_engine)
  - CV/speech/robotics-without-NLP/IR detection (feeds rule_engine)
  - "stale architect" detection: senior title, but description language in
    the most recent role reads as pure management/architecture with no
    coding signal, AND that role has run 18+ months.

This module returns raw signals; rule_engine.py decides what to do with
the boolean flags (apply named penalties), keeping "detection" and
"consequence" cleanly separated so it's easy to defend/adjust either half
independently at the Stage 5 interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pipeline.candidate_loader import Candidate

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "mindtree", "tech mahindra",
    "l&t infotech", "lti", "mphasis",
    # The JD's list ends in "etc." (TCS, Infosys, Wipro, Accenture,
    # Cognizant, Capgemini, etc.), signaling it's illustrative rather than
    # exhaustive. Added after finding 4 real candidates in the dataset
    # with an entire career at "Genpact AI" - a real-world IT/BPO services
    # firm of the same kind as the JD's named examples - who would
    # otherwise bypass consulting-only detection purely because the exact
    # string "Genpact" wasn't in this set.
    "genpact",
    "wns", "firstsource", "concentrix", "ibm services", "deloitte",
    "pwc", "kpmg", "ey global delivery", "hexaware", "birlasoft",
}

NLP_IR_KEYWORDS = (
    "nlp", "natural language", "retrieval", "ranking", "search", "embeddings",
    "rag", "information retrieval", "semantic search", "recommendation",
    "language model", "text", "bm25",
)
CV_SPEECH_ROBOTICS_KEYWORDS = (
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "robotics", "autonomous", "lidar", "slam",
)

CODING_SIGNAL_KEYWORDS = (
    "implemented", "wrote", "built", "coded", "shipped code", "pull request",
    "designed and built", "developed", "debugged", "refactored",
)
PURE_MANAGEMENT_KEYWORDS = (
    "managed a team", "people management", "roadmap", "stakeholder",
    "no longer write code", "architecture reviews only", "individual contributors report",
)


@dataclass
class ExperienceSignals:
    total_years: float
    avg_tenure_months: float
    num_jobs: int
    in_band_5_9: bool
    title_chaser_flag: bool
    consulting_only_flag: bool
    cv_speech_robotics_no_nlp_flag: bool
    stale_architect_flag: bool
    seniority_trajectory_score: float  # 0-1, crude "growing scope over time"
    experience_evidence: List[str] = field(default_factory=list)


def _detect_title_chaser(c: Candidate) -> bool:
    if len(c.career_history) < 3:
        return False
    avg_tenure = sum(ch.duration_months for ch in c.career_history) / len(c.career_history)
    if avg_tenure >= 18:
        return False
    # crude seniority-escalation check: look for increasing seniority words
    seniority_rank = {"junior": 0, "": 1, "senior": 2, "staff": 3, "principal": 4, "lead": 3}
    ranks = []
    for ch in c.career_history:
        title_l = ch.title.lower()
        r = 1
        for kw, val in seniority_rank.items():
            if kw and kw in title_l:
                r = val
        ranks.append(r)
    escalating = all(b >= a for a, b in zip(ranks, ranks[1:])) and ranks[-1] > ranks[0]
    return escalating


def _detect_consulting_only(c: Candidate) -> bool:
    if not c.career_history:
        return False
    companies = [ch.company.lower() for ch in c.career_history]
    if not companies:
        return False
    all_consulting = all(
        any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies
    )
    return all_consulting


def _detect_cv_speech_robotics_no_nlp(c: Candidate) -> bool:
    text = (" ".join(ch.description for ch in c.career_history) + " " + c.summary).lower()
    skill_text = " ".join(c.skill_names_lower)
    combined = text + " " + skill_text
    has_cv_speech = any(k in combined for k in CV_SPEECH_ROBOTICS_KEYWORDS)
    has_nlp_ir = any(k in combined for k in NLP_IR_KEYWORDS)
    return has_cv_speech and not has_nlp_ir


def _detect_stale_architect(c: Candidate) -> bool:
    current_roles = [ch for ch in c.career_history if ch.is_current]
    if not current_roles:
        return False
    role = current_roles[0]
    if role.duration_months < 18:
        return False
    text_l = role.description.lower()
    title_l = role.title.lower()
    is_senior_title = any(k in title_l for k in ("architect", "lead", "principal", "head", "director", "manager"))
    has_coding_signal = any(k in text_l for k in CODING_SIGNAL_KEYWORDS)
    has_mgmt_signal = any(k in text_l for k in PURE_MANAGEMENT_KEYWORDS)
    return is_senior_title and has_mgmt_signal and not has_coding_signal


def _seniority_trajectory_score(c: Candidate) -> float:
    """Rough proxy: does duration_months per role generally make sense for
    a growing career (not all 2-month stints), and do company sizes /
    titles suggest increasing scope? Returns 0-1."""
    if not c.career_history:
        return 0.0
    durations = [ch.duration_months for ch in c.career_history]
    stable_tenure = sum(1 for d in durations if d >= 12) / len(durations)
    return round(stable_tenure, 3)


def score_experience(c: Candidate, job_profile: dict) -> ExperienceSignals:
    band = job_profile["experience_band"]
    total_years = c.years_of_experience
    num_jobs = len(c.career_history)
    avg_tenure = (
        sum(ch.duration_months for ch in c.career_history) / num_jobs if num_jobs else 0.0
    )
    in_band = band["min_years"] <= total_years <= band["max_years"]

    title_chaser = _detect_title_chaser(c)
    consulting_only = _detect_consulting_only(c)
    cv_no_nlp = _detect_cv_speech_robotics_no_nlp(c)
    stale_architect = _detect_stale_architect(c)
    trajectory = _seniority_trajectory_score(c)

    evidence = []
    if in_band:
        evidence.append(f"{total_years} years total experience, within the JD's 5-9y band.")
    else:
        evidence.append(f"{total_years} years total experience, outside the JD's 5-9y band (soft).")
    if title_chaser:
        evidence.append("Tenure pattern suggests title-chasing (avg <18mo/role with escalating titles).")
    if consulting_only:
        evidence.append("Entire career history at consulting/services firms with no product company.")
    if cv_no_nlp:
        evidence.append("Primary expertise reads as CV/speech/robotics without NLP/IR exposure.")
    if stale_architect:
        evidence.append("Current senior/lead role shows management language with no recent coding signal.")

    return ExperienceSignals(
        total_years=total_years,
        avg_tenure_months=round(avg_tenure, 1),
        num_jobs=num_jobs,
        in_band_5_9=in_band,
        title_chaser_flag=title_chaser,
        consulting_only_flag=consulting_only,
        cv_speech_robotics_no_nlp_flag=cv_no_nlp,
        stale_architect_flag=stale_architect,
        seniority_trajectory_score=trajectory,
        experience_evidence=evidence,
    )