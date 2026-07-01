"""
location_scorer.py

Scores logistics fit: location/relocation against the JD's India,
Pune/Noida-preferred, no-visa-sponsorship stance, plus notice period
against the JD's sub-30-day preference (buyout up to 30 days).

Kept as its own small module (rather than folded into signal_scorer)
because it's driven by profile + redrob_signals together and maps
directly to one of the composite weights (location_logistics_fit) called
out explicitly in the JD ("Located in or willing to relocate to Noida or
Pune" is literally one of the five bullet points under "how to read
between the lines").
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.candidate_loader import Candidate


@dataclass
class LocationSignals:
    location_score: float
    notice_period_score: float
    logistics_score: float
    is_preferred_city: bool
    is_india: bool
    visa_would_be_required: bool


def score_location(c: Candidate, redrob_signals: dict, job_profile: dict) -> LocationSignals:
    loc_cfg = job_profile["location"]
    preferred_cities = [city.lower() for city in loc_cfg["preferred_cities"]]
    required_country = loc_cfg["country_required_unless_strong_signal"]

    location_l = c.location.lower()
    is_preferred_city = any(city in location_l for city in preferred_cities)
    is_india = c.country == required_country
    willing_to_relocate = bool(redrob_signals.get("willing_to_relocate", False))
    visa_would_be_required = not is_india

    if is_preferred_city:
        location_score = 1.0
    elif is_india and willing_to_relocate:
        location_score = 0.85
    elif is_india:
        location_score = 0.6
    else:
        # Outside India: JD says "case-by-case, but we don't sponsor work
        # visas." This is explicitly NOT a hard exclude in the JD's own
        # wording - a literal hard-exclude-on-country gate would be less
        # faithful to the source text than a steep penalty, not more.
        # "Case-by-case" does imply a high bar though: it shouldn't read as
        # "still broadly competitive," it should mean only a candidate who
        # is exceptional on every other dimension is worth the visa
        # conversation. Set low enough that only candidates near-maxed on
        # semantic/skill/experience can survive into the top 100; a
        # merely-good non-India candidate should not outrank a solid
        # India-based one.
        location_score = 0.04

    notice_days = int(redrob_signals.get("notice_period_days", 90) or 90)
    notice_cfg = job_profile["notice_period"]
    if notice_days <= notice_cfg["ideal_max_days"]:
        notice_score = 1.0
    elif notice_days <= notice_cfg["soft_cap_days"]:
        notice_score = 0.7
    else:
        # linear decay from 60 to 180 days
        notice_score = max(0.0, 0.7 - 0.6 * ((notice_days - 60) / 120))

    # Location dominates the logistics subscore (0.85) rather than 0.8/0.2
    # - notice period should matter even less relative to location for a
    # role this explicit about India-based hiring.
    preferred_mode = str(redrob_signals.get("preferred_work_mode", "")).lower()
    role_mode = str(loc_cfg.get("work_mode", "hybrid")).lower()
    mode_score = 1.0 if not preferred_mode or preferred_mode == role_mode else (0.7 if preferred_mode == "flexible" else 0.4)
    logistics_score = round(0.78 * location_score + 0.15 * notice_score + 0.07 * mode_score, 4)

    return LocationSignals(
        location_score=round(location_score, 3),
        notice_period_score=round(notice_score, 3),
        logistics_score=logistics_score,
        is_preferred_city=is_preferred_city,
        is_india=is_india,
        visa_would_be_required=visa_would_be_required,
    )
