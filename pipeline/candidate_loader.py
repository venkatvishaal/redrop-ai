"""
candidate_loader.py

Loads the candidate pool from JSONL (optionally gzip'd), with light
normalization so downstream scorers can rely on consistent field shapes.
No network, no heavy deps - just stdlib json/gzip plus dataclasses.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional


DATE_FMT = "%Y-%m-%d"


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, DATE_FMT).date()
    except (ValueError, TypeError):
        return None


@dataclass
class CareerEntry:
    company: str
    title: str
    start_date: Optional[date]
    end_date: Optional[date]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"


@dataclass
class SkillEntry:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0


@dataclass
class Candidate:
    candidate_id: str

    # profile
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""

    career_history: List[CareerEntry] = field(default_factory=list)
    education: List[EducationEntry] = field(default_factory=list)
    skills: List[SkillEntry] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, Any]] = field(default_factory=list)

    redrob_signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def skill_names_lower(self) -> List[str]:
        return [s.name.lower() for s in self.skills]

    @property
    def full_text_blob(self) -> str:
        """Concatenated free text used for embedding / lexical matching.
        Deliberately excludes the bare skills list so a skills-only keyword
        stuffer doesn't get credit equal to someone whose career_history
        actually demonstrates the work (see JD's explicit trap warning)."""
        parts = [self.headline, self.summary]
        for ch in self.career_history:
            parts.append(f"{ch.title} at {ch.company}. {ch.description}")
        for ed in self.education:
            parts.append(f"{ed.degree} {ed.field_of_study}")
        return " ".join(p for p in parts if p)

    @property
    def skills_text_blob(self) -> str:
        return " ".join(s.name for s in self.skills)


def _build_candidate(rec: Dict[str, Any]) -> Candidate:
    profile = rec.get("profile", {})
    rs = rec.get("redrob_signals", {})

    career_history = []
    for ch in rec.get("career_history", []):
        career_history.append(CareerEntry(
            company=ch.get("company", ""),
            title=ch.get("title", ""),
            start_date=_parse_date(ch.get("start_date")),
            end_date=_parse_date(ch.get("end_date")),
            duration_months=int(ch.get("duration_months", 0) or 0),
            is_current=bool(ch.get("is_current", False)),
            industry=ch.get("industry", ""),
            company_size=ch.get("company_size", ""),
            description=ch.get("description", ""),
        ))

    education = []
    for ed in rec.get("education", []):
        education.append(EducationEntry(
            institution=ed.get("institution", ""),
            degree=ed.get("degree", ""),
            field_of_study=ed.get("field_of_study", ""),
            start_year=int(ed.get("start_year", 0) or 0),
            end_year=int(ed.get("end_year", 0) or 0),
            grade=ed.get("grade"),
            tier=ed.get("tier", "unknown"),
        ))

    skills = []
    for sk in rec.get("skills", []):
        skills.append(SkillEntry(
            name=sk.get("name", ""),
            proficiency=sk.get("proficiency", ""),
            endorsements=int(sk.get("endorsements", 0) or 0),
            duration_months=int(sk.get("duration_months", 0) or 0),
        ))

    return Candidate(
        candidate_id=rec["candidate_id"],
        anonymized_name=profile.get("anonymized_name", ""),
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        years_of_experience=float(profile.get("years_of_experience", 0) or 0),
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        current_company_size=profile.get("current_company_size", ""),
        current_industry=profile.get("current_industry", ""),
        career_history=career_history,
        education=education,
        skills=skills,
        certifications=rec.get("certifications", []) or [],
        languages=rec.get("languages", []) or [],
        redrob_signals=rs,
    )


def iter_candidates_raw(path: str):
    """Yields raw dict records from a .jsonl or .jsonl.gz file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[candidate_loader] Skipping malformed JSON on line {lineno}: {e}", flush=True)


def load_candidates(path: str) -> List[Candidate]:
    """Loads the full candidate pool into memory as Candidate objects.

    For 100K candidates at the schema's typical record size this comfortably
    fits the 16GB memory budget (well under 1GB in practice)."""
    candidates = []
    for rec in iter_candidates_raw(path):
        try:
            candidates.append(_build_candidate(rec))
        except (KeyError, TypeError, ValueError) as e:
            # Skip malformed records rather than crash the whole pipeline;
            # log to stderr so issues are visible without halting a 100K run.
            import sys
            print(f"[candidate_loader] skipping malformed record: {e}", file=sys.stderr)
    return candidates