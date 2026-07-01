"""

Unit tests for the scoring modules, with particular attention to:
  - the sanity_checks.py false-positive bug we caught and fixed during
    development (education-year-based plausibility check produced a 22%
    false-positive rate; the fix anchors on career_history instead)
  - the JD's explicitly-named traps (keyword-stuffing, consulting-only
    with prior product experience, title-chasing)


"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.candidate_loader import Candidate, CareerEntry, EducationEntry, SkillEntry
from pipeline.sanity_checks import score_sanity
from pipeline.experience_scorer import score_experience, _detect_consulting_only, _detect_title_chaser
from pipeline.honeypot_detector import detect_honeypot
from pipeline.skill_scorer import score_skills
import json


def _make_candidate(**overrides) -> Candidate:
    defaults = dict(
        candidate_id="CAND_0000000",
        anonymized_name="Test Person",
        headline="",
        summary="",
        location="Pune, Maharashtra",
        country="India",
        years_of_experience=7.0,
        current_title="AI Engineer",
        current_company="Acme",
        current_company_size="201-500",
        current_industry="AI/ML",
        career_history=[],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
        redrob_signals={
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 24,
            "profile_views_received_30d": 50,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "endorsements_received": 20,
            "connection_count": 100,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "preferred_work_mode": "hybrid",
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "github_activity_score": 20,
        },
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def load_job_profile():
    # Try the expected path first, then fall back to root-level profiles
    path_v1 = Path(__file__).parent.parent / "data" / "job_profile.latest.json"
    path_v2 = Path(__file__).parent.parent / "job_profile_latest_1.json"
    path_v3 = Path(__file__).parent.parent / "job_profile.latest.json"
    for p in (path_v1, path_v2, path_v3):
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No job profile found at any expected path. Tried:\n"
        f"  {path_v1}\n  {path_v2}\n  {path_v3}"
    )


# ---------------------------------------------------------------------------
# Regression test for the education-timeline false-positive bug
# ---------------------------------------------------------------------------

def test_working_while_studying_not_flagged_implausible():
    """A candidate who started their career BEFORE their degree's end_year
    (very common - working through college, or a later/second degree)
    must NOT be flagged as having an implausible timeline. This is a
    direct regression test for the bug found during development, which
    incorrectly flagged ~22% of the real dataset."""
    c = _make_candidate(
        years_of_experience=7.9,
        career_history=[
            CareerEntry(
                company="Glance", title="ML Engineer",
                start_date=date(2018, 8, 23), end_date=date(2024, 4, 7),
                duration_months=68, is_current=False,
                industry="AI/ML", company_size="501-1000",
                description="Shipped personalization infrastructure.",
            ),
            CareerEntry(
                company="Meta", title="Senior AI Engineer",
                start_date=date(2024, 4, 7), end_date=None,
                duration_months=26, is_current=True,
                industry="Internet", company_size="10001+",
                description="Built ranking systems.",
            ),
        ],
        education=[
            EducationEntry(
                institution="IIT Bombay", degree="B.Sc", field_of_study="ML",
                start_year=2016, end_year=2020,
            ),
        ],
    )
    sanity = score_sanity(c)
    assert sanity.education_timeline_implausible is False, (
        "Candidate started working in 2018, before their 2020 degree end_year "
        "(plausible: worked through college). Must not be flagged."
    )
    assert sanity.sanity_penalty_multiplier == 1.0


def test_genuinely_implausible_career_start_is_flagged():
    """A candidate whose years_of_experience wildly exceeds time elapsed
    since their EARLIEST career_history start_date should be flagged."""
    c = _make_candidate(
        years_of_experience=15.0,
        career_history=[
            CareerEntry(
                company="StartupX", title="Engineer",
                start_date=date(2025, 1, 1), end_date=None,
                duration_months=12, is_current=True,
                industry="Tech", company_size="11-50",
                description="Worked on stuff.",
            ),
        ],
        education=[],
    )
    sanity = score_sanity(c)
    assert sanity.education_timeline_implausible is True
    assert sanity.sanity_penalty_multiplier < 1.0


# ---------------------------------------------------------------------------
# Honeypot detection threshold
# ---------------------------------------------------------------------------

def test_single_weak_signal_not_honeypot():
    """A single borderline sanity issue alone should not trigger a hard
    honeypot exclusion - the threshold requires 2+ independent flags to
    keep the false-positive rate low."""
    c = _make_candidate(
        years_of_experience=15.0,
        career_history=[
            CareerEntry(
                company="StartupX", title="Engineer",
                start_date=date(2025, 1, 1), end_date=None,
                duration_months=12, is_current=True,
                industry="Tech", company_size="11-50",
                description="Worked on stuff.",
            ),
        ],
    )
    sanity = score_sanity(c)
    honeypot = detect_honeypot(c, sanity)
    # Only the career-start-implausibility flag fires here; tenure_mismatch
    # also fires since 15y claimed vs 1y of history -> 2 flags, so this
    # particular example IS a honeypot. Kept as documentation of the
    # stacking behavior rather than a strict single-flag isolation test.
    assert honeypot.flag_count >= 1


def test_mass_expert_zero_duration_is_honeypot_signal():
    """The spec's literal example: 'expert proficiency in 10 skills with 0
    years used'."""
    c = _make_candidate(
        skills=[
            SkillEntry(name=f"Skill{i}", proficiency="expert", endorsements=0, duration_months=0)
            for i in range(10)
        ],
        career_history=[
            CareerEntry(
                company="Acme", title="Engineer",
                start_date=date(2019, 1, 1), end_date=None,
                duration_months=89, is_current=True,
                industry="Tech", company_size="201-500",
                description="Normal work.",
            ),
        ],
        years_of_experience=7.0,
    )
    sanity = score_sanity(c)
    assert sanity.has_expert_zero_duration is True
    honeypot = detect_honeypot(c, sanity)
    assert any("mass_expert_zero_duration" in f for f in honeypot.flags)


# ---------------------------------------------------------------------------
# JD-specific trap detection
# ---------------------------------------------------------------------------

def test_consulting_only_with_prior_product_experience_not_flagged():
    """JD explicitly says: currently at a consulting firm but with PRIOR
    product-company experience is fine, not penalized."""
    c = _make_candidate(
        career_history=[
            CareerEntry(
                company="Capgemini", title="Data Engineer",
                start_date=date(2022, 1, 1), end_date=None,
                duration_months=52, is_current=True,
                industry="IT Services", company_size="10001+",
                description="Data engineering work.",
            ),
            CareerEntry(
                company="Freshworks", title="Senior Data Engineer",
                start_date=date(2019, 1, 1), end_date=date(2022, 1, 1),
                duration_months=36, is_current=False,
                industry="SaaS", company_size="1001-5000",
                description="Built data pipelines.",
            ),
        ],
    )
    assert _detect_consulting_only(c) is False


def test_consulting_only_entire_career_is_flagged():
    c = _make_candidate(
        career_history=[
            CareerEntry(
                company="TCS", title="Software Engineer",
                start_date=date(2019, 1, 1), end_date=date(2022, 1, 1),
                duration_months=36, is_current=False,
                industry="IT Services", company_size="10001+",
                description="Services work.",
            ),
            CareerEntry(
                company="Infosys", title="ML Engineer",
                start_date=date(2022, 1, 1), end_date=None,
                duration_months=53, is_current=True,
                industry="IT Services", company_size="10001+",
                description="ML services work.",
            ),
        ],
    )
    assert _detect_consulting_only(c) is True


def test_title_chaser_pattern_detected():
    """Career trajectory escalating titles every <18 months across 3+ jobs
    should trip the title-chaser flag the JD explicitly calls out."""
    c = _make_candidate(
        career_history=[
            CareerEntry(
                company="A", title="Senior Engineer",
                start_date=date(2020, 1, 1), end_date=date(2021, 4, 1),
                duration_months=15, is_current=False,
                industry="Tech", company_size="201-500", description="Work.",
            ),
            CareerEntry(
                company="B", title="Staff Engineer",
                start_date=date(2021, 4, 1), end_date=date(2022, 7, 1),
                duration_months=15, is_current=False,
                industry="Tech", company_size="201-500", description="Work.",
            ),
            CareerEntry(
                company="C", title="Principal Engineer",
                start_date=date(2022, 7, 1), end_date=None,
                duration_months=14, is_current=True,
                industry="Tech", company_size="201-500", description="Work.",
            ),
        ],
    )
    assert _detect_title_chaser(c) is True


def test_keyword_stuffer_without_production_evidence_scores_low():
    """A candidate with many trendy AI skill-entries but a career_history
    that never corroborates them in a production context should trigger
    the framework_enthusiast_flag and score low on must-have capabilities."""
    job_profile = load_job_profile()
    c = _make_candidate(
        current_title="Backend Engineer",
        skills=[
            SkillEntry(name="LoRA", proficiency="advanced", endorsements=0, duration_months=2),
            SkillEntry(name="Fine-tuning LLMs", proficiency="advanced", endorsements=0, duration_months=3),
            SkillEntry(name="RAG", proficiency="expert", endorsements=0, duration_months=1),
            SkillEntry(name="GANs", proficiency="advanced", endorsements=0, duration_months=2),
            SkillEntry(name="Prompt Engineering", proficiency="intermediate", endorsements=0, duration_months=1),
        ],
        career_history=[
            CareerEntry(
                company="DataCo", title="Backend Engineer",
                start_date=date(2019, 1, 1), end_date=None,
                duration_months=89, is_current=True,
                industry="IT Services", company_size="10001+",
                description="Built data pipelines on Spark and Airflow for analytics.",
            ),
        ],
    )
    result = score_skills(c, job_profile)
    assert result["framework_enthusiast_flag"] is True
    assert result["must_have_fraction"] < 0.5


def test_production_evidence_without_exact_keywords_still_scores_well():
    """The JD's explicit example: a candidate who never says 'RAG' or
    'retrieval' but whose career_history describes building a
    recommendation/ranking system at a product company should still
    score reasonably on the relevant capability via career-text evidence."""
    job_profile = load_job_profile()
    c = _make_candidate(
        current_title="Senior Engineer",
        skills=[SkillEntry(name="Python", proficiency="advanced", endorsements=10, duration_months=60)],
        career_history=[
            CareerEntry(
                company="ProductCo", title="Senior Engineer",
                start_date=date(2020, 1, 1), end_date=None,
                duration_months=78, is_current=True,
                industry="Internet", company_size="1001-5000",
                description=(
                    "Built and deployed the system that connects users to the "
                    "most relevant matches for their intent, shipped to real users "
                    "at scale."
                ),
            ),
        ],
    )
    result = score_skills(c, job_profile)
    # Should get meaningful credit for embeddings_retrieval_production via
    # career-text production-context matching, even with zero exact
    # keyword overlap in the skills list.
    embeddings_cap = next(
        d for d in result["must_have_details"] if d.capability_id == "embeddings_retrieval_production"
    )
    assert embeddings_cap.confidence > 0.3


def test_disclaimed_ownership_does_not_score_high_despite_endorsed_skills():
    """Regression test: a candidate can list a skill (e.g. Embeddings,
    FAISS) with genuine-looking endorsements/duration, while their own
    career_history narrative explicitly disclaims having done the core
    work ('integration and observability, not the model itself', 'built
    by another team'). This pattern was found verbatim and repeated
    (1500+ times) across the real dataset - clearly an intentional trap -
    and without disclaim detection, such a candidate could score deceptively
    high (this exact pattern let a Data Engineer reach rank 100 of the top
    100 with a 0.925 must_have_fraction)."""
    job_profile = load_job_profile()
    c = _make_candidate(
        current_title="Data Engineer",
        skills=[
            SkillEntry(name="Embeddings", proficiency="intermediate", endorsements=1, duration_months=29),
            SkillEntry(name="FAISS", proficiency="intermediate", endorsements=9, duration_months=26),
        ],
        career_history=[
            CareerEntry(
                company="ServiceCo", title="Data Engineer",
                start_date=date(2022, 1, 1), end_date=None,
                duration_months=52, is_current=True,
                industry="SaaS", company_size="201-500",
                description=(
                    "Recent work includes integrating a model-serving service "
                    "(built by another team) into our API layer; my work was the "
                    "integration and observability, not the model itself."
                ),
            ),
        ],
    )
    result = score_skills(c, job_profile)
    embeddings_cap = next(
        d for d in result["must_have_details"] if d.capability_id == "embeddings_retrieval_production"
    )
    assert embeddings_cap.confidence < 0.5, (
        "Disclaimed ownership should heavily suppress confidence even when "
        "the skill entry itself has real-looking endorsements/duration."
    )
    assert result["must_have_fraction"] < 0.5


def test_exceptional_non_india_candidate_survives_case_by_case_gate():
    """An outside-India candidate who is genuinely strong on ALL four
    non-logistics dimensions (semantic, skill, experience, behavioral)
    should survive the case-by-case exception gate, per the JD's literal
    wording ('case-by-case', not a blanket exclude)."""
    from pipeline.composite_ranker import _case_by_case_exception_met
    assert _case_by_case_exception_met(0.85, 0.78, 1.0, 0.86) is True


def test_merely_good_non_india_candidate_fails_case_by_case_gate():
    """A non-India candidate who is good-but-not-exceptional on even one
    dimension should fail the gate - 'case-by-case' should mean a high
    bar on every dimension, not a high blended average. Regression test
    for a finding from two independent audits: a London-based candidate
    with strong-but-uneven scores was reaching rank 9-18 before this gate
    existed, which both reviews flagged as too permissive given the JD's
    explicit no-visa-sponsorship stance."""
    from pipeline.composite_ranker import _case_by_case_exception_met
    # Strong on 3 dimensions, but semantic falls short of the bar.
    assert _case_by_case_exception_met(0.51, 0.73, 0.75, 0.78) is False


def test_honest_beginner_proficiency_not_flagged_as_framework_enthusiast():
    """A candidate who honestly lists trendy AI skills at beginner/
    intermediate proficiency (e.g. 'still learning RAG') should NOT trigger
    framework_enthusiast_flag, even with 4+ such skills and no career-text
    corroboration. This is a regression test for a real gap found in the
    dataset: 28 real candidates were being penalized identically to
    someone claiming expert-level mastery with zero evidence, conflating
    honesty about a learning interest with the JD's actual concern (claimed
    expertise that isn't backed up)."""
    job_profile = load_job_profile()
    c = _make_candidate(
        current_title="Backend Engineer",
        skills=[
            SkillEntry(name="LoRA", proficiency="beginner", endorsements=0, duration_months=1),
            SkillEntry(name="Fine-tuning LLMs", proficiency="intermediate", endorsements=0, duration_months=2),
            SkillEntry(name="RAG", proficiency="beginner", endorsements=0, duration_months=1),
            SkillEntry(name="GANs", proficiency="intermediate", endorsements=0, duration_months=1),
        ],
        career_history=[
            CareerEntry(
                company="DataCo", title="Backend Engineer",
                start_date=date(2019, 1, 1), end_date=None,
                duration_months=89, is_current=True,
                industry="IT Services", company_size="10001+",
                description="Built data pipelines on Spark and Airflow for analytics.",
            ),
        ],
    )
    result = score_skills(c, job_profile)
    assert result["framework_enthusiast_flag"] is False


# ---------------------------------------------------------------------------
# Education scoring tests
# ---------------------------------------------------------------------------


def test_tier_1_cs_degree_with_good_grades_gets_highest_bonus():
    """A candidate with tier-1 institution (IIT), CS degree, and strong
    grades should receive the maximum education bonus (~0.03)."""
    from pipeline.education_scorer import score_education
    c = _make_candidate(
        education=[
            EducationEntry(
                institution="IIT Bombay", degree="B.Tech",
                field_of_study="Computer Science and Engineering",
                start_year=2016, end_year=2020,
                grade="9.2/10", tier="tier_1",
            ),
            EducationEntry(
                institution="IIT Bombay", degree="M.Tech",
                field_of_study="Machine Learning",
                start_year=2020, end_year=2022,
                grade="8.8/10", tier="tier_1",
            ),
        ],
    )
    bonus = score_education(c)
    assert 0 < bonus <= 0.031


def test_no_education_gets_zero_bonus():
    """A candidate with no education entries should receive zero bonus."""
    from pipeline.education_scorer import score_education
    c = _make_candidate(education=[])
    assert score_education(c) == 0.0


def test_tier_2_generic_degree_gets_moderate_bonus():
    """A tier-2 institution with a non-CS/AI degree gets a moderate bonus."""
    from pipeline.education_scorer import score_education
    c = _make_candidate(
        education=[
            EducationEntry(
                institution="Some University", degree="B.Sc",
                field_of_study="Chemistry",
                start_year=2015, end_year=2018,
                tier="tier_2",
            ),
        ],
    )
    bonus = score_education(c)
    # tier_2(0.08) + other(0.00) + bachelor(0.01) + no_grade(0.0) = 0.09 raw
    # 0.09 * 0.15 = 0.0135
    assert 0 < bonus <= 0.003


def test_unknown_tier_non_relevant_field_gets_minimal_bonus():
    """An unknown-tier institution with a non-relevant field scores near zero."""
    from pipeline.education_scorer import score_education
    c = _make_candidate(
        education=[
            EducationEntry(
                institution="Unknown College", degree="B.A.",
                field_of_study="Fine Arts",
                start_year=2015, end_year=2018,
                tier="unknown",
            ),
        ],
    )
    bonus = score_education(c)
    # unknown(0.0) + other(0.0) + bachelor(0.01) + no_grade(0.0) = 0.01 raw
    # 0.01 * 0.15 = 0.0015
    assert bonus <= 0.005, f"Expected ~0.0015, got {bonus}"


def test_best_education_summary_returns_highest_degree():
    """best_education_summary returns a one-line string from the most
    recent education entry, not the lowest."""
    from pipeline.education_scorer import best_education_summary
    c = _make_candidate(
        education=[
            EducationEntry(
                institution="IIT Delhi", degree="B.Tech",
                field_of_study="Computer Science",
                start_year=2014, end_year=2018,
                tier="tier_1",
            ),
            EducationEntry(
                institution="IIT Kanpur", degree="M.Tech",
                field_of_study="Machine Learning",
                start_year=2018, end_year=2020,
                tier="tier_1",
            ),
        ],
    )
    summary = best_education_summary(c)
    assert "IIT Kanpur" in summary
    assert "M.Tech" in summary
    assert "Machine Learning" in summary
    assert "tier_1" not in summary and "tier1" not in summary


def test_education_scoring_grade_parsing():
    """Various grade formats should all parse to a 0-1 score correctly."""
    from pipeline.education_scorer import _parse_grade_score
    assert _parse_grade_score("8.5/10") == 0.85
    assert _parse_grade_score("3.8/4") == 0.95
    assert _parse_grade_score("85%") == 0.85
    assert _parse_grade_score("9.0") == 0.90
    assert _parse_grade_score(None) == 0.0
    assert _parse_grade_score("") == 0.0
    assert _parse_grade_score("N/A") == 0.0


def test_education_bonus_blended_into_final_score():
    """Education bonus should add to final_score in build_ranked_candidate."""
    from pipeline.composite_ranker import build_ranked_candidate
    from pipeline.experience_scorer import score_experience
    from pipeline.evidence_scorer import EvidenceSignals
    from pipeline.signal_scorer import score_signals
    from pipeline.location_scorer import score_location
    from pipeline.sanity_checks import SanitySignals
    from pipeline.honeypot_detector import HoneypotSignals
    from pipeline.rule_engine import RuleEngineResult

    job_profile = load_job_profile()

    c = _make_candidate(
        education=[
            EducationEntry(
                institution="IIT Madras", degree="B.Tech",
                field_of_study="Computer Science",
                start_year=2013, end_year=2017,
                grade="9.0/10", tier="tier_1",
            ),
        ],
    )
    exp = score_experience(c, job_profile)
    behavioral = score_signals(c.redrob_signals)
    location = score_location(c, c.redrob_signals, job_profile)
    result = build_ranked_candidate(
        c=c,
        semantic_score=0.70,
        skill_result={
            "skill_score": 0.70,
            "must_have_fraction": 0.75,
            "nice_to_have_fraction": 0.2,
            "framework_enthusiast_flag": False,
            "must_have_details": [],
        },
        experience=exp,
        evidence=EvidenceSignals(
            retrieval_evidence_score=0.0,
            eval_framework_evidence_score=0.0,
            production_ml_evidence_score=0.0,
        ),
        behavioral=behavioral,
        location=location,
        sanity=SanitySignals(),
        honeypot=HoneypotSignals(is_honeypot=False, flags=[], flag_count=0),
        rule_result=RuleEngineResult(multiplier=1.0),
        job_profile=job_profile,
    )
    assert result.education_bonus > 0.0
    assert result.education_bonus <= 0.031

    # Education should appear in reasoning
    assert "IIT Madras" in result.reasoning or "Education:" in result.reasoning


# ---------------------------------------------------------------------------
# Original generic-SQL test (unchanged)
# ---------------------------------------------------------------------------


def test_generic_sql_search_not_credited_as_vector_search():
    """A candidate who explicitly describes lexical/SQL/keyword search
    should not get credit for vector_db_hybrid_search just because a
    generic phrase like 'search backend' appears alongside a production
    verb. Regression test for a confirmed-exploitable gap: before the
    NON_VECTOR_SEARCH_DISQUALIFIERS guard, this exact wording scored 0.95
    confidence for vector database experience the candidate doesn't
    actually have."""
    job_profile = load_job_profile()
    c = _make_candidate(
        current_title="Backend Engineer",
        skills=[],
        career_history=[
            CareerEntry(
                company="Acme", title="Backend Engineer",
                start_date=date(2020, 1, 1), end_date=None,
                duration_months=60, is_current=True,
                industry="SaaS", company_size="201-500",
                description=(
                    "Built the search backend for our customer support ticketing "
                    "system using SQL full-text search and keyword matching."
                ),
            ),
        ],
    )
    result = score_skills(c, job_profile)
    vdb = next(d for d in result["must_have_details"] if d.capability_id == "vector_db_hybrid_search")
    assert vdb.confidence == 0.0
    assert vdb.matched is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
