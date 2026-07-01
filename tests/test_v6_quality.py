import copy,json
from pathlib import Path
from pipeline.education_scorer import score_education
from pipeline.jd_parser import compile_job_profile
from pipeline.reranker import rerank_top
from pipeline.composite_ranker import RankedCandidate
from tests.test_pipeline import _make_candidate
from pipeline.candidate_loader import EducationEntry

def base(): return json.loads((Path(__file__).parents[1]/"data/job_profile.latest.json").read_text())

def test_institution_tier_is_counterfactually_invariant():
    a=_make_candidate(education=[EducationEntry("IIT X","B.Tech","Computer Science",2010,2014,"9/10","tier_1")]); b=copy.deepcopy(a)
    a.education[0].tier="tier_1"
    b.education[0].tier="tier_4"
    assert score_education(a)==score_education(b)

def test_different_jobs_compile_different_profiles():
    eng=compile_job_profile("Role: Search Engineer\nExperience: 5-8 years\nLocation: Pune Hybrid",base())
    sales=compile_job_profile("Role: Enterprise Sales Lead\nExperience: 10-14 years\nLocation: Mumbai Remote",base())
    assert eng["role_title"]!=sales["role_title"]
    assert eng["experience_band"]!=sales["experience_band"]
    assert eng["location"]["work_mode"]!=sales["location"]["work_mode"]

def test_jd_flags_soft_band_tension():
    p=compile_job_profile("Role: Engineer. Experience: 5-9 years; this range is not a requirement.",base())
    assert "experience_band_is_soft" in p["jd_intelligence"]["ambiguities_or_tensions"]

def test_reranker_rewards_balanced_evidence():
    def rc(cid,skill,exp):
        return RankedCandidate(_make_candidate(candidate_id=cid),.7,"strong_pass","",.7,skill,exp,.7,.7,1,1,False)
    balanced=rc("A",.8,.8); uneven=rc("B",1.0,.4)
    rerank_top([balanced,uneven],2)
    assert balanced.debug["reranker_bonus"]>uneven.debug["reranker_bonus"]
