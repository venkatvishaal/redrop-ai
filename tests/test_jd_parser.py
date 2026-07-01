import json
from pathlib import Path
from pipeline.jd_parser import compile_job_profile
def test_compiles_explicit_jd_constraints():
    base=json.loads((Path(__file__).parents[1]/"data/job_profile.latest.json").read_text())
    p=compile_job_profile("Job Description: Staff Search Engineer\nCompany: Acme\nExperience Required: 7-11 years\nLocation: Bengaluru (Remote)",base)
    assert p["role_title"]=="Staff Search Engineer" and p["experience_band"]["min_years"]==7
    assert p["location"]["work_mode"]=="remote" and "Bengaluru" in p["location"]["preferred_cities"]
