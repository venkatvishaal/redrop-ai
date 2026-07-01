"""
output_writer.py

Writes:
  - submission.csv: exactly per submission_spec.md Section 2-3 - header
    `candidate_id,rank,score,reasoning`, exactly 100 data rows, ranks
    1-100 each used once, score non-increasing by rank, ties broken by
    candidate_id ascending (matches validate_submission.py exactly).
  - detailed_results.json: richer per-candidate breakdown for the
    dashboard / for your own debugging and Stage 5 interview prep. Not
    part of the graded submission.
"""

from __future__ import annotations

import csv
import json
from typing import List

from pipeline.composite_ranker import RankedCandidate


def _tie_break_sort_key(rc: RankedCandidate):
    # Higher score first; for equal scores, candidate_id ASCENDING
    # (matches validate_submission.py's tie-break check exactly).
    # IMPORTANT: tie-break on the score as it will be WRITTEN (rounded to
    # 4 decimals), not the raw float - otherwise two candidates whose
    # raw scores differ in the 5th decimal but round to the same printed
    # value can end up in descending candidate_id order, which the
    # validator flags as a tie-break violation.
    return (-round(rc.final_score, 4), rc.candidate.candidate_id)


def write_submission_csv(ranked: List[RankedCandidate], path: str, top_n: int = 100) -> None:
    eligible = [rc for rc in ranked if not rc.hard_excluded]
    ranked_sorted = sorted(eligible, key=_tie_break_sort_key)[:top_n]
    if len(ranked_sorted) < top_n:
        raise ValueError(f"Only {len(ranked_sorted)} eligible candidates; need {top_n}")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, rc in enumerate(ranked_sorted, start=1):
            writer.writerow([
                rc.candidate.candidate_id,
                i,
                f"{rc.final_score:.4f}",
                rc.reasoning,
            ])


def write_detailed_results_json(ranked: List[RankedCandidate], path: str, top_n: int = 1000) -> None:
    ranked_sorted = sorted(ranked, key=_tie_break_sort_key)[:top_n]
    out = []
    for i, rc in enumerate(ranked_sorted, start=1):
        out.append({
            "rank": i,
            "candidate_id": rc.candidate.candidate_id,
            "final_score": rc.final_score,
            "core_fit_status": rc.core_fit_status,
            "reasoning": rc.reasoning,
            "current_title": rc.candidate.current_title,
            "current_company": rc.candidate.current_company,
            "location": rc.candidate.location,
            "country": rc.candidate.country,
            "years_of_experience": rc.candidate.years_of_experience,
            "subscores": {
                "semantic": rc.semantic_score,
                "skill": rc.skill_score,
                "experience": rc.experience_score,
                "behavioral": rc.behavioral_score,
                "logistics": rc.logistics_score,
            },
            "rule_multiplier": rc.rule_multiplier,
            "sanity_multiplier": rc.sanity_multiplier,
            "hard_excluded": rc.hard_excluded,
            "hard_exclude_reason": rc.hard_exclude_reason,
            "debug": rc.debug,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
