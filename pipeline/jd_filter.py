"""
jd_filter.py

Fast, deterministic pass that hard-disqualifies candidates BEFORE the
CPU-heavier scoring stages run. Mirrors the architecture doc's intent:
"instantly hard-disqualifies candidates failing mandatory parameters."

What this stage does NOT do: it does not try to be clever about fit
quality - that's composite_ranker's job. This stage only removes
candidates that are categorically out of consideration, so later stages
operate on a smaller, valid pool.

Hard exclusions implemented here:
  1. Salary mismatch: candidate's expected_salary_range_inr_lpa.min is far
     above any reasonable band for the role (no JD salary was published,
     so we use a generous upper bound rather than guessing a number -
     this avoids inventing a JD constraint that doesn't exist).
  2. Pure-research-only career with zero production deployment evidence -
     this is an explicit, named hard disqualifier in the JD text itself
     ("we will not move forward").
  3. Outside-India with no relocation willingness AND no strong
     compensating signal - the JD says "case-by-case" for outside-India,
     not an automatic reject, so this is a soft signal handled in
     composite scoring, NOT a hard filter. (Documented here to explain
     why it's absent from this list.)

Everything else the JD warns about (title-chasers, framework enthusiasts,
consulting-only, CV/speech-without-NLP, closed-source-only) is a PENALTY,
not a hard exclusion, because the JD explicitly says these make someone a
worse fit, not a categorical non-candidate - rule_engine.py applies these
as score multipliers so the reasoning can still surface nuance.
"""

from __future__ import annotations

from typing import List, Tuple

from pipeline.candidate_loader import Candidate

# No salary ceiling was published in the JD. We deliberately do not invent
# one; this filter is a no-op for salary unless a future JD configuration
# supplies max_salary_lpa explicitly. Kept as a documented hook.
MAX_REASONABLE_SALARY_LPA = None  # e.g. 80 if the JD ever specifies a band


def _has_production_evidence(c: Candidate) -> bool:
    """A career_history entry counts as production evidence if its
    description mentions shipping/operating something for real users,
    as opposed to purely research/paper/lab language."""
    production_markers = (
        "production", "shipped", "deployed", "real users", "scale",
        "users", "live system", "in prod", "rolled out", "launched",
    )
    research_only_markers = (
        "research lab", "phd", "academic", "published a paper",
        "research-only", "thesis",
    )
    text = " ".join(ch.description.lower() for ch in c.career_history)
    has_prod = any(m in text for m in production_markers)
    return has_prod


def _is_pure_research_no_production(c: Candidate) -> bool:
    if not c.career_history:
        return False
    text = " ".join(ch.description.lower() for ch in c.career_history)
    titles = " ".join(ch.title.lower() for ch in c.career_history)
    industries = " ".join(ch.industry.lower() for ch in c.career_history)

    research_signal = any(
        kw in text or kw in titles or kw in industries
        for kw in ("research scientist", "research lab", "academic", "phd researcher", "postdoc")
    )
    if not research_signal:
        return False
    return not _has_production_evidence(c)


def apply_hard_exclusions(candidates: List[Candidate]) -> Tuple[List[Candidate], List[Tuple[Candidate, str]]]:
    """Returns (surviving_candidates, excluded_with_reason)."""
    survivors = []
    excluded = []

    for c in candidates:
        # Salary hard-cap (no-op unless configured - see module docstring)
        if MAX_REASONABLE_SALARY_LPA is not None:
            exp_min = (c.redrob_signals.get("expected_salary_range_inr_lpa") or {}).get("min", 0)
            if exp_min and exp_min > MAX_REASONABLE_SALARY_LPA:
                excluded.append((c, "salary_expectation_exceeds_band"))
                continue

        if _is_pure_research_no_production(c):
            excluded.append((c, "pure_research_no_production"))
            continue

        survivors.append(c)

    return survivors, excluded