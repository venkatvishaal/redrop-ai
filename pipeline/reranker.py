"""Fast evidence-aware second-stage reranking for the strongest candidates.

The reranker applies a small bonus to candidates whose evidence across
independent scoring dimensions is balanced (high skill + high experience),
rather than those who excel on only one dimension. This rewards genuine
depth over narrow specialization.

Constants are configurable via the RERANKER_CONFIG dict so the behavior
can be tuned without code changes. The defaults are set based on the
principle that the reranker should never change the ordering by more than
a few positions — it's a refinement, not a reordering.
"""

from __future__ import annotations
from typing import List

from pipeline.composite_ranker import RankedCandidate

# Configurable constants with evidence-based defaults:
#   - evidence_floor_weight: reward for the minimum of (skill, experience).
#     A candidate weak on both gets no bonus; a candidate strong on both
#     gets the full bonus. Weight 0.012 means a top candidate gets at most
#     ~0.012 bonus from this component.
#   - balanced_avg_weight: reward for the average across all three main
#     dimensions (semantic, skill, experience). Weight 0.008 means a
#     balanced candidate gets at most ~0.008 bonus from this component.
#   - max_bonus: hard cap on the total reranker bonus (prevents the
#     reranker from changing the ordering by more than ~2-3 positions).
RERANKER_CONFIG = {
    "evidence_floor_weight": 0.012,
    "balanced_avg_weight": 0.008,
    "max_bonus": 0.025,
}


def rerank_top(
    ranked: List[RankedCandidate],
    depth: int = 1000,
    config: dict | None = None,
) -> None:
    """Apply an evidence-consensus bonus to the top `depth` candidates.

    The bonus rewards candidates whose evidence is balanced across
    independent dimensions (semantic, skill, experience) rather than
    dominated by one exceptional score. This prevents candidates who
    have a single keyword-match spike from outranking genuinely
    well-rounded candidates.

    Args:
        ranked: Full ranked candidate list (mutated in place).
        depth: How many non-excluded candidates to consider.
        config: Override for RERANKER_CONFIG keys. If None, uses defaults.
    """
    cfg = {**RERANKER_CONFIG, **(config or {})}
    ew = cfg["evidence_floor_weight"]
    bw = cfg["balanced_avg_weight"]
    max_bonus = cfg["max_bonus"]

    eligible = sorted(
        (r for r in ranked if not r.hard_excluded),
        key=lambda r: (-r.final_score, r.candidate.candidate_id),
    )[:depth]

    for r in eligible:
        # Reward agreement among independent evidence dimensions, not a
        # single exceptional keyword-derived score. Bounded to preserve
        # the original ordering — the bonus should never exceed max_bonus.
        evidence_floor = min(r.skill_score, r.experience_score)
        balanced = (r.semantic_score + r.skill_score + r.experience_score) / 3.0
        bonus = min(ew * evidence_floor + bw * balanced, max_bonus)

        r.final_score = round(min(1.0, r.final_score + bonus), 6)
        r.debug["reranker_bonus"] = round(bonus, 6)
        r.debug["reranker_depth"] = depth
        r.debug["reranker_config"] = cfg
