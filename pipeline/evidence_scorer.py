"""
evidence_scorer.py

Scans career_history descriptions for verifiable evidence of specific
systems work that the JD cares about most: retrieval systems, production
ML operations, and evaluation-framework rigor. This is distinct from
skill_scorer (which checks capability coverage) - evidence_scorer looks
for the *kind of detail that's hard to fake generically*, i.e. operational
specifics (drift, refresh, regression, A/B, offline/online correlation)
rather than just naming a technology.

The JD explicitly cares about candidates who've "handled embedding drift,
index refresh, retrieval-quality regression in production" - that's
operational language, not a tech-stack checklist, so this module greps
for operational verbs/nouns near the relevant nouns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pipeline.candidate_loader import Candidate
from pipeline.skill_scorer import DISCLAIM_PATTERNS

RETRIEVAL_OPERATIONAL_TERMS = (
    "embedding drift", "index refresh", "retrieval-quality regression",
    "retrieval quality regression", "reindex", "recall@", "precision@",
    "vector index", "hybrid retrieval", "dense retrieval",
)
EVAL_OPERATIONAL_TERMS = (
    "ndcg", "mrr", "map", "offline metric", "online metric",
    "offline-to-online", "a/b test", "ab test", "experimentation",
    "correlat", "evaluation framework", "offline eval",
)
PRODUCTION_ML_OPERATIONAL_TERMS = (
    "production", "real-time", "latency", "throughput", "on-call",
    "monitoring", "drift detection", "retraining cadence", "rollout",
    "feature pipeline", "served", "inference",
)


@dataclass
class EvidenceSignals:
    retrieval_evidence_score: float
    eval_framework_evidence_score: float
    production_ml_evidence_score: float
    evidence_snippets: List[str] = field(default_factory=list)


def _term_density_score(text: str, terms: tuple) -> float:
    text_l = text.lower()
    hits = sum(1 for t in terms if t in text_l)
    return round(min(hits / 3.0, 1.0), 3)  # 3+ distinct operational terms = max credit


def _collect_snippets(c: Candidate, terms: tuple, max_snippets: int = 2) -> List[str]:
    snippets = []
    for ch in c.career_history:
        text_l = ch.description.lower()
        if any(t in text_l for t in terms):
            snippets.append(f"{ch.title} @ {ch.company}: " + ch.description[:160])
        if len(snippets) >= max_snippets:
            break
    return snippets


def score_evidence(c: Candidate) -> EvidenceSignals:
    full_text = " ".join(ch.description for ch in c.career_history)

    retrieval_score = _term_density_score(full_text, RETRIEVAL_OPERATIONAL_TERMS)
    eval_score = _term_density_score(full_text, EVAL_OPERATIONAL_TERMS)
    prod_ml_score = _term_density_score(full_text, PRODUCTION_ML_OPERATIONAL_TERMS)

    # Same disclaim suppression as skill_scorer.py: a candidate whose own
    # narrative explicitly disclaims ownership of the core technical work
    # (e.g. "integration and observability, not the model itself", "built
    # by another team") shouldn't get full evidence credit just because
    # generic operational terms like "A/B test" appear nearby in an
    # unrelated context (e.g. marketing-analytics experimentation, not
    # ranking-system evaluation).
    full_text_l = full_text.lower()
    if any(p in full_text_l for p in DISCLAIM_PATTERNS):
        retrieval_score *= 0.3
        eval_score *= 0.3
        prod_ml_score *= 0.5  # production-ops language is less likely to be the disclaimed part

    snippets = (
        _collect_snippets(c, RETRIEVAL_OPERATIONAL_TERMS, 1)
        + _collect_snippets(c, EVAL_OPERATIONAL_TERMS, 1)
    )

    return EvidenceSignals(
        retrieval_evidence_score=round(retrieval_score, 3),
        eval_framework_evidence_score=round(eval_score, 3),
        production_ml_evidence_score=round(prod_ml_score, 3),
        evidence_snippets=snippets,
    )