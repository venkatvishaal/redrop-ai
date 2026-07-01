"""
skill_scorer.py

Matches a candidate against the JD's must_have / nice_to_have capability
list. This is the module most directly responsible for not falling into
the hackathon's central trap: "find candidates whose skills section
contains the most AI keywords."

Design principle: a skill listed in the `skills` array is WEAK evidence on
its own. It becomes credible evidence only when corroborated by:
  (a) endorsements + duration_months on the skill entry itself,
  (b) the free-text career_history actually describing related work, and
  (c) semantic similarity between the capability description and the
      candidate's career text.

Three-layer matching pipeline:
  1.  Keyword-based: literal evidence_keyword matching with trust-weighting
  2.  Plain-language: curated patterns for candidates who describe the
      work without tech-stack jargon
  3.  Semantic: TF-IDF cosine similarity between capability description
      text and career-history text chunks, capturing conceptual overlap
      that literal keyword matching misses (e.g. a candidate describing
      "two-stage retrieval pipeline with bi-encoder retrieval and
      cross-encoder reranking" who never mentions NDCG/MRR but clearly
      knows evaluation frameworks)

A candidate who lists "RAG, LoRA, Fine-tuning LLMs" with 0 endorsements,
0-duration, and a career_history that's all "Marketing Manager... grew
Instagram engagement" gets near-zero credit here, by design. A candidate
whose skills list doesn't even use the word "RAG" but whose career_history
describes building "a system that retrieves and ranks the most relevant
matches for user intent" gets credit via the production_evidence check
even without an exact keyword hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.candidate_loader import Candidate

# Adjacency map: if a candidate shows strong evidence of the KEY concept,
# credit it partially toward the VALUE capability too. Kept small and
# explicit rather than a black-box similarity.
TRANSFERABLE_SKILL_MAP = {
    "llms": ["embeddings_retrieval_production"],
    "recommendation systems": ["embeddings_retrieval_production", "vector_db_hybrid_search"],
    "search & discovery": ["embeddings_retrieval_production", "vector_db_hybrid_search"],
    "ranking systems": ["embeddings_retrieval_production", "eval_frameworks_ranking"],
    "personalization": ["embeddings_retrieval_production"],
    "information retrieval": ["embeddings_retrieval_production", "vector_db_hybrid_search"],
}

# Plain-language pattern phrases that describe the SAME underlying work as
# a capability's literal evidence_keywords, without using the jargon.
# This exists specifically because the JD warns: "A Tier-5 candidate may
# not use the words 'RAG' or 'Pinecone' in their profile, but if their
# career history shows they built a recommendation system at a product
# company, they're a fit."
PLAIN_LANGUAGE_PATTERNS = {
    "embeddings_retrieval_production": [
        "relevant matches", "connects users to", "most relevant", "recommendation",
        "recommend", "personalization", "matching layer", "collaborative filtering",
        "re-ranking", "reranking", "relevance",
    ],
    "vector_db_hybrid_search": [
        "search and discovery", "search & discovery", "matching layer",
        "relevance", "indexed", "retrieval", "search backend", "search infrastructure",
    ],
}

# Negative-context guard: phrases like "search backend" or "retrieval" are
# ambiguous - they describe vector/embedding-based search just as easily
# as plain SQL/keyword search. If any of these appear in the same
# career_history entry as a plain-language match, the match is rejected.
NON_VECTOR_SEARCH_DISQUALIFIERS = (
    "sql full-text", "full-text search", "keyword matching", "keyword search",
    "regex search", "exact-match search", "boolean search", "string matching",
    "database query", "sql query", "inverted index",
)

# Disclaim/hedge phrases that explicitly walk back ownership of the
# technical work, even when a skill is listed with real-looking
# endorsements/duration. Found via direct inspection of the real dataset:
# ~1500+ occurrences of template phrases describing candidates who touched
# an AI/ML-adjacent system peripherally.
DISCLAIM_PATTERNS = (
    "not the model itself", "built by another team", "wouldn't call myself an ml specialist",
    "not an ml specialist", "integration and observability, not", "not the core",
    "someone else's model", "not my core expertise", "peripheral involvement",
)

# Semantic matching threshold: cosine similarity above which we consider
# the career text semantically related to the capability description.
SEMANTIC_SIMILARITY_THRESHOLD = 0.15


@dataclass
class CapabilityScore:
    capability_id: str
    matched: bool
    confidence: float  # 0-1, how much we trust the match
    via: str  # "skill_entry" | "career_text" | "plain_language" | "semantic" | "transferable" | "none"


def _skill_trust_weight(endorsements: int, duration_months: int) -> float:
    """A skill with no endorsements and no duration is essentially a
    self-reported keyword. A skill with real endorsements and tenure is
    much more credible. Scaled to [0.15, 1.0] so a bare keyword still
    counts for *something* (candidates legitimately under-fill profiles)
    but never dominates the score.
    """
    if endorsements <= 0 and duration_months <= 0:
        return 0.15
    endorsement_factor = min(endorsements / 20.0, 1.0)
    duration_factor = min(duration_months / 24.0, 1.0)
    return 0.15 + 0.85 * max(endorsement_factor, duration_factor)


def _career_text_mentions(text: str, keywords: List[str]) -> bool:
    text_l = text.lower()
    return any(kw.lower() in text_l for kw in keywords)


def _has_production_context_for_keywords(c: Candidate, keywords: List[str]) -> bool:
    """True if the career_history (not skills list) actually describes
    doing this kind of work, ideally with a production-shaped verb nearby."""
    production_verbs = ("built", "shipped", "deployed", "designed", "owned",
                        "implemented", "scaled", "operated", "maintained")
    for ch in c.career_history:
        text_l = ch.description.lower()
        if any(kw.lower() in text_l for kw in keywords):
            if any(v in text_l for v in production_verbs):
                return True
    return False


# ---------------------------------------------------------------------------
# Semantic similarity matching (third layer)
# ---------------------------------------------------------------------------

# Lazy-initialized TF-IDF vectorizer for capability-vs-career-text matching.
# Shared across all calls to avoid re-fitting the same vocabulary.
_semantic_vectorizer: TfidfVectorizer | None = None


def _compute_semantic_similarity(
    capability_text: str,
    career_text: str,
) -> float:
    """Compute cosine similarity between a capability description and
    career-history text using TF-IDF vectors.

    This catches cases where the candidate describes work that is
    conceptually related to a capability without using any of the
    literal evidence keywords. For example, a candidate who describes
    "building a bi-encoder/cross-encoder retrieval pipeline with
    recall@k evaluation" may not mention "NDCG" or "MRR" but is
    clearly doing evaluation-framework work.

    Returns a score in [0, 1].
    """
    global _semantic_vectorizer
    if not capability_text.strip() or not career_text.strip():
        return 0.0

    if _semantic_vectorizer is None:
        _semantic_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )

    try:
        tfidf = _semantic_vectorizer.fit_transform([capability_text, career_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
        return max(0.0, float(sim))
    except ValueError:
        return 0.0


# Cache for semantic scores: (capability_id, career_text_hash) -> score
_semantic_cache: dict[tuple[str, int], float] = {}


def _get_semantic_match_score(capability: dict, career_text: str) -> float:
    """Get the semantic similarity score between a capability description
    and the candidate's career text. Uses caching to avoid recomputation.
    """
    cap_id = capability["id"]
    # Build a descriptive text for the capability from its evidence_keywords
    # and description field for richer semantic matching
    desc_parts = [capability.get("description", "")]
    desc_parts.extend(capability.get("evidence_keywords", []))
    # Add capability name as well
    desc_parts.append(cap_id.replace("_", " "))
    capability_desc = " ".join(desc_parts)

    text_hash = hash(career_text[:500])  # hash first 500 chars as key
    cache_key = (cap_id, text_hash)

    if cache_key in _semantic_cache:
        return _semantic_cache[cache_key]

    score = _compute_semantic_similarity(capability_desc, career_text)
    _semantic_cache[cache_key] = score
    return score


def _has_disclaim_in_text(career_text: str) -> bool:
    """Check if career text contains disclaim/hedge patterns."""
    text_l = career_text.lower()
    return any(p in text_l for p in DISCLAIM_PATTERNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_non_vector_disqualifier(career_text: str, cap_id: str) -> bool:
    """Check if a non-vector-search disqualifier invalidates a match
    for vector/embedding capabilities.
    """
    if cap_id not in ("embeddings_retrieval_production", "vector_db_hybrid_search"):
        return False
    return _career_text_mentions(career_text, list(NON_VECTOR_SEARCH_DISQUALIFIERS))


def score_capability(c: Candidate, capability: dict) -> CapabilityScore:
    cap_id = capability["id"]
    keywords = capability["evidence_keywords"]
    requires_prod = capability.get("evidence_requires_production_context", False)

    career_text = " ".join(ch.description for ch in c.career_history)
    career_text_l = career_text.lower()

    # 1. Direct skill-entry match (trust-weighted)
    best_skill_conf = 0.0
    for s in c.skills:
        if any(kw.lower() in s.name.lower() or s.name.lower() in kw.lower() for kw in keywords):
            trust = _skill_trust_weight(s.endorsements, s.duration_months)
            best_skill_conf = max(best_skill_conf, trust)

    # 2. Career-history textual evidence (keyword-based)
    text_match = _career_text_mentions(career_text, keywords)
    has_non_vector_disq = _check_non_vector_disqualifier(career_text, cap_id)
    if text_match and has_non_vector_disq:
        text_match = False

    prod_context = _has_production_context_for_keywords(c, keywords) if text_match else False

    if requires_prod:
        if prod_context:
            text_conf = 0.95
        elif text_match:
            text_conf = 0.55
        else:
            text_conf = 0.0
    else:
        text_conf = 0.85 if text_match else 0.0

    # 2b. Plain-language pattern match
    plain_patterns = PLAIN_LANGUAGE_PATTERNS.get(cap_id, [])
    plain_match = _career_text_mentions(career_text, plain_patterns) if plain_patterns else False
    if plain_match and _check_non_vector_disqualifier(career_text, cap_id):
        plain_match = False

    plain_prod_context = (
        _has_production_context_for_keywords(c, plain_patterns) if plain_match else False
    )
    plain_conf = 0.75 if (plain_match and plain_prod_context) else (0.4 if plain_match else 0.0)

    # 3. Transferable / adjacent skill credit
    transferable_conf = 0.0
    for skill_name in c.skill_names_lower:
        targets = TRANSFERABLE_SKILL_MAP.get(skill_name, [])
        if cap_id in targets:
            transferable_conf = max(transferable_conf, 0.5)

    # 4. Semantic similarity match (NEW - third layer)
    # Catches candidates who describe conceptually related work without
    # using any literal keywords or plain-language patterns.
    semantic_score = _get_semantic_match_score(capability, career_text)
    if semantic_score >= SEMANTIC_SIMILARITY_THRESHOLD:
        # Scale semantic score: threshold 0.15 -> 0.0, 0.5 -> 1.0
        semantic_conf = min(1.0, (semantic_score - SEMANTIC_SIMILARITY_THRESHOLD) / 0.35)
        # If requires production context, discount semantic-only matches
        if requires_prod:
            # Check if there's any production verb in career text
            has_prod_verb = _has_production_context_for_keywords(c, capability.get("evidence_keywords", []))
            if not has_prod_verb:
                # Check against the capability description words broadly
                desc_words = capability.get("description", "").split()
                has_prod_verb = _has_production_context_for_keywords(c, desc_words) if desc_words else False
            if not has_prod_verb:
                semantic_conf *= 0.5  # discount semantic-only without production context
    else:
        semantic_conf = 0.0

    # Find best confidence across all layers
    confidence = max(best_skill_conf, text_conf, plain_conf, transferable_conf, semantic_conf)
    via = "none"
    if confidence > 0:
        if confidence == best_skill_conf and best_skill_conf > 0:
            via = "skill_entry"
        elif confidence == text_conf and text_conf > 0:
            via = "career_text"
        elif confidence == plain_conf and plain_conf > 0:
            via = "plain_language"
        elif confidence == semantic_conf and semantic_conf > 0:
            via = "semantic"
        elif confidence == transferable_conf and transferable_conf > 0:
            via = "transferable"

    # Disclaim override: if candidate's own career text explicitly disclaims
    # ownership of the core technical work, heavily discount regardless
    # of which path produced the confidence.
    if _has_disclaim_in_text(career_text):
        confidence *= 0.3
        if confidence <= 0.3:
            via = "disclaimed"

    return CapabilityScore(
        capability_id=cap_id,
        matched=confidence > 0.3,
        confidence=round(confidence, 3),
        via=via,
    )


def score_skills(c: Candidate, job_profile: dict) -> Dict[str, object]:
    must_have = job_profile["must_have_capabilities"]
    nice_to_have = job_profile["nice_to_have_capabilities"]

    must_scores = [score_capability(c, cap) for cap in must_have]
    nice_scores = [score_capability(c, cap) for cap in nice_to_have]

    must_have_fraction = (
        sum(s.confidence for s in must_scores) / len(must_scores) if must_scores else 0.0
    )
    nice_to_have_fraction = (
        sum(s.confidence for s in nice_scores) / len(nice_scores) if nice_scores else 0.0
    )

    # Detect "framework enthusiast" keyword-stuffing pattern
    trendy_keywords = [
        "langchain", "lora", "qlora", "peft", "gan", "rag", "prompt engineering",
        "fine-tuning llms", "diffusion",
    ]
    high_proficiency_trendy_hits = sum(
        1 for s in c.skills
        if s.proficiency in ("advanced", "expert") and any(k in s.name.lower() for k in trendy_keywords)
    )
    career_text_l = " ".join(ch.description for ch in c.career_history).lower()
    trendy_corroborated = sum(1 for k in trendy_keywords if k in career_text_l)
    framework_enthusiast_flag = high_proficiency_trendy_hits >= 4 and trendy_corroborated == 0

    skill_score = 0.7 * must_have_fraction + 0.3 * nice_to_have_fraction

    return {
        "skill_score": round(skill_score, 4),
        "must_have_fraction": round(must_have_fraction, 4),
        "nice_to_have_fraction": round(nice_to_have_fraction, 4),
        "must_have_details": must_scores,
        "nice_to_have_details": nice_scores,
        "framework_enthusiast_flag": framework_enthusiast_flag,
        "trendy_skill_hits": high_proficiency_trendy_hits,
    }
