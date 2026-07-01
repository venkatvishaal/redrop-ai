#!/usr/bin/env python3
"""
evaluate.py

Evaluates a submission (ranked candidate list) against recruiter-provided
relevance labels. Reports standard IR metrics: precision, recall, MAP,
NDCG, and Brier score for calibration quality.

Usage:
    python evaluate.py --submission submission.csv --labels recruiter_labels.csv
    python evaluate.py --submission submission.csv --labels recruiter_labels.csv --k 50

Label format (CSV):
    candidate_id,relevance
    CAND_0001000,3
    CAND_0002000,1

Relevance scale (0-3):
    0 = not relevant
    1 = somewhat relevant
    2 = relevant
    3 = highly relevant
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ranked submission against recruiter relevance labels."
    )
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument("--labels", required=True, help="Path to labels CSV (candidate_id,relevance)")
    parser.add_argument("--k", type=int, default=100, help="Depth for precision/recall/NDCG (default: 100)")
    return parser.parse_args()


def load_labels(path: str) -> Dict[str, float]:
    """Load relevance labels from CSV. Returns dict mapping candidate_id -> relevance score."""
    labels: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["candidate_id"]] = float(row["relevance"])
    return labels


def load_submission(path: str, k: int) -> Tuple[List[str], List[float]]:
    """Load the top-k ranked candidates from submission CSV.

    Returns:
        candidate_ids: Ordered list of candidate IDs (ranked).
        scores: Corresponding scores from the submission.
    """
    candidate_ids: List[str] = []
    scores: List[float] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= k:
                break
            candidate_ids.append(row["candidate_id"])
            scores.append(float(row["score"]))
    return candidate_ids, scores


def compute_dcg(relevances: List[float]) -> float:
    """Compute Discounted Cumulative Gain at the given depth."""
    dcg_value = 0.0
    for i, rel in enumerate(relevances):
        # rank position i+1 (1-indexed)
        dcg_value += (2 ** rel - 1) / math.log2(i + 2)
    return dcg_value


def compute_metrics(
    ranked_ids: List[str],
    ranked_scores: List[float],
    labels: Dict[str, float],
    k: int,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a ranked list against labels.

    Metrics computed:
        - precision@k: fraction of top-k that are relevant (relevance > 0)
        - recall@k: fraction of all relevant candidates found in top-k
        - MAP@k: Mean Average Precision
        - NDCG@k: Normalized Discounted Cumulative Gain
        - Brier score: mean squared error between scores and binary relevance

    Returns:
        dict of metric_name -> value
    """
    # Relevance values for top-k (default to 0 if not labeled)
    relevances = [labels.get(cid, 0.0) for cid in ranked_ids[:k]]
    # Binary relevance: relevance > 0 means relevant
    binary_relevance = [1.0 if r > 0 else 0.0 for r in relevances]

    # Total relevant in the full label set
    all_relevant = sum(1 for r in labels.values() if r > 0)

    # Number of labeled candidates found in top-k
    labeled_in_top_k = sum(1 for cid in ranked_ids[:k] if cid in labels)

    # --- Precision@k ---
    precision = sum(binary_relevance) / max(len(binary_relevance), 1)

    # --- Recall@k ---
    recall = sum(binary_relevance) / max(all_relevant, 1)

    # --- MAP@k ---
    running_precision = 0.0
    num_relevant_found = 0
    for i, is_relevant in enumerate(binary_relevance):
        if is_relevant:
            num_relevant_found += 1
            running_precision += num_relevant_found / (i + 1)
    map_score = running_precision / max(all_relevant, 1)

    # --- NDCG@k ---
    ideal_relevances = sorted(labels.values(), reverse=True)[:k]
    dcg_value = compute_dcg(relevances)
    ideal_dcg = compute_dcg(ideal_relevances)
    ndcg = dcg_value / max(ideal_dcg, 1e-10)

    # --- Brier score ---
    # Compares predicted probability (score) against binary relevance
    targets = [1.0 if r > 0 else 0.0 for r in relevances]
    brier_score = sum((p - t) ** 2 for p, t in zip(ranked_scores[:k], targets)) / max(len(ranked_scores[:k]), 1)

    return {
        "evaluated_depth": k,
        "labeled_in_top_k": labeled_in_top_k,
        "total_relevant_in_labels": all_relevant,
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        f"map@{k}": round(map_score, 4),
        f"ndcg@{k}": round(ndcg, 4),
        "brier_score": round(brier_score, 4),
    }


def main() -> None:
    args = parse_args()

    labels = load_labels(args.labels)
    print(f"[evaluate] Loaded {len(labels)} labels from {args.labels}")

    ranked_ids, ranked_scores = load_submission(args.submission, args.k)
    print(f"[evaluate] Loaded top-{len(ranked_ids)} from {args.submission}")

    if len(ranked_ids) < args.k:
        print(f"[evaluate] WARNING: submission has only {len(ranked_ids)} candidates, requested k={args.k}")

    metrics = compute_metrics(ranked_ids, ranked_scores, labels, args.k)

    print("\n--- Evaluation Results ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
