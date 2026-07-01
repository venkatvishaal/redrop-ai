#!/usr/bin/env python3
"""
rank.py

Single entry point for the Redrop AI V8 candidate ranking pipeline.
Produces a submission.csv from candidates.jsonl.

Key improvements in V8:
  - Default semantic backend is sentence-transformers (falls back to TF-IDF)
  - TF-IDF + semantic career-text matching in skill_scorer (three-layer pipeline)
  - Configurable reranker constants with evidence-based defaults
  - Calibration module for learning optimal weights from labeled data
  - Education bonus properly scaled to 0.03 max (meaningful tiebreaker)
  - Cleaned up dead config (weights.json removed)
  - Enhanced JD auto-parser with deeper constraint understanding
  - Comprehensively readable evaluate.py and feedback.py
  - 40+ unit tests covering all modules

Constraints satisfied by design:
  - CPU only, no GPU calls anywhere in this file or pipeline/*
  - No network calls during ranking (sentence-transformers uses cached model;
    TF-IDF fallback has zero network dependency)
  - Designed to finish well inside 5 minutes / 16GB for 100K candidates

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv \\
        --job-profile ./data/job_profile.latest.json \\
        --detailed-out ./detailed_results.json \\
        --embedder sentence-transformers

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv \\
        --calibrate --label-file ./recruiter_labels.csv \\
        --calibrate-metric ndcg --calibrate-k 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline.candidate_loader import load_candidates
from pipeline.embedder import SemanticScorer
from pipeline.jd_filter import apply_hard_exclusions
from pipeline.skill_scorer import score_skills
from pipeline.experience_scorer import score_experience
from pipeline.evidence_scorer import score_evidence
from pipeline.signal_scorer import score_signals
from pipeline.location_scorer import score_location
from pipeline.sanity_checks import score_sanity
from pipeline.honeypot_detector import detect_honeypot
from pipeline.rule_engine import apply_rules
from pipeline.composite_ranker import build_ranked_candidate
from pipeline.output_writer import write_submission_csv, write_detailed_results_json
from pipeline.jd_parser import compile_job_profile
from pipeline.reranker import rerank_top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrop AI V8 candidate ranking pipeline"
    )
    p.add_argument("--candidates", required=True,
                   help="Path to candidates.jsonl (or .jsonl.gz)")
    p.add_argument("--out", required=True,
                   help="Output path for submission.csv")
    p.add_argument("--job-profile",
                   default=str(Path(__file__).parent / "data" / "job_profile.latest.json"))
    p.add_argument("--jd", default=None,
                   help="Raw .txt job description; automatically compiled over the base profile")
    p.add_argument("--detailed-out", default=None,
                   help="Optional path for detailed_results.json")
    p.add_argument("--embedder", default="sentence-transformers",
                   choices=["tfidf", "sentence-transformers"],
                   help="Semantic backend: sentence-transformers (default, requires cached model) or tfidf")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--rerank-depth", type=int, default=1000)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--ablate",
                   choices=["semantic", "core", "experience", "behavioral", "logistics"],
                   default=None,
                   help="Zero one signal family and renormalize weights for controlled ablation")

    # Calibration options
    p.add_argument("--calibrate", action="store_true",
                   help="Run weight calibration after ranking (requires --label-file)")
    p.add_argument("--label-file", default=None,
                   help="CSV with recruiter labels (candidate_id,relevance) for calibration")
    p.add_argument("--calibrate-metric", default="ndcg", choices=["ndcg", "map"],
                   help="Metric to optimize during calibration (default: ndcg)")
    p.add_argument("--calibrate-k", type=int, default=100,
                   help="Depth for calibration metric (default: 100)")
    p.add_argument("--reranker-config", default=None,
                   help="JSON file with reranker config overrides")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Validate input
    candidate_path = Path(args.candidates)
    if not candidate_path.is_file():
        raise SystemExit(
            f"Candidate dataset not found: {candidate_path.resolve()}\n"
            f"Provide --candidates PATH; private challenge data is not bundled."
        )
    if candidate_path.stat().st_size == 0:
        raise SystemExit(
            f"Candidate dataset is empty: {candidate_path.resolve()}"
        )

    # Load job profile
    with open(args.job_profile, "r", encoding="utf-8") as f:
        job_profile = json.load(f)
    if args.jd:
        job_profile = compile_job_profile(
            Path(args.jd).read_text(encoding="utf-8"), job_profile
        )

    # Handle ablation
    if args.ablate:
        ablate_key = {
            "semantic": "semantic_fit",
            "core": "core_fit",
            "experience": "experience_fit",
            "behavioral": "behavioral_fit",
            "logistics": "location_logistics_fit",
        }[args.ablate]
        job_profile["weights"][ablate_key] = 0.0
        total = sum(job_profile["weights"].values())
        job_profile["weights"] = {k: v / total for k, v in job_profile["weights"].items()}

    # Load candidates
    print(f"[rank.py] Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"[rank.py] Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s", file=sys.stderr)

    # Hard exclusions
    print("[rank.py] Applying hard exclusions (jd_filter) ...", file=sys.stderr)
    survivors, excluded = apply_hard_exclusions(candidates)
    print(f"[rank.py] {len(survivors)} survive hard filter, {len(excluded)} excluded.", file=sys.stderr)

    # Semantic similarity
    print(f"[rank.py] Computing semantic similarity (backend={args.embedder}) ...", file=sys.stderr)
    scorer = SemanticScorer(
        job_profile["jd_text_for_embedding"],
        backend=args.embedder,
    )
    texts = [c.full_text_blob for c in survivors]
    semantic_scores = scorer.fit_transform_corpus(texts)
    print(f"[rank.py] Semantic scoring done at {time.time() - t0:.1f}s", file=sys.stderr)

    # Main scoring loop
    ranked = []
    for c, sem_score in zip(survivors, semantic_scores):
        skill_result = score_skills(c, job_profile)
        experience = score_experience(c, job_profile)
        evidence = score_evidence(c)
        behavioral = score_signals(c.redrob_signals)
        location = score_location(c, c.redrob_signals, job_profile)
        sanity = score_sanity(c)
        honeypot = detect_honeypot(c, sanity)
        rule_result = apply_rules(experience, skill_result, honeypot.is_honeypot, job_profile)

        rc = build_ranked_candidate(
            c=c,
            semantic_score=float(sem_score),
            skill_result=skill_result,
            experience=experience,
            evidence=evidence,
            behavioral=behavioral,
            location=location,
            sanity=sanity,
            honeypot=honeypot,
            rule_result=rule_result,
            job_profile=job_profile,
        )
        ranked.append(rc)

    print(f"[rank.py] Scored {len(ranked)} candidates at {time.time() - t0:.1f}s", file=sys.stderr)

    # Rerank
    if not args.no_rerank:
        reranker_config = None
        if args.reranker_config:
            with open(args.reranker_config) as f:
                reranker_config = json.load(f)
        rerank_top(ranked, depth=args.rerank_depth, config=reranker_config)

    # Summary
    n_excluded_total = sum(1 for rc in ranked if rc.hard_excluded) + len(excluded)
    print(f"[rank.py] Hard-excluded total: {n_excluded_total} (jd_filter + honeypot/rule_engine)", file=sys.stderr)

    # Write output
    write_submission_csv(ranked, args.out, top_n=args.top_n)
    print(f"[rank.py] Wrote {args.out}", file=sys.stderr)

    if args.detailed_out:
        write_detailed_results_json(ranked, args.detailed_out)
        print(f"[rank.py] Wrote {args.detailed_out}", file=sys.stderr)

    # Calibration
    if args.calibrate:
        if not args.label_file:
            print("[rank.py] ERROR: --calibrate requires --label-file", file=sys.stderr)
            sys.exit(1)
        _run_calibration(ranked, candidates, args)

    elapsed = time.time() - t0
    print(f"[rank.py] Total runtime: {elapsed:.1f}s", file=sys.stderr)
    if elapsed > 300:
        print("[rank.py] WARNING: exceeded 5-minute budget!", file=sys.stderr)


def _run_calibration(
    ranked: list,
    candidates: list,
    args: argparse.Namespace,
) -> None:
    """Run weight calibration using labeled feedback data."""
    from pipeline.calibrator import Calibrator
    import csv

    print("[rank.py] Running weight calibration ...", file=sys.stderr)

    labels: dict[str, float] = {}
    with open(args.label_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["candidate_id"]] = float(row["relevance"])

    print(f"[rank.py] Loaded {len(labels)} labels for calibration", file=sys.stderr)

    calibrator = Calibrator()
    calibrator.load_data(candidates, labels)

    best_weights = calibrator.grid_search(
        metric=args.calibrate_metric,
        k=args.calibrate_k,
        verbose=True,
    )

    # Save calibrated weights
    cal_path = Path(args.out).parent / "calibrated_weights.json"
    Calibrator.save_weights(best_weights, str(cal_path))


if __name__ == "__main__":
    main()
