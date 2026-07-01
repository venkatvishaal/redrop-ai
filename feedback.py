#!/usr/bin/env python3
"""
feedback.py

Append-only auditable recorder of recruiter outcomes (approve, reject,
interview, offer, hire). This is a data-capture tool — it never mutates
model weights or scoring parameters online. Feedback data can later be
used for offline calibration and evaluation.

Usage:
    python feedback.py --job-id JD_001 --candidate-id CAND_0001000 \\
        --decision approve --reason "Great experience fit"

    python feedback.py --job-id JD_001 --candidate-id CAND_0002000 \\
        --decision reject --reason "No production ML evidence" --stage screening
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import List

# Schema for the feedback CSV file
FEEDBACK_FIELDS = [
    "timestamp",
    "job_id",
    "candidate_id",
    "decision",
    "reason",
    "stage",
]

VALID_DECISIONS = ["approve", "reject", "interview", "offer", "hire"]
VALID_STAGES = ["screening", "shortlist", "interview", "offer", "hire"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record recruiter feedback for a candidate-job pair."
    )
    parser.add_argument(
        "--store",
        default="data/recruiter_feedback.csv",
        help="Path to feedback CSV file (default: data/recruiter_feedback.csv)",
    )
    parser.add_argument("--job-id", required=True, help="Job identifier (e.g., JD_001)")
    parser.add_argument("--candidate-id", required=True, help="Candidate identifier")
    parser.add_argument(
        "--decision",
        choices=VALID_DECISIONS,
        required=True,
        help="Recruiter's decision on this candidate",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Optional free-text reason for the decision",
    )
    parser.add_argument(
        "--stage",
        default="shortlist",
        help="Pipeline stage where this decision was made (default: shortlist)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate stage
    if args.stage not in VALID_STAGES:
        print(
            f"[feedback] WARNING: stage '{args.stage}' not in standard stages "
            f"{VALID_STAGES}. Recording anyway.",
        )

    store_path = Path(args.store)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file is new (needs header)
    is_new_file = not store_path.exists()

    with store_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)

        if is_new_file:
            writer.writeheader()
            print(f"[feedback] Created new feedback store at {store_path}")

        # Build the feedback record
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "job_id": args.job_id,
            "candidate_id": args.candidate_id,
            "decision": args.decision,
            "reason": args.reason,
            "stage": args.stage,
        }

        writer.writerow(record)
        print(
            f"[feedback] Recorded: {args.candidate_id} -> {args.decision} "
            f"(stage={args.stage}, job={args.job_id})",
        )


if __name__ == "__main__":
    main()
