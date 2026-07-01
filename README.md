---
title: Redrop AI Sandbox
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
python_version: 3.12
app_file: app.py
pinned: false
---

# DropNuilAI — Intelligent Candidate Ranking System

An offline, deterministic two-stage candidate ranking engine featuring multi-layer semantic understanding, robust anti-gaming defenses, and data-driven weight calibration. 

Designed for high-precision, equitable hiring, this system goes beyond keyword matching by evaluating career trajectories, operational ML evidence, and behavioral reliability.

## System Architecture

The system is highly modular, ensuring independent evaluation across multiple candidate dimensions before performing a composite, weighted ranking.

```text
rank.py (entry point)
├── jd_filter.py          — Hard disqualifiers (e.g., pure research background, minimum experience)
├── embedder.py           — Semantic similarity engine (Sentence-transformers / TF-IDF fallback)
├── skill_scorer.py       — 3-layer capability matching (exact keyword + plain language + semantic)
├── experience_scorer.py  — Analyzes years of experience, trajectory, and detects "title-chasers"
├── evidence_scorer.py    — Measures density of operational ML/production evidence
├── signal_scorer.py      — Evaluates behavioral signals (recency, recruiter engagement, reliability)
├── location_scorer.py    — Matches city/country, notice period, and remote/hybrid work mode
├── sanity_checks.py      — Internal consistency verification for candidate profiles
├── honeypot_detector.py  — Anti-gaming defense (detects impossible or stuffed profiles)
├── rule_engine.py        — Applies declarative penalties based on configured business logic
├── composite_ranker.py   — Handles weighted score aggregation and generates human-readable reasoning
├── reranker.py           — Performs evidence-consensus second-stage refinement
├── calibrator.py         — Optimizes scoring weights using grid-search against labeled data
├── output_writer.py      — Generates final output (submission.csv + detailed_results.json)
└── jd_parser.py          — Auto-compiles the Job Description and extracts key requirements/insights
```

## Quick Start

To install dependencies and run a basic candidate ranking:

```powershell
# Install required packages
python -m pip install -r requirements.txt

# Run the ranking pipeline
python rank.py --candidates data/jobs.json --out submission.csv

# Run the test suite (70+ tests covering all modules)
python -m pytest tests/ -q
```

*Note: The `sentence-transformers` backend is the default but requires the model to be cached locally. If it's unavailable, the system automatically falls back to a highly performant TF-IDF approach.*

To pre-cache the semantic model:
```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Evaluation & Calibration

You can evaluate the ranking quality against ground-truth recruiter labels using industry-standard metrics (precision@k, recall@k, MAP@k, NDCG@k, Brier score):

```powershell
python evaluate.py --submission submission.csv --labels recruiter_labels.csv
```

To optimize the scoring weights based on empirical feedback:
```powershell
python rank.py --candidates candidates.jsonl --out submission.csv \
    --calibrate --label-file recruiter_labels.csv \
    --calibrate-metric ndcg --calibrate-k 100
```

## Fairness & Integrity

*   **Institution Agnostic:** Institution tier has zero ranking weight (verified by counterfactual invariance tests) to prevent pedigree bias.
*   **Meritocratic Evaluation:** Sensitive demographic attributes are entirely excluded from the evaluation pipeline. 
*   **Education Scaling:** Educational degrees serve strictly as a minor tiebreaker (capped at 0.03 maximum weight).
*   **Anti-Gaming:** The system actively neutralizes "impossible profiles" and keyword-stuffing via its honeypot detectors and sanity checks.
