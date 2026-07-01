# Redrop AI V8 — The Ultimate AI Recruiter

Offline, deterministic two-stage candidate ranking with multi-layer semantic understanding, anti-gaming defenses, and data-driven weight calibration.

## What makes V8 different from V6

| Gap Identified | V6 | V8 Fix |
|---|---|---|
| Semantic layer | TF-IDF only | **sentence-transformers** default, TF-IDF fallback |
| Career-text matching | Keyword/substring only | **3-layer pipeline**: keyword + plain-language + **TF-IDF semantic similarity** |
| Education scaling | 0.01 max (essentially noise) | **0.03 max** — meaningful tiebreaker |
| Reranker constants | Hardcoded 0.012/0.008 | **Configurable** via RERANKER_CONFIG, capped at 0.025 |
| Weights validation | Unvalidated guesses | **Calibration module** — grid-search optimization against labels |
| Evaluate.py | Obfuscated single-letter vars | **Readable** code with docstrings |
| Feedback.py | Obfuscated | **Readable** with validation |
| JD auto-parser | Basic regex extraction | **Enhanced** — trap detection, soft constraint awareness, contradiction detection |
| Dead config | `weights.json` duplicated profile | **Removed** |
| Test coverage | 26 tests | **70+ tests** — every module tested |
| Institution-tier code | Dead code all zeros | **Cleaned** with explicit fairness documentation |

## Architecture

```
rank.py (entry point)
├── jd_filter.py          — Hard disqualifiers (pure research, etc.)
├── embedder.py           — Sentence-transformers / TF-IDF semantic similarity
├── skill_scorer.py       — 3-layer capability matching (keyword + plain + semantic)
├── experience_scorer.py  — Years, trajectory, title-chaser detection
├── evidence_scorer.py    — Operational ML evidence density
├── signal_scorer.py      — Behavioral: recency, engagement, reliability
├── location_scorer.py    — City/country, notice period, work mode
├── sanity_checks.py      — Internal consistency verification
├── honeypot_detector.py  — Impossible-profile detection
├── rule_engine.py        — Declarative penalties from config
├── composite_ranker.py   — Weighted scoring + reasoning generation
├── reranker.py           — Evidence-consensus second-stage refinement
├── calibrator.py         — Weight learning from labeled data (NEW)
├── output_writer.py      — CSV + detailed JSON output
└── jd_parser.py          — JD auto-compilation with insight extraction
```

## Quick Start

```powershell
python -m pip install -r requirements.txt
python rank.py --candidates PATH\candidates.jsonl --out submission.csv
python -m pytest tests/ -q
```

The `sentence-transformers` backend is the default but requires the model to be cached locally. If it's unavailable, TF-IDF is used automatically. To pre-cache the model:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Evaluation

```powershell
python evaluate.py --submission submission.csv --labels recruiter_labels.csv
python calibrate.py --submission submission.csv --labels recruiter_labels.csv
```

Metrics: precision@k, recall@k, MAP@k, NDCG@k, Brier score.

## Calibration

Learn optimal weights from labeled recruiter feedback:

```powershell
python rank.py --candidates candidates.jsonl --out submission.csv \
    --calibrate --label-file recruiter_labels.csv \
    --calibrate-metric ndcg --calibrate-k 100
```

## Controlled Ablations

```powershell
python rank.py --candidates candidates.jsonl --out submission.csv --ablate semantic
python rank.py --candidates candidates.jsonl --out submission.csv --no-rerank
```

## Fairness

- Institution tier has zero ranking weight (verified by counterfactual invariance test)
- Education contributes at most 0.03 for job-relevant field/degree only
- Sensitive attributes are not loaded
- Outcome-based subgroup validation requires lawful, consented demographic labels

## Honest Limits

V8 extracts explicit JD constraints and several nuanced tensions; its capability ontology is reviewable configuration. Measured hiring accuracy, cross-job calibration, and adverse-impact outcomes require real recruiter and demographic labels and are never fabricated.

All 70+ tests pass, but accuracy on specific datasets is unvalidated without labeled recruiter feedback.
