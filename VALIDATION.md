# V8 Validation

- Full pipeline: 70+ automated tests across all 15+ modules
- Semantic backend: sentence-transformers (default) with TF-IDF fallback
- Three-layer skill matching: keyword, plain-language, **TF-IDF semantic similarity**
- Education bonus: correctly scaled to 0.03 max (tiebreaker, not noise)
- Institution-tier invariance: verified by counterfactual test
- Reranker: configurable constants with evidence-based defaults
- Calibration: grid search over weight space supported
- JD auto-parser: enhanced with trap detection, soft constraints, contradictions
- Dead config removed: `weights.json` eliminated
- Code quality: evaluate.py and feedback.py rewritten for readability
- Fairness: tier/grade have zero weight; verified by counterfactual test

## What V8 cannot claim (honest limits)

- **Accuracy**: Unvalidated without recruiter-provided relevance labels
- **Cross-job calibration**: Each JD produces a different profile; weights may need re-calibration per job
- **Adverse impact**: Requires lawful demographic labels the system does not collect

## Reproducibility

```powershell
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- CPU only, no GPU
- No network calls during ranking (model must be cached or TF-IDF used)
- Designed to finish inside 5 minutes / 16GB for 100K candidates
