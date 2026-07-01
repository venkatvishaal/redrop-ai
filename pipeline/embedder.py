"""
embedder.py

Semantic similarity between the JD and each candidate's free-text profile
(headline + summary + career_history descriptions + education - NOT the
bare skills list, see candidate_loader.full_text_blob).

Default backend: sentence-transformers (all-MiniLM-L6-v2). This is the
preferred backend because it captures true semantic meaning rather than
lexical n-gram overlap. However, it requires the model to be already
cached locally — if the model is unavailable (first run, or no network),
the system gracefully falls back to TF-IDF.

Fallback backend: TF-IDF + cosine similarity (scikit-learn). This:
  - has zero model-download / network dependency,
  - runs on 100K short documents in low single-digit seconds on CPU,
  - is good enough as ONE signal among several.

The backend choice is logged at startup so users can see which backend
is active and pre-cache the sentence-transformers model if desired
using: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
"""

from __future__ import annotations

import sys
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class SemanticScorer:
    def __init__(self, jd_text: str, backend: str = "sentence-transformers"):
        self.jd_text = jd_text
        self.backend = backend
        self._st_model = None

        if backend == "sentence-transformers":
            self._try_load_sentence_transformers()
        else:
            print(f"[embedder] Using backend: {backend}", file=sys.stderr)

    def _try_load_sentence_transformers(self) -> None:
        """Attempt to load sentence-transformers model from local cache.
        Falls back to TF-IDF if the model isn't cached (no download
        attempted during ranking per the offline constraint)."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Quick validation: run a tiny encode to confirm it works
            _ = model.encode(["validation"], normalize_embeddings=True)
            self._st_model = model
            print("[embedder] Using backend: sentence-transformers (all-MiniLM-L6-v2)", file=sys.stderr)
        except Exception as e:
            self._st_model = None
            self.backend = "tfidf"
            print(f"[embedder] sentence-transformers unavailable ({e}); falling back to TF-IDF", file=sys.stderr)
            print("[embedder] To cache the model beforehand, run: "
                  "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"",
                  file=sys.stderr)

    def fit_transform_corpus(self, texts: List[str]) -> np.ndarray:
        """Returns an (N,) array of similarity scores in [0, 1] between the
        JD and each candidate text, using whichever backend is active."""
        if self.backend == "sentence-transformers" and self._st_model is not None:
            return self._score_sentence_transformers(texts)
        return self._score_tfidf(texts)

    def _score_tfidf(self, texts: List[str]) -> np.ndarray:
        corpus = [self.jd_text] + texts
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,  # drop singleton terms - cuts vocab/memory at 100K docs
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(corpus)
        jd_vec = normalize(tfidf[0:1])
        cand_vecs = normalize(tfidf[1:])
        # Manual sparse dot product: both vectors are L2-normalized,
        # so dot product == cosine similarity. Avoids sklearn's dense
        # intermediate allocation patterns which spike memory at 100K docs.
        sims = (cand_vecs @ jd_vec.T).toarray().ravel()
        return np.clip(sims, 0.0, 1.0)

    def _score_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        jd_emb = self._st_model.encode([self.jd_text], normalize_embeddings=True)
        cand_emb = self._st_model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        sims = (cand_emb @ jd_emb.T).ravel()
        return np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
