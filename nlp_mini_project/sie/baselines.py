"""
baselines.py - Surface-similarity baselines the engine is compared against.

  * TF-IDF (word)        : classic bag-of-words cosine     (preprocessed: lowercase,
                           punctuation removed, stop-words removed)
  * TF-IDF (char 2-4)    : character n-gram cosine (robust to spelling noise)
  * SBERT (optional)     : all-MiniLM-L6-v2 sentence-transformer cosine, used only
                           if `sentence-transformers` and the model are available.
Each baseline is a 1-D score; a decision threshold is tuned on the training split.
"""
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score


def _cos(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


class CosineBaseline:
    def __init__(self, name):
        self.name, self.threshold = name, 0.5

    def scores(self, pairs):
        raise NotImplementedError

    def fit(self, pairs, labels, corpus):
        s = self.scores(pairs)
        cands = np.unique(np.round(s, 4))
        self.threshold = max(cands, key=lambda t: ((s >= t).astype(int) == labels).mean())
        return self

    def predict(self, pairs):
        return (self.scores(pairs) >= self.threshold).astype(int)


class TfidfBaseline(CosineBaseline):
    def __init__(self, analyzer="word"):
        super().__init__("TF-IDF word" if analyzer == "word" else "TF-IDF char")
        kw = dict(lowercase=True)
        if analyzer == "word":
            kw.update(stop_words=list(ENGLISH_STOP_WORDS))
        else:
            kw.update(analyzer="char_wb", ngram_range=(2, 4))
        self.vec = TfidfVectorizer(**kw)

    def fit(self, pairs, labels, corpus):
        self.vec.fit(corpus)
        return super().fit(pairs, labels, corpus)

    def scores(self, pairs):
        A = self.vec.transform([a for a, _ in pairs]).toarray()
        B = self.vec.transform([b for _, b in pairs]).toarray()
        return _cos(A, B)


class SbertBaseline(CosineBaseline):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__("SBERT MiniLM")
        from sentence_transformers import SentenceTransformer   # may raise
        self.model = SentenceTransformer(model_name)
        self._cache = {}

    def _emb(self, texts):
        new = [t for t in set(texts) if t not in self._cache]
        if new:
            for t, e in zip(new, self.model.encode(new, batch_size=64)):
                self._cache[t] = e
        return np.array([self._cache[t] for t in texts])

    def scores(self, pairs):
        return _cos(self._emb([a for a, _ in pairs]), self._emb([b for _, b in pairs]))


def available_baselines():
    bl = [TfidfBaseline("word"), TfidfBaseline("char")]
    if os.environ.get("USE_SBERT", "1") == "1":
        try:
            bl.append(SbertBaseline())
        except Exception as e:                            # offline / not installed
            print(f"[info] SBERT baseline skipped ({type(e).__name__}).")
    return bl
