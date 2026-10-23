"""
engine.py - Semantic Invariance Engine (SIE): pair -> invariance feature vector.

For each sentence S the engine computes
    pi(S)  : canonical (AGENT, PREDICATE, PATIENT) frame       (syntax module)
    X(S)   : role-aware token cloud on the Poincare ball        (hyperbolic module)
    p(S)   : role-salience mass (frame tokens 1.0, others 0.15)  (length module)
and for a pair (S1, S2) the features
    canon_cos     cosine of TF-IDF over canonical content words
    agent/verb/patient_match, role_cross   frame agreement
    fgw           Fused Gromov-Wasserstein distance between token clouds
    hyp           hyperbolic geodesic distance between weighted centroids
    tda           L2 distance between H0 persistence landscapes
    len_ratio     |log(n1/n2)| (lets the classifier *discount* length)
A logistic-regression head turns these features into P(paraphrase).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .preprocess import Pipeline
from .geometry import (token_matrix, expmap0, poincare_dist, hyperbolic_centroid,
                       fused_gromov_wasserstein, h0_persistence, landscape_distance)

FEATURE_GROUPS = {
    "lexsem": ["canon_cos"],
    "frame":  ["agent_match", "verb_match", "patient_match", "role_cross"],
    "fgw":    ["fgw"],
    "hyp":    ["hyp"],
    "tda":    ["tda"],
    "len":    ["len_ratio"],
}


class SemanticInvarianceEngine:
    def __init__(self, phonetic=True, lexical=True, syntax=True,
                 groups=tuple(FEATURE_GROUPS), alpha=0.5, other_mass=0.15):
        self.pipe = Pipeline(phonetic, lexical, syntax)
        self.groups, self.alpha, self.other_mass = groups, alpha, other_mass
        self.features = [f for g in groups for f in FEATURE_GROUPS[g]]
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self._cache = {}

    # ---------------- sentence-level representation -------------------------
    def represent(self, text):
        if text in self._cache:
            return self._cache[text]
        r = self.pipe.process(text)
        content, roles = r["content"] or ["<empty>"], r["roles"] or ["OTHER"]
        X = expmap0(token_matrix(content, roles))
        w = np.array([self.other_mass if ro == "OTHER" else 1.0 for ro in roles])
        p = w / w.sum()
        n = len(content)
        D = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / max(n - 1, 1)
        r.update(X=X, p=p, D=D, centroid=hyperbolic_centroid(X, p),
                 h0=h0_persistence(poincare_dist(X, X)))
        self._cache[text] = r
        return r

    # ---------------- pair features -----------------------------------------
    def pair_features(self, s1, s2):
        a, b = self.represent(s1), self.represent(s2)
        fa, fb = a["frame"], b["frame"]
        f = {}
        if "lexsem" in self.groups:
            va, vb = self.tfidf.transform([a["canon_text"], b["canon_text"]]).toarray()
            f["canon_cos"] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        if "frame" in self.groups:
            eq = lambda x, y: float(x is not None and x == y)
            f["agent_match"] = eq(fa["agent"], fb["agent"])
            f["verb_match"] = eq(fa["verb"], fb["verb"])
            f["patient_match"] = eq(fa["patient"], fb["patient"])
            f["role_cross"] = eq(fa["agent"], fb["patient"]) * eq(fa["patient"], fb["agent"])
        if "fgw" in self.groups:
            M = poincare_dist(a["X"], b["X"])
            f["fgw"], _ = fused_gromov_wasserstein(M, a["D"], b["D"], a["p"], b["p"], self.alpha)
        if "hyp" in self.groups:
            f["hyp"] = float(poincare_dist(a["centroid"], b["centroid"])[0, 0])
        if "tda" in self.groups:
            f["tda"] = landscape_distance(a["h0"], b["h0"])
        if "len" in self.groups:
            f["len_ratio"] = abs(np.log(len(a["tokens"]) / max(len(b["tokens"]), 1)))
        return [f[k] for k in self.features]

    def featurise(self, pairs):
        return np.array([self.pair_features(s1, s2) for s1, s2 in pairs])

    # ---------------- training / inference ----------------------------------
    def fit(self, pairs, labels, corpus):
        self.tfidf = TfidfVectorizer().fit(self.represent(s)["canon_text"] for s in corpus)
        self.clf.fit(self.featurise(pairs), labels)
        return self

    def predict_proba(self, pairs):
        return self.clf.predict_proba(self.featurise(pairs))[:, 1]

    def predict(self, pairs):
        return (self.predict_proba(pairs) >= 0.5).astype(int)

    def explain(self, s1, s2):
        """Human-readable breakdown for the demo."""
        a, b = self.represent(s1), self.represent(s2)
        feats = dict(zip(self.features, self.pair_features(s1, s2)))
        prob = float(self.predict_proba([(s1, s2)])[0])
        return {"S1_normalised": " ".join(a["tokens"]), "S2_normalised": " ".join(b["tokens"]),
                "S1_frame": a["frame"], "S2_frame": b["frame"],
                "features": {k: round(v, 4) for k, v in feats.items()},
                "P(same meaning)": round(prob, 4)}
