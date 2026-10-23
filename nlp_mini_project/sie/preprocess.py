"""
preprocess.py - Text normalisation and the three symbolic invariance modules.

Pipeline (each stage can be switched off for ablation studies):
    1. Cleaning      : lowercase, strip punctuation (commas kept for parsing), tokenise
    2. Phonetic      : slang dictionary + phonetic-key (Metaphone-style) spelling repair
                       -> handles T_phon  ("da docta prazed teh teecha")
    3. Lexical       : synset lookup, every synonym mapped to a canonical lemma
                       -> handles T_lex   ("physician commended" -> "doctor praise")
    4. Syntactic     : rule-based semantic-role labelling producing an
                       (AGENT, PREDICATE, PATIENT) frame - the "base manifold"
                       projection pi(S) of the readme's fiber-bundle view.
                       -> handles T_syn   (passive, cleft, topicalisation)
    5. Stop-words    : removed when building the bag of content words
                       -> together with role-salience weights handles T_len
"""
import re
from .data import ENTITIES, VERBS

STOPWORDS = {
    "the", "a", "an", "it", "is", "was", "were", "be", "been", "being", "by", "that",
    "who", "which", "of", "to", "in", "on", "at", "for", "with", "as", "and", "or",
    "this", "what", "had", "has", "have", "after", "much", "many", "several", "any",
    "anyone", "can", "point", "time", "far", "rather", "everyone", "end", "an",
    "already", "widely", "according", "familiar", "matter", "earlier", "today",
}
PASSIVE_AUX = {"was", "were", "is", "are", "been", "being", "got", "get"}

# Common chat / ASR respellings of function words.
SLANG = {"da": "the", "teh": "the", "tha": "the", "wuz": "was", "woz": "was",
         "bai": "by", "bi": "by", "dat": "that", "u": "you", "r": "are",
         "gonna": "going", "thru": "through", "wanna": "want"}

# A small general-English dictionary: words that are already correctly spelled
# and must NOT be "repaired" toward the domain lexicon.
COMMON_WORDS = set(STOPWORDS) | {
    "sources", "believed", "tell", "people", "expected", "internal", "deliberation",
    "long", "detailed", "discussion", "lasted", "hours", "described", "unusually",
    "busy", "week", "you", "are", "going", "through", "want", "there", "their",
}


# ---------------------------------------------------------------------------
# Lexicon:  surface word -> (type, canonical lemma)
# ---------------------------------------------------------------------------
class Lexicon:
    def __init__(self):
        self.table = {}
        for ent, syns in ENTITIES.items():
            self.table[ent] = ("ENT", ent)
            for s in syns:
                self.table[s] = ("ENT", ent)
        for base, (past, syns) in VERBS.items():
            for form in [base, past] + syns:
                self.table[form] = ("VERB", base)
        self.vocab = list(self.table) + sorted(STOPWORDS)
        self.keys = {w: phonetic_key(w) for w in self.vocab}

    def lookup(self, w, lexical=True):
        if w not in self.table:
            return None, w
        typ, canon = self.table[w]
        return typ, (canon if lexical else w)


# ---------------------------------------------------------------------------
# Phonetic module
# ---------------------------------------------------------------------------
_PHON_SUBS = [("tion", "xn"), ("tsh", "x"), ("sh", "x"), ("ch", "x"), ("ph", "f"), ("ck", "k"),
              ("th", "0"), ("wh", "w"), ("gh", ""), ("dg", "j"), ("q", "k"),
              ("x", "ks"), ("z", "s")]

def phonetic_key(word):
    """Simplified Metaphone: maps words that *sound* alike to the same key."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    for a, b in _PHON_SUBS:
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    w = re.sub(r"(.)\1+", r"\1", w)                 # collapse doubled letters
    head, tail = w[0], re.sub(r"[aeiouyhw]", "", w[1:])
    return (head + re.sub(r"(.)\1+", r"\1", tail)).upper()

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def phonetic_repair(tok, lex, cache={}):
    """Map an out-of-vocabulary token to the closest-sounding lexicon word."""
    if tok in lex.table or tok in COMMON_WORDS or len(tok) < 3:
        return tok
    if tok in cache:
        return cache[tok]
    k = phonetic_key(tok)
    best, best_score = tok, 1e9
    for w in lex.table:                                  # content words only
        kd = levenshtein(k, lex.keys[w])
        if kd > 2 or abs(len(w) - len(tok)) > 3:
            continue
        score = 0.6 * kd + levenshtein(tok, w) / max(len(w), len(tok))
        if score < best_score and score < 1.35:
            best, best_score = w, score
    cache[tok] = best
    return best


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, phonetic=True, lexical=True, syntax=True):
        self.phonetic, self.lexical, self.syntax = phonetic, lexical, syntax
        self.lex = Lexicon()

    @staticmethod
    def clean(text):
        text = text.lower().replace(",", " , ")
        text = re.sub(r"[^a-z, ]", " ", text)           # drop other punctuation/digits
        return text.split()

    def normalise(self, tokens):
        out = []
        for t in tokens:
            if self.phonetic:
                t = SLANG.get(t, t)
                t = phonetic_repair(t, self.lex)
            out.append(t)
        return out

    def parse_frame(self, toks):
        """Rule-based SRL -> {'agent','verb','patient'} (canonical lemmas)."""
        tagged = [(t,) + self.lex.lookup(t, self.lexical) for t in toks]
        v_idx = next((i for i, (_, typ, _) in enumerate(tagged) if typ == "VERB"), None)
        ents = [(i, c) for i, (_, typ, c) in enumerate(tagged) if typ == "ENT"]
        frame = {"agent": None, "verb": tagged[v_idx][2] if v_idx is not None else None,
                 "patient": None}
        if not self.syntax or v_idx is None:
            # Naive linear SVO assumption: first entity = agent, second = patient
            if ents: frame["agent"] = ents[0][1]
            if len(ents) > 1: frame["patient"] = ents[1][1]
            return frame

        before = [e for e in ents if e[0] < v_idx]
        after = [e for e in ents if e[0] > v_idx]
        is_passive = v_idx > 0 and toks[v_idx - 1] in PASSIVE_AUX
        by_agent = next((c for i, c in after if i > 0 and toks[i - 2:i].count("by")), None)

        if is_passive and by_agent:                      # Y was V-ed by X
            frame["agent"] = by_agent
            frame["patient"] = before[-1][1] if before else None
        elif len(before) >= 2 and "," in toks[before[0][0]:before[1][0]] and not after:
            frame["patient"], frame["agent"] = before[0][1], before[1][1]   # Y, X V-ed
        else:                                            # X V-ed Y  /  it was X that V-ed Y
            frame["agent"] = before[-1][1] if before else None
            frame["patient"] = after[0][1] if after else None
        return frame

    def process(self, text):
        toks = self.normalise(self.clean(text))
        frame = self.parse_frame(toks)
        content = []
        for t in toks:
            if t == "," or t in STOPWORDS:
                continue
            _, c = self.lex.lookup(t, self.lexical)
            content.append(c)
        # Role tagging of content words (used for role-aware transport costs).
        roles = []
        for c in content:
            if c == frame["agent"]: roles.append("AGENT")
            elif c == frame["patient"]: roles.append("PATIENT")
            elif c == frame["verb"]: roles.append("VERB")
            else: roles.append("OTHER")
        return {"tokens": toks, "content": content, "roles": roles, "frame": frame,
                "canon_text": " ".join(content)}
