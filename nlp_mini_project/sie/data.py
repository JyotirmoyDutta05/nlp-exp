"""
data.py - Controlled "Invariance Quadruple" benchmark generator.

Every example is a PAIR (anchor, variant) with a binary label:
    1 = same meaning (semantic paraphrase), 0 = different meaning.

The anchor is always a canonical active sentence  "The <agent> <verb> the <patient>."
Positive variants apply meaning-preserving operators from the readme:
    syntax    : T_syn   (passive / cleft / topicalisation)
    length    : T_len   (pleonastic framing, parentheticals, verbose clauses)
    lexical   : T_lex   (synonym substitution)
    phonetic  : T_phon  (phonetic spelling / ASR-like noise / slang)
    composite : T_lex(T_syn(T_len(T_phon(S))))   - all four at once
Negative variants change the proposition while keeping surface overlap high:
    role_swap          "The B verb the A."             (same words, roles flipped)
    role_swap_passive  "The A was verb-ed by the B."   (same words, passive, flipped)
    verb_change        different predicate
    entity_change      different participant
"""
import random

# ---------------------------------------------------------------------------
# Lexical resources (a tiny WordNet-style synset table).
# All entities are animate so that every role swap is plausible.
# ---------------------------------------------------------------------------
ENTITIES = {
    "committee":  ["panel", "board"],
    "doctor":     ["physician"],
    "teacher":    ["instructor", "tutor"],
    "police":     ["cops"],
    "manager":    ["supervisor", "boss"],
    "lawyer":     ["attorney"],
    "student":    ["learner"],
    "chef":       ["cook"],
    "pilot":      ["aviator"],
    "journalist": ["correspondent"],
}

# base form -> (past / past-participle, [synonymous past forms])
VERBS = {
    "approve":   ("approved",    ["endorsed"]),
    "help":      ("helped",      ["assisted", "aided"]),
    "praise":    ("praised",     ["commended", "applauded"]),
    "call":      ("called",      ["phoned"]),
    "blame":     ("blamed",      ["faulted"]),
    "hire":      ("hired",       ["employed"]),
    "warn":      ("warned",      ["cautioned"]),
    "follow":    ("followed",    ["trailed"]),
    "criticize": ("criticized",  ["condemned"]),
    "interview": ("interviewed", ["questioned"]),
}

POSITIVE_AXES = ["syntax", "length", "lexical", "phonetic", "composite"]
NEGATIVE_AXES = ["role_swap", "role_swap_passive", "verb_change", "entity_change"]

# ---------------------------------------------------------------------------
# Surface realisations
# ---------------------------------------------------------------------------
def active(a, v, p):
    return f"The {a} {v} the {p}."

def passive(a, v, p):
    return f"The {p} was {v} by the {a}."

def cleft(a, v, p):
    return f"It was the {a} that {v} the {p}."

def topicalised(a, v, p):
    return f"The {p}, the {a} {v}."

SYNTAX_OPS = [passive, cleft, topicalised]

# Pleonastic material: carries no new proposition about agent/verb/patient.
PREFIXES = [
    "Earlier today,",
    "According to several sources familiar with the matter, it is widely believed that",
    "As far as anyone can tell at this point in time,",
]
PARENTHETICALS = [
    ", as many people had already expected,",
    ", after much internal deliberation,",
]
SUFFIXES = [
    " after a long and rather detailed discussion that lasted for several hours",
    " at the end of what everyone described as an unusually busy week",
]

def padded(a, v, p, level, rng):
    """Active sentence dilated with `level` (0-4) layers of pleonastic material."""
    subj = f"the {a}"
    if level >= 2:
        subj += rng.choice(PARENTHETICALS)
    core = f"{subj} {v} the {p}"
    if level >= 3:
        core += rng.choice(SUFFIXES)
    if level >= 1:
        prefix = PREFIXES[1] if level >= 4 else rng.choice([PREFIXES[0], PREFIXES[2]])
        core = f"{prefix} {core}"
    return core[0].upper() + core[1:] + "."

# ---------------------------------------------------------------------------
# Phonetic / orthographic noise (simulates phonetic typing and ASR errors)
# ---------------------------------------------------------------------------
FUNCTION_SLANG = {"the": ["da", "teh", "tha"], "was": ["wuz", "woz"],
                  "by": ["bai", "bi"], "that": ["dat"]}
PHON_RULES = [("ph", "f"), ("ck", "k"), ("ch", "tsh"), ("er", "a"), ("or", "a"),
              ("ed", "d"), ("ee", "i"), ("oo", "u"), ("tion", "shun"),
              ("ise", "ize"), ("s", "z"), ("i", "ee"), ("ou", "ow"), ("y", "i")]

def phonetic_noise(sentence, rng, p_word=0.8):
    out = []
    for raw in sentence.split():
        punct = raw[-1] if raw[-1] in ".,!?" else ""
        w = raw[:-1] if punct else raw
        lw = w.lower()
        if lw in FUNCTION_SLANG and rng.random() < p_word:
            w = rng.choice(FUNCTION_SLANG[lw])
        elif len(lw) > 3 and rng.random() < p_word:
            rules = [r for r in PHON_RULES if r[0] in lw]
            rng.shuffle(rules)
            for src, dst in rules[:2]:
                lw = lw.replace(src, dst, 1)
            w = lw
        out.append(w + punct)
    return " ".join(out)

def lexical_sub(a, v, p, rng):
    """Replace every content word by a synonym from the synset table."""
    vb = next(b for b, (past, _) in VERBS.items() if past == v)
    return (rng.choice(ENTITIES[a]), rng.choice(VERBS[vb][1]), rng.choice(ENTITIES[p]))

# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------
def build_benchmark(n_props=150, seed=42):
    """Return a list of dicts {anchor, variant, label, axis}."""
    rng = random.Random(seed)
    ents, verbs = list(ENTITIES), list(VERBS)
    props, seen = [], set()
    while len(props) < n_props:
        a, p = rng.sample(ents, 2)
        v = VERBS[rng.choice(verbs)][0]
        if (a, v, p) not in seen:
            seen.add((a, v, p)); props.append((a, v, p))

    data = []
    for a, v, p in props:
        anchor = active(a, v, p)
        add = lambda s, y, ax: data.append(dict(anchor=anchor, variant=s, label=y, axis=ax))
        # ---- positives ---------------------------------------------------
        add(rng.choice(SYNTAX_OPS)(a, v, p), 1, "syntax")
        add(padded(a, v, p, rng.randint(2, 4), rng), 1, "length")
        add(active(*lexical_sub(a, v, p, rng)), 1, "lexical")
        add(phonetic_noise(active(a, v, p), rng), 1, "phonetic")
        la, lv, lp = lexical_sub(a, v, p, rng)
        comp = rng.choice(SYNTAX_OPS)(la, lv, lp).rstrip(".")
        comp = f"Earlier today, {comp[0].lower() + comp[1:]}."
        add(phonetic_noise(comp, rng, p_word=0.5), 1, "composite")
        # ---- negatives (high surface overlap, different proposition) -----
        add(active(p, v, a), 0, "role_swap")
        add(passive(p, v, a), 0, "role_swap_passive")
        other_v = VERBS[rng.choice([b for b in verbs if VERBS[b][0] != v])][0]
        add(rng.choice([active, passive])(a, other_v, p), 0, "verb_change")
        other_e = rng.choice([e for e in ents if e not in (a, p)])
        add(active(a, v, other_e) if rng.random() < .5 else active(other_e, v, p),
            0, "entity_change")
    return data, props
