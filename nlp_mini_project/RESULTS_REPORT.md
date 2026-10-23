# Experiment 10: Implementation and Presentation of NLP Mini Project
## Semantic Invariance Engine (SIE) — Testing Linguistic Comprehension vs. Surface Similarity

---

## 1. Project Overview & Problem Statement (Exp 1 & Exp 10)

### 1.1 Aim
To design, implement, and evaluate a real-world Natural Language Processing (NLP) mini project—the **Semantic Invariance Engine (SIE)**—that detects whether two sentences express the same underlying proposition across **syntactic transformations, lexical substitutions, length disparities, and phonetic/orthographic variations**, while distinguishing true paraphrases from adversarial meaning-altering perturbations (such as semantic role reversals).

### 1.2 Problem Statement
> Despite the fluency of modern NLP representations, high benchmark scores do not imply linguistic comprehension ($\text{Capability} \neq \text{Understanding}$). Standard vector-space models (TF-IDF, bag-of-words, and surface n-grams) rely heavily on token overlap and sequential position indices. Consequently, they fail in two critical ways:
> 1. **Sensitivity to Meaning-Preserving Perturbations:** Cosine similarity drops sharply when a concise 5-word sentence is compared against a 35-word clause-heavy paraphrase ($\mathcal{T}_{\text{len}}$), when active voice is transformed into passive/clefted structures ($\mathcal{T}_{\text{syn}}$), when synonyms replace surface words ($\mathcal{T}_{\text{lex}}$), or under ASR/phonetic spelling noise ($\mathcal{T}_{\text{phon}}$).
> 2. **Blindness to Meaning-Altering Role Swaps:** Conversely, bag-of-words and n-gram models assign nearly $100\%$ similarity to sentences like *"The doctor praised the teacher"* and *"The teacher praised the doctor"* (or *"The doctor was praised by the teacher"*), completely failing to recognize that the semantic roles ($\text{AGENT} \leftrightarrow \text{PATIENT}$) have inverted.

---

## 2. System Architecture & Methodology

Following the research methodology outlined in [From-Rules-to-Language-Understanding.pdf](file:///Users/jyotirmoy/Desktop/test/From-Rules-to-Language-Understanding.pdf) (**Phenomenon $\rightarrow$ Hypothesis $\rightarrow$ Formulation $\rightarrow$ Experiments**) and the mathematical formulation in [readme.md](file:///Users/jyotirmoy/Desktop/test/readme.md), our implementation ([nlp_mini_project/](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project)) processes sentence pairs through five stages:

```
Raw Sentence Pair (S1, S2)
   │
   ├─► 1. Preprocessing & Cleaning (sie/preprocess.py)
   │      Lowercasing, punctuation normalization, stop-word filtering
   │
   ├─► 2. Phonetic Normalization (T_phon invariance)
   │      Slang dictionary + Metaphone-style phonetic key & edit-distance repair
   │
   ├─► 3. Lexical Canonicalization (T_lex invariance)
   │      Synset lookup mapping surface synonyms to canonical lemmas
   │
   ├─► 4. Syntactic Fiber-Bundle Projection π(S) (T_syn invariance)
   │      Rule-based Semantic Role Labeling (Active / Passive / Cleft / Topicalized)
   │      extracting the canonical (AGENT, PREDICATE, PATIENT) frame
   │
   └─► 5. Geometric & Topological Feature Extraction (sie/geometry.py, sie/engine.py)
          • Hyperbolic Poincaré Ball Embeddings & Geodesic Centroid Distance (d_H)
          • Fused Gromov-Wasserstein (FGW) Optimal Transport (T_len invariance)
          • H0 Persistent Homology & Persistence Landscape Distance (TDA)
          • Logistic Regression Classifier -> P(Same Meaning)
```

### Code Structure
- [`nlp_mini_project/sie/data.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/sie/data.py): Generates the 1,350-pair controlled **Invariance Quadruple Benchmark** across 5 positive invariance axes (`syntax`, `length`, `lexical`, `phonetic`, `composite`) and 4 hard negative axes (`role_swap`, `role_swap_passive`, `verb_change`, `entity_change`).
- [`nlp_mini_project/sie/preprocess.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/sie/preprocess.py): Implements text cleaning, phonetic key generation (`phonetic_key`), phonetic repair (`phonetic_repair`), synset canonicalization (`Lexicon`), and rule-based syntactic role projection (`parse_frame`).
- [`nlp_mini_project/sie/geometry.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/sie/geometry.py): Pure-NumPy implementation of character n-gram hash embeddings, Poincaré ball exponential map (`expmap0`) & geodesic distance (`poincare_dist`), Sinkhorn entropic optimal transport & Fused Gromov-Wasserstein (`fused_gromov_wasserstein`), and 0-dimensional Vietoris-Rips persistent homology landscapes (`h0_persistence`, `landscape_distance`).
- [`nlp_mini_project/sie/engine.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/sie/engine.py): Combines lexical, frame, hyperbolic, FGW, and TDA features into the `SemanticInvarianceEngine` classifier with an interpretable `.explain(s1, s2)` API.
- [`nlp_mini_project/sie/baselines.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/sie/baselines.py): Implements word-level TF-IDF and character n-gram TF-IDF cosine baselines.
- [`nlp_mini_project/main.py`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/main.py): End-to-end experiment runner, proposition-grouped train/test split, ablation study, length-invariance stress test, and figure generator.

---

## 3. Experimental Results

The benchmark consists of **1,350 sentence pairs** (750 true paraphrases, 600 hard non-paraphrases) split **70% train (945 pairs) / 30% test (405 pairs)** grouped strictly by underlying proposition so that no test proposition is ever seen during training.

### 3.1 Overall Test Performance
*(Source: [`nlp_mini_project/results/overall_metrics.csv`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/overall_metrics.csv))*

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF (Word Cosine)** | 0.5556 | 0.5556 | 1.0000 | 0.7143 |
| **TF-IDF (Char 2–4 Cosine)** | 0.5531 | 0.5545 | 0.9956 | 0.7122 |
| **SIE (Proposed Engine)** | **0.9975** | **1.0000** | **0.9956** | **0.9978** |

### 3.2 Per-Axis Accuracy Breakdown
*(Source: [`nlp_mini_project/results/per_axis_accuracy.csv`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/per_axis_accuracy.csv))*

| Axis | Category | Perturbation Type | TF-IDF Word | TF-IDF Char | **SIE (Proposed)** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `syntax` | Positive ($\mathcal{T}_{\text{syn}}$) | Active $\leftrightarrow$ Passive / Cleft / Topicalized | 1.000 | 1.000 | **1.000** |
| `length` | Positive ($\mathcal{T}_{\text{len}}$) | 1:6 Length Dilation (Pleonastic Framing) | 1.000 | 1.000 | **1.000** |
| `lexical` | Positive ($\mathcal{T}_{\text{lex}}$) | Complete Synonym Substitution | 1.000 | 1.000 | **1.000** |
| `phonetic` | Positive ($\mathcal{T}_{\text{phon}}$) | ASR / Phonetic Spelling & Slang | 1.000 | 1.000 | **1.000** |
| `composite` | Positive ($\mathcal{T}_{\text{all}}$) | Combined Syntax + Length + Lexical + Phonetic | 1.000 | 0.978 | **0.978** |
| `role_swap` | Negative (Hard) | Active Role Reversal (*"X verbed Y"* vs *"Y verbed X"*) | 0.000 | 0.000 | **1.000** |
| `role_swap_passive` | Negative (Hard) | Passive Role Reversal (*"X was verbed by Y"*) | 0.000 | 0.000 | **1.000** |
| `verb_change` | Negative | Predicate Replacement | 0.000 | 0.000 | **1.000** |
| `entity_change` | Negative | Participant Replacement | 0.000 | 0.000 | **1.000** |

> **Key Finding:** Because negative pairs share up to $100\%$ of their vocabulary with the anchor (e.g., `role_swap` has identical words in a different order), any threshold that allows TF-IDF to accept lexical/phonetic/length paraphrases forces it to accept all negative pairs as well ($0\%$ accuracy on negatives). **SIE achieves $100\%$ accuracy across all four negative axes and $99.75\%$ overall test accuracy.**

---

### 3.3 Ablation Study
*(Source: [`nlp_mini_project/results/ablation.csv`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/ablation.csv))*

| Configuration | Accuracy | Precision | Recall | F1 Score | Critical Drop Observed |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full SIE** | **0.9975** | **1.0000** | **0.9956** | **0.9978** | — |
| `- syntax module` | 0.7926 | 0.7831 | 0.8667 | 0.8228 | `role_swap_passive` drops to **0.000**, `syntax` drops to **0.622** |
| `- lexical module` | 0.9654 | 0.9774 | 0.9600 | 0.9686 | `lexical` drops to **0.956**, `entity_change` drops to **0.889** |
| `- frame features` | 0.9728 | 1.0000 | 0.9511 | 0.9749 | `composite` drops to **0.867** |
| `- hyperbolic + TDA` | 0.9753 | 1.0000 | 0.9556 | 0.9773 | `composite` drops to **0.889** |
| `geometry only` | 0.9704 | 0.9953 | 0.9511 | 0.9727 | Relies strictly on role-weighted FGW/Hyperbolic/TDA distances |

---

### 3.4 Length Invariance Stress Test (1x to 6x Length Dilation)
*(Source: [`nlp_mini_project/results/length_invariance.csv`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/length_invariance.csv))*

As pleonastic clauses are progressively added (Padding Levels 0 to 4, expanding a 5-word sentence to over 30 words), standard cosine similarity collapses from $1.000$ to $0.304$, whereas SIE's probability score remains completely invariant ($\ge 0.981$):

| Padding Level | Approx. Sentence Length | TF-IDF Word Cosine | TF-IDF Char Cosine | **SIE P(Same Meaning)** |
| :---: | :---: | :---: | :---: | :---: |
| **Level 0** (Canonical) | 5 words | 1.0000 | 1.0000 | **0.9813** |
| **Level 1** (+ Prefix) | ~9 words | 0.6936 | 0.7270 | **0.9984** |
| **Level 2** (+ Parenthetical) | ~16 words | 0.5362 | 0.4703 | **1.0000** |
| **Level 3** (+ Verbose Suffix) | ~26 words | 0.3957 | 0.3387 | **1.0000** |
| **Level 4** (Full Pleonastic) | ~35 words | 0.3438 | 0.3036 | **1.0000** |

---

### 3.5 Qualitative Explainability Demos
*(Source: [`nlp_mini_project/results/example_explanations.json`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/example_explanations.json))*

1. **Syntactic + Lexical Paraphrase:**
   - $S_1$: *"The doctor praised the teacher."* $\rightarrow$ Frame: `{agent: doctor, verb: praise, patient: teacher}`
   - $S_2$: *"The teacher was commended by the physician."* $\rightarrow$ Frame: `{agent: doctor, verb: praise, patient: teacher}`
   - **Result:** `fgw = 0.0`, `role_cross = 0.0` $\rightarrow$ **`P(same meaning) = 0.9958`** (Correctly classified as **Same**)

2. **Adversarial Passive Role Reversal:**
   - $S_1$: *"The doctor praised the teacher."* $\rightarrow$ Frame: `{agent: doctor, verb: praise, patient: teacher}`
   - $S_2$: *"The doctor was praised by the teacher."* $\rightarrow$ Frame: `{agent: teacher, verb: praise, patient: doctor}`
   - **Result:** Even though surface words match (`canon_cos = 1.0`), `role_cross = 1.0` and `fgw = 0.5433` $\rightarrow$ **`P(same meaning) = 0.0164`** (Correctly rejected as **Different**)

3. **Phonetic / ASR Noise:**
   - $S_1$: *"The pilot warned the chef."*
   - $S_2$: *"Da pilat wornd teh shef."* $\rightarrow$ Normalised to *"the pilot warned the chef"*
   - **Result:** `fgw = 0.0`, `agent/verb/patient_match = 1.0` $\rightarrow$ **`P(same meaning) = 0.9813`** (Correctly classified as **Same**)

---

## 4. Generated Visualizations

All figures are saved in [`nlp_mini_project/results/`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results):
- [`fig1_per_axis_accuracy.png`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/fig1_per_axis_accuracy.png): Bar chart of accuracy across all 9 positive and negative perturbation axes.
- [`fig2_overall_metrics.png`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/fig2_overall_metrics.png): Comparison of Accuracy, Precision, Recall, and F1 across models.
- [`fig3_confusion_sie.png`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/fig3_confusion_sie.png): Confusion matrix of the Semantic Invariance Engine on the 405-pair test set.
- [`fig4_ablation.png`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/fig4_ablation.png): Horizontal bar chart showing the F1 impact of ablating each module.
- [`fig5_length_invariance.png`](file:///Users/jyotirmoy/Desktop/test/nlp_mini_project/results/fig5_length_invariance.png): Length invariance curve across padding levels 0–4.

---

## 5. How to Run

```bash
# Run full benchmark, ablations, and regenerate all CSV tables & PNG plots:
USE_SBERT=0 python3 nlp_mini_project/main.py

# Run interactive explainability demo on any two sentences:
python3 nlp_mini_project/main.py --demo "The doctor praised the teacher." "The teacher was commended by the physician."
```
