"""
main.py - Experiment 10: Semantic Invariance Engine (NLP Mini Project)

Usage
    python main.py                     # full experiment -> results/
    python main.py --demo "S1" "S2"    # explain a single sentence pair
    USE_SBERT=0 python main.py         # skip the (optional) SBERT baseline

Pipeline
    1. Data       : generate the controlled Invariance-Quadruple benchmark
    2. Split      : 70 / 30 split *by proposition* (no leakage of meanings)
    3. Train      : baselines (threshold) + SIE (logistic regression on features)
    4. Evaluate   : accuracy / precision / recall / F1, per-axis accuracy
    5. Ablation   : remove one module at a time
    6. Length     : Length-Invariance curve (similarity vs. padding level)
    7. Figures    : results/*.png  and tables results/*.csv
"""
import os, sys, json, random
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

from sie.data import build_benchmark, active, padded, POSITIVE_AXES, NEGATIVE_AXES
from sie.engine import SemanticInvarianceEngine
from sie.baselines import available_baselines

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
AXES = POSITIVE_AXES + NEGATIVE_AXES
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def metrics(y, yhat):
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0))


def per_axis(df, yhat):
    d = df.assign(correct=(yhat == df.label.values))
    return d.groupby("axis").correct.mean().reindex(AXES)


def run():
    # ------------------------------------------------------------ 1. data
    data, props = build_benchmark(n_props=150, seed=42)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    print(f"Benchmark: {len(df)} pairs  ({df.label.sum()} paraphrases / {(1-df.label).sum()} non-paraphrases)")

    # ------------------------------------------------------------ 2. split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    tr_idx, te_idx = next(gss.split(df, df.label, groups=df.anchor))
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    P = lambda d: list(zip(d.anchor, d.variant))
    corpus = list(pd.concat([tr.anchor, tr.variant]).unique())
    print(f"Train pairs: {len(tr)}   Test pairs: {len(te)}")

    # ------------------------------------------------------------ 3/4. models
    models = available_baselines()
    sie = SemanticInvarianceEngine()
    sie.name = "SIE (proposed)"
    models.append(sie)

    rows, axis_tab, preds = [], {}, {}
    for m in models:
        m.fit(P(tr), tr.label.values, corpus)
        yhat = m.predict(P(te))
        preds[m.name] = yhat
        rows.append({"model": m.name, **metrics(te.label, yhat)})
        axis_tab[m.name] = per_axis(te, yhat)
    res = pd.DataFrame(rows).set_index("model").round(4)
    axis_df = pd.DataFrame(axis_tab).round(3)
    res.to_csv(os.path.join(OUT, "overall_metrics.csv"))
    axis_df.to_csv(os.path.join(OUT, "per_axis_accuracy.csv"))
    print("\n=== Overall test metrics ===\n", res.to_string())
    print("\n=== Per-axis accuracy ===\n", axis_df.to_string())

    # ------------------------------------------------------------ 5. ablation
    configs = {
        "Full SIE":           {},
        "- phonetic module":  dict(phonetic=False),
        "- lexical module":   dict(lexical=False),
        "- syntax module":    dict(syntax=False),
        "- frame features":   dict(groups=("lexsem", "fgw", "hyp", "tda", "len")),
        "- FGW transport":    dict(groups=("lexsem", "frame", "hyp", "tda", "len")),
        "- hyperbolic + TDA": dict(groups=("lexsem", "frame", "fgw", "len")),
        "geometry only":      dict(groups=("fgw", "hyp", "tda", "len")),
    }
    abl = []
    for name, kw in configs.items():
        e = SemanticInvarianceEngine(**kw).fit(P(tr), tr.label.values, corpus)
        yhat = e.predict(P(te))
        abl.append({"config": name, **metrics(te.label, yhat),
                    **{f"acc_{a}": v for a, v in per_axis(te, yhat).items()}})
    abl_df = pd.DataFrame(abl).set_index("config").round(4)
    abl_df.to_csv(os.path.join(OUT, "ablation.csv"))
    print("\n=== Ablation (test) ===\n", abl_df[["accuracy", "f1"]].to_string())

    # ------------------------------------------------------------ 6. length curve
    rng = random.Random(7)
    test_props = [p for p in props if active(*p) in set(te.anchor)][:40]
    lic = {m.name: [] for m in models}
    for level in range(5):
        pairs = [(active(*p), padded(*p, level, rng)) for p in test_props]
        for m in models:
            s = m.predict_proba(pairs) if m is sie else m.scores(pairs)
            lic[m.name].append(float(np.mean(s)))
    lic_df = pd.DataFrame(lic, index=pd.Index(range(5), name="padding_level")).round(4)
    lic_df.to_csv(os.path.join(OUT, "length_invariance.csv"))
    print("\n=== Length invariance (mean score vs padding) ===\n", lic_df.to_string())

    # ------------------------------------------------------------ 7. figures
    make_figures(res, axis_df, abl_df, lic_df, te, preds, sie)

    # sample explanations
    demo = [("The doctor praised the teacher.", "The teacher was commended by the physician."),
            ("The doctor praised the teacher.", "The doctor was praised by the teacher."),
            ("The pilot warned the chef.", "Da pilat wornd teh shef."),
            ("The pilot warned the chef.",
             "According to several sources familiar with the matter, it is widely believed "
             "that the aviator, as many people had already expected, cautioned the cook.")]
    ex = [{"S1": a, "S2": b, **sie.explain(a, b)} for a, b in demo]
    with open(os.path.join(OUT, "example_explanations.json"), "w") as f:
        json.dump(ex, f, indent=2)
    print(f"\nAll outputs written to {OUT}")


def make_figures(res, axis_df, abl_df, lic_df, te, preds, sie):
    colors = ["#9aa5b1", "#c4a484", "#7c9eb2", "#2f6f5e"][-len(axis_df.columns):]

    # Fig 1 - per-axis accuracy
    ax = axis_df.plot.bar(figsize=(11, 4.2), color=colors, width=0.8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05); ax.set_xlabel("")
    ax.set_title("Per-axis accuracy: positives (invariance) | negatives (sensitivity)")
    ax.axvline(len(POSITIVE_AXES) - 0.5, color="k", lw=0.8, ls="--")
    plt.xticks(rotation=25, ha="right"); plt.legend(frameon=False, ncol=4)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_per_axis_accuracy.png")); plt.close()

    # Fig 2 - overall metrics
    ax = res.plot.bar(figsize=(8, 4), color=["#2f6f5e", "#7c9eb2", "#c4a484", "#9aa5b1"], width=0.75)
    ax.set_ylim(0, 1.05); ax.set_title("Overall test metrics"); plt.xticks(rotation=0)
    plt.legend(frameon=False, ncol=4, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_overall_metrics.png")); plt.close()

    # Fig 3 - confusion matrix of SIE
    cm = confusion_matrix(te.label, preds[sie.name])
    ConfusionMatrixDisplay(cm, display_labels=["different", "same"]).plot(cmap="Greens", colorbar=False)
    plt.title("SIE confusion matrix (test)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_confusion_sie.png")); plt.close()

    # Fig 4 - ablation
    ax = abl_df["f1"].iloc[::-1].plot.barh(figsize=(7, 4), color="#2f6f5e")
    ax.set_xlim(0, 1.05); ax.set_xlabel("Test F1"); ax.set_title("Ablation study")
    for i, v in enumerate(abl_df["f1"].iloc[::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_ablation.png")); plt.close()

    # Fig 5 - length invariance curve
    ax = lic_df.plot(marker="o", figsize=(7, 4), color=colors)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Mean similarity / P(same)")
    ax.set_title("Length invariance: score vs. pleonastic padding level")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig5_length_invariance.png")); plt.close()


def demo(s1, s2):
    data, _ = build_benchmark(n_props=150, seed=42)
    df = pd.DataFrame(data)
    corpus = list(pd.concat([df.anchor, df.variant]).unique())
    sie = SemanticInvarianceEngine().fit(list(zip(df.anchor, df.variant)), df.label.values, corpus)
    print(json.dumps(sie.explain(s1, s2), indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--demo":
        demo(sys.argv[2], sys.argv[3])
    else:
        run()
