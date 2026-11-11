"""Step 5: choose a decision threshold that targets 80% recall for the at-risk class
(decision D14), then report test performance at that threshold.

Threshold choice uses the TRAINING presentations only: out-of-fold probabilities from
leave-one-presentation-out CV (each training presentation predicted by a model fitted on the
other two), using each model's tuned hyperparameters from Step 3. The threshold is the highest
value whose out-of-fold recall is >= 0.80. The ensemble's out-of-fold probability is the mean of
its three members' (= soft voting). The test presentation is then scored once at that threshold,
using the test probabilities saved in Step 3.

Run:  python src/tune_threshold.py            (add --plot-only to redraw the figure from the CSV)
Writes: results/05_threshold/threshold_results.csv, summary.txt, fig4_recall_precision_tuned.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from config import PROCESSED, RESULTS, CHECKPOINT_WEEKS
from train_models import SEARCH, preprocessor, meta, GROUPS

TARGET_RECALL = 0.80
OUT = RESULTS / "05_threshold"; OUT.mkdir(parents=True, exist_ok=True)
params = json.load(open(RESULTS / "03_models" / "best_params.json"))
test_p = pd.read_pickle(RESULTS / "03_models" / "test_predictions.pkl")

def pick_threshold(y, p, target):
    cands = np.unique(np.round(p, 4))[::-1]          # high -> low
    for t in cands:
        if recall_score(y, (p >= t).astype(int)) >= target:
            return float(t)
    return 0.0

def scores(y, p, t):
    yhat = (p >= t).astype(int)
    return {"recall": recall_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "share_flagged": yhat.mean()}

import sys
PLOT_ONLY = "--plot-only" in sys.argv
rows = []
for w in ([] if PLOT_ONLY else CHECKPOINT_WEEKS):
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    tr = df[df.split == "train"].reset_index(drop=True)
    yte = test_p[f"w{w}_y_test"]
    for g in GROUPS:
        cols = [c for c in meta["groups"][str(w)][g] if tr[c].notna().any()]
        oof = {}
        for a, (est, _) in SEARCH.items():
            pipe = Pipeline([("prep", preprocessor(cols)), ("clf", clone(est))])
            pipe.set_params(**{f"clf__{k}": v for k, v in params[f"w{w}_{g}_{a}"].items()})
            p = np.zeros(len(tr))
            for fit_idx, val_idx in LeaveOneGroupOut().split(tr, groups=tr.code_presentation):
                m = clone(pipe).fit(tr.loc[fit_idx, cols], tr.at_risk.values[fit_idx])
                p[val_idx] = m.predict_proba(tr.loc[val_idx, cols])[:, 1]
            oof[a] = p
        oof["ENS"] = np.mean([oof[a] for a in SEARCH], axis=0)
        for a, p in oof.items():
            t = pick_threshold(tr.at_risk.values, p, TARGET_RECALL)
            pt = test_p[f"w{w}_{g}_{a}"]
            rows.append({"week": w, "group": g, "algorithm": a, "threshold": round(t, 4),
                         "train_oof_recall": recall_score(tr.at_risk, (p >= t).astype(int)),
                         **{f"test_{k}": v for k, v in scores(yte, pt, t).items()},
                         **{f"test_{k}_at_0.5": v for k, v in scores(yte, pt, 0.5).items()}})
        print(f"week {w:>2} {g}: ENS threshold {rows[-1]['threshold']:.3f} -> test recall "
              f"{rows[-1]['test_recall']:.3f}, precision {rows[-1]['test_precision']:.3f}", flush=True)

if PLOT_ONLY:
    res = pd.read_csv(OUT / "threshold_results.csv")
else:
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "threshold_results.csv", index=False)

e = res[(res.group == "M3") & (res.algorithm == "ENS")].set_index("week")
tab = e[["threshold", "test_recall", "test_precision", "test_f1", "test_share_flagged",
         "test_recall_at_0.5", "test_precision_at_0.5", "test_f1_at_0.5"]].round(3)
lines = [f"Ensemble, M3: threshold chosen for {TARGET_RECALL:.0%} recall on training (out-of-fold),"
         " evaluated on test 2014J", tab.to_string(), "",
         "All models, mean test recall at tuned threshold by algorithm:",
         res.groupby("algorithm").test_recall.mean().round(3).to_string()]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# figure: recall and precision of the M3 ensemble, at 0.5 and at the tuned threshold
COL = {"recall": "#2a78d6", "precision": "#eb6834"}
fig, ax = plt.subplots(figsize=(7, 4.4))
for m in ["recall", "precision"]:
    ax.plot(e.index, e[f"test_{m}"], color=COL[m], lw=2, marker="o", ms=5, label=f"{m.capitalize()}, tuned threshold")
    ax.plot(e.index, e[f"test_{m}_at_0.5"], color=COL[m], lw=1.5, ls="--", marker="o", ms=4,
            mfc="white", label=f"{m.capitalize()}, threshold 0.5")
ax.axhline(TARGET_RECALL, color="#8a8984", lw=0.8, ls=":")
ax.text(0, TARGET_RECALL + 0.008, "recall target 0.80", color="#52514e", fontsize=8)
ax.set_xticks(CHECKPOINT_WEEKS); ax.set_ylim(0.4, 1.0)
ax.set_xlabel("Checkpoint (weeks since module start)"); ax.set_ylabel("Test score (at-risk class)")
ax.set_title("Tuned threshold: more at-risk students caught, lower precision (ensemble, M3)",
             loc="left", fontsize=10.5, pad=40)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e4e3df", lw=0.8)
fig.tight_layout(); fig.savefig(OUT / "fig4_recall_precision_tuned.png", dpi=200); plt.close(fig)
