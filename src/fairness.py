"""Step 8: does the model work equally well for different student groups?

For the main model (ensemble, M3) at the tuned threshold (Step 5), on the test presentation,
reports per subgroup: n, at-risk base rate, ROC-AUC, recall (true positive rate), false positive
rate, precision and share flagged. Gaps summarise each attribute:
  - recall gap   = max - min recall across subgroups ("equal opportunity" difference)
  - FPR gap      = max - min false positive rate
  - AUC gap      = max - min ROC-AUC
Subgroups with fewer than 100 students are reported but excluded from the gaps.

Attributes: gender, age_band, disability, imd_band (deprivation), highest_education.
Note: these attributes are also model inputs (M0); this step measures outcomes, it does not
remove them.

Run:  python src/fairness.py
Writes: results/08_fairness/fairness_by_group.csv, fairness_gaps.csv, summary.txt,
        fig6_recall_fpr_by_group_week12.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from config import PROCESSED, RESULTS

OUT = RESULTS / "08_fairness"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_pickle(RESULTS / "03_models" / "test_predictions.pkl")
thr = pd.read_csv(RESULTS / "05_threshold" / "threshold_results.csv")
ATTRS = ["gender", "age_band", "disability", "imd_band", "highest_education"]
WEEKS, MIN_N = [4, 12], 100

rows = []
for w in WEEKS:
    te = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    te = te[te.split == "test"].reset_index(drop=True)   # same order as the saved predictions
    p = P[f"w{w}_M3_ENS"]; y = P[f"w{w}_y_test"]
    assert len(te) == len(p) and (te.at_risk.values == y).all()
    t = thr[(thr.week == w) & (thr.group == "M3") & (thr.algorithm == "ENS")].threshold.iloc[0]
    yhat = (p >= t).astype(int)
    for a in ATTRS:
        vals = te[a].fillna("missing").astype(str)
        for v in sorted(vals.unique()):
            m = (vals == v).values
            yy, pp, hh = y[m], p[m], yhat[m]
            rows.append({"week": w, "attribute": a, "group": v, "n": int(m.sum()),
                         "base_rate": yy.mean(),
                         "roc_auc": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan,
                         "recall": hh[yy == 1].mean() if yy.sum() else np.nan,
                         "fpr": hh[yy == 0].mean() if (yy == 0).sum() else np.nan,
                         "precision": yy[hh == 1].mean() if hh.sum() else np.nan,
                         "share_flagged": hh.mean()})
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "fairness_by_group.csv", index=False)

big = res[res.n >= MIN_N]
gaps = big.groupby(["week", "attribute"]).agg(
    n_groups=("group", "size"),
    recall_gap=("recall", lambda s: s.max() - s.min()),
    fpr_gap=("fpr", lambda s: s.max() - s.min()),
    auc_gap=("roc_auc", lambda s: s.max() - s.min()),
    base_rate_gap=("base_rate", lambda s: s.max() - s.min())).reset_index()
gaps.round(4).to_csv(OUT / "fairness_gaps.csv", index=False)

lines = [f"Ensemble M3, tuned threshold, test 2014J. Gaps over subgroups with n >= {MIN_N}.", "",
         gaps.round(3).to_string(index=False), ""]
for w in WEEKS:
    lines += [f"Week {w} by subgroup:",
              res[res.week == w].drop(columns="week").round(3).to_string(index=False), ""]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:4 + len(gaps)]))

# figure: recall and FPR per subgroup, week 12 (dot plot, one row per subgroup)
d = res[(res.week == 12) & (res.n >= MIN_N)].copy()
d["label"] = d.attribute + ": " + d.group
d = d.iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.6))
yy = np.arange(len(d))
ax.hlines(yy, d.fpr, d.recall, color="#d6d5d0", lw=1.5)
ax.plot(d.recall, yy, "o", color="#2a78d6", ms=6, label="Recall (at-risk caught)")
ax.plot(d.fpr, yy, "o", color="#eb6834", ms=6, label="False positive rate")
ax.set_yticks(yy); ax.set_yticklabels(d.label, fontsize=8)
for k, a in enumerate(d.attribute):
    if k and a != d.attribute.iloc[k - 1]: ax.axhline(k - 0.5, color="#e4e3df", lw=0.8)
ax.set_xlim(0, 1); ax.set_xlabel("Rate (test 2014J, ensemble M3, week 12, tuned threshold)")
ax.set_title("Recall and false positive rate by student group", loc="left", fontsize=10.5, pad=28)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#eeeeea", lw=0.8)
fig.tight_layout(); fig.savefig(OUT / "fig6_recall_fpr_by_group_week12.png", dpi=200); plt.close(fig)
