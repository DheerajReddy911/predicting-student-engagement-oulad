"""Step 7: are the AUC differences real or chance?

Uses DeLong's test for two correlated ROC-AUCs (both models scored on the same test students),
fast implementation after Sun & Xu (2014). 95% confidence intervals for each AUC come from the
same DeLong variance. Because many comparisons are made, p-values are adjusted with the Holm
method within each family of comparisons.

Families:
  A. Feature groups (ensemble): M1 vs M0, M2 vs M1, M3 vs M2, at every checkpoint
  B. Algorithms (M3): ENS vs HGB, ENS vs RF, ENS vs LR, at every checkpoint

Run:  python src/significance_tests.py
Writes: results/07_significance/auc_with_ci.csv, delong_tests.csv, summary.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from config import RESULTS, CHECKPOINT_WEEKS

OUT = RESULTS / "07_significance"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_pickle(RESULTS / "03_models" / "test_predictions.pkl")

def midrank(x):
    order = np.argsort(x); xs = x[order]; n = len(x)
    ranks = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]: j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[order] = ranks
    return out

def delong(y, preds):
    """AUCs and their covariance matrix for k score vectors on the same cases."""
    y = np.asarray(y).astype(bool)
    pos, neg = [p[y] for p in preds], [p[~y] for p in preds]
    m, n = y.sum(), (~y).sum()
    aucs, v01, v10 = [], [], []
    for xp, xn in zip(pos, neg):
        tx, ty, tz = midrank(xp), midrank(xn), midrank(np.concatenate([xp, xn]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)          # per positive case
        v10.append(1 - (tz[m:] - ty) / m)      # per negative case
    s01, s10 = np.cov(np.array(v01)), np.cov(np.array(v10))
    return np.array(aucs), np.atleast_2d(s01) / m + np.atleast_2d(s10) / n

def compare(y, a, b):
    aucs, S = delong(y, [a, b])
    d = aucs[0] - aucs[1]
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = d / se
    return aucs, d, d - 1.96 * se, d + 1.96 * se, 2 * norm.sf(abs(z))

def holm(p):
    p = np.asarray(p); order = np.argsort(p); k = len(p)
    adj = np.empty(k); running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (k - rank) * p[i])); adj[i] = running
    return adj

# AUC with 95% CI for every model
ci_rows = []
for key, p in P.items():
    if "_y_test" in key: continue
    w, g, a = key.split("_"); w = int(w[1:])
    auc, S = delong(P[f"w{w}_y_test"], [p])
    se = np.sqrt(S[0, 0])
    ci_rows.append({"week": w, "group": g, "algorithm": a, "roc_auc": auc[0],
                    "ci_low": auc[0] - 1.96 * se, "ci_high": auc[0] + 1.96 * se})
pd.DataFrame(ci_rows).round(4).to_csv(OUT / "auc_with_ci.csv", index=False)

rows = []
for w in CHECKPOINT_WEEKS:
    y = P[f"w{w}_y_test"]
    for hi, lo in [("M1", "M0"), ("M2", "M1"), ("M3", "M2")]:
        aucs, d, l, h, p = compare(y, P[f"w{w}_{hi}_ENS"], P[f"w{w}_{lo}_ENS"])
        rows.append({"family": "A_feature_groups", "week": w, "comparison": f"{hi} vs {lo} (ENS)",
                     "auc_a": aucs[0], "auc_b": aucs[1], "diff": d, "diff_ci_low": l, "diff_ci_high": h, "p": p})
    for other in ["HGB", "RF", "LR"]:
        aucs, d, l, h, p = compare(y, P[f"w{w}_M3_ENS"], P[f"w{w}_M3_{other}"])
        rows.append({"family": "B_algorithms", "week": w, "comparison": f"ENS vs {other} (M3)",
                     "auc_a": aucs[0], "auc_b": aucs[1], "diff": d, "diff_ci_low": l, "diff_ci_high": h, "p": p})
res = pd.DataFrame(rows)
res["p_holm"] = res.groupby("family")["p"].transform(holm)
res["significant_0.05"] = res.p_holm < 0.05
res.round(5).to_csv(OUT / "delong_tests.csv", index=False)

# sanity: DeLong AUC equals sklearn AUC
from sklearn.metrics import roc_auc_score
chk = max(abs(r["roc_auc"] - roc_auc_score(P[f"w{r['week']}_y_test"], P[f"w{r['week']}_{r['group']}_{r['algorithm']}"])) for r in ci_rows)
lines = [f"Check: max |DeLong AUC - sklearn AUC| = {chk:.2e}", ""]
for fam, g in res.groupby("family"):
    t = g.pivot(index="week", columns="comparison", values="diff").round(3)
    s = g.pivot(index="week", columns="comparison", values="significant_0.05")
    lines += [f"Family {fam}: AUC difference (* = significant after Holm, p<0.05)",
              (t.astype(str) + s.replace({True: "*", False: ""})).to_string(), ""]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
