"""Step 6: which behaviours drive the predictions?

Permutation importance on the TEST presentation for the main model (ensemble, M3) at weeks 4
and 12: shuffle a feature (breaking its link to the outcome), re-score, and record the drop in
ROC-AUC. 10 repeats; mean and standard deviation reported.

Because many features are correlated (e.g. weekly clicks and total clicks), single-feature
importance understates features that share information. Importance is therefore also reported
for FAMILIES of features shuffled together (same row permutation for all columns in a family).

The models are refitted on the training presentations with the tuned hyperparameters from
Step 3; the refit is checked to reproduce the saved test AUC.

Run:  python src/feature_importance.py
Writes: results/06_importance/feature_importance.csv, family_importance.csv, summary.txt,
        fig5_feature_importance_week12.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from config import PROCESSED, RESULTS, SEED
from train_models import SEARCH, preprocessor, meta

OUT = RESULTS / "06_importance"; OUT.mkdir(parents=True, exist_ok=True)
params = json.load(open(RESULTS / "03_models" / "best_params.json"))
saved = pd.read_csv(RESULTS / "03_models" / "results.csv")
WEEKS, REPEATS = [4, 12], 10

def family(c):
    if c in meta["groups"]["0"]["M0"]: return "Background (M0)"
    if c == "total_clicks": return "Total clicks"
    if c.startswith("clicks_week_") or c in ["prestart_clicks", "clicks_last_2_weeks", "weekly_click_trend"]:
        return "Weekly click pattern"
    if c.startswith("share_clicks_") or c == "n_resource_types_used": return "Resource-type mix"
    if c in ["n_assessments_due", "n_submitted", "n_submitted_on_time", "mean_days_early", "share_due_not_submitted"]:
        return "Assignment submission"
    return "Activity regularity & recency"

feat_rows, fam_rows = [], []
rng = np.random.default_rng(SEED)
for w in WEEKS:
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    cols = [c for c in meta["groups"][str(w)]["M3"] if tr[c].notna().any()]
    members = []
    for a, (est, _) in SEARCH.items():
        pipe = Pipeline([("prep", preprocessor(cols)), ("clf", clone(est))])
        pipe.set_params(**{f"clf__{k}": v for k, v in params[f"w{w}_M3_{a}"].items()})
        members.append((a, pipe))
    ens = VotingClassifier(members, voting="soft").fit(tr[cols], tr.at_risk.values)
    X, y = te[cols].copy(), te.at_risk.values
    base = roc_auc_score(y, ens.predict_proba(X)[:, 1])
    ref = saved[(saved.week == w) & (saved.group == "M3") & (saved.algorithm == "ENS")].roc_auc.iloc[0]
    print(f"week {w}: refit test AUC {base:.4f} (saved {ref:.4f})", flush=True)
    assert abs(base - ref) < 1e-6, "refit does not reproduce the saved model"

    def drop_when_shuffled(columns):
        drops = []
        for _ in range(REPEATS):
            Xp = X.copy(); idx = rng.permutation(len(X))
            Xp[columns] = X[columns].values[idx]
            drops.append(base - roc_auc_score(y, ens.predict_proba(Xp)[:, 1]))
        return np.mean(drops), np.std(drops)

    for c in cols:
        m, s = drop_when_shuffled([c])
        feat_rows.append({"week": w, "feature": c, "family": family(c), "auc_drop_mean": m, "auc_drop_std": s})
    fams = pd.Series({c: family(c) for c in cols})
    for f, cs in fams.groupby(fams).groups.items():
        m, s = drop_when_shuffled(list(cs))
        fam_rows.append({"week": w, "family": f, "n_features": len(cs), "auc_drop_mean": m, "auc_drop_std": s})
    print(f"week {w}: done", flush=True)

fi = pd.DataFrame(feat_rows).sort_values(["week", "auc_drop_mean"], ascending=[True, False])
fa = pd.DataFrame(fam_rows).sort_values(["week", "auc_drop_mean"], ascending=[True, False])
fi.round(5).to_csv(OUT / "feature_importance.csv", index=False)
fa.round(5).to_csv(OUT / "family_importance.csv", index=False)
lines = []
for w in WEEKS:
    lines += [f"Week {w}: AUC drop when a FAMILY of features is shuffled (test 2014J, ensemble M3)",
              fa[fa.week == w].drop(columns="week").round(4).to_string(index=False), "",
              f"Week {w}: top 15 single features",
              fi[fi.week == w].head(15).drop(columns="week").round(4).to_string(index=False), ""]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# figure: top 15 single features at week 12, horizontal bars with +-1 sd
d = fi[fi.week == 12].head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.barh(d.feature, d.auc_drop_mean, xerr=d.auc_drop_std, color="#2a78d6", height=0.6,
        error_kw={"ecolor": "#52514e", "lw": 1, "capsize": 2})
ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
ax.set_title("Most important single features, week 12 (ensemble, M3)", loc="left", fontsize=10.5)
ax.tick_params(axis="y", labelsize=8.5)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#eeeeea", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig5_feature_importance_week12.png", dpi=200); plt.close(fig)
