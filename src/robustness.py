"""Step 11: robustness checks for the main model (ensemble, M3, Step 3 hyperparameters).

C1 Split design: the temporal split (train 2013B/2013J/2014B, test 2014J) is compared with a
   random stratified split of all enrolments, with the same test size. A random split lets the
   model see students from the test presentation (and sometimes the same student in another
   module) during training, so any gain shows how much a random split would overstate results.
C2 Per module: test ROC-AUC per module (from Step 3 predictions), to check that the overall
   result is not carried by a few modules.
C3 Stability: refit with 5 different random seeds (RF and HGB) at weeks 4, 8, 12 and report the
   spread of test ROC-AUC.

Run:  python src/robustness.py
Writes: results/11_robustness/split_comparison.csv, auc_by_module.csv, seed_stability.csv,
        summary.txt, fig12_auc_by_module.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import PROCESSED, RESULTS, SEED, CHECKPOINT_WEEKS
from refit import tuned_ensemble, usable_cols
from plot_style import COLORS, INK, finish

OUT = RESULTS / "11_robustness"; OUT.mkdir(parents=True, exist_ok=True)
saved = pd.read_csv(RESULTS / "03_models" / "results.csv")
P = pd.read_pickle(RESULTS / "03_models" / "test_predictions.pkl")
lines = []
def log(s=""):
    print(s, flush=True); lines.append(str(s))
main_auc = lambda w: saved[(saved.week == w) & (saved.group == "M3") & (saved.algorithm == "ENS")].roc_auc.iloc[0]

# ---- C1: random vs temporal split
rows = []
for w in CHECKPOINT_WEEKS:
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl").reset_index(drop=True)
    cols = usable_cols(df[df.split == "train"], w, "M3")
    tr, te = train_test_split(df, test_size=(df.split == "test").mean(), stratify=df.at_risk, random_state=SEED)
    m = tuned_ensemble(w, "M3", cols).fit(tr[cols], tr.at_risk)
    auc_r = roc_auc_score(te.at_risk, m.predict_proba(te[cols])[:, 1])
    shared = len(set(tr.id_student) & set(te.id_student))
    rows.append({"week": w, "temporal_auc": main_auc(w), "random_auc": auc_r,
                 "inflation": auc_r - main_auc(w), "test_students_also_in_train": shared})
c1 = pd.DataFrame(rows)
c1.round(4).to_csv(OUT / "split_comparison.csv", index=False)
log("C1. Temporal vs random split (ensemble M3, same hyperparameters)"); log(c1.round(3).to_string(index=False)); log()

# ---- C2: per-module AUC
rows = []
for w in CHECKPOINT_WEEKS:
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    te = df[df.split == "test"].reset_index(drop=True)
    p, y = P[f"w{w}_M3_ENS"], P[f"w{w}_y_test"]
    assert (te.at_risk.values == y).all()
    for mod, idx in te.groupby("code_module").groups.items():
        rows.append({"week": w, "module": mod, "n": len(idx), "base_rate": y[idx].mean(),
                     "roc_auc": roc_auc_score(y[idx], p[idx])})
c2 = pd.DataFrame(rows)
c2.round(4).to_csv(OUT / "auc_by_module.csv", index=False)
log("C2. Test ROC-AUC per module (ensemble M3)")
log(c2.pivot(index="module", columns="week", values="roc_auc").round(3).to_string()); log()
log("Test size per module (week 12):"); log(c2[c2.week == 12][["module", "n", "base_rate"]].round(3).to_string(index=False)); log()

d = c2[c2.week.isin([4, 12])].pivot(index="module", columns="week", values="roc_auc")
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(d)); bw = 0.38
for i, w in enumerate([4, 12]):
    ax.bar(x + (i - 0.5) * (bw + 0.02), d[w], width=bw, color=COLORS[i], label=f"Week {w}")
for i, w in enumerate([4, 12]):
    ax.axhline(main_auc(w), color=COLORS[i], lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(d.index); ax.set_ylim(0.5, 0.95)
ax.set_xlabel("Module (test presentation 2014J)"); ax.set_ylabel("Test ROC-AUC")
finish(fig, ax, "The model works in every module (dashed = overall AUC)", OUT / "fig12_auc_by_module.png", ncol=2)

# ---- C3: seed stability
rows = []
for w in [4, 8, 12]:
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    cols = usable_cols(tr, w, "M3")
    for s in range(5):
        m = tuned_ensemble(w, "M3", cols, seed=s).fit(tr[cols], tr.at_risk)
        rows.append({"week": w, "seed": s, "roc_auc": roc_auc_score(te.at_risk, m.predict_proba(te[cols])[:, 1])})
    log(f"  seed run week {w} done")
c3 = pd.DataFrame(rows)
c3.round(5).to_csv(OUT / "seed_stability.csv", index=False)
agg = c3.groupby("week").roc_auc.agg(["mean", "std", "min", "max"])
log("C3. Test ROC-AUC over 5 random seeds (ensemble M3)"); log(agg.round(4).to_string())
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
