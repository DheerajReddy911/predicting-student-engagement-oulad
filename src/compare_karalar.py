"""Step 10: comparison with Karalar, Kapucu & Gurueler (2021) ("Paper 1").

Their final model: 5 features (Quiz score, Degree, Lecture_notes, Materials, Video), selected by
SelectKBest, fed to an ensemble of Extra Trees + Random Forest + Logistic Regression with HARD
voting and weights [1, 2, 1]. OULAD has no direct equivalents, so each feature is mapped to its
closest OULAD counterpart (all mappings are approximations and are reported as such):

  Quiz score    -> mean score of non-exam assessments (TMA + CMA) submitted before the checkpoint
                   (AAA and EEE have no CMA quizzes, so TMAs are included). NOTE: this uses GRADES,
                   which this thesis's own feature sets deliberately do not.
  Degree        -> highest_education (prior qualification level)
  Lecture_notes -> clicks on 'resource' items (downloadable files, e.g. PDFs)
  Materials     -> clicks on other course-content items (oucontent, subpage, page, url, folder,
                   dualpane, htmlactivity, sharedsubpage, dataplus, glossary)
  Video         -> clicks on web-conferencing items (ouelluminate, oucollaborate), the closest
                   OULAD has to recorded/live sessions

2 x 2 design at every checkpoint, all on the same rows, split, tuning procedure and test set:
  K-feat + K-model    : Paper 1 as published (mapped features, ET+RF+LR hard voting [1,2,1])
  K-feat + our model  : Paper 1 features, our soft-voting LR+RF+HGB ensemble
  M3     + K-model    : our features, Paper 1 model
  M3     + our model  : our main model (taken from Step 3, not re-run)
Hard voting gives no probabilities; its ROC-AUC uses the weighted vote share (0, .25, .5, .75, 1)
as the score, and its class prediction uses majority (> 0.5).

Run:  python src/compare_karalar.py        (add --plot-only to redraw the figure from the CSV)
Writes: results/10_karalar/comparison.csv, best_params.json, summary.txt, fig11_karalar_comparison.png
"""
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from config import RAW, PROCESSED, RESULTS, SEED, CHECKPOINT_WEEKS
from train_models import SEARCH, preprocessor, metrics, meta
from plot_style import COLORS, INK, finish

warnings.filterwarnings("ignore")
OUT = RESULTS / "10_karalar"; OUT.mkdir(parents=True, exist_ok=True)
MATERIAL_TYPES = ["oucontent", "subpage", "page", "url", "folder", "dualpane", "htmlactivity",
                  "sharedsubpage", "dataplus", "glossary"]
K_FEATURES = ["quiz_score", "highest_education", "lecture_notes", "materials", "video"]

# Paper 1's own hyperparameter spaces (their Table 4), adapted to current sklearn names
K_SEARCH = {
    "ET": (ExtraTreesClassifier(random_state=SEED, n_jobs=2),
           {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [None, 3],
            "clf__class_weight": ["balanced", "balanced_subsample"], "clf__n_estimators": [100, 200, 300, 400]}),
    "RF": (RandomForestClassifier(random_state=SEED, n_jobs=2),
           {"clf__criterion": ["gini", "entropy"], "clf__bootstrap": [False, True], "clf__max_depth": [None, 3],
            "clf__class_weight": ["balanced", "balanced_subsample"], "clf__n_estimators": [100, 200, 300, 400]}),
    "LR": (LogisticRegression(max_iter=3000),
           {"clf__C": [0.1, 1, 10, 100], "clf__class_weight": ["balanced", None]}),
}
K_WEIGHTS = [1, 2, 1]
N_ITER = 6

daily = pd.read_pickle(PROCESSED / "daily_clicks.pkl")
daily["enr"] = (daily.code_module.astype(str) + "_" + daily.code_presentation.astype(str) + "_"
                + daily.id_student.astype(str))
daily["atype"] = daily.activity_type.astype(str)
assess = pd.read_csv(RAW / "assessments.csv", na_values=["?"])
assess = assess[assess.assessment_type != "Exam"]
sub = pd.read_csv(RAW / "studentAssessment.csv", na_values=["?"]).merge(
    assess[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
sub["enr"] = sub.code_module + "_" + sub.code_presentation + "_" + sub.id_student.astype(str)

def karalar_features(df, t):
    X = df[["enr", "highest_education"]].set_index("enr")
    d = daily[daily.date < t]
    X["lecture_notes"] = d[d.atype == "resource"].groupby("enr").sum_click.sum()
    X["materials"] = d[d.atype.isin(MATERIAL_TYPES)].groupby("enr").sum_click.sum()
    X["video"] = d[d.atype.isin(["ouelluminate", "oucollaborate"])].groupby("enr").sum_click.sum()
    X[["lecture_notes", "materials", "video"]] = X[["lecture_notes", "materials", "video"]].fillna(0)
    X["quiz_score"] = sub[sub.date_submitted < t].groupby("enr").score.mean()  # NaN -> imputed + flagged
    return X[K_FEATURES].reset_index(drop=True)

def tune(space, cols, Xtr, ytr, groups):
    best = {}
    for a, (est, grid) in space.items():
        pipe = Pipeline([("prep", preprocessor(cols)), ("clf", est)])
        n = min(N_ITER, int(np.prod([len(v) for v in grid.values()])))
        s = RandomizedSearchCV(pipe, grid, n_iter=n, scoring="roc_auc", cv=LeaveOneGroupOut(),
                               random_state=SEED, n_jobs=1, refit=True).fit(Xtr, ytr, groups=groups)
        best[a] = (s.best_estimator_, {k.replace("clf__", ""): v for k, v in s.best_params_.items()}, s.best_score_)
    return best

def hard_vote_scores(models, X):
    votes = np.array([m.predict(X) for m in models])          # (3, n)
    w = np.array(K_WEIGHTS)[:, None]
    return (votes * w).sum(axis=0) / w.sum()                   # weighted vote share in [0, 1]

import sys
PLOT_ONLY = "--plot-only" in sys.argv
saved = pd.read_csv(RESULTS / "03_models" / "results.csv")
rows, params = [], {}
for w in ([] if PLOT_ONLY else CHECKPOINT_WEEKS):
    t0 = time.time()
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl").reset_index(drop=True)
    K = karalar_features(df, 7 * w)
    trm, tem = (df.split == "train").values, (df.split == "test").values
    ytr, yte, groups = df.at_risk.values[trm], df.at_risk.values[tem], df.code_presentation.values[trm]
    m3cols = [c for c in meta["groups"][str(w)]["M3"] if df.loc[trm, c].notna().any()]
    sets = {"K-feat": (K, K_FEATURES), "M3": (df, m3cols)}
    for fname, (X, cols) in sets.items():
        Xtr, Xte = X.loc[trm, cols], X.loc[tem, cols]
        # Paper 1 model: ET + RF + LR, hard voting [1,2,1]
        kb = tune(K_SEARCH, cols, Xtr, ytr, groups)
        p = hard_vote_scores([kb[a][0] for a in ["ET", "RF", "LR"]], Xte)
        rows.append({"week": w, "features": fname, "model": "K-model (hard vote ET+RF+LR)", **metrics(yte, p)})
        for a in kb: params[f"w{w}_{fname}_Kmodel_{a}"] = kb[a][1]
        # our model on Paper 1 features (on M3 it already exists from Step 3)
        if fname == "K-feat":
            ob = tune(SEARCH, cols, Xtr, ytr, groups)
            ens = VotingClassifier([(a, ob[a][0]) for a in ob], voting="soft").fit(Xtr, ytr)
            rows.append({"week": w, "features": fname, "model": "Our model (soft vote LR+RF+HGB)",
                         **metrics(yte, ens.predict_proba(Xte)[:, 1])})
            for a in ob: params[f"w{w}_{fname}_ours_{a}"] = ob[a][1]
        else:
            s = saved[(saved.week == w) & (saved.group == "M3") & (saved.algorithm == "ENS")].iloc[0]
            rows.append({"week": w, "features": fname, "model": "Our model (soft vote LR+RF+HGB)",
                         **{k: s[k] for k in ["roc_auc", "recall", "precision", "f1", "balanced_accuracy", "accuracy"]}})
    print(f"week {w:>2} done ({time.time() - t0:.0f}s): " + "; ".join(
        f"{r['features']}/{r['model'][:7]} AUC {r['roc_auc']:.3f}" for r in rows[-4:]), flush=True)

if PLOT_ONLY:
    res = pd.read_csv(OUT / "comparison.csv")
else:
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "comparison.csv", index=False)
    json.dump(params, open(OUT / "best_params.json", "w"), indent=1, default=str)
piv = res.pivot_table(index="week", columns=["features", "model"], values="roc_auc").round(3)
lines = ["Test ROC-AUC (2014J). K-feat = Paper 1 features mapped to OULAD; M3 = this thesis.",
         piv.to_string(), "",
         "Recall / precision at default decision rule (0.5 / majority vote):",
         res.pivot_table(index="week", columns=["features", "model"], values="recall").round(3).to_string(),
         res.pivot_table(index="week", columns=["features", "model"], values="precision").round(3).to_string(), "",
         "Balanced accuracy at default decision rule (fair for hard voting, which has coarse scores):",
         res.pivot_table(index="week", columns=["features", "model"], values="balanced_accuracy").round(3).to_string()]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7, 4.4))
series = [("K-feat", "K-model (hard vote ET+RF+LR)", "Paper 1 as published (features + model)"),
          ("K-feat", "Our model (soft vote LR+RF+HGB)", "Paper 1 features, our model"),
          ("M3", "K-model (hard vote ET+RF+LR)", "Our features, Paper 1 model"),
          ("M3", "Our model (soft vote LR+RF+HGB)", "This thesis (M3 + our model)")]
for i, (f, m, lab) in enumerate(series):
    d = res[(res.features == f) & (res.model == m)].sort_values("week")
    ax.plot(d.week, d.roc_auc, color=COLORS[i], lw=2, marker="o", ms=5, label=lab)
ax.set_xticks(CHECKPOINT_WEEKS); ax.set_xlabel("Checkpoint (weeks since module start)")
ax.set_ylabel("Test ROC-AUC")
finish(fig, ax, "Comparison with Karalar et al. (2021) on the same data and split",
       OUT / "fig11_karalar_comparison.png", ncol=2)
