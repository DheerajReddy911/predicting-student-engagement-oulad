"""Step 3: train and evaluate every (checkpoint x feature group x algorithm) combination.

Algorithms (decision D12): logistic regression (baseline), random forest, histogram gradient
boosting, and a soft-voting ensemble of the three tuned models (cf. Karalar et al., 2021).

Tuning: randomized search inside the TRAINING presentations only, with leave-one-presentation-out
cross-validation (3 folds: 2013B, 2013J, 2014B), scored by ROC-AUC. This keeps the temporal logic
of the train/test split inside tuning too. The test presentation (2014J) is used once per model.

Metrics (decision D13): ROC-AUC (headline), recall / precision / F1 for the at-risk class,
balanced accuracy, accuracy. Threshold 0.5.

Run:  python src/train_models.py            (resumes: finished combinations are skipped)
Writes: results/03_models/results.csv, best_params.json, test_predictions.pkl, train_log.txt
"""
import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score,
                             balanced_accuracy_score, accuracy_score)
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import PROCESSED, RESULTS, SEED, CHECKPOINT_WEEKS

warnings.filterwarnings("ignore", category=UserWarning)
OUT = RESULTS / "03_models"; OUT.mkdir(parents=True, exist_ok=True)
RES_FILE, PARAM_FILE, PRED_FILE = OUT / "results.csv", OUT / "best_params.json", OUT / "test_predictions.pkl"
meta = json.load(open(PROCESSED / "feature_groups.json"))
CAT = set(meta["categorical"])
GROUPS = ["M0", "M1", "M2", "M3"]

SEARCH = {
    "LR": (LogisticRegression(max_iter=3000), {"clf__C": [0.01, 0.1, 1, 10]}),
    "RF": (RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=SEED),
           {"clf__max_depth": [None, 12, 20], "clf__min_samples_leaf": [1, 5, 20],
            "clf__max_features": ["sqrt", 0.3]}),
    "HGB": (HistGradientBoostingClassifier(random_state=SEED),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
             "clf__l2_regularization": [0.0, 1.0], "clf__min_samples_leaf": [20, 50]}),
}
N_ITER = 6

def log(s):
    line = f"{time.strftime('%H:%M:%S')}  {s}"
    print(line, flush=True)
    with open(OUT / "train_log.txt", "a", encoding="utf-8") as f: f.write(line + "\n")

def preprocessor(cols):
    cat = [c for c in cols if c in CAT]
    num = [c for c in cols if c not in CAT]
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num)])

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "recall": recall_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat),
            "balanced_accuracy": balanced_accuracy_score(y, yhat), "accuracy": accuracy_score(y, yhat)}

def main():
    done = pd.read_csv(RES_FILE) if RES_FILE.exists() else pd.DataFrame()
    params = json.load(open(PARAM_FILE)) if PARAM_FILE.exists() else {}
    preds = pd.read_pickle(PRED_FILE) if PRED_FILE.exists() else {}
    is_done = lambda w, g, a: len(done) and ((done.week == w) & (done.group == g) & (done.algorithm == a)).any()

    for w in CHECKPOINT_WEEKS:
        df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
        tr, te = df[df.split == "train"], df[df.split == "test"]
        for g in GROUPS:
            if all(is_done(w, g, a) for a in list(SEARCH) + ["ENS"]): continue
            cols = [c for c in meta["groups"][str(w)][g] if tr[c].notna().any()]
            Xtr, ytr, Xte, yte = tr[cols], tr.at_risk.values, te[cols], te.at_risk.values
            best = {}
            for a, (est, grid) in SEARCH.items():
                t0 = time.time()
                pipe = Pipeline([("prep", preprocessor(cols)), ("clf", est)])
                n_iter = min(N_ITER, int(np.prod([len(v) for v in grid.values()])))
                search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=LeaveOneGroupOut(),
                                            random_state=SEED, n_jobs=1 if a == "RF" else 2, refit=True)
                search.fit(Xtr, ytr, groups=tr.code_presentation.values)
                best[a] = search.best_estimator_
                p = best[a].predict_proba(Xte)[:, 1]
                row = {"week": w, "group": g, "algorithm": a, "n_features": len(cols), "n_train": len(tr),
                       "n_test": len(te), "cv_roc_auc": search.best_score_, **metrics(yte, p),
                       "seconds": round(time.time() - t0, 1)}
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                params[f"w{w}_{g}_{a}"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                preds[f"w{w}_{g}_{a}"] = p
                log(f"week {w:>2} {g} {a:4} test AUC {row['roc_auc']:.3f} recall {row['recall']:.3f} "
                    f"F1 {row['f1']:.3f} (cv {search.best_score_:.3f}, {row['seconds']}s)")
            t0 = time.time()
            ens = VotingClassifier([(a, m) for a, m in best.items()], voting="soft").fit(Xtr, ytr)
            p = ens.predict_proba(Xte)[:, 1]
            row = {"week": w, "group": g, "algorithm": "ENS", "n_features": len(cols), "n_train": len(tr),
                   "n_test": len(te), "cv_roc_auc": np.nan, **metrics(yte, p), "seconds": round(time.time() - t0, 1)}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            preds[f"w{w}_{g}_ENS"] = p
            preds[f"w{w}_y_test"] = yte
            log(f"week {w:>2} {g} ENS  test AUC {row['roc_auc']:.3f} recall {row['recall']:.3f} F1 {row['f1']:.3f}")
            done.to_csv(RES_FILE, index=False)
            json.dump(params, open(PARAM_FILE, "w"), indent=1, default=str)
            pd.to_pickle(preds, PRED_FILE)
    log("ALL DONE")


if __name__ == "__main__":
    main()
