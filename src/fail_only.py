"""Step 12: alternative outcome definition, Fail only.

Withdrawn students are removed (train and test); the target is Fail (1) vs Pass/Distinction (0).
This answers "is the model only detecting withdrawal?". Note that in real use one cannot know in
advance who will withdraw, so this is a sensitivity analysis, not a deployable setting.
Uses Step 3's tuned hyperparameters (not re-tuned for the new target), for M0 and M3.

Run:  python src/fail_only.py
Writes: results/12_fail_only/fail_only.csv, summary.txt
"""
import pandas as pd
from config import PROCESSED, RESULTS, CHECKPOINT_WEEKS
from refit import tuned_ensemble, usable_cols
from train_models import metrics

OUT = RESULTS / "12_fail_only"; OUT.mkdir(parents=True, exist_ok=True)
saved = pd.read_csv(RESULTS / "03_models" / "results.csv")
rows = []
for w in CHECKPOINT_WEEKS:
    df = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    df = df[df.final_result != "Withdrawn"].copy()
    df["fail"] = (df.final_result == "Fail").astype(int)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    for g in ["M0", "M3"]:
        cols = usable_cols(tr, w, g)
        m = tuned_ensemble(w, g, cols).fit(tr[cols], tr.fail)
        r = metrics(te.fail.values, m.predict_proba(te[cols])[:, 1])
        main = saved[(saved.week == w) & (saved.group == g) & (saved.algorithm == "ENS")].roc_auc.iloc[0]
        rows.append({"week": w, "group": g, "n_train": len(tr), "n_test": len(te), "fail_rate_test": te.fail.mean(),
                     "fail_only_auc": r["roc_auc"], "main_label_auc": main, **{k: r[k] for k in ["recall", "precision", "f1"]}})
    print(f"week {w} done: M0 {rows[-2]['fail_only_auc']:.3f}, M3 {rows[-1]['fail_only_auc']:.3f}", flush=True)
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "fail_only.csv", index=False)
t = res.pivot(index="week", columns="group", values=["fail_only_auc", "main_label_auc"]).round(3)
lines = ["Test ROC-AUC: Fail-only label (Withdrawn removed) vs main label (Fail+Withdrawn), ensemble",
         t.to_string(), "", res.round(3).to_string(index=False)]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:2]))
