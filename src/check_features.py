"""Step 2c: sanity checks on the feature tables. Recomputes features for random enrolments
directly from the raw CSVs and compares; prints summary statistics.

Run:  python src/check_features.py
Writes: results/02_features/feature_checks.txt
"""
import numpy as np
import pandas as pd
from config import RAW, PROCESSED, RESULTS

out = RESULTS / "02_features"; out.mkdir(parents=True, exist_ok=True)
lines = []
def log(s=""):
    print(s); lines.append(str(s))

w = 6; t = 7 * w
X = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
sample = X.sample(25, random_state=1)
keys = set(zip(sample.code_module, sample.code_presentation, sample.id_student))

raw = []
for ch in pd.read_csv(RAW / "studentVle.csv", chunksize=3_000_000):
    m = [k in keys for k in zip(ch.code_module, ch.code_presentation, ch.id_student)]
    raw.append(ch[m])
raw = pd.concat(raw)
sa = pd.read_csv(RAW / "studentAssessment.csv", na_values=["?"])
asm = pd.read_csv(RAW / "assessments.csv", na_values=["?"])

mism = 0
for _, r in sample.iterrows():
    rr = raw[(raw.code_module == r.code_module) & (raw.code_presentation == r.code_presentation)
             & (raw.id_student == r.id_student) & (raw.date < t)]
    due = asm[(asm.code_module == r.code_module) & (asm.code_presentation == r.code_presentation)
              & (asm.assessment_type != "Exam") & (asm.date < t)]
    subm = sa[(sa.id_student == r.id_student) & sa.id_assessment.isin(due.id_assessment) & (sa.date_submitted < t)]
    checks = {
        "total_clicks": (rr.sum_click.sum(), r.total_clicks),
        "prestart_clicks": (rr[rr.date < 0].sum_click.sum(), r.prestart_clicks),
        "clicks_week_3": (rr[(rr.date >= 21) & (rr.date < 28)].sum_click.sum(), r.clicks_week_3),
        "active_days": (rr.date.nunique(), r.active_days),
        "n_submitted": (len(subm), r.n_submitted),
    }
    for k, (a, b) in checks.items():
        if not np.isclose(a, b):
            mism += 1; log(f"MISMATCH {r.enr} {k}: raw {a} vs feature {b}")
log(f"Week {w}: recomputed 5 features for 25 random enrolments from raw CSVs -> {mism} mismatches")
log()
for w in [0, 6, 12]:
    X = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    log(f"Week {w}: {len(X):,} rows. Missing values per column (non-zero only):")
    na = X.isna().sum(); na = na[na > 0]
    for c, v in na.items(): log(f"  {c}: {v:,}")
    log(f"  n_submitted > n_assessments_due: {(X.n_submitted > X.n_assessments_due).sum()}")
    log()
X = pd.read_pickle(PROCESSED / "features_week6.pkl")
log("Week 6 medians, at-risk vs not:")
cols = ["total_clicks", "active_days", "days_since_last_activity", "longest_gap_days",
        "share_weeks_active", "n_submitted", "mean_days_early"]
log(X.groupby("at_risk")[cols].median().T.to_string())
(out / "feature_checks.txt").write_text("\n".join(lines), encoding="utf-8")
