"""Step 1: build labelled enrolment table, assign train/test split, and count who is
still enrolled at each checkpoint.

Run:  python src/build_cohort.py
Writes: data/processed/cohort.csv, results/01_cohort/cohort_summary.txt
"""
import pandas as pd
from config import RAW, PROCESSED, RESULTS, AT_RISK, KEY, TRAIN_PRESENTATIONS, TEST_PRESENTATIONS, CHECKPOINT_WEEKS

out = RESULTS / "01_cohort"; out.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

info = pd.read_csv(RAW / "studentInfo.csv", na_values=["?"])
reg = pd.read_csv(RAW / "studentRegistration.csv", na_values=["?"])
cohort = info.merge(reg, on=KEY, how="left", validate="one_to_one")

cohort["at_risk"] = cohort["final_result"].map(AT_RISK)
assert cohort["at_risk"].notna().all()
cohort["split"] = cohort["code_presentation"].map(
    {**{p: "train" for p in TRAIN_PRESENTATIONS}, **{p: "test" for p in TEST_PRESENTATIONS}})
assert cohort["split"].notna().all()

lines = []
def log(s=""):
    print(s); lines.append(str(s))

log("Cohort summary")
log("=" * 60)
log(f"Enrolments: {len(cohort):,}")
log(f"At-risk (Fail+Withdrawn): {cohort.at_risk.sum():,} ({cohort.at_risk.mean():.1%})")
log()
log("Split:")
for s, g in cohort.groupby("split"):
    log(f"  {s:6} {len(g):>7,} enrolments, at-risk {g.at_risk.mean():.1%}, "
        f"modules {sorted(g.code_module.unique())}")
log()

# Withdrawn students: when did they leave?
w = cohort[cohort.final_result == "Withdrawn"]
log(f"Withdrawn: {len(w):,}; with an unregistration date: {w.date_unregistration.notna().sum():,}")
log(f"  left before day 0: {(w.date_unregistration < 0).sum():,}")
for wk in CHECKPOINT_WEEKS[1:]:
    log(f"  left by week {wk:>2}: {(w.date_unregistration <= 7*wk).sum():,}")
log(f"Non-withdrawn with an unregistration date (data quirk): "
    f"{cohort[(cohort.final_result != 'Withdrawn') & cohort.date_unregistration.notna()].shape[0]:,}")
log()

# D10: at checkpoint t, only students still enrolled at t are predicted
log("Students still enrolled at each checkpoint (these are the ones we predict for):")
log(f"{'week':>5}{'train n':>10}{'train risk':>12}{'test n':>10}{'test risk':>12}")
for wk in CHECKPOINT_WEEKS:
    t = 7 * wk
    active = cohort[cohort.date_unregistration.isna() | (cohort.date_unregistration > t)]
    tr, te = active[active.split == "train"], active[active.split == "test"]
    log(f"{wk:>5}{len(tr):>10,}{tr.at_risk.mean():>12.1%}{len(te):>10,}{te.at_risk.mean():>12.1%}")

cohort.to_csv(PROCESSED / "cohort.csv", index=False)
(out / "cohort_summary.txt").write_text("\n".join(lines), encoding="utf-8")
