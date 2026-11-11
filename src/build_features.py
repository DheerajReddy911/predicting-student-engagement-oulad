"""Step 2b: build the four nested feature groups for every checkpoint.

M0  background      : module, demographics, prior attempts, credits, registration day
M1  + total clicks  : total VLE clicks before the checkpoint
M2  + weekly pattern: pre-start clicks, clicks per course week, last-2-weeks clicks, trend
M3  + engagement    : active days, recency, longest gap, regularity, resource-type mix,
                      assignment submission behaviour (on time / late / not submitted)

Leakage rule: at checkpoint week w (cut-off day t = 7w) only records with date < t are used,
and only students still enrolled at day t are included (decision D10).

Run:  python src/build_features.py
Writes: data/processed/features_week{w}.pkl, data/processed/feature_groups.json
"""
import json
import numpy as np
import pandas as pd
from config import RAW, PROCESSED, CHECKPOINT_WEEKS

# activity types with >= 1% of all clicks; the rest are pooled as "other" (see research log)
MAIN_TYPES = ["oucontent", "forumng", "quiz", "homepage", "subpage", "resource", "ouwiki", "url"]

def enr_id(df):
    return df.code_module.astype(str) + "_" + df.code_presentation.astype(str) + "_" + df.id_student.astype(str)

cohort = pd.read_csv(PROCESSED / "cohort.csv")
cohort["enr"] = enr_id(cohort)
daily = pd.read_pickle(PROCESSED / "daily_clicks.pkl")
for c in ["code_module", "code_presentation"]:
    daily[c] = daily[c].astype(str)
daily["enr"] = enr_id(daily)
daily["type_group"] = np.where(daily.activity_type.astype(str).isin(MAIN_TYPES),
                               daily.activity_type.astype(str), "other")

assess = pd.read_csv(RAW / "assessments.csv", na_values=["?"])
assess = assess[assess.assessment_type != "Exam"]  # exams are at the end of the module
sub = pd.read_csv(RAW / "studentAssessment.csv", na_values=["?"])
E = "enr"

M0 = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band",
      "disability", "num_of_prev_attempts", "studied_credits", "date_registration"]
CATEGORICAL = ["code_module", "gender", "region", "highest_education", "imd_band",
               "age_band", "disability"]
groups = {}

def per_enrolment(df, col_name, agg):
    return df.groupby(E)[col_name].agg(agg)

for w in CHECKPOINT_WEEKS:
    t = 7 * w
    active = cohort[cohort.date_unregistration.isna() | (cohort.date_unregistration > t)].copy()
    X = active[[E, "code_presentation", "id_student", "split", "at_risk", "final_result"] + M0].set_index(E)
    d = daily[daily.date < t]

    # ---- M1: total clicks
    X["total_clicks"] = per_enrolment(d, "sum_click", "sum")
    m1 = ["total_clicks"]

    # ---- M2: weekly pattern
    X["prestart_clicks"] = per_enrolment(d[d.date < 0], "sum_click", "sum")
    wk_cols = []
    if w > 0:
        dw = d[d.date >= 0].assign(week=lambda x: x.date // 7)
        weekly = dw.groupby([E, "week"])["sum_click"].sum().unstack("week")
        weekly = weekly.reindex(columns=range(w))
        weekly.columns = [f"clicks_week_{k}" for k in range(w)]
        X = X.join(weekly)
        wk_cols = list(weekly.columns)
    m2 = ["prestart_clicks"] + wk_cols
    X[["total_clicks", "prestart_clicks"] + wk_cols] = X[["total_clicks", "prestart_clicks"] + wk_cols].fillna(0)
    if w >= 2:
        X["clicks_last_2_weeks"] = X[wk_cols[-2:]].sum(axis=1)
        wk = X[wk_cols].to_numpy(dtype=float)
        x = np.arange(w) - (w - 1) / 2
        X["weekly_click_trend"] = (wk * x).sum(axis=1) / (x ** 2).sum()  # OLS slope, clicks/week
        m2 += ["clicks_last_2_weeks", "weekly_click_trend"]

    # ---- M3: engagement behaviour
    days = d.groupby([E, "date"])["sum_click"].sum().reset_index()
    X["active_days"] = days.groupby(E).size()
    X["active_days"] = X["active_days"].fillna(0)
    last = days.groupby(E)["date"].max()
    X["days_since_last_activity"] = t - last  # NaN if never active -> imputed + flagged in model
    days = days.sort_values([E, "date"])
    X["longest_gap_days"] = days.groupby(E)["date"].diff().groupby(days[E]).max()
    m3 = ["active_days", "days_since_last_activity", "longest_gap_days"]
    if w >= 2:
        wk = X[wk_cols].to_numpy(dtype=float)
        mean = wk.mean(axis=1)
        X["weekly_click_cv"] = np.where(mean > 0, wk.std(axis=1) / np.where(mean > 0, mean, 1), np.nan)
        X["share_weeks_active"] = (wk > 0).mean(axis=1)
        m3 += ["weekly_click_cv", "share_weeks_active"]
    by_type = d.groupby([E, "type_group"])["sum_click"].sum().unstack("type_group")
    by_type = by_type.reindex(columns=MAIN_TYPES + ["other"]).fillna(0)
    share = by_type.div(by_type.sum(axis=1).replace(0, np.nan), axis=0)
    share.columns = [f"share_clicks_{c}" for c in share.columns]
    X = X.join(share)
    X["n_resource_types_used"] = d.groupby(E)["activity_type"].nunique()
    X["n_resource_types_used"] = X["n_resource_types_used"].fillna(0)
    m3 += list(share.columns) + ["n_resource_types_used"]

    # assignment behaviour: assessments due before t for this module-presentation
    due = assess[assess.date < t][["code_module", "code_presentation", "id_assessment", "date"]]
    s = sub[sub.date_submitted < t].merge(due, on="id_assessment")
    s["days_early"] = s["date"] - s["date_submitted"]
    s[E] = enr_id(s)
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_assessments_due")
    X = X.join(n_due, on=["code_module", "code_presentation"])
    X["n_assessments_due"] = X["n_assessments_due"].fillna(0)
    sg = s.groupby(E)
    X["n_submitted"] = sg.size()
    X["n_submitted_on_time"] = sg["days_early"].apply(lambda v: (v >= 0).sum())
    X["mean_days_early"] = sg["days_early"].mean()
    X[["n_submitted", "n_submitted_on_time"]] = X[["n_submitted", "n_submitted_on_time"]].fillna(0)
    X["share_due_not_submitted"] = np.where(
        X.n_assessments_due > 0, 1 - X.n_submitted.clip(upper=X.n_assessments_due) / X.n_assessments_due.replace(0, 1), np.nan)
    m3 += ["n_assessments_due", "n_submitted", "n_submitted_on_time", "mean_days_early",
           "share_due_not_submitted"]

    X = X.reset_index()
    X.to_pickle(PROCESSED / f"features_week{w}.pkl")
    groups[w] = {"M0": M0, "M1": M0 + m1, "M2": M0 + m1 + m2, "M3": M0 + m1 + m2 + m3}
    print(f"week {w:>2}: {len(X):,} enrolments | features M0 {len(M0)}, M1 {len(groups[w]['M1'])}, "
          f"M2 {len(groups[w]['M2'])}, M3 {len(groups[w]['M3'])}")

json.dump({"groups": groups, "categorical": CATEGORICAL}, open(PROCESSED / "feature_groups.json", "w"), indent=1)
