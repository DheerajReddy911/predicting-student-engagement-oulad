"""Step 9: descriptive analysis of the data, for the Approach chapter.

Run:  python src/eda.py
Writes: results/09_eda/ tables (csv) + figures fig7-fig10 + summary.txt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW, PROCESSED, RESULTS, CHECKPOINT_WEEKS
from plot_style import COLORS, INK, MUTED, finish

OUT = RESULTS / "09_eda"; OUT.mkdir(parents=True, exist_ok=True)
cohort = pd.read_csv(PROCESSED / "cohort.csv")
lines = []
def log(s=""):
    print(s); lines.append(str(s))

# ---- Table: outcomes per module-presentation
t = cohort.pivot_table(index=["code_module", "code_presentation"], columns="final_result",
                       values="id_student", aggfunc="count", fill_value=0)
t = t[["Distinction", "Pass", "Fail", "Withdrawn"]]
t["n"] = t.sum(axis=1)
t["at_risk_rate"] = ((t.Fail + t.Withdrawn) / t.n).round(3)
t["split"] = [cohort[(cohort.code_module == m) & (cohort.code_presentation == p)].split.iloc[0] for m, p in t.index]
t.to_csv(OUT / "table6_outcomes_by_module_presentation.csv")
log("Table 6. Outcomes per module-presentation"); log(t.to_string()); log()

# ---- Table: student background
bg = []
for a in ["gender", "age_band", "highest_education", "disability", "imd_band"]:
    v = cohort[a].fillna("missing")
    g = cohort.groupby(v).agg(n=("at_risk", "size"), at_risk_rate=("at_risk", "mean"))
    g["share"] = g.n / len(cohort); g.insert(0, "attribute", a)
    bg.append(g.reset_index().rename(columns={a: "group"}))
bg = pd.concat(bg).round(3)
bg.to_csv(OUT / "table7_background_profile.csv", index=False)
log("Table 7. Student background and at-risk rate"); log(bg.to_string(index=False)); log()

# ---- Fig 7: outcome mix per module (100% stacked bars)
m = cohort.pivot_table(index="code_module", columns="final_result", values="id_student", aggfunc="count")
m = m[["Distinction", "Pass", "Fail", "Withdrawn"]]
share = m.div(m.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(7, 4))
left = np.zeros(len(share))
for i, c in enumerate(share.columns):
    ax.barh(share.index, share[c], left=left, color=COLORS[i], height=0.6, label=c,
            edgecolor="white", linewidth=1.5)
    for y, (l, w) in enumerate(zip(left, share[c])):
        if w > 0.07: ax.text(l + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=8, color=INK)
    left += share[c].values
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Share of enrolments")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(fig, ax, "Final result by module (all presentations)", OUT / "fig7_outcomes_by_module.png", grid_axis="x")

# ---- Fig 8: mean clicks per week, at-risk vs not (students still enrolled that week)
daily = pd.read_pickle(PROCESSED / "daily_clicks.pkl")
daily["week"] = np.floor(daily.date / 7).astype(int)
wk = daily.groupby(["code_module", "code_presentation", "id_student", "week"], observed=True).sum_click.sum().reset_index()
weeks = range(-3, 39)
rows = []
for w in weeks:
    active = cohort[cohort.date_unregistration.isna() | (cohort.date_unregistration > 7 * w)]
    c = wk[wk.week == w].merge(active[["code_module", "code_presentation", "id_student", "at_risk"]],
                               on=["code_module", "code_presentation", "id_student"], how="right")
    c["sum_click"] = c.sum_click.fillna(0)
    for r, g in c.groupby("at_risk"):
        rows.append({"week": w, "at_risk": r, "mean_clicks": g.sum_click.mean(), "n": len(g),
                     "share_inactive": (g.sum_click == 0).mean()})
cw = pd.DataFrame(rows)
cw.round(3).to_csv(OUT / "table8_weekly_clicks_by_risk.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
for i, (r, lab) in enumerate([(0, "Not at risk (Pass/Distinction)"), (1, "At risk (Fail/Withdrawn)")]):
    d = cw[cw.at_risk == r]
    ax.plot(d.week, d.mean_clicks, color=COLORS[i], lw=2, label=lab)
for w in CHECKPOINT_WEEKS: ax.axvline(w, color="#d6d5d0", lw=0.8, ls=":", zorder=0)
ax.set_xlabel("Week since module start (dotted lines = prediction checkpoints)")
ax.set_ylabel("Mean VLE clicks per student")
finish(fig, ax, "At-risk students click less from the very first weeks", OUT / "fig8_weekly_clicks_by_risk.png", ncol=2)
log("Mean clicks per week (selected weeks):")
log(cw[cw.week.isin([-1, 0, 2, 4, 8, 12, 20, 30])].pivot(index="week", columns="at_risk", values="mean_clicks").round(1).to_string()); log()

# ---- Fig 9: when do withdrawn students leave?
w = cohort[(cohort.final_result == "Withdrawn") & cohort.date_unregistration.notna()].date_unregistration.sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(w.values, np.arange(1, len(w) + 1) / len(w), color=COLORS[0], lw=2)
for k in CHECKPOINT_WEEKS:
    ax.axvline(7 * k, color="#d6d5d0", lw=0.8, ls=":", zorder=0)
ax.set_xlim(-150, 270)  # 61 dates before -150 and 1 data error (day 444, FFF 2013J) fall outside; logged
ax.set_xlabel("Day of unregistration (0 = module start; dotted = checkpoints)")
ax.set_ylabel("Cumulative share of withdrawals")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(fig, ax, "Withdrawals: a quarter happen before the module starts", OUT / "fig9_withdrawal_timing.png", legend=False)
q = {f"<= day {d}": (w <= d).mean() for d in [-1, 14, 28, 56, 84, 140]}
log(f"Unregistration dates outside the plotted range: {(w < -150).sum()} before day -150, "
    f"{(w > 269).sum()} after day 269 (longest module) -> data error, kept in all other steps")
log("Withdrawal timing (share of withdrawn with a date):"); [log(f"  {k}: {v:.1%}") for k, v in q.items()]; log()

# ---- Fig 10: assignment behaviour at week 12 by final result
f12 = pd.read_pickle(PROCESSED / "features_week12.pkl")
order = ["Distinction", "Pass", "Fail", "Withdrawn"]
g = f12.groupby("final_result")[["share_due_not_submitted", "active_days", "days_since_last_activity"]].mean().loc[order]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(order, g.share_due_not_submitted, color=COLORS[0], width=0.55)
for i, v in enumerate(g.share_due_not_submitted): ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=INK)
ax.set_ylabel("Share of due assignments not submitted")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
finish(fig, ax, "By week 12, students who later fail or withdraw have skipped more assignments",
       OUT / "fig10_missed_assignments_week12.png", legend=False)
log("Week 12 means by final result (students still enrolled):"); log(g.round(3).to_string()); log()

# ---- Table: descriptive stats of key M3 features at week 12, by at-risk
key = ["total_clicks", "active_days", "days_since_last_activity", "longest_gap_days",
       "share_weeks_active", "weekly_click_cv", "n_submitted", "share_due_not_submitted", "mean_days_early"]
d = f12.groupby("at_risk")[key].describe(percentiles=[.25, .5, .75]).T
d = d.loc[(slice(None), ["mean", "25%", "50%", "75%"]), :].round(2)
d.columns = ["not_at_risk", "at_risk"]
d.to_csv(OUT / "table9_feature_descriptives_week12.csv")
log("Table 9. Key features at week 12 by at-risk status"); log(d.to_string())
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
