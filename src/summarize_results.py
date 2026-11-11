"""Step 4: turn results/03_models/results.csv into thesis tables and figures.

Run:  python src/summarize_results.py
Writes: results/04_summary/*.csv, *.png, summary.txt
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS

IN, OUT = RESULTS / "03_models" / "results.csv", RESULTS / "04_summary"
OUT.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(IN)

# validated categorical palette, fixed order (blue, orange, aqua, yellow)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})
lines = []
def log(s=""):
    print(s); lines.append(str(s))

def line_chart(data, series_col, series_order, metric, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ends = []
    for i, s in enumerate(series_order):
        d = data[data[series_col] == s].sort_values("week")
        ax.plot(d.week, d[metric], color=COLORS[i], lw=2, marker="o", ms=5, label=s)
        ends.append([d[metric].iloc[-1], s])
    # direct labels at line ends, nudged apart so they never overlap
    lo, hi = data[metric].min(), data[metric].max()
    gap = (hi - lo) * 0.06
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    x_end = data.week.max()
    for y, s in ends:
        ax.text(x_end + 0.35, y, s, va="center", color=INK, fontsize=9)
    ax.set_xticks(sorted(data.week.unique()))
    ax.set_xlabel("Checkpoint (weeks since module start)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series_order), fontsize=9)
    ax.set_xlim(right=data.week.max() + 1.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=28)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)

# Table 1: ensemble AUC, week x feature group
t1 = r[r.algorithm == "ENS"].pivot(index="week", columns="group", values="roc_auc").round(3)
t1.to_csv(OUT / "table1_ensemble_auc_week_by_group.csv")
log("Table 1. Test ROC-AUC of the ensemble, by checkpoint and feature group"); log(t1.to_string()); log()

# Table 2: gain from each added group (ensemble)
gain = t1.diff(axis=1).iloc[:, 1:].round(3)
gain.columns = [f"{a}-{b}" for a, b in zip(t1.columns[1:], t1.columns[:-1])]
gain.to_csv(OUT / "table2_auc_gain_per_group.csv")
log("Table 2. AUC gained by adding each feature group (ensemble)"); log(gain.to_string()); log()

# Table 3: algorithms on the full feature set M3
m = ["roc_auc", "recall", "precision", "f1", "balanced_accuracy"]
t3 = r[r.group == "M3"].pivot_table(index="week", columns="algorithm", values="roc_auc").round(3)
t3.to_csv(OUT / "table3_m3_auc_week_by_algorithm.csv")
log("Table 3. Test ROC-AUC on M3, by checkpoint and algorithm"); log(t3.to_string()); log()
for w in [6, 12]:
    t = r[(r.group == "M3") & (r.week == w)].set_index("algorithm")[m].round(3)
    t.to_csv(OUT / f"table4_m3_week{w}_all_metrics.csv")
    log(f"Table 4.{w}. All metrics, M3, week {w}"); log(t.to_string()); log()

# Best model per checkpoint
best = r.loc[r.groupby("week").roc_auc.idxmax(), ["week", "group", "algorithm"] + m].round(3)
best.to_csv(OUT / "table5_best_per_week.csv", index=False)
log("Table 5. Best model per checkpoint (by test ROC-AUC)"); log(best.to_string(index=False)); log()

ens = r[r.algorithm == "ENS"]
line_chart(ens, "group", ["M0", "M1", "M2", "M3"], "roc_auc",
           "Prediction quality rises as behavioural features are added and weeks pass",
           "Test ROC-AUC (ensemble)", "fig1_auc_by_week_and_group.png")
line_chart(ens, "group", ["M0", "M1", "M2", "M3"], "recall",
           "Share of at-risk students caught (recall), ensemble",
           "Recall, at-risk class", "fig2_recall_by_week_and_group.png")
line_chart(r[r.group == "M3"], "algorithm", ["LR", "RF", "HGB", "ENS"], "roc_auc",
           "Algorithms compared on the full feature set (M3)",
           "Test ROC-AUC", "fig3_auc_by_algorithm_m3.png")
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
