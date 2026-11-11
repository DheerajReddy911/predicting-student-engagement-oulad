"""Step 13: calibration of the main model (ensemble, M3) on the test presentation.

Does a predicted risk of 70% mean about 70% of such students are at risk? Reports the Brier
score, expected calibration error (ECE, 10 equal-width bins), and the gap between the average
predicted risk and the actual at-risk rate. Base rates of train and test are shown because the
test presentation has a lower at-risk rate, which pushes predictions upwards.

Run:  python src/calibration.py
Writes: results/13_calibration/calibration.csv, calibration_bins.csv, summary.txt,
        fig13_calibration.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from config import PROCESSED, RESULTS, CHECKPOINT_WEEKS
from plot_style import COLORS, INK, MUTED, finish

OUT = RESULTS / "13_calibration"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_pickle(RESULTS / "03_models" / "test_predictions.pkl")
bins = np.linspace(0, 1, 11)
rows, brows = [], []
for w in CHECKPOINT_WEEKS:
    p, y = P[f"w{w}_M3_ENS"], P[f"w{w}_y_test"]
    tr = pd.read_pickle(PROCESSED / f"features_week{w}.pkl")
    tr = tr[tr.split == "train"]
    b = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any())
    rows.append({"week": w, "brier": brier_score_loss(y, p), "brier_base_rate_only": brier_score_loss(y, np.full_like(p, y.mean())),
                 "ece": ece, "mean_predicted": p.mean(), "test_at_risk_rate": y.mean(), "train_at_risk_rate": tr.at_risk.mean()})
    for k in range(10):
        if (b == k).any():
            brows.append({"week": w, "bin": f"{bins[k]:.1f}-{bins[k+1]:.1f}", "n": int((b == k).sum()),
                          "mean_predicted": p[b == k].mean(), "observed_rate": y[b == k].mean()})
res, cb = pd.DataFrame(rows), pd.DataFrame(brows)
res.round(4).to_csv(OUT / "calibration.csv", index=False)
cb.round(4).to_csv(OUT / "calibration_bins.csv", index=False)
lines = ["Calibration, ensemble M3, test 2014J", res.round(3).to_string(index=False), "",
         "Reliability bins, weeks 4 and 12:", cb[cb.week.isin([4, 12])].round(3).to_string(index=False)]
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(5.6, 5))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
for i, w in enumerate([4, 12]):
    d = cb[(cb.week == w) & (cb.n >= 30)]
    ax.plot(d.mean_predicted, d.observed_rate, color=COLORS[i], lw=2, marker="o", ms=5, label=f"Week {w}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Mean predicted risk (bin)"); ax.set_ylabel("Observed at-risk rate")
finish(fig, ax, "Calibration of predicted risk (ensemble, M3, test 2014J)", OUT / "fig13_calibration.png", ncol=3)
