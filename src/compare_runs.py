"""Reproducibility check: compare every result table of two complete runs.

Usage (from the thesis folder):
  python 03_Implementation/src/compare_runs.py 03_Implementation 03_Implementation_run2

For every CSV under results/ it reports whether the file exists in both runs and the largest
numeric difference. Saved test predictions are compared too. Figures (PNG) are not compared
(they can differ in metadata); their numbers are in the CSVs. Run-time columns ("seconds") are ignored.
Writes: <run2>/results/reproducibility_check.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-6
IGNORE = {"seconds"}   # run-time measurements, expected to differ between machines
a, b = Path(sys.argv[1]) / "results", Path(sys.argv[2]) / "results"
lines, worst = [], 0.0
files = sorted({p.relative_to(a) for p in a.rglob("*.csv")} | {p.relative_to(b) for p in b.rglob("*.csv")})
for f in files:
    if not (a / f).exists() or not (b / f).exists():
        lines.append(f"MISSING  {f}  (only in {'run 1' if (a / f).exists() else 'run 2'})"); continue
    x, y = pd.read_csv(a / f), pd.read_csv(b / f)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        lines.append(f"DIFFERENT SHAPE  {f}  {x.shape} vs {y.shape}"); continue
    x, y = x.drop(columns=[c for c in IGNORE if c in x]), y.drop(columns=[c for c in IGNORE if c in y])
    num = x.select_dtypes("number").columns
    d = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if len(num) else 0.0
    same_text = x.drop(columns=num).astype(str).equals(y.drop(columns=num).astype(str))
    worst = max(worst, d)
    lines.append(f"{'OK      ' if d <= TOL and same_text else 'DIFFERS '} {f}  max numeric diff {d:.2e}")
pa, pb = a / "03_models" / "test_predictions.pkl", b / "03_models" / "test_predictions.pkl"
if pa.exists() and pb.exists():
    A, B = pd.read_pickle(pa), pd.read_pickle(pb)
    d = max(float(np.max(np.abs(np.asarray(A[k], float) - np.asarray(B[k], float)))) for k in A)
    worst = max(worst, d)
    lines.append(f"{'OK      ' if d <= TOL else 'DIFFERS '} 03_models/test_predictions.pkl  max diff {d:.2e}")
n_ok = sum(l.startswith("OK") for l in lines)
lines.insert(0, f"Reproducibility check: {n_ok} of {len(lines)} outputs identical (tolerance {TOL}); "
                f"largest numeric difference {worst:.2e}\n")
Path(b / "reproducibility_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
