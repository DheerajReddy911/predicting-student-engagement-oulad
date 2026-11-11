"""Run the whole pipeline in order. Stops at the first error.

Usage (from 03_Implementation/):
  python src/run_all.py              run every step (about 75-90 minutes on a 2-core laptop)
  python src/run_all.py --from 6     start at step 6 (earlier outputs must already exist)
  python src/run_all.py --only 9     run a single step
  python src/run_all.py --list       show the steps

Requires the raw OULAD CSVs in data/raw/oulad/ (see README).
"""
import subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("0",  "verify_dataset.py",     "Verify downloaded data against its documentation"),
    ("1",  "build_cohort.py",       "Labels, temporal split, cohort per checkpoint"),
    ("2a", "aggregate_clicks.py",   "Condense 10.6M click rows"),
    ("2b", "build_features.py",     "Feature groups M0-M3 per checkpoint"),
    ("2c", "check_features.py",     "Recompute features from raw data (sanity check)"),
    ("3",  "train_models.py",       "Tune and test all 112 models            [long, ~30 min]"),
    ("4",  "summarize_results.py",  "Main tables and figures 1-3"),
    ("5",  "tune_threshold.py",     "Threshold for 80% recall                [~15 min]"),
    ("6",  "feature_importance.py", "Permutation importance                  [~5 min]"),
    ("7",  "significance_tests.py", "DeLong tests and confidence intervals"),
    ("8",  "fairness.py",           "Performance across student groups"),
    ("9",  "eda.py",                "Descriptive analysis of the data"),
    ("10", "compare_karalar.py",    "Comparison with Karalar et al. (2021)   [~17 min]"),
    ("11", "robustness.py",         "Split design, per-module, seed stability [~8 min]"),
    ("12", "fail_only.py",          "Alternative label: Fail only            [~5 min]"),
    ("13", "calibration.py",        "Calibration of predicted risk"),
]

def main(argv):
    ids = [s[0] for s in STEPS]
    if "--list" in argv:
        for i, f, d in STEPS: print(f"{i:>3}  {f:24} {d}")
        return
    todo = STEPS
    if "--only" in argv:
        todo = [s for s in STEPS if s[0] == argv[argv.index("--only") + 1]]
    elif "--from" in argv:
        todo = STEPS[ids.index(argv[argv.index("--from") + 1]):]
    if not todo: sys.exit("unknown step id; use --list")
    t_all = time.time()
    for i, f, d in todo:
        t0 = time.time()
        print(f"\n=== Step {i}: {d} ({f})", flush=True)
        r = subprocess.run([sys.executable, str(SRC / f)], cwd=SRC.parent)
        if r.returncode != 0:
            sys.exit(f"Step {i} ({f}) failed with exit code {r.returncode}")
        print(f"=== Step {i} finished in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll {len(todo)} steps finished in {(time.time() - t_all) / 60:.1f} min")

if __name__ == "__main__":
    main(sys.argv[1:])
