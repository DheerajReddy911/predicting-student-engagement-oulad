# Predicting Student Engagement Using Behavioral Learning Analytics

Code and results for an MSc dissertation (GISMA University of Applied Sciences, module M599).

The project predicts which students are **at risk** (final result *Fail* or *Withdrawn*) from their
behaviour in the virtual learning environment, **early in the course** (weeks 0–12), using the public
Open University Learning Analytics Dataset (OULAD).

## Main results (test presentation 2014J, never used for training or tuning)

| Checkpoint | Background only (M0) | + total clicks (M1) | + weekly pattern (M2) | + engagement behaviour (M3) |
|---|---|---|---|---|
| Week 0  | 0.660 | 0.719 | 0.720 | 0.723 |
| Week 4  | 0.649 | 0.737 | 0.742 | 0.757 |
| Week 8  | 0.647 | 0.751 | 0.776 | 0.801 |
| Week 12 | 0.647 | 0.760 | 0.785 | **0.812** |

ROC-AUC of a soft-voting ensemble (logistic regression + random forest + gradient boosting).
Main model at week 12: 0.812 [95% CI 0.803–0.822]. With a decision threshold chosen on the training
data for 80% recall, it catches 78–85% of at-risk students on the test presentation.
Further analyses: feature importance, DeLong significance tests, fairness across student groups,
comparison with Karalar et al. (2021), robustness checks, a Fail-only label, and calibration.
All tables and figures are in [`results/`](results/).

## Design in brief
- **Unit:** one row per enrolment (student × module × presentation), 32,593 rows.
- **Temporal split:** train on presentations 2013B, 2013J, 2014B; test on 2014J (all 7 modules).
- **Checkpoints:** weeks 0, 2, 4, 6, 8, 10, 12. Only data before the checkpoint is used, and only
  students still enrolled at that point are predicted.
- **Tuning:** randomized search with leave-one-presentation-out cross-validation inside the training data.

## Data
Open University Learning Analytics Dataset (OULAD): Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2017),
*Scientific Data* 4, 170171, https://doi.org/10.1038/sdata.2017.171.
UCI ML Repository ID 349, DOI 10.24432/C5KK69, licence CC BY 4.0.

The data is **not** included in this repository (one file is about 450 MB). Download it from
https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip
and unzip the CSV files into `data/raw/oulad/`.

## Repository structure
```
src/          all code (one script per step; run_all.py runs them in order)
results/      every table (CSV), figure (PNG) and log produced by the scripts
data/raw/     place the OULAD CSV files here (not committed)
requirements.txt
```

## Steps
| Step | Script | Output |
|---|---|---|
| 0. Verify data against documentation | `python src/verify_dataset.py` | `results/00_data_check/data_check_report.txt` |
| 1. Labels, train/test split, cohort per checkpoint | `python src/build_cohort.py` | `data/processed/cohort.csv`, `results/01_cohort/cohort_summary.txt` |
| 2a. Condense click logs | `python src/aggregate_clicks.py` | `data/processed/daily_clicks.pkl` |
| 2b. Build features M0–M3 per checkpoint | `python src/build_features.py` | `data/processed/features_week*.pkl` |
| 2c. Check features against raw data | `python src/check_features.py` | `results/02_features/feature_checks.txt` |
| 3. Train and evaluate all models | `python src/train_models.py` | `results/03_models/` |
| 4. Tables and figures | `python src/summarize_results.py` | `results/04_summary/` |
| 5. Threshold for 80% recall (chosen on training only) | `python src/tune_threshold.py` | `results/05_threshold/` |
| 6. Feature importance (permutation, weeks 4 & 12) | `python src/feature_importance.py` | `results/06_importance/` |
| 7. Significance tests (DeLong, Holm-corrected) | `python src/significance_tests.py` | `results/07_significance/` |
| 8. Fairness across student groups | `python src/fairness.py` | `results/08_fairness/` |
| 9. Descriptive analysis of the data | `python src/eda.py` | `results/09_eda/` |
| 10. Comparison with Karalar et al. (2021) | `python src/compare_karalar.py` | `results/10_karalar/` |
| 11. Robustness: split design, per module, seeds | `python src/robustness.py` | `results/11_robustness/` |
| 12. Alternative label: Fail only | `python src/fail_only.py` | `results/12_fail_only/` |
| 13. Calibration | `python src/calibration.py` | `results/13_calibration/` |

## Run everything
```
pip install -r requirements.txt
python src/run_all.py            # all steps, about 75-90 min on a 2-core laptop
python src/run_all.py --list     # show the steps
python src/run_all.py --from 6   # resume from a step
```
`train_models.py` skips models already listed in `results/03_models/results.csv`; move that folder away
for a fully fresh run.

## Reproducibility
A complete fresh run on a second machine reproduced 24 of 26 result tables exactly; the other two
differed by at most 0.003 AUC because of floating-point differences between CPUs.
See [`results/reproducibility_check.txt`](results/reproducibility_check.txt).
`src/compare_runs.py` performs this comparison.
