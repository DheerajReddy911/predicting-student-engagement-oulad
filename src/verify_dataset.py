"""Step 0: check the downloaded OULAD files against the dataset documentation (OULAD.names).

Run from 03_Implementation/:  python src/verify_dataset.py
Writes: results/00_data_check/data_check_report.txt
"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/oulad")
OUT = Path("results/00_data_check")
OUT.mkdir(parents=True, exist_ok=True)

# Counts stated in OULAD.names, section 5
DOCUMENTED = {
    "studentInfo.csv": 32953,          # "students in courses"
    "courses.csv": 22,                 # "course-presentations"
    "vle.csv": 6364,                   # "VLE pages"
    "studentVle.csv": 10655280,        # "VLE log entries"
    "studentRegistration.csv": 32953,  # "registration entries"
    "assessments.csv": 206,            # "assessments"
    "studentAssessment.csv": 173912,   # "assessment entries"
}

lines = []
def log(s=""):
    print(s); lines.append(str(s))

log("OULAD data check")
log("=" * 60)
log(f"{'file':28}{'documented':>12}{'actual':>12}  match")
tables = {}
for f, expected in DOCUMENTED.items():
    if f == "studentVle.csv":
        n = sum(1 for _ in open(RAW / f, encoding="utf-8")) - 1  # don't load 450MB just to count
    else:
        tables[f] = pd.read_csv(RAW / f, na_values=["?"])  # OULAD marks missing as "?" or blank
        n = len(tables[f])
    log(f"{f:28}{expected:>12,}{n:>12,}  {'OK' if n == expected else 'MISMATCH'}")

si = tables["studentInfo.csv"]
log()
log("studentInfo.csv")
log(f"  columns: {list(si.columns)}")
log(f"  unique students (id_student): {si['id_student'].nunique():,}")
log(f"  student-module-presentation rows: {len(si):,}")
log(f"  modules: {sorted(si['code_module'].unique())}")
log(f"  presentations: {sorted(si['code_presentation'].unique())}")
log()
log("final_result distribution:")
vc = si["final_result"].value_counts()
for k, v in vc.items():
    log(f"  {k:12}{v:>8,}  ({v / len(si):.1%})")
log()
log("Missing values (blank or '?') per table/column:")
for name, t in tables.items():
    for c, v in t.isna().sum().items():
        if v: log(f"  {name} / {c}: {v:,}")
log("  note: date_unregistration is blank for students who never unregistered (expected)")
log()
log("imd_band labels as stored (note '10-20' lacks the % sign):")
log(f"  {sorted(si['imd_band'].dropna().unique())}")
log()
log("courses.csv:")
log(tables["courses.csv"].to_string(index=False))
log()
log("VLE activity types (vle.csv):")
log(f"  {sorted(tables['vle.csv']['activity_type'].unique())}")
log()
log("Assessment types:")
log(tables["assessments.csv"]["assessment_type"].value_counts().to_string())

(OUT / "data_check_report.txt").write_text("\n".join(lines), encoding="utf-8")
