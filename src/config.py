"""Fixed project settings. Every decision here is logged in 00_RESEARCH_LOG.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 03_Implementation/
RAW = ROOT / "data/raw/oulad"
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"
SEED = 42

# D6: at-risk = Fail or Withdrawn (student decision, 26 Oct 2026)
AT_RISK = {"Fail": 1, "Withdrawn": 1, "Pass": 0, "Distinction": 0}

# D7: unit of analysis = one row per enrolment (student x module x presentation)
KEY = ["code_module", "code_presentation", "id_student"]

# D8: temporal split. Train on earlier presentations, test on the latest one (2014J).
# 2014J contains all 7 modules, so every module is tested on a presentation the model never saw.
TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
TEST_PRESENTATIONS = ["2014J"]

# D9: early-prediction checkpoints, in weeks since module start (day 0).
# Week 0 = before teaching starts (demographics, registration, pre-start activity only).
CHECKPOINT_WEEKS = [0, 2, 4, 6, 8, 10, 12]
