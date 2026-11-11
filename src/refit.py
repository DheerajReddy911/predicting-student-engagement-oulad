"""Helper: rebuild a tuned model from Step 3's best hyperparameters (no new tuning)."""
import json
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from config import RESULTS
from train_models import SEARCH, preprocessor, meta

PARAMS = json.load(open(RESULTS / "03_models" / "best_params.json"))

def usable_cols(df_train, week, group):
    return [c for c in meta["groups"][str(week)][group] if df_train[c].notna().any()]

def tuned_ensemble(week, group, cols, seed=None):
    """Soft-voting LR+RF+HGB with the Step 3 hyperparameters for (week, group).
    seed overrides random_state of RF and HGB (used for the stability check)."""
    members = []
    for a, (est, _) in SEARCH.items():
        pipe = Pipeline([("prep", preprocessor(cols)), ("clf", clone(est))])
        p = {f"clf__{k}": v for k, v in PARAMS[f"w{week}_{group}_{a}"].items()}
        if seed is not None and a in ("RF", "HGB"): p["clf__random_state"] = seed
        members.append((a, pipe.set_params(**p)))
    return VotingClassifier(members, voting="soft")
