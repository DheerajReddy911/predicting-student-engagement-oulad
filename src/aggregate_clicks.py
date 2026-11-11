"""Step 2a: condense studentVle.csv (10.6M rows) to one row per enrolment x day x activity type.
Clicks on different pages of the same type on the same day are summed. No information
used by the features is lost; this only makes later steps fit in memory.

Run:  python src/aggregate_clicks.py
Writes: data/processed/daily_clicks.pkl
"""
import pandas as pd
from config import RAW, PROCESSED, KEY

vle = pd.read_csv(RAW / "vle.csv", usecols=["id_site", "activity_type"])
site_type = dict(zip(vle.id_site, vle.activity_type))

parts = []
dtypes = {"code_module": "category", "code_presentation": "category", "id_student": "int32",
          "id_site": "int32", "date": "int16", "sum_click": "int32"}
for i, chunk in enumerate(pd.read_csv(RAW / "studentVle.csv", dtype=dtypes, chunksize=2_000_000)):
    chunk["activity_type"] = chunk["id_site"].map(site_type).astype("category")
    g = chunk.groupby(KEY + ["date", "activity_type"], observed=True)["sum_click"].sum().reset_index()
    parts.append(g)
    print(f"chunk {i}: {len(chunk):,} rows -> {len(g):,}")

daily = pd.concat(parts, ignore_index=True)
for c in ["code_module", "code_presentation", "activity_type"]:
    daily[c] = daily[c].astype(str)
# a day can be split across two chunks, so aggregate again
daily = daily.groupby(KEY + ["date", "activity_type"])["sum_click"].sum().reset_index()
for c in ["code_module", "code_presentation", "activity_type"]:
    daily[c] = daily[c].astype("category")
print(f"final: {len(daily):,} rows; total clicks {daily.sum_click.sum():,}")
daily.to_pickle(PROCESSED / "daily_clicks.pkl")
