"""Part A: clean the raw export, build features, write modelling table and EDA evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("outputs", exist_ok=True)
raw_shipments = pd.read_csv("data/raw_shipments.csv")
raw_hubs = pd.read_csv("data/hubs.csv")

# "before" measurements (after de-duplication, before any filling)
dedup = raw_shipments.drop_duplicates()
declared_num = pd.to_numeric(
    dedup["declared_value"].astype("string").str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce")
weight_before = dedup["parcel_weight_kg"]
spell_before = dedup["service_type"].nunique()

clean, medians = clean_shipments(raw_shipments, medians=None)
clean_df = build_features(clean, raw_hubs)

clean_df["is_delayed"] = (clean_df["delivery_hours"] > clean_df["promised_hours"]).astype(int)

assert len(clean_df) == len(clean), "hub merge changed row count"
clean_df.to_csv("data/modelling_table.csv", index=False)

dh = clean_df["delivery_hours"]
Q1, Q3 = dh.quantile(0.25), dh.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
dv = clean_df["declared_value_inr"]
ldv = clean_df["log_declared_value"]

metrics = [
    ("raw_rows", len(raw_shipments)),
    ("duplicate_rows_removed", len(raw_shipments) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weight_before.isna().sum())),
    ("sentinel_weight_rows", int((weight_before > 100).sum())),
    ("missing_declared_value_before", int(declared_num.isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(spell_before)),
    ("service_type_spellings_after", int(clean_df["service_type"].nunique())),
    ("declared_value_mean_inr", dv.mean()),
    ("log_declared_value_mean", ldv.mean()),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(clean_df)),
    ("nulls_in_features_and_target", int(clean_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(clean_df["is_delayed"].sum())),
    ("delay_rate", clean_df["is_delayed"].mean()),
    ("hours_q1", Q1), ("hours_q3", Q3),
    ("hours_lower_bound", lower), ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((dh < lower) | (dh > upper)).sum())),
    ("declared_value_median_inr", dv.median()),
    ("log_declared_value_median", ldv.median()),
]

checks = pd.DataFrame(metrics, columns=["metric", "value"])
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# delay evidence by group
clean_df["distance_band"] = pd.cut(clean_df["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = clean_df.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append([dim, str(grp), int(r["shipments"]), int(r["delayed"]), r["delay_rate"], r["mean_hours"]])
by_group = pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns
corr = clean_df.drop(columns=["distance_band"]).select_dtypes(include=[np.number]).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")
print("Part A done:", len(clean_df), "rows; delayed =", int(clean_df["is_delayed"].sum()))