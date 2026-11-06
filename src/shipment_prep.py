"""Shared constants and Data preparation functions used by every part of the pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]

# It takes a raw shipments DataFrame and returns a tuple (clean_df, medians)
def clean_shipments(raw, medians=None) :
  clean_df = raw.copy()

  # 1. drop exact duplicate rows;
  clean_df = raw.drop_duplicates().reset_index(drop=True)
  # 2. service_type strips & use title case
  clean_df["service_type"] = (clean_df["service_type"].str.strip().str.title())
  # 3. Set missing payment_mode to "Unknown"
  clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
  # 4. declared_value: Remove rupee sign and commas -> declared_value_inr
  cleaned_value_inr = (
      clean_df["declared_value"]
      .astype("string")
      .str.replace("₹", "", regex=False)
      .str.replace(",", "", regex=False)
      .str.strip()
  )
  clean_df["declared_value_inr"] = pd.to_numeric(cleaned_value_inr, errors="coerce").astype(float)
  # Dropped declared_value column
  clean_df = clean_df.drop(columns=["declared_value"])
  # 5. dispatch_time: convert to a datetime;
  clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
  # 6. Set sentinel weights (> 100) -> missing
  clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

  # Find the median for distance_km, parcel_weight_kg, declared_value_inr
  medians = {
      "distance_km":
          round(float(clean_df["distance_km"].median()),2),
      "parcel_weight_kg":
          round(float(clean_df["parcel_weight_kg"].median()),2),
      "declared_value_inr":
          round(float(clean_df["declared_value_inr"].median()),2)
  }

  #7. Check & get the medians
  if medians is None:
      medians = {c: float(clean_df[c].median()) for c in MEDIAN_COLUMNS}

  # Fill median respective column's missing values with their medians
  for column in MEDIAN_COLUMNS:
      clean_df[column] = clean_df[column].fillna(medians[column])

  return clean_df, medians

# Attach hub columns and derive model features. Never uses delivery_hours.
def build_features(clean, hubs) :
  clean["promised_hours"] = clean["service_type"].map(PROMISED_HOURS).fillna(np.nan)

  clean["is_cod"] = (
        clean["payment_mode"] == "COD"
    ).astype(int)

  clean["is_weekend"] = (
        clean["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

  clean["dispatch_hour"] = (
        clean["dispatch_time"].dt.hour
    ).astype(int)

  clean["log_declared_value"] = np.log1p(clean["declared_value_inr"])

  merged = clean.merge(hubs, on="hub_id", how="left", validate="many_to_one")

  return merged