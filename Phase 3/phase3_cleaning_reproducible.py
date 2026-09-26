import numpy as np
import pandas as pd

SOURCE = "final_internship_data.xlsx"

# 1) Preserve source exactly.
df_raw = pd.read_excel(SOURCE, engine="openpyxl")
df_clean = df_raw.copy(deep=True)

# 2) Remove invalid target rows only.
df_clean = df_clean.loc[df_clean["fare_amount"] > 0].copy()

# 3) Restore pickup_datetime from Excel serials when necessary.
if pd.api.types.is_numeric_dtype(df_clean["pickup_datetime"]):
    df_clean["pickup_datetime"] = pd.to_datetime(
        df_clean["pickup_datetime"], unit="D", origin="1899-12-30"
    ).dt.round("s")
else:
    df_clean["pickup_datetime"] = pd.to_datetime(
        df_clean["pickup_datetime"], errors="coerce"
    ).dt.round("s")

# 4) Passenger count 0 is treated as an unknown/missing value, not a reason to drop the trip.
df_clean.loc[df_clean["passenger_count"] == 0, "passenger_count"] = np.nan

# 5) Actual coordinate representation is retained as-is (it is consistent with radians).
#    Zero coordinates are treated as sentinel missing values.
coords = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude"
]
for c in coords:
    df_clean.loc[df_clean[c] == 0, c] = np.nan

# 6) Values outside the mathematical radian coordinate domain are invalid.
for c in ["pickup_longitude", "dropoff_longitude"]:
    df_clean.loc[~df_clean[c].between(-np.pi, np.pi), c] = np.nan
for c in ["pickup_latitude", "dropoff_latitude"]:
    df_clean.loc[~df_clean[c].between(-np.pi / 2, np.pi / 2), c] = np.nan

geo_affected = df_clean[coords].isna().any(axis=1)
geo_derived = [
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist",
    "nyc_dist", "distance", "bearing"
]
df_clean.loc[geo_affected, geo_derived] = np.nan

# 7) Keep ordinary statistical outliers; invalidate only clearly implausible NYC-scale distances.
df_clean.loc[df_clean["distance"] > 1000, "distance"] = np.nan
for c in ["jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist"]:
    df_clean.loc[df_clean[c] > 1000, c] = np.nan

# Deliberately NOT done:
# - no Key-based deduplication (Key == pickup_datetime and is not unique)
# - no radians/degrees conversion
# - no category recoding
# - no column renaming
# - no Week column creation
# - no automatic IQR-outlier deletion
