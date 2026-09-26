import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Phase 6 uses the approved Phase 3 dataset as read-only input.
INPUT = "../phase3/df_clean.csv"
OUT = "."
os.makedirs(OUT, exist_ok=True)

df_clean = pd.read_csv(INPUT, parse_dates=["pickup_datetime"])

# Basic integrity check: no mutation is performed.
print("df_clean shape:", df_clean.shape)

# ---------------------------------------------------------------------
# P6-Q1. How does ride frequency vary across hour and day of week?
# ---------------------------------------------------------------------
p6q1_counts = (
    df_clean.groupby(["weekday", "hour"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=range(7), columns=range(24), fill_value=0)
)
p6q1_counts.to_csv(os.path.join(OUT, "P6_Q1_hour_weekday_counts.csv"))

plt.figure(figsize=(13, 5.5))
plt.imshow(p6q1_counts.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="Number of trips")
plt.xlabel("Pickup hour")
plt.ylabel("Weekday (0=Monday, 6=Sunday)")
plt.xticks(range(24))
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.title("P6-Q1: Trip Frequency by Hour and Day of Week")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q1_hour_weekday_demand.png"), dpi=160)
plt.close()

p6q1_long = p6q1_counts.stack().rename("trip_count").reset_index()
p6q1_peak = p6q1_long.loc[p6q1_long["trip_count"].idxmax()]
p6q1_low = p6q1_long.loc[p6q1_long["trip_count"].idxmin()]
print("P6-Q1 peak cell:", p6q1_peak.to_dict())
print("P6-Q1 lowest cell:", p6q1_low.to_dict())
print("P6-Q1 weekday totals:\n", df_clean.groupby("weekday").size())

# ---------------------------------------------------------------------
# P6-Q2. Does typical fare vary across days of the week?
# ---------------------------------------------------------------------
p6q2 = (
    df_clean.groupby("weekday")["fare_amount"]
    .agg(count="count", mean="mean", median="median")
    .reindex(range(7))
)
p6q2.to_csv(os.path.join(OUT, "P6_Q2_weekday_fare.csv"))

plt.figure(figsize=(10, 5.5))
x = np.arange(7)
plt.bar(x - 0.18, p6q2["mean"], width=0.36, label="Mean fare")
plt.bar(x + 0.18, p6q2["median"], width=0.36, label="Median fare")
plt.xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.xlabel("Day of week")
plt.ylabel("Fare amount")
plt.title("P6-Q2: Fare by Day of Week")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q2_weekday_fare.png"), dpi=160)
plt.close()

p6q2_mean_high = p6q2["mean"].idxmax()
p6q2_mean_low = p6q2["mean"].idxmin()
p6q2_median_high = p6q2["median"].idxmax()
p6q2_median_low = p6q2["median"].idxmin()
print("P6-Q2:\n", p6q2)
print("P6-Q2 mean range:", p6q2["mean"].max() - p6q2["mean"].min())
print("P6-Q2 mean high/low weekdays:", p6q2_mean_high, p6q2_mean_low)
print("P6-Q2 median high/low weekdays:", p6q2_median_high, p6q2_median_low)

# ---------------------------------------------------------------------
# P6-Q3. Does traffic/fare relationship vary by pickup hour?
# ---------------------------------------------------------------------
p6q3_mean = (
    df_clean.groupby(["hour", "Traffic Condition"])["fare_amount"]
    .mean()
    .unstack()
    .reindex(range(24))
)
p6q3_count = (
    df_clean.groupby(["hour", "Traffic Condition"])["fare_amount"]
    .count()
    .unstack()
    .reindex(range(24))
)
p6q3_mean.to_csv(os.path.join(OUT, "P6_Q3_traffic_hour_mean_fare.csv"))
p6q3_count.to_csv(os.path.join(OUT, "P6_Q3_traffic_hour_counts.csv"))

plt.figure(figsize=(12, 5.5))
for col in p6q3_mean.columns:
    plt.plot(p6q3_mean.index, p6q3_mean[col], marker="o", markersize=3, label=col)
plt.xlabel("Pickup hour")
plt.ylabel("Mean fare amount")
plt.title("P6-Q3: Mean Fare by Traffic Condition and Pickup Hour")
plt.xticks(range(24))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q3_traffic_hour_fare.png"), dpi=160)
plt.close()

# Within-hour spread across traffic categories; only hours with all 3 categories are compared.
p6q3_spread = p6q3_mean.max(axis=1) - p6q3_mean.min(axis=1)
p6q3_spread.name = "traffic_mean_fare_spread"
p6q3_spread.to_csv(os.path.join(OUT, "P6_Q3_hourly_traffic_spread.csv"), header=True)
print("P6-Q3 hourly mean fare table:\n", p6q3_mean)
print("P6-Q3 largest within-hour traffic mean spread:", p6q3_spread.max(), "at hour", p6q3_spread.idxmax())
print("P6-Q3 smallest within-hour traffic mean spread:", p6q3_spread.min(), "at hour", p6q3_spread.idxmin())

# ---------------------------------------------------------------------
# P6-Q4. How does fare-distance relationship behave across distance ranges,
# and does fare per km vary with trip length?
# ---------------------------------------------------------------------
p6q4 = df_clean[["distance", "fare_amount"]].dropna().copy()
p6q4 = p6q4[p6q4["distance"] > 0].copy()
p6q4["fare_per_km"] = p6q4["fare_amount"] / p6q4["distance"]

# Fixed, interpretable distance ranges; no source-data modification.
bins = [0, 1, 2, 5, 10, 20, np.inf]
labels = ["0–<1", "1–<2", "2–<5", "5–<10", "10–<20", "20+"]
p6q4["distance_range"] = pd.cut(
    p6q4["distance"], bins=bins, labels=labels, right=False, include_lowest=True
)
p6q4_summary = (
    p6q4.groupby("distance_range", observed=False)
    .agg(
        count=("fare_amount", "count"),
        mean_distance=("distance", "mean"),
        mean_fare=("fare_amount", "mean"),
        median_fare=("fare_amount", "median"),
        mean_fare_per_km=("fare_per_km", "mean"),
        median_fare_per_km=("fare_per_km", "median"),
    )
)
p6q4_summary.to_csv(os.path.join(OUT, "P6_Q4_distance_range_summary.csv"))

# Scatterplot of all valid positive-distance observations, with transparency only.
plt.figure(figsize=(10, 6))
plt.scatter(p6q4["distance"], p6q4["fare_amount"], s=4, alpha=0.05)
plt.xlabel("Trip Distance (km)")
plt.ylabel("Fare Amount")
plt.title("P6-Q4: Fare vs Trip Distance (Valid Positive Distances)")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q4_distance_fare_scatter.png"), dpi=160)
plt.close()

# Supplementary fare-per-km median by distance range.
plt.figure(figsize=(10, 5.5))
plt.bar(p6q4_summary.index.astype(str), p6q4_summary["median_fare_per_km"])
plt.xlabel("Distance range (km)")
plt.ylabel("Median fare per km")
plt.title("P6-Q4: Median Fare per Kilometer by Trip Distance Range")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q4_fare_per_km_by_range.png"), dpi=160)
plt.close()

print("P6-Q4 observations after temporary positive-distance filter:", len(p6q4))
print("P6-Q4 excluded from temporary analysis due to missing/nonpositive distance:", len(df_clean) - len(p6q4))
print("P6-Q4 summary:\n", p6q4_summary)

# ---------------------------------------------------------------------
# P6-Q5. Which numerical features contain substantial redundant information?
# ---------------------------------------------------------------------
predictor_numeric = [
    "passenger_count", "hour", "day", "month", "weekday", "year",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist",
    "distance", "bearing"
]
num_corr = df_clean[predictor_numeric].corr()
num_corr.to_csv(os.path.join(OUT, "P6_Q5_predictor_numeric_correlation_matrix.csv"))

plt.figure(figsize=(11, 9))
plt.imshow(num_corr.values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(predictor_numeric)), predictor_numeric, rotation=75, ha="right")
plt.yticks(range(len(predictor_numeric)), predictor_numeric)
plt.title("P6-Q5: Predictor Numerical Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q5_numeric_redundancy_heatmap.png"), dpi=160)
plt.close()

pairs = []
for i, a in enumerate(predictor_numeric):
    for b in predictor_numeric[i+1:]:
        r = num_corr.loc[a, b]
        pairs.append((a, b, r, abs(r)))
pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "pearson_r", "abs_r"])
pairs_df = pairs_df.sort_values("abs_r", ascending=False)
pairs_df.to_csv(os.path.join(OUT, "P6_Q5_numeric_correlation_pairs.csv"), index=False)
print("P6-Q5 top 10 absolute correlation pairs:\n", pairs_df.head(10))

# ---------------------------------------------------------------------
# P6-Q6. Does trip volume and/or typical fare show calendar-time changes?
# ---------------------------------------------------------------------
p6q6 = df_clean[["pickup_datetime", "fare_amount"]].dropna().copy()
p6q6["year_month"] = p6q6["pickup_datetime"].dt.to_period("M").astype(str)
p6q6_monthly = (
    p6q6.groupby("year_month")["fare_amount"]
    .agg(trip_count="count", mean_fare="mean", median_fare="median")
)
p6q6_monthly.to_csv(os.path.join(OUT, "P6_Q6_year_month_summary.csv"))

plt.figure(figsize=(14, 5.5))
plt.plot(p6q6_monthly.index, p6q6_monthly["trip_count"], marker="o", markersize=3)
plt.xlabel("Year-month")
plt.ylabel("Number of trips")
plt.title("P6-Q6: Trip Volume Across Calendar Months")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q6_monthly_trip_volume.png"), dpi=160)
plt.close()

plt.figure(figsize=(14, 5.5))
plt.plot(p6q6_monthly.index, p6q6_monthly["mean_fare"], marker="o", markersize=3, label="Mean fare")
plt.plot(p6q6_monthly.index, p6q6_monthly["median_fare"], marker="o", markersize=3, label="Median fare")
plt.xlabel("Year-month")
plt.ylabel("Fare amount")
plt.title("P6-Q6: Typical Fare Across Calendar Months")
plt.xticks(rotation=75)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "P6_Q6_monthly_fare.png"), dpi=160)
plt.close()

# Identify partial final year explicitly.
print("P6-Q6 monthly summary:\n", p6q6_monthly)
print("P6-Q6 date min/max:", df_clean["pickup_datetime"].min(), df_clean["pickup_datetime"].max())
print("P6-Q6 year counts:\n", df_clean.groupby("year").size())
print("P6-Q6 monthly trip count max:", p6q6_monthly["trip_count"].idxmax(), p6q6_monthly["trip_count"].max())
print("P6-Q6 monthly trip count min:", p6q6_monthly["trip_count"].idxmin(), p6q6_monthly["trip_count"].min())
print("P6-Q6 mean fare max:", p6q6_monthly["mean_fare"].idxmax(), p6q6_monthly["mean_fare"].max())
print("P6-Q6 mean fare min:", p6q6_monthly["mean_fare"].idxmin(), p6q6_monthly["mean_fare"].min())

# Consolidated checks for reproducibility.
checks = {
    "df_clean_rows": len(df_clean),
    "df_clean_cols": df_clean.shape[1],
    "p6q1_total_count": int(p6q1_counts.values.sum()),
    "p6q2_total_count": int(p6q2["count"].sum()),
    "p6q3_total_count": int(p6q3_count.to_numpy().sum()),
    "p6q4_positive_distance_n": len(p6q4),
    "p6q5_feature_count": len(predictor_numeric),
    "p6q6_total_count": int(p6q6_monthly["trip_count"].sum()),
}
pd.Series(checks, name="value").to_csv(os.path.join(OUT, "phase6_reproducibility_checks.csv"))
print("CHECKS:", checks)
