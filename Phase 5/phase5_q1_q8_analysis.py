import pandas as pd
import matplotlib.pyplot as plt

# Approved Phase 3 dataset. df_clean is read-only for this analysis.
df_clean = pd.read_csv("phase3/df_clean.csv", parse_dates=["pickup_datetime"])

# Q1. Is higher Passenger_Count associated with higher Fare_Amount?
# Main visualization: scatter plot using all nonmissing passenger_count/fare_amount rows.
q1 = df_clean[["passenger_count", "fare_amount"]].dropna()
q1_corr = q1["passenger_count"].corr(q1["fare_amount"])

plt.figure(figsize=(10, 6))
plt.scatter(
    q1["passenger_count"],
    q1["fare_amount"],
    s=4,
    alpha=0.08
)
plt.xlabel("Passenger Count")
plt.ylabel("Fare Amount")
plt.title("Q1: Passenger Count vs Fare Amount")
plt.xticks(range(1, 7))
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()

# Supplementary grouped statistics (not used as the main visualization).
q1_grouped = df_clean.groupby("passenger_count")["fare_amount"].agg(
    count="count", mean="mean", median="median"
)
print(q1_grouped)
print("Pearson r =", q1_corr)

# Q2. Does Car Condition influence Fare Amount?
q2 = df_clean.groupby("Car Condition")["fare_amount"].agg(
    count="count", mean="mean", median="median"
)
plt.figure(figsize=(9, 5))
plt.bar(q2.index, q2["mean"])
plt.xlabel("Car Condition")
plt.ylabel("Mean fare amount")
plt.title("Q2: Mean Fare Amount by Car Condition")
plt.xticks(rotation=15)
plt.show()

# Q3. At what hour are fares typically highest?
q3 = df_clean.groupby("hour")["fare_amount"].agg(
    count="count", mean="mean", median="median"
)
plt.figure(figsize=(10, 5))
plt.plot(q3.index, q3["mean"], marker="o", label="Mean fare")
plt.plot(q3.index, q3["median"], marker="o", label="Median fare")
plt.xlabel("Pickup hour")
plt.ylabel("Fare amount")
plt.title("Q3: Typical Fare by Hour")
plt.xticks(range(24))
plt.legend()
plt.show()

# Q4. Does Traffic Condition affect trip fare?
q4 = df_clean.groupby("Traffic Condition")["fare_amount"].agg(
    count="count", mean="mean", median="median"
)
plt.figure(figsize=(9, 5))
plt.bar(q4.index, q4["mean"])
plt.xlabel("Traffic Condition")
plt.ylabel("Mean fare amount")
plt.title("Q4: Mean Fare Amount by Traffic Condition")
plt.xticks(rotation=15)
plt.show()

# Q5. Is trip distance the strongest predictor of the fare amount?
# Main visualization required by the official Task 1 PDF: trip-level scatter plot.
# No aggregation is used; transparency is only for readability.
q5_scatter = df_clean[["distance", "fare_amount"]].dropna()
q5_distance_corr = q5_scatter["distance"].corr(q5_scatter["fare_amount"])

plt.figure(figsize=(10, 6))
plt.scatter(
    q5_scatter["distance"],
    q5_scatter["fare_amount"],
    s=4,
    alpha=0.05
)
plt.xlabel("Trip Distance (km)")
plt.ylabel("Fare Amount")
plt.title("Q5: Trip Distance vs Fare Amount")
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()

# Supporting Pearson correlation comparison (not ML feature importance).
numeric_cols = [
    "distance", "passenger_count", "jfk_dist", "ewr_dist",
    "lga_dist", "sol_dist", "nyc_dist", "bearing"
]
q5 = (
    df_clean[["fare_amount"] + numeric_cols]
    .corr()["fare_amount"]
    .drop("fare_amount")
    .sort_values(key=lambda s: s.abs(), ascending=False)
)
print("Distance vs Fare Pearson r =", q5_distance_corr)
print(q5)

# Q6. At what time of day are ride requests most frequent?
q6 = df_clean.groupby("hour").size()
plt.figure(figsize=(10, 5))
plt.bar(q6.index, q6.values)
plt.xlabel("Pickup hour")
plt.ylabel("Number of trips")
plt.title("Q6: Trip Frequency by Hour")
plt.xticks(range(24))
plt.show()

# Q7. Does Weather Condition influence average trip Distance?
q7 = df_clean.groupby("Weather")["distance"].agg(
    count="count", mean="mean", median="median"
)
plt.figure(figsize=(9, 5))
plt.bar(q7.index, q7["mean"])
plt.xlabel("Weather")
plt.ylabel("Mean trip distance (km)")
plt.title("Q7: Mean Trip Distance by Weather Condition")
plt.xticks(rotation=15)
plt.show()

# Q8. Are rides that start closer to airports generally more expensive?
airport_cols = ["jfk_dist", "ewr_dist", "lga_dist"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, col, label in zip(axes, airport_cols, ["JFK", "EWR", "LGA"]):
    tmp = df_clean[[col, "fare_amount"]].dropna()
    r = tmp[col].corr(tmp["fare_amount"])

    # Reproducible supporting statistics from the full available data.
    q25 = tmp[col].quantile(0.25)
    q75 = tmp[col].quantile(0.75)
    closest_q1 = tmp.loc[tmp[col] <= q25, "fare_amount"]
    farthest_q4 = tmp.loc[tmp[col] >= q75, "fare_amount"]
    print(
        f"{label}: n={len(tmp)}, Pearson r={r:.12f}, "
        f"closest 25% mean={closest_q1.mean():.12f}, median={closest_q1.median():.2f}, "
        f"farthest 25% mean={farthest_q4.mean():.12f}, median={farthest_q4.median():.2f}"
    )

    # Plot sample for readability only; statistics above use full available data.
    plot_df = tmp.sample(min(30000, len(tmp)), random_state=42)
    ax.scatter(plot_df[col], plot_df["fare_amount"], s=4, alpha=0.12)
    ax.set_xlabel(f"{label} distance")
    ax.set_ylabel("Fare amount")
    ax.set_title(f"{label}: r={r:.3f}")

fig.suptitle("Q8: Fare vs Airport Distance")
fig.tight_layout()
plt.show()
