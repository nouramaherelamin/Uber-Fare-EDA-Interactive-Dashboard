import pandas as pd
import matplotlib.pyplot as plt

# Approved Phase 3 dataset; do not modify it during EDA.
df_clean = pd.read_csv("phase3/df_clean.csv", parse_dates=["pickup_datetime"])

# 1) Target: fare_amount
df_clean["fare_amount"].describe()
df_clean["fare_amount"].skew()

plt.figure(figsize=(9, 5))
plt.hist(df_clean["fare_amount"], bins=80)
plt.xlabel("Fare amount")
plt.ylabel("Number of trips")
plt.title("Distribution of Fare Amount")
plt.show()

plt.figure(figsize=(9, 3.8))
plt.boxplot(df_clean["fare_amount"], vert=False)
plt.xlabel("Fare amount")
plt.title("Fare Amount Boxplot")
plt.show()

# 2) Distance
distance = df_clean["distance"].dropna()
p99 = distance.quantile(.99)

plt.figure(figsize=(9, 5))
plt.hist(distance[distance <= p99], bins=70)
plt.xlabel("Distance (km)")
plt.ylabel("Number of trips")
plt.title("Distance Distribution (shown through 99th percentile)")
plt.show()

# 3) Passenger count
counts = df_clean["passenger_count"].value_counts(dropna=False).sort_index()
plt.figure(figsize=(8, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Passenger count")
plt.ylabel("Number of trips")
plt.title("Passenger Count Distribution")
plt.show()

# 4) Categorical variables
for col in ["Car Condition", "Weather", "Traffic Condition"]:
    counts = df_clean[col].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel(col)
    plt.ylabel("Number of trips")
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=20, ha="right")
    plt.show()

# 5) Temporal distributions
hour_counts = df_clean["hour"].value_counts().sort_index()
plt.figure(figsize=(9, 5))
plt.plot(hour_counts.index, hour_counts.values, marker="o")
plt.xlabel("Hour of day")
plt.ylabel("Number of trips")
plt.title("Trips by Hour of Day")
plt.xticks(range(24))
plt.show()

weekday_counts = df_clean["weekday"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(range(7), weekday_counts.values)
plt.xticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.xlabel("Weekday")
plt.ylabel("Number of trips")
plt.title("Trips by Weekday")
plt.show()

month_counts = df_clean["month"].value_counts().sort_index()
plt.figure(figsize=(9, 5))
plt.plot(month_counts.index, month_counts.values, marker="o")
plt.xlabel("Month")
plt.ylabel("Number of trips")
plt.title("Trips by Month")
plt.xticks(range(1, 13))
plt.show()

# 6) Quick baseline correlations — not the deeper Phase 7 analysis
corr_cols = [
    "fare_amount", "distance", "passenger_count",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist", "bearing"
]
corr = df_clean[corr_cols].corr()
print(corr.round(3))

plt.figure(figsize=(9, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Baseline Numeric Correlation Heatmap")
plt.show()

# 7) Fare vs distance
plot_df = df_clean[["fare_amount", "distance"]].dropna()
plot_df = plot_df.sample(min(30000, len(plot_df)), random_state=42)

plt.figure(figsize=(9, 5))
plt.scatter(plot_df["distance"], plot_df["fare_amount"], alpha=0.25, s=8)
plt.xlabel("Distance (km)")
plt.ylabel("Fare amount")
plt.title("Fare Amount vs Distance (30,000-point plotting sample)")
plt.show()
