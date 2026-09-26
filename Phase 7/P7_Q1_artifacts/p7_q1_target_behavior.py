from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path('/mnt/data')
CLEAN_PATH = BASE / 'phase3' / 'df_clean.csv'
OUT = BASE / 'phase7'
OUT.mkdir(parents=True, exist_ok=True)

# Read-only source: df_clean is never overwritten or modified.
df_clean = pd.read_csv(CLEAN_PATH)

# -----------------------------
# P7-Q1: Target behavior
# -----------------------------
q1 = df_clean[['fare_amount']].dropna()

# Full target distribution statistics.
q1_n = len(q1)
q1_missing = int(df_clean['fare_amount'].isna().sum())
q1_quantiles = q1['fare_amount'].quantile([0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]).rename('fare_amount')
q1_stats = pd.DataFrame({
    'metric': ['n', 'missing', 'mean', 'median', 'q1', 'q3', 'iqr', 'iqr_upper_fence', 'p90', 'p95', 'p99', 'p99_5', 'p99_9', 'max', 'skewness'],
    'value': [
        q1_n,
        q1_missing,
        q1['fare_amount'].mean(),
        q1['fare_amount'].median(),
        q1['fare_amount'].quantile(0.25),
        q1['fare_amount'].quantile(0.75),
        q1['fare_amount'].quantile(0.75) - q1['fare_amount'].quantile(0.25),
        q1['fare_amount'].quantile(0.75) + 1.5 * (q1['fare_amount'].quantile(0.75) - q1['fare_amount'].quantile(0.25)),
        q1['fare_amount'].quantile(0.90),
        q1['fare_amount'].quantile(0.95),
        q1['fare_amount'].quantile(0.99),
        q1['fare_amount'].quantile(0.995),
        q1['fare_amount'].quantile(0.999),
        q1['fare_amount'].max(),
        q1['fare_amount'].skew(),
    ]
})
q1_stats.to_csv(OUT / 'P7_Q1_target_distribution_summary.csv', index=False)

# Counts above meaningful upper-tail thresholds.
thresholds = [22.25, 30.0, q1['fare_amount'].quantile(0.99), q1['fare_amount'].quantile(0.995), q1['fare_amount'].quantile(0.999), 100, 200, 300, 400]
threshold_rows = []
for t in thresholds:
    n = int((q1['fare_amount'] > t).sum())
    threshold_rows.append({'threshold': t, 'count_above': n, 'percent_of_nonmissing_fare': 100*n/q1_n})
q1_thresholds = pd.DataFrame(threshold_rows).drop_duplicates(subset=['threshold'])
q1_thresholds.to_csv(OUT / 'P7_Q1_upper_tail_thresholds.csv', index=False)

# Top-fare observations for structural inspection. This is temporary selection only.
inspect_cols = ['fare_amount', 'distance', 'hour', 'weekday', 'month', 'year', 'pickup_datetime']
q1_top = df_clean[inspect_cols].dropna(subset=['fare_amount']).nlargest(100, 'fare_amount').copy()
q1_top.to_csv(OUT / 'P7_Q1_top_100_fares.csv', index=False)

# Upper-tail summary: P99+ observations.
p99 = q1['fare_amount'].quantile(0.99)
upper = df_clean[df_clean['fare_amount'] > p99][inspect_cols].dropna(subset=['fare_amount']).copy()
upper_summary = pd.DataFrame({
    'metric': ['threshold_p99', 'n_upper_tail', 'percent_of_nonmissing_fare', 'mean_fare_upper_tail', 'median_fare_upper_tail', 'min_fare_upper_tail', 'max_fare_upper_tail', 'median_distance_upper_tail', 'mean_distance_upper_tail'],
    'value': [
        p99,
        len(upper),
        100*len(upper)/q1_n,
        upper['fare_amount'].mean(),
        upper['fare_amount'].median(),
        upper['fare_amount'].min(),
        upper['fare_amount'].max(),
        upper['distance'].median(),
        upper['distance'].mean(),
    ]
})
upper_summary.to_csv(OUT / 'P7_Q1_p99_upper_tail_summary.csv', index=False)

# Relationship of high-fare observations to distance: share with missing/available distance,
# and counts by distance availability. No additional row deletion from df_clean.
upper_distance = upper['distance']
upper_distance_avail = upper_distance.notna().sum()
upper_distance_missing = upper_distance.isna().sum()
upper_distance_summary = pd.DataFrame({
    'metric': ['upper_tail_rows', 'distance_available', 'distance_missing', 'distance_available_percent', 'fare_over_100_count', 'fare_over_200_count', 'fare_over_300_count', 'fare_over_400_count'],
    'value': [
        len(upper),
        upper_distance_avail,
        upper_distance_missing,
        100*upper_distance_avail/len(upper),
        int((upper['fare_amount'] > 100).sum()),
        int((upper['fare_amount'] > 200).sum()),
        int((upper['fare_amount'] > 300).sum()),
        int((upper['fare_amount'] > 400).sum()),
    ]
})
upper_distance_summary.to_csv(OUT / 'P7_Q1_upper_tail_distance_summary.csv', index=False)

# Plot 1: full distribution, clipped at P99 for readability only.
plot_fares = q1.loc[q1['fare_amount'] <= p99, 'fare_amount']
plt.figure(figsize=(10, 6))
plt.hist(plot_fares, bins=80)
plt.axvline(p99, linestyle='--', label=f'P99 = {p99:.2f}')
plt.xlabel('Fare Amount')
plt.ylabel('Number of Trips')
plt.title('P7-Q1: Fare Distribution (Visualization clipped at P99)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'P7_Q1_fare_distribution_p99.png', dpi=160)
plt.close()

# Plot 2: upper tail inspection against distance, where distance is available.
upper_plot = upper.dropna(subset=['distance'])
plt.figure(figsize=(10, 6))
plt.scatter(upper_plot['distance'], upper_plot['fare_amount'], s=10, alpha=0.25)
plt.xlabel('Trip Distance (km)')
plt.ylabel('Fare Amount')
plt.title('P7-Q1: Upper-Tail Fares (Above P99) vs Trip Distance')
plt.grid(alpha=0.15)
plt.tight_layout()
plt.savefig(OUT / 'P7_Q1_upper_tail_fare_vs_distance.png', dpi=160)
plt.close()

# Plot 3: boxplot of fare with extreme range compressed visually using log1p only for display.
plt.figure(figsize=(8, 5))
plt.boxplot(np.log1p(q1['fare_amount']), vert=False)
plt.xlabel('log(1 + Fare Amount)')
plt.title('P7-Q1: Fare Distribution on log1p Display Scale')
plt.tight_layout()
plt.savefig(OUT / 'P7_Q1_fare_log1p_boxplot.png', dpi=160)
plt.close()

# Reproducibility checks.
checks = pd.DataFrame([
    {'check': 'df_clean_rows', 'value': len(df_clean)},
    {'check': 'df_clean_columns', 'value': df_clean.shape[1]},
    {'check': 'fare_nonmissing', 'value': q1_n},
    {'check': 'fare_missing', 'value': q1_missing},
    {'check': 'p99_threshold', 'value': p99},
    {'check': 'p99_upper_tail_count', 'value': len(upper)},
    {'check': 'top_100_rows', 'value': len(q1_top)},
    {'check': 'distance_available_in_p99_tail', 'value': upper_distance_avail},
    {'check': 'distance_missing_in_p99_tail', 'value': upper_distance_missing},
])
checks.to_csv(OUT / 'P7_Q1_verification.csv', index=False)

print('P7-Q1 verification')
print(f'df_clean shape: {df_clean.shape}')
print(f'fare nonmissing: {q1_n}')
print(f'fare missing: {q1_missing}')
print(f'mean: {q1["fare_amount"].mean():.10f}')
print(f'median: {q1["fare_amount"].median():.10f}')
print(f'q1: {q1["fare_amount"].quantile(.25):.10f}')
print(f'q3: {q1["fare_amount"].quantile(.75):.10f}')
print(f'iqr upper fence: {q1["fare_amount"].quantile(.75) + 1.5*(q1["fare_amount"].quantile(.75)-q1["fare_amount"].quantile(.25)):.10f}')
print(f'p99: {p99:.10f}')
print(f'p99.5: {q1["fare_amount"].quantile(.995):.10f}')
print(f'p99.9: {q1["fare_amount"].quantile(.999):.10f}')
print(f'max: {q1["fare_amount"].max():.10f}')
print(f'skewness: {q1["fare_amount"].skew():.10f}')
print(f'P99 upper-tail count: {len(upper)} ({100*len(upper)/q1_n:.6f}%)')
print(f'P99 upper-tail mean: {upper["fare_amount"].mean():.10f}')
print(f'P99 upper-tail median: {upper["fare_amount"].median():.10f}')
print(f'P99 upper-tail median distance: {upper["distance"].median():.10f}')
print(f'P99 upper-tail mean distance: {upper["distance"].mean():.10f}')
print(f'P99 tail distance available: {upper_distance_avail}')
print(f'P99 tail distance missing: {upper_distance_missing}')
print(f'fare > 100: {(q1["fare_amount"] > 100).sum()}')
print(f'fare > 200: {(q1["fare_amount"] > 200).sum()}')
print(f'fare > 300: {(q1["fare_amount"] > 300).sum()}')
print(f'fare > 400: {(q1["fare_amount"] > 400).sum()}')
print('Top 10 fares:')
print(q1_top.head(10).to_string(index=False))
