import os, hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = '/mnt/data'
DF_PATH = f'{BASE}/phase3/df_clean.csv'
OUT = f'{BASE}/phase7'
os.makedirs(OUT, exist_ok=True)

# Read-only source load. Never write to df_clean.csv.
df_clean = pd.read_csv(DF_PATH, low_memory=False)
shape_before = df_clean.shape
hash_before = hashlib.sha256(open(DF_PATH, 'rb').read()).hexdigest()

# P7-Q3 temporary analysis object: positive, nonmissing distance/fare only.
q3 = df_clean[['distance', 'fare_amount', 'hour', 'weekday']].copy()
q3 = q3.dropna(subset=['distance', 'fare_amount', 'hour', 'weekday']).copy()
q3 = q3[q3['distance'] > 0].copy()
q3['hour'] = q3['hour'].astype(int)
q3['weekday'] = q3['weekday'].astype(int)

# Four fixed dayparts chosen to summarize the 24 hourly values without repeating Q6.
daypart_order = ['00-05', '06-11', '12-17', '18-23']
q3['daypart'] = pd.cut(
    q3['hour'], bins=[-1, 5, 11, 17, 23], labels=daypart_order, ordered=True
)

# 10 equal-frequency distance bins within each daypart.
q3['distance_bin'] = q3.groupby('daypart', observed=True)['distance'].transform(
    lambda s: pd.qcut(s, q=10, labels=False, duplicates='drop') + 1
)

binned = (
    q3.groupby(['daypart', 'distance_bin'], observed=True)
      .agg(
          observations=('distance', 'size'),
          distance_median=('distance', 'median'),
          fare_median=('fare_amount', 'median'),
          fare_mean=('fare_amount', 'mean'),
      )
      .reset_index()
)
binned.to_csv(f'{OUT}/P7_Q3_distance_fare_by_daypart.csv', index=False)

# Relationship statistics by daypart.
rows = []
for dp in daypart_order:
    sub = q3[q3['daypart'] == dp]
    rows.append({
        'daypart': dp,
        'observations': len(sub),
        'distance_median_km': sub['distance'].median(),
        'fare_mean': sub['fare_amount'].mean(),
        'fare_median': sub['fare_amount'].median(),
        'pearson_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='pearson'),
        'spearman_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='spearman'),
    })
by_daypart = pd.DataFrame(rows)
by_daypart.to_csv(f'{OUT}/P7_Q3_daypart_relationship_summary.csv', index=False)

# Weekday/weekend robustness check using the same positive-distance temporary sample.
q3['week_group'] = np.where(q3['weekday'] < 5, 'Weekday', 'Weekend')
week_rows = []
for wg in ['Weekday', 'Weekend']:
    sub = q3[q3['week_group'] == wg]
    week_rows.append({
        'week_group': wg,
        'observations': len(sub),
        'pearson_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='pearson'),
        'spearman_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='spearman'),
        'fare_median': sub['fare_amount'].median(),
        'fare_mean': sub['fare_amount'].mean(),
    })
week_summary = pd.DataFrame(week_rows)
week_summary.to_csv(f'{OUT}/P7_Q3_weekday_weekend_summary.csv', index=False)

# Hour-level Pearson correlations to check whether the daypart summary hides major hourly changes.
hour_rows = []
for h in range(24):
    sub = q3[q3['hour'] == h]
    hour_rows.append({
        'hour': h,
        'observations': len(sub),
        'pearson_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='pearson'),
        'spearman_r_distance_fare': sub['distance'].corr(sub['fare_amount'], method='spearman'),
    })
hour_summary = pd.DataFrame(hour_rows)
hour_summary.to_csv(f'{OUT}/P7_Q3_hour_relationship_summary.csv', index=False)

# Plot: median fare vs median distance by daypart.
plt.figure(figsize=(10, 6))
for dp in daypart_order:
    sub = binned[binned['daypart'] == dp].sort_values('distance_bin')
    plt.plot(sub['distance_median'], sub['fare_median'], marker='o', linewidth=1.8, label=dp)
plt.xlabel('Median Trip Distance (km)')
plt.ylabel('Median Fare Amount')
plt.title('P7-Q3: Distance–Fare Relationship by Time of Day')
plt.grid(alpha=0.15)
plt.legend(title='Daypart')
plt.tight_layout()
plt.savefig(f'{OUT}/P7_Q3_distance_fare_by_daypart.png', dpi=160)
plt.close()

# Plot: hourly Pearson correlation, descriptive stability check.
plt.figure(figsize=(10, 5.5))
plt.plot(hour_summary['hour'], hour_summary['pearson_r_distance_fare'], marker='o')
plt.axhline(0, linewidth=0.8)
plt.xticks(range(24))
plt.xlabel('Pickup Hour')
plt.ylabel('Pearson r: Distance vs Fare')
plt.title('P7-Q3: Hourly Distance–Fare Pearson Association')
plt.grid(alpha=0.15)
plt.tight_layout()
plt.savefig(f'{OUT}/P7_Q3_hourly_distance_fare_pearson.png', dpi=160)
plt.close()

# Verification table.
excluded = int(shape_before[0] - len(q3))
missing_distance = int(df_clean['distance'].isna().sum())
nonpositive_distance = int((df_clean['distance'].notna() & (df_clean['distance'] <= 0)).sum())
missing_fare = int(df_clean['fare_amount'].isna().sum())

hash_after = hashlib.sha256(open(DF_PATH, 'rb').read()).hexdigest()
verification = pd.DataFrame([
    {'check': 'df_clean_rows', 'value': shape_before[0], 'expected': 499965, 'pass': shape_before[0] == 499965},
    {'check': 'df_clean_columns', 'value': shape_before[1], 'expected': 26, 'pass': shape_before[1] == 26},
    {'check': 'positive_distance_observations_used', 'value': len(q3), 'expected': 484715, 'pass': len(q3) == 484715},
    {'check': 'temporary_exclusions', 'value': excluded, 'expected': 15250, 'pass': excluded == 15250},
    {'check': 'missing_distance', 'value': missing_distance, 'expected': 9996, 'pass': missing_distance == 9996},
    {'check': 'nonpositive_distance', 'value': nonpositive_distance, 'expected': 5254, 'pass': nonpositive_distance == 5254},
    {'check': 'missing_fare', 'value': missing_fare, 'expected': 0, 'pass': missing_fare == 0},
    {'check': 'dayparts', 'value': q3['daypart'].nunique(), 'expected': 4, 'pass': q3['daypart'].nunique() == 4},
    {'check': 'distance_bins_per_daypart', 'value': int(binned.groupby('daypart').size().min()), 'expected': 10, 'pass': bool((binned.groupby('daypart').size() == 10).all())},
    {'check': 'df_clean_sha256_unchanged', 'value': hash_after, 'expected': hash_before, 'pass': hash_after == hash_before},
], dtype=object)
verification.to_csv(f'{OUT}/P7_Q3_verification.csv', index=False)

# Console output for reproducibility.
print('P7-Q3')
print('df_clean shape:', shape_before)
print('positive distance observations:', len(q3))
print('temporary exclusions:', excluded)
print('missing distance:', missing_distance)
print('nonpositive distance:', nonpositive_distance)
print('missing fare:', missing_fare)
print('\nDaypart relationship summary:')
print(by_daypart.to_string(index=False))
print('\nWeekday/weekend summary:')
print(week_summary.to_string(index=False))
print('\nHourly Pearson summary:')
print(hour_summary.to_string(index=False))
print('\nVerification:')
print(verification.to_string(index=False))
print('\nSHA unchanged:', hash_after == hash_before)
