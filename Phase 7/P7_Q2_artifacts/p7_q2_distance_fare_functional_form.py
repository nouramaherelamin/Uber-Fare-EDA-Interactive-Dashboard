from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path('/mnt/data')
CLEAN_PATH = BASE / 'phase3' / 'df_clean.csv'
OUT = BASE / 'phase7'
OUT.mkdir(parents=True, exist_ok=True)

# Read-only source. Never overwrite or modify df_clean.
df_clean = pd.read_csv(CLEAN_PATH)

# P7-Q2 uses only observations with valid positive distance and nonmissing fare.
# This is a temporary analysis dataframe; df_clean remains unchanged.
q2 = df_clean[['distance', 'fare_amount']].dropna().copy()
q2 = q2[q2['distance'] > 0].copy()

n_full = len(df_clean)
n_pair_nonmissing = int(df_clean[['distance', 'fare_amount']].dropna().shape[0])
n_used = len(q2)
n_excluded = n_full - n_used
n_nonpositive_distance = int(((df_clean['distance'].notna()) & (df_clean['distance'] <= 0)).sum())
n_missing_distance = int(df_clean['distance'].isna().sum())
n_missing_fare = int(df_clean['fare_amount'].isna().sum())

# Baseline association and monotonic association on the same positive-distance sample.
pearson_raw = q2['distance'].corr(q2['fare_amount'], method='pearson')
spearman_raw = q2['distance'].corr(q2['fare_amount'], method='spearman')

# Log transforms are analytical views only; source data are not changed.
q2['log1p_distance'] = np.log1p(q2['distance'])
q2['log1p_fare'] = np.log1p(q2['fare_amount'])
pearson_log_distance_fare = q2['log1p_distance'].corr(q2['fare_amount'], method='pearson')
pearson_log_log = q2['log1p_distance'].corr(q2['log1p_fare'], method='pearson')
spearman_log_distance_fare = q2['log1p_distance'].corr(q2['fare_amount'], method='spearman')

# Equal-frequency bins to inspect the functional form without fitting a model.
q2['distance_quantile_bin'] = pd.qcut(q2['distance'], q=20, duplicates='drop')
binned = q2.groupby('distance_quantile_bin', observed=True).agg(
    observations=('distance', 'size'),
    distance_min=('distance', 'min'),
    distance_median=('distance', 'median'),
    distance_max=('distance', 'max'),
    fare_mean=('fare_amount', 'mean'),
    fare_median=('fare_amount', 'median'),
    fare_q1=('fare_amount', lambda s: s.quantile(0.25)),
    fare_q3=('fare_amount', lambda s: s.quantile(0.75)),
    fare_per_km_median=('fare_amount', lambda s: np.nanmedian(s / q2.loc[s.index, 'distance'])),
).reset_index(drop=True)
binned.insert(0, 'bin_number', np.arange(1, len(binned) + 1))
binned.to_csv(OUT / 'P7_Q2_distance_quantile_summary.csv', index=False)

# Compact statistics output.
summary = pd.DataFrame([
    ('df_clean_rows', n_full),
    ('pair_nonmissing_distance_fare', n_pair_nonmissing),
    ('positive_distance_used', n_used),
    ('temporary_excluded_total', n_excluded),
    ('missing_distance_rows', n_missing_distance),
    ('nonpositive_distance_rows', n_nonpositive_distance),
    ('missing_fare_rows', n_missing_fare),
    ('pearson_raw_distance_fare', pearson_raw),
    ('spearman_raw_distance_fare', spearman_raw),
    ('pearson_log1p_distance_vs_raw_fare', pearson_log_distance_fare),
    ('pearson_log1p_distance_vs_log1p_fare', pearson_log_log),
    ('spearman_log1p_distance_vs_raw_fare', spearman_log_distance_fare),
    ('quantile_bins', len(binned)),
    ('first_bin_median_distance_km', binned.iloc[0]['distance_median']),
    ('first_bin_median_fare', binned.iloc[0]['fare_median']),
    ('last_bin_median_distance_km', binned.iloc[-1]['distance_median']),
    ('last_bin_median_fare', binned.iloc[-1]['fare_median']),
    ('first_bin_median_fare_per_km', binned.iloc[0]['fare_per_km_median']),
    ('last_bin_median_fare_per_km', binned.iloc[-1]['fare_per_km_median']),
], columns=['metric', 'value'])
summary.to_csv(OUT / 'P7_Q2_functional_form_summary.csv', index=False)

# Plot 1: raw-distance scatter with equal-frequency binned median overlay.
# Plot all positive-distance observations with low alpha; binned medians summarize the shape.
plt.figure(figsize=(10, 6))
plt.scatter(q2['distance'], q2['fare_amount'], s=3, alpha=0.025)
plt.plot(binned['distance_median'], binned['fare_median'], marker='o', linewidth=2, label='20-bin median fare')
plt.xlabel('Trip Distance (km)')
plt.ylabel('Fare Amount')
plt.title('P7-Q2: Distance vs Fare with Binned Median Relationship')
plt.grid(alpha=0.15)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'P7_Q2_distance_fare_binned_median.png', dpi=160)
plt.close()

# Plot 2: log-log view. This is a transformed analytical visualization only.
plt.figure(figsize=(10, 6))
plt.scatter(q2['log1p_distance'], q2['log1p_fare'], s=3, alpha=0.025)
log_binned = q2.groupby(pd.qcut(q2['log1p_distance'], q=20, duplicates='drop'), observed=True).agg(
    x=('log1p_distance', 'median'),
    y=('log1p_fare', 'median')
).reset_index(drop=True)
plt.plot(log_binned['x'], log_binned['y'], marker='o', linewidth=2, label='20-bin median')
plt.xlabel('log(1 + Trip Distance)')
plt.ylabel('log(1 + Fare Amount)')
plt.title('P7-Q2: Log1p Distance vs Log1p Fare')
plt.grid(alpha=0.15)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'P7_Q2_log1p_distance_fare.png', dpi=160)
plt.close()

# Verification checks.
checks = pd.DataFrame([
    {'check': 'df_clean_shape', 'value': f'{df_clean.shape[0]}x{df_clean.shape[1]}'},
    {'check': 'positive_distance_used', 'value': n_used},
    {'check': 'temporary_excluded_total', 'value': n_excluded},
    {'check': 'missing_distance_rows', 'value': n_missing_distance},
    {'check': 'nonpositive_distance_rows', 'value': n_nonpositive_distance},
    {'check': 'pearson_raw', 'value': pearson_raw},
    {'check': 'spearman_raw', 'value': spearman_raw},
    {'check': 'pearson_log1p_distance_vs_raw_fare', 'value': pearson_log_distance_fare},
    {'check': 'pearson_log1p_distance_vs_log1p_fare', 'value': pearson_log_log},
    {'check': 'quantile_bins', 'value': len(binned)},
    {'check': 'first_bin_median_fare', 'value': binned.iloc[0]['fare_median']},
    {'check': 'last_bin_median_fare', 'value': binned.iloc[-1]['fare_median']},
], columns=['check', 'value'])
checks.to_csv(OUT / 'P7_Q2_verification.csv', index=False)

print('P7-Q2 verification')
print(f'df_clean shape: {df_clean.shape}')
print(f'pair nonmissing distance/fare: {n_pair_nonmissing}')
print(f'positive distance used: {n_used}')
print(f'temporary excluded total: {n_excluded}')
print(f'missing distance rows: {n_missing_distance}')
print(f'nonpositive distance rows: {n_nonpositive_distance}')
print(f'missing fare rows: {n_missing_fare}')
print(f'Pearson raw distance/fare: {pearson_raw:.10f}')
print(f'Spearman raw distance/fare: {spearman_raw:.10f}')
print(f'Pearson log1p(distance) vs raw fare: {pearson_log_distance_fare:.10f}')
print(f'Pearson log1p(distance) vs log1p(fare): {pearson_log_log:.10f}')
print(f'Spearman log1p(distance) vs raw fare: {spearman_log_distance_fare:.10f}')
print(f'quantile bins: {len(binned)}')
print(f'first bin median distance: {binned.iloc[0]["distance_median"]:.10f}')
print(f'first bin median fare: {binned.iloc[0]["fare_median"]:.10f}')
print(f'last bin median distance: {binned.iloc[-1]["distance_median"]:.10f}')
print(f'last bin median fare: {binned.iloc[-1]["fare_median"]:.10f}')
print(f'first bin median fare/km: {binned.iloc[0]["fare_per_km_median"]:.10f}')
print(f'last bin median fare/km: {binned.iloc[-1]["fare_per_km_median"]:.10f}')
print('\nBinned summary:')
print(binned.to_string(index=False))
