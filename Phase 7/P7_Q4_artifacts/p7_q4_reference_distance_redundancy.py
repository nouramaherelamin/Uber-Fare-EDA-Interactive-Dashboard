from pathlib import Path
import hashlib
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path('/mnt/data')
OUT = BASE / 'phase7'
DF_PATH = BASE / 'phase3' / 'df_clean.csv'

REF_COLS = ['jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist']
TARGET = 'fare_amount'
USE_COLS = REF_COLS + [TARGET]

# Read-only source check: hash before analysis.
hash_before = hashlib.sha256(DF_PATH.read_bytes()).hexdigest()
df_clean = pd.read_csv(DF_PATH, usecols=USE_COLS)
shape_before = df_clean.shape

# Target correlations: each reference-distance feature against fare on its own
# nonmissing observations. No row deletion or permanent cleaning is performed.
target_rows = []
for col in REF_COLS:
    sub = df_clean[[col, TARGET]].dropna()
    target_rows.append({
        'feature': col,
        'observations': len(sub),
        'pearson_r_with_fare': sub[col].corr(sub[TARGET], method='pearson'),
        'spearman_r_with_fare': sub[col].corr(sub[TARGET], method='spearman'),
        'mean_feature': sub[col].mean(),
        'median_feature': sub[col].median(),
    })
target_summary = pd.DataFrame(target_rows)
target_summary.to_csv(OUT / 'P7_Q4_reference_distance_target_summary.csv', index=False)

# Pairwise redundancy: Pearson and Spearman associations among reference-distance
# variables, using pairwise complete observations.
pearson = df_clean[REF_COLS].corr(method='pearson')
spearman = df_clean[REF_COLS].corr(method='spearman')
pearson.to_csv(OUT / 'P7_Q4_reference_distance_pearson_matrix.csv')
spearman.to_csv(OUT / 'P7_Q4_reference_distance_spearman_matrix.csv')

# Pairwise overlap counts document how much data supports each redundancy estimate.
overlap = pd.DataFrame(index=REF_COLS, columns=REF_COLS, dtype=int)
for a in REF_COLS:
    for b in REF_COLS:
        overlap.loc[a, b] = int(df_clean[[a, b]].notna().all(axis=1).sum())
overlap.to_csv(OUT / 'P7_Q4_reference_distance_pairwise_overlap.csv')

# Missingness summary for the same features, plus fare as the target reference.
missing_summary = pd.DataFrame({
    'feature': REF_COLS + [TARGET],
    'missing_count': [int(df_clean[c].isna().sum()) for c in REF_COLS + [TARGET]],
})
missing_summary['missing_percent_of_df_clean'] = missing_summary['missing_count'] / len(df_clean) * 100
missing_summary.to_csv(OUT / 'P7_Q4_reference_distance_missingness.csv', index=False)

# Plot 1: redundancy heatmap among reference-distance variables.
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(pearson.values, vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(range(len(REF_COLS)))
ax.set_yticks(range(len(REF_COLS)))
ax.set_xticklabels(REF_COLS, rotation=35, ha='right')
ax.set_yticklabels(REF_COLS)
ax.set_title('P7-Q4: Reference-Distance Pearson Redundancy')
for i in range(len(REF_COLS)):
    for j in range(len(REF_COLS)):
        ax.text(j, i, f'{pearson.iloc[i, j]:.3f}', ha='center', va='center', fontsize=9)
fig.colorbar(im, ax=ax, label='Pearson correlation')
fig.tight_layout()
fig.savefig(OUT / 'P7_Q4_reference_distance_redundancy_heatmap.png', dpi=160)
plt.close(fig)

# Plot 2: target association for each reference-distance feature.
plot_df = target_summary.copy()
x = np.arange(len(plot_df))
width = 0.36
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.bar(x - width/2, plot_df['pearson_r_with_fare'], width, label='Pearson r')
ax.bar(x + width/2, plot_df['spearman_r_with_fare'], width, label='Spearman r')
ax.axhline(0, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(plot_df['feature'], rotation=25, ha='right')
ax.set_ylabel('Correlation with Fare Amount')
ax.set_title('P7-Q4: Reference-Distance Association with Fare')
ax.legend()
ax.grid(axis='y', alpha=0.15)
fig.tight_layout()
fig.savefig(OUT / 'P7_Q4_reference_distance_target_association.png', dpi=160)
plt.close(fig)

# Verification: source shape/hash and expected missingness; all source data remains read-only.
hash_after = hashlib.sha256(DF_PATH.read_bytes()).hexdigest()
expected_missing = 10294
verification = pd.DataFrame([
    {'check': 'df_clean_rows', 'value': shape_before[0], 'expected': 499965, 'pass': shape_before[0] == 499965},
    {'check': 'selected_columns_count', 'value': len(USE_COLS), 'expected': 6, 'pass': len(USE_COLS) == 6},
    {'check': 'fare_missing', 'value': int(df_clean[TARGET].isna().sum()), 'expected': 0, 'pass': int(df_clean[TARGET].isna().sum()) == 0},
    {'check': 'reference_distance_missing_each', 'value': int(df_clean[REF_COLS].isna().sum().min()), 'expected': expected_missing, 'pass': bool((df_clean[REF_COLS].isna().sum() == expected_missing).all())},
    {'check': 'target_summary_rows', 'value': len(target_summary), 'expected': 5, 'pass': len(target_summary) == 5},
    {'check': 'pearson_matrix_shape', 'value': str(pearson.shape), 'expected': '(5, 5)', 'pass': pearson.shape == (5, 5)},
    {'check': 'spearman_matrix_shape', 'value': str(spearman.shape), 'expected': '(5, 5)', 'pass': spearman.shape == (5, 5)},
    {'check': 'pairwise_overlap_diagonal', 'value': int(np.diag(overlap.values).min()), 'expected': 489671, 'pass': bool((np.diag(overlap.values) == 489671).all())},
    {'check': 'df_clean_sha256_unchanged', 'value': hash_after, 'expected': hash_before, 'pass': hash_after == hash_before},
], dtype=object)
verification.to_csv(OUT / 'P7_Q4_verification.csv', index=False)

# Console output for exact reproducibility.
print('P7-Q4')
print('df_clean shape:', shape_before)
print('Reference-distance columns:', REF_COLS)
print('\nTarget association summary:')
print(target_summary.to_string(index=False))
print('\nPearson redundancy matrix:')
print(pearson.to_string())
print('\nSpearman redundancy matrix:')
print(spearman.to_string())
print('\nPairwise complete-observation counts:')
print(overlap.to_string())
print('\nMissingness:')
print(missing_summary.to_string(index=False))
print('\nVerification:')
print(verification.to_string(index=False))
print('\nSHA unchanged:', hash_after == hash_before)

# Package only P7-Q4 artifacts; leave prior Q1-Q3 artifacts untouched.
artifact_names = [
    'p7_q4_reference_distance_redundancy.py',
    'P7_Q4_reference_distance_target_summary.csv',
    'P7_Q4_reference_distance_pearson_matrix.csv',
    'P7_Q4_reference_distance_spearman_matrix.csv',
    'P7_Q4_reference_distance_pairwise_overlap.csv',
    'P7_Q4_reference_distance_missingness.csv',
    'P7_Q4_reference_distance_redundancy_heatmap.png',
    'P7_Q4_reference_distance_target_association.png',
    'P7_Q4_verification.csv',
    'P7_Q4_run_output.txt',
]
# run output is written after print capture externally; create zip after it exists.
zip_path = OUT / 'P7_Q4_artifacts.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for name in artifact_names:
        path = OUT / name
        if path.exists():
            zf.write(path, arcname=name)
print('Artifact zip:', zip_path)
