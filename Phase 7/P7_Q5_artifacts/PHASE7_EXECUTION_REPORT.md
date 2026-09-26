# Phase 7 — Deep-Dive EDA

## P7-Q1 — Fare Target Behavior and Meaningful Extremes

### 1. Question
**What does the fare target distribution look like after the Phase 3 cleaning, and which high-fare observations are genuinely unusual enough to matter for Task 2?**

### 2. Why this question is justified
Phase 4 established that `fare_amount` is strongly right-skewed (skewness about 4.90), with median `$8.50`, P99 about `$52.14`, and maximum `$500`. Phase 5 also showed differences between mean and median fare for several questions. These results justify a deeper inspection of the target's upper tail rather than another basic distribution plot.

### 3. Plot choice and why
Three complementary visualizations were used:

1. **Fare histogram with visualization-only P99 clipping** — shows the main target distribution without allowing the extreme upper tail to compress the majority of the observations.
2. **Upper-tail fare vs distance scatter plot** — checks whether unusually high fares are generally accompanied by longer trips or whether high fares also occur with very small recorded distances.
3. **Horizontal boxplot on a `log1p` display scale** — compresses the long right tail for visual inspection only; it does not transform or modify the source target.

### 4. Data / observations used
- Source: Phase 3 `df_clean`, read-only.
- Full dataset: **499,965 rows × 26 columns**.
- `fare_amount` nonmissing: **499,965**.
- `fare_amount` missing: **0**.
- Full-distribution statistics use all 499,965 fare observations.

### 5. Cleaning / filtering
No additional cleaning was applied and `df_clean` was not modified.

Temporary selections were used only for inspection/visualization:

- P99 visualization clipping: fares **<= P99** were used only for the histogram display; **5,000** observations are above P99 and therefore not shown in that clipped histogram.
- Upper-tail inspection: observations with `fare_amount > P99` = **5,000** observations.
- Upper-tail vs distance plot: of those 5,000 observations, **4,824** had available distance and **176** were excluded from that plot only because distance was missing.
- Top-fare inspection: the **100 highest-fare observations** were selected temporarily for structural inspection.

No rows were removed from `df_clean`.

### 6. Exact analysis code
The complete reproducible code is in `p7_q1_target_behavior.py`. The core calculations are:

```python
q1 = df_clean[['fare_amount']].dropna()

p99 = q1['fare_amount'].quantile(0.99)

q1_stats = {
    'mean': q1['fare_amount'].mean(),
    'median': q1['fare_amount'].median(),
    'q1': q1['fare_amount'].quantile(0.25),
    'q3': q1['fare_amount'].quantile(0.75),
    'p90': q1['fare_amount'].quantile(0.90),
    'p95': q1['fare_amount'].quantile(0.95),
    'p99': p99,
    'p99_5': q1['fare_amount'].quantile(0.995),
    'p99_9': q1['fare_amount'].quantile(0.999),
    'max': q1['fare_amount'].max(),
    'skewness': q1['fare_amount'].skew(),
}

upper = df_clean[df_clean['fare_amount'] > p99].copy()
upper_plot = upper.dropna(subset=['distance'])
```

The script also saves the full statistics, upper-tail thresholds, top 100 observations, upper-tail distance summary, plots, and verification checks.

### 7. Actual numerical results

| Statistic | Result |
|---|---:|
| Observations | **499,965** |
| Missing fare | **0** |
| Mean | **11.3594224596** |
| Median | **8.5000000000** |
| Q1 | **6.0000000000** |
| Q3 | **12.5000000000** |
| P90 | **20.5000000000** |
| P95 | **30.5000000000** |
| P99 | **52.1360000000** |
| P99.5 | **57.3300000000** |
| P99.9 | **80.0000000000** |
| Maximum | **500.0000000000** |
| Skewness | **4.8999007245** |

Upper-tail counts:

| Threshold | Count above threshold |
|---|---:|
| `$100` | **214** |
| `$200` | **26** |
| `$300` | **7** |
| `$400` | **5** |

P99 upper tail (`fare_amount > $52.136`):

- **5,000 observations** (**1.000070%** of nonmissing fares)
- Mean fare: **$64.6380720000**
- Median fare: **$57.3300000000**
- Maximum: **$500.00**
- Median distance among available upper-tail distances: **20.0314796316 km**
- Mean distance among available upper-tail distances: **17.9549169363 km**
- Distance available: **4,824**
- Distance missing: **176**

Within the full dataset, the extreme high-fare counts are:

- `fare_amount > $100`: **214**
- `fare_amount > $200`: **26**
- `fare_amount > $300`: **7**
- `fare_amount > $400`: **5**

The top-fare inspection shows a heterogeneous upper tail. For example, the maximum `$500` observation has recorded distance `0.000000 km`, while another `$495` observation has distance about `140.515 km`; several `$400–$450` observations also have distances close to zero. These are **observed data patterns**, not a conclusion that those records are erroneous.

### 8. Plots

**Main target distribution:**

[P7-Q1 fare distribution — P99-clipped for visualization](P7_Q1_fare_distribution_p99.png)

**Upper-tail inspection:**

[P7-Q1 upper-tail fare vs distance](P7_Q1_upper_tail_fare_vs_distance.png)

**Target on log1p display scale:**

[P7-Q1 log1p fare boxplot](P7_Q1_fare_log1p_boxplot.png)

The P99 clipping and `log1p` scale are visualization choices only. They do not alter `df_clean` or the underlying fare statistics.

### 9. Interpretation
The fare target is strongly right-skewed: the mean (**$11.36**) is substantially above the median (**$8.50**), and the maximum of `$500` is far above the upper quartile of `$12.50`.

The upper tail is small in proportion but numerically substantial. Only about **1%** of observations are above P99, yet that group has a mean fare of about `$64.64`.

The upper-tail scatter also shows that high fares are not exclusively associated with long trips. Some extreme fares occur with very small or zero recorded distance, while others occur with substantially longer distances. Therefore, the upper tail cannot be interpreted simply as a collection of long-trip fares.

### 10. Evidence-based insight
The target distribution has a substantial upper tail that deserves explicit consideration before Task 2. The evidence supports treating fare skewness and extreme observations as **modeling risks to investigate**, rather than automatically deleting or capping them.

The coexistence of very high fares and zero/negligible recorded distance in a subset of observations is particularly important to flag for later Task 2 investigation. It does not, by itself, establish that those records are invalid.

### 11. Limitations
- This is descriptive EDA; it does not determine whether any extreme fare is erroneous.
- P99 is used as an inspection/visualization threshold, not as a data-cleaning rule.
- The upper-tail analysis does not establish causality.
- No model performance or transformation performance was evaluated.
- Distance itself can be missing or zero in some retained rows, so the upper-tail distance comparison is not available for every extreme fare.

### 12. Task 2 relevance
The results support **considering** the following in Task 2:

- whether a target transformation such as a log-scale target should be evaluated;
- whether robust evaluation or explicit treatment of extreme targets is needed;
- whether the combination of extreme fare and very small/zero recorded distance requires additional investigation before modeling.

These are modeling considerations/hypotheses, not confirmed modeling decisions.

## P7-Q1 Verification

- `df_clean` shape read: **499,965 × 26**.
- Fare observations used for full target statistics: **499,965**.
- Temporary P99 upper-tail selection: **5,000**.
- Temporary upper-tail observations excluded from the distance plot because distance was missing: **176**.
- Top-fare inspection selection: **100**.
- Source dataset was read only; no write operation targets `df_clean`.
- Numerical results in this section are reproduced by `p7_q1_target_behavior.py` and the saved CSV outputs.

### P7 Status
**P7-Q1 COMPLETED AND VERIFIED**

**P7-Q2 NOT STARTED.**

---

## P7-Q2 — Distance/Fare Functional Form

### 1. Question
**Is the distance–fare relationship sufficiently nonlinear that a transformed or nonlinear representation should be considered for Task 2?**

### 2. Why this question is justified
Approved Q5 established that `distance` had the strongest observed numerical association with `fare_amount` in the specified Phase 5 comparison. Phase 6 P6-Q4 then showed that total fare rises with distance while median fare per kilometer declines as trip length increases. P7-Q2 therefore examines the functional form more closely rather than repeating the approved Q5 conclusion.

### 3. Plot choice and why
Two complementary visualizations were used:

1. **Distance vs fare scatter plot with 20 equal-frequency binned median fares** — preserves the trip-level observations while making the central relationship easier to inspect without fitting a model.
2. **`log1p(distance)` vs `log1p(fare)` scatter plot with 20-bin median overlay** — evaluates whether a log-scale representation makes the observed relationship more regular while keeping the transformation purely analytical/visual.

### 4. Data / observations used
- Source: Phase 3 `df_clean`, read-only.
- Full dataset: **499,965 rows × 26 columns**.
- Nonmissing distance/fare pairs: **489,969**.
- Positive-distance observations used for this question: **484,715**.
- Missing distance: **9,996**.
- Nonpositive distance: **5,254**.
- Missing fare: **0**.

### 5. Cleaning / filtering
No permanent cleaning was applied and `df_clean` was not modified.

A temporary analysis dataframe retained only rows with:
- nonmissing `distance`;
- `distance > 0`;
- nonmissing `fare_amount`.

Therefore:
- **484,715 observations were analyzed**;
- **15,250 rows were temporarily excluded** from this question;
- the 15,250 consist of **9,996 missing-distance rows + 5,254 nonpositive-distance rows**.

The exclusions exist only in the temporary P7-Q2 analysis object. The source dataset remains unchanged.

### 6. Exact analysis code
The complete reproducible code is in `p7_q2_distance_fare_functional_form.py`. Core calculations:

```python
q2 = df_clean[['distance', 'fare_amount']].dropna().copy()
q2 = q2[q2['distance'] > 0].copy()

pearson_raw = q2['distance'].corr(q2['fare_amount'], method='pearson')
spearman_raw = q2['distance'].corr(q2['fare_amount'], method='spearman')

q2['log1p_distance'] = np.log1p(q2['distance'])
q2['log1p_fare'] = np.log1p(q2['fare_amount'])

pearson_log_distance_fare = q2['log1p_distance'].corr(q2['fare_amount'])
pearson_log_log = q2['log1p_distance'].corr(q2['log1p_fare'])

q2['distance_quantile_bin'] = pd.qcut(
    q2['distance'], q=20, duplicates='drop'
)

binned = q2.groupby('distance_quantile_bin', observed=True).agg(
    observations=('distance', 'size'),
    distance_median=('distance', 'median'),
    fare_median=('fare_amount', 'median'),
    fare_per_km_median=(
        'fare_amount',
        lambda s: np.nanmedian(s / q2.loc[s.index, 'distance'])
    ),
).reset_index(drop=True)
```

The source dataset is never overwritten by this code.

### 7. Actual numerical results

| Measure | Result |
|---|---:|
| Full rows | **499,965** |
| Nonmissing distance/fare pairs | **489,969** |
| Positive-distance observations used | **484,715** |
| Temporarily excluded | **15,250** |
| Missing distance | **9,996** |
| Nonpositive distance | **5,254** |
| Missing fare | **0** |
| Pearson, raw distance vs fare | **0.6826546004** |
| Spearman, raw distance vs fare | **0.8569198647** |
| Pearson, log1p(distance) vs raw fare | **0.7622955843** |
| Pearson, log1p(distance) vs log1p(fare) | **0.8664977770** |
| Spearman, log1p(distance) vs raw fare | **0.8569198647** |
| Equal-frequency bins | **20** |

The first and last equal-frequency bins illustrate the change in the relationship:

| Statistic | First distance bin | Last distance bin |
|---|---:|---:|
| Median distance | **0.4533 km** | **14.5729 km** |
| Median fare | **$4.10** | **$37.83** |
| Median fare/km | **$9.8015/km** | **$2.5055/km** |

Across the 20 bins, median fare increases monotonically from the first bin to the last, while median fare/km generally declines substantially with increasing trip length.

The raw Pearson value here (**0.6826546**) is not expected to equal the approved Phase 5 Q5 value (**0.6708931**) because Q5 used all nonmissing distance/fare pairs, whereas P7-Q2 intentionally restricts the functional-form analysis to **positive distance**, excluding the 5,254 zero-distance observations. This is a documented analytical-sample difference, not a change to Q5.

### 8. Plots

[Distance vs Fare with 20-bin median relationship](P7_Q2_distance_fare_binned_median.png)

[Log1p Distance vs Log1p Fare](P7_Q2_log1p_distance_fare.png)

### 9. Interpretation
The positive distance–fare relationship is clearly monotonic in the positive-distance sample: the median fare rises across all 20 distance quantile bins.

However, the raw-scale relationship is not consistent with a constant fare-per-kilometer pattern. The median fare/km falls from about **$9.80/km** in the shortest distance bin to about **$2.51/km** in the longest distance bin.

The transformed comparisons also show that the association becomes numerically stronger when distance is represented on a `log1p` scale, and the log-distance/log-fare Pearson correlation is **0.8665**. This provides descriptive evidence that a transformed representation may make the relationship more regular.

This does **not** establish that a log transformation will produce a better ML model; model performance has not been evaluated.

### 10. Evidence-based insight
The distance–fare relationship is strongly positive but **not a constant-rate-per-kilometer relationship**. Fare increases with distance while the typical fare per kilometer declines as trips become longer.

The stronger transformed-scale association provides evidence that a logarithmic distance representation is worth **considering** in Task 2 as one candidate representation. It should be evaluated empirically there rather than treated as a predetermined modeling decision.

### 11. Limitations
- The positive-distance filter excludes **5,254 zero/nonpositive-distance observations** and **9,996 missing-distance observations** for this specific analysis.
- The equal-frequency bins describe the relationship but do not fit a predictive model.
- Pearson correlation only measures linear association on the chosen scale.
- The log1p correlations are descriptive evidence, not proof of superior predictive performance.
- Extreme distance values remain visible in the analysis; no additional outlier removal was performed.
- No causality is inferred.

### 12. Task 2 relevance
The findings support **considering**:

1. raw distance versus transformed distance representations;
2. nonlinear treatment of distance;
3. evaluation of whether a transformed target/distance representation improves model behavior later.

These are candidate modeling considerations, not confirmed feature-engineering or modeling decisions.

## P7-Q2 Verification

- `df_clean` shape: **499,965 × 26** before and after analysis.
- Positive-distance observations used: **484,715**.
- Temporary exclusions: **15,250**.
- Missing distance: **9,996**.
- Nonpositive distance: **5,254**.
- 20 equal-frequency bins created successfully.
- Pearson/Spearman and transformed correlations reproduce the saved CSV values.
- First/last-bin median statistics reproduce the saved CSV values.
- Plot data are generated from the same temporary analysis dataframe as the reported statistics.
- SHA-256 of `df_clean.csv` after P7-Q2 matches the hash recorded after P7-Q1.
- No feature selection, feature removal, modeling, or permanent cleaning was performed.

### P7-Q2 Status
**COMPLETED AND VERIFIED**

**P7-Q3 NOT STARTED.**

# P7-Q3 — Distance–Fare Relationship Stability Across Time

## 1. Question
Does the distance–fare relationship remain similar across important time periods, or does its pattern vary by hour/weekday?

## 2. Why this question is justified
Phase 5 established that distance had the strongest observed numerical association with fare in the approved comparison. Phase 6 then identified meaningful temporal structure, including hour × weekday demand patterns and hour-specific traffic/fare differences. P7-Q3 therefore checks whether the principal distance–fare relationship itself is reasonably stable across time rather than assuming that the overall association applies uniformly.

## 3. Plot choice and why
Two complementary descriptive plots were used:

1. **Distance–fare binned-median lines by four dayparts** (`00-05`, `06-11`, `12-17`, `18-23`). Each line uses 10 equal-frequency distance bins within the daypart and plots median fare against median distance. This preserves the distance structure while avoiding a 24-line plot that would largely repeat Phase 6's hourly demand work.
2. **Hourly Pearson correlation line** as a compact stability diagnostic. This checks whether the distance–fare linear association varies materially by individual pickup hour.

No predictive model or fitted regression line was used.

## 4. Data / observations used
- Source: Phase 3 `df_clean`, read-only.
- Full dataset: **499,965 rows × 26 columns**.
- Positive-distance observations used: **484,715**.
- Missing distance: **9,996**.
- Nonpositive distance: **5,254**.
- Missing fare: **0**.
- Temporary exclusions: **15,250**.

The same positive-distance rule used in P7-Q2 was retained so that the functional-form comparison remains comparable.

## 5. Cleaning / filtering
No permanent cleaning was applied.

A temporary analysis dataframe retained rows with nonmissing `distance`, positive `distance`, nonmissing `fare_amount`, and available `hour`/`weekday`.

The source `df_clean` was not modified.

Four descriptive dayparts were used solely to summarize the 24 hours:
- `00-05`
- `06-11`
- `12-17`
- `18-23`

This is a visualization grouping, not a new source-data category or permanent feature.

## 6. Exact analysis code
The complete reproducible code is in `p7_q3_distance_fare_temporal_stability.py`. Core calculations:

```python
q3 = df_clean[['distance', 'fare_amount', 'hour', 'weekday']].copy()
q3 = q3.dropna(subset=['distance', 'fare_amount', 'hour', 'weekday']).copy()
q3 = q3[q3['distance'] > 0].copy()

q3['daypart'] = pd.cut(
    q3['hour'],
    bins=[-1, 5, 11, 17, 23],
    labels=['00-05', '06-11', '12-17', '18-23'],
    ordered=True
)

q3['distance_bin'] = q3.groupby(
    'daypart', observed=True
)['distance'].transform(
    lambda s: pd.qcut(s, q=10, labels=False, duplicates='drop') + 1
)

binned = q3.groupby(
    ['daypart', 'distance_bin'], observed=True
).agg(
    observations=('distance', 'size'),
    distance_median=('distance', 'median'),
    fare_median=('fare_amount', 'median'),
    fare_mean=('fare_amount', 'mean'),
).reset_index()

for dp in ['00-05', '06-11', '12-17', '18-23']:
    sub = q3[q3['daypart'] == dp]
    pearson = sub['distance'].corr(sub['fare_amount'], method='pearson')
    spearman = sub['distance'].corr(sub['fare_amount'], method='spearman')
```

The script also calculates weekday/weekend summary correlations and hourly Pearson correlations as stability checks.

## 7. Actual numerical results

### Daypart summary

| Daypart | Observations | Median distance (km) | Mean fare | Median fare | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|---:|
| 00-05 | 61,780 | 2.8061 | $12.1028 | $9.00 | **0.791382** | **0.895418** |
| 06-11 | 116,925 | 2.0272 | $11.0679 | $8.10 | **0.660364** | **0.846825** |
| 12-17 | 139,283 | 1.9924 | $11.6419 | $8.50 | **0.636924** | **0.837978** |
| 18-23 | 166,727 | 2.2832 | $11.0129 | $8.50 | **0.721940** | **0.868097** |

### Weekday vs weekend stability check

| Group | Observations | Pearson r | Spearman r | Median fare | Mean fare |
|---|---:|---:|---:|---:|---:|
| Weekday | 347,485 | **0.668272** | **0.853016** | $8.50 | $11.3611 |
| Weekend | 137,230 | **0.724686** | **0.868667** | $8.50 | $11.3073 |

### Hourly Pearson range

All 24 hours had positive Pearson associations. The hourly values ranged from:
- minimum: **0.422392 at 17:00**;
- maximum: **0.889246 at 06:00**.

Selected values include:
- 04:00 = **0.872353**
- 11:00 = **0.479178**
- 17:00 = **0.422392**
- 19:00 = **0.629842**
- 21:00 = **0.771530**

## 8. Interpretation
The distance–fare association remains positive across every daypart and every individual pickup hour examined, but its strength is not identical across time.

The strongest daypart Pearson association is overnight (`00-05`, **0.7914**), while the weakest is afternoon (`12-17`, **0.6369**). The hourly diagnostic shows a wider range, with the weakest hourly association at 17:00 (**0.4224**) and the strongest at 06:00 (**0.8892**).

The binned-median plot also shows broadly similar upward distance–fare patterns across dayparts, although their median fare levels and trajectories are not identical.

The weekday/weekend check shows a similar positive relationship in both groups, with somewhat higher Pearson/Spearman values on weekends.

## 9. Evidence-based insight
The overall distance–fare relationship is **not confined to a single time period**: it remains positive across all hours and both weekday/weekend groups.

At the same time, the association strength varies by hour/daypart. Therefore, the approved Q5 overall relationship should not be interpreted as evidence that the exact distance–fare relationship is temporally identical everywhere in the dataset.

This provides descriptive evidence that temporal variables may modify the relationship between distance and fare, but it does not establish a causal interaction or prove that an interaction feature will improve a model.

## 10. Limitations
- **15,250 observations** were temporarily excluded because distance was missing or nonpositive; `df_clean` was not changed.
- Four dayparts are a descriptive grouping of hours, not original dataset categories.
- Correlations describe association and do not establish causation.
- Correlation differences across groups do not by themselves prove a statistical interaction.
- No model comparison or feature-importance analysis was performed.
- The hourly Pearson diagnostic is descriptive and should not be interpreted as a ranking of predictive importance.

## 11. Task 2 relevance
The result supports **considering** whether temporal variables or distance × time interactions should be evaluated later in Task 2. It does not establish that such interactions are necessary or that they will improve predictive performance.

## P7-Q3 Verification

- `df_clean` shape before/after: **499,965 × 26**.
- Positive-distance observations used: **484,715**.
- Temporary exclusions: **15,250**.
- Missing distance: **9,996**.
- Nonpositive distance: **5,254**.
- Missing fare: **0**.
- Four dayparts created; each contains 10 distance bins.
- Daypart statistics reproduce `P7_Q3_daypart_relationship_summary.csv`.
- Distance/fare binned statistics reproduce `P7_Q3_distance_fare_by_daypart.csv`.
- Weekday/weekend values reproduce `P7_Q3_weekday_weekend_summary.csv`.
- Hourly Pearson values reproduce `P7_Q3_hour_relationship_summary.csv`.
- SHA-256 of `df_clean.csv` after P7-Q3 matches the hash recorded after P7-Q2.
- No feature selection, feature removal, modeling, or permanent cleaning was performed.

### P7-Q3 Status
**COMPLETED AND VERIFIED**

**P7-Q4 NOT STARTED.**

## P7-Q4 — Reference/Airport Distance Redundancy vs Target Information

### Status
**COMPLETED AND VERIFIED**

### Question
How redundant are the airport/reference-distance variables with one another, and how much descriptive target information does each provide about `fare_amount`?

### Why this question was justified
Phase 6 identified strong numeric redundancy among some distance/reference-distance variables, while Phase 5 showed that the individual airport/reference-distance variables have different associations with fare. P7-Q4 therefore checks the structure of that redundancy and documents the descriptive relationship of each reference-distance variable with the target without performing feature selection or modeling.

### Source and controls
- Source: `/mnt/data/phase3/df_clean.csv`
- `df_clean` was read-only throughout the analysis.
- Selected columns only: `jfk_dist`, `ewr_dist`, `lga_dist`, `sol_dist`, `nyc_dist`, `fare_amount`.
- `df_clean` shape remained `499,965 × 26`.
- SHA-256 before and after Q4: `1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509`.

### Temporary filtering
For each reference-distance vs fare correlation, rows with a missing value in that reference-distance column were excluded only for that pairwise calculation. `fare_amount` had no missing values.

For the reference-distance redundancy matrices, pairwise complete observations were used. All five reference-distance columns had the same missingness pattern, so every pairwise correlation used `489,671` observations.

No rows were deleted from `df_clean`, no values were imputed, no feature was removed, and no model was fitted.

### Results — reference distance vs fare
| Feature | N | Pearson r | Spearman r |
|---|---:|---:|---:|
| `jfk_dist` | 489,671 | -0.265021 | -0.187893 |
| `ewr_dist` | 489,671 | 0.275987 | 0.147962 |
| `lga_dist` | 489,671 | 0.079585 | 0.054560 |
| `sol_dist` | 489,671 | 0.248432 | 0.108968 |
| `nyc_dist` | 489,671 | 0.290524 | 0.148133 |

The descriptive associations are not identical across the reference-distance variables. `jfk_dist` is negatively associated with fare in this pairwise analysis, while `ewr_dist`, `lga_dist`, `sol_dist`, and `nyc_dist` are positively associated. The magnitudes also differ between Pearson and Spearman measures, so the results should be treated as descriptive associations rather than evidence of a simple linear or monotonic mechanism.

### Results — redundancy among reference-distance variables
Pearson correlations:

| | jfk_dist | ewr_dist | lga_dist | sol_dist | nyc_dist |
|---|---:|---:|---:|---:|---:|
| `jfk_dist` | 1.000000 | 0.367998 | 0.587271 | 0.461160 | 0.447220 |
| `ewr_dist` | 0.367998 | 1.000000 | 0.397319 | 0.977466 | 0.969378 |
| `lga_dist` | 0.587271 | 0.397319 | 1.000000 | 0.380854 | 0.431180 |
| `sol_dist` | 0.461160 | 0.977466 | 0.380854 | 1.000000 | 0.992120 |
| `nyc_dist` | 0.447220 | 0.969378 | 0.431180 | 0.992120 | 1.000000 |

The strongest Pearson redundancy is among `sol_dist`, `ewr_dist`, and `nyc_dist`: `sol_dist`–`nyc_dist` = `0.992120`, `ewr_dist`–`sol_dist` = `0.977466`, and `ewr_dist`–`nyc_dist` = `0.969378`. Other pairs are materially lower by comparison.

Spearman correlations show the same strong association within the EWR/SOL/NYC group, while also showing different rank-order relationships for pairs involving `lga_dist` and `jfk_dist`. This is why the Q4 conclusion is based on the full matrices rather than on one correlation measure alone.

### Missingness
Each reference-distance column has `10,294` missing values, equal to `2.058944%` of `df_clean`. `fare_amount` has `0` missing values. Therefore every reference-distance/fare calculation uses `489,671` complete observations.

### Plots
1. `P7_Q4_reference_distance_redundancy_heatmap.png` — Pearson redundancy matrix for the five reference-distance variables. Visual inspection confirms the very high EWR/SOL/NYC block and the lower relationships involving the other variables.
2. `P7_Q4_reference_distance_target_association.png` — Pearson and Spearman associations of each reference-distance variable with fare.

### Interpretation
P7-Q4 provides descriptive evidence that the reference-distance variables are not equally redundant. In particular, `ewr_dist`, `sol_dist`, and `nyc_dist` form a highly correlated group, while `jfk_dist` and `lga_dist` have weaker Pearson relationships with that group. At the same time, the reference-distance variables have different pairwise associations with the fare target, including the negative Pearson association for `jfk_dist` and positive Pearson associations for the other four variables.

This means the dataset contains substantial overlapping reference-distance information, but the Q4 analysis alone does not establish that any particular variable should be removed. Correlation is descriptive and does not measure unique predictive contribution after accounting for the other variables.

### Task 2 relevance
For later modeling, the strong EWR/SOL/NYC redundancy should be explicitly considered when interpreting coefficients, feature importance, or other model diagnostics. The Q4 results support treating this as a modeling-risk consideration, not as a feature-selection decision. Any decision to keep, transform, combine, or remove variables must be evaluated later with the approved modeling procedure and validation design.

### Limitations
- The analysis is pairwise/descriptive and does not estimate unique contribution after controlling for the other reference-distance variables.
- Correlation does not establish causation.
- Missing reference-distance observations were excluded only from the relevant pairwise calculation.
- No model comparison, feature selection, dimensionality reduction, or target prediction was performed.

### Reproducibility artifacts
- Code: `/mnt/data/phase7/p7_q4_reference_distance_redundancy.py`
- Run output: `/mnt/data/phase7/P7_Q4_run_output.txt`
- Target association CSV: `/mnt/data/phase7/P7_Q4_reference_distance_target_summary.csv`
- Pearson redundancy matrix: `/mnt/data/phase7/P7_Q4_reference_distance_pearson_matrix.csv`
- Spearman redundancy matrix: `/mnt/data/phase7/P7_Q4_reference_distance_spearman_matrix.csv`
- Pairwise overlap CSV: `/mnt/data/phase7/P7_Q4_reference_distance_pairwise_overlap.csv`
- Missingness CSV: `/mnt/data/phase7/P7_Q4_reference_distance_missingness.csv`
- Heatmap: `/mnt/data/phase7/P7_Q4_reference_distance_redundancy_heatmap.png`
- Target-association plot: `/mnt/data/phase7/P7_Q4_reference_distance_target_association.png`
- Verification: `/mnt/data/phase7/P7_Q4_verification.csv`
- Artifact package: `/mnt/data/phase7/P7_Q4_artifacts.zip`

### Verification record
All Q4 verification checks passed:
- `df_clean` rows = `499,965`.
- Selected Q4 columns = `6`.
- Missing fare = `0`.
- Missing count for every reference-distance variable = `10,294`.
- Target-association summary contains `5` variables.
- Pearson redundancy matrix = `5 × 5`.
- Spearman redundancy matrix = `5 × 5`.
- Pairwise complete-observation count = `489,671` for every reference-distance pair.
- `df_clean` SHA-256 unchanged.

### P7-Q4 final status
**COMPLETED — CODE → CSV OUTPUTS → PLOTS → REPORT VERIFIED.**

**P7-Q5: NOT STARTED.**
**MODELING: NOT STARTED.**
**`df_clean`: UNCHANGED / READ-ONLY.**

## P7-Q5 — Consolidated Data-Quality / Modeling-Risk Audit

### Status
**COMPLETED AND VERIFIED**

### Question
After Phase 3 cleaning and the completed Phase 7 analyses, what remaining data-quality conditions and modeling risks should be explicitly documented before Task 2 modeling begins?

### Why this question was justified
P7-Q5 is the final Phase 7 audit. It consolidates the already-established data-quality and modeling-risk areas without performing feature selection or modeling. It is an audit rather than a new exploratory relationship analysis.

### Source and controls
- Source: `/mnt/data/phase3/df_clean.csv`
- `df_clean` was loaded read-only. No values, rows, columns, or schema were modified.
- No P7-Q1, P7-Q2, P7-Q3, or P7-Q4 analysis was rerun.
- The omitted P7-Q4 conditional distance-band diagnostic was not executed.
- No modeling, feature selection, imputation, deduplication, or new cleaning rule was introduced.
- Q1 target-tail values used in the consolidated audit were imported from the already-approved P7-Q1 output rather than recalculated.
- Q4 redundancy evidence was referenced from the already-approved Q4 matrix rather than recalculated.

### Data scope and temporary filtering
The audit used the full `df_clean` for integrity, schema, missingness, identifier, categorical, and temporal checks. Temporary parsing of `pickup_datetime` was used only to compare it with the existing `hour`, `day`, `month`, `weekday`, and `year` columns. Temporary pairwise exclusions were not needed for the core audit; no rows were removed from the source dataset.

### Results — target integrity and approved P7-Q1 evidence
- `fare_amount` missing: **0**.
- `fare_amount <= 0`: **0** after Phase 3 cleaning.
- Maximum fare: **500.00**.
- Approved P7-Q1 P99: **52.136**.
- Approved P7-Q1 P99.5: **57.33**.
- Approved P7-Q1 P99.9: **80.00**.
- Approved P7-Q1 skewness: **4.899901**.

These values document the already-established high/right-skewed target tail. P7-Q5 does not reclassify or remove those observations.

### Results — remaining missingness
| Column | Missing | Percent |
|---|---:|---:|
| `distance` | 9,996 | 1.999340% |
| `passenger_count` | 1,796 | 0.359225% |
| `jfk_dist` | 10,294 | 2.058944% |
| `ewr_dist` | 10,294 | 2.058944% |
| `lga_dist` | 10,294 | 2.058944% |
| `sol_dist` | 10,294 | 2.058944% |
| `nyc_dist` | 10,294 | 2.058944% |
| `bearing` | 9,912 | 1.982539% |
| `pickup_longitude` | 9,439 | 1.887932% |
| `pickup_latitude` | 9,413 | 1.882732% |
| `dropoff_longitude` | 9,464 | 1.892933% |
| `dropoff_latitude` | 9,434 | 1.886932% |

The remaining missing values are intentional outcomes of the documented Phase 3 cleaning process. P7-Q5 does not impute them.

### Results — numeric integrity
All Phase-3-supported integrity checks passed:
- `passenger_count == 0`: **0**.
- `distance < 0`: **0**.
- `distance > 1000`: **0**.
- All five reference-distance columns with values `>1000`: **0 cells**.
- `bearing` outside `[-π, π]`: **0**.
- Non-missing coordinate values equal to zero: **0** for all four coordinate columns.

No new threshold was introduced by P7-Q5.

### Results — temporal consistency and coverage
- Missing `pickup_datetime`: **0**.
- Invalid `hour` values outside `0–23`: **0**.
- Invalid `day` values outside `1–31`: **0**.
- Invalid `month` values outside `1–12`: **0**.
- Invalid `weekday` values outside `0–6`: **0**.
- `hour` vs datetime hour mismatches: **0**.
- `day` vs datetime day-of-month mismatches: **0**.
- `month` vs datetime month mismatches: **0**.
- `weekday` vs datetime weekday mismatches: **0**.
- `year` vs datetime year mismatches: **0**.
- Observed year range: **2009–2015**.
- Datetime range: **2009-01-01 00:31:32** through **2015-06-30 23:38:21**.
- Unique year-month periods: **78**.

The existing temporal fields are internally consistent with `pickup_datetime` in this audit. No new temporal feature was created.

### Results — duplicate / identifier structure
- `key` unique values: **461,173**.
- Distinct duplicated `key` values: **34,066**.
- Excess rows associated with duplicated `key` values: **38,792**.
- Exact duplicate rows: **0**.
- `User ID` unique values: **499,965** with **0** excess duplicates.
- `User Name` unique values: **221,665**.
- `Driver Name` unique values: **221,693**.

The duplicate-key structure is retained by design and is distinct from exact duplicate rows. P7-Q5 does not deduplicate the data.

### Results — categorical representation
The audit preserved the actual observed categories without recoding.

**Car Condition:**
- Very Good: 125,306
- Bad: 124,965
- Good: 124,959
- Excellent: 124,735

**Weather:**
- sunny: 100,424
- cloudy: 100,056
- rainy: 99,967
- stormy: 99,947
- windy: 99,571

**Traffic Condition:**
- Congested Traffic: 166,838
- Dense Traffic: 166,574
- Flow Traffic: 166,553

No unexpected category was created or inferred by the audit.

### Results — redundant numerical variables
P7-Q4 was **not rerun**. The accepted Q4 evidence is referenced only as a prior finding: the strongest Pearson redundancies were `sol_dist ↔ nyc_dist = 0.992120`, `ewr_dist ↔ sol_dist = 0.977466`, and `ewr_dist ↔ nyc_dist = 0.969378`.

This is documented as a modeling-risk consideration rather than a feature-selection decision.

### Results — identifier / possible leakage concern
Identifier-like columns were audited separately from analytical variables. The audit does **not** establish that any column is leakage. It establishes that `key`, user/driver identifiers, and names have identifier-like cardinality/duplication structures that should be reviewed before modeling. No identifier was removed or recoded in Phase 7.

### Consolidated risk register
| Risk | Evidence | Task 2 relevance |
|---|---|---|
| Target tail | Approved P7-Q1 target evidence | Consider target transformation/outlier treatment later; no treatment selected here. |
| Remaining missing predictors | P7-Q5 missingness audit | A missing-value strategy will be needed during modeling. |
| Duplicate keys | P7-Q5 identifier audit | Splitting/validation and identifier handling should account for duplicated keys. |
| Temporal coverage | P7-Q5 temporal audit | Time-aware validation/splitting may need consideration. |
| Reference-distance redundancy | Approved P7-Q4 evidence | Later model diagnostics should account for correlated predictors. |
| Categorical representation | P7-Q5 categorical audit | An encoding strategy will be needed in Task 2. |
| Identifier/leakage concern | P7-Q5 identifier audit | Identifier-like fields should be reviewed before modeling; leakage is not established by this audit. |

### Plots
1. `P7_Q5_missingness_diagnostic.png` — remaining missingness among relevant variables.
2. `P7_Q5_audit_check_status.png` — audit-check status; all 23 numeric/temporal integrity checks passed and no failed checks were recorded.

### Interpretation
The final audit confirms that Phase 3 cleaning produced a structurally consistent `df_clean`, while intentionally retaining several modeling-relevant missing-value patterns and the documented duplicate-key structure. The target retains a pronounced upper tail, and the accepted Q4 evidence establishes strong redundancy among part of the reference-distance feature set. These are modeling considerations, not reasons to alter `df_clean` during Phase 7.

The temporal fields are internally consistent with `pickup_datetime`, and the categorical fields retain their actual observed domains. The audit also distinguishes identifier-like structure from leakage: the presence of identifiers does not itself prove leakage, so no leakage conclusion is made here.

### Limitations
- P7-Q5 is an audit and does not test model performance.
- P7-Q1 and P7-Q4 numerical findings used in the consolidated register were imported from their approved outputs rather than recalculated, preserving the instruction not to rerun earlier questions.
- Correlation redundancy from P7-Q4 does not establish unique predictive contribution.
- The audit does not determine the optimal missing-value, encoding, splitting, target-transformation, or feature-selection strategy.
- No causal conclusions are made.

### Task 2 relevance
P7-Q5 identifies the conditions that should be carried into Task 2 planning: target-tail treatment, handling of intentional missingness, duplicate-key-aware validation, temporal coverage, reference-distance redundancy, categorical encoding, and review of identifier-like variables for leakage risk. These are documented considerations only; no modeling decision is made in Phase 7.

### Reproducibility artifacts
- Code: `/mnt/data/phase7/p7_q5_data_quality_modeling_risk_audit.py`
- Run output: `/mnt/data/phase7/P7_Q5_run_output.txt`
- Schema audit: `/mnt/data/phase7/P7_Q5_schema_audit.csv`
- Missingness audit: `/mnt/data/phase7/P7_Q5_missingness_audit.csv`
- Target audit: `/mnt/data/phase7/P7_Q5_target_audit.csv`
- Numeric integrity audit: `/mnt/data/phase7/P7_Q5_numeric_integrity_audit.csv`
- Temporal consistency audit: `/mnt/data/phase7/P7_Q5_temporal_consistency_audit.csv`
- Temporal coverage audit: `/mnt/data/phase7/P7_Q5_temporal_coverage_audit.csv`
- Identifier/duplicate audit: `/mnt/data/phase7/P7_Q5_identifier_duplicate_audit.csv`
- Categorical audit: `/mnt/data/phase7/P7_Q5_categorical_audit.csv`
- Risk register: `/mnt/data/phase7/P7_Q5_risk_register.csv`
- Missingness plot: `/mnt/data/phase7/P7_Q5_missingness_diagnostic.png`
- Audit-check plot: `/mnt/data/phase7/P7_Q5_audit_check_status.png`
- Verification: `/mnt/data/phase7/P7_Q5_verification.csv`

### Verification record
All P7-Q5 verification checks passed:
- `df_clean` shape = **499,965 × 26**.
- `df_clean` SHA-256 before audit = `1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509`.
- `df_clean` SHA-256 after audit = `1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509`.
- Exact schema and column order match the expected 26-column `df_clean` structure.
- Fare missing/non-positive checks passed.
- Passenger-count zero check passed.
- Distance and reference-distance Phase-3 threshold checks passed.
- Bearing-range check passed.
- Datetime missingness and all temporal consistency checks passed.
- Exact duplicate-row count = **0**.
- P7-Q1–Q4 were not rerun.
- The omitted Q4 conditional distance-band diagnostic was not executed.
- No modeling or presentation work was started.

### P7-Q5 final status
**COMPLETED — CODE → CSV OUTPUTS → PLOTS → REPORT VERIFIED.**

**PHASE 7: P7-Q1 through P7-Q5 COMPLETED AND VERIFIED.**

**TASK 2 MODELING: NOT STARTED.**
**PRESENTATION: NOT STARTED.**
**`df_clean`: UNCHANGED / READ-ONLY.**
