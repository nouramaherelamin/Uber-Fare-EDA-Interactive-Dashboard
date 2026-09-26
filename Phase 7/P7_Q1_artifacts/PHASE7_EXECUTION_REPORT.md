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
