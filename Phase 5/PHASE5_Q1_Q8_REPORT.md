# Phase 5 — Q1–Q8 Official Task Questions

Dataset: `df_clean` from Phase 3, 499,965 rows × 26 columns. No dataset modifications were made during Phase 5.

Task 1 PDF verification:
The official Cellula Technologies Uber Fare Prediction / Exploratory Data Analysis Task 1 PDF has been uploaded and verified. It contains the official Q1–Q8 questions and their required visualization guidance.

## Q1. Is higher Passenger_Count associated with higher Fare_Amount?

**1. Official Question**  
Is higher Passenger_Count associated with higher Fare_Amount?

**2. Appropriate Plot**  
**Scatter plot of `passenger_count` vs `fare_amount`.**

**3. Why This Plot Is Appropriate**  
Both variables are observed at the trip level. A scatter plot preserves the individual trip observations and directly shows the relationship between passenger count and fare. Because `passenger_count` is discrete, vertical bands are expected. Transparency is used to reduce overplotting; no aggregation or jitter is used, so the plotted x/y values remain the actual observations.

**4. Relevant Cleaning / Data Handling Note**  
Phase 3 transformed `passenger_count = 0` to `NaN` and retained those rows. For Q1, the scatter plot and Pearson correlation use all rows in `df_clean` with nonmissing `passenger_count` and `fare_amount`. `df_clean` itself was not modified. The grouped mean/median table remains supplementary evidence only.

**5. Exact Python Code**  
```python
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
```

**6. Actual Result from full `df_clean`**  
The full-data Pearson correlation among nonmissing passenger counts is **r = 0.01275**.

Supplementary grouped results:

| Passenger count | Trips | Mean fare | Median fare |
|---:|---:|---:|---:|
| 1 | 345,983 | 11.2324 | 8.50 |
| 2 | 73,904 | 11.8450 | 8.50 |
| 3 | 21,759 | 11.4870 | 8.50 |
| 4 | 10,613 | 11.7816 | 8.50 |
| 5 | 35,321 | 11.2109 | 8.50 |
| 6 | 10,589 | 12.3286 | 9.00 |

There are **1,796** rows with missing passenger count.

**7. Plot**  
`Q1_passenger_vs_fare.png` — scatter plot using the full available trip-level observations; transparency is used only for readability.

**8. Interpretation**  
The scatter plot forms vertical bands because passenger count takes discrete values. Within those bands, fare values are widely dispersed, and there is no visually clear monotonic increase in fare as passenger count rises. The Pearson correlation of **0.01275** is also very close to zero.

**9. Evidence-based Insight**  
The full trip-level scatter plot, supported by the grouped statistics and Pearson correlation, does not show a clear linear association between passenger count and fare amount.

**10. Limitations / Important Caveats**  
Passenger count is missing for 1,796 rows, and group sizes differ substantially. The scatter plot is dense because the dataset is large; transparency improves readability but does not alter the underlying observations. This is an observational association and does not establish causation.

**11. Relevance to Task 2**  
Passenger count can remain available as a candidate feature, but this EDA alone does not establish it as a strong standalone fare signal. Its eventual usefulness should be evaluated during the later modeling stage.

## Q2. Does Car Condition influence Fare Amount?

**1. Official Question**  
Does Car Condition influence Fare Amount?

**2. Appropriate Plot**  
Bar chart of mean fare by the actual `Car Condition` categories.

**3. Why This Plot Is Appropriate**  
Car Condition is categorical and fare is continuous. Comparing group means directly addresses whether observed average fares differ across categories.

**4. Relevant Cleaning / Data Handling Note**  
The actual dataset categories were preserved: Bad, Excellent, Good, Very Good. No recoding was performed.

**5. Exact Python Code**  
See Q2 section of `phase5_q1_q8_analysis.py`.

**6. Actual Result from full df_clean**

| Car Condition | Trips | Mean fare | Median fare |
|---|---:|---:|---:|
| Bad | 124,965 | 11.3277 | 8.50 |
| Excellent | 124,735 | 11.3559 | 8.50 |
| Good | 124,959 | 11.3376 | 8.50 |
| Very Good | 125,306 | 11.4163 | 8.50 |

Mean-fare range across categories = **0.0886**.

**7. Plot**  
`Q2_car_condition_fare.png`

**8. Interpretation**  
Mean fares are very close across all four categories, and the median is identical at 8.50 in every category.

**9. Evidence-based Insight**  
The observed fare differences by Car Condition are small in this dataset; the group means do not show a large separation.

**10. Limitations / Important Caveats**  
A grouped mean comparison alone does not control for distance, time, traffic, or other variables. Therefore it does not establish that Car Condition itself causes fare differences.

**11. Relevance to Task 2**  
Car Condition can remain available as a candidate categorical feature, but this EDA alone does not show a large marginal fare difference.

---

## Q3. At what hour are fares typically highest?

**1. Official Question**  
At what hour are fares typically highest?

**2. Appropriate Plot**  
Line plot of mean and median fare by pickup hour.

**3. Why This Plot Is Appropriate**  
Hour is ordered and cyclical, so a 24-point line plot makes it easy to compare the typical fare pattern across the full day. Both mean and median are shown because fare is strongly right-skewed.

**4. Relevant Cleaning / Data Handling Note**  
`hour` was validated against `pickup_datetime` in Phase 3. No temporal rows were removed for hour inconsistencies.

**5. Exact Python Code**  
See Q3 section of `phase5_q1_q8_analysis.py`.

**6. Actual Result from full df_clean**  
- Highest **median** fare: **04:00**, median **$10.00**.
- Highest **mean** fare: **05:00**, mean **$15.1696**.
- At 05:00, median fare is **$9.00**.

**7. Plot**  
`Q3_hour_fare.png`

**8. Interpretation**  
The answer depends on the definition of "typically." Using the median, which is less sensitive to extreme fares, 04:00 has the highest typical fare. The mean peaks at 05:00, indicating that high-fare observations at that hour raise the average.

**9. Evidence-based Insight**  
The full-data results show a difference between mean and median at the highest-fare hours; therefore a single mean-only conclusion would hide the skewness of fare values.

**10. Limitations / Important Caveats**  
Some hours have very different trip counts. Hourly fare differences may also reflect other variables such as distance or trip mix. No causal claim is made.

**11. Relevance to Task 2**  
Hour is a potentially useful temporal feature for later modeling because fare distributions vary across hours, but its final predictive value must be evaluated by modeling.

---

## Q4. Does Traffic Condition affect trip fare?

**1. Official Question**  
Does Traffic Condition affect trip fare?

**2. Appropriate Plot**  
Bar chart of mean fare by the actual Traffic Condition categories.

**3. Why This Plot Is Appropriate**  
Traffic Condition is categorical and fare is continuous, so group-level fare comparison directly addresses the question.

**4. Relevant Cleaning / Data Handling Note**  
The actual categories were preserved: Congested Traffic, Dense Traffic, Flow Traffic.

**5. Exact Python Code**  
See Q4 section of `phase5_q1_q8_analysis.py`.

**6. Actual Result from full df_clean**

| Traffic Condition | Trips | Mean fare | Median fare |
|---|---:|---:|---:|
| Congested Traffic | 166,838 | 11.3915 | 8.50 |
| Dense Traffic | 166,574 | 11.3690 | 8.50 |
| Flow Traffic | 166,553 | 11.3177 | 8.50 |

Mean-fare range = **0.0739**.

**7. Plot**  
`Q4_traffic_fare.png`

**8. Interpretation**  
The mean fares are close together and all three medians are 8.50. Congested Traffic has the highest mean, but the difference from Flow Traffic is only about $0.074.

**9. Evidence-based Insight**  
The observed marginal fare differences across Traffic Condition are small in the full cleaned dataset.

**10. Limitations / Important Caveats**  
Traffic categories are observational labels. The comparison does not control for distance, hour, weather, or other trip characteristics and therefore does not establish causation.

**11. Relevance to Task 2**  
Traffic Condition can be considered as a candidate categorical feature, but this one-variable EDA does not indicate a large standalone fare separation.

---

## Q5. Is trip Distance the strongest predictor of the Fare Amount?

**1. Official Question**  
Is trip distance the strongest predictor of the fare amount?

**2. Appropriate Plot**  
Scatter plot of `distance` (x-axis) versus `fare_amount` (y-axis), using the full available `df_clean` observations.

**3. Why This Plot Is Appropriate**  
The official Task 1 PDF specifies a Scatter Plot for Q5 because both `Distance` and `Fare_Amount` are numerical. The plot preserves trip-level observations and allows the visual relationship, direction, spread, and approximate linearity to be examined directly. Because the dataset is large, transparency is used only to reduce overplotting; the observations are not aggregated or replaced by summary values. fileciteturn20file0L109-L114

**4. Relevant Cleaning / Data Handling Note**  
The analysis uses `df_clean` only and does not modify it. For the main scatter plot and the Distance-vs-Fare Pearson correlation, rows with missing `distance` or `fare_amount` are excluded pairwise. This leaves **489,969 available trip observations** for Q5. No additional outlier deletion or transformation was performed. The Pearson correlation comparison across relevant numeric variables is supporting EDA evidence, not ML feature importance.

**5. Exact Python Code**  
```python
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
```

**6. Actual Result from full `df_clean`**  
The main scatter plot uses **489,969** nonmissing `distance`/`fare_amount` observations from the full `df_clean`. The points show an overall positive relationship: higher trip distances are generally associated with higher fares, while substantial vertical spread remains at many distance values. The relationship is not a perfectly tight line.

Pearson correlation between `distance` and `fare_amount` is **0.670893**.

Supporting Pearson correlations with fare are:

| Variable | Pearson r with fare |
|---|---:|
| **distance** | **0.670893** |
| nyc_dist | 0.290524 |
| ewr_dist | 0.275987 |
| jfk_dist | -0.265021 |
| sol_dist | 0.248432 |
| lga_dist | 0.079585 |
| bearing | -0.024115 |
| passenger_count | 0.012749 |

By **absolute Pearson correlation**, `distance` has the largest observed numerical association with `fare_amount` among these specified numeric variables.

**7. Plot**  
`Q5_distance_vs_fare.png` — main Q5 scatter plot.

`Q5_numeric_associations.csv` — supporting Pearson correlation comparison.

**8. Interpretation**  
The scatter plot shows a clear overall upward pattern between trip distance and fare amount, meaning fares generally tend to be higher for longer trips in this cleaned dataset. However, the points are widely dispersed rather than lying tightly around one straight line. The Pearson value of **0.670893** indicates a substantial positive **linear association** in this EDA comparison.

**9. Evidence-based Insight**  
Within the specified numerical EDA comparison, `distance` has the **strongest observed numerical association** with `fare_amount` by absolute Pearson correlation. This supports treating trip distance as an important candidate variable for later modeling. It does **not** establish that `distance` is the strongest ML predictor or feature by model-based feature importance.

**10. Limitations / Important Caveats**  
Pearson correlation measures linear association and does not capture all nonlinear patterns. The scatter plot also shows substantial variability in fare at similar distances, so distance alone does not explain every fare difference. Correlation does not imply causation. Finally, EDA correlation magnitude is not equivalent to machine-learning feature importance or predictive performance; those require a later modeling evaluation.

**11. Relevance to Task 2**  
Trip distance is a strong candidate numerical feature to evaluate in Task 2 because it has the largest observed absolute Pearson association with fare among the specified numeric variables. This is a candidate based on EDA, not a final model-based feature-importance conclusion.

---

## Q6. At what time of day are ride requests most frequent?

**1. Official Question**  
At what time of day are ride requests most frequent?

**2. Appropriate Plot**  
Bar chart of trip count by pickup hour.

**3. Why This Plot Is Appropriate**  
The question concerns frequency/demand, so the correct quantity is the number of recorded trips, not fare amount.

**4. Relevant Cleaning / Data Handling Note**  
Trip counts are calculated from the full `df_clean`. The 35 invalid-fare rows removed in Phase 3 are therefore not included.

**5. Exact Python Code**  
See Q6 section of `phase5_q1_q8_analysis.py`.

**6. Actual Result from full df_clean**  
- Highest trip frequency: **19:00 — 31,380 trips**.
- Next: 18:00 — 30,063.
- Next: 20:00 — 29,193.
- Lowest: **05:00 — 4,981 trips**.

**7. Plot**  
`Q6_trip_frequency_hour.png`

**8. Interpretation**  
Recorded ride frequency is highest at 19:00 and lowest at 05:00.

**9. Evidence-based Insight**  
The hourly demand pattern is concentrated in the evening, with 18:00–22:00 among the highest-volume hours.

**10. Limitations / Important Caveats**  
These are recorded trip frequencies, not necessarily all ride requests in the underlying market. Frequency does not measure fare level.

**11. Relevance to Task 2**  
Hour may be useful for modeling fare or demand-related behavior, but its role should be assessed separately for the target being modeled.

---

## Q7. Does Weather Condition influence average trip Distance?

**1. Official Question**  
Does Weather Condition influence average trip Distance?

**2. Appropriate Plot**  
Bar chart of mean trip distance by the actual Weather categories.

**3. Why This Plot Is Appropriate**  
Weather is categorical and distance is continuous. Grouped means directly compare average trip distance across the observed weather categories.

**4. Relevant Cleaning / Data Handling Note**  
Distance values invalidated during Phase 3 remain missing. Only nonmissing distance values contribute to each weather group's distance statistics.

**5. Exact Python Code**  
See Q7 section of `phase5_q1_q8_analysis.py`.

**6. Actual Result from full df_clean**

| Weather | Trips with distance | Mean distance (km) | Median distance (km) |
|---|---:|---:|---:|
| Stormy | 97,971 | 3.37447 | 2.15408 |
| Sunny | 98,416 | 3.36795 | 2.15759 |
| Rainy | 98,025 | 3.35707 | 2.16342 |
| Windy | 97,570 | 3.35381 | 2.15694 |
| Cloudy | 97,987 | 3.35128 | 2.13568 |

Mean-distance range = **0.02318 km**, about **23 metres**.

**7. Plot**  
`Q7_weather_distance.png`

**8. Interpretation**  
Mean distances are very close across all five weather categories. Stormy has the highest mean at 3.37447 km, while Cloudy has the lowest at 3.35128 km.

**9. Evidence-based Insight**  
The observed average trip-distance differences by weather are small relative to the overall trip-distance scale.

**10. Limitations / Important Caveats**  
This comparison does not establish that weather causes distance changes. It also does not control for time, location, traffic, or trip composition.

**11. Relevance to Task 2**  
Weather can remain a candidate feature, but the marginal distance differences observed here are small.

---

## Q8. Are rides that start closer to airports generally more expensive?

**1. Official Question**  
Are rides that start closer to airports generally more expensive?

**2. Appropriate Plot**  
Three separate fare-vs-distance scatterplots: one for JFK, one for EWR, and one for LGA.

**3. Why This Plot Is Appropriate**  
The question concerns three different airport-distance variables, so they must be analyzed separately. A scatterplot shows the direction and spread of the fare-distance relationship. Smaller airport distance means a ride starts closer to that airport.

**4. Relevant Cleaning / Data Handling Note**  
Phase 3 preserved valid airport-distance values and set clearly implausible reference distances above 1,000 km to missing. Q8 uses all available nonmissing values for each airport. For readability only, the plotted points are a random sample of up to 30,000 observations per airport; all reported statistics use the full available data.

**5. Exact Python Code**  
The following reproducible block is in the Q8 section of `phase5_q1_q8_analysis.py`. The Pearson correlation and closest/farthest 25% statistics are calculated from the full available `df_clean` data; only the plotted points are sampled for readability.
```python
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
```


**6. Actual Result from full df_clean**

| Airport | Available trips | Pearson r | Mean fare: closest 25% | Mean fare: farthest 25% | Median fare: closest 25% | Median fare: farthest 25% |
|---|---:|---:|---:|---:|---:|---:|
| JFK | 489,671 | **-0.2650** | 16.9791 | 10.2061 | 11.30 | 8.00 |
| EWR | 489,671 | **0.2760** | 9.7129 | 16.2028 | 8.00 | 9.50 |
| LGA | 489,671 | **0.0796** | 11.8618 | 14.0837 | 8.50 | 9.00 |

Here, “closest 25%” means the lowest quarter of the respective airport-distance values, and “farthest 25%” means the highest quarter.

**7. Plot**  
`Q8_airport_distance_fare.png`

**8. Interpretation**  
The three airports do not show the same pattern:
- **JFK:** negative correlation; smaller JFK distance is associated with higher fares in this pairwise EDA comparison.
- **EWR:** positive correlation; smaller EWR distance is associated with lower fares in this pairwise comparison.
- **LGA:** weak positive correlation; the relationship is much smaller than for JFK or EWR.

**9. Evidence-based Insight**  
The evidence does not support one uniform “closer to an airport = more expensive” pattern across JFK, EWR, and LGA. The direction differs by airport.

**10. Limitations / Important Caveats**  
These are observational pairwise relationships. Airport distance can be correlated with trip location and trip type, and the analysis does not control for distance, time, traffic, or other factors. Correlation is not causation.

**11. Relevance to Task 2**  
The three airport-distance variables may carry different information and should not be collapsed into one assumption. Their separate usefulness can be evaluated later in feature engineering/modeling.

---

## Phase 5 Scope Check

Completed Q1–Q8 only. No additional questions, deeper Phase 7 analysis, modeling, or Phase 6 work was started.
