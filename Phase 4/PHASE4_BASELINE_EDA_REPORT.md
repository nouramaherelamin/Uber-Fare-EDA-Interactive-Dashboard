# Phase 4 — Baseline EDA

## Scope

This phase uses the approved Phase 3 `df_clean` dataset only. The dataset is **not modified** during EDA.

- Rows: **499,965**
- Columns: **26**
- Exact duplicate rows: **0**
- Target: **`fare_amount`**
- Datetime range: **2009-01-01 00:31:32 → 2015-06-30 23:38:21**
- Phase 4 does **not** answer Q1–Q8 and does not perform the deeper Phase 7 correlation/multicollinearity analysis.

## 1. Baseline overview

The cleaned dataset contains **499,965 trips and 26 columns**. The target `fare_amount` has no missing values. Some explanatory variables intentionally retain missing values from Phase 3:

- `distance`: **9,996 (1.999%)**
- `passenger_count`: **1,796 (0.359%)**
- each airport/reference distance: **10,294 (2.059%)**
- `bearing`: **9,912 (1.983%)**

These missing values are intentional outputs of Phase 3 cleaning and are not imputed in Phase 4.

---

# 2. Target analysis — `fare_amount`

### Target statistics

| Statistic | Value |
|---|---:|
| Count | 499,965 |
| Mean | 11.36 |
| Median | 8.50 |
| Std. dev. | 9.92 |
| Minimum | 0.01 |
| Q1 | 6.00 |
| Q3 | 12.50 |
| P90 | 20.50 |
| P95 | 30.50 |
| P99 | 52.14 |
| Maximum | 500.00 |
| Skewness | 4.90 |

The mean is above the median, and the positive skewness indicates a **right-skewed fare distribution**. The maximum of **500.00** is far above the upper quartile, so the boxplot should be read as an outlier-sensitive view rather than evidence that every high fare is erroneous.

## Plot 1 — Fare distribution

1. **Question / purpose:** What does the overall distribution of the target look like?
2. **Plot choice:** Histogram.
3. **Why this plot:** A histogram shows where fares are concentrated and whether the target is symmetric, skewed, or long-tailed.
4. **Code:**
```python
plt.figure(figsize=(9, 5))
d = df_clean["fare_amount"]
p99 = d.quantile(.99)
plt.hist(d[d <= p99], bins=80)
plt.xlabel("Fare amount")
plt.ylabel("Number of trips")
plt.title("Distribution of Fare Amount")
plt.show()
```
5. **Actual result:** The displayed range covers fares through the **99th percentile (~52.14)** so the common distribution is readable. The full dataset still has median **8.50**, mean **11.36**, and maximum **500.00**.
6. **Interpretation:** Most observations are relatively low fares, while a smaller number of high-fare observations extend the upper tail.
7. **Initial insight:** `fare_amount` is not normally distributed in this baseline view; the right tail is an important feature of the target.
8. **Possible relevance to Task 2:** Later target modeling or fare-focused analysis may need to account for this skewed target distribution rather than assuming a symmetric target.

**Plot:** `02_fare_histogram.png`

## Plot 2 — Fare boxplot

1. **Question / purpose:** How concentrated is the target and how much does the upper tail extend beyond the central distribution?
2. **Plot choice:** Horizontal boxplot.
3. **Why this plot:** A boxplot compactly shows the median, quartiles, and observations beyond the IQR-based whiskers.
4. **Code:**
```python
plt.figure(figsize=(9, 3.8))
plt.boxplot(df_clean["fare_amount"], vert=False)
plt.xlabel("Fare amount")
plt.title("Fare Amount Boxplot")
plt.show()
```
5. **Actual result:** Q1 = **6.00**, median = **8.50**, Q3 = **12.50**; the distribution has many high-side observations beyond the central IQR range.
6. **Interpretation:** The upper tail is substantial. Phase 3 deliberately did not delete statistical fare outliers automatically.
7. **Initial insight:** High fares are present as a real part of the cleaned target distribution and need to remain visible in baseline EDA.
8. **Possible relevance to Task 2:** Any later fare prediction/evaluation should be aware that a small number of high values can affect averages and error metrics.

**Plot:** `03_fare_boxplot.png`

---

# 3. Important numerical variables

## Plot 3 — Distance distribution

1. **Question / purpose:** What is the distribution of recorded trip distance after Phase 3 cleaning?
2. **Plot choice:** Histogram.
3. **Why this plot:** Distance is continuous and strongly right-skewed, so a histogram makes the common trip-distance range easier to see.
4. **Code:**
```python
distance = df_clean["distance"].dropna()
p99 = distance.quantile(.99)

plt.figure(figsize=(9, 5))
plt.hist(distance[distance <= p99], bins=70)
plt.xlabel("Distance (km)")
plt.ylabel("Number of trips")
plt.title("Distance Distribution (shown through 99th percentile)")
plt.show()
```
5. **Actual result:** Among non-missing distances, median = **2.15 km**, Q1 = **1.26 km**, Q3 = **3.93 km**, P99 = **20.35 km**, maximum = **709.85 km**. The mean is **3.36 km**, with skewness **30.50**.
6. **Interpretation:** Most trips are short, with a long upper tail. The displayed chart stops at P99 only to keep the common range readable; no rows are removed from the dataset.
7. **Initial insight:** Distance is highly unevenly distributed and contains a relatively small set of much longer trips.
8. **Possible relevance to Task 2:** Distance is a natural variable to examine alongside fare, while keeping the long tail visible in later modeling/analysis.

**Plot:** `04_distance_histogram.png`

## Plot 4 — Passenger count

1. **Question / purpose:** How are trips distributed by passenger count?
2. **Plot choice:** Bar chart of counts.
3. **Why this plot:** Passenger count is a discrete variable, so a bar chart directly compares the number of observations in each category.
4. **Code:**
```python
counts = df_clean["passenger_count"].value_counts(dropna=False).sort_index()
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Passenger count")
plt.ylabel("Number of trips")
plt.title("Passenger Count Distribution")
plt.show()
```
5. **Actual result:** **1 passenger = 345,983 trips (69.20%)**; 2 = 73,904; 3 = 21,759; 4 = 10,613; 5 = 35,321; 6 = 10,589; missing = **1,796**.
6. **Interpretation:** One-passenger trips are the dominant category. Missing passenger counts remain visible rather than being imputed.
7. **Initial insight:** Passenger count is heavily concentrated at 1 and becomes progressively less common for most other values, with an additional smaller group at 5.
8. **Possible relevance to Task 2:** Passenger count can be considered as a candidate explanatory variable, but its highly concentrated distribution should be kept in mind.

**Plot:** `05_passenger_count.png`

---

# 4. Categorical variables

## Plot 5 — Car Condition

1. **Question / purpose:** How balanced are the observed car-condition categories?
2. **Plot choice:** Count bar chart.
3. **Why this plot:** Car condition is categorical; counts show whether one category dominates the dataset.
4. **Code:**
```python
counts = df_clean["Car Condition"].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel("Car Condition")
plt.ylabel("Number of trips")
plt.title("Car Condition Distribution")
plt.xticks(rotation=20, ha="right")
plt.show()
```
5. **Actual result:** Very Good = **125,306**, Bad = **124,965**, Good = **124,959**, Excellent = **124,735**. Each is close to 25%.
6. **Interpretation:** The four categories are unusually balanced rather than dominated by one condition.
7. **Initial insight:** Category imbalance is not a major baseline issue for this variable.
8. **Possible relevance to Task 2:** Later comparisons can examine whether fare behavior differs by car condition without starting from a severely imbalanced category distribution.

**Plot:** `06_car_condition.png`

## Plot 6 — Weather

1. **Question / purpose:** How are weather categories represented?
2. **Plot choice:** Count bar chart.
3. **Why this plot:** Weather is categorical, so category counts are the clearest baseline view.
4. **Code:**
```python
counts = df_clean["Weather"].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel("Weather")
plt.ylabel("Number of trips")
plt.title("Weather Distribution")
plt.xticks(rotation=20, ha="right")
plt.show()
```
5. **Actual result:** sunny = **100,424**, cloudy = **100,056**, rainy = **99,967**, stormy = **99,947**, windy = **99,571**; each is close to 20%.
6. **Interpretation:** Weather categories are also very evenly represented.
7. **Initial insight:** No single weather category dominates the baseline dataset.
8. **Possible relevance to Task 2:** This gives sufficient representation for later fare comparisons across weather categories.

**Plot:** `07_weather.png`

## Plot 7 — Traffic Condition

1. **Question / purpose:** How balanced are the traffic categories?
2. **Plot choice:** Count bar chart.
3. **Why this plot:** Traffic condition is categorical and the count comparison is the simplest way to inspect representation.
4. **Code:**
```python
counts = df_clean["Traffic Condition"].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel("Traffic Condition")
plt.ylabel("Number of trips")
plt.title("Traffic Condition Distribution")
plt.xticks(rotation=20, ha="right")
plt.show()
```
5. **Actual result:** Congested Traffic = **166,838**, Dense Traffic = **166,574**, Flow Traffic = **166,553**; each is about one third.
6. **Interpretation:** The three traffic categories are almost equally represented.
7. **Initial insight:** Traffic category counts are balanced enough for later comparative analysis.
8. **Possible relevance to Task 2:** Traffic condition can be examined later as a categorical explanatory factor without a severe baseline representation problem.

**Plot:** `08_traffic_condition.png`

---

# 5. Temporal distributions

## Plot 8 — Trips by hour

1. **Question / purpose:** At which hours are trips more or less concentrated?
2. **Plot choice:** Line chart.
3. **Why this plot:** Hour is ordered and cyclical; a line shows the shape of the within-day distribution clearly.
4. **Code:**
```python
hour_counts = df_clean["hour"].value_counts().sort_index()
plt.plot(hour_counts.index, hour_counts.values, marker="o")
plt.xlabel("Hour of day")
plt.ylabel("Number of trips")
plt.title("Trips by Hour of Day")
plt.xticks(range(24))
plt.show()
```
5. **Actual result:** The highest observed hour is **19:00 with 31,380 trips**. The lowest is **05:00 with 4,981 trips**.
6. **Interpretation:** Trip counts vary materially across the day rather than remaining flat.
7. **Initial insight:** Time-of-day is an important dimension of the dataset's trip distribution.
8. **Possible relevance to Task 2:** Hour can provide temporal context when examining fare behavior; this plot itself is a trip-count distribution, not a fare-demand conclusion.

**Plot:** `09_trips_by_hour.png`

## Plot 9 — Trips by weekday

1. **Question / purpose:** How are trips distributed across the seven weekdays?
2. **Plot choice:** Bar chart.
3. **Why this plot:** Weekday is a small discrete ordered category, so bars make differences easy to compare.
4. **Code:**
```python
weekday_counts = df_clean["weekday"].value_counts().sort_index()
plt.bar(range(7), weekday_counts.values)
plt.xticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.xlabel("Weekday")
plt.ylabel("Number of trips")
plt.title("Trips by Weekday")
plt.show()
```
5. **Actual result:** Counts range from **64,233** to **77,217** trips. The highest weekday is **Thursday (74,776)** and the lowest is **Monday (64,233)**.
6. **Interpretation:** The weekday distribution is not perfectly uniform, but differences are much smaller than the hour-to-hour variation.
7. **Initial insight:** Weekday provides some temporal variation in trip counts without one category overwhelming the dataset.
8. **Possible relevance to Task 2:** Weekday can be retained as a candidate temporal factor for later fare-focused comparisons.

**Plot:** `10_trips_by_weekday.png`

## Plot 10 — Trips by month

1. **Question / purpose:** How does the number of recorded trips vary across calendar months?
2. **Plot choice:** Line chart.
3. **Why this plot:** Month is ordered, so a line highlights the broad seasonal pattern.
4. **Code:**
```python
month_counts = df_clean["month"].value_counts().sort_index()
plt.plot(month_counts.index, month_counts.values, marker="o")
plt.xlabel("Month")
plt.ylabel("Number of trips")
plt.title("Trips by Month")
plt.xticks(range(1, 13))
plt.show()
```
5. **Actual result:** The highest month is **May (46,729 trips)** and the lowest is **August (35,867 trips)**.
6. **Interpretation:** Monthly trip counts vary across the year.
7. **Initial insight:** There is visible month-to-month variation, but this plot describes recorded trip volume only; it does not by itself establish seasonal fare effects.
8. **Possible relevance to Task 2:** Month may provide useful temporal context for later analysis of fare patterns.

**Plot:** `11_trips_by_month.png`

**Important temporal note:** 2015 is only represented through **June 30**, so raw yearly totals should not be interpreted as full-year comparisons.

**Plot:** `11_trips_by_month.png`

---

# 6. Baseline numeric relationships

## Plot 11 — Quick correlation heatmap

1. **Question / purpose:** What broad linear relationships are visible among the main numeric variables before deeper analysis?
2. **Plot choice:** Pearson correlation heatmap.
3. **Why this plot:** A heatmap allows a compact baseline scan of many pairwise linear associations.
4. **Code:**
```python
corr_cols = [
    "fare_amount", "distance", "passenger_count",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist", "bearing"
]
corr = df_clean[corr_cols].corr()
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.show()
```
5. **Actual result:** `fare_amount` vs `distance` has Pearson **r = 0.671**. `fare_amount` vs `passenger_count` is about **0.013**. `bearing` has a near-zero linear association with fare (**-0.024**). Some reference-distance variables are strongly correlated with one another, including `sol_dist` vs `nyc_dist` (**~0.992**).
6. **Interpretation:** Distance shows a noticeable positive linear association with fare in this baseline scan. Passenger count and bearing show little linear association with fare. The reference-distance features are highly redundant with one another.
7. **Initial insight:** Distance is a prominent numerical variable for later fare analysis, while the reference-distance variables may contain substantial redundancy.
8. **Possible relevance to Task 2:** These baseline relationships can guide which variables deserve deeper investigation later, without treating correlation as causation or as the final model-selection decision.

**Plot:** `12_baseline_correlation_heatmap.png`

This is intentionally a **quick baseline** correlation view, not the deeper comparative/multicollinearity analysis reserved for the later phase.

## Plot 12 — Fare vs distance

1. **Question / purpose:** Does the point-level data visually support the baseline correlation between distance and fare?
2. **Plot choice:** Scatterplot.
3. **Why this plot:** A scatterplot directly displays the joint relationship between two continuous variables and reveals spread, nonlinearity, and extreme observations.
4. **Code:**
```python
plot_df = df_clean[["fare_amount", "distance"]].dropna()
plot_df = plot_df.sample(min(30000, len(plot_df)), random_state=42)

plt.scatter(
    plot_df["distance"],
    plot_df["fare_amount"],
    alpha=0.25,
    s=8
)
plt.xlabel("Distance (km)")
plt.ylabel("Fare amount")
plt.title("Fare Amount vs Distance (30,000-point plotting sample)")
plt.show()
```
5. **Actual result:** The full-data Pearson correlation is **r = 0.671**, while the plot uses a 30,000-row random sample only to keep the visualization readable.
6. **Interpretation:** The scatter shows an overall upward relationship, but the points are not located on a single line; fare variability exists at similar distances.
7. **Initial insight:** Distance is related to fare, but distance alone does not explain all fare variation visible in the data.
8. **Possible relevance to Task 2:** This supports examining distance as an important candidate explanatory variable while leaving room for other variables and nonlinear effects to be investigated later.

**Plot:** `13_fare_vs_distance_scatter.png`

---

# 7. Baseline EDA conclusions — limited to what was actually analyzed

1. The cleaned target `fare_amount` is **right-skewed** with median **8.50** and mean **11.36**.
2. Distance is also **strongly right-skewed**, with median about **2.15 km** and a long upper tail.
3. Passenger count is dominated by **1-passenger trips**; 1,796 passenger counts remain missing by design from Phase 3.
4. Car condition, weather, and traffic categories are all **roughly balanced** in this dataset.
5. Trip volume varies substantially by **hour**, with the highest observed hour at **19:00** and the lowest at **05:00**.
6. The quick numeric scan shows a **positive Pearson correlation between fare and distance (r ≈ 0.671)**, but this is an association, not evidence of causation.
7. Several reference-distance variables are highly correlated with one another, which is a point to revisit in later multicollinearity analysis.

These are **baseline EDA observations only**. They are not Q1–Q8 answers, not causal conclusions, and not final business insights.

---

## Phase 4 artifacts

- `df_clean.csv` was **not modified**.
- `phase4_baseline_eda.py` — reproducible EDA code.
- `phase4_overview.csv`
- `phase4_numeric_summary.csv`
- `phase4_baseline_correlations.csv`
- distribution CSVs for the major categorical/temporal variables.
- PNG plots `01`–`13`.

**Phase 5 has not started.**
