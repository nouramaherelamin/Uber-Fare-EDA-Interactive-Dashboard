# Uber Fare — Interactive Exploratory Data Analysis

A professional **Exploratory Data Analysis (EDA)** project for Uber trip and fare data, developed as part of the **Cellula Technologies — Machine Learning Internship — Task 1**.

The project combines a documented Jupyter Notebook analysis with a separate **Python + Streamlit interactive EDA application**. It investigates fare behavior, trip characteristics, demand patterns, data quality, distance relationships, weather, traffic, and airport-distance variables using the actual Task 1 dataset.

> **Scope:** This is an EDA project. It does **not** train, evaluate, or deploy a predictive machine-learning model.

---

## 1. Project Overview

The purpose of this project is to understand the structure, quality, distributions, and observed relationships within an Uber fare dataset before moving to later machine-learning work.

The analysis covers:

- Dataset structure and data-quality investigation
- Fare distributions and fare behavior
- Passenger count vs. fare
- Car condition vs. fare
- Traffic condition vs. fare
- Fare patterns by hour
- Ride-request frequency by hour
- Trip distance vs. fare
- Weather vs. trip distance
- Airport distance vs. fare
- Numerical correlation analysis
- Robustness checks for extreme distance and geographic-quality observations
- Interactive exploration through a Streamlit application

The EDA is intended to provide evidence-based observations and identify variables and data-quality considerations that can be evaluated later in a modeling stage. Correlations and grouped comparisons are treated as **observational associations**, not causal effects.

### What this project does not do

This Task 1 project does not include:

- Model training
- Predictions
- Hyperparameter tuning
- SHAP/model-derived feature importance
- Production inference
- Causal analysis

---

## 2. Objectives

The project objectives are to:

1. Inspect the dataset structure and variable types.
2. Investigate missing values, duplicate rows, duplicate keys, and anomalous records.
3. Examine fare behavior and its distribution.
4. Investigate the relationship between passenger count and fare.
5. Compare fares across car-condition categories.
6. Compare fares across traffic-condition categories.
7. Examine how observed fare levels vary by pickup hour.
8. Identify the hour with the highest recorded trip frequency.
9. Investigate the relationship between trip distance and fare.
10. Compare trip distance across weather categories.
11. Examine airport-distance variables in relation to fare.
12. Explore Pearson correlations among numerical variables.
13. Separate raw distance analysis from explicitly labeled robustness views.
14. Preserve analytical decisions and document their limitations.
15. Provide an interactive interface for exploring the verified EDA dataset.

---

## 3. Dataset

### Dataset used

**Uber Fare Dataset — Task 1 sample**

The original notebook loads a **50,000-row sample** and contains **26 columns**. After excluding the 9 non-positive fare records for fare-based EDA, the exported analytical dataset used by the Streamlit application contains:

- **49,991 rows**
- **26 columns**
- **0 missing cells in the exported EDA CSV**
- **0 fully duplicated rows**

The Streamlit application loads the exported file:

```text
uber_fare_eda_clean.csv
```

and treats it as read-only.

### Dataset columns

| Column | Type | Analytical role |
|---|---|---|
| `User ID` | object | User identifier |
| `User Name` | object | User name |
| `Driver Name` | object | Driver name |
| `Car Condition` | object | Vehicle-condition category |
| `Weather` | object | Weather category |
| `Traffic Condition` | object | Traffic category |
| `key` | object | Timestamp-like trip key investigated during data-quality analysis |
| `fare_amount` | float64 | Fare amount / EDA target variable |
| `pickup_datetime` | object | Pickup timestamp |
| `pickup_longitude` | float64 | Pickup longitude |
| `pickup_latitude` | float64 | Pickup latitude |
| `dropoff_longitude` | float64 | Drop-off longitude |
| `dropoff_latitude` | float64 | Drop-off latitude |
| `passenger_count` | int64 | Number of passengers recorded |
| `hour` | int64 | Pickup hour, 0–23 |
| `day` | int64 | Day of month |
| `month` | int64 | Month, 1–12 |
| `weekday` | int64 | Weekday encoding |
| `year` | int64 | Year |
| `jfk_dist` | float64 | Distance/reference variable for JFK |
| `ewr_dist` | float64 | Distance/reference variable for EWR |
| `lga_dist` | float64 | Distance/reference variable for LGA |
| `sol_dist` | float64 | Distance/reference variable for the Statue of Liberty |
| `nyc_dist` | float64 | NYC reference distance |
| `distance` | float64 | Trip distance |
| `bearing` | float64 | Trip direction/bearing |

### Categorical values observed

**Car Condition:** Bad, Excellent, Good, Very Good

**Weather:** cloudy, rainy, stormy, sunny, windy

**Traffic Condition:** Congested Traffic, Dense Traffic, Flow Traffic

**Passenger count:** 0–6

**Years:** 2009–2015

**Hours:** 0–23

---

## 4. Data Cleaning & Quality Checks

The notebook deliberately separates the original loaded dataframe from the analytical dataframe used for fare-based EDA.

### Non-positive fares

The original 50,000-row sample contains **9 non-positive fare records**. These were excluded from fare-based EDA because zero or negative values are not valid fare amounts for fare analysis.

```text
Original sample:       50,000 rows
Positive-fare EDA:     49,991 rows
Excluded non-positive: 9 rows
```

The original dataframe was preserved during the investigation rather than being silently overwritten.

### Zero passenger records

The dataset contains **165 records with `passenger_count = 0`**. These were investigated and retained as documented data-quality anomalies because they can still contain fare and trip-distance information.

### Zero-distance records

The exported dataset contains **1,449 zero-distance records**. The notebook investigates these records by comparing pickup and drop-off coordinates. They were not automatically removed from the EDA dataset.

### Extreme distance observations

The distance distribution contains extreme observations. The notebook uses these observations to motivate a separate robustness analysis rather than silently deleting them from the main EDA dataset.

The Streamlit application provides:

1. **Raw Data**
2. **Robust View — Distance ≤ 50 km**
3. **Robust View — Valid Geo + Distance ≤ 50 km**

The 50 km threshold is an **analytical robustness filter**, not a universal definition of a valid trip.

### Geographic-quality observations

The notebook investigates zero/sentinel geographic coordinates and uses a separate valid-geographic subset for robustness analysis. The Streamlit app preserves this distinction rather than overwriting the source CSV.

### Duplicate rows

The exported EDA CSV contains:

```text
Full duplicate rows: 0
```

### Duplicate `key` values

The `key` column was investigated because it was described as a unique trip identifier. The notebook documents that duplicated key values exist, the duplicate keys are not simply identical full rows, and the `key` values match `pickup_datetime` across the dataset. Therefore, deleting rows solely because `key` is duplicated would not be a defensible deduplication decision.

Duplicate keys were documented and retained rather than arbitrarily removed.

### Analytical principle

The project distinguishes between:

- records removed for a documented analytical reason;
- anomalies investigated but retained;
- robustness subsets created only for comparison;
- the original/exported analytical dataset.

No robustness filter silently replaces the main dataset.

---

# 5. Exploratory Data Analysis

The Task 1 notebook completes eight required analytical questions. The Streamlit application turns the core analyses into interactive views.

## Q1. Is higher passenger count associated with higher fare?

**Question:** Does passenger count show an observed relationship with fare amount?

**Visualization:** Scatter plot of `passenger_count` vs. `fare_amount`.

**Finding:** Pearson correlation in the exported EDA dataset is approximately **0.0165**, indicating a very weak linear association. The scatter forms vertical bands because passenger count is discrete, while fare values remain widely dispersed.

**Interpretation:** The observed trip-level data does not show a clear linear relationship between passenger count and fare. This does not establish that passenger count is irrelevant to prediction; it only describes the marginal linear association observed in this EDA.

## Q2. Does car condition differ in observed fare levels?

**Visualization:** Boxplot of fare amount by `Car Condition`.

| Car Condition | Trips | Mean Fare | Median Fare |
|---|---:|---:|---:|
| Bad | 12,532 | $11.49 | $8.50 |
| Very Good | 12,484 | $11.44 | $8.50 |
| Good | 12,540 | $11.31 | $8.50 |
| Excellent | 12,435 | $11.22 | $8.50 |

**Interpretation:** Average fares are close across the four categories and medians are identical at $8.50. The observed marginal differences are small. The comparison does not control for distance, time, traffic, weather, or other variables, so it does not establish causality.

## Q3. At what hour is the observed average fare highest?

**Visualization:** Interactive line chart of average fare by hour, with the highest observed average highlighted.

**Finding:** The highest observed average fare occurs at **05:00**, at approximately **$15.37**. There are approximately **507 rides** at that hour in the EDA sample.

**Interpretation:** 05:00 has the highest observed mean fare in this sample. This is an observational result, not evidence that the hour itself causes higher fares. The relatively small number of rides at that hour is an important caveat.

## Q4. Does traffic condition differ in observed fare levels?

**Visualization:** Boxplot of fare amount by `Traffic Condition`.

| Traffic Condition | Trips | Mean Fare | Median Fare |
|---|---:|---:|---:|
| Congested Traffic | 16,641 | $11.47 | $8.50 |
| Dense Traffic | 16,656 | $11.32 | $8.50 |
| Flow Traffic | 16,694 | $11.30 | $8.50 |

**Interpretation:** The three traffic categories have similar mean fares, with Congested Traffic having the highest observed mean. The differences are small and medians are identical. The analysis does not establish that traffic condition causes fare changes.

## Q5. How is trip distance related to fare?

**Visualization:** Scatter plot of `distance` vs. `fare_amount` with three clearly labeled analysis views:

- Raw Data
- Robust View — Distance ≤ 50 km
- Robust View — Valid Geo + Distance ≤ 50 km

| Analysis | Pearson | Spearman |
|---|---:|---:|
| Raw distance vs. fare | 0.0165 | 0.8112 |
| Distance ≤ 50 km | 0.8325 | 0.8152 |
| Valid geo + Distance ≤ 50 km | 0.8490 | 0.8444 |

**Interpretation:** Extreme distance observations substantially affect the raw Pearson measure. Spearman correlation is already strong in the raw data, while Pearson becomes much stronger in the explicitly labeled robustness views. Distance is therefore an important EDA candidate variable for later modeling, but correlation is not model-based feature importance and does not establish causality.

## Q6. At what hour are ride requests most frequent?

**Visualization:** Bar chart of trip count by hour.

**Finding:** The highest recorded trip frequency is **19:00**, with **3,117 trips**. The next highest observed hours are 18:00 (3,077), 20:00 (2,859), 21:00 (2,816), and 22:00 (2,808).

**Interpretation:** Recorded ride frequency is concentrated in the evening, with 19:00 being the highest-volume hour in this EDA sample. This is a demand-frequency result, not a fare result.

## Q7. How does observed trip distance vary by weather?

**Visualization:** Boxplot plus mean/median summary table.

| Weather | Trips | Mean Distance | Median Distance |
|---|---:|---:|---:|
| cloudy | 9,903 | 22.96 km | 2.11 km |
| rainy | 10,009 | 19.21 km | 2.17 km |
| sunny | 10,108 | 18.84 km | 2.10 km |
| windy | 9,955 | 18.47 km | 2.12 km |
| stormy | 10,016 | 12.27 km | 2.11 km |

**Interpretation:** Raw means differ substantially, but medians are all close to roughly 2.1–2.2 km. Extreme distance observations therefore have a strong influence on the mean differences. These are observed associations, not evidence that weather causes trip-distance changes.

## Q8. How are airport-distance variables related to fare?

**Visualization:** Separate interactive scatter plots for JFK, EWR, and LGA, selectable through the dashboard's airport selector.

| Airport distance | Pearson correlation with fare |
|---|---:|
| JFK | 0.0034 |
| EWR | 0.0051 |
| LGA | 0.0044 |

**Interpretation:** The raw linear associations are all very close to zero in the exported EDA dataset. The notebook also documents a geographic-quality robustness check. These are pairwise observational relationships and do not control for trip distance, location, time, traffic, or other variables. They should not be interpreted as proof that airport proximity has no effect on fares.

---

# 6. Additional Insights / Questions

The notebook documents four **proposed follow-up questions** for deeper analysis. They are proposals rather than completed additional analyses:

1. Does average fare differ between weekdays and weekends?
2. Does traffic condition affect fare differently at different hours?
3. How does average trip distance change throughout the day?
4. Which day of the week has the highest average fare?

---

# 7. Key Findings

- Passenger count has a **very weak linear association** with fare (`r ≈ 0.0165`).
- Mean fares are close across car-condition categories.
- Mean fares are also close across traffic-condition categories.
- The highest observed average fare occurs at **05:00**, at approximately **$15.37**.
- The highest recorded trip frequency occurs at **19:00**, with **3,117 trips**.
- Raw Pearson distance-vs-fare correlation is distorted by extreme distance observations.
- Distance shows a much stronger linear relationship with fare when the explicitly labeled robustness views are examined.
- Weather-category mean distances differ substantially in the raw data, while median distances remain much closer.
- JFK, EWR, and LGA airport-distance variables show very small raw Pearson correlations with fare.
- The dataset contains documented anomalies including zero-passenger records, zero-distance records, extreme distances, and duplicate `key` values.
- The project keeps these limitations visible instead of silently deleting observations or turning associations into causal claims.

---

# 8. Interactive Streamlit Dashboard

The project includes a separate **Python Streamlit application** for interactive exploration of the exported EDA dataset.

Application title:

> **Uber Fare — Interactive EDA**

The application is implemented in:

```text
app.py
```

and dynamically loads:

```text
uber_fare_eda_clean.csv
```

from the same directory.

### Interactive features

- Global filters for Hour, Car Condition, Weather, Traffic Condition, Passenger Count, Year, and Month.
- Reset All Filters through Streamlit session state.
- Dynamic KPI cards.
- Interactive Plotly charts with hover, zoom, pan, and reset behavior.
- Deterministic scatter sampling for rendering when appropriate while KPI/correlation statistics use all filtered records.
- Raw and robustness distance-analysis views.
- JFK/EWR/LGA airport selector.
- Dynamic Pearson correlation explorer and numerical heatmap.
- Data Explorer with analytical column selection, search, and filtered CSV download.
- Data Quality diagnostics.
- Empty-state handling when filters produce no records.
- Professional dark UI with yellow accents and a taxi visual in the Hero section.
- Responsive CSS and reduced-motion handling.

### Data Explorer

The Data Explorer provides:

- Filtered row count
- Analytical column selection
- Text search
- Interactive dataframe inspection
- CSV download generated from the current filtered analytical records

Direct identifying fields are not included in the default analytical table.

### Performance

The app uses `@st.cache_data` for dataset loading and selected calculations. Large scatter plots can use a deterministic rendering sample to keep the interface responsive, while reported KPIs and correlations continue to use all filtered records.

---

# 9. Dashboard Sections

| Section | Purpose |
|---|---|
| **Overview** | Dynamic KPIs, current filter state, demand by hour, and fare distribution |
| **Fare Analysis** | Fare distribution, average fare by hour, car-condition comparison, and traffic-condition comparison |
| **Demand Analysis** | Trip frequency by hour and selected-hour inspection |
| **Passenger Analysis** | Passenger count vs. fare scatter plot and Pearson correlation |
| **Car & Traffic** | Fare comparisons across car condition and traffic condition |
| **Distance Analysis** | Raw and robustness-filtered distance-vs-fare analysis |
| **Weather Analysis** | Weather vs. trip-distance distributions and summary statistics |
| **Airport Analysis** | JFK/EWR/LGA distance-vs-fare exploration |
| **Correlation Explorer** | Numerical Pearson correlations and heatmap exploration |
| **Data Explorer** | Searchable analytical records and filtered CSV download |
| **Data Quality** | Dataset-quality diagnostics and documented Task 1 findings |

---

# 10. Technologies & Libraries

## Streamlit application

The application's `requirements.txt` contains:

```text
streamlit>=1.36,<2.0
pandas>=2.1,<3.0
numpy>=1.26,<3.0
plotly>=5.20,<7.0
```

The application uses:

- **Python** — application language
- **Streamlit** — interactive application framework
- **Pandas** — dataframe loading, filtering, grouping, and analysis
- **NumPy** — numerical operations
- **Plotly** — interactive visualizations

The application also uses Python standard-library modules including `pathlib`, `io`, and `typing`.

## Notebook analysis stack

The Task 1 notebook additionally uses:

- **Matplotlib**
- **Seaborn**
- **IPython display utilities**

These are notebook dependencies; they are not included in the Streamlit application's `requirements.txt` because the app itself does not import them.

---

# 11. Project Structure

### Streamlit application package

```text
uber_fare_streamlit_polished/
├── app.py
├── requirements.txt
├── README.md
├── README_Interactive_EDA.md
├── uber_fare_eda_clean.csv
└── assets/
    └── taxi.png
```

### Wider Task 1 project workspace

The inspected project workspace also contains the original analysis notebook and supporting phase documentation:

```text
Task1_EDA_Uber_Fare_FINAL.ipynb
phase3/
└── PHASE3_CLEANING_REPORT.md
phase4/
└── PHASE4_BASELINE_EDA_REPORT.md
phase5/
└── PHASE5_Q1_Q8_REPORT.md
phase6/
└── PHASE6_EXECUTION_REPORT.md
phase7/
└── PHASE7_EXECUTION_REPORT.md
```

The separate presentation artifact is kept independent from the Streamlit application.

### Important separation

The Streamlit application is an independent EDA artifact. It does not modify the Task 1 notebook, exported source CSV, or separate presentation during normal use.

---

# 12. How to Run the Streamlit Application

## 1. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start the application

```bash
streamlit run app.py
```

The application expects:

```text
uber_fare_eda_clean.csv
```

to be located beside `app.py`.

---

# 13. Analytical Methodology

The overall workflow is:

```text
Raw Task 1 Sample
        │
        ▼
Dataset Inspection
        │
        ▼
Data Quality Investigation
        │
        ├── Missing values
        ├── Duplicate rows
        ├── Duplicate keys
        ├── Fare anomalies
        ├── Passenger anomalies
        ├── Distance anomalies
        └── Geographic anomalies
        │
        ▼
Positive-Fare EDA Dataset
        │
        ▼
Q1–Q8 Exploratory Analysis
        │
        ├── Passenger ↔ Fare
        ├── Car Condition ↔ Fare
        ├── Hour ↔ Fare
        ├── Traffic ↔ Fare
        ├── Distance ↔ Fare
        ├── Hour ↔ Demand
        ├── Weather ↔ Distance
        └── Airport Distance ↔ Fare
        │
        ▼
Robustness Analysis
        │
        ▼
Interactive Streamlit Exploration
        │
        ▼
Candidate insights for future modeling
```

---

# 14. Important Analytical Limitations

This project is exploratory and observational.

Therefore:

- Correlation does not imply causation.
- Grouped mean differences do not establish causal effects.
- Pearson correlation measures linear association.
- Extreme observations can materially affect means and Pearson correlations.
- The 50 km distance view is an analytical robustness filter, not a universal validity rule.
- Airport-distance relationships are not adjusted for other trip characteristics.
- Demand counts describe the recorded dataset, not necessarily the complete ride-request market.
- EDA association strength is not equivalent to machine-learning feature importance.
- Later modeling is required to evaluate predictive performance.

---

# 15. Reproducibility & Data Integrity

The Streamlit application uses a relative dataset path:

```python
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uber_fare_eda_clean.csv"
```

The dataset is loaded with Pandas and cached with Streamlit using `@st.cache_data`.

The application does not rewrite the CSV during normal use.

The inspected exported dataset has the SHA-256 checksum:

```text
4a07558eefd7496b4d783f493366ce3cd8b93986fff5a85938f4872863005a40
```

This checksum can be used to verify that a local copy matches the inspected project artifact.

---

# 16. Quality Assurance

The application source was checked for:

- Python syntax
- Dataset/schema validation
- Dataset loading
- Session-state filter handling
- Navigation
- Empty filtered-data handling
- Correlation calculations
- Distance robustness views
- Airport selection
- Data Explorer
- CSV download generation
- Data-quality checks
- Duplicate widget-key risks
- Absence of model-training code

The intended local startup command is:

```bash
python -m streamlit run app.py
```

A full browser/runtime verification should be performed in the target environment after installing the dependencies.

---

# 17. Portfolio Value

This project demonstrates practical skills in:

- Exploratory Data Analysis
- Data-quality investigation
- Statistical reasoning
- Pandas
- Numerical analysis
- Data visualization
- Plotly
- Streamlit
- Interactive dashboard development
- Analytical documentation
- Reproducible data workflows
- Responsible interpretation of observational data
- Preparing an EDA workflow for subsequent machine-learning work

The project emphasizes not only visualization, but also **why analytical decisions were made and what their limitations are**.

---

# 18. Author

**Noura Maher Elamin**

- LinkedIn: https://www.linkedin.com/in/nouramaherelamin/
- GitHub: https://github.com/nouramaherelamin

---

# 19. Internship Context

**Cellula Technologies — Machine Learning Internship**  
**Task 1 — Exploratory Data Analysis**

This project contains the Task 1 EDA work and its independent interactive Streamlit exploration layer.

---

## Copyright

© 2026 Noura Maher Elamin. All rights reserved.

This project was created for educational, internship, portfolio, and analytical demonstration purposes.
