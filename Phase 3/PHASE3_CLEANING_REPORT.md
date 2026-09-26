# Phase 3 — Cleaning Report

## Dataset versions
- `df_raw`: original 500,000-row source dataset, preserved and never overwritten.
- `df_clean`: 499,965-row cleaned analysis dataset.

## Summary
- Rows removed: 35
- Rows retained: 499,965
- Percentage removed: 0.0070%
- Columns: 26 before and 26 after
- Columns renamed: none
- Columns added: none
- Columns removed: none
- `Week` created: no
- Categories recoded: no
- Key-based deduplication: no
- Radians/degrees conversion: no

## Major decisions

### 1. Non-positive fares — Remove
35 rows (21 negative, 14 zero) were removed. These are invalid target values for a paid-fare field, and no source field identifies them as legitimate free rides or reversals.

### 2. Passenger count = 0 — Transform to missing
1,796 rows were retained, but `passenger_count` was changed from 0 to NaN. The trip remains useful for fare, distance, and temporal analysis, while the impossible passenger count is not silently treated as valid.

### 3. Geographic zero/invalid values — Transform to missing
Zero coordinates were treated as sentinel missing values. Nonzero coordinate values outside the mathematical radian domain were also invalidated. For affected rows, dependent geographic/distance fields were set to NaN rather than deleting the entire trip.

### 4. Extreme distance > 1,000 km — Transform to missing
Only clearly implausible NYC-scale distances were invalidated. Ordinary IQR outliers were retained. Rows were not deleted.

### 5. Extreme airport/reference distances > 1,000 km — Transform to missing
Implausible derived reference-distance cells were invalidated. The associated trip rows were retained.

### 6. Duplicate Key — Keep / Flag
No rows were removed by Key. `Key` equals `pickup_datetime`, and duplicate Keys are not exact duplicate rows. Therefore Key is not a defensible unique-trip deduplication key for the actual dataset.

### 7. Bearing — Keep / Document
Bearing was left in its observed `[-pi, pi]` representation. No radians-to-degrees conversion was performed.

### 8. Day / weekday — Keep / Document
`day` remains day-of-month (1–31), while `weekday` remains weekday (0–6). Both agree with `pickup_datetime`.

### 9. Schema/category differences — Keep / Document
Actual categories and columns were preserved. No silent mapping to the PDF schema was applied, and no `Week` column was invented.

## Validation
- Exact duplicate rows in `df_clean`: 0
- `fare_amount <= 0`: 0
- `passenger_count == 0`: 0
- `distance > 1,000`: 0
- reference distance > 1,000: 0
- missing `pickup_datetime`: 0
- Hour/Day/Month/weekday/year mismatches vs datetime: 0
- nonmissing Bearing outside `[-pi, pi]`: 0

## Missing values in df_clean
- `pickup_longitude`: 9,439
- `pickup_latitude`: 9,413
- `dropoff_longitude`: 9,464
- `dropoff_latitude`: 9,434
- `passenger_count`: 1,796
- `jfk_dist`: 10,294
- `ewr_dist`: 10,294
- `lga_dist`: 10,294
- `sol_dist`: 10,294
- `nyc_dist`: 10,294
- `distance`: 9,996
- `bearing`: 9,912

These are intentional missing values created by documented Phase 3 transformations; they were not blindly imputed.

## Explicitly deferred
- Q1–Q8
- Baseline EDA
- final insights
- further outlier analysis beyond the Phase 3 cleaning thresholds
- category encoding
- feature engineering
- modeling
