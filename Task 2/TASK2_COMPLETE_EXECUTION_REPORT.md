# TASK 2 — COMPLETE EXECUTION & VERIFICATION RECORD

## Final status
**COMPLETED + VERIFIED. Modeling executed after approved split validation. Presentation not created.**

## 1. Exact split definition
- Train: 2009-01-01 00:00:00 through 2013-12-31 23:59:59 — 390,567 rows
- Validation: 2014-01-01 00:00:00 through 2014-12-31 23:59:59 — 74,607 rows
- Test: 2015-01-01 00:00:00 through 2015-06-30 23:59:59 — 34,791 rows
- Rationale: source evidence documents 2009-01-01 through 2015-06-30, with 2015 partial; latest partial period is reserved for test, preceding complete year for validation, earlier years for training.

## 2. Leakage checks
- Source SHA-256 matched before and after execution.
- 0 duplicate `key` values crossed train/validation/test.
- `fare_amount` excluded from X.
- `key`, `User ID`, `User Name`, and `Driver Name` were excluded from X; no identifier-like field was used without availability/leakage approval.
- Preprocessing was fit only on training data for validation experiments; final preprocessing/model was fit on train+validation only after candidate lock.
- Test set was not used for model selection.

## 3–4. Experiment results / metrics
| experiment_id                   |     MAE |     RMSE |         R2 |
|:--------------------------------|--------:|---------:|-----------:|
| E01_constant_baseline           | 6.52092 | 11.3818  | -0.0339384 |
| E02_ridge_baseline              | 4.00865 |  7.73622 |  0.522328  |
| E03_hgb_baseline                | 2.85605 |  6.00307 |  0.71238   |
| E04_distance_representation     | 2.85605 |  6.00307 |  0.71238   |
| E05_temporal_enhancement        | 2.49825 |  5.44725 |  0.763175  |
| E06_geographic_enhancement      | 2.12957 |  4.80787 |  0.815508  |
| E07_reference_distance          | 2.11399 |  4.74657 |  0.820183  |
| E08_categorical_enhancement     | 2.11247 |  4.74157 |  0.820561  |
| E09_missing_value_strategy      | 2.09455 |  4.73309 |  0.821202  |
| E10_log1p_target                | 2.03965 |  4.96814 |  0.803003  |
| E11_controlled_final_comparison | 2.03965 |  4.96814 |  0.803003  |

## 5. Error analysis
Validation E10 MAE=2.039646; RMSE=4.968136; P95 absolute error=6.406812.
Distance-band validation error saved in `E10_validation_error_by_distance.csv`; upper-tail validation error saved in `E10_validation_upper_tail_error.csv`.
Final test MAE=1.973790; RMSE=5.884111; R2=0.759844.
Final test distance-band and upper-tail error tables were generated only after the final candidate was locked.

## 6. Final model-selection evidence
The predeclared primary selection criterion was validation MAE, with RMSE as tie-break. E10 log1p-target HGB achieved the lowest validation MAE among executed candidates. This does **not** mean it was superior on every metric: E09 had lower validation RMSE and higher validation R². E10 was selected under the declared primary-MAE rule, not by an overall unsupported claim.

## 7. Test-set evaluation
Final locked candidate: E10 configuration — HistGradientBoostingRegressor, log1p(fare_amount) target, log1p(distance), passenger_count, temporal variables, coordinates/bearing, all five reference distances, and three categorical variables with native missing numeric handling. Fit on train+validation; evaluated once on test.
Test MAE=1.973790; RMSE=5.884111; R2=0.759844.

## 8. Artifact verification
- Split manifest, split definition, validation results/ranking, validation predictions, model artifacts, final test predictions, error-analysis tables, configuration, logs, and verification CSV were created.
- Final selected model artifact: `models/E11_final_selected_E10_log1p_target.joblib`.

## 9. df_clean integrity verification
- SHA-256 before: `1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509`
- SHA-256 after: `1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509`
- Hash unchanged: **True**
- Source file was never overwritten or recreated.

## 10. Final Task 2 status
**TASK 2 MODELING: COMPLETED + VERIFIED**
**PRESENTATION: NOT STARTED / NOT CREATED**
**Q1–Q8 and Phases 5–7: UNCHANGED**