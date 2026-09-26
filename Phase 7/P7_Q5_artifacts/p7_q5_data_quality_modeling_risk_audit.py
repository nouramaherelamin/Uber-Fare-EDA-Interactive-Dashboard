from pathlib import Path
import hashlib
import math
import json
import pandas as pd
import numpy as np

BASE = Path('/mnt/data')
PH3 = BASE / 'phase3' / 'df_clean.csv'
OUT = BASE / 'phase7'
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_HASH = '1170f27c73a777dacd0c4712b75088ccdb1cdacb7c955368d5239a77bbad9509'
EXPECTED_SHAPE = (499965, 26)

EXPECTED_COLUMNS = [
    'User ID', 'User Name', 'Driver Name', 'Car Condition', 'Weather',
    'Traffic Condition', 'key', 'fare_amount', 'pickup_datetime',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month',
    'weekday', 'year', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist',
    'nyc_dist', 'distance', 'bearing'
]

CATEGORICALS = ['Car Condition', 'Weather', 'Traffic Condition']
TEMPORAL = ['pickup_datetime', 'hour', 'day', 'month', 'weekday', 'year']
REFERENCE = ['jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist']
COORDS = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
MODEL_RELEVANT = [
    'fare_amount', 'distance', 'passenger_count', *REFERENCE, 'bearing', *COORDS,
    *TEMPORAL, *CATEGORICALS, 'key', 'User ID', 'User Name', 'Driver Name'
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

hash_before = sha256_file(PH3)
print(f'Input SHA256 before audit: {hash_before}')

# READ-ONLY: df_clean is loaded for diagnostics only; no assignment is made to it.
df_clean = pd.read_csv(PH3, low_memory=False)
print(f'Shape: {df_clean.shape}')
print(f'Columns: {len(df_clean.columns)}')

# 1) Schema / shape
schema_rows = []
schema_rows.append({'check':'shape', 'value':f'{df_clean.shape[0]} x {df_clean.shape[1]}', 'status':df_clean.shape == EXPECTED_SHAPE})
schema_rows.append({'check':'column_count', 'value':len(df_clean.columns), 'status':len(df_clean.columns)==26})
schema_rows.append({'check':'column_order_and_names', 'value':'exact_match', 'status':list(df_clean.columns)==EXPECTED_COLUMNS})
pd.DataFrame(schema_rows).to_csv(OUT/'P7_Q5_schema_audit.csv', index=False)

# 2) Missingness across modeling-relevant columns
missing = []
for c in MODEL_RELEVANT:
    n = int(df_clean[c].isna().sum())
    missing.append({'column':c, 'missing_count':n, 'missing_percent':100*n/len(df_clean)})
missing_df = pd.DataFrame(missing)
missing_df.to_csv(OUT/'P7_Q5_missingness_audit.csv', index=False)

# 3) Target / Phase-3 integrity checks. Q1 extreme-tail metrics are imported from approved output,
# not recomputed here, to avoid rerunning P7-Q1.
target_checks = [
    {'metric':'fare_missing', 'value':int(df_clean['fare_amount'].isna().sum())},
    {'metric':'fare_nonpositive', 'value':int((df_clean['fare_amount'] <= 0).sum())},
    {'metric':'fare_max', 'value':float(df_clean['fare_amount'].max())},
]
q1_summary = pd.read_csv(OUT/'P7_Q1_target_distribution_summary.csv')
q1_map = dict(zip(q1_summary['metric'], q1_summary['value']))
for metric in ['p99','p99_5','p99_9','max','skewness']:
    if metric in q1_map:
        target_checks.append({'metric':f'approved_P7_Q1_{metric}', 'value':float(q1_map[metric])})
pd.DataFrame(target_checks).to_csv(OUT/'P7_Q5_target_audit.csv', index=False)

# 4) Phase-3-supported numeric integrity checks; no new thresholds introduced.
num_checks = []
num_checks += [
    {'check':'passenger_count_zero', 'value':int((df_clean['passenger_count'] == 0).sum()), 'status':(df_clean['passenger_count'] == 0).sum()==0},
    {'check':'distance_gt_1000', 'value':int((df_clean['distance'] > 1000).sum()), 'status':(df_clean['distance'] > 1000).sum()==0},
    {'check':'distance_negative', 'value':int((df_clean['distance'] < 0).sum()), 'status':(df_clean['distance'] < 0).sum()==0},
    {'check':'bearing_outside_minus_pi_pi', 'value':int(((df_clean['bearing'] < -math.pi) | (df_clean['bearing'] > math.pi)).sum()), 'status':((df_clean['bearing'] < -math.pi) | (df_clean['bearing'] > math.pi)).sum()==0},
]
for c in REFERENCE:
    num_checks.append({'check':f'{c}_gt_1000', 'value':int((df_clean[c] > 1000).sum()), 'status':(df_clean[c] > 1000).sum()==0})
# Remaining non-missing coordinate values must not contain the documented zero-coordinate placeholders.
for c in COORDS:
    num_checks.append({'check':f'{c}_zero_nonmissing', 'value':int(((df_clean[c] == 0) & df_clean[c].notna()).sum()), 'status':((df_clean[c] == 0) & df_clean[c].notna()).sum()==0})
pd.DataFrame(num_checks).to_csv(OUT/'P7_Q5_numeric_integrity_audit.csv', index=False)

# 5) Temporal consistency. pickup_datetime is parsed temporarily; source column is untouched.
dt = pd.to_datetime(df_clean['pickup_datetime'], errors='coerce')
temporal_rows = [
    {'check':'pickup_datetime_missing','count':int(dt.isna().sum()),'status':dt.isna().sum()==0},
    {'check':'hour_range_invalid','count':int((~df_clean['hour'].between(0,23)).sum()),'status':(~df_clean['hour'].between(0,23)).sum()==0},
    {'check':'day_range_invalid','count':int((~df_clean['day'].between(1,31)).sum()),'status':(~df_clean['day'].between(1,31)).sum()==0},
    {'check':'month_range_invalid','count':int((~df_clean['month'].between(1,12)).sum()),'status':(~df_clean['month'].between(1,12)).sum()==0},
    {'check':'weekday_range_invalid','count':int((~df_clean['weekday'].between(0,6)).sum()),'status':(~df_clean['weekday'].between(0,6)).sum()==0},
]
# Compare against datetime-derived fields only on rows with parseable datetime.
temporal_rows += [
    {'check':'hour_vs_datetime_mismatch','count':int((df_clean.loc[dt.notna(),'hour'].astype(int).to_numpy() != dt.loc[dt.notna()].dt.hour.to_numpy()).sum()),'status':True},
    {'check':'day_vs_datetime_dayofmonth_mismatch','count':int((df_clean.loc[dt.notna(),'day'].astype(int).to_numpy() != dt.loc[dt.notna()].dt.day.to_numpy()).sum()),'status':True},
    {'check':'month_vs_datetime_mismatch','count':int((df_clean.loc[dt.notna(),'month'].astype(int).to_numpy() != dt.loc[dt.notna()].dt.month.to_numpy()).sum()),'status':True},
    {'check':'weekday_vs_datetime_mismatch','count':int((df_clean.loc[dt.notna(),'weekday'].astype(int).to_numpy() != dt.loc[dt.notna()].dt.weekday.to_numpy()).sum()),'status':True},
    {'check':'year_vs_datetime_mismatch','count':int((df_clean.loc[dt.notna(),'year'].astype(int).to_numpy() != dt.loc[dt.notna()].dt.year.to_numpy()).sum()),'status':True},
]
for r in temporal_rows:
    if 'mismatch' in r['check']:
        r['status'] = r['count']==0
temporal_df = pd.DataFrame(temporal_rows)
temporal_df.to_csv(OUT/'P7_Q5_temporal_consistency_audit.csv', index=False)

# 6) Duplicate / identifier audit. This documents risk; it does not deduplicate.
key_dup_mask = df_clean['key'].duplicated(keep=False)
key_dup_values = int(df_clean.loc[key_dup_mask,'key'].nunique(dropna=True))
key_excess = int(df_clean['key'].duplicated(keep='first').sum())
exact_dupes = int(df_clean.duplicated(keep=False).sum())
identifier_rows = []
for c in ['key','User ID','User Name','Driver Name']:
    identifier_rows.append({
        'column':c,
        'missing_count':int(df_clean[c].isna().sum()),
        'nunique':int(df_clean[c].nunique(dropna=True)),
        'duplicate_rows_excess':int(df_clean[c].duplicated(keep='first').sum()),
    })
identifier_rows += [
    {'column':'key__duplicated_distinct_values','missing_count':0,'nunique':key_dup_values,'duplicate_rows_excess':key_excess},
    {'column':'exact_duplicate_rows','missing_count':0,'nunique':0,'duplicate_rows_excess':exact_dupes},
]
identifier_df = pd.DataFrame(identifier_rows)
identifier_df.to_csv(OUT/'P7_Q5_identifier_duplicate_audit.csv', index=False)

# 7) Categorical representation: actual observed domains and counts; no recoding.
cat_rows=[]
for c in CATEGORICALS:
    vc=df_clean[c].value_counts(dropna=False)
    for value,count in vc.items():
        cat_rows.append({'column':c,'category':str(value),'count':int(count),'percent':100*count/len(df_clean)})
cat_df=pd.DataFrame(cat_rows)
cat_df.to_csv(OUT/'P7_Q5_categorical_audit.csv', index=False)

# 8) Temporal coverage summary (descriptive, no new thresholds).
year_min=int(df_clean['year'].min()); year_max=int(df_clean['year'].max())
coverage = pd.DataFrame([
    {'metric':'year_min','value':year_min},
    {'metric':'year_max','value':year_max},
    {'metric':'datetime_min','value':str(dt.min())},
    {'metric':'datetime_max','value':str(dt.max())},
    {'metric':'unique_year_month_periods','value':int(dt.dt.to_period('M').nunique())},
])
coverage.to_csv(OUT/'P7_Q5_temporal_coverage_audit.csv', index=False)

# 9) Consolidated risk register. Numerical values are linked to outputs generated above or approved prior outputs.
risk_rows = [
    {'risk_id':'R1','area':'Target tail','evidence':'P7-Q1 approved output imported without recomputation','finding':'Fare distribution is strongly right-skewed and contains a high upper tail; extreme values were not deleted/capped in Phase 3.','task2_relevance':'Target transformation/outlier treatment should be considered during modeling; no treatment selected here.','source_output':'P7_Q5_target_audit.csv'},
    {'risk_id':'R2','area':'Missing predictors','evidence':'P7-Q5 missingness audit','finding':'Intentional missing predictor values remain after Phase 3 cleaning, especially geographic/reference-distance fields and passenger_count.','task2_relevance':'Task 2 must define a missing-value handling strategy without changing df_clean here.','source_output':'P7_Q5_missingness_audit.csv'},
    {'risk_id':'R3','area':'Duplicate identifiers','evidence':'P7-Q5 identifier audit','finding':'Duplicate Key values remain by design; exact duplicate rows are separately audited.','task2_relevance':'Train/test splitting and identifier handling should account for duplicated keys; no deduplication performed.','source_output':'P7_Q5_identifier_duplicate_audit.csv'},
    {'risk_id':'R4','area':'Temporal coverage','evidence':'P7-Q5 temporal consistency/coverage audit','finding':'Temporal fields are checked against pickup_datetime and the observed coverage is documented.','task2_relevance':'Time-aware splitting/validation may need consideration; no split is performed in Phase 7.','source_output':'P7_Q5_temporal_consistency_audit.csv'},
    {'risk_id':'R5','area':'Redundant numerical variables','evidence':'Approved P7-Q4 output; not recomputed','finding':'The accepted Q4 evidence documents very strong redundancy among EWR/SOL/NYC reference distances.','task2_relevance':'Later modeling diagnostics should account for redundancy; no feature removal is performed.','source_output':'P7_Q4_reference_distance_pearson_matrix.csv'},
    {'risk_id':'R6','area':'Categorical representation','evidence':'P7-Q5 categorical audit','finding':'Actual observed categorical domains and counts are documented without recoding.','task2_relevance':'Task 2 must choose an encoding strategy appropriate to the observed categories.','source_output':'P7_Q5_categorical_audit.csv'},
    {'risk_id':'R7','area':'Identifier/leakage concern','evidence':'P7-Q5 identifier audit','finding':'Identifier-like columns are documented separately from modeling variables; this audit does not establish leakage.','task2_relevance':'Before modeling, identifier-like fields should be reviewed for whether they represent usable predictors or leakage risk; no feature is removed here.','source_output':'P7_Q5_identifier_duplicate_audit.csv'},
]
risk_df=pd.DataFrame(risk_rows)
risk_df.to_csv(OUT/'P7_Q5_risk_register.csv', index=False)

# 10) Planned diagnostic plots: compact missingness and audit-risk overview.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Missingness plot: only columns with >0 missingness, sorted descending.
mi = missing_df[missing_df['missing_count']>0].sort_values('missing_count', ascending=True)
plt.figure(figsize=(9,6))
plt.barh(mi['column'], mi['missing_percent'])
plt.xlabel('Missing values (%)')
plt.ylabel('Column')
plt.title('P7-Q5 Remaining Missingness by Relevant Variable')
plt.tight_layout()
plt.savefig(OUT/'P7_Q5_missingness_diagnostic.png', dpi=160)
plt.close()

# Risk-status plot: counts of failed checks should be zero; show integrity checks only.
checks = pd.concat([
    pd.DataFrame(num_checks),
    temporal_df[['check','count','status']].rename(columns={'count':'value'})
], ignore_index=True)
failed = int((checks['status']==False).sum())
passed = int((checks['status']==True).sum())
plt.figure(figsize=(6,4))
plt.bar(['Passed','Failed'], [passed, failed])
plt.ylabel('Number of audit checks')
plt.title('P7-Q5 Audit Check Status')
plt.tight_layout()
plt.savefig(OUT/'P7_Q5_audit_check_status.png', dpi=160)
plt.close()

# 11) Verification.
hash_after = sha256_file(PH3)
verification = []
verification.append({'check':'df_clean_shape','actual':f'{df_clean.shape[0]} x {df_clean.shape[1]}','expected':'499965 x 26','pass':df_clean.shape==EXPECTED_SHAPE})
verification.append({'check':'df_clean_sha256','actual':hash_after,'expected':EXPECTED_HASH,'pass':hash_after==EXPECTED_HASH})
verification.append({'check':'schema_exact','actual':str(list(df_clean.columns)==EXPECTED_COLUMNS),'expected':'True','pass':list(df_clean.columns)==EXPECTED_COLUMNS})
verification.append({'check':'fare_missing','actual':int(df_clean['fare_amount'].isna().sum()),'expected':'0','pass':df_clean['fare_amount'].isna().sum()==0})
verification.append({'check':'fare_nonpositive','actual':int((df_clean['fare_amount']<=0).sum()),'expected':'0','pass':(df_clean['fare_amount']<=0).sum()==0})
verification.append({'check':'passenger_zero','actual':int((df_clean['passenger_count']==0).sum()),'expected':'0','pass':(df_clean['passenger_count']==0).sum()==0})
verification.append({'check':'distance_gt_1000','actual':int((df_clean['distance']>1000).sum()),'expected':'0','pass':(df_clean['distance']>1000).sum()==0})
verification.append({'check':'reference_gt_1000_total_cells','actual':int(sum((df_clean[c]>1000).sum() for c in REFERENCE)),'expected':'0','pass':sum((df_clean[c]>1000).sum() for c in REFERENCE)==0})
verification.append({'check':'bearing_outside_range','actual':int(((df_clean['bearing']<-math.pi)|(df_clean['bearing']>math.pi)).sum()),'expected':'0','pass':((df_clean['bearing']<-math.pi)|(df_clean['bearing']>math.pi)).sum()==0})
verification.append({'check':'datetime_missing','actual':int(dt.isna().sum()),'expected':'0','pass':dt.isna().sum()==0})
verification.append({'check':'temporal_mismatches_total','actual':int(sum(r['count'] for r in temporal_rows if 'mismatch' in r['check'])),'expected':'0','pass':all(r['status'] for r in temporal_rows if 'mismatch' in r['check'])})
verification.append({'check':'exact_duplicate_rows','actual':exact_dupes,'expected':'0','pass':exact_dupes==0})
verification.append({'check':'hash_unchanged_during_audit','actual':hash_before==hash_after,'expected':'True','pass':hash_before==hash_after})
verification.append({'check':'q4_not_recomputed','actual':'Q4 numerical matrix imported as approved evidence only','expected':'No Q4 rerun','pass':True})
verification.append({'check':'q1_q2_q3_not_recomputed','actual':'Prior approved outputs referenced only where needed','expected':'No rerun','pass':True})
verification_df=pd.DataFrame(verification)
verification_df.to_csv(OUT/'P7_Q5_verification.csv', index=False)

# Write a concise machine-readable run record.
run_record = {
    'status':'COMPLETED AND VERIFIED' if bool(verification_df['pass'].all()) else 'FAIL',
    'source':str(PH3),
    'shape':list(df_clean.shape),
    'sha256_before':hash_before,
    'sha256_after':hash_after,
    'q1_q4_rerun':False,
    'q4_conditional_distance_band_diagnostic':False,
    'df_clean_modified':False,
    'modeling_started':False,
    'presentation_started':False,
}
with open(OUT/'P7_Q5_run_output.txt','w',encoding='utf-8') as f:
    json.dump(run_record,f,indent=2)
    f.write('\n\nVerification table:\n')
    f.write(verification_df.to_string(index=False))

print(json.dumps(run_record, indent=2))
print('\nVerification:')
print(verification_df.to_string(index=False))
