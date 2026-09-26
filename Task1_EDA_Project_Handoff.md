# PROJECT HANDOFF — Uber Fare Prediction, Task 1 (EDA)
### Cellula Technologies ML Internship Program
**Prepared for:** the next AI assistant continuing this project
**Prepared by:** previous Claude session (context transfer — no analysis has been performed on real data yet)

> **READ THIS ENTIRE DOCUMENT BEFORE DOING ANYTHING.** This is the full accumulated context, reasoning, decisions, and execution plan for this project. Do not restart the project, do not redesign the workflow, and do not skip ahead. Work phase-by-phase as laid out below.

---

## 1. PROJECT CONTEXT

**What the internship task is:** This is Task 1 of a multi-task ML internship at Cellula Technologies. The overall project is "Uber Fare Prediction" — the end goal (Task 2, NOT part of this handoff) is to build a model that predicts `Fare_Amount` for NYC Uber rides. Task 1 is exploratory data analysis only — no modeling.

**What Task 1 is trying to teach:** The official PDF states this directly: *"This task trains you to explore, question, and visualize data — and to decide which plot best answers each question."* It is a skill-building exercise in (a) correctly matching visualization type to the analytical question being asked, and (b) extracting and communicating a defensible insight from that visualization — not simply generating charts.

**What EDA means in this project:** Investigating the dataset's structure, quality, distributions, and relationships *before* any modeling occurs. It includes data auditing, cleaning-decision documentation, univariate/bivariate analysis, correlation analysis, outlier investigation, temporal and (if valid) geographic analysis, and turning all of this into notebook + presentation deliverables.

**Why EDA comes before ML:** A model is only as trustworthy as the understanding behind it. Undetected issues (invalid fares, outlier distances, datetime/derived-column mismatches) will silently corrupt a model built on top of them. EDA is where these are caught.

**Actual objective (verbatim spirit of the PDF):** Deeply understand the data, decide which plot best answers each question, and let the resulting insights shape Task 2 modeling decisions. The PDF closes with: *"Remember: The goal is not to produce the most code — it is to produce the most insight."*

**What the evaluator is likely checking (inferred from the PDF's structure and checklist — not verbatim quoted requirements beyond what's cited in Section 2):**
- Correct match of plot type to each question (not a reflexive/random choice)
- Written justification for each plot choice
- Correct interpretation of each plot (not just showing it)
- Clear separation between observation, insight, and unsupported conclusion
- Documented, non-destructive cleaning decisions
- A coherent notebook and a story-driven presentation

**What "good insight" means in this project:** A data-supported statement that has a *consequence* — it changes something about how you would clean, engineer features, or model going forward. Not merely "the chart shows an upward trend," but why that trend matters and what it implies for Task 2.

**Why plot selection matters:** Different plot types reveal or conceal different structure (e.g., a bar chart of group means hides within-group variance that a boxplot would reveal). Choosing the plot that actually exposes the structure relevant to the question is a real analytical skill being tested here.

**Why interpretation matters more than generation:** Anyone can call a plotting function. The differentiator — and the explicit statement in the PDF — is producing insight, not chart volume.

**Distinction discipline (applies throughout this whole handoff and the eventual submission):**
- **Official requirement** = explicitly present in Task1.pdf (Section 2 below marks these)
- **Recommended Enhancement** = professional practice we've layered on top, not literally mandated by the PDF. Must always be labeled as such in the final notebook/presentation, never presented as an official requirement.

---

## 2. OFFICIAL TASK REQUIREMENTS (verbatim/paraphrased from Task1.pdf)

**Title:** "Task 1 — Exploratory Data Analysis" / "Uber Fare Prediction Project • ML Internship Program"

**Objective (from PDF):** "Before building any model, you need to deeply understand your data. This task trains you to explore, question, and visualize data — and to decide which plot best answers each question. The insights you find here will directly shape your decisions in Task 2 (modeling)."

**Dataset description (from PDF):** "The dataset contains real-world Uber ride records from New York City. Each row represents one trip."

**Dataset link (from PDF):** `final_internship_data.csv` — hosted on Google Drive (link was a hyperlink in the PDF, not a plain URL text; the user has since provided the actual file locally as `.xlsx`, see Section 25).

**Target variable (from PDF):** `Fare_Amount` — explicitly marked "TARGET — the fare paid in USD."

**Official column table (from PDF, verbatim column/type/description):**

| Column | Type | Description (from PDF) |
|---|---|---|
| User_ID | Integer | Unique identifier per user — not useful for modeling |
| User_Name | String | Passenger name — not useful for modeling |
| Driver_Name | String | Driver name — not useful for modeling |
| Car_Condition | Categorical | Vehicle condition: Good / Bad |
| Weather | Categorical | Weather at pickup: Sunny / Windy / Winter |
| Traffic_Conditions | Categorical | Traffic level: Normal / Heavy |
| Key | Integer | Unique trip ID — use only for deduplication |
| Fare_Amount | Float | TARGET — the fare paid in USD |
| Pickup_Datetime | DateTime | Timestamp when the ride was requested |
| Pickup_Longitude | Float | Longitude of pickup location |
| Pickup_Latitude | Float | Latitude of pickup location |
| Dropoff_Longitude | Float | Longitude of drop-off location |
| Dropoff_Latitude | Float | Latitude of drop-off location |
| Passenger_Count | Integer | Number of passengers in the ride |
| Hour | Integer | Hour extracted from Pickup_Datetime (0–23) |
| Day | Integer | Day of week (0=Monday … 6=Sunday) |
| Month | Integer | Month of pickup (1–12) |
| Week | Integer | Week number of the year |
| JFK_Dist | Float | Distance from pickup to JFK Airport (km) |
| EWR_Dist | Float | Distance from pickup to EWR Airport (km) |
| LGA_Dist | Float | Distance from pickup to LGA Airport (km) |
| SOL_Dist | Float | Distance from pickup to Statue of Liberty (km) |
| Distance | Float | Total trip distance (km) |
| Bearing | Float | Trip direction angle in degrees (0–360) |

**Visualization Quick Reference (from PDF, verbatim table):**

| Plot | Best For | Example Question | Limitation |
|---|---|---|---|
| Histogram | Distribution of 1 numeric variable | How is Fare_Amount distributed? | Sensitive to bin size |
| Boxplot | Distribution + outliers of numeric variable | Are there extreme fares? | Hides sample size |
| Bar Chart | Compare averages across categories | What is the avg fare per Weather type? | Hides variance within groups |
| Line Chart | Trends over ordered values (time) | How does avg fare change by hour? | Only for ordered/time data |
| Scatter Plot | Relationship between 2 numeric variables | Does Distance predict Fare? | Slow on large datasets |
| Heatmap (corr.) | Correlations between all numeric features | Which features relate most to Fare? | Only linear relationships |
| Heatmap (pivot) | Two-variable patterns (e.g., time x day) | When is demand highest? | Needs aggregation first |
| Count Plot | Frequency of categories | How many rides per Traffic type? | Doesn't show fare |
| KDE Plot | Smooth distribution — compare groups | Does fare differ by Car_Condition? | Harder to read exact values |
| Geo Scatter | Spatial distribution of pickup points | Where in NYC are rides concentrated? | Needs valid coordinates |
| Pair Plot | All pairwise relationships at once | Quick overview of feature relationships | Slow on many columns |

**Training Questions Q1–Q8 (from PDF, verbatim as given, including the PDF's own preliminary variable/plot notes):**

- **Q1)** "Is higher Passenger higher price?" — Passenger count: Numerical; Price: numerical; Suitable plot: scatterplot; What you will see: whether there are relationship by closeness of points or not
- **Q2)** "Does car condition influence the fare amount?" — Car condition: categorical; Fare: numerical; Suitable plot: boxplot/barplot
- **Q3)** "At what hour of the day are fares typically highest?" — Hour: numerical; Fare: numerical; Suitable plot: lineplot
- **Q4)** "Does traffic condition affect the trip fare?" — Traffic_Conditions: Categorical; Fare_Amount: Numerical; Suitable plot: Boxplot/Barplot; What you will see: whether rides during heavy traffic tend to have higher fares compared to normal traffic
- **Q5)** "Is trip distance the strongest predictor of the fare amount?" — Distance: Numerical; Fare_Amount: Numerical; Suitable plot: Scatter Plot; What you will see: whether fare increases as trip distance increases and whether the relationship appears linear
- **Q6)** "At what time of day are ride requests most frequent?" — Hour: Numerical (Discrete); Ride Count: Frequency; Suitable plot: Count Plot/Bar Chart; What you will see: peak demand hours and periods with the highest number of trips. **PDF explicitly notes:** "Different from your current question 'highest fare by hour'; this one analyzes demand rather than price."
- **Q7)** "Does weather condition influence the average trip distance?" — Weather: Categorical; Distance: Numerical; Suitable plot: Boxplot/Barplot; What you will see: whether people tend to travel longer or shorter distances under different weather conditions
- **Q8)** "Are rides that start closer to airports generally more expensive?" — JFK_Dist (or EWR_Dist/LGA_Dist): Numerical; Fare_Amount: Numerical; Suitable plot: Scatter Plot; What you will see: whether proximity to airports is associated with higher fares

**Instruction preceding Q1–Q8 (from PDF):** "For each question below, first identify which plot is most appropriate and why, then write the code to produce it. There may be more than one valid answer — justify your choice."

**Additional questions requirement (from PDF, Submission Checklist):** "Benefit from above questions to analyze and create **new Questions** for understand the data."

**Notebook requirement (from PDF, Submission Checklist):** "Notebook contain: question, plot choice, cleaning notes, code, and interpretation."

**Presentation requirement (from PDF, Submission Checklist):** "Presentation shows: question, plot and the insights in story."

**Code quality requirement (from PDF, Submission Checklist):** "Code is clean and commented."

**Closing statement (from PDF):** "Remember: The goal is not to produce the most code — it is to produce the most insight."

**Full Submission Checklist (from PDF, verbatim):**
```
[ ] Benefit from above questions to analyze and create new Questions for understand the data
[ ] Notebook contain: question, plot choice, cleaning notes, code, and interpretation
[ ] Presentation shows: question, plot and the insights in story
[ ] Code is clean and commented
```

**Extra resources listed in PDF (for reference, not requirements):** 4 YouTube video links + 2 visualization-reference websites (datavizcatalogue.com, datavizproject.com) — informational only, not part of deliverable requirements.

**IMPORTANT — everything NOT explicitly in the checklist above (e.g., specific numbered notebook sections, memory-efficiency workflow, the 12-stage large-file strategy, "df_raw vs df_clean" naming, exact structured question format with 12 sub-fields) is Recommended Enhancement layered on by the assistant for professional quality — not a literal PDF requirement. It should still be followed (it was agreed upon and is good practice) but must never be described to the user or graded as "the PDF requires this."**

---

## 3. COMPLETE DATASET COLUMN MAP

| Column | Type (PDF) | Meaning | Role in EDA | Role in modeling | Special notes | Potential issues to check |
|---|---|---|---|---|---|---|
| User_ID | Integer | Unique passenger identifier | Not directly analyzed; possibly used to check for repeat-user patterns as an *additional* question only | Exclude — identifier, not a ride attribute | PDF explicitly says "not useful for modeling" | Should not be used as a predictive feature |
| User_Name | String | Passenger name | Not analyzed | Exclude | PDF: "not useful for modeling" | PII-like; irrelevant to fare |
| Driver_Name | String | Driver name | Not analyzed | Exclude | PDF: "not useful for modeling" | PII-like; irrelevant to fare |
| Car_Condition | Categorical (Good/Bad) | Vehicle condition | Q2 — compared against Fare_Amount | Possible categorical feature (encode) | Only 2 levels expected | Verify only "Good"/"Bad" appear; check for typos/extra categories |
| Weather | Categorical (Sunny/Windy/Winter) | Weather at pickup | Q7 — compared against Distance | Possible categorical feature (encode) | 3 levels expected | Verify category spelling/imbalance |
| Traffic_Conditions | Categorical (Normal/Heavy) | Traffic level | Q4 — compared against Fare_Amount | Possible categorical feature (encode) | 2 levels expected | Verify category spelling/imbalance |
| Key | Integer | Unique trip ID | Used ONLY for deduplication per PDF | Exclude from modeling | PDF: "use only for deduplication" | Check for duplicate Key values (would indicate duplicate rows) |
| Fare_Amount | Float | **TARGET** | Central to nearly every question; dedicated Target Analysis section | This is what we predict in Task 2 | — | Fare ≤ 0, extreme outliers, distribution shape/skew |
| Pickup_Datetime | DateTime | Timestamp of ride request | Source of truth for Hour/Day/Month/Week; consistency-checked against them | Source for possible engineered temporal features | — | Invalid/unparseable timestamps; mismatch with derived Hour/Day/Month/Week columns |
| Pickup_Longitude | Float | Pickup longitude | Geographic analysis (optional/Recommended Enhancement) | Possible geospatial feature engineering | — | Values outside plausible NYC bounding box |
| Pickup_Latitude | Float | Pickup latitude | Geographic analysis | Possible geospatial feature engineering | — | Values outside plausible NYC bounding box |
| Dropoff_Longitude | Float | Dropoff longitude | Geographic analysis | Possible geospatial feature engineering | — | Values outside plausible NYC bounding box |
| Dropoff_Latitude | Float | Dropoff latitude | Geographic analysis | Possible geospatial feature engineering | — | Values outside plausible NYC bounding box |
| Passenger_Count | Integer | Number of passengers | Q1 — compared against Fare_Amount | Possible numeric/discrete feature | Discrete, low-cardinality | Passenger_Count ≤ 0, unrealistically large values |
| Hour | Integer (0–23) | Hour of pickup | Q3 (fare by hour), Q6 (demand by hour) | Possible cyclical/temporal feature (may need cyclical encoding) | Ordered discrete — treat as sequential for line plots | Values outside 0–23; mismatch vs Pickup_Datetime |
| Day | Integer (0=Mon…6=Sun) | Day of week | Additional questions (weekday/weekend, Hour×Day) | Possible categorical/cyclical feature | Ordered discrete | Values outside 0–6; mismatch vs Pickup_Datetime |
| Month | Integer (1–12) | Month of pickup | Additional questions (monthly trends) | Possible seasonal feature | Ordered discrete | Values outside 1–12; mismatch vs Pickup_Datetime |
| Week | Integer | ISO/week-of-year number | Additional questions (weekly trends) | Possible seasonal feature | — | Implausible week values; mismatch vs Pickup_Datetime |
| JFK_Dist | Float (km) | Distance to JFK Airport | Q8 (one of three airport-distance variables) | Possible predictive feature; check redundancy with other distances | — | Negative values, extreme values |
| EWR_Dist | Float (km) | Distance to EWR Airport | Q8 | Possible predictive feature | — | Negative values, extreme values |
| LGA_Dist | Float (km) | Distance to LGA Airport | Q8 | Possible predictive feature | — | Negative values, extreme values |
| SOL_Dist | Float (km) | Distance to Statue of Liberty | Not in Q1–Q8 explicitly, but available for additional questions/correlation | Possible predictive feature | Not airport-related — likely a general geographic reference point | Negative values, extreme values |
| Distance | Float (km) | Total trip distance | Q1 (context), Q5 (central), correlation analysis | Very likely strong feature | — | Distance < 0, zero-distance-but-nonzero-fare cases (possible flat-fare trips), extreme values |
| Bearing | Float (0–360°) | Trip direction angle | Additional question candidate (fare vs bearing) | Possible cyclical feature (may need sin/cos transform) | Circular data — mean/typical stats need care | Values outside 0–360 |

**Verification requirement:** Per the PDF's own caution and our established rule — *do not assume the actual dataset matches this table.* The real `sample_50000.csv` (and later the full `.xlsx`) must be inspected in Phase 1 to confirm columns exist, are spelled identically, and hold the expected dtypes/categories. Any discrepancy must be reported, not silently resolved.

---

## 4. VARIABLE CATEGORIZATION

| Category | Columns | Why it matters for plot/method choice |
|---|---|---|
| **Target** | Fare_Amount | Everything is ultimately evaluated against this; gets its own dedicated analysis section |
| **Continuous numerical** | Distance, JFK_Dist, EWR_Dist, LGA_Dist, SOL_Dist, Bearing, Pickup/Dropoff Longitude & Latitude | Candidates for scatter plots, histograms, correlation matrix; skew/outlier checks apply |
| **Discrete numerical** | Passenger_Count, Hour, Day, Month, Week | Look numeric but often behave like ordered categories — matters for choosing between scatter (truly continuous) vs. line/bar (ordered/aggregated) |
| **Categorical** | Car_Condition, Weather, Traffic_Conditions | Compared via group statistics (boxplot/barplot/KDE), never via Pearson correlation directly |
| **Datetime** | Pickup_Datetime | Source of truth for Hour/Day/Month/Week; must be cross-validated against them |
| **Identifier** | User_ID, User_Name, Driver_Name, Key | Never used as predictive features; Key used only for dedup |
| **Geographic** | Pickup/Dropoff Longitude & Latitude | Subject to bounding-box validity checks before any geo-plot is trusted |
| **Distance-related** | Distance, JFK_Dist, EWR_Dist, LGA_Dist, SOL_Dist | Likely correlated with each other (multicollinearity candidate) and with Fare_Amount |
| **Temporal (derived)** | Hour, Day, Month, Week | Central to Q3, Q6, and most additional questions; consistency with Pickup_Datetime must be checked |

---

## 5. TOOLS AND TECHNOLOGY

| Tool | What it does | Why needed | Where used |
|---|---|---|---|
| Python | General-purpose language | Standard for data science | Everything |
| Jupyter Notebook | Interactive, cell-based, mixes code/output/Markdown | The deliverable itself is a notebook | Entire submission |
| Pandas | DataFrames, tabular manipulation | Loading, cleaning, grouping, aggregating | Almost every step |
| NumPy | Numerical arrays/math | Underlies Pandas; log transforms, percentiles | Target analysis, outlier detection |
| Matplotlib | Low-level plotting | Fine control over titles/axes/figures | Custom plots, subplots |
| Seaborn | Statistical plotting on top of Matplotlib | Cleaner syntax for boxplots/KDEs/heatmaps | Most visualizations |
| OpenPyXL | Excel file engine for Pandas | The real dataset is `.xlsx` (`final_internship_data.xlsx`), not `.csv` — OpenPyXL is the read engine Pandas needs | Loading the full dataset |

**Confirmed installed/available in user's environment (per user's latest message):**
- Python 3.13.15
- pandas 3.0.6
- openpyxl 3.1.5
- numpy
- matplotlib
- seaborn

**Tools discussed and NOT needed for Task 1 (with reasoning preserved):**
- **Plotly/Bokeh** — interactivity is not required by the PDF; adds unneeded complexity. Skip unless the user specifically wants interactive geographic plots later.
- **Scikit-learn** — this is EDA only (Task 1). No train/test split, no preprocessing pipeline, no model. Explicitly out of scope until Task 2.
- **Dask** — only relevant if Pandas truly cannot hold the data in memory. At ~139MB, this is very unlikely to be a real "big data" problem for Pandas — worth being memory-conscious (dtype optimization, `usecols`, chunked validation where helpful) but full Dask adoption is probably unnecessary. This must be *confirmed*, not assumed, once the full file's exact size/row count is known (Phase 1).

---

## 6. COMPLETE PROJECT WORKFLOW (12-Phase Structure — the plan we will follow)

### PHASE 0 — Project Understanding
- **Objective:** Fully understand the official task before touching data.
- **What we do:** Read Task1.pdf; identify target, columns, Q1–Q8, deliverables; separate official requirements from recommended enhancements.
- **Why:** Prevents scope drift and invented requirements.
- **Output:** Clear project specification (this document, Sections 1–2, already satisfies this).
- **Must understand before moving on:** Exactly what Cellula asks for.
- **STATUS: COMPLETE.**

### PHASE 1 — Initial Dataset Inspection
- **Objective:** Understand the actual structure of the data before cleaning or analyzing.
- **What we do:** Inspect `sample_50000.csv` first (shape, columns, dtypes, missing %, uniques, descriptive stats, obvious issues, match vs. PDF documentation). Then assess the full `final_internship_data.xlsx` only enough to determine exact row/column count, file size, approximate memory requirement, and whether full in-memory loading is practical or chunking/dtype optimization is needed.
- **Why:** No cleaning or analysis should happen blind; the large-file strategy depends on real numbers, not assumptions.
- **Output:** A documented dataset profile (sample + full-file sizing).
- **Must NOT do yet:** Clean data, answer Q1–Q8, draw conclusions, fabricate any numbers.
- **STATUS: NOT STARTED — this is the immediate next step.**

### PHASE 2 — Full Data Quality Audit
- **Objective:** Determine whether the dataset contains quality problems that could distort the EDA.
- **What we do:** Run the full check list in Section 9 below against the real data (sample first for a preview, full dataset for final numbers). For every issue found: Problem → Count → Percentage → Possible reason → Decision → Justification.
- **Why:** You cannot design a defensible cleaning strategy without first knowing what's actually wrong.
- **Output:** A complete data-quality report with real numbers.
- **Must NOT do yet:** Apply the cleaning (that's Phase 3); jump to Q1–Q8.

### PHASE 3 — Cleaning
- **Objective:** Create a trustworthy analysis dataset without destroying valid information.
- **What we do:** Preserve `df_raw`; create `df_clean` only with justified, documented changes, following the Cleaning Decision Framework in Section 10. Compare before/after row counts and stats.
- **Why:** Undocumented deletion is unacceptable and undermines every downstream conclusion.
- **Output:** `df_clean` + a cleaning log + a validation summary.
- **Must NOT do yet:** Automatically delete outliers without investigation; move to Q1–Q8 before validating the clean dataset.

### PHASE 4 — Baseline EDA
- **Objective:** Understand general dataset behavior before answering the specific required questions.
- **What we do:** Target distribution, Distance distribution, Passenger_Count distribution, categorical distributions, temporal distributions, a first look at correlations, a first look at potential outliers.
- **Why:** Context makes Q1–Q8 interpretation sharper and prevents surface-level answers.
- **Output:** General dataset understanding, ready to support Q1–Q8.
- **Must NOT do yet:** Treat this baseline pass as the final answer to any required question.

### PHASE 5 — Official Questions Q1–Q8
- **Objective:** Answer every required question using the structured format (Section 19).
- **What we do:** For each question — Question → Variables → Variable types → Plot choice → Why this plot → Data prep → Code → Visualization → Evidence/numbers → Interpretation → Limitations → Modeling relevance.
- **Why:** This is the core mandatory deliverable content.
- **Output:** Eight complete, evidence-based analyses.
- **Critical reminders carried forward:** Q3 (fare level) ≠ Q6 (ride demand) — must stay separate; Q5 cannot be answered from a single scatterplot alone — needs comparative correlation analysis, and "strongest predictor" language must be scoped to EDA-level association, not ML feature importance; Q8 "closer" = smaller distance value, don't invert the reading.
- **Must NOT do yet:** Skip any of the 8; substitute barplot-only analysis where the PDF or our reasoning calls for boxplot (which shows spread, not just mean).

### PHASE 6 — Additional Data Questions
- **Objective:** Go beyond required questions to demonstrate independent analytical thinking (explicit PDF requirement: "create new Questions for understand the data").
- **What we do:** Use findings from Q1–Q8 to motivate at least 8 new questions, from the candidate list in Section 13. Same structured format as Q1–Q8.
- **Why:** Required by the PDF checklist; also strengthens Task 2 preparation.
- **Output:** A focused set of strong additional analyses (not padding).
- **Must NOT do yet:** Create questions with no analytical purpose just to inflate count; the questions must be answerable with existing columns.

### PHASE 7 — Deep-Dive Analysis
- **Objective:** Investigate the most important patterns surfaced so far in more depth.
- **What we do:** Full correlation matrix + heatmap, outlier investigation (IQR-based), temporal patterns (Hour×Day pivot heatmap, monthly/weekly trends), geographic patterns (only if coordinates validate), multicollinearity discussion among distance columns, category imbalance checks.
- **Why:** Some questions (correlation ranking for Q5, multicollinearity for Task 2) need dedicated deeper treatment beyond a single plot.
- **Output:** Deeper understanding of the strongest findings.
- **Must NOT do yet:** Treat exploratory deep-dive results as final without cross-checking against the cleaned, full dataset.

### PHASE 8 — Final Insights
- **Objective:** Convert charts/statistics into clear, evidence-based insights.
- **What we do:** For every important finding, separate Observation → Insight → Modeling implication, following Section 11/12 frameworks. Avoid unsupported/causal claims.
- **Output:** A concise, evidence-based insights list.
- **Must NOT do yet:** State anything not traceable to an actual number or plot already produced.

### PHASE 9 — Task 2 / Modeling Preparation
- **Objective:** Translate Task 1 findings into modeling guidance (without modeling).
- **What we do:** Document useful features, irrelevant identifiers, categorical encoding needs, missing-data strategy, outlier strategy, transformation candidates, multicollinearity, leakage risks, promising relationships — see Section 21.
- **Output:** A modeling-preparation section in the notebook.
- **Must NOT do:** Actually build, train, or evaluate any model — that's Task 2, explicitly out of scope here.

### PHASE 10 — Final Notebook
- **Objective:** Assemble everything into one professional, reproducible `.ipynb`.
- **What we do:** Follow the architecture in Section 19.
- **Output:** Final notebook, runs top-to-bottom without errors.
- **Must NOT do:** Leave any section's conclusions unsupported by code/output actually present in the notebook.

### PHASE 11 — Presentation
- **Objective:** Convert the strongest findings into a clear, story-driven presentation.
- **What we do:** Question → Plot → Insight per slide, ~10–14 slides, following Section 20.
- **Output:** Final presentation.
- **Must NOT do:** Copy the entire notebook into the deck; include every plot.

### PHASE 12 — Final QA
- **Objective:** Review the entire submission before calling it done.
- **What we do:** Run through the full checklist in Section 22/27 (data, EDA, code, visualization, insights, presentation).
- **Output:** Submission-ready Task 1.

---

## 7. Q1–Q8 COMPLETE ANALYSIS PLAN

> No actual results are included below — only the analytical plan, reasoning, and pitfalls established so far. All real numbers/plots must come from Phase 1 onward using actual data.

### Q1 — Is higher Passenger higher price?
- **Variables:** Passenger_Count (numerical, discrete) vs Fare_Amount (numerical, continuous)
- **What it asks:** Does carrying more passengers correlate with a higher fare?
- **Recommended plot:** Scatter plot
- **Why:** Checks the relationship between two numeric variables; shows the raw point cloud rather than a summary that could hide structure.
- **What it should reveal:** Whether the cloud trends upward as passenger count rises, or is flat/random. Expect "vertical stripe" clustering since Passenger_Count is low-cardinality discrete (likely 1–6 values).
- **Pitfalls:** Overplotting due to discreteness of x-axis; may need slight jitter or alpha transparency to read density.
- **What NOT to conclude:** Causation — NYC fares are normally metered by distance/time, not headcount, so any correlation is likely incidental (via trip type), not a direct passenger-count effect.
- **Modeling relevance:** If no real relationship exists, Passenger_Count may add little predictive value; if variance is large at every passenger count, that signals general target noise.

### Q2 — Does car condition influence the fare amount?
- **Variables:** Car_Condition (categorical: Good/Bad) vs Fare_Amount (numerical)
- **Recommended plot:** Boxplot (preferred over barplot for this comparison)
- **Why:** Boxplot shows the full distribution (median, spread, outliers) per group, not just the mean — more informative for judging whether groups genuinely differ.
- **What to look for:** Overlap between the two boxes; shift in median; concentration of outliers in one group; sample size per group.
- **What NOT to conclude:** Causation — car condition doesn't mechanically set a metered fare, so any difference likely reflects a confound (trip type, driver segment).
- **Modeling relevance:** Even without a causal story, a genuine association means Car_Condition could still add predictive value.

### Q3 — At what hour of the day are fares typically highest?
- **Variables:** Hour (ordered discrete, 0–23) vs Fare_Amount (numerical), **aggregated** (mean and/or median per hour)
- **Recommended plot:** Line plot on the aggregated series
- **Why:** Hour is ordered/sequential — a line plot shows trend across that order clearly, unlike scatter/bar.
- **Data prep required:** Must aggregate first — `groupby('Hour')['Fare_Amount'].mean()` (and consider median too) — never plot raw per-ride points on a line chart.
- **CRITICAL — must stay distinct from Q6:** Q3 is about fare *level* per hour, not trip *volume*. An hour can have very few, expensive rides and look like a "high fare hour" while having very low demand. Do not conflate.

### Q4 — Does traffic condition affect the trip fare?
- **Variables:** Traffic_Conditions (Normal/Heavy) vs Fare_Amount
- **Recommended plot:** Boxplot (primary) / Barplot (secondary, for a quick mean comparison)
- **What to compare:** Mean, median, spread, outliers, and — critically — **sample size per group** (if the split is very imbalanced, e.g., 95%/5%, that limits confidence in the "Heavy" conclusion).

### Q5 — Is trip distance the strongest predictor of the fare amount?
- **Variables:** Distance vs Fare_Amount
- **Recommended plot:** Scatter plot first, but **this question cannot be answered from a scatter plot alone.**
- **Required extended analysis:**
  1. Scatter Distance vs Fare — assess shape (linear? curved? noisy?)
  2. Compute Pearson correlation between Distance and Fare_Amount
  3. Compute correlation of *every other relevant numeric feature* against Fare_Amount for comparison
  4. Build a full correlation heatmap
  5. Explicitly state correlation's limitations (linear-only, sensitive to outliers, no causal claim)
  6. **Explicitly distinguish EDA-level association from ML-stage feature importance** — "strongest predictor" is properly a modeling-stage claim (could involve nonlinear/interaction effects a correlation coefficient can't capture); EDA can only speak to linear association observed here.
- **Watch for:** A cluster of near-zero-distance-but-nonzero(-large)-fare points — possible signature of flat-rate airport fares, worth cross-referencing with Q8.

### Q6 — At what time of day are ride requests most frequent?
- **Variables:** Hour vs ride count (a frequency count, NOT Fare_Amount)
- **Recommended plot:** Count plot / bar chart
- **Why:** Directly counts occurrences per hour — exactly what count/bar plots are for.
- **CRITICAL — must stay distinct from Q3:** This is a demand question, not a price question. High-demand hours (e.g., rush hour) might have *lower* average fares if trips are short/congested — do not assume high demand implies high fares.

### Q7 — Does weather condition influence the average trip distance?
- **Variables:** Weather (Sunny/Windy/Winter — 3 categories) vs Distance
- **Recommended plot:** Boxplot / Barplot
- **What to compare:** Mean/median distance per weather type, spread, and sample size per category (verify "Winter" isn't a tiny sliver before drawing conclusions from it).

### Q8 — Are rides that start closer to airports generally more expensive?
- **Variables:** JFK_Dist, EWR_Dist, LGA_Dist (each numerical) vs Fare_Amount
- **Recommended plot:** Scatter plot per airport-distance variable (3 separate plots, or small multiples)
- **"Closer" means:** A *smaller* distance value — do not invert the reading of the axis.
- **Why separate per airport:** Each NYC airport (especially JFK) commonly has its own flat-fare policy and distinct geography; combining the three distance variables into one blurs three potentially different patterns.
- **What to look for:** A downward trend as distance→0, and/or a cluster/floor pattern near zero distance suggesting a flat fare regardless of exact proximity.

---

## 8. VISUALIZATION DECISION FRAMEWORK

| Plot | Best use | Why useful | What it reveals | Limitation | Example in this project |
|---|---|---|---|---|---|
| Histogram | Distribution of 1 numeric variable | Shows shape/spread at a glance | Skewness, modality, spread | Sensitive to bin width | Fare_Amount distribution (Target Analysis) |
| Boxplot | Distribution + outliers of a numeric variable (optionally split by group) | Shows median, quartiles, outliers together | Central tendency + spread + outliers per group | Hides sample size | Q2, Q4, Q7; Distance/Fare outlier checks |
| Bar Chart | Compare an aggregated value (usually mean) across categories | Simple, quick group comparison | Group-level average differences | Hides within-group variance | Secondary view for Q2/Q4/Q7 |
| Line Chart | Trends over ordered values (typically time) | Shows trend/trajectory clearly | Peaks, troughs, trend direction | Only valid for ordered/time data | Q3 (fare by hour), monthly/weekly trend additional questions |
| Scatter Plot | Relationship between 2 numeric variables | Shows raw relationship shape | Linearity, clusters, outliers, noise | Slow/overplotted on very large datasets | Q1, Q5, Q8 |
| Correlation Heatmap | Correlations across all numeric features at once | Efficient overview of many pairwise relationships | Which features move together with Fare_Amount | Captures linear relationships only | Q5 extended analysis, multicollinearity check |
| Pivot Heatmap | Two-dimension patterns (e.g., Hour × Day) after aggregation | Reveals interaction patterns | Where combined conditions produce highs/lows | Needs aggregation done first | Additional question: Hour × Day demand |
| Count Plot | Frequency of categories | Simple, direct frequency comparison | Category imbalance, peak/low periods | Says nothing about fare or any other variable | Q6, category imbalance checks |
| KDE Plot | Smooth distribution comparison across groups | Easier to compare overlapping distributions than histograms | Distribution shape differences between groups | Harder to read exact values than histogram | Optional alternative for Q2/Q4/Q7 |
| Geo Scatter | Spatial distribution of pickup/dropoff points | Shows geographic concentration | Where rides cluster in NYC | Requires validated coordinates first; garbage-in-garbage-out | Optional geographic analysis (Phase 7), only if coordinates validate |
| Pair Plot | All pairwise relationships at once | Fast overview across several variables | Broad relationship patterns | Slow/cluttered beyond ~5–6 columns | Not planned as a primary tool given the column count — use selectively if at all |

**Core rule to carry forward:** plots are selected because they match what the question is actually asking, never by default habit.

---

## 9. DATA QUALITY AUDIT PLAN

| Check | Why it matters |
|---|---|
| Missing values (count + %) | Determines cleaning strategy per column; some missingness may be acceptable, some may block analysis |
| Duplicate rows | Inflates counts/statistics if unaddressed |
| Duplicate `Key` values | `Key` is meant to be a unique trip ID — duplicates likely indicate duplicate/erroneous rows |
| Fare_Amount ≤ 0 | A ride cannot have zero or negative cost — likely data errors |
| Negative Distance | Physically impossible — data error |
| Passenger_Count ≤ 0 | Physically impossible for an actual ride |
| Hour outside 0–23 | Invalid — violates stated valid range |
| Day outside 0–6 | Invalid — violates stated valid range |
| Month outside 1–12 | Invalid — violates stated valid range |
| Invalid/implausible Week values | Should correspond to a real ISO week number |
| Bearing outside 0–360 | Invalid — violates stated valid range for a direction angle |
| Invalid geographic coordinates | Longitude/latitude far outside NYC bounds invalidates any geo-based conclusion |
| Suspicious/extreme Distance values | Could be data errors or legitimately long trips — must investigate, not assume |
| Extreme Fare_Amount values | Could be data errors, premium/long trips, or fraud — must investigate |
| Pickup_Datetime parse validity | Malformed timestamps break all derived temporal analysis |
| Hour consistency vs Pickup_Datetime | Hour should match what's computable directly from the timestamp |
| Day consistency vs Pickup_Datetime | Same logic |
| Month consistency vs Pickup_Datetime | Same logic |
| Week consistency vs Pickup_Datetime | Same logic |

For every issue found, report: **Problem → Number of affected rows → Percentage of dataset → Possible explanation → Decision → Reason.**

---

## 10. CLEANING DECISION FRAMEWORK

**Do NOT** simply say "delete bad rows." Instead, for every issue, walk through:

```
Problem
  → Number of affected rows
  → Percentage of dataset
  → Possible explanation
  → Decision (Keep / Remove / Correct / Transform / Investigate / Document as limitation)
  → Reason
```

**Why unexplained deletion is unacceptable:** Silent deletion can introduce bias (e.g., dropping all high-fare rows narrows the fare distribution artificially) and makes the analysis non-reproducible/non-defensible — nobody, including the analyst later, can judge whether the cleaning was reasonable.

**Why outliers should not be auto-deleted:** An extreme value could be a genuine rare but valid trip (e.g., an unusually long ride) rather than an error. Removing it without investigation destroys real signal and could bias Task 2 modeling. Outliers must be classified as: data error / valid rare observation / suspicious-needs-more-info before any action is taken.

**Data version discipline:**
- `df_raw` — preserved, untouched original data (loaded fresh, never overwritten)
- `df_clean` — created only with justified, documented, logged changes

This lets every downstream number be traced back to a specific, explainable decision.

---

## 11. PLOT INTERPRETATION FRAMEWORK

For **every** plot produced in the notebook, apply this framework (write the answers, don't just show the chart):

1. What do I literally observe?
2. Is the relationship/pattern strong or weak?
3. Is the pattern consistent across the range, or does it break down somewhere?
4. Are there outliers, and do they look like errors or genuine rare cases?
5. How large is each group being compared (sample size)?
6. Could sample size be driving the apparent pattern?
7. Is this correlation or causation? (Default assumption: correlation, unless there's a mechanical/metered reason to think otherwise — e.g., distance mechanically drives a metered fare.)
8. What does this mean for the dataset generally (cleaning implication? modeling opportunity?)?
9. Does it matter for Task 2 (would it change a feature-engineering or modeling decision)?

**How to use this in the final notebook:** each required/additional question's "Interpretation" subsection should visibly work through the relevant subset of these 9 questions — not just describe the picture.

---

## 12. OBSERVATION vs INSIGHT vs CONCLUSION vs PREDICTION vs CAUSATION

- **Observation** — a literal description of what's on the chart. *Example: "The boxplot for 'Heavy' traffic sits higher than 'Normal.'"*
- **Insight** — an observation connected to why it matters. *Example: "Heavy-traffic trips show a higher median fare, consistent with a time-based fare component — slower trips under traffic may cost more if time factors into the fare model."*
- **Conclusion** — a stronger claim you're willing to stand behind, still scoped to *this dataset*. *Example: "Traffic condition is associated with fare differences in this dataset and is worth retaining as a feature."*
- **Prediction** — an inferential claim about future/unseen data; belongs to Task 2, not Task 1. *Example: "A model will likely find Traffic_Conditions useful for predicting fare"* — may be stated as a modeling *recommendation*, never tested here.
- **Causation** — a claim that one variable directly produces changes in another; almost never justified by EDA alone (no controlled conditions / causal inference method applied here).

**Rule for the next assistant:** never use causal language when the EDA in front of us only supports association. When in doubt, say "associated with," not "causes" or "leads to."

---

## 13. ADDITIONAL QUESTIONS

**What makes a good additional question:**
- Answerable with columns that actually exist in the dataset
- Ties back to something useful for cleaning, feature engineering, or Task 2 modeling
- Specific enough that one clear plot can answer it

**What makes a useless question:**
- Not answerable with available columns
- Purely decorative — no modeling or data-quality implication
- Duplicates a required question with no new angle

**Why required:** explicit PDF checklist item — "Benefit from above questions to analyze and create new Questions for understand the data."

**How many planned:** at least 8 (per PDF wording "at least 8" established in our prior discussion) — a reasonable, non-padded set, not more than needed to add real value.

**Candidate areas discussed (none answered yet — planning only):**
- Fare by day of week
- Weekday vs weekend fare differences
- Hour × Day demand (pivot heatmap)
- Monthly fare trends
- Weekly trends
- Traffic × Hour interaction
- Weather × Demand
- Passenger_Count vs Distance
- Fare per kilometer (derived ratio) across categories
- Airport proximity relationships (deeper than Q8's single-variable view)
- Distance-variable redundancy (JFK_Dist/EWR_Dist/LGA_Dist/SOL_Dist/Distance correlations)
- Category imbalance (Weather/Traffic_Conditions/Car_Condition distribution)
- Geographic pickup concentration (if coordinates validate)

**Clear distinction to maintain:** everything in the list above is a *candidate*, not an answered question. No results exist for any of them yet.

---

## 14. CORRELATION ANALYSIS

- **Why it matters for Q5:** a single scatterplot can't establish "strongest predictor" — correlation coefficients let us *compare* Distance's linear association with Fare_Amount against other numeric features' associations.
- **What Pearson correlation captures:** linear association strength and direction between two numeric variables (range -1 to 1).
- **What it does NOT capture:** nonlinear relationships (e.g., a strong curved relationship can show a weak Pearson correlation), interaction effects between features, or causal direction/strength.
- **Why correlation ≠ causation:** an association can arise from a third confounding variable, from reverse causation, or from pure coincidence in this particular dataset.
- **Why correlation ≠ model feature importance:** feature importance is computed *within a fitted model* and can capture nonlinear/interaction effects that raw correlation misses entirely — this is explicitly a Task 2 concept, not something EDA alone can determine.
- **Multicollinearity concern:** JFK_Dist, EWR_Dist, LGA_Dist, SOL_Dist, and Distance are likely correlated with each other (all measure some form of geographic distance) — this matters especially for linear-type models in Task 2, less so for tree-based models, but should be flagged either way.

---

## 15. OUTLIER ANALYSIS

- **Where to look:** Fare_Amount, Distance, Passenger_Count, JFK_Dist, EWR_Dist, LGA_Dist, SOL_Dist.
- **Method:** IQR-based flagging (Q1/Q3, IQR = Q3−Q1, flag points beyond Q1−1.5×IQR or Q3+1.5×IQR), supported visually by boxplots, and cross-checked against percentiles and scatterplots.
- **Core discipline:** never auto-delete. For each outlier cluster, determine: data error vs. valid rare observation vs. suspicious/needs more info — and document the decision (see Section 10 framework).
- **Effect on Task 2:** outlier decisions here directly shape whether Task 2 needs robust scalers, log transforms, capping/winsorizing, or explicit outlier-removal steps — this section's findings should feed directly into the Modeling Recommendations (Section 21).

---

## 16. TEMPORAL ANALYSIS

**Columns involved:** Hour, Day, Month, Week, Pickup_Datetime.

**Planned analyses:**
- Hour vs Fare (Q3) — fare *level* per hour, aggregated (mean/median)
- Hour vs Ride Count (Q6) — demand/volume per hour
- Hour × Day pivot heatmap (additional question) — interaction of time-of-day and day-of-week on demand and/or fare
- Monthly patterns (additional question) — seasonal fare/demand trends
- Weekly patterns (additional question) — trend across weeks of the year
- Pickup_Datetime consistency checks (data quality) — does the timestamp actually match the derived Hour/Day/Month/Week columns row by row?

**Non-negotiable distinction to preserve throughout:** **Q3 = price/fare pattern. Q6 = demand pattern.** These must never be merged or conflated in any write-up.

---

## 17. GEOGRAPHIC ANALYSIS

**Columns involved:** Pickup_Longitude/Latitude, Dropoff_Longitude/Latitude, JFK_Dist, EWR_Dist, LGA_Dist, SOL_Dist.

**Status:** This is **not explicitly required** by the PDF checklist — it is a **Recommended Enhancement**, valuable because the dataset has rich geographic columns and Q8 touches on airport proximity, but it must be labeled as optional/recommended in the final notebook, not presented as an official requirement.

**Plan if pursued:**
- First validate coordinates fall within a plausible NYC bounding box (data quality step, Section 9) — do not assume validity.
- If valid: pickup scatter, dropoff scatter, geographic density/concentration plots.
- Because the dataset is large, avoid plotting every point if performance suffers — use a representative sample for the visualization while still reporting summary statistics computed on the full cleaned data.

---

## 18. TARGET ANALYSIS

**Why Fare_Amount deserves dedicated analysis:** it's the prediction target for Task 2 — understanding its distribution, spread, and outliers directly shapes modeling decisions (e.g., whether a transform is needed).

**Planned checks (no fabricated numbers — all must come from real computation in Phase 1+):**
- Distribution shape (histogram)
- Boxplot (spread + outliers)
- Count, mean, median, standard deviation, min, max, Q1, Q3, IQR
- Skewness (to assess transform-worthiness)
- Selected percentiles
- Explicit assessment: is the target skewed? Do extreme values exist? Would a log or other transform be worth testing in Task 2? (State this as a *recommendation for Task 2*, never actually apply it in Task 1 unless explicitly asked.)

---

## 19. NOTEBOOK ARCHITECTURE (final target structure)

```
1.  Title
2.  Objective
3.  Dataset Description
4.  Imports
5.  Data Loading (with documented memory strategy)
6.  Initial Inspection
7.  Data Quality Audit
8.  Cleaning Strategy (df_raw preserved, df_clean created + logged)
9.  Cleaned Dataset Validation
10. Baseline EDA
11. Target Analysis (Fare_Amount deep dive)
12. Q1  — full structured format
13. Q2  — full structured format
14. Q3  — full structured format
15. Q4  — full structured format
16. Q5  — full structured format
17. Q6  — full structured format
18. Q7  — full structured format
19. Q8  — full structured format
20. Additional Questions (≥8) — full structured format
21. Correlation Analysis
22. Outlier Analysis
23. Temporal Analysis
24. Geographic Analysis (only if coordinates validate — labeled as Recommended Enhancement)
25. Key Insights
26. Modeling Implications
27. Final Conclusion
```

**Structured format used for every Q1–Q8 and additional question:**
```
Question
→ Variables
→ Variable Types
→ Plot choice
→ Why this plot?
→ Data Preparation
→ Code
→ Result / Visualization
→ Numerical Evidence
→ Interpretation
→ Limitations
→ Modeling Relevance
```

Markdown cells carry all explanations; code cells stay clean and commented; the notebook must run top-to-bottom without errors.

---

## 20. PRESENTATION ARCHITECTURE

- **Length:** roughly 10–14 slides — enough to cover objective, dataset overview, data quality, target distribution, the strongest 4–6 required/additional findings, correlation, and modeling recommendations, without becoming a slide-per-plot dump.
- **Per-slide structure (matches PDF requirement exactly):** Question → Plot → Insight.
- **Selection rule:** only include a plot if it supports a specific, stated insight — not "here's a chart." Everything else stays in the notebook only.
- **Text:** minimal — a short question/insight statement plus the plot. The notebook already carries the full prose.
- **Suggested slide flow:** Title → Objective → Dataset overview → Data quality summary → Target distribution → 4–6 key findings (mix of required + additional questions) → Correlation/relationships → Geographic/time patterns (if used) → Modeling recommendations → Final takeaways.
- **Explicit labeling requirement:** state clearly that slide count/structure is a *recommendation*, not something verbatim mandated by the PDF beyond "question, plot, insight" and "don't include every plot."

---

## 21. TASK 2 CONNECTION

Task 1 findings should inform (not pre-empt) the following Task 2 decisions:

- **Useful features:** columns showing real association with Fare_Amount from Q1–Q8 and correlation analysis (Distance and airport proximity are strong domain-logic candidates, to be confirmed/challenged by actual EDA).
- **Irrelevant identifiers:** User_ID, User_Name, Driver_Name, Key — excluded from modeling; using them risks the model memorizing IDs instead of learning general patterns.
- **Categorical encoding:** decisions for Car_Condition, Weather, Traffic_Conditions (one-hot vs ordinal, depending on modeling approach used later).
- **Missing-data strategy:** informed by how much is missing and where (from Phase 2 audit).
- **Outlier strategy:** whether extreme Fare_Amount/Distance values are errors to remove or rare-but-real cases to keep (possibly with a robust transform) — from Section 15.
- **Transformations:** if Fare_Amount is heavily right-skewed, a log transform is a candidate to *test* in Task 2 (not applied here).
- **Multicollinearity:** the distance-related columns (JFK_Dist/EWR_Dist/LGA_Dist/SOL_Dist/Distance) are likely correlated with each other — matters more for linear models, less for tree-based ones, but should be flagged regardless.
- **Leakage risks:** anything derived from Fare_Amount itself, or not knowable at prediction time, must be excluded.
- **Temporal features:** Hour/Day/Month/Week may benefit from cyclical encoding (e.g., sin/cos) in Task 2 — a recommendation, not something implemented in Task 1.
- **Distance-related features:** likely strong predictors — to be confirmed by actual correlation numbers in Phase 7.
- **Promising relationships:** flagged from Q1–Q8/additional-question findings — to be filled in with real evidence, never invented.

**Rule:** Task 1 informs Task 2. No model is built, trained, or evaluated in this task.

---

## 22. INTERNSHIP-LEVEL QUALITY STANDARD

- Every plot has a stated reason for being the chosen type
- Every plot has written interpretation — never left to "speak for itself"
- Cleaning decisions are documented with real numbers (never "removed some outliers")
- No plot exists just because it *could* — each answers a specific stated question
- Code has short comments explaining *intent*, not just mechanically what the line does
- The notebook runs top-to-bottom without errors, in order
- Insights are traceable to specific numbers/plots actually present in the notebook
- No fabricated results, no unsupported conclusions
- Presentation tells a coherent story using only the strongest findings
- Code is clean and reproducible

---

## 23. COMMON MISTAKES TO AVOID

- Picking a chart type reflexively instead of matching it to the question
- Showing a plot with zero written interpretation
- Confusing "highest average fare" (Q3, price) with "highest ride count" (Q6, demand)
- Treating correlation as causation anywhere in the notebook
- Deleting outliers with no documented justification
- Using User_ID / Driver_Name / User_Name / Key as model features
- Any form of target leakage (using something derived from Fare_Amount to help "predict" Fare_Amount)
- Overloading the notebook with redundant plots
- Putting every plot into the presentation instead of the strongest few
- Writing a conclusion not backed by a specific number or chart already in the notebook
- Fabricating or guessing at results instead of computing them from real data
- **Treating `sample_50000.csv` as the full dataset for final conclusions** — it is explicitly for initial inspection only; all final numbers/conclusions must come from the actual full dataset (`final_internship_data.xlsx`, ~139MB)
- Reversing the "closer to airport" logic in Q8 (smaller distance = closer, not the reverse)
- Using a barplot alone (mean only) where a boxplot (full distribution) is needed to see spread/outliers, for Q2/Q4/Q7-type comparisons

---

## 24. PRACTICAL ROADMAP

| Step | Action | Tool | Expected output | Decision before moving forward |
|---|---|---|---|---|
| 1 | Inspect `sample_50000.csv` structure | Pandas | Column list, dtypes, shape | Do actual columns match the PDF's description? Any discrepancies to report? |
| 2 | Load and inspect the full `.xlsx`, checking memory footprint | Pandas + OpenPyXL | Row count, memory usage, file size | Can it load fully in memory, or does it need chunking/dtype optimization/Dask? |
| 3 | Data quality audit on the full (or best-available) data | Pandas | Missing %, duplicates, invalid-value counts, per Section 9 | Which issues are real vs. negligible? |
| 4 | Design + apply cleaning, preserving `df_raw`, creating `df_clean` | Pandas | Cleaning log with Problem→Count→%→Reason→Decision | Why each decision was made — documented |
| 5 | Analyze the target (Fare_Amount) | Pandas/Seaborn | Histogram, boxplot, full stats | Is it skewed? Any transform candidates flagged for Task 2? |
| 6 | Answer Q1–Q8 in full structured format | Pandas/Matplotlib/Seaborn | 8 plots + interpretations | Each question's specific pitfall (Section 7) respected |
| 7 | Design and answer ≥8 additional questions | Same | Plots + interpretations | Why each question matters for modeling |
| 8 | Correlation + outlier + temporal + (optional) geographic deep-dive | Pandas/Seaborn | Heatmap, plots, stats | What multicollinearity/outlier patterns actually exist |
| 9 | Write Key Insights + Modeling Recommendations | Markdown | Bullet summary | Everything ties back to evidence already shown |
| 10 | Assemble final notebook structure | Jupyter | Complete `.ipynb` | Runs cleanly start to finish |
| 11 | Build the presentation (question → plot → insight) | Slide tool of choice | ~10–14 slides | Only the strongest findings included |
| 12 | Run the Final Quality Check (Section 22/27) | Manual review | Checklist ticked off | Nothing fabricated, everything documented |

---

## 25. CURRENT PROJECT STATUS

**Completed:**
- Task1.pdf reviewed in full detail
- Official requirements extracted and separated from recommended enhancements
- Full EDA strategy, phased workflow, and quality standards established (this handoff)
- Dataset identified: `final_internship_data.xlsx` (~139 MB)
- Full dataset local location confirmed: `C:\Users\noura\OneDrive\Desktop\ZCellula-ML\Week 1\final_internship_data.xlsx`
- Initial 50,000-row sample created: `sample_50000.csv`
- Sample local location confirmed: `C:\Users\noura\OneDrive\Desktop\ZCellula-ML\Week 1\sample_50000.csv`
- Sample dimensions stated by user: 50,000 rows × 26 columns (**not yet independently verified by actually reading the file — this is Phase 1's first job**)
- Python environment confirmed: Python 3.13.15; pandas 3.0.6, openpyxl 3.1.5, numpy, matplotlib, seaborn all available
- Full Q1–Q8 reasoning/plan developed (Section 7)
- Full notebook and presentation architecture agreed (Sections 19–20)
- Full data-quality, cleaning, outlier, correlation, and interpretation frameworks agreed (Sections 9–15)

**NOT completed (must not be assumed or fabricated):**
- Actual inspection of `sample_50000.csv` (columns/dtypes/missing/etc. not yet run)
- Actual inspection of the full `.xlsx` (exact row count, memory requirement, loading strategy not yet determined)
- Data quality audit (no real counts/percentages yet)
- Any cleaning applied
- Any Q1–Q8 results
- Any additional-question results
- Correlation matrix / heatmap
- Outlier counts
- Any final insights
- Notebook file
- Presentation file

**Current phase: PHASE 1 — INITIAL DATASET INSPECTION** (not yet started)

---

## 26. EXACT NEXT STEP

Start with **Phase 1** only.

1. Inspect `sample_50000.csv` and report:
   - Shape
   - Exact columns (compare against Section 3's expected list — report any mismatch, don't silently reconcile)
   - Data types
   - Missing values + percentages
   - Unique value counts per column
   - Which columns are numerical
   - Which columns are categorical
   - Which columns are datetime
   - Which columns are identifiers
   - Confirm the target
   - Descriptive statistics (numeric summary)
   - Any obvious quality issues spotted at this stage
   - Whether actual columns match Task1.pdf's documentation

2. Then inspect the full `final_internship_data.xlsx` — but **only enough** to determine:
   - Exact row count
   - Exact column count
   - File size
   - Approximate memory requirement if loaded fully
   - Whether full Pandas in-memory loading is practical
   - Whether dtype optimization is needed
   - Whether chunked/partial processing is needed

**Do NOT, in this next step:**
- Clean the data
- Answer Q1–Q8
- Create the final notebook
- Create the presentation
- Fabricate any results

**STOP after Phase 1** and report findings before proceeding to Phase 2 (Full Data Quality Audit).

---

## 27. HANDOFF RULES FOR THE NEXT AI

- Read this entire handoff before acting.
- Treat Task1.pdf as the official specification (Section 2 preserves its content verbatim/paraphrased).
- Treat this handoff as the accumulated project context — do not restart the project or ask the user to re-explain it.
- Do not replace the established 12-phase workflow with a different generic one without a stated reason.
- Do not invent results — every number must come from actually running code against the real data.
- Do not treat `sample_50000.csv` results as final-dataset results; the sample is for initial inspection only.
- Do not jump ahead in the phase sequence — later phases depend on earlier validation/cleaning decisions (see Section 31 dependency map, effectively embedded in the phase order above).
- Work phase-by-phase: explain what's about to happen and why, execute, show actual results, interpret them, record decisions, state what remains, then move on.
- Ask for confirmation after each major phase when appropriate (this was the user's established working style).
- Use actual computed values for all numerical claims — never estimate/guess a statistic and present it as real.
- Clearly distinguish official requirements (Section 2) from recommended enhancements (labeled throughout this document) in every response to the user.
- Preserve reproducibility (clean, commented, top-to-bottom-runnable code).
- Prioritize insight over code volume, per the PDF's own closing statement.

---

## 28. THINGS THAT MUST REMAIN UNKNOWN (do not guess these — verify in later phases)

- Full exact row count of `final_internship_data.xlsx`
- Full dataset memory usage
- Actual missing-value percentages (any column)
- Actual duplicate row count
- Actual duplicate `Key` count
- Actual count of invalid `Fare_Amount` values (≤0 or extreme)
- Actual count of invalid `Distance` values (negative or extreme)
- Actual outlier counts (Fare, Distance, Passenger_Count, airport distances)
- Actual correlation coefficients (Distance vs Fare, or any other pair)
- Actual Q1–Q8 results/plots/numbers
- Actual additional-question results
- Actual final insights or modeling recommendations with specific numbers attached
- Whether the actual dataset's columns/categories perfectly match the PDF's documented list

**These must be calculated from the real dataset in Phase 1 onward. Do not guess or infer them from general Uber-fare domain knowledge.**

---

## 29. FINAL HANDOFF SUMMARY

```
PROJECT:            Uber Fare Prediction — Cellula Technologies ML Internship
TASK:                Task 1 — Exploratory Data Analysis (EDA)
TARGET:              Fare_Amount
DATASET:             final_internship_data.xlsx (~139 MB, full dataset)
                     sample_50000.csv (50,000 rows × 26 cols, inspection-only sample)
CURRENT PHASE:       Phase 1 — Initial Dataset Inspection (not yet started)
COMPLETED:           Phase 0 (full task/requirement understanding); complete project
                     planning across all 12 phases; environment confirmed ready
NEXT STEP:           Inspect sample_50000.csv fully, then size/profile the full .xlsx
                     to determine loading strategy — STOP after Phase 1 for confirmation
FINAL DELIVERABLES:  (1) A complete, reproducible Jupyter notebook covering data audit,
                     cleaning, target analysis, Q1–Q8, ≥8 additional questions,
                     correlation, outliers, temporal (and optional geographic) analysis,
                     insights, and modeling recommendations.
                     (2) A ~10–14 slide presentation following Question → Plot → Insight.
MAIN RULES:          No fabricated results. No causal language for correlational
                     findings. Sample ≠ full dataset. Document every cleaning decision
                     (never silently delete). Distinguish official PDF requirements
                     from recommended enhancements at all times. Q3 (fare level) and
                     Q6 (ride demand) must never be conflated. Q5 requires comparative
                     correlation analysis, not a single scatterplot, to address
                     "strongest predictor." Work phase-by-phase with user confirmation
                     between major phases.
```

---

## 30. COMPLETE EXECUTION PLAN — ONE-PAGE ROADMAP (Project Control Panel)

```
CURRENT STATUS → Phase 0 complete, Phase 1 not started
       ↓
PHASE 1  → Inspect sample_50000.csv + size/profile the full .xlsx, to know what we're actually working with
       ↓
PHASE 2  → Run the full data-quality audit, to know exactly what's wrong and how much
       ↓
PHASE 3  → Apply documented, justified cleaning (df_raw preserved, df_clean created), to get a trustworthy analysis dataset
       ↓
PHASE 4  → Run baseline EDA (target, distributions, first-pass correlations), to build context before the required questions
       ↓
PHASE 5  → Answer Q1–Q8 in full structured format, to satisfy the core mandatory deliverable
       ↓
PHASE 6  → Create and answer ≥8 additional questions, to satisfy the "create new questions" requirement and deepen understanding
       ↓
PHASE 7  → Deep-dive: full correlation matrix, outliers, temporal/geographic patterns, multicollinearity, to resolve open analytical questions (especially Q5's "strongest predictor")
       ↓
PHASE 8  → Convert findings into evidence-based Observation → Insight → Modeling-implication statements, to produce defensible conclusions
       ↓
PHASE 9  → Document Task 2 modeling preparation (features, encoding, leakage, transforms) without building any model, to bridge to the next task
       ↓
PHASE 10 → Assemble the final notebook per the agreed architecture, to produce the primary deliverable
       ↓
PHASE 11 → Build the ~10–14 slide presentation (Question → Plot → Insight), to produce the second deliverable
       ↓
PHASE 12 → Run the Final QA checklist across data/EDA/code/visualization/insights/presentation, to confirm submission readiness
       ↓
FINAL SUBMISSION → Notebook + Presentation, both internship-level quality, both fully evidence-based
```

---

**END OF HANDOFF.** The next assistant should now proceed directly to Phase 1 as described in Section 26, working incrementally and pausing for user confirmation after each major phase, per Section 27.
