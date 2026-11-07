# Customer churn: Weeks 5, 6 and 7 preprocessing lab

## What you receive
- customer_churn_10000.csv: 10,000 independent synthetic customers, 40 columns (38 candidate predictors, customer_id and churn).
- lab.py: complete dataset generator, feature builder, pipelines and measured comparison.
- baseline.py: run the baseline alone.
- preprocessed.py: run the preprocessed model alone.
- teaching_notebook.ipynb: guided classroom notebook.
- data_dictionary.csv: every input column explained.
- results_all.csv: actual measured metrics and timings.
- environment.json and requirements.txt: reproducibility information.

## Run locally
Python 3.10+ recommended. Extract this folder, open a terminal in it:

```bash
pip install -r requirements.txt
python lab.py
```

Or run the two stages independently:

```bash
python baseline.py
python preprocessed.py
```

For consistent single-thread timing on Linux/macOS:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python lab.py
```

For Google Colab: upload lab.py and the CSV, then run `!python lab.py`.
The notebook imports lab.py, so keep them together. Delete the CSV and run
lab.py to regenerate exactly the same synthetic data using seed 42.

## Scenario and target
Predict churn at a fixed snapshot (2026-11-01), using only customer history
available before that snapshot. churn=1 means leaves during the subsequent
hypothetical observation window. No actual people or financial records are used.
All predictors precede the outcome. Monthly income and monthly debt use the
same unspecified currency. Spend figures are totals/variation over the prior
90 days. Each customer occurs once, so no customer crosses the split.

This dataset deliberately makes churn depend on plausible nonlinear signals:
debt burden, a debt threshold, early-tenure interaction, returns, complaint
language, night activity, recency and spending variability. Labels are sampled
probabilistically, so perfect classification is neither expected nor intended.
The known generation formula is visible in generate_data(). This is a teaching
demonstration of model specification, not evidence of real-world uplift.
The features and thresholds are fixed domain hypotheses before evaluation;
there is no test-driven threshold search or model tuning.

## Mapping to the uploaded slides

| Week | Concepts in deck | Implementation |
|---|---|---|
| 5 | Time extraction | Purchase hour, weekday, weekend, night flag and recency |
| 5 | Text extraction | Text length, rule-based complaint flag and simple sentiment proxy |
| 5 | Aggregation and behavioural summaries | Existing 90-day totals, return ratio, average order value and spend variability |
| 5 | Categorical encoding | Baseline one-hot encodes whole strings/categories; the selected final model uses numeric signals |
| 6 | Ratios | Debt/income, returns/orders, spend/orders, spending standard deviation/spend |
| 6 | Interactions | Debt-to-income multiplied by new-customer flag |
| 6 | Flags | Tenure <=6 months and debt-to-income >=0.35 |
| 6 | Leakage avoidance | Stratified split before fitted transformations; imputer and scaler fit inside training pipeline |
| 7 | Irrelevant dimensions | Remove 20 independent noise columns and irrelevant demographic/device/preferences |
| 7 | Redundancy/multicollinearity | Remove income_copy and constant_field; retain one complaint representation |
| 7 | Stability and training cost | Report numeric-feature ablation, selected-feature model and timing |

The feature builder constructs more signals than the final model retains.
This is intentional: Week 7 asks which signals deserve to stay. Selection is
explicit domain selection, not automated target-based feature selection.
The simple sentiment proxy is tailored to this small synthetic phrase set;
it is not a general sentiment analysis model. Seasonal/month features are
not demonstrated because the observation period is too short. No raw event-log
aggregation is claimed: the CSV already contains per-customer 90-day summaries.
PCA and anomaly detectors are not required by these particular decks.

## Fair comparison
Same LogisticRegression(C=1, max_iter=2000, random_state=42), same 7,500/2,500
stratified split, same labels and fixed 0.5 decision threshold in every run.
A literal zero-preprocessing logistic regression cannot consume missing values,
timestamps and text. The baseline therefore uses only necessary type conversion,
median imputation, numeric scaling and one-hot encoding. It uses no semantic
feature extraction, ratios, interactions or domain feature selection, and drops
customer_id in every model. Calling it 'minimal preprocessing' is accurate.

No preprocessing is fitted on the full dataset. FeatureBuilder applies only
fixed row-wise rules. All learned medians, scaling statistics and category
vocabularies are fitted exclusively on training rows inside the pipeline.
The test set is used only for reporting, never to choose features or tune a model.

The intermediate `engineered_all` run retains all numeric original and engineered
features (46), including numeric noise/duplicates. It omits categorical columns
and raw text/date after extracting signals. It is a numeric-feature ablation,
not a controlled one-change experiment against the categorical baseline.
Comparing it with the selected final run isolates the numeric selection step.

## Actual results from the supplied run

| Run | Model inputs | Accuracy | F1 | ROC AUC | Fit pipeline seconds | Overall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Minimal-preprocessing baseline | 56 | 77.72% | 0.8291 | 0.8438 | 0.0592 | 0.0887 |
| Engineered numeric features, including noise | 46 | 83.20% | 0.8678 | 0.9002 | 0.0901 | 0.1778 |
| Extraction + construction + domain selection | 9 | 83.00% | 0.8660 | 0.9005 | 0.0682 | 0.1538 |

Accuracy improves by 5.28 percentage points over baseline. Majority-class
accuracy is 61.84%. Test confusion matrices (rows actual 0/1, columns predicted
0/1): baseline [[592,362],[195,1351]], final [[702,252],[173,1373]].

Week 7 selection reduces dimensions and intermediate runtime, with a small
accuracy decrease of 0.20 percentage points versus keeping all numeric features.
Do not claim feature removal always improves accuracy. The final pipeline is
slower overall than the baseline because feature construction costs time.

Timings are medians of three repeated fits on the same split, not best-of-three
accuracy trials. `fit_seconds` includes fitted preprocessing and model training.
`predict_and_probability_seconds` includes both predict and predict_proba, and
therefore transforms the test data twice. `setup_seconds` includes building
pipeline objects and deterministic schema/type conversions; final pipelines
also inspect deterministic feature output to choose the schema. Total is the
sum of those three median components. CSV loading/generation and metric
calculation are excluded. Hardware, BLAS threads and versions change runtime;
these short measurements are illustrative, not a benchmark or guaranteed speedup.

## Classroom flow
1. Inspect shape, types, missingness and class balance. Identify tempting but irrelevant columns.
2. Run baseline.py. Discuss why whole feedback strings and an epoch timestamp lack useful structure.
3. Trace FeatureBuilder line by line and calculate the ratios for one customer.
4. Ask why new-customer debt risk requires an interaction and a threshold flag.
5. Run preprocessed.py. Compare accuracy, F1, AUC, confusion matrices and elapsed times.
6. Run lab.py for the intermediate ablation. Explain the dimensionality/cost tradeoff honestly.
7. Discuss why future events, target-derived fields, fitting on all rows and tuning against test accuracy leak information.
8. Optional: propose a different feature set and assess it by cross-validation on training rows only before using a fresh test set.
