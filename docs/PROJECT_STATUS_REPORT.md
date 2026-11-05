# Project Status and Team Handoff

**Reviewed:** 2026-11-05
**Repository branch:** `codex/project-handoff`
**Purpose:** Give the frontend and ML workstreams a shared, evidence-based starting point.

## Executive status

This repository contains a working batch data pipeline for US domestic flight records from October through December 2024. It uses Hadoop HDFS for storage, YARN to run Spark jobs, Spark SQL for holiday analysis, and Spark MLlib plus separate host-side experiments for flight-delay models. The cleaned dataset contains 1,781,482 flights; 1,765,279 eligible as-of feature rows are reported by the current OPS V2 state export.

The data validation, cleaning, holiday analysis, schedule-only ML work, UTC-correct T−120 operations features, validation/fold model comparisons, and required **FOURTH DESIGN EVALUATION** are complete and documented. XGBoost with UTC-correct operations features passed the validation and rolling-fold gate; the deep-learning candidate did not. The frozen XGBoost test PR-AUC was 0.41980 versus 0.40524 for the UTC-LR refit reference.

There is currently no frontend application, web server, prediction API, or UI component in the tracked source tree. The frontend teammate can build the presentation layer independently using the reports, plots, and explicitly labelled results in this repository. The ML evaluation is closed for this dataset; only work on an independent future test window can support additional model-selection claims.

## Repository map

| Area | Contents | State |
|---|---|---|
| `src/00_validate_data.py` | Schema, row, date, null, eligibility, and holiday-calendar checks | Implemented; actual-data validation is documented |
| `src/01_clean.py` | Reads raw BTS monthly CSVs from HDFS, normalizes and types fields, writes cleaned Parquet | Implemented; pipeline and actual-data results documented |
| `src/02_holiday_analysis.py` | Spark SQL summaries by holiday, date, airline, origin, and departure hour; delay causes are descriptive | Implemented |
| `src/03_ml.py` | Spark MLlib schedule/history models, controlled feature experiments, validation selection, frozen final evaluation; optional XGBoost | Implemented and exercised on actual data; OPS V2 has a separate UTC runner |
| `src/asof_ops*.py` | As-of operational feature construction and UTC ordering/state helpers | Implemented; UTC synthetic leakage checks are documented as passing |
| `src/tests/` | Synthetic local Spark checks for ML, temporal history, operations features, UTC behavior, and deep-learning model code | Test evidence is in `docs/TEST_RUN_LOG.md`; tests use synthetic data, not model-quality evidence |
| `analysis/` | Audits, bootstrap calculations, diagnostics, host-side model evaluation and OPS V2 orchestration | Research/evaluation tools; OPS V2 scripts are under `analysis/ops_v2/` |
| `docker/` | Docker Compose cluster, Hadoop/Spark configuration, and container startup | Three-container local cluster: master plus two workers |
| `docs/` | Data, ML, UTC operations, test, and handoff reports | Main evidence and known limitations are documented here |
| `screenshots/ml_actual/` | Existing report-ready plots for the earlier actual-data ML evaluation | Available for reuse in the frontend/report |

Generated data, model binaries, YARN output, and experiment results are intentionally outside Git or ignored. The GitHub checkout alone is not the runtime dataset; see the data staging instructions in `README.md`.

## Current ML evidence

### Existing schedule/history design

The chronological split is train 2024-10-01–11-24, validation 11-25–12-03, and test 12-04–12-31. The existing actual-data report records 523,983 eligible test flights and a 21.318% delayed-flight rate. It reports schedule/history LR test PR-AUC 0.2995 and RF 0.2825 in the earlier evaluation. The schedule-only models are useful baselines, but their scores are modest and the class prevalence rises across time windows.

### UTC-correct operations design

At T−120, the accepted feature set is variant (d), containing all 16 operations features ordered in UTC. Airport time zones were resolved for 339 of 340 observed airport codes; PBI has no metadata match and receives unavailable-time/zero-observation and training-prior behavior rather than a guessed offset. The UTC design passed the validation gate against schedule-only E6: PR-AUC 0.28752 versus 0.24220, a +0.04532 gain. It also beat origin-only features by +0.01469 on validation, while it did not beat the older local-clock variant. The report documents the limitations and synthetic cross-time-zone/DST leakage tests.

The third-design UTC test report is already present in `docs/ASOF_TZ_REPORT.md`: LR test PR-AUC 0.40508 and RF 0.37924. This is historical evidence from the third design evaluation, not the pending fourth evaluation.

### OPS V2 model comparison and frozen decision

The committed comparison in `docs/OPS_V2_REPORT.md` uses the accepted UTC feature set and validation plus four rolling-origin folds:

| Candidate | Validation PR-AUC | Status |
|---|---:|---|
| Frozen UTC LR | 0.28784 | Baseline |
| LR with explicit operations interactions | 0.28314 | Did not improve baseline |
| LightGBM | 0.29921 | Below XGBoost |
| CatBoost | 0.30118 | Below XGBoost |
| XGBoost | **0.30590** | Phase 1 gate passed; frozen candidate |
| Five-seed deep model ensemble | 0.30414 | Phase 2 gate failed on rolling folds; not eligible for test scoring |

XGBoost's validation gain over frozen LR was +0.01806 PR-AUC; paired date-block bootstrap 95% CI was [+0.01031, +0.02766], and mean rolling-fold gain was +0.03103. Its train-validation gap is material (0.08182), so overfitting and temporal instability remain concerns. The interaction LR did not explain the boosted-model gain. The XGBoost choice and test guard were frozen and committed before the fourth-design test scoring stage.

## Fourth evaluation: current state

The guarded fourth evaluation completed. Its frozen XGBoost model scored test PR-AUC 0.41980, ROC-AUC 0.70198, top-decile precision 0.52349, and top-decile lift 2.456. Compared with the UTC-LR refit on the same test rows, the paired 28-date bootstrap PR-AUC gain was +0.01455 (95% CI [+0.00689, +0.02273]). Holiday-slice PR-AUC was 0.48138 for XGBoost versus 0.47308 for UTC-LR during Christmas/New Year, and 0.33147 versus 0.30762 on regular days. Full metrics and interpretation are in `docs/OPS_V2_REPORT.md`; the machine-readable output is in the ignored local path `results/experiments/ml_redesign/ops_v2/fourth_design/test_results.json`.

The test window was opened and scored for the fourth time. The validation/fold freeze remains the model decision boundary; these test results do not authorize additional tuning. Do not use this test window to select another model. A new time window is needed for further generalization claims.

The export took about 30 minutes and reconciled 1,765,279 eligible rows with zero duplicate technical keys and no non-label nulls. An earlier attempt ran out of driver memory; the successful run used a 2 GB driver and 512 MB executors. The first export wrote the supposed local Parquet path to HDFS because the cluster default filesystem is HDFS. The data was copied to `C:\bda\ops_v2\test` for host scoring, and the exporter now uses an explicit `file://` URI for host output. The Docker cluster was stopped after scoring; its named HDFS volume remains intact.

## Frontend teammate handoff

The current tracked files contain no frontend assets or API framework: no `package.json`, web page, React/Vue application, Streamlit/Flask/FastAPI service, or existing UI routes were found. This is a backend batch project today, not an integrated interactive app.

The frontend teammate can start without waiting for the final ML run:

1. Build the UI as a separate frontend component or folder, and document its local start command and any runtime requirements.
2. Use the committed plots in `screenshots/ml_actual/` and the tables/reports under `docs/` for an initial results page. Label third-design results and OPS V2 validation results distinctly; do not present validation metrics as fourth-design test results.
3. For a project demo, make the current interface clearly historical/batch-oriented. A live prediction form should wait for an agreed input schema and a serving/inference contract; no such endpoint or packaged serving model is currently defined.
4. If the UI needs structured data immediately, create a small, checked-in demo fixture with provenance and a clear “sample / historical results” label. Do not commit raw flight data, generated model directories, or large result dumps.
5. Keep UI changes scoped so they can be merged with the ML branch later. Agree on field names and result JSON/CSV shape before wiring a prediction workflow to an eventual backend.

Suggested initial screens: project overview and pipeline, dataset/split summary, model comparison and baselines, top-risk lift, holiday slices, limitations, and a demo/results view. Existing evidence is static output; the UI should not imply that this repo currently serves live predictions.

## ML teammate / owner next steps

1. Do not tune on the reused test window; obtain a later independent time period before making further model-selection or deployment claims.
2. Review repository hygiene before pushing: commit only intended source and documentation. Do not add local data, model artifacts, Docker state, or generated experiment dumps.
3. Coordinate with the frontend teammate on a shared display schema; no prediction-serving API exists yet.

## Key documentation

- [README.md](../README.md): setup, data staging, commands, and standard pipeline outputs.
- [DATA_VALIDATION_REPORT.md](DATA_VALIDATION_REPORT.md): real dataset and split validation.
- [ACTUAL_DATA_ML_RESULTS.md](ACTUAL_DATA_ML_RESULTS.md): earlier schedule/history model results.
- [ASOF_TZ_REPORT.md](ASOF_TZ_REPORT.md): UTC time-zone correction, third evaluation, coverage, and leakage checks.
- [OPS_V2_REPORT.md](OPS_V2_REPORT.md): Phase 1/2 comparisons, frozen choice, and completed fourth evaluation.
- [TEST_RUN_LOG.md](TEST_RUN_LOG.md): what the synthetic correctness suite actually covered and test markers.
- [DIAGNOSTICS_REPORT.md](DIAGNOSTICS_REPORT.md): threshold transfer, slices, and ranking diagnostics.
