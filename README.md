# Big Data Analytics: Holiday Flight Delay Pipeline

Spark/Hadoop mini-project using 1,781,482 US domestic flight records from October–December 2024. The ML stage predicts `ArrDel15` for non-cancelled, non-diverted flights using only information available before scheduled departure.

The actual-data pipeline has been validated and executed on Spark 3.5.9 over YARN with two worker containers. See [the data report](docs/DATA_VALIDATION_REPORT.md), [actual-data ML results](docs/ACTUAL_DATA_ML_RESULTS.md), and [test evidence](docs/TEST_RUN_LOG.md).

## Design and leakage controls

- Train: 2024-10-01–2024-11-24; validation: 2024-11-25–2024-12-03; test: 2024-12-04–2024-12-31.
- Feature/hyperparameter/model/threshold selection uses train and validation only.
- `--mode all` writes `frozen_config.json` before it reads test outcomes, then evaluates the test once.
- Predictors are scheduled time/duration/distance, carrier, airport, weekday, holiday calendar, schedule-only demand counts, and smoothed historical rates.
- Every fitting-row historical rate uses labels from dates strictly before that row's date. Validation/test lookup tables are frozen from their completed fitting window.
- Departure/arrival outcomes, actual times, delay causes, cancellation/diversion flags, and other post-departure fields are blocked as predictors.

## 1. Start the three-container cluster

From the repository root:

```powershell
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.yml --progress plain build master
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
```

On a fresh named volume, the master waits for both DataNodes and uploads the Spark runtime jars required by YARN. Confirm both workers:

```powershell
docker exec master bash -lc 'hdfs dfsadmin -report | grep "Live datanodes"; yarn node -list -all'
```

## 2. Stage the actual handoff

Extract the supplied package under `results/data_handoff`, producing `results/data_handoff/clean`, `results/data_handoff/holidays.csv`, and the completion marker. The data and generated results are ignored by Git.

Upload to the canonical HDFS locations:

```powershell
docker exec master bash -lc 'hdfs dfs -mkdir -p /user/root/flights/clean /user/root/flights/reference && hdfs dfs -put -f /results/data_handoff/clean/* /user/root/flights/clean/ && hdfs dfs -put -f /results/data_handoff/holidays.csv /user/root/flights/reference/holidays.csv && hdfs fsck /user/root/flights/clean -files -blocks -locations'
```

## 3. Validate the dataset

```powershell
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/00_validate_data.py'
```

Expected terminal marker: `DATA_VALIDATION_STATUS=PASSED`. JSON and daily counts are written under `results/data_validation` and the corresponding HDFS results directory.

## 4. Run correctness tests

Local Spark is intentionally used only for small synthetic correctness tests:

```powershell
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py'
```

## 5. Reproduce controlled actual-data experiments

Each selection run has an isolated namespace and does not read test labels:

```powershell
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_history_ablation --experiments E0_reference,E1_no_history'
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_temporal_features --experiments E0_reference,E3_history_warmup7,E4_history_trailing14,E9_recent7_history'
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_calendar_schedule --experiments E0_reference,E2_no_holiday_features,E5_plus_scheduled_demand,E10_extra_demand,E11_schedule_extras'
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_extended_history --experiments E0_reference,E8_extended_history'
```

The 1 GB driver is required for combined feature plans; executors remain at the configured 512 MB. Selection uses validation PR-AUC with a predeclared 0.002 improvement margin. Thresholds maximize delayed-class F1 on validation only.

## 6. Run the frozen final evaluation

```powershell
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1536m /src/03_ml.py --mode all --run-tag actual_final --experiments E0_reference,E3_history_warmup7,E8_extended_history,E9_recent7_history,E10_extra_demand'
```

Expected terminal marker: `MLLIB_JOB_STATUS=SUCCEEDED`. Host outputs are written to `results/experiments/ml_redesign/actual_final`; HDFS results and models use `/user/root/flights/experiments/ml_redesign/actual_final`. Existing production output paths are not overwritten.

Generate report plots and verify that fitted artifacts reload:

```powershell
python src/04_make_plots.py --results-dir results/experiments/ml_redesign/actual_final
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client /src/05_verify_saved_models.py --models-uri hdfs:///user/root/flights/experiments/ml_redesign/actual_final/models'
```

## Outputs

The final directory contains experiment/hyperparameter tables, validation and test metrics, confusion counts, baselines, ranking lift, holiday slices, threshold-transfer diagnostics, feature importance, frozen configuration, preprocessing metadata, and a run summary. HDFS stores the fitted preprocessing `PipelineModel`, Logistic Regression, Random Forest, and every historical-rate lookup table needed to reproduce feature construction.

Raw CSV/Parquet, Docker volumes, HDFS data, model binaries, Spark logs, and generated result directories must not be committed. The tracked documentation contains the reproducible numerical summary and selected plots.
