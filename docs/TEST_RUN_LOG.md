# Test and execution evidence

Run date: 2026-10-30. Spark 3.5.9, Hadoop 3.4.3, Java 11, Docker Desktop, one YARN ResourceManager and two registered NodeManagers/DataNodes.

Final SHA-256: `src/03_ml.py` = `F89483358D1A2FBD7AEE4F1EB231D69D6C9658E7F72BDE972D56235BC708291B`; `src/tests/test_ml_pipeline.py` = `9702AA70B10BFA906DB94BAD44911489B7B474577C9FEAAC3C8631C01BC207EC`.

## Synthetic correctness suite

Synthetic tests verify code properties; they are not evidence of predictive performance. They ran in the project container with `local[2]` and a 2 GB driver:

- `TEMPORAL_HISTORY_TEST_STATUS=PASSED` from `src/tests/test_temporal_history_features.py`.
- All 19 unit components in `src/tests/test_ml_pipeline.py` passed.
- `test_select_mode_never_reads_test_and_is_invariant_to_test_labels` passed in 456.4 seconds, including two ten-experiment runs with altered test labels.
- `test_all_mode_reconciles_and_freezes_before_test` passed in 180.7 seconds.
- Failure-status and unsupported-model reporting tests passed.
- `test_extra_models_full_pipeline_and_decision_invariance` passed in 267.1 seconds, exercising LR, RF, GBT and MLP on synthetic data.

The components passed after the final fixes, although they were run in focused invocations rather than misrepresented as one uninterrupted suite run.

Coverage includes eligibility reconciliation, split boundaries, predictor provenance, strict past-only expanding and trailing history, warm-up behaviour, frozen lookup tables, schedule-only demand counts, threshold selection, namespace protection, no test-label reads in selection mode, freeze-before-test ordering, result/model persistence, and invariance of selected decisions to changed test labels.

## Failures found and fixed

1. A fresh cluster could not distribute Spark because `spark.yarn.jars` pointed at an empty HDFS directory. The entrypoint now waits for both DataNodes and seeds `/spark-jars` on first start.
2. Integration tests passed bare temporary paths while Docker's default filesystem was HDFS. Tests now use explicit `file://` URIs for local temporary results/models.
3. Repeated deep experiment plans exhausted the driver in Adaptive Query Execution's plan-update rendering path. The ML session disables AQE for this deterministic batch pipeline and caps SQL/UI retention. The focused synthetic integration tests then passed with a 2 GB driver.
4. The first actual temporal combination run exhausted a 512 MB driver while canonicalizing the combined plan. The exact rerun used a 1 GB driver and succeeded; executors remained at 512 MB. This resource requirement is documented in the reproduction command.

No test was skipped or weakened to obtain a pass.

## Actual-data / YARN evidence

- Data validation ran as YARN application `application_1793344136681_0001`, used both workers, and ended `DATA_VALIDATION_STATUS=PASSED`.
- Controlled actual-data selection jobs ran in isolated result/HDFS namespaces. The reference/history, temporal, calendar/schedule, and extended-history runs all ended `MLLIB_JOB_STATUS=SUCCEEDED` (the initial 512 MB temporal-combination attempt failed as described above and was rerun successfully).
- The final `--mode all` job freezes configuration before reading test outcomes, evaluates the held-out test once, and writes fitted preprocessing and classifier artifacts. Its application ID and final status are recorded in `docs/ACTUAL_DATA_ML_RESULTS.md`.
- `src/05_verify_saved_models.py` reloads the persisted PipelineModel, LogisticRegressionModel and RandomForestClassificationModel and checks a consistent positive feature width.

## Reproduce the synthetic tests

```powershell
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py'
```

Generated result directories, Spark logs, Parquet data and model binaries remain excluded from Git.
