# OPS V2 evaluation artifacts and reproduction record

The artifacts under `results/experiments/ml_redesign/ops_v2/` are the retained outputs for the **FOURTH DESIGN EVALUATION**. The test window was scored once in that evaluation and must not be reused to select or tune another model. `artifact_manifest.json` records byte sizes and SHA-256 hashes for the pushed files.

## Included result artifacts

| Artifact | Purpose |
|---|---|
| `host_models/xgboost_best.json` | Original selected XGBoost search result, including trial, validation metrics, threshold, and best iteration |
| `frozen_ops_v2_config.complete.json` | Self-contained model/config/feature/dependency/source-hash manifest, including 387 boosting rounds and the ordered 183-feature list |
| `host_export/pretest_metadata.json` | Train/validation-only feature export metadata; `test_opened=false` |
| `host_export/test_metadata.json` | Fourth-evaluation export row counts, null/duplicate checks, feature list, and runtime metadata |
| `fourth_design/test_results.json` | Final held-out test ranking metrics and paired date-block bootstrap |
| `fourth_design/xgboost_test_scores.npy`, `fourth_design/utc_lr_refit_test_scores.npy` | Per-row prediction scores in the same test-row order as the export |
| `host_models/phase1_decision.json`, `phase1_folds.json`, `phase1_validation_summary.json` | Original as-run Phase-1 decision and validation/fold evidence. Read alongside the retrospective-fold audit in `OPS_V2_REPORT.md`; these files preserve the historical outputs and are not independently valid forward-validation claims. |
| `host_export/pretest_yarn.log`, `host_export/test_yarn_final.log` | Full Spark/YARN execution logs for the successful host-feature exports |
| `fourth_design/final_host.log` | Final host-scoring execution log with success marker and reported metrics |

The compressed test feature Parquet is about 609 MB and is not included in the Git repository. It remains in the original HDFS/host workspace when available. The successful test export is recorded as YARN application **`application_1793854644166_0004`** (`HOST_EXPORT_STATUS=SUCCEEDED`). The pretest export is **`application_1793455790052_0017`** (`HOST_EXPORT_STATUS=SUCCEEDED`). Their complete logs are in the paths above. The final scoring log ends with `FOURTH_DESIGN_EVALUATION_STATUS=SUCCEEDED`.

The two score arrays are the actual float32 outputs written by `final_host.py`, each with 523,983 values. No YARN application ID exists for that final model fit/scoring step: it ran in the host Python environment after the YARN feature export. The arrays are in the test dataset scan order. A separate keyed prediction table was not retained, so the arrays alone are not a durable row-keyed prediction export; do not join them to a newly generated Parquet scan unless you independently verify identical row ordering.

## Export regeneration command

This command regenerates the test feature export from the existing clean HDFS input; it does not train or score a model. It requires Docker Compose, the project containers, clean flight Parquet and holiday reference data in HDFS, and the committed frozen decision file mounted at `/results/experiments/ml_redesign/ops_v2/frozen_ops_v2_config.json`.

```powershell
docker compose -f docker/docker-compose.yml up -d
docker cp analysis/ops_v2/export_host_features.py master:/tmp/export_host_features.py
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 2g --executor-memory 512m --num-executors 2 --executor-cores 1 --conf spark.driver.maxResultSize=512m --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --conf spark.sql.shuffle.partitions=64 /tmp/export_host_features.py --scope test'
```

The exporter refuses test scope unless the frozen config marks decisions frozen, committed before test, authorized, and at least one phase passed. It writes the canonical export under `hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/host_export/test`. The successful historical export was then copied from HDFS to `C:\bda\ops_v2\test` for host scoring. Do not rerun `final_host.py` against this test export for model-selection purposes.

## Point-in-time limitations

The fourth result uses hour-bucketed T−120 features, with T derived from `ScheduledDepHour`, not the exact CRS departure minute. The backlog features also use final `Cancelled`/`Diverted` flags without status-event timestamps. This assumes those final statuses were known at prediction time; the assumption is unverified and diversion status can be post-departure information. See `ASOF_TZ_REPORT.md` for the precise cutoff and recommended future correction. Therefore the result is reproducible for the implemented feature definition but is not strictly point-in-time certified.
