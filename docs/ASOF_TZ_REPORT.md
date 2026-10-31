# UTC-Correct As-Of Operations Report

## Verdict

The operations gain survived the UTC correction on validation, but UTC correction did **not** improve the already-used held-out test result. At T-120, E13 Logistic Regression with all UTC-correct operations features reached validation PR-AUC **0.28752**, versus **0.24220** for the frozen schedule-only E6 reference: a gain of **+0.04532**, above the required +0.03 gate. The adopted feature set is therefore variant (d), all 16 UTC-correct operations features.

Variant (d) was slightly worse than the old local-clock definition on validation (-0.00200 PR-AUC), and the paired date-block interval includes zero. In the labelled **THIRD DESIGN EVALUATION, UTC-corrected**, LR test PR-AUC fell from 0.41032 to 0.40508 and RF fell from 0.38237 to 0.37924. These are negative results, not improvements.

## Time model and observability rule

`airportsdata==20261030` supplies each IATA airport's IANA time-zone name. `tzdata==2026.6` supplies the time-zone database inside the isolated `.venv-ops-v2` environment. The generated table has one airport/local-date row and 48 integer UTC-offset columns: fold 0 and fold 1 for each local hour. This represents offset changes by date and ambiguous DST hours without asking Spark/JVM to infer them.

For each flight:

- scheduled departure UTC = origin-local `FlightDate + CRSDepTime`, converted with the origin/date/hour offset;
- actual departure UTC = scheduled departure UTC + `DepDelay`;
- scheduled arrival UTC = scheduled departure UTC + `CRSElapsedTime`;
- actual arrival UTC = scheduled arrival UTC + `ArrDelay`;
- target cutoff T = scheduled departure UTC - 120 minutes.

Every departure, arrival, carrier-network, destination, origin-departure, and origin-inbound aggregate is ordered in UTC and admits only events from UTC hours strictly before T. For an ambiguous fall-back hour, an event uses the later UTC interpretation and a target uses the earlier UTC interpretation. This is deliberately conservative: it can omit an observable event but cannot admit a post-T event. Spring-forward nonexistent local hours have no valid offset.

The origin features retain their original meaning (activity at the target's origin), but their event-time comparison is now also performed in UTC. The code is in `src/asof_ops_utc.py`; `analysis/ops_v2/run_utc_ml.py` injects it into the existing ML stage without changing `src/03_ml.py`.

## Airport-offset coverage

| Item | Measured value |
|---|---:|
| Distinct airport codes in clean real data | 340 |
| Matched by `airportsdata` | 339 |
| Unmatched | 1 |
| Airport/date rows | 34,917 |
| Rows containing an offset/DST transition | 320 |
| Covered local dates | 2024-09-25 through 2025-01-05 |

The requested table is `results/experiments/ml_redesign/asof_ops/airport_tz.csv`; it was also uploaded to `hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/reference/airport_tz.csv`.

| Unmatched code | Reason | Treatment |
|---|---|---|
| PBI | No IATA record in `airportsdata==20261030` | UTC target/event timestamps remain unavailable; observed counts are zero and smoothed rate features use priors fitted on the training window. No guessed offset was inserted. |

Real-data pre-test coverage was 1,247,073 / 1,252,000 event rows and 1,228,200 / 1,233,037 distinct target groups. Final-test target coverage was 517,521 / 520,187 groups. Coverage loss is dominated by PBI.

## Validation-only ablation

The ablation used the E13 schedule/history specification, T-120, LR `regParam=0.001`, 1,067,747 training rows and 173,549 validation rows. The test window was not read by this ablation.

| Variant | Operations features | Validation ROC-AUC | Validation PR-AUC | Top-10% precision | Top-10% recall | Top-10% lift |
|---|---:|---:|---:|---:|---:|---:|
| (a) origin-only, UTC | 5 | 0.63570 | 0.27284 | 0.33211 | 0.19620 | 1.95209 |
| (b) origin + inbound-at-origin, UTC | 8 | 0.63599 | 0.27379 | 0.33473 | 0.19759 | 1.96749 |
| (c) all operations, old local clocks | 16 | 0.65081 | 0.28953 | 0.35762 | 0.21131 | 2.10202 |
| (d) all operations, new UTC ordering | 16 | 0.65166 | 0.28752 | 0.35460 | 0.20951 | 2.08428 |
| Frozen schedule-only E6 reference | 0 | not rerun here | 0.24220 | not measured | not measured | not measured |

Measured gate: 0.28752 - 0.24220 = **+0.04532**, so variant (d) is adopted. Variant (d) improves on origin-only by +0.01469 measured PR-AUC, but trails the old local-clock variant by -0.00200.

### Paired date-block bootstrap

The bootstrap resampled all 9 validation dates as whole paired blocks, 2,000 times with seed 20261101. PR curves were reconstructed from the saved 1,000-bin per-date score histograms; intervals are percentile 95% CIs, so their small binning approximation is a limitation.

| Difference | Bootstrap mean | 95% CI | P(difference > 0) |
|---|---:|---:|---:|
| (d) UTC all - (a) UTC origin-only | +0.01437 | [+0.00473, +0.02271] | 0.9990 |
| (d) UTC all - (c) old local-clock all | -0.00180 | [-0.00641, +0.00254] | 0.2295 |

The cross-time-zone features add signal beyond origin-only, but this validation window does not establish that UTC ordering itself has better predictive ranking than the old implementation. UTC is adopted because it satisfies the leakage/observability definition and passes the schedule-only gate, not because it wins (d) versus (c).

## Synthetic leakage tests

| Test | Expected property | Result |
|---|---|---|
| Westbound JFK target / LAX arrival | Event is two hours after T in UTC, although its LAX wall clock is earlier than T's JFK wall clock; UTC implementation must exclude it | Passed; new destination landed count is 0, while the old local-clock implementation incorrectly counts it |
| DST fall-back ambiguity | Conservative folds must not admit an event whose UTC ordering is uncertain/post-T | Passed |
| Existing future-outcome invariance | Changing outcomes after T must not change features | Passed |
| Own-outcome exclusion and row preservation | Target flight cannot contribute its own outcome and joins cannot change row count | Passed |

Test markers: `ASOF_UTC_TESTS_STATUS=PASSED`, `ASOF_OPS_TESTS_STATUS=PASSED`, and `ML_PIPELINE_TESTS_STATUS=PASSED`.

## Third design evaluation (test scored once in this phase)

The selection-only run wrote and committed the pre-test freeze as commit `19cd9e4` before the final job read test outcomes. The test window (2024-12-04 through 2024-12-31) had already been used by two earlier design evaluations; this is use number **3**. It contains 523,983 eligible flights and 111,703 delays (21.318% prevalence).

| Model | Threshold | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.15 | 0.68704 | 0.40508 | 0.28189 | 0.76166 | 0.41148 | 0.53554 |
| Random Forest | 0.15 | 0.66504 | 0.37924 | 0.28049 | 0.69830 | 0.40022 | 0.55382 |

| Model | Top-10% precision | Top-10% recall | Top-10% lift |
|---|---:|---:|---:|
| Logistic Regression | 0.50051 | 0.23643 | 2.34783 |
| Random Forest | 0.48140 | 0.22733 | 2.25819 |

| Model / slice | Flights | Prevalence | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LR, Christmas/New Year | 225,777 | 0.26799 | 0.47290 | 0.31761 | 0.85776 | 0.46357 |
| LR, regular day | 298,206 | 0.17169 | 0.30749 | 0.23971 | 0.64809 | 0.34998 |
| RF, Christmas/New Year | 225,777 | 0.26799 | 0.44763 | 0.33529 | 0.75309 | 0.46400 |
| RF, regular day | 298,206 | 0.17169 | 0.28381 | 0.22812 | 0.63354 | 0.33545 |

### Comparison with the earlier local-clock evaluation

| Model | Old local-clock test PR-AUC | UTC-correct test PR-AUC | Delta | Old test F1 | UTC test F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.41032 | 0.40508 | -0.00524 | 0.41996 | 0.41148 |
| Random Forest | 0.38237 | 0.37924 | -0.00313 | 0.40137 | 0.40022 |

The honest conclusion is that the strong validation gain over schedule-only survives, but the previous test score was mildly optimistic relative to the corrected temporal definition. The corrected LR still provides useful ranking (50.1% delayed among the top 10% versus 21.3% prevalence), but it is not a large-accuracy classifier.

### Freeze reproducibility note

The committed selection run froze LR threshold 0.15 and RF threshold 0.14. The final `--mode all` job repeated validation selection before opening test and reproduced every selected feature/model hyperparameter, but Spark RF's distributed validation scores changed enough to choose RF threshold 0.15. No test value influenced this change. Consequently, LR results and all threshold-free RF metrics (ROC-AUC, PR-AUC, ranking) conform to the committed model freeze; RF threshold-dependent precision/recall/F1/accuracy use the final job's validation-only threshold 0.15 and should not be represented as an exact byte-for-byte replay of the committed threshold configuration.

## Failures retained

| YARN application | Stage | Failure | Resolution |
|---|---|---|---|
| `application_1793455790052_0008` | Ablation preparation | Driver/event-log listener OOM on the very wide logical plan | Disabled event logging and inserted an auditable HDFS checkpoint; retry succeeded |
| `application_1793455790052_0010` | Ablation model | 512 MB executors were externally killed while caching the wide prepared frame | Streamed source Parquet, persisted narrower derivatives `DISK_ONLY`, disabled vectorized Parquet reader, used one 512 MB executor |
| `application_1793455790052_0011` | Ablation model | AUC evaluator could not find `probability` after an early scalar-score projection | Retained both probability vector and scalar score; no metric from the failed run was accepted |

Successful real-cluster applications were preparation `..._0009`, ablation `..._0012`, selection/freeze `..._0013`, and third evaluation `..._0014`. Failed logs are retained under `results/experiments/ml_redesign/ops_v2/tz_ablation/`.

## Exact commands

```powershell
py -3 -m venv .venv-ops-v2
.\.venv-ops-v2\Scripts\python.exe -m pip install -r analysis\ops_v2\requirements.txt

docker cp analysis/ops_v2/export_airport_codes.py master:/tmp/export_airport_codes.py
docker exec master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /tmp/export_airport_codes.py'
.\.venv-ops-v2\Scripts\python.exe analysis\ops_v2\build_airport_tz.py

docker cp analysis/ops_v2/tz_ablation.py master:/tmp/tz_ablation.py
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/tz_ablation.py --phase prepare --reuse-old'
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --conf spark.sql.parquet.enableVectorizedReader=false --driver-memory 512m --executor-memory 512m --num-executors 1 --executor-cores 1 /tmp/tz_ablation.py --phase model'
.\.venv-ops-v2\Scripts\python.exe analysis\ops_v2\bootstrap_tz_ablation.py --input results\experiments\ml_redesign\ops_v2\tz_ablation\validation_score_histograms_1000bin.csv --output results\experiments\ml_redesign\ops_v2\tz_ablation\bootstrap_ci.json --replicates 2000 --seed 20261101

docker cp analysis/ops_v2/run_utc_ml.py master:/tmp/run_utc_ml.py
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/run_utc_ml.py --mode select --experiments E0_reference,E13_e6_plus_asof --asof-lead-minutes 120 --local-results /results/experiments/ml_redesign/ops_v2/freeze_select --hdfs-results hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/freeze_select/results --hdfs-models hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/freeze_select/models'
git commit -m "Add UTC-correct as-of operations features"
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/run_utc_ml.py --mode all --experiments E0_reference,E13_e6_plus_asof --asof-lead-minutes 120 --local-results /results/experiments/ml_redesign/ops_v2/third_design_utc --hdfs-results hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/third_design_utc/results --hdfs-models hdfs://master:9000/user/root/flights/experiments/ml_redesign/ops_v2/third_design_utc/models'

docker exec master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 2g /src/tests/test_asof_ops_utc.py'
docker exec master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 2g /src/tests/test_asof_ops_features.py'
docker exec master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 2g /src/tests/test_ml_pipeline.py'
```

## Limitations

- `CRSElapsedTime` is used to derive scheduled arrival UTC; that is preferable to comparing unrelated airport wall clocks, but it inherits any source-data schedule anomalies.
- Ambiguous DST handling is conservative rather than attempting to infer an aircraft's exact fold from missing actual timestamps.
- PBI has no metadata match in the pinned airport database and therefore receives fallback features.
- Validation contains only nine dates, so its date-block interval is informative but not high-powered.
- Christmas/New Year is unseen before the test window. Its category effect cannot be learned; slice performance partly reflects its higher prevalence.
- The test window has now been used three times. Further model choices must not be justified from these test results.
