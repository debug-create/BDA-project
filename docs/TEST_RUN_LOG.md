# Test evidence: SYNTHETIC / LOCAL ONLY

Environment: pyspark 3.5.9 (same version as the cluster), Python 3.11, OpenJDK 21, `local[2]`, 8 GB sandbox, **2 GB driver heap**.
No real BTS data, no YARN, no HDFS, no 512 MB executors. These runs show code-level properties only; they say nothing
about model quality or about runtime and memory on 1.77 M rows.

Files tested (md5): `src/03_ml.py` de4aac247a0873db6938b316c3b063da,
`src/tests/test_ml_pipeline.py` 93f26154fc9b48443288b0b8a1abad61 (hashes were checked unchanged after the run).

## Final run: 24 of 24 tests passed (single uninterrupted invocation, about 30 minutes)
`ML_PIPELINE_TESTS_STATUS=PASSED`, plus `TEMPORAL_HISTORY_TEST_STATUS=PASSED` for `test_temporal_history_features.py`.

Unit (21): minute-of-day parsing incl. 2400; split windows equal the agreed baseline windows; schema type check;
eligibility categories exclusive and summing to the input; leakage guard and feature provenance (all specs);
past-only property over every date (all-earlier and trailing windows); the property check flags a deliberately leaky
leave-one-day-out design; key isolation and first-day fallback; exact trailing-window values; frozen-table application
with unseen keys; scheduled-demand counts include cancelled flights and joins preserve rows; threshold search and
confusion counts; warm-up rows excluded but kept as history; namespace safety and production guard; **extended and
short-memory history groups are past-only and row-preserving; recent group uses only the last N earlier dates;
destination-hour / origin-day / previous-hour demand and scheduled speed have hand-computed values and null block time is
imputed; combining feature specs; unsupported extra model name rejected.**

Integration (3 + 2 on synthetic parquet in local mode): select mode never reads the test window and gives identical
decisions when only test labels change (all ten feature experiments); all mode freezes decisions before the test is
scored, reconciles counts, writes the documented files and saved models; failure is reported as FAILED with exit code 2;
**all four model families (LR, RF, GBT, MLP) are tuned on validation, thresholded, refit, scored once on the test window,
saved, appear in every result table, and their frozen choices do not change when only test labels change.**

## Failures met on the way
1. Earlier sessions: Java heap exhaustion in a 1 GB local driver; fixed by capping Spark job/UI metadata retention
   (`SPARK_SESSION_CONF`); float-exact comparison replaced by a 1e-9 tolerance.
2. This session: the script on disk had an unfinished edit (`main` still used the removed `ctx.with_gbt` /
   `GBT_MAX_ITER` and a `fit_gbt` call with the old signature). The extra models were connected into `main`.
3. With ten feature experiments, a 1 GB driver ran out of heap again in the select-mode test (two full runs in one JVM);
   with 2 GB the whole suite passes. **The cluster driver has 512 MB. Whether a real select run fits is untested.**
   Mitigation if it does not: run fewer experiments per run (`--experiments ...`), and give the driver more memory if the
   Docker/WSL budget allows.
4. My own test bug (pipeline fixture lacked history columns) fixed in the test, not in the script.

## Known gaps
Not covered: real-schema edge cases, YARN, 512 MB executors, runtime on real data, the `E6` combined-spec path inside a
full run (only `combine_specs` itself is unit-tested), saved-model reload. XGBoost, LightGBM and CatBoost are not
included; an attempt to run XGBoost's Spark API in this sandbox crashed (Arrow on Java 21), so no XGBoost code was added.
