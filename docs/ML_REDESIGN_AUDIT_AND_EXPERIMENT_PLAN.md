# ML stage redesign: audit, experiment plan and hand-off

Scope: `src/03_ml.py` and `src/tests/` only. `01_clean.py`, `02_holiday_analysis.py`, Docker/Hadoop/YARN
configuration and `data/holidays.csv` are untouched.

## 0. What has and has not been verified

| Claim | Status |
|---|---|
| Replacement `03_ml.py` compiles and runs end to end in **local Spark 3.5.9 on synthetic data** (select mode and all mode) | Verified here (see section 8 for the log). |
| Leakage-sensitive helpers behave as specified (past-only history, frozen-table application, demand counts, row preservation, windows, eligibility accounting, namespace guard) | Verified here, **synthetic/local only**. |
| Final decisions do not depend on test-window labels | Verified here on synthetic data by flipping only test labels and comparing outputs. |
| Any model-quality number on the real BTS data | **Not available.** The real Parquet and cluster were not accessible. No validation or test metric from the real data appears in this document; the baseline numbers quoted below come from the handover. |
| Runs under `spark-submit --master yarn` with 512 MB executors/driver | **Not tested.** Needs the owner's cluster. |
| Runtime and memory on 1.77 M rows | **Unknown.** Synthetic timings are meaningless for this. |

**Provenance note (2026-10-29).** The `03_ml.py` on disk when this session resumed was not the file I delivered on
2026-10-28: it had been edited afterwards (extended and short-memory history keys, extra demand and schedule features,
GBT/MLP helper functions, `--extra-models`) but the edit was unfinished: `main` still called a removed `ctx.with_gbt`
and a `fit_gbt` with the old signature, so every run would have failed at that point. I did not write that part. I kept
it, connected the extra models into `main` (tuning, validation thresholds, frozen config, single test scoring, model
saving, result tables), added tests for it, and re-ran the whole suite on the result. A copy of the file as found is
not part of the bundle.

No pull request has been opened: no repository write access was available in this session. The files are laid
out in repository paths so they can be copied onto a branch (`git add src/03_ml.py src/tests docs`).

## 1. Audit of the current `src/03_ml.py` (baseline candidate)

The current script is better than the handover text suggests: training-row history is already past-only, the
design avoids raw dates, and the final models are refit on train+validation. The findings below are about what
it does not prove or does not protect against. Line numbers refer to the file in the repository zip.

| # | Finding | Evidence | Risk | Change made |
|---|---|---|---|---|
| A1 | Outputs always go to the production paths. | Lines 73-77 hard-code `.../results/ml`, `.../models`, `/results/ml`. | Any trial run overwrites the baseline results and models. | Default namespace is `experiments`; production needs `--namespace production --confirm-overwrite-baseline`. |
| A2 | Nothing structural stops test labels reaching decisions. | Test rows are cached in `eligible` and test positives/prevalence are computed and logged before any tuning (lines 565-607). | Not a leak today, but a later edit could use them without anyone noticing. | `--mode select` filters test rows out before caching; `all` writes `frozen_config.json` and logs `DECISIONS_FROZEN=1` before the test is scored; a test flips only test labels and asserts identical decisions. |
| A3 | Threshold picked on a train-only model, applied to a train+validation refit, under a drifting base rate. | Lines 676-708; prevalence 13.74 % / 17.01 % / 21.32 % (handover). | Calibration may not transfer. | Kept the design (it matches the agreed protocol) but added `threshold_transfer.csv` (validation vs test predicted-positive rate) so the effect is visible. |
| A4 | F1-optimal thresholds are close to degenerate. | Baseline test: "flag every flight" F1 = 2*0.2132/1.2132 = 0.3515; LR F1 0.3779 (+0.026), RF 0.3695 (+0.018); thresholds 0.10/0.13, recall 0.83/0.68, accuracy 0.42/0.50 (handover). | F1 barely separates the models from a trivial rule; this will look weak to a reader. | Script now warns when validation F1 is within 0.005 of the trivial F1. The full threshold table, lift table and the trivial baselines are reported so the trade-off is visible. Threshold criterion itself is unchanged (F1, as agreed); changing it is an owner decision. |
| A5 | The holiday features can only extrapolate. | `holidays.csv`: training window contains one holiday date (Halloween); validation is mostly Thanksgiving week; test is Christmas/New Year. | Stage-A validation measures Halloween-to-Thanksgiving transfer, the test measures Halloween+Thanksgiving-to-Christmas transfer. Neither shows a learned Christmas effect. | Not fixable by code. Added experiment E2 (holiday features off) so the report can say whether the flag helps at all on validation, and the script logs the unseen-category notes as before. |
| A6 | History rates: three design choices that are risks, not bugs. (a) Day 1 of the fitting window gets a constant 0.5 for all five rates (about 1.8 % of training rows). (b) Early fitting rows see thin history while validation/test rows see the full window. (c) Frozen tables go stale while prevalence drifts. | `apply_history_training_past_only` / `apply_history` (lines 222-295). | May cost validation AUC-PR. | Experiments E3 (drop first 7 days as rows but keep them as history) and E4 (trailing 14-day history). |
| A7 | The existing regression test cannot detect key mixing and does not cover validation/test application. | Fixture has one airline, one route, one block for all rows (test file lines 32-52); only `apply_history_training_past_only` is called. | A bug in `apply_history` or in key handling would pass. | New tests: multi-key property test over every date, key isolation, frozen-table application with unseen keys, exact trailing-window values, and a deliberately leaky implementation that the property test must flag. The old test is kept and still passes. |
| A8 | Eligibility and joins are not reconciled at run time. | Eligible count (line 600) is never compared with the category count; no row-count check after the holiday or history joins. | A join that duplicated rows would go unnoticed. | `check_rows` after every join; eligibility categories must sum to the input row count; eligible frame must equal the independent eligible count. |
| A9 | Only presence, not type, of required columns is checked. | Lines 501-503. | A schema change would fail late with a confusing error. | `validate_schema` checks types up front. |
| A10 | Threshold grid stops at 0.60. | `THRESHOLD_CANDIDATES`. | Class-weighted models produce inflated probabilities and would hit the edge. | Grid widened to 0.02-0.90 (superset of the old range). |
| A11 | Driver memory risk from job metadata (new, found while testing). | In local mode with a 1 GB heap, two consecutive `select` runs with Spark's default retention ran out of Java heap; with `spark.ui.retainedJobs/Stages=100`, `spark.sql.ui.retainedExecutions=50` the same test completed. | The production driver has 512 MB. I have **not** shown this happens on the cluster, only that the cap removed the failure locally. | `SPARK_SESSION_CONF` applied when the script creates its own session. |

Not changed on purpose: target, eligibility rule, chronological split (reproduced exactly by a test:
train 2024-10-01..11-24, validation 11-25..12-03, test 12-04..12-31 for 92 dates), the Logistic Regression and Random
Forest model families and their grids, history smoothing constant, seed, AUC-PR as the selection metric, F1 as the
threshold criterion, output filenames.

## 2. Experiment plan

All experiments fit on the training window and are scored on the validation window. The test window is not read.
Hypotheses are about direction and are things the validation numbers can contradict.

Feature experiments use Logistic Regression with a fixed `regParam=0.01` (cheap, deterministic) so only the feature
factor varies. Each differs from E0 in **one** factor.

| Id | Factor changed vs E0 | Hypothesis |
|---|---|---|
| E0 | none: the previous script's features (cyclic times, carrier, day of week, holiday flag/category, five expanding past-only history rates) | Reference. |
| E1 | remove history rates | If the history rates carry signal, AUC-PR drops. |
| E2 | remove holiday flag and category | Holiday features cannot generalise to unseen holidays, so removing them should not hurt validation. |
| E3 | drop first 7 fitting days as training rows (still used as history) | Removes the constant-0.5 and thin-history rows, so may help slightly. |
| E4 | history from trailing 14 days instead of all earlier days | Tracks the drifting base rate better. |
| E5 | add scheduled origin-hour demand (count of scheduled flights from the same origin, date and hour, including cancelled/diverted ones) | A congestion proxy known in advance; may help. |
| E6 | combination of every challenger that beat E0 by the margin (only built if two or more did) | Gains add up. |
| E8 | five more past-only rate keys: airline x origin, origin x weekday, airline x departure block, destination x arrival hour, airline x route | Finer groups capture hub/carrier/time-of-week patterns the five base keys average away. |
| E9 | add short-memory rates (last 7 dates) for airline, origin, origin x block, next to the long-memory ones | Follows the drifting delay rate (13.7 % to 21.3 % across the windows) better than all-time rates. |
| E10 | add scheduled destination-hour, origin-day and previous-hour demand | Congestion at both ends and a queue-from-the-previous-hour proxy, all from the schedule. |
| E11 | add scheduled speed (distance / block time) and log distance | Tight block times and short/long-haul shape are visible in the schedule. |

**Selection rule.** A challenger replaces E0 only if its validation AUC-PR is at least `SELECTION_MARGIN = 0.002`
higher; among those the best wins. The margin is a deliberate guard against chasing noise, not a significance test.

On the chosen feature set the script then runs: LR `regParam` in {0.001, 0.01, 0.1}; one class-weighted LR at the best
`regParam` (kept only if it beats the unweighted one by the margin); Random Forest with the two existing
configurations (`maxDepth` 5 or 7, `minInstancesPerNode` 100, 30 trees). Optional `--extra-models gbt,mlp` adds
`GBTClassifier` (2 configurations: 30 iterations, depth 4 or 6) and a small `MultilayerPerceptronClassifier` (hidden
layers 16, or 32-16; inputs standardised inside its pipeline). Extras are tuned on validation AUC-PR, thresholded on
validation F1, refit on train+validation and scored once on the test window in the same pass as LR and RF; they are
extra rows in `metrics.csv` and never replace the required LR/RF comparison. Thresholds are chosen by validation F1.

Everything lands in `experiment_comparison.csv` with validation AUC-ROC, AUC-PR, best F1/threshold, difference from E0,
and runtime per experiment. Test AUC-PR/F1 columns are filled only for the final model of each family in `--mode all`.

### Why not deep learning, XGBoost, LightGBM or CatBoost here

- **Deep learning:** on tabular schedule data, gradient-boosted trees normally match or beat neural networks, and 512 MB
  executors cannot train a serious one. The `mlp` option is MLlib's small feed-forward network, included so the question
  can be answered with a number rather than an opinion.
- **XGBoost:** has a Spark API (`xgboost.spark`), but needs the `xgboost`, `pyarrow` and `scikit-learn` pip packages on the
  driver and both workers (a Docker image change that needs owner approval) and runs training in barrier mode across
  workers. I tried it in the sandbox and it did not run (Arrow on the sandbox's Java 21 crashed a Python worker), so no
  XGBoost code is included: it would be untested.
- **LightGBM / CatBoost:** need extra JARs (SynapseML) or packages on the cluster; same approval issue, nothing tested.
- **Spark's own `GBTClassifier`** is the boosted-tree option that needs no changes to the cluster.

## 3. Leakage audit

Allow-list of source columns (`FEATURE_SOURCE_COLUMNS`): `FlightDate` (calendar key only), `CRSDepTime`, `CRSArrTime`,
`CRSElapsedTime`, `Distance`, `Reporting_Airline`, `DayOfWeek`, `Origin`, `Dest`, `DepTimeBlk`, `HolidayPeriod`.
A runtime guard rejects anything in `FORBIDDEN_SOURCE_COLUMNS` (arrival/departure delay and actual times, taxi/wheels,
delay causes, cancellation and diversion outcome fields, `Tail_Number`, any `Div*` column). A test also checks every
derived feature has a provenance entry within the allow-list and that no outcome column is present in the frame given
to the models.

| Feature | Built from | Why it is pre-departure |
|---|---|---|
| time sin/cos, scheduled hours | scheduled HHMM (2400 maps to minute 0) | schedule |
| elapsed time, distance, carrier, day of week | schedule | schedule |
| holiday flag / category | calendar file | known in advance |
| origin-hour demand (optional) | count of scheduled flights, **including cancelled/diverted** | schedule; excluding cancelled rows would leak an outcome |
| history rates (optional): 5 base keys, 5 extended keys (E8), 3 short-memory keys over the last 7 dates (E9) | labels of **strictly earlier dates** | outcomes of past flights |
| destination-hour, origin-day, previous-hour demand (E10) | counts of scheduled flights, **including cancelled/diverted** | schedule only |
| scheduled speed, log distance (E11) | distance and scheduled block time | schedule only; non-positive block time gives a null that is imputed with the fitting-window median |

History rules: a fitting row on date D uses only dates before D (all earlier dates, or the trailing N dates before D);
validation rows use rates from the training window only; test rows use rates from train+validation only; a row's own
label is never used. Preprocessing (imputer medians, category indexers) is fitted on the fitting window only; unseen
categories map to an extra bucket.

## 4. Output interface

Preserved (same names and columns): `metrics.csv`, `validation_threshold_search.csv`, `confusion_counts.csv`,
`random_forest_feature_importance_top30.csv`, `class_counts.csv`, `run_summary.txt`, plus the extras of the earlier
script (`eligibility_counts.csv`, `holiday_category_counts.csv`, `hyperparameter_search.csv`, `baseline_metrics.csv`,
`ranking_metrics.csv`, `metrics_by_holiday_slice.csv`, `ablation_validation_metrics.csv`, `preprocessing_metadata.json`).

Documented changes: `validation_threshold_search.csv` has 89 candidate thresholds instead of 58 (wider grid, same
columns). With `--extra-models`, the extra families appear as additional `Model` values in `metrics.csv`,
`confusion_counts.csv`, `validation_threshold_search.csv`, `hyperparameter_search.csv`, `ranking_metrics.csv`,
`metrics_by_holiday_slice.csv` and `threshold_transfer.csv` (LR and RF rows are unchanged and always present), and their
fitted models are saved under `.../models/gbt` and `.../models/mlp`. `experiment_comparison.csv` has four extra columns for the
new feature factors. `preprocessing_metadata.json` lists `history_groups` instead of `history_keys`. New files: `experiment_comparison.csv`, `validation_metrics.csv`, `frozen_config.json`,
`threshold_transfer.csv`. In `--mode select` the test row of `class_counts.csv` is blank and `metrics.csv` is not written.

## 5. Commands

Synthetic/local tests (no data, no cluster; roughly 15-20 minutes because each pipeline run issues thousands of small
Spark jobs):

```powershell
docker exec -it master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
docker exec -it master bash -lc 'spark-submit --master "local[2]" --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false --driver-memory 2g /src/tests/test_ml_pipeline.py'
```

Real-data development run on the owner's cluster (writes only under `.../experiments/ml_redesign`, never touches the
test window):

```powershell
docker exec -it master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 512m --executor-memory 512m --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag trial1'
```

Smaller first trial if the cluster is slow: add `--experiments E0_reference,E8_extended_history,E9_recent7_history`.
Each experiment is a full feature build plus a Logistic Regression fit on about 1.07 M rows, so the full list of ten
feature experiments is the slowest part; nothing here has been timed on the cluster.

Add the extra models only after the plain select run works: `--extra-models gbt,mlp` (the neural network and the boosted
trees are the most memory-hungry parts; if executors die, drop `mlp` first, then reduce `GBT_CONFIGS` in the script).

Full candidate run (select, freeze, single test evaluation), still in the experiments namespace:

```powershell
docker exec -it master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 512m --executor-memory 512m --num-executors 2 --executor-cores 1 /src/03_ml.py --mode all --run-tag candidate1'
```

Production overwrite, only after review and after backing up `C:\bda\results\ml` and the HDFS models folder:

```powershell
docker exec -it master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 512m --executor-memory 512m --num-executors 2 --executor-cores 1 /src/03_ml.py --mode all --namespace production --confirm-overwrite-baseline'
```

Note this differs from the command in the handover: the bare command now writes to the experiments namespace.

## 6. Limitations

- No real-data result exists yet; whether any experiment helps is unknown until the select run is done.
- One season, three months. Validation (mostly Thanksgiving week) and test (Christmas/New Year) are different regimes,
  so a validation win may not transfer to the test window.
- Holiday features cannot show a Christmas effect; they extrapolate from other holidays.
- No weather, inbound aircraft or live network state (not known before departure).
- Frozen history tables go stale across the 9-day validation and 28-day test windows.
- The F1 criterion yields low-threshold, high-recall operating points (see A4).
- The AUC-PR margin (0.002) is a heuristic.
- The largest known lever is outside the current rules and **not implemented**: information from earlier flights on the
  same day (the same aircraft's previous leg, or delays at the origin in the preceding hours). It would need `Tail_Number`
  and exact departure times in the clean Parquet (an upstream change), a defined prediction time (for example two hours
  before departure), and either walk-forward evaluation or a protocol change, because the agreed rule forbids using test
  labels for validation/test features. It needs the owner's and the instructor's decision.
- The short-memory rates (E9) are fitted per window as frozen tables like the others, so for validation and test rows they
  describe the last 7 days of the fitting window, not the 7 days before each row.
- GBT/MLP memory use and runtime on 1.77 M rows with 512 MB executors are untested.
- Runtime and 512 MB memory behaviour of seven feature builds on 1.77 M rows are untested; the select run is
  substantially heavier than the baseline's single feature stage.

## 7. Decisions needed from the owner

1. Is it acceptable that the bare command now writes to the experiments namespace and production needs two extra flags?
2. Please back up the baseline results and models before any production run.
3. The handover refers to `docs/FULL_MLLIB_PIPELINE_REBUILD_HANDOVER.md` and a `reference/` folder
   (`current_candidate_metrics.csv`, `current_candidate_split_counts.csv`, `CURRENT_RUN_NOTES.md`); neither was in the
   zip I received (it contains `docs/HANDOVER_TEMPORAL_HISTORY_ISSUE.md`). Baseline numbers here are from the handover tables.
4. `README.md` still describes the earlier issue-only task; it was not edited.
5. Keep F1 as the threshold criterion, or switch to a recall- or top-k-based rule for the report?
6. Should the earlier-flights lever (see Limitations) be pursued? It is the only change likely to move the scores a lot.

## 8. Test evidence (synthetic/local)

See `docs/TEST_RUN_LOG.md`: 24 of 24 synthetic/local tests passed with a 2 GB driver. The cluster driver has 512 MB; that is untested and flagged there.
