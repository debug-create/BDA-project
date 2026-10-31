# Operations-aware flight-delay prediction at T−120 minutes

## Executive result

The feasibility gate passed, the implementation and validation-only selection completed on YARN, and one frozen final run scored the test window. Operations-aware features materially improved ranking over the existing schedule-only design. This is a changed prediction problem: predict at **T = scheduled departure minus 120 minutes**, using schedule information plus operational events observable by T.

The test window has now been used for a **SECOND DESIGN EVALUATION**. No feature, model, hyperparameter, threshold, or lead-time decision in this work used its results. Nevertheless, the earlier schedule-only test evaluation means the window is no longer a pristine one-use benchmark; external replication is required.

| Status | Design | Model | Test ROC-AUC | Test PR-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SECOND DESIGN EVALUATION | schedule-only (reused) | Logistic Regression | 0.6266 | 0.3036 | 0.3787 | 0.2436 | 0.8498 | 0.4056 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | Logistic Regression | **0.6906** | **0.4103** | **0.4200** | **0.2966** | 0.7191 | **0.5766** |
| SECOND DESIGN EVALUATION | schedule-only (reused) | Random Forest | 0.6094 | 0.2869 | 0.3722 | 0.2364 | 0.8742 | 0.3713 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | Random Forest | **0.6667** | **0.3824** | **0.4014** | **0.2816** | 0.6985 | **0.5558** |

The LR PR-AUC gain is +0.1067 using exact reported metrics; the paired 1,000-resample date-block histogram estimate is +0.1069 with 95% CI [0.0723, 0.1305]. RF gains +0.0955 exactly; its histogram estimate is +0.0959 with CI [0.0641, 0.1191].

## Governance and split

- Data: existing cleaned Parquet only; no download or source-data rewrite.
- Train: 2024-10-01 through 2024-11-24; validation: 2024-11-25 through 2024-12-03; test: 2024-12-04 through 2024-12-31.
- Eligible rows: train 1,067,747; validation 173,549; test 523,983. Cancelled/diverted target flights remain excluded.
- Primary selection metric: validation PR-AUC; existing replacement margin: +0.002.
- Outputs are isolated under local and HDFS `experiments/ml_redesign/asof_ops/`. `actual_final`, `actual_xgb_combined`, `diagnostics`, original models, and baseline outputs were not overwritten.
- Executor memory was 512 MB, two one-core executors; driver memory was 1,536 MB for every real-data YARN run.

## As-of definition and implementation

For target flight F, T is the scheduled-departure local hour minus the configured lead. Hourly aggregation is conservative: only event hours **strictly before T's hour** are eligible.

- Departed: scheduled departure plus `DepDelay` is before T; delay is then observable.
- Landed: approximated scheduled arrival plus `ArrDelay` is before T; arrival delay is then observable.
- Backlog: scheduled before T but not departed by T, including cancelled/diverted flights; no cancellation cause is used.
- Cancelled/diverted flights never enter delay averages, only backlog counts.
- Target outcomes are excluded. Because the clean table has no flight identifier, departures at least the entire lead early and arrivals before their own scheduled-departure hour are conservatively excluded so a target cannot supply its own outcome.
- Scheduled arrival date is approximated as `FlightDate + 1 day` when `CRSArrTime < CRSDepTime`, then `ArrDelay` is added.
- Empty windows have observed count zero. Means/shares use 20 training-fitted pseudo-events toward train-only priors; no validation/test outcome enters a prior.
- Operations are pre-aggregated by airport/carrier, date, and hour. All six joins passed one-to-one row-count checks. An HDFS checkpoint truncates Spark lineage before the large E6 history plan; it changes neither rows nor values.

The clean handoff contains airport codes but no UTC offsets or time-zone table. Airport-level origin and inbound tables remain isolated on each airport's local clock. Cross-airport carrier-network and destination timing compare the available local wall-clock hour and are therefore an approximation, not a UTC-certified global as-of ordering. This limitation is material and should be fixed with airport time zones before deployment.

### New feature factor

| Feature | Definition at T−lead | Missing/empty handling |
|---|---|---|
| `OpsOriginDepartedCount3h` | departed at origin in prior 3 complete hours | 0 |
| `OpsOriginMeanDepDelay3h` | mean known departure delay in that window | 20-event smoothed train mean |
| `OpsOriginDepDelay15Share3h` | share of known departures delayed ≥15 | 20-event smoothed train share |
| `OpsOriginBacklogCount3h` | scheduled-before-T origin flights not departed | 0 |
| `OpsOriginBacklogShare3h` | backlog / scheduled-before-T origin flights | 20-event smoothed train share |
| `OpsOriginInboundLandedCount3h` | flights landed at origin in prior 3 complete hours | 0 |
| `OpsOriginInboundMeanArrDelay3h` | mean known arrival delay at origin | 20-event smoothed train mean |
| `OpsOriginInboundArrDelay15Share3h` | delayed-arrival share at origin | 20-event smoothed train share |
| `OpsCarrierDepartedCountToday` | carrier departures observed so far that local day | 0 |
| `OpsCarrierMeanDepDelayToday` | carrier mean known departure delay | 20-event smoothed train mean |
| `OpsCarrierDepDelay15ShareToday` | carrier known departure-delay share | 20-event smoothed train share |
| `OpsCarrierScheduledBeforeCountToday` | carrier flights scheduled before T | 0 |
| `OpsCarrierBacklogShareToday` | carrier backlog / scheduled-before-T | 20-event smoothed train share |
| `OpsDestLandedCount3h` | flights landed at target destination in prior 3 complete hours | 0 |
| `OpsDestMeanArrDelay3h` | destination mean known arrival delay | 20-event smoothed train mean |
| `OpsDestArrDelay15Share3h` | destination delayed-arrival share | 20-event smoothed train share |

## Phase 1 — feasibility and gate

### Coverage

Coverage is the share of eligible flights with at least five observed origin departures in the three-hour window.

| Scheduled hour | Train coverage | Validation coverage | Scheduled hour | Train coverage | Validation coverage |
|---:|---:|---:|---:|---:|---:|
| 00 | 91.1% | 91.3% | 12 | 89.1% | 88.7% |
| 01 | 44.3% | 55.9% | 13 | 89.0% | 89.7% |
| 02 | 45.5% | 57.8% | 14 | 89.6% | 89.9% |
| 03 | 6.1% | 28.6% | 15 | 92.2% | 92.6% |
| 04 | 0.0% | 7.7% | 16 | 93.9% | 93.9% |
| 05 | 10.2% | 13.3% | 17 | 91.8% | 92.0% |
| 06 | 1.0% | 2.0% | 18 | 92.1% | 91.7% |
| 07 | 2.2% | 1.4% | 19 | 95.6% | 94.5% |
| 08 | 81.3% | 83.3% | 20 | 97.9% | 97.7% |
| 09 | 94.8% | 94.6% | 21 | 98.8% | 97.5% |
| 10 | 94.1% | 94.7% | 22 | 99.1% | 98.2% |
| 11 | 92.3% | 92.2% | 23 | 97.7% | 95.3% |

Early-morning sparsity is handled by explicit counts plus smoothed priors; it is not silently imputed from validation/test outcomes.

### Training prevalence by operational bucket

| Feature | Bucket | Flights | Delayed ≥15 | Prevalence |
|---|---|---:|---:|---:|
| origin mean departure delay | ≤0 min | 132,190 | 13,612 | 10.30% |
| origin mean departure delay | 1–15 min | 892,850 | 122,075 | 13.67% |
| origin mean departure delay | 16–60 min | 41,653 | 10,503 | 25.22% |
| origin mean departure delay | >60 min | 1,054 | 545 | 51.71% |
| origin backlog share | (0, 0.10] | 1,019,108 | 132,027 | 12.96% |
| origin backlog share | (0.10, 0.25] | 45,310 | 12,437 | 27.45% |
| origin backlog share | (0.25, 0.50] | 2,911 | 1,979 | 67.98% |
| origin backlog share | >0.50 | 418 | 292 | 69.86% |

### Validation power of each feature alone

| Feature | PR-AUC | Top-10% precision | Lift |
|---|---:|---:|---:|
| Origin departed count 3h | 0.1876 | 0.1958 | 1.151 |
| Origin mean departure delay 3h | 0.2138 | 0.2542 | 1.494 |
| Origin departure-delay share 3h | 0.2335 | 0.2871 | 1.688 |
| Origin backlog count 3h | 0.2241 | 0.2577 | 1.515 |
| Origin backlog share 3h | **0.2346** | **0.2873** | **1.689** |
| Origin inbound landed count 3h | 0.1875 | 0.1904 | 1.119 |
| Origin inbound mean arrival delay 3h | 0.1961 | 0.2121 | 1.247 |
| Origin inbound arrival-delay share 3h | 0.2069 | 0.2365 | 1.390 |
| Carrier departed count today | 0.1868 | 0.1866 | 1.097 |
| Carrier mean departure delay today | 0.1895 | 0.1692 | 0.994 |
| Carrier departure-delay share today | 0.1966 | 0.1976 | 1.162 |
| Carrier scheduled-before count today | 0.1878 | 0.1886 | 1.108 |
| Carrier backlog share today | 0.2089 | 0.2303 | 1.354 |
| Destination landed count 3h | 0.1825 | 0.1854 | 1.089 |
| Destination mean arrival delay 3h | 0.1912 | 0.1921 | 1.129 |
| Destination arrival-delay share 3h | 0.2004 | 0.2194 | 1.290 |

### Quick frozen-E6 LR test and gate

| Design | Validation ROC-AUC | Validation PR-AUC | Top-10% precision | Top-10% lift |
|---|---:|---:|---:|---:|
| schedule-only E6 | 0.6142 | 0.2422 | 0.2933 | 1.724 |
| operations-aware (T−120), E6 + operations | **0.6508** | **0.2896** | **0.3566** | **2.096** |

Gain over the stipulated 0.2422 reference: **+0.04736**, exceeding the +0.010 gate. Decision: **PASS**.

## Phase 2 — implementation and tests

`E12_asof_ops` adds operations to the old E0 feature set; `E13_e6_plus_asof` adds them to frozen E6. Existing experiment defaults exclude both, so old behavior is unchanged unless an as-of experiment is explicitly named. The lead is constrained to 60, 120, or 180 minutes.

Synthetic local-Spark tests cover: future outcome/label/cancellation invariance; observed-event sensitivity; target-own-outcome invariance including an extreme early departure; one-to-one row preservation; midnight arrival rollover; airport-local clock isolation; and a deliberately leaky comparator that the invariance mutation detects. The dedicated test marker is `ASOF_OPS_TESTS_STATUS=PASSED`. The whole pre-existing ML suite is recorded in `test_ml_pipeline_full.log`; interrupted/failed verification attempts remain retained and are not represented as passes.

## Phase 3 — validation-only selection

The test window was not read by any run in this section.

### Factor progression at T−120

| Design | Model/configuration | Validation PR-AUC | Interpretation |
|---|---|---:|---|
| schedule-only frozen E6 | LR, reg=0.001 | 0.2422 | comparison baseline |
| E12 operations + E0 | tuned LR, reg=0.001 | 0.2774 | operations factor helps |
| E13 operations + frozen E6 | tuned LR, reg=0.001 | **0.2895** | selected primary spec |
| E13 operations + frozen E6 | RF depth 7, 30 trees | 0.2745 | retained required model |

The pipeline's internal E0 reference was 0.2222; the table above uses the requested frozen-E6 schedule-only comparison, 0.2422.

### Lead-time sensitivity (E13)

| Lead | LR PR-AUC | LR F1 threshold | RF PR-AUC | RF F1 threshold | Role |
|---:|---:|---:|---:|---:|---|
| 60 min | **0.3031** | 0.16 | **0.2844** | 0.15 | sensitivity only |
| 120 min | 0.2895 | 0.16 | 0.2745 | 0.14 in the selection retry; 0.15 in frozen rerun | pre-specified primary |
| 180 min | 0.2797 | 0.16 | 0.2714 | 0.16 | sensitivity only |

T−60 performed best, but the brief defined T−120 as the primary problem and required every primary comparison to use that label. It was therefore not changed after sensitivity analysis.

### Optional nonlinear comparator at T−120

| Model | Configuration | Validation ROC-AUC | Validation PR-AUC | Threshold | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | reg=0.001 | **0.6508** | **0.2895** | 0.16 | 0.3458 |
| Random Forest | depth 7, 30 trees | 0.6336 | 0.2752 | 0.15 | 0.3316 |
| GBT | depth 6, 30 iterations | 0.6487 | 0.2802 | 0.16 | **0.3458** |

GBT lost on the primary metric and was not taken into Phase 4. XGBoost was not attempted under the 512 MB executor constraint; the Spark GBT comparator satisfied the optional nonlinear comparison without changing the container image.

## Phase 4 — single final test evaluation

`frozen_config.json` was written before test outcomes were read. Frozen decisions: E13, T−120, unweighted LR with `regParam=0.001`, LR threshold 0.16, RF depth 7 / 30 trees / minimum node size 100, RF threshold 0.15. The final run was executed once as application `application_1793455790052_0007`.

### Ranking at operational review budgets

| Status | Design | Model | Top fraction | Precision | Recall | Lift |
|---|---|---|---:|---:|---:|---:|
| SECOND DESIGN EVALUATION | schedule-only | LR | 10% | 0.3575 | 0.1679 | 1.677 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | LR | 10% | **0.5083** | **0.2403** | **2.384** |
| SECOND DESIGN EVALUATION | schedule-only | LR | 20% | 0.3248 | 0.3053 | 1.523 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | LR | 20% | **0.4181** | **0.3935** | **1.961** |
| SECOND DESIGN EVALUATION | schedule-only | LR | 30% | 0.3046 | 0.4284 | 1.429 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | LR | 30% | **0.3647** | **0.5143** | **1.711** |
| SECOND DESIGN EVALUATION | schedule-only | RF | 10% | 0.3332 | 0.1567 | 1.563 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | RF | 10% | **0.4873** | **0.2302** | **2.286** |
| SECOND DESIGN EVALUATION | schedule-only | RF | 20% | 0.3003 | 0.2817 | 1.409 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | RF | 20% | **0.4020** | **0.3783** | **1.886** |
| SECOND DESIGN EVALUATION | schedule-only | RF | 30% | 0.2887 | 0.4097 | 1.354 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | RF | 30% | **0.3505** | **0.4943** | **1.644** |

### Christmas/New Year versus regular days

| Status | Design | Model | Slice | Flights | Prevalence | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| SECOND DESIGN EVALUATION | schedule-only | LR | Christmas/New Year | 225,777 | 0.2680 | 0.6230 | 0.3606 | 0.4458 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | LR | Christmas/New Year | 225,777 | 0.2680 | **0.6945** | **0.4805** | **0.4754** |
| SECOND DESIGN EVALUATION | schedule-only | LR | Regular day | 298,206 | 0.1717 | 0.6227 | 0.2537 | 0.3197 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | LR | Regular day | 298,206 | 0.1717 | **0.6611** | **0.3096** | **0.3538** |
| SECOND DESIGN EVALUATION | schedule-only | RF | Christmas/New Year | 225,777 | 0.2680 | 0.6076 | 0.3445 | 0.4414 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | RF | Christmas/New Year | 225,777 | 0.2680 | **0.6722** | **0.4484** | **0.4637** |
| SECOND DESIGN EVALUATION | schedule-only | RF | Regular day | 298,206 | 0.1717 | 0.6132 | 0.2436 | 0.3142 |
| SECOND DESIGN EVALUATION | operations-aware (T−120) | RF | Regular day | 298,206 | 0.1717 | **0.6396** | **0.2893** | **0.3370** |

Christmas/New Year was unseen in fitting as a holiday category. The improvement therefore comes from same-day operational state and transferable schedule/history patterns, not from a learned Christmas coefficient.

### Paired date-block uncertainty

| Status | Model | Schedule-only PR-AUC (histogram) | Operations T−120 PR-AUC (histogram) | Improvement | 95% CI | Dates / resamples |
|---|---|---:|---:|---:|---|---|
| SECOND DESIGN EVALUATION | LR | 0.3033 | 0.4101 | +0.1069 | [0.0723, 0.1305] | 28 / 1,000 |
| SECOND DESIGN EVALUATION | RF | 0.2859 | 0.3818 | +0.0959 | [0.0641, 0.1191] | 28 / 1,000 |

These are paired whole-date resamples with seed 42 and 1,000 score bins. Histogram point estimates differ slightly from exact Spark evaluator values because of binning.

## What ran and what failed

| Phase | Application(s) | Result |
|---|---|---|
| feasibility | `...0013` | deliberately stopped after identifying lineage/materialization issue; log kept |
| feasibility | `...0014` | failed with driver heap while rebuilding baseline; log kept |
| feasibility | `...0015`, `...0016` | checkpoint preparation and quick LR/gate succeeded |
| selection | `...0001` | E12 T−120 succeeded |
| selection | `...0002` | E13 failed at 1,536 MB driver due plan-lineage heap; log kept |
| selection | `...0003` | HDFS-lineage-checkpoint retry succeeded |
| sensitivity | `...0004`, `...0005` | T−60 and T−180 succeeded |
| optional comparator | `...0006` | LR/RF/GBT validation run succeeded |
| final | `...0007` | frozen T−120 LR/RF test evaluation succeeded once |

Application IDs, outcomes, and log paths are in `results/experiments/ml_redesign/asof_ops/yarn_runs.csv`. Exact submitted command forms are in `results/experiments/ml_redesign/asof_ops/exact_commands.txt`. Failed logs were retained, not overwritten.

## Measured versus inferred

| Claim | Status | Evidence / boundary |
|---|---|---|
| Phase-1 operational signal and gate | measured | real cleaned Parquet, YARN, train/validation only |
| E12/E13, LR/RF/GBT and lead sensitivity | measured | real cleaned Parquet, YARN, validation only |
| final LR/RF metrics, rankings and slices | measured | real cleaned Parquet, one frozen Phase-4 run |
| improvement uncertainty | measured approximation | paired date bootstrap from Spark 1,000-bin histograms |
| leakage invariants and midnight behavior | measured | small synthetic local-Spark tests |
| causal holiday effect | **not inferred** | this remains observational prediction |
| future-month/general-production gain | unknown | only Oct–Dec 2024 was evaluated |
| exact cross-time-zone global ordering | unknown/approximated | airport UTC offsets absent from clean data |

## Limitations and decision

1. This is the test window's **SECOND DESIGN EVALUATION**. The large gain and positive date-block CIs are encouraging, but a new untouched period is needed for a clean confirmation.
2. Local scheduled/actual time fields and no airport time-zone reference make cross-airport carrier/destination ordering approximate. Origin-local operational features are the most defensible subset.
3. Arrival date uses the next-day rule when scheduled arrival clock time precedes scheduled departure; multi-day and pathological records are not fully reconstructable.
4. The clean data has no stable flight identifier. Conservative impossible-early-event filters protect a target's own features but may discard a small number of genuinely very-early observations.
5. Hourly conservatism discards events earlier within T's own hour. This reduces leakage risk but loses some valid signal.
6. Thresholds intentionally trade accuracy for delayed-flight recall. Ranking and capacity-based precision/lift are the better operational readout.

Decision: the operations-aware design is a real predictive improvement on this dataset and is suitable as the project's stronger result, with the changed T−120 problem statement and limitations stated exactly as above. It should not be described as production-ready until replicated on a new time window with airport time-zone metadata.
