# Operations model comparison (OPS V2)

## Evaluation boundary

All model choices below were made from the train, validation, and four rolling-origin validation folds. The sealed test split was not read for Phase 1 or Phase 2. The freeze artifact was committed before the authorized final test export and evaluation. The one post-freeze run is complete and recorded below; its results did not change the frozen choice.

## Phase 1: boosted models

The accepted UTC-correct feature set (variant d, all 16 operations features) was used. Validation contains 173,549 rows with delayed-flight prevalence 0.17013.

| Candidate | Validation PR-AUC | ROC-AUC | Top-10% precision | Top-10% lift |
|---|---:|---:|---:|---:|
| Frozen UTC logistic regression | 0.28784 | 0.65166 | 0.35494 | 2.086 |
| Logistic regression with explicit ops interactions | 0.28314 | — | — | — |
| LightGBM | 0.29921 | 0.65643 | 0.36474 | 2.144 |
| CatBoost | 0.30118 | 0.66442 | 0.36860 | 2.167 |
| XGBoost | **0.30590** | **0.66635** | **0.37338** | **2.195** |

XGBoost improved validation PR-AUC by 0.01806 over frozen UTC LR. The paired date-block bootstrap gain was 0.01815, 95% CI [0.01031, 0.02766]. Its mean rolling-fold PR-AUC gain over UTC LR was 0.03103 (SD 0.02544). The Phase-1 decision gate passed. Spark evaluator parity on 173,549 rows gave PR-AUC 0.30562 and ROC-AUC 0.66635.

The requested explicit LR interactions (operations backlog share × departure-hour block and delay share × carrier) scored 0.28314 validation PR-AUC, below the UTC LR baseline. Thus the boosted-model gain is not explained by those simple interactions. XGBoost factor-removal refits showed the largest small gains from `RouteEarlierToday` (+0.00292 PR-AUC) and six-hour state summaries (+0.00165); missingness/count masks contributed +0.00093. Removing the explicit interaction columns changed PR-AUC by only +0.00026, and removing trend features or one-hour windows did not reduce the score. These are single validation refits and should be read as directional rather than precise causal attribution.

Grouped permutation importance for XGBoost (five shuffles per group) showed validation PR-AUC drops of 0.09052 for operations, 0.03419 for recent history, 0.02002 for categoricals, 0.00362 for schedule fields, 0.00274 for demand, and 0.00140 for explicit interactions. The train-to-validation AP gap was 0.08182, so overfitting remains a limitation.

## Phase 2: deep model

The 12-configuration search selected a compact convolutional model under the stated 0.002 simplicity margin. Its selected single-search validation PR-AUC was 0.30287 (absolute best 0.30394). The five-seed logit ensemble scored 0.30414 PR-AUC, ROC-AUC 0.66563, and top-10% precision 0.37021 (lift 2.176). Per-seed best AP mean was 0.29978 (SD 0.00181).

| Evidence | DL | XGBoost reference | Difference |
|---|---:|---:|---:|
| Validation PR-AUC | 0.30414 | 0.30590 | -0.00176 |
| Validation top-10% precision | 0.37021 | 0.37338 | -0.00317 |
| Rolling mean PR-AUC gain (four folds) | — | reference | -0.03826 (SD 0.02281) |

The validation date-block bootstrap for DL minus XGBoost had mean -0.00143 and 95% CI [-0.00977, 0.00699]. Fold gains for weeks starting Oct 28, Nov 4, Nov 11, and Nov 18 were -0.02516, -0.07187, -0.03296, and -0.02304 respectively. DL also had lower top-10% precision on each fold. Although ablations show the embeddings and sequence model contribute to the DL score, the full model did not meet the required validation-and-fold rule. Phase 2 therefore fails, and DL is not eligible for test scoring.

| Ablation | Ensemble validation PR-AUC | Drop vs full |
|---|---:|---:|
| Full model | 0.30414 | — |
| No auxiliary losses | 0.30334 | 0.00080 |
| No state streams | 0.30285 | 0.00129 |
| No LR offset | 0.30357 | 0.00057 |
| No embeddings | 0.29884 | 0.00530 |
| GRU state encoder | 0.30381 | 0.00033 |
| Plain MLP | 0.29731 | 0.00683 |

## Frozen choice and limitations

The final candidate is XGBoost, with UTC-correct variant d and all 16 operations features. The Phase-1 pass authorized the one final test evaluation; Phase 2 did not. The model is a ranking aid: validation top-decile precision is about 37%, not a reliable per-flight guarantee. Fold performance varies substantially by week, and the train-validation gap indicates overfitting risk. The completed test result is below; it did not change the model choice.

## FOURTH DESIGN EVALUATION

The committed freeze selected XGBoost with the UTC-correct feature set and all 16 operations features. The Phase 2 deep model was not run on test because it failed the validation/fold gate. The post-freeze host export reconciled 1,067,747 train rows, 173,549 validation rows, and 523,983 test rows; it found zero duplicate technical flight keys and zero nulls in non-label exported features. Test prevalence was 0.21318. Scoring used the frozen XGBoost configuration and iteration count. The UTC-LR comparator was refit on train plus validation using the same UTC feature export.

| Model | Test PR-AUC | Test ROC-AUC | Top-10% precision | Top-10% recall | Top-10% lift |
|---|---:|---:|---:|---:|---:|
| UTC-LR refit reference | 0.40524 | 0.68708 | 0.50134 | 0.23517 | 2.352 |
| Frozen XGBoost | **0.41980** | **0.70198** | **0.52349** | **0.24556** | **2.456** |

The paired 28-date block bootstrap for XGBoost minus UTC-LR test PR-AUC had mean **+0.01455**, with 1,000 replicates and 95% CI **[+0.00689, +0.02273]**. Both scores exceed the 0.21318 test prevalence baseline. The top decile selected by XGBoost contains 52.35% delayed flights, versus 50.13% for UTC-LR and 21.32% overall.

| Test slice | Flights | Delay prevalence | UTC-LR PR-AUC | XGBoost PR-AUC |
|---|---:|---:|---:|---:|
| Christmas/New Year | 225,777 | 0.26799 | 0.47308 | **0.48138** |
| Regular day | 298,206 | 0.17169 | 0.30762 | **0.33147** |

XGBoost ranked better than UTC-LR on this test run, including both reported slices. This is a descriptive result for the already-used test period, not evidence for further tuning or a new model-selection decision. PR-AUC depends on prevalence, and test prevalence is higher than validation prevalence; compare the two models on the same test rows rather than interpreting the absolute validation-to-test change as a clean measure of generalization. The validation train-to-validation gap and week-to-week fold variation remain material limitations. The test window has now been used for the fourth design evaluation; use a new time period for further generalization claims.

Execution markers: `HOST_EXPORT_STATUS=SUCCEEDED` and `FOURTH_DESIGN_EVALUATION_STATUS=SUCCEEDED`. The export required an HDFS-to-host copy because the cluster's default filesystem treated the script's absolute `/results` Parquet output as an HDFS path. The exporter now writes its host output with an explicit `file://` URI. The canonical HDFS export remains under `/user/root/flights/experiments/ml_redesign/ops_v2/host_export/test`.

