# Flight-delay model diagnostics

This is a diagnostics-only audit of the frozen `actual_final` run. It does not change `src/03_ml.py`, retrain either saved final model, rerun the frozen final evaluation, or overwrite `actual_final`, `actual_xgb_combined`, or the baseline. Large-data work ran on YARN with a 1536 MB driver and two 512 MB, one-core executors. Test labels had already been seen once; every test-derived table row is labelled exactly **TEST - DESCRIPTIVE ONLY, not used for any decision** and is descriptive only.

PR-AUC/ROC computed by this diagnostic job uses a bounded-memory 1,000-bin score histogram unless a source table explicitly states otherwise. Calibration bins use approximate score deciles (relative quantile error 0.001) to avoid a single-partition global sort on the constrained executors.

## A. Consistency and inventory

### A1. Headline consistency checks

All values were recomputed from saved final-run files; tolerance and pass/fail are explicit.

| Check | Entity | Field | ReportedOrSaved | Recomputed | Result |
| --- | --- | --- | --- | --- | --- |
| metrics.csv vs confusion_counts.csv | LogisticRegression | Precision_class1 | 0.24363100646267408 | 0.24363100646267408 | MATCH |
| metrics.csv vs confusion_counts.csv | LogisticRegression | Recall_class1 | 0.8497891730750293 | 0.8497891730750293 | MATCH |
| metrics.csv vs confusion_counts.csv | LogisticRegression | F1_class1 | 0.37869246496783526 | 0.37869246496783526 | MATCH |
| metrics.csv vs confusion_counts.csv | LogisticRegression | Accuracy | 0.40555895897385985 | 0.40555895897385985 | MATCH |
| metrics.csv vs confusion_counts.csv | RandomForest | Precision_class1 | 0.23641403687160306 | 0.23641403687160306 | MATCH |
| metrics.csv vs confusion_counts.csv | RandomForest | Recall_class1 | 0.8742110775896798 | 0.8742110775896798 | MATCH |
| metrics.csv vs confusion_counts.csv | RandomForest | F1_class1 | 0.3721791759249025 | 0.3721791759249025 | MATCH |
| metrics.csv vs confusion_counts.csv | RandomForest | Accuracy | 0.37125059400782084 | 0.37125059400782084 | MATCH |
| baseline_metrics.csv vs class_counts.csv | MajorityClass_AlwaysNotDelayed | AUC_PR | 0.21318058028600165 | 0.21318058028600165 | MATCH |
| baseline_metrics.csv vs class_counts.csv | MajorityClass_AlwaysNotDelayed | F1_class1 | 0.0 | 0.0 | MATCH |
| baseline_metrics.csv vs class_counts.csv | MajorityClass_AlwaysNotDelayed | Accuracy | 0.7868194197139984 | 0.7868194197139984 | MATCH |
| baseline_metrics.csv vs class_counts.csv | AlwaysDelayed | AUC_PR | 0.21318058028600165 | 0.21318058028600165 | MATCH |
| baseline_metrics.csv vs class_counts.csv | AlwaysDelayed | F1_class1 | 0.3514408056807921 | 0.3514408056807921 | MATCH |
| baseline_metrics.csv vs class_counts.csv | AlwaysDelayed | Accuracy | 0.21318058028600165 | 0.21318058028600165 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.1 | ActualFlaggedFraction | 0.10009485040545209 | 0.10009485040545209 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.1 | Lift | 1.6770571474366482 | 1.6770571474366482 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.2 | ActualFlaggedFraction | 0.20039199745029895 | 0.20039199745029895 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.2 | Lift | 1.5233832204743827 | 1.5233832204743827 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.3 | ActualFlaggedFraction | 0.2998017111242159 | 0.2998017111242159 | MATCH |
| ranking_metrics.csv internal | LogisticRegression@0.3 | Lift | 1.4290470450999087 | 1.4290470450999087 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.1 | ActualFlaggedFraction | 0.1002246256080827 | 0.1002246256080827 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.1 | Lift | 1.5630538894274022 | 1.5630538894274022 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.2 | ActualFlaggedFraction | 0.19999312954809603 | 0.19999312954809603 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.2 | Lift | 1.4087392969346773 | 1.4087392969346773 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.3 | ActualFlaggedFraction | 0.3025403495914944 | 0.3025403495914944 | MATCH |
| ranking_metrics.csv internal | RandomForest@0.3 | Lift | 1.354326206410408 | 1.354326206410408 | MATCH |
| metrics_by_holiday_slice.csv internal | LogisticRegression/ALL TEST | Prevalence | 0.21318058028600165 | 0.21318058028600165 | MATCH |
| metrics_by_holiday_slice.csv internal | LogisticRegression/Christmas/New Year period | Prevalence | 0.26798566727346007 | 0.26798566727346007 | MATCH |
| metrics_by_holiday_slice.csv internal | LogisticRegression/Regular day | Prevalence | 0.1716866863845798 | 0.1716866863845798 | MATCH |
| metrics_by_holiday_slice.csv internal | RandomForest/ALL TEST | Prevalence | 0.21318058028600165 | 0.21318058028600165 | MATCH |
| metrics_by_holiday_slice.csv internal | RandomForest/Christmas/New Year period | Prevalence | 0.26798566727346007 | 0.26798566727346007 | MATCH |
| metrics_by_holiday_slice.csv internal | RandomForest/Regular day | Prevalence | 0.1716866863845798 | 0.1716866863845798 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | validation/LogisticRegression | AUC_PR | 0.24215 | 0.2421530474820992 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | validation/RandomForest | AUC_PR | 0.22366 | 0.22366307239535177 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | test/LogisticRegression | AUC_PR | 0.30359 | 0.30358960979984206 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | test/RandomForest | AUC_PR | 0.28687 | 0.28687309823937623 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | test/LogisticRegression | AUC_ROC | 0.62664 | 0.6266447181476484 | MATCH |
| ACTUAL_DATA_ML_RESULTS.md headline | test/RandomForest | AUC_ROC | 0.60936 | 0.6093565155910344 | MATCH |

#### A1 summary

```json
{
  "checks": 38,
  "mismatches": 0,
  "mismatch_rows": []
}
```

### A2. Full frozen-run contents

The requested files follow in full, not excerpts.

#### frozen_config.json

```json
{
  "extra_models": {},
  "feature_spec": {
    "extra_demand": true,
    "history_set": "extended",
    "name": "E6_combined_winners",
    "recent_days": 7,
    "sched_extras": false,
    "use_demand": false,
    "use_history": true,
    "use_holiday": true,
    "warmup_days": 7,
    "window_days": null
  },
  "feature_spec_label": "history=expanding,warmup=7,holiday=on,demand=off,historyset=extended,recent=7,extrademand=on,schedextras=off",
  "frozen_before_test_scoring": true,
  "logistic_regression": {
    "class_weighted": false,
    "label": "regParam=0.001",
    "maxIter": 30,
    "regParam": 0.001
  },
  "mode": "all",
  "random_forest": {
    "featureSubsetStrategy": "onethird",
    "label": "maxDepth=7,minInstancesPerNode=100,numTrees=30",
    "maxBins": 32,
    "maxDepth": 7,
    "minInstancesPerNode": 100,
    "numTrees": 30,
    "subsamplingRate": 0.5
  },
  "seed": 42,
  "selection_margin_auc_pr": 0.002,
  "thresholds_from_validation_F1": {
    "LogisticRegression": 0.11,
    "RandomForest": 0.12
  },
  "windows": {
    "test": [
      "2024-12-04",
      "2024-12-31"
    ],
    "train": [
      "2024-10-01",
      "2024-11-24"
    ],
    "validation": [
      "2024-11-25",
      "2024-12-03"
    ]
  }
}
```

#### experiment_comparison.csv

| Experiment | Description | UsesHistory | HistoryWindowDays | WarmupDays | UsesHolidayFeatures | UsesScheduledDemand | HistorySet | RecentWindowDays | UsesExtraDemand | UsesScheduleExtras | Model | Role | ValidationAUC_ROC | ValidationAUC_PR | ValidationBestF1 | ValidationBestThreshold | DeltaAUC_PR_vs_Reference | BeatsReferenceByMargin | Selected | Seconds | TestAUC_PR | TestF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_reference | none (previous script's feature set) | yes | all earlier dates | 0 | yes | no | base |  | no | no | LogisticRegression(regParam=0.01) | feature experiment | 0.5942832720922261 | 0.2221810651791513 | 0.3128226291059418 | 0.1 | 0.0 | reference | no | 87.2 |  |  |
| E3_history_warmup7 | drop first 7 fitting days as rows (still used as history) | yes | all earlier dates | 7 | yes | no | base |  | no | no | LogisticRegression(regParam=0.01) | feature experiment | 0.5965366585213676 | 0.2249028850621771 | 0.31399766958812503 | 0.1 | 0.0027218198830258056 | yes | no | 62.7 |  |  |
| E8_extended_history | add 5 more past-only rate keys (airline x origin, origin x weekday, airline x block, dest x arrival hour, airline x route) | yes | all earlier dates | 0 | yes | no | extended |  | no | no | LogisticRegression(regParam=0.01) | feature experiment | 0.6028695049666478 | 0.23261175475070137 | 0.31540189314936623 | 0.11 | 0.010430689571550078 | yes | no | 112.8 |  |  |
| E9_recent7_history | add short-memory (last 7 dates) rates for airline, origin, origin x block | yes | all earlier dates | 0 | yes | no | base | 7 | no | no | LogisticRegression(regParam=0.01) | feature experiment | 0.5979802056140554 | 0.2255673848182342 | 0.3137812359718689 | 0.11 | 0.003386319639082924 | yes | no | 121.5 |  |  |
| E10_extra_demand | add scheduled dest-hour, origin-day and previous-hour demand | yes | all earlier dates | 0 | yes | no | base |  | yes | no | LogisticRegression(regParam=0.01) | feature experiment | 0.5970434078382478 | 0.22449925826111738 | 0.31307632134137187 | 0.11 | 0.0023181930819660967 | yes | no | 61.2 |  |  |
| E6_combined_winners | combination of every challenger that beat the reference by the margin | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | LogisticRegression(regParam=0.01) | feature experiment | 0.6123414305186805 | 0.24054038140803669 | 0.319564913004709 | 0.11 | 0.018359316228885397 | yes | feature spec | 194.5 |  |  |
| tune_LR | L2 strength grid | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | LogisticRegression(regParam=0.001) | model tuning | 0.6142441026690141 | 0.24215281561444046 |  |  | 0.019971750435289176 |  | final candidate | 38.4 | 0.30358960979984206 | 0.37869246496783526 |
| tune_LR | L2 strength grid | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | LogisticRegression(regParam=0.01) | model tuning | 0.612341942814835 | 0.24053999908239962 |  |  | 0.01835893390324833 |  | no | 37.3 |  |  |
| tune_LR | L2 strength grid | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | LogisticRegression(regParam=0.1) | model tuning | 0.6099988617595452 | 0.23916080999358483 |  |  | 0.01697974481443354 |  | no | 35.3 |  |  |
| E7_LR_class_weighted | balanced class weights | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | LogisticRegression(regParam=0.001,classWeight=balanced) | model tuning | 0.6158833052693575 | 0.24356219399686088 |  |  | 0.021381128817709594 |  | no | 47.1 |  |  |
| tune_RF | small RF grid | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | RandomForest(maxDepth=5,minInstancesPerNode=100,numTrees=30) | model tuning | 0.6002852112012491 | 0.21777887332522217 |  |  | -0.004402191853929122 |  | no | 48.9 |  |  |
| tune_RF | small RF grid | yes | all earlier dates | 7 | yes | no | extended | 7 | yes | no | RandomForest(maxDepth=7,minInstancesPerNode=100,numTrees=30) | model tuning | 0.6053473243785868 | 0.2236654027521047 |  |  | 0.0014843375729534092 |  | final candidate | 71.1 | 0.28687309823937623 | 0.3721791759249025 |

#### hyperparameter_search.csv

| Model | Config | ValidationAUC_ROC | ValidationAUC_PR | Selected |
| --- | --- | --- | --- | --- |
| LogisticRegression | regParam=0.001 | 0.6142441026690141 | 0.24215281561444046 | yes |
| LogisticRegression | regParam=0.01 | 0.612341942814835 | 0.24053999908239962 | no |
| LogisticRegression | regParam=0.1 | 0.6099988617595452 | 0.23916080999358483 | no |
| LogisticRegression | regParam=0.001,classWeight=balanced | 0.6158833052693575 | 0.24356219399686088 | no |
| RandomForest | maxDepth=5,minInstancesPerNode=100,numTrees=30 | 0.6002852112012491 | 0.21777887332522217 | no |
| RandomForest | maxDepth=7,minInstancesPerNode=100,numTrees=30 | 0.6053473243785868 | 0.2236654027521047 | yes |

#### validation_threshold_search.csv

| Model | Threshold | TP | FP | FN | Precision_class1 | Recall_class1 | F1_class1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.02 | 29526 | 144016 | 0 | 0.17013748833135495 | 1.0 | 0.29079914117438493 |
| LogisticRegression | 0.03 | 29509 | 143690 | 17 | 0.17037627238032552 | 0.9994242362663416 | 0.291123443087927 |
| LogisticRegression | 0.04 | 29420 | 141738 | 106 | 0.17188796316853433 | 0.9964099437783648 | 0.29319726535249446 |
| LogisticRegression | 0.05 | 29050 | 137205 | 476 | 0.17473158701993924 | 0.9838786154575628 | 0.29676015547984735 |
| LogisticRegression | 0.06 | 28348 | 130374 | 1178 | 0.17860158012121824 | 0.9601029601029601 | 0.30117717054098847 |
| LogisticRegression | 0.07 | 27356 | 121868 | 2170 | 0.1833217176861631 | 0.9265054528212423 | 0.3060811188811189 |
| LogisticRegression | 0.08 | 26120 | 112266 | 3406 | 0.18874741664619255 | 0.8846440425387794 | 0.31111534613368913 |
| LogisticRegression | 0.09 | 24655 | 102146 | 4871 | 0.19443852966459255 | 0.8350267560793877 | 0.3154285568072054 |
| LogisticRegression | 0.1 | 23037 | 92151 | 6489 | 0.19999479112407542 | 0.7802275960170697 | 0.31837970065093907 |
| LogisticRegression | 0.11 | 21344 | 82149 | 8182 | 0.20623617056226026 | 0.7228883018356702 | 0.32091656079206726 |
| LogisticRegression | 0.12 | 19551 | 72886 | 9975 | 0.21150621504375952 | 0.6621621621621622 | 0.32060542951550874 |
| LogisticRegression | 0.13 | 17800 | 64049 | 11726 | 0.21747364048430648 | 0.6028584975953397 | 0.3196408529741863 |
| LogisticRegression | 0.14 | 16086 | 56015 | 13440 | 0.2231037017517094 | 0.5448079658605974 | 0.3165694156080569 |
| LogisticRegression | 0.15 | 14542 | 48661 | 14984 | 0.23008401499928802 | 0.4925150714624399 | 0.31364513798272387 |
| LogisticRegression | 0.16 | 13007 | 42034 | 16519 | 0.23631474718845952 | 0.44052699315857213 | 0.3076140811427625 |
| LogisticRegression | 0.17 | 11598 | 36116 | 17928 | 0.24307331181623842 | 0.39280634017476124 | 0.30031071983428276 |
| LogisticRegression | 0.18 | 10324 | 30929 | 19202 | 0.2502605871088163 | 0.34965792860529704 | 0.29172494666497123 |
| LogisticRegression | 0.19 | 9146 | 26425 | 20380 | 0.2571195636895224 | 0.30976088870825713 | 0.2809960520454092 |
| LogisticRegression | 0.2 | 8031 | 22522 | 21495 | 0.26285471148496053 | 0.2719975614712457 | 0.2673479918107825 |
| LogisticRegression | 0.21 | 7067 | 19207 | 22459 | 0.26897312932937506 | 0.23934837092731828 | 0.25329749103942656 |
| LogisticRegression | 0.22 | 6289 | 16371 | 23237 | 0.27753751103265667 | 0.212998712998713 | 0.24102249645498797 |
| LogisticRegression | 0.23 | 5576 | 13869 | 23950 | 0.2867575212136796 | 0.18885050463997832 | 0.22772661371015498 |
| LogisticRegression | 0.24 | 4932 | 11642 | 24594 | 0.29757451429950527 | 0.16703921967079863 | 0.21396963123644253 |
| LogisticRegression | 0.25 | 4343 | 9822 | 25183 | 0.30660077656194845 | 0.14709069972227867 | 0.1988052459316564 |
| LogisticRegression | 0.26 | 3832 | 8329 | 25694 | 0.3151056656524957 | 0.12978391925760346 | 0.18384628301388922 |
| LogisticRegression | 0.27 | 3349 | 7091 | 26177 | 0.32078544061302683 | 0.11342545553071869 | 0.16759245358554772 |
| LogisticRegression | 0.28 | 2896 | 6011 | 26630 | 0.3251375322779836 | 0.09808304545146651 | 0.15070382223609918 |
| LogisticRegression | 0.29 | 2491 | 5008 | 27035 | 0.33217762368315773 | 0.08436632120842646 | 0.13455773126266035 |
| LogisticRegression | 0.3 | 2166 | 4193 | 27360 | 0.34061959427582955 | 0.07335907335907337 | 0.12071896335516233 |
| LogisticRegression | 0.31 | 1878 | 3537 | 27648 | 0.34681440443213296 | 0.06360495834180045 | 0.10749549240147678 |
| LogisticRegression | 0.32 | 1625 | 3005 | 27901 | 0.3509719222462203 | 0.05503623924676556 | 0.09515165710270523 |
| LogisticRegression | 0.33 | 1371 | 2552 | 28155 | 0.34947744073413206 | 0.04643365169680959 | 0.0819755448593381 |
| LogisticRegression | 0.34 | 1182 | 2129 | 28344 | 0.3569918453639384 | 0.04003251371672424 | 0.07199196028869873 |
| LogisticRegression | 0.35 | 1007 | 1777 | 28519 | 0.36170977011494254 | 0.0341055341055341 | 0.0623336428350356 |
| LogisticRegression | 0.36 | 853 | 1495 | 28673 | 0.36328790459965926 | 0.028889792047686786 | 0.05352324778816591 |
| LogisticRegression | 0.37 | 736 | 1284 | 28790 | 0.36435643564356435 | 0.024927182821919663 | 0.04666201737145755 |
| LogisticRegression | 0.38 | 614 | 1062 | 28912 | 0.36634844868735084 | 0.020795231321547113 | 0.03935645150951863 |
| LogisticRegression | 0.39 | 516 | 902 | 29010 | 0.3638928067700987 | 0.017476122739280633 | 0.03335056876938987 |
| LogisticRegression | 0.4 | 440 | 730 | 29086 | 0.37606837606837606 | 0.01490212016527806 | 0.028668230388324212 |
| LogisticRegression | 0.41 | 375 | 601 | 29151 | 0.38422131147540983 | 0.012700670595407438 | 0.02458855157038883 |
| LogisticRegression | 0.42 | 329 | 510 | 29197 | 0.39213349225268174 | 0.011142721669037459 | 0.021669685493166474 |
| LogisticRegression | 0.43 | 264 | 434 | 29262 | 0.37822349570200575 | 0.008941272099166836 | 0.017469560614081524 |
| LogisticRegression | 0.44 | 227 | 368 | 29299 | 0.3815126050420168 | 0.007688139267086636 | 0.015072540752299062 |
| LogisticRegression | 0.45 | 176 | 303 | 29350 | 0.3674321503131524 | 0.005960848066111224 | 0.01173137810364939 |
| LogisticRegression | 0.46 | 145 | 260 | 29381 | 0.35802469135802467 | 0.004910925963557543 | 0.009688951254552138 |
| LogisticRegression | 0.47 | 123 | 229 | 29403 | 0.3494318181818182 | 0.00416581995529364 | 0.00823348283017605 |
| LogisticRegression | 0.48 | 104 | 203 | 29422 | 0.33876221498371334 | 0.003522319311792996 | 0.006972144940166929 |
| LogisticRegression | 0.49 | 90 | 162 | 29436 | 0.35714285714285715 | 0.003048160942897785 | 0.006044731009470078 |
| LogisticRegression | 0.5 | 78 | 132 | 29448 | 0.37142857142857144 | 0.002641739483844747 | 0.005246166263115415 |
| LogisticRegression | 0.51 | 67 | 108 | 29459 | 0.38285714285714284 | 0.0022691864797127954 | 0.004511632604962796 |
| LogisticRegression | 0.52 | 60 | 91 | 29466 | 0.3973509933774834 | 0.00203210729526519 | 0.0040435353977827946 |
| LogisticRegression | 0.53 | 52 | 79 | 29474 | 0.3969465648854962 | 0.001761159655896498 | 0.0035067606298681588 |
| LogisticRegression | 0.54 | 43 | 67 | 29483 | 0.39090909090909093 | 0.0014563435616067196 | 0.0029018760966392225 |
| LogisticRegression | 0.55 | 36 | 55 | 29490 | 0.3956043956043956 | 0.001219264377159114 | 0.002431036229192693 |
| LogisticRegression | 0.56 | 31 | 46 | 29495 | 0.4025974025974026 | 0.0010499221025536815 | 0.0020943823261155964 |
| LogisticRegression | 0.57 | 24 | 38 | 29502 | 0.3870967741935484 | 0.000812842918106076 | 0.0016222793024199 |
| LogisticRegression | 0.58 | 19 | 33 | 29507 | 0.36538461538461536 | 0.0006435006435006435 | 0.001284738657110014 |
| LogisticRegression | 0.59 | 18 | 28 | 29508 | 0.391304347826087 | 0.000609632188579557 | 0.001217367780332747 |
| LogisticRegression | 0.6 | 17 | 22 | 29509 | 0.4358974358974359 | 0.0005757637336584705 | 0.0011500084559445292 |
| LogisticRegression | 0.61 | 12 | 17 | 29514 | 0.41379310344827586 | 0.000406421459053038 | 0.0008120453391981052 |
| LogisticRegression | 0.62 | 9 | 10 | 29517 | 0.47368421052631576 | 0.0003048160942897785 | 0.0006092401421560332 |
| LogisticRegression | 0.63 | 8 | 7 | 29518 | 0.5333333333333333 | 0.000270947639368692 | 0.0005416201211875021 |
| LogisticRegression | 0.64 | 6 | 5 | 29520 | 0.5454545454545454 | 0.000203210729526519 | 0.00040627010190608386 |
| LogisticRegression | 0.65 | 4 | 4 | 29522 | 0.5 | 0.000135473819684346 | 0.00027087424663100153 |
| LogisticRegression | 0.66 | 3 | 3 | 29523 | 0.5 | 0.0001016053647632595 | 0.00020316944331572534 |
| LogisticRegression | 0.67 | 1 | 3 | 29525 | 0.25 | 3.38684549210865e-05 | 6.772773450728073e-05 |
| LogisticRegression | 0.68 | 0 | 2 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.69 | 0 | 2 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.7 | 0 | 2 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.71 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.72 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.73 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.74 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.75 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.76 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.77 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.78 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.79 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.8 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.81 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.82 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.83 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.84 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.85 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.86 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.87 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.88 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.89 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| LogisticRegression | 0.9 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.02 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.03 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.04 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.05 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.06 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.07 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.08 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.09 | 29526 | 144023 | 0 | 0.1701306259327337 | 1.0 | 0.29078911732118673 |
| RandomForest | 0.1 | 26290 | 113816 | 3236 | 0.18764364124305882 | 0.8904016798753641 | 0.3099651009243539 |
| RandomForest | 0.11 | 23870 | 97489 | 5656 | 0.1966891619080579 | 0.8084400189663348 | 0.31639990721410344 |
| RandomForest | 0.12 | 22047 | 86708 | 7479 | 0.20272171394418648 | 0.746697825645194 | 0.3188724408993282 |
| RandomForest | 0.13 | 20381 | 77949 | 9145 | 0.20727143292992983 | 0.6902729797466639 | 0.3188117882617945 |
| RandomForest | 0.14 | 18274 | 67465 | 11252 | 0.21313521268034383 | 0.6189121452279347 | 0.3170780375656097 |
| RandomForest | 0.15 | 15927 | 56584 | 13599 | 0.21964943249989657 | 0.5394228815281447 | 0.31218087556474616 |
| RandomForest | 0.16 | 13181 | 44453 | 16345 | 0.22870180796057882 | 0.44642010431484114 | 0.30245525470399265 |
| RandomForest | 0.17 | 10114 | 31841 | 19412 | 0.24106781074961267 | 0.34254555307186885 | 0.28298428953148386 |
| RandomForest | 0.18 | 8476 | 25363 | 21050 | 0.2504802151363811 | 0.28706902391112915 | 0.26752939319813773 |
| RandomForest | 0.19 | 7005 | 19928 | 22521 | 0.2600898525971856 | 0.23724852672221095 | 0.2481446713544342 |
| RandomForest | 0.2 | 4806 | 13312 | 24720 | 0.26526106634286345 | 0.1627717943507417 | 0.20174628494668795 |
| RandomForest | 0.21 | 2756 | 7409 | 26770 | 0.27112641416625677 | 0.0933414617625144 | 0.13887279232067723 |
| RandomForest | 0.22 | 923 | 3115 | 28603 | 0.22857850421000495 | 0.03126058389216284 | 0.05499940412346562 |
| RandomForest | 0.23 | 450 | 1986 | 29076 | 0.18472906403940886 | 0.015240804714488925 | 0.02815843814529754 |
| RandomForest | 0.24 | 317 | 1397 | 29209 | 0.18494749124854143 | 0.010736300209984421 | 0.02029449423815621 |
| RandomForest | 0.25 | 244 | 1048 | 29282 | 0.18885448916408668 | 0.008263903000745107 | 0.015834901680835876 |
| RandomForest | 0.26 | 202 | 828 | 29324 | 0.19611650485436893 | 0.006841427894059473 | 0.013221625867260112 |
| RandomForest | 0.27 | 160 | 653 | 29366 | 0.1968019680196802 | 0.00541895278737384 | 0.010547480141072546 |
| RandomForest | 0.28 | 124 | 514 | 29402 | 0.19435736677115986 | 0.004199688410214726 | 0.0082217212571277 |
| RandomForest | 0.29 | 96 | 417 | 29430 | 0.1871345029239766 | 0.003251371672424304 | 0.0063916908019574555 |
| RandomForest | 0.3 | 78 | 345 | 29448 | 0.18439716312056736 | 0.002641739483844747 | 0.005208855053591105 |
| RandomForest | 0.31 | 63 | 300 | 29463 | 0.17355371900826447 | 0.0021337126600284497 | 0.004215597711532671 |
| RandomForest | 0.32 | 51 | 250 | 29475 | 0.16943521594684385 | 0.0017272912009754116 | 0.0034197203875683106 |
| RandomForest | 0.33 | 41 | 224 | 29485 | 0.15471698113207547 | 0.0013886066517645465 | 0.002752509147057836 |
| RandomForest | 0.34 | 29 | 190 | 29497 | 0.1324200913242009 | 0.0009821851927115084 | 0.0019499075474869723 |
| RandomForest | 0.35 | 25 | 156 | 29501 | 0.13812154696132597 | 0.0008467113730271625 | 0.0016831049920894067 |
| RandomForest | 0.36 | 16 | 104 | 29510 | 0.13333333333333333 | 0.000541895278737384 | 0.0010794036294947042 |
| RandomForest | 0.37 | 7 | 55 | 29519 | 0.11290322580645161 | 0.0002370791844476055 | 0.0004731647965391374 |
| RandomForest | 0.38 | 2 | 28 | 29524 | 0.06666666666666667 | 6.7736909842173e-05 | 0.00013533631073216944 |
| RandomForest | 0.39 | 1 | 6 | 29525 | 0.14285714285714285 | 3.38684549210865e-05 | 6.772085463718552e-05 |
| RandomForest | 0.4 | 1 | 3 | 29525 | 0.25 | 3.38684549210865e-05 | 6.772773450728073e-05 |
| RandomForest | 0.41 | 1 | 0 | 29525 | 1.0 | 3.38684549210865e-05 | 6.773461577539202e-05 |
| RandomForest | 0.42 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.43 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.44 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.45 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.46 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.47 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.48 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.49 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.5 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.51 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.52 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.53 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.54 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.55 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.56 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.57 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.58 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.59 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.6 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.61 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.62 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.63 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.64 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.65 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.66 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.67 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.68 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.69 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.7 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.71 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.72 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.73 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.74 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.75 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.76 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.77 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.78 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.79 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.8 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.81 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.82 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.83 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.84 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.85 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.86 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.87 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.88 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.89 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |
| RandomForest | 0.9 | 0 | 0 | 29526 | 0.0 | 0.0 | 0.0 |

#### validation_metrics.csv

| Model | Config | DecisionThreshold | AUC_ROC | AUC_PR | F1_class1 | Precision_class1 | Recall_class1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | regParam=0.001 | 0.11 | 0.6142431122924915 | 0.2421530474820992 | 0.32091656079206726 | 0.20623617056226026 | 0.7228883018356702 | 0.4795072285060703 |
| RandomForest | maxDepth=7,minInstancesPerNode=100,numTrees=30 | 0.12 | 0.6053570070933709 | 0.22366307239535177 | 0.3188724408993282 | 0.20272171394418648 | 0.746697825645194 | 0.45728871961232853 |

#### ranking_metrics.csv

| Model | TargetFlaggedFraction | ActualFlaggedFraction | FlaggedFlights | Precision_class1 | Recall_class1 | Lift |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.1 | 0.10009485040545209 | 52448 | 0.3575160158633313 | 0.1678647842940655 | 1.6770571474366482 |
| LogisticRegression | 0.2 | 0.20039199745029895 | 105002 | 0.3247557189386869 | 0.30527380643313073 | 1.5233832204743827 |
| LogisticRegression | 0.3 | 0.2998017111242159 | 157091 | 0.3046450783303945 | 0.42843074939795706 | 1.4290470450999087 |
| RandomForest | 0.1 | 0.1002246256080827 | 52516 | 0.33321273516642547 | 0.1566564908731189 | 1.5630538894274022 |
| RandomForest | 0.2 | 0.19999312954809603 | 104793 | 0.3003158607922285 | 0.28173818071135065 | 1.4087392969346773 |
| RandomForest | 0.3 | 0.3025403495914944 | 158526 | 0.28871604657911004 | 0.40973832394832727 | 1.354326206410408 |

#### metrics_by_holiday_slice.csv

| Model | Slice | Flights | Delayed15Plus | Prevalence | AUC_ROC | AUC_PR | DecisionThreshold | Precision_class1 | Recall_class1 | F1_class1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | ALL TEST | 523983 | 111703 | 0.21318058028600165 | 0.6266447181476484 | 0.30358960979984206 | 0.11 | 0.24363100646267408 | 0.8497891730750293 | 0.37869246496783526 | 0.40555895897385985 |
| LogisticRegression | Christmas/New Year period | 225777 | 60505 | 0.26798566727346007 | 0.6230244483040931 | 0.3606047624849566 | 0.11 | 0.3004628165694081 | 0.863746797785307 | 0.44583688790308823 | 0.42457380512629717 |
| LogisticRegression | Regular day | 298206 | 51198 | 0.1716866863845798 | 0.6226871056391311 | 0.25372911416644944 | 0.11 | 0.1978005164891718 | 0.8332942693073948 | 0.31971073683421697 | 0.391162484993595 |
| RandomForest | ALL TEST | 523983 | 111703 | 0.21318058028600165 | 0.6093565155910344 | 0.28687309823937623 | 0.12 | 0.23641403687160306 | 0.8742110775896798 | 0.3721791759249025 | 0.37125059400782084 |
| RandomForest | Christmas/New Year period | 225777 | 60505 | 0.26798566727346007 | 0.6076038402899802 | 0.34448913819369287 | 0.12 | 0.29534918833075174 | 0.872936120981737 | 0.44136645858925183 | 0.4078227631689676 |
| RandomForest | Regular day | 298206 | 51198 | 0.1716866863845798 | 0.6132167040660813 | 0.24364580849126094 | 0.12 | 0.19141769060650823 | 0.8757178014766202 | 0.31416419081787095 | 0.34356116241792584 |

#### threshold_transfer.csv

| Model | Threshold | ValidationPrevalence | ValidationPredictedPositiveRate | TestPrevalence | TestPredictedPositiveRate |
| --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.11 | 0.1701306259327337 | 0.596333024102703 | 0.21318058028600165 | 0.743577558813931 |
| RandomForest | 0.12 | 0.1701306259327337 | 0.6266529913741941 | 0.21318058028600165 | 0.788298475332215 |

#### run_summary.txt

```text
BDA MLlib flight-arrival-delay classification (redesigned ML stage)
Mode: all; namespace: experiments
Target: ArrDel15; positive class: arrival delay >= 15 minutes
Cancelled and diverted flights excluded; valid 0/1 target required.
Predictors: pre-departure schedule, carrier, cyclic time features, holiday calendar, optional
  scheduled origin-hour demand and smoothed historical delay rates (strictly earlier dates).
No post-departure or delay-cause columns, no Tail_Number, no raw Year/Month/DayOfYear.
Input rows: 1781482; eligible rows in input: 1765279
Distinct dates: 92
Training date range: 2024-10-01 through 2024-11-24
Validation date range (tuning and threshold selection): 2024-11-25 through 2024-12-03
Final test date range (held out, scored once): 2024-12-04 through 2024-12-31
Train records: 1067747; positive: 146735 (13.74%)
Validation records: 173549; positive: 29526 (17.01%)
Test records: 523983; positive: 111703 (21.32%)
Chosen feature spec: E6_combined_winners [history=expanding,warmup=7,holiday=on,demand=off,historyset=extended,recent=7,extrademand=on,schedextras=off] (margin 0.002 AUC-PR over E0)
Selected Logistic Regression: regParam=0.001 (maxIter=30)
Selected Random Forest: maxDepth=7,minInstancesPerNode=100,numTrees=30 (maxBins=32, subsamplingRate=0.5, featureSubsetStrategy=onethird)
Hyperparameters chosen by validation AUC-PR; thresholds chosen by validation F1 only.
AUC-ROC and AUC-PR use the class-1 probability; F1, precision and recall refer to class 1
at the validation-selected threshold. Holiday categories absent from the fitting windows are
treated as unseen categories. Results are observational, not causal.

Validation metrics at the chosen thresholds:
Model, Config, DecisionThreshold, AUC_ROC, AUC_PR, F1_class1, Precision_class1, Recall_class1, Accuracy
LogisticRegression, "regParam=0.001", 0.110000, 0.614243, 0.242153, 0.320917, 0.206236, 0.722888, 0.479507
RandomForest, "maxDepth=7,minInstancesPerNode=100,numTrees=30", 0.120000, 0.605357, 0.223663, 0.318872, 0.202722, 0.746698, 0.457289

Trivial references on test: majority class accuracy=0.7868; 'flag every flight' F1=0.3514; random-ranking AUC-PR equals prevalence=0.2132.
Held-out test metrics (single evaluation after decisions were frozen):
Model, DecisionThreshold, AUC_ROC, AUC_PR, F1_class1, Precision_class1, Recall_class1, Accuracy
LogisticRegression, 0.110000, 0.626645, 0.303590, 0.378692, 0.243631, 0.849789, 0.405559
RandomForest, 0.120000, 0.609357, 0.286873, 0.372179, 0.236414, 0.874211, 0.371251
LogisticRegression: threshold transfer validation predicted-positive rate=0.5963 vs test=0.7436 (prevalence 0.1701 -> 0.2132)
RandomForest: threshold transfer validation predicted-positive rate=0.6267 vs test=0.7883 (prevalence 0.1701 -> 0.2132)
```

#### preprocessing_metadata.json

```json
{
  "frozen_before_test_scoring": true,
  "mode": "all",
  "feature_spec": {
    "use_history": true,
    "window_days": null,
    "warmup_days": 7,
    "use_holiday": true,
    "use_demand": false,
    "history_set": "extended",
    "recent_days": 7,
    "extra_demand": true,
    "sched_extras": false,
    "name": "E6_combined_winners"
  },
  "feature_spec_label": "history=expanding,warmup=7,holiday=on,demand=off,historyset=extended,recent=7,extrademand=on,schedextras=off",
  "selection_margin_auc_pr": 0.002,
  "logistic_regression": {
    "label": "regParam=0.001",
    "regParam": 0.001,
    "class_weighted": false,
    "maxIter": 30
  },
  "random_forest": {
    "label": "maxDepth=7,minInstancesPerNode=100,numTrees=30",
    "maxDepth": 7,
    "minInstancesPerNode": 100,
    "numTrees": 30,
    "maxBins": 32,
    "subsamplingRate": 0.5,
    "featureSubsetStrategy": "onethird"
  },
  "extra_models": {},
  "thresholds_from_validation_F1": {
    "LogisticRegression": 0.11,
    "RandomForest": 0.12
  },
  "windows": {
    "train": [
      "2024-10-01",
      "2024-11-24"
    ],
    "validation": [
      "2024-11-25",
      "2024-12-03"
    ],
    "test": [
      "2024-12-04",
      "2024-12-31"
    ]
  },
  "seed": 42,
  "finished_utc": "2026-10-30T10:40:48Z",
  "spark_version": "3.5.9",
  "target": "ArrDel15 (1 = arrival delay >= 15 minutes); cancelled/diverted excluded",
  "final_fit_window": [
    "2024-10-01",
    "2024-12-03"
  ],
  "feature_source_columns": [
    "FlightDate",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Reporting_Airline",
    "DayOfWeek",
    "Origin",
    "Dest",
    "DepTimeBlk",
    "HolidayPeriod"
  ],
  "features": [
    "CRSElapsedTime__imputed",
    "Distance__imputed",
    "DepMinuteSin",
    "DepMinuteCos",
    "ArrMinuteSin",
    "ArrMinuteCos",
    "ScheduledDepHour",
    "ScheduledArrHour",
    "IsHolidayWindow",
    "DestHourDemandLog",
    "OriginDayDemandLog",
    "OriginPrevHourDemandLog",
    "hist_rate_airline",
    "hist_rate_origin",
    "hist_rate_dest",
    "hist_rate_route",
    "hist_rate_origin_depblk",
    "hist_rate_airline_origin",
    "hist_rate_origin_dow",
    "hist_rate_airline_depblk",
    "hist_rate_dest_arrhour",
    "hist_rate_airline_route",
    "recent_rate_airline",
    "recent_rate_origin",
    "recent_rate_origin_depblk",
    "Reporting_Airline__ohe_WN",
    "Reporting_Airline__ohe_DL",
    "Reporting_Airline__ohe_AA",
    "Reporting_Airline__ohe_OO",
    "Reporting_Airline__ohe_UA",
    "Reporting_Airline__ohe_YX",
    "Reporting_Airline__ohe_MQ",
    "Reporting_Airline__ohe_OH",
    "Reporting_Airline__ohe_NK",
    "Reporting_Airline__ohe_AS",
    "Reporting_Airline__ohe_B6",
    "Reporting_Airline__ohe_F9",
    "Reporting_Airline__ohe_9E",
    "Reporting_Airline__ohe_G4",
    "Reporting_Airline__ohe_HA",
    "Reporting_Airline__ohe___unknown",
    "Reporting_Airline__ohe_invalidValues",
    "DayOfWeekCat__ohe_DOW_7",
    "DayOfWeekCat__ohe_DOW_1",
    "DayOfWeekCat__ohe_DOW_2",
    "DayOfWeekCat__ohe_DOW_5",
    "DayOfWeekCat__ohe_DOW_4",
    "DayOfWeekCat__ohe_DOW_3",
    "DayOfWeekCat__ohe_DOW_6",
    "DayOfWeekCat__ohe___unknown",
    "DayOfWeekCat__ohe_invalidValues",
    "HolidayPeriod__ohe_Regular day",
    "HolidayPeriod__ohe_Thanksgiving week",
    "HolidayPeriod__ohe_Halloween",
    "HolidayPeriod__ohe___unknown",
    "HolidayPeriod__ohe_invalidValues"
  ],
  "features_stage_a": [
    "CRSElapsedTime__imputed",
    "Distance__imputed",
    "DepMinuteSin",
    "DepMinuteCos",
    "ArrMinuteSin",
    "ArrMinuteCos",
    "ScheduledDepHour",
    "ScheduledArrHour",
    "IsHolidayWindow",
    "DestHourDemandLog",
    "OriginDayDemandLog",
    "OriginPrevHourDemandLog",
    "hist_rate_airline",
    "hist_rate_origin",
    "hist_rate_dest",
    "hist_rate_route",
    "hist_rate_origin_depblk",
    "hist_rate_airline_origin",
    "hist_rate_origin_dow",
    "hist_rate_airline_depblk",
    "hist_rate_dest_arrhour",
    "hist_rate_airline_route",
    "recent_rate_airline",
    "recent_rate_origin",
    "recent_rate_origin_depblk",
    "Reporting_Airline__ohe_WN",
    "Reporting_Airline__ohe_DL",
    "Reporting_Airline__ohe_AA",
    "Reporting_Airline__ohe_OO",
    "Reporting_Airline__ohe_UA",
    "Reporting_Airline__ohe_YX",
    "Reporting_Airline__ohe_MQ",
    "Reporting_Airline__ohe_OH",
    "Reporting_Airline__ohe_NK",
    "Reporting_Airline__ohe_AS",
    "Reporting_Airline__ohe_B6",
    "Reporting_Airline__ohe_F9",
    "Reporting_Airline__ohe_9E",
    "Reporting_Airline__ohe_G4",
    "Reporting_Airline__ohe_HA",
    "Reporting_Airline__ohe___unknown",
    "Reporting_Airline__ohe_invalidValues",
    "DayOfWeekCat__ohe_DOW_7",
    "DayOfWeekCat__ohe_DOW_5",
    "DayOfWeekCat__ohe_DOW_4",
    "DayOfWeekCat__ohe_DOW_3",
    "DayOfWeekCat__ohe_DOW_2",
    "DayOfWeekCat__ohe_DOW_1",
    "DayOfWeekCat__ohe_DOW_6",
    "DayOfWeekCat__ohe___unknown",
    "DayOfWeekCat__ohe_invalidValues",
    "HolidayPeriod__ohe_Regular day",
    "HolidayPeriod__ohe_Halloween",
    "HolidayPeriod__ohe___unknown",
    "HolidayPeriod__ohe_invalidValues"
  ],
  "history_groups": {
    "base": {
      "keys": {
        "hist_rate_airline": [
          "Reporting_Airline"
        ],
        "hist_rate_origin": [
          "Origin"
        ],
        "hist_rate_dest": [
          "Dest"
        ],
        "hist_rate_route": [
          "Origin",
          "Dest"
        ],
        "hist_rate_origin_depblk": [
          "Origin",
          "DepTimeBlk"
        ],
        "hist_rate_airline_origin": [
          "Reporting_Airline",
          "Origin"
        ],
        "hist_rate_origin_dow": [
          "Origin",
          "DayOfWeekCat"
        ],
        "hist_rate_airline_depblk": [
          "Reporting_Airline",
          "DepTimeBlk"
        ],
        "hist_rate_dest_arrhour": [
          "Dest",
          "ScheduledArrHour"
        ],
        "hist_rate_airline_route": [
          "Reporting_Airline",
          "Origin",
          "Dest"
        ]
      },
      "window_days": null
    },
    "recent": {
      "keys": {
        "recent_rate_airline": [
          "Reporting_Airline"
        ],
        "recent_rate_origin": [
          "Origin"
        ],
        "recent_rate_origin_depblk": [
          "Origin",
          "DepTimeBlk"
        ]
      },
      "window_days": 7
    }
  },
  "history_smoothing": 100.0,
  "history_prior_stage_a_train": 0.13742487686689825,
  "history_prior_final_fit": 0.1419975573916294
}
```

### A3. YARN runtime inventory

Commands and resource settings come from Spark History Server environment metadata. Failure/retry evidence is cross-checked against the captured application logs. For diagnostic attempts `_0006`-`_0010`, YARN reports `SUCCEEDED` because the shell pipeline ended in `tee`; the explicit `DIAGNOSTICS_YARN_STATUS=FAILED` marker and executor-loss evidence are authoritative.

| ApplicationId | Name | StartUTC | EndUTC | WallTimeMs | DriverMemory | ExecutorMemory | ExecutorInstances | ExecutorCores | YARNState | YARNFinalStatus | FailureOrRetryEvidence | Command | HistoryLogCompletion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| application_1793344136681_0001 | BDA-00-Validate-Actual-Data | 2026-10-30T07:10:13.814GMT | 2026-10-30T07:11:10.064GMT | 56250 | 512m | 512m | 2 | 1 | not retained by current RM | not retained | PASSED; documented data-validation marker; no retry documented | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=512m --executor-memory 512m --num-executors 2 --executor-cores 1 /src/00_validate_data.py | completed |
| application_1793344136681_0002 | BDA-03-Flight-Delay-MLlib | 2026-10-30T09:29:37.585GMT | 2026-10-30T09:35:30.833GMT | 353248 | 1g | 512m | 2 | 1 | not retained by current RM | not retained | SUCCEEDED; documented selection run; no retry documented | spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_history_ablation --experiments E0_reference,E1_no_history | completed |
| application_1793344136681_0003 | BDA-03-Flight-Delay-MLlib | 2026-10-30T09:35:52.782GMT | 2026-10-30T09:43:35.870GMT | 463088 | 512m | 512m | 2 | 1 | not retained by current RM | not retained | FAILED; documented 512 MB driver OOM during initial temporal-combination run | spark-submit --master yarn --deploy-mode client --driver-memory 512m /src/03_ml.py --mode select --run-tag actual_temporal_features --experiments E0_reference,E3_history_warmup7,E4_history_trailing14,E9_recent7_history | completed |
| application_1793344136681_0004 | BDA-03-Flight-Delay-MLlib | 2026-10-30T09:44:26.522GMT | 2026-10-30T09:57:24.699GMT | 778177 | 1g | 512m | 2 | 1 | not retained by current RM | not retained | SUCCEEDED; exact temporal rerun with 1 GB driver | spark-submit --master yarn --deploy-mode client --driver-memory 1g /src/03_ml.py --mode select --run-tag actual_temporal_features --experiments E0_reference,E3_history_warmup7,E4_history_trailing14,E9_recent7_history | completed |
| application_1793344136681_0005 | BDA-03-Flight-Delay-MLlib | 2026-10-30T09:57:49.836GMT | 2026-10-30T10:09:19.673GMT | 689837 | 1g | 512m | 2 | 1 | not retained by current RM | not retained | SUCCEEDED; documented selection run; no retry documented | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.executor.memory=512m --conf spark.driver.memory=1g /src/03_ml.py --mode select --run-tag actual_calendar_schedule --experiments E0_reference,E2_no_holiday_features,E5_plus_scheduled_demand,E10_extra_demand,E11_schedule_extras | completed |
| application_1793344136681_0006 | BDA-03-Flight-Delay-MLlib | 2026-10-30T10:09:41.279GMT | 2026-10-30T10:18:10.109GMT | 508830 | 1g | 512m | 2 | 1 | not retained by current RM | not retained | SUCCEEDED; documented selection run; no retry documented | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.executor.memory=512m --conf spark.driver.memory=1g /src/03_ml.py --mode select --run-tag actual_extended_history --experiments E0_reference,E8_extended_history | completed |
| application_1793344136681_0007 | BDA-03-Flight-Delay-MLlib | 2026-10-30T10:18:34.644GMT | 2026-10-30T10:40:49.306GMT | 1334662 | 1536m | 512m | 2 | 1 | not retained by current RM | not retained | SUCCEEDED; documented final marker; no retry documented | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.executor.memory=512m --conf spark.driver.memory=1536m /src/03_ml.py --mode all --run-tag actual_final --experiments E0_reference,E3_history_warmup7,E8_extended_history,E9_recent7_history,E10_extra_demand | completed |
| application_1793344136681_0008 | BDA-05-Verify-Saved-Models | 2026-10-30T10:41:15.945GMT | 2026-10-30T11:12:37.970GMT | 1882025 | 1g | 512m | 2 | 1 | not retained by current RM | not retained | PASSED; documented saved-model verification marker; no retry documented | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1g /src/05_verify_saved_models.py --models-uri hdfs:///user/root/flights/experiments/ml_redesign/actual_final/models | completed |
| application_1793392054406_0001 | BDA-03-Flight-Delay-MLlib | 2026-10-30T20:28:11.343GMT | 2026-10-30T20:39:08.809GMT | 657466 | 1536m | 1g | 2 | 1 | FINISHED | SUCCEEDED | no retained run log; no failure inferred | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1536m --executor-memory 1g --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag actual_gbt_probe2 --experiments E0_reference,E8_extended_history --extra-models gbt | completed |
| application_1793392054406_0002 | BDA-03-Flight-Delay-MLlib | 2026-10-30T20:39:26.467GMT | 2026-10-30T20:52:48.796GMT | 802329 | 1536m | 1g | 2 | 1 | FINISHED | SUCCEEDED | no retained run log; no failure inferred | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1536m --executor-memory 1g --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag actual_mlp_probe --experiments E0_reference,E8_extended_history --extra-models mlp | completed |
| application_1793392054406_0003 | BDA-03-Flight-Delay-MLlib | 2026-10-30T21:16:58.278GMT | 2026-10-30T21:24:14.979GMT | 436701 | 1536m | 1g | 2 | 1 | FINISHED | SUCCEEDED | no retained run log; no failure inferred | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1536m --executor-memory 1g --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag actual_xgb_probe --experiments E0_reference,E8_extended_history --extra-models xgb | completed |
| application_1793392054406_0004 | BDA-03-Flight-Delay-MLlib | 2026-10-30T21:24:38.014GMT | 2026-10-30T21:33:37.883GMT | 539869 | 1536m | 1g | 2 | 1 | FINISHED | SUCCEEDED | no retained run log; no failure inferred | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1536m --executor-memory 1g --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag actual_xgb_probe2 --experiments E0_reference,E8_extended_history --extra-models xgb | completed |
| application_1793392054406_0005 | BDA-03-Flight-Delay-MLlib | 2026-10-30T21:35:45.274GMT | 2026-10-30T21:55:12.415GMT | 1167141 | 1536m | 1g | 2 | 1 | FINISHED | SUCCEEDED | no retained run log; no failure inferred | org.apache.spark.deploy.SparkSubmit --master yarn --deploy-mode client --conf spark.driver.memory=1536m --executor-memory 1g --num-executors 2 --executor-cores 1 /src/03_ml.py --mode select --run-tag actual_xgb_combined --experiments E0_reference,E3_history_warmup7,E8_extended_history,E9_recent7_history,E10_extra_demand --extra-models xgb | completed |
| application_1793392054406_0006 | BDA-Flight-ML-Diagnostics | 2026-10-31T06:39:31.673GMT | 2026-10-31T06:49:59.638GMT | 627965 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | FAILED; Lost executor messages=6; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0007 | BDA-Flight-ML-Diagnostics | 2026-10-31T06:50:55.314GMT | 2026-10-31T07:01:02.015GMT | 606701 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | FAILED; Lost executor messages=6; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0008 | BDA-Flight-ML-Diagnostics | 2026-10-31T07:02:03.962GMT | 2026-10-31T07:12:05.075GMT | 601113 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | FAILED; Lost executor messages=7; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0009 | BDA-Flight-ML-Diagnostics | 2026-10-31T07:12:59.313GMT | 2026-10-31T07:23:08.681GMT | 609368 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | FAILED; Lost executor messages=6; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0010 | BDA-Flight-ML-Diagnostics | 2026-10-31T07:24:00.087GMT | 2026-10-31T07:36:19.584GMT | 739497 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | FAILED; Lost executor messages=7; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0011 | BDA-Flight-ML-Diagnostics | 2026-10-31T07:38:25.880GMT | 2026-10-31T07:46:39.620GMT | 493740 | 1536m | 512m | 2 | 1 | KILLED | KILLED | no explicit status marker; Lost executor messages=0; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |
| application_1793392054406_0012 | BDA-Flight-ML-Diagnostics | 2026-10-31T07:47:27.986GMT | 2026-10-31T08:08:58.047GMT | 1290061 | 1536m | 512m | 2 | 1 | FINISHED | SUCCEEDED | SUCCEEDED; Lost executor messages=0; log files=1 | spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py | completed |

## B. Model diagnostics

Validation is the only decision-bearing split. Test rows are retained solely to describe the already-scored final window.

### B1. Train versus validation PR-AUC

The train/validation diagnostic models use the frozen feature specification and hyperparameters and are fitted on train only. The test rows load the saved final models.

| Status | Split | Model | Flights | Delayed15Plus | Prevalence | AUC_ROC | AUC_PR | Method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN - diagnostic fit assessment | train | LogisticRegression | 929645 | 127569 | 0.13722334869762112 | 0.6620381279290899 | 0.23646061267428783 | 1000-bin histogram approximation (bounded-memory diagnostic) |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 173549 | 29526 | 0.1701306259327337 | 0.6142484745293798 | 0.24220024706703233 | 1000-bin histogram approximation (bounded-memory diagnostic) |
| TRAIN - diagnostic fit assessment | train | RandomForest | 929645 | 127569 | 0.13722334869762112 | 0.642730455307552 | 0.23275947906385214 | 1000-bin histogram approximation (bounded-memory diagnostic) |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 173549 | 29526 | 0.1701306259327337 | 0.6033638517076834 | 0.2236689050041484 | 1000-bin histogram approximation (bounded-memory diagnostic) |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 523983 | 111703 | 0.21318058028600165 | 0.6266473478510162 | 0.30359073964831573 | 1000-bin histogram approximation (bounded-memory diagnostic) |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 523983 | 111703 | 0.21318058028600165 | 0.6093943761712336 | 0.28686706732980555 | 1000-bin histogram approximation (bounded-memory diagnostic) |

### B2. Logistic Regression calibration deciles

Bins are approximate equal-count deciles as documented above.

| Status | Split | EqualCountBin | Flights | MeanPredictedProbability | ObservedDelayRate |
| --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | validation | 1 | 17359 | 0.05052939607954351 | 0.08128348407166311 |
| VALIDATION - used for diagnostics/decisions | validation | 2 | 17330 | 0.07149261638543655 | 0.11211771494518176 |
| VALIDATION - used for diagnostics/decisions | validation | 3 | 17315 | 0.08711188822236328 | 0.12879006641640198 |
| VALIDATION - used for diagnostics/decisions | validation | 4 | 17339 | 0.10199315278900623 | 0.144068285368245 |
| VALIDATION - used for diagnostics/decisions | validation | 5 | 17345 | 0.11718052414246256 | 0.1616604208705679 |
| VALIDATION - used for diagnostics/decisions | validation | 6 | 17320 | 0.1337484679012922 | 0.171189376443418 |
| VALIDATION - used for diagnostics/decisions | validation | 7 | 17337 | 0.15293481789361468 | 0.18515314068177885 |
| VALIDATION - used for diagnostics/decisions | validation | 8 | 17340 | 0.17673736121336345 | 0.20063437139561707 |
| VALIDATION - used for diagnostics/decisions | validation | 9 | 17353 | 0.21152456780791132 | 0.22186365469947558 |
| VALIDATION - used for diagnostics/decisions | validation | 10 | 17511 | 0.29841381102141334 | 0.2933013534349837 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 1 | 52354 | 0.064518625290437 | 0.09729915574741185 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 2 | 52173 | 0.09022687213395267 | 0.13552987177275602 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 3 | 52492 | 0.10877377098015341 | 0.15758591785414922 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 4 | 52374 | 0.12630659851852316 | 0.17558330469316838 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 5 | 52360 | 0.14436709162055844 | 0.19358288770053475 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 6 | 52294 | 0.16394795122368472 | 0.21621983401537462 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 7 | 52360 | 0.18635186127615183 | 0.24163483575248282 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 8 | 52372 | 0.21373077388442469 | 0.26382418086000153 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 9 | 52322 | 0.2521484379865025 | 0.2920377661404381 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | 10 | 52882 | 0.3430469894740857 | 0.3569834726371922 |

### B3. Gains and lift

Fractions 1%, 2%, 5%, 10%, 20%, 30%, 40%, and 50% are shown for both models.

| Status | Split | Model | TargetFlaggedFraction | ActualFlaggedFraction | FlaggedFlights | Precision | Recall | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.01 | 0.010994013218168933 | 1908 | 0.36792452830188677 | 0.023775655354602722 | 2.1626002154800563 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.02 | 0.020973903623760438 | 3640 | 0.35412087912087914 | 0.0436564383932805 | 2.081464622724021 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.05 | 0.05097695751632104 | 8847 | 0.32496891601672884 | 0.09737180789812369 | 1.910114150436472 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.1 | 0.10090521985145405 | 17512 | 0.29328460484239377 | 0.17394838447470026 | 1.7238789502740837 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.2 | 0.20089427193472736 | 34865 | 0.25773698551556 | 0.30434193592088327 | 1.5149358565074822 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.3 | 0.30080841721934437 | 52205 | 0.23877023273632794 | 0.4221702905913432 | 1.4034523850557805 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.4 | 0.40070527631965613 | 69542 | 0.22540335336918696 | 0.5308880308880309 | 1.3248840538464075 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | 0.5 | 0.5005041803755712 | 86862 | 0.21459326287674702 | 0.6313079997290524 | 1.2613441095643354 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.01 | 0.010976727033863634 | 1905 | 0.2141732283464567 | 0.013818329607803293 | 1.2588752152780334 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.02 | 0.020979665685195537 | 3641 | 0.23098049986267508 | 0.028483370588633747 | 1.357665608977423 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.05 | 0.05114981935937401 | 8877 | 0.2766700461867748 | 0.08318092528618845 | 1.6262212912574876 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.1 | 0.10096284046580505 | 17522 | 0.2638968154320283 | 0.156607735555104 | 1.5511423295201883 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.2 | 0.20096917873338366 | 34878 | 0.24981363610298757 | 0.2950958477274267 | 1.468363704261918 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.3 | 0.30106194792248875 | 52249 | 0.23248291833336523 | 0.4114001219264377 | 1.3664965790773285 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.4 | 0.40065341776674024 | 69533 | 0.21983806250269655 | 0.5177132019237283 | 1.2921721841522888 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | 0.5 | 0.5008556661231122 | 86923 | 0.21190018752228984 | 0.6238230711914923 | 1.2455146529941705 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.01 | 0.010992723046358375 | 5760 | 0.44947916666666665 | 0.023177533280216288 | 2.108443302216592 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.02 | 0.020994955943227166 | 11001 | 0.42150713571493503 | 0.04151186628828232 | 1.9772304548071118 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.05 | 0.0509711192920381 | 26708 | 0.38471618990564627 | 0.09198499592669848 | 1.8046493230739573 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.1 | 0.10092503001051562 | 52883 | 0.35697672219805987 | 0.16900172779603054 | 1.6745273970037153 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.2 | 0.2007794145993286 | 105205 | 0.3246803859132171 | 0.3057930404733982 | 1.5230298438892886 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.3 | 0.30072922213125236 | 157577 | 0.30446067636774404 | 0.42949607441160936 | 1.428182041531558 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.4 | 0.4006580366156917 | 209938 | 0.28879002372128915 | 0.5427607136782361 | 1.3546732227384426 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | 0.5 | 0.5004570758975004 | 262231 | 0.2743153936796183 | 0.6439755422862412 | 1.286774777100234 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.01 | 0.011059519106535898 | 5795 | 0.41345987920621224 | 0.021449737249671003 | 1.9394819108359553 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.02 | 0.02441109730659201 | 12791 | 0.3925416308341803 | 0.0449495537272947 | 1.8413573614798735 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.05 | 0.0513528110644811 | 26908 | 0.36517020960309204 | 0.08796540827014494 | 1.7129618894609542 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.1 | 0.10114259432080812 | 52997 | 0.3329811121384229 | 0.15798143290690492 | 1.561967378509326 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.2 | 0.20083857682405726 | 105236 | 0.3003154813941997 | 0.2829288380795502 | 1.4087375172320973 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.3 | 0.3025403495914944 | 158526 | 0.28871604657911004 | 0.40973832394832727 | 1.354326206410408 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.4 | 0.4014424132080621 | 210349 | 0.2789602042320144 | 0.5253126594630404 | 1.3085629275319697 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | 0.5 | 0.5004608928152249 | 262233 | 0.2662098210370167 | 0.6249518813281648 | 1.24875268037957 |

### B4. History-feature distributions

Training is past-only; validation uses frozen training tables; test is descriptive and uses final saved history tables.

| Status | Split | Feature | Mean | StdDev | P05 | P50 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN - diagnostic fit assessment | train | hist_rate_airline | 0.13433896526366118 | 0.031716860584389335 | 0.08661121703749576 | 0.13525818253193175 | 0.1670191789503393 |
| TRAIN - diagnostic fit assessment | train | hist_rate_origin | 0.13462844415343198 | 0.03411863778774282 | 0.09472316039616635 | 0.13016981572990027 | 0.1822353766040474 |
| TRAIN - diagnostic fit assessment | train | hist_rate_dest | 0.13453097233747868 | 0.029017876846949252 | 0.0974355272257034 | 0.130889290272361 | 0.17626611202491987 |
| TRAIN - diagnostic fit assessment | train | hist_rate_route | 0.13541651222173398 | 0.036186127306414975 | 0.08820879426235802 | 0.1306686807631987 | 0.19596688407043444 |
| TRAIN - diagnostic fit assessment | train | hist_rate_origin_depblk | 0.13518030341418966 | 0.04769979569231711 | 0.0689623848670614 | 0.13017993833284966 | 0.21979487968441802 |
| TRAIN - diagnostic fit assessment | train | hist_rate_airline_origin | 0.13474913343302863 | 0.043257866079571265 | 0.07656944520201082 | 0.12942875087463018 | 0.20414561406437934 |
| TRAIN - diagnostic fit assessment | train | hist_rate_origin_dow | 0.13654971035030633 | 0.04681940952297636 | 0.0815030173923002 | 0.128920868428809 | 0.2179434314048854 |
| TRAIN - diagnostic fit assessment | train | hist_rate_airline_depblk | 0.1338556251111783 | 0.051523107339592686 | 0.057927990608068115 | 0.1282093643007603 | 0.22123298834709765 |
| TRAIN - diagnostic fit assessment | train | hist_rate_dest_arrhour | 0.13444458911502247 | 0.046305059278209805 | 0.06904317631560189 | 0.13020298545111275 | 0.21624448032223456 |
| TRAIN - diagnostic fit assessment | train | hist_rate_airline_route | 0.13373609641065926 | 0.03282363386430844 | 0.08811863685796542 | 0.12970231808887836 | 0.1916841976666855 |
| TRAIN - diagnostic fit assessment | train | recent_rate_airline | 0.13542653065737031 | 0.038763903717696134 | 0.0800128342476169 | 0.13408169119822277 | 0.20411176356760716 |
| TRAIN - diagnostic fit assessment | train | recent_rate_origin | 0.13598807440307165 | 0.0458021117850794 | 0.08490630569807431 | 0.1281879964025282 | 0.20331833217346876 |
| TRAIN - diagnostic fit assessment | train | recent_rate_origin_depblk | 0.13695591501311502 | 0.04610513059688062 | 0.08014850420441061 | 0.12988532204446285 | 0.2142931524314073 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline | 0.13752173877997595 | 0.028526506067382943 | 0.0997804611712773 | 0.14187536230061842 | 0.16890647598013064 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin | 0.1373754433726491 | 0.027250499276672436 | 0.10328329935584007 | 0.1312516526291645 | 0.18092702960055923 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_dest | 0.13737182645178864 | 0.025772891113417767 | 0.10343823592339427 | 0.13313051260871542 | 0.19999709298861149 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_route | 0.13818849888527876 | 0.03760236373684504 | 0.0864733372117484 | 0.1338225012044505 | 0.20797056754065518 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin_depblk | 0.13740579895843155 | 0.048125958731440065 | 0.06569693126193601 | 0.13463277104850122 | 0.22110678400221517 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_origin | 0.13749439274230682 | 0.040107522917815594 | 0.08226708289992615 | 0.1350440242352513 | 0.20729696349108406 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin_dow | 0.13101515887924184 | 0.038860759602297866 | 0.07827285058319068 | 0.126794496320257 | 0.20685287979869701 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_depblk | 0.1367027031253458 | 0.0505729732654641 | 0.060105790991858934 | 0.1339886334227661 | 0.21669054615423985 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_dest_arrhour | 0.13758028473673722 | 0.048069358124585085 | 0.06967467541520382 | 0.13342221055038667 | 0.21982124235947972 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_route | 0.1369777547237826 | 0.037193400942098054 | 0.08524002545261532 | 0.13215499585428284 | 0.2062040064565496 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_airline | 0.16471909128825601 | 0.03467597837149854 | 0.11756005591252772 | 0.1722690238505589 | 0.22693597682688915 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_origin | 0.1658433988348074 | 0.038779419750088535 | 0.12112602058942072 | 0.15852411916457054 | 0.25823614647415133 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_origin_depblk | 0.1664642125831245 | 0.04349530836722955 | 0.10672181244121152 | 0.16125065603156052 | 0.24885860932896522 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline | 0.1419659801024474 | 0.027744876474796967 | 0.10486005328693283 | 0.1434755451372477 | 0.17608668947159006 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin | 0.1420334139478371 | 0.025577473593211168 | 0.10831831652019809 | 0.1353004280461654 | 0.18066210802796243 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_dest | 0.14183163882843103 | 0.02588031491200645 | 0.1050154246225745 | 0.13797356579820289 | 0.213534020756508 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_route | 0.1422517101133229 | 0.03746752317882833 | 0.08940626215924351 | 0.13814349051258742 | 0.21173053436457973 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin_depblk | 0.14221662508344926 | 0.047966656680047605 | 0.06935455730318311 | 0.1401702442711195 | 0.2210692842083544 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_origin | 0.1415820797393954 | 0.039119264906430515 | 0.08518720993835717 | 0.13715852534173473 | 0.20531205810468667 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin_dow | 0.1431765724935987 | 0.037166017026683686 | 0.0925501662529434 | 0.13896231090369474 | 0.20761717584840822 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_depblk | 0.14165248496496868 | 0.049699543087763656 | 0.0679428603426602 | 0.1369579062172134 | 0.21799280158526216 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_dest_arrhour | 0.1425096789712126 | 0.048625287661161906 | 0.07579117011940245 | 0.13770479669108937 | 0.22284520188209006 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_route | 0.14089505615149614 | 0.03771574627510148 | 0.08879257702405806 | 0.1370355292540922 | 0.21027571438404324 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_airline | 0.17691466312139903 | 0.044156051166069334 | 0.12265031736838819 | 0.16108808710316136 | 0.24936340258473702 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_origin | 0.17596852096221116 | 0.04649873817987871 | 0.11094580905523271 | 0.17402351464056434 | 0.259008856325885 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_origin_depblk | 0.17689354287083278 | 0.04321471036110477 | 0.1137718877239471 | 0.17331860819875664 | 0.25887363874186287 |

### B4. History-feature shift ranking

Absolute mean shifts are normalized by the training standard deviation; this is a descriptive ranking, not a causal test.

| Status | Split | Feature | TrainingMean | ServingMean | MeanShift | AbsoluteMeanShiftInTrainingStdDevs |
| --- | --- | --- | --- | --- | --- | --- |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_airline | 0.13542653065737031 | 0.17691466312139903 | 0.041488132464028715 | 1.0702774613767534 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_origin | 0.13598807440307165 | 0.17596852096221116 | 0.039980446559139515 | 0.8728952661995738 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | recent_rate_origin_depblk | 0.13695591501311502 | 0.17689354287083278 | 0.03993762785771776 | 0.8662295788057015 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_dest | 0.13453097233747868 | 0.14183163882843103 | 0.0073006664909523455 | 0.25159202823344706 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline | 0.13433896526366118 | 0.1419659801024474 | 0.0076270148387862224 | 0.24047193506094197 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_route | 0.13373609641065926 | 0.14089505615149614 | 0.007158959740836884 | 0.2181038141734011 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin | 0.13462844415343198 | 0.1420334139478371 | 0.007404969794405136 | 0.2170359156913757 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_route | 0.13541651222173398 | 0.1422517101133229 | 0.006835197891588929 | 0.18889000841980688 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_dest_arrhour | 0.13444458911502247 | 0.1425096789712126 | 0.008065089856190144 | 0.17417297336201462 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_origin | 0.13474913343302863 | 0.1415820797393954 | 0.006832946306366772 | 0.15795846919026973 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_airline_depblk | 0.1338556251111783 | 0.14165248496496868 | 0.007796859853790389 | 0.15132743843263757 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin_depblk | 0.13518030341418966 | 0.14221662508344926 | 0.007036321669259604 | 0.14751261650357397 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | hist_rate_origin_dow | 0.13654971035030633 | 0.1431765724935987 | 0.006626862143292361 | 0.14154091670122979 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_airline | 0.13542653065737031 | 0.16471909128825601 | 0.0292925606308857 | 0.7556659113646838 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_origin | 0.13598807440307165 | 0.1658433988348074 | 0.029855324431735747 | 0.6518329235959266 |
| VALIDATION - used for diagnostics/decisions | validation | recent_rate_origin_depblk | 0.13695591501311502 | 0.1664642125831245 | 0.029508297570009473 | 0.6400219929537722 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin_dow | 0.13654971035030633 | 0.13101515887924184 | -0.005534551471064492 | 0.11821062092524771 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline | 0.13433896526366118 | 0.13752173877997595 | 0.0031827735163147686 | 0.10034957614566974 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_route | 0.13373609641065926 | 0.1369777547237826 | 0.003241658313123341 | 0.09875988522551232 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_dest | 0.13453097233747868 | 0.13737182645178864 | 0.002840854114309954 | 0.09790013684645653 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin | 0.13462844415343198 | 0.1373754433726491 | 0.0027469992192171167 | 0.08051315636651768 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_route | 0.13541651222173398 | 0.13818849888527876 | 0.0027719866635447787 | 0.07660357352065604 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_dest_arrhour | 0.13444458911502247 | 0.13758028473673722 | 0.003135695621714757 | 0.06771820769896628 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_origin | 0.13474913343302863 | 0.13749439274230682 | 0.0027452593092781863 | 0.06346266143198978 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_airline_depblk | 0.1338556251111783 | 0.1367027031253458 | 0.0028470780141675034 | 0.055258274610694526 |
| VALIDATION - used for diagnostics/decisions | validation | hist_rate_origin_depblk | 0.13518030341418966 | 0.13740579895843155 | 0.0022254955442418956 | 0.04665629091154264 |

### B4. Thin training history

Thin means fewer than 100 strictly earlier flights for that lookup key.

| Feature | TrainingRows | RowsWithFewerThan100EarlierFlights | Share |
| --- | --- | --- | --- |
| hist_rate_airline | 929645 | 0 | 0.0 |
| hist_rate_origin | 929645 | 12345 | 0.013279262514185522 |
| hist_rate_dest | 929645 | 12362 | 0.01329754906442782 |
| hist_rate_route | 929645 | 298136 | 0.3206987613551409 |
| hist_rate_origin_depblk | 929645 | 149888 | 0.1612314378069048 |
| hist_rate_airline_origin | 929645 | 77448 | 0.08330922018620011 |
| hist_rate_origin_dow | 929645 | 91242 | 0.09814714218868492 |
| hist_rate_airline_depblk | 929645 | 1204 | 0.0012951180289250198 |
| hist_rate_dest_arrhour | 929645 | 158177 | 0.1701477445691635 |
| hist_rate_airline_route | 929645 | 514543 | 0.553483318901301 |
| recent_rate_airline | 929645 | 0 | 0.0 |
| recent_rate_origin | 929645 | 41034 | 0.044139429567200385 |
| recent_rate_origin_depblk | 929645 | 393708 | 0.4235035954584815 |

### B5. Staleness by validation day and test week

Validation day 1 is 2024-11-25. Test week 1 starts 2024-12-04 and is descriptive only.

| Status | Split | Model | PeriodUnit | Period | Flights | Prevalence | AUC_ROC | AUC_PR | PredictedPositiveRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 1 | 20948 | 0.14163643307236967 | 0.6784404790683911 | 0.30123870697985633 | 0.6513270956654573 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 2 | 21105 | 0.1544657664060649 | 0.6581346444416558 | 0.2616853970122706 | 0.3824212271973466 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 3 | 21239 | 0.16168369508922265 | 0.6448259856462876 | 0.2735337113198707 | 0.5460709072931871 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 4 | 11201 | 0.08454602267654673 | 0.6277300495605906 | 0.14217288823928595 | 0.7354700473172038 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 5 | 17656 | 0.13315586769370186 | 0.5810429949039879 | 0.17197083899132273 | 0.812301767104667 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 6 | 20551 | 0.2213517590384896 | 0.6222465487043837 | 0.31429491482526417 | 0.5596321346893095 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 7 | 21690 | 0.2292761641309359 | 0.5911123392273898 | 0.29362626058410685 | 0.7030889810972798 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 8 | 21234 | 0.21814071771686916 | 0.6038788465567645 | 0.2873119128246625 | 0.6679853065837807 |
| VALIDATION - used for diagnostics/decisions | validation | LogisticRegression | day | 9 | 17925 | 0.13461645746164574 | 0.5785629354876145 | 0.16729765153724419 | 0.3718270571827057 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 1 | 20948 | 0.14163643307236967 | 0.6575462561340871 | 0.2668564922710982 | 0.6297498567882376 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 2 | 21105 | 0.1544657664060649 | 0.6365531236087165 | 0.22728703160939356 | 0.6288083392561005 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 3 | 21239 | 0.16168369508922265 | 0.6381367209023788 | 0.2593352874198898 | 0.6313856584584961 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 4 | 11201 | 0.08454602267654673 | 0.5898415206242953 | 0.10902282886712182 | 0.4666547629675922 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 5 | 17656 | 0.13315586769370186 | 0.5650915991318451 | 0.16170788764941074 | 0.6553013140009062 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 6 | 20551 | 0.2213517590384896 | 0.6027929392426736 | 0.29402771729474203 | 0.6328645807989879 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 7 | 21690 | 0.2292761641309359 | 0.5706892111221256 | 0.2755890296705617 | 0.6374827109266943 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 8 | 21234 | 0.21814071771686916 | 0.5859971339054597 | 0.2667263163309695 | 0.6298860318357351 |
| VALIDATION - used for diagnostics/decisions | validation | RandomForest | day | 9 | 17925 | 0.13461645746164574 | 0.5607456932931834 | 0.16104189368941857 | 0.6133891213389121 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | week | 1 | 129221 | 0.16830081797850194 | 0.6216673969951138 | 0.24858656213647917 | 0.7202234930854892 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | week | 2 | 130517 | 0.1655646390891608 | 0.6323919486490251 | 0.2524454284027152 | 0.7209712144778075 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | week | 3 | 133646 | 0.2507594690450893 | 0.5931202499690719 | 0.32598523709126614 | 0.7592819837481107 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LogisticRegression | week | 4 | 130599 | 0.2667172030413709 | 0.6464392171626149 | 0.37224700529524957 | 0.7732065329749845 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | week | 1 | 129221 | 0.16830081797850194 | 0.6087195483076284 | 0.23590105198180616 | 0.7847718250129623 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | week | 2 | 130517 | 0.1655646390891608 | 0.6225478661793162 | 0.244387457730019 | 0.7852080571879525 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | week | 3 | 133646 | 0.2507594690450893 | 0.5896702395213701 | 0.3131953817147756 | 0.7896308157370965 |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RandomForest | week | 4 | 130599 | 0.2667172030413709 | 0.622135380470044 | 0.35397690964555206 | 0.7935129671743275 |

### B6. Walk-forward validation upper bound

Validation only. Earlier validation labels are used only for later validation dates; preprocessing and LR remain fitted on train.

| Status | HistoryServingMode | AUC_ROC | AUC_PR | DeltaAUC_PR_vsFrozen |
| --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | frozen_at_training_end | 0.6142484745293798 | 0.24220024706703233 | 0.0 |
| VALIDATION - used for diagnostics/decisions | daily_past_only_refresh | 0.6155098901685063 | 0.24341232837165608 | 0.0012120813046237466 |

### B7. Single-feature validation ranking

Each feature is used directly as the score; this is univariate association, not a causal contribution.

| Status | FeatureUsedAlone | AUC_ROC | AUC_PR |
| --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | hist_rate_airline | 0.542644829623207 | 0.21170866173516484 |
| VALIDATION - used for diagnostics/decisions | hist_rate_origin | 0.5146925933897273 | 0.18278132402444483 |
| VALIDATION - used for diagnostics/decisions | hist_rate_dest | 0.538245404973106 | 0.197137408524509 |
| VALIDATION - used for diagnostics/decisions | hist_rate_route | 0.5584103328327843 | 0.20673253981219633 |
| VALIDATION - used for diagnostics/decisions | hist_rate_origin_depblk | 0.585424691670697 | 0.21951698031111339 |
| VALIDATION - used for diagnostics/decisions | hist_rate_airline_origin | 0.5543256179303169 | 0.20525884820290324 |
| VALIDATION - used for diagnostics/decisions | hist_rate_origin_dow | 0.518317684812839 | 0.17479626931992612 |
| VALIDATION - used for diagnostics/decisions | hist_rate_airline_depblk | 0.5914755574728562 | 0.21866291302337074 |
| VALIDATION - used for diagnostics/decisions | hist_rate_dest_arrhour | 0.5927649778512235 | 0.23315423038895391 |
| VALIDATION - used for diagnostics/decisions | hist_rate_airline_route | 0.575458221608032 | 0.2185161578073562 |
| VALIDATION - used for diagnostics/decisions | recent_rate_airline | 0.5556646189818999 | 0.2159054342636311 |
| VALIDATION - used for diagnostics/decisions | recent_rate_origin | 0.5532467580910504 | 0.20105294832645634 |
| VALIDATION - used for diagnostics/decisions | recent_rate_origin_depblk | 0.5876521258374564 | 0.21653662301672397 |
| VALIDATION - used for diagnostics/decisions | ScheduledDepHour | 0.5783957983053926 | 0.20582074539847622 |

### B8. Date-block bootstrap uncertainty

1,000 whole-date resamples from Spark-produced per-date, 1,000-bin histograms; seed 42. During the completed run Spark resolved the nominal local histogram path through the default HDFS filesystem; its part file was copied to the local diagnostics folder before bootstrap. The committed script now uses an explicit `file://` URI for reproducibility.

| Status | Split | Metric | PointEstimateFromHistogram | CI95_Lower | CI95_Upper | BootstrapResamples | DateBlocks | Method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | validation | LR_AUC_PR | 0.24194902027514406 | 0.19820924322311978 | 0.2819571915637224 | 1000 | 9 | date-block bootstrap; 1000 score bins |
| VALIDATION - used for diagnostics/decisions | validation | RF_AUC_PR | 0.22323392599356828 | 0.18905963244757762 | 0.2572315146906137 | 1000 | 9 | date-block bootstrap; 1000 score bins |
| VALIDATION - used for diagnostics/decisions | validation | LR_minus_RF | 0.018715094281575778 | 0.008018028012067777 | 0.027037920112234368 | 1000 | 9 | date-block bootstrap; 1000 score bins |
| VALIDATION - used for diagnostics/decisions | validation | LR_minus_prevalence | 0.07181839434241036 | 0.048528295108578014 | 0.1016171454498827 | 1000 | 9 | date-block bootstrap; 1000 score bins |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LR_AUC_PR | 0.3032718740988617 | 0.2623365025606679 | 0.34214326346799684 | 1000 | 28 | date-block bootstrap; 1000 score bins |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | RF_AUC_PR | 0.2859233289064573 | 0.24764136117111896 | 0.32446669939875644 | 1000 | 28 | date-block bootstrap; 1000 score bins |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LR_minus_RF | 0.01734854519240442 | 0.00775175941768878 | 0.026078571764089432 | 1000 | 28 | date-block bootstrap; 1000 score bins |
| TEST - DESCRIPTIVE ONLY, not used for any decision | test | LR_minus_prevalence | 0.09009129381286005 | 0.07257274319520418 | 0.10603072025765253 | 1000 | 28 | date-block bootstrap; 1000 score bins |

### B9. Validation signal by operational group

Groups require at least 2,000 validation flights. Group PR-AUC is the bounded 1,000-bin diagnostic estimate.

| Status | GroupType | Group | Flights | Delayed15Plus | Prevalence | AUC_PR | Top10PctPrecision | Top10PctLift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | carrier | DL | 24807 | 4244 | 0.1710807433385738 | 0.23247197210930176 | 0.26536312849162014 | 1.5510987578915223 |
| VALIDATION - used for diagnostics/decisions | carrier | YX | 7889 | 1078 | 0.13664596273291926 | 0.22303984212248 | 0.30138713745271123 | 2.2056058695402956 |
| VALIDATION - used for diagnostics/decisions | carrier | B6 | 6156 | 1399 | 0.22725795971410007 | 0.3071904412663209 | 0.34299516908212563 | 1.5092768126301397 |
| VALIDATION - used for diagnostics/decisions | carrier | OH | 5849 | 1004 | 0.17165327406394257 | 0.21676299506111393 | 0.25 | 1.4564243027888446 |
| VALIDATION - used for diagnostics/decisions | carrier | UA | 18777 | 2884 | 0.1535921606220376 | 0.25049395518400003 | 0.33122362869198313 | 2.156513895960252 |
| VALIDATION - used for diagnostics/decisions | carrier | 9E | 4921 | 664 | 0.1349319244056086 | 0.23843240719469067 | 0.26666666666666666 | 1.976305220883534 |
| VALIDATION - used for diagnostics/decisions | carrier | WN | 33989 | 4147 | 0.12201006207890788 | 0.18052463533102583 | 0.20279720279720279 | 1.6621350677294733 |
| VALIDATION - used for diagnostics/decisions | carrier | OO | 19049 | 4475 | 0.2349204682660507 | 0.29375352766327106 | 0.34355509355509356 | 1.4624315032694921 |
| VALIDATION - used for diagnostics/decisions | carrier | F9 | 5419 | 1348 | 0.2487543827274405 | 0.29791177718762146 | 0.2967032967032967 | 1.1927560569993805 |
| VALIDATION - used for diagnostics/decisions | carrier | NK | 6185 | 1218 | 0.19692805173807598 | 0.30665702107269804 | 0.3798076923076923 | 1.9286622142225591 |
| VALIDATION - used for diagnostics/decisions | carrier | MQ | 6511 | 1119 | 0.17186300107510366 | 0.25381721233256677 | 0.308868501529052 | 1.7971785643035367 |
| VALIDATION - used for diagnostics/decisions | carrier | AA | 23068 | 3670 | 0.15909485000867002 | 0.21446560708700105 | 0.2525817555938038 | 1.587617421808683 |
| VALIDATION - used for diagnostics/decisions | carrier | AS | 5717 | 1264 | 0.22109497988455484 | 0.2691888821254804 | 0.3076923076923077 | 1.3916747809152872 |
| VALIDATION - used for diagnostics/decisions | carrier | G4 | 3323 | 827 | 0.24887150165513092 | 0.3197759493088302 | 0.36607142857142855 | 1.4709254620832612 |
| VALIDATION - used for diagnostics/decisions | top20_origin | BOS | 3660 | 626 | 0.17103825136612022 | 0.24702525833954433 | 0.2702702702702703 | 1.580174423624903 |
| VALIDATION - used for diagnostics/decisions | top20_origin | JFK | 2997 | 546 | 0.18218218218218218 | 0.23991318072413279 | 0.264026402640264 | 1.449243825481449 |
| VALIDATION - used for diagnostics/decisions | top20_origin | PHX | 5132 | 816 | 0.15900233826968044 | 0.20830096634517034 | 0.21621621621621623 | 1.359830418653948 |
| VALIDATION - used for diagnostics/decisions | top20_origin | MIA | 2746 | 535 | 0.1948288419519301 | 0.29316480525761623 | 0.36462093862815886 | 1.871493640136307 |
| VALIDATION - used for diagnostics/decisions | top20_origin | EWR | 3145 | 717 | 0.22798092209856916 | 0.3776745624438204 | 0.4056603773584906 | 1.779361069445541 |
| VALIDATION - used for diagnostics/decisions | top20_origin | DFW | 7462 | 1335 | 0.17890645939426428 | 0.24655452292008917 | 0.29388297872340424 | 1.6426627619730654 |
| VALIDATION - used for diagnostics/decisions | top20_origin | MCO | 3902 | 673 | 0.17247565351101998 | 0.2606387776941171 | 0.29336734693877553 | 1.7009203384176852 |
| VALIDATION - used for diagnostics/decisions | top20_origin | DCA | 3338 | 528 | 0.15817855002995806 | 0.19711371075777326 | 0.20597014925373133 | 1.302137042062415 |
| VALIDATION - used for diagnostics/decisions | top20_origin | DEN | 7483 | 1271 | 0.16985166377121474 | 0.17390969776241233 | 0.20611702127659576 | 1.2135119356512716 |
| VALIDATION - used for diagnostics/decisions | top20_origin | SEA | 3675 | 742 | 0.2019047619047619 | 0.24307522456918343 | 0.2830188679245283 | 1.401744393022428 |
| VALIDATION - used for diagnostics/decisions | top20_origin | ORD | 6654 | 1220 | 0.1833483618875864 | 0.2667011487604295 | 0.3153961136023916 | 1.720201426155995 |
| VALIDATION - used for diagnostics/decisions | top20_origin | IAH | 2862 | 375 | 0.1310272536687631 | 0.1974726050921973 | 0.25517241379310346 | 1.9474758620689654 |
| VALIDATION - used for diagnostics/decisions | top20_origin | LAX | 4741 | 555 | 0.11706391056739085 | 0.16728037230435397 | 0.20842105263157895 | 1.7804039829302987 |
| VALIDATION - used for diagnostics/decisions | top20_origin | MSP | 3075 | 736 | 0.23934959349593496 | 0.32967785168817215 | 0.3851132686084142 | 1.6089990502321654 |
| VALIDATION - used for diagnostics/decisions | top20_origin | CLT | 5330 | 1028 | 0.19287054409005627 | 0.25789852067958385 | 0.29906542056074764 | 1.5506018400669115 |
| VALIDATION - used for diagnostics/decisions | top20_origin | LAS | 4516 | 555 | 0.1228963684676705 | 0.20272328154309777 | 0.22857142857142856 | 1.8598712998712998 |
| VALIDATION - used for diagnostics/decisions | top20_origin | LGA | 3941 | 427 | 0.108348134991119 | 0.14914016391788845 | 0.17676767676767677 | 1.631478721642656 |
| VALIDATION - used for diagnostics/decisions | top20_origin | DTW | 3326 | 1000 | 0.30066145520144316 | 0.40147924568044646 | 0.4388059701492537 | 1.459468656716418 |
| VALIDATION - used for diagnostics/decisions | top20_origin | ATL | 8648 | 1580 | 0.18270120259019426 | 0.2782428045146598 | 0.3360091743119266 | 1.8391185692718617 |
| VALIDATION - used for diagnostics/decisions | top20_origin | SFO | 3625 | 669 | 0.18455172413793103 | 0.26191427008549123 | 0.3085399449035813 | 1.671834529559764 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 12 | 10484 | 1658 | 0.15814574589851202 | 0.2065027541300633 | 0.22851746931067043 | 1.4449801859186182 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 18 | 10200 | 2362 | 0.2315686274509804 | 0.3057111148816973 | 0.3705140640155189 | 1.6000183966800563 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 16 | 9753 | 2006 | 0.2056803034963601 | 0.25188843924117676 | 0.300507614213198 | 1.4610422539488135 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 10 | 10433 | 1569 | 0.1503881913160165 | 0.20689310345499484 | 0.24667931688804554 | 1.6402838196895977 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 5 | 5429 | 496 | 0.0913612083256585 | 0.1332611786924028 | 0.163302752293578 | 1.7874408108907962 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 20 | 7986 | 1787 | 0.22376659153518658 | 0.2870138746770829 | 0.33707865168539325 | 1.506385065674063 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 22 | 4413 | 946 | 0.2143666440063449 | 0.28135902648153704 | 0.33783783783783783 | 1.5759813724930003 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 13 | 10229 | 1680 | 0.16423892853651384 | 0.19986302747781287 | 0.23814133591481124 | 1.4499688839717881 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 14 | 9899 | 1714 | 0.17314880290938478 | 0.2271141726488149 | 0.2782782782782783 | 1.6071625884928102 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 17 | 10471 | 2214 | 0.2114411231019005 | 0.2771693857123173 | 0.3431001890359168 | 1.6226748326084395 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 6 | 12295 | 1344 | 0.109312728751525 | 0.14380415712768282 | 0.17756255044390637 | 1.6243538375802298 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 9 | 9907 | 1355 | 0.13677197940849903 | 0.18453428165944932 | 0.226 | 1.6523852398523988 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 11 | 11259 | 1685 | 0.1496580513367084 | 0.18458612143206193 | 0.2323943661971831 | 1.5528357086137001 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 7 | 12046 | 1553 | 0.1289224638884277 | 0.16509823093139903 | 0.19325657894736842 | 1.4990140051513199 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 15 | 9667 | 1840 | 0.1903382641977863 | 0.22519153561068525 | 0.2441025641025641 | 1.2824671125975473 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 19 | 9157 | 2107 | 0.23009719340395327 | 0.3104087009931005 | 0.3837837837837838 | 1.6679203170897523 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 8 | 11910 | 1647 | 0.13828715365239294 | 0.17916175137626597 | 0.19051580698835274 | 1.377682611555119 |
| VALIDATION - used for diagnostics/decisions | scheduled_departure_hour | 21 | 5790 | 1234 | 0.21312607944732298 | 0.28168136985970704 | 0.3384879725085911 | 1.5882053167137296 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_2 | 39030 | 5673 | 0.14534973097617218 | 0.21660649135689236 | 0.2463179278821737 | 1.6946569231872446 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_4 | 11201 | 947 | 0.08454602267654673 | 0.14217288823928595 | 0.15459363957597172 | 1.8285146324080879 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_1 | 42182 | 7599 | 0.18014793039685173 | 0.2852463756264431 | 0.3559521014322611 | 1.9758878197941359 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_6 | 20551 | 4549 | 0.2213517590384896 | 0.31429491482526417 | 0.3757838880849011 | 1.6976774420823924 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_7 | 21690 | 4973 | 0.2292761641309359 | 0.29362626058410685 | 0.3401826484018265 | 1.4837244407471581 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_5 | 17656 | 2351 | 0.13315586769370186 | 0.17197083899132273 | 0.20426487093153758 | 1.5340283118533506 |
| VALIDATION - used for diagnostics/decisions | weekday | DOW_3 | 21239 | 3434 | 0.16168369508922265 | 0.2735337113198707 | 0.3252449836677555 | 2.011612757169324 |

### B9. Excluded validation groups

Excluded groups are listed with the reason.

| GroupType | Group | Flights | Reason |
| --- | --- | --- | --- |
| carrier | HA | 1889 | fewer than 2000 flights |
| origin | SLC | 2664 | outside 20 largest or fewer than 2000 flights |
| origin | BNA | 2389 | outside 20 largest or fewer than 2000 flights |
| origin | SAN | 2386 | outside 20 largest or fewer than 2000 flights |
| origin | BWI | 2381 | outside 20 largest or fewer than 2000 flights |
| origin | FLL | 2357 | outside 20 largest or fewer than 2000 flights |
| origin | PHL | 2278 | outside 20 largest or fewer than 2000 flights |
| origin | AUS | 2080 | outside 20 largest or fewer than 2000 flights |
| origin | TPA | 2019 | outside 20 largest or fewer than 2000 flights |
| origin | DAL | 1784 | outside 20 largest or fewer than 2000 flights |
| origin | MDW | 1774 | outside 20 largest or fewer than 2000 flights |
| origin | STL | 1615 | outside 20 largest or fewer than 2000 flights |
| origin | RDU | 1537 | outside 20 largest or fewer than 2000 flights |
| origin | PDX | 1480 | outside 20 largest or fewer than 2000 flights |
| origin | HNL | 1464 | outside 20 largest or fewer than 2000 flights |
| origin | SMF | 1392 | outside 20 largest or fewer than 2000 flights |
| origin | IAD | 1373 | outside 20 largest or fewer than 2000 flights |
| origin | HOU | 1362 | outside 20 largest or fewer than 2000 flights |
| origin | SJC | 1283 | outside 20 largest or fewer than 2000 flights |
| origin | MSY | 1268 | outside 20 largest or fewer than 2000 flights |
| origin | MCI | 1232 | outside 20 largest or fewer than 2000 flights |
| origin | IND | 1201 | outside 20 largest or fewer than 2000 flights |
| origin | CMH | 1092 | outside 20 largest or fewer than 2000 flights |
| origin | SNA | 1057 | outside 20 largest or fewer than 2000 flights |
| origin | SAT | 1035 | outside 20 largest or fewer than 2000 flights |
| origin | PIT | 1024 | outside 20 largest or fewer than 2000 flights |
| origin | CVG | 969 | outside 20 largest or fewer than 2000 flights |
| origin | CLE | 964 | outside 20 largest or fewer than 2000 flights |
| origin | OAK | 960 | outside 20 largest or fewer than 2000 flights |
| origin | RSW | 943 | outside 20 largest or fewer than 2000 flights |
| origin | SJU | 912 | outside 20 largest or fewer than 2000 flights |
| origin | BUR | 824 | outside 20 largest or fewer than 2000 flights |
| origin | JAX | 795 | outside 20 largest or fewer than 2000 flights |
| origin | PBI | 781 | outside 20 largest or fewer than 2000 flights |
| origin | MKE | 666 | outside 20 largest or fewer than 2000 flights |
| origin | OGG | 637 | outside 20 largest or fewer than 2000 flights |
| origin | ONT | 633 | outside 20 largest or fewer than 2000 flights |
| origin | CHS | 597 | outside 20 largest or fewer than 2000 flights |
| origin | ABQ | 592 | outside 20 largest or fewer than 2000 flights |
| origin | BOI | 590 | outside 20 largest or fewer than 2000 flights |
| origin | OMA | 587 | outside 20 largest or fewer than 2000 flights |
| origin | OKC | 572 | outside 20 largest or fewer than 2000 flights |
| origin | BDL | 566 | outside 20 largest or fewer than 2000 flights |
| origin | SDF | 551 | outside 20 largest or fewer than 2000 flights |
| origin | MEM | 543 | outside 20 largest or fewer than 2000 flights |
| origin | ORF | 538 | outside 20 largest or fewer than 2000 flights |
| origin | RNO | 531 | outside 20 largest or fewer than 2000 flights |
| origin | TUS | 525 | outside 20 largest or fewer than 2000 flights |
| origin | BUF | 523 | outside 20 largest or fewer than 2000 flights |
| origin | SAV | 484 | outside 20 largest or fewer than 2000 flights |
| origin | BHM | 471 | outside 20 largest or fewer than 2000 flights |
| origin | RIC | 469 | outside 20 largest or fewer than 2000 flights |
| origin | ELP | 465 | outside 20 largest or fewer than 2000 flights |
| origin | GRR | 461 | outside 20 largest or fewer than 2000 flights |
| origin | LGB | 439 | outside 20 largest or fewer than 2000 flights |
| origin | GEG | 437 | outside 20 largest or fewer than 2000 flights |
| origin | SRQ | 434 | outside 20 largest or fewer than 2000 flights |
| origin | TYS | 424 | outside 20 largest or fewer than 2000 flights |
| origin | TUL | 424 | outside 20 largest or fewer than 2000 flights |
| origin | GSP | 412 | outside 20 largest or fewer than 2000 flights |
| origin | PVD | 399 | outside 20 largest or fewer than 2000 flights |
| origin | PSP | 393 | outside 20 largest or fewer than 2000 flights |
| origin | KOA | 387 | outside 20 largest or fewer than 2000 flights |
| origin | DSM | 384 | outside 20 largest or fewer than 2000 flights |
| origin | XNA | 382 | outside 20 largest or fewer than 2000 flights |
| origin | LIH | 375 | outside 20 largest or fewer than 2000 flights |
| origin | ANC | 326 | outside 20 largest or fewer than 2000 flights |
| origin | COS | 315 | outside 20 largest or fewer than 2000 flights |
| origin | FAT | 314 | outside 20 largest or fewer than 2000 flights |
| origin | LIT | 313 | outside 20 largest or fewer than 2000 flights |
| origin | MYR | 306 | outside 20 largest or fewer than 2000 flights |
| origin | SYR | 305 | outside 20 largest or fewer than 2000 flights |
| origin | PNS | 295 | outside 20 largest or fewer than 2000 flights |
| origin | GSO | 292 | outside 20 largest or fewer than 2000 flights |
| origin | ALB | 289 | outside 20 largest or fewer than 2000 flights |
| origin | ICT | 286 | outside 20 largest or fewer than 2000 flights |
| origin | HPN | 285 | outside 20 largest or fewer than 2000 flights |
| origin | MSN | 284 | outside 20 largest or fewer than 2000 flights |
| origin | AVL | 282 | outside 20 largest or fewer than 2000 flights |
| origin | ROC | 276 | outside 20 largest or fewer than 2000 flights |
| origin | SFB | 261 | outside 20 largest or fewer than 2000 flights |
| origin | SGF | 245 | outside 20 largest or fewer than 2000 flights |
| origin | PIE | 242 | outside 20 largest or fewer than 2000 flights |
| origin | PWM | 238 | outside 20 largest or fewer than 2000 flights |
| origin | MAF | 237 | outside 20 largest or fewer than 2000 flights |
| origin | LEX | 224 | outside 20 largest or fewer than 2000 flights |
| origin | CHA | 219 | outside 20 largest or fewer than 2000 flights |
| origin | PGD | 215 | outside 20 largest or fewer than 2000 flights |
| origin | CID | 212 | outside 20 largest or fewer than 2000 flights |
| origin | EYW | 209 | outside 20 largest or fewer than 2000 flights |
| origin | HSV | 208 | outside 20 largest or fewer than 2000 flights |
| origin | AZA | 198 | outside 20 largest or fewer than 2000 flights |
| origin | EUG | 195 | outside 20 largest or fewer than 2000 flights |
| origin | BTV | 193 | outside 20 largest or fewer than 2000 flights |
| origin | SBA | 193 | outside 20 largest or fewer than 2000 flights |
| origin | ILM | 192 | outside 20 largest or fewer than 2000 flights |
| origin | VPS | 191 | outside 20 largest or fewer than 2000 flights |
| origin | CAE | 186 | outside 20 largest or fewer than 2000 flights |
| origin | FSD | 184 | outside 20 largest or fewer than 2000 flights |
| origin | ATW | 183 | outside 20 largest or fewer than 2000 flights |
| origin | DAY | 181 | outside 20 largest or fewer than 2000 flights |
| origin | JAN | 180 | outside 20 largest or fewer than 2000 flights |
| origin | BZN | 178 | outside 20 largest or fewer than 2000 flights |
| origin | FAR | 174 | outside 20 largest or fewer than 2000 flights |
| origin | SBN | 172 | outside 20 largest or fewer than 2000 flights |
| origin | PSC | 164 | outside 20 largest or fewer than 2000 flights |
| origin | ITO | 158 | outside 20 largest or fewer than 2000 flights |
| origin | LBB | 157 | outside 20 largest or fewer than 2000 flights |
| origin | SBP | 156 | outside 20 largest or fewer than 2000 flights |
| origin | ECP | 150 | outside 20 largest or fewer than 2000 flights |
| origin | FWA | 140 | outside 20 largest or fewer than 2000 flights |
| origin | MDT | 136 | outside 20 largest or fewer than 2000 flights |
| origin | TLH | 134 | outside 20 largest or fewer than 2000 flights |
| origin | RAP | 132 | outside 20 largest or fewer than 2000 flights |
| origin | MHT | 131 | outside 20 largest or fewer than 2000 flights |
| origin | SHV | 130 | outside 20 largest or fewer than 2000 flights |
| origin | BTR | 128 | outside 20 largest or fewer than 2000 flights |
| origin | MFE | 127 | outside 20 largest or fewer than 2000 flights |
| origin | STS | 127 | outside 20 largest or fewer than 2000 flights |
| origin | RDM | 127 | outside 20 largest or fewer than 2000 flights |
| origin | MRY | 126 | outside 20 largest or fewer than 2000 flights |
| origin | AMA | 122 | outside 20 largest or fewer than 2000 flights |
| origin | HRL | 116 | outside 20 largest or fewer than 2000 flights |
| origin | MLI | 112 | outside 20 largest or fewer than 2000 flights |
| origin | ISP | 112 | outside 20 largest or fewer than 2000 flights |
| origin | STT | 110 | outside 20 largest or fewer than 2000 flights |
| origin | PIA | 108 | outside 20 largest or fewer than 2000 flights |
| origin | BIS | 108 | outside 20 largest or fewer than 2000 flights |
| origin | GJT | 107 | outside 20 largest or fewer than 2000 flights |
| origin | AGS | 106 | outside 20 largest or fewer than 2000 flights |
| origin | ASE | 106 | outside 20 largest or fewer than 2000 flights |
| origin | BIL | 104 | outside 20 largest or fewer than 2000 flights |
| origin | CRP | 102 | outside 20 largest or fewer than 2000 flights |
| origin | DRO | 102 | outside 20 largest or fewer than 2000 flights |
| origin | MFR | 98 | outside 20 largest or fewer than 2000 flights |
| origin | ABE | 98 | outside 20 largest or fewer than 2000 flights |
| origin | IDA | 94 | outside 20 largest or fewer than 2000 flights |
| origin | ROA | 94 | outside 20 largest or fewer than 2000 flights |
| origin | SGU | 91 | outside 20 largest or fewer than 2000 flights |
| origin | PVU | 90 | outside 20 largest or fewer than 2000 flights |
| origin | CRW | 89 | outside 20 largest or fewer than 2000 flights |
| origin | JNU | 88 | outside 20 largest or fewer than 2000 flights |
| origin | GRB | 86 | outside 20 largest or fewer than 2000 flights |
| origin | LFT | 85 | outside 20 largest or fewer than 2000 flights |
| origin | MSO | 84 | outside 20 largest or fewer than 2000 flights |
| origin | CAK | 84 | outside 20 largest or fewer than 2000 flights |
| origin | SAF | 82 | outside 20 largest or fewer than 2000 flights |
| origin | BGR | 81 | outside 20 largest or fewer than 2000 flights |
| origin | CHO | 81 | outside 20 largest or fewer than 2000 flights |
| origin | BFL | 81 | outside 20 largest or fewer than 2000 flights |
| origin | GPT | 79 | outside 20 largest or fewer than 2000 flights |
| origin | GNV | 74 | outside 20 largest or fewer than 2000 flights |
| origin | MLB | 73 | outside 20 largest or fewer than 2000 flights |
| origin | MGM | 73 | outside 20 largest or fewer than 2000 flights |
| origin | FAI | 69 | outside 20 largest or fewer than 2000 flights |
| origin | MOB | 67 | outside 20 largest or fewer than 2000 flights |
| origin | JAC | 66 | outside 20 largest or fewer than 2000 flights |
| origin | DAB | 65 | outside 20 largest or fewer than 2000 flights |
| origin | FNT | 65 | outside 20 largest or fewer than 2000 flights |
| origin | MOT | 65 | outside 20 largest or fewer than 2000 flights |
| origin | LNK | 62 | outside 20 largest or fewer than 2000 flights |
| origin | BQN | 61 | outside 20 largest or fewer than 2000 flights |
| origin | CPR | 61 | outside 20 largest or fewer than 2000 flights |
| origin | TRI | 61 | outside 20 largest or fewer than 2000 flights |
| origin | TVC | 60 | outside 20 largest or fewer than 2000 flights |
| origin | FCA | 59 | outside 20 largest or fewer than 2000 flights |
| origin | FLG | 59 | outside 20 largest or fewer than 2000 flights |
| origin | MBS | 57 | outside 20 largest or fewer than 2000 flights |
| origin | ACY | 57 | outside 20 largest or fewer than 2000 flights |
| origin | DLH | 57 | outside 20 largest or fewer than 2000 flights |
| origin | GTF | 57 | outside 20 largest or fewer than 2000 flights |
| origin | EVV | 53 | outside 20 largest or fewer than 2000 flights |
| origin | GRK | 52 | outside 20 largest or fewer than 2000 flights |
| origin | YUM | 50 | outside 20 largest or fewer than 2000 flights |
| origin | MLU | 50 | outside 20 largest or fewer than 2000 flights |
| origin | XWA | 50 | outside 20 largest or fewer than 2000 flights |
| origin | KTN | 49 | outside 20 largest or fewer than 2000 flights |
| origin | ACV | 49 | outside 20 largest or fewer than 2000 flights |
| origin | AEX | 49 | outside 20 largest or fewer than 2000 flights |
| origin | OAJ | 49 | outside 20 largest or fewer than 2000 flights |
| origin | BLI | 46 | outside 20 largest or fewer than 2000 flights |
| origin | LRD | 44 | outside 20 largest or fewer than 2000 flights |
| origin | AVP | 42 | outside 20 largest or fewer than 2000 flights |
| origin | ABI | 42 | outside 20 largest or fewer than 2000 flights |
| origin | COU | 41 | outside 20 largest or fewer than 2000 flights |
| origin | FAY | 41 | outside 20 largest or fewer than 2000 flights |
| origin | RDD | 41 | outside 20 largest or fewer than 2000 flights |
| origin | BMI | 41 | outside 20 largest or fewer than 2000 flights |
| origin | BRO | 39 | outside 20 largest or fewer than 2000 flights |
| origin | EGE | 36 | outside 20 largest or fewer than 2000 flights |
| origin | ORH | 35 | outside 20 largest or fewer than 2000 flights |
| origin | MTJ | 35 | outside 20 largest or fewer than 2000 flights |
| origin | HLN | 34 | outside 20 largest or fewer than 2000 flights |
| origin | TTN | 34 | outside 20 largest or fewer than 2000 flights |
| origin | HDN | 34 | outside 20 largest or fewer than 2000 flights |
| origin | LAN | 32 | outside 20 largest or fewer than 2000 flights |
| origin | GFK | 30 | outside 20 largest or fewer than 2000 flights |
| origin | ELM | 27 | outside 20 largest or fewer than 2000 flights |
| origin | LWS | 27 | outside 20 largest or fewer than 2000 flights |
| origin | SUN | 26 | outside 20 largest or fewer than 2000 flights |
| origin | MHK | 25 | outside 20 largest or fewer than 2000 flights |
| origin | VLD | 25 | outside 20 largest or fewer than 2000 flights |
| origin | LAW | 25 | outside 20 largest or fewer than 2000 flights |
| origin | ABY | 25 | outside 20 largest or fewer than 2000 flights |
| origin | ROW | 25 | outside 20 largest or fewer than 2000 flights |
| origin | AZO | 25 | outside 20 largest or fewer than 2000 flights |
| origin | BQK | 25 | outside 20 largest or fewer than 2000 flights |
| origin | CSG | 25 | outside 20 largest or fewer than 2000 flights |
| origin | ACT | 25 | outside 20 largest or fewer than 2000 flights |
| origin | FSM | 25 | outside 20 largest or fewer than 2000 flights |
| origin | HHH | 25 | outside 20 largest or fewer than 2000 flights |
| origin | DHN | 25 | outside 20 largest or fewer than 2000 flights |
| origin | RST | 25 | outside 20 largest or fewer than 2000 flights |
| origin | CLL | 25 | outside 20 largest or fewer than 2000 flights |
| origin | SJT | 25 | outside 20 largest or fewer than 2000 flights |
| origin | LCH | 25 | outside 20 largest or fewer than 2000 flights |
| origin | CMI | 25 | outside 20 largest or fewer than 2000 flights |
| origin | GRI | 24 | outside 20 largest or fewer than 2000 flights |
| origin | SIT | 24 | outside 20 largest or fewer than 2000 flights |
| origin | STX | 24 | outside 20 largest or fewer than 2000 flights |
| origin | BLV | 24 | outside 20 largest or fewer than 2000 flights |
| origin | TYR | 24 | outside 20 largest or fewer than 2000 flights |
| origin | USA | 23 | outside 20 largest or fewer than 2000 flights |
| origin | LCK | 22 | outside 20 largest or fewer than 2000 flights |
| origin | PAE | 22 | outside 20 largest or fewer than 2000 flights |
| origin | RFD | 20 | outside 20 largest or fewer than 2000 flights |
| origin | GUM | 18 | outside 20 largest or fewer than 2000 flights |
| origin | GTR | 18 | outside 20 largest or fewer than 2000 flights |
| origin | HOB | 18 | outside 20 largest or fewer than 2000 flights |
| origin | MQT | 18 | outside 20 largest or fewer than 2000 flights |
| origin | CWA | 17 | outside 20 largest or fewer than 2000 flights |
| origin | SPS | 17 | outside 20 largest or fewer than 2000 flights |
| origin | ABR | 17 | outside 20 largest or fewer than 2000 flights |
| origin | PRC | 17 | outside 20 largest or fewer than 2000 flights |
| origin | RKS | 17 | outside 20 largest or fewer than 2000 flights |
| origin | PLN | 17 | outside 20 largest or fewer than 2000 flights |
| origin | SWF | 17 | outside 20 largest or fewer than 2000 flights |
| origin | GGG | 17 | outside 20 largest or fewer than 2000 flights |
| origin | BTM | 17 | outside 20 largest or fewer than 2000 flights |
| origin | COD | 17 | outside 20 largest or fewer than 2000 flights |
| origin | IMT | 17 | outside 20 largest or fewer than 2000 flights |
| origin | BPT | 17 | outside 20 largest or fewer than 2000 flights |
| origin | SUX | 17 | outside 20 largest or fewer than 2000 flights |
| origin | CYS | 17 | outside 20 largest or fewer than 2000 flights |
| origin | GCC | 17 | outside 20 largest or fewer than 2000 flights |
| origin | CIU | 17 | outside 20 largest or fewer than 2000 flights |
| origin | VCT | 17 | outside 20 largest or fewer than 2000 flights |
| origin | SLN | 17 | outside 20 largest or fewer than 2000 flights |
| origin | CMX | 17 | outside 20 largest or fewer than 2000 flights |
| origin | JST | 17 | outside 20 largest or fewer than 2000 flights |
| origin | GCK | 17 | outside 20 largest or fewer than 2000 flights |
| origin | TXK | 17 | outside 20 largest or fewer than 2000 flights |
| origin | DIK | 17 | outside 20 largest or fewer than 2000 flights |
| origin | SHR | 17 | outside 20 largest or fewer than 2000 flights |
| origin | RIW | 17 | outside 20 largest or fewer than 2000 flights |
| origin | BJI | 16 | outside 20 largest or fewer than 2000 flights |
| origin | HYS | 16 | outside 20 largest or fewer than 2000 flights |
| origin | JLN | 16 | outside 20 largest or fewer than 2000 flights |
| origin | SCK | 16 | outside 20 largest or fewer than 2000 flights |
| origin | LAR | 16 | outside 20 largest or fewer than 2000 flights |
| origin | BRD | 16 | outside 20 largest or fewer than 2000 flights |
| origin | DDC | 16 | outside 20 largest or fewer than 2000 flights |
| origin | APN | 16 | outside 20 largest or fewer than 2000 flights |
| origin | RHI | 16 | outside 20 largest or fewer than 2000 flights |
| origin | CDC | 16 | outside 20 largest or fewer than 2000 flights |
| origin | PIH | 16 | outside 20 largest or fewer than 2000 flights |
| origin | PIB | 16 | outside 20 largest or fewer than 2000 flights |
| origin | GUC | 16 | outside 20 largest or fewer than 2000 flights |
| origin | LBL | 16 | outside 20 largest or fewer than 2000 flights |
| origin | MCW | 16 | outside 20 largest or fewer than 2000 flights |
| origin | TWF | 16 | outside 20 largest or fewer than 2000 flights |
| origin | LBF | 16 | outside 20 largest or fewer than 2000 flights |
| origin | DVL | 16 | outside 20 largest or fewer than 2000 flights |
| origin | BET | 16 | outside 20 largest or fewer than 2000 flights |
| origin | MEI | 16 | outside 20 largest or fewer than 2000 flights |
| origin | ITH | 16 | outside 20 largest or fewer than 2000 flights |
| origin | FOD | 16 | outside 20 largest or fewer than 2000 flights |
| origin | EAR | 16 | outside 20 largest or fewer than 2000 flights |
| origin | SWO | 16 | outside 20 largest or fewer than 2000 flights |
| origin | JMS | 16 | outside 20 largest or fewer than 2000 flights |
| origin | BFF | 16 | outside 20 largest or fewer than 2000 flights |
| origin | DEC | 16 | outside 20 largest or fewer than 2000 flights |
| origin | WRG | 15 | outside 20 largest or fewer than 2000 flights |
| origin | INL | 15 | outside 20 largest or fewer than 2000 flights |
| origin | PSG | 15 | outside 20 largest or fewer than 2000 flights |
| origin | ADQ | 15 | outside 20 largest or fewer than 2000 flights |
| origin | YAK | 15 | outside 20 largest or fewer than 2000 flights |
| origin | SCC | 15 | outside 20 largest or fewer than 2000 flights |
| origin | ESC | 15 | outside 20 largest or fewer than 2000 flights |
| origin | CDV | 15 | outside 20 largest or fewer than 2000 flights |
| origin | LSE | 14 | outside 20 largest or fewer than 2000 flights |
| origin | PBG | 14 | outside 20 largest or fewer than 2000 flights |
| origin | PSE | 14 | outside 20 largest or fewer than 2000 flights |
| origin | HIB | 14 | outside 20 largest or fewer than 2000 flights |
| origin | PSM | 13 | outside 20 largest or fewer than 2000 flights |
| origin | EKO | 9 | outside 20 largest or fewer than 2000 flights |
| origin | BGM | 9 | outside 20 largest or fewer than 2000 flights |
| origin | PQI | 9 | outside 20 largest or fewer than 2000 flights |
| origin | TOL | 9 | outside 20 largest or fewer than 2000 flights |
| origin | SPN | 9 | outside 20 largest or fewer than 2000 flights |
| origin | SCE | 9 | outside 20 largest or fewer than 2000 flights |
| origin | OTZ | 8 | outside 20 largest or fewer than 2000 flights |
| origin | HTS | 8 | outside 20 largest or fewer than 2000 flights |
| origin | BRW | 8 | outside 20 largest or fewer than 2000 flights |
| origin | LBE | 8 | outside 20 largest or fewer than 2000 flights |
| origin | IAG | 7 | outside 20 largest or fewer than 2000 flights |
| origin | OME | 7 | outside 20 largest or fewer than 2000 flights |
| origin | HGR | 6 | outside 20 largest or fewer than 2000 flights |
| origin | OTH | 5 | outside 20 largest or fewer than 2000 flights |
| origin | STC | 4 | outside 20 largest or fewer than 2000 flights |
| origin | CKB | 3 | outside 20 largest or fewer than 2000 flights |
| origin | SMX | 3 | outside 20 largest or fewer than 2000 flights |
| origin | PPG | 3 | outside 20 largest or fewer than 2000 flights |
| origin | ADK | 2 | outside 20 largest or fewer than 2000 flights |
| origin | SPI | 2 | outside 20 largest or fewer than 2000 flights |

### B10. Standardized LR coefficients

Twenty most positive and twenty most negative coefficients; coefficient multiplied by training feature standard deviation.

| Status | Direction | Rank | Feature | RawCoefficient | TrainingStdDev | StandardizedCoefficient |
| --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | positive | 1 | Distance__imputed | 0.00033154131691876486 | 590.4959503625466 | 0.1957738050183963 |
| VALIDATION - used for diagnostics/decisions | positive | 2 | hist_rate_dest_arrhour | 3.57600029835311 | 0.046305059278209805 | 0.1655869057941367 |
| VALIDATION - used for diagnostics/decisions | positive | 3 | hist_rate_airline_route | 4.354226261593547 | 0.03282363386430844 | 0.1429215285729031 |
| VALIDATION - used for diagnostics/decisions | positive | 4 | hist_rate_origin_depblk | 2.5395788338967877 | 0.0476997956923171 | 0.12113739152140969 |
| VALIDATION - used for diagnostics/decisions | positive | 5 | OriginPrevHourDemandLog | 0.09753099497218219 | 1.1738980536086019 | 0.11449144516435501 |
| VALIDATION - used for diagnostics/decisions | positive | 6 | hist_rate_airline_origin | 2.587618244873219 | 0.043257866079571265 | 0.11193484350178096 |
| VALIDATION - used for diagnostics/decisions | positive | 7 | DayOfWeekCat__ohe_DOW_4 | 0.308331514939179 | 0.358895877920751 | 0.11065890974473179 |
| VALIDATION - used for diagnostics/decisions | positive | 8 | DayOfWeekCat__ohe_DOW_5 | 0.2645031597703306 | 0.3623186819411028 | 0.09583443621724311 |
| VALIDATION - used for diagnostics/decisions | positive | 9 | Reporting_Airline__ohe_F9 | 0.533497923272494 | 0.1675808041520681 | 0.08940401099546287 |
| VALIDATION - used for diagnostics/decisions | positive | 10 | hist_rate_airline_depblk | 1.6187314823826833 | 0.051523107339592686 | 0.08340207592078097 |
| VALIDATION - used for diagnostics/decisions | positive | 11 | recent_rate_origin | 1.7656487029004755 | 0.04580211178507939 | 0.080870439263428 |
| VALIDATION - used for diagnostics/decisions | positive | 12 | DestHourDemandLog | 0.07381431614838851 | 1.0068485856282445 | 0.07431983981312107 |
| VALIDATION - used for diagnostics/decisions | positive | 13 | Reporting_Airline__ohe_OO | 0.22242048388599422 | 0.31645105751762753 | 0.0703851973393053 |
| VALIDATION - used for diagnostics/decisions | positive | 14 | recent_rate_airline | 1.1645969829552232 | 0.038763903717696134 | 0.04514432531719568 |
| VALIDATION - used for diagnostics/decisions | positive | 15 | Reporting_Airline__ohe_AS | 0.21190275038212564 | 0.17994634689349764 | 0.03813112582794822 |
| VALIDATION - used for diagnostics/decisions | positive | 16 | Reporting_Airline__ohe_AA | 0.10130284376749323 | 0.34488886924249973 | 0.03493822323802035 |
| VALIDATION - used for diagnostics/decisions | positive | 17 | recent_rate_origin_depblk | 0.7449536548948098 | 0.04610513059688062 | 0.034346185547548744 |
| VALIDATION - used for diagnostics/decisions | positive | 18 | ScheduledDepHour | 0.006366565448603965 | 4.851423291521072 | 0.030886903904350577 |
| VALIDATION - used for diagnostics/decisions | positive | 19 | hist_rate_route | 0.8438646892759873 | 0.036186127306414975 | 0.030536195075529193 |
| VALIDATION - used for diagnostics/decisions | positive | 20 | DayOfWeekCat__ohe_DOW_7 | 0.0672466323622724 | 0.36285490596883757 | 0.02440077046253334 |
| VALIDATION - used for diagnostics/decisions | negative | 1 | CRSElapsedTime__imputed | -0.002722497331042092 | 71.5507230031871 | -0.1947966524103089 |
| VALIDATION - used for diagnostics/decisions | negative | 2 | hist_rate_origin | -5.129536053204441 | 0.03411863778774282 | -0.1750127826184502 |
| VALIDATION - used for diagnostics/decisions | negative | 3 | hist_rate_airline | -5.369688394908138 | 0.031716860584389335 | -0.17030965820291477 |
| VALIDATION - used for diagnostics/decisions | negative | 4 | DayOfWeekCat__ohe_DOW_2 | -0.40363581349190863 | 0.3423659477460032 | -0.1381911578303863 |
| VALIDATION - used for diagnostics/decisions | negative | 5 | Reporting_Airline__ohe_DL | -0.2451247674015846 | 0.34995756135553807 | -0.08578326582770204 |
| VALIDATION - used for diagnostics/decisions | negative | 6 | Reporting_Airline__ohe_9E | -0.34542385370686207 | 0.16483715186279985 | -0.05693868423051158 |
| VALIDATION - used for diagnostics/decisions | negative | 7 | DayOfWeekCat__ohe_DOW_3 | -0.16379912397016294 | 0.3447521986321454 | -0.05647010812273302 |
| VALIDATION - used for diagnostics/decisions | negative | 8 | OriginDayDemandLog | -0.0425121760745749 | 1.2450474677637264 | -0.05292967717077515 |
| VALIDATION - used for diagnostics/decisions | negative | 9 | DayOfWeekCat__ohe_DOW_6 | -0.1413763979167804 | 0.33680017566681664 | -0.04761559565351341 |
| VALIDATION - used for diagnostics/decisions | negative | 10 | Reporting_Airline__ohe_YX | -0.18243838280170924 | 0.21080274236061164 | -0.03845851140643535 |
| VALIDATION - used for diagnostics/decisions | negative | 11 | hist_rate_dest | -1.3152074047220503 | 0.029017876846949252 | -0.0381645264984202 |
| VALIDATION - used for diagnostics/decisions | negative | 12 | ArrMinuteSin | -0.057182285226907825 | 0.6277213215265588 | -0.0358945396505432 |
| VALIDATION - used for diagnostics/decisions | negative | 13 | Reporting_Airline__ohe_WN | -0.08451960604236122 | 0.39566852526652546 | -0.03344174787888878 |
| VALIDATION - used for diagnostics/decisions | negative | 14 | DepMinuteSin | -0.04284800913550283 | 0.7424117477815169 | -0.03181086535124706 |
| VALIDATION - used for diagnostics/decisions | negative | 15 | Reporting_Airline__ohe_MQ | -0.14167724928326234 | 0.1914936528205719 | -0.027130293986822658 |
| VALIDATION - used for diagnostics/decisions | negative | 16 | ScheduledArrHour | -0.0018469304561036662 | 5.136991148565948 | -0.009487665405021403 |
| VALIDATION - used for diagnostics/decisions | negative | 17 | ArrMinuteCos | -0.009154976438442791 | 0.6888778660564353 | -0.006306660632711414 |
| VALIDATION - used for diagnostics/decisions | negative | 18 | Reporting_Airline__ohe_HA | -0.016105249097383065 | 0.10399391563777009 | -0.0016748479159585275 |
| VALIDATION - used for diagnostics/decisions | negative | 19 | IsHolidayWindow | -0.011611727906996428 | 0.13481059053232125 | -0.001565383896242823 |
| VALIDATION - used for diagnostics/decisions | negative | 20 | HolidayPeriod__ohe_Halloween | -0.011611727906996428 | 0.13481059053232125 | -0.001565383896242823 |

### B11. Validation-selected operating points

Every threshold is selected on validation only, then applied unchanged to the descriptive test scores.

| Status | Model | ValidationChosenOperatingPoint | Threshold | FlaggedShare | Precision | Recall | AppliedSplit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION - used for diagnostics/decisions | LogisticRegression | validation_precision_at_least_30pct | 0.2429 | 0.0912710531319685 | 0.3 | 0.16094289778500304 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | LogisticRegression | validation_precision_at_least_30pct | 0.2429 | 0.16919632888853264 | 0.3321377007760332 | 0.2636097508571838 | test |
| VALIDATION - used for diagnostics/decisions | LogisticRegression | validation_precision_at_least_40pct | 0.5247 | 0.00082397478521916 | 0.3916083916083916 | 0.001896633475580844 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | LogisticRegression | validation_precision_at_least_40pct | 0.5247 | 0.0020687694066410553 | 0.5396678966789668 | 0.0052371019578704246 | test |
| VALIDATION - used for diagnostics/decisions | LogisticRegression | validation_precision_at_least_50pct | 0.6282 | 9.219298296158434e-05 | 0.5625 | 0.0003048160942897785 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | LogisticRegression | validation_precision_at_least_50pct | 0.6282 | 0.00023474044005244445 | 0.5365853658536586 | 0.000590852528580253 | test |
| VALIDATION - used for diagnostics/decisions | LogisticRegression | validation_top_10pct | 0.2370517382970907 | 0.10009853125054019 | 0.29363343311075296 | 0.17276298855246225 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | LogisticRegression | validation_top_10pct | 0.2370517382970907 | 0.1830880009465956 | 0.3282535049773284 | 0.2819172269321325 | test |
| VALIDATION - used for diagnostics/decisions | LogisticRegression | validation_top_20pct | 0.19161949804069178 | 0.20009334539524862 | 0.2578471462304901 | 0.3032581453634085 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | LogisticRegression | validation_top_20pct | 0.19161949804069178 | 0.3284152348454053 | 0.29990004881337023 | 0.46201086810560144 | test |
| VALIDATION - used for diagnostics/decisions | RandomForest | validation_precision_at_least_30pct | 0.39 | 1.7286184305297065e-05 | 0.3333333333333333 | 3.38684549210865e-05 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | RandomForest | validation_precision_at_least_30pct | 0.39 | 0.0 | 0.0 | 0.0 | test |
| VALIDATION - used for diagnostics/decisions | RandomForest | validation_precision_at_least_40pct | UNREACHABLE |  |  |  |  |
| TEST - DESCRIPTIVE ONLY, not used for any decision | RandomForest | validation_precision_at_least_40pct | UNREACHABLE |  |  |  |  |
| VALIDATION - used for diagnostics/decisions | RandomForest | validation_precision_at_least_50pct | UNREACHABLE |  |  |  |  |
| TEST - DESCRIPTIVE ONLY, not used for any decision | RandomForest | validation_precision_at_least_50pct | UNREACHABLE |  |  |  |  |
| VALIDATION - used for diagnostics/decisions | RandomForest | validation_top_10pct | 0.20361712747669572 | 0.10011005537341039 | 0.2640727523886267 | 0.15538847117794485 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | RandomForest | validation_top_10pct | 0.20361712747669572 | 0.0838958515829712 | 0.34494995450409466 | 0.13575284459683268 | test |
| VALIDATION - used for diagnostics/decisions | RandomForest | validation_top_20pct | 0.17516924125947123 | 0.2001106315795539 | 0.24961847447378271 | 0.29360563571089887 | validation |
| TEST - DESCRIPTIVE ONLY, not used for any decision | RandomForest | validation_top_20pct | 0.17516924125947123 | 0.20833118631711334 | 0.2990692731903043 | 0.2922660984933261 | test |

## C. Earlier-flights feasibility gate

Part C stopped at C1 because the required raw BTS CSVs were absent from both permitted locations. Nothing was downloaded, nothing was staged, and C2-C5 were not run.

### C1. Raw-file availability

| RequiredFile | ExpectedLocalPath | ExistsLocally | ExpectedHDFSPath | ExistsInHDFS |
| --- | --- | --- | --- | --- |
| flights_2024_10.csv | C:\bda\data\flights_2024_10.csv | 0 | /user/root/flights/raw/flights_2024_10.csv | 0 |
| flights_2024_11.csv | C:\bda\data\flights_2024_11.csv | 0 | /user/root/flights/raw/flights_2024_11.csv | 0 |
| flights_2024_12.csv | C:\bda\data\flights_2024_12.csv | 0 | /user/root/flights/raw/flights_2024_12.csv | 0 |

#### C1 status

```text
PART_C_GATE=STOPPED
Reason: one or more required raw BTS CSVs are absent from both allowed locations.
Expected local directory: C:\bda\data
Expected HDFS directory: /user/root/flights/raw
No files were downloaded or staged; C2-C5 were not run.
```

## D. Repository and course-readiness audit

### D1. Local test executions

Both tests use small synthetic data under local Spark. Full logs are preserved in the diagnostics directory.

| Test | StartUTC | WallTimeMs | ExitCode | ExpectedMarker | MarkerSeen | Result | Command | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_history_features | 2026-10-31T08:12:46.440804+00:00 | 33633 | 0 | TEMPORAL_HISTORY_TEST_STATUS=PASSED | 1 | PASSED | spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py | results\experiments\ml_redesign\diagnostics\local_test_temporal_history_features.log |
| ml_pipeline | 2026-10-31T08:13:20.078439+00:00 | 4033220 | 0 | ML_PIPELINE_TESTS_STATUS=PASSED | 1 | PASSED | spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py | results\experiments\ml_redesign\diagnostics\local_test_ml_pipeline.log |

### D2. Tracked files over 5 MB

| Path | Bytes | Finding |
| --- | --- | --- |
| None |  |  |

### D2. Generated or local artifacts that must not be committed

| Path | GitStatus | Recommendation |
| --- | --- | --- |
| analysis/__pycache__/ | ignored | do not commit generated data/artifacts/cache |
| docker/conf.zip | ignored | do not commit generated data/artifacts/cache |
| ml_actual_data_handoff/ | ignored | do not commit generated data/artifacts/cache |
| results/data_handoff/ | ignored | do not commit generated data/artifacts/cache |
| results/data_validation/ | ignored | do not commit generated data/artifacts/cache |
| results/experiments/ | ignored | do not commit generated data/artifacts/cache |
| src/__pycache__/ | ignored | do not commit generated data/artifacts/cache |
| src/ml_redesign_candidate/__pycache__/ | ignored | do not commit generated data/artifacts/cache |
| src/ml_redesign_candidate/tests/__pycache__/ | ignored | do not commit generated data/artifacts/cache |
| src/tests/__pycache__/ | ignored | do not commit generated data/artifacts/cache |

### D3. README command audit

| Workflow | READMEStatus | AuditResult | EvidenceOrIssue |
| --- | --- | --- | --- |
| cluster startup | documented | consistent | docker compose config/build/up/ps is valid from repository root |
| handoff staging | documented | consistent | matches current /results bind mount and canonical HDFS clean/reference paths |
| data validation | documented | consistent | actual evidence identifies application ..._0001 and PASSED marker |
| local tests | documented | consistent | commands match the two tracked src/tests scripts |
| four controlled selection runs | documented | consistent | run tags match actual-data results narrative |
| XGBoost selection | documented | wording mismatch | command explicitly requests 1g executors, but following prose says executors remain at 512 MB |
| frozen final evaluation | documented | consistent | 1536m driver; executor memory inherited as 512m from spark-defaults |
| plot generation | documented | consistent | tracked script and actual screenshots exist |
| saved-model verification | documented | consistent | command is valid; application ..._0008 and the PASSED marker are recorded in the actual-data report |

### D4. Course-topic coverage

`Covered` requires demonstrable repository workload/evidence; configuration alone is not counted as a project workload.

| Topic | Coverage | RepositoryEvidence |
| --- | --- | --- |
| HDFS | covered | canonical inputs, output tables, saved models, fsck and two DataNodes are implemented/documented |
| MapReduce | demonstrated only | handover records Hadoop MapReduce Pi success; no project MapReduce job is tracked |
| Pig | not covered | no Pig scripts or documented Pig run |
| Hive | not covered | no Hive DDL/query or documented Hive run |
| HBase | not covered | no HBase schema/client or documented HBase run |
| Spark RDD | not directly covered | project uses DataFrame/Spark SQL APIs; no explicit RDD workload |
| SparkSQL | covered | 02_holiday_analysis.py registers views and executes Spark SQL |
| SparkML | covered | 03_ml.py implements leakage-controlled MLlib pipelines and persisted models |
| Streaming | not covered | no Structured Streaming/DStream source or documented streaming run |

## Findings (measured)

- LogisticRegression: train PR-AUC 0.23646, validation PR-AUC 0.24220; gap -0.00574.
- RandomForest: train PR-AUC 0.23276, validation PR-AUC 0.22367; gap 0.00909.
- Validation rolling-history estimate: PR-AUC 0.24341, a 0.00121 change versus frozen history.
- Validation LR_minus_RF: 0.01872, 95% date-block interval [0.00802, 0.02704].
- Validation LR_minus_prevalence: 0.07182, 95% date-block interval [0.04853, 0.10162].
- Largest validation mean shifts in training-standard-deviation units: recent_rate_airline (0.756), recent_rate_origin (0.652), recent_rate_origin_depblk (0.640).
- LR validation decile expected calibration error (count-weighted absolute bin gap): 0.03083.
- On validation, the LR top-10% risk group has precision 0.29328, recall 0.17395, and lift 1.724.
- Strongest univariate validation score: hist_rate_dest_arrhour with PR-AUC 0.23315.
- Thinnest lookup: hist_rate_airline_route; 0.553 of training rows have fewer than 100 earlier flights for its key.
- Saved-result consistency audit: 38 checks, 0 mismatches.
- No tracked file exceeds 5 MiB; generated data, logs, caches, and handoff artifacts remain ignored.
- The raw-file gate found none of the three required BTS CSVs locally or in HDFS raw staging.

## Inferences (not proven)

- A walk-forward improvement, if measured, estimates the benefit of refreshing historical-rate lookups; it does not prove production benefit or justify changing the frozen model without a new validation protocol.
- Group and coefficient differences identify where the fitted score varies and ranks better or worse. They do not establish that a carrier, airport, hour, weekday, or history lookup causes delays.
- Wide date-block intervals or intervals crossing zero mean the observed validation comparison may not be stable across dates; they are not formal population-level guarantees.

## What could not be done and why

- Part C2-C5 could not be measured because `C:\bda\data\flights_2024_10.csv`, `..._11.csv`, and `..._12.csv` were missing and `/user/root/flights/raw` contained none of them. The hard stop rule prohibited downloading or reconstructing them.
- Exact, globally sorted PR-AUC and exact `ntile(10)` calibration were not used in the new diagnostic job: repeated attempts exceeded the fixed 512 MB executor envelope. Bounded 1,000-bin AUC and approximate quantile cut points are explicitly identified wherever used.
- Diagnostic YARN attempts `_0006` through `_0010` failed from executor loss/heap pressure. `_0011` was intentionally killed after a deterministic diagnostic column-selection defect was found. Their logs are preserved; no final artifacts were altered.

## Reproduction commands

```powershell
python analysis/inventory_audit.py
python analysis/part_c_probe.py
python analysis/repo_audit.py
docker cp analysis/model_diagnostics.py master:/tmp/model_diagnostics.py
docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py'
python analysis/bootstrap_uncertainty.py
python analysis/postprocess_diagnostics.py
docker cp analysis/runtime_inventory.py master:/tmp/runtime_inventory.py
docker exec master python3 /tmp/runtime_inventory.py
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'
docker exec master bash -lc 'spark-submit --master "local[2]" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py'
python analysis/build_diagnostics_report.py
```
