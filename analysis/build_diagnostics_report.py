#!/usr/bin/env python3
"""Build the self-contained diagnostics report from measured artifacts."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "experiments" / "ml_redesign" / "diagnostics"
FINAL = ROOT / "results" / "experiments" / "ml_redesign" / "actual_final"
REPORT = ROOT / "docs" / "DIAGNOSTICS_REPORT.md"
TEST_STATUS = "TEST - DESCRIPTIVE ONLY, not used for any decision"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_table(path):
    rows = read_csv(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    lines = ["| " + " | ".join(map(md_cell, header)) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    if rows:
        lines.extend("| " + " | ".join(md_cell(row.get(h, "")) for h in header) + " |" for row in rows)
    else:
        lines.append("| " + " | ".join(["None"] + [""] * (len(header) - 1)) + " |")
    return "\n".join(lines)


def table_section(title, filename, note=""):
    path = DIAG / filename
    body = [f"### {title}", ""]
    if note:
        body += [note, ""]
    body += [md_table(path) if path.exists() else f"Not produced: `{filename}`.", ""]
    return body


def exact_file(title, path):
    if path.suffix.lower() == ".csv":
        return [f"#### {title}", "", md_table(path), ""]
    language = "json" if path.suffix.lower() == ".json" else "text"
    return [f"#### {title}", "", f"```{language}", path.read_text(encoding="utf-8-sig").rstrip(), "```", ""]


def fnum(value, digits=5):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


lines = [
    "# Flight-delay model diagnostics",
    "",
    "This is a diagnostics-only audit of the frozen `actual_final` run. It does not change `src/03_ml.py`, "
    "retrain either saved final model, rerun the frozen final evaluation, or overwrite `actual_final`, "
    "`actual_xgb_combined`, or the baseline. Large-data work ran on YARN with a 1536 MB driver and two "
    "512 MB, one-core executors. Test labels had already been seen once; every test-derived table row is "
    f"labelled exactly **{TEST_STATUS}** and is descriptive only.",
    "",
    "PR-AUC/ROC computed by this diagnostic job uses a bounded-memory 1,000-bin score histogram unless a "
    "source table explicitly states otherwise. Calibration bins use approximate score deciles (relative "
    "quantile error 0.001) to avoid a single-partition global sort on the constrained executors.",
    "",
    "## A. Consistency and inventory",
    "",
]

lines += table_section("A1. Headline consistency checks", "consistency_checks.csv",
                       "All values were recomputed from saved final-run files; tolerance and pass/fail are explicit.")
summary = DIAG / "consistency_summary.json"
if summary.exists():
    lines += exact_file("A1 summary", summary)

lines += ["### A2. Full frozen-run contents", "",
          "The requested files follow in full, not excerpts.", ""]
for name in ("frozen_config.json", "experiment_comparison.csv", "hyperparameter_search.csv",
             "validation_threshold_search.csv", "validation_metrics.csv", "ranking_metrics.csv",
             "metrics_by_holiday_slice.csv", "threshold_transfer.csv", "run_summary.txt",
             "preprocessing_metadata.json"):
    lines += exact_file(name, FINAL / name)

lines += table_section("A3. YARN runtime inventory", "yarn_runtime_inventory.csv",
                       "Commands and resource settings come from Spark History Server environment metadata. "
                       "Failure/retry evidence is cross-checked against the captured application logs. For diagnostic "
                       "attempts `_0006`-`_0010`, YARN reports `SUCCEEDED` because the shell pipeline ended in `tee`; "
                       "the explicit `DIAGNOSTICS_YARN_STATUS=FAILED` marker and executor-loss evidence are authoritative.")

lines += ["## B. Model diagnostics", "",
          "Validation is the only decision-bearing split. Test rows are retained solely to describe the already-scored final window.", ""]

tables = [
    ("B1. Train versus validation PR-AUC", "train_validation_test_auc.csv", "The train/validation diagnostic models use the frozen feature specification and hyperparameters and are fitted on train only. The test rows load the saved final models."),
    ("B2. Logistic Regression calibration deciles", "lr_calibration_deciles.csv", "Bins are approximate equal-count deciles as documented above."),
    ("B3. Gains and lift", "gains_lift.csv", "Fractions 1%, 2%, 5%, 10%, 20%, 30%, 40%, and 50% are shown for both models."),
    ("B4. History-feature distributions", "history_feature_skew.csv", "Training is past-only; validation uses frozen training tables; test is descriptive and uses final saved history tables."),
    ("B4. History-feature shift ranking", "history_feature_shift_ranking.csv", "Absolute mean shifts are normalized by the training standard deviation; this is a descriptive ranking, not a causal test."),
    ("B4. Thin training history", "history_thin_share.csv", "Thin means fewer than 100 strictly earlier flights for that lookup key."),
    ("B5. Staleness by validation day and test week", "staleness_by_period.csv", "Validation day 1 is 2024-11-25. Test week 1 starts 2024-12-04 and is descriptive only."),
    ("B6. Walk-forward validation upper bound", "walk_forward_validation.csv", "Validation only. Earlier validation labels are used only for later validation dates; preprocessing and LR remain fitted on train."),
    ("B7. Single-feature validation ranking", "single_feature_ranking_validation.csv", "Each feature is used directly as the score; this is univariate association, not a causal contribution."),
    ("B8. Date-block bootstrap uncertainty", "bootstrap_uncertainty.csv", "1,000 whole-date resamples from Spark-produced per-date, 1,000-bin histograms; seed 42. During the completed run Spark resolved the nominal local histogram path through the default HDFS filesystem; its part file was copied to the local diagnostics folder before bootstrap. The committed script now uses an explicit `file://` URI for reproducibility."),
    ("B9. Validation signal by operational group", "validation_signal_by_group.csv", "Groups require at least 2,000 validation flights. Group PR-AUC is the bounded 1,000-bin diagnostic estimate."),
    ("B9. Excluded validation groups", "validation_groups_excluded.csv", "Excluded groups are listed with the reason."),
    ("B10. Standardized LR coefficients", "lr_standardized_coefficients_top20.csv", "Twenty most positive and twenty most negative coefficients; coefficient multiplied by training feature standard deviation."),
    ("B11. Validation-selected operating points", "operating_points.csv", "Every threshold is selected on validation only, then applied unchanged to the descriptive test scores."),
]
for title, filename, note in tables:
    lines += table_section(title, filename, note)

lines += ["## C. Earlier-flights feasibility gate", "",
          "Part C stopped at C1 because the required raw BTS CSVs were absent from both permitted locations. "
          "Nothing was downloaded, nothing was staged, and C2-C5 were not run.", ""]
lines += table_section("C1. Raw-file availability", "part_c_raw_availability.csv")
status_path = DIAG / "part_c_status.txt"
if status_path.exists():
    lines += exact_file("C1 status", status_path)

lines += ["## D. Repository and course-readiness audit", ""]
lines += table_section("D1. Local test executions", "local_test_runs.csv",
                       "Both tests use small synthetic data under local Spark. Full logs are preserved in the diagnostics directory.")
lines += table_section("D2. Tracked files over 5 MB", "tracked_files_over_5mb.csv")
lines += table_section("D2. Generated or local artifacts that must not be committed", "noncommit_artifacts.csv")
lines += table_section("D3. README command audit", "readme_command_audit.csv")
lines += table_section("D4. Course-topic coverage", "course_topic_coverage.csv",
                       "`Covered` requires demonstrable repository workload/evidence; configuration alone is not counted as a project workload.")

lines += ["## Findings (measured)", ""]
auc_path = DIAG / "train_validation_test_auc.csv"
if auc_path.exists():
    auc = read_csv(auc_path)
    for model in ("LogisticRegression", "RandomForest"):
        train = next(r for r in auc if r["Model"] == model and r["Split"] == "train")
        val = next(r for r in auc if r["Model"] == model and r["Split"] == "validation")
        lines.append(f"- {model}: train PR-AUC {fnum(train['AUC_PR'])}, validation PR-AUC {fnum(val['AUC_PR'])}; gap {fnum(float(train['AUC_PR'])-float(val['AUC_PR']))}.")
walk_path = DIAG / "walk_forward_validation.csv"
if walk_path.exists():
    walk = read_csv(walk_path)
    rolling = next(r for r in walk if r["HistoryServingMode"] == "daily_past_only_refresh")
    lines.append(f"- Validation rolling-history estimate: PR-AUC {fnum(rolling['AUC_PR'])}, a {fnum(rolling['DeltaAUC_PR_vsFrozen'])} change versus frozen history.")
boot_path = DIAG / "bootstrap_uncertainty.csv"
if boot_path.exists():
    for row in read_csv(boot_path):
        if row["Split"] == "validation" and row["Metric"] in ("LR_minus_RF", "LR_minus_prevalence"):
            lines.append(f"- Validation {row['Metric']}: {fnum(row['PointEstimateFromHistogram'])}, 95% date-block interval [{fnum(row['CI95_Lower'])}, {fnum(row['CI95_Upper'])}].")
shift_path = DIAG / "history_feature_shift_ranking.csv"
if shift_path.exists():
    shifts = [r for r in read_csv(shift_path) if r["Split"] == "validation"][:3]
    lines.append("- Largest validation mean shifts in training-standard-deviation units: " + ", ".join(
        f"{r['Feature']} ({fnum(r['AbsoluteMeanShiftInTrainingStdDevs'], 3)})" for r in shifts) + ".")
calibration_path = DIAG / "lr_calibration_deciles.csv"
if calibration_path.exists():
    cal = [r for r in read_csv(calibration_path) if r["Split"] == "validation"]
    ece = sum(int(r["Flights"]) * abs(float(r["MeanPredictedProbability"]) - float(r["ObservedDelayRate"])) for r in cal) / sum(int(r["Flights"]) for r in cal)
    lines.append(f"- LR validation decile expected calibration error (count-weighted absolute bin gap): {fnum(ece)}.")
gain_path = DIAG / "gains_lift.csv"
if gain_path.exists():
    gain = next(r for r in read_csv(gain_path) if r["Split"] == "validation" and r["Model"] == "LogisticRegression" and float(r["TargetFlaggedFraction"]) == 0.10)
    lines.append(f"- On validation, the LR top-10% risk group has precision {fnum(gain['Precision'])}, recall {fnum(gain['Recall'])}, and lift {fnum(gain['Lift'], 3)}.")
single_path = DIAG / "single_feature_ranking_validation.csv"
if single_path.exists():
    best = max(read_csv(single_path), key=lambda r: float(r["AUC_PR"]))
    lines.append(f"- Strongest univariate validation score: {best['FeatureUsedAlone']} with PR-AUC {fnum(best['AUC_PR'])}.")
thin_path = DIAG / "history_thin_share.csv"
if thin_path.exists():
    thin = max(read_csv(thin_path), key=lambda r: float(r["Share"]))
    lines.append(f"- Thinnest lookup: {thin['Feature']}; {fnum(thin['Share'], 3)} of training rows have fewer than 100 earlier flights for its key.")
consistency = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
if consistency:
    lines.append(f"- Saved-result consistency audit: {consistency.get('checks')} checks, {consistency.get('mismatches')} mismatches.")
lines += [
    "- No tracked file exceeds 5 MiB; generated data, logs, caches, and handoff artifacts remain ignored.",
    "- The raw-file gate found none of the three required BTS CSVs locally or in HDFS raw staging.",
    "",
    "## Inferences (not proven)",
    "",
    "- A walk-forward improvement, if measured, estimates the benefit of refreshing historical-rate lookups; it does not prove production benefit or justify changing the frozen model without a new validation protocol.",
    "- Group and coefficient differences identify where the fitted score varies and ranks better or worse. They do not establish that a carrier, airport, hour, weekday, or history lookup causes delays.",
    "- Wide date-block intervals or intervals crossing zero mean the observed validation comparison may not be stable across dates; they are not formal population-level guarantees.",
    "",
    "## What could not be done and why",
    "",
    "- Part C2-C5 could not be measured because `C:\\bda\\data\\flights_2024_10.csv`, `..._11.csv`, and `..._12.csv` were missing and `/user/root/flights/raw` contained none of them. The hard stop rule prohibited downloading or reconstructing them.",
    "- Exact, globally sorted PR-AUC and exact `ntile(10)` calibration were not used in the new diagnostic job: repeated attempts exceeded the fixed 512 MB executor envelope. Bounded 1,000-bin AUC and approximate quantile cut points are explicitly identified wherever used.",
    "- Diagnostic YARN attempts `_0006` through `_0010` failed from executor loss/heap pressure. `_0011` was intentionally killed after a deterministic diagnostic column-selection defect was found. Their logs are preserved; no final artifacts were altered.",
    "",
    "## Reproduction commands",
    "",
    "```powershell",
    "python analysis/inventory_audit.py",
    "python analysis/part_c_probe.py",
    "python analysis/repo_audit.py",
    "docker cp analysis/model_diagnostics.py master:/tmp/model_diagnostics.py",
    "docker exec master bash -lc 'spark-submit --master yarn --deploy-mode client --driver-memory 1536m --executor-memory 512m --num-executors 2 --executor-cores 1 /tmp/model_diagnostics.py'",
    "python analysis/bootstrap_uncertainty.py",
    "python analysis/postprocess_diagnostics.py",
    "docker cp analysis/runtime_inventory.py master:/tmp/runtime_inventory.py",
    "docker exec master python3 /tmp/runtime_inventory.py",
    "docker exec master bash -lc 'spark-submit --master \"local[2]\" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_temporal_history_features.py'",
    "docker exec master bash -lc 'spark-submit --master \"local[2]\" --driver-memory 2g --conf spark.eventLog.enabled=false --conf spark.ui.enabled=false /src/tests/test_ml_pipeline.py'",
    "python analysis/build_diagnostics_report.py",
    "```",
    "",
]

REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"REPORT_STATUS=SUCCEEDED path={REPORT}")
