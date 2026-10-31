#!/usr/bin/env python3
"""Audit saved final-run tables without reading row-level flight data."""

import csv
import json
import os
from pathlib import Path

FINAL = Path("results/experiments/ml_redesign/actual_final")
OUT = Path("results/experiments/ml_redesign/diagnostics")
FILES = [
    "frozen_config.json", "experiment_comparison.csv", "hyperparameter_search.csv",
    "validation_threshold_search.csv", "validation_metrics.csv", "ranking_metrics.csv",
    "metrics_by_holiday_slice.csv", "threshold_transfer.csv", "run_summary.txt",
    "preprocessing_metadata.json",
]


def rows(name):
    with (FINAL / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a, b, tolerance=5e-6):
    return abs(float(a) - float(b)) <= tolerance


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "inventory_full.txt").open("w", encoding="utf-8") as output:
        for name in FILES:
            output.write(f"===== {name} =====\n")
            output.write((FINAL / name).read_text(encoding="utf-8"))
            output.write("\n\n")

    metrics = {r["Model"]: r for r in rows("metrics.csv")}
    validation = {r["Model"]: r for r in rows("validation_metrics.csv")}
    confusion = rows("confusion_counts.csv")
    class_counts = {r["Split"]: r for r in rows("class_counts.csv")}
    checks = []

    # Recompute test threshold metrics from saved confusion counts.
    for model, metric in metrics.items():
        counts = {(r["ActualLabel"], r["PredictedLabel"]): int(r["Count"]) for r in confusion if r["Model"] == model}
        tn, fp = counts[("0", "0")], counts[("0", "1")]
        fn, tp = counts[("1", "0")], counts[("1", "1")]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        for field, computed in (("Precision_class1", precision), ("Recall_class1", recall),
                                ("F1_class1", f1), ("Accuracy", accuracy)):
            checks.append(["metrics.csv vs confusion_counts.csv", model, field, metric[field], computed,
                           "MATCH" if close(metric[field], computed, 1e-12) else "MISMATCH"])

    test_n = int(class_counts["test"]["Flights"])
    test_pos = int(class_counts["test"]["Delayed15Plus"])
    prevalence = test_pos / test_n
    baselines = {r["Model"]: r for r in rows("baseline_metrics.csv")}
    expected = {
        "MajorityClass_AlwaysNotDelayed": {"AUC_PR": prevalence, "F1_class1": 0.0, "Accuracy": 1 - prevalence},
        "AlwaysDelayed": {"AUC_PR": prevalence, "F1_class1": 2 * prevalence / (1 + prevalence), "Accuracy": prevalence},
    }
    for model, values in expected.items():
        for field, computed in values.items():
            saved = baselines[model][field]
            checks.append(["baseline_metrics.csv vs class_counts.csv", model, field, saved, computed,
                           "MATCH" if close(saved, computed, 1e-12) else "MISMATCH"])

    for r in rows("ranking_metrics.csv"):
        flagged = int(r["FlaggedFlights"])
        precision = float(r["Precision_class1"])
        expected_share = flagged / test_n
        expected_lift = precision / prevalence
        for field, saved, computed in (("ActualFlaggedFraction", r["ActualFlaggedFraction"], expected_share),
                                       ("Lift", r["Lift"], expected_lift)):
            checks.append(["ranking_metrics.csv internal", r["Model"] + "@" + r["TargetFlaggedFraction"], field,
                           saved, computed, "MATCH" if close(saved, computed, 1e-12) else "MISMATCH"])

    for r in rows("metrics_by_holiday_slice.csv"):
        computed = int(r["Delayed15Plus"]) / int(r["Flights"])
        checks.append(["metrics_by_holiday_slice.csv internal", r["Model"] + "/" + r["Slice"], "Prevalence",
                       r["Prevalence"], computed, "MATCH" if close(r["Prevalence"], computed, 1e-12) else "MISMATCH"])

    # Headline values printed in ACTUAL_DATA_ML_RESULTS.md, compared at displayed precision.
    doc_expected = {
        ("validation", "LogisticRegression", "AUC_PR"): 0.24215,
        ("validation", "RandomForest", "AUC_PR"): 0.22366,
        ("test", "LogisticRegression", "AUC_PR"): 0.30359,
        ("test", "RandomForest", "AUC_PR"): 0.28687,
        ("test", "LogisticRegression", "AUC_ROC"): 0.62664,
        ("test", "RandomForest", "AUC_ROC"): 0.60936,
    }
    sources = {"validation": validation, "test": metrics}
    for (split, model, field), displayed in doc_expected.items():
        saved = float(sources[split][model][field])
        checks.append(["ACTUAL_DATA_ML_RESULTS.md headline", f"{split}/{model}", field, displayed, saved,
                       "MATCH" if round(saved, 5) == displayed else "MISMATCH"])

    with (OUT / "consistency_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Check", "Entity", "Field", "ReportedOrSaved", "Recomputed", "Result"])
        writer.writerows(checks)
    mismatches = [r for r in checks if r[-1] == "MISMATCH"]
    summary = {"checks": len(checks), "mismatches": len(mismatches), "mismatch_rows": mismatches}
    (OUT / "consistency_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
